# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MThreads (MUSA) specialization of the shifted mHC post -> pre boundary.

The generic path spends five launches on the boundary and, on MUSA, the skinny
projection dominates every shape: the shared NVIDIA tune space picks a small
M tile and walks the whole ``hc * H`` reduction axis with the transposed ``fn``
weight, so a single token costs ~250 us in the projection alone.

This module changes the dataflow while keeping the post arithmetic verbatim:

1. one CTA per token computes the post, the collapse, the stream mean and the
   row sum of squares while the just-rounded bf16 post values are still in
   registers.  Narrow rows keep the whole row live and apply the RMSNorm in the
   same pass; wide rows stream H in tiles and normalise in a second pass.  The
   sum of squares is CTA-local, so the coefficient launch never folds tiles.
2. ``fn`` is packed once into a contiguous ``(hc*H, 32)`` bf16 operand (the
   same cached transpose-cast the MetaX/PPU paths use), and the projection
   becomes a pure split-K GEMM that can pipeline with ``num_stages >= 2``.
   The K split is sized from the 60 MTT S5000 multiprocessors instead of a
   single CTA walking a 20K-28K axis, and the split is aligned to 256 K
   elements so the kernel uses the unmasked load path (MUSA predicates cost
   2-4x on the long-K shapes).
3. the split-K reduction and the Sinkhorn coefficients share one launch, so
   the whole boundary is three launches (post/collapse, project, coefficients).
4. after the first (tuning) call the kernels are launched through a cached
   compiled-kernel path.  A bare MUSA kernel launch is ~2 us, but LibEntry's
   per-call key reconstruction for a 20-argument kernel costs ~57 us, which is
   the real floor of the decode-sized shapes; the cached path cuts it to
   ~21 us without changing specialization (alignment and every varying scalar
   are in the cache key, and the LibEntry path is the fallback).

Every reduction runs in a fixed order (split index ascending, H tiles
ascending), so repeated calls stay bitwise deterministic; the tests compare
two calls with ``rtol=0, atol=0``.

The post fma chain and the coefficient arithmetic are copied verbatim from
``mhc_post`` and ``_mhc_shifted_coeff_kernel``: the post output has to stay
bitwise equal to ``mhc_post``.

This module is intentionally *not* re-exported from ``_mthreads.ops.__init__``:
``SpecOpRegistrar`` would then replace the public operator with this private
launcher.  ``flaggems_vllm.ops.mhc.mhc_shifted_post_pre`` imports it lazily.
"""

import weakref

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

_FN_PACK_CACHE: dict = {}
_FN_STRIDE = 32

# The flat token kernel keeps the whole row live in registers; wider rows use
# the H-streamed loop kernel.  Same boundary the MetaX path measured: H=4095
# still fits one 4096-wide tile, but the 4-stream register footprint starts
# spilling, and the streamed variant wins.
_TOKEN_FLAT_MAX_HIDDEN = 2048

# Smallest BLOCK_K the split-K policy will hand to a CTA.  Splitting further
# only adds reduce work.
_MIN_K_PER_SPLIT = 128

# A split is "even" when every K tile is full.  Aligning the split to 256
# elements covers every BLOCK_K in the tune space (64/128/256) without the
# per-element K mask, which is very expensive on MUSA.
_ALIGN_K = 256

# MTT S5000 has 60 multiprocessors.  The projection is memory bound, so aim a
# few waves past one full machine before giving up parallelism.
_PROJECT_TARGET_CTAS = 240
_PROJECT_MAX_SPLIT = 32


@triton.jit
def _pack_fn_bf16_kernel(
    src_ptr,  # (mix_size, K) float32, contiguous
    dst_ptr,  # (K, 32) bfloat16, zero-padded
    numel,
    K,
    mix_size,
    dst_stride,
    BLOCK: tl.constexpr,
):
    """One-time ``fn`` transpose-cast for the project kernel's B operands.

    Rows are written with a power-of-two ``dst_stride`` so every B tile starts
    on a 64-byte boundary (the pack also zero-pads the unused columns).
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    n = offs // K
    k = offs % K
    value = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + k * dst_stride + n, value.to(tl.bfloat16), mask=mask)


def _fn_bf16_packed(fn: torch.Tensor) -> torch.Tensor:
    """``fn`` as a contiguous ``(hc*H, mix_size)`` bfloat16 tensor, cached."""
    key = id(fn)
    version = None if fn.is_inference() else fn._version
    ptr = fn.data_ptr()
    entry = _FN_PACK_CACHE.get(key)
    if entry is not None:
        ref, cached_version, cached_ptr, cached = entry
        if ref() is fn and cached_version == version and cached_ptr == ptr:
            return cached

    mix_size, K = fn.shape
    # ``mix_size`` is hc*(hc+2) = 24 for hc=4; the power-of-two 32-column
    # stride keeps every B row 64-byte aligned and lets the project N tile
    # load 32 columns without masking (the padding columns are zero).
    mix_stride = _FN_STRIDE
    out = torch.zeros(K, mix_stride, dtype=torch.bfloat16, device=fn.device)
    if fn.numel():
        _pack_fn_bf16_kernel[lambda meta: (triton.cdiv(fn.numel(), meta["BLOCK"]),)](
            fn, out, fn.numel(), K, mix_size, mix_stride, BLOCK=1024
        )
    _FN_PACK_CACHE[key] = (
        weakref.ref(fn, lambda ref, key=key: _FN_PACK_CACHE.pop(key, None)),
        version,
        ptr,
        out,
    )
    return out


@triton.jit
def _project_tiles(
    a_ptr,
    bt_ptr,
    acc,
    offs_m,
    offs_n,
    k_begin,
    k_end,
    a_stride_n,
    bt_stride,
    num_tokens,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    M_FULL: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """K loop with the masks hoisted out of the full-tile paths.

    MUSA predicates are expensive: masking every A/B load costs 2-4x on the
    long-K shapes.  ``M_FULL`` and ``EVEN_K`` are uniform across the CTA (full
    M tile / split-aligned K), so the unmasked loop is selected once per
    program and the masked variant only runs for the last M tile or an
    unaligned K split.  The packed weight zero-pads the columns past
    ``mix_size``, so the B load never needs a column mask.
    """
    if M_FULL and EVEN_K:
        for k0 in range(k_begin, k_end, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :])
            b = tl.load(bt_ptr + offs_k[:, None] * bt_stride + offs_n[None, :])
            acc += tl.dot(a, b)
    elif M_FULL:
        for k0 in range(k_begin, k_end, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < k_end
            a = tl.load(
                a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
                mask=k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                bt_ptr + offs_k[:, None] * bt_stride + offs_n[None, :],
                mask=k_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(a, b)
    elif EVEN_K:
        m_mask = offs_m < num_tokens
        for k0 in range(k_begin, k_end, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(
                a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
                mask=m_mask[:, None],
                other=0.0,
            )
            b = tl.load(bt_ptr + offs_k[:, None] * bt_stride + offs_n[None, :])
            acc += tl.dot(a, b)
    else:
        m_mask = offs_m < num_tokens
        for k0 in range(k_begin, k_end, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < k_end
            a = tl.load(
                a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                bt_ptr + offs_k[:, None] * bt_stride + offs_n[None, :],
                mask=k_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(a, b)
    return acc


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_project_mthreads"),
    key=["num_tokens", "hc_hidden_size", "split_k"],
    warmup=5,
    rep=10,
)
@triton.jit
def _mhc_shifted_project_mthreads_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (hc_hidden_size, mix_size) bfloat16, contiguous, 32-col padded
    out_ptr,  # (split_k, num_tokens, mix_size) float32 partials
    num_tokens,
    hc_hidden_size,
    mix_size,
    a_stride_n,
    bt_stride,
    split_k,
    k_per_split,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Pure ``a @ fn^T`` with the K dimension split across CTAs."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    k_begin = pid_k * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, hc_hidden_size)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if (pid_m + 1) * BLOCK_M <= num_tokens:
        acc = _project_tiles(
            a_ptr,
            bt_ptr,
            acc,
            offs_m,
            offs_n,
            k_begin,
            k_end,
            a_stride_n,
            bt_stride,
            num_tokens,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
            M_FULL=True,
            EVEN_K=EVEN_K,
        )
        # The packed weight zero-pads the columns past mix_size, but the
        # partials buffer stops at mix_size: mask only the N columns.
        tl.store(
            out_ptr
            + (pid_k * num_tokens + offs_m[:, None]) * mix_size
            + offs_n[None, :],
            acc,
            mask=offs_n[None, :] < mix_size,
        )
    else:
        acc = _project_tiles(
            a_ptr,
            bt_ptr,
            acc,
            offs_m,
            offs_n,
            k_begin,
            k_end,
            a_stride_n,
            bt_stride,
            num_tokens,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
            M_FULL=False,
            EVEN_K=EVEN_K,
        )
        # Write the split-K partials; a separate fixed-order reduction makes
        # the result bitwise deterministic (the epsilon-wiring test compares
        # two calls with rtol=0, atol=0).
        tl.store(
            out_ptr
            + (pid_k * num_tokens + offs_m[:, None]) * mix_size
            + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < mix_size),
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_post_collapse_norm_mthreads"),
    key=["H", "num_tokens"],
    warmup=5,
    rep=10,
)
@triton.jit
def _mhc_shifted_post_collapse_norm_kernel(
    a_ptr,  # comb_res_mix : (N, 4, 4), float32 -- vLLM (j, i) orientation
    b_ptr,  # residual     : (N, 4, H), bfloat16
    c_ptr,  # post_layer_mix: (N, 4), float32
    d_ptr,  # x            : (N, H), bfloat16
    pre_ptr,  # pre_mix    : (N, 4), float32, or a dummy when USE_PRE is off
    out_ptr,  # residual_cur: (N, 4, H), bfloat16
    layer_ptr,  # layer_input: (N, H) bfloat16, normalised when HAS_W
    aux_ptr,  # aux         : (N, H), bfloat16, or a dummy when WRITE_AUX is off
    ssq_ptr,  # (N,) float32 row sum of squares of residual_cur
    w_ptr,  # norm weight: (H,) bfloat16, or a dummy when HAS_W is off
    a_stride_n,
    a_stride_j,
    a_stride_i,
    num_tokens,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
    HAS_W: tl.constexpr,
    EPS: tl.constexpr,
):
    """One CTA per token: post + collapse + sum of squares + optional norm.

    ``BLOCK_H`` covers the whole row (H <= 2048), so the row reductions are
    CTA-local, the sum of squares is a single value per token, and the norm
    reads the just-rounded bf16 values from registers instead of a second
    global pass.
    """
    pid_n = tl.program_id(0)
    h_off = tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    a_base = pid_n * a_stride_n
    c_base = pid_n * 4
    b_base = pid_n * 4 * H
    d_base = pid_n * H
    out_base = pid_n * 4 * H

    c0 = tl.load(c_ptr + c_base + 0).to(tl.float32)
    c1 = tl.load(c_ptr + c_base + 1).to(tl.float32)
    c2 = tl.load(c_ptr + c_base + 2).to(tl.float32)
    c3 = tl.load(c_ptr + c_base + 3).to(tl.float32)

    a00 = tl.load(a_ptr + a_base).to(tl.float32)
    a01 = tl.load(a_ptr + a_base + 1 * a_stride_i).to(tl.float32)
    a02 = tl.load(a_ptr + a_base + 2 * a_stride_i).to(tl.float32)
    a03 = tl.load(a_ptr + a_base + 3 * a_stride_i).to(tl.float32)
    a10 = tl.load(a_ptr + a_base + 1 * a_stride_j).to(tl.float32)
    a11 = tl.load(a_ptr + a_base + 1 * a_stride_j + 1 * a_stride_i).to(tl.float32)
    a12 = tl.load(a_ptr + a_base + 1 * a_stride_j + 2 * a_stride_i).to(tl.float32)
    a13 = tl.load(a_ptr + a_base + 1 * a_stride_j + 3 * a_stride_i).to(tl.float32)
    a20 = tl.load(a_ptr + a_base + 2 * a_stride_j).to(tl.float32)
    a21 = tl.load(a_ptr + a_base + 2 * a_stride_j + 1 * a_stride_i).to(tl.float32)
    a22 = tl.load(a_ptr + a_base + 2 * a_stride_j + 2 * a_stride_i).to(tl.float32)
    a23 = tl.load(a_ptr + a_base + 2 * a_stride_j + 3 * a_stride_i).to(tl.float32)
    a30 = tl.load(a_ptr + a_base + 3 * a_stride_j).to(tl.float32)
    a31 = tl.load(a_ptr + a_base + 3 * a_stride_j + 1 * a_stride_i).to(tl.float32)
    a32 = tl.load(a_ptr + a_base + 3 * a_stride_j + 2 * a_stride_i).to(tl.float32)
    a33 = tl.load(a_ptr + a_base + 3 * a_stride_j + 3 * a_stride_i).to(tl.float32)

    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b0 = tl.load(b_ptr + b_base + 0 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b1 = tl.load(b_ptr + b_base + 1 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b2 = tl.load(b_ptr + b_base + 2 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b3 = tl.load(b_ptr + b_base + 3 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)

    # Identical fma chain to mhc_post_kernel_hc_mult_4: the post output must
    # stay bitwise equal to mhc_post.
    acc0 = tl.fma(
        a30, b3, tl.fma(a20, b2, tl.fma(a10, b1, tl.fma(a00, b0, c0 * d_vals)))
    )
    acc1 = tl.fma(
        a31, b3, tl.fma(a21, b2, tl.fma(a11, b1, tl.fma(a01, b0, c1 * d_vals)))
    )
    acc2 = tl.fma(
        a32, b3, tl.fma(a22, b2, tl.fma(a12, b1, tl.fma(a02, b0, c2 * d_vals)))
    )
    acc3 = tl.fma(
        a33, b3, tl.fma(a23, b2, tl.fma(a13, b1, tl.fma(a03, b0, c3 * d_vals)))
    )

    v0 = acc0.to(tl.bfloat16)
    v1 = acc1.to(tl.bfloat16)
    v2 = acc2.to(tl.bfloat16)
    v3 = acc3.to(tl.bfloat16)
    tl.store(out_ptr + out_base + 0 * H + h_off, v0, mask=h_mask)
    tl.store(out_ptr + out_base + 1 * H + h_off, v1, mask=h_mask)
    tl.store(out_ptr + out_base + 2 * H + h_off, v2, mask=h_mask)
    tl.store(out_ptr + out_base + 3 * H + h_off, v3, mask=h_mask)

    # The generic collapse loads the materialised bf16 residual_cur; converting
    # the stored values back to fp32 keeps the same rounding boundary.
    f0 = v0.to(tl.float32)
    f1 = v1.to(tl.float32)
    f2 = v2.to(tl.float32)
    f3 = v3.to(tl.float32)
    if USE_PRE:
        p0 = tl.load(pre_ptr + pid_n * 4 + 0)
        p1 = tl.load(pre_ptr + pid_n * 4 + 1)
        p2 = tl.load(pre_ptr + pid_n * 4 + 2)
        p3 = tl.load(pre_ptr + pid_n * 4 + 3)
        col = f0 * p0
        col += f1 * p1
        col += f2 * p2
        col += f3 * p3
    else:
        col = f0
    col_bf = col.to(tl.bfloat16)
    if HAS_W:
        # Match the generic path's rounding boundary: the norm sees the bf16
        # collapsed value, and the pre-norm row is not an output, so only the
        # normalised value is stored.
        value = col_bf.to(tl.float32)
        layer_ssq = tl.sum(value * value, axis=0)
        inv = tl.rsqrt(layer_ssq / H + EPS)
        weight = tl.load(w_ptr + h_off, mask=h_mask, other=0.0).to(tl.float32)
        tl.store(
            layer_ptr + pid_n * H + h_off,
            (value * inv * weight).to(tl.bfloat16),
            mask=h_mask,
        )
    else:
        tl.store(layer_ptr + pid_n * H + h_off, col_bf, mask=h_mask)
    if WRITE_AUX:
        total = f0 + f1
        total += f2
        total += f3
        total = total / 4.0
        tl.store(aux_ptr + pid_n * H + h_off, total.to(tl.bfloat16), mask=h_mask)

    ssq = f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3
    tl.store(ssq_ptr + pid_n, tl.sum(ssq))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_post_collapse_norm_loop_mthreads"),
    key=["H", "num_tokens"],
    warmup=5,
    rep=10,
)
@triton.jit
def _mhc_shifted_post_collapse_norm_loop_kernel(
    a_ptr,  # comb_res_mix : (N, 4, 4), float32 -- vLLM (j, i) orientation
    b_ptr,  # residual     : (N, 4, H), bfloat16
    c_ptr,  # post_layer_mix: (N, 4), float32
    d_ptr,  # x            : (N, H), bfloat16
    pre_ptr,  # pre_mix    : (N, 4), float32, or a dummy when USE_PRE is off
    out_ptr,  # residual_cur: (N, 4, H), bfloat16
    layer_ptr,  # layer_input: (N, H) bfloat16, normalised when HAS_W
    aux_ptr,  # aux         : (N, H), bfloat16, or a dummy when WRITE_AUX is off
    ssq_ptr,  # (N,) float32 row sum of squares of residual_cur
    w_ptr,  # norm weight: (H,) bfloat16, or a dummy when HAS_W is off
    a_stride_n,
    a_stride_j,
    a_stride_i,
    num_tokens,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
    HAS_W: tl.constexpr,
    EPS: tl.constexpr,
):
    """One CTA per token, H streamed in tiles (wide rows).

    The flat kernel keeps the whole row live in registers; this variant keeps
    only ``BLOCK_H`` live and lets the kernel-level ``num_stages`` pipeline the
    next residual/x tile behind the current post/collapse work.  The norm needs
    the full row's sum of squares, so it runs a second pass over the stored
    bf16 ``layer_input``.
    """
    pid_n = tl.program_id(0)

    a_base = pid_n * a_stride_n
    c_base = pid_n * 4
    b_base = pid_n * 4 * H
    d_base = pid_n * H
    out_base = pid_n * 4 * H

    c0 = tl.load(c_ptr + c_base + 0).to(tl.float32)
    c1 = tl.load(c_ptr + c_base + 1).to(tl.float32)
    c2 = tl.load(c_ptr + c_base + 2).to(tl.float32)
    c3 = tl.load(c_ptr + c_base + 3).to(tl.float32)

    a00 = tl.load(a_ptr + a_base).to(tl.float32)
    a01 = tl.load(a_ptr + a_base + 1 * a_stride_i).to(tl.float32)
    a02 = tl.load(a_ptr + a_base + 2 * a_stride_i).to(tl.float32)
    a03 = tl.load(a_ptr + a_base + 3 * a_stride_i).to(tl.float32)
    a10 = tl.load(a_ptr + a_base + 1 * a_stride_j).to(tl.float32)
    a11 = tl.load(a_ptr + a_base + 1 * a_stride_j + 1 * a_stride_i).to(tl.float32)
    a12 = tl.load(a_ptr + a_base + 1 * a_stride_j + 2 * a_stride_i).to(tl.float32)
    a13 = tl.load(a_ptr + a_base + 1 * a_stride_j + 3 * a_stride_i).to(tl.float32)
    a20 = tl.load(a_ptr + a_base + 2 * a_stride_j).to(tl.float32)
    a21 = tl.load(a_ptr + a_base + 2 * a_stride_j + 1 * a_stride_i).to(tl.float32)
    a22 = tl.load(a_ptr + a_base + 2 * a_stride_j + 2 * a_stride_i).to(tl.float32)
    a23 = tl.load(a_ptr + a_base + 2 * a_stride_j + 3 * a_stride_i).to(tl.float32)
    a30 = tl.load(a_ptr + a_base + 3 * a_stride_j).to(tl.float32)
    a31 = tl.load(a_ptr + a_base + 3 * a_stride_j + 1 * a_stride_i).to(tl.float32)
    a32 = tl.load(a_ptr + a_base + 3 * a_stride_j + 2 * a_stride_i).to(tl.float32)
    a33 = tl.load(a_ptr + a_base + 3 * a_stride_j + 3 * a_stride_i).to(tl.float32)

    sq = 0.0
    layer_sq = 0.0
    for h0 in range(0, H, BLOCK_H):
        h_off = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_off < H

        d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
        b0 = tl.load(b_ptr + b_base + 0 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        b1 = tl.load(b_ptr + b_base + 1 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        b2 = tl.load(b_ptr + b_base + 2 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        b3 = tl.load(b_ptr + b_base + 3 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )

        acc0 = tl.fma(
            a30, b3, tl.fma(a20, b2, tl.fma(a10, b1, tl.fma(a00, b0, c0 * d_vals)))
        )
        acc1 = tl.fma(
            a31, b3, tl.fma(a21, b2, tl.fma(a11, b1, tl.fma(a01, b0, c1 * d_vals)))
        )
        acc2 = tl.fma(
            a32, b3, tl.fma(a22, b2, tl.fma(a12, b1, tl.fma(a02, b0, c2 * d_vals)))
        )
        acc3 = tl.fma(
            a33, b3, tl.fma(a23, b2, tl.fma(a13, b1, tl.fma(a03, b0, c3 * d_vals)))
        )

        v0 = acc0.to(tl.bfloat16)
        v1 = acc1.to(tl.bfloat16)
        v2 = acc2.to(tl.bfloat16)
        v3 = acc3.to(tl.bfloat16)
        tl.store(out_ptr + out_base + 0 * H + h_off, v0, mask=h_mask)
        tl.store(out_ptr + out_base + 1 * H + h_off, v1, mask=h_mask)
        tl.store(out_ptr + out_base + 2 * H + h_off, v2, mask=h_mask)
        tl.store(out_ptr + out_base + 3 * H + h_off, v3, mask=h_mask)

        f0 = v0.to(tl.float32)
        f1 = v1.to(tl.float32)
        f2 = v2.to(tl.float32)
        f3 = v3.to(tl.float32)
        if USE_PRE:
            p0 = tl.load(pre_ptr + pid_n * 4 + 0)
            p1 = tl.load(pre_ptr + pid_n * 4 + 1)
            p2 = tl.load(pre_ptr + pid_n * 4 + 2)
            p3 = tl.load(pre_ptr + pid_n * 4 + 3)
            col = f0 * p0
            col += f1 * p1
            col += f2 * p2
            col += f3 * p3
        else:
            col = f0
        col_bf = col.to(tl.bfloat16)
        # Keep layer_input in the default cache policy: the norm's second pass
        # reads it back immediately.
        tl.store(layer_ptr + pid_n * H + h_off, col_bf, mask=h_mask)
        if WRITE_AUX:
            total = f0 + f1
            total += f2
            total += f3
            total = total / 4.0
            tl.store(aux_ptr + pid_n * H + h_off, total.to(tl.bfloat16), mask=h_mask)

        ssq = f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3
        sq += tl.sum(ssq)
        if HAS_W:
            value = col_bf.to(tl.float32)
            layer_sq += tl.sum(value * value)

    tl.store(ssq_ptr + pid_n, sq)

    if HAS_W:
        # The first loop's layer_input stores must be visible to the second
        # loop's loads across the CTA's threads.
        tl.debug_barrier()
        inv = tl.rsqrt(layer_sq / H + EPS)
        for h0 in range(0, H, BLOCK_H):
            h_off = h0 + tl.arange(0, BLOCK_H)
            h_mask = h_off < H
            value = tl.load(layer_ptr + pid_n * H + h_off, mask=h_mask, other=0.0).to(
                tl.float32
            )
            weight = tl.load(w_ptr + h_off, mask=h_mask, other=0.0).to(tl.float32)
            tl.store(
                layer_ptr + pid_n * H + h_off,
                (value * inv * weight).to(tl.bfloat16),
                mask=h_mask,
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_reduce_coeff_mthreads"),
    key=["num_tokens"],
    warmup=5,
    rep=10,
)
@triton.jit
def _mhc_shifted_reduce_coeff_mthreads_kernel(
    part_ptr,  # (split_k, num_tokens, MIX_SIZE) float32
    ssq_ptr,  # (num_tokens,) float32
    scale_ptr,  # (3,) float32
    base_ptr,  # (MIX_SIZE,) float32
    pre_out_ptr,  # (num_tokens, HC) float32
    post_out_ptr,  # (num_tokens, HC) float32
    comb_out_ptr,  # (num_tokens, HC, HC) float32
    num_tokens,
    hc_hidden_size,
    split_k,
    RSUM_EPS: tl.constexpr,
    HC: tl.constexpr,
    MIX_SIZE: tl.constexpr,
    HC_PAD: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINK_REPEAT: tl.constexpr,
):
    """Reduce the split-K partials and build the three coefficient groups."""
    n = tl.program_id(0)
    idx = tl.arange(0, HC_PAD)
    valid = idx < HC
    comb_idx = idx[:, None] * HC + idx[None, :]
    comb_valid = valid[:, None] & valid[None, :]

    pre = tl.zeros((HC_PAD,), dtype=tl.float32)
    post = tl.zeros((HC_PAD,), dtype=tl.float32)
    comb = tl.zeros((HC_PAD, HC_PAD), dtype=tl.float32)
    for s in range(split_k):
        base = (s * num_tokens + n) * MIX_SIZE
        pre += tl.load(part_ptr + base + idx, mask=valid, other=0.0)
        post += tl.load(part_ptr + base + HC + idx, mask=valid, other=0.0)
        comb += tl.load(part_ptr + base + 2 * HC + comb_idx, mask=comb_valid, other=0.0)

    rms_inv = tl.rsqrt(tl.load(ssq_ptr + n) / hc_hidden_size + RSUM_EPS)

    scale_0 = tl.load(scale_ptr + 0)
    scale_1 = tl.load(scale_ptr + 1)
    scale_2 = tl.load(scale_ptr + 2)

    pre_logits = pre * rms_inv * scale_0
    pre_logits += tl.load(base_ptr + idx, mask=valid, other=0.0)
    tl.store(
        pre_out_ptr + n * HC + idx,
        tl.sigmoid(pre_logits) + PRE_EPS,
        mask=valid,
    )

    post_logits = post * rms_inv * scale_1
    post_logits += tl.load(base_ptr + HC + idx, mask=valid, other=0.0)
    tl.store(
        post_out_ptr + n * HC + idx,
        tl.sigmoid(post_logits) * POST_MULT,
        mask=valid,
    )

    comb = comb * rms_inv * scale_2
    comb += tl.load(base_ptr + 2 * HC + comb_idx, mask=comb_valid, other=0.0)
    comb = tl.where(comb_valid, comb, float("-inf"))
    comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None]
    comb = tl.where(comb_valid, comb + SINK_EPS, 0.0)
    comb = comb / (tl.sum(comb, axis=0)[None, :] + SINK_EPS)
    for _ in range(SINK_REPEAT - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + SINK_EPS)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + SINK_EPS)
    tl.store(
        comb_out_ptr + n * HC * HC + comb_idx,
        comb,
        mask=comb_valid,
    )


def _split_k_for(num_tokens, hc_hidden_size):
    """Split K until the project grid fills the 60 MTT S5000 SMs.

    The projection is a memory-bound skinny GEMM; one CTA walking the whole
    ``hc * H`` axis is latency bound, so the split targets a few waves of
    CTAs while keeping at least ``_MIN_K_PER_SPLIT`` elements per split.

    Returns ``(split_k, k_per_split, even_k)``.  When ``hc * H`` is a multiple
    of ``_ALIGN_K`` the split is chosen as a divisor of the K-block count, so
    every CTA walks full ``BLOCK_K`` tiles and the projection can use the
    unmasked load path.
    """
    est_m = max(1, triton.cdiv(num_tokens, 64))
    target = max(1, min(_PROJECT_MAX_SPLIT, triton.cdiv(_PROJECT_TARGET_CTAS, est_m)))
    if hc_hidden_size % _ALIGN_K == 0:
        units = hc_hidden_size // _ALIGN_K
        target = min(target, units)
        best, best_dist = 1, abs(1 - target)
        for split in range(2, units + 1):
            if units % split == 0:
                dist = abs(split - target)
                if dist < best_dist:
                    best, best_dist = split, dist
        return best, (units // best) * _ALIGN_K, True
    split = min(target, max(1, hc_hidden_size // _MIN_K_PER_SPLIT))
    return split, triton.cdiv(hc_hidden_size, split), False


def _launch_context():
    """Temporarily stop FlagTree aabs from rewriting the tuned tiles."""
    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    return aabs


def _restore_context(aabs):
    triton.knobs.autotuning.adjust_block_size = aabs


def _align_bits(tensors):
    bits = 0
    for i, tensor in enumerate(tensors):
        if tensor.data_ptr() % 16 == 0:
            bits |= 1 << i
    return bits


def _make_template(arg_names, num_args, kwargs, constexprs):
    """Map every kernel parameter to a call-time source.

    ``(0, i)`` reads positional arg ``i``, ``(1, name)`` reads ``kwargs[name]``
    and ``(2, value)`` is a constant captured at the first launch (the tuner's
    META and num_warps/num_stages are baked into the compiled kernel).
    """
    template = []
    for i, name in enumerate(arg_names):
        if i < num_args:
            template.append((0, i))
        elif name in kwargs:
            template.append((1, name))
        elif name in constexprs:
            template.append((2, constexprs[name]))
        else:
            return None
    return template


class _FastLauncher:
    """Launch the already-compiled kernel without LibEntry's per-call key work.

    LibEntry rebuilds a specialization key from every argument on each call
    (~57 us for the mHC kernels on MUSA, where tensor ``data_ptr`` and dict
    bookkeeping are expensive).  The kernels here are fixed-shape and their
    tuner config is stable per key, so after the first (tuning) launch the
    compiled kernel is called directly (~21 us).  The fallback is the normal
    LibEntry path whenever the template cannot be reconstructed.
    """

    def __init__(self, name, kernel):
        self.name = name
        self.kernel = kernel
        self.cache: dict = {}
        self.disabled = False

    def __call__(self, scalars, grid, args, kwargs):
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        key = (
            self.name,
            scalars,
            _align_bits(tensors),
            tuple(tensor.dtype for tensor in tensors),
            runtime.torch_device_fn.current_device(),
        )
        entry = self.cache.get(key)
        if entry is None:
            if self.disabled:
                self.kernel[grid](*args, **kwargs)
                return
            compiled, constexprs = self.kernel[grid](*args, **kwargs)
            template = _make_template(
                self.kernel.arg_names, len(args), kwargs, constexprs
            )
            if template is None:
                self.disabled = True
                return
            if callable(grid):
                meta = {
                    **dict(zip(self.kernel.arg_names, args)),
                    **kwargs,
                    **constexprs,
                }
                resolved = grid(meta)
            else:
                resolved = grid
            self.cache[key] = (compiled, template, tuple(resolved) + (1, 1))
            return
        compiled, template, full_grid = entry
        all_args = []
        for kind, value in template:
            if kind == 0:
                all_args.append(args[value])
            elif kind == 1:
                all_args.append(kwargs[value])
            else:
                all_args.append(value)
        compiled[full_grid](*all_args)


_FAST_POST_NORM = _FastLauncher("post_norm", _mhc_shifted_post_collapse_norm_kernel)
_FAST_POST_NORM_LOOP = _FastLauncher(
    "post_norm_loop", _mhc_shifted_post_collapse_norm_loop_kernel
)
_FAST_PROJECT = _FastLauncher("project", _mhc_shifted_project_mthreads_kernel)
_FAST_REDUCE_COEFF = _FastLauncher(
    "reduce_coeff", _mhc_shifted_reduce_coeff_mthreads_kernel
)


def mhc_shifted_post_collapse_norm(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    pre_mix,
    norm_weight,
    norm_eps,
    num_tokens,
    hidden_size,
    capture_aux,
    residual_cur=None,
    layer_input=None,
    aux=None,
    ssq=None,
):
    """Token-fused post + collapse + ssq + optional norm (H <= 2048)."""
    device = residual.device
    if residual_cur is None:
        residual_cur = torch.empty_like(residual)
    if layer_input is None:
        layer_input = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )
    if aux is None:
        aux = (
            torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
            if capture_aux
            else residual_cur.new_empty(0, hidden_size)
        )
    if ssq is None:
        ssq = torch.empty(num_tokens, dtype=torch.float32, device=device)
    block_h = triton.next_power_of_2(hidden_size)

    st = comb_res_mix.stride()
    a_sn = st[0]
    # vLLM orientation (MixLayout.TRANSPOSED): stored mix is [j, i].
    a_sj, a_si = st[1], st[2]

    args = (
        comb_res_mix,
        residual,
        post_layer_mix,
        x,
        pre_mix if pre_mix is not None else post_layer_mix,
        residual_cur,
        layer_input,
        aux if capture_aux else layer_input,
        ssq,
        norm_weight if norm_weight is not None else layer_input,
        a_sn,
        a_sj,
        a_si,
        num_tokens,
    )
    kwargs = dict(
        H=hidden_size,
        BLOCK_H=block_h,
        USE_PRE=pre_mix is not None,
        WRITE_AUX=capture_aux,
        HAS_W=norm_weight is not None,
        EPS=norm_eps,
    )
    _FAST_POST_NORM(
        (
            num_tokens,
            hidden_size,
            a_sn,
            a_sj,
            a_si,
            pre_mix is not None,
            capture_aux,
            norm_weight is not None,
            norm_eps,
        ),
        (num_tokens,),
        args,
        kwargs,
    )

    return residual_cur, layer_input, aux, ssq


def mhc_shifted_post_collapse_norm_loop(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    pre_mix,
    norm_weight,
    norm_eps,
    num_tokens,
    hidden_size,
    capture_aux,
    residual_cur=None,
    layer_input=None,
    aux=None,
    ssq=None,
):
    """Token-fused variant for wide rows; same outputs as the flat kernel."""
    device = residual.device
    if residual_cur is None:
        residual_cur = torch.empty_like(residual)
    if layer_input is None:
        layer_input = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )
    if aux is None:
        aux = (
            torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
            if capture_aux
            else residual_cur.new_empty(0, hidden_size)
        )
    if ssq is None:
        ssq = torch.empty(num_tokens, dtype=torch.float32, device=device)

    st = comb_res_mix.stride()
    a_sn = st[0]
    # vLLM orientation (MixLayout.TRANSPOSED): stored mix is [j, i].
    a_sj, a_si = st[1], st[2]

    args = (
        comb_res_mix,
        residual,
        post_layer_mix,
        x,
        pre_mix if pre_mix is not None else post_layer_mix,
        residual_cur,
        layer_input,
        aux if capture_aux else layer_input,
        ssq,
        norm_weight if norm_weight is not None else layer_input,
        a_sn,
        a_sj,
        a_si,
        num_tokens,
    )
    kwargs = dict(
        H=hidden_size,
        USE_PRE=pre_mix is not None,
        WRITE_AUX=capture_aux,
        HAS_W=norm_weight is not None,
        EPS=norm_eps,
    )
    _FAST_POST_NORM_LOOP(
        (
            num_tokens,
            hidden_size,
            a_sn,
            a_sj,
            a_si,
            pre_mix is not None,
            capture_aux,
            norm_weight is not None,
            norm_eps,
        ),
        (num_tokens,),
        args,
        kwargs,
    )

    return residual_cur, layer_input, aux, ssq


def mhc_shifted_project_mthreads(
    a_flat,
    fn_packed,
    num_tokens,
    mix_size,
    split_k=None,
    k_per_split=None,
    even_k=None,
    partials=None,
):
    """Return ``(split_k_partials, split_k)`` for ``a @ fn^T``."""
    hc_hidden_size = a_flat.shape[1]
    if split_k is None:
        split_k, k_per_split, even_k = _split_k_for(num_tokens, hc_hidden_size)
    if partials is None:
        partials = torch.empty(
            split_k, num_tokens, mix_size, dtype=torch.float32, device=a_flat.device
        )
    a_stride = a_flat.stride(0)
    fn_stride = fn_packed.stride(0)
    _FAST_PROJECT(
        (
            num_tokens,
            hc_hidden_size,
            split_k,
            k_per_split,
            even_k,
            mix_size,
            a_stride,
            fn_stride,
        ),
        lambda meta: (
            triton.cdiv(num_tokens, meta["BLOCK_M"]),
            triton.cdiv(mix_size, meta["BLOCK_N"]),
            split_k,
        ),
        (
            a_flat,
            fn_packed,
            partials,
            num_tokens,
            hc_hidden_size,
            mix_size,
            a_stride,
            fn_stride,
            split_k,
            k_per_split,
        ),
        dict(EVEN_K=even_k),
    )
    return partials, split_k


def mhc_shifted_reduce_coeff_mthreads(
    partials,
    ssq,
    split_k,
    hc_scale,
    hc_base,
    num_tokens,
    mix_size,
    hc,
    hidden_size,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    post_mix=None,
    comb_mix=None,
    next_pre=None,
):
    """Return ``(post_mix, comb_mix, next_pre)`` from one fused launch."""
    device = partials.device
    if post_mix is None:
        post_mix = torch.empty(num_tokens, hc, 1, dtype=torch.float32, device=device)
    if comb_mix is None:
        comb_mix = torch.empty(num_tokens, hc, hc, dtype=torch.float32, device=device)
    if next_pre is None:
        next_pre = torch.empty(num_tokens, hc, dtype=torch.float32, device=device)

    hc_hidden_size = hc * hidden_size
    hc_pad = triton.next_power_of_2(hc)
    _FAST_REDUCE_COEFF(
        (
            num_tokens,
            hc_hidden_size,
            split_k,
            hc,
            mix_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        ),
        (num_tokens,),
        (
            partials,
            ssq,
            hc_scale,
            hc_base,
            next_pre,
            post_mix,
            comb_mix,
            num_tokens,
            hc_hidden_size,
            split_k,
        ),
        dict(
            RSUM_EPS=rms_eps,
            HC=hc,
            MIX_SIZE=mix_size,
            HC_PAD=hc_pad,
            PRE_EPS=hc_pre_eps,
            SINK_EPS=hc_sinkhorn_eps,
            POST_MULT=hc_post_mult_value,
            SINK_REPEAT=sinkhorn_repeat,
        ),
    )
    return post_mix, comb_mix, next_pre


def mhc_shifted_post_pre_mthreads(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    pre_mix,
    norm_weight,
    norm_eps,
    capture_aux,
    num_tokens,
    hc,
    hidden_size,
    mix_size,
):
    """Full hc=4 boundary on MUSA, shared by every MThreads shape."""
    # Disable FlagTree aabs for the whole call so it cannot rewrite the tuned
    # tiles between the three launches.
    aabs = _launch_context()
    try:
        return _mhc_shifted_post_pre_staged(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix,
            norm_weight,
            norm_eps,
            capture_aux,
            num_tokens,
            hc,
            hidden_size,
            mix_size,
        )
    finally:
        _restore_context(aabs)


def _mhc_shifted_post_pre_staged(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    pre_mix,
    norm_weight,
    norm_eps,
    capture_aux,
    num_tokens,
    hc,
    hidden_size,
    mix_size,
):
    """Three-launch path for every MThreads shape.

    All outputs come from two backing allocations (one bfloat16, one float32):
    each ``torch.empty`` costs ~3 us on MUSA and the decode shapes are
    launch/allocation bound, so the eight separate allocations of the naive
    dataflow are collapsed into two contiguous buffers plus metadata views.
    """
    device = residual.device
    hc_hidden_size = hc * hidden_size
    split_k, k_per_split, even_k = _split_k_for(num_tokens, hc_hidden_size)

    n_res = num_tokens * hc_hidden_size
    n_layer = num_tokens * hidden_size
    n_aux = n_layer if capture_aux else 0
    out_bf = torch.empty(n_res + n_layer + n_aux, dtype=torch.bfloat16, device=device)
    residual_cur = out_bf[:n_res].view(num_tokens, hc, hidden_size)
    layer_input = out_bf[n_res : n_res + n_layer].view(num_tokens, hidden_size)
    aux = out_bf[n_res + n_layer :].view(
        0 if not capture_aux else num_tokens, hidden_size
    )

    n_ssq = num_tokens
    n_part = split_k * num_tokens * mix_size
    n_post = num_tokens * hc
    n_comb = num_tokens * hc * hc
    out_f32 = torch.empty(
        n_ssq + n_part + n_post + n_comb + n_post,
        dtype=torch.float32,
        device=device,
    )
    off = 0
    ssq = out_f32[off : off + n_ssq].view(num_tokens)
    off += n_ssq
    project_partials = out_f32[off : off + n_part].view(split_k, num_tokens, mix_size)
    off += n_part
    post_mix = out_f32[off : off + n_post].view(num_tokens, hc, 1)
    off += n_post
    comb_mix = out_f32[off : off + n_comb].view(num_tokens, hc, hc)
    off += n_comb
    next_pre = out_f32[off : off + n_post].view(num_tokens, hc)

    # (1)+(4) post, collapse, row sum of squares and (for narrow rows) norm.
    token_kernel = (
        mhc_shifted_post_collapse_norm
        if hidden_size <= _TOKEN_FLAT_MAX_HIDDEN
        else mhc_shifted_post_collapse_norm_loop
    )
    residual_cur, layer_input, aux, ssq = token_kernel(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        pre_mix,
        norm_weight,
        norm_eps,
        num_tokens,
        hidden_size,
        capture_aux,
        residual_cur=residual_cur,
        layer_input=layer_input,
        aux=aux,
        ssq=ssq,
    )

    # (2)+(3) packed split-K projection, then the fused reduce/coefficient
    # launch.  The projection is a pure GEMM over the packed fn weight, so it
    # can pipeline; the reduction order stays fixed.
    a_flat = residual_cur.view(num_tokens, hc_hidden_size)
    fn_packed = _fn_bf16_packed(fn)
    project_partials, split_k = mhc_shifted_project_mthreads(
        a_flat,
        fn_packed,
        num_tokens,
        mix_size,
        split_k,
        k_per_split,
        even_k,
        partials=project_partials,
    )
    post_mix, comb_mix, next_pre = mhc_shifted_reduce_coeff_mthreads(
        project_partials,
        ssq,
        split_k,
        hc_scale,
        hc_base,
        num_tokens,
        mix_size,
        hc,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        post_mix=post_mix,
        comb_mix=comb_mix,
        next_pre=next_pre,
    )

    return residual_cur, post_mix, comb_mix, layer_input, next_pre, aux


__all__ = ["mhc_shifted_post_pre_mthreads"]
