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

"""MetaX C550 specialization of the shifted mHC post -> pre boundary.

The generic path writes ``residual_cur`` in the post kernel and reads it back
in the collapse kernel, folds the row sum of squares into the pipelined project
K loop (which pins ``num_stages`` to 1), and spends six launches on the
boundary.  On C550 the decode-sized shapes are launch bound, so the launch
count is what decides the acceptance speedup.

This module keeps the post arithmetic verbatim (the post output has to stay
bitwise equal to ``mhc_post``) and changes the dataflow:

1. small batches (``num_tokens <= 512``) compute the post, the collapse, the
   row sum of squares and the RMSNorm in one one-CTA-per-token kernel: the flat
   kernel keeps the whole row in registers for ``H <= 2048``, the loop kernel
   streams wider rows in pipelined tiles and normalises in a second pass.
   Wide rows with at most 128 tokens leave that grid almost idle, so they take
   the tiled post grid instead and the norm runs inside the coefficient
   launch (the tile partial squares are folded there);
2. the project stage is a pure split-K GEMM.  C550 needs substantially more
   K splits than the old ``min(8, CUs / M tiles)`` heuristic: a single CTA
   walking a long K loop is latency bound, so the split targets roughly 512
   CTAs while keeping at least 256 K elements per split;
3. the split-K reduction, the sum-of-squares reduction and the Sinkhorn
   coefficients always collapse into one launch;
4. ``fn`` is packed once into a 32-column zero-padded bfloat16 operand, so
   every B row is 64-byte aligned and the project K loop needs no column
   mask.  The post kernels tune per ``(H, num_tokens)`` because the optimal
   warp count depends on the batch.

All reductions run in a fixed order, so repeated calls stay bitwise
deterministic.
"""

import weakref

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

_FN_PACK_CACHE: dict = {}

# Launch-bound decode shapes: one CTA per token keeps the boundary at a single
# launch.  Above this the tiled post kernel already exposes enough parallelism
# and the extra norm launch is amortised.
_TOKEN_FUSED_MAX_TOKENS = 512

# The flat token kernel keeps the whole row live in registers; wider rows use
# the H-streamed loop kernel.
_TOKEN_FLAT_MAX_HIDDEN = 2048

# Wide rows with very few tokens leave the one-CTA-per-token loop kernel with
# ~1.2 CTAs per C550 CU.  Below this token count they use the tiled post grid
# and fold the norm into the coefficient launch instead.
_TILED_POST_MAX_TOKENS = 128

# Smallest BLOCK_H in the tiled post tune space.  The per-token sum-of-squares
# scratch is sized for it, so every tuned tile partition fits one buffer.
_MIN_BLOCK_H = 128

_SSQ_BLOCK = 32


@triton.jit
def _pack_fn_bf16_kernel(
    src_ptr,  # (mix_size, K) float32, contiguous
    dst_ptr,  # (K, next_pow2(mix_size)) bfloat16, zero-padded
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
    # ``mix_size`` is hc*(hc+2) = 24 for hc=4; the padded 32-column stride
    # keeps every B row 64-byte aligned and lets the N tile skip masking.
    mix_stride = triton.next_power_of_2(mix_size)
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_project_metax"),
    key=["num_tokens", "hc_hidden_size", "split_k"],
    warmup=5,
    rep=10,
)
@triton.jit
def _mhc_shifted_project_metax_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (hc_hidden_size, mix_size) bfloat16, contiguous
    out_ptr,  # (split_k, num_tokens, mix_size) float32 partials
    num_tokens,
    hc_hidden_size,
    mix_size,
    a_stride_n,
    bt_stride,
    split_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Pure ``a @ fn^T`` with the K dimension optionally split across CTAs."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < num_tokens
    n_mask = offs_n < mix_size

    k_per_split = tl.cdiv(hc_hidden_size, split_k)
    k_begin = pid_k * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, hc_hidden_size)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_end
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # The packed weight zero-pads the columns past mix_size, so the B
        # load needs no column mask (the padded dot columns are exactly 0).
        b = tl.load(
            bt_ptr + offs_k[:, None] * bt_stride + offs_n[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(a, b)

    # Write the split-K partials; a separate fixed-order reduction makes the
    # result bitwise deterministic (the epsilon-wiring test compares two calls
    # with rtol=0, atol=0).
    tl.store(
        out_ptr + (pid_k * num_tokens + offs_m[:, None]) * mix_size + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_post_collapse"),
    key=["H", "WRITE_LSSQ"],
    warmup=5,
    rep=10,
)
@triton.jit
def _mhc_shifted_post_collapse_kernel(
    a_ptr,  # comb_res_mix : (N, 4, 4), float32 -- vLLM (j, i) orientation
    b_ptr,  # residual     : (N, 4, H), bfloat16
    c_ptr,  # post_layer_mix: (N, 4), float32
    d_ptr,  # x            : (N, H), bfloat16
    pre_ptr,  # pre_mix    : (N, 4), float32, or a dummy when USE_PRE is off
    out_ptr,  # residual_cur: (N, 4, H), bfloat16
    layer_ptr,  # layer_input: (N, H), bfloat16
    aux_ptr,  # aux         : (N, H), bfloat16, or a dummy when WRITE_AUX is off
    part_ptr,  # (N, max_tiles) float32, zero-initialised partial sums
    lpart_ptr,  # (N, max_tiles) float32 layer-input partial squares, or dummy
    a_stride_n,
    a_stride_j,
    a_stride_i,
    max_tiles,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
    WRITE_LSSQ: tl.constexpr,
):
    """Fused hc=4 post + collapse + row sum of squares (H-tiled)."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
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
    tl.store(layer_ptr + pid_n * H + h_off, col_bf, mask=h_mask)
    if WRITE_LSSQ:
        # Partial sum of squares of the stored bf16 layer_input; the
        # coefficient kernel folds the tiles and applies the norm.
        lval = col_bf.to(tl.float32)
        tl.store(lpart_ptr + pid_n * max_tiles + pid_h, tl.sum(lval * lval))
    if WRITE_AUX:
        total = f0 + f1
        total += f2
        total += f3
        total = total / 4.0
        tl.store(aux_ptr + pid_n * H + h_off, total.to(tl.bfloat16), mask=h_mask)

    # Per-(token, h-tile) row sum of squares; a separate fixed-order reduction
    # combines the tiles deterministically.
    ssq = f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3
    tl.store(part_ptr + pid_n * max_tiles + pid_h, tl.sum(ssq))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_post_collapse_norm_metax"),
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
    configs=runtime.get_tuned_config("mhc_shifted_post_collapse_norm_loop_metax"),
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
    bf16 ``layer_input`` (one extra ``NH`` read) and normalises in place.
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
    configs=runtime.get_tuned_config("mhc_shifted_reduce_coeff_metax"),
    key=["num_tokens", "hidden_size", "HAS_NORM"],
    warmup=5,
    rep=10,
)
@triton.jit
def _mhc_shifted_reduce_coeff_kernel(
    part_ptr,  # (split_k, num_tokens, MIX_SIZE) float32
    ssq_ptr,  # (num_tokens, ssq_stride) float32 single value or tile partials
    lpart_ptr,  # (num_tokens, lssq_stride) float32 layer tile partials, or dummy
    layer_ptr,  # (num_tokens, hidden_size) bfloat16, the pre-norm collapse
    lnorm_ptr,  # (num_tokens, hidden_size) bfloat16, the normalised output
    w_ptr,  # (hidden_size,) bfloat16 norm weight, or a dummy
    scale_ptr,  # (3,) float32
    base_ptr,  # (MIX_SIZE,) float32
    pre_out_ptr,  # (num_tokens, HC) float32
    post_out_ptr,  # (num_tokens, HC) float32
    comb_out_ptr,  # (num_tokens, HC, HC) float32
    num_tokens,
    hc_hidden_size,
    hidden_size,
    ssq_stride,
    ssq_tiles,
    lssq_stride,
    lssq_tiles,
    split_k,
    RSUM_EPS: tl.constexpr,
    HC: tl.constexpr,
    MIX_SIZE: tl.constexpr,
    HC_PAD: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINK_REPEAT: tl.constexpr,
    SSQ_BLOCK: tl.constexpr,
    NORM_BLOCK: tl.constexpr,
    HAS_NORM: tl.constexpr,
    EPS: tl.constexpr,
):
    """Reduce the project partials and the sum-of-squares tiles, then build the
    three coefficient groups -- all in one launch.

    The whole boundary otherwise ends with three tiny launches (split-K
    reduce, ssq reduce, coefficients) that dominate the decode-sized shapes.
    Every reduction keeps the fixed split / tile order of the separate
    kernels, so repeated calls stay bitwise deterministic.
    """
    n = tl.program_id(0)
    idx = tl.arange(0, HC_PAD)
    valid = idx < HC
    comb_idx = idx[:, None] * HC + idx[None, :]
    comb_valid = valid[:, None] & valid[None, :]

    ssq = 0.0
    for t0 in range(0, ssq_tiles, SSQ_BLOCK):
        ti = t0 + tl.arange(0, SSQ_BLOCK)
        ssq += tl.sum(
            tl.load(
                ssq_ptr + n * ssq_stride + ti,
                mask=ti < ssq_tiles,
                other=0.0,
            ),
            axis=0,
        )

    pre = tl.zeros((HC_PAD,), dtype=tl.float32)
    post = tl.zeros((HC_PAD,), dtype=tl.float32)
    comb = tl.zeros((HC_PAD, HC_PAD), dtype=tl.float32)
    for s in range(split_k):
        base = (s * num_tokens + n) * MIX_SIZE
        pre += tl.load(part_ptr + base + idx, mask=valid, other=0.0)
        post += tl.load(part_ptr + base + HC + idx, mask=valid, other=0.0)
        comb += tl.load(part_ptr + base + 2 * HC + comb_idx, mask=comb_valid, other=0.0)

    rms_inv = tl.rsqrt(ssq / hc_hidden_size + RSUM_EPS)

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

    if HAS_NORM:
        # Fold the layer_input tile partial squares and normalise the row in
        # place, so the tiled-post path does not need a separate norm launch.
        lssq = 0.0
        for t0 in range(0, lssq_tiles, SSQ_BLOCK):
            ti = t0 + tl.arange(0, SSQ_BLOCK)
            lssq += tl.sum(
                tl.load(
                    lpart_ptr + n * lssq_stride + ti,
                    mask=ti < lssq_tiles,
                    other=0.0,
                ),
                axis=0,
            )
        inv = tl.rsqrt(lssq / hidden_size + EPS)
        for h0 in range(0, hidden_size, NORM_BLOCK):
            offs = h0 + tl.arange(0, NORM_BLOCK)
            h_mask = offs < hidden_size
            value = tl.load(
                layer_ptr + n * hidden_size + offs, mask=h_mask, other=0.0
            ).to(tl.float32)
            weight = tl.load(w_ptr + offs, mask=h_mask, other=0.0).to(tl.float32)
            tl.store(
                lnorm_ptr + n * hidden_size + offs,
                (value * inv * weight).to(tl.bfloat16),
                mask=h_mask,
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_norm_metax"),
    key=["D"],
    warmup=5,
    rep=10,
)
@triton.jit
def _mhc_shifted_norm_metax_kernel(
    X,  # (num_tokens, D) bfloat16
    W,  # (D,) bfloat16
    Y,  # (num_tokens, D) bfloat16
    D,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Single-pass RMSNorm: load the row once, reduce, normalise and store."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    value = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ssq = tl.sum(value * value, axis=0)
    inv = tl.rsqrt(ssq / D + EPS)
    result = value * inv * tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * D + offs, result.to(tl.bfloat16), mask=mask)


def mhc_shifted_norm_metax(x, weight, eps):
    """One-pass RMSNorm when the row fits in one CTA (D <= 8192)."""
    D = x.shape[1]
    out = torch.empty_like(x)
    block_d = triton.next_power_of_2(D)
    # BLOCK_D must cover the whole row; aabs must not rewrite it to a divisor.
    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    try:
        _mhc_shifted_norm_metax_kernel[(x.shape[0],)](
            x, weight, out, D, EPS=eps, BLOCK_D=block_d
        )
    finally:
        triton.knobs.autotuning.adjust_block_size = aabs
    return out


def _split_k_for(num_tokens, hc_hidden_size):
    """Split K until the project grid hides the C550 memory latency.

    A CTA walking a long K loop is latency bound on C550 (one CTA over the
    full 20K-28K axis costs 60-90 us), so the split is sized for roughly 512
    CTAs instead of one wave of 104 CUs.  At least 256 K elements stay in
    each split, and tiny batches cap at 16 (more splits only add reduce work
    once the grid already covers the machine).
    """
    est_m = max(1, triton.cdiv(num_tokens, 64))
    split = max(1, min(32, triton.cdiv(512, est_m)))
    if num_tokens <= 16:
        split = min(split, 16)
    return min(split, max(1, hc_hidden_size // 256))


def _launch_context():
    """Temporarily stop FlagTree aabs from rewriting the tuned tiles."""
    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    return aabs


def _restore_context(aabs):
    triton.knobs.autotuning.adjust_block_size = aabs


_DUMMY_CACHE: dict = {}


def _dummy_fp32(device):
    """One-element scratch for a disabled optional output.

    The tuner's ``reset_to_zero`` hook zeroes the named pointer arguments;
    a shared one-element buffer keeps that harmless when the buffer is off.
    """
    buf = _DUMMY_CACHE.get(device)
    if buf is None:
        buf = torch.zeros(1, dtype=torch.float32, device=device)
        _DUMMY_CACHE[device] = buf
    return buf


def mhc_shifted_post_collapse(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    pre_mix,
    num_tokens,
    hidden_size,
    capture_aux,
    write_lssq=False,
):
    """Tiled fused post + collapse + ssq; return the scratch ssq partials.

    ``write_lssq`` additionally stores the layer_input tile partial squares so
    the coefficient kernel can fold them and normalise the row.  The tiled
    grid gives wide rows the parallelism the one-CTA-per-token loop kernel
    lacks at small token counts.
    """
    device = residual.device
    residual_cur = torch.empty_like(residual)
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    aux = (
        torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        if capture_aux
        else residual_cur.new_empty(0, hidden_size)
    )
    max_tiles = triton.cdiv(hidden_size, _MIN_BLOCK_H)
    # No zeroing memset: the grid closure records the BLOCK_H the tuner
    # actually launched (on the tuning run and on cached launches alike), so
    # the reductions only read the slots this launch wrote.
    partials = torch.empty(num_tokens, max_tiles, dtype=torch.float32, device=device)
    lpartials = (
        torch.empty(num_tokens, max_tiles, dtype=torch.float32, device=device)
        if write_lssq
        else _dummy_fp32(device)
    )
    selected: dict = {}

    st = comb_res_mix.stride()
    a_sn = st[0]
    # vLLM orientation (MixLayout.TRANSPOSED): stored mix is [j, i].
    a_sj, a_si = st[1], st[2]

    def grid(meta):
        selected["block_h"] = meta["BLOCK_H"]
        return (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_H"]))

    aabs = _launch_context()
    try:
        _mhc_shifted_post_collapse_kernel[grid](
            comb_res_mix,
            residual,
            post_layer_mix,
            x,
            pre_mix if pre_mix is not None else post_layer_mix,
            residual_cur,
            layer_input,
            aux if capture_aux else layer_input,
            partials,
            lpartials,
            a_sn,
            a_sj,
            a_si,
            max_tiles,
            H=hidden_size,
            USE_PRE=pre_mix is not None,
            WRITE_AUX=capture_aux,
            WRITE_LSSQ=write_lssq,
        )
    finally:
        _restore_context(aabs)

    block_h = selected.get("block_h")
    if block_h is None:
        # Defensive fallback: without the launched tile every slot must be
        # valid, so zero the scratch and read the widest tile count.
        partials.zero_()
        if write_lssq:
            lpartials.zero_()
        tiles = max_tiles
    else:
        tiles = triton.cdiv(hidden_size, block_h)
    return residual_cur, layer_input, aux, partials, max_tiles, tiles, lpartials


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
):
    """Token-fused post + collapse + ssq + optional norm (H <= 2048)."""
    device = residual.device
    residual_cur = torch.empty_like(residual)
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    aux = (
        torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        if capture_aux
        else residual_cur.new_empty(0, hidden_size)
    )
    ssq = torch.empty(num_tokens, dtype=torch.float32, device=device)
    block_h = triton.next_power_of_2(hidden_size)

    st = comb_res_mix.stride()
    a_sn = st[0]
    # vLLM orientation (MixLayout.TRANSPOSED): stored mix is [j, i].
    a_sj, a_si = st[1], st[2]

    aabs = _launch_context()
    try:
        _mhc_shifted_post_collapse_norm_kernel[(num_tokens,)](
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
            H=hidden_size,
            BLOCK_H=block_h,
            USE_PRE=pre_mix is not None,
            WRITE_AUX=capture_aux,
            HAS_W=norm_weight is not None,
            EPS=norm_eps,
        )
    finally:
        _restore_context(aabs)

    return residual_cur, layer_input, aux, ssq, 1, 1


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
):
    """Token-fused variant for wide rows; same outputs as the flat kernel."""
    device = residual.device
    residual_cur = torch.empty_like(residual)
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    aux = (
        torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        if capture_aux
        else residual_cur.new_empty(0, hidden_size)
    )
    ssq = torch.empty(num_tokens, dtype=torch.float32, device=device)

    st = comb_res_mix.stride()
    a_sn = st[0]
    # vLLM orientation (MixLayout.TRANSPOSED): stored mix is [j, i].
    a_sj, a_si = st[1], st[2]

    aabs = _launch_context()
    try:
        _mhc_shifted_post_collapse_norm_loop_kernel[(num_tokens,)](
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
            H=hidden_size,
            USE_PRE=pre_mix is not None,
            WRITE_AUX=capture_aux,
            HAS_W=norm_weight is not None,
            EPS=norm_eps,
        )
    finally:
        _restore_context(aabs)

    return residual_cur, layer_input, aux, ssq, 1, 1


def mhc_shifted_project_partials_metax(a_flat, fn_packed, num_tokens, mix_size):
    """Return ``(split_k_partials, split_k)`` for ``a @ fn^T``."""
    split_k = _split_k_for(num_tokens, a_flat.shape[1])
    partials = torch.empty(
        split_k, num_tokens, mix_size, dtype=torch.float32, device=a_flat.device
    )
    # See mhc_shifted_post_collapse: keep aabs from rewriting the tuned tiles.
    aabs = _launch_context()
    try:
        _mhc_shifted_project_metax_kernel[
            lambda meta: (
                triton.cdiv(num_tokens, meta["BLOCK_M"]),
                triton.cdiv(mix_size, meta["BLOCK_N"]),
                split_k,
            )
        ](
            a_flat,
            fn_packed,
            partials,
            num_tokens,
            a_flat.shape[1],
            mix_size,
            a_flat.stride(0),
            fn_packed.stride(0),
            split_k,
        )
    finally:
        _restore_context(aabs)
    return partials, split_k


def mhc_shifted_reduce_coeff_metax(
    partials,
    ssq,
    ssq_stride,
    ssq_tiles,
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
    layer_input=None,
    layer_partials=None,
    norm_weight=None,
    norm_eps=1e-6,
):
    """Return ``(post_mix, comb_mix, next_pre)`` from one fused launch.

    When ``layer_input`` and ``norm_weight`` are given, the same launch also
    folds the layer_input tile partial squares and writes the normalised row
    to a separate output (the tiled-post path has no fused norm of its own).
    The output is not in place: the tuner re-runs the kernel against the same
    input, and normalising twice would not be idempotent.
    """
    device = partials.device
    post_mix = torch.empty(num_tokens, hc, 1, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc, hc, dtype=torch.float32, device=device)
    next_pre = torch.empty(num_tokens, hc, dtype=torch.float32, device=device)
    has_norm = layer_input is not None and norm_weight is not None
    lpart = layer_partials if has_norm else _dummy_fp32(device)
    layer = layer_input if has_norm else ssq
    normed = (
        torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        if has_norm
        else None
    )
    weight = norm_weight if has_norm else ssq

    aabs = _launch_context()
    try:
        _mhc_shifted_reduce_coeff_kernel[(num_tokens,)](
            partials,
            ssq,
            lpart,
            layer,
            normed if has_norm else layer,
            weight,
            hc_scale,
            hc_base,
            next_pre,
            post_mix,
            comb_mix,
            num_tokens,
            hc * hidden_size,
            hidden_size,
            ssq_stride,
            ssq_tiles,
            ssq_stride if has_norm else 0,
            ssq_tiles if has_norm else 0,
            split_k,
            RSUM_EPS=rms_eps,
            HC=hc,
            MIX_SIZE=mix_size,
            HC_PAD=triton.next_power_of_2(hc),
            PRE_EPS=hc_pre_eps,
            SINK_EPS=hc_sinkhorn_eps,
            POST_MULT=hc_post_mult_value,
            SINK_REPEAT=sinkhorn_repeat,
            SSQ_BLOCK=_SSQ_BLOCK,
            NORM_BLOCK=1024,
            HAS_NORM=has_norm,
            EPS=norm_eps,
        )
    finally:
        _restore_context(aabs)
    return post_mix, comb_mix, next_pre, normed


def mhc_shifted_post_pre_metax(
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
    """Full hc=4 boundary on C550, shared by every MetaX shape."""
    device = residual.device

    # (1)+(4) post, collapse, row sum of squares and (for small batches) norm.
    # Wide rows with a small token count are the one place the one-CTA-per-
    # token loop kernel loses to the tiled grid (only ~1.2 CTAs per C550 CU),
    # so they take the tiled post and fold the norm into the coefficient
    # launch instead.
    use_token_fused = num_tokens <= _TOKEN_FUSED_MAX_TOKENS
    use_tiled_post = (not use_token_fused) or (
        hidden_size > _TOKEN_FLAT_MAX_HIDDEN and num_tokens <= _TILED_POST_MAX_TOKENS
    )
    norm_in_reduce = (
        use_tiled_post
        and norm_weight is not None
        and hidden_size > _TOKEN_FLAT_MAX_HIDDEN
        and num_tokens <= _TILED_POST_MAX_TOKENS
    )
    if use_tiled_post:
        (
            residual_cur,
            layer_input,
            aux,
            ssq,
            ssq_stride,
            ssq_tiles,
            layer_partials,
        ) = mhc_shifted_post_collapse(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            pre_mix,
            num_tokens,
            hidden_size,
            capture_aux,
            write_lssq=norm_in_reduce,
        )
    else:
        token_kernel = (
            mhc_shifted_post_collapse_norm
            if hidden_size <= _TOKEN_FLAT_MAX_HIDDEN
            else mhc_shifted_post_collapse_norm_loop
        )
        residual_cur, layer_input, aux, ssq, ssq_stride, ssq_tiles = token_kernel(
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
        )
        layer_partials = None

    # (2)+(3) projection and coefficients.  A full-K project per M tile was
    # measured on C550 and loses badly (one CTA over the whole K axis is
    # latency bound), so the split-K project is used for every shape.  The
    # reduce and coefficient stages share one launch.
    a_flat = residual_cur.view(num_tokens, hc * hidden_size)
    fn_packed = _fn_bf16_packed(fn)
    project_partials, split_k = mhc_shifted_project_partials_metax(
        a_flat, fn_packed, num_tokens, mix_size
    )
    post_mix, comb_mix, next_pre, normed = mhc_shifted_reduce_coeff_metax(
        project_partials,
        ssq,
        ssq_stride,
        ssq_tiles,
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
        layer_input=layer_input if norm_in_reduce else None,
        layer_partials=layer_partials,
        norm_weight=norm_weight if norm_in_reduce else None,
        norm_eps=norm_eps,
    )
    if norm_in_reduce:
        layer_input = normed

    # (5) optional RMSNorm of the collapsed input.  The token-fused path has
    # already applied it; the tiled path keeps the separate norm kernel.
    if norm_weight is not None and not use_token_fused and not norm_in_reduce:
        from flaggems_vllm.ops.mhc.mhc_shifted_post_pre import _mhc_shifted_norm_kernel

        if 4096 <= hidden_size <= 8192:
            layer_input = mhc_shifted_norm_metax(layer_input, norm_weight, norm_eps)
        else:
            normed = torch.empty(
                num_tokens, hidden_size, dtype=torch.bfloat16, device=device
            )
            _mhc_shifted_norm_kernel[(num_tokens,)](
                layer_input,
                norm_weight,
                normed,
                hidden_size,
                EPS=norm_eps,
                HAS_W=True,
            )
            layer_input = normed

    return residual_cur, post_mix, comb_mix, layer_input, next_pre, aux
