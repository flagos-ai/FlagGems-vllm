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

"""PPU (_thead) specialization of the shifted mHC post -> pre boundary.

The generic path writes ``residual_cur`` in the post kernel and reads it back
in the collapse kernel, and it folds the row sum of squares into the project
K loop.  On the PPU that costs two extra HBM round trips for large batches and
serialises the skinny projection for decode-sized ones (one CTA walks the whole
``hc * H`` reduction axis at ``num_stages=1``).

This module changes the dataflow while keeping the post arithmetic verbatim:

1. one fused kernel computes the post values, the collapse, the stream mean and
   the per-tile sum of squares while the just-rounded bf16 post values are still
   in registers;
2. the project stage is a pure split-K GEMM over a packed contiguous ``fn``
   weight, so it can pipeline at ``num_stages >= 2`` and fill the PPU's 64 SMs
   even for a single token;
3. one per-token kernel reduces the split-K partials, the sum-of-squares tiles
   and computes the Sinkhorn coefficients;
4. a one-pass RMSNorm replaces the generic two-pass kernel when the row fits a
   single CTA.

Every reduction runs in a fixed order, so repeated calls stay bitwise
deterministic (the tests compare two calls with ``rtol=0, atol=0``).

This module is intentionally *not* re-exported from ``_thead.ops.__init__``:
``SpecOpRegistrar`` would then replace the public operator with this private
launcher.  ``flaggems_vllm.ops.mhc.mhc_shifted_post_pre`` imports it lazily.
"""

import weakref

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

# Smallest BLOCK_H in the tuned space.  The per-token sum-of-squares scratch is
# sized for it, so every tuned tile partition fits the same buffer.
_MIN_BLOCK_H = 128

# PPU-ZW810E has 64 SMs.  Split-K aims to fill them without shattering each
# split below one useful K tile.
_TARGET_CTAS = 64

_FN_PACK_CACHE: dict = {}


@triton.jit
def _pack_fn_bf16_kernel(
    src_ptr,  # (mix_size, K) float32, contiguous
    dst_ptr,  # (K, mix_stride) bfloat16, zero-padded columns
    numel,
    K,
    mix_size,
    mix_stride,
    BLOCK: tl.constexpr,
):
    """One-time ``fn`` transpose-cast for the project kernel's B operand."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    n = offs // K
    k = offs % K
    value = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + k * mix_stride + n, value.to(tl.bfloat16), mask=mask)


def _fn_bf16_packed(fn: torch.Tensor) -> torch.Tensor:
    """``fn`` as a zero-padded ``(hc*H, mix_stride)`` bfloat16 tensor, cached.

    ``mix_stride`` is the next power of two above ``mix_size``.  The PPU B
    loads then start on a 64-byte boundary for ``BLOCK_N=32`` instead of the
    24-column (48-byte) stride, and the dot's N tile needs no column mask.

    ``_version`` is skipped for inference tensors: they do not track one and
    reading it raises.  Inference tensors cannot be edited in place either, so
    the cache stays valid.
    """
    key = id(fn)
    version = None if fn.is_inference() else fn._version
    ptr = fn.data_ptr()
    entry = _FN_PACK_CACHE.get(key)
    if entry is not None:
        ref, cached_version, cached_ptr, cached = entry
        if ref() is fn and cached_version == version and cached_ptr == ptr:
            return cached

    mix_size, K = fn.shape
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
    configs=runtime.get_tuned_config("mhc_shifted_post_collapse"),
    key=["H"],
    # The tuner benchmarks configs with different BLOCK_H, hence different
    # tile counts, against the same partials buffer.  A short config must not
    # leave a long config's slots behind for the sum-of-squares reduction.
    reset_to_zero=["part_ptr"],
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
    a_stride_n,
    a_stride_j,
    a_stride_i,
    max_tiles,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
):
    """Fused hc=4 post + collapse + row sum of squares."""
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

    d_vals = tl.load(
        d_ptr + d_base + h_off,
        mask=h_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    b0 = tl.load(
        b_ptr + b_base + 0 * H + h_off,
        mask=h_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    b1 = tl.load(
        b_ptr + b_base + 1 * H + h_off,
        mask=h_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    b2 = tl.load(
        b_ptr + b_base + 2 * H + h_off,
        mask=h_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    b3 = tl.load(
        b_ptr + b_base + 3 * H + h_off,
        mask=h_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

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

    # The generic collapse reads the materialised bf16 residual_cur; converting
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
    tl.store(layer_ptr + pid_n * H + h_off, col.to(tl.bfloat16), mask=h_mask)
    if WRITE_AUX:
        total = f0 + f1
        total += f2
        total += f3
        total = total / 4.0
        tl.store(aux_ptr + pid_n * H + h_off, total.to(tl.bfloat16), mask=h_mask)

    # Per-(token, h-tile) row sum of squares; the coefficient kernel combines
    # the tiles in a fixed order.
    ssq = f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3
    tl.store(part_ptr + pid_n * max_tiles + pid_h, tl.sum(ssq))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_post_collapse_norm_thead"),
    key=["H"],
)
@triton.jit
def _mhc_shifted_post_collapse_norm_kernel(
    a_ptr,  # comb_res_mix : (N, 4, 4), float32 -- vLLM (j, i) orientation
    b_ptr,  # residual     : (N, 4, H), bfloat16
    c_ptr,  # post_layer_mix: (N, 4), float32
    d_ptr,  # x            : (N, H), bfloat16
    pre_ptr,  # pre_mix    : (N, 4), float32, or a dummy when USE_PRE is off
    out_ptr,  # residual_cur: (N, 4, H), bfloat16
    layer_ptr,  # layer_input: (N, H), bfloat16
    aux_ptr,  # aux         : (N, H), bfloat16, or a dummy when WRITE_AUX is off
    ssq_ptr,  # (N,) float32, row sum of squares of residual_cur
    w_ptr,  # norm weight: (H,) bfloat16, or a dummy when HAS_W is off
    norm_ptr,  # normed layer_input: (N, H), bfloat16, or a dummy
    a_stride_n,
    a_stride_j,
    a_stride_i,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
    HAS_W: tl.constexpr,
    EPS: tl.constexpr,
):
    """One CTA per token: post + collapse + sum of squares + optional norm.

    ``BLOCK_H`` covers the whole row (H <= 2048), so the row reductions are
    CTA-local and the kernel writes a single sum-of-squares value per token.
    This is the launch-bound path for decode-sized batches.
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
    if WRITE_AUX:
        total = f0 + f1
        total += f2
        total += f3
        total = total / 4.0
        tl.store(aux_ptr + pid_n * H + h_off, total.to(tl.bfloat16), mask=h_mask)

    ssq = f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3
    tl.store(ssq_ptr + pid_n, tl.sum(ssq))

    if HAS_W:
        # Match the generic path's rounding boundary: the norm reads the bf16
        # layer_input that was just stored.
        value = col_bf.to(tl.float32)
        layer_ssq = tl.sum(value * value, axis=0)
        inv = tl.rsqrt(layer_ssq / H + EPS)
        weight = tl.load(w_ptr + h_off, mask=h_mask, other=0.0).to(tl.float32)
        tl.store(
            norm_ptr + pid_n * H + h_off,
            (value * inv * weight).to(tl.bfloat16),
            mask=h_mask,
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_project_thead"),
    key=["num_tokens", "hc_hidden_size", "split_k"],
)
@triton.jit
def _mhc_shifted_project_thead_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (hc_hidden_size, mix_stride) bfloat16, zero-padded
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
        b = tl.load(
            bt_ptr + offs_k[:, None] * bt_stride + offs_n[None, :],
            mask=k_mask[:, None] & (offs_n[None, :] < bt_stride),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    # A separate fixed-order reduction combines the splits, so the result stays
    # bitwise deterministic even though the project stage is parallel over K.
    tl.store(
        out_ptr + (pid_k * num_tokens + offs_m[:, None]) * mix_size + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_coeff_thead"),
    key=["num_tokens"],
)
@triton.jit
def _mhc_shifted_reduce_coeff_thead_kernel(
    part_ptr,  # (split_k, num_tokens, MIX_SIZE) float32
    ssq_part_ptr,  # (num_tokens, max_tiles) float32
    scale_ptr,  # (3,) float32
    base_ptr,  # (MIX_SIZE,) float32
    pre_out_ptr,  # (num_tokens, HC) float32
    post_out_ptr,  # (num_tokens, HC) float32
    comb_out_ptr,  # (num_tokens, HC, HC) float32
    num_tokens,
    hc_hidden_size,
    ssq_stride,
    ssq_tiles,
    split_k,
    RSUM_EPS: tl.constexpr,
    HC: tl.constexpr,
    MIX_SIZE: tl.constexpr,
    HC_PAD: tl.constexpr,
    SPLIT_BLOCK: tl.constexpr,
    SSQ_BLOCK: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINK_REPEAT: tl.constexpr,
):
    """Reduce split-K and sum-of-squares partials, then build coefficients."""
    n = tl.program_id(0)
    idx = tl.arange(0, HC_PAD)
    valid = idx < HC
    comb_idx = idx[:, None] * HC + idx[None, :]
    comb_valid = valid[:, None] & valid[None, :]

    # Vectorised split-K reduction: one masked load per coefficient group.
    si = tl.arange(0, SPLIT_BLOCK)
    s_valid = si < split_k
    split_base = (si * num_tokens + n) * MIX_SIZE
    pre = tl.sum(
        tl.load(
            part_ptr + split_base[:, None] + idx[None, :],
            mask=s_valid[:, None] & valid[None, :],
            other=0.0,
        ),
        axis=0,
    )
    post = tl.sum(
        tl.load(
            part_ptr + split_base[:, None] + (HC + idx)[None, :],
            mask=s_valid[:, None] & valid[None, :],
            other=0.0,
        ),
        axis=0,
    )
    comb = tl.sum(
        tl.load(
            part_ptr + split_base[:, None, None] + (2 * HC + comb_idx)[None, :, :],
            mask=s_valid[:, None, None] & comb_valid[None, :, :],
            other=0.0,
        ),
        axis=0,
    )

    # Fixed-order tile reduction: bitwise deterministic across calls.  The
    # runtime bounds keep the varying tile counts out of the constexpr
    # signature (the persistent tuner cache treats those as config columns).
    ssq = 0.0
    for t0 in range(0, ssq_tiles, SSQ_BLOCK):
        ti = t0 + tl.arange(0, SSQ_BLOCK)
        ssq += tl.sum(
            tl.load(
                ssq_part_ptr + n * ssq_stride + ti,
                mask=ti < ssq_tiles,
                other=0.0,
            )
        )

    scale_0 = tl.load(scale_ptr + 0)
    scale_1 = tl.load(scale_ptr + 1)
    scale_2 = tl.load(scale_ptr + 2)
    rms_inv = tl.rsqrt(ssq / hc_hidden_size + RSUM_EPS)

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
    """Split K until the grid covers the PPU's 64 SMs, without shattering K."""
    # A prefill-sized batch already fills the machine with M tiles; splitting K
    # there only adds a reduction.
    if num_tokens >= 4096:
        return 1
    est_m = max(1, triton.cdiv(num_tokens, 64))
    occupancy = max(1, _TARGET_CTAS // est_m)
    split = max(1, min(32, occupancy))
    return min(split, max(1, hc_hidden_size // 128))


def mhc_shifted_post_collapse(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    pre_mix,
    num_tokens,
    hidden_size,
    capture_aux,
):
    """Return ``(residual_cur, layer_input, aux, ssq_partials, stride, tiles)``."""
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
    partials = torch.empty(num_tokens, max_tiles, dtype=torch.float32, device=device)

    st = comb_res_mix.stride()
    a_sn = st[0]
    # vLLM orientation (MixLayout.TRANSPOSED): stored mix is [j, i].
    a_sj, a_si = st[1], st[2]

    # FlagTree aabs rewrites the tile constexprs during autotuning; the sweep
    # below is what we actually want to compare, so keep it off for the launch
    # (and therefore for the tuner's benchmark runs) and restore it afterwards.
    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    try:
        _mhc_shifted_post_collapse_kernel[
            lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_H"]))
        ](
            comb_res_mix,
            residual,
            post_layer_mix,
            x,
            pre_mix if pre_mix is not None else post_layer_mix,
            residual_cur,
            layer_input,
            aux if capture_aux else layer_input,
            partials,
            a_sn,
            a_sj,
            a_si,
            max_tiles,
            H=hidden_size,
            USE_PRE=pre_mix is not None,
            WRITE_AUX=capture_aux,
        )
    finally:
        triton.knobs.autotuning.adjust_block_size = aabs

    # Read back the tile the tuner actually launched so the coefficient kernel
    # only sums the slots this launch wrote.  Allocating ``torch.zeros`` here
    # instead would add a memset kernel to every call, which dominates the
    # decode-sized shapes.  The fallback zeroes the buffer if the tuner does
    # not expose the selected config.
    best = getattr(_mhc_shifted_post_collapse_kernel.fn, "best_config", None)
    block_h = None if best is None else best.kwargs.get("BLOCK_H")
    if block_h is None:
        partials.zero_()
        ssq_tiles = max_tiles
    else:
        ssq_tiles = triton.cdiv(hidden_size, block_h)

    return residual_cur, layer_input, aux, partials, max_tiles, ssq_tiles


def mhc_shifted_project_thead(a_flat, fn_packed, num_tokens, mix_size):
    """Return ``(split_k_partials, split_k)`` for ``a @ fn^T``."""
    split_k = _split_k_for(num_tokens, a_flat.shape[1])
    partials = torch.empty(
        split_k, num_tokens, mix_size, dtype=torch.float32, device=a_flat.device
    )
    # See mhc_shifted_post_collapse: keep aabs from rewriting the tuned tiles.
    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    try:
        _mhc_shifted_project_thead_kernel[
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
        triton.knobs.autotuning.adjust_block_size = aabs
    return partials, split_k


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
    """Token-fused post + collapse + sum of squares + optional RMSNorm.

    Used for ``H <= 8192``: BLOCK_H covers the whole row, so one CTA per token
    keeps the boundary at a single launch and writes a single sum-of-squares
    value per token.
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
    ssq = torch.empty(num_tokens, 1, dtype=torch.float32, device=device)
    normed = torch.empty_like(layer_input) if norm_weight is not None else layer_input

    st = comb_res_mix.stride()
    a_sn = st[0]
    # vLLM orientation (MixLayout.TRANSPOSED): stored mix is [j, i].
    a_sj, a_si = st[1], st[2]
    block_h = triton.next_power_of_2(hidden_size)

    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
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
            normed,
            a_sn,
            a_sj,
            a_si,
            H=hidden_size,
            BLOCK_H=block_h,
            USE_PRE=pre_mix is not None,
            WRITE_AUX=capture_aux,
            HAS_W=norm_weight is not None,
            EPS=norm_eps,
        )
    finally:
        triton.knobs.autotuning.adjust_block_size = aabs

    layer_out = normed if norm_weight is not None else layer_input
    return residual_cur, layer_out, aux, ssq, 1, 1


def mhc_shifted_post_pre_thead(
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
    """Full hc=4 boundary on the PPU, shared by every PPU shape."""
    device = residual.device

    # (1)+(4) post, collapse and the row sum of squares.  Narrow rows use the
    # one-CTA-per-token kernel that also applies the norm, which removes a
    # launch from the decode-sized shapes.
    use_token_fused = hidden_size <= 8192
    if use_token_fused:
        residual_cur, layer_input, aux, ssq_partials, ssq_stride, ssq_tiles = (
            mhc_shifted_post_collapse_norm(
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
        )
    else:
        residual_cur, layer_input, aux, ssq_partials, ssq_stride, ssq_tiles = (
            mhc_shifted_post_collapse(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                pre_mix,
                num_tokens,
                hidden_size,
                capture_aux,
            )
        )

    # (2) pure split-K projection.
    a_flat = residual_cur.view(num_tokens, hc * hidden_size)
    project_partials, split_k = mhc_shifted_project_thead(
        a_flat, _fn_bf16_packed(fn), num_tokens, mix_size
    )

    # (3) split-K reduction + sum-of-squares reduction + coefficients.
    post_mix = torch.empty(num_tokens, hc, 1, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc, hc, dtype=torch.float32, device=device)
    next_pre = torch.empty(num_tokens, hc, dtype=torch.float32, device=device)
    # aabs rewrites launch constexprs into the tuner config; keep it off here
    # as well so only the declared square brackets are persisted.
    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    try:
        _mhc_shifted_reduce_coeff_thead_kernel[(num_tokens,)](
            project_partials,
            ssq_partials,
            hc_scale,
            hc_base,
            next_pre,
            post_mix,
            comb_mix,
            num_tokens,
            hc * hidden_size,
            ssq_stride,
            ssq_tiles,
            split_k,
            RSUM_EPS=rms_eps,
            HC=hc,
            MIX_SIZE=mix_size,
            HC_PAD=triton.next_power_of_2(hc),
            SPLIT_BLOCK=32,
            SSQ_BLOCK=64,
            PRE_EPS=hc_pre_eps,
            SINK_EPS=hc_sinkhorn_eps,
            POST_MULT=hc_post_mult_value,
            SINK_REPEAT=sinkhorn_repeat,
        )
    finally:
        triton.knobs.autotuning.adjust_block_size = aabs

    # (5) optional RMSNorm of the collapsed input.  The token-fused path has
    # already applied it; wider rows fall back to the generic two-pass kernel.
    if norm_weight is not None and not use_token_fused:
        from flaggems_vllm.ops.mhc.mhc_shifted_post_pre import _mhc_shifted_norm_kernel

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
