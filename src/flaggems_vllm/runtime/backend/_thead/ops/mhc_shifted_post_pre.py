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
   the row sum of squares while the just-rounded bf16 post values are still in
   registers; narrow rows keep the whole row live, wide rows stream H in
   pipelined tiles; the RMSNorm is applied in place in the same kernel when the
   row fits one CTA;
2. the project stage is a pure GEMM over a packed contiguous ``fn`` weight, so
   it can pipeline at ``num_stages >= 2``; above 256 tokens the four residual
   streams run as separate K slabs, below that a split-K kernel fills the
   PPU's SMs even for a single token;
3. small batches fuse the project with the coefficient epilogue into a single
   launch: H <= 1280 keeps the whole K in one CTA, wider small rows keep the
   K split and elect the last CTA per M tile to reduce the partials and run the
   Sinkhorn epilogue.  This removes the separate reduce launch that dominates
   decode-sized shapes;
4. larger batches keep one per-token kernel that reduces the project partials,
   the sum-of-squares tiles and computes the Sinkhorn coefficients.

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

# TLE async loads need both the language surface and the PPU pass that lowers
# them to AIU descriptors.  The regular load paths stay available when either
# is missing.
try:
    import triton.experimental.tle.language as tle
    from triton._C.libtriton import ppu as _ppu

    _HAS_AIU = _ppu.passes.ttppugpuir.add_tle_promote_async_load_to_aiu is not None
except Exception:  # noqa: BLE001 - optional compiler feature
    tle = None
    _HAS_AIU = False

# Smallest BLOCK_H in the tuned space.  The per-token sum-of-squares scratch is
# sized for it, so every tuned tile partition fits the same buffer.
_MIN_BLOCK_H = 128

# PPU-ZW810E has 64 SMs.  Split-K aims to fill them without shattering each
# split below one useful K tile.
_TARGET_CTAS = 64

# AIU staging only breaks even once the K loop is long enough to amortize the
# descriptor setup: at 4H <= 20480 (H <= 5120) the M tiles already keep the
# machine busy and the stream-split kernel is as fast or faster, while at
# 4H = 28672 (H = 7168) the AIU A load is 1.05-1.18x ahead.  Gate on K, not on
# H, because K is the loop that pays for the staging.
_AIU_MIN_K = 24576

# Small batches: fuse the project with the coefficient epilogue and drop the
# split-K reduce launch.  The K axis is not split, so this only pays off while
# the full-K weight read stays cheap.  Measured crossover on the PPU:
# H=1280 wins 1.5-2.7x, H=4096/5120 loses because one CTA can no longer fill
# the machine with the full-K weight read; those keep the split-K project plus
# the combined reduce kernel.
_FUSED_COEFF_MAX_TOKENS = 512
_FUSED_COEFF_MAX_HIDDEN = 1280

# Wider rows cannot use the no-split fusion: one CTA per M tile would read the
# whole 4H-weight slab.  The split-K project still pays off, so it keeps the
# split across CTAs and elects the last CTA per M tile to reduce the partials
# and run the coefficient epilogue, removing the separate reduce launch.
_SPLITK_COEFF_MAX_TOKENS = 512
_SPLITK_COEFF_MAX_HIDDEN = 5120
_NTHREADS = 128  # num_warps is fixed at 4 in the split-K fused tune space
_SPLITK_MIN_BLOCK_M = 16  # smallest BLOCK_M in the split-K fused tune space
_SPLITK_COUNTER_CACHE: dict = {}
_SPLITK_FENCE_CACHE: dict = {}

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
    configs=runtime.get_tuned_config("mhc_shifted_post_collapse_norm_loop_thead"),
    key=["H"],
)
@triton.jit
def _mhc_shifted_post_collapse_norm_loop_kernel(
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
    PIPE_STAGES: tl.constexpr,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
    HAS_W: tl.constexpr,
    EPS: tl.constexpr,
):
    """Token-loop shape: one CTA per token, H streamed in pipelined tiles.

    The flat kernel keeps the whole row live in registers.  This variant keeps
    only ``BLOCK_H`` live and pipelines the next residual/x tile behind the
    current post/collapse work.  The norm needs the full row's sum of squares,
    so it runs a second pass over the stored bf16 ``layer_input`` (one extra
    ``NH`` read) instead of holding the row in registers.
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
    for h0 in tl.range(0, H, BLOCK_H, num_stages=PIPE_STAGES):
        h_off = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_off < H

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
        tl.store(
            out_ptr + out_base + 0 * H + h_off,
            v0,
            mask=h_mask,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + out_base + 1 * H + h_off,
            v1,
            mask=h_mask,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + out_base + 2 * H + h_off,
            v2,
            mask=h_mask,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + out_base + 3 * H + h_off,
            v3,
            mask=h_mask,
            eviction_policy="evict_first",
        )

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
            tl.store(
                aux_ptr + pid_n * H + h_off,
                total.to(tl.bfloat16),
                mask=h_mask,
                eviction_policy="evict_first",
            )

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
        for h0 in tl.range(0, H, BLOCK_H, num_stages=PIPE_STAGES):
            h_off = h0 + tl.arange(0, BLOCK_H)
            h_mask = h_off < H
            value = tl.load(layer_ptr + pid_n * H + h_off, mask=h_mask, other=0.0).to(
                tl.float32
            )
            weight = tl.load(w_ptr + h_off, mask=h_mask, other=0.0).to(tl.float32)
            tl.store(
                norm_ptr + pid_n * H + h_off,
                (value * inv * weight).to(tl.bfloat16),
                mask=h_mask,
                eviction_policy="evict_first",
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
    configs=runtime.get_tuned_config("mhc_shifted_project_thead_aiu"),
    key=["num_tokens", "hc_hidden_size", "split_k"],
)
@triton.jit
def _mhc_shifted_project_thead_aiu_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16, contiguous rows
    bt_ptr,  # (hc_hidden_size, mix_stride) bfloat16, zero-padded
    out_ptr,  # (split_k, num_tokens, mix_size) float32
    num_tokens,
    hc_hidden_size,
    mix_size,
    a_stride_n,
    bt_stride,
    split_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    """Full-K project with an AIU async A load.

    The AIU descriptor needs the whole logical row, so this variant only covers
    the contiguous ``(N, hc*H)`` matrix.  The stream-split kernel's A is a
    strided ``(N, H)`` sub-matrix and must keep regular loads.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < num_tokens
    n_mask = offs_n < mix_size

    k_per_split = tl.cdiv(hc_hidden_size, split_k)
    k_begin = pid_k * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, hc_hidden_size)

    a_bp = tl.make_block_ptr(
        base=a_ptr,
        shape=(num_tokens, hc_hidden_size),
        strides=(a_stride_n, 1),
        offsets=(pid_m * BLOCK_M, k_begin),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(k_begin, k_end, BLOCK_K, num_stages=PIPE_STAGES):
        a = tle.load(a_bp, boundary_check=(0, 1), padding_option="zero", is_async=True)
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_end
        b = tl.load(
            bt_ptr + offs_k[:, None] * bt_stride + offs_n[None, :],
            mask=k_mask[:, None] & (offs_n[None, :] < bt_stride),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))

    tl.store(
        out_ptr + (pid_k * num_tokens + offs_m[:, None]) * mix_size + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_project_thead_stream"),
    key=["num_tokens", "hidden_size", "split_k"],
)
@triton.jit
def _mhc_shifted_project_thead_stream_kernel(
    a_ptr,  # (num_tokens, 4 * hidden_size) bfloat16
    bt_ptr,  # (4 * hidden_size, mix_stride) bfloat16, zero-padded
    out_ptr,  # (N_STREAMS * split_k, num_tokens, mix_size) float32
    num_tokens,
    hidden_size,
    mix_size,
    a_stride_n,
    bt_stride,
    split_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_STREAMS: tl.constexpr,
):
    """Project each residual stream as an independent K slab.

    The full-K kernel walks ``4 * H`` per CTA.  Splitting the four streams into
    separate grid slices shortens that loop to ``H / split_k`` and keeps each
    stream's weight tile in L2, which wins once the M grid has enough tiles.
    The reduce kernel sums the stream partials in a fixed order.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    stream = pid_s // split_k
    split = pid_s % split_k

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < num_tokens
    n_mask = offs_n < mix_size

    k_per_split = tl.cdiv(hidden_size, split_k)
    k_begin = split * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, hidden_size)

    stream_off = stream * hidden_size
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_end
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_n + stream_off + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            bt_ptr + (stream_off + offs_k)[:, None] * bt_stride + offs_n[None, :],
            mask=k_mask[:, None] & (offs_n[None, :] < bt_stride),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    tl.store(
        out_ptr + (pid_s * num_tokens + offs_m[:, None]) * mix_size + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _mhc_shifted_coeff_epilogue(
    pre,  # (BLOCK_M, HC_PAD) float32 raw projection, first HC columns valid
    post,  # (BLOCK_M, HC_PAD) float32 raw projection, first HC columns valid
    comb,  # (BLOCK_M, HC * HC) float32 raw projection, row-major [i, j]
    ssq,  # (BLOCK_M,) float32 row sum of squares of residual_cur
    scale_ptr,  # (3,) float32
    base_ptr,  # (MIX_SIZE,) float32
    pre_out_ptr,  # (num_tokens, HC) float32
    post_out_ptr,  # (num_tokens, HC) float32
    comb_out_ptr,  # (num_tokens, HC, HC) float32
    offs_m,  # (BLOCK_M,) int32
    m_mask,  # (BLOCK_M,) bool
    hc_hidden_size,
    RSUM_EPS: tl.constexpr,
    HC: tl.constexpr,
    HC_PAD: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINK_REPEAT: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Turn the raw project columns into the three coefficient groups.

    Shared by the no-split fused kernel and the split-K last-CTA kernel so both
    keep exactly the same Sinkhorn arithmetic and rounding.
    """
    idx = tl.arange(0, HC_PAD)
    valid = idx < HC
    comb_idx = idx[:, None] * HC + idx[None, :]
    comb_valid = valid[:, None] & valid[None, :]

    comb = tl.reshape(comb, (BLOCK_M, HC_PAD, HC_PAD))
    rms_inv = tl.rsqrt(ssq / hc_hidden_size + RSUM_EPS)

    scale_0 = tl.load(scale_ptr + 0)
    scale_1 = tl.load(scale_ptr + 1)
    scale_2 = tl.load(scale_ptr + 2)

    pre_logits = pre * rms_inv[:, None] * scale_0
    pre_logits += tl.load(base_ptr + idx, mask=valid, other=0.0)[None, :]
    tl.store(
        pre_out_ptr + offs_m[:, None] * HC + idx[None, :],
        tl.sigmoid(pre_logits) + PRE_EPS,
        mask=m_mask[:, None] & valid[None, :],
    )

    post_logits = post * rms_inv[:, None] * scale_1
    post_logits += tl.load(base_ptr + HC + idx, mask=valid, other=0.0)[None, :]
    tl.store(
        post_out_ptr + offs_m[:, None] * HC + idx[None, :],
        tl.sigmoid(post_logits) * POST_MULT,
        mask=m_mask[:, None] & valid[None, :],
    )

    comb = comb * rms_inv[:, None, None] * scale_2
    comb += tl.load(base_ptr + 2 * HC + comb_idx, mask=comb_valid, other=0.0)[
        None, :, :
    ]
    comb = tl.where(comb_valid[None, :, :], comb, float("-inf"))
    comb = tl.exp(comb - tl.max(comb, axis=2)[:, :, None])
    comb = comb / tl.sum(comb, axis=2)[:, :, None]
    comb = tl.where(comb_valid[None, :, :], comb + SINK_EPS, 0.0)
    comb = comb / (tl.sum(comb, axis=1)[:, None, :] + SINK_EPS)
    for _ in range(SINK_REPEAT - 1):
        comb = comb / (tl.sum(comb, axis=2)[:, :, None] + SINK_EPS)
        comb = comb / (tl.sum(comb, axis=1)[:, None, :] + SINK_EPS)
    tl.store(
        comb_out_ptr + offs_m[:, None, None] * (HC * HC) + comb_idx[None, :, :],
        comb,
        mask=m_mask[:, None, None] & comb_valid[None, :, :],
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_project_coeff_thead"),
    key=["num_tokens", "hc_hidden_size"],
)
@triton.jit
def _mhc_shifted_project_coeff_thead_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (hc_hidden_size, mix_stride) bfloat16, zero-padded
    scratch_ptr,  # (num_tokens, mix_stride) float32 raw projection
    ssq_ptr,  # (num_tokens, 1) float32
    scale_ptr,  # (3,) float32
    base_ptr,  # (MIX_SIZE,) float32
    pre_out_ptr,  # (num_tokens, HC) float32
    post_out_ptr,  # (num_tokens, HC) float32
    comb_out_ptr,  # (num_tokens, HC, HC) float32
    num_tokens,
    hc_hidden_size,
    mix_size,
    a_stride_n,
    bt_stride,
    RSUM_EPS: tl.constexpr,
    HC: tl.constexpr,
    MIX_SIZE: tl.constexpr,
    HC_PAD: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINK_REPEAT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Full-K project plus the coefficient epilogue in one launch.

    Small batches are host-bound: the split-K reduce launch costs more than
    the extra parallelism saves.  One CTA owns its ``BLOCK_M`` tokens for the
    whole K reduction, so the raw projection never leaves the CTA.  The
    projection is staged through the scratch buffer and reloaded by the same
    CTA (barrier-ordered), which avoids cutting the dot result into the
    pre / post / comb column groups in registers.

    ``BLOCK_N`` must equal ``bt_stride`` (32 in the tuned space), because the
    scratch row stride is the packed weight stride.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < num_tokens
    n_mask = offs_n < mix_size

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, hc_hidden_size, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hc_hidden_size
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

    tl.store(
        scratch_ptr + offs_m[:, None] * bt_stride + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )
    # The same CTA reads the rows it just wrote; a CTA barrier is enough.
    tl.debug_barrier()

    idx = tl.arange(0, HC_PAD)
    valid = idx < HC
    cflat = tl.arange(0, HC * HC)

    pre = tl.load(
        scratch_ptr + offs_m[:, None] * bt_stride + idx[None, :],
        mask=m_mask[:, None] & valid[None, :],
        other=0.0,
    )
    post = tl.load(
        scratch_ptr + offs_m[:, None] * bt_stride + (HC + idx)[None, :],
        mask=m_mask[:, None] & valid[None, :],
        other=0.0,
    )
    comb = tl.load(
        scratch_ptr + offs_m[:, None] * bt_stride + (2 * HC + cflat)[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )
    ssq = tl.load(ssq_ptr + offs_m, mask=m_mask, other=0.0)

    _mhc_shifted_coeff_epilogue(
        pre,
        post,
        comb,
        ssq,
        scale_ptr,
        base_ptr,
        pre_out_ptr,
        post_out_ptr,
        comb_out_ptr,
        offs_m,
        m_mask,
        hc_hidden_size,
        RSUM_EPS,
        HC,
        HC_PAD,
        PRE_EPS,
        SINK_EPS,
        POST_MULT,
        SINK_REPEAT,
        BLOCK_M,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_project_coeff_splitk_thead"),
    key=["num_tokens", "hc_hidden_size", "split_k"],
)
@triton.jit
def _mhc_shifted_project_coeff_splitk_thead_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (hc_hidden_size, mix_stride) bfloat16, zero-padded
    part_ptr,  # (split_k, num_tokens, BLOCK_N) float32 split partials
    counter_ptr,  # (num_m_tiles,) int32, zero before the launch
    fence_ptr,  # (num_m_tiles * NTHREADS,) int32 release/acquire slots
    ssq_ptr,  # (num_tokens, 1) float32
    scale_ptr,  # (3,) float32
    base_ptr,  # (MIX_SIZE,) float32
    pre_out_ptr,  # (num_tokens, HC) float32
    post_out_ptr,  # (num_tokens, HC) float32
    comb_out_ptr,  # (num_tokens, HC, HC) float32
    num_tokens,
    hc_hidden_size,
    mix_size,
    a_stride_n,
    bt_stride,
    split_k,
    RSUM_EPS: tl.constexpr,
    HC: tl.constexpr,
    MIX_SIZE: tl.constexpr,
    HC_PAD: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINK_REPEAT: tl.constexpr,
    NTHREADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Split-K project plus a last-CTA coefficient epilogue in one launch.

    The split partials are written normally; every thread then publishes its
    own stores with a release xchg on a per-(M tile, thread) fence slot, and
    the CTA that sees the final counter value acquires the matching slots and
    reduces the partials in a fixed split order.  The atomic only elects the
    last CTA; the numerical reduction order is fixed, so repeated calls stay
    bitwise deterministic.  The summed tile reuses split 0's partial slot, so
    the epilogue needs no separate scratch buffer.  ``num_warps`` is fixed at 4
    in the tuned space because the fence width is ``NTHREADS == 128``.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
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

    tl.store(
        part_ptr + (pid_k * num_tokens + offs_m[:, None]) * BLOCK_N + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )
    tl.debug_barrier()

    lane = tl.arange(0, NTHREADS)
    fence = fence_ptr + pid_m * NTHREADS + lane
    # Per-thread release: each thread's partial stores are visible before the
    # counter increment below.
    tl.atomic_xchg(fence, 0, sem="release", scope="gpu")

    one = tl.arange(0, 1)
    old = tl.atomic_add(
        counter_ptr + pid_m + one * 0,
        1,
        mask=one == 0,
        sem="acq_rel",
        scope="gpu",
    )
    is_last = tl.max(tl.where(one == 0, old, 0), axis=0) == split_k - 1

    if is_last:
        # Acquire the matching slots before reading the other splits' partials.
        tl.atomic_xchg(fence, 0, sem="acquire", scope="gpu")
        summed = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for s in range(split_k):
            summed += tl.load(
                part_ptr
                + (s * num_tokens + offs_m[:, None]) * BLOCK_N
                + offs_n[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
        # Reuse split 0's partial slot as the summed projection staging area.
        tl.store(
            part_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :],
            summed,
            mask=m_mask[:, None] & n_mask[None, :],
        )
        tl.debug_barrier()

        idx = tl.arange(0, HC_PAD)
        valid = idx < HC
        cflat = tl.arange(0, HC * HC)
        pre = tl.load(
            part_ptr + offs_m[:, None] * BLOCK_N + idx[None, :],
            mask=m_mask[:, None] & valid[None, :],
            other=0.0,
        )
        post = tl.load(
            part_ptr + offs_m[:, None] * BLOCK_N + (HC + idx)[None, :],
            mask=m_mask[:, None] & valid[None, :],
            other=0.0,
        )
        comb = tl.load(
            part_ptr + offs_m[:, None] * BLOCK_N + (2 * HC + cflat)[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        ssq = tl.load(ssq_ptr + offs_m, mask=m_mask, other=0.0)

        _mhc_shifted_coeff_epilogue(
            pre,
            post,
            comb,
            ssq,
            scale_ptr,
            base_ptr,
            pre_out_ptr,
            post_out_ptr,
            comb_out_ptr,
            offs_m,
            m_mask,
            hc_hidden_size,
            RSUM_EPS,
            HC,
            HC_PAD,
            PRE_EPS,
            SINK_EPS,
            POST_MULT,
            SINK_REPEAT,
            BLOCK_M,
        )
        # Re-arm the counter for the next call on this M tile.
        tl.store(counter_ptr + pid_m + one * 0, 0, mask=one == 0)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_coeff_thead"),
    key=["num_tokens"],
)
@triton.jit
def _mhc_shifted_reduce_coeff_thead_kernel(
    part_ptr,  # (split_k, num_tokens, MIX_SIZE) float32
    ssq_part_ptr,  # (num_tokens, ssq_stride) float32
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
    TOKEN_TILE: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINK_REPEAT: tl.constexpr,
):
    """Reduce project partials and build the Sinkhorn coefficients.

    ``TOKEN_TILE`` tokens share one program, so the serial Sinkhorn chain runs
    as ``[T]``-vector math and the grid shrinks by the same factor.  The
    per-token arithmetic and the fixed reduction order are unchanged.

    hc=4 only (the thead dispatch): ``HC_PAD == HC``, and the stored comb block
    is ``HC * HC`` contiguous values, so it is loaded flat and reshaped.
    """
    pid = tl.program_id(0)
    offs_t = pid * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
    t_mask = offs_t < num_tokens

    idx = tl.arange(0, HC_PAD)
    valid = idx < HC
    comb_idx = idx[:, None] * HC + idx[None, :]
    comb_valid = valid[:, None] & valid[None, :]
    cflat = tl.arange(0, HC * HC)

    # Vectorised split reduction: one masked load per coefficient group.
    si = tl.arange(0, SPLIT_BLOCK)
    s_valid = si < split_k
    split_base = (si[:, None] * num_tokens + offs_t[None, :]) * MIX_SIZE
    pre = tl.sum(
        tl.load(
            part_ptr + split_base[:, :, None] + idx[None, None, :],
            mask=s_valid[:, None, None] & t_mask[None, :, None] & valid[None, None, :],
            other=0.0,
        ),
        axis=0,
    )
    post = tl.sum(
        tl.load(
            part_ptr + split_base[:, :, None] + (HC + idx)[None, None, :],
            mask=s_valid[:, None, None] & t_mask[None, :, None] & valid[None, None, :],
            other=0.0,
        ),
        axis=0,
    )
    comb_flat = tl.sum(
        tl.load(
            part_ptr + split_base[:, :, None] + (2 * HC + cflat)[None, None, :],
            mask=s_valid[:, None, None] & t_mask[None, :, None],
            other=0.0,
        ),
        axis=0,
    )
    comb = tl.reshape(comb_flat, (TOKEN_TILE, HC_PAD, HC_PAD))

    # Fixed-order tile reduction: bitwise deterministic across calls.  The
    # runtime bounds keep the varying tile counts out of the constexpr
    # signature (the persistent tuner cache treats those as config columns).
    ssq = tl.zeros((TOKEN_TILE,), dtype=tl.float32)
    for t0 in range(0, ssq_tiles, SSQ_BLOCK):
        ti = t0 + tl.arange(0, SSQ_BLOCK)
        ssq += tl.sum(
            tl.load(
                ssq_part_ptr + offs_t[:, None] * ssq_stride + ti[None, :],
                mask=t_mask[:, None] & (ti[None, :] < ssq_tiles),
                other=0.0,
            ),
            axis=1,
        )

    scale_0 = tl.load(scale_ptr + 0)
    scale_1 = tl.load(scale_ptr + 1)
    scale_2 = tl.load(scale_ptr + 2)
    rms_inv = tl.rsqrt(ssq / hc_hidden_size + RSUM_EPS)

    pre_logits = pre * rms_inv[:, None] * scale_0
    pre_logits += tl.load(base_ptr + idx, mask=valid, other=0.0)[None, :]
    tl.store(
        pre_out_ptr + offs_t[:, None] * HC + idx[None, :],
        tl.sigmoid(pre_logits) + PRE_EPS,
        mask=t_mask[:, None] & valid[None, :],
    )

    post_logits = post * rms_inv[:, None] * scale_1
    post_logits += tl.load(base_ptr + HC + idx, mask=valid, other=0.0)[None, :]
    tl.store(
        post_out_ptr + offs_t[:, None] * HC + idx[None, :],
        tl.sigmoid(post_logits) * POST_MULT,
        mask=t_mask[:, None] & valid[None, :],
    )

    comb = comb * rms_inv[:, None, None] * scale_2
    comb += tl.load(base_ptr + 2 * HC + comb_idx, mask=comb_valid, other=0.0)[
        None, :, :
    ]
    comb = tl.where(comb_valid[None, :, :], comb, float("-inf"))
    comb = tl.exp(comb - tl.max(comb, axis=2)[:, :, None])
    comb = comb / tl.sum(comb, axis=2)[:, :, None]
    comb = tl.where(comb_valid[None, :, :], comb + SINK_EPS, 0.0)
    comb = comb / (tl.sum(comb, axis=1)[:, None, :] + SINK_EPS)
    for _ in range(SINK_REPEAT - 1):
        comb = comb / (tl.sum(comb, axis=2)[:, :, None] + SINK_EPS)
        comb = comb / (tl.sum(comb, axis=1)[:, None, :] + SINK_EPS)
    tl.store(
        comb_out_ptr + offs_t[:, None, None] * (HC * HC) + comb_idx[None, :, :],
        comb,
        mask=t_mask[:, None, None] & comb_valid[None, :, :],
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


def mhc_shifted_project_thead_aiu(a_flat, fn_packed, num_tokens, mix_size):
    """Full-K project with AIU A loads; return ``(partials, split_k)``."""
    split_k = _split_k_for(num_tokens, a_flat.shape[1])
    partials = torch.empty(
        split_k, num_tokens, mix_size, dtype=torch.float32, device=a_flat.device
    )
    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    try:
        _mhc_shifted_project_thead_aiu_kernel[
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


def _stream_split_for(num_tokens, hidden_size):
    """Per-stream K splits for the four-stream project.

    Medium batches with wide rows still leave the stream grid under-filled;
    one K split per stream shortens each CTA's loop without adding a large
    reduction.  Larger batches already expose enough M tiles.
    """
    return 4 if num_tokens < 512 and hidden_size >= 4096 else 1


def mhc_shifted_project_thead_stream(a_flat, fn_packed, num_tokens, mix_size):
    """Return ``(stream_partials, num_partial_planes)`` for ``a @ fn^T``.

    Each of the four residual streams becomes its own grid slice, so the
    reduce kernel sums ``4 * split_k`` fixed-order partials.
    """
    n_streams = 4
    hidden_size = a_flat.shape[1] // n_streams
    split_k = _stream_split_for(num_tokens, hidden_size)
    num_partials = n_streams * split_k
    partials = torch.empty(
        num_partials, num_tokens, mix_size, dtype=torch.float32, device=a_flat.device
    )
    # See mhc_shifted_project_thead: keep aabs from rewriting the tuned tiles.
    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    try:
        _mhc_shifted_project_thead_stream_kernel[
            lambda meta: (
                triton.cdiv(num_tokens, meta["BLOCK_M"]),
                triton.cdiv(mix_size, meta["BLOCK_N"]),
                num_partials,
            )
        ](
            a_flat,
            fn_packed,
            partials,
            num_tokens,
            hidden_size,
            mix_size,
            a_flat.stride(0),
            fn_packed.stride(0),
            split_k,
            N_STREAMS=n_streams,
        )
    finally:
        triton.knobs.autotuning.adjust_block_size = aabs
    return partials, num_partials


def mhc_shifted_project_coeff_thead(
    a_flat,
    fn_packed,
    ssq,
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
):
    """Return ``(post_mix, comb_mix, next_pre)`` from one fused launch."""
    device = a_flat.device
    # ``BLOCK_N`` in the tuned space equals the packed weight stride (32), so
    # the scratch row stride can reuse ``fn_packed.stride(0)``.
    scratch = torch.empty(
        num_tokens, fn_packed.shape[1], dtype=torch.float32, device=device
    )
    post_mix = torch.empty(num_tokens, hc, 1, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc, hc, dtype=torch.float32, device=device)
    next_pre = torch.empty(num_tokens, hc, dtype=torch.float32, device=device)

    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    try:
        _mhc_shifted_project_coeff_thead_kernel[
            lambda meta: (triton.cdiv(num_tokens, meta["BLOCK_M"]),)
        ](
            a_flat,
            fn_packed,
            scratch,
            ssq,
            hc_scale,
            hc_base,
            next_pre,
            post_mix,
            comb_mix,
            num_tokens,
            hc * hidden_size,
            mix_size,
            a_flat.stride(0),
            fn_packed.stride(0),
            RSUM_EPS=rms_eps,
            HC=hc,
            MIX_SIZE=mix_size,
            HC_PAD=triton.next_power_of_2(hc),
            PRE_EPS=hc_pre_eps,
            SINK_EPS=hc_sinkhorn_eps,
            POST_MULT=hc_post_mult_value,
            SINK_REPEAT=sinkhorn_repeat,
        )
    finally:
        triton.knobs.autotuning.adjust_block_size = aabs
    return post_mix, comb_mix, next_pre


def _splitk_counter(device, num_m_tiles):
    key = (device, num_m_tiles)
    buf = _SPLITK_COUNTER_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(num_m_tiles, dtype=torch.int32, device=device)
        _SPLITK_COUNTER_CACHE[key] = buf
    return buf


def _splitk_fence(device, num_m_tiles):
    key = (device, num_m_tiles)
    buf = _SPLITK_FENCE_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(num_m_tiles * _NTHREADS, dtype=torch.int32, device=device)
        _SPLITK_FENCE_CACHE[key] = buf
    return buf


def mhc_shifted_project_coeff_splitk_thead(
    a_flat,
    fn_packed,
    ssq,
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
):
    """Return ``(post_mix, comb_mix, next_pre)`` from one split-K launch."""
    device = a_flat.device
    split_k = _split_k_for(num_tokens, a_flat.shape[1])
    mix_stride = fn_packed.shape[1]
    partials = torch.empty(
        split_k, num_tokens, mix_stride, dtype=torch.float32, device=device
    )
    post_mix = torch.empty(num_tokens, hc, 1, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc, hc, dtype=torch.float32, device=device)
    next_pre = torch.empty(num_tokens, hc, dtype=torch.float32, device=device)

    # The counter and fence buffers are reused across calls: the last CTA
    # re-arms its counter, and the fence slots are xchg-ed back to zero.  Size
    # them for the smallest BLOCK_M in the tune space (largest tile grid).
    num_m_tiles = triton.cdiv(num_tokens, _SPLITK_MIN_BLOCK_M)
    counter = _splitk_counter(device, num_m_tiles)
    fence = _splitk_fence(device, num_m_tiles)

    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    try:
        _mhc_shifted_project_coeff_splitk_thead_kernel[
            lambda meta: (
                triton.cdiv(num_tokens, meta["BLOCK_M"]),
                split_k,
            )
        ](
            a_flat,
            fn_packed,
            partials,
            counter,
            fence,
            ssq,
            hc_scale,
            hc_base,
            next_pre,
            post_mix,
            comb_mix,
            num_tokens,
            hc * hidden_size,
            mix_size,
            a_flat.stride(0),
            fn_packed.stride(0),
            split_k,
            RSUM_EPS=rms_eps,
            HC=hc,
            MIX_SIZE=mix_size,
            HC_PAD=triton.next_power_of_2(hc),
            PRE_EPS=hc_pre_eps,
            SINK_EPS=hc_sinkhorn_eps,
            POST_MULT=hc_post_mult_value,
            SINK_REPEAT=sinkhorn_repeat,
            NTHREADS=_NTHREADS,
        )
    finally:
        triton.knobs.autotuning.adjust_block_size = aabs
    return post_mix, comb_mix, next_pre


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

    Used for ``H <= 2048``: BLOCK_H covers the whole row, so one CTA per token
    keeps the boundary at a single launch and writes a single sum-of-squares
    value per token.  Wider rows use the token-loop variant.
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
    # Normalise in place: the pre-norm collapsed row is not an output, and the
    # kernel reads the tile from registers before storing the normalised value.
    normed = layer_input

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

    layer_out = layer_input
    return residual_cur, layer_out, aux, ssq, 1, 1


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
    """Token-loop variant of the fused boundary; same outputs as the flat one."""
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
    # Normalise in place: the pre-norm collapsed row is not an output, and the
    # kernel reads the tile from registers before storing the normalised value.
    normed = layer_input

    st = comb_res_mix.stride()
    a_sn = st[0]
    # vLLM orientation (MixLayout.TRANSPOSED): stored mix is [j, i].
    a_sj, a_si = st[1], st[2]

    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
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
            normed,
            a_sn,
            a_sj,
            a_si,
            H=hidden_size,
            USE_PRE=pre_mix is not None,
            WRITE_AUX=capture_aux,
            HAS_W=norm_weight is not None,
            EPS=norm_eps,
        )
    finally:
        triton.knobs.autotuning.adjust_block_size = aabs

    layer_out = layer_input
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
    # launch from the decode-sized shapes.  Wide rows stream the hidden axis in
    # pipelined tiles: the flat whole-row tile spills at H >= 4096.
    use_token_fused = hidden_size <= 8192
    if use_token_fused:
        token_kernel = (
            mhc_shifted_post_collapse_norm_loop
            if hidden_size >= 4096
            else mhc_shifted_post_collapse_norm
        )
        residual_cur, layer_input, aux, ssq_partials, ssq_stride, ssq_tiles = (
            token_kernel(
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

    # (2) projection.  Small batches fuse the project with the coefficient
    # epilogue and skip the split-K reduce launch, because the host launch cost
    # is the floor there.  Larger batches keep the split-K variants: above 256
    # tokens the four-stream split shortens each CTA's K loop, and the AIU A
    # load replaces it only for long-K batches, where its staging is amortized
    # (see _AIU_MIN_K).
    a_flat = residual_cur.view(num_tokens, hc * hidden_size)
    fn_packed = _fn_bf16_packed(fn)
    if num_tokens <= _FUSED_COEFF_MAX_TOKENS and hidden_size <= _FUSED_COEFF_MAX_HIDDEN:
        post_mix, comb_mix, next_pre = mhc_shifted_project_coeff_thead(
            a_flat,
            fn_packed,
            ssq_partials,
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
        )
    elif (
        num_tokens <= _SPLITK_COEFF_MAX_TOKENS
        and hidden_size <= _SPLITK_COEFF_MAX_HIDDEN
    ):
        # Wider small rows keep the K split, but the last CTA reduces the
        # partials and emits the coefficients in the same launch.
        post_mix, comb_mix, next_pre = mhc_shifted_project_coeff_splitk_thead(
            a_flat,
            fn_packed,
            ssq_partials,
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
        )
    else:
        if _HAS_AIU and 256 <= num_tokens < 1024 and hc * hidden_size >= _AIU_MIN_K:
            project_partials, split_k = mhc_shifted_project_thead_aiu(
                a_flat, fn_packed, num_tokens, mix_size
            )
        elif num_tokens >= 256:
            project_partials, split_k = mhc_shifted_project_thead_stream(
                a_flat, fn_packed, num_tokens, mix_size
            )
        else:
            project_partials, split_k = mhc_shifted_project_thead(
                a_flat, fn_packed, num_tokens, mix_size
            )

        # (3) split-K reduction + sum-of-squares reduction + coefficients.
        post_mix = torch.empty(num_tokens, hc, 1, dtype=torch.float32, device=device)
        comb_mix = torch.empty(num_tokens, hc, hc, dtype=torch.float32, device=device)
        next_pre = torch.empty(num_tokens, hc, dtype=torch.float32, device=device)
        # aabs rewrites launch constexprs into the tuner config; keep it off
        # here as well so only the declared square brackets are persisted.
        aabs = triton.knobs.autotuning.adjust_block_size
        triton.knobs.autotuning.adjust_block_size = False
        try:
            _mhc_shifted_reduce_coeff_thead_kernel[
                lambda meta: (triton.cdiv(num_tokens, meta["TOKEN_TILE"]),)
            ](
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
