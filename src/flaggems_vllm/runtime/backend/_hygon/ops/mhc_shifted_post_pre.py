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

"""Hygon (gfx93x DCU) specialization of the shifted mHC post -> pre boundary.

The generic path spends five launches on the boundary, and on Hygon the host
side is the real cost: LibEntry rebuilds a specialization key for every tensor
argument on every call, and the Hygon key path also asks the HIP backend for a
per-tensor specialization.  Measured on BW1000 that is 50-150 us per kernel,
which is the whole latency of the decode-sized shapes.

This module keeps the arithmetic verbatim and changes the dataflow:

1. one CTA per token computes post, collapse, the stream mean and the row sum
   of squares while the just-rounded bf16 post values are still in registers.
   Narrow rows keep the whole row live and apply the RMSNorm in the same pass;
   wide rows stream H in tiles and normalise in a second pass.
2. ``fn`` is packed once into a contiguous ``(32, hc*H)`` bf16 operand (the
   unused rows are zero) and the projection becomes a pure split-K GEMM that
   pipelines with ``num_stages >= 2``.
3. the split-K reduction and the Sinkhorn coefficients share the projection
   launch: the last CTA of each token tile reduces the partials and writes the
   three coefficient groups, so the boundary is two launches.  The arrival
   counter is a per-tile ``tl.atomic_add`` with acq_rel semantics, the same
   pattern ``_hygon/ops/mhc_pre.py`` uses.
4. for decode-sized batches (``num_tokens <= _ONE_LAUNCH_MAX_TOKENS``, with a
   smaller cap once the row no longer fits one flat tile) the token kernel also
   runs the projection and the coefficients: every token owns its whole row, so
   no cross-CTA reduction is needed and the boundary is one launch.  The
   trade-off is that each CTA re-reads ``fn`` from L2, which is cheaper than a
   launch until the batch (or the row) is large enough that the per-CTA re-read
   dominates.
5. after the first (tuning) call the kernels are launched through a cached
   compiled-kernel path that skips LibEntry's per-call key reconstruction and
   the driver's stream lookup (~10 us -> ~6 us per launch here).

Every reduction runs in a fixed order (split index ascending, H tiles
ascending), so repeated calls stay bitwise deterministic; the tests compare
two calls with ``rtol=0, atol=0``.

The post fma chain, the collapse weighting and the coefficient arithmetic are
copied verbatim from ``mhc_post`` and ``_mhc_shifted_coeff_kernel``: the post
output has to stay bitwise equal to ``mhc_post``.

This module is intentionally *not* re-exported from ``_hygon.ops.__init__``:
``SpecOpRegistrar`` would then replace the public operator with this private
launcher.  ``flaggems_vllm.ops.mhc.mhc_shifted_post_pre`` imports it lazily.
"""

import weakref

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

_FN_PACK_CACHE: dict = {}
_FN_STRIDE = 32

# The flat token kernel keeps the whole row live in registers; wider rows use
# the H-streamed loop kernel.  Same boundary the MetaX/MThreads paths measured:
# H <= 2048 fits one tile, past that the 4-stream footprint starts spilling.
_TOKEN_FLAT_MAX_HIDDEN = 2048

# Shortest K slice worth giving a CTA.  Splitting past this only adds partial
# traffic and a longer last-CTA reduction without buying parallelism the 80 CUs
# can use.
_MIN_K_PER_SPLIT = 1024

# A split is "even" when every K tile is full.  Aligning the split to 256
# elements covers every BLOCK_K in the tune space (64/128/256) without the
# per-element K mask.
_ALIGN_K = 256

# BW1000 has 80 CUs.  The projection is memory bound; measured across the
# benchmark shapes the best split fills roughly one wave, and going to several
# waves costs more in partial traffic than it recovers in parallelism.
_PROJECT_TARGET_CTAS = 64
_PROJECT_MAX_SPLIT = 32

# Up to this batch the all-in-one kernel beats the two-launch path: the extra
# per-CTA ``fn`` traffic stays inside L2 while a second launch costs ~6 us.
_ONE_LAUNCH_MAX_TOKENS = 64
# Wide rows make the fused projection re-read ``fn`` (48 * H bytes) per CTA
# from L2; past a handful of tokens the split-K project wins even with the
# extra launch.
_ONE_LAUNCH_MAX_TOKENS_WIDE = 16

_COUNTER_CACHE: dict = {}


@triton.jit
def _pack_fn_bf16_kernel(
    src_ptr,  # (mix_size, K) float32, contiguous
    dst_ptr,  # (32, K) bfloat16, zero-padded rows
    numel,
    K,
    dst_stride,
    BLOCK: tl.constexpr,
):
    """One-time ``fn`` cast into the project kernel's B operand."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    n = offs // K
    k = offs % K
    value = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + n * dst_stride + k, value.to(tl.bfloat16), mask=mask)


def _fn_bf16_packed(fn: torch.Tensor) -> torch.Tensor:
    """``fn`` as a contiguous ``(32, hc*H)`` bfloat16 tensor, cached.

    ``mix_size`` is hc*(hc+2) = 24 for hc=4, so the operand is padded to 32
    rows; the dot and the elementwise projection both read the padded layout
    and never mask the mix axis.
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
    out = torch.zeros(_FN_STRIDE, K, dtype=torch.bfloat16, device=fn.device)
    if fn.numel():
        _pack_fn_bf16_kernel[lambda meta: (triton.cdiv(fn.numel(), meta["BLOCK"]),)](
            fn, out, fn.numel(), K, out.stride(0), BLOCK=1024
        )
    _FN_PACK_CACHE[key] = (
        weakref.ref(fn, lambda ref, key=key: _FN_PACK_CACHE.pop(key, None)),
        version,
        ptr,
        out,
    )
    return out


@triton.jit
def _post_collapse_tile(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    pre_ptr,
    out_ptr,
    layer_ptr,
    aux_ptr,
    w_ptr,
    token,
    h_off,
    h_mask,
    a_base,
    c_base,
    b_base,
    d_base,
    out_base,
    H: tl.constexpr,
    a_stride_i,
    a_stride_j,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
    HAS_W: tl.constexpr,
    EPS: tl.constexpr,
    STORE_RAW: tl.constexpr,
    HC: tl.constexpr,
):
    """Post + collapse + stream mean + (optional) RMSNorm for one H tile.

    Returns the bf16-rounded post values and the fp32 accumulator as
    ``(f0, f1, f2, f3, col_bf, layer_sq)``; ``layer_sq`` is the row sum of
    squares of the collapsed value and is 0 when the norm is off.
    """
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
        p0 = tl.load(pre_ptr + token * HC + 0)
        p1 = tl.load(pre_ptr + token * HC + 1)
        p2 = tl.load(pre_ptr + token * HC + 2)
        p3 = tl.load(pre_ptr + token * HC + 3)
        col = f0 * p0
        col += f1 * p1
        col += f2 * p2
        col += f3 * p3
    else:
        col = f0
    col_bf = col.to(tl.bfloat16)
    if (not HAS_W) or STORE_RAW:
        tl.store(layer_ptr + token * H + h_off, col_bf, mask=h_mask)
    if WRITE_AUX:
        total = f0 + f1
        total += f2
        total += f3
        total = total / 4.0
        tl.store(aux_ptr + token * H + h_off, total.to(tl.bfloat16), mask=h_mask)

    layer_sq = tl.zeros((1,), dtype=tl.float32)
    if HAS_W:
        value = col_bf.to(tl.float32)
        layer_sq += tl.sum(value * value, axis=0)
    return f0, f1, f2, f3, col_bf, layer_sq


@triton.jit
def _project_rows(
    res_ptr,
    fn_ptr,
    token,
    fn_stride,
    H: tl.constexpr,
    BLOCK_F: tl.constexpr,
    HC: tl.constexpr,
):
    """Project one token's stored row onto the packed ``fn`` operand.

    ``residual_cur`` was just written by this CTA, so the caller has to issue a
    ``tl.debug_barrier`` before this helper.  The reduction order is H tiles
    ascending and stream index ascending, which keeps repeated calls bitwise
    deterministic.  Every tile stays ``BLOCK_F`` wide, so the combined register
    footprint is bounded by the tile size, not by H.
    """
    n_idx = tl.arange(0, HC)
    comb_rows = 2 * HC + n_idx[:, None] * HC + n_idx[None, :]
    pre_acc = tl.zeros((HC,), dtype=tl.float32)
    post_acc = tl.zeros((HC,), dtype=tl.float32)
    comb_acc = tl.zeros((HC, HC), dtype=tl.float32)
    base = token * (HC * H)
    for f0 in range(0, H, BLOCK_F):
        offs = f0 + tl.arange(0, BLOCK_F)
        f_mask = offs < H
        for j in tl.static_range(HC):
            a = tl.load(res_ptr + base + j * H + offs, mask=f_mask, other=0.0).to(
                tl.float32
            )
            cols = j * H + offs
            pre_t = tl.load(
                fn_ptr + n_idx[:, None] * fn_stride + cols[None, :],
                mask=f_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            pre_acc += tl.sum(pre_t * a[None, :], axis=1)
            post_t = tl.load(
                fn_ptr + (HC + n_idx)[:, None] * fn_stride + cols[None, :],
                mask=f_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            post_acc += tl.sum(post_t * a[None, :], axis=1)
            comb_t = tl.load(
                fn_ptr + comb_rows[:, :, None] * fn_stride + cols[None, None, :],
                mask=f_mask[None, None, :],
                other=0.0,
            ).to(tl.float32)
            comb_acc += tl.sum(comb_t * a[None, None, :], axis=2)
    return pre_acc, post_acc, comb_acc


@triton.jit
def _coeff_epilogue(
    pre_proj,
    post_proj,
    comb_proj,
    ssq,
    token,
    scale_ptr,
    base_ptr,
    pre_out_ptr,
    post_out_ptr,
    comb_out_ptr,
    K,
    RSUM_EPS: tl.constexpr,
    HC: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINK_REPEAT: tl.constexpr,
):
    """Turn one token's projection into the three coefficient groups.

    The arithmetic mirrors ``_mhc_shifted_coeff_kernel`` statement for
    statement; only the source of the raw projection differs.
    """
    idx = tl.arange(0, HC)
    comb_idx = idx[:, None] * HC + idx[None, :]
    scale_0 = tl.load(scale_ptr + 0)
    scale_1 = tl.load(scale_ptr + 1)
    scale_2 = tl.load(scale_ptr + 2)
    rms_inv = tl.rsqrt(ssq / K + RSUM_EPS)

    pre_logits = pre_proj * rms_inv * scale_0
    pre_logits += tl.load(base_ptr + idx)
    tl.store(pre_out_ptr + token * HC + idx, tl.sigmoid(pre_logits) + PRE_EPS)

    post_logits = post_proj * rms_inv
    post_logits = post_logits * scale_1 + tl.load(base_ptr + HC + idx)
    tl.store(post_out_ptr + token * HC + idx, tl.sigmoid(post_logits) * POST_MULT)

    comb = comb_proj * rms_inv
    comb = comb * scale_2 + tl.load(base_ptr + 2 * HC + comb_idx)
    comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None]
    comb = comb + SINK_EPS
    comb = comb / (tl.sum(comb, axis=0)[None, :] + SINK_EPS)
    for _ in range(SINK_REPEAT - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + SINK_EPS)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + SINK_EPS)
    tl.store(comb_out_ptr + token * HC * HC + comb_idx, comb)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_post_collapse_norm_hygon"),
    key=["H", "num_tokens"],
)
@triton.jit
def _mhc_shifted_post_collapse_norm_hygon_kernel(
    a_ptr,  # comb_res_mix : (N, 4, 4), float32 -- vLLM (j, i) orientation
    b_ptr,  # residual     : (N, 4, H), bfloat16
    c_ptr,  # post_layer_mix: (N, 4), float32
    d_ptr,  # x            : (N, H), bfloat16
    pre_ptr,  # pre_mix    : (N, 4), float32, or a dummy when USE_PRE is off
    out_ptr,  # residual_cur: (N, 4, H), bfloat16
    layer_ptr,  # layer_input: (N, H) bfloat16
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

    f0, f1, f2, f3, col_bf, layer_sq = _post_collapse_tile(
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        pre_ptr,
        out_ptr,
        layer_ptr,
        aux_ptr,
        w_ptr,
        pid_n,
        h_off,
        h_mask,
        a_base,
        c_base,
        b_base,
        d_base,
        out_base,
        H,
        a_stride_i,
        a_stride_j,
        USE_PRE,
        WRITE_AUX,
        HAS_W,
        EPS,
        STORE_RAW=False,
        HC=4,
    )

    if HAS_W:
        value = col_bf.to(tl.float32)
        inv = tl.rsqrt(tl.sum(layer_sq) / H + EPS)
        weight = tl.load(w_ptr + h_off, mask=h_mask, other=0.0).to(tl.float32)
        tl.store(
            layer_ptr + pid_n * H + h_off,
            (value * inv * weight).to(tl.bfloat16),
            mask=h_mask,
        )

    ssq = f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3
    tl.store(ssq_ptr + pid_n, tl.sum(ssq))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_post_collapse_norm_loop_hygon"),
    key=["H", "num_tokens"],
)
@triton.jit
def _mhc_shifted_post_collapse_norm_loop_hygon_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    pre_ptr,
    out_ptr,
    layer_ptr,
    aux_ptr,
    ssq_ptr,
    w_ptr,
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

    Only ``BLOCK_H`` stays live; the norm needs the full row's sum of squares,
    so it runs a second pass over the stored bf16 ``layer_input``.
    """
    pid_n = tl.program_id(0)

    a_base = pid_n * a_stride_n
    c_base = pid_n * 4
    b_base = pid_n * 4 * H
    d_base = pid_n * H
    out_base = pid_n * 4 * H

    sq = 0.0
    layer_sq = tl.zeros((1,), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        h_off = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_off < H
        f0, f1, f2, f3, col_bf, part_sq = _post_collapse_tile(
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr,
            pre_ptr,
            out_ptr,
            layer_ptr,
            aux_ptr,
            w_ptr,
            pid_n,
            h_off,
            h_mask,
            a_base,
            c_base,
            b_base,
            d_base,
            out_base,
            H,
            a_stride_i,
            a_stride_j,
            USE_PRE,
            WRITE_AUX,
            HAS_W,
            EPS,
            STORE_RAW=True,
            HC=4,
        )
        sq += tl.sum(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3)
        if HAS_W:
            layer_sq += part_sq

    tl.store(ssq_ptr + pid_n, sq)

    if HAS_W:
        # The first loop's layer_input stores must be visible to the second
        # loop's loads across the CTA's threads.
        tl.debug_barrier()
        inv = tl.rsqrt(tl.sum(layer_sq) / H + EPS)
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
    configs=runtime.get_tuned_config("mhc_shifted_all_hygon"),
    key=["H"],
)
@triton.jit
def _mhc_shifted_all_hygon_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    pre_ptr,
    out_ptr,
    layer_ptr,
    aux_ptr,
    w_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    pre_out_ptr,
    post_out_ptr,
    comb_out_ptr,
    a_stride_n,
    a_stride_j,
    a_stride_i,
    fn_stride,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_F: tl.constexpr,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
    HAS_W: tl.constexpr,
    EPS: tl.constexpr,
    RSUM_EPS: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINK_REPEAT: tl.constexpr,
):
    """Decode-sized batches: the whole boundary in one launch (flat rows)."""
    pid_n = tl.program_id(0)
    h_off = tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    a_base = pid_n * a_stride_n
    c_base = pid_n * 4
    b_base = pid_n * 4 * H
    d_base = pid_n * H
    out_base = pid_n * 4 * H

    f0, f1, f2, f3, col_bf, layer_sq = _post_collapse_tile(
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        pre_ptr,
        out_ptr,
        layer_ptr,
        aux_ptr,
        w_ptr,
        pid_n,
        h_off,
        h_mask,
        a_base,
        c_base,
        b_base,
        d_base,
        out_base,
        H,
        a_stride_i,
        a_stride_j,
        USE_PRE,
        WRITE_AUX,
        HAS_W,
        EPS,
        STORE_RAW=False,
        HC=4,
    )

    if HAS_W:
        value = col_bf.to(tl.float32)
        inv = tl.rsqrt(tl.sum(layer_sq) / H + EPS)
        weight = tl.load(w_ptr + h_off, mask=h_mask, other=0.0).to(tl.float32)
        tl.store(
            layer_ptr + pid_n * H + h_off,
            (value * inv * weight).to(tl.bfloat16),
            mask=h_mask,
        )

    ssq = tl.sum(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3)

    # The projection re-reads the row this CTA has just stored.
    tl.debug_barrier()
    pre_proj, post_proj, comb_proj = _project_rows(
        out_ptr, fn_ptr, pid_n, fn_stride, H, BLOCK_F, HC=4
    )
    _coeff_epilogue(
        pre_proj,
        post_proj,
        comb_proj,
        ssq,
        pid_n,
        scale_ptr,
        base_ptr,
        pre_out_ptr,
        post_out_ptr,
        comb_out_ptr,
        4 * H,
        RSUM_EPS,
        HC=4,
        PRE_EPS=PRE_EPS,
        SINK_EPS=SINK_EPS,
        POST_MULT=POST_MULT,
        SINK_REPEAT=SINK_REPEAT,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_all_loop_hygon"),
    key=["H"],
)
@triton.jit
def _mhc_shifted_all_loop_hygon_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    pre_ptr,
    out_ptr,
    layer_ptr,
    aux_ptr,
    w_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    pre_out_ptr,
    post_out_ptr,
    comb_out_ptr,
    a_stride_n,
    a_stride_j,
    a_stride_i,
    fn_stride,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_F: tl.constexpr,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
    HAS_W: tl.constexpr,
    EPS: tl.constexpr,
    RSUM_EPS: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINK_REPEAT: tl.constexpr,
):
    """Decode-sized batches with wide rows: the boundary in one launch."""
    pid_n = tl.program_id(0)

    a_base = pid_n * a_stride_n
    c_base = pid_n * 4
    b_base = pid_n * 4 * H
    d_base = pid_n * H
    out_base = pid_n * 4 * H

    sq = 0.0
    layer_sq = tl.zeros((1,), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        h_off = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_off < H
        f0, f1, f2, f3, col_bf, part_sq = _post_collapse_tile(
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr,
            pre_ptr,
            out_ptr,
            layer_ptr,
            aux_ptr,
            w_ptr,
            pid_n,
            h_off,
            h_mask,
            a_base,
            c_base,
            b_base,
            d_base,
            out_base,
            H,
            a_stride_i,
            a_stride_j,
            USE_PRE,
            WRITE_AUX,
            HAS_W,
            EPS,
            STORE_RAW=True,
            HC=4,
        )
        sq += tl.sum(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3)
        if HAS_W:
            layer_sq += part_sq

    if HAS_W:
        tl.debug_barrier()
        inv = tl.rsqrt(tl.sum(layer_sq) / H + EPS)
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

    tl.debug_barrier()
    pre_proj, post_proj, comb_proj = _project_rows(
        out_ptr, fn_ptr, pid_n, fn_stride, H, BLOCK_F, HC=4
    )
    _coeff_epilogue(
        pre_proj,
        post_proj,
        comb_proj,
        sq,
        pid_n,
        scale_ptr,
        base_ptr,
        pre_out_ptr,
        post_out_ptr,
        comb_out_ptr,
        4 * H,
        RSUM_EPS,
        HC=4,
        PRE_EPS=PRE_EPS,
        SINK_EPS=SINK_EPS,
        POST_MULT=POST_MULT,
        SINK_REPEAT=SINK_REPEAT,
    )


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

    The packed weight zero-pads the mix axis to ``BLOCK_N`` columns, so the B
    load never needs a column mask.
    """
    if M_FULL and EVEN_K:
        for k0 in range(k_begin, k_end, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :])
            b = tl.load(bt_ptr + offs_n[None, :] * bt_stride + offs_k[:, None])
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
                bt_ptr + offs_n[None, :] * bt_stride + offs_k[:, None],
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
            b = tl.load(bt_ptr + offs_n[None, :] * bt_stride + offs_k[:, None])
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
                bt_ptr + offs_n[None, :] * bt_stride + offs_k[:, None],
                mask=k_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(a, b)
    return acc


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_project_coeff_hygon"),
    key=["num_tokens", "hc_hidden_size", "split_k"],
)
@triton.jit
def _mhc_shifted_project_coeff_hygon_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (32, hc_hidden_size) bfloat16, contiguous
    part_ptr,  # (split_k, num_tokens, mix_size) float32
    cnt_ptr,  # (num_m_tiles,) int32, zeroed
    ssq_ptr,  # (num_tokens,) float32
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
    k_per_split,
    RSUM_EPS: tl.constexpr,
    HC: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINK_REPEAT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Split-K projection, then the last CTA of each row tile emits the
    coefficients, so the reduction and the Sinkhorn epilogue share one launch.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < num_tokens

    k_begin = pid_k * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, hc_hidden_size)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
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
        M_FULL=(pid_m + 1) * BLOCK_M <= num_tokens,
        EVEN_K=EVEN_K,
    )
    tl.store(
        part_ptr + (pid_k * num_tokens + offs_m[:, None]) * mix_size + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & (offs_n[None, :] < mix_size),
    )

    if split_k > 1:
        old = tl.atomic_add(cnt_ptr + pid_m, 1, sem="acq_rel", scope="gpu")
        last = old == split_k - 1
        if last:
            tl.store(cnt_ptr + pid_m, 0)
    else:
        last = True

    if last:
        # This CTA's own partial has to be visible to every warp before the
        # reduction reads it back; the atomics order the other CTAs.
        tl.debug_barrier()
        idx = tl.arange(0, HC)
        comb_idx = idx[:, None] * HC + idx[None, :]
        pre = tl.zeros((BLOCK_M, HC), dtype=tl.float32)
        post = tl.zeros((BLOCK_M, HC), dtype=tl.float32)
        comb = tl.zeros((BLOCK_M, HC, HC), dtype=tl.float32)
        for s in range(split_k):
            row = part_ptr + (s * num_tokens + offs_m[:, None]) * mix_size
            pre += tl.load(row + idx[None, :], mask=m_mask[:, None], other=0.0)
            post += tl.load(row + HC + idx[None, :], mask=m_mask[:, None], other=0.0)
            comb += tl.load(
                row[:, :, None] + 2 * HC + comb_idx[None, :, :],
                mask=m_mask[:, None, None],
                other=0.0,
            )

        rms_inv = tl.rsqrt(
            tl.load(ssq_ptr + offs_m, mask=m_mask, other=1.0) / hc_hidden_size
            + RSUM_EPS
        )
        scale_0 = tl.load(scale_ptr + 0)
        scale_1 = tl.load(scale_ptr + 1)
        scale_2 = tl.load(scale_ptr + 2)

        pre_logits = pre * rms_inv[:, None] * scale_0
        pre_logits += tl.load(base_ptr + idx)[None, :]
        tl.store(
            pre_out_ptr + offs_m[:, None] * HC + idx[None, :],
            tl.sigmoid(pre_logits) + PRE_EPS,
            mask=m_mask[:, None],
        )

        post_logits = post * rms_inv[:, None]
        post_logits = post_logits * scale_1 + tl.load(base_ptr + HC + idx)[None, :]
        tl.store(
            post_out_ptr + offs_m[:, None] * HC + idx[None, :],
            tl.sigmoid(post_logits) * POST_MULT,
            mask=m_mask[:, None],
        )

        comb = comb * rms_inv[:, None, None]
        comb = comb * scale_2 + tl.load(base_ptr + 2 * HC + comb_idx)[None, :, :]
        comb = tl.exp(comb - tl.max(comb, axis=2)[:, :, None])
        comb = comb / tl.sum(comb, axis=2)[:, :, None]
        comb = comb + SINK_EPS
        comb = comb / (tl.sum(comb, axis=1)[:, None, :] + SINK_EPS)
        for _ in range(SINK_REPEAT - 1):
            comb = comb / (tl.sum(comb, axis=2)[:, :, None] + SINK_EPS)
            comb = comb / (tl.sum(comb, axis=1)[:, None, :] + SINK_EPS)
        tl.store(
            comb_out_ptr + offs_m[:, None, None] * HC * HC + comb_idx[None, :, :],
            comb,
            mask=m_mask[:, None, None],
        )


def _split_k_for(num_tokens, hc_hidden_size):
    """Split K until the project grid fills the 80 BW1000 CUs.

    The projection is a memory-bound skinny GEMM; one CTA walking the whole
    ``hc * H`` axis is latency bound, so the split fills about one wave of the
    80 CUs while keeping at least ``_MIN_K_PER_SPLIT`` elements per split.

    Returns ``(split_k, k_per_split, even_k)``.  When ``hc * H`` is a multiple
    of ``_ALIGN_K`` the split is a divisor of the K-block count, so every CTA
    walks full ``BLOCK_K`` tiles and the projection uses the unmasked path.
    """
    est_m = max(1, triton.cdiv(num_tokens, 64))
    target = max(1, min(_PROJECT_MAX_SPLIT, triton.cdiv(_PROJECT_TARGET_CTAS, est_m)))
    target = min(target, max(1, hc_hidden_size // _MIN_K_PER_SPLIT))
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

    LibEntry rebuilds a specialization key from every argument on each call and
    the Hygon key path additionally asks the HIP backend for a per-tensor
    specialization; for these 15-20 argument kernels that costs 80-150 us, which
    dwarfs the kernels themselves on decode shapes.  The kernels here are
    fixed-shape and their tuner config is stable per key, so after the first
    (tuning) launch the compiled kernel is invoked directly (~6 us).  The
    fallback is the normal LibEntry path whenever the template cannot be
    reconstructed.
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
            full_grid = tuple(resolved) + (1, 1)
            if len(self.cache) >= 256:
                self.cache.clear()
            self.cache[key] = (
                compiled,
                template,
                full_grid,
            )
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
        stream = driver.active.get_current_stream(driver.active.get_current_device())
        enter = triton.knobs.runtime.launch_enter_hook
        exit_ = triton.knobs.runtime.launch_exit_hook
        launch_metadata = None
        if enter is not None:
            launch_metadata = compiled.launch_metadata(full_grid, stream, *all_args)
        compiled.run(
            full_grid[0],
            full_grid[1],
            full_grid[2],
            stream,
            compiled.function,
            compiled.packed_metadata,
            launch_metadata,
            enter,
            exit_,
            *all_args,
        )


_FAST_TOKEN = _FastLauncher("post_norm", _mhc_shifted_post_collapse_norm_hygon_kernel)
_FAST_TOKEN_LOOP = _FastLauncher(
    "post_norm_loop", _mhc_shifted_post_collapse_norm_loop_hygon_kernel
)
_FAST_ALL = _FastLauncher("all", _mhc_shifted_all_hygon_kernel)
_FAST_ALL_LOOP = _FastLauncher("all_loop", _mhc_shifted_all_loop_hygon_kernel)
_FAST_PROJECT_COEFF = _FastLauncher(
    "project_coeff", _mhc_shifted_project_coeff_hygon_kernel
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
    _FAST_TOKEN(
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
            block_h,
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
    """Wide-row variant; same outputs as the flat token kernel."""
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
    _FAST_TOKEN_LOOP(
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


def _all_stage(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn_packed,
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
    hidden_size,
    residual_cur,
    layer_input,
    aux,
    post_mix,
    comb_mix,
    next_pre,
):
    """One-launch boundary for decode-sized batches."""
    fn_stride = fn_packed.stride(0)
    st = comb_res_mix.stride()
    a_sn = st[0]
    a_sj, a_si = st[1], st[2]
    block_h = triton.next_power_of_2(hidden_size)
    loop = hidden_size > _TOKEN_FLAT_MAX_HIDDEN

    args = (
        comb_res_mix,
        residual,
        post_layer_mix,
        x,
        pre_mix if pre_mix is not None else post_layer_mix,
        residual_cur,
        layer_input,
        aux if capture_aux else layer_input,
        norm_weight if norm_weight is not None else layer_input,
        fn_packed,
        hc_scale,
        hc_base,
        next_pre,
        post_mix,
        comb_mix,
        a_sn,
        a_sj,
        a_si,
        fn_stride,
    )
    kwargs = dict(
        H=hidden_size,
        USE_PRE=pre_mix is not None,
        WRITE_AUX=capture_aux,
        HAS_W=norm_weight is not None,
        EPS=norm_eps,
        RSUM_EPS=rms_eps,
        PRE_EPS=hc_pre_eps,
        SINK_EPS=hc_sinkhorn_eps,
        POST_MULT=hc_post_mult_value,
        SINK_REPEAT=sinkhorn_repeat,
    )
    if not loop:
        kwargs["BLOCK_H"] = block_h
    scalars = (
        num_tokens,
        hidden_size,
        a_sn,
        a_sj,
        a_si,
        fn_stride,
        pre_mix is not None,
        capture_aux,
        norm_weight is not None,
        norm_eps,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    if loop:
        _FAST_ALL_LOOP(scalars + (block_h,), (num_tokens,), args, kwargs)
    else:
        _FAST_ALL(scalars + (block_h,), (num_tokens,), args, kwargs)


def _counter_buffer(num_tokens, device):
    """Zeroed arrival counters, one per M tile, reused across calls.

    The kernel resets a counter before it returns, so the buffer is zero at
    every entry; a shape that appears again reuses the same allocation.
    """
    key = num_tokens
    buf = _COUNTER_CACHE.get(key)
    if buf is not None and buf.device == device:
        return buf
    if len(_COUNTER_CACHE) >= 64:
        _COUNTER_CACHE.clear()
    buf = torch.zeros(triton.cdiv(num_tokens, 16), dtype=torch.int32, device=device)
    _COUNTER_CACHE[key] = buf
    return buf


def _project_coeff_stage(
    a_flat,
    fn_packed,
    ssq,
    split_k,
    k_per_split,
    even_k,
    hc_scale,
    hc_base,
    num_tokens,
    hc,
    hidden_size,
    mix_size,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    partials,
    post_mix,
    comb_mix,
    next_pre,
):
    """Split-K projection with the reduce/coefficient epilogue folded in."""
    hc_hidden_size = hc * hidden_size
    a_stride = a_flat.stride(0)
    fn_stride = fn_packed.stride(0)
    counters = _counter_buffer(num_tokens, a_flat.device)

    _FAST_PROJECT_COEFF(
        (
            num_tokens,
            hc_hidden_size,
            mix_size,
            a_stride,
            fn_stride,
            split_k,
            k_per_split,
            even_k,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        ),
        lambda meta: (
            triton.cdiv(num_tokens, meta["BLOCK_M"]),
            split_k,
        ),
        (
            a_flat,
            fn_packed,
            partials,
            counters,
            ssq,
            hc_scale,
            hc_base,
            next_pre,
            post_mix,
            comb_mix,
            num_tokens,
            hc_hidden_size,
            mix_size,
            a_stride,
            fn_stride,
            split_k,
            k_per_split,
        ),
        dict(
            RSUM_EPS=rms_eps,
            HC=hc,
            PRE_EPS=hc_pre_eps,
            SINK_EPS=hc_sinkhorn_eps,
            POST_MULT=hc_post_mult_value,
            SINK_REPEAT=sinkhorn_repeat,
            EVEN_K=even_k,
        ),
    )


def mhc_shifted_post_pre_hygon(
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
    """Full hc=4 boundary on Hygon, shared by every shape."""
    # Disable FlagTree aabs for the whole call so it cannot rewrite the tuned
    # tiles between the launches.
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
    """One or two launches for every Hygon shape.

    All outputs come from two backing allocations (one bfloat16, one float32)
    plus metadata views: each ``torch.empty`` and each ``as_strided`` costs a
    few microseconds here, and the decode shapes are launch/host bound.
    """
    device = residual.device
    hc_hidden_size = hc * hidden_size

    n_res = num_tokens * hc_hidden_size
    n_layer = num_tokens * hidden_size
    n_aux = n_layer if capture_aux else 0
    out_bf = torch.empty(n_res + n_layer + n_aux, dtype=torch.bfloat16, device=device)
    residual_cur = out_bf.as_strided(
        (num_tokens, hc, hidden_size),
        (hc_hidden_size, hidden_size, 1),
        0,
    )
    layer_input = out_bf.as_strided((num_tokens, hidden_size), (hidden_size, 1), n_res)
    aux = out_bf.as_strided(
        (0 if not capture_aux else num_tokens, hidden_size),
        (hidden_size, 1),
        n_res + n_layer,
    )

    n_post = num_tokens * hc
    n_comb = num_tokens * hc * hc
    fn_packed = _fn_bf16_packed(fn)

    one_launch_max = (
        _ONE_LAUNCH_MAX_TOKENS
        if hidden_size <= _TOKEN_FLAT_MAX_HIDDEN
        else _ONE_LAUNCH_MAX_TOKENS_WIDE
    )
    if num_tokens <= one_launch_max:
        out_f32 = torch.empty(
            n_post + n_comb + n_post, dtype=torch.float32, device=device
        )
        post_mix = out_f32.as_strided((num_tokens, hc, 1), (hc, 1, 1), 0)
        comb_mix = out_f32.as_strided((num_tokens, hc, hc), (hc * hc, hc, 1), n_post)
        next_pre = out_f32.as_strided((num_tokens, hc), (hc, 1), n_post + n_comb)
        _all_stage(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn_packed,
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
            hidden_size,
            residual_cur,
            layer_input,
            aux,
            post_mix,
            comb_mix,
            next_pre,
        )
        return residual_cur, post_mix, comb_mix, layer_input, next_pre, aux

    split_k, k_per_split, even_k = _split_k_for(num_tokens, hc_hidden_size)
    n_ssq = num_tokens
    n_part = split_k * num_tokens * mix_size
    out_f32 = torch.empty(
        n_ssq + n_part + n_post + n_comb + n_post,
        dtype=torch.float32,
        device=device,
    )
    ssq = out_f32.as_strided((num_tokens,), (1,), 0)
    partials = out_f32.as_strided(
        (split_k, num_tokens, mix_size),
        (num_tokens * mix_size, mix_size, 1),
        n_ssq,
    )
    post_mix = out_f32.as_strided((num_tokens, hc, 1), (hc, 1, 1), n_ssq + n_part)
    comb_mix = out_f32.as_strided(
        (num_tokens, hc, hc),
        (hc * hc, hc, 1),
        n_ssq + n_part + n_post,
    )
    next_pre = out_f32.as_strided(
        (num_tokens, hc),
        (hc, 1),
        n_ssq + n_part + n_post + n_comb,
    )

    token_kernel = (
        mhc_shifted_post_collapse_norm
        if hidden_size <= _TOKEN_FLAT_MAX_HIDDEN
        else mhc_shifted_post_collapse_norm_loop
    )
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
        residual_cur=residual_cur,
        layer_input=layer_input,
        aux=aux,
        ssq=ssq,
    )

    a_flat = residual_cur.view(num_tokens, hc_hidden_size)
    _project_coeff_stage(
        a_flat,
        fn_packed,
        ssq,
        split_k,
        k_per_split,
        even_k,
        hc_scale,
        hc_base,
        num_tokens,
        hc,
        hidden_size,
        mix_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        partials,
        post_mix,
        comb_mix,
        next_pre,
    )
    return residual_cur, post_mix, comb_mix, layer_input, next_pre, aux


__all__ = ["mhc_shifted_post_pre_hygon"]
