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

"""Ascend NPU specialization of the shifted mHC post -> pre boundary.

Only ``hc_mult == 4`` is specialized; every other ``hc_mult`` keeps the generic
Triton stages.  Four launches replace the generic five (post, project, coeff,
collapse, norm):

1. ``post + collapse + aux + ssq`` -- one TLE kernel.  It reuses the persistent
   ``mhc_post`` data flow (DSA bulk copies, one token per pipeline step) and,
   while the post-mapped streams are still in UB, applies the carried
   ``pre_mix``, writes the stream mean and accumulates the row sum of squares
   the coefficient stage needs.  ``pre_mix is None`` selects stream zero, as in
   the generic operator.
2. ``project`` -- ``mhc_pre``'s AB-swapped TLE projection (``fn @ a^T``) without
   its private sum-of-squares pass (step 1 already produced it).
3. ``coefficients`` -- ``mhc_pre``'s vectorised gate + Sinkhorn kernel.  The
   Sinkhorn rounds are element-wise vector arithmetic over the token axis.
4. ``epilogue`` -- re-emits ``next_pre`` from the coefficient kernel's
   mix-major layout and, when a weight is present, RMSNormalises the collapsed
   row.  One program per token, two passes over the hidden axis.

The ``comb_res_mix``/``comb_mix`` layout is the generic operator's vLLM
TRANSPOSED convention: ``comb[t, in_stream, out_head]``, which is exactly what
the AscendC/TLE post kernel reads as ``h_res[t, j, i]``.

For hidden sizes without a clean DSA tile (``_pick_block_d`` returns ``None``,
e.g. 127/1000/4095), the fused post falls back to a plain Triton variant with
the same arithmetic; production widths 1280/5120/7168 use the TLE kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._ascend.ops import mhc_pre as _mhc_pre
from flaggems_vllm.runtime.backend._ascend.ops.mhc_post import (
    _HAS_DSA,
    _aiv_core_count,
    _pick_block_d,
    tle,
)

# TLE is required: without ``triton.experimental.tle.dsa`` the host dispatch
# keeps the generic stages.
HAS_ASCEND_TLE = _HAS_DSA


# ---------------------------------------------------------------------------
# (1) Fused post + collapse + aux + row sum of squares
# ---------------------------------------------------------------------------
@triton.jit
def _ascend_fused_post_tle_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    pre_ptr,
    out_ptr,
    li_ptr,
    aux_ptr,
    sq_ptr,
    w_ptr,
    T,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
    HAS_W: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    """Persistent TLE post + collapse + aux + ssq, with optional fused RMSNorm.

    Only clean DSA tiles are launched here (``_pick_block_d`` guarantees
    ``D == BLOCK_D * NUM_D_BLOCKS``), so no tail masks are needed.  When
    ``HAS_W`` is set the collapsed row is kept in UB across the D loop, the
    row sum of squares is accumulated from the bf16-rounded values and a
    second pass writes the normalised row to ``li_ptr``; otherwise pass one
    writes the raw collapsed row there.  The residual output keeps the Ascend
    mhc_post FMA order, so it stays bitwise identical to
    ``flaggems_vllm.mhc_post``.
    """
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    x_dt = res_ptr.dtype.element_ty
    off = tl.arange(0, BLOCK_D)
    for tok in tle.dsa.pipeline(pid, T, npg, num_stages=2):
        xb = tok * 4 * D
        ob = tok * D
        hb = tok * 16
        pb = tok * 4
        hp0 = tl.load(post_ptr + pb + 0)
        hp1 = tl.load(post_ptr + pb + 1)
        hp2 = tl.load(post_ptr + pb + 2)
        hp3 = tl.load(post_ptr + pb + 3)
        c00 = tl.load(comb_ptr + hb + 0)
        c01 = tl.load(comb_ptr + hb + 1)
        c02 = tl.load(comb_ptr + hb + 2)
        c03 = tl.load(comb_ptr + hb + 3)
        c10 = tl.load(comb_ptr + hb + 4)
        c11 = tl.load(comb_ptr + hb + 5)
        c12 = tl.load(comb_ptr + hb + 6)
        c13 = tl.load(comb_ptr + hb + 7)
        c20 = tl.load(comb_ptr + hb + 8)
        c21 = tl.load(comb_ptr + hb + 9)
        c22 = tl.load(comb_ptr + hb + 10)
        c23 = tl.load(comb_ptr + hb + 11)
        c30 = tl.load(comb_ptr + hb + 12)
        c31 = tl.load(comb_ptr + hb + 13)
        c32 = tl.load(comb_ptr + hb + 14)
        c33 = tl.load(comb_ptr + hb + 15)
        if USE_PRE:
            p0 = tl.load(pre_ptr + pb + 0)
            p1 = tl.load(pre_ptr + pb + 1)
            p2 = tl.load(pre_ptr + pb + 2)
            p3 = tl.load(pre_ptr + pb + 3)
        if HAS_W:
            li_ub = tle.dsa.alloc([D], x_dt, tle.dsa.ascend.UB)
        sqv = tl.zeros([BLOCK_D], dtype=tl.float32)
        li_sqv = tl.zeros([BLOCK_D], dtype=tl.float32)
        for d in tl.static_range(0, NUM_D_BLOCKS):
            d_start = d * BLOCK_D
            d_off = d_start + off
            ho_ub = tle.dsa.alloc([BLOCK_D], x_dt, tle.dsa.ascend.UB)
            x0_ub = tle.dsa.alloc([BLOCK_D], x_dt, tle.dsa.ascend.UB)
            x1_ub = tle.dsa.alloc([BLOCK_D], x_dt, tle.dsa.ascend.UB)
            x2_ub = tle.dsa.alloc([BLOCK_D], x_dt, tle.dsa.ascend.UB)
            x3_ub = tle.dsa.alloc([BLOCK_D], x_dt, tle.dsa.ascend.UB)
            tle.dsa.copy(x_ptr + ob + d_off, ho_ub, [BLOCK_D], inter_no_alias=True)
            tle.dsa.copy(
                res_ptr + xb + 0 * D + d_off, x0_ub, [BLOCK_D], inter_no_alias=True
            )
            tle.dsa.copy(
                res_ptr + xb + 1 * D + d_off, x1_ub, [BLOCK_D], inter_no_alias=True
            )
            tle.dsa.copy(
                res_ptr + xb + 2 * D + d_off, x2_ub, [BLOCK_D], inter_no_alias=True
            )
            tle.dsa.copy(
                res_ptr + xb + 3 * D + d_off, x3_ub, [BLOCK_D], inter_no_alias=True
            )
            ho = tle.dsa.to_tensor(ho_ub, writable=False).to(tl.float32)
            x0 = tle.dsa.to_tensor(x0_ub, writable=False).to(tl.float32)
            x1 = tle.dsa.to_tensor(x1_ub, writable=False).to(tl.float32)
            x2 = tle.dsa.to_tensor(x2_ub, writable=False).to(tl.float32)
            x3 = tle.dsa.to_tensor(x3_ub, writable=False).to(tl.float32)
            y0 = tl.fma(c00, x0, hp0 * ho)
            y0 = tl.fma(c10, x1, y0)
            y0 = tl.fma(c20, x2, y0)
            y0 = tl.fma(c30, x3, y0)
            y1 = tl.fma(c01, x0, hp1 * ho)
            y1 = tl.fma(c11, x1, y1)
            y1 = tl.fma(c21, x2, y1)
            y1 = tl.fma(c31, x3, y1)
            y2 = tl.fma(c02, x0, hp2 * ho)
            y2 = tl.fma(c12, x1, y2)
            y2 = tl.fma(c22, x2, y2)
            y2 = tl.fma(c32, x3, y2)
            y3 = tl.fma(c03, x0, hp3 * ho)
            y3 = tl.fma(c13, x1, y3)
            y3 = tl.fma(c23, x2, y3)
            y3 = tl.fma(c33, x3, y3)
            b0 = y0.to(x_dt)
            b1 = y1.to(x_dt)
            b2 = y2.to(x_dt)
            b3 = y3.to(x_dt)
            tle.dsa.copy(
                tle.dsa.to_buffer(b0, tle.dsa.ascend.UB),
                out_ptr + xb + 0 * D + d_off,
                [BLOCK_D],
                inter_no_alias=True,
            )
            tle.dsa.copy(
                tle.dsa.to_buffer(b1, tle.dsa.ascend.UB),
                out_ptr + xb + 1 * D + d_off,
                [BLOCK_D],
                inter_no_alias=True,
            )
            tle.dsa.copy(
                tle.dsa.to_buffer(b2, tle.dsa.ascend.UB),
                out_ptr + xb + 2 * D + d_off,
                [BLOCK_D],
                inter_no_alias=True,
            )
            tle.dsa.copy(
                tle.dsa.to_buffer(b3, tle.dsa.ascend.UB),
                out_ptr + xb + 3 * D + d_off,
                [BLOCK_D],
                inter_no_alias=True,
            )
            r0 = b0.to(tl.float32)
            r1 = b1.to(tl.float32)
            r2 = b2.to(tl.float32)
            r3 = b3.to(tl.float32)
            if USE_PRE:
                li = p0 * r0 + p1 * r1 + p2 * r2 + p3 * r3
            else:
                li = r0
            lb = li.to(x_dt)
            lbf = lb.to(tl.float32)
            if HAS_W:
                sub = tle.dsa.subview(li_ub, (d_start,), (BLOCK_D,), (1,))
                tle.dsa.to_buffer(lb, tle.dsa.ascend.UB, sub)
                li_sqv += lbf * lbf
            else:
                tle.dsa.copy(
                    tle.dsa.to_buffer(lb, tle.dsa.ascend.UB),
                    li_ptr + ob + d_off,
                    [BLOCK_D],
                    inter_no_alias=True,
                )
            if WRITE_AUX:
                tb = ((r0 + r1 + r2 + r3) * 0.25).to(x_dt)
                tle.dsa.copy(
                    tle.dsa.to_buffer(tb, tle.dsa.ascend.UB),
                    aux_ptr + ob + d_off,
                    [BLOCK_D],
                    inter_no_alias=True,
                )
            sqv += r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3
        tl.store(sq_ptr + tok, tl.sum(sqv))
        if HAS_W:
            inv = tl.rsqrt(tl.sum(li_sqv) / D + NORM_EPS)
            li_t = tle.dsa.to_tensor(li_ub)
            for d in tl.static_range(0, NUM_D_BLOCKS):
                d_start = d * BLOCK_D
                d_off = d_start + off
                v = tl.reshape(
                    tle.dsa.extract_slice(li_t, (d_start,), (BLOCK_D,), (1,)), [BLOCK_D]
                ).to(tl.float32)
                wt = tl.load(w_ptr + d_off).to(tl.float32)
                r = v * inv * wt
                tle.dsa.copy(
                    tle.dsa.to_buffer(r.to(x_dt), tle.dsa.ascend.UB),
                    li_ptr + ob + d_off,
                    [BLOCK_D],
                    inter_no_alias=True,
                )


@triton.jit
def _ascend_fused_post_plain_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    pre_ptr,
    out_ptr,
    li_ptr,
    aux_ptr,
    sq_ptr,
    T,
    D,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Plain fallback for hidden sizes without a clean DSA tile."""
    tok = tl.program_id(0)
    hb = tok * 16
    pb = tok * 4
    hp0 = tl.load(post_ptr + pb + 0)
    hp1 = tl.load(post_ptr + pb + 1)
    hp2 = tl.load(post_ptr + pb + 2)
    hp3 = tl.load(post_ptr + pb + 3)
    c00 = tl.load(comb_ptr + hb + 0)
    c01 = tl.load(comb_ptr + hb + 1)
    c02 = tl.load(comb_ptr + hb + 2)
    c03 = tl.load(comb_ptr + hb + 3)
    c10 = tl.load(comb_ptr + hb + 4)
    c11 = tl.load(comb_ptr + hb + 5)
    c12 = tl.load(comb_ptr + hb + 6)
    c13 = tl.load(comb_ptr + hb + 7)
    c20 = tl.load(comb_ptr + hb + 8)
    c21 = tl.load(comb_ptr + hb + 9)
    c22 = tl.load(comb_ptr + hb + 10)
    c23 = tl.load(comb_ptr + hb + 11)
    c30 = tl.load(comb_ptr + hb + 12)
    c31 = tl.load(comb_ptr + hb + 13)
    c32 = tl.load(comb_ptr + hb + 14)
    c33 = tl.load(comb_ptr + hb + 15)
    if USE_PRE:
        p0 = tl.load(pre_ptr + pb + 0)
        p1 = tl.load(pre_ptr + pb + 1)
        p2 = tl.load(pre_ptr + pb + 2)
        p3 = tl.load(pre_ptr + pb + 3)

    sq = tl.zeros((1,), dtype=tl.float32)
    ob = tok * D
    xb = tok * 4 * D
    for d0 in range(0, D, BLOCK_D):
        off = d0 + tl.arange(0, BLOCK_D)
        m = off < D
        ho = tl.load(x_ptr + ob + off, mask=m, other=0.0).to(tl.float32)
        x0 = tl.load(res_ptr + xb + 0 * D + off, mask=m, other=0.0).to(tl.float32)
        x1 = tl.load(res_ptr + xb + 1 * D + off, mask=m, other=0.0).to(tl.float32)
        x2 = tl.load(res_ptr + xb + 2 * D + off, mask=m, other=0.0).to(tl.float32)
        x3 = tl.load(res_ptr + xb + 3 * D + off, mask=m, other=0.0).to(tl.float32)
        y0 = tl.fma(c00, x0, hp0 * ho)
        y0 = tl.fma(c10, x1, y0)
        y0 = tl.fma(c20, x2, y0)
        y0 = tl.fma(c30, x3, y0)
        y1 = tl.fma(c01, x0, hp1 * ho)
        y1 = tl.fma(c11, x1, y1)
        y1 = tl.fma(c21, x2, y1)
        y1 = tl.fma(c31, x3, y1)
        y2 = tl.fma(c02, x0, hp2 * ho)
        y2 = tl.fma(c12, x1, y2)
        y2 = tl.fma(c22, x2, y2)
        y2 = tl.fma(c32, x3, y2)
        y3 = tl.fma(c03, x0, hp3 * ho)
        y3 = tl.fma(c13, x1, y3)
        y3 = tl.fma(c23, x2, y3)
        y3 = tl.fma(c33, x3, y3)
        b0 = y0.to(tl.bfloat16)
        b1 = y1.to(tl.bfloat16)
        b2 = y2.to(tl.bfloat16)
        b3 = y3.to(tl.bfloat16)
        tl.store(out_ptr + xb + 0 * D + off, b0, mask=m)
        tl.store(out_ptr + xb + 1 * D + off, b1, mask=m)
        tl.store(out_ptr + xb + 2 * D + off, b2, mask=m)
        tl.store(out_ptr + xb + 3 * D + off, b3, mask=m)
        r0 = b0.to(tl.float32)
        r1 = b1.to(tl.float32)
        r2 = b2.to(tl.float32)
        r3 = b3.to(tl.float32)
        if USE_PRE:
            li = p0 * r0 + p1 * r1 + p2 * r2 + p3 * r3
        else:
            li = r0
        tl.store(li_ptr + ob + off, li.to(tl.bfloat16), mask=m)
        if WRITE_AUX:
            tl.store(
                aux_ptr + ob + off,
                ((r0 + r1 + r2 + r3) * 0.25).to(tl.bfloat16),
                mask=m,
            )
        sq += tl.sum(r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3)
    tl.store(sq_ptr + tok, tl.sum(sq))


@triton.jit
def _tle_row(t, r, BLOCK_T: tl.constexpr):
    """Row ``r`` of a ``(NC, BLOCK_T)`` UB tensor as a ``(BLOCK_T,)`` vector."""
    return tl.reshape(tle.dsa.extract_slice(t, (r, 0), (1, BLOCK_T), (1, 1)), [BLOCK_T])


@triton.jit
def _tle_comb_logit(
    t, r, rms_inv, scale, base, CLAMP_LOGITS: tl.constexpr, BLOCK_T: tl.constexpr
):
    """One comb logit over the token axis, clamped when the model requires it."""
    v = _tle_row(t, r, BLOCK_T) * rms_inv * scale + base
    if CLAMP_LOGITS:
        v = tl.clamp(v, -30.0, 30.0)
    return v


@triton.jit
def mhc_pre_ascend_tle_proj_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (hc_hidden_size, NC) bfloat16, fn^T zero padded to NC columns
    out_t_ptr,  # (NC, out_stride) float32, transposed mix layout
    sq_ptr,  # (num_tokens,) float32 -- row sum of squares of a
    num_tokens,
    hc_hidden_size,
    a_stride_n,
    bt_stride_k,
    out_stride,
    NC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_K_SQ: tl.constexpr,
):
    """``fn @ x^T`` (AB swapped), i.e. the transposed mix, plus the RMS sumsq.

    Swapping A/B is what makes the transpose free: the cube computes
    ``(NC, BLOCK_M)`` instead of ``(BLOCK_M, NC)``, so the epilogue reads one
    contiguous row per mix column and never transposes a tile.
    """
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, NC)
    m_mask = offs_m < num_tokens
    acc_t = tl.zeros((NC, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, hc_hidden_size, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hc_hidden_size
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # bt is zero padded to NC columns, so no column mask is needed.
        b = tl.load(
            bt_ptr + offs_k[:, None] * bt_stride_k + offs_n[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )
        acc_t += tl.dot(tl.trans(b), tl.trans(a))
    tl.store(
        out_t_ptr + offs_n[:, None] * out_stride + offs_m[None, :],
        acc_t,
        mask=m_mask[None, :],
    )
    # The sum of squares needs its own loop.  A vector reduction inside the dot
    # loop makes the compiler mis-schedule the MIX (AIC/AIV) loop: with a large
    # BLOCK_K it fails to tile (UB overflow + "'scf.for' op Failed to collect
    # vector loop tiling info"), and with a small one it compiles but the
    # reduction reads stale UB - sq comes back NaN/inf and changes between
    # identical runs, while the dot output stays bit-identical.  Keep the two
    # loops separate.
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, hc_hidden_size, BLOCK_K_SQ):
        offs_k = k0 + tl.arange(0, BLOCK_K_SQ)
        k_mask = offs_k < hc_hidden_size
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        a_f32 = a.to(tl.float32)
        sq += tl.sum(a_f32 * a_f32, 1)
    tl.store(sq_ptr + offs_m, sq, mask=m_mask)


@triton.jit
def _ascend_mix_kernel(
    mix_t_ptr,  # (NC, mix_stride) float32, transposed mix
    sq_ptr,  # (num_tokens,) float32
    hc_scale_ptr,  # (3,) float32
    hc_base_ptr,  # (hc_mult3,) float32
    post_t_ptr,  # (num_tokens, 4) float32, token-major public layout
    comb_t_ptr,  # (num_tokens, 16) float32, token-major public layout
    next_pre_ptr,  # (num_tokens, 4) float32, token-major public layout
    num_tokens,
    hc_hidden_size,
    mix_stride,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat,
    NC: tl.constexpr,
    BLOCK_T: tl.constexpr,
    CLAMP_LOGITS: tl.constexpr,
):
    """Gates + Sinkhorn for a tile of ``BLOCK_T`` tokens, one row per mix column.

    Every quantity here is a vector over the token axis, so the 20 Sinkhorn
    rounds are pure element-wise arithmetic: 4 row sums and 4 column sums are
    the only "reductions" and they are written as explicit vector adds (never
    ``tl.sum``), which is what keeps the cost down to ~35us at 4096x1280.
    """
    pid = tl.program_id(0)
    t0 = pid * BLOCK_T
    t_off = t0 + tl.arange(0, BLOCK_T)
    rows = tl.arange(0, NC)
    mask = t_off < num_tokens
    # The caller pads the sequence to BLOCK_T, so every program copies a full
    # tile: a partial inner extent changes the UB row stride and would scramble
    # the tile. Padding lanes are masked on store and never read back.
    mix_ub = tle.dsa.alloc(
        [NC, BLOCK_T], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(
            mix_t_ptr + rows[:, None] * mix_stride + t_off[None, :],
            mix_ub,
            [NC, BLOCK_T],
        )
    mt = tle.dsa.to_tensor(mix_ub)

    sq_ub = tle.dsa.alloc([BLOCK_T], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(sq_ptr + t_off, sq_ub, [BLOCK_T])
    sq = tle.dsa.to_tensor(sq_ub)
    rms_inv = tl.rsqrt(sq / hc_hidden_size + rms_eps)

    s0 = tl.load(hc_scale_ptr + 0)
    s1 = tl.load(hc_scale_ptr + 1)
    s2 = tl.load(hc_scale_ptr + 2)

    b0 = tl.load(hc_base_ptr + 0)
    b1 = tl.load(hc_base_ptr + 1)
    b2 = tl.load(hc_base_ptr + 2)
    b3 = tl.load(hc_base_ptr + 3)
    b4 = tl.load(hc_base_ptr + 4)
    b5 = tl.load(hc_base_ptr + 5)
    b6 = tl.load(hc_base_ptr + 6)
    b7 = tl.load(hc_base_ptr + 7)
    b8 = tl.load(hc_base_ptr + 8)
    b9 = tl.load(hc_base_ptr + 9)
    b10 = tl.load(hc_base_ptr + 10)
    b11 = tl.load(hc_base_ptr + 11)
    b12 = tl.load(hc_base_ptr + 12)
    b13 = tl.load(hc_base_ptr + 13)
    b14 = tl.load(hc_base_ptr + 14)
    b15 = tl.load(hc_base_ptr + 15)
    b16 = tl.load(hc_base_ptr + 16)
    b17 = tl.load(hc_base_ptr + 17)
    b18 = tl.load(hc_base_ptr + 18)
    b19 = tl.load(hc_base_ptr + 19)
    b20 = tl.load(hc_base_ptr + 20)
    b21 = tl.load(hc_base_ptr + 21)
    b22 = tl.load(hc_base_ptr + 22)
    b23 = tl.load(hc_base_ptr + 23)

    p0 = tl.sigmoid(_tle_row(mt, 0, BLOCK_T) * rms_inv * s0 + b0) + hc_pre_eps
    p1 = tl.sigmoid(_tle_row(mt, 1, BLOCK_T) * rms_inv * s0 + b1) + hc_pre_eps
    p2 = tl.sigmoid(_tle_row(mt, 2, BLOCK_T) * rms_inv * s0 + b2) + hc_pre_eps
    p3 = tl.sigmoid(_tle_row(mt, 3, BLOCK_T) * rms_inv * s0 + b3) + hc_pre_eps

    q0 = tl.sigmoid(_tle_row(mt, 4, BLOCK_T) * rms_inv * s1 + b4) * hc_post_mult_value
    q1 = tl.sigmoid(_tle_row(mt, 5, BLOCK_T) * rms_inv * s1 + b5) * hc_post_mult_value
    q2 = tl.sigmoid(_tle_row(mt, 6, BLOCK_T) * rms_inv * s1 + b6) * hc_post_mult_value
    q3 = tl.sigmoid(_tle_row(mt, 7, BLOCK_T) * rms_inv * s1 + b7) * hc_post_mult_value

    # Model contract: clamp the comb logits before Sinkhorn (CLAMP_LOGITS).
    c00 = _tle_comb_logit(mt, 8, rms_inv, s2, b8, CLAMP_LOGITS, BLOCK_T)
    c01 = _tle_comb_logit(mt, 9, rms_inv, s2, b9, CLAMP_LOGITS, BLOCK_T)
    c02 = _tle_comb_logit(mt, 10, rms_inv, s2, b10, CLAMP_LOGITS, BLOCK_T)
    c03 = _tle_comb_logit(mt, 11, rms_inv, s2, b11, CLAMP_LOGITS, BLOCK_T)
    c10 = _tle_comb_logit(mt, 12, rms_inv, s2, b12, CLAMP_LOGITS, BLOCK_T)
    c11 = _tle_comb_logit(mt, 13, rms_inv, s2, b13, CLAMP_LOGITS, BLOCK_T)
    c12 = _tle_comb_logit(mt, 14, rms_inv, s2, b14, CLAMP_LOGITS, BLOCK_T)
    c13 = _tle_comb_logit(mt, 15, rms_inv, s2, b15, CLAMP_LOGITS, BLOCK_T)
    c20 = _tle_comb_logit(mt, 16, rms_inv, s2, b16, CLAMP_LOGITS, BLOCK_T)
    c21 = _tle_comb_logit(mt, 17, rms_inv, s2, b17, CLAMP_LOGITS, BLOCK_T)
    c22 = _tle_comb_logit(mt, 18, rms_inv, s2, b18, CLAMP_LOGITS, BLOCK_T)
    c23 = _tle_comb_logit(mt, 19, rms_inv, s2, b19, CLAMP_LOGITS, BLOCK_T)
    c30 = _tle_comb_logit(mt, 20, rms_inv, s2, b20, CLAMP_LOGITS, BLOCK_T)
    c31 = _tle_comb_logit(mt, 21, rms_inv, s2, b21, CLAMP_LOGITS, BLOCK_T)
    c32 = _tle_comb_logit(mt, 22, rms_inv, s2, b22, CLAMP_LOGITS, BLOCK_T)
    c33 = _tle_comb_logit(mt, 23, rms_inv, s2, b23, CLAMP_LOGITS, BLOCK_T)

    # row softmax over j, then the first column normalisation
    m0 = tl.maximum(tl.maximum(c00, c01), tl.maximum(c02, c03))
    e00 = tl.exp(c00 - m0)
    e01 = tl.exp(c01 - m0)
    e02 = tl.exp(c02 - m0)
    e03 = tl.exp(c03 - m0)
    r0 = e00 + e01 + e02 + e03
    c00 = e00 / r0 + hc_sinkhorn_eps
    c01 = e01 / r0 + hc_sinkhorn_eps
    c02 = e02 / r0 + hc_sinkhorn_eps
    c03 = e03 / r0 + hc_sinkhorn_eps

    m1 = tl.maximum(tl.maximum(c10, c11), tl.maximum(c12, c13))
    e10 = tl.exp(c10 - m1)
    e11 = tl.exp(c11 - m1)
    e12 = tl.exp(c12 - m1)
    e13 = tl.exp(c13 - m1)
    r1 = e10 + e11 + e12 + e13
    c10 = e10 / r1 + hc_sinkhorn_eps
    c11 = e11 / r1 + hc_sinkhorn_eps
    c12 = e12 / r1 + hc_sinkhorn_eps
    c13 = e13 / r1 + hc_sinkhorn_eps

    m2 = tl.maximum(tl.maximum(c20, c21), tl.maximum(c22, c23))
    e20 = tl.exp(c20 - m2)
    e21 = tl.exp(c21 - m2)
    e22 = tl.exp(c22 - m2)
    e23 = tl.exp(c23 - m2)
    r2 = e20 + e21 + e22 + e23
    c20 = e20 / r2 + hc_sinkhorn_eps
    c21 = e21 / r2 + hc_sinkhorn_eps
    c22 = e22 / r2 + hc_sinkhorn_eps
    c23 = e23 / r2 + hc_sinkhorn_eps

    m3 = tl.maximum(tl.maximum(c30, c31), tl.maximum(c32, c33))
    e30 = tl.exp(c30 - m3)
    e31 = tl.exp(c31 - m3)
    e32 = tl.exp(c32 - m3)
    e33 = tl.exp(c33 - m3)
    r3 = e30 + e31 + e32 + e33
    c30 = e30 / r3 + hc_sinkhorn_eps
    c31 = e31 / r3 + hc_sinkhorn_eps
    c32 = e32 / r3 + hc_sinkhorn_eps
    c33 = e33 / r3 + hc_sinkhorn_eps

    cs0 = c00 + c10 + c20 + c30
    cs1 = c01 + c11 + c21 + c31
    cs2 = c02 + c12 + c22 + c32
    cs3 = c03 + c13 + c23 + c33
    i0 = 1.0 / (cs0 + hc_sinkhorn_eps)
    i1 = 1.0 / (cs1 + hc_sinkhorn_eps)
    i2 = 1.0 / (cs2 + hc_sinkhorn_eps)
    i3 = 1.0 / (cs3 + hc_sinkhorn_eps)
    c00 *= i0
    c10 *= i0
    c20 *= i0
    c30 *= i0
    c01 *= i1
    c11 *= i1
    c21 *= i1
    c31 *= i1
    c02 *= i2
    c12 *= i2
    c22 *= i2
    c32 *= i2
    c03 *= i3
    c13 *= i3
    c23 *= i3
    c33 *= i3

    for _ in range(sinkhorn_repeat - 1):
        r0 = c00 + c01 + c02 + c03
        i0 = 1.0 / (r0 + hc_sinkhorn_eps)
        c00 *= i0
        c01 *= i0
        c02 *= i0
        c03 *= i0
        r1 = c10 + c11 + c12 + c13
        i1 = 1.0 / (r1 + hc_sinkhorn_eps)
        c10 *= i1
        c11 *= i1
        c12 *= i1
        c13 *= i1
        r2 = c20 + c21 + c22 + c23
        i2 = 1.0 / (r2 + hc_sinkhorn_eps)
        c20 *= i2
        c21 *= i2
        c22 *= i2
        c23 *= i2
        r3 = c30 + c31 + c32 + c33
        i3 = 1.0 / (r3 + hc_sinkhorn_eps)
        c30 *= i3
        c31 *= i3
        c32 *= i3
        c33 *= i3

        cs0 = c00 + c10 + c20 + c30
        cs1 = c01 + c11 + c21 + c31
        cs2 = c02 + c12 + c22 + c32
        cs3 = c03 + c13 + c23 + c33
        i0 = 1.0 / (cs0 + hc_sinkhorn_eps)
        i1 = 1.0 / (cs1 + hc_sinkhorn_eps)
        i2 = 1.0 / (cs2 + hc_sinkhorn_eps)
        i3 = 1.0 / (cs3 + hc_sinkhorn_eps)
        c00 *= i0
        c10 *= i0
        c20 *= i0
        c30 *= i0
        c01 *= i1
        c11 *= i1
        c21 *= i1
        c31 *= i1
        c02 *= i2
        c12 *= i2
        c22 *= i2
        c32 *= i2
        c03 *= i3
        c13 *= i3
        c23 *= i3
        c33 *= i3

    # ``pre`` stays mix-major: it is the only gate the reduce stage reads, and
    # one contiguous row per head is exactly what that stage wants.
    pre_il = tl.reshape(
        tl.interleave(tl.interleave(p0, p2), tl.interleave(p1, p3)),
        [BLOCK_T, 4],
    )
    tl.store(
        next_pre_ptr + t_off[:, None] * 4 + tl.arange(0, 4)[None, :],
        pre_il,
        mask=mask[:, None],
    )

    # ``post`` / ``comb`` go back to the token-major public contract here, with
    # interleave trees whose leaves are the bit-reversed pairs (v_j, v_j + NC/2):
    # a single contiguous store each, instead of 20 scattered 4-byte stores
    # (measured ~20x slower) or an ATen transpose afterwards (~175us at 256
    # tokens, which would dominate short sequences).
    post_il = tl.reshape(
        tl.interleave(tl.interleave(q0, q2), tl.interleave(q1, q3)), [BLOCK_T, 4]
    )
    e00i = tl.interleave(c00, c20)
    e01i = tl.interleave(c10, c30)
    e10i = tl.interleave(c02, c22)
    e11i = tl.interleave(c12, c32)
    o00i = tl.interleave(c01, c21)
    o01i = tl.interleave(c11, c31)
    o10i = tl.interleave(c03, c23)
    o11i = tl.interleave(c13, c33)
    eei = tl.interleave(tl.interleave(e00i, e01i), tl.interleave(e10i, e11i))
    ooi = tl.interleave(tl.interleave(o00i, o01i), tl.interleave(o10i, o11i))
    comb_il = tl.reshape(tl.interleave(eei, ooi), [BLOCK_T, 16])

    tl.store(
        post_t_ptr + t_off[:, None] * 4 + tl.arange(0, 4)[None, :],
        post_il,
        mask=mask[:, None],
    )
    tl.store(
        comb_t_ptr + t_off[:, None] * 16 + tl.arange(0, 16)[None, :],
        comb_il,
        mask=mask[:, None],
    )


# ---------------------------------------------------------------------------
# (2) Projection without the sum-of-squares pass
# ---------------------------------------------------------------------------
@triton.jit
def _ascend_project_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (hc_hidden_size, NC) bfloat16, fn^T zero padded
    out_t_ptr,  # (NC, out_stride) float32
    sq_ptr,  # (num_tokens,) float32, or a dummy when COMPUTE_SQ is off
    num_tokens,
    hc_hidden_size,
    a_stride_n,
    bt_stride_k,
    out_stride,
    NC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_K_SQ: tl.constexpr,
    COMPUTE_SQ: tl.constexpr,
):
    """``fn @ a^T`` (AB swapped), i.e. the transposed mix.

    This is ``mhc_pre``'s projection with the row sum of squares made optional:
    the shifted boundary already accumulated it while the post-mapped streams
    were in UB, and the extra K pass costs a full re-read of ``a``.
    """
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, NC)
    m_mask = offs_m < num_tokens
    acc_t = tl.zeros((NC, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, hc_hidden_size, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hc_hidden_size
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            bt_ptr + offs_k[:, None] * bt_stride_k + offs_n[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )
        acc_t += tl.dot(tl.trans(b), tl.trans(a))
    tl.store(
        out_t_ptr + offs_n[:, None] * out_stride + offs_m[None, :],
        acc_t,
        mask=m_mask[None, :],
    )
    if COMPUTE_SQ:
        sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, hc_hidden_size, BLOCK_K_SQ):
            offs_k = k0 + tl.arange(0, BLOCK_K_SQ)
            k_mask = offs_k < hc_hidden_size
            a = tl.load(
                a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            a_f32 = a.to(tl.float32)
            sq += tl.sum(a_f32 * a_f32, 1)
        tl.store(sq_ptr + offs_m, sq, mask=m_mask)


# ---------------------------------------------------------------------------
# (3) Epilogue: next_pre (mix-major -> token-major) + optional RMSNorm
# ---------------------------------------------------------------------------
@triton.jit
def _ascend_norm_kernel(
    x_ptr,  # (T, D) bfloat16 collapsed row
    w_ptr,  # (D,) bfloat16
    out_ptr,  # (T, D) bfloat16
    num_tokens,
    hidden_size,
    norm_eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Two-pass RMSNorm, used only by the non-clean-tile fallback.

    Clean hidden sizes already normalise inside the fused post kernel.
    """
    row = tl.program_id(0)
    keep = row < num_tokens
    base = row * hidden_size
    ssq = tl.zeros((1,), dtype=tl.float32)
    for d0 in range(0, hidden_size, BLOCK_D):
        off = d0 + tl.arange(0, BLOCK_D)
        m = (off < hidden_size) & keep
        v = tl.load(x_ptr + base + off, mask=m, other=0.0).to(tl.float32)
        ssq += tl.sum(v * v, axis=0)
    inv = tl.rsqrt(ssq / hidden_size + norm_eps)
    for d0 in range(0, hidden_size, BLOCK_D):
        off = d0 + tl.arange(0, BLOCK_D)
        m = (off < hidden_size) & keep
        v = tl.load(x_ptr + base + off, mask=m, other=0.0).to(tl.float32)
        r = v * inv * tl.load(w_ptr + off, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + base + off, r.to(tl.bfloat16), mask=m)


def _plain_block_d(hidden_size: int) -> int:
    return min(4096, triton.next_power_of_2(hidden_size))


def mhc_shifted_post_pre_ascend(
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
    """Ascend ``hc_mult == 4`` path; arguments are already validated."""
    if hc != 4:
        raise ValueError(f"mhc_shifted_post_pre_ascend expects hc=4, got {hc}")
    device = residual.device

    # (1) post + collapse + stream mean + row sum of squares, one launch.
    residual_cur = torch.empty_like(residual)
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    aux = (
        torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        if capture_aux
        else residual_cur.new_empty(0, hidden_size)
    )
    # Token tile for the gate/Sinkhorn kernel: many small programs for short
    # sequences, wide tiles for long ones (measured on 910B4: 16 tokens below
    # 256, 64 up to 1024, 128 above).
    if num_tokens <= 256:
        block_t = 16
    elif num_tokens <= 1024:
        block_t = 64
    else:
        block_t = 128
    t_pad = triton.cdiv(num_tokens, block_t) * block_t
    # The coefficient kernel copies whole BLOCK_T tokens, so the sq padding
    # lanes must be addressable; only [0, num_tokens) is ever written.
    sq = torch.empty(t_pad, dtype=torch.float32, device=device)
    post_flat = post_layer_mix.reshape(num_tokens, hc)
    comb_flat = comb_res_mix.reshape(num_tokens, hc * hc)
    pre_flat = pre_mix if pre_mix is not None else post_flat

    has_norm = norm_weight is not None
    pick = _pick_block_d(hidden_size)
    if pick is not None:
        block_d, num_d_blocks = pick
        npg = min(num_tokens, _aiv_core_count())
        _ascend_fused_post_tle_kernel[(npg,)](
            x,
            residual,
            post_flat,
            comb_flat,
            pre_flat,
            residual_cur,
            layer_input,
            aux if capture_aux else layer_input,
            sq,
            norm_weight if has_norm else layer_input,
            num_tokens,
            hidden_size,
            block_d,
            num_d_blocks,
            pre_mix is not None,
            capture_aux,
            has_norm,
            norm_eps,
        )
    else:
        block_d = _plain_block_d(hidden_size)
        _ascend_fused_post_plain_kernel[(num_tokens,)](
            x,
            residual,
            post_flat,
            comb_flat,
            pre_flat,
            residual_cur,
            layer_input,
            aux if capture_aux else layer_input,
            sq,
            num_tokens,
            hidden_size,
            pre_mix is not None,
            capture_aux,
            block_d,
        )

    # (2) projection: fn @ a^T -> (NC, T), squares already computed above.
    a_flat = residual_cur.view(num_tokens, hc * hidden_size)
    # The cached pack checks ``fn._version``; inference tensors do not track a
    # version, so pack without the cache for them (they cannot be edited in
    # place anyway).
    if fn.is_inference():
        bt = _mhc_pre._build_bt_padded(fn)
    else:
        bt = _mhc_pre._get_bt_padded_cached(fn)
    mix_stride = t_pad
    mix_t = torch.empty(
        _mhc_pre._TLE_NC, mix_stride, dtype=torch.float32, device=device
    )
    block_m = _mhc_pre._tle_proj_block_m(num_tokens)
    # Wide K tiles only pay off while the M tile is small; at BLOCK_M=128 the
    # 1024-wide K tile is ~40% slower (the a tile no longer fits the cube).
    block_k = 1024 if block_m <= 32 else _mhc_pre._TLE_PROJ_BLOCK_K
    _ascend_project_kernel[(triton.cdiv(num_tokens, block_m),)](
        a_flat,
        bt,
        mix_t,
        sq,
        num_tokens,
        hc * hidden_size,
        a_flat.stride(0),
        bt.stride(0),
        mix_stride,
        NC=_mhc_pre._TLE_NC,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_K_SQ=_mhc_pre._TLE_PROJ_BLOCK_K_SQ,
        COMPUTE_SQ=False,
    )

    # (3) gates + Sinkhorn for the next boundary; next_pre is emitted
    # token-major by the same kernel.
    post_mix = torch.empty(num_tokens, hc, 1, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc, hc, dtype=torch.float32, device=device)
    next_pre = torch.empty(num_tokens, hc, dtype=torch.float32, device=device)
    _ascend_mix_kernel[(t_pad // block_t,)](
        mix_t,
        sq,
        hc_scale,
        hc_base,
        post_mix.view(num_tokens, hc),
        comb_mix.view(num_tokens, hc * hc),
        next_pre,
        num_tokens,
        hc * hidden_size,
        mix_stride,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        NC=_mhc_pre._TLE_NC,
        BLOCK_T=block_t,
        CLAMP_LOGITS=False,
        num_warps=2 if block_t >= 128 else 4,
    )

    # (4) Normalise the collapsed row for hidden sizes that could not use the
    # fused TLE post kernel.  Clean tiles already did it in kernel (1).
    if pick is None and norm_weight is not None:
        normed = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )
        _ascend_norm_kernel[(num_tokens,)](
            layer_input,
            norm_weight,
            normed,
            num_tokens,
            hidden_size,
            norm_eps=norm_eps,
            BLOCK_D=_plain_block_d(hidden_size),
        )
        layer_input = normed

    return residual_cur, post_mix, comb_mix, layer_input, next_pre, aux


__all__ = ["HAS_ASCEND_TLE", "mhc_shifted_post_pre_ascend"]
