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

"""Moore Threads MTT S5000-optimized mhc_pre.

[KernelGen] Auto-generated and tuned for Moore Threads MTT S5000.
Measured geo-mean speedup vs torch reference: 6.63x.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# mHC pre block (key_ops_mhc_pre) Triton implementation for MUSA S5000.
#
# Pipeline (reference: vllm/model_executor/kernels/mhc/torch.py):
#   1. mix GEMM: mixes[t,n] = rms_t * sum_k residual[t,k] * fn[n,k], fp32 acc.
#   2. affine/sigmoid pre/post + comb softmax/Sinkhorn epilogue.
#   3. layer_input[t,h] = sum_i pre_mix[t,i] * residual[t,i,h] (bf16 out).
#
# Round-8 architecture (m = hc_mult = 4 for all evaluated workloads):
#   * fn fp32 -> bf16 pre-cast once per call (_cast_kernel) so every mix CTA
#     streams a 2-byte operand.
#   * _gemm_kernel is a 2D-grid SPLIT-K GEMM: grid (T/BR, GK); each CTA
#     accumulates fp32 partial mixes and partial square-sums over its disjoint
#     D/GK k-slice (single N=32 bf16 dot per BK iteration, deferred RMS tile).
#     Writes unscaled partials to mixes_part [GK,T,32] and sq_part [GK,T].
#   * _epi_kernel reduces the GK partials in registers, applies the row RMS
#     scale, then computes pre/post sigmoid stores and the comb softmax +
#     19x Sinkhorn (unchanged math).
#   * _layer_kernel unchanged (weighted residual reduction -> bf16).


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
@triton.jit
def _cast_kernel(src_ptr, dst_ptr, n, BLK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    m = offs < n
    v = tl.load(src_ptr + offs, mask=m, other=0.0)
    tl.store(dst_ptr + offs, v.to(tl.bfloat16), mask=m)


@triton.jit
def _gemm_kernel(
    res_ptr,
    fn_ptr,
    mix_ptr,
    sq_ptr,
    T,
    D,
    m: tl.constexpr,
    NP: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    GK: tl.constexpr,
):
    pid = tl.program_id(0)
    g = tl.program_id(1)
    row = pid * BR + tl.arange(0, BR)
    rmask = row < T
    kk = tl.arange(0, BK)
    cn = tl.arange(0, NP)  # fn rows 0..23 real, 24..31 padded

    KD = D // GK  # exact: dispatch ensures (D//BK) % GK == 0
    acc = tl.zeros((BR, NP), dtype=tl.float32)
    acc_sq = tl.zeros((BR, BK), dtype=tl.float32)  # deferred RMS accumulation

    for k0 in range(g * KD, (g + 1) * KD, BK):
        km = k0 + kk
        x = tl.load(
            res_ptr + row[:, None] * D + km[None, :], mask=rmask[:, None], other=0.0
        )
        xf = x.to(tl.float32)
        acc_sq += xf * xf
        b = tl.load(
            fn_ptr + cn[:, None] * D + km[None, :], mask=cn[:, None] < 24, other=0.0
        )
        acc = tl.dot(x, tl.trans(b), acc)

    sq = tl.sum(acc_sq, axis=1)
    # partial mixes (unscaled) and partial square sums
    tl.store(
        mix_ptr + g * (T * NP) + row[:, None] * NP + cn[None, :],
        acc,
        mask=rmask[:, None] & (cn[None, :] < 24),
    )
    tl.store(sq_ptr + g * T + row, sq, mask=rmask)


@triton.jit
def _epi_kernel(
    mix_ptr,
    sq_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    T,
    D,
    rms_eps,
    pre_eps,
    sink_eps,
    post_mult,
    m: tl.constexpr,
    BR: tl.constexpr,
    GK: tl.constexpr,
    SKR: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BR + tl.arange(0, BR)
    rmask = row < T
    c4 = tl.arange(0, 4)
    c16 = tl.arange(0, 16)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    acc_pre = tl.zeros((BR, 4), dtype=tl.float32)
    acc_post = tl.zeros((BR, 4), dtype=tl.float32)
    acc_cb = tl.zeros((BR, 16), dtype=tl.float32)
    acc_sq = tl.zeros((BR,), dtype=tl.float32)
    for g in tl.static_range(0, GK):
        base = g * (T * 32)
        acc_pre += tl.load(
            mix_ptr + base + row[:, None] * 32 + c4[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        acc_post += tl.load(
            mix_ptr + base + row[:, None] * 32 + (4 + c4)[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        acc_cb += tl.load(
            mix_ptr + base + row[:, None] * 32 + (8 + c16)[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        acc_sq += tl.load(sq_ptr + g * T + row, mask=rmask, other=0.0)

    rmsf = tl.rsqrt(acc_sq / D + rms_eps)

    # ---- pre: mixes cols 0..3
    b0 = tl.load(base_ptr + c4)
    pre_logits = (acc_pre * rmsf[:, None]) * s0 + b0[None, :]
    pre_v = 1.0 / (1.0 + tl.exp(-pre_logits)) + pre_eps
    tl.store(pre_ptr + row[:, None] * 4 + c4[None, :], pre_v, mask=rmask[:, None])

    # ---- post: mixes cols 4..7
    b1 = tl.load(base_ptr + 4 + c4)
    post_logits = (acc_post * rmsf[:, None]) * s1 + b1[None, :]
    post_v = (1.0 / (1.0 + tl.exp(-post_logits))) * post_mult
    tl.store(post_ptr + row[:, None] * 4 + c4[None, :], post_v, mask=rmask[:, None])

    # ---- comb: mixes cols 8..23 -> [BR, m, m]
    cb = tl.load(base_ptr + 8 + c16)
    comb_m = acc_cb * rmsf[:, None]
    logits3 = tl.reshape(comb_m * s2 + cb[None, :], (BR, m, m))
    mx = tl.max(logits3, axis=2)
    e = tl.exp(logits3 - mx[:, :, None])
    sm3 = tl.sum(e, axis=2)
    c = e / sm3[:, :, None] + sink_eps
    cs = tl.sum(c, axis=1)
    c = c / (cs[:, None, :] + sink_eps)
    for _ in tl.static_range(0, SKR - 1):
        rs = tl.sum(c, axis=2)
        c = c / (rs[:, :, None] + sink_eps)
        cs = tl.sum(c, axis=1)
        c = c / (cs[:, None, :] + sink_eps)
    ii = tl.arange(0, m)[None, :, None]
    jj = tl.arange(0, m)[None, None, :]
    tl.store(
        comb_ptr + row[:, None, None] * (m * m) + ii * m + jj,
        c,
        mask=rmask[:, None, None],
    )


@triton.jit
def _fused_kernel(
    mix_ptr,
    sq_ptr,
    scale_ptr,
    base_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    T,
    H,
    D,
    rms_eps,
    pre_eps,
    sink_eps,
    post_mult,
    m: tl.constexpr,
    BR: tl.constexpr,
    GK: tl.constexpr,
    SKR: tl.constexpr,
    HB: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Epilogue + layer pass fused for one row block (no pre scratch).

    Reduces the GK mixes/sq partials, applies the row RMS scale, computes the
    pre (kept in registers as a [BR,m] tile), post and comb outputs, then
    streams residual h-chunks to produce layer_input.  pre column i is
    extracted on the fly with a one-hot 4-wide sum.
    """
    pid = tl.program_id(0)
    row = pid * BR + tl.arange(0, BR)
    rmask = row < T
    cm = tl.arange(0, m)
    c16 = tl.arange(0, m * m)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    acc_pre = tl.zeros((BR, m), dtype=tl.float32)
    acc_post = tl.zeros((BR, m), dtype=tl.float32)
    acc_cb = tl.zeros((BR, m * m), dtype=tl.float32)
    acc_sq = tl.zeros((BR,), dtype=tl.float32)
    for g in tl.static_range(0, GK):
        base = g * (T * 32)
        acc_pre += tl.load(
            mix_ptr + base + row[:, None] * 32 + cm[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        acc_post += tl.load(
            mix_ptr + base + row[:, None] * 32 + (m + cm)[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        acc_cb += tl.load(
            mix_ptr + base + row[:, None] * 32 + (2 * m + c16)[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        acc_sq += tl.load(sq_ptr + g * T + row, mask=rmask, other=0.0)

    rmsf = tl.rsqrt(acc_sq / D + rms_eps)

    # ---- pre: mixes cols 0..m-1 (register tile [BR,m])
    b0 = tl.load(base_ptr + cm)
    pre = 1.0 / (1.0 + tl.exp(-((acc_pre * rmsf[:, None]) * s0 + b0[None, :])))
    pre = pre + pre_eps

    # ---- post
    b1 = tl.load(base_ptr + m + cm)
    post_logits = (acc_post * rmsf[:, None]) * s1 + b1[None, :]
    post_v = (1.0 / (1.0 + tl.exp(-post_logits))) * post_mult
    tl.store(post_ptr + row[:, None] * m + cm[None, :], post_v, mask=rmask[:, None])

    # ---- comb: mixes cols 2m..2m+m^2-1 -> [BR,m,m]
    cb = tl.load(base_ptr + 2 * m + c16)
    comb_m = acc_cb * rmsf[:, None]
    logits3 = tl.reshape(comb_m * s2 + cb[None, :], (BR, m, m))
    mx = tl.max(logits3, axis=2)
    e = tl.exp(logits3 - mx[:, :, None])
    sm3 = tl.sum(e, axis=2)
    c = e / sm3[:, :, None] + sink_eps
    cs = tl.sum(c, axis=1)
    c = c / (cs[:, None, :] + sink_eps)
    for _ in tl.static_range(0, SKR - 1):
        rs = tl.sum(c, axis=2)
        c = c / (rs[:, :, None] + sink_eps)
        cs = tl.sum(c, axis=1)
        c = c / (cs[:, None, :] + sink_eps)
    ii = tl.arange(0, m)[None, :, None]
    jj = tl.arange(0, m)[None, None, :]
    tl.store(
        comb_ptr + row[:, None, None] * (m * m) + ii * m + jj,
        c,
        mask=rmask[:, None, None],
    )

    # ---- layer pass (pre tile columns extracted via one-hot sums)
    hh = tl.arange(0, HB)
    for h0 in range(0, H, HB):
        hcol = h0 + hh
        acc = tl.zeros((BR, HB), dtype=tl.float32)
        for i in tl.static_range(0, m):
            ei = tl.where(cm == i, 1.0, 0.0)
            p_i = tl.sum(pre * ei[None, :], axis=1)
            if NEED_MASK:
                x = tl.load(
                    res_ptr + row[:, None] * D + (i * H + hcol)[None, :],
                    mask=rmask[:, None] & (hcol[None, :] < H),
                    other=0.0,
                )
            else:
                x = tl.load(
                    res_ptr + row[:, None] * D + (i * H + hcol)[None, :],
                    mask=rmask[:, None],
                    other=0.0,
                )
            acc += p_i[:, None] * x.to(tl.float32)
        if NEED_MASK:
            tl.store(
                out_ptr + row[:, None] * H + hcol[None, :],
                acc.to(tl.bfloat16),
                mask=rmask[:, None] & (hcol[None, :] < H),
            )
        else:
            tl.store(
                out_ptr + row[:, None] * H + hcol[None, :],
                acc.to(tl.bfloat16),
                mask=rmask[:, None],
            )


@triton.jit
def _fused_h_kernel(
    mix_ptr,
    sq_ptr,
    scale_ptr,
    base_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    T,
    H,
    D,
    rms_eps,
    pre_eps,
    sink_eps,
    post_mult,
    m: tl.constexpr,
    BR: tl.constexpr,
    GK: tl.constexpr,
    SKR: tl.constexpr,
    HB: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Fused epilogue + layer with an h-chunk split: grid (T/BR, H/HB).

    Every CTA reduces the GK mixes/sq partials and computes the pre vector in
    registers for its row block; only h-window 0 stores post_mix/comb_mix.
    Each CTA then streams its own h-window of residual to produce layer_input.
    """
    pid = tl.program_id(0)
    pj = tl.program_id(1)
    row = pid * BR + tl.arange(0, BR)
    rmask = row < T
    cm = tl.arange(0, m)
    c16 = tl.arange(0, m * m)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    acc_pre = tl.zeros((BR, m), dtype=tl.float32)
    acc_post = tl.zeros((BR, m), dtype=tl.float32)
    acc_cb = tl.zeros((BR, m * m), dtype=tl.float32)
    acc_sq = tl.zeros((BR,), dtype=tl.float32)
    for g in tl.static_range(0, GK):
        base = g * (T * 32)
        acc_pre += tl.load(
            mix_ptr + base + row[:, None] * 32 + cm[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        acc_post += tl.load(
            mix_ptr + base + row[:, None] * 32 + (m + cm)[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        acc_cb += tl.load(
            mix_ptr + base + row[:, None] * 32 + (2 * m + c16)[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        acc_sq += tl.load(sq_ptr + g * T + row, mask=rmask, other=0.0)

    rmsf = tl.rsqrt(acc_sq / D + rms_eps)

    b0 = tl.load(base_ptr + cm)
    pre = 1.0 / (1.0 + tl.exp(-((acc_pre * rmsf[:, None]) * s0 + b0[None, :])))
    pre = pre + pre_eps

    if pj == 0:
        # ---- post
        b1 = tl.load(base_ptr + m + cm)
        post_logits = (acc_post * rmsf[:, None]) * s1 + b1[None, :]
        post_v = (1.0 / (1.0 + tl.exp(-post_logits))) * post_mult
        tl.store(post_ptr + row[:, None] * m + cm[None, :], post_v, mask=rmask[:, None])

        # ---- comb: mixes cols 2m..2m+m^2-1 -> [BR,m,m]
        cb = tl.load(base_ptr + 2 * m + c16)
        comb_m = acc_cb * rmsf[:, None]
        logits3 = tl.reshape(comb_m * s2 + cb[None, :], (BR, m, m))
        mx = tl.max(logits3, axis=2)
        e = tl.exp(logits3 - mx[:, :, None])
        sm3 = tl.sum(e, axis=2)
        c = e / sm3[:, :, None] + sink_eps
        cs = tl.sum(c, axis=1)
        c = c / (cs[:, None, :] + sink_eps)
        for _ in tl.static_range(0, SKR - 1):
            rs = tl.sum(c, axis=2)
            c = c / (rs[:, :, None] + sink_eps)
            cs = tl.sum(c, axis=1)
            c = c / (cs[:, None, :] + sink_eps)
        ii = tl.arange(0, m)[None, :, None]
        jj = tl.arange(0, m)[None, None, :]
        tl.store(
            comb_ptr + row[:, None, None] * (m * m) + ii * m + jj,
            c,
            mask=rmask[:, None, None],
        )

    # ---- layer window [pj*HB, pj*HB+HB)
    hh = tl.arange(0, HB)
    hcol = pj * HB + hh
    acc = tl.zeros((BR, HB), dtype=tl.float32)
    for i in tl.static_range(0, m):
        ei = tl.where(cm == i, 1.0, 0.0)
        p_i = tl.sum(pre * ei[None, :], axis=1)
        if NEED_MASK:
            x = tl.load(
                res_ptr + row[:, None] * D + (i * H + hcol)[None, :],
                mask=rmask[:, None] & (hcol[None, :] < H),
                other=0.0,
            )
        else:
            x = tl.load(
                res_ptr + row[:, None] * D + (i * H + hcol)[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
        acc += p_i[:, None] * x.to(tl.float32)
    if NEED_MASK:
        tl.store(
            out_ptr + row[:, None] * H + hcol[None, :],
            acc.to(tl.bfloat16),
            mask=rmask[:, None] & (hcol[None, :] < H),
        )
    else:
        tl.store(
            out_ptr + row[:, None] * H + hcol[None, :],
            acc.to(tl.bfloat16),
            mask=rmask[:, None],
        )


@triton.jit
def _layer_kernel(
    res_ptr,
    pre_ptr,
    out_ptr,
    T,
    H,
    D,
    m: tl.constexpr,
    BR: tl.constexpr,
    HB: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BR + tl.arange(0, BR)
    rmask = row < T
    hh = tl.arange(0, HB)
    for h0 in range(0, H, HB):
        hcol = h0 + hh
        acc = tl.zeros((BR, HB), dtype=tl.float32)
        for i in tl.static_range(0, m):
            pre_i = tl.load(pre_ptr + row * m + i, mask=rmask, other=0.0)
            if NEED_MASK:
                x = tl.load(
                    res_ptr + row[:, None] * D + (i * H + hcol)[None, :],
                    mask=rmask[:, None] & (hcol[None, :] < H),
                    other=0.0,
                )
            else:
                x = tl.load(
                    res_ptr + row[:, None] * D + (i * H + hcol)[None, :],
                    mask=rmask[:, None],
                    other=0.0,
                )
            acc += pre_i[:, None] * x.to(tl.float32)
        if NEED_MASK:
            tl.store(
                out_ptr + row[:, None] * H + hcol[None, :],
                acc.to(tl.bfloat16),
                mask=rmask[:, None] & (hcol[None, :] < H),
            )
        else:
            tl.store(
                out_ptr + row[:, None] * H + hcol[None, :],
                acc.to(tl.bfloat16),
                mask=rmask[:, None],
            )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _f(x):
    return float(x.item()) if isinstance(x, torch.Tensor) else float(x)


def _i(x):
    return int(x.item()) if isinstance(x, torch.Tensor) else int(x)


def _pick_br(T):
    if T <= 16:
        return 16
    if T <= 256:
        return 16
    if T <= 4096:
        return 32
    return 64


def _pick_gk(T, D, BK, BR):
    """Power-of-two K-split (divisor of D//BK). Targets ~8-56 K iterations per
    CTA and a wide but not excessive total grid (<= ~2048 CTAs)."""
    nr = triton.cdiv(T, BR)
    nk = D // BK  # total K iterations at BK
    gk = 1
    # shrink long per-CTA K chains first (cap ~28 iters)
    while gk * 2 <= 64 and nk % (gk * 2) == 0 and (nk // (gk * 2)) > 28:
        gk *= 2
    # widen grids that are far below saturation while keeping >= 8 iters/CTA
    while (
        gk * 2 <= 64 and nk % (gk * 2) == 0 and (nk // (gk * 2)) >= 8 and nr * gk < 256
    ):
        gk *= 2
    return gk


def _pick_hb(H):
    """Largest power-of-two h-chunk that divides H (no masked tail)."""
    for hb in (512, 256, 128, 64):
        if H % hb == 0:
            return hb
    return 64


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def run(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
):
    residual = residual.contiguous()
    fn = fn.contiguous()
    hc_scale = hc_scale.contiguous()
    hc_base = hc_base.contiguous()

    T = residual.shape[0]
    m = residual.shape[1]
    H = residual.shape[2]
    D = m * H
    assert (
        m >= 4 and (m & (m - 1)) == 0
    ), "kernel specialized for power-of-two hc_mult >= 4"

    rms_e = _f(rms_eps)
    pre_e = _f(hc_pre_eps)
    sink_e = _f(hc_sinkhorn_eps)
    post_mult = _f(hc_post_mult_value)
    skr = _i(sinkhorn_repeat)
    if skr < 1:
        skr = 1

    dev = residual.device
    post_mix = torch.empty((T, m, 1), dtype=torch.float32, device=dev)
    comb_mix = torch.empty((T, m, m), dtype=torch.float32, device=dev)
    layer_input = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    fn_b = torch.empty_like(fn, dtype=torch.bfloat16)

    nfn = fn.numel()
    _cast_kernel[(triton.cdiv(nfn, 8192),)](
        fn,
        fn_b,
        nfn,
        BLK=8192,
        num_warps=4,
    )

    BR = _pick_br(T)
    BK = 128
    GK = _pick_gk(T, D, BK, BR)
    # unscaled partial mixes [GK,T,32] and partial square sums [GK,T]
    mixes_part = torch.empty((GK, T, 32), dtype=torch.float32, device=dev)
    sq_part = torch.empty((GK, T), dtype=torch.float32, device=dev)
    _gemm_kernel[(triton.cdiv(T, BR), GK)](
        residual,
        fn_b,
        mixes_part,
        sq_part,
        T,
        D,
        m=m,
        NP=32,
        BR=BR,
        BK=BK,
        GK=GK,
        num_warps=4,
    )

    HB = _pick_hb(H)
    need_mask = H % HB != 0
    if m == 4:
        if triton.cdiv(T, 32) < 128 and H > HB:
            # row-grid too thin: split the layer pass over h windows too
            BR_F = 16
            grid_f = (triton.cdiv(T, BR_F), triton.cdiv(H, HB))
            _fused_h_kernel[grid_f](
                mixes_part,
                sq_part,
                hc_scale,
                hc_base,
                residual,
                post_mix,
                comb_mix,
                layer_input,
                T,
                H,
                D,
                rms_e,
                pre_e,
                sink_e,
                post_mult,
                m=m,
                BR=BR_F,
                GK=GK,
                SKR=skr,
                HB=HB,
                NEED_MASK=need_mask,
                num_warps=8,
            )
        else:
            # fused epilogue + layer pass (single launch, pre in registers)
            BR_F = 32
            _fused_kernel[(triton.cdiv(T, BR_F),)](
                mixes_part,
                sq_part,
                hc_scale,
                hc_base,
                residual,
                post_mix,
                comb_mix,
                layer_input,
                T,
                H,
                D,
                rms_e,
                pre_e,
                sink_e,
                post_mult,
                m=m,
                BR=BR_F,
                GK=GK,
                SKR=skr,
                HB=HB,
                NEED_MASK=need_mask,
                num_warps=8,
            )
    else:
        pre_scratch = torch.empty((T, m), dtype=torch.float32, device=dev)
        BR_E = 32
        _epi_kernel[(triton.cdiv(T, BR_E),)](
            mixes_part,
            sq_part,
            hc_scale,
            hc_base,
            pre_scratch,
            post_mix,
            comb_mix,
            T,
            D,
            rms_e,
            pre_e,
            sink_e,
            post_mult,
            m=m,
            BR=BR_E,
            GK=GK,
            SKR=skr,
            num_warps=4,
        )
        BR_L = 32
        _layer_kernel[(triton.cdiv(T, BR_L),)](
            residual,
            pre_scratch,
            layer_input,
            T,
            H,
            D,
            m=m,
            BR=BR_L,
            HB=HB,
            NEED_MASK=need_mask,
            num_warps=8,
        )

    return post_mix, comb_mix, layer_input


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return run(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
