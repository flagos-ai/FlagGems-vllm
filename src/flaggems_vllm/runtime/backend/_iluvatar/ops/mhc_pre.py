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

"""Iluvatar BI-V150-optimized mhc_pre.

[KernelGen] Auto-generated and tuned for Iluvatar BI-V150.
Measured geo-mean speedup vs torch reference: 14.59x.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# mHC pre block (key_ops_mhc_pre) in Triton for Iluvatar BI-V150.
#
# Architecture (v3)
# -----------------
# Same-session probes showed (a) predicated masks roughly double the cost of the
# bf16 dot/r2 loops and (b) a fine-grained 2D C-sliced K2 (BT2=32, BC2=64,
# num_warps=4) beats 1D token-grid K2 at every T (2.57ms vs 3.73ms at T=20480).
# So:
#
#   K0  : fn fp32 [h3, D] -> fnt bf16 [D, BN] (padded rows, contiguous)
#   Full-T path (T % 64 == 0, D % 64 == 0):
#     K1a : mixes_raw[t, j] = x[t] . fn[j]   maskless bf16 dot, fp32 acc
#     K1b : r2buf[t] = sum_d x[t,d]^2        maskless elementwise accumulate
#   Small/irregular-T path: split-K partial mixes + partial r2 + one reduce.
#   K2  : 2D grid (token tiles x C slices), BT2=32 BC2=64; cpid==0 programs
#         emit post/comb after sinkhorn; every program streams its residual
#         C-slice once for the pre-weighted bf16 layer reduction; gates are
#         scaled by r = rsqrt(r2buf/D + rms_eps).
#
# All accumulation fp32; GEMM operands bf16 (x native, fn cast in K0).


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 1 else 1


# ---------------------------------------------------------------------------
# K0: fn [h3, D] fp32 -> fnt [D, BN] bf16 (rows padded to BN, contiguous)
# ---------------------------------------------------------------------------
@triton.jit
def _fn_t_kernel(fn_ptr, fnt_ptr, D, h3, BD: tl.constexpr, BNJ: tl.constexpr):
    pid = tl.program_id(0)
    k0 = pid * BD
    kk = k0 + tl.arange(0, BD)
    jj = tl.arange(0, BNJ)
    kvalid = kk < D
    jmask = jj < h3
    src_off = jj[:, None] * D + kk[None, :]
    x = tl.load(fn_ptr + src_off, mask=jmask[:, None] & kvalid[None, :], other=0.0)
    x = tl.trans(x)
    xb = x.to(tl.bfloat16)
    dst_off = kk[:, None] * BNJ + jj[None, :]
    tl.store(fnt_ptr + dst_off, xb, mask=kvalid[:, None] & jmask[None, :])


# ---------------------------------------------------------------------------
# Full path, maskless (requires T % BT == 0, D % BK == 0)
# ---------------------------------------------------------------------------
@triton.jit
def _mix_dot_kernel(
    x_ptr,
    fnt_ptr,
    mix_ptr,
    T,
    D,
    h3,
    BT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    tt = pid * BT + tl.arange(0, BT)
    jj = tl.arange(0, BN)
    jmask = jj < h3
    acc = tl.zeros([BT, BN], dtype=tl.float32)
    kk0 = tl.arange(0, BK)
    for k in range(0, D, BK):
        xc = tl.load(x_ptr + tt[:, None] * D + (k + kk0)[None, :])
        fb = tl.load(fnt_ptr + (k + kk0)[:, None] * BN + jj[None, :])
        acc = tl.dot(xc, fb, acc=acc, out_dtype=tl.float32)
    tl.store(mix_ptr + tt[:, None] * h3 + jj[None, :], acc, mask=jmask[None, :])


@triton.jit
def _r2buf_kernel(x_ptr, r2_ptr, T, D, BT: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    tt = pid * BT + tl.arange(0, BT)
    r2acc = tl.zeros([BT, BK], dtype=tl.float32)
    kk0 = tl.arange(0, BK)
    for k in range(0, D, BK):
        xc = tl.load(x_ptr + tt[:, None] * D + (k + kk0)[None, :])
        xf = xc.to(tl.float32)
        r2acc += xf * xf
    tl.store(r2_ptr + tt, tl.sum(r2acc, axis=1))


# Full path, single launch: r2 pass then dot pass (two sequential loops)
@triton.jit
def _mix_r2_dot_kernel(
    x_ptr,
    fnt_ptr,
    mix_ptr,
    r2_ptr,
    T,
    D,
    h3,
    BT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    tt = pid * BT + tl.arange(0, BT)
    jj = tl.arange(0, BN)
    jmask = jj < h3
    kk0 = tl.arange(0, BK)
    # pass 1: r2buf (maskless, divisibility required)
    r2acc = tl.zeros([BT, BK], dtype=tl.float32)
    for k in range(0, D, BK):
        xc = tl.load(x_ptr + tt[:, None] * D + (k + kk0)[None, :])
        xf = xc.to(tl.float32)
        r2acc += xf * xf
    tl.store(r2_ptr + tt, tl.sum(r2acc, axis=1))
    # pass 2: mixes raw dot
    acc = tl.zeros([BT, BN], dtype=tl.float32)
    for k in range(0, D, BK):
        xc = tl.load(x_ptr + tt[:, None] * D + (k + kk0)[None, :])
        fb = tl.load(fnt_ptr + (k + kk0)[:, None] * BN + jj[None, :])
        acc = tl.dot(xc, fb, acc=acc, out_dtype=tl.float32)
    tl.store(mix_ptr + tt[:, None] * h3 + jj[None, :], acc, mask=jmask[None, :])


# ---------------------------------------------------------------------------
# Small / irregular-T path: split-K raw partials + partial r2 + reduce
# ---------------------------------------------------------------------------
@triton.jit
def _mix_partial_kernel(
    x_ptr,
    fnt_ptr,
    part_mix_ptr,
    part_r2_ptr,
    T,
    D,
    h3,
    num_tt,
    S: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tpid = tl.program_id(0)
    spid = tl.program_id(1)
    t0 = tpid * BT
    tt = t0 + tl.arange(0, BT)
    tmask = tt < T
    jj = tl.arange(0, BN)
    jmask = jj < h3
    acc = tl.zeros([BT, BN], dtype=tl.float32)
    r2acc = tl.zeros([BT, BK], dtype=tl.float32)
    nchunks = (D + BK - 1) // BK
    spb = (nchunks + S - 1) // S
    kstart = spid * spb * BK
    kend = tl.minimum(kstart + spb * BK, D)
    kk0 = tl.arange(0, BK)
    for k in range(kstart, kend, BK):
        kpos = k + kk0
        kvalid = kpos < kend
        xc = tl.load(
            x_ptr + tt[:, None] * D + kpos[None, :],
            mask=tmask[:, None] & kvalid[None, :],
            other=0.0,
        )
        fb = tl.load(
            fnt_ptr + kpos[:, None] * BN + jj[None, :],
            mask=kvalid[:, None] & jmask[None, :],
            other=0.0,
        )
        acc = tl.dot(xc, fb, acc=acc, out_dtype=tl.float32)
        xf = xc.to(tl.float32)
        r2acc += xf * xf
    p = spid * num_tt + tpid
    tl.store(
        part_mix_ptr + p * (BT * BN) + (tt - t0)[:, None] * BN + jj[None, :],
        acc,
        mask=tmask[:, None] & jmask[None, :],
    )
    r2 = tl.sum(r2acc, axis=1)
    tl.store(part_r2_ptr + p * BT + (tt - t0), r2, mask=tmask)


@triton.jit
def _mix_reduce_kernel(
    part_mix_ptr,
    part_r2_ptr,
    mix_ptr,
    r2_ptr,
    T,
    D,
    h3,
    num_tt,
    S,
    BT: tl.constexpr,
    BN: tl.constexpr,
):
    tpid = tl.program_id(0)
    tt = tpid * BT + tl.arange(0, BT)
    tmask = tt < T
    jj = tl.arange(0, BN)
    jmask = jj < h3
    acc = tl.zeros([BT, BN], dtype=tl.float32)
    r2 = tl.zeros([BT], dtype=tl.float32)
    for s in range(0, S):
        p = s * num_tt + tpid
        acc += tl.load(
            part_mix_ptr + p * (BT * BN) + tl.arange(0, BT)[:, None] * BN + jj[None, :],
            mask=tmask[:, None] & jmask[None, :],
            other=0.0,
        )
        r2 += tl.load(part_r2_ptr + p * BT + tl.arange(0, BT), mask=tmask, other=0.0)
    tl.store(
        mix_ptr + tt[:, None] * h3 + jj[None, :],
        acc,
        mask=tmask[:, None] & jmask[None, :],
    )
    tl.store(r2_ptr + tt, r2, mask=tmask)


# ---------------------------------------------------------------------------
# Mid/small exact-T path: single-launch maskless split-K with atomic partials
# (requires T % BT == 0, D % BK == 0, S | (D // BK)); no separate reduce pass
# ---------------------------------------------------------------------------
@triton.jit
def _mix_atomic_kernel(
    x_ptr,
    fnt_ptr,
    mix_ptr,
    r2_ptr,
    D,
    h3,
    num_tt,
    S: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tpid = tl.program_id(0)
    spid = tl.program_id(1)
    tt = tpid * BT + tl.arange(0, BT)
    jj = tl.arange(0, BN)
    jmask = jj < h3
    spb = (D // BK) // S
    kstart = spid * spb * BK
    kk0 = tl.arange(0, BK)
    acc = tl.zeros([BT, BN], dtype=tl.float32)
    for k in range(kstart, kstart + spb * BK, BK):
        xc = tl.load(x_ptr + tt[:, None] * D + (k + kk0)[None, :])
        fb = tl.load(fnt_ptr + (k + kk0)[:, None] * BN + jj[None, :])
        acc = tl.dot(xc, fb, acc=acc, out_dtype=tl.float32)
    tl.atomic_add(mix_ptr + tt[:, None] * h3 + jj[None, :], acc, mask=jmask[None, :])
    r2acc = tl.zeros([BT, BK], dtype=tl.float32)
    for k in range(kstart, kstart + spb * BK, BK):
        xc = tl.load(x_ptr + tt[:, None] * D + (k + kk0)[None, :])
        xf = xc.to(tl.float32)
        r2acc += xf * xf
    tl.atomic_add(r2_ptr + tt, tl.sum(r2acc, axis=1))


# ---------------------------------------------------------------------------
# Masked atomic split-K (tiny / irregular T): handles T % BT != 0 and D % BK
# != 0 via predicated loads; atomically accumulates mixes + r2 into zeroed
# global buffers so no separate reduce kernel is needed.
# ---------------------------------------------------------------------------
@triton.jit
def _mix_atomic_masked_kernel(
    x_ptr,
    fnt_ptr,
    mix_ptr,
    r2_ptr,
    T,
    D,
    h3,
    num_tt,
    S: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tpid = tl.program_id(0)
    spid = tl.program_id(1)
    t0 = tpid * BT
    tt = t0 + tl.arange(0, BT)
    tmask = tt < T
    jj = tl.arange(0, BN)
    jmask = jj < h3
    nchunks = (D + BK - 1) // BK
    spb = (nchunks + S - 1) // S
    kstart = spid * spb * BK
    kend = tl.minimum(kstart + spb * BK, D)
    kk0 = tl.arange(0, BK)
    acc = tl.zeros([BT, BN], dtype=tl.float32)
    for k in range(kstart, kend, BK):
        kpos = k + kk0
        kv = kpos < kend
        xc = tl.load(
            x_ptr + tt[:, None] * D + kpos[None, :],
            mask=tmask[:, None] & kv[None, :],
            other=0.0,
        )
        fb = tl.load(
            fnt_ptr + kpos[:, None] * BN + jj[None, :],
            mask=kv[:, None] & jmask[None, :],
            other=0.0,
        )
        acc = tl.dot(xc, fb, acc=acc, out_dtype=tl.float32)
    tl.atomic_add(
        mix_ptr + tt[:, None] * h3 + jj[None, :],
        acc,
        mask=tmask[:, None] & jmask[None, :],
    )
    r2acc = tl.zeros([BT, BK], dtype=tl.float32)
    for k in range(kstart, kend, BK):
        kpos = k + kk0
        kv = kpos < kend
        xc = tl.load(
            x_ptr + tt[:, None] * D + kpos[None, :],
            mask=tmask[:, None] & kv[None, :],
            other=0.0,
        )
        xf = xc.to(tl.float32)
        r2acc += xf * xf
    tl.atomic_add(r2_ptr + tt, tl.sum(r2acc, axis=1), mask=tmask)


# ---------------------------------------------------------------------------
# K2: gates (post / comb sinkhorn) + layer_input (2D: token tiles x C slices)
# ---------------------------------------------------------------------------
@triton.jit
def _gates_layer_kernel(
    mix_ptr,
    r2_ptr,
    x_ptr,
    base_ptr,
    scale_ptr,
    post_ptr,
    comb_ptr,
    layer_ptr,
    T,
    C,
    D,
    h,
    h3,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    H: tl.constexpr,
    HP: tl.constexpr,
    BT2: tl.constexpr,
    BC2: tl.constexpr,
):
    tpid = tl.program_id(0)
    cpid = tl.program_id(1)
    t0 = tpid * BT2
    tt = t0 + tl.arange(0, BT2)
    tmask = tt < T

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)
    pre_eps = hc_pre_eps
    seps = hc_sinkhorn_eps
    pm = hc_post_mult_value

    r2v = tl.load(r2_ptr + tt, mask=tmask, other=1.0)
    r = tl.rsqrt(r2v * (1.0 / D) + rms_eps)  # [BT2]

    if cpid == 0:
        # post_mix
        nn = tl.arange(0, HP)
        nmask = nn < H
        mix_post = tl.load(
            mix_ptr + tt[:, None] * h3 + (h + nn)[None, :],
            mask=tmask[:, None] & nmask[None, :],
            other=0.0,
        )
        b_post = tl.load(base_ptr + (h + nn), mask=nmask, other=0.0)
        post = tl.sigmoid((mix_post * r[:, None]) * s1 + b_post[None, :]) * pm
        tl.store(
            post_ptr + tt[:, None] * h + nn[None, :],
            post,
            mask=tmask[:, None] & nmask[None, :],
        )

        # comb_mix: softmax over b + eps, column-norm, repeat row/col norms
        aa = tl.arange(0, HP)
        bb = tl.arange(0, HP)
        amask = aa < H
        bmask = bb < H
        c2 = 2 * h
        mix_comb = tl.load(
            mix_ptr
            + tt[:, None, None] * h3
            + c2
            + aa[None, :, None] * h
            + bb[None, None, :],
            mask=tmask[:, None, None] & amask[None, :, None] & bmask[None, None, :],
            other=0.0,
        )
        b2d = tl.load(
            base_ptr + c2 + aa[:, None] * h + bb[None, :],
            mask=amask[:, None] & bmask[None, :],
            other=0.0,
        )
        b2r = tl.reshape(b2d, (1, HP, HP))
        m3 = tmask[:, None, None] & amask[None, :, None] & bmask[None, None, :]
        e = tl.exp((mix_comb * r[:, None, None]) * s2 + b2r)
        e = tl.where(m3, e, 0.0)
        rowsum = tl.sum(e, axis=2, keep_dims=True)
        comb = e / rowsum
        comb = tl.where(m3, comb, 0.0) + seps
        comb = tl.where(m3, comb, 0.0)
        colsum = tl.sum(comb, axis=1, keep_dims=True)
        comb = comb / (colsum + seps)
        for _ in range(sinkhorn_repeat - 1):
            rsum = tl.sum(comb, axis=2, keep_dims=True)
            comb = comb / (rsum + seps)
            csum = tl.sum(comb, axis=1, keep_dims=True)
            comb = comb / (csum + seps)
        tl.store(
            comb_ptr
            + tt[:, None, None] * (h * h)
            + aa[None, :, None] * h
            + bb[None, None, :],
            comb,
            mask=m3,
        )

    # layer_input over this C slice
    cc = cpid * BC2 + tl.arange(0, BC2)
    cmask = cc < C
    acc = tl.zeros([BT2, BC2], dtype=tl.float32)
    for n in range(0, HP):
        if n < H:
            m_n = tl.load(mix_ptr + tt * h3 + n, mask=tmask, other=0.0)
            b_n = tl.load(base_ptr + n)
            p_n = tl.sigmoid((m_n * r) * s0 + b_n) + pre_eps
            p2 = tl.reshape(p_n, (BT2, 1))
            res = tl.load(
                x_ptr + tt[:, None] * D + n * C + cc[None, :],
                mask=tmask[:, None] & cmask[None, :],
                other=0.0,
            )
            acc += res.to(tl.float32) * p2
    tl.store(
        layer_ptr + tt[:, None] * C + cc[None, :],
        acc.to(tl.bfloat16),
        mask=tmask[:, None] & cmask[None, :],
    )


# ---------------------------------------------------------------------------
# K2b: token-per-CTA variant: one program owns one token, computes post/comb
# gates, then streams the whole residual row in BC2-wide chunks (contiguous
# 2*BC2-byte segments per stream).  Faster at large T than the C-sliced grid.
# ---------------------------------------------------------------------------
@triton.jit
def _gates_layer_token_kernel(
    mix_ptr,
    r2_ptr,
    x_ptr,
    base_ptr,
    scale_ptr,
    post_ptr,
    comb_ptr,
    layer_ptr,
    T,
    C,
    D,
    h,
    h3,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    H: tl.constexpr,
    HP: tl.constexpr,
    BC2: tl.constexpr,
):
    t = tl.program_id(0)
    if t < T:
        s0 = tl.load(scale_ptr + 0)
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)
        seps = hc_sinkhorn_eps
        pm = hc_post_mult_value
        r = tl.rsqrt(tl.load(r2_ptr + t) * (1.0 / D) + rms_eps)

        nn = tl.arange(0, HP)
        nmask = nn < H
        mixrow = tl.load(mix_ptr + t * h3 + nn, mask=nmask, other=0.0)
        b_post = tl.load(base_ptr + h + nn, mask=nmask, other=0.0)
        post = tl.sigmoid((mixrow * r) * s1 + b_post) * pm
        tl.store(post_ptr + t * h + nn, post, mask=nmask)

        aa = tl.arange(0, HP)
        bb = tl.arange(0, HP)
        amask = aa < H
        bmask = bb < H
        mixc = tl.load(
            mix_ptr + t * h3 + 2 * h + aa[:, None] * h + bb[None, :],
            mask=amask[:, None] & bmask[None, :],
            other=0.0,
        )
        bc = tl.load(
            base_ptr + 2 * h + aa[:, None] * h + bb[None, :],
            mask=amask[:, None] & bmask[None, :],
            other=0.0,
        )
        m3 = amask[:, None] & bmask[None, :]
        e = tl.exp((mixc * r) * s2 + bc)
        e = tl.where(m3, e, 0.0)
        rowsum = tl.sum(e, axis=1, keep_dims=True)
        comb = e / rowsum
        comb = tl.where(m3, comb, 0.0) + seps
        comb = tl.where(m3, comb, 0.0)
        colsum = tl.sum(comb, axis=0, keep_dims=True)
        comb = comb / (colsum + seps)
        for _ in range(sinkhorn_repeat - 1):
            rsum = tl.sum(comb, axis=1, keep_dims=True)
            comb = comb / (rsum + seps)
            csum = tl.sum(comb, axis=0, keep_dims=True)
            comb = comb / (csum + seps)
        tl.store(comb_ptr + t * (h * h) + aa[:, None] * h + bb[None, :], comb, mask=m3)

        for c0 in range(0, C, BC2):
            cc = c0 + tl.arange(0, BC2)
            cmask = cc < C
            acc = tl.zeros([BC2], dtype=tl.float32)
            for n in range(0, HP):
                if n < H:
                    m_n = tl.load(mix_ptr + t * h3 + n)
                    b_n = tl.load(base_ptr + n)
                    p_n = tl.sigmoid((m_n * r) * s0 + b_n) + hc_pre_eps
                    res = tl.load(x_ptr + t * D + n * C + cc, mask=cmask, other=0.0)
                    acc += res.to(tl.float32) * p_n
            tl.store(layer_ptr + t * C + cc, acc.to(tl.bfloat16), mask=cmask)


# ---------------------------------------------------------------------------
# Host wrapper
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
    T, h, C = residual.shape
    H = int(h)
    D = H * C
    h3 = 2 * H + H * H
    device = residual.device

    rms_eps = float(rms_eps)
    hc_pre_eps = float(hc_pre_eps)
    hc_sinkhorn_eps = float(hc_sinkhorn_eps)
    hc_post_mult_value = float(hc_post_mult_value)
    sinkhorn_repeat = int(sinkhorn_repeat)

    BN = _next_pow2(h3)
    HP = _next_pow2(H)

    fnt = torch.empty((D, BN), device=device, dtype=torch.bfloat16)
    _fn_t_kernel[(triton.cdiv(D, 64),)](fn, fnt, D, h3, BD=64, BNJ=BN, num_warps=4)

    mixes = torch.empty((T, h3), device=device, dtype=torch.float32)
    r2buf = torch.empty((T,), device=device, dtype=torch.float32)

    BT1 = 32
    BK = 64
    num_tt = triton.cdiv(T, BT1)
    if (T % BT1 == 0) and (D % BK == 0) and T >= 64 and num_tt >= 256:
        # big exact: single-launch two-loop maskless (r2 pass + dot pass)
        _mix_r2_dot_kernel[(num_tt,)](
            residual,
            fnt,
            mixes,
            r2buf,
            T,
            D,
            h3,
            BT=BT1,
            BN=BN,
            BK=BK,
            num_warps=4,
            num_stages=2,
        )
    else:
        # single zeroed flat scratch holding mixes [T,h3] then r2buf [T]
        flat = torch.zeros((T * h3 + T,), device=device, dtype=torch.float32)
        mixes = flat[: T * h3].view(T, h3)
        r2buf = flat[T * h3 :]
        if (T % BT1 == 0) and (D % BK == 0) and T >= 64:
            # exact mid/small: maskless split-K with atomics. S = largest
            # divisor of (D//BK) yielding ~512-1024 CTAs, >=4 chunks/split.
            nchunks = D // BK
            cap = max(1, min(nchunks // 4, (1024 + num_tt - 1) // num_tt))
            S = 1
            for d in range(cap, 0, -1):
                if nchunks % d == 0:
                    S = d
                    break
            _mix_atomic_kernel[(num_tt, S)](
                residual,
                fnt,
                mixes,
                r2buf,
                D,
                h3,
                num_tt,
                S=S,
                BT=BT1,
                BN=BN,
                BK=BK,
                num_warps=4,
                num_stages=2,
            )
        else:
            # tiny/irregular: masked split-K with atomics (64-way parallel)
            nchunks = triton.cdiv(D, BK)
            S = min(64, nchunks)
            _mix_atomic_masked_kernel[(num_tt, S)](
                residual,
                fnt,
                mixes,
                r2buf,
                T,
                D,
                h3,
                num_tt,
                S=S,
                BT=BT1,
                BN=BN,
                BK=BK,
                num_warps=4,
                num_stages=2,
            )

    # outputs
    post_mix = torch.empty((T, h, 1), device=device, dtype=torch.float32)
    comb_mix = torch.empty((T, h, h), device=device, dtype=torch.float32)
    layer_input = torch.empty((T, C), device=device, dtype=torch.bfloat16)

    BT2 = 4
    BC2 = 256
    if T >= 1024:
        _gates_layer_token_kernel[(T,)](
            mixes,
            r2buf,
            residual,
            hc_base,
            hc_scale,
            post_mix,
            comb_mix,
            layer_input,
            T,
            C,
            D,
            h,
            h3,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            H=H,
            HP=HP,
            BC2=512,
        )
    else:
        _gates_layer_kernel[(triton.cdiv(T, BT2), triton.cdiv(C, BC2))](
            mixes,
            r2buf,
            residual,
            hc_base,
            hc_scale,
            post_mix,
            comb_mix,
            layer_input,
            T,
            C,
            D,
            h,
            h3,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            H=H,
            HP=HP,
            BT2=BT2,
            BC2=BC2,
            num_warps=2,
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
