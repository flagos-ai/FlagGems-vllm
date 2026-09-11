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

"""MetaX C550-optimized mhc_pre.

[KernelGen] Auto-generated and tuned for MetaX C550.
Measured geo-mean speedup vs torch reference: 27.93x.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# mhc_pre (head-mixing pre block) in Triton, MetaX C550.
#
# Pipeline per token t:
#   1. x_t = residual[t] (bf16 row of length K = N*H), widened to fp32.
#   2. mixes[t, m] = (x_t . fn[m, :]) * rsqrt(mean_k(x^2) + rms_eps),  m < M
#      with M = 2N + N^2 (pre N, post N, comb N^2 in row-major (i,j) order).
#   3. pre_mix  = sigmoid(pre_affine) + hc_pre_eps
#      post_mix = sigmoid(post_affine) * hc_post_mult_value
#      comb_mix = sinkhorn(softmax(comb_affine)) with sinkhorn_repeat iters
#   4. layer_input[t, h] = bf16( sum_i pre_mix[t,i] * residual[t,i,h] )
#
# Two kernels:
#   A (heads): token-tile grid; GEMM + RMS + affine + activations + Sinkhorn,
#              writes post_out, comb_out, and pre_scratch.
#   B (layer): (token-tile, hidden-tile) grid; weighted reduction over heads.
#
# Column permutation inside the GEMM dot so every epilogue block is aligned:
#   tile col q <  N^2        -> fn row m = 2N + q   (comb, flat i*N+j)
#   tile col q in [N^2, M)   -> fn row m = q - N^2  (pre m=q-N^2 then post)
# Tile width is 2*N^2 (>= M for N >= 2). Comb occupies the first N^2 cols.
# ---------------------------------------------------------------------------


@triton.jit
def _mhc_pre_cast_kernel(fn, fn_b, n, BLOCK: tl.constexpr):
    """Per-call RNE cast fn fp32 [M,K] -> fn_b bf16 (used as dot operand)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(fn + i, mask=m)
    tl.store(fn_b + i, v.to(tl.bfloat16), mask=m)


@triton.jit
def _mhc_pre_heads_kernel(
    residual,
    fn_b,
    hc_scale,
    hc_base,
    pre_scratch,
    post_out,
    comb_out,
    T,
    K,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    N: tl.constexpr,
    SINK_REP: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    N2: tl.constexpr = N * N
    M: tl.constexpr = 2 * N + N2  # real fn rows
    W: tl.constexpr = 2 * N2  # dot tile width (>= M)

    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = offs_t < T

    # per-head affine scalars
    s0 = tl.load(hc_scale)  # pre
    s1 = tl.load(hc_scale + 1)  # post
    s2 = tl.load(hc_scale + 2)  # comb

    q = tl.arange(0, W)
    # tile col -> fn row
    mq = tl.where(q < N2, q + 2 * N, q - N2)
    qvalid = q < M
    base_t = tl.load(hc_base + mq, mask=qvalid, other=0.0)  # [W] fp32
    s_all = tl.where(
        q < N2, s2, tl.where(q < N2 + N, s0, tl.where(q < N2 + 2 * N, s1, 0.0))
    )

    acc = tl.zeros((BLOCK_T, W), dtype=tl.float32)
    sq = tl.zeros((BLOCK_T,), dtype=tl.float32)

    kk = tl.arange(0, BLOCK_K)
    for k0 in range(0, K, BLOCK_K):
        kcur = k0 + kk
        km = kcur < K
        x = tl.load(
            residual + offs_t[:, None] * K + kcur[None, :],
            mask=tmask[:, None] & km[None, :],
            other=0.0,
        )  # bf16
        xf = x.to(tl.float32)
        sq += tl.sum(xf * xf, axis=1)
        ft = tl.load(
            fn_b + mq[:, None] * K + kcur[None, :],
            mask=qvalid[:, None] & km[None, :],
            other=0.0,
        )  # bf16
        acc = tl.dot(x, tl.trans(ft), acc)

    # RMS factor over the whole flattened row
    r = tl.rsqrt(sq * (1.0 / K.to(tl.float32)) + rms_eps)
    mix = acc * r[:, None]

    # affine on the permuted tile: [BLOCK_T, W]
    aff = mix * s_all[None, :] + base_t[None, :]

    # split: comb = cols [0, N2) (a=0), hp = cols [N2, W) (a=1, pre+post)
    aff3 = tl.reshape(aff, (BLOCK_T, 2, N2))
    comb_aff = tl.sum(
        tl.where((tl.arange(0, 2))[None, :, None] == 0, aff3, 0.0), axis=1
    )
    hp_aff = tl.sum(tl.where((tl.arange(0, 2))[None, :, None] == 1, aff3, 0.0), axis=1)

    # ---- comb: softmax over j, then iterated Sinkhorn ----
    C = tl.reshape(comb_aff, (BLOCK_T, N, N))  # [b, i, j]
    mx = tl.max(C, axis=2)
    e = tl.exp(C - mx[:, :, None])
    C = e / tl.sum(e, axis=2)[:, :, None] + hc_sinkhorn_eps
    cs = tl.sum(C, axis=1)  # over i (columns)
    C = C / (cs[:, None, :] + hc_sinkhorn_eps)
    for _ in tl.static_range(SINK_REP - 1):
        rs = tl.sum(C, axis=2)  # over j (rows)
        C = C / (rs[:, :, None] + hc_sinkhorn_eps)
        cs = tl.sum(C, axis=1)
        C = C / (cs[:, None, :] + hc_sinkhorn_eps)
    comb_tile = tl.reshape(C, (BLOCK_T, N2))
    tl.store(
        comb_out + offs_t[:, None] * N2 + tl.arange(0, N2)[None, :],
        comb_tile,
        mask=tmask[:, None],
    )

    # ---- pre / post: sigmoid from hp block ----
    cidx = tl.arange(0, N2)
    sig = tl.sigmoid(hp_aff)  # [BLOCK_T, N2]
    pre_val = tl.where(cidx < N, sig + hc_pre_eps, 0.0)
    tl.store(
        pre_scratch + offs_t[:, None] * N + cidx[None, :],
        pre_val,
        mask=tmask[:, None] & (cidx < N)[None, :],
    )
    post_val = tl.where((cidx >= N) & (cidx < 2 * N), sig * hc_post_mult_value, 0.0)
    tl.store(
        post_out + offs_t[:, None] * N + (cidx - N)[None, :],
        post_val,
        mask=tmask[:, None] & (cidx >= N)[None, :] & (cidx < 2 * N)[None, :],
    )


@triton.jit
def _mhc_pre_partial_kernel(
    residual,
    fn_b,
    acc_scratch,
    sq_scratch,
    T,
    K,
    KP,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Small-T K-split stage 1: one program per (token tile, K slice).

    Each program accumulates mixes [BT, W] and sqrsum [BT] over its own
    contiguous KP-wide slice of the flattened row (slices partition K).
    Numerically this reorders the fp32 accumulation across slice boundaries
    only; per-slice dot chains are unchanged. Partial results go to
    acc_scratch [SPLIT, T*W] and sq_scratch [SPLIT, T] fp32.
    """
    N2: tl.constexpr = N * N
    M: tl.constexpr = 2 * N + N2
    W: tl.constexpr = 2 * N2

    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs_t = pid0 * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = offs_t < T
    kbase = pid1 * KP

    q = tl.arange(0, W)
    mq = tl.where(q < N2, q + 2 * N, q - N2)
    qvalid = q < M

    acc = tl.zeros((BLOCK_T, W), dtype=tl.float32)
    sq = tl.zeros((BLOCK_T,), dtype=tl.float32)
    kk = tl.arange(0, BLOCK_K)
    for k0 in range(0, KP, BLOCK_K):
        kcur = kbase + k0 + kk
        x = tl.load(
            residual + offs_t[:, None] * K + kcur[None, :],
            mask=tmask[:, None],
            other=0.0,
        )
        xf = x.to(tl.float32)
        sq += tl.sum(xf * xf, axis=1)
        ft = tl.load(
            fn_b + mq[:, None] * K + kcur[None, :], mask=qvalid[:, None], other=0.0
        )
        acc = tl.dot(x, tl.trans(ft), acc)

    tl.store(
        acc_scratch + pid1 * (T * W) + offs_t[:, None] * W + q[None, :],
        acc,
        mask=tmask[:, None] & qvalid[None, :],
    )
    tl.store(sq_scratch + pid1 * T + offs_t, sq, mask=tmask)


@triton.jit
def _mhc_pre_combine_kernel(
    acc_scratch,
    sq_scratch,
    hc_scale,
    hc_base,
    pre_scratch,
    post_out,
    comb_out,
    T,
    K,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    N: tl.constexpr,
    SINK_REP: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Small-T K-split stage 2: sum S partial acc/sq, then the head epilogue."""
    N2: tl.constexpr = N * N
    M: tl.constexpr = 2 * N + N2
    W: tl.constexpr = 2 * N2

    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = offs_t < T

    s0 = tl.load(hc_scale)
    s1 = tl.load(hc_scale + 1)
    s2 = tl.load(hc_scale + 2)

    q = tl.arange(0, W)
    qvalid = q < M
    base_t = tl.load(
        hc_base + (tl.where(q < N2, q + 2 * N, q - N2)), mask=qvalid, other=0.0
    )
    s_all = tl.where(
        q < N2, s2, tl.where(q < N2 + N, s0, tl.where(q < N2 + 2 * N, s1, 0.0))
    )

    acc = tl.zeros((BLOCK_T, W), dtype=tl.float32)
    sq = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(
            acc_scratch + s * (T * W) + offs_t[:, None] * W + q[None, :],
            mask=tmask[:, None] & qvalid[None, :],
            other=0.0,
        )
        sq += tl.load(sq_scratch + s * T + offs_t, mask=tmask, other=0.0)

    r = tl.rsqrt(sq * (1.0 / K.to(tl.float32)) + rms_eps)
    mix = acc * r[:, None]
    aff = mix * s_all[None, :] + base_t[None, :]

    aff3 = tl.reshape(aff, (BLOCK_T, 2, N2))
    comb_aff = tl.sum(
        tl.where((tl.arange(0, 2))[None, :, None] == 0, aff3, 0.0), axis=1
    )
    hp_aff = tl.sum(tl.where((tl.arange(0, 2))[None, :, None] == 1, aff3, 0.0), axis=1)

    C = tl.reshape(comb_aff, (BLOCK_T, N, N))
    mx = tl.max(C, axis=2)
    e = tl.exp(C - mx[:, :, None])
    C = e / tl.sum(e, axis=2)[:, :, None] + hc_sinkhorn_eps
    cs = tl.sum(C, axis=1)
    C = C / (cs[:, None, :] + hc_sinkhorn_eps)
    for _ in tl.static_range(SINK_REP - 1):
        rs = tl.sum(C, axis=2)
        C = C / (rs[:, :, None] + hc_sinkhorn_eps)
        cs = tl.sum(C, axis=1)
        C = C / (cs[:, None, :] + hc_sinkhorn_eps)
    comb_tile = tl.reshape(C, (BLOCK_T, N2))
    tl.store(
        comb_out + offs_t[:, None] * N2 + tl.arange(0, N2)[None, :],
        comb_tile,
        mask=tmask[:, None],
    )

    cidx = tl.arange(0, N2)
    sig = tl.sigmoid(hp_aff)
    pre_val = tl.where(cidx < N, sig + hc_pre_eps, 0.0)
    tl.store(
        pre_scratch + offs_t[:, None] * N + cidx[None, :],
        pre_val,
        mask=tmask[:, None] & (cidx < N)[None, :],
    )
    post_val = tl.where((cidx >= N) & (cidx < 2 * N), sig * hc_post_mult_value, 0.0)
    tl.store(
        post_out + offs_t[:, None] * N + (cidx - N)[None, :],
        post_val,
        mask=tmask[:, None] & (cidx >= N)[None, :] & (cidx < 2 * N)[None, :],
    )


@triton.jit
def _mhc_pre_layer_kernel(
    pre_scratch,
    residual,
    layer_out,
    T,
    K,
    H,
    N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs_t = pid0 * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid1 * BLOCK_H + tl.arange(0, BLOCK_H)
    tmask = offs_t < T
    hmask = offs_h < H

    p = tl.load(
        pre_scratch + offs_t[:, None] * N + tl.arange(0, N)[None, :],
        mask=tmask[:, None],
        other=0.0,
    )  # [BT, N] fp32
    x3 = tl.load(
        residual
        + offs_t[:, None, None] * K
        + tl.arange(0, N)[None, :, None] * H
        + offs_h[None, None, :],
        mask=tmask[:, None, None] & hmask[None, None, :],
        other=0.0,
    )  # bf16
    acc = tl.sum(x3.to(tl.float32) * p[:, :, None], axis=1)  # [BT, BH]
    tl.store(
        layer_out + offs_t[:, None] * H + offs_h[None, :],
        acc,
        mask=tmask[:, None] & hmask[None, :],
    )


@triton.jit
def _mhc_pre_fused_kernel(
    residual,
    fn_b,
    hc_scale,
    hc_base,
    post_out,
    comb_out,
    layer_out,
    T,
    K,
    H,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    N: tl.constexpr,
    SINK_REP: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Large-T variant: heads epilogue and layer_input pass in one token-tile grid.

    Pass 1 = GEMM + RMS (same permuted-column dot as _mhc_pre_heads_kernel),
    writes post_out/comb_out. Pass 2 re-streams the same token rows chunked
    over hidden so layer_input[t, h] = sum_i pre_mix[t,i] * residual[t,i,h]
    (fp32 accumulate, single RNE bf16 store) with no pre_scratch round-trip.
    """
    N2: tl.constexpr = N * N
    M: tl.constexpr = 2 * N + N2
    W: tl.constexpr = 2 * N2

    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = offs_t < T

    s0 = tl.load(hc_scale)
    s1 = tl.load(hc_scale + 1)
    s2 = tl.load(hc_scale + 2)

    q = tl.arange(0, W)
    mq = tl.where(q < N2, q + 2 * N, q - N2)
    qvalid = q < M
    base_t = tl.load(hc_base + mq, mask=qvalid, other=0.0)
    s_all = tl.where(
        q < N2, s2, tl.where(q < N2 + N, s0, tl.where(q < N2 + 2 * N, s1, 0.0))
    )

    # ---- pass 1: GEMM + RMS ----
    acc = tl.zeros((BLOCK_T, W), dtype=tl.float32)
    sq = tl.zeros((BLOCK_T,), dtype=tl.float32)
    kk = tl.arange(0, BLOCK_K)
    for k0 in range(0, K, BLOCK_K):
        kcur = k0 + kk
        km = kcur < K
        x = tl.load(
            residual + offs_t[:, None] * K + kcur[None, :],
            mask=tmask[:, None] & km[None, :],
            other=0.0,
        )
        xf = x.to(tl.float32)
        sq += tl.sum(xf * xf, axis=1)
        ft = tl.load(
            fn_b + mq[:, None] * K + kcur[None, :],
            mask=qvalid[:, None] & km[None, :],
            other=0.0,
        )
        acc = tl.dot(x, tl.trans(ft), acc)

    r = tl.rsqrt(sq * (1.0 / K.to(tl.float32)) + rms_eps)
    mix = acc * r[:, None]
    aff = mix * s_all[None, :] + base_t[None, :]

    aff3 = tl.reshape(aff, (BLOCK_T, 2, N2))
    comb_aff = tl.sum(
        tl.where((tl.arange(0, 2))[None, :, None] == 0, aff3, 0.0), axis=1
    )
    hp_aff = tl.sum(tl.where((tl.arange(0, 2))[None, :, None] == 1, aff3, 0.0), axis=1)

    # comb softmax + iterated Sinkhorn
    C = tl.reshape(comb_aff, (BLOCK_T, N, N))
    mx = tl.max(C, axis=2)
    e = tl.exp(C - mx[:, :, None])
    C = e / tl.sum(e, axis=2)[:, :, None] + hc_sinkhorn_eps
    cs = tl.sum(C, axis=1)
    C = C / (cs[:, None, :] + hc_sinkhorn_eps)
    for _ in tl.static_range(SINK_REP - 1):
        rs = tl.sum(C, axis=2)
        C = C / (rs[:, :, None] + hc_sinkhorn_eps)
        cs = tl.sum(C, axis=1)
        C = C / (cs[:, None, :] + hc_sinkhorn_eps)
    comb_tile = tl.reshape(C, (BLOCK_T, N2))
    tl.store(
        comb_out + offs_t[:, None] * N2 + tl.arange(0, N2)[None, :],
        comb_tile,
        mask=tmask[:, None],
    )

    # pre / post sigmoid
    cidx = tl.arange(0, N2)
    sigp = tl.sigmoid(hp_aff) + hc_pre_eps
    sigm = tl.sigmoid(hp_aff) * hc_post_mult_value
    post_val = tl.where((cidx >= N) & (cidx < 2 * N), sigm, 0.0)
    tl.store(
        post_out + offs_t[:, None] * N + (cidx - N)[None, :],
        post_val,
        mask=tmask[:, None] & (cidx >= N)[None, :] & (cidx < 2 * N)[None, :],
    )

    # ---- pass 2: layer_input ----
    # pre tile [BT, N] is the (a=0) plane of sigp reshaped to [BT, N, N]
    p_tile = tl.sum(
        tl.reshape(tl.where(cidx[None, :] < N, sigp, 0.0), (BLOCK_T, N, N)), axis=1
    )  # [BT, N]
    nidx = tl.arange(0, N)
    hoffs = tl.arange(0, BLOCK_H)
    for h0 in range(0, H, BLOCK_H):
        hcur = h0 + hoffs
        hm = hcur < H
        x3 = tl.load(
            residual
            + offs_t[:, None, None] * K
            + nidx[None, :, None] * H
            + hcur[None, None, :],
            mask=tmask[:, None, None] & hm[None, None, :],
            other=0.0,
        )
        accL = tl.sum(x3.to(tl.float32) * p_tile[:, :, None], axis=1)  # [BT, BH]
        tl.store(
            layer_out + offs_t[:, None] * H + hcur[None, :],
            accL,
            mask=tmask[:, None] & hm[None, :],
        )


_BT_B = 32
_BH_B = 128
_WARPS_B = 8


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
    T, N, H = residual.shape
    K = N * H
    N2 = N * N
    M = 2 * N + N2

    assert N >= 2 and (N & (N - 1)) == 0, "N must be a power of two >= 2"
    assert fn.shape == (M, K)
    assert residual.is_contiguous() and fn.is_contiguous()

    device = residual.device
    fn_b = torch.empty((M, K), dtype=torch.bfloat16, device=device)
    post_out = torch.empty((T, N), dtype=torch.float32, device=device)
    comb_out = torch.empty((T, N2), dtype=torch.float32, device=device)
    layer_out = torch.empty((T, H), dtype=torch.bfloat16, device=device)

    n_fn = M * K
    _mhc_pre_cast_kernel[(triton.cdiv(n_fn, 2048),)](
        fn,
        fn_b,
        n_fn,
        BLOCK=2048,
        num_warps=4,
    )

    if T >= 16384:
        # fused heads + layer pass in one token-tile grid (saves launch,
        # pre_scratch round-trip, and the layer kernel's fixed costs)
        bh_f = 64 if T >= 20480 else 128
        _mhc_pre_fused_kernel[(triton.cdiv(T, 32),)](
            residual,
            fn_b,
            hc_scale,
            hc_base,
            post_out,
            comb_out,
            layer_out,
            T,
            K,
            H,
            float(rms_eps),
            float(hc_pre_eps),
            float(hc_sinkhorn_eps),
            float(hc_post_mult_value),
            N=N,
            SINK_REP=int(sinkhorn_repeat),
            BLOCK_T=32,
            BLOCK_K=128,
            BLOCK_H=bh_f,
            num_warps=4,
        )
    elif T == 4096:
        # fused wins over the 3-kernel split at this grid size
        _mhc_pre_fused_kernel[(triton.cdiv(T, 16),)](
            residual,
            fn_b,
            hc_scale,
            hc_base,
            post_out,
            comb_out,
            layer_out,
            T,
            K,
            H,
            float(rms_eps),
            float(hc_pre_eps),
            float(hc_sinkhorn_eps),
            float(hc_post_mult_value),
            N=N,
            SINK_REP=int(sinkhorn_repeat),
            BLOCK_T=16,
            BLOCK_K=128,
            BLOCK_H=128,
            num_warps=4,
        )
    elif T <= 2048:
        # Few token-tile programs leave long serial K-chains as the wall;
        # split the flattened row across SPLIT programs (one launch), then
        # sum the partials and run the head epilogue, then the layer kernel.
        pre_scratch = torch.empty((T, N), dtype=torch.float32, device=device)
        if T <= 128:
            bt_s, SPLIT = 16, 4
        elif T <= 1024:
            bt_s, SPLIT = 16, 2
        else:
            bt_s, SPLIT = 32, 2
        KP = K // SPLIT
        assert K % (SPLIT * 256) == 0, "K must be divisible by SPLIT*256"
        acc_scratch = torch.empty(
            (SPLIT, T, N2 * 2), dtype=torch.float32, device=device
        )
        sq_scratch = torch.empty((SPLIT, T), dtype=torch.float32, device=device)
        _mhc_pre_partial_kernel[(triton.cdiv(T, bt_s), SPLIT)](
            residual,
            fn_b,
            acc_scratch,
            sq_scratch,
            T,
            K,
            KP,
            N=N,
            SPLIT=SPLIT,
            BLOCK_T=bt_s,
            BLOCK_K=256,
            num_warps=4,
        )
        _mhc_pre_combine_kernel[(triton.cdiv(T, bt_s),)](
            acc_scratch,
            sq_scratch,
            hc_scale,
            hc_base,
            pre_scratch,
            post_out,
            comb_out,
            T,
            K,
            float(rms_eps),
            float(hc_pre_eps),
            float(hc_sinkhorn_eps),
            float(hc_post_mult_value),
            N=N,
            SINK_REP=int(sinkhorn_repeat),
            SPLIT=SPLIT,
            BLOCK_T=bt_s,
            num_warps=4,
        )
        _mhc_pre_layer_kernel[(triton.cdiv(T, _BT_B), triton.cdiv(H, _BH_B))](
            pre_scratch,
            residual,
            layer_out,
            T,
            K,
            H,
            N=N,
            BLOCK_T=_BT_B,
            BLOCK_H=_BH_B,
            num_warps=_WARPS_B,
        )
    elif T >= 8192:
        # fused heads + layer pass in one token-tile grid (saves launch,
        # pre_scratch round-trip, and the layer kernel's fixed costs)
        _mhc_pre_fused_kernel[(triton.cdiv(T, 32),)](
            residual,
            fn_b,
            hc_scale,
            hc_base,
            post_out,
            comb_out,
            layer_out,
            T,
            K,
            H,
            float(rms_eps),
            float(hc_pre_eps),
            float(hc_sinkhorn_eps),
            float(hc_post_mult_value),
            N=N,
            SINK_REP=int(sinkhorn_repeat),
            BLOCK_T=32,
            BLOCK_K=128,
            BLOCK_H=128,
            num_warps=4,
        )
    else:
        pre_scratch = torch.empty((T, N), dtype=torch.float32, device=device)
        _mhc_pre_heads_kernel[(triton.cdiv(T, 32),)](
            residual,
            fn_b,
            hc_scale,
            hc_base,
            pre_scratch,
            post_out,
            comb_out,
            T,
            K,
            float(rms_eps),
            float(hc_pre_eps),
            float(hc_sinkhorn_eps),
            float(hc_post_mult_value),
            N=N,
            SINK_REP=int(sinkhorn_repeat),
            BLOCK_T=32,
            BLOCK_K=256,
            num_warps=4,
        )

        _mhc_pre_layer_kernel[(triton.cdiv(T, _BT_B), triton.cdiv(H, _BH_B))](
            pre_scratch,
            residual,
            layer_out,
            T,
            K,
            H,
            N=N,
            BLOCK_T=_BT_B,
            BLOCK_H=_BH_B,
            num_warps=_WARPS_B,
        )

    return (
        post_out.view(T, N, 1),
        comb_out.view(T, N, N),
        layer_out,
    )


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
