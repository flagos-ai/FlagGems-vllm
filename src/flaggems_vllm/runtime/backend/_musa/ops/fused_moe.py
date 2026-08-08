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

"""Optimized fused MoE kernel for MUSA.

This module provides a MUSA-optimized implementation of the fused MoE
(Mixture of Experts) operator, designed for vLLM inference workloads.
The kernel performs expert routing, GEMM operations, and activation fusion
with optimizations tailored for MUSA architecture.

[KernelGen] This kernel was generated and optimized using automated kernel
generation and tuning infrastructure.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@triton.jit
def _moe_sort_init_kernel(
    topk_ids_ptr,
    sorted_ptr,
    em_ptr,
    num_valid,
    topk,
    E,
    BM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EM: tl.constexpr,
):
    # Single-kernel counting sort with disjoint fill/scatter write sets:
    #  - fill writes the sentinel route index (num_valid) only to NON-target
    #    slots (mask = ~target), where a slot is a target iff it lies inside
    #    some expert's real (unpadded) placement range
    #  - scatter writes real routes only to target slots (dst)
    # so no slot is written by both stores (race-free, no kernel boundary
    # needed).
    ridx = tl.arange(0, BLOCK_R)
    rmask = ridx < num_valid
    idx64 = (ridx // topk).to(tl.int64) * topk + (ridx % topk).to(tl.int64)
    e = tl.load(topk_ids_ptr + idx64, mask=rmask, other=BLOCK_E).to(tl.int32)

    ee = tl.arange(0, BLOCK_E)
    cnt = tl.sum((e[:, None] == ee[None, :]).to(tl.int32), axis=0)
    pcnt = ((cnt + BM - 1) // BM) * BM
    pexcl = tl.cumsum(pcnt, axis=0) - pcnt
    total = tl.sum(pcnt, axis=0)
    tl.store(em_ptr, total)

    # Sentinel fill of every non-target slot over the full padded range.
    em_offs = tl.arange(0, BLOCK_EM)
    is_target = tl.sum(
        (
            (em_offs[:, None] >= pexcl[None, :])
            & (em_offs[:, None] < pexcl[None, :] + cnt[None, :])
        ).to(tl.int32),
        axis=1,
    )
    tl.store(sorted_ptr + em_offs.to(tl.int64), num_valid, mask=(is_target == 0))

    # Stable counting-sort scatter: place each route at
    # padded_start(expert) + position_within_expert.
    pos = tl.zeros((BLOCK_R,), dtype=tl.int32)
    for j in tl.static_range(BLOCK_E):
        m = (e == j).to(tl.int32)
        exc = tl.cumsum(m, axis=0) - m
        pos = tl.where(e == j, exc, pos)

    # Padded start offset of each route's expert, gathered per route:
    # sum of pcnt over expert ids smaller than the route's expert.
    pstart_r = tl.sum((ee[None, :] < e[:, None]).to(tl.int32) * pcnt[None, :], axis=1)
    dst = pstart_r + pos
    tl.store(sorted_ptr + dst.to(tl.int64), ridx, mask=rmask)


@triton.jit
def _moe_gemm1_kernel(
    a_ptr,
    w_ptr,
    w_scale_ptr,
    mid_ptr,
    tw_ptr,
    topk_ids_ptr,
    sorted_ptr,
    em_ptr,
    num_valid,
    K,
    N1,
    N2,
    stride_am,
    stride_ak,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_tm,
    stride_tn,
    topk,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    QUANT: tl.constexpr,
    APPLY_W_IN: tl.constexpr,
    A_DTYPE: tl.constexpr,
    MID_DTYPE: tl.constexpr,
    K_DIV: tl.constexpr,
    N_DIV: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N1, 2 * BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    em = tl.load(em_ptr)
    if pid_m * BM >= em:
        return
    r0 = tl.load(sorted_ptr + pid_m * BM)
    if r0 >= num_valid:
        return
    e = tl.load(
        topk_ids_ptr + (r0 // topk).to(tl.int64) * topk + (r0 % topk).to(tl.int64)
    ).to(tl.int32)

    offs_tok = pid_m * BM + tl.arange(0, BM)
    routes = tl.load(sorted_ptr + offs_tok.to(tl.int64)).to(tl.int64)
    tok_mask = routes < num_valid
    tokens = routes // topk

    offs_pair = tl.arange(0, 2 * BN)
    offs_n = pid_n * BN + tl.arange(0, BN)
    pair_cols = tl.where(
        offs_pair < BN, pid_n * BN + offs_pair, N2 + pid_n * BN + (offs_pair - BN)
    )
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + tokens[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        w_ptr
        + e.to(tl.int64) * stride_we
        + offs_k[:, None] * stride_wk
        + pair_cols[None, :] * stride_wn
    )
    acc = tl.zeros((BM, 2 * BN), dtype=tl.float32)
    for k in range(0, K, BK):
        if K_DIV:
            a = tl.load(a_ptrs, mask=tok_mask[:, None], other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=tok_mask[:, None] & (offs_k[None, :] < K - k),
                other=0.0,
            )
        if K_DIV and N_DIV:
            b = tl.load(b_ptrs)
        elif K_DIV:
            b = tl.load(b_ptrs, mask=pair_cols[None, :] < N1, other=0.0)
        elif N_DIV:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        else:
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K - k) & (pair_cols[None, :] < N1),
                other=0.0,
            )
        if APPLY_W_IN:
            wts = tl.load(
                tw_ptr + tokens * stride_tm + (routes % topk) * stride_tn,
                mask=tok_mask,
                other=0.0,
            )
            a = a.to(tl.float32) * wts.to(tl.float32)
            if QUANT == 0:
                a = a.to(A_DTYPE)
        if QUANT == 1:
            # w8a16/w4a16 path: dequantize weights to bf16 and use bf16 MMA
            # (the reference does not quantize activations here, so A stays
            # as-is; measured ~25% faster than the fp32-emulated dot).
            sc = tl.load(
                w_scale_ptr + e.to(tl.int64) * stride_se + pair_cols * stride_sn
            )
            b = (b.to(tl.float32) * sc[None, :]).to(tl.bfloat16)
            a = a.to(tl.bfloat16)
        elif QUANT == 2:
            sc = tl.load(w_scale_ptr + e.to(tl.int64) * stride_se)
            b = b.to(tl.float32) * sc
            a = a.to(tl.float32)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_wk

    acc2 = tl.reshape(acc, (BM, 2, BN))
    gate, up = tl.split(tl.trans(acc2, (0, 2, 1)))
    act = gate * tl.sigmoid(gate) * up
    mid_ptrs = mid_ptr + routes[:, None] * N2 + offs_n[None, :]
    if N_DIV:
        tl.store(mid_ptrs, act.to(MID_DTYPE), mask=tok_mask[:, None])
    else:
        tl.store(
            mid_ptrs,
            act.to(MID_DTYPE),
            mask=tok_mask[:, None] & (offs_n[None, :] < N2),
        )


@triton.jit
def _moe_gemm2_kernel(
    mid_ptr,
    w_ptr,
    w_scale_ptr,
    out_ptr,
    tw_ptr,
    topk_ids_ptr,
    sorted_ptr,
    em_ptr,
    num_valid,
    K2,
    NOUT,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_tm,
    stride_tn,
    topk,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    QUANT: tl.constexpr,
    APPLY_W_IN: tl.constexpr,
    K_DIV: tl.constexpr,
    N_DIV: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(NOUT, BN)
    num_k = tl.cdiv(tl.cdiv(K2, BK), SPLIT_K)
    pid_m = pid // (num_pid_n * SPLIT_K)
    rem = pid % (num_pid_n * SPLIT_K)
    pid_n = rem % num_pid_n
    pid_k = rem // num_pid_n
    em = tl.load(em_ptr)
    if pid_m * BM >= em:
        return
    r0 = tl.load(sorted_ptr + pid_m * BM)
    if r0 >= num_valid:
        return
    e = tl.load(
        topk_ids_ptr + (r0 // topk).to(tl.int64) * topk + (r0 % topk).to(tl.int64)
    ).to(tl.int32)

    offs_tok = pid_m * BM + tl.arange(0, BM)
    routes = tl.load(sorted_ptr + offs_tok.to(tl.int64)).to(tl.int64)
    tok_mask = routes < num_valid
    tokens = routes // topk

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    k_begin = pid_k * num_k * BK
    a_ptrs = mid_ptr + routes[:, None] * K2 + k_begin + offs_k[None, :]
    b_ptrs = (
        w_ptr
        + e.to(tl.int64) * stride_we
        + k_begin
        + offs_k[:, None] * stride_wk
        + offs_n[None, :] * stride_wn
    )
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, num_k * BK, BK):
        if K_DIV:
            a = tl.load(a_ptrs, mask=tok_mask[:, None], other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=tok_mask[:, None] & (offs_k[None, :] < K2 - k_begin - k),
                other=0.0,
            )
        if K_DIV and N_DIV:
            b = tl.load(b_ptrs)
        elif K_DIV:
            b = tl.load(b_ptrs, mask=offs_n[None, :] < NOUT, other=0.0)
        elif N_DIV:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K2 - k_begin - k, other=0.0)
        else:
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K2 - k_begin - k) & (offs_n[None, :] < NOUT),
                other=0.0,
            )
        if QUANT == 1:
            # w8a16/w4a16 path: bf16 MMA on dequantized weights.
            sc = tl.load(w_scale_ptr + e.to(tl.int64) * stride_se + offs_n * stride_sn)
            b = (b.to(tl.float32) * sc[None, :]).to(tl.bfloat16)
            a = a.to(tl.bfloat16)
        elif QUANT == 2:
            sc = tl.load(w_scale_ptr + e.to(tl.int64) * stride_se)
            b = b.to(tl.float32) * sc
            a = a.to(tl.float32)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BK
        b_ptrs += BK * stride_wk

    if not APPLY_W_IN:
        wts = tl.load(
            tw_ptr + tokens * stride_tm + (routes % topk) * stride_tn,
            mask=tok_mask,
            other=0.0,
        )
        acc = acc * wts.to(tl.float32)[:, None]
    # Each K-split block writes its own contrib plane; the sum kernel adds
    # all planes. Invalid (masked) rows may be garbage but are never read.
    out_ptrs = (
        out_ptr + pid_k * num_valid * NOUT + routes[:, None] * NOUT + offs_n[None, :]
    )
    if N_DIV:
        tl.store(out_ptrs, acc, mask=tok_mask[:, None])
    else:
        tl.store(out_ptrs, acc, mask=tok_mask[:, None] & (offs_n[None, :] < NOUT))


@triton.jit
def _moe_quant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    K,
    BLOCK_K: tl.constexpr,
):
    # Per-row int8 fake-quantization: scale = amax(K) / 127 (clamped),
    # q = round(x / scale).clamp(-128, 127).  Matches the reference's
    # per-token activation quantization for the w8a8 path.
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    xbase = x_ptr + pid.to(tl.int64) * K
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(xbase + k + offs, mask=offs < K - k, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))
    m = tl.max(amax, axis=0)
    iscale = 127.0 / tl.maximum(m, 1e-10)
    for k in range(0, K, BLOCK_K):
        x = tl.load(xbase + k + offs, mask=offs < K - k, other=0.0).to(tl.float32)
        q = tl.minimum(tl.maximum(tl.floor(x * iscale + 0.5), -128.0), 127.0)
        tl.store(
            q_ptr + pid.to(tl.int64) * K + k + offs,
            q.to(tl.int8),
            mask=offs < K - k,
        )
    tl.store(scale_ptr + pid, m / 127.0)


@triton.jit
def _moe_gemm1_i8_kernel(
    aq_ptr,
    a_scale_ptr,
    w_ptr,
    w_scale_ptr,
    mid_ptr,
    topk_ids_ptr,
    sorted_ptr,
    em_ptr,
    num_valid,
    K,
    N1,
    N2,
    stride_am,
    stride_ak,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    topk,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    K_DIV: tl.constexpr,
    N_DIV: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N1, 2 * BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    em = tl.load(em_ptr)
    if pid_m * BM >= em:
        return
    r0 = tl.load(sorted_ptr + pid_m * BM)
    if r0 >= num_valid:
        return
    e = tl.load(
        topk_ids_ptr + (r0 // topk).to(tl.int64) * topk + (r0 % topk).to(tl.int64)
    ).to(tl.int32)

    offs_tok = pid_m * BM + tl.arange(0, BM)
    routes = tl.load(sorted_ptr + offs_tok.to(tl.int64)).to(tl.int64)
    tok_mask = routes < num_valid
    tokens = routes // topk

    offs_pair = tl.arange(0, 2 * BN)
    pair_cols = tl.where(
        offs_pair < BN, pid_n * BN + offs_pair, N2 + pid_n * BN + (offs_pair - BN)
    )
    offs_k = tl.arange(0, BK)
    a_ptrs = aq_ptr + tokens[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        w_ptr
        + e.to(tl.int64) * stride_we
        + offs_k[:, None] * stride_wk
        + pair_cols[None, :] * stride_wn
    )
    acc = tl.zeros((BM, 2 * BN), dtype=tl.int32)
    for k in range(0, K, BK):
        if K_DIV:
            a = tl.load(a_ptrs, mask=tok_mask[:, None], other=0).to(tl.int8)
        else:
            a = tl.load(
                a_ptrs,
                mask=tok_mask[:, None] & (offs_k[None, :] < K - k),
                other=0,
            ).to(tl.int8)
        if K_DIV and N_DIV:
            b = tl.load(b_ptrs).to(tl.int8)
        elif K_DIV:
            b = tl.load(b_ptrs, mask=pair_cols[None, :] < N1, other=0).to(tl.int8)
        elif N_DIV:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0).to(tl.int8)
        else:
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K - k) & (pair_cols[None, :] < N1),
                other=0,
            ).to(tl.int8)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_wk

    asc = tl.load(a_scale_ptr + tokens, mask=tok_mask, other=0.0)
    sc = tl.load(w_scale_ptr + e.to(tl.int64) * stride_se + pair_cols * stride_sn)
    accf = acc.to(tl.float32) * asc[:, None] * sc[None, :]
    acc2 = tl.reshape(accf, (BM, 2, BN))
    gate, up = tl.split(tl.trans(acc2, (0, 2, 1)))
    act = gate * tl.sigmoid(gate) * up
    offs_n = pid_n * BN + tl.arange(0, BN)
    mid_ptrs = mid_ptr + routes[:, None] * N2 + offs_n[None, :]
    if N_DIV:
        tl.store(mid_ptrs, act, mask=tok_mask[:, None])
    else:
        tl.store(
            mid_ptrs,
            act,
            mask=tok_mask[:, None] & (offs_n[None, :] < N2),
        )


@triton.jit
def _moe_gemm2_i8_kernel(
    midq_ptr,
    m_scale_ptr,
    w_ptr,
    w_scale_ptr,
    out_ptr,
    tw_ptr,
    topk_ids_ptr,
    sorted_ptr,
    em_ptr,
    num_valid,
    K2,
    NOUT,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_tm,
    stride_tn,
    topk,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    APPLY_W_IN: tl.constexpr,
    K_DIV: tl.constexpr,
    N_DIV: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(NOUT, BN)
    num_k = tl.cdiv(tl.cdiv(K2, BK), SPLIT_K)
    pid_m = pid // (num_pid_n * SPLIT_K)
    rem = pid % (num_pid_n * SPLIT_K)
    pid_n = rem % num_pid_n
    pid_k = rem // num_pid_n
    em = tl.load(em_ptr)
    if pid_m * BM >= em:
        return
    r0 = tl.load(sorted_ptr + pid_m * BM)
    if r0 >= num_valid:
        return
    e = tl.load(
        topk_ids_ptr + (r0 // topk).to(tl.int64) * topk + (r0 % topk).to(tl.int64)
    ).to(tl.int32)

    offs_tok = pid_m * BM + tl.arange(0, BM)
    routes = tl.load(sorted_ptr + offs_tok.to(tl.int64)).to(tl.int64)
    tok_mask = routes < num_valid
    tokens = routes // topk

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    k_begin = pid_k * num_k * BK
    a_ptrs = midq_ptr + routes[:, None] * K2 + k_begin + offs_k[None, :]
    b_ptrs = (
        w_ptr
        + e.to(tl.int64) * stride_we
        + k_begin
        + offs_k[:, None] * stride_wk
        + offs_n[None, :] * stride_wn
    )
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, num_k * BK, BK):
        if K_DIV:
            a = tl.load(a_ptrs, mask=tok_mask[:, None], other=0).to(tl.int8)
        else:
            a = tl.load(
                a_ptrs,
                mask=tok_mask[:, None] & (offs_k[None, :] < K2 - k_begin - k),
                other=0,
            ).to(tl.int8)
        if K_DIV and N_DIV:
            b = tl.load(b_ptrs).to(tl.int8)
        elif K_DIV:
            b = tl.load(b_ptrs, mask=offs_n[None, :] < NOUT, other=0).to(tl.int8)
        elif N_DIV:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K2 - k_begin - k, other=0).to(
                tl.int8
            )
        else:
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K2 - k_begin - k) & (offs_n[None, :] < NOUT),
                other=0,
            ).to(tl.int8)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
        a_ptrs += BK
        b_ptrs += BK * stride_wk

    msc = tl.load(m_scale_ptr + routes, mask=tok_mask, other=0.0)
    sc = tl.load(w_scale_ptr + e.to(tl.int64) * stride_se + offs_n * stride_sn)
    accf = acc.to(tl.float32) * msc[:, None] * sc[None, :]
    if not APPLY_W_IN:
        wts = tl.load(
            tw_ptr + tokens * stride_tm + (routes % topk) * stride_tn,
            mask=tok_mask,
            other=0.0,
        )
        accf = accf * wts.to(tl.float32)[:, None]
    out_ptrs = (
        out_ptr + pid_k * num_valid * NOUT + routes[:, None] * NOUT + offs_n[None, :]
    )
    if N_DIV:
        tl.store(out_ptrs, accf, mask=tok_mask[:, None])
    else:
        tl.store(out_ptrs, accf, mask=tok_mask[:, None] & (offs_n[None, :] < NOUT))


@triton.jit
def _moe_sum_kernel(
    contrib_ptr,
    out_ptr,
    num_tokens,
    NOUT,
    topk,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TOPK_CE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(NOUT, BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    tok = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    tmask = tok < num_tokens
    nmask = offs_n < NOUT
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for r in tl.static_range(TOPK_CE):
        route = tok * topk + r
        base = route.to(tl.int64)[:, None] * NOUT + offs_n[None, :]
        for s in tl.static_range(SPLIT_K):
            acc += tl.load(
                contrib_ptr + s * num_tokens * topk * NOUT + base,
                mask=tmask[:, None] & nmask[None, :],
                other=0.0,
            )
    out_ptrs = out_ptr + tok.to(tl.int64)[:, None] * NOUT + offs_n[None, :]
    tl.store(out_ptrs, acc.to(OUT_DTYPE), mask=tmask[:, None] & nmask[None, :])


def run(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme=None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map=None,
    w1_scale=None,
    w2_scale=None,
    w1_zp=None,
    w2_zp=None,
    a1_scale=None,
    a2_scale=None,
    block_shape=None,
    w1_bias=None,
    w2_bias=None,
) -> torch.Tensor:
    assert activation == "silu"
    assert ocp_mx_scheme is None
    assert expert_map is None
    assert w1_zp is None and w2_zp is None
    assert a1_scale is None and a2_scale is None
    assert w1_bias is None and w2_bias is None
    assert block_shape is None
    assert global_num_experts in (-1, w1.shape[0])

    num_tokens, K = hidden_states.shape
    E, N1, K1 = w1.shape
    NOUT, K2 = w2.shape[1], w2.shape[2]
    N2 = N1 // 2
    topk = topk_ids.shape[1]
    R = num_tokens * topk
    assert K1 == K and K2 == N2 and NOUT == K
    assert R <= 4096

    use_fp8 = use_fp8_w8a8
    use_int = use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16
    quant_mode = 1 if use_int else (2 if use_fp8 else 0)
    # Only w8a8 fake-quantizes activations in the reference, so only it may
    # use the int8 x int8 MMA path; w8a16/w4a16 keep unquantized activations
    # and fall back to the fp32-dot kernels.
    use_i8_mma = use_int8_w8a8
    is_f32 = quant_mode > 0

    hidden_dtype = hidden_states.dtype
    mid_dtype = torch.float32 if is_f32 else hidden_dtype
    tl_mid = _TL_DTYPE[mid_dtype]
    tl_out = _TL_DTYPE[hidden_dtype]
    tl_a = _TL_DTYPE[hidden_dtype]

    if quant_mode == 0:
        # Sweep on MUSA: plain gemm1/gemm2 with num_stages=1 is the pipeline
        # optimum at every M — the small smem tiles need no multi-stage
        # pipelining and st1 maximizes occupancy for the weight-streaming
        # GEMMs (gemm1: 0.732->0.700/1.591->1.565/2.082->2.057ms; gemm2:
        # 0.697->0.695/1.576->1.534/2.099->2.012ms for M=1/4/16).
        cfg1 = (32, 32, 32, 4, 1)  # BM, BK, BN, warps, stages
        cfg2 = (32, 64, 32, 4, 1)
    else:
        # int8/fp32-dot path: 8 warps helps gemm1 (0.872 -> 0.694ms) but
        # strongly hurts gemm2 (0.431 -> 1.549ms), so only gemm1 gets 8 warps.
        cfg1 = (32, 64, 32, 8, 2)
        cfg2 = (32, 64, 32, 4, 2)
    BM_A, BK_A, BN_A, W1, S1 = cfg1
    BM_B, BK_B, BN_B, W2, S2 = cfg2
    if quant_mode == 1 and not use_i8_mma:
        # The bf16-MMA gemm2 (w8a16/w4a16) prefers st1: occupancy win measured
        # at pipeline level (0.787 -> 0.712ms); the i8-MMA gemm2 keeps st2
        # (0.446 vs 0.469ms at st1).
        S2 = 1
    BM_S, BN_S = 16, 64
    # Split the gemm2 K-loop across SPLIT_K blocks: small-M workloads are
    # parallelism-starved and this roughly doubles gemm2 throughput for M=1.
    # Choose the largest divisor of the K-loop iteration count (capped) so each
    # split covers an exact number of BK-wide iterations (no ragged k-ranges).
    # The bf16-MMA path (w8a16/w4a16) measured best with SPLIT_K=2 at M=1
    # (0.800 -> 0.782ms); the int8-MMA path is flat between 2 and 8
    # (0.444 vs 0.447ms) so it uses 2 as well to halve contrib-plane traffic;
    # the plain half path keeps the deep split (cap 8).
    iters2 = (K2 + BK_B - 1) // BK_B
    _cap = 2 if quant_mode else 8
    SPLIT_K = 1
    for _d in range(_cap, 1, -1):
        if iters2 % _d == 0:
            SPLIT_K = _d
            break

    dev = hidden_states.device
    em = torch.empty(1, dtype=torch.int32, device=dev)
    # At most min(E, R) experts are non-empty; each is padded to BM_A slots.
    EM_upper = R + min(E, R) * BM_A + BM_A
    BLOCK_EM = triton.next_power_of_2(EM_upper)
    # The sort kernel fills every slot in [0, BLOCK_EM), so allocate the full
    # padded power-of-two range.
    sorted_ids = torch.empty(BLOCK_EM, dtype=torch.int32, device=dev)
    mid = torch.empty((R, N2), dtype=mid_dtype, device=dev)
    contrib = torch.empty((SPLIT_K, R, NOUT), dtype=torch.float32, device=dev)
    out = hidden_states if inplace else torch.empty_like(hidden_states)

    BLOCK_R = triton.next_power_of_2(R)
    BLOCK_E = triton.next_power_of_2(E)
    _moe_sort_init_kernel[(1,)](
        topk_ids,
        sorted_ids,
        em,
        R,
        topk,
        E,
        BM=BM_A,
        BLOCK_R=BLOCK_R,
        BLOCK_E=BLOCK_E,
        BLOCK_EM=BLOCK_EM,
        num_warps=4,
    )

    s_se = w1_scale.stride(0) if quant_mode else 0
    s_sn = w1_scale.stride(1) if quant_mode == 1 else 0
    s2_se = w2_scale.stride(0) if quant_mode else 0
    s2_sn = w2_scale.stride(1) if quant_mode == 1 else 0

    num_pid_m = triton.cdiv(EM_upper, BM_A)

    if use_i8_mma:
        # Real int8 MMA path: per-token fake-quantize activations, then
        # int8 x int8 dot with int32 accumulation (measured ~1.8x faster
        # than the fp32-emulated dot path on this backend).
        aq = torch.empty((num_tokens, K), dtype=torch.int8, device=dev)
        a_scale = torch.empty((num_tokens,), dtype=torch.float32, device=dev)
        _moe_quant_kernel[(num_tokens,)](
            hidden_states, aq, a_scale, K, BLOCK_K=2048, num_warps=4
        )
        _moe_gemm1_i8_kernel[(num_pid_m * triton.cdiv(N1, 2 * BN_A),)](
            aq,
            a_scale,
            w1,
            w1_scale,
            mid,
            topk_ids,
            sorted_ids,
            em,
            R,
            K,
            N1,
            N2,
            aq.stride(0),
            aq.stride(1),
            w1.stride(0),
            w1.stride(1),
            w1.stride(2),
            s_se,
            s_sn,
            topk,
            BM=BM_A,
            BN=BN_A,
            BK=BK_A,
            K_DIV=(K % BK_A == 0),
            N_DIV=(N2 % BN_A == 0),
            # 4 warps beat 8 for the int8-MMA gemm1 (0.284 -> 0.263ms);
            # the bf16-MMA (w8a16/w4a16) gemm1 keeps 8 warps.
            num_warps=4,
            num_stages=S1,
        )

        midq = torch.empty((R, N2), dtype=torch.int8, device=dev)
        m_scale = torch.empty((R,), dtype=torch.float32, device=dev)
        _moe_quant_kernel[(R,)](mid, midq, m_scale, N2, BLOCK_K=2048, num_warps=4)
        _moe_gemm2_i8_kernel[(num_pid_m * triton.cdiv(NOUT, BN_B) * SPLIT_K,)](
            midq,
            m_scale,
            w2,
            w2_scale,
            contrib,
            topk_weights,
            topk_ids,
            sorted_ids,
            em,
            R,
            K2,
            NOUT,
            w2.stride(0),
            w2.stride(1),
            w2.stride(2),
            s2_se,
            s2_sn,
            topk_weights.stride(0),
            topk_weights.stride(1),
            topk,
            BM=BM_B,
            BN=BN_B,
            BK=BK_B,
            SPLIT_K=SPLIT_K,
            APPLY_W_IN=apply_router_weight_on_input,
            K_DIV=(K2 % BK_B == 0),
            N_DIV=(NOUT % BN_B == 0),
            num_warps=W2,
            num_stages=S2,
        )
    else:
        grid1 = (num_pid_m * triton.cdiv(N1, 2 * BN_A),)
        _moe_gemm1_kernel[grid1](
            hidden_states,
            w1,
            w1_scale if quant_mode else w1,
            mid,
            topk_weights,
            topk_ids,
            sorted_ids,
            em,
            R,
            K,
            N1,
            N2,
            hidden_states.stride(0),
            hidden_states.stride(1),
            w1.stride(0),
            w1.stride(1),
            w1.stride(2),
            s_se,
            s_sn,
            topk_weights.stride(0),
            topk_weights.stride(1),
            topk,
            BM=BM_A,
            BN=BN_A,
            BK=BK_A,
            QUANT=quant_mode,
            APPLY_W_IN=apply_router_weight_on_input,
            A_DTYPE=tl_a,
            MID_DTYPE=tl_mid,
            K_DIV=(K % BK_A == 0),
            N_DIV=(N2 % BN_A == 0),
            num_warps=W1,
            num_stages=S1,
        )

        grid2 = (num_pid_m * triton.cdiv(NOUT, BN_B) * SPLIT_K,)
        _moe_gemm2_kernel[grid2](
            mid,
            w2,
            w2_scale if quant_mode else w2,
            contrib,
            topk_weights,
            topk_ids,
            sorted_ids,
            em,
            R,
            K2,
            NOUT,
            w2.stride(0),
            w2.stride(1),
            w2.stride(2),
            s2_se,
            s2_sn,
            topk_weights.stride(0),
            topk_weights.stride(1),
            topk,
            BM=BM_B,
            BN=BN_B,
            BK=BK_B,
            SPLIT_K=SPLIT_K,
            QUANT=quant_mode,
            APPLY_W_IN=apply_router_weight_on_input,
            K_DIV=(K2 % BK_B == 0),
            N_DIV=(NOUT % BN_B == 0),
            num_warps=W2,
            num_stages=S2,
        )

    grid3 = (triton.cdiv(num_tokens, BM_S) * triton.cdiv(NOUT, BN_S),)
    _moe_sum_kernel[grid3](
        contrib,
        out,
        num_tokens,
        NOUT,
        topk,
        BM=BM_S,
        BN=BN_S,
        TOPK_CE=topk,
        SPLIT_K=SPLIT_K,
        OUT_DTYPE=tl_out,
        num_warps=4,
    )
    return out


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    override_config: Optional[dict] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MUSA-optimized fused experts implementation.

    Main entry point matching the standard FlagGems-vllm interface.
    Dispatches to the optimized MUSA kernel.
    """
    return run(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation="silu",
        apply_router_weight_on_input=False,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=False,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
    )


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    override_config: Optional[dict] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> None:
    """In-place variant of fused experts for MUSA.

    Performs the MoE computation in-place, modifying hidden_states directly.
    """
    fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=True,
        override_config=override_config,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
    )


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    override_config: Optional[dict] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Out-of-place variant of fused experts for MUSA.

    Allocates a new tensor for the output, leaving hidden_states unchanged.
    """
    return fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        override_config=override_config,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
    )
