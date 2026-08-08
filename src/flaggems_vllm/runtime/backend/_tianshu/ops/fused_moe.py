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

"""Optimized fused MoE kernel for Tianshu.

This module provides a Tianshu-optimized implementation of the fused MoE
(Mixture of Experts) operator, designed for vLLM inference workloads.
The kernel performs expert routing, GEMM operations, and activation fusion
with optimizations tailored for Tianshu architecture.

[KernelGen] This kernel was generated and optimized using automated kernel
generation and tuning infrastructure.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 0: expert grouping / alignment.
# Given flat topk_ids[r] (r = token*topk + route), produce:
#   sorted_ids[pos] = r           (rows grouped per expert, padded to BM with -1)
#   block_expert[b] = expert id for m-block b (or -1 for padding blocks)
# ---------------------------------------------------------------------------
@triton.jit
def _moe_align_kernel(
    topk_ids_ptr,
    sorted_ptr,
    block_expert_ptr,
    expert_rank_ptr,
    used_ptr,
    count_ptr,
    num_valid,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    N_POW2: tl.constexpr,
    BM: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_BLOCKS_POW2: tl.constexpr,
):
    offs = tl.arange(0, N_POW2)
    mask = offs < num_valid
    ids = tl.load(topk_ids_ptr + offs, mask=mask, other=-1).to(tl.int32)
    ids = tl.where(mask, ids, -1)

    idxv = tl.arange(0, E_POW2)
    e_mask = idxv < E
    eq = (ids[:, None] == idxv[None, :]) & mask[:, None]
    counts = tl.sum(eq.to(tl.int32), axis=0)
    counts = tl.where(e_mask, counts, 0)

    # padded counts (per-expert groups rounded up to BM) and prefix offsets
    pc = ((counts + BM - 1) // BM) * BM
    pc = tl.where(e_mask, pc, 0)
    lt_e = tl.arange(0, E_POW2)[:, None] < tl.arange(0, E_POW2)[None, :]
    offs_e = tl.sum(lt_e.to(tl.int32) * pc[None, :], axis=1)

    # compact rank for each used expert (for dequantizing only used experts)
    used = (counts > 0).to(tl.int32)
    rank_e = tl.sum(lt_e.to(tl.int32) * used[None, :], axis=1)
    tl.store(
        expert_rank_ptr + idxv,
        tl.where(used == 1, rank_e, -1),
        mask=e_mask,
    )
    # compact list of raw expert ids: used_ptr[r] = id of the r-th used expert
    tl.store(used_ptr + rank_e, idxv, mask=(used == 1) & e_mask)
    tl.store(count_ptr, tl.sum(used))

    # rank of each row within its expert (number of earlier rows with same id)
    lt = tl.arange(0, N_POW2)[:, None] < tl.arange(0, N_POW2)[None, :]
    same = ids[:, None] == ids[None, :]
    rank = tl.sum((lt & same).to(tl.int32), axis=0)

    expert_off = tl.sum(offs_e[None, :] * eq.to(tl.int32), axis=1)
    pos = rank + expert_off
    pos = tl.where(mask, pos, 0)
    tl.store(sorted_ptr + pos, offs, mask=mask)

    # padding sentinels (-1 rows) at the end of every expert group
    pad_start = offs_e + counts
    pad_len = pc - counts
    pad_pos = pad_start[:, None] + tl.arange(0, BM)[None, :]
    pad_valid = (tl.arange(0, BM)[None, :] < pad_len[:, None]) & e_mask[:, None]
    tl.store(
        sorted_ptr + pad_pos,
        tl.zeros((E_POW2, BM), dtype=tl.int32) - 1,
        mask=pad_valid,
    )

    # per-m-block expert ids
    total = tl.sum(pc)
    b_off = tl.arange(0, NUM_BLOCKS_POW2) * BM
    in_range = (b_off[:, None] >= offs_e[None, :]) & (
        b_off[:, None] < (offs_e + pc)[None, :]
    )
    be = tl.sum(in_range.to(tl.int32) * idxv[None, :], axis=1)
    be = tl.where(b_off < total, be, -1)
    tl.store(
        block_expert_ptr + tl.arange(0, NUM_BLOCKS_POW2),
        be,
        mask=tl.arange(0, NUM_BLOCKS_POW2) < NUM_BLOCKS,
    )


# ---------------------------------------------------------------------------
# Kernel: fused int8 weight dequantization for BOTH weight matrices (w1 and
# w2) in one launch. Kept separate from the GEMM kernels because the Iluvatar
# backend miscompiles tl.dot when an operand is produced by an in-kernel
# int8->float conversion; operands must be loaded natively.
# Grid (E_used, N1/BN + N2/BN, SPLIT): only routed experts are processed
# (compact ranks via used_ptr), K is split across programs for memory-level
# parallelism, and w1/w2 blocks interleave in one grid so both streams
# overlap. Outputs [E_used, N, K] indexed by compact rank.
# ---------------------------------------------------------------------------
@triton.jit
def _dequant_weight_kernel(
    w1_ptr,
    s1_ptr,
    o1_ptr,
    w2_ptr,
    s2_ptr,
    o2_ptr,
    used_ptr,
    N1,
    K1,
    N2,
    K2,
    B1,
    OUT_DTYPE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    N1_DIV: tl.constexpr,
    N2_DIV: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    raw = tl.load(used_ptr + pid_e)
    if pid_n < B1:
        K = K1
        w_base = w1_ptr + raw * (N1 * K1)
        s_base = s1_ptr + raw * N1
        o_base = o1_ptr + pid_e * (N1 * K1)
        n = pid_n * BN
        n_mask = n + tl.arange(0, BN) < N1
        n_div = N1_DIV
    else:
        K = K2
        w_base = w2_ptr + raw * (N2 * K2)
        s_base = s2_ptr + raw * N2
        o_base = o2_ptr + pid_e * (N2 * K2)
        n = (pid_n - B1) * BN
        n_mask = n + tl.arange(0, BN) < N2
        n_div = N2_DIV
    offs_n = n + tl.arange(0, BN)
    s = tl.load(s_base + offs_n, mask=n_mask, other=0.0)
    iters = K // BK
    per = iters // SPLIT
    k_start = pid_k * per * BK
    for k0 in range(0, per):
        kk = k_start + k0 * BK + tl.arange(0, BK)
        if n_div:
            w = tl.load(w_base + offs_n[:, None] * K + kk[None, :])
        else:
            w = tl.load(
                w_base + offs_n[:, None] * K + kk[None, :],
                mask=n_mask[:, None],
                other=0,
            )
        deq = (w.to(tl.float32) * s[:, None]).to(OUT_DTYPE)
        if n_div:
            tl.store(o_base + offs_n[:, None] * K + kk[None, :], deq)
        else:
            tl.store(
                o_base + offs_n[:, None] * K + kk[None, :], deq, mask=n_mask[:, None]
            )


# ---------------------------------------------------------------------------
# Kernel: per-row fake quantization (round to int8 grid, dequantize back) in
# the compute dtype. Used for w8a8 input and intermediate activations.
# ---------------------------------------------------------------------------
@triton.jit
def _fake_quant_kernel(
    src_ptr,
    dst_ptr,
    M,
    K,
    stride_sm,
    stride_sk,
    OUT_DTYPE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M

    amax = tl.zeros((BM,), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BK)):
        kk = k0 * BK + tl.arange(0, BK)
        a = tl.load(
            src_ptr + offs_m[:, None] * stride_sm + kk[None, :] * stride_sk,
            mask=m_mask[:, None] & (kk[None, :] < K),
            other=0.0,
        )
        amax = tl.maximum(amax, tl.max(tl.abs(a.to(tl.float32)), axis=1))
    scale = tl.maximum(amax, 1e-10) / 127.0

    for k0 in range(0, tl.cdiv(K, BK)):
        kk = k0 * BK + tl.arange(0, BK)
        k_mask = kk < K
        a = tl.load(
            src_ptr + offs_m[:, None] * stride_sm + kk[None, :] * stride_sk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        q = tl.floor(a.to(tl.float32) / scale[:, None] + 0.5)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        dq = (q * scale[:, None]).to(OUT_DTYPE)
        tl.store(
            dst_ptr + offs_m[:, None] * K + kk[None, :],
            dq,
            mask=m_mask[:, None] & k_mask[None, :],
        )


# ===========================================================================
# GEMM1 kernel (hidden x w1^T) + SiLU. Output out1[route_row, 0:I] fp32.
# Used by all modes: plain (native weights), int8 (pre-dequantized weights),
# with optional per-input router weight.
# ===========================================================================
@triton.jit
def _gemm1_kernel(
    hidden_ptr,
    w1_ptr,
    out1_ptr,
    topk_weights_ptr,
    sorted_ptr,
    block_expert_ptr,
    expert_map_ptr,
    H,
    N_inter,
    stride_hm,
    stride_hk,
    stride_w1e,
    stride_w1n,
    stride_w1k,
    stride_o1r,
    stride_o1n,
    topk: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    APPLY_INPUT_WEIGHT: tl.constexpr,
    K_DIV: tl.constexpr,
    N_DIV: tl.constexpr,
    USE_EXPERT_MAP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    expert = tl.load(block_expert_ptr + pid_m)
    if expert < 0:
        return
    if USE_EXPERT_MAP:
        expert = tl.load(expert_map_ptr + expert)

    rows = tl.load(sorted_ptr + pid_m * BM + tl.arange(0, BM))
    row_mask = rows >= 0
    token = rows // topk

    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N_inter

    weight = tl.load(topk_weights_ptr + rows, mask=row_mask, other=0.0)

    acc_gate = tl.zeros((BM, BN), dtype=tl.float32)
    acc_up = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(H, BK)):
        kk = k0 * BK + tl.arange(0, BK)
        if K_DIV:
            a = tl.load(
                hidden_ptr + token[:, None] * stride_hm + kk[None, :] * stride_hk
            )
        else:
            a = tl.load(
                hidden_ptr + token[:, None] * stride_hm + kk[None, :] * stride_hk,
                mask=row_mask[:, None] & (kk[None, :] < H),
                other=0.0,
            )
        b_base = w1_ptr + expert * stride_w1e
        if N_DIV and K_DIV:
            b_gate = tl.load(
                b_base + kk[:, None] * stride_w1k + offs_n[None, :] * stride_w1n
            )
            b_up = tl.load(
                b_base
                + kk[:, None] * stride_w1k
                + (offs_n + N_inter)[None, :] * stride_w1n
            )
        else:
            b_gate = tl.load(
                b_base + kk[:, None] * stride_w1k + offs_n[None, :] * stride_w1n,
                mask=(kk[:, None] < H) & n_mask[None, :],
                other=0.0,
            )
            b_up = tl.load(
                b_base
                + kk[:, None] * stride_w1k
                + (offs_n + N_inter)[None, :] * stride_w1n,
                mask=(kk[:, None] < H) & n_mask[None, :],
                other=0.0,
            )
        acc_gate += tl.dot(a, b_gate)
        acc_up += tl.dot(a, b_up)

    if APPLY_INPUT_WEIGHT:
        acc_gate = acc_gate * weight[:, None]
        acc_up = acc_up * weight[:, None]

    act = acc_gate * tl.sigmoid(acc_gate) * acc_up

    tl.store(
        out1_ptr + rows[:, None] * stride_o1r + offs_n[None, :] * stride_o1n,
        act,
        mask=row_mask[:, None] & n_mask[None, :],
    )


# ===========================================================================
# GEMM2 kernel (activated x w2^T) with topk-weight scaling.
# Output out2[route_row, 0:N2] fp32. Used by all modes.
# ===========================================================================
@triton.jit
def _gemm2_kernel(
    out1_ptr,
    w2_ptr,
    out2_ptr,
    topk_weights_ptr,
    sorted_ptr,
    block_expert_ptr,
    expert_map_ptr,
    K2,
    N2,
    stride_o1r,
    stride_o1k,
    stride_w2e,
    stride_w2n,
    stride_w2k,
    stride_o2r,
    stride_o2n,
    topk: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    K_DIV: tl.constexpr,
    N_DIV: tl.constexpr,
    USE_EXPERT_MAP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    expert = tl.load(block_expert_ptr + pid_m)
    if expert < 0:
        return
    if USE_EXPERT_MAP:
        expert = tl.load(expert_map_ptr + expert)

    rows = tl.load(sorted_ptr + pid_m * BM + tl.arange(0, BM))
    row_mask = rows >= 0
    rows_safe = tl.where(row_mask, rows, 0)
    weight = tl.load(topk_weights_ptr + rows, mask=row_mask, other=0.0)

    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N2

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K2, BK)):
        kk = k0 * BK + tl.arange(0, BK)
        if K_DIV:
            a = tl.load(
                out1_ptr + rows_safe[:, None] * stride_o1r + kk[None, :] * stride_o1k
            )
        else:
            a = tl.load(
                out1_ptr + rows_safe[:, None] * stride_o1r + kk[None, :] * stride_o1k,
                mask=row_mask[:, None] & (kk[None, :] < K2),
                other=0.0,
            )
        if N_DIV and K_DIV:
            b = tl.load(
                w2_ptr
                + expert * stride_w2e
                + kk[:, None] * stride_w2k
                + offs_n[None, :] * stride_w2n
            )
        else:
            b = tl.load(
                w2_ptr
                + expert * stride_w2e
                + kk[:, None] * stride_w2k
                + offs_n[None, :] * stride_w2n,
                mask=(kk[:, None] < K2) & n_mask[None, :],
                other=0.0,
            )
        acc += tl.dot(a.to(COMPUTE_DTYPE), b)

    acc = acc * weight[:, None]
    tl.store(
        out2_ptr + rows[:, None] * stride_o2r + offs_n[None, :] * stride_o2n,
        acc,
        mask=row_mask[:, None] & n_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Kernel: sum topk route contributions per token, cast to output dtype.
# ---------------------------------------------------------------------------
@triton.jit
def _sum_kernel(
    out2_ptr,
    out_ptr,
    T,
    H,
    stride_o2r,
    stride_o2n,
    stride_om,
    stride_on,
    topk: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < T
    n_mask = offs_n < H
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for route in range(topk):
        r = offs_m * topk + route
        acc += tl.load(
            out2_ptr + r[:, None] * stride_o2r + offs_n[None, :] * stride_o2n,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(OUT_DTYPE),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _dequant_cfg2(K1: int, K2: int):
    """Pick (BK, SPLIT) for the fused split-K dequant kernel: BK divides both
    K dims and SPLIT (<=4) divides both iteration counts K//BK."""
    bk = 128
    while bk > 1 and (K1 % bk != 0 or K2 % bk != 0):
        bk //= 2
    i1, i2 = K1 // bk, K2 // bk
    split = 4
    while split > 1 and (i1 % split != 0 or i2 % split != 0):
        split //= 2
    return bk, split


def _gemm1_cfg(N_inter: int, H: int):
    bn = 128 if N_inter % 128 == 0 else 64
    bk = 128 if H % 128 == 0 else 64
    return bn, bk


def _gemm2_cfg(N2: int, K2: int):
    bn = 256 if N2 % 256 == 0 else (128 if N2 % 128 == 0 else 64)
    bk = 128 if K2 % 128 == 0 else 64
    return bn, bk


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
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert activation == "silu"
    assert sum((use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16)) <= 1
    assert ocp_mx_scheme is None
    assert global_num_experts in (-1, w1.shape[0])
    assert expert_map is None
    assert w1_zp is None and w2_zp is None
    assert a1_scale is None and a2_scale is None
    assert w1_bias is None and w2_bias is None
    assert block_shape is None

    if use_fp8_w8a8:
        raise NotImplementedError("fp8_w8a8 is not supported by this implementation")

    T, H = hidden_states.shape
    E = w1.shape[0]
    N1 = w1.shape[1]
    N_inter = N1 // 2
    N2 = w2.shape[1]
    K2 = w2.shape[2]
    topk = topk_ids.shape[1]
    num_valid = T * topk

    if hidden_states.dtype == torch.bfloat16:
        compute_dtype = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_dtype = tl.float16
    else:
        raise ValueError(f"unsupported hidden dtype {hidden_states.dtype}")

    if use_int8_w8a8:
        assert per_channel_quant
        assert w1_scale is not None and w2_scale is not None
        mode = 1
    elif use_int8_w8a16 or use_int4_w4a16:
        assert per_channel_quant
        assert w1_scale is not None and w2_scale is not None
        mode = 2
    else:
        assert not per_channel_quant
        assert w1_scale is None and w2_scale is None
        mode = 0

    BM = 16
    BM_ALIGN = 16

    device = hidden_states.device
    out1 = torch.empty((num_valid, N_inter), device=device, dtype=torch.float32)
    out2 = torch.empty((num_valid, N2), device=device, dtype=torch.float32)
    output = hidden_states if inplace else torch.empty_like(hidden_states)

    alloc = num_valid + E * BM_ALIGN
    nblocks = triton.cdiv(alloc, BM_ALIGN)
    sorted_ids = torch.empty((alloc,), device=device, dtype=torch.int32)
    block_expert = torch.empty((nblocks,), device=device, dtype=torch.int32)
    expert_rank = torch.empty((E,), device=device, dtype=torch.int32)
    used_experts = torch.empty((E,), device=device, dtype=torch.int32)
    count_buf = torch.empty((1,), device=device, dtype=torch.int32)

    E_POW2 = _pow2(E)
    N_POW2 = _pow2(num_valid)
    NB_POW2 = _pow2(nblocks)

    _moe_align_kernel[(1,)](
        topk_ids,
        sorted_ids,
        block_expert,
        expert_rank,
        used_experts,
        count_buf,
        num_valid,
        E=E,
        E_POW2=E_POW2,
        N_POW2=N_POW2,
        BM=BM_ALIGN,
        NUM_BLOCKS=nblocks,
        NUM_BLOCKS_POW2=NB_POW2,
    )

    # Preprocessing for quantized modes: dequantize int8 weights into the
    # compute dtype so GEMM dot operands are loaded natively (the backend
    # miscompiles dots with in-kernel int8->float converted operands).
    # Only experts actually used by the routing are dequantized; w1 and w2
    # are dequantized in ONE fused launch so their memory streams overlap.
    w1_gemm = w1
    w2_gemm = w2
    if mode in (1, 2):
        num_used = int(count_buf.item())
        w1_deq = torch.empty((E, N1, H), device=device, dtype=hidden_states.dtype)
        w2_deq = torch.empty((E, N2, K2), device=device, dtype=hidden_states.dtype)
        bk, split = _dequant_cfg2(H, K2)
        b1 = triton.cdiv(N1, 64)
        b2 = triton.cdiv(N2, 64)
        _dequant_weight_kernel[(num_used, b1 + b2, split)](
            w1,
            w1_scale,
            w1_deq,
            w2,
            w2_scale,
            w2_deq,
            used_experts,
            N1,
            H,
            N2,
            K2,
            b1,
            OUT_DTYPE=compute_dtype,
            BN=64,
            BK=bk,
            SPLIT=split,
            N1_DIV=(N1 % 64 == 0),
            N2_DIV=(N2 % 64 == 0),
            num_warps=4,
            num_stages=3,
        )
        w1_gemm = w1_deq
        w2_gemm = w2_deq

    if mode == 1:
        q_hidden = torch.empty((T, H), device=device, dtype=hidden_states.dtype)
        q_out1 = torch.empty(
            (num_valid, N_inter), device=device, dtype=hidden_states.dtype
        )
        _fake_quant_kernel[(triton.cdiv(T, 16),)](
            hidden_states,
            q_hidden,
            T,
            H,
            hidden_states.stride(0),
            hidden_states.stride(1),
            OUT_DTYPE=compute_dtype,
            BM=16,
            BK=64,
            num_warps=4,
        )
        gemm1_in = q_hidden
    else:
        gemm1_in = hidden_states

    BN1, BK1 = _gemm1_cfg(N_inter, H)
    W1 = 8 if (BN1 >= 128 and BK1 >= 128) else 4
    grid1 = (triton.cdiv(alloc, BM), triton.cdiv(N_inter, BN1))
    _gemm1_kernel[grid1](
        gemm1_in,
        w1_gemm,
        out1,
        topk_weights,
        sorted_ids,
        block_expert,
        expert_rank,
        H,
        N_inter,
        gemm1_in.stride(0),
        gemm1_in.stride(1),
        w1_gemm.stride(0),
        w1_gemm.stride(1),
        w1_gemm.stride(2),
        out1.stride(0),
        out1.stride(1),
        topk=topk,
        BM=BM,
        BN=BN1,
        BK=BK1,
        APPLY_INPUT_WEIGHT=apply_router_weight_on_input and mode == 0,
        K_DIV=(H % BK1 == 0),
        N_DIV=(N_inter % BN1 == 0),
        USE_EXPERT_MAP=(mode != 0),
        num_warps=W1,
        num_stages=3,
    )

    if mode == 1:
        _fake_quant_kernel[(triton.cdiv(num_valid, 16),)](
            out1,
            q_out1,
            num_valid,
            N_inter,
            out1.stride(0),
            out1.stride(1),
            OUT_DTYPE=compute_dtype,
            BM=16,
            BK=64,
            num_warps=4,
        )
        gemm2_in = q_out1
    else:
        gemm2_in = out1

    BN2, BK2 = _gemm2_cfg(N2, K2)
    W2 = 8 if (BN2 >= 128 and BK2 >= 128) else 4
    grid2 = (triton.cdiv(alloc, BM), triton.cdiv(N2, BN2))
    _gemm2_kernel[grid2](
        gemm2_in,
        w2_gemm,
        out2,
        topk_weights,
        sorted_ids,
        block_expert,
        expert_rank,
        K2,
        N2,
        gemm2_in.stride(0),
        gemm2_in.stride(1),
        w2_gemm.stride(0),
        w2_gemm.stride(1),
        w2_gemm.stride(2),
        out2.stride(0),
        out2.stride(1),
        topk=topk,
        BM=BM,
        BN=BN2,
        BK=BK2,
        COMPUTE_DTYPE=compute_dtype,
        K_DIV=(K2 % BK2 == 0),
        N_DIV=(N2 % BN2 == 0),
        USE_EXPERT_MAP=(mode != 0),
        num_warps=W2,
        num_stages=3,
    )

    grid3 = (triton.cdiv(T, 16), triton.cdiv(N2, 128))
    _sum_kernel[grid3](
        out2,
        output,
        T,
        N2,
        out2.stride(0),
        out2.stride(1),
        output.stride(0),
        output.stride(1),
        topk=topk,
        BM=16,
        BN=128,
        OUT_DTYPE=compute_dtype,
        num_warps=4,
    )

    return output


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
    """Tianshu-optimized fused experts implementation.

    Main entry point matching the standard FlagGems-vllm interface.
    Dispatches to the optimized Tianshu kernel.
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
    """In-place variant of fused experts for Tianshu.

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
    """Out-of-place variant of fused experts for Tianshu.

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
