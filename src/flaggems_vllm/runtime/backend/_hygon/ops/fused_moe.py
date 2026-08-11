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

"""Optimized fused MoE kernel for Hygon.

This module provides a Hygon-optimized implementation of the fused MoE
(Mixture of Experts) operator, designed for vLLM inference workloads.
The kernel performs expert routing, GEMM operations, and activation fusion
with optimizations tailored for Hygon architecture.

[KernelGen] This kernel was generated and optimized using automated kernel
generation and tuning infrastructure.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fused MoE (fused_experts_impl), Triton implementation.
#
# Pipeline (per run() call):
#   1. (w8a8 modes only) per-token quantize hidden_states -> hidden_q + a1_scale
#   2. route rows -> used-expert list (count + append via int32 atomics)
#   3. GEMM1: for each *used* expert, masked rows over (token, route),
#      gate/up projections + SiLU fusion -> scratch2 [E, R, N_inter]
#   4. (w8a8 modes only) per-row quantize scratch2 -> scratch2_q + a2_scale
#   5. GEMM2: masked rows, dot with w2, scale by topk weight -> scratch3 [R, Ko]
#   6. reduce-sum over topk routes -> output [M, Ko]
#
# Rows that do not belong to the current expert are masked to 0 so a single
# tiled GEMM per expert covers every route; tensor cores are fed with padded
# BLOCK_M rows while the used-expert filter avoids reading weights of experts
# that no token routed to.
# ---------------------------------------------------------------------------


@triton.jit
def _quantize_input_kernel(
    a_ptr,
    a_q_ptr,
    a_scale_ptr,
    M,
    H,
    stride_am,
    stride_ak,
    aq_stride_am,
    aq_stride_ak,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QMAX: tl.constexpr,
    QMIN: tl.constexpr,
    IS_INT8: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, H, BLOCK_K):
        kk = k + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + kk[None, :] * stride_ak,
            mask=m_mask[:, None] & (kk[None, :] < H),
            other=0.0,
        )
        amax = tl.maximum(amax, tl.max(tl.abs(a.to(tl.float32)), axis=1))
    scale = tl.maximum(amax / QMAX, 1e-10)
    tl.store(a_scale_ptr + offs_m, scale, mask=m_mask)
    for k in range(0, H, BLOCK_K):
        kk = k + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + kk[None, :] * stride_ak,
            mask=m_mask[:, None] & (kk[None, :] < H),
            other=0.0,
        )
        if IS_INT8:
            aq = tl.minimum(
                tl.maximum(
                    tl.math.floor(a.to(tl.float32) / scale[:, None] + 0.5), QMIN
                ),
                QMAX,
            )
            aq = aq.to(tl.int8)
        else:
            aq = tl.minimum(tl.maximum(a.to(tl.float32) / scale[:, None], QMIN), QMAX)
            aq = aq.to(tl.float8e4m3fn)
        tl.store(
            a_q_ptr + offs_m[:, None] * aq_stride_am + kk[None, :] * aq_stride_ak,
            aq,
            mask=m_mask[:, None] & (kk[None, :] < H),
        )


@triton.jit
def _zero_kernel(ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offs, 0, mask=offs < n)


@triton.jit
def _moe_count_kernel(
    topk_ids_ptr,
    count_ptr,
    R,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < R
    e = tl.load(topk_ids_ptr + offs, mask=m, other=-1).to(tl.int64)
    tl.atomic_add(count_ptr + e, 1, mask=m & (e >= 0))


@triton.jit
def _moe_offset_scatter_kernel(
    topk_ids_ptr,
    count_ptr,
    off_ptr,
    sorted_ptr,
    R,
    E,
):
    # Serial exclusive prefix over per-expert counts; count is reused as the
    # scatter cursor (reset to 0 here). This avoids relying on atomic return
    # values, which are not dependable on this backend for dependent stores.
    total = 0
    for e in range(E):
        tl.store(off_ptr + e, total)
        c = tl.load(count_ptr + e)
        tl.store(count_ptr + e, 0)
        total += c
    for i in range(R):
        e = tl.load(topk_ids_ptr + i).to(tl.int32)
        cur = tl.load(count_ptr + e)
        tl.store(sorted_ptr + tl.load(off_ptr + e) + cur, i)
        tl.store(count_ptr + e, cur + 1)


@triton.jit
def _moe_gemm1_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    out_ptr,
    count_ptr,
    off_ptr,
    sorted_ptr,
    M,
    H,
    N_inter,
    TOPK,
    R,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_bse,
    stride_bsn,
    stride_oe,
    stride_or,
    stride_oi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RBLOCK: tl.constexpr,
    MODE: tl.constexpr,
    STORE_F32: tl.constexpr,
    SCALE_PER_EXPERT: tl.constexpr,
    APPLY_WEIGHT_ON_INPUT: tl.constexpr,
    USE_INT_DOT: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    N_DIV: tl.constexpr,
    K_DIV: tl.constexpr,
    COMPACT: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    expert = pid_e.to(tl.int64)
    if COMPACT:
        count_e = tl.load(count_ptr + expert)
        if count_e == 0:
            return
        if pid_m * BLOCK_M >= count_e:
            return
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = offs_m < count_e
        r = tl.load(
            sorted_ptr + tl.load(off_ptr + expert) + tl.minimum(offs_m, count_e - 1),
            mask=valid,
            other=-1,
        ).to(tl.int64)
        r = tl.maximum(r, 0)
        token = r // TOPK
    else:
        offs_r = tl.arange(0, RBLOCK)
        eids_r = tl.load(topk_ids_ptr + offs_r, mask=offs_r < R, other=-1).to(tl.int64)
        if tl.max(tl.where(eids_r == expert, 1, 0)) == 0:
            return

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = offs_m < M * TOPK
        eid = tl.load(topk_ids_ptr + offs_m, mask=row_mask, other=-1).to(tl.int64)
        valid = row_mask & (eid == expert)
        token = offs_m // TOPK
        r = offs_m

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + token[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_g_ptrs = (
        b_ptr
        + expert * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    b_u_ptrs = b_g_ptrs + N_inter * stride_bn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        kk = k + offs_k
        if K_DIV:
            a = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
            b_g = tl.load(b_g_ptrs)
            b_u = tl.load(b_u_ptrs)
        else:
            a = tl.load(a_ptrs, mask=valid[:, None] & (kk[None, :] < H), other=0.0)
            b_g = tl.load(b_g_ptrs, mask=kk[:, None] < H, other=0.0)
            b_u = tl.load(b_u_ptrs, mask=kk[:, None] < H, other=0.0)
        if APPLY_WEIGHT_ON_INPUT:
            w = tl.load(topk_weights_ptr + r, mask=valid, other=0.0).to(COMPUTE_DTYPE)
            a = a * w[:, None]
        if MODE == 1 or MODE == 2:
            if USE_INT_DOT:
                acc_g = tl.dot(a, b_g, acc=acc_g, out_dtype=tl.float32)
                acc_u = tl.dot(a, b_u, acc=acc_u, out_dtype=tl.float32)
            else:
                acc_g = tl.dot(a.to(COMPUTE_DTYPE), b_g.to(COMPUTE_DTYPE), acc=acc_g)
                acc_u = tl.dot(a.to(COMPUTE_DTYPE), b_u.to(COMPUTE_DTYPE), acc=acc_u)
        elif MODE == 3:
            acc_g = tl.dot(a, b_g.to(COMPUTE_DTYPE), acc=acc_g)
            acc_u = tl.dot(a, b_u.to(COMPUTE_DTYPE), acc=acc_u)
        else:
            acc_g = tl.dot(a, b_g, acc=acc_g)
            acc_u = tl.dot(a, b_u, acc=acc_u)
        a_ptrs += BLOCK_K * stride_ak
        b_g_ptrs += BLOCK_K * stride_bk
        b_u_ptrs += BLOCK_K * stride_bk

    if MODE == 1 or MODE == 2:
        a_scale = tl.load(a_scale_ptr + token, mask=valid, other=0.0)
        if SCALE_PER_EXPERT:
            b_scale = tl.load(b_scale_ptr + expert)
            acc_g = acc_g * a_scale[:, None] * b_scale
            acc_u = acc_u * a_scale[:, None] * b_scale
        else:
            b_scale_g = tl.load(b_scale_ptr + expert * stride_bse + offs_n * stride_bsn)
            b_scale_u = tl.load(
                b_scale_ptr + expert * stride_bse + (offs_n + N_inter) * stride_bsn
            )
            acc_g = acc_g * a_scale[:, None] * b_scale_g[None, :]
            acc_u = acc_u * a_scale[:, None] * b_scale_u[None, :]
    elif MODE == 3:
        b_scale_g = tl.load(b_scale_ptr + expert * stride_bse + offs_n * stride_bsn)
        b_scale_u = tl.load(
            b_scale_ptr + expert * stride_bse + (offs_n + N_inter) * stride_bsn
        )
        acc_g = acc_g * b_scale_g[None, :]
        acc_u = acc_u * b_scale_u[None, :]

    act = tl.sigmoid(acc_g) * acc_g * acc_u
    if STORE_F32:
        out = act
    else:
        out = act.to(COMPUTE_DTYPE)
    o_ptrs = (
        out_ptr
        + expert * stride_oe
        + r[:, None] * stride_or
        + offs_n[None, :] * stride_oi
    )
    if N_DIV:
        tl.store(o_ptrs, out, mask=valid[:, None])
    else:
        tl.store(o_ptrs, out, mask=valid[:, None] & (offs_n[None, :] < N_inter))


@triton.jit
def _quantize_scratch_kernel(
    in_ptr,
    out_ptr,
    scale_ptr,
    topk_ids_ptr,
    count_ptr,
    off_ptr,
    sorted_ptr,
    M,
    N_inter,
    TOPK,
    R,
    stride_ie,
    stride_ir,
    stride_ii,
    stride_oe,
    stride_or,
    stride_oi,
    stride_se,
    stride_sr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QMAX: tl.constexpr,
    QMIN: tl.constexpr,
    IS_INT8: tl.constexpr,
    COMPACT: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    if COMPACT:
        count_e = tl.load(count_ptr + pid_e)
        if count_e == 0:
            return
        if pid_m * BLOCK_M >= count_e:
            return
        valid = offs_m < count_e
        r = tl.load(
            sorted_ptr + tl.load(off_ptr + pid_e) + tl.minimum(offs_m, count_e - 1),
            mask=valid,
            other=-1,
        ).to(tl.int64)
        r = tl.maximum(r, 0)
    else:
        row_mask = offs_m < M * TOPK
        eid = tl.load(topk_ids_ptr + offs_m, mask=row_mask, other=-1).to(tl.int64)
        valid = row_mask & (eid == pid_e)
        r = offs_m

    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, N_inter, BLOCK_K):
        kk = k + offs_k
        x = tl.load(
            in_ptr
            + pid_e * stride_ie
            + r[:, None] * stride_ir
            + kk[None, :] * stride_ii,
            mask=valid[:, None] & (kk[None, :] < N_inter),
            other=0.0,
        )
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=1))
    scale = tl.maximum(amax / QMAX, 1e-10)
    tl.store(scale_ptr + pid_e * stride_se + r * stride_sr, scale, mask=valid)

    for k in range(0, N_inter, BLOCK_K):
        kk = k + offs_k
        x = tl.load(
            in_ptr
            + pid_e * stride_ie
            + r[:, None] * stride_ir
            + kk[None, :] * stride_ii,
            mask=valid[:, None] & (kk[None, :] < N_inter),
            other=0.0,
        )
        if IS_INT8:
            xq = tl.minimum(
                tl.maximum(tl.math.floor(x / scale[:, None] + 0.5), QMIN), QMAX
            ).to(tl.int8)
        else:
            xq = tl.minimum(tl.maximum(x / scale[:, None], QMIN), QMAX).to(
                tl.float8e4m3fn
            )
        tl.store(
            out_ptr
            + pid_e * stride_oe
            + r[:, None] * stride_or
            + kk[None, :] * stride_oi,
            xq,
            mask=valid[:, None] & (kk[None, :] < N_inter),
        )


@triton.jit
def _moe_gemm2_kernel(
    a2_ptr,
    a2_scale_ptr,
    b2_ptr,
    b2_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    out_ptr,
    count_ptr,
    off_ptr,
    sorted_ptr,
    M,
    N_inter,
    Ko,
    TOPK,
    R,
    stride_a2e,
    stride_a2r,
    stride_a2i,
    stride_a2se,
    stride_a2sr,
    stride_b2e,
    stride_b2n,
    stride_b2k,
    stride_b2se,
    stride_b2sn,
    stride_or,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RBLOCK: tl.constexpr,
    MODE: tl.constexpr,
    SCALE_PER_EXPERT: tl.constexpr,
    USE_INT_DOT: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    APPLY_WEIGHT_ON_INPUT: tl.constexpr,
    N_DIV: tl.constexpr,
    K_DIV: tl.constexpr,
    COMPACT: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    expert = pid_e.to(tl.int64)
    if COMPACT:
        count_e = tl.load(count_ptr + expert)
        if count_e == 0:
            return
        if pid_m * BLOCK_M >= count_e:
            return
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = offs_m < count_e
        r = tl.load(
            sorted_ptr + tl.load(off_ptr + expert) + tl.minimum(offs_m, count_e - 1),
            mask=valid,
            other=-1,
        ).to(tl.int64)
        r = tl.maximum(r, 0)
    else:
        offs_r = tl.arange(0, RBLOCK)
        eids_r = tl.load(topk_ids_ptr + offs_r, mask=offs_r < R, other=-1).to(tl.int64)
        if tl.max(tl.where(eids_r == expert, 1, 0)) == 0:
            return

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = offs_m < M * TOPK
        eid = tl.load(topk_ids_ptr + offs_m, mask=row_mask, other=-1).to(tl.int64)
        valid = row_mask & (eid == expert)
        r = offs_m

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_ptrs = (
        a2_ptr
        + expert * stride_a2e
        + r[:, None] * stride_a2r
        + offs_k[None, :] * stride_a2i
    )
    b_ptrs = (
        b2_ptr
        + expert * stride_b2e
        + offs_n[None, :] * stride_b2n
        + offs_k[:, None] * stride_b2k
    )
    for k in range(0, N_inter, BLOCK_K):
        kk = k + offs_k
        if K_DIV:
            a2 = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
            b2 = tl.load(b_ptrs)
        else:
            a2 = tl.load(
                a_ptrs, mask=valid[:, None] & (kk[None, :] < N_inter), other=0.0
            )
            b2 = tl.load(b_ptrs, mask=kk[:, None] < N_inter, other=0.0)
        if MODE == 1 or MODE == 2:
            if USE_INT_DOT:
                acc = tl.dot(a2, b2, acc=acc, out_dtype=tl.float32)
            else:
                acc = tl.dot(a2.to(COMPUTE_DTYPE), b2.to(COMPUTE_DTYPE), acc=acc)
        elif MODE == 3:
            acc = tl.dot(a2, b2.to(COMPUTE_DTYPE), acc=acc)
        else:
            acc = tl.dot(a2, b2, acc=acc)
        a_ptrs += BLOCK_K * stride_a2i
        b_ptrs += BLOCK_K * stride_b2k

    if MODE == 1 or MODE == 2:
        a2s = tl.load(
            a2_scale_ptr + expert * stride_a2se + r * stride_a2sr,
            mask=valid,
            other=0.0,
        )
        if SCALE_PER_EXPERT:
            b2s = tl.load(b2_scale_ptr + expert)
            acc = acc * a2s[:, None] * b2s
        else:
            b2s = tl.load(b2_scale_ptr + expert * stride_b2se + offs_n * stride_b2sn)
            acc = acc * a2s[:, None] * b2s[None, :]
    elif MODE == 3:
        b2s = tl.load(b2_scale_ptr + expert * stride_b2se + offs_n * stride_b2sn)
        acc = acc * b2s[None, :]

    if not APPLY_WEIGHT_ON_INPUT:
        w = tl.load(topk_weights_ptr + r, mask=valid, other=0.0).to(tl.float32)
        acc = acc * w[:, None]

    o_ptrs = out_ptr + r[:, None] * stride_or + offs_n[None, :]
    if N_DIV:
        tl.store(o_ptrs, acc.to(OUT_DTYPE), mask=valid[:, None])
    else:
        tl.store(
            o_ptrs, acc.to(OUT_DTYPE), mask=valid[:, None] & (offs_n[None, :] < Ko)
        )


@triton.jit
def _moe_sum_kernel(
    src_ptr,
    out_ptr,
    M,
    Ko,
    stride_sr,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for r in tl.static_range(TOPK):
        rows = offs_m * TOPK + r
        x = tl.load(
            src_ptr + rows[:, None] * stride_sr + offs_n[None, :],
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < Ko),
            other=0.0,
        )
        acc += x
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        acc.to(OUT_DTYPE),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < Ko),
    )


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
    assert ocp_mx_scheme is None
    assert expert_map is None
    assert w1_zp is None and w2_zp is None
    assert a1_scale is None and a2_scale is None
    assert w1_bias is None and w2_bias is None
    assert block_shape is None
    assert sum((use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16)) <= 1

    M, H = hidden_states.shape
    E, Nw, _ = w1.shape
    _, Ko, N_inter = w2.shape
    TOPK = topk_ids.shape[1]
    R = M * TOPK
    assert E == w2.shape[0]
    assert Nw == 2 * N_inter
    assert Ko == H
    assert global_num_experts in (-1, E)

    if hidden_states.dtype == torch.bfloat16:
        compute_dtype, out_dtype = tl.bfloat16, tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_dtype, out_dtype = tl.float16, tl.float16
    else:
        compute_dtype, out_dtype = tl.float32, tl.float32

    if use_fp8_w8a8:
        mode = 2
        assert not per_channel_quant
    elif use_int8_w8a8:
        mode = 1
        assert per_channel_quant
    elif use_int8_w8a16 or use_int4_w4a16:
        mode = 3
        assert per_channel_quant
    else:
        mode = 0
        assert not per_channel_quant

    quant_in = mode in (1, 2)
    dev = hidden_states.device
    qdtype = torch.int8 if mode == 1 else torch.float8_e4m3fn

    if quant_in:
        hidden_q = torch.empty((M, H), device=dev, dtype=qdtype)
        a1_scale_t = torch.empty((M,), device=dev, dtype=torch.float32)
    scratch2 = torch.empty(
        (E, R, N_inter),
        device=dev,
        dtype=torch.float32 if quant_in else hidden_states.dtype,
    )
    if quant_in:
        scratch2_q = torch.empty((E, R, N_inter), device=dev, dtype=qdtype)
        a2_scale_t = torch.empty((E, R), device=dev, dtype=torch.float32)
    scratch3 = torch.empty((R, Ko), device=dev, dtype=hidden_states.dtype)
    output = hidden_states if inplace else torch.empty_like(hidden_states)

    BM = 32 if R >= 32 else 16
    RBLOCK = triton.next_power_of_2(R)

    # Per-expert row compaction was measured to cost more than it saves on this
    # target: for R=32 (one m-tile per expert at BM=32) the B-side weights are
    # already read exactly once, and the 3 serial index kernels add ~0.15ms to
    # the bandwidth-bound time-2 path. Disabled; GEMM kernels use their in-kernel
    # route-scan (COMPACT=False) path instead.
    compact = False
    count = torch.empty((E,), device=dev, dtype=torch.int32)
    off = torch.empty((E + 1,), device=dev, dtype=torch.int32)
    sorted_ids = torch.empty((R,), device=dev, dtype=torch.int32)
    if compact:
        _zero_kernel[(triton.cdiv(E, 128),)](count, E, BLOCK=128, num_warps=4)
        _moe_count_kernel[(triton.cdiv(R, 128),)](
            topk_ids, count, R, BLOCK=128, num_warps=4
        )
        _moe_offset_scatter_kernel[(1,)](
            topk_ids, count, off, sorted_ids, R, E, num_warps=4
        )

    if quant_in:
        QMAX = 127.0 if mode == 1 else 448.0
        QMIN = -128.0 if mode == 1 else -448.0
        _quantize_input_kernel[(triton.cdiv(M, 16),)](
            hidden_states,
            hidden_q,
            a1_scale_t,
            M,
            H,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_q.stride(0),
            hidden_q.stride(1),
            BLOCK_M=16,
            BLOCK_K=256,
            QMAX=QMAX,
            QMIN=QMIN,
            IS_INT8=(mode == 1),
            num_warps=4,
        )

    # Small single-token plain batches are B-memory-latency bound: narrow tiles
    # (BN=32) double blocks/CU and improve effective bandwidth. Larger/quantized
    # batches prefer wide tiles (BN=64, deeper K) for MFMA efficiency.
    small = (R < 8) and (mode == 0)
    if small:
        BK1, BN1 = 64, 32
        BK2 = 64 if N_inter % 64 == 0 else 32
        BN2 = 32
    else:
        BK1, BN1 = 64, 64
        BK2 = 128 if N_inter % 128 == 0 else 64
        BN2 = 64
    grid1 = (E, triton.cdiv(R, BM), triton.cdiv(N_inter, BN1))
    _moe_gemm1_kernel[grid1](
        hidden_q if quant_in else hidden_states,
        a1_scale_t if quant_in else hidden_states,
        w1,
        w1_scale if w1_scale is not None else w1,
        topk_weights,
        topk_ids,
        scratch2,
        count,
        off,
        sorted_ids,
        M,
        H,
        N_inter,
        TOPK,
        R,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        w1_scale.stride(0) if (w1_scale is not None and w1_scale.dim() >= 2) else 0,
        w1_scale.stride(1) if (w1_scale is not None and w1_scale.dim() >= 2) else 0,
        scratch2.stride(0),
        scratch2.stride(1),
        scratch2.stride(2),
        BLOCK_M=BM,
        BLOCK_N=BN1,
        BLOCK_K=BK1,
        RBLOCK=RBLOCK,
        MODE=mode,
        STORE_F32=quant_in,
        SCALE_PER_EXPERT=(mode == 2),
        APPLY_WEIGHT_ON_INPUT=(apply_router_weight_on_input and mode == 0),
        USE_INT_DOT=False,
        COMPUTE_DTYPE=compute_dtype,
        N_DIV=(N_inter % BN1 == 0),
        K_DIV=(H % BK1 == 0),
        COMPACT=compact,
        num_warps=4,
        num_stages=2,
    )

    if quant_in:
        BMq = 32 if R >= 32 else 16
        _quantize_scratch_kernel[(E, triton.cdiv(R, BMq))](
            scratch2,
            scratch2_q,
            a2_scale_t,
            topk_ids,
            count,
            off,
            sorted_ids,
            M,
            N_inter,
            TOPK,
            R,
            scratch2.stride(0),
            scratch2.stride(1),
            scratch2.stride(2),
            scratch2_q.stride(0),
            scratch2_q.stride(1),
            scratch2_q.stride(2),
            a2_scale_t.stride(0),
            a2_scale_t.stride(1),
            BLOCK_M=BMq,
            BLOCK_K=256,
            QMAX=QMAX,
            QMIN=QMIN,
            IS_INT8=(mode == 1),
            COMPACT=compact,
            num_warps=4,
        )

    grid2 = (E, triton.cdiv(R, BM), triton.cdiv(Ko, BN2))
    _moe_gemm2_kernel[grid2](
        scratch2_q if quant_in else scratch2,
        a2_scale_t if quant_in else scratch2,
        w2,
        w2_scale if w2_scale is not None else w2,
        topk_weights,
        topk_ids,
        scratch3,
        count,
        off,
        sorted_ids,
        M,
        N_inter,
        Ko,
        TOPK,
        R,
        scratch2.stride(0),
        scratch2.stride(1),
        scratch2.stride(2),
        a2_scale_t.stride(0) if quant_in else 0,
        a2_scale_t.stride(1) if quant_in else 0,
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        w2_scale.stride(0) if (w2_scale is not None and w2_scale.dim() >= 2) else 0,
        w2_scale.stride(1) if (w2_scale is not None and w2_scale.dim() >= 2) else 0,
        scratch3.stride(0),
        BLOCK_M=BM,
        BLOCK_N=BN2,
        BLOCK_K=BK2,
        RBLOCK=RBLOCK,
        MODE=mode,
        SCALE_PER_EXPERT=(mode == 2),
        USE_INT_DOT=False,
        COMPUTE_DTYPE=compute_dtype,
        OUT_DTYPE=out_dtype,
        APPLY_WEIGHT_ON_INPUT=apply_router_weight_on_input,
        N_DIV=(Ko % BN2 == 0),
        K_DIV=(N_inter % BK2 == 0),
        COMPACT=compact,
        num_warps=4,
        num_stages=2,
    )

    BMs = 32 if M >= 32 else 16
    BNs = 256 if Ko % 256 == 0 else 128
    _moe_sum_kernel[(triton.cdiv(M, BMs), triton.cdiv(Ko, BNs))](
        scratch3,
        output,
        M,
        Ko,
        scratch3.stride(0),
        output.stride(0),
        BLOCK_M=BMs,
        BLOCK_N=BNs,
        TOPK=TOPK,
        OUT_DTYPE=out_dtype,
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
    """Hygon-optimized fused experts implementation.

    Main entry point matching the standard FlagGems-vllm interface.
    Dispatches to the optimized Hygon kernel.
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
    """In-place variant of fused experts for Hygon.

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
    """Out-of-place variant of fused experts for Hygon.

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
