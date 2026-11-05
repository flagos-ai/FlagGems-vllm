# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""GPU-selected continuous-KV Split-KV for bounded page32/GQA8 Mixed."""
import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._metax.ops.flash_attention.common import (
    tn_compile_scenario,
)
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.single_prefill import (
    _pv_layout_jit,
    _scaled_online_softmax_t,
    _scatter_decode,
    _single_prefill_gather,
    _single_prefill_gather_q,
)
from flaggems_vllm.utils import libentry


@libentry()
@_pv_layout_jit(do_not_specialize=["total_q", "host_q_len", "host_k_len"])
def flash_varlen_fwd_d256_tn_contiguous_splitkv_kernel(
    DESC,
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    seqused_k_ptr,
    IS_CU_SEQLENS_K: tl.constexpr,
    IS_SEQUSED_K: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_RATIO: tl.constexpr,
    total_q,
    host_q_len,
    host_k_len,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    k_page_stride,
    v_page_stride,
    worklist_ptr,
    COMPACT_WORKLIST: tl.constexpr,
    GRID_ORDER: tl.constexpr,
    REVERSE_Q_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QK_BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    PACK_GROUP: tl.constexpr,
    TN_SCENARIO: tl.constexpr,
    TP2_FIRST_TILE: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 32,
    MASK_K_TAIL: tl.constexpr = True,
    FULL_Q: tl.constexpr = False,
    PAD_K: tl.constexpr = 0,
    PAD_Q: tl.constexpr = 0,
    INTERNAL_SPLITS: tl.constexpr = 12,
):
    tl.static_assert(
        BLOCK_K == 128 and QK_BLOCK_K == 64 and (BLOCK_N == 128),
        ("D256 TN requires QK BK64, PV D128, BN128"),
    )
    tl.static_assert(not COMPACT_WORKLIST, "Fast attention requires a rectangular grid")
    if GRID_ORDER == 1:
        head_pid = tl.program_id(0)
        task_pid = tl.program_id(2)
    else:
        task_pid = tl.program_id(0)
        head_pid = tl.program_id(2)
    if REVERSE_Q_TILES:
        if GRID_ORDER == 1:
            task_pid = tl.num_programs(2) - 1 - task_pid
        else:
            task_pid = tl.num_programs(0) - 1 - task_pid
    split_id = head_pid % INTERNAL_SPLITS
    hid = head_pid // INTERNAL_SPLITS
    if COMPACT_WORKLIST:
        descriptor_offset = task_pid * 2
        bid = tl.load(DESC)
        if bid < 0:
            return
        m_block = tl.load(worklist_ptr + descriptor_offset + 1).to(tl.int32)
    else:
        bid = tl.load(DESC)
        if bid < 0:
            return
        m_block = task_pid
    q_bos = tl.load(cu_seqlens_q_ptr + bid)
    q_len = tl.load(cu_seqlens_q_ptr + bid + 1) - q_bos
    q_bos = 0
    o_ptr += split_id * total_q * NUM_HEADS * 256
    softmax_lse_ptr += split_id * total_q * NUM_HEADS
    effective_q_len = q_len * PACK_GROUP
    if m_block * BLOCK_M >= effective_q_len:
        return
    k_len = tl.load(seqused_k_ptr + bid)
    packed_row = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_row = tl.max_contiguous(tl.multiple_of(packed_row, BLOCK_M), BLOCK_M)
    row_idx = packed_row // PACK_GROUP
    query_head = hid * PACK_GROUP + packed_row % PACK_GROUP
    kv_head = hid // (HEAD_RATIO // PACK_GROUP)
    q_block_end = tl.cdiv((m_block + 1) * BLOCK_M, PACK_GROUP)
    n_block_min = 0
    n_block_max = min(
        tl.cdiv(k_len, BLOCK_N), tl.cdiv(q_block_end + k_len - q_len, BLOCK_N)
    )
    n_block_max = tl.maximum(n_block_max, 0)
    blocks_per_split = tl.cdiv(n_block_max, INTERNAL_SPLITS)
    n_block_min = tl.minimum(n_block_max, split_id * blocks_per_split)
    n_block_max = tl.minimum(n_block_max, n_block_min + blocks_per_split)
    n_masking_steps = min(n_block_max - n_block_min, 2)
    q_base = q_ptr
    o_base = o_ptr + q_bos * o_row_stride
    k_base = k_ptr + kv_head * k_head_stride
    v_base = v_ptr + kv_head * v_head_stride
    page_table_ptr += bid * page_table_batch_stride
    d_idx = tl.arange(0, BLOCK_K)
    output_d_offset = 0
    acc0 = tl.zeros((BLOCK_K, BLOCK_M), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_K, BLOCK_M), dtype=tl.float32)
    row_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    qk_d_idx = tl.arange(0, QK_BLOCK_K)
    n_block = n_block_max - 1
    for mask_step in tl.static_range(0, 2):
        if mask_step < n_masking_steps:
            start_n = n_block * BLOCK_N
            col_idx = tl.max_contiguous(
                tl.multiple_of(start_n + tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N
            )
            k_cache_offset = col_idx * 64
            scores_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            q_tile_ptr = (
                q_base
                + (row_idx[None, :] * NUM_HEADS + query_head[None, :]) * QK_BLOCK_K
                + qk_d_idx[:, None]
            )
            k_tile_ptr = k_base + qk_d_idx[None, :] + k_cache_offset[:, None]
            for _d_block in tl.range(0, 256 // QK_BLOCK_K):
                k_tile = tl.load(k_tile_ptr)
                q_tile = tl.load(q_tile_ptr)
                scores_t = tl.dot(k_tile, q_tile, scores_t)
                q_tile_ptr += PAD_Q * NUM_HEADS * QK_BLOCK_K
                k_tile_ptr += PAD_K * QK_BLOCK_K
            causal_end = tl.minimum(k_len - 1, row_idx + k_len - q_len)
            scores_t = tl.where(
                col_idx[:, None] > causal_end[None, :], float("-inf"), scores_t
            )
            v_cache_offset = col_idx * 128
            if PAGE_SIZE == 16 or MASK_K_TAIL:
                v0 = tl.load(
                    v_base + v_cache_offset[:, None] + d_idx[None, :],
                    mask=col_idx[:, None] < k_len,
                    other=0.0,
                )
            else:
                v0 = tl.load(v_base + v_cache_offset[:, None] + d_idx[None, :])
            (acc_scale, probabilities, row_max, row_sum) = _scaled_online_softmax_t(
                scores_t,
                row_max,
                row_sum,
                scale_softmax_log2=scale_softmax_log2,
                IS_BORDER=True,
            )
            probabilities = probabilities.to(v_ptr.type.element_ty)
            if not mask_step == 0:
                acc0 = acc0 * acc_scale[None, :]
            if mask_step == 0:
                acc0 = tl.dot(tl.trans(v0), probabilities)
            else:
                acc0 = tl.dot(tl.trans(v0), probabilities, acc0)
            if not mask_step == 0:
                acc1 = acc1 * acc_scale[None, :]
            if PAGE_SIZE == 16 or MASK_K_TAIL:
                v1 = tl.load(
                    v_base
                    + v_cache_offset[:, None]
                    + (PAD_K * BLOCK_K + d_idx)[None, :],
                    mask=col_idx[:, None] < k_len,
                    other=0.0,
                )
            else:
                v1 = tl.load(
                    v_base
                    + v_cache_offset[:, None]
                    + (PAD_K * BLOCK_K + d_idx)[None, :]
                )
            if mask_step == 0:
                acc1 = tl.dot(tl.trans(v1), probabilities)
            else:
                acc1 = tl.dot(tl.trans(v1), probabilities, acc1)
            pass
            n_block -= 1
    main_start = n_block_max - n_masking_steps - 1
    for n_block in tl.range(main_start, n_block_min - 1, step=-1):
        start_n = n_block * BLOCK_N
        col_idx = tl.max_contiguous(
            tl.multiple_of(start_n + tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N
        )
        k_cache_offset = col_idx * 64
        scores_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        q_tile_ptr = (
            q_base
            + (row_idx[None, :] * NUM_HEADS + query_head[None, :]) * QK_BLOCK_K
            + qk_d_idx[:, None]
        )
        k_tile_ptr = k_base + qk_d_idx[None, :] + k_cache_offset[:, None]
        for _d_block in tl.range(0, 256 // QK_BLOCK_K):
            k_tile = tl.load(k_tile_ptr)
            q_tile = tl.load(q_tile_ptr)
            scores_t = tl.dot(k_tile, q_tile, scores_t)
            q_tile_ptr += PAD_Q * NUM_HEADS * QK_BLOCK_K
            k_tile_ptr += PAD_K * QK_BLOCK_K
        v_cache_offset = col_idx * 128
        v0 = tl.load(v_base + v_cache_offset[:, None] + d_idx[None, :])
        (acc_scale, probabilities, row_max, row_sum) = _scaled_online_softmax_t(
            scores_t,
            row_max,
            row_sum,
            scale_softmax_log2=scale_softmax_log2,
            IS_BORDER=False,
        )
        probabilities = probabilities.to(v_ptr.type.element_ty)
        acc0 = acc0 * acc_scale[None, :]
        acc0 = tl.dot(tl.trans(v0), probabilities, acc0)
        acc1 = acc1 * acc_scale[None, :]
        v1 = tl.load(
            v_base + v_cache_offset[:, None] + (PAD_K * BLOCK_K + d_idx)[None, :]
        )
        acc1 = tl.dot(tl.trans(v1), probabilities, acc1)
        pass
    o_mask = True if FULL_Q else row_idx[:, None] < q_len
    o0_ptr = (
        o_base
        + row_idx[:, None] * o_row_stride
        + query_head[:, None] * o_head_stride
        + output_d_offset
        + d_idx[None, :]
    )
    o1_ptr = (
        o_base
        + row_idx[:, None] * o_row_stride
        + query_head[:, None] * o_head_stride
        + BLOCK_K
        + d_idx[None, :]
    )
    lse = tl.where(
        (row_sum == 0) | (row_sum != row_sum),
        float(("-inf")),
        row_max * (scale_softmax / scale_softmax_log2) + tl.log(row_sum),
    )
    inv_sum = tl.where((row_sum == 0) | (row_sum != row_sum), 1.0, 1.0 / row_sum)
    acc0_f32 = tl.trans(acc0) * inv_sum[:, None]
    acc1_f32 = tl.trans(acc1) * inv_sum[:, None]
    tl.store(o0_ptr, acc0_f32.to(o_ptr.type.element_ty), mask=o_mask)
    tl.store(o1_ptr, acc1_f32.to(o_ptr.type.element_ty), mask=o_mask)
    lse_mask = True if FULL_Q else row_idx < q_len
    tl.store(
        softmax_lse_ptr + query_head * total_q + q_bos + row_idx, lse, mask=lse_mask
    )


@triton.jit
def _select_contiguous_splitkv(
    CU,
    USED,
    DESC,
    DCU,
    DU,
    B: tl.constexpr,
    R: tl.constexpr,
    MAX_Q: tl.constexpr,
    MAX_K: tl.constexpr,
    TOTAL: tl.constexpr,
):
    b = tl.arange(0, R)
    start = tl.load(CU + b, b <= B, other=0)
    end = tl.load(CU + b + 1, b < B, other=0)
    q = end - start
    k = tl.load(USED + b, b < B, other=0)
    large = (b < B) & (q > 128)
    valid = (tl.sum(large.to(tl.int32), 0) == 1) & (
        tl.sum(((b < B) & (q != 1) & ~large).to(tl.int32), 0) == 0
    )
    invalid = (b < B) & ((q < 0) | (q > MAX_Q) | (k < 0) | (k > MAX_K))
    valid = (
        valid
        & (tl.sum(invalid.to(tl.int32), 0) == 0)
        & (tl.load(CU) == 0)
        & (tl.load(CU + B) == TOTAL)
    )
    bid = tl.max(tl.where(large, b, -1), 0)
    tl.store(DESC, tl.where(valid, bid, -1))
    tl.store(DCU + b, b, b <= B)
    tl.store(DU + b, tl.where(valid & (q == 1), k, 0), b < B)


@triton.jit
def _copy_selected_decode_q(Q, DQ, CU, DESC, QR: tl.constexpr, QH: tl.constexpr):
    b = tl.program_id(0)
    start = tl.load(CU + b)
    end = tl.load(CU + b + 1)
    x = tl.arange(0, 2048)
    valid = (tl.load(DESC) >= 0) & (end - start == 1)
    values = tl.load(Q + start * QR + x // 256 * QH + x % 256, valid, other=0.0)
    tl.store(DQ + b * 2048 + x, values)


@triton.jit
def _merge_selected_splitkv(
    OP,
    LP,
    O,
    L,
    CU,
    DESC,
    QR: tl.constexpr,
    S: tl.constexpr,
    SS: tl.constexpr,
    TOTAL,
    OR,
    OH,
):
    r = tl.program_id(0)
    h = tl.program_id(1)
    bid = tl.load(DESC)
    if bid < 0:
        return
    start = tl.load(CU + bid)
    length = tl.load(CU + bid + 1) - start
    if r >= length:
        return
    ss = tl.arange(0, SS)
    dd = tl.arange(0, 256)
    ls = tl.load(LP + ss * 8 * QR + h * QR + r, ss < S, other=float("-inf"))
    valid = (ss < S) & (ls > float("-inf")) & (ls < float("inf"))
    maximum = tl.max(tl.where(valid, ls, float("-inf")), 0)
    maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    weight = tl.where(valid, tl.exp(ls - maximum), 0.0)
    den = tl.sum(weight, 0)
    values = tl.load(
        OP + ss[:, None] * QR * 2048 + r * 2048 + h * 256 + dd[None, :],
        ss[:, None] < S,
        other=0.0,
    )
    result = tl.sum(values * weight[:, None], 0) / tl.where(den > 0, den, 1.0)
    tl.store(O + (start + r) * OR + h * OH + dd, result.to(O.type.element_ty))
    tl.store(
        L + h * TOTAL + start + r,
        tl.where(den > 0, maximum + tl.log(den), float("-inf")),
    )


def can_use_contiguous_splitkv(p, num_splits):
    return (
        num_splits == 0
        and 2 <= p.b <= 8
        and p.h == 8
        and p.hk == 1
        and p.d == 256
        and p.block_size == 32
        and p.is_paged
        and p.is_cu_seqlens_q
        and p.is_seqused_k
        and not p.seqlenq_ngroups_swapped
        and 128 < p.seqlen_q <= 132
        and p.total_q == p.seqlen_q + p.b - 1
        and 32768 <= p.seqlen_k <= 67584
        and p.is_causal
        and not p.is_local
        and not p.is_dropout
        and not p.is_alibi
        and not p.is_softcap
        and p.q_ptr.dtype == torch.bfloat16
        and all(x.stride(-1) == 1 for x in (p.q_ptr, p.k_ptr, p.v_ptr, p.o_ptr))
    )


def launch_contiguous_splitkv(p, *, fallback):
    from flaggems_vllm.ops.attention import flash_attn_varlen_func

    q, k, v, out = p.q_ptr, p.k_ptr, p.v_ptr, p.o_ptr
    cu, used, table = p.cu_seqlens_q_ptr, p.seqused_k_ptr, p.page_table_ptr
    b, total, maxq, maxk = p.b, p.total_q, p.seqlen_q, p.seqlen_k
    pad = triton.cdiv(maxk, 128) * 128
    padq = triton.cdiv(maxq, 128) * 128
    splits = 12
    desc = torch.empty((1,), dtype=torch.int32, device=q.device)
    dcu = torch.empty((b + 1,), dtype=torch.int32, device=q.device)
    du = torch.empty((b,), dtype=torch.int32, device=q.device)
    dq = torch.empty((b, 8, 256), dtype=q.dtype, device=q.device)
    do = torch.empty_like(dq)
    kc = torch.empty((pad, 1, 256), dtype=k.dtype, device=k.device)
    vc = torch.empty_like(kc)
    qc = torch.empty((4, padq * 8, 64), dtype=q.dtype, device=q.device)
    op = torch.empty((splits, maxq, 8, 256), dtype=torch.float32, device=q.device)
    lp = torch.empty((splits, 8, maxq), dtype=torch.float32, device=q.device)
    _select_contiguous_splitkv[1,](
        cu,
        used,
        desc,
        dcu,
        du,
        b,
        triton.next_power_of_2(b + 1),
        maxq,
        maxk,
        total,
        num_warps=1,
    )
    _single_prefill_gather[triton.cdiv(pad * 256, 1024),](
        k,
        v,
        kc,
        vc,
        table,
        used,
        desc,
        k.stride(0),
        v.stride(0),
        k.stride(1),
        v.stride(1),
        table.stride(0),
        pad,
        1024,
        num_warps=4,
        num_stages=1,
        PAGE_SIZE=32,
    )
    _single_prefill_gather_q[triton.cdiv(padq * 8 * 256, 1024),](
        q,
        qc,
        cu,
        desc,
        q.stride(0),
        q.stride(1),
        padq,
        1024,
        num_warps=4,
        num_stages=1,
        HEADS=8,
    )
    _copy_selected_decode_q[b,](q, dq, cu, desc, q.stride(0), q.stride(1), num_warps=4)
    _, dl = flash_attn_varlen_func(
        q=dq,
        k=k,
        v=v,
        cu_seqlens_q=dcu,
        seqused_k=du,
        max_seqlen_q=1,
        max_seqlen_k=maxk,
        block_table=table,
        out=do,
        return_softmax_lse=True,
        causal=True,
        num_splits=0,
        softmax_scale=p.scale_softmax,
    )
    _scatter_decode[b,](
        desc,
        out,
        p.softmax_lse_ptr,
        do,
        dl,
        cu,
        out.stride(0),
        out.stride(1),
        total,
        b,
        dl.stride(0),
        dl.stride(1),
        num_warps=4,
        HEADS=8,
    )
    scenario = tn_compile_scenario(qc, kc, vc, op, lp, cu, used, table, desc)
    result = flash_varlen_fwd_d256_tn_contiguous_splitkv_kernel[
        triton.cdiv(maxq * 8, 64), 1, splits
    ](
        desc,
        qc,
        kc,
        vc,
        op,
        lp,
        q.stride(0),
        256,
        256,
        q.stride(1),
        256,
        256,
        2048,
        256,
        p.scale_softmax,
        p.scale_softmax_log2,
        cu,
        None,
        used,
        False,
        True,
        8,
        8,
        maxq,
        maxq,
        maxk,
        table,
        table.stride(0),
        0,
        0,
        table,
        COMPACT_WORKLIST=False,
        GRID_ORDER=0,
        REVERSE_Q_TILES=True,
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=128,
        QK_BLOCK_K=64,
        num_warps=4,
        num_stages=4,
        PACK_GROUP=8,
        TN_SCENARIO=scenario,
        TP2_FIRST_TILE=False,
        PAGE_SIZE=32,
        MASK_K_TAIL=True,
        FULL_Q=False,
        PAD_K=pad,
        PAD_Q=padq,
        INTERNAL_SPLITS=splits,
        pipeline="cpasync",
        scenario=scenario,
        inner_stages=(0, 0),
        pipeline_load_num=-1,
    )
    _merge_selected_splitkv[maxq, 8](
        op,
        lp,
        out,
        p.softmax_lse_ptr,
        cu,
        desc,
        maxq,
        splits,
        triton.next_power_of_2(splits),
        total,
        out.stride(0),
        out.stride(1),
        num_warps=4,
    )
    fallback(p, num_splits=0, _skip_single_prefill=True, _skip_selection_desc=desc)
    return result
