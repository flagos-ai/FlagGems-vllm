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

import logging

import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

from .common import apply_mask, tn_compile_scenario

logger = logging.getLogger(__name__)


_D256_FRAGMENT = 128


_QK_FRAGMENT = 64


@triton.jit
def _direct_paged_tile_coords(
    start_n,
    k_len,
    page_table_ptr,
    BLOCK_N: tl.constexpr,
    BOUNDARY_CHECK: tl.constexpr,
    PAGE_SIZE: tl.constexpr = 32,
):
    """Resolve physical rows from the page table."""
    if PAGE_SIZE == 16:
        tl.static_assert(BLOCK_N == 128, "page16 scalar gather requires BN128")
        page_slot = tl.arange(0, BLOCK_N) // 16
        virtual_page = start_n // 16
        if BOUNDARY_CHECK:
            page0 = tl.load(
                page_table_ptr + virtual_page + 0, mask=start_n + 0 < k_len, other=0
            )
        else:
            page0 = tl.load(page_table_ptr + virtual_page + 0)
        if BOUNDARY_CHECK:
            page1 = tl.load(
                page_table_ptr + virtual_page + 1, mask=start_n + 16 < k_len, other=0
            )
        else:
            page1 = tl.load(page_table_ptr + virtual_page + 1)
        if BOUNDARY_CHECK:
            page2 = tl.load(
                page_table_ptr + virtual_page + 2, mask=start_n + 32 < k_len, other=0
            )
        else:
            page2 = tl.load(page_table_ptr + virtual_page + 2)
        if BOUNDARY_CHECK:
            page3 = tl.load(
                page_table_ptr + virtual_page + 3, mask=start_n + 48 < k_len, other=0
            )
        else:
            page3 = tl.load(page_table_ptr + virtual_page + 3)
        if BOUNDARY_CHECK:
            page4 = tl.load(
                page_table_ptr + virtual_page + 4, mask=start_n + 64 < k_len, other=0
            )
        else:
            page4 = tl.load(page_table_ptr + virtual_page + 4)
        if BOUNDARY_CHECK:
            page5 = tl.load(
                page_table_ptr + virtual_page + 5, mask=start_n + 80 < k_len, other=0
            )
        else:
            page5 = tl.load(page_table_ptr + virtual_page + 5)
        if BOUNDARY_CHECK:
            page6 = tl.load(
                page_table_ptr + virtual_page + 6, mask=start_n + 96 < k_len, other=0
            )
        else:
            page6 = tl.load(page_table_ptr + virtual_page + 6)
        if BOUNDARY_CHECK:
            page7 = tl.load(
                page_table_ptr + virtual_page + 7, mask=start_n + 112 < k_len, other=0
            )
        else:
            page7 = tl.load(page_table_ptr + virtual_page + 7)
        page_id = tl.where(
            page_slot < 4,
            tl.where(
                page_slot < 2,
                tl.where(page_slot == 0, page0, page1),
                tl.where(page_slot == 2, page2, page3),
            ),
            tl.where(
                page_slot < 6,
                tl.where(page_slot == 4, page4, page5),
                tl.where(page_slot == 6, page6, page7),
            ),
        ).to(tl.int64)
        col_idx = start_n + tl.arange(0, BLOCK_N)
        col_idx = tl.max_contiguous(tl.multiple_of(col_idx, BLOCK_N), BLOCK_N)
        return (col_idx, page_id, col_idx % 16)
    else:
        page_slot = tl.arange(0, BLOCK_N) // 32
        virtual_page = start_n // 32
        if BOUNDARY_CHECK:
            page0 = tl.load(
                page_table_ptr + virtual_page, mask=start_n < k_len, other=0
            ).to(tl.int64)
        else:
            page0 = tl.load(page_table_ptr + virtual_page).to(tl.int64)
        if BLOCK_N == 32:
            page_id = page0
        elif BLOCK_N == 64:
            if BOUNDARY_CHECK:
                page1 = tl.load(
                    page_table_ptr + virtual_page + 1,
                    mask=start_n + 32 < k_len,
                    other=0,
                ).to(tl.int64)
            else:
                page1 = tl.load(page_table_ptr + virtual_page + 1).to(tl.int64)
            page_id = tl.where(page_slot == 0, page0, page1)
        elif BLOCK_N == 128:
            if BOUNDARY_CHECK:
                page1 = tl.load(
                    page_table_ptr + virtual_page + 1,
                    mask=start_n + 32 < k_len,
                    other=0,
                ).to(tl.int64)
                page2 = tl.load(
                    page_table_ptr + virtual_page + 2,
                    mask=start_n + 64 < k_len,
                    other=0,
                ).to(tl.int64)
                page3 = tl.load(
                    page_table_ptr + virtual_page + 3,
                    mask=start_n + 96 < k_len,
                    other=0,
                ).to(tl.int64)
            else:
                page1 = tl.load(page_table_ptr + virtual_page + 1).to(tl.int64)
                page2 = tl.load(page_table_ptr + virtual_page + 2).to(tl.int64)
                page3 = tl.load(page_table_ptr + virtual_page + 3).to(tl.int64)
            page_id = tl.where(
                page_slot == 0,
                page0,
                tl.where(page_slot == 1, page1, tl.where(page_slot == 2, page2, page3)),
            )
        else:
            tl.static_assert(False, "Async-TN BLOCK_N must be 32, 64, or 128")
        col_idx = start_n + tl.arange(0, BLOCK_N)
        col_idx = tl.max_contiguous(tl.multiple_of(col_idx, BLOCK_N), BLOCK_N)
        page_offset = col_idx % 32
        return (col_idx, page_id, page_offset)


@triton.jit
def _direct_online_softmax_stats(
    scores, row_max, row_sum, scale_softmax_log2, IS_BORDER: tl.constexpr
):
    previous_max = row_max
    row_max = tl.maximum(row_max, tl.max(scores, 1))
    if IS_BORDER:
        current_max = tl.where(row_max == float("-inf"), 0.0, row_max)
    else:
        current_max = row_max
    accumulator_scale = tl.math.exp2((previous_max - current_max) * scale_softmax_log2)
    row_sum *= accumulator_scale
    max_scaled = tl.where(row_max == float("-inf"), 0.0, row_max * scale_softmax_log2)
    probabilities = tl.math.exp2(scores * scale_softmax_log2 - max_scaled[:, None])
    row_sum += tl.sum(probabilities, 1)
    return (accumulator_scale, probabilities, row_max, row_sum)


@libentry()
@triton.jit(do_not_specialize=["total_q", "fast_q_len", "fast_k_len"])
def flash_varlen_fwd_d256_tn_kernel(
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
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    k_page_stride,
    v_page_stride,
    worklist_ptr,
    COMPACT_WORKLIST: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    Q_TILES: tl.constexpr,
    GRID_ORDER: tl.constexpr,
    REVERSE_Q_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QK_BLOCK_K: tl.constexpr,
    FAST_CAUSAL_ALIGNED: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    BULK_TAIL: tl.constexpr = False,
    TAIL_ROWS: tl.constexpr = 0,
    TAIL_UNALIGNED_ONLY: tl.constexpr = False,
    PEEL_Q: tl.constexpr = False,
    PREFIX_ONLY: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 32,
    FALLBACK_FOR_FAST: tl.constexpr = False,
    fast_q_len=0,
    fast_k_len=0,
):
    tl.static_assert(
        BLOCK_K == 128 and QK_BLOCK_K == 64 and (BLOCK_N == 128),
        "D256 TN requires QK BK64, PV D128, BN128",
    )
    tl.static_assert(
        (not SPLIT_KV) | (not COMPACT_WORKLIST) & (NUM_SPLITS > 1),
        "Async-TN Split-KV requires a rectangular multi-split grid",
    )
    tl.static_assert(
        SPLIT_KV | (NUM_SPLITS == 1) & (Q_TILES > 0),
        "Async-TN Direct requires the neutral split configuration",
    )
    if GRID_ORDER == 1:
        head_pid = tl.program_id(0)
        rectangular_bid = tl.program_id(1)
        task_pid = tl.program_id(2)
    else:
        task_pid = tl.program_id(0)
        rectangular_bid = tl.program_id(1)
        head_pid = tl.program_id(2)
    if (not SPLIT_KV) & REVERSE_Q_TILES:
        if GRID_ORDER == 1:
            task_pid = tl.num_programs(2) - 1 - task_pid
        else:
            task_pid = tl.num_programs(0) - 1 - task_pid
    hid = head_pid
    if COMPACT_WORKLIST:
        descriptor_offset = task_pid * 2
        bid = tl.load(worklist_ptr + descriptor_offset).to(tl.int32)
        if bid < 0:
            return
        m_block = tl.load(worklist_ptr + descriptor_offset + 1).to(tl.int32)
        split_id = 0
    elif SPLIT_KV:
        bid = rectangular_bid
        split_id = task_pid // Q_TILES
        m_block = task_pid % Q_TILES
    else:
        bid = rectangular_bid
        m_block = task_pid
        split_id = 0
    q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
    q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
    q_len = q_eos - q_bos
    effective_q_len = q_len * HEAD_RATIO
    if m_block * BLOCK_M >= effective_q_len:
        return
    if IS_SEQUSED_K:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        tl.static_assert(IS_CU_SEQLENS_K, "Async-TN requires a KV length source")
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
        k_len = k_eos - k_bos
    if FALLBACK_FOR_FAST:
        # Exactly one of this kernel and fast owns the request on every replay.
        if q_bos == 0 and q_len == fast_q_len and k_len == fast_k_len:
            return
    if PEEL_Q:
        prefix_q = min(q_len, q_len - k_len & 64 // HEAD_RATIO - 1)
        q_bos += prefix_q
        q_len -= prefix_q
    effective_q_len = q_len * HEAD_RATIO
    q_valid_len = q_len
    if PREFIX_ONLY:
        q_valid_len = min(q_len, q_len - k_len & 64 // HEAD_RATIO - 1)
        if m_block * BLOCK_M >= q_valid_len * HEAD_RATIO:
            return
    output_total_q = total_q
    output_q_bos = q_bos
    if PREFIX_ONLY & SPLIT_KV:
        output_total_q = TAIL_ROWS
        output_q_bos = bid * (64 // HEAD_RATIO)
    if BULK_TAIL:
        aligned_tail = (k_len - q_len) % (64 // HEAD_RATIO) == 0
        if TAIL_UNALIGNED_ONLY:
            if aligned_tail:
                return
        if SPLIT_KV:
            if not aligned_tail:
                return
            full_q = max(0, min(q_len, k_len // BLOCK_N * BLOCK_N - (k_len - q_len)))
            full_q = full_q // (64 // HEAD_RATIO) * (64 // HEAD_RATIO)
            m_block += full_q * HEAD_RATIO // BLOCK_M
            if m_block * BLOCK_M >= effective_q_len:
                return
            output_total_q = TAIL_ROWS
            output_q_bos = bid * 128 - full_q
    packed_row = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_row = tl.max_contiguous(tl.multiple_of(packed_row, BLOCK_M), BLOCK_M)
    row_idx = packed_row // HEAD_RATIO
    query_head = hid * HEAD_RATIO + packed_row % HEAD_RATIO
    kv_head = hid
    q_block_end = tl.cdiv((m_block + 1) * BLOCK_M, HEAD_RATIO)
    if BULK_TAIL:
        bulk_q_end = (m_block * BLOCK_M // 64 + 1) * (64 // HEAD_RATIO)
        bulk_valid = (
            (bulk_q_end <= q_len)
            & (bulk_q_end + k_len - q_len <= k_len // BLOCK_N * BLOCK_N)
            & ((k_len - q_len) % (64 // HEAD_RATIO) == 0)
        )
        if bulk_valid:
            return
    n_block_min = 0
    n_block_max = min(
        tl.cdiv(k_len, BLOCK_N), tl.cdiv(q_block_end + k_len - q_len, BLOCK_N)
    )
    if SPLIT_KV:
        span_blocks = max(n_block_max - n_block_min, 0)
        blocks_per_split = tl.cdiv(span_blocks, NUM_SPLITS)
        split_block_min = min(n_block_max, n_block_min + split_id * blocks_per_split)
        split_block_max = min(n_block_max, split_block_min + blocks_per_split)
        n_block_min = split_block_min
        n_block_max = split_block_max
    if FAST_CAUSAL_ALIGNED:
        n_masking_steps = min(n_block_max - n_block_min, 1)
    else:
        n_masking_steps = min(n_block_max - n_block_min, tl.cdiv(BLOCK_M, BLOCK_N) + 1)
    if SPLIT_KV:
        o_ptr += split_id * output_total_q * NUM_HEADS * 256
        softmax_lse_ptr += split_id * NUM_HEADS * output_total_q
    q_base = q_ptr + q_bos * q_row_stride
    o_base = o_ptr + output_q_bos * o_row_stride
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
    for _ in tl.range(0, n_masking_steps):
        start_n = n_block * BLOCK_N
        (col_idx, page_id, page_offset) = _direct_paged_tile_coords(
            start_n,
            k_len,
            page_table_ptr,
            BLOCK_N=BLOCK_N,
            BOUNDARY_CHECK=True,
            PAGE_SIZE=PAGE_SIZE,
        )
        k_cache_offset = page_id * k_page_stride + page_offset * k_row_stride
        scores_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        q_tile_ptr = (
            q_base
            + row_idx[None, :] * q_row_stride
            + query_head[None, :] * q_head_stride
            + qk_d_idx[:, None]
        )
        k_tile_ptr = k_base + qk_d_idx[None, :] + k_cache_offset[:, None]
        for _d_block in tl.range(0, 256 // QK_BLOCK_K):
            if FAST_CAUSAL_ALIGNED:
                k_tile = tl.load(k_tile_ptr)
            else:
                k_tile = tl.load(k_tile_ptr, mask=col_idx[:, None] < k_len, other=0.0)
            q_tile = tl.load(q_tile_ptr, mask=row_idx[None, :] < q_len, other=0.0)
            scores_t = tl.dot(k_tile, q_tile, scores_t)
            q_tile_ptr += QK_BLOCK_K
            k_tile_ptr += QK_BLOCK_K
        scores = tl.trans(scores_t)
        scores = apply_mask(
            scores,
            col_idx,
            row_idx,
            q_len,
            k_len,
            0,
            0,
            is_even_mn=False,
            is_causal=True,
            is_local=False,
        )
        v_cache_offset = page_id * v_page_stride + page_offset * v_row_stride
        if FAST_CAUSAL_ALIGNED:
            v0 = tl.load(v_base + v_cache_offset[:, None] + d_idx[None, :])
        else:
            v0 = tl.load(
                v_base + v_cache_offset[:, None] + d_idx[None, :],
                mask=col_idx[:, None] < k_len,
                other=0.0,
            )
        (acc_scale, probabilities, row_max, row_sum) = _direct_online_softmax_stats(
            scores,
            row_max,
            row_sum,
            scale_softmax_log2=scale_softmax_log2,
            IS_BORDER=True,
        )
        probabilities = probabilities.to(v_ptr.type.element_ty)
        acc0 = acc0 * acc_scale[None, :]
        acc0 = tl.dot(tl.trans(v0), tl.trans(probabilities), acc0)
        acc1 = acc1 * acc_scale[None, :]
        if FAST_CAUSAL_ALIGNED:
            v1 = tl.load(v_base + v_cache_offset[:, None] + (BLOCK_K + d_idx)[None, :])
        else:
            v1 = tl.load(
                v_base + v_cache_offset[:, None] + (BLOCK_K + d_idx)[None, :],
                mask=col_idx[:, None] < k_len,
                other=0.0,
            )
        acc1 = tl.dot(tl.trans(v1), tl.trans(probabilities), acc1)
        pass
        n_block -= 1
    for n_block in tl.range(
        n_block_max - n_masking_steps - 1, n_block_min - 1, step=-1
    ):
        start_n = n_block * BLOCK_N
        (col_idx, page_id, page_offset) = _direct_paged_tile_coords(
            start_n,
            k_len,
            page_table_ptr,
            BLOCK_N=BLOCK_N,
            BOUNDARY_CHECK=False,
            PAGE_SIZE=PAGE_SIZE,
        )
        k_cache_offset = page_id * k_page_stride + page_offset * k_row_stride
        scores_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        q_tile_ptr = (
            q_base
            + row_idx[None, :] * q_row_stride
            + query_head[None, :] * q_head_stride
            + qk_d_idx[:, None]
        )
        k_tile_ptr = k_base + qk_d_idx[None, :] + k_cache_offset[:, None]
        for _d_block in tl.range(0, 256 // QK_BLOCK_K):
            k_tile = tl.load(k_tile_ptr)
            q_tile = tl.load(q_tile_ptr, mask=row_idx[None, :] < q_len, other=0.0)
            scores_t = tl.dot(k_tile, q_tile, scores_t)
            q_tile_ptr += QK_BLOCK_K
            k_tile_ptr += QK_BLOCK_K
        scores = tl.trans(scores_t)
        v_cache_offset = page_id * v_page_stride + page_offset * v_row_stride
        v0 = tl.load(v_base + v_cache_offset[:, None] + d_idx[None, :])
        (acc_scale, probabilities, row_max, row_sum) = _direct_online_softmax_stats(
            scores,
            row_max,
            row_sum,
            scale_softmax_log2=scale_softmax_log2,
            IS_BORDER=False,
        )
        probabilities = probabilities.to(v_ptr.type.element_ty)
        acc0 = acc0 * acc_scale[None, :]
        acc0 = tl.dot(tl.trans(v0), tl.trans(probabilities), acc0)
        acc1 = acc1 * acc_scale[None, :]
        v1 = tl.load(v_base + v_cache_offset[:, None] + (BLOCK_K + d_idx)[None, :])
        acc1 = tl.dot(tl.trans(v1), tl.trans(probabilities), acc1)
        pass
    o_mask = row_idx[:, None] < q_valid_len
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
        float("-inf"),
        row_max * scale_softmax + tl.log(row_sum),
    )
    inv_sum = tl.where((row_sum == 0) | (row_sum != row_sum), 1.0, 1.0 / row_sum)
    acc0_f32 = tl.trans(acc0) * inv_sum[:, None]
    acc1_f32 = tl.trans(acc1) * inv_sum[:, None]
    tl.store(o0_ptr, acc0_f32.to(o_ptr.type.element_ty), mask=o_mask)
    tl.store(o1_ptr, acc1_f32.to(o_ptr.type.element_ty), mask=o_mask)
    lse_mask = row_idx < q_valid_len
    tl.store(
        softmax_lse_ptr + query_head * output_total_q + output_q_bos + row_idx,
        lse,
        mask=lse_mask,
    )


def _launch_d256_tn_base(
    params,
    *,
    max_seqlen_q,
    max_seqlen_k,
    batch_size,
    num_heads,
    total_q,
    compact_worklist=None,
    block_m,
    block_n,
    grid_order,
    bulk_tail=False,
    tail_unaligned_only=False,
    fallback_for_fast=False,
):
    """Launch the fixed MetaX cpasync/s4 async-TN specialization."""
    if params.block_size not in (16, 32) or params.d != 256:
        raise ValueError("Async-TN requires paged D256 with page size 16 or 32")
    if params.cu_seqlens_q_ptr is None:
        raise ValueError("Async-TN requires varlen Q prefix sums")
    if params.cu_seqlens_k_ptr is None and params.seqused_k_ptr is None:
        raise ValueError("Async-TN requires KV lengths")
    if block_m != 32 or block_n != 128:
        raise ValueError("Async-TN kernel requires BM32 x BN128")
    use_compact_worklist = compact_worklist is not None
    fast_causal_aligned = (
        not fallback_for_fast
        and not use_compact_worklist
        and batch_size == 1
        and (total_q == max_seqlen_q)
        and (max_seqlen_k >= max_seqlen_q)
        and ((max_seqlen_k - max_seqlen_q) % block_m == 0)
    )
    row_tiles = (
        triton.cdiv(total_q * params.h_hk_ratio, block_m) + batch_size - 1
        if use_compact_worklist
        else triton.cdiv(max_seqlen_q * params.h_hk_ratio, block_m)
    )
    output_fragment_d = _D256_FRAGMENT
    head_programs = num_heads // params.h_hk_ratio
    grid = (row_tiles, 1 if use_compact_worklist else batch_size, head_programs)
    worklist_ptr = compact_worklist if use_compact_worklist else params.page_table_ptr
    cfg = {
        "PAGE_SIZE": params.block_size,
        "COMPACT_WORKLIST": use_compact_worklist,
        "BULK_TAIL": bulk_tail,
        "TAIL_UNALIGNED_ONLY": tail_unaligned_only,
        "SPLIT_KV": False,
        "NUM_SPLITS": 1,
        "Q_TILES": row_tiles,
        "GRID_ORDER": 0,
        "REVERSE_Q_TILES": max_seqlen_q <= 4108,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": output_fragment_d,
        "QK_BLOCK_K": _QK_FRAGMENT,
        "FAST_CAUSAL_ALIGNED": fast_causal_aligned,
        "FALLBACK_FOR_FAST": fallback_for_fast,
        "num_warps": 4,
        "num_stages": 4,
        "pipeline": "cpasync",
        "scenario": tn_compile_scenario(
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            params.o_ptr,
            params.softmax_lse_ptr,
            params.page_table_ptr,
            params.cu_seqlens_q_ptr,
            params.cu_seqlens_k_ptr,
            params.seqused_k_ptr,
        ),
        "pipeline_load_num": -1,
        "inner_stages": (0, 0),
    }
    logger.debug("Running D256 async-TN Direct kernel with config: %s", cfg)
    return flash_varlen_fwd_d256_tn_kernel[grid](
        params.q_ptr,
        params.k_ptr,
        params.v_ptr,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.q_row_stride,
        params.k_row_stride,
        params.v_row_stride,
        params.q_head_stride,
        params.k_head_stride,
        params.v_head_stride,
        params.o_row_stride,
        params.o_head_stride,
        params.scale_softmax,
        params.scale_softmax_log2,
        params.cu_seqlens_q_ptr,
        params.cu_seqlens_k_ptr,
        params.seqused_k_ptr,
        params.is_cu_seqlens_k,
        params.is_seqused_k,
        num_heads,
        params.h_hk_ratio,
        total_q,
        params.page_table_ptr,
        params.page_table_batch_stride,
        params.k_page_stride,
        params.v_ptr.stride(0),
        worklist_ptr,
        fast_q_len=max_seqlen_q if fallback_for_fast else 0,
        fast_k_len=max_seqlen_k if fallback_for_fast else 0,
        **cfg,
    )


def _d256_tn_bulk_policy(params, *, max_seqlen_q, max_seqlen_k, batch_size, total_q):
    """Share the exact bulk decision between the worklist adapter and launcher."""
    rectangular_tiles = triton.cdiv(max_seqlen_q * params.h_hk_ratio, 64) * batch_size
    compact_tiles = triton.cdiv(total_q * params.h_hk_ratio, 64) + batch_size - 1
    # Count only complete BM64 tiles after prefix/tail removal (GQA4: 16 Q rows).
    tp4_bulk_tiles = max(0, total_q // 16 - batch_size * 9)
    bulk_wave_floor = (
        728
        if (
            2048 < max_seqlen_k <= 8192
            and "noaddropt" not in tn_compile_scenario(params.k_ptr, params.v_ptr)
        )
        else 832
    )
    tp4_bulk_waves = (
        params.block_size == 16
        and params.h_hk_ratio == 4
        and (max_seqlen_k > 2048)
        and (tp4_bulk_tiles >= bulk_wave_floor)
    )
    sparse_bulk = rectangular_tiles > 4 * compact_tiles and (
        max_seqlen_q > 4108 or tp4_bulk_waves
    )
    supports_bulk = params.h_hk_ratio == 8 or (
        params.block_size == 16 and params.h_hk_ratio == 4
    )
    use_bulk = supports_bulk and not (
        rectangular_tiles > 4 * compact_tiles and (not sparse_bulk)
    )
    return use_bulk, sparse_bulk, compact_tiles


def launch_d256_tn_direct(
    params,
    *,
    max_seqlen_q,
    max_seqlen_k,
    batch_size,
    num_heads,
    total_q,
    compact_worklist=None,
    block_m,
    block_n,
    grid_order,
):
    """Peel Q into aligned BM64 bulk and disjoint Split-KV boundaries."""
    use_bulk, sparse_bulk, compact_tiles = _d256_tn_bulk_policy(
        params,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        batch_size=batch_size,
        total_q=total_q,
    )
    if not use_bulk:
        if max_seqlen_k >= 32768 and batch_size > 1:
            from .ragged import maybe_repack

            params = maybe_repack(
                params,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                batch_size=batch_size,
                total_q=total_q,
            )
        return _launch_d256_tn_base(
            params,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            batch_size=batch_size,
            num_heads=num_heads,
            total_q=total_q,
            compact_worklist=compact_worklist,
            block_m=32,
            block_n=block_n,
            grid_order=grid_order,
        )
    launch_bulk = launch_d256_tn_direct_bulk

    common = dict(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        batch_size=batch_size,
        num_heads=num_heads,
        total_q=total_q,
        block_n=block_n,
        grid_order=grid_order,
    )
    bulk_worklist = None
    if sparse_bulk:
        import torch

        from .ragged import build_compact_worklist_kernel

        bulk_worklist = torch.empty(
            (compact_tiles, 2), dtype=torch.int32, device=params.q_ptr.device
        )
        build_compact_worklist_kernel[(compact_tiles,)](
            params.cu_seqlens_q_ptr,
            bulk_worklist,
            batch_size,
            BLOCK_M=64 // params.h_hk_ratio,
            num_warps=1,
            num_stages=1,
        )
    bulk_params = params
    if max_seqlen_k >= 32768 and batch_size > 1:
        from .ragged import maybe_repack

        bulk_params = maybe_repack(
            params,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            batch_size=batch_size,
            total_q=total_q,
            bulk_only=True,
        )
    launch_bulk(
        bulk_params, block_m=64, compact_worklist=bulk_worklist, peel_q=True, **common
    )
    from .tn_boundary import launch_tail_split

    if params.block_size == 16 and params.h_hk_ratio == 4 and (max_seqlen_k <= 2048):
        from .tn_boundary import launch_boundary_merge

        boundary_args = dict(
            max_seqlen_q=max_seqlen_q,
            batch_size=batch_size,
            num_heads=num_heads,
            total_q=total_q,
            defer_merge=True,
        )
        tail_splits = (
            3
            if (
                triton.cdiv(max_seqlen_k, 128) == 9
                and "noaddropt" not in tn_compile_scenario(params.k_ptr, params.v_ptr)
            )
            else 4
        )
        tail = launch_tail_split(
            params,
            num_splits=tail_splits,
            peel_q=True,
            aligned_single_mask="noaddropt"
            not in tn_compile_scenario(params.k_ptr, params.v_ptr),
            **boundary_args,
        )
        prefix_single = "noaddropt" not in tn_compile_scenario(
            params.k_ptr, params.v_ptr
        )
        prefix = launch_tail_split(
            params,
            num_splits=max(2, triton.cdiv(max_seqlen_k, 128)) if prefix_single else 16,
            prefix_only=True,
            single_tile=prefix_single,
            **boundary_args,
        )
        return launch_boundary_merge(
            params,
            batch_size=batch_size,
            num_heads=num_heads,
            total_q=total_q,
            tail=tail,
            prefix=prefix,
        )
    launch_tail_split(
        bulk_params,
        max_seqlen_q=max_seqlen_q,
        batch_size=batch_size,
        num_heads=num_heads,
        total_q=total_q,
        num_splits=(
            8
            if params.block_size == 16
            and params.h_hk_ratio == 4
            and (max_seqlen_k >= 32768)
            else 4
        ),
        peel_q=True,
        aligned_single_mask=(
            params.block_size == 16
            and params.h_hk_ratio == 4
            and max_seqlen_k >= 32768
            and "noaddropt"
            not in tn_compile_scenario(bulk_params.k_ptr, bulk_params.v_ptr)
        ),
    )
    return launch_tail_split(
        params,
        max_seqlen_q=max_seqlen_q,
        batch_size=batch_size,
        num_heads=num_heads,
        total_q=total_q,
        num_splits=32 if max_seqlen_k >= 32768 else 16,
        prefix_only=True,
    )


@triton.jit
def _bulk_paged_tile_coords(
    start_n,
    k_len,
    page_table_ptr,
    BLOCK_N: tl.constexpr,
    BOUNDARY_CHECK: tl.constexpr,
    PAGE_SIZE: tl.constexpr = 32,
):
    """Resolve physical rows from the page table."""
    if PAGE_SIZE == 16:
        tl.static_assert(BLOCK_N == 128, "page16 scalar gather requires BN128")
        page_slot = tl.arange(0, BLOCK_N) // 16
        virtual_page = start_n // 16
        if BOUNDARY_CHECK:
            page0 = tl.load(
                page_table_ptr + virtual_page + 0, mask=start_n + 0 < k_len, other=0
            )
        else:
            page0 = tl.load(page_table_ptr + virtual_page + 0)
        if BOUNDARY_CHECK:
            page1 = tl.load(
                page_table_ptr + virtual_page + 1, mask=start_n + 16 < k_len, other=0
            )
        else:
            page1 = tl.load(page_table_ptr + virtual_page + 1)
        if BOUNDARY_CHECK:
            page2 = tl.load(
                page_table_ptr + virtual_page + 2, mask=start_n + 32 < k_len, other=0
            )
        else:
            page2 = tl.load(page_table_ptr + virtual_page + 2)
        if BOUNDARY_CHECK:
            page3 = tl.load(
                page_table_ptr + virtual_page + 3, mask=start_n + 48 < k_len, other=0
            )
        else:
            page3 = tl.load(page_table_ptr + virtual_page + 3)
        if BOUNDARY_CHECK:
            page4 = tl.load(
                page_table_ptr + virtual_page + 4, mask=start_n + 64 < k_len, other=0
            )
        else:
            page4 = tl.load(page_table_ptr + virtual_page + 4)
        if BOUNDARY_CHECK:
            page5 = tl.load(
                page_table_ptr + virtual_page + 5, mask=start_n + 80 < k_len, other=0
            )
        else:
            page5 = tl.load(page_table_ptr + virtual_page + 5)
        if BOUNDARY_CHECK:
            page6 = tl.load(
                page_table_ptr + virtual_page + 6, mask=start_n + 96 < k_len, other=0
            )
        else:
            page6 = tl.load(page_table_ptr + virtual_page + 6)
        if BOUNDARY_CHECK:
            page7 = tl.load(
                page_table_ptr + virtual_page + 7, mask=start_n + 112 < k_len, other=0
            )
        else:
            page7 = tl.load(page_table_ptr + virtual_page + 7)
        page_id = tl.where(
            page_slot < 4,
            tl.where(
                page_slot < 2,
                tl.where(page_slot == 0, page0, page1),
                tl.where(page_slot == 2, page2, page3),
            ),
            tl.where(
                page_slot < 6,
                tl.where(page_slot == 4, page4, page5),
                tl.where(page_slot == 6, page6, page7),
            ),
        ).to(tl.int64)
        col_idx = start_n + tl.arange(0, BLOCK_N)
        col_idx = tl.max_contiguous(tl.multiple_of(col_idx, BLOCK_N), BLOCK_N)
        return (col_idx, page_id, col_idx % 16)
    else:
        page_slot = tl.arange(0, BLOCK_N) // 32
        virtual_page = start_n // 32
        if BOUNDARY_CHECK:
            page0 = tl.load(
                page_table_ptr + virtual_page, mask=start_n < k_len, other=0
            ).to(tl.int64)
        else:
            page0 = tl.load(page_table_ptr + virtual_page).to(tl.int64)
        if BLOCK_N == 32:
            page_id = page0
        elif BLOCK_N == 64:
            if BOUNDARY_CHECK:
                page1 = tl.load(
                    page_table_ptr + virtual_page + 1,
                    mask=start_n + 32 < k_len,
                    other=0,
                ).to(tl.int64)
            else:
                page1 = tl.load(page_table_ptr + virtual_page + 1).to(tl.int64)
            page_id = tl.where(page_slot == 0, page0, page1)
        elif BLOCK_N == 128:
            if BOUNDARY_CHECK:
                page1 = tl.load(
                    page_table_ptr + virtual_page + 1,
                    mask=start_n + 32 < k_len,
                    other=0,
                ).to(tl.int64)
                page2 = tl.load(
                    page_table_ptr + virtual_page + 2,
                    mask=start_n + 64 < k_len,
                    other=0,
                ).to(tl.int64)
                page3 = tl.load(
                    page_table_ptr + virtual_page + 3,
                    mask=start_n + 96 < k_len,
                    other=0,
                ).to(tl.int64)
            else:
                page1 = tl.load(page_table_ptr + virtual_page + 1).to(tl.int64)
                page2 = tl.load(page_table_ptr + virtual_page + 2).to(tl.int64)
                page3 = tl.load(page_table_ptr + virtual_page + 3).to(tl.int64)
            page_id = tl.where(
                page_slot == 0,
                page0,
                tl.where(page_slot == 1, page1, tl.where(page_slot == 2, page2, page3)),
            )
        else:
            tl.static_assert(False, "Async-TN BLOCK_N must be 32, 64, or 128")
        col_idx = start_n + tl.arange(0, BLOCK_N)
        col_idx = tl.max_contiguous(tl.multiple_of(col_idx, BLOCK_N), BLOCK_N)
        page_offset = col_idx % 32
        return (col_idx, page_id, page_offset)


@triton.jit
def _bulk_online_softmax_stats(
    scores, row_max, row_sum, scale_softmax_log2, IS_BORDER: tl.constexpr
):
    previous_max = row_max
    row_max = tl.maximum(row_max, tl.max(scores, 1))
    if IS_BORDER:
        current_max = tl.where(row_max == float("-inf"), 0.0, row_max)
    else:
        current_max = row_max
    accumulator_scale = tl.math.exp2((previous_max - current_max) * scale_softmax_log2)
    row_sum *= accumulator_scale
    max_scaled = tl.where(row_max == float("-inf"), 0.0, row_max * scale_softmax_log2)
    probabilities = tl.math.exp2(scores * scale_softmax_log2 - max_scaled[:, None])
    row_sum += tl.sum(probabilities, 1)
    return (accumulator_scale, probabilities, row_max, row_sum)


@triton.jit
def _scaled_online_softmax_stats(
    scores, row_max, row_sum, scale_softmax_log2, IS_BORDER: tl.constexpr
):
    scores *= scale_softmax_log2
    previous_max = row_max
    row_max = tl.maximum(row_max, tl.max(scores, 1))
    if IS_BORDER:
        current_max = tl.where(row_max == float("-inf"), 0.0, row_max)
    else:
        current_max = row_max
    accumulator_scale = tl.math.exp2(previous_max - current_max)
    row_sum *= accumulator_scale
    max_scaled = tl.where(row_max == float("-inf"), 0.0, row_max)
    probabilities = tl.math.exp2(scores - max_scaled[:, None])
    row_sum += tl.sum(probabilities, 1)
    return (accumulator_scale, probabilities, row_max, row_sum)


@libentry()
@triton.jit(do_not_specialize=["total_q"])
def flash_varlen_fwd_d256_tn_bulk_kernel(
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
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    k_page_stride,
    v_page_stride,
    worklist_ptr,
    COMPACT_WORKLIST: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    Q_TILES: tl.constexpr,
    GRID_ORDER: tl.constexpr,
    REVERSE_Q_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QK_BLOCK_K: tl.constexpr,
    FAST_CAUSAL_ALIGNED: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    PEEL_Q: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 32,
    SHORT_SOFTMAX: tl.constexpr = False,
):
    tl.static_assert(
        BLOCK_K == 128 and QK_BLOCK_K == 64 and (BLOCK_N == 128),
        "D256 TN requires QK BK64, PV D128, BN128",
    )
    tl.static_assert(
        (not SPLIT_KV) | (not COMPACT_WORKLIST) & (NUM_SPLITS > 1),
        "Async-TN Split-KV requires a rectangular multi-split grid",
    )
    tl.static_assert(
        SPLIT_KV | (NUM_SPLITS == 1) & (Q_TILES > 0),
        "Async-TN Direct requires the neutral split configuration",
    )
    if GRID_ORDER == 1:
        head_pid = tl.program_id(0)
        rectangular_bid = tl.program_id(1)
        task_pid = tl.program_id(2)
    else:
        task_pid = tl.program_id(0)
        rectangular_bid = tl.program_id(1)
        head_pid = tl.program_id(2)
    if (not SPLIT_KV) & REVERSE_Q_TILES:
        if GRID_ORDER == 1:
            task_pid = tl.num_programs(2) - 1 - task_pid
        else:
            task_pid = tl.num_programs(0) - 1 - task_pid
    hid = head_pid
    if COMPACT_WORKLIST:
        descriptor_offset = task_pid * 2
        bid = tl.load(worklist_ptr + descriptor_offset).to(tl.int32)
        if bid < 0:
            return
        m_block = tl.load(worklist_ptr + descriptor_offset + 1).to(tl.int32)
        split_id = 0
    elif SPLIT_KV:
        bid = rectangular_bid
        split_id = task_pid // Q_TILES
        m_block = task_pid % Q_TILES
    else:
        bid = rectangular_bid
        m_block = task_pid
        split_id = 0
    q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
    q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
    q_len = q_eos - q_bos
    effective_q_len = q_len * HEAD_RATIO
    if m_block * BLOCK_M >= effective_q_len:
        return
    if IS_SEQUSED_K:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        tl.static_assert(IS_CU_SEQLENS_K, "Async-TN requires a KV length source")
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
        k_len = k_eos - k_bos
    if PEEL_Q:
        prefix_q = min(q_len, q_len - k_len & 64 // HEAD_RATIO - 1)
        q_bos += prefix_q
        q_len -= prefix_q
    packed_row = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_row = tl.max_contiguous(tl.multiple_of(packed_row, BLOCK_M), BLOCK_M)
    row_idx = packed_row // HEAD_RATIO
    query_head = hid * HEAD_RATIO + packed_row % HEAD_RATIO
    kv_head = hid
    q_block_end = tl.cdiv((m_block + 1) * BLOCK_M, HEAD_RATIO)
    if (q_block_end > q_len) | (
        q_block_end + k_len - q_len > k_len // BLOCK_N * BLOCK_N
    ):
        return
    if (k_len - q_len) % (BLOCK_M // HEAD_RATIO) != 0:
        return
    n_block_min = 0
    n_block_max = min(
        tl.cdiv(k_len, BLOCK_N), tl.cdiv(q_block_end + k_len - q_len, BLOCK_N)
    )
    if SPLIT_KV:
        span_blocks = max(n_block_max - n_block_min, 0)
        blocks_per_split = tl.cdiv(span_blocks, NUM_SPLITS)
        split_block_min = min(n_block_max, n_block_min + split_id * blocks_per_split)
        split_block_max = min(n_block_max, split_block_min + blocks_per_split)
        n_block_min = split_block_min
        n_block_max = split_block_max
    if FAST_CAUSAL_ALIGNED:
        n_masking_steps = min(n_block_max - n_block_min, 1)
    else:
        n_masking_steps = min(n_block_max - n_block_min, 1)
    if SPLIT_KV:
        o_ptr += split_id * total_q * NUM_HEADS * 256
        softmax_lse_ptr += split_id * NUM_HEADS * total_q
    q_base = q_ptr + q_bos * q_row_stride
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
    for _ in tl.range(0, n_masking_steps):
        start_n = n_block * BLOCK_N
        (col_idx, page_id, page_offset) = _bulk_paged_tile_coords(
            start_n,
            k_len,
            page_table_ptr,
            BLOCK_N=BLOCK_N,
            BOUNDARY_CHECK=True,
            PAGE_SIZE=PAGE_SIZE,
        )
        k_cache_offset = page_id * k_page_stride + page_offset * k_row_stride
        scores_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        q_tile_ptr = (
            q_base
            + row_idx[None, :] * q_row_stride
            + query_head[None, :] * q_head_stride
            + qk_d_idx[:, None]
        )
        k_tile_ptr = k_base + qk_d_idx[None, :] + k_cache_offset[:, None]
        for _d_block in tl.range(0, 256 // QK_BLOCK_K):
            if FAST_CAUSAL_ALIGNED:
                k_tile = tl.load(k_tile_ptr)
            else:
                k_tile = tl.load(k_tile_ptr)
            q_tile = tl.load(q_tile_ptr)
            scores_t = tl.dot(k_tile, q_tile, scores_t)
            q_tile_ptr += QK_BLOCK_K
            k_tile_ptr += QK_BLOCK_K
        scores = tl.trans(scores_t)
        scores = apply_mask(
            scores,
            col_idx,
            row_idx,
            q_len,
            k_len,
            0,
            0,
            is_even_mn=False,
            is_causal=True,
            is_local=False,
        )
        v_cache_offset = page_id * v_page_stride + page_offset * v_row_stride
        if FAST_CAUSAL_ALIGNED:
            v0 = tl.load(v_base + v_cache_offset[:, None] + d_idx[None, :])
        else:
            v0 = tl.load(v_base + v_cache_offset[:, None] + d_idx[None, :])
        if SHORT_SOFTMAX:
            scores *= scale_softmax_log2
            row_max = tl.max(scores, 1)
            max_scaled = tl.where(row_max == float("-inf"), 0.0, row_max)
            probabilities = tl.exp2(scores - max_scaled[:, None])
            row_sum = tl.sum(probabilities, 1)
            probabilities = probabilities.to(v_ptr.type.element_ty)
            acc0 = tl.dot(tl.trans(v0), tl.trans(probabilities))
            v1 = tl.load(v_base + v_cache_offset[:, None] + (BLOCK_K + d_idx)[None, :])
            acc1 = tl.dot(tl.trans(v1), tl.trans(probabilities))
        else:
            (acc_scale, probabilities, row_max, row_sum) = _bulk_online_softmax_stats(
                scores,
                row_max,
                row_sum,
                scale_softmax_log2=scale_softmax_log2,
                IS_BORDER=True,
            )
            probabilities = probabilities.to(v_ptr.type.element_ty)
            acc0 = acc0 * acc_scale[None, :]
            acc0 = tl.dot(tl.trans(v0), tl.trans(probabilities), acc0)
            acc1 = acc1 * acc_scale[None, :]
            if FAST_CAUSAL_ALIGNED:
                v1 = tl.load(
                    v_base + v_cache_offset[:, None] + (BLOCK_K + d_idx)[None, :]
                )
            else:
                v1 = tl.load(
                    v_base + v_cache_offset[:, None] + (BLOCK_K + d_idx)[None, :]
                )
            acc1 = tl.dot(tl.trans(v1), tl.trans(probabilities), acc1)
        pass
        n_block -= 1
    for n_block in tl.range(
        n_block_max - n_masking_steps - 1, n_block_min - 1, step=-1
    ):
        start_n = n_block * BLOCK_N
        (col_idx, page_id, page_offset) = _bulk_paged_tile_coords(
            start_n,
            k_len,
            page_table_ptr,
            BLOCK_N=BLOCK_N,
            BOUNDARY_CHECK=False,
            PAGE_SIZE=PAGE_SIZE,
        )
        k_cache_offset = page_id * k_page_stride + page_offset * k_row_stride
        scores_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        q_tile_ptr = (
            q_base
            + row_idx[None, :] * q_row_stride
            + query_head[None, :] * q_head_stride
            + qk_d_idx[:, None]
        )
        k_tile_ptr = k_base + qk_d_idx[None, :] + k_cache_offset[:, None]
        for _d_block in tl.range(0, 256 // QK_BLOCK_K):
            k_tile = tl.load(k_tile_ptr)
            q_tile = tl.load(q_tile_ptr)
            scores_t = tl.dot(k_tile, q_tile, scores_t)
            q_tile_ptr += QK_BLOCK_K
            k_tile_ptr += QK_BLOCK_K
        scores = tl.trans(scores_t)
        v_cache_offset = page_id * v_page_stride + page_offset * v_row_stride
        v0 = tl.load(v_base + v_cache_offset[:, None] + d_idx[None, :])
        if SHORT_SOFTMAX:
            (acc_scale, probabilities, row_max, row_sum) = _scaled_online_softmax_stats(
                scores,
                row_max,
                row_sum,
                scale_softmax_log2=scale_softmax_log2,
                IS_BORDER=False,
            )
        else:
            (acc_scale, probabilities, row_max, row_sum) = _bulk_online_softmax_stats(
                scores,
                row_max,
                row_sum,
                scale_softmax_log2=scale_softmax_log2,
                IS_BORDER=False,
            )
        probabilities = probabilities.to(v_ptr.type.element_ty)
        acc0 = acc0 * acc_scale[None, :]
        acc0 = tl.dot(tl.trans(v0), tl.trans(probabilities), acc0)
        acc1 = acc1 * acc_scale[None, :]
        v1 = tl.load(v_base + v_cache_offset[:, None] + (BLOCK_K + d_idx)[None, :])
        acc1 = tl.dot(tl.trans(v1), tl.trans(probabilities), acc1)
        pass
    o_mask = row_idx[:, None] < q_len
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
        float("-inf"),
        row_max
        * (scale_softmax / scale_softmax_log2 if SHORT_SOFTMAX else scale_softmax)
        + tl.log(row_sum),
    )
    inv_sum = tl.where((row_sum == 0) | (row_sum != row_sum), 1.0, 1.0 / row_sum)
    acc0_f32 = tl.trans(acc0) * inv_sum[:, None]
    acc1_f32 = tl.trans(acc1) * inv_sum[:, None]
    tl.store(o0_ptr, acc0_f32.to(o_ptr.type.element_ty), mask=o_mask)
    tl.store(o1_ptr, acc1_f32.to(o_ptr.type.element_ty), mask=o_mask)
    lse_mask = row_idx < q_len
    tl.store(
        softmax_lse_ptr + query_head * total_q + q_bos + row_idx, lse, mask=lse_mask
    )


def launch_d256_tn_direct_bulk(
    params,
    *,
    max_seqlen_q,
    max_seqlen_k,
    batch_size,
    num_heads,
    total_q,
    compact_worklist=None,
    block_m,
    block_n,
    grid_order,
    peel_q=False,
):
    """Launch the fixed MetaX cpasync/s4 async-TN specialization."""
    if params.block_size not in (16, 32) or params.d != 256:
        raise ValueError("Async-TN requires paged D256 with page size 16 or 32")
    if params.cu_seqlens_q_ptr is None:
        raise ValueError("Async-TN requires varlen Q prefix sums")
    if params.cu_seqlens_k_ptr is None and params.seqused_k_ptr is None:
        raise ValueError("Async-TN requires KV lengths")
    if block_m != 64 or block_n != 128:
        raise ValueError("Async-TN bulk kernel requires BM64 x BN128")
    use_compact_worklist = compact_worklist is not None
    fast_causal_aligned = (
        not use_compact_worklist
        and batch_size == 1
        and (total_q == max_seqlen_q)
        and (max_seqlen_k >= max_seqlen_q)
        and ((max_seqlen_k - max_seqlen_q) % block_m == 0)
    )
    row_tiles = (
        triton.cdiv(total_q * params.h_hk_ratio, block_m) + batch_size - 1
        if use_compact_worklist
        else triton.cdiv(max_seqlen_q * params.h_hk_ratio, block_m)
    )
    output_fragment_d = _D256_FRAGMENT
    head_programs = num_heads // params.h_hk_ratio
    batch_first = (
        not use_compact_worklist
        and params.block_size == 16
        and params.h_hk_ratio == 4
        and max_seqlen_k <= 2048
        and total_q >= 8192
        and row_tiles * batch_size
        <= 2 * (triton.cdiv(total_q * params.h_hk_ratio, block_m) + batch_size - 1)
        and "noaddropt" not in tn_compile_scenario(params.k_ptr, params.v_ptr)
    )
    grid = (
        (head_programs, batch_size, row_tiles)
        if batch_first
        else (row_tiles, 1 if use_compact_worklist else batch_size, head_programs)
    )
    worklist_ptr = compact_worklist if use_compact_worklist else params.page_table_ptr
    cfg = {
        "PAGE_SIZE": params.block_size,
        "SHORT_SOFTMAX": (
            params.block_size == 16
            and params.h_hk_ratio == 4
            and max_seqlen_k <= 2048
            and "noaddropt" not in tn_compile_scenario(params.k_ptr, params.v_ptr)
        ),
        "COMPACT_WORKLIST": use_compact_worklist,
        "PEEL_Q": peel_q,
        "SPLIT_KV": False,
        "NUM_SPLITS": 1,
        "Q_TILES": row_tiles,
        "GRID_ORDER": 1 if batch_first else 0,
        "REVERSE_Q_TILES": max_seqlen_q <= 4108,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": output_fragment_d,
        "QK_BLOCK_K": _QK_FRAGMENT,
        "FAST_CAUSAL_ALIGNED": fast_causal_aligned,
        "num_warps": 4,
        "num_stages": 4,
        "pipeline": "cpasync",
        "scenario": tn_compile_scenario(
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            params.o_ptr,
            params.softmax_lse_ptr,
            params.page_table_ptr,
            params.cu_seqlens_q_ptr,
            params.cu_seqlens_k_ptr,
            params.seqused_k_ptr,
        ),
        "pipeline_load_num": -1,
        "inner_stages": (0, 0),
    }
    logger.debug("Running D256 async-TN Direct kernel with config: %s", cfg)
    return flash_varlen_fwd_d256_tn_bulk_kernel[grid](
        params.q_ptr,
        params.k_ptr,
        params.v_ptr,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.q_row_stride,
        params.k_row_stride,
        params.v_row_stride,
        params.q_head_stride,
        params.k_head_stride,
        params.v_head_stride,
        params.o_row_stride,
        params.o_head_stride,
        params.scale_softmax,
        params.scale_softmax_log2,
        params.cu_seqlens_q_ptr,
        params.cu_seqlens_k_ptr,
        params.seqused_k_ptr,
        params.is_cu_seqlens_k,
        params.is_seqused_k,
        num_heads,
        params.h_hk_ratio,
        total_q,
        params.page_table_ptr,
        params.page_table_batch_stride,
        params.k_page_stride,
        params.v_ptr.stride(0),
        worklist_ptr,
        **cfg,
    )


@triton.jit
def _fast_paged_tile_coords(
    start_n,
    k_len,
    page_table_ptr,
    BLOCK_N: tl.constexpr,
    BOUNDARY_CHECK: tl.constexpr,
    PAGE_SIZE: tl.constexpr = 32,
):
    """Resolve physical rows from the page table."""
    if PAGE_SIZE == 16:
        tl.static_assert(BLOCK_N == 128, "page16 scalar gather requires BN128")
        page_slot = tl.arange(0, BLOCK_N) // 16
        virtual_page = start_n // 16
        if BOUNDARY_CHECK:
            page0 = tl.load(
                page_table_ptr + virtual_page + 0, mask=start_n + 0 < k_len, other=0
            )
        else:
            page0 = tl.load(page_table_ptr + virtual_page + 0)
        if BOUNDARY_CHECK:
            page1 = tl.load(
                page_table_ptr + virtual_page + 1, mask=start_n + 16 < k_len, other=0
            )
        else:
            page1 = tl.load(page_table_ptr + virtual_page + 1)
        if BOUNDARY_CHECK:
            page2 = tl.load(
                page_table_ptr + virtual_page + 2, mask=start_n + 32 < k_len, other=0
            )
        else:
            page2 = tl.load(page_table_ptr + virtual_page + 2)
        if BOUNDARY_CHECK:
            page3 = tl.load(
                page_table_ptr + virtual_page + 3, mask=start_n + 48 < k_len, other=0
            )
        else:
            page3 = tl.load(page_table_ptr + virtual_page + 3)
        if BOUNDARY_CHECK:
            page4 = tl.load(
                page_table_ptr + virtual_page + 4, mask=start_n + 64 < k_len, other=0
            )
        else:
            page4 = tl.load(page_table_ptr + virtual_page + 4)
        if BOUNDARY_CHECK:
            page5 = tl.load(
                page_table_ptr + virtual_page + 5, mask=start_n + 80 < k_len, other=0
            )
        else:
            page5 = tl.load(page_table_ptr + virtual_page + 5)
        if BOUNDARY_CHECK:
            page6 = tl.load(
                page_table_ptr + virtual_page + 6, mask=start_n + 96 < k_len, other=0
            )
        else:
            page6 = tl.load(page_table_ptr + virtual_page + 6)
        if BOUNDARY_CHECK:
            page7 = tl.load(
                page_table_ptr + virtual_page + 7, mask=start_n + 112 < k_len, other=0
            )
        else:
            page7 = tl.load(page_table_ptr + virtual_page + 7)
        page_id = tl.where(
            page_slot < 4,
            tl.where(
                page_slot < 2,
                tl.where(page_slot == 0, page0, page1),
                tl.where(page_slot == 2, page2, page3),
            ),
            tl.where(
                page_slot < 6,
                tl.where(page_slot == 4, page4, page5),
                tl.where(page_slot == 6, page6, page7),
            ),
        ).to(tl.int64)
        col_idx = start_n + tl.arange(0, BLOCK_N)
        col_idx = tl.max_contiguous(tl.multiple_of(col_idx, BLOCK_N), BLOCK_N)
        return (col_idx, page_id, col_idx % 16)
    else:
        page_slot = tl.arange(0, BLOCK_N) // 32
        virtual_page = start_n // 32
        if BOUNDARY_CHECK:
            page0 = tl.load(
                page_table_ptr + virtual_page, mask=start_n < k_len, other=0
            ).to(tl.int64)
        else:
            page0 = tl.load(page_table_ptr + virtual_page).to(tl.int64)
        if BLOCK_N == 32:
            page_id = page0
        elif BLOCK_N == 64:
            if BOUNDARY_CHECK:
                page1 = tl.load(
                    page_table_ptr + virtual_page + 1,
                    mask=start_n + 32 < k_len,
                    other=0,
                ).to(tl.int64)
            else:
                page1 = tl.load(page_table_ptr + virtual_page + 1).to(tl.int64)
            page_id = tl.where(page_slot == 0, page0, page1)
        elif BLOCK_N == 128:
            if BOUNDARY_CHECK:
                page1 = tl.load(
                    page_table_ptr + virtual_page + 1,
                    mask=start_n + 32 < k_len,
                    other=0,
                ).to(tl.int64)
                page2 = tl.load(
                    page_table_ptr + virtual_page + 2,
                    mask=start_n + 64 < k_len,
                    other=0,
                ).to(tl.int64)
                page3 = tl.load(
                    page_table_ptr + virtual_page + 3,
                    mask=start_n + 96 < k_len,
                    other=0,
                ).to(tl.int64)
            else:
                page1 = tl.load(page_table_ptr + virtual_page + 1).to(tl.int64)
                page2 = tl.load(page_table_ptr + virtual_page + 2).to(tl.int64)
                page3 = tl.load(page_table_ptr + virtual_page + 3).to(tl.int64)
            page_id = tl.where(
                page_slot == 0,
                page0,
                tl.where(page_slot == 1, page1, tl.where(page_slot == 2, page2, page3)),
            )
        else:
            tl.static_assert(False, "Async-TN BLOCK_N must be 32, 64, or 128")
        col_idx = start_n + tl.arange(0, BLOCK_N)
        col_idx = tl.max_contiguous(tl.multiple_of(col_idx, BLOCK_N), BLOCK_N)
        page_offset = col_idx % 32
        return (col_idx, page_id, page_offset)


@triton.jit
def _fast_online_softmax_stats(
    scores, row_max, row_sum, scale_softmax_log2, IS_BORDER: tl.constexpr
):
    scores *= scale_softmax_log2
    previous_max = row_max
    row_max = tl.maximum(row_max, tl.max(scores, 1))
    if IS_BORDER:
        current_max = tl.where(row_max == float("-inf"), 0.0, row_max)
    else:
        current_max = row_max
    accumulator_scale = tl.math.exp2(previous_max - current_max)
    row_sum *= accumulator_scale
    max_scaled = tl.where(row_max == float("-inf"), 0.0, row_max)
    probabilities = tl.math.exp2(scores - max_scaled[:, None])
    row_sum += tl.sum(probabilities, 1)
    return (accumulator_scale, probabilities, row_max, row_sum)


@libentry()
@triton.jit(do_not_specialize=["total_q", "host_q_len", "host_k_len"])
def flash_varlen_fwd_d256_tn_fast_kernel(
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
    PAGE_SIZE: tl.constexpr = 32,
    MASK_K_TAIL: tl.constexpr = True,
):
    tl.static_assert(
        BLOCK_K == 128 and QK_BLOCK_K == 64 and (BLOCK_N == 128),
        "D256 TN requires QK BK64, PV D128, BN128",
    )
    tl.static_assert(not COMPACT_WORKLIST, "Fast attention requires a rectangular grid")
    if GRID_ORDER == 1:
        head_pid = tl.program_id(0)
        rectangular_bid = tl.program_id(1)
        task_pid = tl.program_id(2)
    else:
        task_pid = tl.program_id(0)
        rectangular_bid = tl.program_id(1)
        head_pid = tl.program_id(2)
    if REVERSE_Q_TILES:
        if GRID_ORDER == 1:
            task_pid = tl.num_programs(2) - 1 - task_pid
        else:
            task_pid = tl.num_programs(0) - 1 - task_pid
    hid = head_pid
    if COMPACT_WORKLIST:
        descriptor_offset = task_pid * 2
        bid = tl.load(worklist_ptr + descriptor_offset).to(tl.int32)
        if bid < 0:
            return
        m_block = tl.load(worklist_ptr + descriptor_offset + 1).to(tl.int32)
    else:
        bid = rectangular_bid
        m_block = task_pid
    actual_q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
    actual_q_len = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32) - actual_q_bos
    if IS_SEQUSED_K:
        actual_k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        actual_k_len = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32) - tl.load(
            cu_seqlens_k_ptr + bid
        ).to(tl.int32)
    # Host maxima may be graph bounds. Generic handles all other actual lengths.
    if actual_q_bos != 0 or actual_q_len != host_q_len or actual_k_len != host_k_len:
        return
    q_bos = 0
    q_len = host_q_len
    effective_q_len = q_len * PACK_GROUP
    if m_block * BLOCK_M >= effective_q_len:
        return
    k_len = host_k_len
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
    n_masking_steps = min(n_block_max - n_block_min, 1)
    q_base = q_ptr + q_bos * q_row_stride
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
    for _ in tl.range(0, n_masking_steps):
        start_n = n_block * BLOCK_N
        (col_idx, page_id, page_offset) = _fast_paged_tile_coords(
            start_n,
            k_len,
            page_table_ptr,
            BLOCK_N=BLOCK_N,
            BOUNDARY_CHECK=True,
            PAGE_SIZE=PAGE_SIZE,
        )
        k_cache_offset = page_id * k_page_stride + page_offset * k_row_stride
        scores_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        q_tile_ptr = (
            q_base
            + row_idx[None, :] * q_row_stride
            + query_head[None, :] * q_head_stride
            + qk_d_idx[:, None]
        )
        k_tile_ptr = k_base + qk_d_idx[None, :] + k_cache_offset[:, None]
        for _d_block in tl.range(0, 256 // QK_BLOCK_K):
            k_tile = tl.load(k_tile_ptr)
            q_tile = tl.load(q_tile_ptr, mask=row_idx[None, :] < q_len, other=0.0)
            scores_t = tl.dot(k_tile, q_tile, scores_t)
            q_tile_ptr += QK_BLOCK_K
            k_tile_ptr += QK_BLOCK_K
        scores = tl.trans(scores_t)
        scores = apply_mask(
            scores,
            col_idx,
            row_idx,
            q_len,
            k_len,
            0,
            0,
            is_even_mn=False,
            is_causal=True,
            is_local=False,
        )
        v_cache_offset = page_id * v_page_stride + page_offset * v_row_stride
        if PAGE_SIZE == 16 or MASK_K_TAIL:
            v0 = tl.load(
                v_base + v_cache_offset[:, None] + d_idx[None, :],
                mask=col_idx[:, None] < k_len,
                other=0.0,
            )
        else:
            v0 = tl.load(v_base + v_cache_offset[:, None] + d_idx[None, :])
        (acc_scale, probabilities, row_max, row_sum) = _fast_online_softmax_stats(
            scores,
            row_max,
            row_sum,
            scale_softmax_log2=scale_softmax_log2,
            IS_BORDER=True,
        )
        probabilities = probabilities.to(v_ptr.type.element_ty)
        acc0 = acc0 * acc_scale[None, :]
        acc0 = tl.dot(tl.trans(v0), tl.trans(probabilities), acc0)
        acc1 = acc1 * acc_scale[None, :]
        if PAGE_SIZE == 16 or MASK_K_TAIL:
            v1 = tl.load(
                v_base + v_cache_offset[:, None] + (BLOCK_K + d_idx)[None, :],
                mask=col_idx[:, None] < k_len,
                other=0.0,
            )
        else:
            v1 = tl.load(v_base + v_cache_offset[:, None] + (BLOCK_K + d_idx)[None, :])
        acc1 = tl.dot(tl.trans(v1), tl.trans(probabilities), acc1)
        pass
        n_block -= 1
    main_start = n_block_max - n_masking_steps - 1
    for n_block in tl.range(main_start, n_block_min - 1, step=-1):
        start_n = n_block * BLOCK_N
        (col_idx, page_id, page_offset) = _fast_paged_tile_coords(
            start_n,
            k_len,
            page_table_ptr,
            BLOCK_N=BLOCK_N,
            BOUNDARY_CHECK=False,
            PAGE_SIZE=PAGE_SIZE,
        )
        k_cache_offset = page_id * k_page_stride + page_offset * k_row_stride
        scores_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        q_tile_ptr = (
            q_base
            + row_idx[None, :] * q_row_stride
            + query_head[None, :] * q_head_stride
            + qk_d_idx[:, None]
        )
        k_tile_ptr = k_base + qk_d_idx[None, :] + k_cache_offset[:, None]
        for _d_block in tl.range(0, 256 // QK_BLOCK_K):
            k_tile = tl.load(k_tile_ptr)
            q_tile = tl.load(q_tile_ptr, mask=row_idx[None, :] < q_len, other=0.0)
            scores_t = tl.dot(k_tile, q_tile, scores_t)
            q_tile_ptr += QK_BLOCK_K
            k_tile_ptr += QK_BLOCK_K
        scores = tl.trans(scores_t)
        v_cache_offset = page_id * v_page_stride + page_offset * v_row_stride
        v0 = tl.load(v_base + v_cache_offset[:, None] + d_idx[None, :])
        (acc_scale, probabilities, row_max, row_sum) = _fast_online_softmax_stats(
            scores,
            row_max,
            row_sum,
            scale_softmax_log2=scale_softmax_log2,
            IS_BORDER=False,
        )
        probabilities = probabilities.to(v_ptr.type.element_ty)
        acc0 = acc0 * acc_scale[None, :]
        acc0 = tl.dot(tl.trans(v0), tl.trans(probabilities), acc0)
        acc1 = acc1 * acc_scale[None, :]
        v1 = tl.load(v_base + v_cache_offset[:, None] + (BLOCK_K + d_idx)[None, :])
        acc1 = tl.dot(tl.trans(v1), tl.trans(probabilities), acc1)
        pass
    o_mask = row_idx[:, None] < q_len
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
        float("-inf"),
        row_max * (scale_softmax / scale_softmax_log2) + tl.log(row_sum),
    )
    inv_sum = tl.where((row_sum == 0) | (row_sum != row_sum), 1.0, 1.0 / row_sum)
    acc0_f32 = tl.trans(acc0) * inv_sum[:, None]
    acc1_f32 = tl.trans(acc1) * inv_sum[:, None]
    tl.store(o0_ptr, acc0_f32.to(o_ptr.type.element_ty), mask=o_mask)
    tl.store(o1_ptr, acc1_f32.to(o_ptr.type.element_ty), mask=o_mask)
    lse_mask = row_idx < q_len
    tl.store(
        softmax_lse_ptr + query_head * total_q + q_bos + row_idx, lse, mask=lse_mask
    )


def launch_d256_tn_direct_fast(
    params,
    *,
    max_seqlen_q,
    max_seqlen_k,
    batch_size,
    num_heads,
    total_q,
    compact_worklist=None,
    block_m,
    block_n,
    grid_order,
):
    """Launch the fixed MetaX cpasync/s4 async-TN specialization."""
    if params.block_size not in (16, 32) or params.d != 256:
        raise ValueError("Async-TN requires paged D256 with page size 16 or 32")
    if params.cu_seqlens_q_ptr is None:
        raise ValueError("Async-TN requires varlen Q prefix sums")
    if params.cu_seqlens_k_ptr is None and params.seqused_k_ptr is None:
        raise ValueError("Async-TN requires KV lengths")
    if batch_size != 1 or total_q != max_seqlen_q:
        raise ValueError("Async-TN fast path requires one dense varlen request")
    long_fast = max_seqlen_q > 4108 and (
        params.h_hk_ratio == 8 or (params.block_size == 16 and params.h_hk_ratio == 4)
    )
    if block_m == 32 and long_fast:
        block_m = 64
    if block_m not in (32, 64) or block_n != 128:
        raise ValueError("Async-TN fast kernel requires BM32/BM64 x BN128")
    pack_group = 4 if block_m == 64 and params.h_hk_ratio == 8 else params.h_hk_ratio
    use_compact_worklist = compact_worklist is not None
    output_fragment_d = _D256_FRAGMENT
    head_programs = num_heads // pack_group
    row_tiles = triton.cdiv(max_seqlen_q * pack_group, block_m)
    grid = (row_tiles, 1 if use_compact_worklist else batch_size, head_programs)
    worklist_ptr = compact_worklist if use_compact_worklist else params.page_table_ptr
    cfg = {
        "PAGE_SIZE": params.block_size,
        # The device guard proves that host K equals the active K length.
        "MASK_K_TAIL": max_seqlen_k % block_n != 0,
        "COMPACT_WORKLIST": use_compact_worklist,
        "PACK_GROUP": pack_group,
        "GRID_ORDER": 0,
        "REVERSE_Q_TILES": max_seqlen_q <= 4108 or (block_m == 64 and long_fast),
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": output_fragment_d,
        "QK_BLOCK_K": _QK_FRAGMENT,
        "num_warps": 4,
        "num_stages": 4,
        "pipeline": "cpasync",
        "scenario": tn_compile_scenario(
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            params.o_ptr,
            params.softmax_lse_ptr,
            params.page_table_ptr,
            params.cu_seqlens_q_ptr,
            params.cu_seqlens_k_ptr,
            params.seqused_k_ptr,
        ),
        "pipeline_load_num": -1,
        "inner_stages": (0, 0),
    }
    logger.debug("Running D256 async-TN Direct kernel with config: %s", cfg)
    fast_result = flash_varlen_fwd_d256_tn_fast_kernel[grid](
        params.q_ptr,
        params.k_ptr,
        params.v_ptr,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.q_row_stride,
        params.k_row_stride,
        params.v_row_stride,
        params.q_head_stride,
        params.k_head_stride,
        params.v_head_stride,
        params.o_row_stride,
        params.o_head_stride,
        params.scale_softmax,
        params.scale_softmax_log2,
        params.cu_seqlens_q_ptr,
        params.cu_seqlens_k_ptr,
        params.seqused_k_ptr,
        params.is_cu_seqlens_k,
        params.is_seqused_k,
        num_heads,
        params.h_hk_ratio,
        total_q,
        max_seqlen_q,
        max_seqlen_k,
        params.page_table_ptr,
        params.page_table_batch_stride,
        params.k_page_stride,
        params.v_ptr.stride(0),
        worklist_ptr,
        **cfg,
    )

    _launch_d256_tn_base(
        params,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        batch_size=batch_size,
        num_heads=num_heads,
        total_q=total_q,
        block_m=32,
        block_n=128,
        grid_order=grid_order,
        fallback_for_fast=True,
    )
    return fast_result
