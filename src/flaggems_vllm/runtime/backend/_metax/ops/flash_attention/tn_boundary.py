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

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._metax.ops.flash_attention.common import (
    apply_mask,
    tn_compile_scenario,
)
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.tn_direct import (
    flash_varlen_fwd_d256_tn_kernel,
)
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


_D256_FRAGMENT = 128


_QK_FRAGMENT = 64


@triton.jit
def _prefix_paged_tile_coords(
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
def _prefix_online_softmax_stats(
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
def flash_varlen_fwd_d256_tn_prefix_kernel(
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
        span_blocks = max(n_block_max - n_block_min, 0)  # noqa: F841
        blocks_per_split = 1
        split_block_min = min(n_block_max, n_block_min + split_id * blocks_per_split)
        split_block_max = min(n_block_max, split_block_min + blocks_per_split)
        n_block_min = split_block_min
        n_block_max = split_block_max
    if FAST_CAUSAL_ALIGNED:
        n_masking_steps = min(n_block_max - n_block_min, 1)  # noqa: F841
    else:
        n_masking_steps = min(  # noqa: F841
            n_block_max - n_block_min, tl.cdiv(BLOCK_M, BLOCK_N) + 1
        )  # noqa: F841
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
    tl.static_assert(
        PREFIX_ONLY
        and SPLIT_KV
        and (NUM_SPLITS >= 2)
        and (NUM_SPLITS <= 16)
        and PAGE_SIZE == 16
    )
    if n_block_min < n_block_max:
        start_n = n_block * BLOCK_N
        (col_idx, page_id, page_offset) = _prefix_paged_tile_coords(
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
        row_max = tl.max(scores, 1)
        max_scaled = tl.where(
            row_max == float("-inf"), 0.0, row_max * scale_softmax_log2
        )
        probabilities = tl.exp2(scores * scale_softmax_log2 - max_scaled[:, None])
        row_sum = tl.sum(probabilities, 1)
        probabilities = probabilities.to(v_ptr.type.element_ty)
        acc0 = tl.dot(tl.trans(v0), tl.trans(probabilities))
        if FAST_CAUSAL_ALIGNED:
            v1 = tl.load(v_base + v_cache_offset[:, None] + (BLOCK_K + d_idx)[None, :])
        else:
            v1 = tl.load(
                v_base + v_cache_offset[:, None] + (BLOCK_K + d_idx)[None, :],
                mask=col_idx[:, None] < k_len,
                other=0.0,
            )
        acc1 = tl.dot(tl.trans(v1), tl.trans(probabilities))
        pass
        n_block -= 1
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


@triton.jit
def _aligned_paged_tile_coords(
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
def _aligned_online_softmax_stats(
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
def flash_varlen_fwd_d256_tn_aligned_kernel(
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
        tl.static_assert(BULK_TAIL and PEEL_Q and SPLIT_KV and HEAD_RATIO == 4)
        # shift is a multiple of16, packed BM32 covers8 query rows: only the
        # final N128 tile can intersect a valid query's causal boundary.
        n_masking_steps = min(n_block_max - n_block_min, 1)
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
        (col_idx, page_id, page_offset) = _aligned_paged_tile_coords(
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
        (acc_scale, probabilities, row_max, row_sum) = _aligned_online_softmax_stats(
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
        (col_idx, page_id, page_offset) = _aligned_paged_tile_coords(
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
        (acc_scale, probabilities, row_max, row_sum) = _aligned_online_softmax_stats(
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


def launch_tail_split(
    params,
    *,
    max_seqlen_q,
    batch_size,
    num_heads,
    total_q,
    num_splits,
    peel_q=False,
    prefix_only=False,
    defer_merge=False,
    single_tile=False,
    aligned_single_mask=False,
):
    """Split bounded prefix/tail rows per request and merge FP32 partials."""
    if prefix_only and peel_q:
        raise ValueError("Prefix Split-KV must retain the original Q base and length")
    if params.q_ptr.dtype != torch.bfloat16:
        raise ValueError("Async-TN tail Split-KV requires BF16 Q/K/V")
    if not params.is_paged or params.block_size not in (16, 32) or params.d != 256:
        raise ValueError(
            "Async-TN tail Split-KV requires paged D256 with page size 16 or 32"
        )
    if not params.is_causal or params.is_local:
        raise ValueError("Async-TN tail Split-KV only supports causal global attention")
    if params.is_dropout or params.is_alibi or params.is_softcap:
        raise ValueError(
            "Async-TN tail Split-KV does not support optional attention features"
        )
    if params.cu_seqlens_q_ptr is None:
        raise ValueError("Async-TN tail Split-KV requires varlen Q prefix sums")
    if params.cu_seqlens_k_ptr is None and params.seqused_k_ptr is None:
        raise ValueError("Async-TN tail Split-KV requires KV lengths")
    if not 2 <= num_splits <= 32:
        raise ValueError("Async-TN tail Split-KV requires 2 to 32 splits")
    if num_heads % params.h_hk_ratio != 0:
        raise ValueError("Async-TN tail Split-KV received an invalid GQA ratio")
    if params.h_hk_ratio != 8 and (
        not (params.block_size == 16 and params.h_hk_ratio == 4)
    ):
        raise ValueError("TN tail Split-KV requires GQA8 or page16/GQA4")
    block_m = 32
    block_n = 128
    block_k = 128
    rows_per_request = 64 // params.h_hk_ratio if prefix_only else 128
    q_tiles = rows_per_request * params.h_hk_ratio // block_m
    tail_rows = batch_size * rows_per_request
    head_programs = num_heads // params.h_hk_ratio
    original_out = params.o_ptr
    original_lse = params.softmax_lse_ptr
    out_splits = torch.empty(
        (num_splits, tail_rows, num_heads, 256),
        dtype=torch.float32,
        device=original_out.device,
    )
    lse_splits = torch.empty(
        (num_splits, num_heads, tail_rows),
        dtype=torch.float32,
        device=original_out.device,
    )
    grid = (head_programs, batch_size, num_splits * q_tiles)
    cfg = {
        "PAGE_SIZE": params.block_size,
        "COMPACT_WORKLIST": False,
        "BULK_TAIL": not prefix_only,
        "PREFIX_ONLY": prefix_only,
        "PEEL_Q": peel_q,
        "TAIL_ROWS": tail_rows,
        "SPLIT_KV": True,
        "NUM_SPLITS": num_splits,
        "Q_TILES": q_tiles,
        "GRID_ORDER": 1,
        "REVERSE_Q_TILES": False,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "QK_BLOCK_K": 64,
        "FAST_CAUSAL_ALIGNED": False,
        "num_warps": 4,
        "num_stages": 4,
        "pipeline": "cpasync",
        "scenario": tn_compile_scenario(
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            out_splits,
            lse_splits,
            params.page_table_ptr,
            params.cu_seqlens_q_ptr,
            params.cu_seqlens_k_ptr,
            params.seqused_k_ptr,
        ),
        "pipeline_load_num": -1,
        "inner_stages": (0, 0),
    }
    logger.debug("Running D256 async-TN Split-KV with config: %s", cfg)
    kernel = flash_varlen_fwd_d256_tn_kernel
    if single_tile:
        assert prefix_only and 2 <= num_splits <= 16 and params.block_size == 16
        prefix_kernel = flash_varlen_fwd_d256_tn_prefix_kernel

        kernel = prefix_kernel
    if aligned_single_mask:
        assert not prefix_only and peel_q and params.h_hk_ratio == 4
        aligned_kernel = flash_varlen_fwd_d256_tn_aligned_kernel

        kernel = aligned_kernel
    partial = kernel[grid](
        params.q_ptr,
        params.k_ptr,
        params.v_ptr,
        out_splits,
        lse_splits,
        params.q_row_stride,
        params.k_row_stride,
        params.v_row_stride,
        params.q_head_stride,
        params.k_head_stride,
        params.v_head_stride,
        out_splits.stride(1),
        out_splits.stride(2),
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
        params.page_table_ptr,
        **cfg,
    )
    if defer_merge:
        return (partial, out_splits, lse_splits)
    merge = tail_split_merge_kernel[rows_per_request, num_heads, batch_size]
    merge(
        original_out,
        original_lse,
        out_splits,
        lse_splits,
        params.o_row_stride,
        params.o_head_stride,
        total_q,
        num_heads,
        256,
        num_splits,
        triton.next_power_of_2(num_splits),
        256,
        params.cu_seqlens_q_ptr,
        params.cu_seqlens_k_ptr,
        params.seqused_k_ptr,
        params.is_seqused_k,
        tail_rows,
        peel_q,
        prefix_only,
        64 // params.h_hk_ratio,
        num_warps=1,
        num_stages=1,
    )
    return partial


@libentry()
@triton.jit
def tail_split_merge_kernel(
    out_ptr,
    lse_ptr,
    out_splits_ptr,
    lse_splits_ptr,
    o_row_stride,
    o_head_stride,
    total_q: tl.constexpr,
    h: tl.constexpr,
    d: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    MAX_N_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    cu_q_ptr,
    cu_k_ptr,
    used_k_ptr,
    IS_SEQUSED_K: tl.constexpr,
    TAIL_ROWS: tl.constexpr,
    PEEL_Q: tl.constexpr,
    PREFIX_ONLY: tl.constexpr,
    Q_ALIGNMENT: tl.constexpr = 8,
):
    tail_idx = tl.program_id(0)
    bid = tl.program_id(2)
    q_bos = tl.load(cu_q_ptr + bid)
    q_len = tl.load(cu_q_ptr + bid + 1) - q_bos
    if IS_SEQUSED_K:
        k_len = tl.load(used_k_ptr + bid)
    else:
        k_len = tl.load(cu_k_ptr + bid + 1) - tl.load(cu_k_ptr + bid)
    if PEEL_Q:
        prefix_q = min(q_len, q_len - k_len & Q_ALIGNMENT - 1)
        q_bos += prefix_q
        q_len -= prefix_q
    if PREFIX_ONLY:
        prefix_q = min(q_len, q_len - k_len & Q_ALIGNMENT - 1)
        if tail_idx >= prefix_q:
            return
        q_idx = q_bos + tail_idx
        compact_idx = bid * Q_ALIGNMENT + tail_idx
    else:
        shift = k_len - q_len
        if shift % Q_ALIGNMENT != 0:
            return
        full_q = max(0, min(q_len, k_len // 128 * 128 - shift))
        full_q = full_q // Q_ALIGNMENT * Q_ALIGNMENT
        if full_q + tail_idx >= q_len:
            return
        q_idx = q_bos + full_q + tail_idx
        compact_idx = bid * 128 + tail_idx
    head_idx = tl.program_id(1)
    split_idx = tl.arange(0, MAX_N_SPLITS)
    split_mask = split_idx < NUM_SPLITS
    lse_offsets = split_idx * h * TAIL_ROWS + head_idx * TAIL_ROWS + compact_idx
    split_lse = tl.load(
        lse_splits_ptr + lse_offsets, mask=split_mask, other=float("-inf")
    )
    valid_lse = split_mask & (split_lse > float("-inf")) & (split_lse < float("inf"))
    max_lse = tl.max(tl.where(valid_lse, split_lse, float("-inf")), axis=0)
    has_valid_lse = max_lse > float("-inf")
    safe_max_lse = tl.where(has_valid_lse, max_lse, 0.0)
    split_scale = tl.where(valid_lse, tl.exp(split_lse - safe_max_lse), 0.0)
    scale_sum = tl.sum(split_scale, axis=0)
    safe_scale_sum = tl.where(has_valid_lse, scale_sum, 1.0)
    weights = split_scale / safe_scale_sum
    col = tl.arange(0, BLOCK_K)
    split_out_offsets = (
        split_idx[:, None] * TAIL_ROWS * h * d
        + (compact_idx * h + head_idx) * d
        + col[None, :]
    )
    split_out = tl.load(
        out_splits_ptr + split_out_offsets,
        mask=split_mask[:, None] & (col[None, :] < d),
        other=0.0,
    )
    out = tl.sum(weights[:, None] * split_out, axis=0)
    out_offsets = q_idx * o_row_stride + head_idx * o_head_stride + col
    tl.store(out_ptr + out_offsets, out, mask=col < d)
    merged_lse = tl.where(
        has_valid_lse, tl.log(safe_scale_sum) + safe_max_lse, float("-inf")
    )
    tl.store(lse_ptr + head_idx * total_q + q_idx, merged_lse)


@triton.jit
def _merge_region(
    out_ptr,
    lse_ptr,
    out_splits_ptr,
    lse_splits_ptr,
    o_row_stride,
    o_head_stride,
    total_q: tl.constexpr,
    h: tl.constexpr,
    d: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    MAX_N_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    cu_q_ptr,
    cu_k_ptr,
    used_k_ptr,
    IS_SEQUSED_K: tl.constexpr,
    TAIL_ROWS: tl.constexpr,
    PEEL_Q: tl.constexpr,
    PREFIX_ONLY: tl.constexpr,
    Q_ALIGNMENT: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
):
    first_idx = tl.program_id(0) * 4 - ROW_OFFSET
    tail_idx = first_idx + tl.arange(0, 4)
    bid = tl.program_id(2)
    q_bos = tl.load(cu_q_ptr + bid)
    q_len = tl.load(cu_q_ptr + bid + 1) - q_bos
    if IS_SEQUSED_K:
        k_len = tl.load(used_k_ptr + bid)
    else:
        k_len = tl.load(cu_k_ptr + bid + 1) - tl.load(cu_k_ptr + bid)
    if PEEL_Q:
        prefix_q = min(q_len, (q_len - k_len) & (Q_ALIGNMENT - 1))
        q_bos += prefix_q
        q_len -= prefix_q
    if PREFIX_ONLY:
        prefix_q = min(q_len, (q_len - k_len) & (Q_ALIGNMENT - 1))
        if first_idx >= prefix_q:
            return
        row_mask = tail_idx < prefix_q
        q_idx = q_bos + tail_idx
        compact_idx = bid * Q_ALIGNMENT + tail_idx
    else:
        shift = k_len - q_len
        if shift % Q_ALIGNMENT != 0:
            return
        full_q = max(0, min(q_len, (k_len // 128) * 128 - shift))
        full_q = (full_q // Q_ALIGNMENT) * Q_ALIGNMENT
        if full_q + first_idx >= q_len:
            return
        row_mask = full_q + tail_idx < q_len
        q_idx = q_bos + full_q + tail_idx
        compact_idx = bid * 128 + tail_idx
    head_idx = tl.program_id(1)
    split_idx = tl.arange(0, MAX_N_SPLITS)
    split_mask = split_idx < NUM_SPLITS
    valid = split_mask[:, None] & row_mask[None, :]
    lse_offsets = (
        split_idx[:, None] * h * TAIL_ROWS + head_idx * TAIL_ROWS + compact_idx[None, :]
    )
    split_lse = tl.load(lse_splits_ptr + lse_offsets, mask=valid, other=float("-inf"))
    valid_lse = valid & (split_lse > float("-inf")) & (split_lse < float("inf"))
    max_lse = tl.max(tl.where(valid_lse, split_lse, float("-inf")), axis=0)
    has_valid_lse = max_lse > float("-inf")
    safe_max_lse = tl.where(has_valid_lse, max_lse, 0.0)
    split_scale = tl.where(valid_lse, tl.exp(split_lse - safe_max_lse[None, :]), 0.0)
    scale_sum = tl.sum(split_scale, axis=0)
    safe_scale_sum = tl.where(has_valid_lse, scale_sum, 1.0)
    col = tl.arange(0, BLOCK_K)
    out = tl.full((4, BLOCK_K), 0.0, tl.float32)
    for split in range(NUM_SPLITS):
        lse_i = tl.load(
            lse_splits_ptr + split * h * TAIL_ROWS + head_idx * TAIL_ROWS + compact_idx,
            mask=row_mask,
            other=float("-inf"),
        )
        valid_i = row_mask & (lse_i > float("-inf")) & (lse_i < float("inf"))
        weight = tl.where(valid_i, tl.exp(lse_i - safe_max_lse) / safe_scale_sum, 0.0)
        offsets = (
            split * TAIL_ROWS * h * d
            + (compact_idx[:, None] * h + head_idx) * d
            + col[None, :]
        )
        partial = tl.load(
            out_splits_ptr + offsets,
            mask=valid_i[:, None] & (col[None, :] < d),
            other=0.0,
        )
        out += weight[:, None] * partial
    out_offsets = (
        q_idx[:, None] * o_row_stride + head_idx * o_head_stride + col[None, :]
    )
    tl.store(out_ptr + out_offsets, out, mask=row_mask[:, None] & (col[None, :] < d))
    merged_lse = tl.where(
        has_valid_lse, tl.log(safe_scale_sum) + safe_max_lse, float("-inf")
    )
    tl.store(lse_ptr + head_idx * total_q + q_idx, merged_lse, mask=row_mask)


@libentry()
@triton.jit
def fused_boundary_merge_kernel(
    out_ptr,
    lse_ptr,
    tail_out_ptr,
    tail_lse_ptr,
    prefix_out_ptr,
    prefix_lse_ptr,
    o_row_stride,
    o_head_stride,
    total_q: tl.constexpr,
    h: tl.constexpr,
    cu_q_ptr,
    cu_k_ptr,
    used_k_ptr,
    IS_SEQUSED_K: tl.constexpr,
    TAIL_ROWS: tl.constexpr,
    PREFIX_ROWS: tl.constexpr,
    PREFIX_SPLITS: tl.constexpr,
    TAIL_SPLITS: tl.constexpr,
):
    # Each CTA uses the original split count and merge arithmetic for its region.
    if tl.program_id(0) >= 32:
        _merge_region(
            out_ptr,
            lse_ptr,
            prefix_out_ptr,
            prefix_lse_ptr,
            o_row_stride,
            o_head_stride,
            total_q,
            h,
            256,
            PREFIX_SPLITS,
            16,
            256,
            cu_q_ptr,
            cu_k_ptr,
            used_k_ptr,
            IS_SEQUSED_K,
            PREFIX_ROWS,
            False,
            True,
            16,
            128,
        )
    else:
        _merge_region(
            out_ptr,
            lse_ptr,
            tail_out_ptr,
            tail_lse_ptr,
            o_row_stride,
            o_head_stride,
            total_q,
            h,
            256,
            TAIL_SPLITS,
            4,
            256,
            cu_q_ptr,
            cu_k_ptr,
            used_k_ptr,
            IS_SEQUSED_K,
            TAIL_ROWS,
            True,
            False,
            16,
            0,
        )


def launch_boundary_merge(params, *, batch_size, num_heads, total_q, tail, prefix):
    """Merge S4 aligned tails and S16 prefixes after both producers finish."""
    assert params.block_size == 16 and params.h_hk_ratio == 4
    return fused_boundary_merge_kernel[(36, num_heads, batch_size)](
        params.o_ptr,
        params.softmax_lse_ptr,
        tail[1],
        tail[2],
        prefix[1],
        prefix[2],
        params.o_row_stride,
        params.o_head_stride,
        total_q,
        num_heads,
        params.cu_seqlens_q_ptr,
        params.cu_seqlens_k_ptr,
        params.seqused_k_ptr,
        params.is_seqused_k,
        batch_size * 128,
        batch_size * 16,
        prefix[1].shape[0],
        tail[1].shape[0],
        num_warps=1,
        num_stages=1,
    )
