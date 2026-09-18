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

"""D256 async-TN bulk kernel for complete, aligned paged attention tiles."""
import logging

import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

from .common import apply_mask
from .tn_addressing import tn_compile_scenario

logger = logging.getLogger(__name__)
_D256_FRAGMENT = 128
_QK_FRAGMENT = 64


@triton.jit
def _paged_tile_coords(
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
def _online_softmax_stats(
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
        (col_idx, page_id, page_offset) = _paged_tile_coords(
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
            (acc_scale, probabilities, row_max, row_sum) = _online_softmax_stats(
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
        (col_idx, page_id, page_offset) = _paged_tile_coords(
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
            (acc_scale, probabilities, row_max, row_sum) = _online_softmax_stats(
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
        **cfg,
    )


__all__ = ["flash_varlen_fwd_d256_tn_kernel", "launch_d256_tn_direct"]
