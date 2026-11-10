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


"""Fused SFA Cube/Vector pipeline, including independent cross-query states."""

import triton
import triton.experimental.tle as tle
import triton.language as tl

from flaggems_vllm.runtime.backend._ascend.ops.sparse_flash_attention.common import (
    ASCEND_PIPE,
    TL_FAST_PAGE_OFFSET_MASK,
    TL_FAST_PAGE_SIZE_LOG2,
    TL_FAST_SPARSE_ENTRIES,
    TL_KV_DIM,
    TL_MASKED_GATHER_ROWS,
    TL_NUM_PIPELINE_SLOTS,
    TL_OUTPUT_STAGE_LAG,
    TL_PV_SYNC_EVENT_BASE,
    TL_ROPE_DIM,
    TL_SOFTMAX_ALPHA_STATE_BASE,
    TL_SOFTMAX_STATE_ROWS,
    TL_VECTOR_SUBCORES_PER_CUBE,
    TL_VLLM_FINITE_SOFTMAX_SENTINEL,
    _sfa_query_meta,
)


@triton.jit
def _sfa_pipeline(
    QUERY,
    ACTUAL_QUERY_HEADS: tl.constexpr,
    PIPELINE_HEADS: tl.constexpr,
    CROSS_QUERY_PIPELINE: tl.constexpr,
    QUERY_ROPE,
    KV,
    KEY_ROPE,
    LOGICAL_SPARSE_INDICES,
    PHYSICAL_ROW_INDICES,
    BLOCK_TABLE,
    KV_WORKSPACE,
    ROPE_WORKSPACE,
    SCORE_WORKSPACE,
    UNNORMALIZED_WEIGHTS_WORKSPACE,
    BLOCK_WEIGHTED_VALUE_SUM_WORKSPACE,
    SOFTMAX_STATE_WORKSPACE,
    OUTPUT_ACCUMULATOR_WORKSPACE,
    OUT,
    SPLIT_WEIGHTED_VALUE_SUM,
    SPLIT_SOFTMAX_MAX,
    SPLIT_SOFTMAX_SUM,
    QUERY_LENGTHS,
    PACKED_QUERY_META,
    KV_LENGTHS,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    USE_PREPARED_PHYSICAL_INDICES: tl.constexpr,
    CACHE_STORAGE_ROW_SPAN: tl.constexpr,
    NUM_QUERY_TOKENS: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    QUERY_SEQ_CAPACITY: tl.constexpr,
    QUERY_IS_TND: tl.constexpr,
    HAS_QUERY_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr,
    RETURN_SOFTMAX_STATS: tl.constexpr,
    NUM_SPARSE_BLOCKS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGES_PER_REQUEST: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    KV_BLOCK_ROWS: tl.constexpr,
    QK_BLOCK_COLUMNS: tl.constexpr,
    GATHER_BLOCK_ROWS: tl.constexpr,
    USE_INT32_OFFSETS: tl.constexpr,
    SOFTMAX_BLOCK_HEADS: tl.constexpr,
    VALUE_BLOCK_COLUMNS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    OUTPUT_BLOCK_COLUMNS: tl.constexpr,
    OUTPUT_BLOCK_HEADS: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    WORKSPACE_PADDING: tl.constexpr,
):
    # Three generations overlap gather/QK, softmax/PV and output update.
    # Each producer signals after its stores; the consumer waits before loads.
    # The ordinary schedule drains after each query. The cross-query schedule
    # drains once per core, keeping gather/QK, softmax/PV and output busy across
    # query boundaries. Each stage reads its own query's metadata. Two state
    # buffers isolate the next query's softmax from the current query's output.
    # Prepared indices address physical storage rows; otherwise indices name
    # logical KV rows. Keep both inputs separate from the query-length tensor.
    if USE_PREPARED_PHYSICAL_INDICES:
        selected_row_indices = PHYSICAL_ROW_INDICES
    else:
        selected_row_indices = LOGICAL_SPARSE_INDICES
    # The host requires two Vector subcores per Cube; synchronization is explicit.
    NUM_VECTOR_SUBCORES: tl.constexpr = TL_VECTOR_SUBCORES_PER_CUBE
    program_id = tl.program_id(0)
    vector_subcore_id = tle.dsa.ascend.sub_vec_id()
    BLOCKS_PER_QUERY: tl.constexpr = NUM_SPARSE_BLOCKS // KV_BLOCK_ROWS // NUM_KV_SPLITS
    if CROSS_QUERY_PIPELINE:
        tl.static_assert(USE_PREPARED_PHYSICAL_INDICES)
        tl.static_assert(NUM_KV_SPLITS == 1)
        num_query_tasks = tl.cdiv(NUM_QUERY_TOKENS - program_id, NUM_CORES)
        total_blocks = num_query_tasks * BLOCKS_PER_QUERY
    else:
        total_blocks = BLOCKS_PER_QUERY
    for task_idx in range(
        program_id,
        program_id + 1 if CROSS_QUERY_PIPELINE else NUM_QUERY_TOKENS * NUM_KV_SPLITS,
        NUM_CORES,
    ):
        if CROSS_QUERY_PIPELINE:
            kv_split_idx = 0
        else:
            query_token_idx = (task_idx // NUM_KV_SPLITS).to(tl.int64)
            kv_split_idx = task_idx % NUM_KV_SPLITS
            if USE_PREPARED_PHYSICAL_INDICES:
                packed_query_meta = tl.load(PACKED_QUERY_META + query_token_idx).to(
                    tl.uint32
                )
                kv_end_exclusive = (packed_query_meta >> 1).to(tl.int32)
                all_sparse_entries_valid = (packed_query_meta & 1) != 0
            else:
                request_idx, kv_end_exclusive, _ = _sfa_query_meta(
                    query_token_idx,
                    QUERY_LENGTHS,
                    KV_LENGTHS,
                    NUM_REQUESTS,
                    QUERY_SEQ_CAPACITY,
                    0,
                    QUERY_IS_TND,
                    False,
                    HAS_QUERY_LENGTHS,
                    True,
                    CAUSAL,
                )
                selection_positions = tl.arange(0, TL_FAST_SPARSE_ENTRIES)
                logical_indices_to_check = tl.load(
                    selected_row_indices
                    + query_token_idx * TL_FAST_SPARSE_ENTRIES
                    + selection_positions
                )
                all_sparse_entries_valid = (
                    (kv_end_exclusive >= TL_FAST_SPARSE_ENTRIES)
                    & (tl.min(logical_indices_to_check, 0) >= 0)
                    & (tl.max(logical_indices_to_check, 0) < kv_end_exclusive)
                    & (
                        tl.max(logical_indices_to_check, 0) // PAGE_SIZE
                        < PAGES_PER_REQUEST
                    )
                )
            if not USE_PREPARED_PHYSICAL_INDICES:
                page_table_indices = tl.arange(
                    0, triton.next_power_of_2(PAGES_PER_REQUEST)
                )
                request_page_table = tl.load(
                    BLOCK_TABLE + request_idx * PAGES_PER_REQUEST + page_table_indices,
                    mask=page_table_indices < PAGES_PER_REQUEST,
                    other=0,
                )
        for pipeline_tick in range(total_blocks + TL_OUTPUT_STAGE_LAG):
            gather_slot = pipeline_tick % TL_NUM_PIPELINE_SLOTS
            if pipeline_tick < total_blocks:
                if CROSS_QUERY_PIPELINE:
                    query_task = pipeline_tick // BLOCKS_PER_QUERY
                    query_token_idx = (program_id + query_task * NUM_CORES).to(tl.int64)
                    state_program_id = program_id * 2 + query_task % 2
                    packed_query_meta = tl.load(PACKED_QUERY_META + query_token_idx).to(
                        tl.uint32
                    )
                    kv_end_exclusive = (packed_query_meta >> 1).to(tl.int32)
                    all_sparse_entries_valid = (packed_query_meta & 1) != 0
                gather_block = (
                    pipeline_tick % BLOCKS_PER_QUERY
                    if CROSS_QUERY_PIPELINE
                    else pipeline_tick
                )
                if all_sparse_entries_valid:
                    for gather_group_idx in range(
                        KV_BLOCK_ROWS // GATHER_BLOCK_ROWS // NUM_VECTOR_SUBCORES
                    ):
                        kv_feature_indices = tl.arange(0, TL_KV_DIM)
                        rope_feature_indices = tl.arange(0, TL_ROPE_DIM)
                        if USE_PREPARED_PHYSICAL_INDICES:
                            selection_start = (
                                kv_split_idx * (NUM_SPARSE_BLOCKS // NUM_KV_SPLITS)
                                + gather_block * KV_BLOCK_ROWS
                                + (
                                    gather_group_idx * NUM_VECTOR_SUBCORES
                                    + vector_subcore_id
                                )
                                * GATHER_BLOCK_ROWS
                            )
                            physical_indices_block_ptr = tl.make_block_ptr(
                                selected_row_indices
                                + query_token_idx * NUM_SPARSE_BLOCKS,
                                (NUM_SPARSE_BLOCKS, 1),
                                (1, 1),
                                (selection_start.to(tl.int32), 0),
                                (GATHER_BLOCK_ROWS, 1),
                                (1, 0),
                            )
                            kv_cache_block_ptr = tl.make_block_ptr(
                                KV,
                                (CACHE_STORAGE_ROW_SPAN, TL_KV_DIM),
                                (TL_KV_DIM, 1),
                                (0, 0),
                                (GATHER_BLOCK_ROWS, TL_KV_DIM),
                                (1, 0),
                            )
                            rope_cache_block_ptr = tl.make_block_ptr(
                                KEY_ROPE,
                                (CACHE_STORAGE_ROW_SPAN, TL_ROPE_DIM),
                                (TL_ROPE_DIM, 1),
                                (0, 0),
                                (GATHER_BLOCK_ROWS, TL_ROPE_DIM),
                                (1, 0),
                            )
                            gathered_kv_tile = tl.full(
                                (GATHER_BLOCK_ROWS, TL_KV_DIM), 0, KV.dtype.element_ty
                            )
                            gathered_rope_tile = tl.full(
                                (GATHER_BLOCK_ROWS, TL_ROPE_DIM), 0, KV.dtype.element_ty
                            )
                            gathered_kv_tile = tle.dsa.ascend.raw(
                                "gather_gm_to_ub",
                                kv_cache_block_ptr,
                                physical_indices_block_ptr,
                                GATHER_BLOCK_ROWS,
                                TL_KV_DIM,
                                out=gathered_kv_tile,
                            )
                            gathered_rope_tile = tle.dsa.ascend.raw(
                                "gather_gm_to_ub",
                                rope_cache_block_ptr,
                                physical_indices_block_ptr,
                                GATHER_BLOCK_ROWS,
                                TL_ROPE_DIM,
                                out=gathered_rope_tile,
                            )
                        else:
                            sparse_positions = (
                                kv_split_idx * (NUM_SPARSE_BLOCKS // NUM_KV_SPLITS)
                                + gather_block * KV_BLOCK_ROWS
                                + (
                                    gather_group_idx * NUM_VECTOR_SUBCORES
                                    + vector_subcore_id
                                )
                                * GATHER_BLOCK_ROWS
                                + tl.arange(0, GATHER_BLOCK_ROWS)
                            )
                            selected_indices = tl.load(
                                selected_row_indices
                                + query_token_idx * NUM_SPARSE_BLOCKS
                                + sparse_positions
                            )
                            # Preserve int32 page-ID bits through float32 gather; this is
                            # not a numeric cast. The original lowering rationale is unverified.
                            physical_page_ids = tl.gather(
                                request_page_table.to(tl.float32, bitcast=True),
                                selected_indices >> TL_FAST_PAGE_SIZE_LOG2,
                                0,
                            ).to(tl.int32, bitcast=True)
                            physical_row_indices = physical_page_ids.to(
                                tl.int32 if USE_INT32_OFFSETS else tl.int64
                            ) * PAGE_SIZE + (
                                selected_indices & TL_FAST_PAGE_OFFSET_MASK
                            )
                            gathered_kv_tile = tl.load(
                                KV
                                + physical_row_indices[:, None] * TL_KV_DIM
                                + kv_feature_indices[None, :]
                            )
                            gathered_rope_tile = tl.load(
                                KEY_ROPE
                                + physical_row_indices[:, None] * TL_ROPE_DIM
                                + rope_feature_indices[None, :]
                            )
                        workspace_row_indices = (
                            (program_id * TL_NUM_PIPELINE_SLOTS + gather_slot)
                            * (KV_BLOCK_ROWS + WORKSPACE_PADDING)
                            + (
                                gather_group_idx * NUM_VECTOR_SUBCORES
                                + vector_subcore_id
                            )
                            * GATHER_BLOCK_ROWS
                            + tl.arange(0, GATHER_BLOCK_ROWS)
                        )
                        tl.store(
                            KV_WORKSPACE
                            + workspace_row_indices[:, None] * TL_KV_DIM
                            + kv_feature_indices[None, :],
                            gathered_kv_tile,
                        )
                        tl.store(
                            ROPE_WORKSPACE
                            + workspace_row_indices[:, None] * TL_ROPE_DIM
                            + rope_feature_indices[None, :],
                            gathered_rope_tile,
                        )
                else:
                    for gather_group_idx in range(
                        KV_BLOCK_ROWS // TL_MASKED_GATHER_ROWS // NUM_VECTOR_SUBCORES
                    ):
                        sparse_positions = (
                            kv_split_idx * (NUM_SPARSE_BLOCKS // NUM_KV_SPLITS)
                            + gather_block * KV_BLOCK_ROWS
                            + (
                                gather_group_idx * NUM_VECTOR_SUBCORES
                                + vector_subcore_id
                            )
                            * TL_MASKED_GATHER_ROWS
                            + tl.arange(0, TL_MASKED_GATHER_ROWS)
                        )
                        selected_indices = tl.load(
                            selected_row_indices
                            + query_token_idx * NUM_SPARSE_BLOCKS
                            + sparse_positions
                        )
                        valid_kv_mask = (selected_indices >= 0) & (
                            sparse_positions < kv_end_exclusive
                        )
                        if not USE_PREPARED_PHYSICAL_INDICES:
                            valid_kv_mask &= (selected_indices < kv_end_exclusive) & (
                                selected_indices // PAGE_SIZE < PAGES_PER_REQUEST
                            )
                        selected_indices = tl.where(valid_kv_mask, selected_indices, 0)
                        if USE_PREPARED_PHYSICAL_INDICES:
                            physical_row_indices = selected_indices.to(
                                tl.int32 if USE_INT32_OFFSETS else tl.int64
                            )
                        else:
                            # Preserve int32 page-ID bits through float32 gather; this is
                            # not a numeric cast. The original lowering rationale is unverified.
                            physical_page_ids = tl.gather(
                                request_page_table.to(tl.float32, bitcast=True),
                                selected_indices >> TL_FAST_PAGE_SIZE_LOG2,
                                0,
                            ).to(tl.int32, bitcast=True)
                            physical_row_indices = physical_page_ids.to(
                                tl.int32 if USE_INT32_OFFSETS else tl.int64
                            ) * PAGE_SIZE + (
                                selected_indices & TL_FAST_PAGE_OFFSET_MASK
                            )
                        kv_feature_indices = tl.arange(0, TL_KV_DIM)
                        rope_feature_indices = tl.arange(0, TL_ROPE_DIM)
                        gathered_kv_tile = tl.load(
                            KV
                            + physical_row_indices[:, None] * TL_KV_DIM
                            + kv_feature_indices[None, :],
                            valid_kv_mask[:, None],
                            other=0,
                        )
                        gathered_rope_tile = tl.load(
                            KEY_ROPE
                            + physical_row_indices[:, None] * TL_ROPE_DIM
                            + rope_feature_indices[None, :],
                            valid_kv_mask[:, None],
                            other=0,
                        )
                        workspace_row_indices = (
                            (program_id * TL_NUM_PIPELINE_SLOTS + gather_slot)
                            * (KV_BLOCK_ROWS + WORKSPACE_PADDING)
                            + (
                                gather_group_idx * NUM_VECTOR_SUBCORES
                                + vector_subcore_id
                            )
                            * TL_MASKED_GATHER_ROWS
                            + tl.arange(0, TL_MASKED_GATHER_ROWS)
                        )
                        tl.store(
                            KV_WORKSPACE
                            + workspace_row_indices[:, None] * TL_KV_DIM
                            + kv_feature_indices[None, :],
                            gathered_kv_tile,
                        )
                        tl.store(
                            ROPE_WORKSPACE
                            + workspace_row_indices[:, None] * TL_ROPE_DIM
                            + rope_feature_indices[None, :],
                            gathered_rope_tile,
                        )
                if ACTUAL_QUERY_HEADS == PIPELINE_HEADS:
                    query_tile = tl.load(
                        tl.make_block_ptr(
                            QUERY + query_token_idx * PIPELINE_HEADS * TL_KV_DIM,
                            (PIPELINE_HEADS, TL_KV_DIM),
                            (TL_KV_DIM, 1),
                            (0, 0),
                            (PIPELINE_HEADS, TL_KV_DIM),
                            (1, 0),
                        )
                    )
                else:
                    query_tile = tl.load(
                        tl.make_block_ptr(
                            QUERY + query_token_idx * ACTUAL_QUERY_HEADS * TL_KV_DIM,
                            (ACTUAL_QUERY_HEADS, TL_KV_DIM),
                            (TL_KV_DIM, 1),
                            (0, 0),
                            (PIPELINE_HEADS, TL_KV_DIM),
                            (1, 0),
                        ),
                        boundary_check=(0,),
                        padding_option="zero",
                    )
                if ACTUAL_QUERY_HEADS == PIPELINE_HEADS:
                    query_rope_tile = tl.load(
                        tl.make_block_ptr(
                            QUERY_ROPE + query_token_idx * PIPELINE_HEADS * TL_ROPE_DIM,
                            (PIPELINE_HEADS, TL_ROPE_DIM),
                            (TL_ROPE_DIM, 1),
                            (0, 0),
                            (PIPELINE_HEADS, TL_ROPE_DIM),
                            (1, 0),
                        )
                    )
                else:
                    query_rope_tile = tl.load(
                        tl.make_block_ptr(
                            QUERY_ROPE
                            + query_token_idx * ACTUAL_QUERY_HEADS * TL_ROPE_DIM,
                            (ACTUAL_QUERY_HEADS, TL_ROPE_DIM),
                            (TL_ROPE_DIM, 1),
                            (0, 0),
                            (PIPELINE_HEADS, TL_ROPE_DIM),
                            (1, 0),
                        ),
                        boundary_check=(0,),
                        padding_option="zero",
                    )
                tle.dsa.ascend.sync_block_set(
                    "vector",
                    "cube",
                    gather_slot,
                    ASCEND_PIPE.PIPE_MTE3,
                    ASCEND_PIPE.PIPE_MTE2,
                )
                tle.dsa.ascend.sync_block_wait(
                    "vector",
                    "cube",
                    gather_slot,
                    ASCEND_PIPE.PIPE_MTE3,
                    ASCEND_PIPE.PIPE_MTE2,
                )
                for qk_block_idx in range(KV_BLOCK_ROWS // QK_BLOCK_COLUMNS):
                    kv_tile = tl.load(
                        tl.make_block_ptr(
                            KV_WORKSPACE
                            + (program_id * TL_NUM_PIPELINE_SLOTS + gather_slot)
                            * (KV_BLOCK_ROWS + WORKSPACE_PADDING)
                            * TL_KV_DIM,
                            (KV_BLOCK_ROWS, TL_KV_DIM),
                            (TL_KV_DIM, 1),
                            (qk_block_idx * QK_BLOCK_COLUMNS, 0),
                            (QK_BLOCK_COLUMNS, TL_KV_DIM),
                            (1, 0),
                        )
                    )
                    key_rope_tile = tl.load(
                        tl.make_block_ptr(
                            ROPE_WORKSPACE
                            + (program_id * TL_NUM_PIPELINE_SLOTS + gather_slot)
                            * (KV_BLOCK_ROWS + WORKSPACE_PADDING)
                            * TL_ROPE_DIM,
                            (KV_BLOCK_ROWS, TL_ROPE_DIM),
                            (TL_ROPE_DIM, 1),
                            (qk_block_idx * QK_BLOCK_COLUMNS, 0),
                            (QK_BLOCK_COLUMNS, TL_ROPE_DIM),
                            (1, 0),
                        )
                    )
                    attention_scores = tl.dot(query_tile, tl.trans(kv_tile))
                    attention_scores = tl.dot(
                        query_rope_tile, tl.trans(key_rope_tile), attention_scores
                    )
                    tl.store(
                        tl.make_block_ptr(
                            SCORE_WORKSPACE
                            + (program_id * TL_NUM_PIPELINE_SLOTS + gather_slot)
                            * (
                                PIPELINE_HEADS * KV_BLOCK_ROWS
                                + WORKSPACE_PADDING * PIPELINE_HEADS
                            ),
                            (PIPELINE_HEADS, KV_BLOCK_ROWS),
                            (KV_BLOCK_ROWS, 1),
                            (0, qk_block_idx * QK_BLOCK_COLUMNS),
                            (PIPELINE_HEADS, QK_BLOCK_COLUMNS),
                            (1, 0),
                        ),
                        attention_scores,
                    )
                tle.dsa.ascend.sync_block_set(
                    "cube",
                    "vector",
                    gather_slot,
                    ASCEND_PIPE.PIPE_FIX,
                    ASCEND_PIPE.PIPE_MTE2,
                )
            if pipeline_tick >= 1 and pipeline_tick < total_blocks + 1:
                if CROSS_QUERY_PIPELINE:
                    query_task = (pipeline_tick - 1) // BLOCKS_PER_QUERY
                    query_token_idx = (program_id + query_task * NUM_CORES).to(tl.int64)
                    state_program_id = program_id * 2 + query_task % 2
                    packed_query_meta = tl.load(PACKED_QUERY_META + query_token_idx).to(
                        tl.uint32
                    )
                    kv_end_exclusive = (packed_query_meta >> 1).to(tl.int32)
                    all_sparse_entries_valid = (packed_query_meta & 1) != 0
                else:
                    state_program_id = program_id
                softmax_block = (
                    (pipeline_tick - 1) % BLOCKS_PER_QUERY
                    if CROSS_QUERY_PIPELINE
                    else pipeline_tick - 1
                )
                softmax_slot = (pipeline_tick - 1) % TL_NUM_PIPELINE_SLOTS
                tle.dsa.ascend.sync_block_wait(
                    "cube",
                    "vector",
                    softmax_slot,
                    ASCEND_PIPE.PIPE_FIX,
                    ASCEND_PIPE.PIPE_MTE2,
                )
                additive_attention_mask = tl.full((KV_BLOCK_ROWS,), 0.0, tl.float32)
                if not all_sparse_entries_valid:
                    mask_block_positions = tl.arange(0, KV_BLOCK_ROWS)
                    mask_sparse_positions = (
                        kv_split_idx * (NUM_SPARSE_BLOCKS // NUM_KV_SPLITS)
                        + softmax_block * KV_BLOCK_ROWS
                        + mask_block_positions
                    )
                    mask_selected_indices = tl.load(
                        selected_row_indices
                        + query_token_idx * NUM_SPARSE_BLOCKS
                        + mask_sparse_positions
                    )
                    valid_kv_mask = (mask_selected_indices >= 0) & (
                        mask_sparse_positions < kv_end_exclusive
                    )
                    if not USE_PREPARED_PHYSICAL_INDICES:
                        valid_kv_mask &= (mask_selected_indices < kv_end_exclusive) & (
                            mask_selected_indices // PAGE_SIZE < PAGES_PER_REQUEST
                        )
                    additive_attention_mask = tl.where(
                        valid_kv_mask, 0.0, float("-inf")
                    )
                for head_block_idx in range(
                    PIPELINE_HEADS // SOFTMAX_BLOCK_HEADS // NUM_VECTOR_SUBCORES
                ):
                    head_indices = (
                        head_block_idx * NUM_VECTOR_SUBCORES + vector_subcore_id
                    ) * SOFTMAX_BLOCK_HEADS + tl.arange(0, SOFTMAX_BLOCK_HEADS)
                    sparse_positions = tl.arange(0, KV_BLOCK_ROWS)
                    scaled_attention_scores = (
                        tl.load(
                            SCORE_WORKSPACE
                            + (program_id * TL_NUM_PIPELINE_SLOTS + softmax_slot)
                            * (
                                PIPELINE_HEADS * KV_BLOCK_ROWS
                                + WORKSPACE_PADDING * PIPELINE_HEADS
                            )
                            + head_indices[:, None] * KV_BLOCK_ROWS
                            + sparse_positions[None, :]
                        )
                        * SOFTMAX_SCALE
                    )
                    previous_softmax_max = tl.full(
                        (SOFTMAX_BLOCK_HEADS,),
                        TL_VLLM_FINITE_SOFTMAX_SENTINEL,
                        tl.float32,
                    )
                    previous_softmax_sum = tl.zeros((SOFTMAX_BLOCK_HEADS,), tl.float32)
                    if softmax_block > 0:
                        previous_softmax_max = tl.load(
                            SOFTMAX_STATE_WORKSPACE
                            + state_program_id
                            * (TL_SOFTMAX_STATE_ROWS + WORKSPACE_PADDING)
                            * PIPELINE_HEADS
                            + head_indices
                        )
                        previous_softmax_sum = tl.load(
                            SOFTMAX_STATE_WORKSPACE
                            + state_program_id
                            * (TL_SOFTMAX_STATE_ROWS + WORKSPACE_PADDING)
                            * PIPELINE_HEADS
                            + PIPELINE_HEADS
                            + head_indices
                        )
                    if not all_sparse_entries_valid:
                        scaled_attention_scores = (
                            scaled_attention_scores + additive_attention_mask[None, :]
                        )
                    next_softmax_max = tl.maximum(
                        previous_softmax_max, tl.max(scaled_attention_scores, 1)
                    )
                    previous_state_scale = tl.exp(
                        previous_softmax_max - next_softmax_max
                    )
                    unnormalized_weights = tl.exp(
                        scaled_attention_scores - next_softmax_max[:, None]
                    )
                    next_softmax_sum = (
                        previous_softmax_sum * previous_state_scale
                        + tl.sum(unnormalized_weights, 1)
                    )
                    tl.store(
                        SOFTMAX_STATE_WORKSPACE
                        + state_program_id
                        * (TL_SOFTMAX_STATE_ROWS + WORKSPACE_PADDING)
                        * PIPELINE_HEADS
                        + head_indices,
                        next_softmax_max,
                    )
                    tl.store(
                        SOFTMAX_STATE_WORKSPACE
                        + state_program_id
                        * (TL_SOFTMAX_STATE_ROWS + WORKSPACE_PADDING)
                        * PIPELINE_HEADS
                        + PIPELINE_HEADS
                        + head_indices,
                        next_softmax_sum,
                    )
                    tl.store(
                        SOFTMAX_STATE_WORKSPACE
                        + state_program_id
                        * (TL_SOFTMAX_STATE_ROWS + WORKSPACE_PADDING)
                        * PIPELINE_HEADS
                        + (TL_SOFTMAX_ALPHA_STATE_BASE + softmax_slot) * PIPELINE_HEADS
                        + head_indices,
                        previous_state_scale,
                    )
                    tl.store(
                        UNNORMALIZED_WEIGHTS_WORKSPACE
                        + (program_id * TL_NUM_PIPELINE_SLOTS + softmax_slot)
                        * (
                            PIPELINE_HEADS * KV_BLOCK_ROWS
                            + WORKSPACE_PADDING * PIPELINE_HEADS
                        )
                        + head_indices[:, None] * KV_BLOCK_ROWS
                        + sparse_positions[None, :],
                        unnormalized_weights.to(KV.dtype.element_ty),
                    )
                tle.dsa.ascend.sync_block_set(
                    "vector",
                    "cube",
                    softmax_slot + TL_PV_SYNC_EVENT_BASE,
                    ASCEND_PIPE.PIPE_MTE3,
                    ASCEND_PIPE.PIPE_MTE2,
                )
                tle.dsa.ascend.sync_block_wait(
                    "vector",
                    "cube",
                    softmax_slot + TL_PV_SYNC_EVENT_BASE,
                    ASCEND_PIPE.PIPE_MTE3,
                    ASCEND_PIPE.PIPE_MTE2,
                )
                weights_tile = tl.load(
                    tl.make_block_ptr(
                        UNNORMALIZED_WEIGHTS_WORKSPACE
                        + (program_id * TL_NUM_PIPELINE_SLOTS + softmax_slot)
                        * (
                            PIPELINE_HEADS * KV_BLOCK_ROWS
                            + WORKSPACE_PADDING * PIPELINE_HEADS
                        ),
                        (PIPELINE_HEADS, KV_BLOCK_ROWS),
                        (KV_BLOCK_ROWS, 1),
                        (0, 0),
                        (PIPELINE_HEADS, KV_BLOCK_ROWS),
                        (1, 0),
                    )
                )
                for value_block_idx in range(TL_KV_DIM // VALUE_BLOCK_COLUMNS):
                    value_tile = tl.load(
                        tl.make_block_ptr(
                            KV_WORKSPACE
                            + (program_id * TL_NUM_PIPELINE_SLOTS + softmax_slot)
                            * (KV_BLOCK_ROWS + WORKSPACE_PADDING)
                            * TL_KV_DIM,
                            (KV_BLOCK_ROWS, TL_KV_DIM),
                            (TL_KV_DIM, 1),
                            (0, value_block_idx * VALUE_BLOCK_COLUMNS),
                            (KV_BLOCK_ROWS, VALUE_BLOCK_COLUMNS),
                            (1, 0),
                        )
                    )
                    block_weighted_value_sum = tl.dot(weights_tile, value_tile)
                    tl.store(
                        tl.make_block_ptr(
                            BLOCK_WEIGHTED_VALUE_SUM_WORKSPACE
                            + (program_id * TL_NUM_PIPELINE_SLOTS + softmax_slot)
                            * (
                                PIPELINE_HEADS * TL_KV_DIM
                                + WORKSPACE_PADDING * PIPELINE_HEADS
                            ),
                            (PIPELINE_HEADS, TL_KV_DIM),
                            (TL_KV_DIM, 1),
                            (0, value_block_idx * VALUE_BLOCK_COLUMNS),
                            (PIPELINE_HEADS, VALUE_BLOCK_COLUMNS),
                            (1, 0),
                        ),
                        block_weighted_value_sum,
                    )
                tle.dsa.ascend.sync_block_set(
                    "cube",
                    "vector",
                    softmax_slot + TL_PV_SYNC_EVENT_BASE,
                    ASCEND_PIPE.PIPE_FIX,
                    ASCEND_PIPE.PIPE_MTE2,
                )
            if pipeline_tick >= TL_OUTPUT_STAGE_LAG:
                if CROSS_QUERY_PIPELINE:
                    query_task = (
                        pipeline_tick - TL_OUTPUT_STAGE_LAG
                    ) // BLOCKS_PER_QUERY
                    query_token_idx = (program_id + query_task * NUM_CORES).to(tl.int64)
                    state_program_id = program_id * 2 + query_task % 2
                    packed_query_meta = tl.load(PACKED_QUERY_META + query_token_idx).to(
                        tl.uint32
                    )
                    kv_end_exclusive = (packed_query_meta >> 1).to(tl.int32)
                    all_sparse_entries_valid = (packed_query_meta & 1) != 0
                else:
                    state_program_id = program_id
                output_block = (
                    (pipeline_tick - TL_OUTPUT_STAGE_LAG) % BLOCKS_PER_QUERY
                    if CROSS_QUERY_PIPELINE
                    else pipeline_tick - TL_OUTPUT_STAGE_LAG
                )
                output_slot = (
                    pipeline_tick - TL_OUTPUT_STAGE_LAG
                ) % TL_NUM_PIPELINE_SLOTS
                tle.dsa.ascend.sync_block_wait(
                    "cube",
                    "vector",
                    output_slot + TL_PV_SYNC_EVENT_BASE,
                    ASCEND_PIPE.PIPE_FIX,
                    ASCEND_PIPE.PIPE_MTE2,
                )
                for head_block_idx in range(
                    PIPELINE_HEADS // OUTPUT_BLOCK_HEADS // NUM_VECTOR_SUBCORES
                ):
                    head_indices = (
                        head_block_idx * NUM_VECTOR_SUBCORES + vector_subcore_id
                    ) * OUTPUT_BLOCK_HEADS + tl.arange(0, OUTPUT_BLOCK_HEADS)
                    previous_state_scale = tl.load(
                        SOFTMAX_STATE_WORKSPACE
                        + state_program_id
                        * (TL_SOFTMAX_STATE_ROWS + WORKSPACE_PADDING)
                        * PIPELINE_HEADS
                        + (TL_SOFTMAX_ALPHA_STATE_BASE + output_slot) * PIPELINE_HEADS
                        + head_indices
                    )
                    softmax_sum = tl.load(
                        SOFTMAX_STATE_WORKSPACE
                        + state_program_id
                        * (TL_SOFTMAX_STATE_ROWS + WORKSPACE_PADDING)
                        * PIPELINE_HEADS
                        + PIPELINE_HEADS
                        + head_indices
                    )
                    inverse_softmax_sum = 1.0 / tl.where(
                        softmax_sum > 0, softmax_sum, 1.0
                    )
                    for value_block_idx in range(TL_KV_DIM // OUTPUT_BLOCK_COLUMNS):
                        kv_columns = value_block_idx * OUTPUT_BLOCK_COLUMNS + tl.arange(
                            0, OUTPUT_BLOCK_COLUMNS
                        )
                        previous_weighted_value_sum = tl.zeros(
                            (OUTPUT_BLOCK_HEADS, OUTPUT_BLOCK_COLUMNS), tl.float32
                        )
                        if output_block > 0:
                            previous_weighted_value_sum = tl.load(
                                OUTPUT_ACCUMULATOR_WORKSPACE
                                + state_program_id
                                * PIPELINE_HEADS
                                * (TL_KV_DIM + WORKSPACE_PADDING)
                                + head_indices[:, None] * TL_KV_DIM
                                + kv_columns[None, :]
                            )
                        block_weighted_value_sum = tl.load(
                            BLOCK_WEIGHTED_VALUE_SUM_WORKSPACE
                            + (program_id * TL_NUM_PIPELINE_SLOTS + output_slot)
                            * (
                                PIPELINE_HEADS * TL_KV_DIM
                                + WORKSPACE_PADDING * PIPELINE_HEADS
                            )
                            + head_indices[:, None] * TL_KV_DIM
                            + kv_columns[None, :]
                        )
                        weighted_value_sum = (
                            previous_weighted_value_sum * previous_state_scale[:, None]
                            + block_weighted_value_sum
                        )
                        if output_block == BLOCKS_PER_QUERY - 1:
                            if NUM_KV_SPLITS == 1:
                                tl.store(
                                    OUT
                                    + query_token_idx * ACTUAL_QUERY_HEADS * TL_KV_DIM
                                    + head_indices[:, None] * TL_KV_DIM
                                    + kv_columns[None, :],
                                    weighted_value_sum * inverse_softmax_sum[:, None],
                                    head_indices[:, None] < ACTUAL_QUERY_HEADS,
                                )
                                if RETURN_SOFTMAX_STATS:
                                    softmax_max = tl.load(
                                        SOFTMAX_STATE_WORKSPACE
                                        + state_program_id
                                        * (TL_SOFTMAX_STATE_ROWS + WORKSPACE_PADDING)
                                        * PIPELINE_HEADS
                                        + head_indices
                                    )
                                    tl.store(
                                        SOFTMAX_MAX
                                        + query_token_idx * ACTUAL_QUERY_HEADS
                                        + head_indices,
                                        tl.where(
                                            softmax_sum > 0,
                                            softmax_max,
                                            tl.where(
                                                kv_end_exclusive > 0,
                                                TL_VLLM_FINITE_SOFTMAX_SENTINEL,
                                                0.0,
                                            ),
                                        ),
                                        head_indices < ACTUAL_QUERY_HEADS,
                                    )
                                    tl.store(
                                        SOFTMAX_SUM
                                        + query_token_idx * ACTUAL_QUERY_HEADS
                                        + head_indices,
                                        tl.where(
                                            softmax_sum > 0,
                                            softmax_sum,
                                            tl.minimum(
                                                NUM_SPARSE_BLOCKS, kv_end_exclusive
                                            ).to(tl.float32),
                                        ),
                                        head_indices < ACTUAL_QUERY_HEADS,
                                    )
                            else:
                                tl.store(
                                    SPLIT_WEIGHTED_VALUE_SUM
                                    + task_idx * PIPELINE_HEADS * TL_KV_DIM
                                    + head_indices[:, None] * TL_KV_DIM
                                    + kv_columns[None, :],
                                    weighted_value_sum,
                                )
                                softmax_max = tl.load(
                                    SOFTMAX_STATE_WORKSPACE
                                    + state_program_id
                                    * (TL_SOFTMAX_STATE_ROWS + WORKSPACE_PADDING)
                                    * PIPELINE_HEADS
                                    + head_indices
                                )
                                tl.store(
                                    SPLIT_SOFTMAX_MAX
                                    + task_idx * PIPELINE_HEADS
                                    + head_indices,
                                    softmax_max,
                                )
                                tl.store(
                                    SPLIT_SOFTMAX_SUM
                                    + task_idx * PIPELINE_HEADS
                                    + head_indices,
                                    softmax_sum,
                                )
                        else:
                            tl.store(
                                OUTPUT_ACCUMULATOR_WORKSPACE
                                + state_program_id
                                * PIPELINE_HEADS
                                * (TL_KV_DIM + WORKSPACE_PADDING)
                                + head_indices[:, None] * TL_KV_DIM
                                + kv_columns[None, :],
                                weighted_value_sum,
                            )
