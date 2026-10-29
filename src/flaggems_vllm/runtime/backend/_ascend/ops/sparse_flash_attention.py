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

"""Shared-KV MLA with the vLLM-Ascend SFA calling convention.

The small-batch path uses head/Split-KV parallelism. The multi-batch path uses
128-head Cube tiles and 16-head Vector tiles with explicit 1:2 C/V scheduling.
Schedules use fixed Ascend configurations; no NVIDIA autotune is applied.
"""

import math

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.language.extra.cann.extension import sort as ascend_sort

from flaggems_vllm.runtime.backend._ascend.device import get_device_core_counts

INT32_MAX = 2**31 - 1
# Triton requires constexpr globals; keep host comparisons as plain Python ints.
TRITON_INT32_MAX = tl.constexpr(INT32_MAX)

# Input contracts and numeric bounds.
INT64_MAX = 2**63 - 1
KV_DIM = 512
ROPE_DIM = 64
FAST_QUERY_HEADS = 128
FAST_SPARSE_ENTRIES = 2048
FAST_PAGE_SIZE = 128
FAST_PAGE_OFFSET_MASK = FAST_PAGE_SIZE - 1
FAST_PAGE_SIZE_LOG2 = FAST_PAGE_SIZE.bit_length() - 1
TND_TENSOR_RANK = 3
BATCHED_TENSOR_RANK = 4
SUPPORTED_QUERY_HEADS = (1, 2, 4, 8, 16, 32, 64, 128)
SUPPORTED_SPARSE_BLOCK_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
MAX_SUPPORTED_PAGE_SIZE = 1024
PAGE_SIZE_ALIGNMENT = 16
MLA_ABSORB_ATTENTION_MODE = 2
SPARSE_MODE_ALL = 0
SPARSE_MODE_CAUSAL = 3
INVALID_SPARSE_INDEX = -1
# Preserve the installed vLLM template's finite initial/empty-row maximum.
VLLM_FINITE_SOFTMAX_SENTINEL = -2.0e38
VLLM_FINITE_MASK_MAX_BLOCK_SIZE = 4
# Numeric float32 sorting must preserve every integer physical row index.
FP32_EXACT_INT_LIMIT = 2**24
FP32_ELEMENT_BYTES = 4
FP16_BF16_ELEMENT_BYTES = 2
# Retain the existing 2-GiB cache-span guard; the tested path uses 2-byte elements.
MAX_CACHE_BYTES_FOR_INT32_OFFSETS = 2**31
MAX_CACHE_ELEMENTS_FOR_INT32_OFFSETS = (
    MAX_CACHE_BYTES_FOR_INT32_OFFSETS // FP16_BF16_ELEMENT_BYTES
)
CACHE_SPEC_INT64_OFFSETS_INDEX = 4

# Pipeline state layout and producer/consumer event ranges.
VECTOR_SUBCORES_PER_CUBE = 2
NUM_PIPELINE_SLOTS = 3
OUTPUT_STAGE_LAG = NUM_PIPELINE_SLOTS - 1
# State rows 0/1 hold max/sum; subsequent rows hold alpha for each ring slot.
SOFTMAX_ALPHA_STATE_BASE = 2
SOFTMAX_STATE_ROWS = SOFTMAX_ALPHA_STATE_BASE + NUM_PIPELINE_SLOTS
PV_SYNC_EVENT_BASE = NUM_PIPELINE_SLOTS
COMBINE_TWO_SPLITS = 2

# Ascend schedules coordinate Cube and Vector stages explicitly. These fixed
# schedules are validated on 910B; they do not modify NVIDIA autotune configs.
SMALL_QUERY_MAX_TOKENS = 16
SMALL_QUERY_HEAD_BLOCK = 16
SMALL_QUERY_KV_BLOCK = 256
SMALL_QUERY_MIN_SPARSE_ENTRIES = 256
SMALL_QUERY_MAX_SPARSE_ENTRIES = 4096
SMALL_PIPELINE_VECTOR_HEADS = 8

# Existing fixed scheduling choices; their optimality is not established here.
# The original commit couples the >=16-token choices; no threshold sweep
# is recorded. Preserve the existing schedule instead of claiming a hardware rule.
LARGE_QUERY_BATCH_THRESHOLD = 16
LARGE_BATCH_WORKSPACE_PADDING = 1
SINGLE_QUERY_KV_SPLITS = 4
PIPELINE_FOUR_QUERY_SPLITS = 4
PIPELINE_EIGHT_QUERY_SPLITS = 2
GENERAL_HEAD_BLOCK_SIZE = 16
GENERAL_WIDE_OFFSET_KV_ROWS = 4
GENERAL_STRIDED_KV_ROWS = 8
GENERAL_CONTIGUOUS_KV_ROWS = 32
GATHERED_HEADS_PER_GROUP = 32
GATHERED_HEAD_GROUPS = FAST_QUERY_HEADS // GATHERED_HEADS_PER_GROUP
GATHERED_KV_BLOCK_ROWS = 256
PIPELINE_KV_BLOCK_ROWS = 512
WIDE_OFFSET_GATHER_ROWS = 4
STRIDED_GATHER_ROWS = 16
MASKED_GATHER_ROWS = 16
CONTIGUOUS_GATHER_ROWS = 32
SMALL_BATCH_GATHER_ROWS = 64
LARGE_BATCH_QK_COLUMNS = 256
SMALL_BATCH_QK_COLUMNS = 128
PIPELINE_SOFTMAX_HEADS = 16
PIPELINE_VALUE_COLUMNS = 256
LARGE_BATCH_OUTPUT_COLUMNS = 256
SMALL_BATCH_OUTPUT_COLUMNS = KV_DIM
LARGE_BATCH_OUTPUT_HEADS = 32
SMALL_BATCH_OUTPUT_HEADS = 16
COMBINE_HEADS_PER_PROGRAM = 16
COMBINE_PROGRAMS_PER_QUERY = FAST_QUERY_HEADS // COMBINE_HEADS_PER_PROGRAM
COMBINE_TWO_HEADS_PER_PROGRAM = 32
COMBINE_TWO_PROGRAMS_PER_QUERY = FAST_QUERY_HEADS // COMBINE_TWO_HEADS_PER_PROGRAM
COMBINE_TWO_VALUE_COLUMNS = 256
COMBINE_TWO_VALUE_TILES = KV_DIM // COMBINE_TWO_VALUE_COLUMNS
SORT_MIN_LOGICAL_SPAN = 4096
STRIDED_COPY_BLOCK_ELEMENTS = 1024

# Scratch start-address offsets modulo a period; not buffer sizes or alignment guarantees.
# Hardware motivation for this address pattern remains unverified.
SCRATCH_ADDRESS_PERIOD_BYTES = 64 * 1024
SCRATCH_ADDRESS_STEP_BYTES = 8 * 1024
KV_WORKSPACE_ADDRESS_OFFSET_BYTES = SCRATCH_ADDRESS_STEP_BYTES
ROPE_WORKSPACE_ADDRESS_OFFSET_BYTES = 2 * SCRATCH_ADDRESS_STEP_BYTES
SCORE_WORKSPACE_ADDRESS_OFFSET_BYTES = 3 * SCRATCH_ADDRESS_STEP_BYTES
WEIGHTS_WORKSPACE_ADDRESS_OFFSET_BYTES = 4 * SCRATCH_ADDRESS_STEP_BYTES
PV_WORKSPACE_ADDRESS_OFFSET_BYTES = 5 * SCRATCH_ADDRESS_STEP_BYTES
SOFTMAX_WORKSPACE_ADDRESS_OFFSET_BYTES = 6 * SCRATCH_ADDRESS_STEP_BYTES
OUTPUT_WORKSPACE_ADDRESS_OFFSET_BYTES = 7 * SCRATCH_ADDRESS_STEP_BYTES

# TL_* are Triton constexpr views; host comparisons and shapes use Python values.
TL_KV_DIM = tl.constexpr(KV_DIM)
TL_ROPE_DIM = tl.constexpr(ROPE_DIM)
TL_FAST_QUERY_HEADS = tl.constexpr(FAST_QUERY_HEADS)
TL_FAST_SPARSE_ENTRIES = tl.constexpr(FAST_SPARSE_ENTRIES)
TL_FAST_PAGE_SIZE = tl.constexpr(FAST_PAGE_SIZE)
TL_FAST_PAGE_OFFSET_MASK = tl.constexpr(FAST_PAGE_OFFSET_MASK)
TL_FAST_PAGE_SIZE_LOG2 = tl.constexpr(FAST_PAGE_SIZE_LOG2)
TL_INVALID_SPARSE_INDEX = tl.constexpr(INVALID_SPARSE_INDEX)
TL_VLLM_FINITE_SOFTMAX_SENTINEL = tl.constexpr(VLLM_FINITE_SOFTMAX_SENTINEL)
TL_VLLM_FINITE_MASK_MAX_BLOCK_SIZE = tl.constexpr(VLLM_FINITE_MASK_MAX_BLOCK_SIZE)
TL_CACHE_SPEC_INT64_OFFSETS_INDEX = tl.constexpr(CACHE_SPEC_INT64_OFFSETS_INDEX)
TL_VECTOR_SUBCORES_PER_CUBE = tl.constexpr(VECTOR_SUBCORES_PER_CUBE)
TL_NUM_PIPELINE_SLOTS = tl.constexpr(NUM_PIPELINE_SLOTS)
TL_OUTPUT_STAGE_LAG = tl.constexpr(OUTPUT_STAGE_LAG)
TL_SOFTMAX_ALPHA_STATE_BASE = tl.constexpr(SOFTMAX_ALPHA_STATE_BASE)
TL_SOFTMAX_STATE_ROWS = tl.constexpr(SOFTMAX_STATE_ROWS)
TL_PV_SYNC_EVENT_BASE = tl.constexpr(PV_SYNC_EVENT_BASE)
TL_COMBINE_TWO_SPLITS = tl.constexpr(COMBINE_TWO_SPLITS)
TL_GATHERED_HEADS_PER_GROUP = tl.constexpr(GATHERED_HEADS_PER_GROUP)
TL_GATHERED_KV_BLOCK_ROWS = tl.constexpr(GATHERED_KV_BLOCK_ROWS)
TL_WIDE_OFFSET_GATHER_ROWS = tl.constexpr(WIDE_OFFSET_GATHER_ROWS)
TL_MASKED_GATHER_ROWS = tl.constexpr(MASKED_GATHER_ROWS)
TL_COMBINE_HEADS_PER_PROGRAM = tl.constexpr(COMBINE_HEADS_PER_PROGRAM)
TL_COMBINE_PROGRAMS_PER_QUERY = tl.constexpr(COMBINE_PROGRAMS_PER_QUERY)
TL_COMBINE_TWO_HEADS_PER_PROGRAM = tl.constexpr(COMBINE_TWO_HEADS_PER_PROGRAM)
TL_COMBINE_TWO_PROGRAMS_PER_QUERY = tl.constexpr(COMBINE_TWO_PROGRAMS_PER_QUERY)
TL_COMBINE_TWO_VALUE_COLUMNS = tl.constexpr(COMBINE_TWO_VALUE_COLUMNS)
TL_COMBINE_TWO_VALUE_TILES = tl.constexpr(COMBINE_TWO_VALUE_TILES)
TL_SORT_MIN_LOGICAL_SPAN = tl.constexpr(SORT_MIN_LOGICAL_SPAN)

ASCEND_PIPE = tle.dsa.ascend.PIPE


def _get_core_num(device):
    """Return physical cores for the input device; this schedule needs 1:2 C/V."""
    num_cube_cores, num_vector_cores = get_device_core_counts(device)
    if num_vector_cores != VECTOR_SUBCORES_PER_CUBE * num_cube_cores:
        raise NotImplementedError(
            "sparse_flash_attention requires 1:2 Cube/Vector cores"
        )
    return num_cube_cores, num_vector_cores


@triton.jit
def _sfa_query_meta(
    query_token_idx,
    QUERY_LENGTHS,
    KV_LENGTHS,
    NUM_REQUESTS: tl.constexpr,
    QUERY_SEQ_CAPACITY: tl.constexpr,
    KV_SEQ_CAPACITY: tl.constexpr,
    QUERY_IS_TND: tl.constexpr,
    KV_IS_TND: tl.constexpr,
    HAS_QUERY_LENGTHS: tl.constexpr,
    HAS_KV_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # Equal token/request counts do not imply one query per request:
    # individual query lengths can include zero and two.
    if QUERY_IS_TND:
        request_indices = tl.arange(0, triton.next_power_of_2(NUM_REQUESTS))
        query_end_offsets = tl.load(
            QUERY_LENGTHS + request_indices,
            request_indices < NUM_REQUESTS,
            other=TRITON_INT32_MAX,
        )
        one_query_per_request = (
            tl.sum(
                (
                    (query_end_offsets == request_indices + 1)
                    & (request_indices < NUM_REQUESTS)
                ).to(tl.int32),
                0,
            )
            == NUM_REQUESTS
        )
        if one_query_per_request & (query_token_idx < NUM_REQUESTS):
            request_idx = query_token_idx.to(tl.int32)
            query_length = tl.full((), 1, tl.int32)
            query_position = tl.full((), 0, tl.int64)
        else:
            request_idx = tl.minimum(
                tl.sum((query_token_idx >= query_end_offsets).to(tl.int32), 0),
                NUM_REQUESTS - 1,
            )
            query_start_offset = tl.load(
                QUERY_LENGTHS + request_idx - 1, request_idx > 0, other=0
            )
            query_end_offset = tl.load(QUERY_LENGTHS + request_idx)
            query_length = query_end_offset - query_start_offset
            query_position = query_token_idx.to(tl.int64) - query_start_offset
    else:
        request_idx = query_token_idx // QUERY_SEQ_CAPACITY
        query_position = query_token_idx % QUERY_SEQ_CAPACITY
        query_length = QUERY_SEQ_CAPACITY
        if HAS_QUERY_LENGTHS:
            query_length = tl.load(QUERY_LENGTHS + request_idx)
    query_position_valid = (query_position >= 0) & (query_position < query_length)
    kv_start_offset = 0
    kv_length = KV_SEQ_CAPACITY
    if HAS_KV_LENGTHS:
        kv_length = tl.load(KV_LENGTHS + request_idx)
        if KV_IS_TND:
            kv_start_offset = tl.load(
                KV_LENGTHS + request_idx - 1, request_idx > 0, other=0
            )
            kv_length -= kv_start_offset
    kv_end_exclusive = kv_length
    if CAUSAL:
        kv_end_exclusive = tl.minimum(
            kv_length, kv_length - query_length + query_position + 1
        )
    kv_end_exclusive = tl.where(
        query_position_valid, tl.maximum(kv_end_exclusive, 0), 0
    )
    return request_idx.to(tl.int64), kv_end_exclusive, kv_start_offset.to(tl.int64)


@triton.jit
def _sfa_load_cache(
    CACHE,
    row_indices,
    column_indices,
    ROW_WIDTH: tl.constexpr,
    CACHE_ACCESS_SPEC: tl.constexpr,
    valid_row_mask=None,
):
    if CACHE_ACCESS_SPEC is None:
        if valid_row_mask is None:
            return tl.load(
                CACHE + row_indices[:, None] * ROW_WIDTH + column_indices[None, :]
            )
        else:
            return tl.load(
                CACHE + row_indices[:, None] * ROW_WIDTH + column_indices[None, :],
                valid_row_mask[:, None],
                other=0,
            )
    else:
        rows_per_group: tl.constexpr = CACHE_ACCESS_SPEC[0]
        group_stride_elems: tl.constexpr = CACHE_ACCESS_SPEC[1]
        row_stride_elems: tl.constexpr = CACHE_ACCESS_SPEC[2]
        column_stride_elems: tl.constexpr = CACHE_ACCESS_SPEC[3]
        requires_int64_element_offsets: tl.constexpr = CACHE_ACCESS_SPEC[
            TL_CACHE_SPEC_INT64_OFFSETS_INDEX
        ]
        address_row_indices = row_indices.to(
            tl.int64 if requires_int64_element_offsets else tl.int32
        )
        row_base_offsets = (
            address_row_indices // rows_per_group
        ) * group_stride_elems + (
            address_row_indices % rows_per_group
        ) * row_stride_elems
        element_offsets = (
            row_base_offsets[:, None]
            + column_indices[None, :].to(
                tl.int64 if requires_int64_element_offsets else tl.int32
            )
            * column_stride_elems
        )
        flat_element_offsets = tl.reshape(
            element_offsets, (row_indices.shape[0] * column_indices.shape[0],)
        )
        if valid_row_mask is None:
            cache_values = tl.load(CACHE + flat_element_offsets)
        else:
            flat_valid_mask = tl.reshape(
                tl.broadcast_to(valid_row_mask[:, None], element_offsets.shape),
                flat_element_offsets.shape,
            )
            cache_values = tl.load(
                CACHE + flat_element_offsets, flat_valid_mask, other=0
            )
        return tl.reshape(cache_values, element_offsets.shape)


@triton.jit
def _sfa_load_cache_rowblock(
    CACHE,
    row_indices,
    column_indices,
    ROW_WIDTH: tl.constexpr,
    CACHE_ACCESS_SPEC: tl.constexpr,
    valid_row_mask=None,
):
    if CACHE_ACCESS_SPEC is None:
        if valid_row_mask is None:
            return tl.load(
                CACHE + row_indices[:, None] * ROW_WIDTH + column_indices[None, :]
            )
        else:
            return tl.load(
                CACHE + row_indices[:, None] * ROW_WIDTH + column_indices[None, :],
                valid_row_mask[:, None],
                other=0,
            )
    else:
        rows_per_group: tl.constexpr = CACHE_ACCESS_SPEC[0]
        group_stride_elems: tl.constexpr = CACHE_ACCESS_SPEC[1]
        row_stride_elems: tl.constexpr = CACHE_ACCESS_SPEC[2]
        column_stride_elems: tl.constexpr = CACHE_ACCESS_SPEC[3]
        requires_int64_element_offsets: tl.constexpr = CACHE_ACCESS_SPEC[
            TL_CACHE_SPEC_INT64_OFFSETS_INDEX
        ]
        address_row_indices = row_indices.to(
            tl.int64 if requires_int64_element_offsets else tl.int32
        )
        row_base_offsets = (
            address_row_indices // rows_per_group
        ) * group_stride_elems + (
            address_row_indices % rows_per_group
        ) * row_stride_elems
        if column_stride_elems == 0:
            if valid_row_mask is None:
                cache_values = tl.load(CACHE + row_base_offsets)
            else:
                cache_values = tl.load(
                    CACHE + row_base_offsets, valid_row_mask, other=0
                )
            return tl.broadcast_to(
                cache_values[:, None], (row_indices.shape[0], ROW_WIDTH)
            )
        else:
            row_positions = tl.arange(0, row_indices.shape[0])
            cache_tile = tl.full(
                (row_indices.shape[0], ROW_WIDTH), 0, CACHE.dtype.element_ty
            )
            for row_position in range(row_indices.shape[0]):
                row_base_offset = tl.sum(
                    tl.where(row_positions == row_position, row_base_offsets, 0), 0
                )
                row_is_valid = True
                if valid_row_mask is not None:
                    row_is_valid = (
                        tl.sum(
                            ((row_positions == row_position) & valid_row_mask).to(
                                tl.int32
                            ),
                            0,
                        )
                        > 0
                    )
                if row_is_valid:
                    cache_row_ptr = tl.make_block_ptr(
                        CACHE + row_base_offset,
                        (ROW_WIDTH,),
                        (column_stride_elems,),
                        (0,),
                        (ROW_WIDTH,),
                        (0,),
                    )
                    cache_values = tl.load(cache_row_ptr)
                    cache_tile = tl.where(
                        row_positions[:, None] == row_position,
                        cache_values[None, :],
                        cache_tile,
                    )
            return cache_tile


@triton.jit
def _sfa_general(
    QUERY,
    QUERY_ROPE,
    KV,
    KEY_ROPE,
    SPARSE_INDICES,
    BLOCK_TABLE,
    QUERY_LENGTHS,
    KV_LENGTHS,
    OUT,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    NUM_REQUESTS: tl.constexpr,
    NUM_QUERY_TOKENS: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    QUERY_SEQ_CAPACITY: tl.constexpr,
    KV_SEQ_CAPACITY: tl.constexpr,
    NUM_SPARSE_BLOCKS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGES_PER_REQUEST: tl.constexpr,
    SPARSE_BLOCK_SIZE: tl.constexpr,
    QUERY_IS_TND: tl.constexpr,
    KV_IS_TND: tl.constexpr,
    PAGED: tl.constexpr,
    HAS_QUERY_LENGTHS: tl.constexpr,
    HAS_KV_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    RETURN_SOFTMAX_STATS: tl.constexpr,
    HEAD_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_ROWS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    KV_ACCESS_SPEC: tl.constexpr = None,
    KEY_ROPE_ACCESS_SPEC: tl.constexpr = None,
):
    program_id = tl.program_id(0)
    for task_idx in range(
        program_id,
        NUM_QUERY_TOKENS * triton.cdiv(NUM_QUERY_HEADS, HEAD_BLOCK_SIZE),
        NUM_CORES,
    ):
        query_token_idx = (
            task_idx // triton.cdiv(NUM_QUERY_HEADS, HEAD_BLOCK_SIZE)
        ).to(tl.int64)
        head_indices = task_idx % triton.cdiv(
            NUM_QUERY_HEADS, HEAD_BLOCK_SIZE
        ) * HEAD_BLOCK_SIZE + tl.arange(0, HEAD_BLOCK_SIZE)
        kv_columns = tl.arange(0, TL_KV_DIM)
        rope_columns = tl.arange(0, TL_ROPE_DIM)
        request_idx, kv_end_exclusive, kv_start_offset = _sfa_query_meta(
            query_token_idx,
            QUERY_LENGTHS,
            KV_LENGTHS,
            NUM_REQUESTS,
            QUERY_SEQ_CAPACITY,
            KV_SEQ_CAPACITY,
            QUERY_IS_TND,
            KV_IS_TND,
            HAS_QUERY_LENGTHS,
            HAS_KV_LENGTHS,
            CAUSAL,
        )
        if not KV_IS_TND and not PAGED:
            kv_start_offset = request_idx * KV_SEQ_CAPACITY
        query_tile = tl.load(
            QUERY
            + (query_token_idx * NUM_QUERY_HEADS + head_indices[:, None]) * TL_KV_DIM
            + kv_columns[None, :],
            head_indices[:, None] < NUM_QUERY_HEADS,
            other=0,
        )
        query_rope_tile = tl.load(
            QUERY_ROPE
            + (query_token_idx * NUM_QUERY_HEADS + head_indices[:, None]) * TL_ROPE_DIM
            + rope_columns[None, :],
            head_indices[:, None] < NUM_QUERY_HEADS,
            other=0,
        )
        softmax_max = tl.full((HEAD_BLOCK_SIZE,), float("-inf"), tl.float32)
        softmax_sum = tl.zeros((HEAD_BLOCK_SIZE,), tl.float32)
        weighted_value_sum = tl.zeros((HEAD_BLOCK_SIZE, TL_KV_DIM), tl.float32)
        for kv_block_idx in range(
            triton.cdiv(NUM_SPARSE_BLOCKS * SPARSE_BLOCK_SIZE, KV_BLOCK_ROWS)
        ):
            sparse_positions = kv_block_idx * KV_BLOCK_ROWS + tl.arange(
                0, KV_BLOCK_ROWS
            )
            sparse_entry_positions = sparse_positions // SPARSE_BLOCK_SIZE
            logical_sparse_block_indices = tl.load(
                SPARSE_INDICES
                + query_token_idx * NUM_SPARSE_BLOCKS
                + sparse_entry_positions,
                sparse_entry_positions < NUM_SPARSE_BLOCKS,
                other=TL_INVALID_SPARSE_INDEX,
            )
            logical_kv_indices = (
                logical_sparse_block_indices * SPARSE_BLOCK_SIZE
                + sparse_positions % SPARSE_BLOCK_SIZE
            )
            valid_kv_mask = (
                (sparse_entry_positions < NUM_SPARSE_BLOCKS)
                & (
                    sparse_entry_positions
                    < tl.cdiv(kv_end_exclusive, SPARSE_BLOCK_SIZE)
                )
                & (logical_sparse_block_indices >= 0)
            )
            valid_kv_mask &= (logical_kv_indices >= 0) & (
                logical_kv_indices < kv_end_exclusive
            )
            if PAGED:
                valid_kv_mask &= logical_kv_indices // PAGE_SIZE < PAGES_PER_REQUEST
                physical_page_ids = tl.load(
                    BLOCK_TABLE
                    + request_idx * PAGES_PER_REQUEST
                    + logical_kv_indices // PAGE_SIZE,
                    valid_kv_mask,
                    other=0,
                )
                cache_row_indices = (
                    physical_page_ids.to(tl.int64) * PAGE_SIZE
                    + logical_kv_indices % PAGE_SIZE
                )
            else:
                cache_row_indices = kv_start_offset + logical_kv_indices.to(tl.int64)
            kv_tile = _sfa_load_cache(
                KV,
                cache_row_indices,
                kv_columns,
                TL_KV_DIM,
                KV_ACCESS_SPEC,
                valid_kv_mask,
            )
            key_rope_tile = _sfa_load_cache(
                KEY_ROPE,
                cache_row_indices,
                rope_columns,
                TL_ROPE_DIM,
                KEY_ROPE_ACCESS_SPEC,
                valid_kv_mask,
            )
            attention_scores = tl.dot(query_tile, tl.trans(kv_tile))
            attention_scores = (
                tl.dot(query_rope_tile, tl.trans(key_rope_tile), attention_scores)
                * SOFTMAX_SCALE
            )
            attention_scores = tl.where(
                valid_kv_mask[None, :], attention_scores, float("-inf")
            )
            updated_softmax_max = tl.maximum(softmax_max, tl.max(attention_scores, 1))
            softmax_exp_base = tl.where(
                updated_softmax_max == float("-inf"), 0.0, updated_softmax_max
            )
            previous_state_scale = tl.where(
                softmax_sum > 0, tl.exp(softmax_max - softmax_exp_base), 0.0
            )
            unnormalized_weights = tl.exp(attention_scores - softmax_exp_base[:, None])
            softmax_sum = softmax_sum * previous_state_scale + tl.sum(
                unnormalized_weights, 1
            )
            weighted_value_sum = weighted_value_sum * previous_state_scale[
                :, None
            ] + tl.dot(unnormalized_weights.to(KV.dtype.element_ty), kv_tile)
            softmax_max = updated_softmax_max
        tl.store(
            OUT
            + (query_token_idx * NUM_QUERY_HEADS + head_indices[:, None]) * TL_KV_DIM
            + kv_columns[None, :],
            weighted_value_sum / tl.where(softmax_sum > 0, softmax_sum, 1.0)[:, None],
            head_indices[:, None] < NUM_QUERY_HEADS,
        )
        if RETURN_SOFTMAX_STATS:
            # The installed vLLM template for sparse blocks of up to four tokens
            # uses a finite mask. Preserve its all-masked max/sum convention.
            empty_row_softmax_max = 0.0
            empty_row_softmax_sum = 0.0
            if SPARSE_BLOCK_SIZE <= TL_VLLM_FINITE_MASK_MAX_BLOCK_SIZE:
                empty_row_softmax_max = tl.where(
                    kv_end_exclusive > 0, TL_VLLM_FINITE_SOFTMAX_SENTINEL, 0.0
                )
                empty_row_softmax_sum = tl.minimum(
                    NUM_SPARSE_BLOCKS * SPARSE_BLOCK_SIZE, kv_end_exclusive
                ).to(tl.float32)
            tl.store(
                SOFTMAX_MAX + query_token_idx * NUM_QUERY_HEADS + head_indices,
                tl.where(softmax_sum > 0, softmax_max, empty_row_softmax_max),
                head_indices < NUM_QUERY_HEADS,
            )
            tl.store(
                SOFTMAX_SUM + query_token_idx * NUM_QUERY_HEADS + head_indices,
                tl.where(softmax_sum > 0, softmax_sum, empty_row_softmax_sum),
                head_indices < NUM_QUERY_HEADS,
            )


@triton.jit
def _sfa_small_gather(
    KV,
    KEY_ROPE,
    SPARSE_INDICES,
    BLOCK_TABLE,
    GATHERED_KV,
    GATHERED_KEY_ROPE,
    ADDITIVE_ATTENTION_MASK,
    KV_END_EXCLUSIVE,
    QUERY_LENGTHS,
    KV_LENGTHS,
    NUM_SPARSE_BLOCKS: tl.constexpr,
    PADDED_SPARSE_BLOCKS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_QUERY_TOKENS: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    QUERY_SEQ_CAPACITY: tl.constexpr,
    QUERY_IS_TND: tl.constexpr,
    HAS_QUERY_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr,
    PAGES_PER_REQUEST: tl.constexpr,
    GATHER_BLOCK_ROWS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    KV_ACCESS_SPEC: tl.constexpr = None,
    KEY_ROPE_ACCESS_SPEC: tl.constexpr = None,
):
    MASKED_GATHER_BLOCK_ROWS: tl.constexpr = (
        TL_WIDE_OFFSET_GATHER_ROWS
        if GATHER_BLOCK_ROWS == TL_WIDE_OFFSET_GATHER_ROWS
        else TL_MASKED_GATHER_ROWS
    )
    program_id = tl.program_id(0)
    for token_idx in range(NUM_QUERY_TOKENS):
        query_token_idx = token_idx.to(tl.int64)
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
        page_table_indices = tl.arange(0, triton.next_power_of_2(PAGES_PER_REQUEST))
        request_page_table = tl.load(
            BLOCK_TABLE + request_idx * PAGES_PER_REQUEST + page_table_indices,
            mask=page_table_indices < PAGES_PER_REQUEST,
            other=0,
        )
        if program_id == 0:
            tl.store(KV_END_EXCLUSIVE + query_token_idx, kv_end_exclusive)
        first_gather_block = (
            program_id
            + NUM_CORES
            - (token_idx * (PADDED_SPARSE_BLOCKS // GATHER_BLOCK_ROWS)) % NUM_CORES
        ) % NUM_CORES
        for gather_block_idx in range(
            first_gather_block, PADDED_SPARSE_BLOCKS // GATHER_BLOCK_ROWS, NUM_CORES
        ):
            sparse_positions = gather_block_idx * GATHER_BLOCK_ROWS + tl.arange(
                0, GATHER_BLOCK_ROWS
            )
            logical_kv_indices = tl.load(
                SPARSE_INDICES + query_token_idx * NUM_SPARSE_BLOCKS + sparse_positions,
                sparse_positions < NUM_SPARSE_BLOCKS,
                other=-1,
            )
            valid_kv_mask = (
                (logical_kv_indices >= 0)
                & (logical_kv_indices < kv_end_exclusive)
                & (sparse_positions < kv_end_exclusive)
            )
            valid_kv_mask &= logical_kv_indices // PAGE_SIZE < PAGES_PER_REQUEST
            tl.store(
                ADDITIVE_ATTENTION_MASK
                + query_token_idx * PADDED_SPARSE_BLOCKS
                + sparse_positions,
                tl.where(valid_kv_mask, 0.0, float("-inf")),
            )
            logical_kv_indices = tl.where(valid_kv_mask, logical_kv_indices, 0)
            # Preserve int32 page-ID bits through float32 gather; this is
            # not a numeric cast. The original lowering rationale is unverified.
            physical_page_ids = tl.gather(
                request_page_table.to(tl.float32, bitcast=True),
                logical_kv_indices // PAGE_SIZE,
                0,
            ).to(tl.int32, bitcast=True)
            physical_row_indices = (
                physical_page_ids.to(tl.int64) * PAGE_SIZE
                + logical_kv_indices % PAGE_SIZE
            )
            kv_columns = tl.arange(0, TL_KV_DIM)
            rope_columns = tl.arange(0, TL_ROPE_DIM)
            if tl.sum(valid_kv_mask.to(tl.int32), 0) == GATHER_BLOCK_ROWS:
                kv_tile = _sfa_load_cache_rowblock(
                    KV, physical_row_indices, kv_columns, TL_KV_DIM, KV_ACCESS_SPEC
                )
                key_rope_tile = _sfa_load_cache_rowblock(
                    KEY_ROPE,
                    physical_row_indices,
                    rope_columns,
                    TL_ROPE_DIM,
                    KEY_ROPE_ACCESS_SPEC,
                )
                tl.store(
                    GATHERED_KV
                    + (
                        query_token_idx * PADDED_SPARSE_BLOCKS
                        + sparse_positions[:, None]
                    )
                    * TL_KV_DIM
                    + kv_columns[None, :],
                    kv_tile,
                )
                tl.store(
                    GATHERED_KEY_ROPE
                    + (
                        query_token_idx * PADDED_SPARSE_BLOCKS
                        + sparse_positions[:, None]
                    )
                    * TL_ROPE_DIM
                    + rope_columns[None, :],
                    key_rope_tile,
                )

            else:
                for subblock_idx in range(
                    GATHER_BLOCK_ROWS // MASKED_GATHER_BLOCK_ROWS
                ):
                    subblock_sparse_positions = (
                        gather_block_idx * GATHER_BLOCK_ROWS
                        + subblock_idx * MASKED_GATHER_BLOCK_ROWS
                        + tl.arange(0, MASKED_GATHER_BLOCK_ROWS)
                    )
                    subblock_logical_kv_indices = tl.load(
                        SPARSE_INDICES
                        + query_token_idx * NUM_SPARSE_BLOCKS
                        + subblock_sparse_positions,
                        subblock_sparse_positions < NUM_SPARSE_BLOCKS,
                        other=-1,
                    )
                    subblock_valid_kv_mask = (
                        (subblock_logical_kv_indices >= 0)
                        & (subblock_logical_kv_indices < kv_end_exclusive)
                        & (subblock_sparse_positions < kv_end_exclusive)
                    )
                    subblock_valid_kv_mask &= (
                        subblock_logical_kv_indices // PAGE_SIZE < PAGES_PER_REQUEST
                    )
                    subblock_logical_kv_indices = tl.where(
                        subblock_valid_kv_mask, subblock_logical_kv_indices, 0
                    )
                    # Preserve int32 page-ID bits through float32 gather; this is
                    # not a numeric cast. The original lowering rationale is unverified.
                    subblock_physical_page_ids = tl.gather(
                        request_page_table.to(tl.float32, bitcast=True),
                        subblock_logical_kv_indices // PAGE_SIZE,
                        0,
                    ).to(tl.int32, bitcast=True)
                    subblock_physical_row_indices = (
                        subblock_physical_page_ids.to(tl.int64) * PAGE_SIZE
                        + subblock_logical_kv_indices % PAGE_SIZE
                    )
                    subblock_kv_tile = _sfa_load_cache_rowblock(
                        KV,
                        subblock_physical_row_indices,
                        kv_columns,
                        TL_KV_DIM,
                        KV_ACCESS_SPEC,
                        subblock_valid_kv_mask,
                    )
                    tl.store(
                        GATHERED_KV
                        + (
                            query_token_idx * PADDED_SPARSE_BLOCKS
                            + subblock_sparse_positions[:, None]
                        )
                        * TL_KV_DIM
                        + kv_columns[None, :],
                        subblock_kv_tile,
                    )
                    subblock_key_rope_tile = _sfa_load_cache_rowblock(
                        KEY_ROPE,
                        subblock_physical_row_indices,
                        rope_columns,
                        TL_ROPE_DIM,
                        KEY_ROPE_ACCESS_SPEC,
                        subblock_valid_kv_mask,
                    )
                    tl.store(
                        GATHERED_KEY_ROPE
                        + (
                            query_token_idx * PADDED_SPARSE_BLOCKS
                            + subblock_sparse_positions[:, None]
                        )
                        * TL_ROPE_DIM
                        + rope_columns[None, :],
                        subblock_key_rope_tile,
                    )


@triton.jit
def _sfa_small_attention(
    QUERY,
    QUERY_ROPE,
    GATHERED_KV,
    GATHERED_KEY_ROPE,
    MASK,
    PARTIAL_OUT,
    PARTIAL_MAX,
    PARTIAL_SUM,
    NUM_QUERY_TOKENS: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    PADDED_SPARSE_BLOCKS: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
):
    HEAD_GROUPS: tl.constexpr = triton.cdiv(NUM_QUERY_HEADS, HEAD_BLOCK)
    NUM_KV_BLOCKS: tl.constexpr = PADDED_SPARSE_BLOCKS // KV_BLOCK
    for task in range(
        tl.program_id(0), NUM_QUERY_TOKENS * HEAD_GROUPS * NUM_SPLITS, NUM_CORES
    ):
        split = task % NUM_SPLITS
        hg = (task // NUM_SPLITS) % HEAD_GROUPS
        token = (task // (NUM_SPLITS * HEAD_GROUPS)).to(tl.int64)
        heads = hg * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
        cols = tl.arange(0, TL_KV_DIM)
        q = tl.load(
            QUERY
            + (token * NUM_QUERY_HEADS + heads[:, None]) * TL_KV_DIM
            + cols[None, :],
            heads[:, None] < NUM_QUERY_HEADS,
            other=0,
        )
        qr = tl.load(
            QUERY_ROPE
            + (token * NUM_QUERY_HEADS + heads[:, None]) * TL_ROPE_DIM
            + tl.arange(0, TL_ROPE_DIM)[None, :],
            heads[:, None] < NUM_QUERY_HEADS,
            other=0,
        )
        sm = tl.full((HEAD_BLOCK,), TL_VLLM_FINITE_SOFTMAX_SENTINEL, tl.float32)
        ss = tl.zeros((HEAD_BLOCK,), tl.float32)
        acc = tl.zeros((HEAD_BLOCK, TL_KV_DIM), tl.float32)
        for block in range(split, NUM_KV_BLOCKS, NUM_SPLITS):
            k = tl.load(
                tl.make_block_ptr(
                    GATHERED_KV + token * PADDED_SPARSE_BLOCKS * TL_KV_DIM,
                    (PADDED_SPARSE_BLOCKS, TL_KV_DIM),
                    (TL_KV_DIM, 1),
                    (block * KV_BLOCK, 0),
                    (KV_BLOCK, TL_KV_DIM),
                    (1, 0),
                )
            )
            kr = tl.load(
                tl.make_block_ptr(
                    GATHERED_KEY_ROPE + token * PADDED_SPARSE_BLOCKS * TL_ROPE_DIM,
                    (PADDED_SPARSE_BLOCKS, TL_ROPE_DIM),
                    (TL_ROPE_DIM, 1),
                    (block * KV_BLOCK, 0),
                    (KV_BLOCK, TL_ROPE_DIM),
                    (1, 0),
                )
            )
            score = tl.dot(q, tl.trans(k))
            score = tl.dot(qr, tl.trans(kr), score) * SOFTMAX_SCALE
            valid = (
                tl.load(
                    MASK
                    + token * PADDED_SPARSE_BLOCKS
                    + block * KV_BLOCK
                    + tl.arange(0, KV_BLOCK)
                )
                == 0
            )
            score = tl.where(valid[None, :], score, float("-inf"))
            sm_next = tl.maximum(sm, tl.max(score, 1))
            alpha = tl.exp(sm - sm_next)
            p = tl.exp(score - sm_next[:, None])
            ss = ss * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(GATHERED_KV.dtype.element_ty), k)
            sm = sm_next
        partial_row = (token * NUM_SPLITS + split) * NUM_QUERY_HEADS + heads
        tl.store(
            PARTIAL_OUT + partial_row[:, None] * TL_KV_DIM + cols[None, :],
            acc,
            heads[:, None] < NUM_QUERY_HEADS,
        )
        tl.store(PARTIAL_MAX + partial_row, sm, heads < NUM_QUERY_HEADS)
        tl.store(PARTIAL_SUM + partial_row, ss, heads < NUM_QUERY_HEADS)


@triton.jit
def _sfa_small_combine(
    PARTIAL_OUT,
    PARTIAL_MAX,
    PARTIAL_SUM,
    OUT,
    MAX_OUT,
    SUM_OUT,
    KV_END,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPARSE_BLOCKS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    RETURN_STATS: tl.constexpr,
):
    GROUPS: tl.constexpr = triton.cdiv(NUM_QUERY_HEADS, HEAD_BLOCK)
    token = tl.program_id(0) // GROUPS
    hg = tl.program_id(0) % GROUPS
    heads = hg * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    cols = tl.arange(0, TL_KV_DIM)
    sm = tl.full((HEAD_BLOCK,), TL_VLLM_FINITE_SOFTMAX_SENTINEL, tl.float32)
    for split in range(NUM_SPLITS):
        row = (token * NUM_SPLITS + split) * NUM_QUERY_HEADS + heads
        part_max = tl.load(
            PARTIAL_MAX + row,
            heads < NUM_QUERY_HEADS,
            other=TL_VLLM_FINITE_SOFTMAX_SENTINEL,
        )
        sm = tl.maximum(sm, part_max)
    acc = tl.zeros((HEAD_BLOCK, TL_KV_DIM), tl.float32)
    ss = tl.zeros((HEAD_BLOCK,), tl.float32)
    for split in range(NUM_SPLITS):
        row = (token * NUM_SPLITS + split) * NUM_QUERY_HEADS + heads
        part_max = tl.load(
            PARTIAL_MAX + row,
            heads < NUM_QUERY_HEADS,
            other=TL_VLLM_FINITE_SOFTMAX_SENTINEL,
        )
        part_sum = tl.load(PARTIAL_SUM + row, heads < NUM_QUERY_HEADS, other=0)
        part_out = tl.load(
            PARTIAL_OUT + row[:, None] * TL_KV_DIM + cols[None, :],
            heads[:, None] < NUM_QUERY_HEADS,
            other=0,
        )
        alpha = tl.exp(part_max - sm)
        ss += part_sum * alpha
        acc += part_out * alpha[:, None]
    tl.store(
        OUT + (token * NUM_QUERY_HEADS + heads[:, None]) * TL_KV_DIM + cols[None, :],
        acc / tl.where(ss > 0, ss, 1.0)[:, None],
        heads[:, None] < NUM_QUERY_HEADS,
    )
    if RETURN_STATS:
        kv_end = tl.load(KV_END + token)
        tl.store(
            MAX_OUT + token * NUM_QUERY_HEADS + heads,
            tl.where(
                ss > 0, sm, tl.where(kv_end > 0, TL_VLLM_FINITE_SOFTMAX_SENTINEL, 0.0)
            ),
            heads < NUM_QUERY_HEADS,
        )
        tl.store(
            SUM_OUT + token * NUM_QUERY_HEADS + heads,
            tl.where(ss > 0, ss, tl.minimum(NUM_SPARSE_BLOCKS, kv_end).to(tl.float32)),
            heads < NUM_QUERY_HEADS,
        )


@triton.jit
def _sfa_gather_sparse_kv(
    KV,
    KEY_ROPE,
    SPARSE_INDICES,
    BLOCK_TABLE,
    GATHERED_KV,
    GATHERED_KEY_ROPE,
    ADDITIVE_ATTENTION_MASK,
    KV_END_EXCLUSIVE,
    QUERY_LENGTHS,
    KV_LENGTHS,
    NUM_QUERY_TOKENS: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    QUERY_SEQ_CAPACITY: tl.constexpr,
    QUERY_IS_TND: tl.constexpr,
    HAS_QUERY_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr,
    PAGES_PER_REQUEST: tl.constexpr,
    GATHER_BLOCK_ROWS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    KV_ACCESS_SPEC: tl.constexpr = None,
    KEY_ROPE_ACCESS_SPEC: tl.constexpr = None,
):
    MASKED_GATHER_BLOCK_ROWS: tl.constexpr = (
        TL_WIDE_OFFSET_GATHER_ROWS
        if GATHER_BLOCK_ROWS == TL_WIDE_OFFSET_GATHER_ROWS
        else TL_MASKED_GATHER_ROWS
    )
    program_id = tl.program_id(0)
    for token_idx in range(NUM_QUERY_TOKENS):
        query_token_idx = token_idx.to(tl.int64)
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
        page_table_indices = tl.arange(0, triton.next_power_of_2(PAGES_PER_REQUEST))
        request_page_table = tl.load(
            BLOCK_TABLE + request_idx * PAGES_PER_REQUEST + page_table_indices,
            mask=page_table_indices < PAGES_PER_REQUEST,
            other=0,
        )
        if program_id == 0:
            tl.store(KV_END_EXCLUSIVE + query_token_idx, kv_end_exclusive)
        first_gather_block = (
            program_id
            + NUM_CORES
            - (token_idx * (TL_FAST_SPARSE_ENTRIES // GATHER_BLOCK_ROWS)) % NUM_CORES
        ) % NUM_CORES
        for gather_block_idx in range(
            first_gather_block, TL_FAST_SPARSE_ENTRIES // GATHER_BLOCK_ROWS, NUM_CORES
        ):
            sparse_positions = gather_block_idx * GATHER_BLOCK_ROWS + tl.arange(
                0, GATHER_BLOCK_ROWS
            )
            logical_kv_indices = tl.load(
                SPARSE_INDICES
                + query_token_idx * TL_FAST_SPARSE_ENTRIES
                + sparse_positions
            )
            valid_kv_mask = (
                (logical_kv_indices >= 0)
                & (logical_kv_indices < kv_end_exclusive)
                & (sparse_positions < kv_end_exclusive)
            )
            valid_kv_mask &= logical_kv_indices // TL_FAST_PAGE_SIZE < PAGES_PER_REQUEST
            tl.store(
                ADDITIVE_ATTENTION_MASK
                + query_token_idx * TL_FAST_SPARSE_ENTRIES
                + sparse_positions,
                tl.where(valid_kv_mask, 0.0, float("-inf")),
            )
            logical_kv_indices = tl.where(valid_kv_mask, logical_kv_indices, 0)
            # Preserve int32 page-ID bits through float32 gather; this is
            # not a numeric cast. The original lowering rationale is unverified.
            physical_page_ids = tl.gather(
                request_page_table.to(tl.float32, bitcast=True),
                logical_kv_indices // TL_FAST_PAGE_SIZE,
                0,
            ).to(tl.int32, bitcast=True)
            physical_row_indices = (
                physical_page_ids.to(tl.int64) * TL_FAST_PAGE_SIZE
                + logical_kv_indices % TL_FAST_PAGE_SIZE
            )
            kv_columns = tl.arange(0, TL_KV_DIM)
            rope_columns = tl.arange(0, TL_ROPE_DIM)
            if tl.sum(valid_kv_mask.to(tl.int32), 0) == GATHER_BLOCK_ROWS:
                kv_tile = _sfa_load_cache_rowblock(
                    KV, physical_row_indices, kv_columns, TL_KV_DIM, KV_ACCESS_SPEC
                )
                key_rope_tile = _sfa_load_cache_rowblock(
                    KEY_ROPE,
                    physical_row_indices,
                    rope_columns,
                    TL_ROPE_DIM,
                    KEY_ROPE_ACCESS_SPEC,
                )
                tl.store(
                    GATHERED_KV
                    + (
                        query_token_idx * TL_FAST_SPARSE_ENTRIES
                        + sparse_positions[:, None]
                    )
                    * TL_KV_DIM
                    + kv_columns[None, :],
                    kv_tile,
                )
                tl.store(
                    GATHERED_KEY_ROPE
                    + (
                        query_token_idx * TL_FAST_SPARSE_ENTRIES
                        + sparse_positions[:, None]
                    )
                    * TL_ROPE_DIM
                    + rope_columns[None, :],
                    key_rope_tile,
                )

            else:
                for subblock_idx in range(
                    GATHER_BLOCK_ROWS // MASKED_GATHER_BLOCK_ROWS
                ):
                    subblock_sparse_positions = (
                        gather_block_idx * GATHER_BLOCK_ROWS
                        + subblock_idx * MASKED_GATHER_BLOCK_ROWS
                        + tl.arange(0, MASKED_GATHER_BLOCK_ROWS)
                    )
                    subblock_logical_kv_indices = tl.load(
                        SPARSE_INDICES
                        + query_token_idx * TL_FAST_SPARSE_ENTRIES
                        + subblock_sparse_positions
                    )
                    subblock_valid_kv_mask = (
                        (subblock_logical_kv_indices >= 0)
                        & (subblock_logical_kv_indices < kv_end_exclusive)
                        & (subblock_sparse_positions < kv_end_exclusive)
                    )
                    subblock_valid_kv_mask &= (
                        subblock_logical_kv_indices // TL_FAST_PAGE_SIZE
                        < PAGES_PER_REQUEST
                    )
                    subblock_logical_kv_indices = tl.where(
                        subblock_valid_kv_mask, subblock_logical_kv_indices, 0
                    )
                    # Preserve int32 page-ID bits through float32 gather; this is
                    # not a numeric cast. The original lowering rationale is unverified.
                    subblock_physical_page_ids = tl.gather(
                        request_page_table.to(tl.float32, bitcast=True),
                        subblock_logical_kv_indices // TL_FAST_PAGE_SIZE,
                        0,
                    ).to(tl.int32, bitcast=True)
                    subblock_physical_row_indices = (
                        subblock_physical_page_ids.to(tl.int64) * TL_FAST_PAGE_SIZE
                        + subblock_logical_kv_indices % TL_FAST_PAGE_SIZE
                    )
                    subblock_kv_tile = _sfa_load_cache_rowblock(
                        KV,
                        subblock_physical_row_indices,
                        kv_columns,
                        TL_KV_DIM,
                        KV_ACCESS_SPEC,
                        subblock_valid_kv_mask,
                    )
                    tl.store(
                        GATHERED_KV
                        + (
                            query_token_idx * TL_FAST_SPARSE_ENTRIES
                            + subblock_sparse_positions[:, None]
                        )
                        * TL_KV_DIM
                        + kv_columns[None, :],
                        subblock_kv_tile,
                    )
                    subblock_key_rope_tile = _sfa_load_cache_rowblock(
                        KEY_ROPE,
                        subblock_physical_row_indices,
                        rope_columns,
                        TL_ROPE_DIM,
                        KEY_ROPE_ACCESS_SPEC,
                        subblock_valid_kv_mask,
                    )
                    tl.store(
                        GATHERED_KEY_ROPE
                        + (
                            query_token_idx * TL_FAST_SPARSE_ENTRIES
                            + subblock_sparse_positions[:, None]
                        )
                        * TL_ROPE_DIM
                        + rope_columns[None, :],
                        subblock_key_rope_tile,
                    )


@triton.jit
def _sfa_gathered_attention(
    QUERY,
    QUERY_ROPE,
    GATHERED_KV,
    GATHERED_KEY_ROPE,
    SPLIT_WEIGHTED_VALUE_SUM,
    SPLIT_SOFTMAX_MAX,
    SPLIT_SOFTMAX_SUM,
    OUT,
    ADDITIVE_ATTENTION_MASK,
    KV_END_EXCLUSIVE,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    NUM_QUERY_TOKENS: tl.constexpr,
    RETURN_SOFTMAX_STATS: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
):
    # Fixed specialization; head groups and KV blocks are compute tiles.
    NUM_QUERY_HEADS: tl.constexpr = TL_FAST_QUERY_HEADS
    KV_DIM: tl.constexpr = TL_KV_DIM
    ROPE_DIM: tl.constexpr = TL_ROPE_DIM
    HEADS_PER_GROUP: tl.constexpr = TL_GATHERED_HEADS_PER_GROUP
    NUM_HEAD_GROUPS: tl.constexpr = NUM_QUERY_HEADS // HEADS_PER_GROUP
    NUM_GATHERED_KV_ROWS: tl.constexpr = TL_FAST_SPARSE_ENTRIES
    KV_BLOCK_ROWS: tl.constexpr = TL_GATHERED_KV_BLOCK_ROWS
    NUM_KV_BLOCKS: tl.constexpr = NUM_GATHERED_KV_ROWS // KV_BLOCK_ROWS
    program_id = tl.program_id(0)
    for task_idx in range(
        program_id, NUM_QUERY_TOKENS * NUM_HEAD_GROUPS * NUM_KV_SPLITS, NUM_CORES
    ):
        kv_split_idx = task_idx % NUM_KV_SPLITS
        head_group_idx = (task_idx // NUM_KV_SPLITS) % NUM_HEAD_GROUPS
        query_token_idx = (task_idx // (NUM_HEAD_GROUPS * NUM_KV_SPLITS)).to(tl.int64)
        if RETURN_SOFTMAX_STATS:
            kv_end_exclusive = tl.load(KV_END_EXCLUSIVE + query_token_idx)
        query_tile = tl.load(
            tl.make_block_ptr(
                QUERY + query_token_idx * NUM_QUERY_HEADS * KV_DIM,
                (NUM_QUERY_HEADS, KV_DIM),
                (KV_DIM, 1),
                (head_group_idx * HEADS_PER_GROUP, 0),
                (HEADS_PER_GROUP, KV_DIM),
                (1, 0),
            )
        )
        query_rope_tile = tl.load(
            tl.make_block_ptr(
                QUERY_ROPE + query_token_idx * NUM_QUERY_HEADS * ROPE_DIM,
                (NUM_QUERY_HEADS, ROPE_DIM),
                (ROPE_DIM, 1),
                (head_group_idx * HEADS_PER_GROUP, 0),
                (HEADS_PER_GROUP, ROPE_DIM),
                (1, 0),
            )
        )
        softmax_max = tl.full(
            (HEADS_PER_GROUP,), TL_VLLM_FINITE_SOFTMAX_SENTINEL, tl.float32
        )
        softmax_sum = tl.zeros((HEADS_PER_GROUP,), tl.float32)
        weighted_value_sum = tl.zeros((HEADS_PER_GROUP, KV_DIM), tl.float32)
        KV_BLOCKS_PER_SPLIT: tl.constexpr = triton.cdiv(NUM_KV_BLOCKS, NUM_KV_SPLITS)
        for kv_block_idx in range(
            kv_split_idx * KV_BLOCKS_PER_SPLIT,
            tl.minimum((kv_split_idx + 1) * KV_BLOCKS_PER_SPLIT, NUM_KV_BLOCKS),
        ):
            kv_tile = tl.load(
                tl.make_block_ptr(
                    GATHERED_KV + query_token_idx * NUM_GATHERED_KV_ROWS * KV_DIM,
                    (NUM_GATHERED_KV_ROWS, KV_DIM),
                    (KV_DIM, 1),
                    (kv_block_idx * KV_BLOCK_ROWS, 0),
                    (KV_BLOCK_ROWS, KV_DIM),
                    (1, 0),
                )
            )
            key_rope_tile = tl.load(
                tl.make_block_ptr(
                    GATHERED_KEY_ROPE
                    + query_token_idx * NUM_GATHERED_KV_ROWS * ROPE_DIM,
                    (NUM_GATHERED_KV_ROWS, ROPE_DIM),
                    (ROPE_DIM, 1),
                    (kv_block_idx * KV_BLOCK_ROWS, 0),
                    (KV_BLOCK_ROWS, ROPE_DIM),
                    (1, 0),
                )
            )
            attention_scores = tl.dot(query_tile, tl.trans(kv_tile))
            attention_scores = (
                tl.dot(query_rope_tile, tl.trans(key_rope_tile), attention_scores)
                * SOFTMAX_SCALE
            )
            sparse_positions = kv_block_idx * KV_BLOCK_ROWS + tl.arange(
                0, KV_BLOCK_ROWS
            )
            additive_mask = tl.load(
                ADDITIVE_ATTENTION_MASK
                + query_token_idx * NUM_GATHERED_KV_ROWS
                + sparse_positions
            )
            attention_scores = attention_scores + additive_mask[None, :]
            updated_softmax_max = tl.maximum(softmax_max, tl.max(attention_scores, 1))
            softmax_exp_base = updated_softmax_max
            previous_state_scale = tl.exp(softmax_max - softmax_exp_base)
            unnormalized_weights = tl.exp(attention_scores - softmax_exp_base[:, None])
            softmax_sum = softmax_sum * previous_state_scale + tl.sum(
                unnormalized_weights, 1
            )
            weighted_value_sum = weighted_value_sum * previous_state_scale[
                :, None
            ] + tl.dot(unnormalized_weights.to(GATHERED_KV.dtype.element_ty), kv_tile)
            softmax_max = updated_softmax_max
        head_indices = head_group_idx * HEADS_PER_GROUP + tl.arange(0, HEADS_PER_GROUP)
        kv_columns = tl.arange(0, KV_DIM)
        if NUM_KV_SPLITS == 1:
            tl.store(
                OUT
                + query_token_idx * NUM_QUERY_HEADS * KV_DIM
                + head_indices[:, None] * KV_DIM
                + kv_columns[None, :],
                weighted_value_sum
                / tl.where(softmax_sum > 0, softmax_sum, 1.0)[:, None],
            )
            if RETURN_SOFTMAX_STATS:
                tl.store(
                    SOFTMAX_MAX + query_token_idx * NUM_QUERY_HEADS + head_indices,
                    tl.where(
                        softmax_sum > 0,
                        softmax_max,
                        tl.where(
                            kv_end_exclusive > 0, TL_VLLM_FINITE_SOFTMAX_SENTINEL, 0.0
                        ),
                    ),
                )
                tl.store(
                    SOFTMAX_SUM + query_token_idx * NUM_QUERY_HEADS + head_indices,
                    tl.where(
                        softmax_sum > 0,
                        softmax_sum,
                        tl.minimum(NUM_GATHERED_KV_ROWS, kv_end_exclusive).to(
                            tl.float32
                        ),
                    ),
                )
        else:
            tl.store(
                SPLIT_WEIGHTED_VALUE_SUM
                + (query_token_idx * NUM_KV_SPLITS + kv_split_idx)
                * NUM_QUERY_HEADS
                * KV_DIM
                + head_indices[:, None] * KV_DIM
                + kv_columns[None, :],
                weighted_value_sum,
            )
            tl.store(
                SPLIT_SOFTMAX_MAX
                + (query_token_idx * NUM_KV_SPLITS + kv_split_idx) * NUM_QUERY_HEADS
                + head_indices,
                softmax_max,
            )
            tl.store(
                SPLIT_SOFTMAX_SUM
                + (query_token_idx * NUM_KV_SPLITS + kv_split_idx) * NUM_QUERY_HEADS
                + head_indices,
                softmax_sum,
            )


@triton.jit
def _sfa_prepare_indices(
    SPARSE_INDICES,
    BLOCK_TABLE,
    QUERY_LENGTHS,
    KV_LENGTHS,
    PHYSICAL_ROW_INDICES,
    PACKED_QUERY_META,
    PAGES_PER_REQUEST: tl.constexpr,
    NUM_QUERY_TOKENS: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    QUERY_SEQ_CAPACITY: tl.constexpr,
    QUERY_IS_TND: tl.constexpr,
    HAS_QUERY_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr,
    NUM_CORES: tl.constexpr,
    ALLOW_EXACT_SORT: tl.constexpr = False,
):
    program_id = tl.program_id(0)
    for query_token_idx in range(program_id, NUM_QUERY_TOKENS, NUM_CORES):
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
        kv_end_exclusive = kv_end_exclusive.to(tl.int32)
        sparse_positions = tl.arange(0, TL_FAST_SPARSE_ENTRIES)
        logical_kv_indices = tl.load(
            SPARSE_INDICES + query_token_idx * TL_FAST_SPARSE_ENTRIES + sparse_positions
        )
        valid_kv_mask = (
            (logical_kv_indices >= 0)
            & (logical_kv_indices < kv_end_exclusive)
            & (sparse_positions < kv_end_exclusive)
        )
        valid_kv_mask &= logical_kv_indices // TL_FAST_PAGE_SIZE < PAGES_PER_REQUEST
        all_sparse_entries_valid = (
            tl.sum(valid_kv_mask.to(tl.int32), 0) == TL_FAST_SPARSE_ENTRIES
        )
        page_table_indices = tl.arange(0, triton.next_power_of_2(PAGES_PER_REQUEST))
        request_page_table = tl.load(
            BLOCK_TABLE + request_idx * PAGES_PER_REQUEST + page_table_indices,
            page_table_indices < PAGES_PER_REQUEST,
            other=0,
        )
        safe_logical_kv_indices = tl.where(valid_kv_mask, logical_kv_indices, 0)
        # Preserve int32 page-ID bits through float32 gather; this is
        # not a numeric cast. The original lowering rationale is unverified.
        physical_page_ids = tl.gather(
            request_page_table.to(tl.float32, bitcast=True),
            safe_logical_kv_indices >> TL_FAST_PAGE_SIZE_LOG2,
            0,
        ).to(tl.int32, bitcast=True)
        physical_row_indices = physical_page_ids * TL_FAST_PAGE_SIZE + (
            safe_logical_kv_indices & TL_FAST_PAGE_OFFSET_MASK
        )
        prepared_row_indices = tl.where(
            valid_kv_mask, physical_row_indices, TL_INVALID_SPARSE_INDEX
        )
        if ALLOW_EXACT_SORT:
            # Only full, widely scattered selections benefit from reordering.
            # Short windows and already ordered selections retain their order.
            if (kv_end_exclusive > TL_SORT_MIN_LOGICAL_SPAN) & all_sparse_entries_valid:
                logical_index_span = tl.max(safe_logical_kv_indices, 0) - tl.min(
                    safe_logical_kv_indices, 0
                )
                if logical_index_span > TL_SORT_MIN_LOGICAL_SPAN:
                    previous_logical_kv_indices = tl.load(
                        SPARSE_INDICES
                        + query_token_idx * TL_FAST_SPARSE_ENTRIES
                        + sparse_positions
                        - 1,
                        sparse_positions > 0,
                        other=TL_INVALID_SPARSE_INDEX,
                    )
                    indices_are_sorted = (
                        tl.sum(
                            (
                                (sparse_positions > 0)
                                & (logical_kv_indices < previous_logical_kv_indices)
                            ).to(tl.int32),
                            0,
                        )
                        == 0
                    )
                    if not indices_are_sorted:
                        prepared_row_indices = ascend_sort(
                            physical_row_indices.to(tl.float32), descending=False
                        ).to(tl.int32)
        tl.store(
            PHYSICAL_ROW_INDICES
            + query_token_idx * TL_FAST_SPARSE_ENTRIES
            + sparse_positions,
            prepared_row_indices,
        )
        # Bits 31:1 hold the exclusive KV end; bit 0 marks a full valid selection.
        packed_query_meta = (
            kv_end_exclusive.to(tl.uint32) << 1
        ) | all_sparse_entries_valid.to(tl.uint32)
        tl.store(PACKED_QUERY_META + query_token_idx, packed_query_meta)


@triton.jit
def _sfa_prepare_strided_indices(
    SPARSE_INDICES,
    BLOCK_TABLE,
    QUERY_LENGTHS,
    KV_LENGTHS,
    PHYSICAL_ROW_INDICES,
    PACKED_QUERY_META,
    PAGES_PER_REQUEST: tl.constexpr,
    NUM_QUERY_TOKENS: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    QUERY_SEQ_CAPACITY: tl.constexpr,
    QUERY_IS_TND: tl.constexpr,
    HAS_QUERY_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr,
    NUM_CORES: tl.constexpr,
    SHARED_KV_ROW_LAYOUT: tl.constexpr = None,
):
    program_id = tl.program_id(0)
    for query_token_idx in range(program_id, NUM_QUERY_TOKENS, NUM_CORES):
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
        kv_end_exclusive = kv_end_exclusive.to(tl.int32)
        sparse_positions = tl.arange(0, TL_FAST_SPARSE_ENTRIES)
        logical_kv_indices = tl.load(
            SPARSE_INDICES + query_token_idx * TL_FAST_SPARSE_ENTRIES + sparse_positions
        )
        valid_kv_mask = (
            (logical_kv_indices >= 0)
            & (logical_kv_indices < kv_end_exclusive)
            & (sparse_positions < kv_end_exclusive)
        )
        valid_kv_mask &= logical_kv_indices // TL_FAST_PAGE_SIZE < PAGES_PER_REQUEST
        all_sparse_entries_valid = (
            tl.sum(valid_kv_mask.to(tl.int32), 0) == TL_FAST_SPARSE_ENTRIES
        )
        page_table_indices = tl.arange(0, triton.next_power_of_2(PAGES_PER_REQUEST))
        request_page_table = tl.load(
            BLOCK_TABLE + request_idx * PAGES_PER_REQUEST + page_table_indices,
            page_table_indices < PAGES_PER_REQUEST,
            other=0,
        )
        safe_logical_kv_indices = tl.where(valid_kv_mask, logical_kv_indices, 0)
        # Preserve int32 page-ID bits through float32 gather; this is
        # not a numeric cast. The original lowering rationale is unverified.
        physical_page_ids = tl.gather(
            request_page_table.to(tl.float32, bitcast=True),
            safe_logical_kv_indices >> TL_FAST_PAGE_SIZE_LOG2,
            0,
        ).to(tl.int32, bitcast=True)
        physical_row_indices = physical_page_ids * TL_FAST_PAGE_SIZE + (
            safe_logical_kv_indices & TL_FAST_PAGE_OFFSET_MASK
        )
        if SHARED_KV_ROW_LAYOUT is not None:
            rows_per_page: tl.constexpr = SHARED_KV_ROW_LAYOUT[0]
            page_stride_rows: tl.constexpr = SHARED_KV_ROW_LAYOUT[1]
            row_stride_rows: tl.constexpr = SHARED_KV_ROW_LAYOUT[2]
            physical_row_indices = (
                physical_row_indices // rows_per_page
            ) * page_stride_rows + (
                physical_row_indices % rows_per_page
            ) * row_stride_rows
        prepared_row_indices = tl.where(
            valid_kv_mask, physical_row_indices, TL_INVALID_SPARSE_INDEX
        )
        tl.store(
            PHYSICAL_ROW_INDICES
            + query_token_idx * TL_FAST_SPARSE_ENTRIES
            + sparse_positions,
            prepared_row_indices,
        )
        # Bits 31:1 hold the exclusive KV end; bit 0 marks a full valid selection.
        packed_query_meta = (
            kv_end_exclusive.to(tl.uint32) << 1
        ) | all_sparse_entries_valid.to(tl.uint32)
        tl.store(PACKED_QUERY_META + query_token_idx, packed_query_meta)


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
    MANUAL_BLOCK_SYNC: tl.constexpr,
    OUTPUT_BLOCK_COLUMNS: tl.constexpr,
    OUTPUT_BLOCK_HEADS: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    WORKSPACE_PADDING: tl.constexpr,
    NUM_VECTOR_SUBCORES: tl.constexpr,
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
    program_id = tl.program_id(0)
    vector_subcore_id = 0
    if NUM_VECTOR_SUBCORES == TL_VECTOR_SUBCORES_PER_CUBE:
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
                if MANUAL_BLOCK_SYNC:
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
                if MANUAL_BLOCK_SYNC:
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
                if MANUAL_BLOCK_SYNC:
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
                if MANUAL_BLOCK_SYNC:
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
                if MANUAL_BLOCK_SYNC:
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
                if MANUAL_BLOCK_SYNC:
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


@triton.jit
def _sfa_combine(
    SPLIT_WEIGHTED_VALUE_SUM,
    SPLIT_SOFTMAX_MAX,
    SPLIT_SOFTMAX_SUM,
    OUT,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    QUERY_LENGTHS,
    KV_LENGTHS,
    NUM_KV_SPLITS: tl.constexpr,
    RETURN_SOFTMAX_STATS: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    QUERY_SEQ_CAPACITY: tl.constexpr,
    QUERY_IS_TND: tl.constexpr,
    HAS_QUERY_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    task_idx = tl.program_id(0)
    query_token_idx = task_idx // TL_COMBINE_PROGRAMS_PER_QUERY
    head_indices = (
        task_idx % TL_COMBINE_PROGRAMS_PER_QUERY * TL_COMBINE_HEADS_PER_PROGRAM
        + tl.arange(0, TL_COMBINE_HEADS_PER_PROGRAM)
    )
    kv_columns = tl.arange(0, TL_KV_DIM)
    softmax_max = tl.full((TL_COMBINE_HEADS_PER_PROGRAM,), float("-inf"), tl.float32)
    for kv_split_idx in range(NUM_KV_SPLITS):
        split_softmax_max = tl.load(
            SPLIT_SOFTMAX_MAX
            + (query_token_idx * NUM_KV_SPLITS + kv_split_idx) * TL_FAST_QUERY_HEADS
            + head_indices
        )
        softmax_max = tl.maximum(softmax_max, split_softmax_max)
    combined_weighted_value_sum = tl.zeros(
        (TL_COMBINE_HEADS_PER_PROGRAM, TL_KV_DIM), tl.float32
    )
    softmax_sum = tl.zeros((TL_COMBINE_HEADS_PER_PROGRAM,), tl.float32)
    for kv_split_idx in range(NUM_KV_SPLITS):
        split_softmax_max = tl.load(
            SPLIT_SOFTMAX_MAX
            + (query_token_idx * NUM_KV_SPLITS + kv_split_idx) * TL_FAST_QUERY_HEADS
            + head_indices
        )
        split_softmax_sum = tl.load(
            SPLIT_SOFTMAX_SUM
            + (query_token_idx * NUM_KV_SPLITS + kv_split_idx) * TL_FAST_QUERY_HEADS
            + head_indices
        )
        split_weighted_value_sum = tl.load(
            SPLIT_WEIGHTED_VALUE_SUM
            + (query_token_idx * NUM_KV_SPLITS + kv_split_idx)
            * TL_FAST_QUERY_HEADS
            * TL_KV_DIM
            + head_indices[:, None] * TL_KV_DIM
            + kv_columns[None, :]
        )
        split_rescale_factor = tl.exp(split_softmax_max - softmax_max)
        combined_weighted_value_sum += (
            split_weighted_value_sum * split_rescale_factor[:, None]
        )
        softmax_sum += split_softmax_sum * split_rescale_factor
    tl.store(
        OUT
        + query_token_idx * TL_FAST_QUERY_HEADS * TL_KV_DIM
        + head_indices[:, None] * TL_KV_DIM
        + kv_columns[None, :],
        combined_weighted_value_sum
        / tl.where(softmax_sum > 0, softmax_sum, 1.0)[:, None],
    )

    if RETURN_SOFTMAX_STATS:
        _, kv_end_exclusive, _ = _sfa_query_meta(
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
        tl.store(
            SOFTMAX_MAX + query_token_idx * TL_FAST_QUERY_HEADS + head_indices,
            tl.where(
                softmax_sum > 0,
                softmax_max,
                tl.where(kv_end_exclusive > 0, TL_VLLM_FINITE_SOFTMAX_SENTINEL, 0.0),
            ),
        )
        tl.store(
            SOFTMAX_SUM + query_token_idx * TL_FAST_QUERY_HEADS + head_indices,
            tl.where(
                softmax_sum > 0,
                softmax_sum,
                tl.minimum(TL_FAST_SPARSE_ENTRIES, kv_end_exclusive).to(tl.float32),
            ),
        )


@triton.jit
def _sfa_combine_two(
    SPLIT_WEIGHTED_VALUE_SUM,
    SPLIT_SOFTMAX_MAX,
    SPLIT_SOFTMAX_SUM,
    OUT,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    QUERY_LENGTHS,
    KV_LENGTHS,
    NUM_KV_SPLITS: tl.constexpr,
    RETURN_SOFTMAX_STATS: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    QUERY_SEQ_CAPACITY: tl.constexpr,
    QUERY_IS_TND: tl.constexpr,
    HAS_QUERY_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    task_idx = tl.program_id(0)
    query_token_idx = task_idx // TL_COMBINE_TWO_PROGRAMS_PER_QUERY
    head_indices = (
        task_idx % TL_COMBINE_TWO_PROGRAMS_PER_QUERY * TL_COMBINE_TWO_HEADS_PER_PROGRAM
        + tl.arange(0, TL_COMBINE_TWO_HEADS_PER_PROGRAM)
    )
    split_max_0 = tl.load(
        SPLIT_SOFTMAX_MAX
        + query_token_idx * TL_COMBINE_TWO_SPLITS * TL_FAST_QUERY_HEADS
        + head_indices
    )
    split_max_1 = tl.load(
        SPLIT_SOFTMAX_MAX
        + (query_token_idx * TL_COMBINE_TWO_SPLITS + 1) * TL_FAST_QUERY_HEADS
        + head_indices
    )
    softmax_max = tl.maximum(split_max_0, split_max_1)
    split_weight_0 = tl.exp(split_max_0 - softmax_max)
    split_weight_1 = tl.exp(split_max_1 - softmax_max)
    split_sum_0 = tl.load(
        SPLIT_SOFTMAX_SUM
        + query_token_idx * TL_COMBINE_TWO_SPLITS * TL_FAST_QUERY_HEADS
        + head_indices
    )
    split_sum_1 = tl.load(
        SPLIT_SOFTMAX_SUM
        + (query_token_idx * TL_COMBINE_TWO_SPLITS + 1) * TL_FAST_QUERY_HEADS
        + head_indices
    )
    softmax_sum = split_sum_0 * split_weight_0 + split_sum_1 * split_weight_1
    inverse_softmax_sum = 1.0 / tl.where(softmax_sum > 0, softmax_sum, 1.0)
    split_weight_0 = split_weight_0 * inverse_softmax_sum
    split_weight_1 = split_weight_1 * inverse_softmax_sum
    for value_block_idx in range(TL_COMBINE_TWO_VALUE_TILES):
        kv_columns = value_block_idx * TL_COMBINE_TWO_VALUE_COLUMNS + tl.arange(
            0, TL_COMBINE_TWO_VALUE_COLUMNS
        )
        weighted_value_sum_0 = tl.load(
            SPLIT_WEIGHTED_VALUE_SUM
            + query_token_idx * TL_COMBINE_TWO_SPLITS * TL_FAST_QUERY_HEADS * TL_KV_DIM
            + head_indices[:, None] * TL_KV_DIM
            + kv_columns[None, :]
        )
        weighted_value_sum_1 = tl.load(
            SPLIT_WEIGHTED_VALUE_SUM
            + (query_token_idx * TL_COMBINE_TWO_SPLITS + 1)
            * TL_FAST_QUERY_HEADS
            * TL_KV_DIM
            + head_indices[:, None] * TL_KV_DIM
            + kv_columns[None, :]
        )
        output_tile = (
            weighted_value_sum_0 * split_weight_0[:, None]
            + weighted_value_sum_1 * split_weight_1[:, None]
        )
        tl.store(
            OUT
            + query_token_idx * TL_FAST_QUERY_HEADS * TL_KV_DIM
            + head_indices[:, None] * TL_KV_DIM
            + kv_columns[None, :],
            output_tile,
        )
    if RETURN_SOFTMAX_STATS:
        _, kv_end_exclusive, _ = _sfa_query_meta(
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
        tl.store(
            SOFTMAX_MAX + query_token_idx * TL_FAST_QUERY_HEADS + head_indices,
            tl.where(
                softmax_sum > 0,
                softmax_max,
                tl.where(kv_end_exclusive > 0, TL_VLLM_FINITE_SOFTMAX_SENTINEL, 0.0),
            ),
        )
        tl.store(
            SOFTMAX_SUM + query_token_idx * TL_FAST_QUERY_HEADS + head_indices,
            tl.where(
                softmax_sum > 0,
                softmax_sum,
                tl.minimum(TL_FAST_SPARSE_ENTRIES, kv_end_exclusive).to(tl.float32),
            ),
        )


@triton.jit
def _sfa_copy_strided(
    INPUT,
    OUTPUT,
    NUM_ELEMENTS: tl.constexpr,
    INPUT_SHAPE: tl.constexpr,
    INPUT_STRIDES: tl.constexpr,
    COPY_BLOCK_ELEMENTS: tl.constexpr,
):
    for copy_block_idx in range(
        tl.program_id(0),
        triton.cdiv(NUM_ELEMENTS, COPY_BLOCK_ELEMENTS),
        tl.num_programs(0),
    ):
        flat_indices = copy_block_idx.to(tl.int64) * COPY_BLOCK_ELEMENTS + tl.arange(
            0, COPY_BLOCK_ELEMENTS
        ).to(tl.int64)
        remaining_indices = flat_indices
        input_element_offsets = tl.full((COPY_BLOCK_ELEMENTS,), 0, tl.int64)
        for axis in tl.static_range(len(INPUT_SHAPE) - 1, -1, -1):
            input_element_offsets += (
                remaining_indices % INPUT_SHAPE[axis]
            ) * INPUT_STRIDES[axis]
            remaining_indices = remaining_indices // INPUT_SHAPE[axis]
        tl.store(
            OUTPUT + flat_indices,
            tl.load(
                INPUT + input_element_offsets, flat_indices < NUM_ELEMENTS, other=0
            ),
            flat_indices < NUM_ELEMENTS,
        )


def _sfa_make_contiguous(tensor):
    if tensor is None or tensor.is_contiguous():
        return tensor
    contiguous_tensor = torch.empty(
        tuple(tensor.shape), dtype=tensor.dtype, device=tensor.device
    )
    num_programs = min(
        triton.cdiv(tensor.numel(), STRIDED_COPY_BLOCK_ELEMENTS),
        _get_core_num(tensor.device)[1],
    )
    _sfa_copy_strided[(num_programs,)](
        tensor,
        contiguous_tensor,
        tensor.numel(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        STRIDED_COPY_BLOCK_ELEMENTS,
    )
    return contiguous_tensor


def sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=SPARSE_MODE_CAUSAL,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    attention_mode=MLA_ABSORB_ATTENTION_MODE,
    return_softmax_lse=False,
):
    """vLLM-Ascend SFA interface for shared-KV MLA inference on Ascend 910B.

    Returns (attention_out, softmax_max, softmax_sum). The output has query's
    shape and dtype; optional max/sum are FP32, or empty tensors when disabled.
    FP16/BF16, BSND/TND queries, BSND/TND/paged caches, block sparsity and
    right-down causal masking are supported. Main/RoPE dimensions are 512/64.
    TND lengths are cumulative; other layouts use per-request lengths.

    value must alias key with the same shape/strides (MLA absorb). Independent
    V and backward are outside this shared-KV contract. Sparse indices name
    blocks, with valid entries followed by -1 padding; duplicates count again.
    Lengths, indices and page mappings must be valid for their input storage.
    Logical pages outside the per-request table are defensively masked; physical
    page IDs must still refer to valid KV storage.
    Device metadata is consumed in kernels, including for T==batch_size.
    Strided KV is gathered directly; only noncontiguous Q/metadata is materialized.
    Contiguous inputs stay zero-copy. No host length readback or Torch compute
    fallback is used.
    """
    required_tensors = (query, key, value, sparse_indices)
    if not all(isinstance(tensor, torch.Tensor) for tensor in required_tensors):
        raise TypeError("query, key, value and sparse_indices must be tensors")
    if layout_query not in ("BSND", "TND") or layout_kv not in (
        "BSND",
        "TND",
        "PA_BSND",
    ):
        raise NotImplementedError("unsupported query/cache layout")
    if layout_kv != "PA_BSND" and layout_kv != layout_query:
        raise ValueError("nonpaged query and cache layouts must match")
    if attention_mode != MLA_ABSORB_ATTENTION_MODE or sparse_mode not in (
        SPARSE_MODE_ALL,
        SPARSE_MODE_CAUSAL,
    ):
        raise NotImplementedError("only MLA absorb and sparse_mode 0/3 are supported")
    if pre_tokens != INT64_MAX or next_tokens != INT64_MAX:
        raise NotImplementedError("pre_tokens and next_tokens require their defaults")
    if (
        not isinstance(sparse_block_size, int)
        or sparse_block_size not in SUPPORTED_SPARSE_BLOCK_SIZES
    ):
        raise NotImplementedError("sparse_block_size must be a power of two in [1,128]")
    if not isinstance(return_softmax_lse, bool):
        raise TypeError("return_softmax_lse must be a bool")
    if query_rope is None or key_rope is None:
        raise NotImplementedError("this MLA interface requires separate query/key RoPE")
    optional_tensors = (
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
    )
    input_tensors = required_tensors + tuple(
        tensor for tensor in optional_tensors if tensor is not None
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in input_tensors):
        raise TypeError("optional tensor arguments must be tensors or None")
    if query.device.type != "npu":
        raise NotImplementedError("sparse_flash_attention requires an Ascend NPU")
    if any(tensor.device != query.device for tensor in input_tensors):
        raise ValueError("all tensors must be on query's device")
    if any(tensor.requires_grad for tensor in input_tensors):
        raise NotImplementedError("sparse_flash_attention is forward-only")
    all_inputs_contiguous = all(tensor.is_contiguous() for tensor in input_tensors)
    if any(tensor.numel() == 0 for tensor in input_tensors):
        raise ValueError("provided tensors must be nonempty")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("floating inputs must be FP16 or BF16")
    if any(
        tensor.dtype != query.dtype for tensor in (key, value, query_rope, key_rope)
    ):
        raise ValueError("all floating inputs must have the same dtype")
    metadata_tensors = (
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
    )
    if any(
        tensor is not None and tensor.dtype != torch.int32
        for tensor in metadata_tensors
    ):
        raise NotImplementedError("indices, page tables and lengths must be int32")
    if (
        value.shape != key.shape
        or value.stride() != key.stride()
        or value.data_ptr() != key.data_ptr()
    ):
        raise NotImplementedError("shared-KV MLA requires value to alias key")
    if (
        query.ndim
        != (TND_TENSOR_RANK if layout_query == "TND" else BATCHED_TENSOR_RANK)
        or query.shape[-1] != KV_DIM
    ):
        raise ValueError("query rank must match its layout, with main dimension 512")
    num_query_heads = query.shape[-2]
    if num_query_heads not in SUPPORTED_QUERY_HEADS:
        raise NotImplementedError("query heads must be 1,2,4,8,16,32,64 or 128")
    if query_rope.shape != (*query.shape[:-1], ROPE_DIM):
        raise ValueError("query_rope must match query with dimension 64")
    if key.ndim != (
        TND_TENSOR_RANK if layout_kv == "TND" else BATCHED_TENSOR_RANK
    ) or key.shape[-2:] != (1, KV_DIM):
        raise ValueError(
            "cache rank must match its layout, with one KV head and dimension 512"
        )
    if key_rope.shape != (*key.shape[:-1], ROPE_DIM):
        raise ValueError("key_rope must match key with dimension 64")
    if layout_query == "TND":
        if actual_seq_lengths_query is None or actual_seq_lengths_query.ndim != 1:
            raise ValueError("TND query requires cumulative query lengths")
        num_requests = actual_seq_lengths_query.numel()
        query_seq_capacity = 0
        num_query_tokens = query.shape[0]
    else:
        num_requests, query_seq_capacity = query.shape[:2]
        num_query_tokens = num_requests * query_seq_capacity
    for lengths in (actual_seq_lengths_query, actual_seq_lengths_kv):
        if lengths is not None and lengths.shape != (num_requests,):
            raise ValueError("length tensors must have shape [batch_size]")
    if (
        sparse_indices.ndim != query.ndim
        or sparse_indices.shape[:-2] != query.shape[:-2]
        or sparse_indices.shape[-2] != 1
    ):
        raise ValueError("sparse_indices must match query tokens with one KV head")
    num_sparse_blocks = sparse_indices.shape[-1]
    paged = layout_kv == "PA_BSND"
    page_size, pages_per_request, kv_seq_capacity = 0, 0, 0
    if paged:
        page_size = key.shape[1]
        if (
            page_size > MAX_SUPPORTED_PAGE_SIZE
            or page_size % PAGE_SIZE_ALIGNMENT
            or page_size % sparse_block_size
        ):
            raise ValueError(
                "page size must be 16-aligned, <=1024 and divisible by sparse_block_size"
            )
        if (
            block_table is None
            or block_table.ndim != 2
            or block_table.shape[0] != num_requests
        ):
            raise ValueError(
                "paged KV requires block_table [batch_size,pages_per_request]"
            )
        pages_per_request = block_table.shape[1]
    else:
        if block_table is not None:
            raise ValueError("block_table is only used with paged KV")
        if layout_kv == "BSND":
            if key.shape[0] != num_requests:
                raise ValueError("query and cache batch sizes must match")
            kv_seq_capacity = key.shape[1]
    if layout_kv in ("TND", "PA_BSND") and actual_seq_lengths_kv is None:
        raise ValueError("TND/paged KV requires actual_seq_lengths_kv")
    if not isinstance(scale_value, (int, float)) or not math.isfinite(scale_value):
        raise ValueError("scale_value must be a finite scalar")

    with torch.npu.device(query.device):
        logical_sparse_indices = sparse_indices
        if not all_inputs_contiguous:
            query = _sfa_make_contiguous(query)
            query_rope = _sfa_make_contiguous(query_rope)
            logical_sparse_indices = _sfa_make_contiguous(logical_sparse_indices)
            block_table = _sfa_make_contiguous(block_table)
            actual_seq_lengths_query = _sfa_make_contiguous(actual_seq_lengths_query)
            actual_seq_lengths_kv = _sfa_make_contiguous(actual_seq_lengths_kv)

        def get_cache_access_spec(tensor):
            # (rows_per_group, group_stride_elems, row_stride_elems,
            #  column_stride_elems, requires_int64_element_offsets).
            # Relative offsets are measured in elements, not bytes.
            if tensor.is_contiguous():
                return None
            requires_int64_element_offsets = (
                sum(
                    (size - 1) * dimension_stride
                    for size, dimension_stride in zip(tensor.shape, tensor.stride())
                )
                > INT32_MAX
            )
            if layout_kv == "TND":
                return (
                    1,
                    tensor.stride(0),
                    0,
                    tensor.stride(-1),
                    requires_int64_element_offsets,
                )
            return (
                tensor.shape[1],
                tensor.stride(0),
                tensor.stride(1),
                tensor.stride(-1),
                requires_int64_element_offsets,
            )

        kv_access_spec, key_rope_access_spec = get_cache_access_spec(
            key
        ), get_cache_access_spec(key_rope)
        cache_access_kwargs = dict(
            KV_ACCESS_SPEC=kv_access_spec, KEY_ROPE_ACCESS_SPEC=key_rope_access_spec
        )
        generic_access_requires_int64_offsets = (
            kv_access_spec is not None
            and kv_access_spec[CACHE_SPEC_INT64_OFFSETS_INDEX]
        ) or (
            key_rope_access_spec is not None
            and key_rope_access_spec[CACHE_SPEC_INT64_OFFSETS_INDEX]
        )

        def allocate_empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=query.device)

        def allocate_workspace(shape, dtype, relative_address_offset_bytes):
            # Retain the empirically tuned scratch address layout for >=16-token
            # Ascend pipelines: offsets within a 64-KiB window stabilize placement
            # relative to KV storage. The hardware cause has not been established;
            # this is not a UB-bank or synchronization requirement.
            # as_strided only views uninitialized storage; it launches no copy/cast.
            if num_query_tokens < LARGE_QUERY_BATCH_THRESHOLD:
                return allocate_empty(shape, dtype)
            address_period_bytes = SCRATCH_ADDRESS_PERIOD_BYTES
            element_size_bytes = (
                FP32_ELEMENT_BYTES
                if dtype == torch.float32
                else FP16_BF16_ELEMENT_BYTES
            )
            num_elements = math.prod(shape)
            workspace_storage = allocate_empty(
                (num_elements + address_period_bytes // element_size_bytes,), dtype
            )
            storage_offset_elements = (
                (
                    key.data_ptr()
                    + relative_address_offset_bytes
                    - workspace_storage.data_ptr()
                )
                % address_period_bytes
            ) // element_size_bytes
            contiguous_strides = []
            running_stride = 1
            for dimension_size in reversed(shape):
                contiguous_strides.insert(0, running_stride)
                running_stride *= dimension_size
            return workspace_storage.as_strided(
                shape, tuple(contiguous_strides), storage_offset_elements
            )

        attention_output = allocate_empty(query.shape, query.dtype)
        softmax_stats_shape = (
            (
                (1, num_query_tokens, num_query_heads)
                if layout_query == "TND"
                else (num_requests, 1, query_seq_capacity, num_query_heads)
            )
            if return_softmax_lse
            else (0,)
        )
        softmax_max, softmax_sum = allocate_empty(
            softmax_stats_shape, torch.float32
        ), allocate_empty(softmax_stats_shape, torch.float32)
        # Unused length pointers are placeholders guarded by HAS_*_LENGTHS.
        query_lengths = (
            actual_seq_lengths_query
            if actual_seq_lengths_query is not None
            else logical_sparse_indices
        )
        kv_lengths = (
            actual_seq_lengths_kv
            if actual_seq_lengths_kv is not None
            else logical_sparse_indices
        )
        num_cube_cores, num_vector_cores = _get_core_num(query.device)
        # Small head counts need padding to Cube's minimum head tile. Keeping
        # all stages in the pipeline avoids materializing the full selection.
        use_small_head_pipeline = (
            paged
            and num_query_heads == 4
            and 2 < num_query_tokens <= SMALL_QUERY_MAX_TOKENS
            and num_sparse_blocks == FAST_SPARSE_ENTRIES
            and page_size == FAST_PAGE_SIZE
            and sparse_block_size == 1
            and kv_access_spec is None
            and key_rope_access_spec is None
        )
        # Long H64 batches have enough queries to pipeline across query
        # boundaries. Prepared physical indices and two state generations keep
        # masking and normalization independent for every query.
        use_cross_query_pipeline = (
            paged
            and num_query_heads == 64
            and num_query_tokens > SMALL_QUERY_MAX_TOKENS
            and num_sparse_blocks == FAST_SPARSE_ENTRIES
            and page_size == FAST_PAGE_SIZE
            and sparse_block_size == 1
            and kv_access_spec is None
            and key_rope_access_spec is None
            and key.numel() <= MAX_CACHE_ELEMENTS_FOR_INT32_OFFSETS
        )
        if (
            not use_small_head_pipeline
            and paged
            and num_query_tokens <= SMALL_QUERY_MAX_TOKENS
            and num_query_heads <= 64
            and sparse_block_size == 1
            and num_sparse_blocks >= SMALL_QUERY_MIN_SPARSE_ENTRIES
            and num_sparse_blocks <= SMALL_QUERY_MAX_SPARSE_ENTRIES
            and kv_access_spec is None
            and key_rope_access_spec is None
        ):
            head_block, kv_block = SMALL_QUERY_HEAD_BLOCK, SMALL_QUERY_KV_BLOCK
            padded_sparse = triton.cdiv(num_sparse_blocks, kv_block) * kv_block
            head_groups = triton.cdiv(num_query_heads, head_block)
            splits = min(
                padded_sparse // kv_block,
                max(1, triton.cdiv(num_cube_cores, num_query_tokens * head_groups)),
            )
            if num_query_tokens > 2:
                # Balance complete Cube waves against KV blocks per split;
                # excess splits otherwise leave cores idle in the final wave.
                kv_blocks = padded_sparse // kv_block
                splits = min(
                    range(1, kv_blocks + 1),
                    key=lambda count: (
                        triton.cdiv(
                            num_query_tokens * head_groups * count, num_cube_cores
                        )
                        * triton.cdiv(kv_blocks, count),
                        count,
                    ),
                )
            programs = min(num_cube_cores, num_query_tokens * head_groups * splits)
            gathered_kv = allocate_empty(
                (num_query_tokens, padded_sparse, KV_DIM), query.dtype
            )
            gathered_rope = allocate_empty(
                (num_query_tokens, padded_sparse, ROPE_DIM), query.dtype
            )
            mask = allocate_empty((num_query_tokens, padded_sparse), torch.float32)
            kv_end = allocate_empty((num_query_tokens,), torch.int32)
            partial_out = allocate_empty(
                (num_query_tokens * splits, num_query_heads, KV_DIM), torch.float32
            )
            partial_max = allocate_empty(
                (num_query_tokens * splits, num_query_heads), torch.float32
            )
            partial_sum = allocate_empty(
                (num_query_tokens * splits, num_query_heads), torch.float32
            )
            _sfa_small_gather[(num_vector_cores,)](
                key,
                key_rope,
                logical_sparse_indices,
                block_table,
                gathered_kv,
                gathered_rope,
                mask,
                kv_end,
                query_lengths,
                kv_lengths,
                NUM_SPARSE_BLOCKS=num_sparse_blocks,
                PADDED_SPARSE_BLOCKS=padded_sparse,
                PAGE_SIZE=page_size,
                NUM_QUERY_TOKENS=num_query_tokens,
                NUM_REQUESTS=num_requests,
                QUERY_SEQ_CAPACITY=query_seq_capacity,
                QUERY_IS_TND=layout_query == "TND",
                HAS_QUERY_LENGTHS=actual_seq_lengths_query is not None,
                CAUSAL=sparse_mode == SPARSE_MODE_CAUSAL,
                PAGES_PER_REQUEST=pages_per_request,
                GATHER_BLOCK_ROWS=32,
                NUM_CORES=num_vector_cores,
            )
            _sfa_small_attention[(programs,)](
                query,
                query_rope,
                gathered_kv,
                gathered_rope,
                mask,
                partial_out,
                partial_max,
                partial_sum,
                NUM_QUERY_TOKENS=num_query_tokens,
                NUM_QUERY_HEADS=num_query_heads,
                PADDED_SPARSE_BLOCKS=padded_sparse,
                HEAD_BLOCK=head_block,
                KV_BLOCK=kv_block,
                NUM_SPLITS=splits,
                NUM_CORES=programs,
                SOFTMAX_SCALE=scale_value,
                limit_auto_multi_buffer_only_for_local_buffer=False,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
            )
            _sfa_small_combine[(num_query_tokens * head_groups,)](
                partial_out,
                partial_max,
                partial_sum,
                attention_output,
                softmax_max,
                softmax_sum,
                kv_end,
                NUM_QUERY_HEADS=num_query_heads,
                NUM_SPARSE_BLOCKS=num_sparse_blocks,
                NUM_SPLITS=splits,
                HEAD_BLOCK=head_block,
                RETURN_STATS=return_softmax_lse,
            )
            return attention_output, softmax_max, softmax_sum
        if not (
            paged
            and (
                num_query_heads == FAST_QUERY_HEADS
                or use_small_head_pipeline
                or use_cross_query_pipeline
            )
            and num_sparse_blocks == FAST_SPARSE_ENTRIES
            and page_size == FAST_PAGE_SIZE
            and sparse_block_size == 1
        ):
            num_programs = min(
                num_cube_cores,
                num_query_tokens
                * triton.cdiv(num_query_heads, GENERAL_HEAD_BLOCK_SIZE),
            )
            _sfa_general[(num_programs,)](
                query,
                query_rope,
                key,
                key_rope,
                logical_sparse_indices,
                block_table if paged else logical_sparse_indices,
                query_lengths,
                kv_lengths,
                attention_output,
                softmax_max,
                softmax_sum,
                NUM_REQUESTS=num_requests,
                NUM_QUERY_TOKENS=num_query_tokens,
                NUM_QUERY_HEADS=num_query_heads,
                QUERY_SEQ_CAPACITY=query_seq_capacity,
                KV_SEQ_CAPACITY=kv_seq_capacity,
                NUM_SPARSE_BLOCKS=num_sparse_blocks,
                PAGE_SIZE=page_size,
                PAGES_PER_REQUEST=pages_per_request,
                SPARSE_BLOCK_SIZE=sparse_block_size,
                QUERY_IS_TND=layout_query == "TND",
                KV_IS_TND=layout_kv == "TND",
                PAGED=paged,
                HAS_QUERY_LENGTHS=actual_seq_lengths_query is not None,
                HAS_KV_LENGTHS=actual_seq_lengths_kv is not None,
                CAUSAL=sparse_mode == SPARSE_MODE_CAUSAL,
                SOFTMAX_SCALE=scale_value,
                RETURN_SOFTMAX_STATS=return_softmax_lse,
                **cache_access_kwargs,
                HEAD_BLOCK_SIZE=GENERAL_HEAD_BLOCK_SIZE,
                # Sparse address tensors coexist with the FP32 accumulator in UB.
                KV_BLOCK_ROWS=(
                    GENERAL_WIDE_OFFSET_KV_ROWS
                    if generic_access_requires_int64_offsets
                    else (
                        GENERAL_STRIDED_KV_ROWS
                        if kv_access_spec is not None
                        or key_rope_access_spec is not None
                        else GENERAL_CONTIGUOUS_KV_ROWS
                    )
                ),
                NUM_CORES=num_programs,
                limit_auto_multi_buffer_only_for_local_buffer=False,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
            )
            return attention_output, softmax_max, softmax_sum
        query_meta_kwargs = dict(
            NUM_REQUESTS=num_requests,
            QUERY_SEQ_CAPACITY=query_seq_capacity,
            QUERY_IS_TND=layout_query == "TND",
            HAS_QUERY_LENGTHS=actual_seq_lengths_query is not None,
            CAUSAL=sparse_mode == SPARSE_MODE_CAUSAL,
        )
        shared_kv_row_layout = None
        cache_storage_row_span = key.shape[0] * FAST_PAGE_SIZE
        # Row-contiguous paged views can reuse native gather after remapping rows.
        if (
            num_query_tokens >= LARGE_QUERY_BATCH_THRESHOLD
            and key.shape[0] * FAST_PAGE_SIZE <= INT32_MAX
            and (kv_access_spec is not None or key_rope_access_spec is not None)
        ):

            def get_native_gather_row_layout(tensor, row_width):
                if (
                    tensor.stride(-1) != 1
                    or tensor.stride(0) % row_width
                    or tensor.stride(1) % row_width
                ):
                    return None
                # (rows_per_page, page_stride_rows, row_stride_rows), in rows.
                return (
                    FAST_PAGE_SIZE,
                    tensor.stride(0) // row_width,
                    tensor.stride(1) // row_width,
                )

            kv_row_layout, key_rope_row_layout = get_native_gather_row_layout(
                key, KV_DIM
            ), get_native_gather_row_layout(key_rope, ROPE_DIM)
            if kv_row_layout is not None and kv_row_layout == key_rope_row_layout:
                storage_row_span = (
                    (key.shape[0] - 1) * kv_row_layout[1]
                    + FAST_PAGE_OFFSET_MASK * kv_row_layout[2]
                    + 1
                )
                if storage_row_span <= INT32_MAX:
                    shared_kv_row_layout, cache_storage_row_span = (
                        kv_row_layout,
                        storage_row_span,
                    )
                    kv_access_spec = key_rope_access_spec = None
                    cache_access_kwargs = dict(
                        KV_ACCESS_SPEC=None, KEY_ROPE_ACCESS_SPEC=None
                    )
                    generic_access_requires_int64_offsets = False
        if (
            num_query_tokens == 1
            or kv_access_spec is not None
            or key_rope_access_spec is not None
        ):
            additive_attention_mask = allocate_empty(
                (num_query_tokens, FAST_SPARSE_ENTRIES), torch.float32
            )
            kv_end_exclusive = allocate_empty((num_query_tokens,), torch.int32)
            num_kv_splits = SINGLE_QUERY_KV_SPLITS if num_query_tokens == 1 else 1
            gathered_kv = allocate_empty(
                (num_query_tokens, FAST_SPARSE_ENTRIES, KV_DIM), query.dtype
            )
            gathered_key_rope = allocate_empty(
                (num_query_tokens, FAST_SPARSE_ENTRIES, ROPE_DIM), query.dtype
            )
            _sfa_gather_sparse_kv[(num_vector_cores,)](
                key,
                key_rope,
                logical_sparse_indices,
                block_table,
                gathered_kv,
                gathered_key_rope,
                additive_attention_mask,
                kv_end_exclusive,
                query_lengths,
                kv_lengths,
                NUM_QUERY_TOKENS=num_query_tokens,
                PAGES_PER_REQUEST=pages_per_request,
                **cache_access_kwargs,
                GATHER_BLOCK_ROWS=(
                    WIDE_OFFSET_GATHER_ROWS
                    if generic_access_requires_int64_offsets
                    else (
                        STRIDED_GATHER_ROWS
                        if kv_access_spec is not None
                        or key_rope_access_spec is not None
                        else CONTIGUOUS_GATHER_ROWS
                    )
                ),
                NUM_CORES=num_vector_cores,
                **query_meta_kwargs,
            )
            split_weighted_value_sum = (
                allocate_empty(
                    (num_query_tokens * num_kv_splits, FAST_QUERY_HEADS, KV_DIM),
                    torch.float32,
                )
                if num_kv_splits > 1
                else None
            )
            split_softmax_max = (
                allocate_empty(
                    (num_query_tokens * num_kv_splits, FAST_QUERY_HEADS), torch.float32
                )
                if num_kv_splits > 1
                else softmax_max
            )
            split_softmax_sum = (
                allocate_empty(
                    (num_query_tokens * num_kv_splits, FAST_QUERY_HEADS), torch.float32
                )
                if num_kv_splits > 1
                else softmax_sum
            )
            num_programs = min(
                num_cube_cores, num_query_tokens * GATHERED_HEAD_GROUPS * num_kv_splits
            )
            _sfa_gathered_attention[(num_programs,)](
                query,
                query_rope,
                gathered_kv,
                gathered_key_rope,
                split_weighted_value_sum,
                split_softmax_max,
                split_softmax_sum,
                attention_output,
                additive_attention_mask,
                kv_end_exclusive,
                softmax_max,
                softmax_sum,
                NUM_QUERY_TOKENS=num_query_tokens,
                NUM_KV_SPLITS=num_kv_splits,
                NUM_CORES=num_programs,
                SOFTMAX_SCALE=scale_value,
                RETURN_SOFTMAX_STATS=return_softmax_lse,
                limit_auto_multi_buffer_only_for_local_buffer=False,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
            )
        else:
            pipeline_heads = (
                SMALL_QUERY_HEAD_BLOCK if use_small_head_pipeline else num_query_heads
            )
            num_kv_splits = (
                1
                if use_small_head_pipeline
                else (
                    PIPELINE_FOUR_QUERY_SPLITS
                    if num_query_tokens == 4
                    else (PIPELINE_EIGHT_QUERY_SPLITS if num_query_tokens == 8 else 1)
                )
            )
            num_programs = min(num_cube_cores, num_query_tokens * num_kv_splits)
            workspace_padding = (
                LARGE_BATCH_WORKSPACE_PADDING
                if not (use_small_head_pipeline or use_cross_query_pipeline)
                and num_query_tokens >= LARGE_QUERY_BATCH_THRESHOLD
                else 0
            )
            kv_workspace = allocate_workspace(
                (
                    num_programs,
                    NUM_PIPELINE_SLOTS,
                    PIPELINE_KV_BLOCK_ROWS + workspace_padding,
                    KV_DIM,
                ),
                query.dtype,
                KV_WORKSPACE_ADDRESS_OFFSET_BYTES,
            )
            rope_workspace = allocate_workspace(
                (
                    num_programs,
                    NUM_PIPELINE_SLOTS,
                    PIPELINE_KV_BLOCK_ROWS + workspace_padding,
                    ROPE_DIM,
                ),
                query.dtype,
                ROPE_WORKSPACE_ADDRESS_OFFSET_BYTES,
            )
            score_workspace = allocate_workspace(
                (
                    num_programs,
                    NUM_PIPELINE_SLOTS,
                    pipeline_heads,
                    PIPELINE_KV_BLOCK_ROWS + workspace_padding,
                ),
                torch.float32,
                SCORE_WORKSPACE_ADDRESS_OFFSET_BYTES,
            )
            unnormalized_weights_workspace = allocate_workspace(
                (
                    num_programs,
                    NUM_PIPELINE_SLOTS,
                    pipeline_heads,
                    PIPELINE_KV_BLOCK_ROWS + workspace_padding,
                ),
                query.dtype,
                WEIGHTS_WORKSPACE_ADDRESS_OFFSET_BYTES,
            )
            block_weighted_value_sum_workspace = allocate_workspace(
                (
                    num_programs,
                    NUM_PIPELINE_SLOTS,
                    pipeline_heads,
                    KV_DIM + workspace_padding,
                ),
                torch.float32,
                PV_WORKSPACE_ADDRESS_OFFSET_BYTES,
            )
            softmax_state_workspace = allocate_workspace(
                (
                    num_programs * (2 if use_cross_query_pipeline else 1),
                    SOFTMAX_STATE_ROWS + workspace_padding,
                    pipeline_heads,
                ),
                torch.float32,
                SOFTMAX_WORKSPACE_ADDRESS_OFFSET_BYTES,
            )
            output_accumulator_workspace = allocate_workspace(
                (
                    num_programs * (2 if use_cross_query_pipeline else 1),
                    pipeline_heads,
                    KV_DIM + workspace_padding,
                ),
                torch.float32,
                OUTPUT_WORKSPACE_ADDRESS_OFFSET_BYTES,
            )
            split_weighted_value_sum = (
                allocate_empty(
                    (num_query_tokens * num_kv_splits, pipeline_heads, KV_DIM),
                    torch.float32,
                )
                if num_kv_splits > 1
                else None
            )
            split_softmax_max = (
                allocate_empty(
                    (num_query_tokens * num_kv_splits, pipeline_heads), torch.float32
                )
                if num_kv_splits > 1
                else softmax_max
            )
            split_softmax_sum = (
                allocate_empty(
                    (num_query_tokens * num_kv_splits, pipeline_heads), torch.float32
                )
                if num_kv_splits > 1
                else softmax_sum
            )
            packed_query_meta = None
            physical_row_indices = None
            use_prepared_physical_indices = (
                not use_small_head_pipeline
                and num_query_tokens >= LARGE_QUERY_BATCH_THRESHOLD
                and key.shape[0] * FAST_PAGE_SIZE <= INT32_MAX
            )
            if use_prepared_physical_indices:
                physical_row_indices = allocate_empty(
                    logical_sparse_indices.shape, torch.int32
                )
                packed_query_meta = allocate_empty((num_query_tokens,), torch.int32)
                prepare_indices_kernel = (
                    _sfa_prepare_indices
                    if shared_kv_row_layout is None
                    else _sfa_prepare_strided_indices
                )
                prepare_indices_kwargs = (
                    dict(
                        ALLOW_EXACT_SORT=not use_cross_query_pipeline
                        and cache_storage_row_span <= FP32_EXACT_INT_LIMIT
                    )
                    if shared_kv_row_layout is None
                    else dict(SHARED_KV_ROW_LAYOUT=shared_kv_row_layout)
                )
                prepare_indices_kernel[(min(num_vector_cores, num_query_tokens),)](
                    logical_sparse_indices,
                    block_table,
                    query_lengths,
                    kv_lengths,
                    physical_row_indices,
                    packed_query_meta,
                    NUM_QUERY_TOKENS=num_query_tokens,
                    PAGES_PER_REQUEST=pages_per_request,
                    NUM_CORES=min(num_vector_cores, num_query_tokens),
                    **prepare_indices_kwargs,
                    **query_meta_kwargs,
                )
            _sfa_pipeline[(num_programs,)](
                query,
                num_query_heads,
                pipeline_heads,
                use_cross_query_pipeline,
                query_rope,
                key,
                key_rope,
                logical_sparse_indices,
                physical_row_indices,
                block_table,
                kv_workspace,
                rope_workspace,
                score_workspace,
                unnormalized_weights_workspace,
                block_weighted_value_sum_workspace,
                softmax_state_workspace,
                output_accumulator_workspace,
                attention_output,
                split_weighted_value_sum,
                split_softmax_max,
                split_softmax_sum,
                query_lengths,
                packed_query_meta,
                kv_lengths,
                softmax_max,
                softmax_sum,
                USE_PREPARED_PHYSICAL_INDICES=use_prepared_physical_indices,
                CACHE_STORAGE_ROW_SPAN=cache_storage_row_span,
                NUM_QUERY_TOKENS=num_query_tokens,
                NUM_SPARSE_BLOCKS=FAST_SPARSE_ENTRIES,
                PAGE_SIZE=FAST_PAGE_SIZE,
                PAGES_PER_REQUEST=pages_per_request,
                SOFTMAX_SCALE=scale_value,
                KV_BLOCK_ROWS=PIPELINE_KV_BLOCK_ROWS,
                QK_BLOCK_COLUMNS=(
                    LARGE_BATCH_QK_COLUMNS
                    if not use_small_head_pipeline
                    and num_query_tokens >= LARGE_QUERY_BATCH_THRESHOLD
                    else SMALL_BATCH_QK_COLUMNS
                ),
                # Narrow offsets only when the complete cache fits in 2 GiB.
                # Wider gather rows fit UB with 32-bit offsets; keep the 64-bit path.
                GATHER_BLOCK_ROWS=(
                    WIDE_OFFSET_GATHER_ROWS
                    if generic_access_requires_int64_offsets
                    else (
                        STRIDED_GATHER_ROWS
                        if kv_access_spec is not None
                        or key_rope_access_spec is not None
                        else (
                            CONTIGUOUS_GATHER_ROWS
                            if not use_small_head_pipeline
                            and num_query_tokens >= LARGE_QUERY_BATCH_THRESHOLD
                            else SMALL_BATCH_GATHER_ROWS
                        )
                    )
                ),
                USE_INT32_OFFSETS=num_query_tokens >= LARGE_QUERY_BATCH_THRESHOLD
                and cache_storage_row_span * KV_DIM
                <= MAX_CACHE_ELEMENTS_FOR_INT32_OFFSETS
                and kv_access_spec is None
                and key_rope_access_spec is None,
                SOFTMAX_BLOCK_HEADS=(
                    SMALL_PIPELINE_VECTOR_HEADS
                    if use_small_head_pipeline
                    else PIPELINE_SOFTMAX_HEADS
                ),
                VALUE_BLOCK_COLUMNS=PIPELINE_VALUE_COLUMNS,
                NUM_CORES=num_programs,
                MANUAL_BLOCK_SYNC=True,
                OUTPUT_BLOCK_COLUMNS=(
                    LARGE_BATCH_OUTPUT_COLUMNS
                    if not use_small_head_pipeline
                    and num_query_tokens >= LARGE_QUERY_BATCH_THRESHOLD
                    else SMALL_BATCH_OUTPUT_COLUMNS
                ),
                OUTPUT_BLOCK_HEADS=(
                    SMALL_PIPELINE_VECTOR_HEADS
                    if use_small_head_pipeline
                    else (
                        LARGE_BATCH_OUTPUT_HEADS
                        if not use_small_head_pipeline
                        and num_query_tokens >= LARGE_QUERY_BATCH_THRESHOLD
                        else SMALL_BATCH_OUTPUT_HEADS
                    )
                ),
                WORKSPACE_PADDING=workspace_padding,
                NUM_KV_SPLITS=num_kv_splits,
                NUM_VECTOR_SUBCORES=VECTOR_SUBCORES_PER_CUBE,
                RETURN_SOFTMAX_STATS=return_softmax_lse,
                **query_meta_kwargs,
                disable_auto_cv_work_space_manage=use_prepared_physical_indices,
                disable_auto_inject_block_sync=True,
                multibuffer=False,
            )
        if num_kv_splits > 1:
            if num_kv_splits == COMBINE_TWO_SPLITS:
                _sfa_combine_two[(num_query_tokens * COMBINE_TWO_PROGRAMS_PER_QUERY,)](
                    split_weighted_value_sum,
                    split_softmax_max,
                    split_softmax_sum,
                    attention_output,
                    softmax_max,
                    softmax_sum,
                    query_lengths,
                    kv_lengths,
                    NUM_KV_SPLITS=num_kv_splits,
                    RETURN_SOFTMAX_STATS=return_softmax_lse,
                    **query_meta_kwargs,
                )
            else:
                _sfa_combine[(num_query_tokens * COMBINE_PROGRAMS_PER_QUERY,)](
                    split_weighted_value_sum,
                    split_softmax_max,
                    split_softmax_sum,
                    attention_output,
                    softmax_max,
                    softmax_sum,
                    query_lengths,
                    kv_lengths,
                    NUM_KV_SPLITS=num_kv_splits,
                    RETURN_SOFTMAX_STATS=return_softmax_lse,
                    **query_meta_kwargs,
                )
        return attention_output, softmax_max, softmax_sum
