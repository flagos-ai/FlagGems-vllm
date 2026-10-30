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


"""Shared SFA metadata, paged addressing, allocation-copy kernels and constants."""

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
INT64_ELEMENT_BYTES = 8
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
GENERAL_MIN_GATHER_ROWS = 256
GENERAL_MAX_GATHER_WORKSPACE_BYTES = 64 * 1024 * 1024
GENERAL_GATHERED_HEAD_BLOCK = 32
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
    if not HAS_KV_LENGTHS and KV_SEQ_CAPACITY > TRITON_INT32_MAX:
        # Large constexpr capacities otherwise infer uint32, which cannot be
        # mixed with the signed sparse-block divisor in address calculations.
        kv_length = tl.full((), KV_SEQ_CAPACITY, tl.int64)
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
def _sfa_sparse_cache_rows(
    SPARSE_INDICES,
    BLOCK_TABLE,
    query_token_idx,
    request_idx,
    kv_end_exclusive,
    kv_start_offset,
    sparse_positions,
    NUM_SPARSE_BLOCKS: tl.constexpr,
    SPARSE_BLOCK_SIZE: tl.constexpr,
    PAGED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGES_PER_REQUEST: tl.constexpr,
):
    """Expand global sparse positions and resolve their masked cache row addresses."""
    sparse_entry_positions = sparse_positions // SPARSE_BLOCK_SIZE
    logical_sparse_blocks = tl.load(
        SPARSE_INDICES + query_token_idx * NUM_SPARSE_BLOCKS + sparse_entry_positions,
        sparse_entry_positions < NUM_SPARSE_BLOCKS,
        other=TL_INVALID_SPARSE_INDEX,
    )
    logical_kv_indices = (
        logical_sparse_blocks.to(tl.int64) * SPARSE_BLOCK_SIZE
        + sparse_positions % SPARSE_BLOCK_SIZE
    )
    valid_kv_mask = (
        (sparse_entry_positions < NUM_SPARSE_BLOCKS)
        & (sparse_entry_positions < tl.cdiv(kv_end_exclusive, SPARSE_BLOCK_SIZE))
        & (logical_sparse_blocks >= 0)
        & (logical_kv_indices < kv_end_exclusive)
    )
    if PAGED:
        valid_kv_mask &= logical_kv_indices // PAGE_SIZE < PAGES_PER_REQUEST
    logical_kv_indices = tl.where(valid_kv_mask, logical_kv_indices, 0)
    if PAGED:
        physical_page_ids = tl.load(
            BLOCK_TABLE
            + request_idx * PAGES_PER_REQUEST
            + logical_kv_indices // PAGE_SIZE,
            valid_kv_mask,
            other=0,
        )
        cache_row_indices = (
            physical_page_ids.to(tl.int64) * PAGE_SIZE + logical_kv_indices % PAGE_SIZE
        )
    else:
        cache_row_indices = kv_start_offset + logical_kv_indices
    return cache_row_indices, valid_kv_mask


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
    GENERAL: tl.constexpr = False,
    SPARSE_BLOCK_SIZE: tl.constexpr = 1,
    PAGED: tl.constexpr = True,
    KV_IS_TND: tl.constexpr = False,
    KV_SEQ_CAPACITY: tl.constexpr = 0,
    HAS_KV_LENGTHS: tl.constexpr = True,
):
    MASKED_GATHER_BLOCK_ROWS: tl.constexpr = (
        TL_WIDE_OFFSET_GATHER_ROWS
        if GATHER_BLOCK_ROWS == TL_WIDE_OFFSET_GATHER_ROWS
        else TL_MASKED_GATHER_ROWS
    )
    program_id = tl.program_id(0)
    for token_idx in range(NUM_QUERY_TOKENS):
        query_token_idx = token_idx.to(tl.int64)
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
        if not PAGED and not KV_IS_TND:
            kv_start_offset = request_idx * KV_SEQ_CAPACITY
        if not GENERAL:
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
            if GENERAL:
                physical_row_indices, valid_kv_mask = _sfa_sparse_cache_rows(
                    SPARSE_INDICES,
                    BLOCK_TABLE,
                    query_token_idx,
                    request_idx,
                    kv_end_exclusive,
                    kv_start_offset,
                    sparse_positions,
                    NUM_SPARSE_BLOCKS,
                    SPARSE_BLOCK_SIZE,
                    PAGED,
                    PAGE_SIZE,
                    PAGES_PER_REQUEST,
                )
            else:
                logical_kv_indices = tl.load(
                    SPARSE_INDICES
                    + query_token_idx * NUM_SPARSE_BLOCKS
                    + sparse_positions,
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
            if not GENERAL:
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
            valid_rows = tl.sum(valid_kv_mask.to(tl.int32), 0)
            if GENERAL and valid_rows == 0:
                tl.store(
                    GATHERED_KV
                    + (
                        query_token_idx * PADDED_SPARSE_BLOCKS
                        + sparse_positions[:, None]
                    )
                    * TL_KV_DIM
                    + kv_columns[None, :],
                    0.0,
                )
                tl.store(
                    GATHERED_KEY_ROPE
                    + (
                        query_token_idx * PADDED_SPARSE_BLOCKS
                        + sparse_positions[:, None]
                    )
                    * TL_ROPE_DIM
                    + rope_columns[None, :],
                    0.0,
                )
            elif valid_rows == GATHER_BLOCK_ROWS:
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
                    if GENERAL:
                        subblock_physical_row_indices, subblock_valid_kv_mask = (
                            _sfa_sparse_cache_rows(
                                SPARSE_INDICES,
                                BLOCK_TABLE,
                                query_token_idx,
                                request_idx,
                                kv_end_exclusive,
                                kv_start_offset,
                                subblock_sparse_positions,
                                NUM_SPARSE_BLOCKS,
                                SPARSE_BLOCK_SIZE,
                                PAGED,
                                PAGE_SIZE,
                                PAGES_PER_REQUEST,
                            )
                        )
                    else:
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
                    if not GENERAL:
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
