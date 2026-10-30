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

"""Select compressed QSA key groups for packed BF16/FP8 prefill on NVIDIA SM90.

API and semantics follow vllm-project/vllm commit
7a877ae34dcf4a7d46de08eb16db4699e1e0047c,
vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py.

Scores are accumulated in FP32 and TopK uses the repository implementation.
Selection order and tie-breaking are unspecified. Device-resident metadata
must satisfy the public API preconditions. Autograd is not supported.
"""

import os

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.persistent_topk import HAS_TLE, persistent_topk
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.libentry import LibEntry


class _QSAScoreEntry(LibEntry):
    """Include score-tuner parameters in the outer runtime cache key."""

    def key(self, spec_args, dns_args, const_args):
        named = {
            self.arg_names[index]: value
            for index, value in zip(self.specialize_indices, spec_args)
        }
        named.update(
            (self.arg_names[index], value)
            for index, value in zip(self.do_not_specialize_indices, dns_args)
        )
        constexpr_names = (
            param.name for param in self.jit_function.params if param.is_constexpr
        )
        named.update(zip(constexpr_names, const_args))
        return super().key(spec_args, dns_args, const_args) + (self.fn.get_key(named),)


def _prune_head_group_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    eligible = not args["IS_FP8"] and args["NUM_HEADS"] == 4 and args["HEAD_DIM"] == 128
    return [
        c
        for c in configs
        if not c.kwargs["HEAD_GROUP"]
        or (eligible and c.kwargs["TILE_R"] == 64 and c.num_warps == 4)
    ]


def _prune_fp8_store_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    paired = args["IS_FP8"] and args["NUM_HEADS"] == 4 and args["HEAD_DIM"] == 128
    result = []
    for config in configs:
        meta = config.kwargs
        split = meta["QUERY_PARTS"] == 2
        if split and (not args["IS_FP8"] or meta["TILE_R"] < 32):
            continue
        added = (
            meta["TILE_R"] == 32
            and meta["BLOCK_N"] == 64
            and meta["K_TILES"] == 16
            and split
            and config.num_warps == 4
        )
        if not paired and added:
            continue
        if (
            paired
            and split
            and not (
                meta["TILE_R"] == 32 and meta["BLOCK_N"] == 64 and config.num_warps == 4
            )
        ):
            continue
        result.append(config)
    return result


@triton.jit
def _fp8_bulk_tail_store(ptrs, values, valid_rows, columns, visible, logits_width):
    limit = tl.minimum(visible, logits_width)
    full = tl.multiple_of(limit // 4 * 4, (4, 4))
    tl.store(ptrs, values, mask=valid_rows & (columns < full))
    tl.store(ptrs, values, mask=valid_rows & (columns >= full) & (columns < limit))


@_QSAScoreEntry
@libtuner(
    configs=runtime.get_tuned_config("qsa_select_paged_prefill_fp8_store_score"),
    key=[
        "num_rows",
        "max_query_len",
        "logits_width",
        "num_requests",
        "PAGE_SIZE",
        "NUM_HEADS",
        "HEAD_DIM",
        "IS_FP8",
        "stride_q_row",
        "stride_q_head",
        "stride_cache_block",
        "stride_cache_token",
        "stride_table_req",
        "stride_table_page",
    ],
    strategy="default",
    prune_configs_by={"early_config_prune": _prune_fp8_store_configs},
)
@triton.jit(do_not_specialize=["num_rows", "query_offset", "max_query_len"])
def _qsa_prefill_fp8_store_score_kernel(
    q_ptr,
    cache_ptr,
    table_ptr,
    starts_ptr,
    visible_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_table_page,
    num_rows,
    query_offset,
    max_query_len,
    logits_width,
    num_requests,
    num_pages,
    table_width,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    QUERY_PARTS: tl.constexpr,
):
    request = tl.program_id(0)
    query_base = tl.load(starts_ptr)
    request_start = tl.maximum(tl.load(starts_ptr + request) - query_base, query_offset)
    request_end = tl.minimum(
        tl.load(starts_ptr + request + 1) - query_base, query_offset + num_rows
    )
    row_start = request_start + tl.program_id(1) * TILE_R
    if row_start >= request_end:
        return

    HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires N>=16 even when TILE_R * HEADS_PADDED is smaller.
    # Fake rows do not read queries/metadata or write outside this query tile.
    BLOCK_R: tl.constexpr = max(TILE_R, triton.cdiv(16, HEADS_PADDED))
    SPLIT_QUERY: tl.constexpr = IS_FP8 and TILE_R >= 32 and QUERY_PARTS == 2
    PAIR_DOT: tl.constexpr = (
        NUM_HEADS == 4 and HEAD_DIM == 128 and TILE_R == 32 and BLOCK_N == 64
    )
    row_lanes = tl.arange(0, BLOCK_R)
    rows = row_start + row_lanes
    valid_rows = (row_lanes < TILE_R) & (rows < request_end)
    visible = tl.load(visible_ptr + rows, mask=valid_rows, other=0)
    max_visible = tl.minimum(tl.max(visible, axis=0), logits_width)
    first_tile = tl.program_id(2) * K_TILES
    if first_tile * BLOCK_N >= max_visible:
        return
    last_tile = tl.minimum(first_tile + K_TILES, tl.cdiv(max_visible, BLOCK_N))

    # Padded heads are adjacent in the dot result. Mask them before reduction.
    query_lanes = tl.arange(0, BLOCK_R * HEADS_PADDED)
    query_row_lanes = query_lanes // HEADS_PADDED
    query_rows = row_start + query_row_lanes
    heads = query_lanes % HEADS_PADDED
    # Pad the reduction dimension to the native dot instruction's minimum.
    BLOCK_D: tl.constexpr = max(32 if IS_FP8 else 16, triton.next_power_of_2(HEAD_DIM))
    dims = tl.arange(0, BLOCK_D)
    if SPLIT_QUERY:
        HALF_R: tl.constexpr = TILE_R // 2
        half_lanes = tl.arange(0, HALF_R * HEADS_PADDED)
        half_rows = half_lanes // HEADS_PADDED
        half_heads = half_lanes % HEADS_PADDED
        query_left = tl.load(
            q_ptr
            + (row_start + 0 + half_rows)[None, :] * stride_q_row
            + half_heads[None, :] * stride_q_head
            + dims[:, None],
            mask=((row_start + 0 + half_rows)[None, :] < request_end)
            & (half_heads[None, :] < NUM_HEADS)
            & (dims[:, None] < HEAD_DIM),
            other=0.0,
        )
        query_right = tl.load(
            q_ptr
            + (row_start + HALF_R + half_rows)[None, :] * stride_q_row
            + half_heads[None, :] * stride_q_head
            + dims[:, None],
            mask=((row_start + HALF_R + half_rows)[None, :] < request_end)
            & (half_heads[None, :] < NUM_HEADS)
            & (dims[:, None] < HEAD_DIM),
            other=0.0,
        )
        visible_left, visible_right = tl.split(
            tl.trans(tl.reshape(visible, (2, HALF_R)))
        )
    else:
        query = tl.load(
            q_ptr
            + query_rows[None, :] * stride_q_row
            + heads[None, :] * stride_q_head
            + dims[:, None],
            mask=(query_row_lanes[None, :] < TILE_R)
            & (query_rows[None, :] < request_end)
            & (heads[None, :] < NUM_HEADS)
            & (dims[:, None] < HEAD_DIM),
            other=0.0,
        )
    column_lanes = tl.arange(0, BLOCK_N)
    for tile in range(first_tile, last_tile):
        columns = tile * BLOCK_N + column_lanes
        logical_page = columns // PAGE_SIZE
        page_offset = columns % PAGE_SIZE
        live = (columns < max_visible) & (logical_page < table_width)
        if not IS_FP8 and BLOCK_N <= PAGE_SIZE:
            first_page = tile * BLOCK_N // PAGE_SIZE
            second_page = first_page + 1
            first_id = tl.load(
                table_ptr + request * stride_table_req + first_page * stride_table_page,
                mask=(tile * BLOCK_N < max_visible) & (first_page < table_width),
                other=0,
            )
            second_id = tl.load(
                table_ptr
                + request * stride_table_req
                + second_page * stride_table_page,
                mask=(second_page < table_width)
                & (
                    second_page * PAGE_SIZE
                    < tl.minimum((tile + 1) * BLOCK_N, max_visible)
                ),
                other=0,
            )
            physical_page = tl.where(logical_page == first_page, first_id, second_id)
        else:
            physical_page = tl.load(
                table_ptr
                + request * stride_table_req
                + logical_page * stride_table_page,
                mask=live,
                other=0,
            )
        live &= (physical_page >= 0) & (physical_page < num_pages)
        keys = tl.load(
            cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        )
        if SPLIT_QUERY:
            scores_left = tl.dot(keys, query_left, out_dtype=tl.float32)
            if PAIR_DOT:
                scores_right = tl.dot(keys, query_right, out_dtype=tl.float32)
            scores_left = tl.reshape(scores_left, (BLOCK_N, HALF_R, HEADS_PADDED))
            real_heads_left = tl.arange(0, HEADS_PADDED) < NUM_HEADS
            scores_left = tl.where(
                real_heads_left[None, None, :], tl.maximum(scores_left, 0.0), 0.0
            )
            scores_left = tl.sum(scores_left, axis=2)
            rows_left = row_start + 0 + tl.arange(0, HALF_R)
            _fp8_bulk_tail_store(
                logits_ptr
                + (rows_left - query_offset)[None, :] * logits_width
                + columns[:, None],
                tl.where(live[:, None], scores_left, -float("inf")),
                valid_rows=(rows_left[None, :] < request_end),
                columns=columns[:, None],
                visible=visible_left[None, :],
                logits_width=logits_width,
            )
            if not PAIR_DOT:
                scores_right = tl.dot(keys, query_right, out_dtype=tl.float32)
            scores_right = tl.reshape(scores_right, (BLOCK_N, HALF_R, HEADS_PADDED))
            real_heads_right = tl.arange(0, HEADS_PADDED) < NUM_HEADS
            scores_right = tl.where(
                real_heads_right[None, None, :], tl.maximum(scores_right, 0.0), 0.0
            )
            scores_right = tl.sum(scores_right, axis=2)
            rows_right = row_start + HALF_R + tl.arange(0, HALF_R)
            _fp8_bulk_tail_store(
                logits_ptr
                + (rows_right - query_offset)[None, :] * logits_width
                + columns[:, None],
                tl.where(live[:, None], scores_right, -float("inf")),
                valid_rows=(rows_right[None, :] < request_end),
                columns=columns[:, None],
                visible=visible_right[None, :],
                logits_width=logits_width,
            )
        else:
            scores = tl.dot(keys, query, out_dtype=tl.float32)
            scores = tl.reshape(scores, (BLOCK_N, BLOCK_R, HEADS_PADDED))
            real_heads = tl.arange(0, HEADS_PADDED) < NUM_HEADS
            scores = tl.where(real_heads[None, None, :], tl.maximum(scores, 0.0), 0.0)
            scores = tl.sum(scores, axis=2)
            _fp8_bulk_tail_store(
                logits_ptr
                + (rows - query_offset)[None, :] * logits_width
                + columns[:, None],
                tl.where(live[:, None], scores, -float("inf")),
                valid_rows=valid_rows[None, :],
                columns=columns[:, None],
                visible=visible[None, :],
                logits_width=logits_width,
            )


# Fixed-size bandwidth-only fills: no autotune. They do not perform selection.
@triton.jit
def _select_all_kernel(visible_ptr, output_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    visible = tl.load(visible_ptr + row)
    tl.store(
        output_ptr + row * K + columns,
        tl.where(columns < visible, columns, -1),
        mask=columns < K,
    )


@_QSAScoreEntry
@libtuner(
    configs=runtime.get_tuned_config("qsa_select_paged_prefill_head_group_score"),
    key=[
        "num_rows",
        "max_query_len",
        "logits_width",
        "num_requests",
        "PAGE_SIZE",
        "NUM_HEADS",
        "HEAD_DIM",
        "IS_FP8",
        "stride_q_row",
        "stride_q_head",
        "stride_cache_block",
        "stride_cache_token",
        "stride_table_req",
        "stride_table_page",
    ],
    strategy="default",
    prune_configs_by={"early_config_prune": _prune_head_group_configs},
)
@triton.jit(do_not_specialize=["num_rows", "query_offset", "max_query_len"])
def _qsa_prefill_score_kernel(
    q_ptr,
    cache_ptr,
    table_ptr,
    starts_ptr,
    visible_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_table_page,
    num_rows,
    query_offset,
    max_query_len,
    logits_width,
    num_requests,
    num_pages,
    table_width,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    HEAD_GROUP: tl.constexpr = False,
):
    request = tl.program_id(0)
    query_base = tl.load(starts_ptr)
    request_start = tl.maximum(tl.load(starts_ptr + request) - query_base, query_offset)
    request_end = tl.minimum(
        tl.load(starts_ptr + request + 1) - query_base, query_offset + num_rows
    )
    row_start = request_start + tl.program_id(1) * TILE_R
    if row_start >= request_end:
        return

    GROUP_HEADS: tl.constexpr = (
        HEAD_GROUP
        and not IS_FP8
        and NUM_HEADS == 4
        and HEAD_DIM == 128
        and TILE_R == 64
    )
    HEADS_PADDED: tl.constexpr = 2 if GROUP_HEADS else triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires N>=16 even when TILE_R * HEADS_PADDED is smaller.
    # Fake rows do not read queries/metadata or write outside this query tile.
    BLOCK_R: tl.constexpr = max(TILE_R, triton.cdiv(16, HEADS_PADDED))
    row_lanes = tl.arange(0, BLOCK_R)
    rows = row_start + row_lanes
    valid_rows = (row_lanes < TILE_R) & (rows < request_end)
    visible = tl.load(visible_ptr + rows, mask=valid_rows, other=0)
    max_visible = tl.minimum(tl.max(visible, axis=0), logits_width)
    first_tile = tl.program_id(2) * K_TILES
    if first_tile * BLOCK_N >= max_visible:
        return
    last_tile = tl.minimum(first_tile + K_TILES, tl.cdiv(max_visible, BLOCK_N))

    # Padded heads are adjacent in the dot result. Mask them before reduction.
    query_lanes = tl.arange(0, BLOCK_R * HEADS_PADDED)
    query_row_lanes = query_lanes // HEADS_PADDED
    query_rows = row_start + query_row_lanes
    heads = query_lanes % HEADS_PADDED
    # Pad the reduction dimension to the native dot instruction's minimum.
    BLOCK_D: tl.constexpr = max(32 if IS_FP8 else 16, triton.next_power_of_2(HEAD_DIM))
    dims = tl.arange(0, BLOCK_D)
    query = tl.load(
        q_ptr
        + query_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None],
        mask=(query_row_lanes[None, :] < TILE_R)
        & (query_rows[None, :] < request_end)
        & (heads[None, :] < NUM_HEADS)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    if GROUP_HEADS:
        query_high = tl.load(
            q_ptr
            + query_rows[None, :] * stride_q_row
            + (heads[None, :] + 2) * stride_q_head
            + dims[:, None],
            mask=(query_row_lanes[None, :] < TILE_R)
            & (query_rows[None, :] < request_end)
            & (heads[None, :] < NUM_HEADS)
            & (dims[:, None] < HEAD_DIM),
            other=0.0,
        )
    column_lanes = tl.arange(0, BLOCK_N)
    for tile in range(first_tile, last_tile):
        columns = tile * BLOCK_N + column_lanes
        logical_page = columns // PAGE_SIZE
        page_offset = columns % PAGE_SIZE
        live = (columns < max_visible) & (logical_page < table_width)
        physical_page = tl.load(
            table_ptr + request * stride_table_req + logical_page * stride_table_page,
            mask=live,
            other=0,
        )
        live &= (physical_page >= 0) & (physical_page < num_pages)
        keys = tl.load(
            cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        )
        if GROUP_HEADS:
            low = tl.dot(keys, query, out_dtype=tl.float32)
            low = tl.sum(
                tl.maximum(tl.reshape(low, (BLOCK_N, BLOCK_R, 2)), 0.0), axis=2
            )
            high = tl.dot(keys, query_high, out_dtype=tl.float32)
            high = tl.sum(
                tl.maximum(tl.reshape(high, (BLOCK_N, BLOCK_R, 2)), 0.0), axis=2
            )
            scores = low + high
        else:
            scores = tl.dot(keys, query, out_dtype=tl.float32)
            scores = tl.reshape(scores, (BLOCK_N, BLOCK_R, HEADS_PADDED))
            real_heads = tl.arange(0, HEADS_PADDED) < NUM_HEADS
            scores = tl.where(real_heads[None, None, :], tl.maximum(scores, 0.0), 0.0)
            scores = tl.sum(scores, axis=2)
        tl.store(
            logits_ptr
            + (rows - query_offset)[None, :] * logits_width
            + columns[:, None],
            tl.where(live[:, None], scores, -float("inf")),
            mask=valid_rows[None, :]
            & (columns[:, None] < visible[None, :])
            & (columns[:, None] < logits_width),
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("qsa_select_paged_prefill_score"),
    key=[
        "num_rows",
        "max_query_len",
        "logits_width",
        "num_requests",
        "PAGE_SIZE",
        "NUM_HEADS",
        "HEAD_DIM",
        "IS_FP8",
        "stride_q_row",
        "stride_q_head",
        "stride_cache_block",
        "stride_cache_token",
        "stride_table_req",
        "stride_table_page",
    ],
    strategy="default",
)
@triton.jit(do_not_specialize=["num_rows", "query_offset", "max_query_len"])
def _qsa_prefill_compact_score_kernel(
    q_ptr,
    cache_ptr,
    table_ptr,
    starts_ptr,
    visible_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_table_page,
    num_rows,
    query_offset,
    max_query_len,
    logits_width,
    num_requests,
    num_pages,
    table_width,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
):
    # Monotone slot boundaries include one padding slot per request. This
    # avoids the rectangular requests * max_query_len grid without a prefix
    # scan, host metadata reads, atomics or an extra launch/workspace.
    slot = tl.program_id(0)
    query_base = tl.load(starts_ptr)
    low, high = 0, num_requests
    while low + 1 < high:
        middle = (low + high) // 2
        start = tl.load(starts_ptr + middle) - query_base - query_offset
        start = tl.minimum(tl.maximum(start, 0), num_rows)
        boundary = start // TILE_R + middle
        if boundary <= slot:
            low = middle
        else:
            high = middle
    request = low
    start = tl.load(starts_ptr + request) - query_base - query_offset
    clipped_start = tl.minimum(tl.maximum(start, 0), num_rows)
    first_slot = clipped_start // TILE_R + request
    request_start = query_offset + clipped_start
    request_end = tl.minimum(
        tl.load(starts_ptr + request + 1) - query_base, query_offset + num_rows
    )
    row_start = request_start + (slot - first_slot) * TILE_R
    if row_start >= request_end:
        return

    HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires M>=16 even when TILE_R * HEADS_PADDED is smaller.
    # Fake rows do not read queries/metadata or write outside this query tile.
    BLOCK_R: tl.constexpr = max(TILE_R, triton.cdiv(16, HEADS_PADDED))
    row_lanes = tl.arange(0, BLOCK_R)
    rows = row_start + row_lanes
    valid_rows = (row_lanes < TILE_R) & (rows < request_end)
    visible = tl.load(visible_ptr + rows, mask=valid_rows, other=0)
    max_visible = tl.minimum(tl.max(visible, axis=0), logits_width)
    first_tile = tl.program_id(2) * K_TILES
    if first_tile * BLOCK_N >= max_visible:
        return
    last_tile = tl.minimum(first_tile + K_TILES, tl.cdiv(max_visible, BLOCK_N))

    # Padded heads are adjacent in the dot result. Mask them before reduction.
    query_lanes = tl.arange(0, BLOCK_R * HEADS_PADDED)
    query_row_lanes = query_lanes // HEADS_PADDED
    query_rows = row_start + query_row_lanes
    heads = query_lanes % HEADS_PADDED
    # Pad the reduction dimension to the native dot instruction's minimum.
    BLOCK_D: tl.constexpr = max(32 if IS_FP8 else 16, triton.next_power_of_2(HEAD_DIM))
    dims = tl.arange(0, BLOCK_D)
    query = tl.load(
        q_ptr
        + query_rows[:, None] * stride_q_row
        + heads[:, None] * stride_q_head
        + dims[None, :],
        mask=(query_row_lanes[:, None] < TILE_R)
        & (query_rows[:, None] < request_end)
        & (heads[:, None] < NUM_HEADS)
        & (dims[None, :] < HEAD_DIM),
        other=0.0,
    )
    column_lanes = tl.arange(0, BLOCK_N)
    for tile in range(first_tile, last_tile):
        columns = tile * BLOCK_N + column_lanes
        logical_page = columns // PAGE_SIZE
        page_offset = columns % PAGE_SIZE
        live = (columns < max_visible) & (logical_page < table_width)
        physical_page = tl.load(
            table_ptr + request * stride_table_req + logical_page * stride_table_page,
            mask=live,
            other=0,
        )
        live &= (physical_page >= 0) & (physical_page < num_pages)
        keys = tl.load(
            cache_ptr
            + physical_page[None, :].to(tl.int64) * stride_cache_block
            + page_offset[None, :] * stride_cache_token
            + dims[:, None],
            mask=live[None, :] & (dims[:, None] < HEAD_DIM),
            other=0.0,
        )
        scores = tl.dot(query, keys, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_R, HEADS_PADDED, BLOCK_N))
        real_heads = tl.arange(0, HEADS_PADDED) < NUM_HEADS
        scores = tl.where(real_heads[None, :, None], tl.maximum(scores, 0.0), 0.0)
        scores = tl.sum(scores, axis=1)
        tl.store(
            logits_ptr
            + (rows - query_offset)[:, None] * logits_width
            + columns[None, :],
            tl.where(live[None, :], scores, -float("inf")),
            mask=valid_rows[:, None]
            & (columns[None, :] < visible[:, None])
            & (columns[None, :] < logits_width),
        )


def _compact_eligible(total_rows, requests, max_query_len, chunk_rows):
    # Use compact grids when rectangular capacity is at least four times the
    # chunk size, so fewer programs can offset the request-lookup overhead.
    return (
        requests > 1
        and total_rows < requests * max_query_len
        and chunk_rows > max_query_len
        and requests * max_query_len >= 4 * chunk_rows
    )


def _check_nonoverlapping_strides(tensor, name):
    # Empty tensors also undergo the stride check: contiguity alone does not
    # establish that their declared dimensions have nonoverlapping strides.
    if tensor.is_contiguous() and tensor.numel():
        return
    extent = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size > 1:
            if stride < extent:
                raise NotImplementedError(f"{name} must have nonoverlapping strides")
            extent += (size - 1) * stride


def _output_overlaps_inputs(output, inputs):
    """Check conservative byte hulls after output contiguity validation.

    Bounds are local to this call: no tensor/storage metadata is cached across
    invocations. Padding holes in a strided input remain part of its hull.
    """
    count = output.numel()
    if not count:
        return False
    output_start = output.data_ptr()
    output_end = output_start + count * output.element_size()
    for tensor in inputs:
        count = tensor.numel()
        if not count:
            continue
        start = tensor.data_ptr()
        if start >= output_end:
            continue
        if not tensor.is_contiguous():
            count = 1 + sum(
                (size - 1) * stride
                for size, stride in zip(tensor.shape, tensor.stride())
            )
        if output_start < start + count * tensor.element_size():
            return True
    return False


def _validate_inputs(
    q,
    k_cache,
    page_table,
    query_start_loc,
    visible_blocks,
    token_topk,
    compress_ratio,
    max_query_len,
    block_indices,
    max_seq_len,
):
    tensors = (q, k_cache, page_table, query_start_loc, visible_blocks, block_indices)
    if q.device.type != "cuda" or runtime.device.vendor_name != "nvidia":
        raise NotImplementedError("QSA prefill currently requires NVIDIA CUDA SM90")
    if any(t.device != q.device for t in tensors):
        raise ValueError("QSA prefill tensors must be on the same device")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise NotImplementedError("QSA prefill currently supports SM90 (H800/H100)")
    if q.dtype != k_cache.dtype:
        raise ValueError("Q and the compressed K cache must match dtype")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise NotImplementedError("QSA prefill supports BF16/FP8 E4M3FN query/cache")
    if q.ndim != 3:
        raise NotImplementedError("QSA prefill query must be [tokens, heads, head_dim]")
    if q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA prefill heads and head_dim must be positive")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, q.shape[2]):
        raise NotImplementedError(
            "QSA prefill cache must be [pages, page_size, 1, D] matching Q"
        )
    if q.requires_grad or k_cache.requires_grad:
        raise NotImplementedError("QSA prefill is inference-only")
    for name, value in (
        ("token_topk", token_topk),
        ("compress_ratio", compress_ratio),
        ("max_query_len", max_query_len),
        ("max_seq_len", max_seq_len),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if compress_ratio <= 0 or token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("token_topk must be positive and divisible by compress_ratio")
    k = token_topk // compress_ratio
    if k not in (512, 1024, 2048):
        raise NotImplementedError("QSA prefill supports block Top-K 512/1024/2048")
    rows = q.shape[0]
    if page_table.ndim != 2:
        raise ValueError("page_table must be [requests, max_pages]")
    if page_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("page_table must use int32/int64")
    if query_start_loc.shape != (page_table.shape[0] + 1,):
        raise ValueError("query_start_loc must contain requests + 1 offsets")
    if query_start_loc.dtype not in (torch.int32, torch.int64):
        raise TypeError("query_start_loc must use int32/int64")
    if visible_blocks.shape != (rows,) or visible_blocks.dtype != torch.int32:
        raise ValueError("visible_blocks must be int32 [tokens]")
    if block_indices.shape != (rows, k) or block_indices.dtype != torch.int32:
        raise ValueError("block_indices must be int32 [tokens, block_topk]")
    if max_query_len < 0 or max_seq_len < 0:
        raise ValueError("maximum lengths must be nonnegative")
    if rows and (
        not all(k_cache.shape[:2])
        or not all(page_table.shape)
        or max_query_len == 0
        or max_seq_len == 0
    ):
        raise ValueError(
            "nonempty queries require positive cache/table/maximum lengths"
        )
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1:
        raise NotImplementedError("QSA query/cache require unit innermost stride")
    for name, tensor in (("q", q), ("cache", k_cache), ("page_table", page_table)):
        _check_nonoverlapping_strides(tensor, name)
    for tensor in (query_start_loc, visible_blocks, block_indices):
        if not tensor.is_contiguous():
            raise NotImplementedError("QSA metadata/output must be contiguous")
    if _output_overlaps_inputs(block_indices, tensors[:-1]):
        raise ValueError("block_indices must not overlap any input")
    return rows, k


def _prefill_score_metadata(q, k_cache, page_table):
    """Scalar launch parameters, scoped to one validated public call."""
    return (
        (
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            page_table.stride(0),
            page_table.stride(1),
        ),
        (page_table.shape[0], k_cache.shape[0], page_table.shape[1]),
        dict(
            PAGE_SIZE=k_cache.shape[1],
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            IS_FP8=q.dtype == torch.float8_e4m3fn,
        ),
    )


def _fp8_store_eligible(q, logits, logits_width):
    return (
        q.dtype == torch.float8_e4m3fn
        and logits.dtype == torch.float32
        and logits.is_contiguous()
        and logits.ndim == 2
        and logits.shape[1] == logits_width
        and logits_width > 0
        and logits_width % 4 == 0
        and logits.data_ptr() % 16 == 0
    )


def _prefill_logits(
    q,
    k_cache,
    page_table,
    query_start_loc,
    visible_blocks,
    logits_workspace,
    max_query_len,
    logits_width,
    query_offset,
    num_queries,
    *,
    launch_metadata=None,
):
    """Internal score launcher; inputs have already passed host validation."""
    if logits_workspace.ndim == 2:
        logits = (
            logits_workspace
            if num_queries == logits_workspace.shape[0]
            else logits_workspace[:num_queries]
        )
    else:
        # A flat workspace contains num_queries consecutive logits rows.
        logits = logits_workspace[: num_queries * logits_width].view(
            num_queries, logits_width
        )
    if launch_metadata is None:
        launch_metadata = _prefill_score_metadata(q, k_cache, page_table)
    strides, layout, constants = launch_metadata
    # Request lookup adds a binary search, so use a compact grid only when
    # the rectangular grid leaves substantial capacity unused.
    compact = _compact_eligible(q.shape[0], layout[0], max_query_len, num_queries)
    if compact:
        kernel = _qsa_prefill_compact_score_kernel
        grid = lambda meta: (
            num_queries // meta["TILE_R"] + layout[0],
            1,
            triton.cdiv(logits_width, meta["BLOCK_N"] * meta["K_TILES"]),
        )
    else:
        kernel = (
            _qsa_prefill_fp8_store_score_kernel
            if _fp8_store_eligible(q, logits, logits_width)
            else _qsa_prefill_score_kernel
        )
        grid = lambda meta: (
            layout[0],
            triton.cdiv(min(num_queries, max_query_len), meta["TILE_R"]),
            triton.cdiv(logits_width, meta["BLOCK_N"] * meta["K_TILES"]),
        )
    kernel[grid](
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        logits,
        *strides,
        num_queries,
        query_offset,
        max_query_len,
        logits_width,
        *layout,
        **constants,
    )
    return logits


def qsa_select_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_query_len: int,
    block_indices: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Write logical compressed-block indices selected for packed prefill Q.

    Matching BF16/FP8 Q is [tokens,H,D], K is [pages,page_size,1,D], H,D>0.
    Heads and dimensions are inferred from Q; compilation must fit device
    resources. Padded head/dimension lanes do not contribute to the score.
    ``query_start_loc`` is [requests+1] and its first entry may be nonzero.
    Caller-provided ``visible_blocks`` is int32 [tokens]. ``block_indices`` is
    contiguous int32 [tokens,token_topk//compress_ratio]. Short rows are padded with -1.
    The caller must supply valid device-resident metadata and finite Q/K.
    Only logits workspace is limited by VLLM_SPARSE_INDEXER_MAX_LOGITS_MB;
    a zero budget selects one row per chunk, as in vLLM. At least one row is
    allocated, and the existing Top-K operator allocates additional scratch.
    """
    rows, k = _validate_inputs(
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        token_topk,
        compress_ratio,
        max_query_len,
        block_indices,
        max_seq_len,
    )
    if rows == 0:
        return
    capacity = page_table.shape[1] * k_cache.shape[1]
    visible_bound = min(capacity, triton.cdiv(max_seq_len, compress_ratio))
    with torch.cuda.device(q.device):
        if visible_bound <= k:
            _select_all_kernel[(rows, triton.cdiv(k, 256))](
                visible_blocks, block_indices, K=k, BLOCK=256, num_warps=4
            )
            return
        if not HAS_TLE:
            raise NotImplementedError("QSA prefill Top-K requires Triton TLE")
        try:
            budget_mb = int(os.environ.get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512"))
        except ValueError as exc:
            raise ValueError(
                "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB must be an integer"
            ) from exc
        if budget_mb < 0:
            raise ValueError("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB must be nonnegative")
        logits_width = min(capacity, triton.cdiv(max(64, visible_bound), 64) * 64)
        rows_per_chunk = min(
            rows, max(1, budget_mb * 1024 * 1024 // (logits_width * 4))
        )
        workspace = torch.empty(
            (rows_per_chunk, logits_width), dtype=torch.float32, device=q.device
        )
        topk_workspace = torch.empty((1024 * 1024,), dtype=torch.uint8, device=q.device)
        launch_metadata = _prefill_score_metadata(q, k_cache, page_table)
        for query_offset in range(0, rows, rows_per_chunk):
            num_queries = min(rows_per_chunk, rows - query_offset)
            logits = _prefill_logits(
                q,
                k_cache,
                page_table,
                query_start_loc,
                visible_blocks,
                workspace,
                max_query_len,
                logits_width,
                query_offset,
                num_queries,
                launch_metadata=launch_metadata,
            )
            query_slice = slice(query_offset, query_offset + num_queries)
            persistent_topk(
                logits,
                visible_blocks[query_slice],
                block_indices[query_slice],
                topk_workspace,
                k=k,
                max_seq_len=logits_width,
            )


__all__ = ["qsa_select_paged_prefill"]
