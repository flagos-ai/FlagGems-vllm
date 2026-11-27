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

"""Hygon QSA prefill: row-wise scoring plus per-row top-k.

The generic implementation scores BLOCK_R rows of a request with a tiled
``tl.dot`` and selects with ``persistent_topk``.  On the DCU backend the
vendor prefill path is a per-row, per-head elementwise score followed by
``top_k_per_row_decode``; this module mirrors that pipeline so the benchmark
compares the same shape of computation on both sides.
"""

import os

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.qsa_select_paged_prefill import _validate_inputs
from flaggems_vllm.ops.top_k_per_row_decode import top_k_per_row_decode
from flaggems_vllm.runtime import torch_device_fn

_BLOCK_N = 32
_NUM_WARPS = 4


@triton.jit
def _prefill_rows_score_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    visible_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    query_offset,
    logits_width,
    num_pages,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    first = tl.load(query_start_loc_ptr)
    token = first + query_offset + row
    request = tl.zeros((), tl.int32)
    for r in tl.static_range(0, NUM_REQUESTS):
        request += (token >= tl.load(query_start_loc_ptr + r + 1)).to(tl.int32)
    visible = tl.load(visible_ptr + query_offset + row)
    if tile * BLOCK_N >= visible:
        tl.store(
            logits_ptr + row * stride_logits_row + cols,
            tl.full((BLOCK_N,), -float("inf"), tl.float32),
            mask=cols < logits_width,
        )
        return
    live = (cols < logits_width) & (cols < visible)
    logical_page = cols // PAGE_SIZE
    page_offset = cols % PAGE_SIZE
    physical_page = tl.load(
        page_table_ptr + request * stride_table_req + logical_page * stride_table_page,
        mask=live & (logical_page < PAGE_TABLE_WIDTH),
        other=-1,
    )
    live &= (physical_page >= 0) & (physical_page < num_pages)
    keys = tl.load(
        k_cache_ptr
        + physical_page[:, None].to(tl.int64) * stride_cache_block
        + page_offset[:, None] * stride_cache_token
        + dims[None, :],
        mask=live[:, None] & (dims[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    score = tl.zeros((BLOCK_N,), tl.float32)
    for head in tl.static_range(0, NUM_HEADS):
        query = tl.load(
            q_ptr + (query_offset + row) * stride_q_row + head * stride_q_head + dims,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)
    tl.store(
        logits_ptr + row * stride_logits_row + cols,
        tl.where(live, score * SCALE, -float("inf")),
        mask=cols < logits_width,
    )


def _score_rows(
    q,
    k_cache,
    page_table,
    query_start_loc,
    visible_blocks,
    logits,
    query_offset,
    num_queries,
    logits_width,
):
    grid = (num_queries, triton.cdiv(logits_width, _BLOCK_N))
    with torch_device_fn.device(q.device):
        _prefill_rows_score_kernel[grid](
            q,
            k_cache,
            page_table,
            query_start_loc,
            visible_blocks,
            logits,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            page_table.stride(0),
            page_table.stride(1),
            logits.stride(0),
            query_offset,
            logits_width,
            k_cache.shape[0],
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=page_table.shape[1],
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            NUM_REQUESTS=query_start_loc.numel() - 1,
            BLOCK_N=_BLOCK_N,
            BLOCK_D=max(16, triton.next_power_of_2(q.shape[2])),
            SCALE=q.shape[2] ** -0.5,
            num_warps=_NUM_WARPS,
        )


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
    """Write logical compressed-block indices selected for packed prefill Q."""
    rows, block_topk = _validate_inputs(
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
    try:
        budget_mb = int(os.environ.get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512"))
    except ValueError as exc:
        raise ValueError(
            "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB must be an integer"
        ) from exc
    if budget_mb < 0:
        raise ValueError("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB must be nonnegative")
    logits_width = min(capacity, triton.cdiv(max(64, visible_bound), 64) * 64)
    rows_per_chunk = min(rows, max(1, budget_mb * 1024 * 1024 // (logits_width * 4)))
    for query_offset in range(0, rows, rows_per_chunk):
        num_queries = min(rows_per_chunk, rows - query_offset)
        logits = torch.empty(
            (num_queries, logits_width), dtype=torch.float32, device=q.device
        )
        _score_rows(
            q,
            k_cache,
            page_table,
            query_start_loc,
            visible_blocks,
            logits,
            query_offset,
            num_queries,
            logits_width,
        )
        query_slice = slice(query_offset, query_offset + num_queries)
        with torch_device_fn.device(q.device):
            top_k_per_row_decode(
                logits,
                1,
                visible_blocks[query_slice],
                block_indices[query_slice],
                num_queries,
                logits.stride(0),
                logits.stride(1),
                block_topk,
            )


__all__ = ["qsa_select_paged_prefill"]
