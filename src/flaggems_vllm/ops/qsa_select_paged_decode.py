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

from flaggems_vllm.runtime import torch_device_fn

try:
    from flaggems_vllm.ops.top_k_per_row_decode import top_k_per_row_decode

    _HAS_TOP_K_PER_ROW = True
except Exception:
    _HAS_TOP_K_PER_ROW = False

logger = logging.getLogger(__name__)

_MAX_BLOCK_TOPK = 4096
_QSA_SUPPORTED_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)

_SCORE_BLOCK_N = 128
_SCORE_NUM_WARPS = 4
_SMALL_POOL_MAX_BLOCKS = 1000


_LOGITS_CACHE: dict = {}
_LOGITS_CACHE_MAX = 64


def _get_logits(rows: int, columns: int, device) -> torch.Tensor:
    key = (rows, columns, device)
    buf = _LOGITS_CACHE.get(key)
    if buf is None:
        if len(_LOGITS_CACHE) >= _LOGITS_CACHE_MAX:
            _LOGITS_CACHE.clear()
        buf = torch.empty((rows, columns), dtype=torch.float32, device=device)
        _LOGITS_CACHE[key] = buf
    return buf


def _pick_score_plan(num_blocks: int, num_requests: int):
    """Score-variant dispatch table over (num_blocks, num_requests),
    measured on H800 across the production shape trace.

    Returns ("loop", block_n, tiles_per_prog, num_warps, stages) for the
    pipelined-loop kernel or ("tile", block_n, num_warps) for the
    one-tile-per-program kernel.
    """
    if num_blocks <= _SMALL_POOL_MAX_BLOCKS:
        buckets = (
            (16, ("loop", 64, 1, 4, 2)),
            (80, ("loop", 256, 4, 4, 2)),
            (192, ("loop", 64, 8, 2, 2)),
            (367, ("loop", 64, 8, 4, 2)),
            (None, ("loop", 128, 8, 2, 2)),
        )
    else:
        buckets = (
            (6, ("loop", 64, 1, 4, 2)),
            (44, ("loop", 64, 2, 4, 2)),
            (80, ("loop", 64, 16, 4, 2)),
            (112, ("loop", 64, 16, 2, 3)),
            (148, ("loop", 64, 16, 2, 2)),
            (212, ("loop", 64, 16, 4, 3)),
            (288, ("loop", 64, 8, 2, 2)),
            (385, ("loop", 256, 8, 4, 2)),
            (None, ("tile", _SCORE_BLOCK_N, _SCORE_NUM_WARPS)),
        )
    for limit, plan in buckets:
        if limit is None or num_requests <= limit:
            return plan
    raise AssertionError("unreachable")


def _eager_impl(
    q,
    k_cache,
    page_table,
    visible_blocks,
    block_indices,
    decode_query_len,
    block_topk,
):
    """Pure eager body — no graph dispatch, no validation."""
    rows = q.shape[0]
    num_requests = rows // decode_query_len
    columns = page_table.shape[1] * k_cache.shape[1]
    logits = _get_logits(rows, columns, q.device)
    plan = _pick_score_plan(k_cache.shape[0], num_requests)

    with torch_device_fn.device(q.device):
        if plan[0] == "loop":
            _, block_n, tpp, num_warps, stages = plan
            _qsa_mqa_paged_score_loop_kernel[
                (num_requests, triton.cdiv(columns, block_n * tpp))
            ](
                q,
                k_cache,
                page_table,
                visible_blocks,
                logits,
                q.stride(0),
                q.stride(1),
                k_cache.stride(0),
                k_cache.stride(1),
                page_table.stride(0),
                logits.stride(0),
                PAGE_SIZE=k_cache.shape[1],
                PAGE_TABLE_WIDTH=page_table.shape[1],
                NUM_HEADS=q.shape[1],
                HEAD_DIM=q.shape[2],
                DECODE_QUERY_LEN=decode_query_len,
                BLOCK_N=block_n,
                TILES_PER_PROG=tpp,
                STAGES=stages,
                num_warps=num_warps,
            )
        else:
            _, block_n, num_warps = plan
            _qsa_mqa_paged_score_tile_kernel[(rows, triton.cdiv(columns, block_n))](
                q,
                k_cache,
                page_table,
                visible_blocks,
                logits,
                q.stride(0),
                q.stride(1),
                k_cache.stride(0),
                k_cache.stride(1),
                page_table.stride(0),
                logits.stride(0),
                columns,
                decode_query_len,
                PAGE_SIZE=k_cache.shape[1],
                PAGE_TABLE_WIDTH=page_table.shape[1],
                NUM_HEADS=q.shape[1],
                HEAD_DIM=q.shape[2],
                BLOCK_H=max(16, triton.next_power_of_2(q.shape[1])),
                BLOCK_N=block_n,
                BLOCK_D=max(16, triton.next_power_of_2(q.shape[2])),
                num_warps=num_warps,
            )
        top_k_per_row_decode(
            logits,
            1,
            visible_blocks,
            block_indices,
            rows,
            logits.stride(0),
            logits.stride(1),
            block_topk,
        )


@triton.jit(do_not_specialize=["columns", "decode_query_len"])
def _qsa_mqa_paged_score_tile_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    columns,
    decode_query_len,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile_columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    visible = tl.load(visible_blocks_ptr + row)
    if tl.program_id(1) * BLOCK_N >= visible:
        return
    request = row // decode_query_len
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, BLOCK_H)

    live = (tile_columns < columns) & (tile_columns < visible)
    logical_page = tl.minimum(tile_columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
    page_offset = tile_columns % PAGE_SIZE
    physical_page = tl.load(
        page_table_ptr + request * stride_table_req + logical_page,
        mask=live,
        other=-1,
    )
    live &= physical_page >= 0
    safe_page = tl.maximum(physical_page, 0).to(tl.int64)

    query = tl.load(
        q_ptr + row * stride_q_row + heads[:, None] * stride_q_head + dims[None, :],
        mask=(heads[:, None] < NUM_HEADS) & (dims[None, :] < HEAD_DIM),
        other=0.0,
    )
    keys = tl.load(
        k_cache_ptr
        + safe_page[None, :] * stride_cache_block
        + page_offset[None, :] * stride_cache_token
        + dims[:, None],
        mask=(dims[:, None] < HEAD_DIM) & live[None, :],
        other=0.0,
    )
    dots = tl.dot(query, keys, out_dtype=tl.float32)
    score = tl.sum(tl.maximum(dots, 0.0), axis=0)
    tl.store(
        logits_ptr + row * stride_logits_row + tile_columns,
        score,
        mask=live,
    )


@triton.jit
def _qsa_mqa_paged_score_loop_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    """Pipelined-loop score kernel (structure mirrors vllm's
    ``_qsa_mqa_paged_uniform_kernel``): each program covers
    ``TILES_PER_PROG`` consecutive column tiles, the query is loaded once,
    K tiles are loaded with ``eviction_policy="evict_first"`` and prefetched
    ``num_stages`` deep, and the dot keeps columns on the M side and heads
    on the N side (``[BLOCK_N, BLOCK_D] x [BLOCK_D, DQL*H]``).
    """
    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    DQL_P: tl.constexpr = triton.next_power_of_2(DECODE_QUERY_LEN)
    HEADS_P: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    query_offsets = tl.arange(0, DQL_P)
    valid_qoff = query_offsets < DECODE_QUERY_LEN
    rows = request * DECODE_QUERY_LEN + query_offsets
    visible = tl.load(visible_blocks_ptr + rows, mask=valid_qoff, other=0)
    max_visible = tl.max(visible, axis=0)
    if tile_start * BLOCK_N >= max_visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(max_visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(NUM_COLUMNS, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    n = tl.arange(0, DQL_P * HEADS_P)
    query_offset = n // HEADS_P
    head = n % HEADS_P
    valid_query = (query_offset < DECODE_QUERY_LEN) & (head < NUM_HEADS)
    query = tl.load(
        q_ptr
        + (request * DECODE_QUERY_LEN + query_offset)[None, :] * stride_q_row
        + head[None, :] * stride_q_head
        + dims[:, None],
        mask=valid_query[None, :] & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        # clamp keeps a malformed (-1) page-table entry memory-safe without
        # adding a data-dependent mask to the pipelined K load
        physical_page = tl.maximum(physical_page, 0)
        keys = tl.load(
            k_cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(valid_query[None, :], tl.maximum(scores, 0.0), 0.0)
        scores = tl.reshape(scores, (BLOCK_N, DQL_P, HEADS_P))
        score = tl.sum(scores, axis=2)
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=valid_qoff[None, :]
            & (columns[:, None] < NUM_COLUMNS)
            & (columns[:, None] < visible[None, :]),
        )


def qsa_select_paged_decode(
    q,
    k_cache,
    page_table,
    visible_blocks,
    token_topk,
    compress_ratio,
    decode_query_len,
    block_indices,
) -> None:
    """See module docstring."""
    logger.debug("GEMS QSA_SELECT_PAGED_DECODE")
    if q.dtype != k_cache.dtype:
        raise ValueError("QSA decode: q and k_cache must have the same dtype")
    if q.dtype not in _QSA_SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"QSA decode: unsupported dtype {q.dtype}; "
            f"supported: {_QSA_SUPPORTED_DTYPES}"
        )
    if q.ndim != 3:
        raise ValueError("QSA decode: q must be [rows, heads, head_dim]")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA decode: k_cache must be [blocks, page_size, 1, dim]")
    if page_table.ndim != 2 or page_table.dtype != torch.int32:
        raise ValueError("QSA decode: page_table must be int32 [requests, pages]")
    if visible_blocks.dtype != torch.int32:
        raise NotImplementedError(
            f"QSA decode: visible_blocks must be int32, got " f"{visible_blocks.dtype}"
        )
    if block_indices.dtype != torch.int32:
        raise TypeError("QSA decode: block_indices must be int32")
    if (
        q.stride(-1) != 1
        or k_cache.stride(-1) != 1
        or page_table.stride(-1) != 1
        or visible_blocks.stride(0) != 1
        or block_indices.stride(-1) != 1
    ):
        raise ValueError(
            "QSA decode: tensors must be contiguous in their innermost dim"
        )
    if compress_ratio <= 0 or token_topk % compress_ratio != 0:
        raise ValueError("QSA decode: token_topk must be a multiple of ratio")
    block_topk = token_topk // compress_ratio
    if block_topk < 1 or block_topk > _MAX_BLOCK_TOPK:
        raise NotImplementedError(
            f"QSA decode: block_topk must be in [1, {_MAX_BLOCK_TOPK}], "
            f"got {block_topk}"
        )
    rows = q.shape[0]
    if decode_query_len <= 0 or rows % decode_query_len != 0:
        raise ValueError("QSA decode: rows must be a multiple of query len")
    num_requests = rows // decode_query_len
    if page_table.shape[0] != num_requests:
        raise ValueError("QSA decode: page_table rows must equal num_requests")
    if visible_blocks.shape != (rows,):
        raise ValueError("QSA decode: visible_blocks must match query rows")
    if block_indices.shape != (rows, block_topk):
        raise ValueError(
            f"QSA decode: block_indices must be {(rows, block_topk)}, "
            f"got {tuple(block_indices.shape)}"
        )
    if rows == 0:
        return

    if not _HAS_TOP_K_PER_ROW:
        raise NotImplementedError(
            "QSA decode: top_k_per_row_decode backend is unavailable"
        )

    _eager_impl(
        q,
        k_cache,
        page_table,
        visible_blocks,
        block_indices,
        decode_query_len,
        block_topk,
    )


__all__ = ["qsa_select_paged_decode"]
