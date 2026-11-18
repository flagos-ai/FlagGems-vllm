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
"""
MUSA QSA compressed-block selection, supporting bf16 and fp8.

Scores are unscaled sum_h relu(K_h . q_h), matching the vLLM decode baseline.
A uniform positive scale does not change top-k selection. Only visible logits
are written; persistent_topk reads each row exclusively within [0, visible).
Every live page_table entry must be in [0, num_pages). Defensive clamping does
not define malformed-input semantics or guarantee tile/loop agreement there.
Logits and top-k workspaces are reused by serialized calls; concurrent calls
sharing a device require independent workspaces.
"""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

_PERSISTENT_TOPK_IMPORT_ERROR = ""
try:
    from flaggems_vllm.runtime.backend._mthreads.ops.persistent_topk import (
        persistent_topk,
    )

    _HAS_PERSISTENT_TOPK = True
except ImportError as exc:
    persistent_topk = None
    _HAS_PERSISTENT_TOPK = False
    _PERSISTENT_TOPK_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

logger = logging.getLogger(__name__)

_MTHREADS_QSA_BLOCK_TOPK = (512, 1024, 2048)
_QSA_SUPPORTED_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)

_SCORE_BLOCK_N = 128
_SCORE_NUM_WARPS = 4
_SMALL_POOL_MAX_BLOCKS = 1000

_LOGITS_CACHE: dict = {}
_LOGITS_CACHE_MAX = 64
_TOPK_WORKSPACE_CACHE: dict = {}
_TOPK_WORKSPACE_CACHE_MAX = 16
_TOPK_WORKSPACE_BYTES = 1024 * 1024


def _get_topk_workspace(device) -> torch.Tensor:
    workspace = _TOPK_WORKSPACE_CACHE.get(device)
    if workspace is None:
        if len(_TOPK_WORKSPACE_CACHE) >= _TOPK_WORKSPACE_CACHE_MAX:
            _TOPK_WORKSPACE_CACHE.clear()
        workspace = torch.empty(
            (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=device
        )
        _TOPK_WORKSPACE_CACHE[device] = workspace
    return workspace


def _get_logits(rows: int, columns: int, device) -> torch.Tensor:
    key = (rows, columns, device)
    buf = _LOGITS_CACHE.get(key)
    if buf is None:
        if len(_LOGITS_CACHE) >= _LOGITS_CACHE_MAX:
            _LOGITS_CACHE.clear()
        buf = torch.empty((rows, columns), dtype=torch.float32, device=device)
        _LOGITS_CACHE[key] = buf
    return buf


def _pick_score_plan(num_blocks: int, num_requests: int, dtype=torch.bfloat16):
    """
    MUSA score dispatch measured on MTT S5000 in November 2026.

    The BF16 small-pool exception from 193 through 367 rows is the measured
    loop-kernel region; tile wins outside it. Only BF16 was swept for this
    exception; FP8 uses the remaining plans without a separate plan sweep.
    Retune these thresholds when the hardware or compiler changes.

    Returns ("loop", block_n, tiles_per_prog, num_warps, stages) for the
    pipelined-loop kernel or ("tile", block_n, num_warps) for the
    one-tile-per-program kernel.
    """
    if num_blocks <= _SMALL_POOL_MAX_BLOCKS:
        if num_requests >= 128 and (
            dtype != torch.bfloat16 or not 193 <= num_requests <= 367
        ):
            return "tile", _SCORE_BLOCK_N, _SCORE_NUM_WARPS
        buckets = (
            (16, ("loop", 64, 1, 4, 2)),
            (80, ("loop", 256, 4, 4, 2)),
            (192, ("loop", 64, 8, 2, 2)),
            (367, ("loop", 64, 8, 4, 2)),
            (None, ("loop", 128, 8, 2, 2)),
        )
    else:
        if num_requests >= 128:
            return "tile", _SCORE_BLOCK_N, _SCORE_NUM_WARPS
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
    raise RuntimeError(
        f"unreachable score plan: {num_blocks=}, {num_requests=}, {dtype=}"
    )


def _score_impl(q, k_cache, page_table, visible_blocks, decode_query_len):
    rows = q.shape[0]
    num_requests = rows // decode_query_len
    columns = page_table.shape[1] * k_cache.shape[1]
    logits = _get_logits(rows, columns, q.device)
    plan = _pick_score_plan(k_cache.shape[0], num_requests, q.dtype)

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
                k_cache.shape[0],
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
                k_cache.shape[0],
                PAGE_SIZE=k_cache.shape[1],
                PAGE_TABLE_WIDTH=page_table.shape[1],
                NUM_HEADS=q.shape[1],
                HEAD_DIM=q.shape[2],
                BLOCK_H=max(16, triton.next_power_of_2(q.shape[1])),
                BLOCK_N=block_n,
                BLOCK_D=max(16, triton.next_power_of_2(q.shape[2])),
                num_warps=num_warps,
            )
    return logits


def _topk_impl(logits, visible_blocks, block_indices, block_topk):
    with torch_device_fn.device(logits.device):
        persistent_topk(
            logits,
            visible_blocks,
            block_indices,
            _get_topk_workspace(logits.device),
            k=block_topk,
            max_seq_len=logits.shape[1],
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
    num_pages,
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
    live &= (physical_page >= 0) & (physical_page < num_pages)
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
    num_pages,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    """
    Pipelined-loop score kernel (structure mirrors vllm's
    ``_qsa_mqa_paged_uniform_kernel``): each program covers
    ``TILES_PER_PROG`` consecutive column tiles, the query is loaded once,
    K tiles are loaded with ``eviction_policy="evict_first"`` and prefetched
    ``num_stages`` deep, and the dot keeps columns on the M side and heads
    on the N side (``[BLOCK_N, BLOCK_D] x [BLOCK_D, DQL*H]``).
    """
    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    DQL_P: tl.constexpr = triton.next_power_of_2(DECODE_QUERY_LEN)
    HEADS_P: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
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
        physical_page = tl.minimum(tl.maximum(physical_page, 0), num_pages - 1)
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
    logger.debug("GEMS_MTHREADS QSA_SELECT_PAGED_DECODE")
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
            f"QSA decode: visible_blocks must be int32, got {visible_blocks.dtype}"
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
    if block_topk not in _MTHREADS_QSA_BLOCK_TOPK:
        raise NotImplementedError(
            f"QSA decode on MUSA supports block_topk={_MTHREADS_QSA_BLOCK_TOPK}, "
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
    if not _HAS_PERSISTENT_TOPK:
        detail = (
            f" ({_PERSISTENT_TOPK_IMPORT_ERROR})"
            if _PERSISTENT_TOPK_IMPORT_ERROR
            else ""
        )
        raise NotImplementedError(
            "QSA decode on MUSA requires persistent_topk and FlagTree TLE" + detail
        )

    logits = _score_impl(q, k_cache, page_table, visible_blocks, decode_query_len)
    _topk_impl(logits, visible_blocks, block_indices, block_topk)


__all__ = ["qsa_select_paged_decode"]
