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

from flaggems_vllm.ops.top_k_per_row_decode import top_k_per_row_decode
from flaggems_vllm.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_HYGON_QSA_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
_MAX_BLOCK_TOPK = 4096
_SCORE_BLOCK_N = 128
_SCORE_NUM_WARPS = 4

_LOGITS_CACHE = {}
_LOGITS_CACHE_MAX = 64


def _get_logits(rows, columns, device):
    key = (rows, columns, device)
    buf = _LOGITS_CACHE.get(key)
    if buf is None:
        if len(_LOGITS_CACHE) >= _LOGITS_CACHE_MAX:
            _LOGITS_CACHE.clear()
        buf = torch.empty((rows, columns), dtype=torch.float32, device=device)
        _LOGITS_CACHE[key] = buf
    return buf


@triton.jit
def _qsa_mqa_paged_score_ew_kernel(
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
    num_rows,
    num_pages,
    decode_query_len,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_TILES: tl.constexpr,
    ROW_PROGS: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    tile = pid % NUM_TILES
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    for row in range(pid // NUM_TILES, num_rows, ROW_PROGS):
        visible = tl.load(visible_blocks_ptr + row)
        if tile * BLOCK_N < visible:
            request = row // decode_query_len
            live = (cols < columns) & (cols < visible)
            logical_page = tl.minimum(cols // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
            page_offset = cols % PAGE_SIZE
            physical_page = tl.load(
                page_table_ptr + request * stride_table_req + logical_page,
                mask=live,
                other=-1,
            )
            live &= (physical_page >= 0) & (physical_page < num_pages)
            safe_page = tl.maximum(physical_page, 0).to(tl.int64)

            keys = tl.load(
                k_cache_ptr
                + safe_page[:, None] * stride_cache_block
                + page_offset[:, None] * stride_cache_token
                + dims[None, :],
                mask=live[:, None] & (dims[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            score = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for head in tl.static_range(0, NUM_HEADS):
                query = tl.load(
                    q_ptr + row * stride_q_row + head * stride_q_head + dims,
                    mask=dims < HEAD_DIM,
                    other=0.0,
                ).to(tl.float32)
                dot = tl.sum(keys * query[None, :], axis=1)
                score += tl.maximum(dot, 0.0)
            tl.store(
                logits_ptr + row * stride_logits_row + cols,
                tl.where(live, score / score_divisor, -float("inf")),
                mask=cols < columns,
            )
        else:
            tl.store(
                logits_ptr + row * stride_logits_row + cols,
                tl.full((BLOCK_N,), -float("inf"), dtype=tl.float32),
                mask=cols < columns,
            )


def _score_impl(q, k_cache, page_table, visible_blocks, decode_query_len):
    rows = q.shape[0]
    columns = page_table.shape[1] * k_cache.shape[1]
    logits = _get_logits(rows, columns, q.device)
    tiles = triton.cdiv(columns, _SCORE_BLOCK_N)
    row_progs = min(rows, max(1, 32768 // tiles))
    with torch_device_fn.device(q.device):
        _qsa_mqa_paged_score_ew_kernel[(row_progs * tiles,)](
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
            rows,
            k_cache.shape[0],
            decode_query_len,
            float(q.shape[2] ** 0.5),
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=page_table.shape[1],
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            BLOCK_N=_SCORE_BLOCK_N,
            BLOCK_D=max(16, triton.next_power_of_2(q.shape[2])),
            NUM_TILES=tiles,
            ROW_PROGS=row_progs,
            num_warps=_SCORE_NUM_WARPS,
        )
    return logits


def _topk_impl(logits, visible_blocks, block_indices, block_topk):
    with torch_device_fn.device(logits.device):
        top_k_per_row_decode(
            logits,
            1,
            visible_blocks,
            block_indices,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            block_topk,
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
    """Hygon QSA compressed-block selection with Triton scoring and top-k.

    Scores are sum_h relu(K_h . q_h) / sqrt(head_dim), accumulated in FP32.
    visible_blocks contains compressed-column counts in [0, columns]. The
    baseline adapter's raw-token metadata reconstructs the same actual_visible.
    Logits scratch buffers are reused by serialized calls on each device;
    concurrent calls sharing a shape and device require separate workspaces.
    """
    logger.debug("GEMS_HYGON QSA_SELECT_PAGED_DECODE")
    if q.dtype != k_cache.dtype:
        raise ValueError("QSA decode: q and k_cache must have the same dtype")
    if q.dtype not in _HYGON_QSA_DTYPES:
        raise NotImplementedError(
            f"QSA decode: unsupported dtype {q.dtype}; "
            f"supported: {_HYGON_QSA_DTYPES}"
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

    logits = _score_impl(q, k_cache, page_table, visible_blocks, decode_query_len)
    _topk_impl(logits, visible_blocks, block_indices, block_topk)


__all__ = ["qsa_select_paged_decode"]
