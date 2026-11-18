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
QSA decode compressed-block selection (MetaX), supporting bf16 and fp8.

visible_blocks contains compressed-column counts in [0, columns]. Logits and
top-k workspaces are reused by serialized calls; concurrent calls sharing a
device require independent workspaces.

The stock MetaX Triton backend rejects the narrow-N MMA in dot-based score
kernels, so scoring uses elementwise multiply + sum. Each K tile is loaded once
per (row, tile) and reused across the head loop. Selection uses MetaX
persistent_topk.
"""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

_PERSISTENT_TOPK_IMPORT_ERROR = ""
try:
    from flaggems_vllm.runtime.backend._metax.ops.persistent_topk import persistent_topk

    _HAS_PERSISTENT_TOPK = True
except ImportError as exc:
    persistent_topk = None
    _HAS_PERSISTENT_TOPK = False
    _PERSISTENT_TOPK_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

logger = logging.getLogger(__name__)

_METAX_QSA_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
_METAX_QSA_BLOCK_TOPK = (512, 1024, 2048)

_SCORE_BLOCK_N = 128
_SCORE_NUM_WARPS = 4

_LOGITS_CACHE: dict = {}
_LOGITS_CACHE_MAX = 64
_TOPK_WORKSPACE_CACHE: dict = {}
_TOPK_WORKSPACE_CACHE_MAX = 16
_TOPK_WORKSPACE_BYTES = 1024 * 1024


def _get_logits(rows: int, columns: int, device) -> torch.Tensor:
    key = (rows, columns, device)
    buf = _LOGITS_CACHE.get(key)
    if buf is None:
        if len(_LOGITS_CACHE) >= _LOGITS_CACHE_MAX:
            _LOGITS_CACHE.clear()
        buf = torch.empty((rows, columns), dtype=torch.float32, device=device)
        _LOGITS_CACHE[key] = buf
    return buf


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
    """
    Elementwise-multiply score kernel (no tl.dot — the MetaX backend
    rejects narrow-N MMA).  K tile is loaded once per (row, tile) and
    reused across the head loop; -inf is stored for non-live columns so
    every consumer of the logits buffer sees defined values.
    """
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
                qv = tl.load(
                    q_ptr + row * stride_q_row + head * stride_q_head + dims,
                    mask=dims < HEAD_DIM,
                    other=0.0,
                ).to(tl.float32)
                dot = tl.sum(keys * qv[None, :], axis=1)
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
        persistent_topk(
            logits,
            visible_blocks,
            block_indices,
            _get_topk_workspace(logits.device),
            k=block_topk,
            max_seq_len=logits.shape[1],
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
    """Select top-k compressed blocks with MetaX Triton kernels."""
    logger.debug("GEMS_METAX QSA_SELECT_PAGED_DECODE")
    if q.dtype != k_cache.dtype:
        raise ValueError("QSA decode: q and k_cache must have the same dtype")
    if q.dtype not in _METAX_QSA_DTYPES:
        raise NotImplementedError(
            f"QSA decode: unsupported dtype {q.dtype}; "
            f"supported: {_METAX_QSA_DTYPES}"
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
            "QSA decode: tensors must be contiguous in their innermost dim; "
            f"strides: q={tuple(q.stride())}, k_cache={tuple(k_cache.stride())}, "
            f"page_table={tuple(page_table.stride())}, "
            f"visible_blocks={tuple(visible_blocks.stride())}, "
            f"block_indices={tuple(block_indices.stride())}"
        )
    if compress_ratio <= 0 or token_topk % compress_ratio != 0:
        raise ValueError("QSA decode: token_topk must be a multiple of ratio")
    block_topk = token_topk // compress_ratio
    if block_topk not in _METAX_QSA_BLOCK_TOPK:
        raise NotImplementedError(
            f"QSA decode on MetaX supports block_topk={_METAX_QSA_BLOCK_TOPK}, "
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
            "QSA decode on MetaX requires persistent_topk and FlagTree TLE" + detail
        )

    logits = _score_impl(q, k_cache, page_table, visible_blocks, decode_query_len)
    _topk_impl(logits, visible_blocks, block_indices, block_topk)


__all__ = ["qsa_select_paged_decode"]
