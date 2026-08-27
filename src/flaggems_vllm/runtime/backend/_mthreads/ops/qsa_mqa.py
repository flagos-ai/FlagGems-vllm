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

"""Mthreads S-series-optimized Qwen4 QSA MQA paged dot product.

[KernelGen] Auto-generated and tuned for Mthreads S-series.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_mqa_paged_dot_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    logits_ptr,
    visible_ptr,
    num_requests,
    num_pages,
    page_table_width,
    num_columns,
    compress_ratio: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # --- per-row metadata ---
    request = tl.load(token_to_req_ptr + pid_row)  # int32
    qpos = tl.load(query_positions_ptr + pid_row).to(tl.int64)

    valid_req = (request >= 0) & (request < num_requests)
    req_safe = tl.where(valid_req, request, 0)
    seqlen = tl.load(sequence_lengths_ptr + req_safe).to(tl.int64)

    visible = tl.minimum((qpos + 1) // compress_ratio, seqlen // compress_ratio)
    visible = tl.where(valid_req, visible, 0)
    visible_i32 = visible.to(tl.int32)

    if pid_col == 0:
        tl.store(visible_ptr + pid_row, visible_i32)

    # --- per-page (block of columns) work ---
    cols = pid_col * page_size + tl.arange(0, page_size)

    logical_page = pid_col
    lp_valid = logical_page < page_table_width

    phys = tl.load(
        page_table_ptr + req_safe * page_table_width + logical_page,
        mask=lp_valid,
        other=-1,
    )

    page_valid = valid_req & lp_valid & (phys >= 0) & (phys < num_pages)
    col_valid = cols < visible_i32
    valid = col_valid & page_valid  # [page_size]

    # Contiguous flat load of the whole page, then reshape.
    phys_safe = tl.where(page_valid, phys, 0)
    flat = tl.arange(0, page_size * head_dim)
    # k pages are each read exactly once: mark streaming so they evict early
    # and leave cache room for the hot q rows.
    k_flat = tl.load(
        k_cache_ptr + phys_safe * (page_size * head_dim) + flat,
        eviction_policy="evict_first",
    ).to(tl.float32)
    k_block = tl.reshape(k_flat, (page_size, head_dim))

    d = tl.arange(0, head_dim)
    q_base = pid_row * (num_heads * head_dim)
    # num_heads is a fixed invariant of the op (q is always [R, 4, 128]):
    # compute the four head dots independently and combine with a balanced
    # tree so the reduction dependency chain stays short. The q rows are
    # re-read by every column program of a row, so keep them resident.
    q0 = tl.load(q_ptr + q_base + 0 * head_dim + d, eviction_policy="evict_last").to(
        tl.float32
    )
    q1 = tl.load(q_ptr + q_base + 1 * head_dim + d, eviction_policy="evict_last").to(
        tl.float32
    )
    q2 = tl.load(q_ptr + q_base + 2 * head_dim + d, eviction_policy="evict_last").to(
        tl.float32
    )
    q3 = tl.load(q_ptr + q_base + 3 * head_dim + d, eviction_policy="evict_last").to(
        tl.float32
    )
    d0 = tl.sum(q0[None, :] * k_block, axis=1)
    d1 = tl.sum(q1[None, :] * k_block, axis=1)
    d2 = tl.sum(q2[None, :] * k_block, axis=1)
    d3 = tl.sum(q3[None, :] * k_block, axis=1)
    score = (tl.maximum(d0, 0.0) + tl.maximum(d1, 0.0)) + (
        tl.maximum(d2, 0.0) + tl.maximum(d3, 0.0)
    )

    score = score * (1.0 / math.sqrt(head_dim))

    result = tl.where(valid, score, float("-inf"))
    tl.store(logits_ptr + pid_row * num_columns + cols, result, mask=cols < num_columns)


def qwen4_qsa_mqa_paged_dot(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int = 4,
    num_columns: int | None = None,
    score_scale: float | None = None,
):
    if not all(
        t.device.type not in ("cpu", "meta")
        for t in (
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
        )
    ):
        raise RuntimeError("Qwen4 QSA MQA dot requires a Triton accelerator")
    if q.ndim != 3 or q.shape[1:] != (4, 128) or q.dtype != torch.bfloat16:
        raise ValueError("Qwen4 QSA MQA dot requires BF16 q shaped [rows, 4, 128]")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, 128):
        raise ValueError("Qwen4 QSA MQA cache must be [pages, page_size, 1, 128]")
    if k_cache.dtype != q.dtype:
        raise ValueError("Qwen4 QSA query and cache must have the same dtype")
    if page_table.ndim != 2:
        raise ValueError("Qwen4 QSA MQA page table must be rank-2")
    if page_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 QSA page table must use int32 or int64")

    rows = q.shape[0]
    if rows and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError(
            "Qwen4 QSA MQA cache and page table must be nonempty for nonempty q"
        )
    if token_to_req.shape != (rows,) or query_positions.shape != (rows,):
        raise ValueError("Qwen4 QSA request metadata must match query rows")
    if token_to_req.dtype not in (
        torch.int32,
        torch.int64,
    ) or query_positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 QSA request metadata must use int32 or int64")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("Qwen4 QSA sequence lengths must match page-table requests")
    if sequence_lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 QSA sequence lengths must use int32 or int64")
    if compress_ratio <= 0:
        raise ValueError("Qwen4 QSA compression ratio must be positive")

    if score_scale is not None:
        raise NotImplementedError(
            "Vendor QSA MQA uses hardcoded scale 1/sqrt(128); custom score_scale not supported"
        )

    if num_columns is None:
        num_columns = triton.cdiv(
            sequence_lengths.max().item() if sequence_lengths.numel() else 0,
            compress_ratio,
        )

    rows = q.shape[0]
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_requests = page_table.shape[0]
    page_table_width = page_table.shape[1]
    num_columns = int(num_columns)
    compress_ratio = int(compress_ratio)

    logits = torch.empty((rows, num_columns), dtype=torch.float32, device=q.device)
    visible = torch.empty((rows,), dtype=torch.int32, device=q.device)

    grid = (rows, triton.cdiv(num_columns, page_size))
    _qsa_mqa_paged_dot_kernel[grid](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        logits,
        visible,
        num_requests,
        num_pages,
        page_table_width,
        num_columns,
        compress_ratio=compress_ratio,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
    )
    return logits, visible
