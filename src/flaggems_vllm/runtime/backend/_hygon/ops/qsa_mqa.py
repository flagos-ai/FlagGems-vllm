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

"""Hygon DCU-optimized Qwen4 QSA MQA paged dot product.

[KernelGen] Auto-generated and tuned for Hygon DCU.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_mqa_paged_dot_kernel(
    q_ptr,
    k_ptr,
    pt_ptr,
    t2r_ptr,
    qpos_ptr,
    seq_ptr,
    logits_ptr,
    visible_ptr,
    num_columns,
    num_pages,
    num_requests,
    page_table_width,
    compress_ratio,
    inv_sqrt_dim,
    BLOCK_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    row = tl.program_id(0)
    page_block = tl.program_id(1)

    request = tl.load(t2r_ptr + row)
    qpos = tl.load(qpos_ptr + row).to(tl.int32)
    valid_req = (request >= 0) & (request < num_requests)
    safe_req = tl.where(valid_req, request, 0)
    seqlen = tl.load(seq_ptr + safe_req).to(tl.int32)
    seqlen = tl.where(valid_req, seqlen, 0)
    visible = tl.minimum((qpos + 1) // compress_ratio, seqlen // compress_ratio)
    if page_block == 0:
        tl.store(visible_ptr + row, visible)

    offs_p = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)

    logical_page = page_block * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    lp_valid = logical_page < page_table_width
    safe_lp = tl.minimum(logical_page, page_table_width - 1)
    phys = tl.load(
        pt_ptr + safe_req * page_table_width + safe_lp, mask=lp_valid, other=-1
    )
    phys_valid = (phys >= 0) & (phys < num_pages)
    page_valid = lp_valid & phys_valid

    k_offsets = (
        phys[:, None, None] * (PAGE_SIZE * HEAD_DIM)
        + offs_p[None, :, None] * HEAD_DIM
        + offs_d[None, None, :]
    )
    k_tile = tl.load(k_ptr + k_offsets, mask=page_valid[:, None, None], other=0.0).to(
        tl.float32
    )

    acc = tl.zeros([BLOCK_PAGES, PAGE_SIZE], dtype=tl.float32)
    for h in tl.static_range(NUM_HEADS):
        qh = tl.load(q_ptr + row * (NUM_HEADS * HEAD_DIM) + h * HEAD_DIM + offs_d).to(
            tl.float32
        )
        dots_h = tl.sum(k_tile * qh[None, None, :], axis=2)
        acc += tl.maximum(dots_h, 0.0)
    score = acc * inv_sqrt_dim

    cols = logical_page[:, None] * PAGE_SIZE + offs_p[None, :]
    col_mask = cols < num_columns
    valid = page_valid[:, None] & (cols < visible) & col_mask
    out = tl.where(valid, score, float("-inf"))
    tl.store(logits_ptr + row * num_columns + cols, out, mask=col_mask)


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
    num_requests = page_table.shape[0]
    page_table_width = page_table.shape[1]
    num_columns = int(num_columns)
    compress_ratio = int(compress_ratio)
    head_dim = q.shape[2]
    num_heads = q.shape[1]

    logits = torch.empty((rows, num_columns), dtype=torch.float32, device=q.device)
    visible = torch.empty((rows,), dtype=torch.int32, device=q.device)

    # Low-row workloads are launch-bound: use a smaller per-program tile.
    if rows <= 8:
        BLOCK_PAGES = 2
    else:
        BLOCK_PAGES = 4

    pages_per_row = triton.cdiv(num_columns, page_size)
    num_page_blocks = triton.cdiv(pages_per_row, BLOCK_PAGES)
    grid = (rows, num_page_blocks)
    inv_sqrt_dim = 1.0 / math.sqrt(head_dim)

    _qsa_mqa_paged_dot_kernel[grid](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        logits,
        visible,
        num_columns,
        num_pages,
        num_requests,
        page_table_width,
        compress_ratio,
        inv_sqrt_dim,
        BLOCK_PAGES=BLOCK_PAGES,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        NUM_HEADS=num_heads,
        num_warps=4,
    )
    return logits, visible
