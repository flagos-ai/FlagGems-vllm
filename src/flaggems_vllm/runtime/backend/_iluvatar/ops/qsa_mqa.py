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

"""Iluvatar BI/MR series-optimized Qwen4 QSA MQA paged dot product.

[KernelGen] Auto-generated and tuned for Iluvatar BI/MR series.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_elementwise_kernel(
    Q_PTR,
    K_PTR,
    PT_PTR,
    T2R_PTR,
    QPOS_PTR,
    SL_PTR,
    LOG_PTR,
    VIS_PTR,
    NUM_REQUESTS,
    NUM_PAGES,
    q_stride_row,
    q_stride_head,
    k_stride_page,
    k_stride_off,
    pt_stride_req,
    log_stride_row,
    COMPRESS_RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_COLS: tl.constexpr,
    D_C: tl.constexpr,
    HEADS_C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INV_SQRT: tl.constexpr,
):
    r = tl.program_id(0)
    t = tl.program_id(1)
    col_start = t * BLOCK_C

    # ---- Row metadata (int32; values are small) ----
    req = tl.load(T2R_PTR + r).to(tl.int32)
    qpos = tl.load(QPOS_PTR + r).to(tl.int32)
    valid_req = (req >= 0) & (req < NUM_REQUESTS)
    safe_req = tl.where(valid_req, req, 0)
    seqlen = tl.load(SL_PTR + safe_req).to(tl.int32)
    seqlen = tl.where(valid_req, seqlen, 0)
    visible = tl.minimum((qpos + 1) // COMPRESS_RATIO, seqlen // COMPRESS_RATIO)
    visible = tl.where(valid_req, visible, 0)

    if t == 0:
        tl.store(VIS_PTR + r, visible)

    d_idx = tl.arange(0, D_C)

    # ---- Column tile ----
    col_offs = col_start + tl.arange(0, BLOCK_C)
    col_mask = col_offs < NUM_COLS
    logical_page = col_offs // PAGE_SIZE
    page_off = col_offs % PAGE_SIZE
    lp_valid = (logical_page < PAGE_TABLE_WIDTH) & col_mask
    valid_col = valid_req & (col_offs < visible) & lp_valid
    safe_lp = tl.where(lp_valid, logical_page, 0)
    phys = tl.load(
        PT_PTR + safe_req * pt_stride_req + safe_lp, mask=valid_col, other=-1
    )
    phys_valid = valid_col & (phys >= 0) & (phys < NUM_PAGES)
    safe_phys = tl.where(phys_valid, phys, 0)
    safe_off = tl.where(phys_valid, page_off, 0)

    # ---- Load k [BLOCK_C, D] via paged indirection ----
    k_row_off = safe_phys * k_stride_page + safe_off * k_stride_off  # [BLOCK_C]
    k_ptrs = K_PTR + k_row_off[:, None] + d_idx[None, :]
    k_bf = tl.load(k_ptrs, mask=phys_valid[:, None], other=0.0)
    k_f32 = k_bf.to(tl.float32)  # [BLOCK_C, D]

    # ---- Per-head dot ----
    scores = tl.zeros([BLOCK_C], dtype=tl.float32)
    for h in tl.static_range(HEADS_C):
        q_h = tl.load(Q_PTR + r * q_stride_row + h * q_stride_head + d_idx).to(
            tl.float32
        )
        dot_h = tl.sum(k_f32 * q_h[None, :], axis=1)  # [BLOCK_C]
        scores += tl.maximum(dot_h, 0.0)
    scores = scores * INV_SQRT

    out = tl.where(phys_valid, scores, float("-inf"))
    tl.store(LOG_PTR + r * log_stride_row + col_offs, out, mask=col_mask)


@triton.jit
def _qsa_mma_kernel(
    Q_PTR,
    K_PTR,
    PT_PTR,
    T2R_PTR,
    QPOS_PTR,
    SL_PTR,
    LOG_PTR,
    VIS_PTR,
    NUM_REQUESTS,
    NUM_PAGES,
    q_stride_row,
    q_stride_head,
    k_stride_page,
    k_stride_off,
    pt_stride_req,
    log_stride_row,
    COMPRESS_RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_COLS: tl.constexpr,
    D_C: tl.constexpr,
    HEADS_C: tl.constexpr,
    M_PAD: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INV_SQRT: tl.constexpr,
):
    r = tl.program_id(0)
    t = tl.program_id(1)
    col_start = t * BLOCK_C

    req = tl.load(T2R_PTR + r).to(tl.int32)
    qpos = tl.load(QPOS_PTR + r).to(tl.int32)
    valid_req = (req >= 0) & (req < NUM_REQUESTS)
    safe_req = tl.where(valid_req, req, 0)
    seqlen = tl.load(SL_PTR + safe_req).to(tl.int32)
    seqlen = tl.where(valid_req, seqlen, 0)
    visible = tl.minimum((qpos + 1) // COMPRESS_RATIO, seqlen // COMPRESS_RATIO)
    visible = tl.where(valid_req, visible, 0)

    if t == 0:
        tl.store(VIS_PTR + r, visible)

    d_idx = tl.arange(0, D_C)

    col_offs = col_start + tl.arange(0, BLOCK_C)
    col_mask = col_offs < NUM_COLS
    logical_page = col_offs // PAGE_SIZE
    page_off = col_offs % PAGE_SIZE
    lp_valid = (logical_page < PAGE_TABLE_WIDTH) & col_mask
    valid_col = valid_req & (col_offs < visible) & lp_valid
    safe_lp = tl.where(lp_valid, logical_page, 0)
    phys = tl.load(
        PT_PTR + safe_req * pt_stride_req + safe_lp, mask=valid_col, other=-1
    )
    phys_valid = valid_col & (phys >= 0) & (phys < NUM_PAGES)
    safe_phys = tl.where(phys_valid, phys, 0)
    safe_off = tl.where(phys_valid, page_off, 0)

    k_row_off = safe_phys * k_stride_page + safe_off * k_stride_off  # [BLOCK_C]
    k_ptrs = K_PTR + k_row_off[:, None] + d_idx[None, :]
    k_bf = tl.load(k_ptrs, mask=phys_valid[:, None], other=0.0)  # [BLOCK_C, D] bf16

    h_idx = tl.arange(0, M_PAD)
    q16 = tl.load(
        Q_PTR + r * q_stride_row + h_idx[:, None] * q_stride_head + d_idx[None, :],
        mask=h_idx[:, None] < HEADS_C,
        other=0.0,
    )  # [M_PAD, D] bf16

    k_T = tl.trans(k_bf)  # [D, BLOCK_C] bf16
    dots = tl.dot(q16, k_T)  # [M_PAD, BLOCK_C] fp32
    scores = tl.sum(tl.maximum(dots, 0.0), axis=0)  # [BLOCK_C]
    scores = scores * INV_SQRT

    out = tl.where(phys_valid, scores, float("-inf"))
    tl.store(LOG_PTR + r * log_stride_row + col_offs, out, mask=col_mask)


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
    heads = q.shape[1]
    d = q.shape[2]
    num_pages, page_size, _, _ = k_cache.shape
    num_requests, page_table_width = page_table.shape

    compress_ratio = int(compress_ratio)
    num_columns = int(num_columns)

    device = q.device
    logits = torch.empty((rows, num_columns), device=device, dtype=torch.float32)
    visible = torch.empty((rows,), device=device, dtype=torch.int32)

    if rows == 0 or num_columns == 0:
        return logits, visible

    # Shape dispatch: elementwise for memory-bound (many rows) and
    # latency-bound (1 row) workloads; bf16 MMA for the occupancy-limited
    # mid-size workload.
    use_mma = False
    if num_columns <= 64:
        BLOCK_C = triton.next_power_of_2(num_columns)
        NW = 2 if BLOCK_C <= 32 else 4
    elif rows == 1:
        BLOCK_C = 32
        NW = 1
    elif rows <= 8:
        use_mma = True
        BLOCK_C = 128
        NW = 8
    else:
        BLOCK_C = 128
        NW = 2

    num_tiles = triton.cdiv(num_columns, BLOCK_C)
    grid = (rows, num_tiles)

    if use_mma:
        M_PAD = 8
        _qsa_mma_kernel[grid](
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
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            page_table.stride(0),
            logits.stride(0),
            COMPRESS_RATIO=compress_ratio,
            PAGE_SIZE=page_size,
            PAGE_TABLE_WIDTH=page_table_width,
            NUM_COLS=num_columns,
            D_C=d,
            HEADS_C=heads,
            M_PAD=M_PAD,
            BLOCK_C=BLOCK_C,
            INV_SQRT=1.0 / math.sqrt(d),
            num_warps=NW,
        )
    else:
        _qsa_elementwise_kernel[grid](
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
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            page_table.stride(0),
            logits.stride(0),
            COMPRESS_RATIO=compress_ratio,
            PAGE_SIZE=page_size,
            PAGE_TABLE_WIDTH=page_table_width,
            NUM_COLS=num_columns,
            D_C=d,
            HEADS_C=heads,
            BLOCK_C=BLOCK_C,
            INV_SQRT=1.0 / math.sqrt(d),
            num_warps=NW,
        )
    return logits, visible
