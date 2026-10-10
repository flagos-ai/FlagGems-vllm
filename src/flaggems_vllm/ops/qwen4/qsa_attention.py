# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged sparse QSA attention with optional caller-owned split workspace."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.qwen4.qsa import _is_triton_device
from flaggems_vllm.runtime import device as runtime_device

_QSA_SPLIT_ALLOWED = (1, 2, 4, 8, 16, 32)


def qsa_sparse_triton_launch_config(
    device: torch.device,
) -> tuple[int, int, int]:
    """Return the measured sparse-QSA config, guarded by GPU architecture.

    ``(64, 8, 3)`` is validated on SM90 H100/H200 shapes.  Every other
    NVIDIA architecture and every non-NVIDIA accelerator keeps the original
    conservative Triton configuration until it has its own benchmark.
    """

    if runtime_device.vendor_name != "nvidia" or device.type != "cuda":
        return 16, 4, 2
    try:
        if torch.cuda.get_device_capability(device) == (9, 0):
            return 64, 8, 3
    except (AssertionError, RuntimeError):
        # No usable capability query: retain the conservative launch config.
        pass
    return 16, 4, 2


def qsa_sparse_split_count(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    topk: int,
) -> int:
    """Select a graph-static split count for the measured H100 c64 shape.

    The default is deliberately narrow: only SM90 ``[rows, 3, 256]`` over a
    single KV head and a sufficiently wide selection use split=8.  Shape or
    Architecture misses stay on the single CTA path. Callers select single or
    split execution by omitting or supplying fixed workspace, respectively.
    """

    requested = 8
    if q.device.type != "cuda" or q.shape[0] <= 0:
        return 1
    if q.shape[1] != 3 or k_cache.shape[2] != 1 or q.shape[2] != 256:
        return 1
    if q.shape[0] > 64 or topk < 512:
        return 1
    try:
        # The measured H100 selector is the only selector currently approved
        # for split dispatch.  Avoid introducing a second architecture ABI.
        block_n, num_warps, num_stages = qsa_sparse_triton_launch_config(q.device)
        if (block_n, num_warps, num_stages) != (64, 8, 3):
            return 1
    except (AssertionError, RuntimeError):
        return 1
    num_tiles = triton.cdiv(topk, 64)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    return min(requested, max_useful_splits)


@triton.jit
def _qsa_sparse_paged_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    gate_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_indices_row,
    stride_indices_column,
    stride_table_req,
    stride_table_page,
    stride_token_to_req,
    stride_gate_row,
    stride_gate_head,
    stride_gate_dim,
    stride_output_row,
    stride_output_head,
    stride_output_dim,
    num_rows,
    num_cache_blocks,
    num_requests,
    softmax_scale,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    APPLY_GATE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.load(token_to_req_ptr + row * stride_token_to_req)
    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=(head_offsets[:, None] < GROUP_SIZE) & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    query = (query * softmax_scale * 1.4426950408889634).to(query.dtype)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)

    for start in tl.range(0, TOPK, BLOCK_N):
        columns = start + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns * stride_indices_column,
            mask=columns < TOPK,
            other=-1,
        )
        logical_page = tl.maximum(logical_token, 0) // PAGE_SIZE
        page_offset = tl.maximum(logical_token, 0) % PAGE_SIZE
        valid = (
            (row < num_rows)
            & (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + tl.minimum(tl.maximum(request, 0), num_requests - 1) * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=(dim_offsets[:, None] < HEAD_DIM) & valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )
        scores = tl.dot(query, keys)
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    output = tl.where(
        normalizer[:, None] > 0,
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    if APPLY_GATE:
        gate = tl.load(
            gate_ptr
            + row * stride_gate_row
            + (first_head + head_offsets[:, None]) * stride_gate_head
            + dim_offsets[None, :] * stride_gate_dim,
            mask=(row < num_rows)
            & (head_offsets[:, None] < GROUP_SIZE)
            & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )
        # Preserve the unfused BF16 rounding points: attention is stored as
        # BF16, sigmoid returns BF16 for a BF16 gate, then the product rounds
        # to BF16 before the original output buffer is consumed by o_proj.
        output = (
            output.to(tl.bfloat16) * tl.sigmoid(gate.to(tl.float32)).to(tl.bfloat16)
        ).to(tl.bfloat16)
    tl.store(
        output_ptr
        + row * stride_output_row
        + (first_head + head_offsets[:, None]) * stride_output_head
        + dim_offsets[None, :] * stride_output_dim,
        output,
        mask=(row < num_rows)
        & (head_offsets[:, None] < GROUP_SIZE)
        & (dim_offsets[None, :] < HEAD_DIM),
    )


@triton.jit
def _qsa_sparse_paged_gqa_split_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_indices_row,
    stride_indices_column,
    stride_table_req,
    stride_table_page,
    stride_token_to_req,
    stride_partial_output_row,
    stride_partial_output_head,
    stride_partial_output_split,
    stride_partial_output_dim,
    stride_partial_max_row,
    stride_partial_max_head,
    stride_partial_max_split,
    stride_partial_sum_row,
    stride_partial_sum_head,
    stride_partial_sum_split,
    num_rows,
    num_cache_blocks,
    num_requests,
    softmax_scale,
    TOPK: tl.constexpr,
    SPLIT_TOPK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Compute one contiguous TopK split into FP32 partials.

    The split axis is deliberately part of the launch grid, so every program
    locates its own token range from ``program_id(2)``.  No host-side index
    expansion or reduction metadata is needed during graph replay.
    """

    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row * stride_token_to_req)
    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=(head_offsets[:, None] < GROUP_SIZE) & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    # Preserve the single-kernel path's BF16 query scaling/rounding point.
    query = (query * softmax_scale * 1.4426950408889634).to(query.dtype)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)
    split_start = split * SPLIT_TOPK

    for start in tl.range(0, SPLIT_TOPK, BLOCK_N):
        columns = split_start + start + column_offsets
        valid_column = (columns < TOPK) & (start + column_offsets < SPLIT_TOPK)
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns * stride_indices_column,
            mask=valid_column,
            other=-1,
        )
        logical_page = tl.maximum(logical_token, 0) // PAGE_SIZE
        page_offset = tl.maximum(logical_token, 0) % PAGE_SIZE
        valid = (
            (row < num_rows)
            & (request >= 0)
            & (request < num_requests)
            & valid_column
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
        safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + safe_logical_page * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=(dim_offsets[:, None] < HEAD_DIM) & valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )
        scores = tl.dot(query, keys)
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    query_heads = first_head + head_offsets
    max_offsets = (
        row * stride_partial_max_row
        + query_heads * stride_partial_max_head
        + split * stride_partial_max_split
    )
    sum_offsets = (
        row * stride_partial_sum_row
        + query_heads * stride_partial_sum_head
        + split * stride_partial_sum_split
    )
    stat_mask = (row < num_rows) & (head_offsets < GROUP_SIZE)
    tl.store(partial_max_ptr + max_offsets, max_value, mask=stat_mask)
    tl.store(partial_sum_ptr + sum_offsets, normalizer, mask=stat_mask)
    output_offsets = (
        row * stride_partial_output_row
        + query_heads[:, None] * stride_partial_output_head
        + split * stride_partial_output_split
        + dim_offsets[None, :] * stride_partial_output_dim
    )
    tl.store(
        partial_output_ptr + output_offsets,
        accumulator,
        mask=stat_mask[:, None] & (dim_offsets[None, :] < HEAD_DIM),
    )


@triton.jit
def _qsa_sparse_paged_gqa_split_reduce_kernel(
    partial_output_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    gate_ptr,
    output_ptr,
    stride_partial_output_row,
    stride_partial_output_head,
    stride_partial_output_split,
    stride_partial_output_dim,
    stride_partial_max_row,
    stride_partial_max_head,
    stride_partial_max_split,
    stride_partial_sum_row,
    stride_partial_sum_head,
    stride_partial_sum_split,
    stride_gate_row,
    stride_gate_head,
    stride_gate_dim,
    stride_output_row,
    stride_output_head,
    stride_output_dim,
    num_rows,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    APPLY_GATE: tl.constexpr,
) -> None:
    """Stable FP32 merge and BF16 output-gate epilogue."""

    row = tl.program_id(0)
    query_head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_S)
    dim_offsets = tl.arange(0, BLOCK_D)
    split_mask = split_offsets < NUM_SPLITS
    max_offsets = (
        row * stride_partial_max_row
        + query_head * stride_partial_max_head
        + split_offsets * stride_partial_max_split
    )
    sum_offsets = (
        row * stride_partial_sum_row
        + query_head * stride_partial_sum_head
        + split_offsets * stride_partial_sum_split
    )
    partial_max = tl.load(partial_max_ptr + max_offsets, mask=split_mask, other=-1.0e20)
    partial_sum = tl.load(partial_sum_ptr + sum_offsets, mask=split_mask, other=0.0)
    global_max = tl.max(partial_max, axis=0)
    split_scale = tl.math.exp2(partial_max - global_max)
    denominator = tl.sum(partial_sum * split_scale, axis=0)
    partial_offsets = (
        row * stride_partial_output_row
        + query_head * stride_partial_output_head
        + split_offsets[:, None] * stride_partial_output_split
        + dim_offsets[None, :] * stride_partial_output_dim
    )
    partial_output = tl.load(
        partial_output_ptr + partial_offsets,
        mask=split_mask[:, None] & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    numerator = tl.sum(partial_output * split_scale[:, None], axis=0)
    output = tl.where(denominator > 0, numerator / denominator, 0.0)
    if APPLY_GATE:
        gate = tl.load(
            gate_ptr
            + row * stride_gate_row
            + query_head * stride_gate_head
            + dim_offsets * stride_gate_dim,
            mask=(row < num_rows)
            & (query_head < NUM_QUERY_HEADS)
            & (dim_offsets < HEAD_DIM),
            other=0.0,
        )
        # Match the existing single-kernel observable BF16 rounding points.
        output = (
            output.to(tl.bfloat16) * tl.sigmoid(gate.to(tl.float32)).to(tl.bfloat16)
        ).to(tl.bfloat16)
    tl.store(
        output_ptr
        + row * stride_output_row
        + query_head * stride_output_head
        + dim_offsets * stride_output_dim,
        output,
        mask=(row < num_rows)
        & (query_head < NUM_QUERY_HEADS)
        & (dim_offsets < HEAD_DIM),
    )


def _qsa_sparse_paged_attention_split(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    partial_output: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    *,
    num_splits: int,
    softmax_scale: float | None,
    out: torch.Tensor,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    """Run graph-safe split-TopK sparse attention with caller-owned partials."""

    if num_splits not in _QSA_SPLIT_ALLOWED[1:]:
        raise ValueError("QSA split count must be a power of two from 2 through 32")
    expected_output = (q.shape[0], q.shape[1], num_splits, q.shape[2])
    expected_stats = (q.shape[0], q.shape[1], num_splits)
    if partial_output.shape != expected_output or partial_output.dtype != torch.float32:
        raise ValueError("split QSA partial output workspace has an invalid layout")
    if partial_max.shape != expected_stats or partial_max.dtype != torch.float32:
        raise ValueError("split QSA partial max workspace has an invalid layout")
    if partial_sum.shape != expected_stats or partial_sum.dtype != torch.float32:
        raise ValueError("split QSA partial sum workspace has an invalid layout")
    if out.shape != q.shape:
        raise ValueError("split QSA output must match its query")
    if partial_output.device != q.device or partial_max.device != q.device:
        raise ValueError("split QSA workspace must share the query device")
    if partial_sum.device != q.device:
        raise ValueError("split QSA workspace must share the query device")

    scale = q.shape[2] ** -0.5 if softmax_scale is None else softmax_scale
    group_size = q.shape[1] // k_cache.shape[2]
    block_m = max(8, triton.next_power_of_2(group_size))
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    block_n, num_warps, num_stages = qsa_sparse_triton_launch_config(q.device)
    split_topk = triton.cdiv(logical_indices.shape[1], num_splits)
    _qsa_sparse_paged_gqa_split_kernel[(q.shape[0], k_cache.shape[2], num_splits)](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_max,
        partial_sum,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        logical_indices.stride(0),
        logical_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        token_to_req.stride(0),
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        partial_sum.stride(0),
        partial_sum.stride(1),
        partial_sum.stride(2),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        float(scale),
        TOPK=logical_indices.shape[1],
        SPLIT_TOPK=split_topk,
        NUM_SPLITS=num_splits,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        NUM_KV_HEADS=k_cache.shape[2],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _qsa_sparse_paged_gqa_split_reduce_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_max,
        partial_sum,
        gate if gate is not None else out,
        out,
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        partial_sum.stride(0),
        partial_sum.stride(1),
        partial_sum.stride(2),
        gate.stride(0) if gate is not None else 0,
        gate.stride(1) if gate is not None else 0,
        gate.stride(2) if gate is not None else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        q.shape[0],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        HEAD_DIM=q.shape[2],
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=block_d,
        APPLY_GATE=gate is not None,
        num_warps=4,
        num_stages=2,
    )
    return out


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    split_workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 K/V caches.

    ``split_workspace`` is optional so existing callers keep the ABI.  A
    caller-provided workspace enables the split path; otherwise the single
    kernel runs without temporary allocations. Graph ownership stays with the caller.
    Request mappings and each workspace tensor use their own strides.
    """

    if not _is_triton_device(q):
        raise NotImplementedError(
            "paged QSA sparse attention requires an accelerator Triton backend"
        )
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    scale = q.shape[2] ** -0.5 if softmax_scale is None else softmax_scale
    if scale <= 0:
        raise ValueError("QSA softmax scale must be positive")
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    if gate is not None and gate.shape != q.shape:
        raise ValueError("QSA output gate must match its query")
    if gate is not None and gate.dtype != q.dtype:
        raise ValueError("QSA output gate must match its query dtype")
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    # Triton accepts an 8-row dot tile on the supported accelerator backend.
    # TP8 executes three Q heads per replicated KV head, so a minimum of 16
    # spends most of the M tile on padding.  Eight retains Triton's matrix-dot
    # lowering while reducing that waste and is neutral to the device vendor.
    block_m = max(8, triton.next_power_of_2(group_size))
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    block_n, num_warps, num_stages = qsa_sparse_triton_launch_config(q.device)
    num_splits = split_workspace[0].shape[2] if split_workspace is not None else 1
    if num_splits > 1:
        return _qsa_sparse_paged_attention_split(
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            *split_workspace,
            num_splits=num_splits,
            softmax_scale=scale,
            out=out,
            gate=gate,
        )
    _qsa_sparse_paged_gqa_kernel[(q.shape[0], k_cache.shape[2])](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        gate if gate is not None else out,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        logical_indices.stride(0),
        logical_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        token_to_req.stride(0),
        gate.stride(0) if gate is not None else 0,
        gate.stride(1) if gate is not None else 0,
        gate.stride(2) if gate is not None else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        float(scale),
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        NUM_KV_HEADS=k_cache.shape[2],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        APPLY_GATE=gate is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = ["qsa_sparse_paged_attention", "qsa_sparse_split_count"]
