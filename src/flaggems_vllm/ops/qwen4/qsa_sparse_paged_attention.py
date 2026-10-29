# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors

"""QSA sparse paged attention ported from vLLM f47792af7f.

Launch geometry: both the split count and the main kernel's
BLOCK_N / num_warps / num_stages come from the backend heuristics system
(`runtime.get_heuristic_config("qsa_sparse_paged_attention")`), which keeps
the per-device policies out of the op. Both kernels are launched directly
(no libentry/libtuner on the host path).

Autotune exemption (AGENTS.md rule 3): the launch tables are static backend
policies, so this op keeps no libtuner candidate lists in tune_configs.yaml.

NUM_SPLITS is fixed by the host before any launch; the main kernel records,
per row, how many splits actually wrote partials, and the merge kernel
applies exactly that count (early exit at <= 1, plus tl.where discard on the
loads) so the two kernels cannot disagree about the workspace layout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime


@triton.jit(do_not_specialize=["num_rows", "num_requests", "num_cache_blocks"])
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    split_counts_ptr,
    output_ptr,
    softmax_scale,
    output_scale,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP8: tl.constexpr,
    DEFER_V_LOAD: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)

    # The packed selection buffer carries one TRAILING COUNT COLUMN per row
    # (column TOPK of a TOPK+1-wide buffer): the row's valid-entry count,
    # written by the expand kernel. It is never a token index — the tile loop
    # and the index load below only ever cover columns [0, TOPK).
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    tile_end = tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N)
    # Splits without work exit before touching Q; split 0 always stays so
    # rows with zero or one tile still get their (zero or gated) final output.
    if split_id > 0 and split_id >= tile_end:
        return
    if NUM_SPLITS > 1 and split_id == 0:
        # How many splits wrote mergeable partials for this row: 0 when the
        # row was finalized directly below. The merge kernel loads the full
        # allocated split range and discards entries at or above this count
        # with tl.where.
        tl.store(
            split_counts_ptr + row,
            tl.where(tile_end <= 1, 0, tl.minimum(tile_end, NUM_SPLITS)),
        )

    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    # The host passes the gate through a no-copy contiguous view normalized to
    # Q's shape, so its strides are fixed by the shape.
    stride_output_gate_row = NUM_QUERY_HEADS * HEAD_DIM
    stride_output_gate_head = HEAD_DIM

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    # softmax_scale is the host-side attention scale (1/sqrt(head_dim), with the
    # fp8 K dequant scale already pre-multiplied in); convert to log2 units once
    # here for the exp2-based online softmax.
    score_scale = softmax_scale * 1.4426950408889634

    for tile in range(split_id, tile_end, NUM_SPLITS):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        if NUM_SPLITS == 1:
            keys = tl.load(
                k_cache_ptr
                + safe_page[:, None] * stride_k_block
                + page_offset[:, None] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            keys = tl.permute(keys, (1, 0))
        else:
            keys = tl.load(
                k_cache_ptr
                + safe_page[:, None] * stride_k_block
                + page_offset[:, None] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            keys = tl.permute(keys, (1, 0))
        if not DEFER_V_LOAD:
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
        if IS_FP8:
            # e4m3 -> Q dtype is exact; keep the QK dot in Q's dtype.
            keys = keys.to(query.dtype)
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16; for fp8
        # caches the K dequant scale is already folded into softmax_scale on the
        # host.
        scores *= score_scale
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        if DEFER_V_LOAD:
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
        if IS_FP8:
            # P <= 1 after online softmax, so fp16 has the range and a wider
            # mantissa than bf16 for the PV dot.
            values = values.to(tl.float16)
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    # Fold the fp8 V dequant scale (output_scale, 1.0 for bf16) into a per-row
    # reciprocal normalizer, so the output is a per-row multiply rather than a
    # HEAD_DIM-wide scale. The split-K LSE below keeps the unscaled normalizer,
    # so the merge stays correct.
    inv_normalizer = output_scale / tl.maximum(normalizer, 1.0e-20)
    normalized_output = tl.where(
        has_values[:, None],
        accumulator * inv_normalizer[:, None],
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    # Rows with at most one tile (including empty selections) are finalized
    # directly by split 0 — BF16 rounding, then the gate in FP32 — and the
    # merge kernel skips them.
    if NUM_SPLITS == 1 or tile_end <= 1:
        # Preserve the unfused path's BF16 attention-output rounding before
        # applying the gate in FP32.
        normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + (first_head + head_offsets[:, None]) * stride_output_gate_head
            + dim_offsets[None, :],
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        tl.store(
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    split_counts_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    # Number of splits that wrote mergeable partials for this row (0 when the
    # main kernel finalized the row directly, including empty selections).
    # Written by the same launch; rows at or above this count are discarded
    # below before they can contribute, so stale workspace never leaks in.
    merge_splits = tl.load(split_counts_ptr + row)
    if merge_splits <= 1:
        return
    # Gate strides are fixed by the normalized contiguous view (see main).
    stride_output_gate_row = NUM_QUERY_HEADS * HEAD_DIM
    stride_output_gate_head = HEAD_DIM
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    # Loads are bounded by the constexpr NUM_SPLITS (the allocated workspace
    # size, always in-bounds) so they do not serialize behind the split_counts
    # load; the count is applied afterwards with tl.where, which provably
    # discards every value at or above merge_splits before it can contribute
    # (lse -> -inf, partial -> 0.0, so NaN bit patterns cannot propagate).
    bounds_mask = split_offsets < NUM_SPLITS
    split_mask = split_offsets < merge_splits
    lse_raw = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=bounds_mask,
        other=-float("inf"),
    )
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    partial_raw = tl.load(
        partial_output_ptr
        + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=bounds_mask[:, None],
        other=0.0,
    )
    lse = tl.where(split_mask, lse_raw, -float("inf"))
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.where(split_mask[:, None], partial_raw, 0.0)
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    # Preserve the unfused path's BF16 attention-output rounding before
    # applying the gate in FP32.
    merged = merged.to(output_ptr.dtype.element_ty)
    output_gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


# Reference tile width for the NUM_SPLITS clamp below. More splits than the
# tile count only widen the merge kernel's per-row read without adding
# main-kernel work they can parallelize.
_SPLIT_CAP_TILE_WIDTH = 64


# Launch policies (split count and the main kernel's launch config) come from
# the backend heuristics system; resolved once at import (the heuristics
# table is static backend configuration, never call data).
_QSA_HEURISTICS = runtime.get_heuristic_config("qsa_sparse_paged_attention")
_QSA_MIN_BLOCK_M = (_QSA_HEURISTICS or {}).get("MIN_BLOCK_M", 1)

# Explicit conservative defaults when the backend has no policy entry (a
# single split and a mid-size tile are always correct); the NVIDIA backend
# installs the real policies through HEURISTICS_CONFIGS.
_QSA_SPLIT_HEUR_DEFAULT = 1
_QSA_LAUNCH_CONFIG_DEFAULT = (64, 4, 1)


def _select_splits(
    num_rows: int,
    num_kv_heads: int,
    use_prefill_config: bool,
    is_fp8: bool,
) -> int:
    """Resolve the target split count from the backend heuristics system."""
    heuristics = _QSA_HEURISTICS
    if not heuristics or "NUM_SPLITS" not in heuristics:
        return _QSA_SPLIT_HEUR_DEFAULT
    return heuristics["NUM_SPLITS"](
        {
            "num_rows": num_rows,
            "num_kv_heads": num_kv_heads,
            "use_prefill_config": use_prefill_config,
            "is_fp8": is_fp8,
        }
    )


def _select_launch_config(
    num_rows: int, num_splits: int, group_size: int, is_fp8: bool
) -> tuple[int, int, int]:
    """(BLOCK_N, num_warps, num_stages) from the backend heuristics system."""
    heuristics = _QSA_HEURISTICS
    if not heuristics or "LAUNCH_CONFIG" not in heuristics:
        return _QSA_LAUNCH_CONFIG_DEFAULT
    return heuristics["LAUNCH_CONFIG"](
        {
            "num_rows": num_rows,
            "num_splits": num_splits,
            "group_size": group_size,
            "is_fp8": is_fp8,
        }
    )


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    *,
    output_gate: torch.Tensor,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 or FP8-e4m3 K/V caches.

    With fp8 caches, k_scale/v_scale are the layer's per-tensor dequant scales
    as host floats (e.g. layer._k_scale_float/_v_scale_float): k_scale is
    pre-multiplied into the softmax scale and v_scale becomes the kernel's
    output scale, so the kernel needs no device scale buffers.

    logical_indices is the PACKED selection buffer: [rows, selection_width + 1]
    with the trailing column holding each row's valid-entry count (written by
    the expand kernel; never a token index). The kernel reads it as the
    tile-loop bound. use_prefill_config only steers the backend split policy;
    see runtime.get_heuristic_config("qsa_sparse_paged_attention").

    out= may be any non-overlapping tensor with matching shape/dtype/device;
    storage shared with an input is rejected because the kernels read the
    inputs and write out in the same launch, with no ordering between them.
    """
    q_shape = q.shape
    k_shape = k_cache.shape
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q_shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q_shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] < 2:
        raise ValueError(
            "QSA packed indices need selection columns plus the count column"
        )
    if q_shape[2] != k_shape[3] or q_shape[1] % k_shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q_shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == v_cache.dtype
    is_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if is_fp8:
        assert k_scale is not None and v_scale is not None
        # Host pre-multiply: fold the K dequant scale into the attention scale
        # and pass V's dequant scale as the kernel's output scale. Computed on
        # every call so a changed scale always takes effect.
        softmax_scale = (head_dim**-0.5) * float(k_scale)
        output_scale = float(v_scale)
    else:
        assert k_cache.dtype == torch.bfloat16
        softmax_scale = head_dim**-0.5
        output_scale = 1.0
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    device = q.device
    assert (
        k_cache.device == v_cache.device == device
        and logical_indices.device == block_table.device == device
        and token_to_req.device == device
    )
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1

    # Gate contract: exactly one value per (row, head, dim), contiguous so the
    # no-copy view below is legal. BF16 and FP32 gates are both supported.
    if output_gate.numel() != q.numel():
        raise ValueError("QSA output gate must have one value per query element")
    if output_gate.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("QSA output gate must be bfloat16 or float32")
    assert output_gate.device == device
    assert output_gate.is_contiguous()
    # No-copy normalization to Q's shape: for a contiguous [T, Hq*D] gate this
    # yields exactly the [T, Hq, D] strides the kernel indexes with.
    output_gate_view = output_gate.view_as(q)

    out_is_caller_provided = out is not None
    if out is None:
        out = torch.empty_like(q)
    else:
        if out.shape != q_shape:
            raise ValueError("QSA sparse output must match its query")
        assert out.dtype == q.dtype and out.device == device
        assert out.stride(2) == 1
    num_rows = q_shape[0]
    if not num_rows:
        return out
    if out_is_caller_provided and out.numel():
        # The kernels read the inputs and write out in the same launch with
        # no ordering guarantee, so an out that overlaps an input corrupts
        # the input while it is still being read.
        out_storage = out.untyped_storage().data_ptr()
        for tensor in (
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            output_gate,
        ):
            if tensor.untyped_storage().data_ptr() == out_storage:
                raise ValueError(
                    "QSA sparse attention out must not share storage with an input"
                )

    group_size = q_shape[1] // k_shape[2]
    block_m = max(_QSA_MIN_BLOCK_M, triton.next_power_of_2(group_size))
    selection_width = logical_indices.shape[1] - 1  # trailing column is the count
    target_splits = _select_splits(num_rows, k_shape[2], use_prefill_config, is_fp8)
    # Clamp splits to the tile count at the reference width so the merge
    # kernel never reads more partials per row than this cap allows. Splits
    # beyond the launch BLOCK_N's tile count exit empty in the kernel, and
    # split_counts keeps the merge correct for any clamp outcome.
    num_splits = min(target_splits, triton.cdiv(selection_width, _SPLIT_CAP_TILE_WIDTH))
    block_n, num_warps, num_stages = _select_launch_config(
        num_rows, num_splits, group_size, is_fp8
    )

    q_stride = q.stride()
    k_stride = k_cache.stride()
    v_stride = v_cache.stride()
    out_stride = out.stride()

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
        split_counts = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits. Freshly allocated on every call; nothing call-specific is
        # retained beyond the launch.
        partial_output = torch.empty(
            (num_splits, *q_shape), dtype=torch.float32, device=device
        )
        partial_lse = torch.empty(
            (num_splits, num_rows, q_shape[1]),
            dtype=torch.float32,
            device=device,
        )
        split_counts = torch.empty(num_rows, dtype=torch.int32, device=device)

    _qsa_sparse_paged_gqa_splitk_kernel[(num_rows, k_shape[2], num_splits)](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        split_counts,
        out,
        softmax_scale,
        output_scale,
        output_gate_view,
        q_stride[0],
        q_stride[1],
        k_stride[0],
        k_stride[1],
        k_stride[2],
        v_stride[0],
        v_stride[1],
        v_stride[2],
        logical_indices.stride(0),
        block_table.stride(0),
        out_stride[0],
        out_stride[1],
        num_rows,
        k_shape[0],
        block_table.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=k_shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q_shape[2],
        NUM_QUERY_HEADS=q_shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_FP8=is_fp8,
        DEFER_V_LOAD=num_splits > 1 or num_rows >= 4096,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(num_rows, q_shape[1])](
        partial_output,
        partial_lse,
        split_counts,
        out,
        output_gate_view,
        out_stride[0],
        out_stride[1],
        num_rows,
        HEAD_DIM=q_shape[2],
        NUM_QUERY_HEADS=q_shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out
