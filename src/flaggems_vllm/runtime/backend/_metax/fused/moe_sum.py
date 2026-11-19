# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.moe_sum import moe_sum as generic_moe_sum

logger = logging.getLogger(__name__)

_QWEN_TOPKS = (8, 10)
_QWEN_HIDDEN_SIZES = (2048, 4096)


@triton.jit
def _qwen_moe_sum_kernel(
    input_ptr,
    output_ptr,
    router_weights_ptr,
    router_weights_stride_token,
    router_weights_stride_topk,
    num_tokens,
    hidden_size: tl.constexpr,
    TOPK: tl.constexpr,
    APPLY_ROUTER_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size

    input_base = input_ptr + token_idx * TOPK * hidden_size + hidden_offsets
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for expert_idx in tl.static_range(0, TOPK):
        values = tl.load(
            input_base + expert_idx * hidden_size,
            mask=hidden_mask,
            other=0.0,
        )
        if APPLY_ROUTER_WEIGHT:
            router_weight = tl.load(
                router_weights_ptr
                + token_idx * router_weights_stride_token
                + expert_idx * router_weights_stride_topk
            )
            values = values.to(tl.float32) * router_weight.to(tl.float32)
        acc += values.to(tl.float32)

    output_offsets = token_idx * hidden_size + hidden_offsets
    tl.store(
        output_ptr + output_offsets,
        acc.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


def _metax_moe_sum_block_size(hidden_size: int) -> int:
    if hidden_size <= 256:
        return 256
    if hidden_size <= 512:
        return 512
    return 1024


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
    router_weights: torch.Tensor | None = None,
):
    """Use a statically-unrolled reduction for contiguous Qwen MoE outputs."""
    num_tokens, topk, hidden_size = input.shape
    if (
        topk not in _QWEN_TOPKS
        or hidden_size not in _QWEN_HIDDEN_SIZES
        or not input.is_contiguous()
        or not output.is_contiguous()
    ):
        assert router_weights is None
        return generic_moe_sum(input, output)
    if router_weights is not None:
        assert router_weights.shape == input.shape[:2]
        router_weights_strides = router_weights.stride()
    else:
        router_weights_strides = (0, 0)

    logger.debug("GEMS_METAX QWEN MOE SUM")
    use_qwen3_5_launch = topk == 10
    block_size = 256 if use_qwen3_5_launch else _metax_moe_sum_block_size(hidden_size)
    grid = (num_tokens, triton.cdiv(hidden_size, block_size))
    launch_kwargs = {"num_warps": 4 if use_qwen3_5_launch else 8}
    if use_qwen3_5_launch:
        launch_kwargs["num_stages"] = 1
    _qwen_moe_sum_kernel[grid](
        input,
        output,
        input if router_weights is None else router_weights,
        router_weights_strides[0],
        router_weights_strides[1],
        num_tokens,
        hidden_size,
        TOPK=topk,
        APPLY_ROUTER_WEIGHT=router_weights is not None,
        BLOCK_SIZE=block_size,
        **launch_kwargs,
    )


@triton.jit
def _weighted_sum(
    ROUTE_OUTPUT,
    WEIGHTS,
    OUTPUT,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    RT: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUTE_H: tl.constexpr = 0,
    OUTPUT_START: tl.constexpr = 0,
):
    RH: tl.constexpr = H if ROUTE_H == 0 else ROUTE_H
    tl.static_assert(0 <= OUTPUT_START and OUTPUT_START + RH <= H)
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slots = tl.arange(0, RT)
    values = tl.load(
        ROUTE_OUTPUT + (token * TOPK + slots[:, None]) * RH + columns[None, :],
        (slots[:, None] < TOPK) & (columns[None, :] < RH),
        0,
    ).to(tl.float32)
    weights = tl.load(WEIGHTS + token * TOPK + slots, slots < TOPK, 0).to(tl.float32)
    result = tl.sum(values * weights[:, None], axis=0)
    tl.store(
        OUTPUT + token * H + OUTPUT_START + columns,
        result.to(tl.bfloat16),
        columns < RH,
    )


def moe_weighted_sum_out(route_output, weights, out, *, column_start=0):
    """Weight BF16 expert outputs in FP32, then store one BF16 output slice.

    route_output has shape [tokens * topk, width]. Only columns
    [column_start, column_start + width) of out are modified. Input buffers
    must not share storage with out. No cross-call workspace is retained.
    """
    if not route_output.is_cuda:
        raise NotImplementedError("This reduction is qualified on MetaX")
    if type(column_start) is not int:
        raise TypeError("column_start must be an integer column index")
    if weights.ndim != 2 or out.ndim != 2 or route_output.ndim != 2:
        raise ValueError("Expected matrix inputs and output")
    tokens, topk = weights.shape
    rows, width = route_output.shape
    if (
        rows != tokens * topk
        or out.shape[0] != tokens
        or not 1 <= topk <= 16
        or width <= 0
        or column_start < 0
        or column_start + width > out.shape[1]
    ):
        raise ValueError("Invalid route shape or output slice")
    if (
        route_output.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
        or weights.dtype != torch.float32
    ):
        raise ValueError("Expected BF16 values/output and FP32 route weights")
    if not all(
        t.is_contiguous() and t.device == out.device
        for t in (out, route_output, weights)
    ):
        raise ValueError("All tensors must be contiguous on one device")
    # Triton computes the flattened row/column indices with int32 arithmetic.
    if max(route_output.numel(), weights.numel(), out.numel()) >= 2**31:
        raise ValueError("Weighted reduction exceeds the supported index range")
    if tokens == 0:
        return
    if out.untyped_storage().data_ptr() in (
        route_output.untyped_storage().data_ptr(),
        weights.untyped_storage().data_ptr(),
    ):
        raise ValueError("Output may not alias an input")
    _weighted_sum[(tokens, triton.cdiv(width, 256))](
        route_output,
        weights,
        out,
        out.shape[1],
        topk,
        triton.next_power_of_2(topk),
        256,
        ROUTE_H=width,
        OUTPUT_START=column_start,
        num_warps=4,
        enable_fp_fusion=False,
    )
