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
