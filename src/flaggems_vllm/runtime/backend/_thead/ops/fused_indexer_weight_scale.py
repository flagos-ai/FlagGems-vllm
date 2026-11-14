# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import device, torch_device_fn
from flaggems_vllm.utils import libentry


@libentry()
@triton.jit(do_not_specialize=["scale"])
def fused_indexer_weight_scale_kernel(
    weights,
    query_scale,
    output,
    total,
    scale,
    HEADS: tl.constexpr,
    WEIGHT_STRIDE_T: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    QUERY_STRIDE_T: tl.constexpr,
    QUERY_STRIDE_H: tl.constexpr,
    DENSE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    if DENSE:
        weight_offsets = offsets
        query_offsets = offsets
    else:
        token = offsets // HEADS
        head = offsets % HEADS
        weight_offsets = token * WEIGHT_STRIDE_T + head * WEIGHT_STRIDE_H
        query_offsets = token * QUERY_STRIDE_T + head * QUERY_STRIDE_H
    weight = tl.load(weights + weight_offsets, mask, other=0).to(tl.float32)
    query = tl.load(query_scale + query_offsets, mask, other=0)
    # Preserve the native FP32 intermediate and the order of the two products.
    result = (weight * query) * scale
    tl.store(output + offsets, result, mask)


def fused_indexer_weight_scale(
    weights: torch.Tensor, q_scale: torch.Tensor, scale: float
) -> torch.Tensor:
    """Scale [T,H] weights by FP32 [T,H,1] query scales and a scalar.

    FP16/BF16 weights are promoted to FP32. The output is a new contiguous
    FP32 [T,H] tensor. Strided inputs are supported for inference forward.
    """
    if weights.ndim != 2:
        raise ValueError("weights must have shape [tokens, heads]")
    tokens, heads = weights.shape
    if q_scale.shape != (tokens, heads, 1):
        raise ValueError("q_scale must have shape [tokens, heads, 1]")
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("weights must be FP16, BF16, or FP32")
    if q_scale.dtype != torch.float32:
        raise TypeError("q_scale must be FP32")
    if weights.device.type != device.name or q_scale.device != weights.device:
        raise ValueError("weights and q_scale must be on the same accelerator")
    if weights.requires_grad or q_scale.requires_grad:
        raise NotImplementedError("indexer weight scaling supports inference only")
    if not isinstance(scale, (int, float)):
        raise TypeError("scale must be a Python scalar")
    output = torch.empty((tokens, heads), dtype=torch.float32, device=weights.device)
    total = tokens * heads
    if total == 0:
        return output
    # This pointwise kernel has no reduction or shared-memory tuning dimension.
    # Use an explicit launch policy instead of tuning every decode token count.
    block_size = 256
    with torch_device_fn.device(weights.device):
        fused_indexer_weight_scale_kernel[(triton.cdiv(total, block_size),)](
            weights,
            q_scale,
            output,
            total,
            float(scale),
            heads,
            weights.stride(0),
            weights.stride(1),
            q_scale.stride(0),
            q_scale.stride(1),
            weights.is_contiguous() and q_scale.is_contiguous(),
            block_size,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
