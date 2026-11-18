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

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn


@triton.jit(do_not_specialize=["scale"])
def indexer_weight_scale_kernel(
    weights_ptr,
    query_scale_ptr,
    output_ptr,
    total,
    scale,
    HEADS: tl.constexpr,
    WEIGHT_STRIDE_T: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    QUERY_STRIDE_T: tl.constexpr,
    QUERY_STRIDE_H: tl.constexpr,
    IS_DENSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    if IS_DENSE:
        weight_offsets = offsets
        query_offsets = offsets
    else:
        tokens = offsets // HEADS
        heads = offsets % HEADS
        weight_offsets = tokens * WEIGHT_STRIDE_T + heads * WEIGHT_STRIDE_H
        query_offsets = tokens * QUERY_STRIDE_T + heads * QUERY_STRIDE_H
    weights = tl.load(weights_ptr + weight_offsets, mask=mask, other=0).to(tl.float32)
    query_scale = tl.load(query_scale_ptr + query_offsets, mask=mask, other=0)
    scaled_weights = weights * query_scale
    output = scaled_weights * scale
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit(do_not_specialize=["scale"])
def dense_indexer_weight_scale_kernel(
    weights_ptr,
    query_scale_ptr,
    output_ptr,
    total,
    scale,
    BLOCK: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = True if FULL_BLOCKS else offsets < total
    weights = tl.load(weights_ptr + offsets, mask=mask, other=0)
    query_scale = tl.load(query_scale_ptr + offsets, mask=mask, other=0)
    scaled_weights = weights * query_scale
    output = scaled_weights * scale
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit(do_not_specialize=["scale"])
def tiny_indexer_weight_scale_kernel(
    weights_ptr,
    query_scale_ptr,
    output_ptr,
    scale,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = True if TOTAL == BLOCK else offsets < TOTAL
    weights = tl.load(weights_ptr + offsets, mask=mask, other=0)
    query_scale = tl.load(query_scale_ptr + offsets, mask=mask, other=0)
    scaled_weights = weights * query_scale
    output = scaled_weights * scale
    tl.store(output_ptr + offsets, output, mask=mask)


def fused_indexer_weight_scale(
    weights: torch.Tensor, q_scale: torch.Tensor, scale: float
) -> torch.Tensor:
    """Scale [T,H] floating weights with FP32 [T,H,1] query scales."""
    if weights.ndim != 2:
        raise ValueError("weights must have shape [tokens, heads]")
    tokens, heads = weights.shape
    if q_scale.shape != (tokens, heads, 1):
        raise ValueError("q_scale must have shape [tokens, heads, 1]")
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("weights must be FP16, BF16, or FP32")
    if q_scale.dtype != torch.float32:
        raise TypeError("q_scale must be FP32")
    if weights.device.type != "musa":
        raise NotImplementedError("indexer weight scaling requires MUSA")
    if q_scale.device != weights.device:
        raise ValueError("weights and q_scale must be on the same device")
    output = torch.empty((tokens, heads), dtype=torch.float32, device=weights.device)
    total = tokens * heads
    if total == 0:
        return output
    is_dense = weights.is_contiguous() and q_scale.is_contiguous()
    is_dense_fp32 = is_dense and weights.dtype == torch.float32
    if is_dense_fp32 and total <= 64:
        block = 32 if total <= 32 else 64
        with torch_device_fn.device(weights.device):
            tiny_indexer_weight_scale_kernel[(1,)](
                weights,
                q_scale,
                output,
                scale,
                total,
                block,
                num_warps=1,
                num_stages=1,
                enable_fp_fusion=False,
            )
        return output
    should_use_dense_kernel = is_dense_fp32 and total <= 8192
    # Measured small-grid schedules keep adjacent CTA ranges separate.
    if is_dense_fp32 and 128 < total <= 512:
        block, num_warps = 32, 1
    elif is_dense_fp32 and 512 < total <= 768:
        block, num_warps = 64, 2
    elif is_dense_fp32 and (total <= 2048 or 4096 < total <= 8192):
        block, num_warps = 128, 4
    else:
        block, num_warps = 256, 4
    with torch_device_fn.device(weights.device):
        if should_use_dense_kernel:
            dense_indexer_weight_scale_kernel[(triton.cdiv(total, block),)](
                weights,
                q_scale,
                output,
                total,
                scale,
                block,
                total % block == 0,
                num_warps=num_warps,
                num_stages=1,
                enable_fp_fusion=False,
            )
        else:
            indexer_weight_scale_kernel[(triton.cdiv(total, block),)](
                weights,
                q_scale,
                output,
                total,
                scale,
                heads,
                weights.stride(0),
                weights.stride(1),
                q_scale.stride(0),
                q_scale.stride(1),
                is_dense,
                block,
                num_warps=num_warps,
                enable_fp_fusion=False,
            )
    return output
