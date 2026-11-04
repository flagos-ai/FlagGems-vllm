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
from flaggems_vllm.utils import libentry


@libentry()
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
    block = 256
    with torch_device_fn.device(weights.device):
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
            weights.is_contiguous() and q_scale.is_contiguous(),
            block,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
