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

from flaggems_vllm.utils import libentry
from flaggems_vllm.utils.triton_version_utils import HAS_TLE

if HAS_TLE:
    from triton.experimental.tle import language as tle

    @triton.jit
    def indexer_scale_tle_tile(
        weights_ptr,
        q_scale_ptr,
        output_ptr,
        total,
        begin,
        SCALE: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_TAIL: tl.constexpr,
    ):
        offsets = begin + tl.arange(0, BLOCK)
        if HAS_TAIL and begin + BLOCK > total:
            mask = offsets < total
            weights = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
            query = tl.load(q_scale_ptr + offsets, mask=mask, other=0.0)
            tl.store(output_ptr + offsets, weights * query * SCALE, mask=mask)
        else:
            weights_ub = tle.dsa.alloc([BLOCK], tl.float32, tle.dsa.ascend.UB)
            query_ub = tle.dsa.alloc([BLOCK], tl.float32, tle.dsa.ascend.UB)
            tle.dsa.copy(
                weights_ptr + offsets, weights_ub, [BLOCK], inter_no_alias=True
            )
            tle.dsa.copy(q_scale_ptr + offsets, query_ub, [BLOCK], inter_no_alias=True)
            product = tle.dsa.to_tensor(weights_ub, writable=False) * tle.dsa.to_tensor(
                query_ub, writable=False
            )
            output_ub = tle.dsa.to_buffer(product * SCALE, space=tle.dsa.ascend.UB)
            # Current TLE copy lowering uses full tiles; partial writes stay masked.
            tle.dsa.copy(output_ub, output_ptr + offsets, [BLOCK], inter_no_alias=True)

    @libentry()
    @triton.jit
    def fused_indexer_weight_scale_tle_kernel(
        weights_ptr,
        q_scale_ptr,
        output_ptr,
        total,
        SCALE: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_TAIL: tl.constexpr,
        PIPELINED: tl.constexpr,
    ):
        if PIPELINED:
            for begin in tle.dsa.pipeline(
                tl.program_id(0) * BLOCK,
                total,
                tl.num_programs(0) * BLOCK,
                num_stages=2,
            ):
                indexer_scale_tle_tile(
                    weights_ptr,
                    q_scale_ptr,
                    output_ptr,
                    total,
                    begin,
                    SCALE,
                    BLOCK,
                    HAS_TAIL,
                )
        else:
            indexer_scale_tle_tile(
                weights_ptr,
                q_scale_ptr,
                output_ptr,
                total,
                tl.program_id(0) * BLOCK,
                SCALE,
                BLOCK,
                HAS_TAIL,
            )


@libentry()
@triton.jit
def indexer_scale_dense_kernel(
    weights_ptr,
    q_scale_ptr,
    output_ptr,
    total,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    weights = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
    query = tl.load(q_scale_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, weights * query * SCALE, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["scale"])
def fused_indexer_weight_scale_kernel(
    weights_ptr,
    q_scale_ptr,
    output_ptr,
    total,
    scale,
    HEADS: tl.constexpr,
    WEIGHT_STRIDE_T: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    QUERY_STRIDE_T: tl.constexpr,
    QUERY_STRIDE_H: tl.constexpr,
    DENSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    if DENSE:
        weight_offsets = offsets
        query_offsets = offsets
    else:
        tokens = offsets // HEADS
        heads = offsets % HEADS
        weight_offsets = tokens * WEIGHT_STRIDE_T + heads * WEIGHT_STRIDE_H
        query_offsets = tokens * QUERY_STRIDE_T + heads * QUERY_STRIDE_H
    weights = tl.load(weights_ptr + weight_offsets, mask=mask, other=0.0).to(tl.float32)
    query_scale = tl.load(q_scale_ptr + query_offsets, mask=mask, other=0.0)
    output = weights * query_scale * scale
    tl.store(output_ptr + offsets, output, mask=mask)


def fused_indexer_weight_scale(
    weights: torch.Tensor, q_scale: torch.Tensor, scale: float
) -> torch.Tensor:
    """Scale floating [T,H] weights with FP32 [T,H,1] query scales.

    FP16/BF16 weights are promoted to FP32. The new contiguous output is
    FP32 [T,H]. Supports ordinary strides and inference forward only.
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
    if weights.device.type != "npu":
        raise NotImplementedError("indexer weight scaling requires Ascend")
    if q_scale.device != weights.device:
        raise ValueError("weights and q_scale must be on the same device")
    output = torch.empty((tokens, heads), dtype=torch.float32, device=weights.device)
    total = tokens * heads
    if total == 0:
        return output
    is_dense = weights.is_contiguous() and q_scale.is_contiguous()
    if is_dense and weights.dtype == torch.float32:
        block = min(8192, triton.next_power_of_2(total))
        if (
            HAS_TLE
            and total >= 8
            and weights.data_ptr() % 32 == 0
            and q_scale.data_ptr() % 32 == 0
        ):
            floor_block = block // 2 if block > total else block
            cores = (
                torch.npu.get_device_properties(weights.device).vector_core_num
                if total > 8192
                else 1
            )
            is_pipelined = total > 8192 and total > cores * 8192
            if is_pipelined:
                block = 4096
                grid = min(triton.cdiv(total, block), cores)
            elif total % floor_block == 0 or total - floor_block <= floor_block // 4:
                block = floor_block
                grid = triton.cdiv(total, block)
            else:
                indexer_scale_dense_kernel[(triton.cdiv(total, block),)](
                    weights,
                    q_scale,
                    output,
                    total,
                    scale,
                    block,
                )
                return output
            fused_indexer_weight_scale_tle_kernel[(grid,)](
                weights,
                q_scale,
                output,
                total,
                scale,
                block,
                total % block != 0,
                is_pipelined,
            )
        else:
            indexer_scale_dense_kernel[(triton.cdiv(total, block),)](
                weights,
                q_scale,
                output,
                total,
                scale,
                block,
            )
        return output
    block = min(8192 if is_dense else 1024, triton.next_power_of_2(total))
    fused_indexer_weight_scale_kernel[(triton.cdiv(total, block),)](
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
    )
    return output
