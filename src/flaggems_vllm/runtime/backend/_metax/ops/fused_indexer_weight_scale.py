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


@triton.jit
def _indexer_weight_scale_vector_tile(
    weights_ptr,
    query_ptr,
    output_ptr,
    offsets,
    TOTAL: tl.constexpr,
    SCALE: tl.constexpr,
    MASKED: tl.constexpr,
):
    mask = offsets < TOTAL if MASKED else True
    weights = tl.load(weights_ptr + offsets, mask, other=0)
    query = tl.load(query_ptr + offsets, mask, other=0)
    output = (weights * query) * SCALE
    scale_value = tl.full((), SCALE, tl.float32)
    zero_sign = (
        weights.to(tl.uint32, bitcast=True)
        ^ query.to(tl.uint32, bitcast=True)
        ^ scale_value.to(tl.uint32, bitcast=True)
    ) & 0x80000000
    # Preserve IEEE zero signs when MACA folds multiplication by a power of two.
    output_bits = (output.to(tl.uint32, bitcast=True) & 0x7FFFFFFF) | zero_sign
    tl.store(output_ptr + offsets, output_bits.to(tl.float32, bitcast=True), mask)


@libentry()
@triton.jit
def indexer_weight_scale_vector_kernel(
    weights_ptr,
    query_ptr,
    output_ptr,
    TOTAL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _indexer_weight_scale_vector_tile(
        weights_ptr, query_ptr, output_ptr, offsets, TOTAL, SCALE, TOTAL % BLOCK != 0
    )


@libentry()
@triton.jit
def indexer_weight_scale_short_tail_kernel(
    weights_ptr,
    query_ptr,
    output_ptr,
    TOTAL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # The branch is uniform within a CTA; only the last CTA needs a mask.
    if tl.program_id(0) < TOTAL // BLOCK:
        _indexer_weight_scale_vector_tile(
            weights_ptr, query_ptr, output_ptr, offsets, TOTAL, SCALE, False
        )
    else:
        _indexer_weight_scale_vector_tile(
            weights_ptr, query_ptr, output_ptr, offsets, TOTAL, SCALE, True
        )


@libentry()
@triton.jit
def indexer_weight_scale_kernel(
    weights_ptr,
    query_ptr,
    output_ptr,
    TOTAL: tl.constexpr,
    HEADS: tl.constexpr,
    WEIGHT_STRIDE_T: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    QUERY_STRIDE_T: tl.constexpr,
    QUERY_STRIDE_H: tl.constexpr,
    SCALE: tl.constexpr,
    IS_DENSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    if IS_DENSE:
        weight_offsets = offsets
        query_offsets = offsets
    else:
        tokens = offsets // HEADS
        heads = offsets % HEADS
        weight_offsets = tokens * WEIGHT_STRIDE_T + heads * WEIGHT_STRIDE_H
        query_offsets = tokens * QUERY_STRIDE_T + heads * QUERY_STRIDE_H
    weights = tl.load(weights_ptr + weight_offsets, mask, other=0).to(tl.float32)
    query = tl.load(query_ptr + query_offsets, mask, other=0)
    output = (weights * query) * SCALE
    # Preserve IEEE zero signs even when MACA folds multiplication by a power of two.
    scale_value = tl.full((), SCALE, tl.float32)
    zero_sign = (
        weights.to(tl.uint32, bitcast=True)
        ^ query.to(tl.uint32, bitcast=True)
        ^ scale_value.to(tl.uint32, bitcast=True)
    ) & 0x80000000
    output_bits = (output.to(tl.uint32, bitcast=True) & 0x7FFFFFFF) | zero_sign
    output = output_bits.to(tl.float32, bitcast=True)
    tl.store(output_ptr + offsets, output, mask)


def fused_indexer_weight_scale(
    weights: torch.Tensor, q_scale: torch.Tensor, scale: float
) -> torch.Tensor:
    """Return FP32 [T,H] indexer weights; inference forward only."""
    if weights.ndim != 2:
        raise ValueError("weights must have shape [tokens, heads]")
    tokens, heads = weights.shape
    if q_scale.shape != (tokens, heads, 1):
        raise ValueError("q_scale must have shape [tokens, heads, 1]")
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("weights must be FP16, BF16, or FP32")
    if q_scale.dtype != torch.float32:
        raise TypeError("q_scale must be FP32")
    if weights.device.type != "cuda":
        raise NotImplementedError("indexer weight scaling requires a GPU")
    if q_scale.device != weights.device:
        raise ValueError("weights and q_scale must be on the same device")
    output = torch.empty((tokens, heads), dtype=torch.float32, device=weights.device)
    total = tokens * heads
    if total == 0:
        return output
    is_dense = weights.is_contiguous() and q_scale.is_contiguous()
    if (
        is_dense
        and weights.dtype == torch.float32
        and scale > 0.0
        and total > 8192
        and weights.data_ptr() % 16 == 0
        and q_scale.data_ptr() % 16 == 0
    ):
        # C550 tuning: short tails benefit from larger CTAs and an unmasked body.
        split_tail = 262144 <= total <= 425984 and 0 < total % 1024 <= 256
        block = 1024 if total < 65536 or split_tail else 512
        pipeline = "cpasync" if total >= 262144 else "basic"
        warps = 2 if total < 65536 or split_tail else 1
        kernel = (
            indexer_weight_scale_short_tail_kernel
            if split_tail
            else indexer_weight_scale_vector_kernel
        )
        kernel[(triton.cdiv(total, block),)](
            weights,
            q_scale,
            output,
            total,
            scale,
            block,
            num_warps=warps,
            enable_fp_fusion=False,
            pipeline=pipeline,
        )
        return output
    # Multiple small blocks add scheduling overhead once decode exceeds 2K items.
    block = 128 if total <= 2048 else (512 if total < 262144 else 1024)
    indexer_weight_scale_kernel[(triton.cdiv(total, block),)](
        weights,
        q_scale,
        output,
        total,
        heads,
        weights.stride(0),
        weights.stride(1),
        q_scale.stride(0),
        q_scale.stride(1),
        scale,
        is_dense,
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
