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

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.moe_sum import moe_sum as generic_moe_sum

logger = logging.getLogger(__name__)

_MTHREADS_TOPK = (8, 10)
_MTHREADS_HIDDEN_SIZES = (2048, 4096, 7168)


@triton.jit
def _mthreads_moe_sum_kernel(
    input_ptr,
    output_ptr,
    router_weights_ptr,
    router_weights_stride_token,
    router_weights_stride_topk,
    num_tokens,
    hidden_size,
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


def _mthreads_moe_sum_block_size(hidden_size: int) -> int:
    if hidden_size <= 2048:
        return 256
    if hidden_size <= 4096:
        return 512
    return 1024


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
    router_weights: torch.Tensor | None = None,
):
    """MUSA-specialized MoE reduction with a statically-unrolled TOPK loop.

    Uses a fixed block size derived from hidden_size (no @triton.autotune,
    which is unreliable on MUSA) and supports an optional router-weight
    multiply for the deferred-GEMM2 path.
    """
    num_tokens, topk, hidden_size = input.shape
    if (
        topk not in _MTHREADS_TOPK
        or hidden_size not in _MTHREADS_HIDDEN_SIZES
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

    logger.debug("GEMS_MTHREADS MOE SUM")
    block_size = _mthreads_moe_sum_block_size(hidden_size)
    grid = (num_tokens, triton.cdiv(hidden_size, block_size))
    _mthreads_moe_sum_kernel[grid](
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
    )
