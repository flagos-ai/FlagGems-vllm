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

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("moe_sum_pair"),
    key=["hidden_size", "topk", "ELEM_SIZE"],
)
@triton.jit
def _thead_moe_sum_pair_kernel(
    input_ptr,
    output_ptr,
    router_weights_ptr,
    skip_ptr,
    bias_ptr,
    src2dst_ptr,
    expert_ptr,
    num_tokens,
    topk: tl.constexpr,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    output_stride_token,
    skip_stride_token,
    bias_stride_expert,
    APPLY_ROUTER_WEIGHT: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IDENTITY_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    hidden_start = block_idx * BLOCK_SIZE
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE)

    hidden_offsets = tl.max_contiguous(
        tl.multiple_of(hidden_offsets, BLOCK_SIZE), BLOCK_SIZE
    )

    hidden_mask = hidden_offsets < hidden_size
    if token_idx >= num_tokens:
        return
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_base = input_ptr + token_idx * input_stride_token

    for expert_idx in tl.static_range(0, topk, 2):
        if IDENTITY_MAP:
            input_ptrs0 = input_base + expert_idx * input_stride_topk + hidden_offsets
        else:
            source_row0 = tl.load(src2dst_ptr + expert_idx * num_tokens + token_idx)
            input_ptrs0 = input_ptr + source_row0 * input_stride_topk + hidden_offsets
        expert_data0 = tl.load(input_ptrs0, mask=hidden_mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            expert_id0 = tl.load(expert_ptr + token_idx * topk + expert_idx)
            expert_data0 += tl.load(
                bias_ptr + expert_id0 * bias_stride_expert + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
        if APPLY_ROUTER_WEIGHT:
            router_weight0 = tl.load(router_weights_ptr + token_idx * topk + expert_idx)
            expert_data0 = expert_data0 * router_weight0.to(tl.float32)

        if expert_idx + 1 < topk:
            if IDENTITY_MAP:
                input_ptrs1 = (
                    input_base + (expert_idx + 1) * input_stride_topk + hidden_offsets
                )
            else:
                source_row1 = tl.load(
                    src2dst_ptr + (expert_idx + 1) * num_tokens + token_idx
                )
                input_ptrs1 = (
                    input_ptr + source_row1 * input_stride_topk + hidden_offsets
                )
            expert_data1 = tl.load(input_ptrs1, mask=hidden_mask, other=0.0).to(
                tl.float32
            )
            if HAS_BIAS:
                expert_id1 = tl.load(expert_ptr + token_idx * topk + (expert_idx + 1))
                expert_data1 += tl.load(
                    bias_ptr + expert_id1 * bias_stride_expert + hidden_offsets,
                    mask=hidden_mask,
                    other=0.0,
                ).to(tl.float32)
            if APPLY_ROUTER_WEIGHT:
                router_weight1 = tl.load(
                    router_weights_ptr + token_idx * topk + (expert_idx + 1)
                )
                expert_data1 = expert_data1 * router_weight1.to(tl.float32)
            acc += expert_data0 + expert_data1
        else:
            acc += expert_data0

    if HAS_SKIP:
        acc += tl.load(
            skip_ptr + token_idx * skip_stride_token + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)

    output_ptr_pos = output_ptr + token_idx * output_stride_token + hidden_offsets
    tl.store(
        output_ptr_pos,
        acc.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("moe_sum_mt"),
    key=["hidden_size", "topk", "token_bucket", "ELEM_SIZE"],
)
@triton.jit
def _thead_moe_sum_mt_kernel(
    input_ptr,
    output_ptr,
    router_weights_ptr,
    skip_ptr,
    bias_ptr,
    src2dst_ptr,
    expert_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    input_stride_token,
    input_stride_topk,
    output_stride_token,
    skip_stride_token,
    bias_stride_expert,
    token_bucket,
    topk: tl.constexpr,
    APPLY_ROUTER_WEIGHT: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IDENTITY_MAP: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    token_pid = tl.program_id(1)

    token_offsets = token_pid * TOKENS + tl.arange(0, TOKENS)
    hidden_offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_offsets = tl.max_contiguous(tl.multiple_of(hidden_offsets, BLOCK_H), BLOCK_H)

    token_mask = token_offsets < num_tokens
    EVEN_H: tl.constexpr = hidden_size % BLOCK_H == 0
    if EVEN_H:
        mask = tl.broadcast_to(token_mask[:, None], (TOKENS, BLOCK_H))
    else:
        mask = token_mask[:, None] & (hidden_offsets[None, :] < hidden_size)

    acc = tl.zeros((TOKENS, BLOCK_H), dtype=tl.float32)
    input_base = (
        input_ptr
        + token_offsets[:, None] * input_stride_token
        + hidden_offsets[None, :]
    )

    for expert_idx in tl.static_range(topk):
        if IDENTITY_MAP:
            input_ptrs = input_base + expert_idx * input_stride_topk
        else:
            source_row = tl.load(
                src2dst_ptr + expert_idx * num_tokens + token_offsets,
                mask=token_mask,
                other=0,
            )
            input_ptrs = (
                input_ptr
                + source_row[:, None] * input_stride_topk
                + hidden_offsets[None, :]
            )
        expert_data = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)

        if HAS_BIAS:
            expert_id = tl.load(
                expert_ptr + token_offsets * topk + expert_idx,
                mask=token_mask,
                other=0,
            )
            expert_data += tl.load(
                bias_ptr
                + expert_id[:, None] * bias_stride_expert
                + hidden_offsets[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)

        if APPLY_ROUTER_WEIGHT:
            router_weight = tl.load(
                router_weights_ptr + token_offsets[:, None] * topk + expert_idx,
                mask=token_mask[:, None],
                other=0.0,
            )
            expert_data = expert_data * router_weight.to(tl.float32)
        acc += expert_data

    if HAS_SKIP:
        acc += tl.load(
            skip_ptr
            + token_offsets[:, None] * skip_stride_token
            + hidden_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

    output_ptr_pos = (
        output_ptr
        + token_offsets[:, None] * output_stride_token
        + hidden_offsets[None, :]
    )
    tl.store(output_ptr_pos, acc.to(output_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("moe_sum_general"),
    key=["hidden_size", "topk", "ELEM_SIZE"],
)
@triton.jit
def _thead_moe_sum_general_kernel(
    input_ptr,
    output_ptr,
    router_weights_ptr,
    skip_ptr,
    bias_ptr,
    src2dst_ptr,
    expert_ptr,
    router_weights_stride_token,
    router_weights_stride_topk,
    num_tokens,
    topk,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    skip_stride_token,
    bias_stride_expert,
    APPLY_ROUTER_WEIGHT: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IDENTITY_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    hidden_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    if token_idx >= num_tokens:
        return
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_base = (
        input_ptr
        + token_idx * input_stride_token
        + hidden_offsets * input_stride_hidden
    )

    for expert_idx in range(topk):
        if IDENTITY_MAP:
            input_ptrs = input_base + expert_idx * input_stride_topk
        else:
            source_row = tl.load(src2dst_ptr + expert_idx * num_tokens + token_idx)
            input_ptrs = (
                input_ptr
                + source_row * input_stride_topk
                + hidden_offsets * input_stride_hidden
            )
        expert_data = tl.load(input_ptrs, mask=hidden_mask, other=0.0).to(tl.float32)

        if HAS_BIAS:
            expert_id = tl.load(expert_ptr + token_idx * topk + expert_idx)
            expert_data += tl.load(
                bias_ptr + expert_id * bias_stride_expert + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)

        if APPLY_ROUTER_WEIGHT:
            router_weight = tl.load(
                router_weights_ptr
                + token_idx * router_weights_stride_token
                + expert_idx * router_weights_stride_topk
            )
            expert_data = expert_data * router_weight.to(tl.float32)
        acc += expert_data

    if HAS_SKIP:
        acc += tl.load(
            skip_ptr + token_idx * skip_stride_token + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)

    output_ptr_pos = (
        output_ptr
        + token_idx * output_stride_token
        + hidden_offsets * output_stride_hidden
    )
    tl.store(
        output_ptr_pos,
        acc.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


def _token_bucket(num_tokens: int) -> int:
    if num_tokens < 1024:
        return 0
    if num_tokens < 8192:
        return 1
    return 2


def _min_vec_elems(elem_size: int, vec_bytes: int = 16) -> int:
    return max(1, vec_bytes // elem_size)


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
    router_weights: torch.Tensor | None = None,
    skip: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    expanded_source_row_to_expanded_dest_row: torch.Tensor | None = None,
    expert_for_source_row: torch.Tensor | None = None,
):
    """moe_sum with optional T-Head-native-aligned parameters.

        output[i] = skip[i]
                  + sum_j router_weights[i, j]
                           * (input_flat[src2dst[j*num_tokens + i]]
                              + bias[expert[i*topk + j]])

    where input_flat is input viewed as [num_tokens*topk, hidden_size].

    Args:
        router_weights: [num_tokens, topk] routing weights, read
            row-major at [i*topk + j]; None means a plain sum.
        skip: [num_tokens, hidden_size] residual added to the output;
            the hidden dim must be contiguous. None means no residual.
        bias: [num_experts, hidden_size] per-expert bias table, added
            to each route before weighting; the hidden dim must be
            contiguous. Requires expert_for_source_row when given.
            None means no bias.
        expanded_source_row_to_expanded_dest_row: contiguous int32
            table in column-major order: flat[j*num_tokens + i] holds
            the source row of token i's j-th route in the flattened
            input. None means the identity map (generic mode, the table
            is never read).
        expert_for_source_row: contiguous int32 [num_tokens, topk]
            expert ids, read row-major at [i*topk + j].
    """
    logger.debug("GEMS_THEAD MOE SUM")
    src2dst = expanded_source_row_to_expanded_dest_row
    expert = expert_for_source_row

    num_tokens, topk, hidden_size = input.shape

    if router_weights is not None:
        router_weights_strides = router_weights.stride()
    else:
        router_weights_strides = (0, 0)

    input_stride = input.stride()
    output_stride = output.stride()
    elem_size = input.element_size()

    vec_ok = hidden_size % _min_vec_elems(elem_size, vec_bytes=16) == 0
    contiguous = input.is_contiguous() and output.is_contiguous()
    weights_contiguous = router_weights is None or router_weights.is_contiguous()

    # Shared launch arguments for the three kernels. Optional parameters
    # that are None are replaced by `input` as a dummy pointer.
    ptr_args = (
        input,
        output,
        router_weights if router_weights is not None else input,
        skip if skip is not None else input,
        bias if bias is not None else input,
        src2dst if src2dst is not None else input,
        expert if expert is not None else input,
    )
    skip_stride = skip.stride(0) if skip is not None else 0
    bias_stride = bias.stride(0) if bias is not None else 0
    flags = dict(
        APPLY_ROUTER_WEIGHT=router_weights is not None,
        HAS_SKIP=skip is not None,
        HAS_BIAS=bias is not None,
        IDENTITY_MAP=src2dst is None,
        ELEM_SIZE=elem_size,
    )

    if (
        contiguous
        and weights_contiguous
        and topk <= 16
        and num_tokens >= 128
        and vec_ok
    ):
        grid = lambda meta: (
            triton.cdiv(hidden_size, meta["BLOCK_H"]),
            triton.cdiv(num_tokens, meta["TOKENS"]),
        )
        _thead_moe_sum_mt_kernel[grid](
            *ptr_args,
            num_tokens,
            hidden_size,
            input_stride[0],
            input_stride[1],
            output_stride[0],
            skip_stride,
            bias_stride,
            _token_bucket(num_tokens),
            topk,
            **flags,
        )
    elif contiguous and weights_contiguous and topk <= 16 and num_tokens < 128:
        grid = lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
        _thead_moe_sum_pair_kernel[grid](
            *ptr_args,
            num_tokens,
            topk,
            hidden_size,
            input_stride[0],
            input_stride[1],
            output_stride[0],
            skip_stride,
            bias_stride,
            **flags,
        )
    else:
        grid = lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
        _thead_moe_sum_general_kernel[grid](
            *ptr_args,
            router_weights_strides[0],
            router_weights_strides[1],
            num_tokens,
            topk,
            hidden_size,
            input_stride[0],
            input_stride[1],
            input_stride[2],
            output_stride[0],
            output_stride[1],
            skip_stride,
            bias_stride,
            **flags,
        )
