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
def _mthreads_moe_sum_pair_kernel(
    input_ptr,
    output_ptr,
    router_weights_ptr,
    num_tokens,
    topk: tl.constexpr,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    output_stride_token,
    APPLY_ROUTER_WEIGHT: tl.constexpr,
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

    for i in tl.static_range(0, topk, 2):
        x0 = tl.load(
            input_base + i * input_stride_topk + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        )
        if APPLY_ROUTER_WEIGHT:
            w0 = tl.load(router_weights_ptr + token_idx * topk + i)
            x0 = x0.to(tl.float32) * w0.to(tl.float32)
        if i + 1 < topk:
            x1 = tl.load(
                input_base + (i + 1) * input_stride_topk + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            )
            if APPLY_ROUTER_WEIGHT:
                w1 = tl.load(router_weights_ptr + token_idx * topk + (i + 1))
                x1 = x1.to(tl.float32) * w1.to(tl.float32)
            acc += x0.to(tl.float32) + x1.to(tl.float32)
        else:
            acc += x0.to(tl.float32)

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
def _mthreads_moe_sum_mt_kernel(
    input_ptr,
    output_ptr,
    router_weights_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    input_stride_token,
    input_stride_topk,
    output_stride_token,
    token_bucket,
    topk: tl.constexpr,
    APPLY_ROUTER_WEIGHT: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    token_idx = tl.program_id(1)

    token_offsets = token_idx * TOKENS + tl.arange(0, TOKENS)
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

    for k in tl.static_range(topk):
        x = tl.load(input_base + k * input_stride_topk, mask=mask, other=0.0)
        if APPLY_ROUTER_WEIGHT:
            w = tl.load(
                router_weights_ptr + token_offsets[:, None] * topk + k,
                mask=token_mask[:, None],
                other=0.0,
            )
            x = x.to(tl.float32) * w.to(tl.float32)
        acc += x.to(tl.float32)

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
def _mthreads_moe_sum_general_kernel(
    input_ptr,
    output_ptr,
    router_weights_ptr,
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
    APPLY_ROUTER_WEIGHT: tl.constexpr,
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
        expert_data = tl.load(
            input_base + expert_idx * input_stride_topk,
            mask=hidden_mask,
            other=0.0,
        )
        if APPLY_ROUTER_WEIGHT:
            router_weight = tl.load(
                router_weights_ptr
                + token_idx * router_weights_stride_token
                + expert_idx * router_weights_stride_topk
            )
            expert_data = expert_data.to(tl.float32) * router_weight.to(tl.float32)
        acc += expert_data.to(tl.float32)

    tl.store(
        output_ptr
        + token_idx * output_stride_token
        + hidden_offsets * output_stride_hidden,
        acc.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


def _token_bucket(num_tokens: int) -> int:
    if num_tokens < 1024:
        return 0
    if num_tokens < 8192:
        return 1
    return 2


def _check_moe_sum_inputs(input: torch.Tensor, output: torch.Tensor):
    assert input.dim() == 3, (
        f"moe_sum: input must be 3D [num_tokens, topk, hidden], "
        f"got {input.dim()}D shape {tuple(input.shape)}"
    )
    assert output.dim() == 2, (
        f"moe_sum: output must be 2D [num_tokens, hidden], "
        f"got {output.dim()}D shape {tuple(output.shape)}"
    )
    num_tokens, topk, hidden_size = input.shape
    assert topk >= 1, f"moe_sum: topk must be >= 1, got {topk}"
    assert output.shape == (num_tokens, hidden_size), (
        f"moe_sum: output shape {tuple(output.shape)} mismatch, "
        f"expected ({num_tokens}, {hidden_size}) from input shape"
    )
    assert (
        input.dtype == output.dtype
    ), f"moe_sum: dtype mismatch, input {input.dtype} vs output {output.dtype}"
    assert input.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"moe_sum: unsupported dtype {input.dtype}, "
        f"expected float16 / bfloat16 / float32"
    )
    assert (
        input.device == output.device
    ), f"moe_sum: device mismatch, input {input.device} vs output {output.device}"


def _min_vec_elems(elem_size: int, vec_bytes: int = 16) -> int:
    return max(1, vec_bytes // elem_size)


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
    router_weights: torch.Tensor | None = None,
):
    logger.debug("GEMS_MTHREADS MOE SUM")
    _check_moe_sum_inputs(input, output)
    num_tokens, topk, hidden_size = input.shape

    if router_weights is not None:
        assert router_weights.shape == (num_tokens, topk), (
            f"moe_sum: router_weights shape {tuple(router_weights.shape)} mismatch, "
            f"expected ({num_tokens}, {topk}) from input shape"
        )
        router_weights_strides = router_weights.stride()
    else:
        router_weights_strides = (0, 0)

    input_stride = input.stride()
    output_stride = output.stride()
    elem_size = input.element_size()

    vec_ok = hidden_size % _min_vec_elems(elem_size, vec_bytes=16) == 0
    contiguous = input.is_contiguous() and output.is_contiguous()
    weights_contiguous = router_weights is None or router_weights.is_contiguous()

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
        _mthreads_moe_sum_mt_kernel[grid](
            input,
            output,
            input if router_weights is None else router_weights,
            num_tokens,
            hidden_size,
            input_stride[0],
            input_stride[1],
            output_stride[0],
            _token_bucket(num_tokens),
            topk,
            APPLY_ROUTER_WEIGHT=router_weights is not None,
            ELEM_SIZE=elem_size,
        )
    elif contiguous and weights_contiguous and topk <= 16 and num_tokens < 128:
        grid = lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
        _mthreads_moe_sum_pair_kernel[grid](
            input,
            output,
            input if router_weights is None else router_weights,
            num_tokens,
            topk,
            hidden_size,
            input_stride[0],
            input_stride[1],
            output_stride[0],
            APPLY_ROUTER_WEIGHT=router_weights is not None,
            ELEM_SIZE=elem_size,
        )
    else:
        grid = lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
        _mthreads_moe_sum_general_kernel[grid](
            input,
            output,
            input if router_weights is None else router_weights,
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
            APPLY_ROUTER_WEIGHT=router_weights is not None,
            ELEM_SIZE=elem_size,
        )
