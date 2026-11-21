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
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import tl_extra_shim

sigmoid = tl.sigmoid
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow

logger = logging.getLogger(__name__)


@triton.jit
def swiglu_kernel(
    input_ptr,
    output_ptr,
    M,
    H,
    stride_in_m,
    stride_in_h,
    stride_out_m,
    stride_out_h,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)

    input_a_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + offs_h[None, :] * stride_in_h
    )
    input_b_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + (offs_h[None, :] + H) * stride_in_h
    )
    output_ptr = (
        output_ptr + offs_m[:, None] * stride_out_m + offs_h[None, :] * stride_out_h
    )

    x_a = tl.load(input_a_ptr, mask=mask, other=0.0).to(tl.float32)
    x_b = tl.load(input_b_ptr, mask=mask, other=0.0).to(tl.float32)

    silu_x_a = x_a * sigmoid(x_a)
    out = silu_x_a * x_b

    tl.store(output_ptr, out.to(x_a.dtype), mask=mask)


@triton.jit
def dswiglu_kernel(
    grad_out_ptr,
    input_ptr,
    grad_in_ptr,
    M,
    H,
    stride_grad_out_m,
    stride_grad_out_h,
    stride_in_m,
    stride_in_h,
    stride_grad_in_m,
    stride_grad_in_h,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)

    grad_out_ptr = (
        grad_out_ptr
        + offs_m[:, None] * stride_grad_out_m
        + offs_h[None, :] * stride_grad_out_h
    )
    input_a_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + offs_h[None, :] * stride_in_h
    )
    input_b_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + (offs_h[None, :] + H) * stride_in_h
    )
    grad_a_ptr = (
        grad_in_ptr
        + offs_m[:, None] * stride_grad_in_m
        + offs_h[None, :] * stride_grad_in_h
    )
    grad_b_ptr = (
        grad_in_ptr
        + offs_m[:, None] * stride_grad_in_m
        + (offs_h[None, :] + H) * stride_grad_in_h
    )

    grad_out = tl.load(grad_out_ptr, mask=mask, other=0.0).to(tl.float32)
    x_a = tl.load(input_a_ptr, mask=mask, other=0.0).to(tl.float32)
    x_b = tl.load(input_b_ptr, mask=mask, other=0.0).to(tl.float32)

    sig = sigmoid(x_a)
    silu = x_a * sig
    d_silu = sig + x_a * sig * (1 - sig)

    grad_a = grad_out * x_b * d_silu
    grad_b = grad_out * silu

    tl.store(grad_a_ptr, grad_a.to(x_a.dtype), mask=mask)
    tl.store(grad_b_ptr, grad_b.to(x_a.dtype), mask=mask)


# Fixed-grid SP count of the target GCU (L600/GCU400 class): CTA count above
# this only adds wave-quantization cost for this streaming kernel.
_NUM_SPC = 48


def _next_pow2(n: int) -> int:
    if n <= 2:
        return 2
    return 1 << (n - 1).bit_length()


def _swiglu_tiling(M: int, H: int, itemsize: int):
    # Per-CTA element budget: 64Ki elements for <=2-byte dtypes (~384KB DSM with
    # three live bf16 streams, keeps the 448KB 48-way line), halved for 4-byte
    # dtypes. Narrow rows (H <= 64) cannot widen the contiguous load segment, so
    # they are dispatch-bound: cap BLOCK_M to keep ~one wave of CTAs instead.
    target_elems = 65536 if itemsize <= 2 else 32768
    block_h = max(2, min(_next_pow2(H), 1024))
    block_m = max(
        32,
        min(target_elems // block_h, _next_pow2((M + _NUM_SPC - 1) // _NUM_SPC)),
    )
    num_tiles = triton.cdiv(M, block_m) * triton.cdiv(H, block_h)
    if itemsize > 2:
        num_warps = 2
    elif block_h <= 64:
        num_warps = 1
    else:
        num_warps = 4 if num_tiles <= _NUM_SPC else 2
    return block_m, block_h, num_warps


def swiglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    logger.debug("GEMS SWIGLU")
    if input_tensor.shape[-1] % 2 != 0:
        raise ValueError(
            f"The last dimension of must be even number, got {input_tensor.shape[-1]}"
        )
    if not input_tensor.is_cuda:
        raise ValueError("Only CUDA tensor is supported by SwiGLU")

    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)
    input_2d = input_tensor.contiguous().view(M, 2 * H)
    output_2d = torch.empty(M, H, device=input_tensor.device, dtype=input_tensor.dtype)
    if M == 0:
        return output_2d.view(*shape[:-1], H)

    block_m, block_h, num_warps = _swiglu_tiling(M, H, input_tensor.element_size())
    grid = (triton.cdiv(M, block_m), triton.cdiv(H, block_h))

    swiglu_kernel[grid](
        input_2d,
        output_2d,
        M,
        H,
        input_2d.stride(0),
        input_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_H=block_h,
        num_warps=num_warps,
    )

    return output_2d.view(*shape[:-1], H)


def dswiglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    logger.debug("GEMS DSWIGLU")
    shape = input_tensor.shape
    assert (
        shape[-1] % 2 == 0
    ), f"The last dimension of input_tensor must be an even number, got {shape[-1]}"
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)
    grad_out_2d = grad_output.contiguous().view(M, H)
    input_2d = input_tensor.contiguous().view(M, 2 * H)
    grad_in_2d = torch.empty_like(input_2d)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(H, META["BLOCK_SIZE_H"]),
    )

    dswiglu_kernel[grid](
        grad_out_2d,
        input_2d,
        grad_in_2d,
        M,
        H,
        grad_out_2d.stride(0),
        grad_out_2d.stride(1),
        input_2d.stride(0),
        input_2d.stride(1),
        grad_in_2d.stride(0),
        grad_in_2d.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_H=64,
    )

    return grad_in_2d.view_as(input_tensor)


__all__ = ["swiglu", "dswiglu"]
