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

"""Triton implementation of weighted RMS normalization."""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SINGLE_BLOCK_LIMIT = 4096
_DX_LOOP_BLOCK_SIZE = 1024
_DW_ROW_BLOCK_SIZE = 16
_DW_COL_BLOCK_SIZE = 64
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_RMS_NORM_LOOP_CONFIGS = runtime.get_tuned_config("rms_norm_loop")
if not _RMS_NORM_LOOP_CONFIGS:
    # Keep the wide-row path usable on backends that have not supplied a
    # platform-specific tuning table yet.
    _RMS_NORM_LOOP_CONFIGS = [
        triton.Config({"TILE_N": 1024}, num_warps=4),
    ]


@triton.jit
def _previous_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _rms_norm_kernel(
    output,
    inverse_rms,
    x,
    weight,
    row_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tle.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < row_size

    values = tl.load(x + row * row_size + columns, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / row_size
    reciprocal_rms = tl.rsqrt(variance + eps)

    scales = tl.load(weight + columns, mask=mask, other=0.0)
    # vLLM casts the normalized values back to the input dtype before the
    # affine multiplication. Preserve that rounding order for low precision.
    normalized = (values * reciprocal_rms).to(x.dtype.element_ty)
    tl.store(output + row * row_size + columns, normalized * scales, mask=mask)
    tl.store(inverse_rms + row, reciprocal_rms)


@libentry()
@libtuner(
    configs=_RMS_NORM_LOOP_CONFIGS,
    key=["row_size"],
    strategy=["log"],
)
@triton.jit(do_not_specialize=["eps"])
def _rms_norm_loop_kernel(
    output,
    inverse_rms,
    x,
    weight,
    row_size,
    eps,
    TILE_N: tl.constexpr,
):
    row = tle.program_id(0)
    row_offset = row * row_size
    accumulator = tl.zeros((TILE_N,), dtype=tl.float32)
    steps = tl.cdiv(row_size, TILE_N)

    for step in range(0, steps - 1):
        columns = step * TILE_N + tl.arange(0, TILE_N)
        values = tl.load(x + row_offset + columns).to(tl.float32)
        accumulator += values * values

    columns = (steps - 1) * TILE_N + tl.arange(0, TILE_N)
    mask = columns < row_size
    values = tl.load(x + row_offset + columns, mask=mask, other=0.0).to(tl.float32)
    accumulator += values * values

    variance = tl.sum(accumulator, axis=0) / row_size
    reciprocal_rms = tl.rsqrt(variance + eps)
    tl.store(inverse_rms + row, reciprocal_rms)

    last_tile = _previous_multiple_of(row_size, TILE_N)
    for tile_offset in range(0, TILE_N, TILE_N):
        columns = last_tile - tile_offset + tl.arange(0, TILE_N)
        mask = columns < row_size
        values = tl.load(
            x + row_offset + columns,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        scales = tl.load(weight + columns, mask=mask, other=0.0)
        normalized = (values * reciprocal_rms).to(x.dtype.element_ty)
        tl.store(output + row_offset + columns, normalized * scales, mask=mask)

    for tile_offset in range(TILE_N, row_size, TILE_N):
        columns = last_tile - tile_offset + tl.arange(0, TILE_N)
        values = tl.load(
            x + row_offset + columns,
            eviction_policy="evict_first",
        ).to(tl.float32)
        scales = tl.load(weight + columns)
        normalized = (values * reciprocal_rms).to(x.dtype.element_ty)
        tl.store(output + row_offset + columns, normalized * scales)


@libentry()
@triton.jit
def _rms_norm_grad_x_kernel(
    x,
    grad_output,
    inverse_rms,
    grad_x,
    weight,
    row_size,
    GRAD_OUTPUT_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tle.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < row_size
    offset = row * row_size + columns

    values = tl.load(x + offset, mask=mask, other=0.0).to(tl.float32)
    output_gradients = tl.load(
        grad_output + offset * GRAD_OUTPUT_STRIDE, mask=mask, other=0.0
    ).to(tl.float32)
    scales = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
    reciprocal_rms = tl.load(inverse_rms + row).to(tl.float32)

    affine_gradients = output_gradients * scales
    normalized = values * reciprocal_rms
    projection = tl.sum(normalized * affine_gradients, axis=0)
    gradients = (
        affine_gradients - normalized * (projection / row_size)
    ) * reciprocal_rms
    tl.store(grad_x + offset, gradients, mask=mask)


@libentry()
@triton.jit
def _rms_norm_grad_x_loop_kernel(
    x,
    grad_output,
    inverse_rms,
    grad_x,
    weight,
    row_size,
    GRAD_OUTPUT_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tle.program_id(0)
    row_offset = row * row_size
    reciprocal_rms = tl.load(inverse_rms + row).to(tl.float32)
    projection_accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in range(0, row_size, BLOCK_SIZE):
        columns = start + tl.arange(0, BLOCK_SIZE)
        mask = columns < row_size
        values = tl.load(x + row_offset + columns, mask=mask, other=0.0).to(tl.float32)
        output_gradients = tl.load(
            grad_output + (row_offset + columns) * GRAD_OUTPUT_STRIDE,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scales = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
        projection_accumulator += values * reciprocal_rms * output_gradients * scales

    projection = tl.sum(projection_accumulator, axis=0)
    for start in range(0, row_size, BLOCK_SIZE):
        columns = start + tl.arange(0, BLOCK_SIZE)
        mask = columns < row_size
        values = tl.load(x + row_offset + columns, mask=mask, other=0.0).to(tl.float32)
        output_gradients = tl.load(
            grad_output + (row_offset + columns) * GRAD_OUTPUT_STRIDE,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scales = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
        affine_gradients = output_gradients * scales
        normalized = values * reciprocal_rms
        gradients = (
            affine_gradients - normalized * (projection / row_size)
        ) * reciprocal_rms
        tl.store(grad_x + row_offset + columns, gradients, mask=mask)


@libentry()
@triton.jit
def _rms_norm_grad_weight_partial_kernel(
    x,
    grad_output,
    inverse_rms,
    partial_grad_weight,
    row_count,
    row_size,
    GRAD_OUTPUT_STRIDE: tl.constexpr,
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    row_block = tle.program_id(0)
    column_block = tle.program_id(1)
    rows = row_block * ROW_BLOCK_SIZE + tl.arange(0, ROW_BLOCK_SIZE)
    columns = column_block * COL_BLOCK_SIZE + tl.arange(0, COL_BLOCK_SIZE)
    mask = (rows[:, None] < row_count) & (columns[None, :] < row_size)
    offsets = rows[:, None] * row_size + columns[None, :]

    values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    output_gradients = tl.load(
        grad_output + offsets * GRAD_OUTPUT_STRIDE, mask=mask, other=0.0
    ).to(tl.float32)
    reciprocal_rms = tl.load(inverse_rms + rows, mask=rows < row_count, other=0.0).to(
        tl.float32
    )
    normalized = (values * reciprocal_rms[:, None]).to(x.dtype.element_ty)
    partial = tl.sum(normalized.to(tl.float32) * output_gradients, axis=0)
    tl.store(
        partial_grad_weight + row_block * row_size + columns,
        partial,
        mask=columns < row_size,
    )


@libentry()
@triton.jit
def _rms_norm_grad_weight_reduce_kernel(
    partial_grad_weight,
    grad_weight,
    partial_count,
    row_size,
    COL_BLOCK_SIZE: tl.constexpr,
):
    column_block = tle.program_id(0)
    columns = column_block * COL_BLOCK_SIZE + tl.arange(0, COL_BLOCK_SIZE)
    mask = columns < row_size
    accumulator = tl.zeros((COL_BLOCK_SIZE,), dtype=tl.float32)

    for partial_row in range(0, partial_count):
        accumulator += tl.load(
            partial_grad_weight + partial_row * row_size + columns,
            mask=mask,
            other=0.0,
        )
    tl.store(grad_weight + columns, accumulator, mask=mask)


def _canonical_normalized_shape(normalized_shape) -> tuple[int, ...]:
    if isinstance(normalized_shape, numbers.Integral):
        shape = (int(normalized_shape),)
    elif isinstance(normalized_shape, Sequence):
        shape = tuple(normalized_shape)
    else:
        raise TypeError("normalized_shape must be an int or a sequence of ints")

    if not shape or any(not isinstance(size, numbers.Integral) for size in shape):
        raise TypeError("normalized_shape must contain at least one integer")
    shape = tuple(int(size) for size in shape)
    if any(size <= 0 for size in shape):
        raise ValueError("normalized_shape dimensions must be greater than zero")
    return shape


def _validate_inputs(
    x: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor,
) -> tuple[tuple[int, ...], int, int]:
    shape = _canonical_normalized_shape(normalized_shape)
    if weight is None:
        raise NotImplementedError("rms_norm requires a weight tensor")
    if x.ndim < len(shape) or tuple(x.shape[-len(shape) :]) != shape:
        raise ValueError(
            f"input shape {tuple(x.shape)} must end with normalized_shape {shape}"
        )
    if tuple(weight.shape) != shape:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} must equal normalized_shape {shape}"
        )
    if x.device != weight.device:
        raise ValueError("input and weight must be on the same device")
    if x.dtype not in _SUPPORTED_DTYPES or weight.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("rms_norm only supports float16, bfloat16, and float32")
    if x.dtype != weight.dtype:
        raise TypeError("input and weight must have the same dtype")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise NotImplementedError("rms_norm currently requires contiguous tensors")

    row_size = math.prod(shape)
    row_count = x.numel() // row_size
    if row_count == 0:
        raise NotImplementedError("rms_norm does not support empty inputs")
    return shape, row_count, row_size


def _rms_norm_forward(
    x: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    shape, row_count, row_size = _validate_inputs(x, normalized_shape, weight)
    if eps is None:
        raise NotImplementedError("rms_norm requires an explicit epsilon")

    output = torch.empty_like(x)
    inverse_rms = torch.empty((row_count,), dtype=torch.float32, device=x.device)
    with torch_device_fn.device(x.device):
        if row_size <= _SINGLE_BLOCK_LIMIT:
            block_size = triton.next_power_of_2(row_size)
            _rms_norm_kernel[(row_count,)](
                output,
                inverse_rms,
                x,
                weight,
                row_size,
                eps,
                BLOCK_SIZE=block_size,
            )
        else:
            _rms_norm_loop_kernel[(row_count,)](
                output,
                inverse_rms,
                x,
                weight,
                row_size,
                eps,
            )
    return output, inverse_rms, shape


def _rms_norm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    inverse_rms: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, row_count, row_size = _validate_inputs(x, normalized_shape, weight)
    if grad_output.shape != x.shape or grad_output.device != x.device:
        raise ValueError("grad_output must have the same shape and device as input")
    if grad_output.dtype != x.dtype:
        raise TypeError("grad_output and input must have the same dtype")
    if grad_output.is_contiguous():
        grad_output_stride = 1
    elif all(
        size == 1 or stride == 0
        for size, stride in zip(grad_output.shape, grad_output.stride())
    ):
        # Autograd supplies an expanded scalar with zero strides for common
        # reductions such as output.sum(). Kernels can read that scalar
        # directly without a prohibited layout-conversion copy.
        grad_output_stride = 0
    else:
        raise NotImplementedError(
            "rms_norm backward requires contiguous or scalar-expanded grad_output"
        )

    grad_x = torch.empty_like(x)
    partial_count = triton.cdiv(row_count, _DW_ROW_BLOCK_SIZE)
    partial_grad_weight = torch.empty(
        (partial_count, row_size), dtype=torch.float32, device=x.device
    )
    grad_weight = torch.empty_like(weight)

    with torch_device_fn.device(x.device):
        if row_size <= _SINGLE_BLOCK_LIMIT:
            block_size = triton.next_power_of_2(row_size)
            _rms_norm_grad_x_kernel[(row_count,)](
                x,
                grad_output,
                inverse_rms,
                grad_x,
                weight,
                row_size,
                GRAD_OUTPUT_STRIDE=grad_output_stride,
                BLOCK_SIZE=block_size,
            )
        else:
            _rms_norm_grad_x_loop_kernel[(row_count,)](
                x,
                grad_output,
                inverse_rms,
                grad_x,
                weight,
                row_size,
                GRAD_OUTPUT_STRIDE=grad_output_stride,
                BLOCK_SIZE=_DX_LOOP_BLOCK_SIZE,
            )

        column_blocks = triton.cdiv(row_size, _DW_COL_BLOCK_SIZE)
        _rms_norm_grad_weight_partial_kernel[(partial_count, column_blocks)](
            x,
            grad_output,
            inverse_rms,
            partial_grad_weight,
            row_count,
            row_size,
            GRAD_OUTPUT_STRIDE=grad_output_stride,
            ROW_BLOCK_SIZE=_DW_ROW_BLOCK_SIZE,
            COL_BLOCK_SIZE=_DW_COL_BLOCK_SIZE,
        )
        _rms_norm_grad_weight_reduce_kernel[(column_blocks,)](
            partial_grad_weight,
            grad_weight,
            partial_count,
            row_size,
            COL_BLOCK_SIZE=_DW_COL_BLOCK_SIZE,
        )

    return grad_x, grad_weight


class _RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        logger.debug("GEMS RMS_NORM FORWARD")
        output, inverse_rms, shape = _rms_norm_forward(x, normalized_shape, weight, eps)
        ctx.save_for_backward(x, inverse_rms, weight)
        ctx.normalized_shape = shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS RMS_NORM BACKWARD")
        x, inverse_rms, weight = ctx.saved_tensors
        grad_x, grad_weight = _rms_norm_backward(
            grad_output, x, inverse_rms, ctx.normalized_shape, weight
        )
        return grad_x, None, grad_weight, None


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    """Apply weighted RMSNorm over the trailing ``normalized_shape`` dimensions."""

    return _RmsNorm.apply(x, normalized_shape, weight, eps)


__all__ = ["rms_norm"]
