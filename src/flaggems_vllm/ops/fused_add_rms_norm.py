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

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    x,
    residual,
    weight,
    row_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tle.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < row_size
    offsets = row * row_size + columns

    values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    residual_values = tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    added = values + residual_values
    tl.store(residual + offsets, added, mask=mask)

    variance = tl.sum(added * added, axis=0) / row_size
    reciprocal_rms = tl.rsqrt(variance + eps)
    scales = tl.load(weight + columns, mask=mask, other=0.0)
    normalized = (added * reciprocal_rms).to(x.dtype.element_ty)
    tl.store(x + offsets, normalized * scales, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_loop_kernel(
    x,
    residual,
    weight,
    row_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tle.program_id(0)
    row_offset = row * row_size
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # The operation is stateful, so regular autotuning would execute it more
    # than once and corrupt the inputs. Use a fixed conservative tile here.
    for start in range(0, row_size, BLOCK_SIZE):
        columns = start + tl.arange(0, BLOCK_SIZE)
        mask = columns < row_size
        values = tl.load(x + row_offset + columns, mask=mask, other=0.0).to(tl.float32)
        residual_values = tl.load(
            residual + row_offset + columns, mask=mask, other=0.0
        ).to(tl.float32)
        added = values + residual_values
        accumulator += added * added

    variance = tl.sum(accumulator, axis=0) / row_size
    reciprocal_rms = tl.rsqrt(variance + eps)

    for start in range(0, row_size, BLOCK_SIZE):
        columns = start + tl.arange(0, BLOCK_SIZE)
        mask = columns < row_size
        values = tl.load(x + row_offset + columns, mask=mask, other=0.0).to(tl.float32)
        residual_values = tl.load(
            residual + row_offset + columns, mask=mask, other=0.0
        ).to(tl.float32)
        added = values + residual_values
        tl.store(residual + row_offset + columns, added, mask=mask)
        scales = tl.load(weight + columns, mask=mask, other=0.0)
        normalized = (added * reciprocal_rms).to(x.dtype.element_ty)
        tl.store(x + row_offset + columns, normalized * scales, mask=mask)


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


def _storage_ranges_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two nonempty contiguous tensors share storage bytes."""
    left_start = left.data_ptr()
    right_start = right.data_ptr()
    left_end = left_start + left.numel() * left.element_size()
    right_end = right_start + right.numel() * right.element_size()
    return left_start < right_end and right_start < left_end


def _validate_inputs(x, residual, normalized_shape, weight) -> tuple[int, int]:
    shape = _canonical_normalized_shape(normalized_shape)
    if x.shape != residual.shape:
        raise ValueError("input and residual must have the same shape")
    if x.ndim < len(shape) or tuple(x.shape[-len(shape) :]) != shape:
        raise ValueError(
            f"input shape {tuple(x.shape)} must end with normalized_shape {shape}"
        )
    if tuple(weight.shape) != shape:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} must equal normalized_shape {shape}"
        )
    if x.device != residual.device or x.device != weight.device:
        raise ValueError("input, residual, and weight must be on the same device")
    if x.dtype != residual.dtype:
        raise TypeError("input and residual must have the same dtype")
    if x.dtype not in _SUPPORTED_DTYPES or weight.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "fused_add_rms_norm only supports float16, bfloat16, and float32"
        )
    if weight.dtype != x.dtype:
        raise TypeError("input, residual, and weight must have the same dtype")
    if (
        not x.is_contiguous()
        or not residual.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise NotImplementedError(
            "fused_add_rms_norm currently requires contiguous tensors"
        )
    if torch.is_grad_enabled() and (
        x.requires_grad or residual.requires_grad or weight.requires_grad
    ):
        raise NotImplementedError("fused_add_rms_norm is an inference-only operation")

    row_size = math.prod(shape)
    row_count = x.numel() // row_size
    if row_count == 0:
        raise NotImplementedError("fused_add_rms_norm does not support empty inputs")
    tensor_pairs = ((x, residual), (x, weight), (residual, weight))
    if any(_storage_ranges_overlap(left, right) for left, right in tensor_pairs):
        raise ValueError("input, residual, and weight must not overlap in memory")
    return row_count, row_size


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified. The operation is inference-only
    and rejects tensors requiring gradients while gradient recording is enabled.
    """
    logger.debug(
        "GEMS FUSED_ADD_RMS_NORM FORWARD, [input shape]: %s, [residual shape]: %s, [weight shape]: %s",
        x.size(),
        residual.size(),
        weight.size(),
    )
    row_count, row_size = _validate_inputs(x, residual, normalized_shape, weight)
    if eps is None:
        raise NotImplementedError("fused_add_rms_norm requires an explicit epsilon")

    with torch_device_fn.device(x.device):
        if row_size <= 4096:
            block_size = triton.next_power_of_2(row_size)
            fused_add_rms_norm_kernel[(row_count,)](
                x,
                residual,
                weight,
                row_size,
                eps,
                BLOCK_SIZE=block_size,
            )
        else:
            fused_add_rms_norm_loop_kernel[(row_count,)](
                x,
                residual,
                weight,
                row_size,
                eps,
                BLOCK_SIZE=1024,
            )
    return x, residual
