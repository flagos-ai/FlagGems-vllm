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

"""Explicit data movement for the varlen host wrapper; views never copy."""

import math

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def _copy_or_fill(
    source,
    destination,
    count,
    SHAPE: tl.constexpr,
    SOURCE_STRIDES: tl.constexpr,
    DESTINATION_STRIDES: tl.constexpr,
    FILL: tl.constexpr,
    VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Widen before multiplying by strides: paged tensors may span above 4 GiB.
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    remaining = index
    source_offset = tl.full((BLOCK,), 0, tl.int64)
    destination_offset = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coordinate = remaining % SHAPE[axis]
        remaining = remaining // SHAPE[axis]
        destination_offset += coordinate * DESTINATION_STRIDES[axis]
        if not FILL:
            source_offset += coordinate * SOURCE_STRIDES[axis]
    if FILL:
        value = tl.full((BLOCK,), VALUE, tl.float32)
    else:
        value = tl.load(source + source_offset, index < count, other=0)
    tl.store(destination + destination_offset, value, index < count)


def _launch(source, destination, *, fill=False, value=0.0):
    count = destination.numel()
    if count == 0:
        return destination
    with torch_device_fn.device(destination.device):
        # A memory-only helper: 256 C550 threads, no MMA/staging or tuning search.
        _copy_or_fill[(triton.cdiv(count, 1024),)](
            source,
            destination,
            count,
            tuple(destination.shape),
            tuple(source.stride()),
            tuple(destination.stride()),
            fill,
            value,
            1024,
            num_warps=4,
            num_stages=1,
        )
    return destination


def copy_tensor(source, destination):
    assert source.shape == destination.shape
    assert source.dtype == destination.dtype and source.device == destination.device
    return _launch(source, destination)


def fill_tensor(destination, value):
    return _launch(destination, destination, fill=True, value=value)


def _contiguous_copy(source):
    destination = torch.empty(source.shape, dtype=source.dtype, device=source.device)
    return copy_tensor(source, destination)


def ensure_last_dim_contiguous(source):
    if source.stride(-1) == 1:
        return source
    return _contiguous_copy(source)


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * max(shape[axis + 1], 1)
    return tuple(strides)


def _view_strides(source, shape):
    """Partition contiguous stride chunks using metadata only."""
    assert all(size >= 0 for size in shape)
    assert math.prod(shape) == source.numel()
    sizes, strides = tuple(source.shape), source.stride()
    if source.numel() == 0:
        return strides if sizes == shape else _contiguous_strides(shape)
    if not sizes:
        return (1,) * len(shape)
    result = [0] * len(shape)
    view_axis = len(shape) - 1
    chunk_stride = strides[-1]
    chunk_elements = view_elements = 1
    for axis in range(len(sizes) - 1, -1, -1):
        chunk_elements *= sizes[axis]
        if axis == 0 or (
            sizes[axis - 1] != 1 and strides[axis - 1] != chunk_elements * chunk_stride
        ):
            while view_axis >= 0 and (
                view_elements < chunk_elements or shape[view_axis] == 1
            ):
                result[view_axis] = view_elements * chunk_stride
                view_elements *= shape[view_axis]
                view_axis -= 1
            if view_elements != chunk_elements:
                return None
            if axis > 0:
                chunk_stride = strides[axis - 1]
                chunk_elements = view_elements = 1
    return tuple(result) if view_axis == -1 else None


def view_tensor(source, shape):
    shape = tuple(shape)
    strides = _view_strides(source, shape)
    if strides is None:
        raise ValueError("output layout cannot be represented as a no-copy view")
    return source.as_strided(shape, strides)


def reshape_view_or_copy(source, shape):
    shape = tuple(shape)
    strides = _view_strides(source, shape)
    if strides is not None:
        return source.as_strided(shape, strides)
    # mcPyTorch may silently copy inside Tensor.view. Decide from strides first.
    return _contiguous_copy(source).as_strided(shape, _contiguous_strides(shape))
