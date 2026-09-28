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

import math

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


@triton.jit
def compact_ragged_tile_coords(
    tile_idx,
    cu_seqlens_q_ptr,
    batch_size,
    BLOCK_M: tl.constexpr,
):
    """Map one compact Q-tile id to request-local ``(m_block, bid)``."""

    # tl.cumsum benefits from a power-of-two vector.  Use 31 requests per
    # group so bids + 1 remains a naturally masked cu_seqlens_q load.
    lane = tl.arange(0, 32)
    group_bid = 0
    group_start = 0
    found = False
    selected_bid = 0
    selected_m_block = 0

    while (group_bid < batch_size) & (~found):
        bids = group_bid + lane
        valid = (lane < 31) & (bids < batch_size)
        q_bos = tl.load(cu_seqlens_q_ptr + bids, mask=valid, other=0).to(tl.int32)
        q_eos = tl.load(cu_seqlens_q_ptr + bids + 1, mask=valid, other=0).to(tl.int32)
        m_blocks = tl.cdiv(q_eos - q_bos, BLOCK_M)
        group_prefix = tl.cumsum(tl.where(valid, m_blocks, 0), axis=0)
        work_before = group_prefix - m_blocks
        group_work = tl.sum(tl.where(valid, m_blocks, 0), axis=0)
        local_idx = tile_idx - group_start
        in_group = (local_idx >= 0) & (local_idx < group_work)
        hit = valid & (local_idx >= work_before) & (local_idx < group_prefix)

        selected_bid = tl.where(
            in_group, tl.sum(tl.where(hit, bids, 0), axis=0), selected_bid
        )
        selected_m_block = tl.where(
            in_group,
            tl.sum(tl.where(hit, local_idx - work_before, 0), axis=0),
            selected_m_block,
        )
        found = found | in_group
        group_start += group_work
        group_bid += 31

    return selected_m_block, selected_bid, found


@triton.jit
def paged_tile_coords(
    start_n,
    k_len,
    page_table_ptr,
    BLOCK_N: tl.constexpr,
    BOUNDARY_CHECK: tl.constexpr,
    PAGE_SIZE: tl.constexpr = 32,
):
    """Resolve physical rows from the page table."""
    if PAGE_SIZE == 16:
        tl.static_assert(BLOCK_N == 128, "page16 scalar gather requires BN128")
        page_slot = tl.arange(0, BLOCK_N) // 16
        virtual_page = start_n // 16
        if BOUNDARY_CHECK:
            page0 = tl.load(
                page_table_ptr + virtual_page + 0, mask=start_n + 0 < k_len, other=0
            )
        else:
            page0 = tl.load(page_table_ptr + virtual_page + 0)
        if BOUNDARY_CHECK:
            page1 = tl.load(
                page_table_ptr + virtual_page + 1, mask=start_n + 16 < k_len, other=0
            )
        else:
            page1 = tl.load(page_table_ptr + virtual_page + 1)
        if BOUNDARY_CHECK:
            page2 = tl.load(
                page_table_ptr + virtual_page + 2, mask=start_n + 32 < k_len, other=0
            )
        else:
            page2 = tl.load(page_table_ptr + virtual_page + 2)
        if BOUNDARY_CHECK:
            page3 = tl.load(
                page_table_ptr + virtual_page + 3, mask=start_n + 48 < k_len, other=0
            )
        else:
            page3 = tl.load(page_table_ptr + virtual_page + 3)
        if BOUNDARY_CHECK:
            page4 = tl.load(
                page_table_ptr + virtual_page + 4, mask=start_n + 64 < k_len, other=0
            )
        else:
            page4 = tl.load(page_table_ptr + virtual_page + 4)
        if BOUNDARY_CHECK:
            page5 = tl.load(
                page_table_ptr + virtual_page + 5, mask=start_n + 80 < k_len, other=0
            )
        else:
            page5 = tl.load(page_table_ptr + virtual_page + 5)
        if BOUNDARY_CHECK:
            page6 = tl.load(
                page_table_ptr + virtual_page + 6, mask=start_n + 96 < k_len, other=0
            )
        else:
            page6 = tl.load(page_table_ptr + virtual_page + 6)
        if BOUNDARY_CHECK:
            page7 = tl.load(
                page_table_ptr + virtual_page + 7, mask=start_n + 112 < k_len, other=0
            )
        else:
            page7 = tl.load(page_table_ptr + virtual_page + 7)
        page_id = tl.where(
            page_slot < 4,
            tl.where(
                page_slot < 2,
                tl.where(page_slot == 0, page0, page1),
                tl.where(page_slot == 2, page2, page3),
            ),
            tl.where(
                page_slot < 6,
                tl.where(page_slot == 4, page4, page5),
                tl.where(page_slot == 6, page6, page7),
            ),
        ).to(tl.int64)
        col_idx = start_n + tl.arange(0, BLOCK_N)
        col_idx = tl.max_contiguous(tl.multiple_of(col_idx, BLOCK_N), BLOCK_N)
        return (col_idx, page_id, col_idx % 16)
    else:
        page_slot = tl.arange(0, BLOCK_N) // 32
        virtual_page = start_n // 32
        if BOUNDARY_CHECK:
            page0 = tl.load(
                page_table_ptr + virtual_page, mask=start_n < k_len, other=0
            ).to(tl.int64)
        else:
            page0 = tl.load(page_table_ptr + virtual_page).to(tl.int64)
        if BLOCK_N == 32:
            page_id = page0
        elif BLOCK_N == 64:
            if BOUNDARY_CHECK:
                page1 = tl.load(
                    page_table_ptr + virtual_page + 1,
                    mask=start_n + 32 < k_len,
                    other=0,
                ).to(tl.int64)
            else:
                page1 = tl.load(page_table_ptr + virtual_page + 1).to(tl.int64)
            page_id = tl.where(page_slot == 0, page0, page1)
        elif BLOCK_N == 128:
            if BOUNDARY_CHECK:
                page1 = tl.load(
                    page_table_ptr + virtual_page + 1,
                    mask=start_n + 32 < k_len,
                    other=0,
                ).to(tl.int64)
                page2 = tl.load(
                    page_table_ptr + virtual_page + 2,
                    mask=start_n + 64 < k_len,
                    other=0,
                ).to(tl.int64)
                page3 = tl.load(
                    page_table_ptr + virtual_page + 3,
                    mask=start_n + 96 < k_len,
                    other=0,
                ).to(tl.int64)
            else:
                page1 = tl.load(page_table_ptr + virtual_page + 1).to(tl.int64)
                page2 = tl.load(page_table_ptr + virtual_page + 2).to(tl.int64)
                page3 = tl.load(page_table_ptr + virtual_page + 3).to(tl.int64)
            page_id = tl.where(
                page_slot == 0,
                page0,
                tl.where(page_slot == 1, page1, tl.where(page_slot == 2, page2, page3)),
            )
        else:
            tl.static_assert(False, "Async-TN BLOCK_N must be 32, 64, or 128")
        col_idx = start_n + tl.arange(0, BLOCK_N)
        col_idx = tl.max_contiguous(tl.multiple_of(col_idx, BLOCK_N), BLOCK_N)
        page_offset = col_idx % 32
        return (col_idx, page_id, page_offset)


@triton.jit
def online_softmax_stats(
    scores, row_max, row_sum, scale_softmax_log2, IS_BORDER: tl.constexpr
):
    previous_max = row_max
    row_max = tl.maximum(row_max, tl.max(scores, 1))
    if IS_BORDER:
        current_max = tl.where(row_max == float("-inf"), 0.0, row_max)
    else:
        current_max = row_max
    accumulator_scale = tl.math.exp2((previous_max - current_max) * scale_softmax_log2)
    row_sum *= accumulator_scale
    max_scaled = tl.where(row_max == float("-inf"), 0.0, row_max * scale_softmax_log2)
    probabilities = tl.math.exp2(scores * scale_softmax_log2 - max_scaled[:, None])
    row_sum += tl.sum(probabilities, 1)
    return (accumulator_scale, probabilities, row_max, row_sum)


@triton.jit
def scaled_online_softmax_stats(
    scores, row_max, row_sum, scale_softmax_log2, IS_BORDER: tl.constexpr
):
    scores *= scale_softmax_log2
    previous_max = row_max
    row_max = tl.maximum(row_max, tl.max(scores, 1))
    if IS_BORDER:
        current_max = tl.where(row_max == float("-inf"), 0.0, row_max)
    else:
        current_max = row_max
    accumulator_scale = tl.math.exp2(previous_max - current_max)
    row_sum *= accumulator_scale
    max_scaled = tl.where(row_max == float("-inf"), 0.0, row_max)
    probabilities = tl.math.exp2(scores - max_scaled[:, None])
    row_sum += tl.sum(probabilities, 1)
    return (accumulator_scale, probabilities, row_max, row_sum)


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


def tn_compile_scenario(*tensors):
    """Enable the 4 GiB address optimization only for bounded byte spans."""
    for tensor in tensors:
        if tensor is None or tensor.numel() == 0:
            continue
        # numel() misses holes in noncontiguous caches. The origin is the
        # pointer passed to the kernel, so storage_offset is not added again.
        last_element = sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        if (last_element + 1) * tensor.element_size() > (1 << 32):
            # cpasync otherwise enables metaxgpu-aggressive-4g-addr-opt,
            # which truncates large K/V byte offsets even with int64 source.
            return "storeCoalesce;noaddropt"
    return "storeCoalesce"
