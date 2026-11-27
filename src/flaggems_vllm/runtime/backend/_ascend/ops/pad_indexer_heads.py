# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Bounded-grid contiguous head padding for Ascend."""

import math
import operator
from contextlib import nullcontext
from functools import lru_cache

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.pad_indexer_heads import (
    _STORAGE_TYPES,
    _launch_pad_indexer_heads,
    _prepare_pad_indexer_heads,
)
from flaggems_vllm.utils import libentry


@lru_cache(maxsize=None)
def _vector_cores(device_index):
    cores = operator.index(
        triton.runtime.driver.active.utils.get_device_properties(device_index)[
            "num_vectorcore"
        ]
    )
    if cores <= 0:
        raise NotImplementedError(
            "pad_indexer_heads requires a positive vector-core count"
        )
    return cores


@libentry()
@triton.jit(do_not_specialize=["T", "TILES"])
def _pad_indexer_heads_contiguous(
    X,
    Y,
    T,
    TILES,
    IN_WIDTH: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    TILES_PER_ROW: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    row_lane = tl.arange(0, ROWS)
    # Divide scalar tile IDs, leaving the vector load/store addresses affine.
    for tile in range(
        tl.program_id(0).to(tl.int64),
        TILES.to(tl.int64),
        tl.num_programs(0).to(tl.int64),
    ):
        row = (tile // TILES_PER_ROW).to(tl.int32) * ROWS + row_lane
        col = (tile % TILES_PER_ROW).to(tl.int32) * BLOCK + lane
        if IN_WIDTH:
            value = tl.load(
                X + row[:, None] * IN_WIDTH + col[None, :],
                (row[:, None] < T) & (col[None, :] < IN_WIDTH),
                other=0,
            )
        else:
            # Do not present an empty input's null pointer to CANN.
            value = tl.full((ROWS, BLOCK), 0, tl.int32)
        tl.store(
            Y + row[:, None] * OUT_WIDTH + col[None, :],
            value,
            (row[:, None] < T) & (col[None, :] < OUT_WIDTH),
        )


def pad_indexer_heads(x, pad):
    out, pad = _prepare_pad_indexer_heads(x, pad)
    itemsize = x.element_size()
    unit_size = min(itemsize, 8)
    units = itemsize // unit_size
    count = out.numel() * units
    if count == 0:
        return out
    device = x.device
    guard = (
        nullcontext()
        if device.index == runtime.torch_device_fn.current_device()
        else runtime.torch_device_fn.device(device)
    )
    with guard:
        cores = _vector_cores(device.index)
        width = math.prod(x.shape[2:]) * units
        output_width = out.shape[1] * width
        # Offline policy: batch short rows without starving the device grid.
        block = min(4096, triton.next_power_of_2(output_width))
        tile_units = min(16384, 65536 // unit_size)
        rows = min(
            128,
            tile_units // block,
            triton.next_power_of_2(triton.cdiv(x.shape[0], cores)),
        )
        # Include masked rows and columns in the signed-index range proof.
        max_index = count + (rows - 1) * output_width + block - 1
        if not x.is_contiguous() or max_index >= 2**31:
            return _launch_pad_indexer_heads(x, out, pad, grid_limit=cores)
        tiles_per_row = triton.cdiv(output_width, block)
        tiles = triton.cdiv(x.shape[0], rows) * tiles_per_row
        _pad_indexer_heads_contiguous[(min(cores, tiles),)](
            triton.reinterpret(x, _STORAGE_TYPES[unit_size]),
            triton.reinterpret(out, _STORAGE_TYPES[unit_size]),
            x.shape[0],
            tiles,
            x.shape[1] * width,
            output_width,
            tiles_per_row,
            ROWS=rows,
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
    return out
