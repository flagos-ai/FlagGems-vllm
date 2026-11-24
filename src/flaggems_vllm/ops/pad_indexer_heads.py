# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Materialize GLM5Next indexer head padding with bit-preserving copies."""

import logging
import math
import operator
from contextlib import nullcontext
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_IS_ASCEND = runtime.device.vendor_name == "ascend"
_STORAGE_TYPES = {1: tl.int8, 2: tl.int16, 4: tl.int32, 8: tl.int64}
# Keep these values in the tuning key, without generating Ascend binaries for
# incidental scalar divisibility/one-value specializations.
_DYNAMIC_INDEX_DIMS = ["T", "H", "PAD"] if _IS_ASCEND else []


@lru_cache(None)
def _vector_cores(device_index):
    return triton.runtime.driver.active.utils.get_device_properties(device_index)[
        "num_vectorcore"
    ]


@libentry()
@triton.jit(do_not_specialize=["N"])
def _pad_indexer_heads_contiguous_kernel(
    X, Y, N, ROW: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    for tile in range(pid, tl.cdiv(N.to(tl.int64), BLOCK), tl.num_programs(0)):
        source = tile * BLOCK
        if ROW:
            # Each complete input tile supplies one copy and one zero tile.
            # ROW is a multiple of BLOCK, so neither store crosses a token.
            token = tile // (ROW // BLOCK)
            dest = source + token * ROW
            value = tl.load(X + source + lane)
            tl.store(Y + dest + lane, value)
            tl.store(Y + dest + ROW + lane, 0)
        else:
            valid = source + lane < N
            value = tl.load(X + source + lane, mask=valid, other=0)
            tl.store(Y + source + lane, value, mask=valid)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("pad_indexer_heads"),
    key=["T", "H", "PAD", "TAIL", "ITEMSIZE", "CONTIGUOUS"],
)
@triton.jit(do_not_specialize=_DYNAMIC_INDEX_DIMS)
def _pad_indexer_heads_kernel(
    X,
    Y,
    T,
    H,
    PAD,
    TAIL: tl.constexpr,
    ITEMSIZE: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    SHAPE: tl.constexpr = LAYOUT[0]
    STRIDES: tl.constexpr = LAYOUT[1]
    UNITS: tl.constexpr = LAYOUT[2]
    STATIC_HEADS: tl.constexpr = LAYOUT[3]
    HAS_INPUT: tl.constexpr = LAYOUT[4]
    width = TAIL * UNITS
    pid = tl.program_id(0).to(tl.int64)
    GRID_STRIDE: tl.constexpr = LAYOUT[5]
    if GRID_STRIDE:
        heads = H.to(tl.int64) + PAD
        real_width = H.to(tl.int64) * width
        total = T.to(tl.int64) * heads * width
        # The Ascend launch grid is bounded. Each program owns tiles separated
        # by that grid size, including the final partial tile.
        programs = tl.num_programs(0)
        steps = tl.cdiv(tl.cdiv(total, BLOCK) - pid, programs)
    else:
        heads = STATIC_HEADS
        real_width = H * width
        total = T * heads * width
        programs = 1
        steps = 1
    for step in range(steps):
        i = (pid + step * programs) * BLOCK + tl.arange(0, BLOCK)
        head = i // width % heads
        token = i // (width * heads)
        tail = i % width
        if CONTIGUOUS:
            source = (token * H + head) * width + tail
        else:
            source = (token * STRIDES[0] + head * STRIDES[1]) * UNITS
            source += tail % UNITS
            remaining = tail // UNITS
            for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
                source += remaining % SHAPE[axis] * STRIDES[axis + 2] * UNITS
                remaining //= SHAPE[axis]
        valid = i < total
        # X/Y have integer pointer types: no FP8 conversion or float canonicalization.
        # Bound the flattened source-head span, rather than the head coordinate.
        # FlagTree AABS otherwise mistakes H for the flat tile's extent and shrinks
        # a 1024-element copy tile to 16 elements for the real GLM5Next workload.
        if HAS_INPUT:
            real_head = i % (heads * width) < real_width
            value = tl.load(X + source, valid & real_head, other=0)
        else:
            # CANN can validate even a fully masked null input address. Eliminate
            # the load at compile time for zero-head inputs with nonempty padding.
            value = tl.full((BLOCK,), 0, tl.int32)
        tl.store(Y + i, value, valid)


def pad_indexer_heads(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Append ``pad`` zero heads to ``[T,H,*tail]`` in fresh contiguous storage.

    All strided layouts with nonnegative strides, including broadcast views,
    are read directly. The inference API preserves storage bits, not autograd.
    """
    logger.debug("GEMS PAD_INDEXER_HEADS")
    pad = operator.index(pad)
    sizes = tuple(x.shape)
    if len(sizes) == 0:
        raise IndexError("pad_indexer_heads requires a token axis")
    if len(sizes) == 1:
        raise RuntimeError("pad_indexer_heads requires a head axis")
    if pad < 0:
        raise RuntimeError("pad_indexer_heads received a negative padding dimension")
    if x.layout != torch.strided or x.is_quantized:
        raise NotImplementedError(
            "pad_indexer_heads requires unquantized strided storage"
        )
    strides = tuple(x.stride())
    if any(s < 0 for s in strides) or x.is_conj() or x.is_neg():
        raise NotImplementedError(
            "pad_indexer_heads does not support lazy or negative-stride views"
        )
    device = x.device
    if device.type != runtime.device.name:
        raise NotImplementedError(
            "pad_indexer_heads requires the active accelerator backend"
        )
    itemsize = x.element_size()
    if itemsize not in (1, 2, 4, 8, 16):
        raise NotImplementedError("pad_indexer_heads requires byte-addressable storage")
    shape = (sizes[0], sizes[1] + pad, *sizes[2:])
    out = torch.empty(shape, dtype=x.dtype, device=device)
    count = out.numel()
    if count == 0:
        return out
    unit_size = min(itemsize, 8)
    storage_type = _STORAGE_TYPES[unit_size]
    units = itemsize // unit_size
    contiguous = x.is_contiguous()
    guard = (
        nullcontext()
        if device.index == runtime.torch_device_fn.current_device()
        else runtime.torch_device_fn.device(device)
    )
    with guard:
        grid_limit = _vector_cores(device.index) if _IS_ASCEND else count * units
        src = triton.reinterpret(x, storage_type)
        dst = triton.reinterpret(out, storage_type)
        tail = math.prod(sizes[2:])
        if _IS_ASCEND and contiguous:
            # Use up to 8 KiB per transfer. With equal real and padded spans,
            # select a power-of-two tile that divides a whole input row.
            block = 8192 // unit_size
            row = sizes[1] * tail * units if pad else 0
            if row:
                block = math.gcd(row, block)
            if pad == 0 or (pad == sizes[1] and block * unit_size >= 128):
                n = x.numel() * units
                grid = (min(grid_limit, triton.cdiv(n, block)),)
                _pad_indexer_heads_contiguous_kernel[grid](src, dst, n, row, block)
                return out
        _pad_indexer_heads_kernel[
            lambda meta: (min(grid_limit, triton.cdiv(count * units, meta["BLOCK"])),)
        ](
            src,
            dst,
            sizes[0],
            sizes[1],
            pad,
            tail,
            itemsize,
            contiguous,
            (
                () if contiguous else sizes[2:],
                () if contiguous else strides,
                units,
                0 if _IS_ASCEND else shape[1],
                sizes[1] != 0,
                _IS_ASCEND,
            ),
        )
    return out
