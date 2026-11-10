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

"""Ascend-optimized mHC stream expansion."""

from typing import Tuple

import torch
import triton
import triton.language as tl

__all__ = ["hc_expand"]

_MAX_WORKERS = 40
_DEFAULT_BLOCK_H = 512
_WIDE_ROW_BLOCK_H = 1024
_SMALL_STRIDED_ELEMS = 1024
_SMALL_ROW_TILE_ELEMS_CAP = 1024
# Measured UB-safe input tile footprint for this kernel on Ascend910B4.
_TILE_BYTES_CAP = 64 * 1024

_VECTOR_CORE_NUM = None


def _vector_core_num() -> int:
    global _VECTOR_CORE_NUM
    if _VECTOR_CORE_NUM is None:
        try:
            device = torch.npu.current_device()
            _VECTOR_CORE_NUM = torch.npu.get_device_limit(device)["vector_core_num"]
        except (AttributeError, KeyError, TypeError, RuntimeError):
            try:
                utils = triton.runtime.driver.active.utils
                _VECTOR_CORE_NUM = utils.get_aivector_core_num()
            except (AttributeError, RuntimeError):
                _VECTOR_CORE_NUM = _MAX_WORKERS
    return max(1, min(_MAX_WORKERS, _VECTOR_CORE_NUM))


def _select_persistent_tile(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    element_size: int,
) -> Tuple[int, int]:
    if hidden_size <= 32:
        # Expanded stores share UB; FP32 with 8 streams needs BLOCK_H <= 16.
        stream_bytes = element_size * max(hc_mult, 1)
        block_h = 32 if stream_bytes <= 16 else 16 if stream_bytes <= 32 else 8
    elif hidden_size > 4096 and num_tokens <= 64:
        block_h = _WIDE_ROW_BLOCK_H
    else:
        block_h = _DEFAULT_BLOCK_H
    block_h = min(triton.next_power_of_2(hidden_size), block_h)

    if num_tokens <= 16:
        block_r = min(16, triton.next_power_of_2(num_tokens))
    elif num_tokens <= 1024:
        block_r = 32
    else:
        block_r = 64
    block_r = min(
        block_r,
        max(1, _TILE_BYTES_CAP // (block_h * element_size)),
    )

    # Keep at least half a worker wave without making tiles too fragmented.
    min_tasks = max(1, _vector_core_num() // 2)
    hidden_tiles = triton.cdiv(hidden_size, block_h)
    while (
        block_r > 1
        and triton.cdiv(num_tokens, block_r) * hidden_tiles < min_tasks
    ):
        block_r //= 2
    return block_r, block_h


@triton.jit
def _hc_expand_persistent_kernel(
    x,
    out,
    M,
    H: tl.constexpr,
    N: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    workers = tl.num_programs(0)
    hidden_tiles: tl.constexpr = tl.cdiv(H, BLOCK_H)
    total_tiles = tl.cdiv(M, BLOCK_R) * hidden_tiles

    for tile_id in tl.range(
        pid,
        total_tiles,
        workers,
        disable_licm=True,
        loop_unroll_factor=1,
    ):
        row_block = tile_id // hidden_tiles
        hidden_block = tile_id - row_block * hidden_tiles
        rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
        offsets = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
        mask = (rows < M) & (offsets < H)
        value = tl.load(
            x + rows * STRIDE_M + offsets * STRIDE_H,
            mask=mask,
            other=0,
        )
        for stream in tl.static_range(N):
            tl.store(
                out + (rows * N + stream) * H + offsets,
                value,
                mask=mask,
            )


@triton.jit
def _hc_expand_small_hidden_kernel(
    x,
    out,
    H: tl.constexpr,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_H)
    hidden_mask = offsets < H
    value = tl.load(
        x + row * STRIDE_M + offsets * STRIDE_H,
        mask=hidden_mask,
        other=0,
    )

    streams = tl.arange(0, N_PAD)[:, None]
    output_offsets = (row * N + streams) * H + offsets[None, :]
    output_mask = (streams < N) & hidden_mask[None, :]
    output_value = tl.broadcast_to(value[None, :], (N_PAD, BLOCK_H))
    tl.store(out + output_offsets, output_value, mask=output_mask)


def hc_expand(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """Expand ``x`` to ``[num_tokens, hc_mult, hidden_size]`` on Ascend.

    A two-dimensional grid supplies the row directly and keeps the hidden
    dimension contiguous. This avoids the one-dimensional ``index // H`` and
    ``index % H`` pattern, which FlagTree lowers through discrete scalar memory
    accesses on Ascend.
    """
    if x.ndim != 2:
        raise ValueError("hc_expand expects a rank-2 input")
    if hc_mult < 0:
        raise ValueError("hc_mult must be non-negative")
    if x.device.type != "npu":
        raise RuntimeError("the Ascend hc_expand implementation requires an NPU")

    # A strided hidden dimension requires repeated gather operations on Ascend.
    # Compacting once is substantially faster than expanding from the original
    # strided storage directly.
    if (
        x.stride(1) != 1
        and x.numel() > _SMALL_STRIDED_ELEMS
    ):
        x = x.contiguous()

    num_tokens, hidden_size = x.shape
    out = x.new_empty((num_tokens, hc_mult, hidden_size))
    if not out.numel():
        return out

    if hidden_size <= 32 and num_tokens <= _MAX_WORKERS:
        block_h = triton.next_power_of_2(hidden_size)
        stream_pad = triton.next_power_of_2(hc_mult)
        if stream_pad * block_h <= _SMALL_ROW_TILE_ELEMS_CAP:
            _hc_expand_small_hidden_kernel[(num_tokens,)](
                x,
                out,
                hidden_size,
                hc_mult,
                stream_pad,
                *x.stride(),
                BLOCK_H=block_h,
                num_warps=2,
                num_stages=1,
            )
            return out

    element_size = x.element_size()
    block_r, block_h = _select_persistent_tile(
        num_tokens,
        hidden_size,
        hc_mult,
        element_size,
    )

    total_tiles = (
        triton.cdiv(num_tokens, block_r)
        * triton.cdiv(hidden_size, block_h)
    )
    grid = (min(total_tiles, _vector_core_num()),)
    _hc_expand_persistent_kernel[grid](
        x,
        out,
        num_tokens,
        hidden_size,
        hc_mult,
        *x.stride(),
        BLOCK_R=block_r,
        BLOCK_H=block_h,
        num_warps=2,
        num_stages=2,
    )
    return out
