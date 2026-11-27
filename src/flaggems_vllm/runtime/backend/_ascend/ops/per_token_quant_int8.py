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

import shutil
from functools import lru_cache

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice

from flaggems_vllm.runtime import torch_device_fn

try:
    import triton.experimental.tle.language.raw as tle_raw
    from triton.experimental.tle import dsa
    from triton.experimental.tle.raw import dialect

    _HAS_RAW_QUANTIZE = hasattr(dsa.ascend, "UB")
except (AttributeError, ImportError):
    _HAS_RAW_QUANTIZE = False

# Keep the FP32 tile, cast scratch and outputs within the 910B's 192 KiB UB.
_TILE_ELEMENTS = 16384
_HALF_TILE_ELEMENTS = 32768
_CAST_SPAN = 8192

if _HAS_RAW_QUANTIZE:

    @dialect(
        name="cann",
        file="per_token_quant_int8.cpp",
        extern_func_name="per_token_quant_tile",
        pipeline={"core": "vector", "pipe": "PIPE_V"},
        extra_buffers=(tl.float16, _CAST_SPAN),
    )
    def _quantize_tile(*args, **kwargs): ...

    @dialect(
        name="cann",
        file="per_token_quant_int8.cpp",
        extern_func_name="per_token_quant_half_tile",
        pipeline={"core": "vector", "pipe": "PIPE_V"},
        extra_buffers=[(tl.float32, _CAST_SPAN), (tl.float16, _CAST_SPAN)],
    )
    def _quantize_half_tile(*args, **kwargs): ...


@triton.jit
def _max_pair(left, right):
    return tl.maximum(left, right)


@triton.jit
def _quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    PERSISTENT: tl.constexpr,
    PACKED_HALF: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(M, ROWS)
    pid = tl.program_id(0)
    if PERSISTENT:
        tiles_per_core = tl.cdiv(tiles, tl.num_programs(0))
        first = pid * tiles_per_core
        last = tl.minimum(tiles, first + tiles_per_core)
    else:
        first = pid
        last = pid + 1

    # Non-power-of-two rows remain contiguous. Reduce aligned segments in
    # parallel, then finish each row's max in the scalar scale loop.
    if N < COLS and ROWS > 1:
        CHUNK: tl.constexpr = N & -N
    elif COLS > 8192 and PERSISTENT:
        CHUNK: tl.constexpr = 4096
    else:
        CHUNK: tl.constexpr = COLS
    PIECES: tl.constexpr = triton.cdiv(N, CHUNK)
    offsets = tl.arange(0, ROWS * COLS)
    for tile in range(first, last):
        row = tile * ROWS
        base = row * N
        if M % ROWS:
            mask = offsets < tl.minimum(ROWS, M - row) * N
        elif N < COLS:
            mask = offsets < ROWS * N
        else:
            mask = True

        values = tl.load(x_ptr + base + offsets, mask, 0)
        if values.dtype != tl.float16:
            values = values.to(tl.float32)
        values = tl.where(values == values, values, 0.0)
        magnitudes = tl.abs(tl.reshape(values, (ROWS * COLS // CHUNK, CHUNK)))
        # Keep the FP16 reduction in its input dtype. The maximum is one
        # of the input magnitudes, so no precision is lost.
        if values.dtype == tl.float16:
            partials = tl.reduce(magnitudes, 1, _max_pair).to(tl.float32)
        else:
            partials = tl.max(magnitudes, 1)
        q = dsa.to_tensor(
            dsa.alloc([ROWS * COLS], tl.int8, mem_addr_space=dsa.ascend.UB)
        )
        scale = dsa.to_tensor(
            dsa.alloc([ROWS], tl.float32, mem_addr_space=dsa.ascend.UB)
        )
        width = tl.full((), N, tl.int32)
        pieces = tl.full((), PIECES, tl.int32)
        if PACKED_HALF:
            q, scale = tle_raw.call(
                _quantize_half_tile,
                [partials, width, pieces, values, q, scale],
                output_indices=[4, 5],
            )
        else:
            # Reuse the FP32 tile as scratch after the row reduction.
            q, scale, _ = tle_raw.call(
                _quantize_tile,
                [partials, width, pieces, q, scale, values.to(tl.float32)],
                output_indices=[3, 4, 5],
            )
        tl.store(q_ptr + base + offsets, q, mask)
        rows = row + tl.arange(0, ROWS)
        tl.store(scale_ptr + rows, scale, rows < M if M % ROWS else True)


@triton.jit
def _absmax(x):
    magnitude = tl.abs(x)
    return tl.max(tl.where(magnitude == magnitude, magnitude, 0.0), 0).to(tl.float32)


@triton.jit
def _round_to_int8(x, inverse):
    scaled = x.to(tl.float32) * inverse
    rounded = libdevice.round(tl.where(scaled == scaled, scaled, 0.0))
    return rounded.to(tl.float16).to(tl.int8)


@triton.jit
def _per_token_quant_int8_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    m,
    N: tl.constexpr,
    STRIDE_ROW: tl.constexpr,
    STRIDE_COL: tl.constexpr,
    BLOCK: tl.constexpr,
    LARGE: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    if LARGE:
        cols = cols.to(tl.int64)
    for row in range(tl.program_id(0), m, tl.num_programs(0)):
        row_index = row.to(tl.int64) if LARGE else row
        x_row = x_ptr + row_index * STRIDE_ROW
        q_row = q_ptr + row_index * N
        if N <= BLOCK:
            x = tl.load(x_row + cols * STRIDE_COL, cols < N, 0)
            maximum = tl.maximum(_absmax(x), 1e-10)
        else:
            maximum = tl.full((), 1e-10, tl.float32)
            for start in range(0, N, BLOCK):
                chunk = start + cols
                x = tl.load(x_row + chunk * STRIDE_COL, chunk < N, 0)
                maximum = tl.maximum(maximum, _absmax(x))

        # Scalar division preserves FP32 rounding on Ascend. A vector divf
        # can differ by one ULP even when requested through tl.div_rn.
        scale = maximum / 127.0
        inverse = 127.0 / maximum
        if N <= BLOCK:
            tl.store(q_row + cols, _round_to_int8(x, inverse), cols < N)
        else:
            for start in range(0, N, BLOCK):
                chunk = start + cols
                mask = chunk < N
                x = tl.load(x_row + chunk * STRIDE_COL, mask, 0)
                tl.store(q_row + chunk, _round_to_int8(x, inverse), mask)
        tl.store(scale_ptr + row, scale)


@lru_cache(maxsize=None)
def _supports_row_tiles(device: torch.device) -> bool:
    return (
        _HAS_RAW_QUANTIZE
        and torch_device_fn.get_device_name(device).startswith("Ascend910B")
        and shutil.which("ccec") is not None
    )


def per_token_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize last-dimension rows using half-away-from-zero rounding."""
    shape = x.shape
    if x.ndim > 2:
        x = x.view(-1, shape[-1])
    n = x.shape[-1]
    m = x.numel() // n
    if x.ndim < 2:
        raise IndexError("per_token_quant_int8 expects at least two dimensions")
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)

    q = torch.empty(shape, device=x.device, dtype=torch.int8)
    scale = torch.empty(shape[:-1] + (1,), device=x.device, dtype=torch.float32)
    if not m:
        return q, scale

    with torch_device_fn.device(x.device):
        cores = triton.runtime.driver.active.utils.get_aivector_core_num()
        block = triton.next_power_of_2(n)
        large = m * n >= 2**31
        if (
            128 <= n <= _TILE_ELEMENTS
            and not large
            and x.is_contiguous()
            and _supports_row_tiles(x.device)
        ):
            half_rows = _HALF_TILE_ELEMENTS // block
            # Full, aligned tiles let the FP16 path double row batching.
            # Use it when each core can process several tiles.
            packed_half = (
                x.dtype == torch.float16
                and 2048 <= n <= _CAST_SPAN
                and n % 512 == 0
                and m % half_rows == 0
                and m >= (2 if n < block else 4) * cores * half_rows
            )
            capacity = _TILE_ELEMENTS // block
            rows = 1
            if packed_half:
                rows = half_rows
            elif m > cores and (
                n == block or n % 128 == 0 and m >= 2 * cores * capacity
            ):
                rows = min(
                    capacity,
                    triton.next_power_of_2(triton.cdiv(m, max(1, cores // 2))),
                )
            tiles = triton.cdiv(m, rows)
            _quantize_rows_kernel[(min(tiles, cores),)](
                x,
                q,
                scale,
                m,
                n,
                ROWS=rows,
                COLS=block,
                PERSISTENT=tiles > cores,
                PACKED_HALF=packed_half,
                num_warps=1,
                enable_fp_fusion=False,
                multibuffer=False,
            )
        else:
            stride_row, stride_col = x.stride()
            max_input_offset = (m - 1) * stride_row + (n - 1) * stride_col
            # Non-unit column strides also need UB space for gather indices.
            block_limit = 8192 if stride_col == 1 else 2048
            _per_token_quant_int8_kernel[(min(m, cores),)](
                x,
                q,
                scale,
                m,
                n,
                stride_row,
                stride_col,
                BLOCK=min(block_limit, block),
                LARGE=large or max_input_offset >= 2**31,
                num_warps=1,
                enable_fp_fusion=False,
            )
    return q, scale
