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

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

_I8_MIN = tl.constexpr(-128.0)
_I8_MAX = tl.constexpr(127.0)
_I32_MIN = tl.constexpr(-2147483648.0)
_I32_MAX = tl.constexpr(2147483647.0)
_INF = tl.constexpr(1e30)
_NEG_INF = tl.constexpr(-1e30)


@triton.jit
def _round_i8_sat(x):
    return tl.clamp(tldevice.nearbyint(x), _I8_MIN, _I8_MAX).to(tl.int8)


@triton.jit
def _round_i32_sat(x):
    return tl.clamp(tldevice.nearbyint(x), _I32_MIN, _I32_MAX).to(tl.int32)


@triton.jit
def _sat_i32_to_i8(x):
    return tl.minimum(tl.maximum(x, -128), 127).to(tl.int8)


@triton.jit
def _static_flat_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    azp_ptr,
    numel,
    SYMMETRIC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    src = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)
    inv_s = 1.0 / scale
    if SYMMETRIC:
        dst = _round_i8_sat(src * inv_s)
    else:
        azp = tl.load(azp_ptr)
        dst = _sat_i32_to_i8(_round_i32_sat(src * inv_s) + azp)
    tl.store(output_ptr + offs, dst, mask=mask)


@triton.jit
def _dynamic_quant_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    azp_out_ptr,
    hidden_size,
    SYMMETRIC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_base = input_ptr + pid * hidden_size

    if SYMMETRIC:
        row_absmax = 0.0
        for start in range(0, hidden_size, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < hidden_size
            src = tl.load(row_base + offs, mask=mask, other=0.0).to(tl.float32)
            row_absmax = tl.maximum(row_absmax, tl.max(tl.abs(src)))
        scale = row_absmax / 127.0
        inv_s = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)
        tl.store(scale_out_ptr + pid, scale)
        for start in range(0, hidden_size, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < hidden_size
            src = tl.load(row_base + offs, mask=mask, other=0.0).to(tl.float32)
            dst = _round_i8_sat(src * inv_s)
            tl.store(output_ptr + pid * hidden_size + offs, dst, mask=mask)
    else:
        row_min = _INF
        row_max = _NEG_INF
        for start in range(0, hidden_size, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < hidden_size
            src = tl.load(row_base + offs, mask=mask, other=0.0).to(tl.float32)
            row_min = tl.minimum(row_min, tl.min(tl.where(mask, src, _INF)))
            row_max = tl.maximum(row_max, tl.max(tl.where(mask, src, _NEG_INF)))
        scale = (row_max - row_min) / 255.0
        inv_s = 1.0 / scale
        azp = _round_i32_sat(-128.0 - row_min * inv_s)
        tl.store(scale_out_ptr + pid, scale)
        tl.store(azp_out_ptr + pid, azp)
        for start in range(0, hidden_size, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < hidden_size
            src = tl.load(row_base + offs, mask=mask, other=0.0).to(tl.float32)
            dst = _sat_i32_to_i8(_round_i32_sat(src * inv_s) + azp)
            tl.store(output_ptr + pid * hidden_size + offs, dst, mask=mask)


@triton.jit
def _dyn_sym_reduce_kernel(
    input_ptr,
    partial_ptr,
    hidden_size,
    num_chunks,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pc = tl.program_id(1)
    start = pc * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < hidden_size
    src = tl.load(input_ptr + pid * hidden_size + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    m = tl.max(tl.abs(src))
    tl.store(partial_ptr + pid * num_chunks + pc, m)


@triton.jit
def _dyn_sym_quant_kernel(
    input_ptr,
    output_ptr,
    partial_ptr,
    scale_out_ptr,
    hidden_size,
    num_chunks,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pc = tl.program_id(1)
    m = 0.0
    for c in range(num_chunks):
        m = tl.maximum(m, tl.load(partial_ptr + pid * num_chunks + c))
    scale = m / 127.0
    inv_s = tl.where(m == 0.0, 0.0, 127.0 / m)
    tl.store(scale_out_ptr + pid, scale)
    start = pc * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < hidden_size
    src = tl.load(input_ptr + pid * hidden_size + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    dst = _round_i8_sat(src * inv_s)
    tl.store(output_ptr + pid * hidden_size + offs, dst, mask=mask)


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _static_block_for(numel):
    if numel >= 4 * 1024 * 1024:
        return 4096
    return min(_next_pow2(numel), 2048)


def _static_warps_for(block):
    return min(max(block // 512, 1), 4)


def _dyn_block_for(hidden, num_rows):
    if hidden == 4096:
        return 4096
    return min(_next_pow2(hidden), 2048)


def _dyn_warps_for(block, num_rows):
    target = 8 if num_rows == 1 else 4
    return min(target, max(block // 64, 1))


def scaled_int8_quant(input, scale, azp, symmetric):
    num_rows, hidden_size = input.shape[0], input.shape[1]
    numel = num_rows * hidden_size

    if scale is not None:
        block = _static_block_for(numel)
        grid = (triton.cdiv(numel, block),)
        azp_arg = azp
        if azp_arg is None:
            azp_arg = torch.empty(1, dtype=torch.int32, device=input.device)
        output = torch.empty_like(input, dtype=torch.int8)
        _static_flat_kernel[grid](
            input,
            output,
            scale,
            azp_arg,
            numel,
            SYMMETRIC=symmetric,
            BLOCK=block,
            num_warps=_static_warps_for(block),
        )
        return output, scale, azp

    output = torch.empty_like(input, dtype=torch.int8)
    scale_out = torch.empty((num_rows, 1), dtype=torch.float32, device=input.device)

    if symmetric and num_rows == 1 and hidden_size > 2048:
        block = _dyn_block_for(hidden_size, num_rows)
        num_chunks = triton.cdiv(hidden_size, block)
        partial = torch.empty((num_chunks,), dtype=torch.float32, device=input.device)
        _dyn_sym_reduce_kernel[(1, num_chunks)](
            input,
            partial,
            hidden_size,
            num_chunks,
            BLOCK=block,
            num_warps=8,
        )
        _dyn_sym_quant_kernel[(1, num_chunks)](
            input,
            output,
            partial,
            scale_out,
            hidden_size,
            num_chunks,
            BLOCK=block,
            num_warps=8,
        )
        return output, scale_out, None

    block = _dyn_block_for(hidden_size, num_rows)
    if symmetric:
        azp_dummy = torch.empty(1, dtype=torch.int32, device=input.device)
        _dynamic_quant_kernel[(num_rows,)](
            input,
            output,
            scale_out,
            azp_dummy,
            hidden_size,
            SYMMETRIC=True,
            BLOCK=block,
            num_warps=_dyn_warps_for(block, num_rows),
            num_stages=2,
        )
        return output, scale_out, None

    azp_out = torch.empty((num_rows, 1), dtype=torch.int32, device=input.device)
    _dynamic_quant_kernel[(num_rows,)](
        input,
        output,
        scale_out,
        azp_out,
        hidden_size,
        SYMMETRIC=False,
        BLOCK=block,
        num_warps=_dyn_warps_for(block, num_rows),
        num_stages=2,
    )
    return output, scale_out, azp_out
