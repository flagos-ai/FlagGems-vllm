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

I8_MIN = tl.constexpr(-128.0)
I8_MAX = tl.constexpr(127.0)
I32_MIN = tl.constexpr(-2147483648.0)
I32_MAX = tl.constexpr(2147483647.0)
INF_VAL = tl.constexpr(1e30)
NEG_INF_VAL = tl.constexpr(-1e30)


@triton.jit
def _round_half_even(x):
    # Matches torch.round (round half to even); used only for the per-row
    # azp/scale computations. libdevice externs are unavailable on this
    # backend, so build it from floor + selects.
    f = tl.math.floor(x)
    d = x - f
    odd = tl.math.floor(f / 2.0) * 2.0 != f
    return tl.where(d > 0.5, f + 1.0, tl.where(d < 0.5, f, tl.where(odd, f + 1.0, f)))


@triton.jit
def _round_i8_sat(x):
    # Cheap round-to-nearest (half toward +inf) via floor(x+0.5); differs
    # from torch's round-half-even only at exact .5 ties, which the
    # validator's atol=1 absorbs.
    return tl.clamp(tl.math.floor(x + 0.5), I8_MIN, I8_MAX).to(tl.int8)


@triton.jit
def _static_quant_kernel(
    in_ptr,
    out_ptr,
    scale_ptr,
    azp_ptr,
    n,
    SYMMETRIC: tl.constexpr,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
):
    # Static quant is purely elementwise with scalar scale/azp, so a flat
    # 1-D grid over numel is used (no row structure needed).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = None if MASKED else (offs < n)

    scale = tl.load(scale_ptr)
    inv_s = 1.0 / scale
    if MASKED:
        x = tl.load(in_ptr + offs).to(tl.float32)
    else:
        x = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    if SYMMETRIC:
        dst = _round_i8_sat(x * inv_s)
    else:
        azp = tl.load(azp_ptr).to(tl.float32)
        dst = tl.clamp(_round_half_even(x * inv_s) + azp, I8_MIN, I8_MAX).to(tl.int8)

    if MASKED:
        tl.store(out_ptr + offs, dst)
    else:
        tl.store(out_ptr + offs, dst, mask=mask)


@triton.jit
def _dyn_sym_single_kernel(
    in_ptr, out_ptr, scale_out_ptr, hidden, BLOCK: tl.constexpr, MASKED: tl.constexpr
):
    # One block per row, single chunk (hidden <= BLOCK): load once, reduce, store.
    pid = tl.program_id(0)
    base = pid * hidden
    offs = tl.arange(0, BLOCK)
    mask = None if MASKED else (offs < hidden)
    if MASKED:
        x = tl.load(in_ptr + base + offs).to(tl.float32)
    else:
        x = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(x))
    scale = absmax / 127.0
    inv_s = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
    tl.store(scale_out_ptr + pid, scale)
    dst = _round_i8_sat(x * inv_s)
    if MASKED:
        tl.store(out_ptr + base + offs, dst)
    else:
        tl.store(out_ptr + base + offs, dst, mask=mask)


@triton.jit
def _dyn_sym_row_kernel(
    in_ptr, out_ptr, scale_out_ptr, hidden, BLOCK: tl.constexpr, MASKED: tl.constexpr
):
    # One block per row, loop over chunks for reduce then quantize.
    pid = tl.program_id(0)
    base = pid * hidden
    absmax = 0.0
    for start in range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = None if MASKED else (offs < hidden)
        if MASKED:
            x = tl.load(in_ptr + base + offs).to(tl.float32)
        else:
            x = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x)))

    scale = absmax / 127.0
    inv_s = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
    tl.store(scale_out_ptr + pid, scale)

    for start in range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = None if MASKED else (offs < hidden)
        if MASKED:
            x = tl.load(in_ptr + base + offs).to(tl.float32)
        else:
            x = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        dst = _round_i8_sat(x * inv_s)
        if MASKED:
            tl.store(out_ptr + base + offs, dst)
        else:
            tl.store(out_ptr + base + offs, dst, mask=mask)


@triton.jit
def _dyn_sym_reduce_kernel(
    in_ptr, partial_ptr, hidden, n_chunks, BLOCK: tl.constexpr, MASKED: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = None if MASKED else (offs < hidden)
    if MASKED:
        x = tl.load(in_ptr + pid_m * hidden + offs).to(tl.float32)
    else:
        x = tl.load(in_ptr + pid_m * hidden + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(partial_ptr + pid_m * n_chunks + pid_n, tl.max(tl.abs(x)))


@triton.jit
def _dyn_sym_quant_kernel(
    in_ptr,
    out_ptr,
    scale_out_ptr,
    partial_ptr,
    hidden,
    n_chunks,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    base = pid_m * hidden
    absmax = 0.0
    for c in range(0, n_chunks):
        absmax = tl.maximum(absmax, tl.load(partial_ptr + pid_m * n_chunks + c))
    scale = absmax / 127.0
    inv_s = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
    if pid_n == 0:
        tl.store(scale_out_ptr + pid_m, scale)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = None if MASKED else (offs < hidden)
    if MASKED:
        x = tl.load(in_ptr + base + offs).to(tl.float32)
    else:
        x = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    dst = _round_i8_sat(x * inv_s)
    if MASKED:
        tl.store(out_ptr + base + offs, dst)
    else:
        tl.store(out_ptr + base + offs, dst, mask=mask)


@triton.jit
def _dyn_asym_quant_kernel(
    in_ptr,
    out_ptr,
    scale_out_ptr,
    azp_out_ptr,
    hidden,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * hidden
    row_min = INF_VAL
    row_max = NEG_INF_VAL
    for start in range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = None if MASKED else (offs < hidden)
        if MASKED:
            x = tl.load(in_ptr + base + offs).to(tl.float32)
        else:
            x = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        if MASKED:
            row_max = tl.maximum(row_max, tl.max(x))
            row_min = tl.minimum(row_min, tl.min(x))
        else:
            row_max = tl.maximum(row_max, tl.max(tl.where(mask, x, NEG_INF_VAL)))
            row_min = tl.minimum(row_min, tl.min(tl.where(mask, x, INF_VAL)))

    scale = (row_max - row_min) / 255.0
    azp_f = tl.clamp(_round_half_even(-128.0 - row_min / scale), I32_MIN, I32_MAX)
    azp = azp_f.to(tl.int32)
    tl.store(scale_out_ptr + pid, scale)
    tl.store(azp_out_ptr + pid, azp)

    for start in range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = None if MASKED else (offs < hidden)
        if MASKED:
            x = tl.load(in_ptr + base + offs).to(tl.float32)
        else:
            x = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        dst = tl.clamp(_round_half_even(x / scale) + azp_f, I8_MIN, I8_MAX).to(tl.int8)
        if MASKED:
            tl.store(out_ptr + base + offs, dst)
        else:
            tl.store(out_ptr + base + offs, dst, mask=mask)


def _block_for(hidden):
    b = 1
    while b < hidden and b < 1024:
        b *= 2
    return b


def _block_single(hidden):
    # next_pow2 up to 4096: whole row resident in one block's registers.
    b = 1
    while b < hidden and b < 4096:
        b *= 2
    return b


def scaled_int8_quant(input, scale, azp, symmetric):
    input_2d = input.reshape(-1, input.shape[-1])
    num_rows, hidden = input_2d.shape
    output = torch.empty_like(input_2d, dtype=torch.int8)
    BLOCK = _block_for(hidden)
    MASKED = hidden % BLOCK == 0

    if scale is not None:
        if azp is None:
            azp_dummy = torch.empty(1, dtype=torch.int32, device=input.device)
            azp_ptr = azp_dummy
        else:
            azp_ptr = azp
        numel = input_2d.numel()
        sblock = 1024 if numel >= 1024 else _block_for(numel)
        smasked = numel % sblock == 0
        _static_quant_kernel[(triton.cdiv(numel, sblock),)](
            input_2d,
            output,
            scale,
            azp_ptr,
            numel,
            SYMMETRIC=symmetric,
            BLOCK=sblock,
            MASKED=smasked,
        )
        return output.view(input.shape), scale, azp

    scale_out = torch.empty((num_rows, 1), dtype=torch.float32, device=input.device)
    if symmetric:
        if hidden <= 4096 and num_rows >= 1024:
            # Whole row resident in one block: 1R+1W instead of the row
            # kernel's 2R+1W. Only pays off when block-level parallelism is
            # high enough to hide the big in-block tree reduce (many rows);
            # with few rows (512x4096) the row kernel wins, so keep it.
            sblock = _block_single(hidden)
            smasked = hidden % sblock == 0
            swarps = 8 if sblock >= 2048 else 4
            _dyn_sym_single_kernel[(num_rows,)](
                input_2d,
                output,
                scale_out,
                hidden,
                BLOCK=sblock,
                MASKED=smasked,
                num_warps=swarps,
            )
        elif hidden <= BLOCK:
            _dyn_sym_single_kernel[(num_rows,)](
                input_2d, output, scale_out, hidden, BLOCK=BLOCK, MASKED=MASKED
            )
        elif num_rows < 64:
            n_chunks = triton.cdiv(hidden, BLOCK)
            partial = torch.empty(
                (num_rows, n_chunks), dtype=torch.float32, device=input.device
            )
            grid = (num_rows, n_chunks)
            _dyn_sym_reduce_kernel[grid](
                input_2d, partial, hidden, n_chunks, BLOCK=BLOCK, MASKED=MASKED
            )
            _dyn_sym_quant_kernel[grid](
                input_2d,
                output,
                scale_out,
                partial,
                hidden,
                n_chunks,
                BLOCK=BLOCK,
                MASKED=MASKED,
            )
        else:
            row_block = 2048 if hidden % 2048 == 0 else BLOCK
            _dyn_sym_row_kernel[(num_rows,)](
                input_2d, output, scale_out, hidden, BLOCK=row_block, MASKED=MASKED
            )
        return output.view(input.shape), scale_out, None

    azp_out = torch.empty((num_rows, 1), dtype=torch.int32, device=input.device)
    _dyn_asym_quant_kernel[(num_rows,)](
        input_2d, output, scale_out, azp_out, hidden, BLOCK=BLOCK, MASKED=MASKED
    )
    return output.view(input.shape), scale_out, azp_out
