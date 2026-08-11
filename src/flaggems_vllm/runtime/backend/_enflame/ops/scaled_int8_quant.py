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


@triton.jit
def _round_even(x):
    # torch.round semantics: round half to even. This Triton build exposes no
    # round/rint intrinsic, so build it from fmod + floor.
    f = x % 1.0  # fractional part, sign follows x
    u = x - f  # integer part (trunc)
    is_half = (f == 0.5) | (f == -0.5)
    up = f > 0.5
    dn = f < -0.5
    r = u + tl.where(up, 1.0, tl.where(dn, -1.0, 0.0))
    half_u = u * 0.5
    u_odd = half_u != tl.math.floor(half_u)
    sign = tl.where(f > 0.0, 1.0, -1.0)
    r = tl.where(is_half & u_odd, u + sign, r)
    return r


# ---------------------------------------------------------------------------
# Dynamic quantization (scale/azp computed per row from the input).
# One program handles one row. When ROW_SIZE <= 16384 the whole row is held
# in registers (single HBM read); otherwise a two-pass chunked loop is used.
# ---------------------------------------------------------------------------


@triton.jit
def _dyn_sym_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    INLINE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = row * ROW_SIZE

    if INLINE:
        offs = tl.arange(0, BLOCK)
        mask = offs < ROW_SIZE
        x = tl.load(input_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.max(tl.abs(x), axis=0)
        scale = absmax / 127.0
        inv = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
        tl.store(scale_out_ptr + row, scale)
        q = _round_even(x * inv)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(output_ptr + base + offs, q.to(tl.int8), mask=mask)
    else:
        offs = tl.arange(0, CHUNK)
        acc = tl.zeros([CHUNK], dtype=tl.float32)
        for off in range(0, ROW_SIZE, CHUNK):
            m = off + offs < ROW_SIZE
            x = tl.load(input_ptr + base + off + offs, mask=m, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(x))
        absmax = tl.max(acc, axis=0)
        scale = absmax / 127.0
        inv = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
        tl.store(scale_out_ptr + row, scale)
        for off in range(0, ROW_SIZE, CHUNK):
            m = off + offs < ROW_SIZE
            x = tl.load(input_ptr + base + off + offs, mask=m, other=0.0).to(tl.float32)
            q = _round_even(x * inv)
            q = tl.minimum(tl.maximum(q, -128.0), 127.0)
            tl.store(output_ptr + base + off + offs, q.to(tl.int8), mask=m)


@triton.jit
def _dyn_asym_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    azp_out_ptr,
    ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    INLINE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = row * ROW_SIZE

    if INLINE:
        offs = tl.arange(0, BLOCK)
        mask = offs < ROW_SIZE
        x = tl.load(input_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        xmax = tl.where(mask, x, float("-inf"))
        xmin = tl.where(mask, x, float("inf"))
        row_max = tl.max(xmax, axis=0)
        row_min = tl.min(xmin, axis=0)
        scale = (row_max - row_min) / 255.0
        safe = tl.where(scale == 0.0, 1.0, scale)
        azp = _round_even(-128.0 - row_min / safe)
        azp = tl.minimum(tl.maximum(azp, -2147483648.0), 2147483647.0)
        azp_i = azp.to(tl.int32)
        tl.store(scale_out_ptr + row, scale)
        tl.store(azp_out_ptr + row, azp_i)
        q = _round_even(x / safe) + azp_i.to(tl.float32)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(output_ptr + base + offs, q.to(tl.int8), mask=mask)
    else:
        offs = tl.arange(0, CHUNK)
        acc_max = tl.full([CHUNK], float("-inf"), tl.float32)
        acc_min = tl.full([CHUNK], float("inf"), tl.float32)
        for off in range(0, ROW_SIZE, CHUNK):
            m = off + offs < ROW_SIZE
            x = tl.load(input_ptr + base + off + offs, mask=m, other=0.0).to(tl.float32)
            xm = tl.where(m, x, float("-inf"))
            xn = tl.where(m, x, float("inf"))
            acc_max = tl.maximum(acc_max, xm)
            acc_min = tl.minimum(acc_min, xn)
        row_max = tl.max(acc_max, axis=0)
        row_min = tl.min(acc_min, axis=0)
        scale = (row_max - row_min) / 255.0
        safe = tl.where(scale == 0.0, 1.0, scale)
        azp = _round_even(-128.0 - row_min / safe)
        azp = tl.minimum(tl.maximum(azp, -2147483648.0), 2147483647.0)
        azp_i = azp.to(tl.int32)
        tl.store(scale_out_ptr + row, scale)
        tl.store(azp_out_ptr + row, azp_i)
        for off in range(0, ROW_SIZE, CHUNK):
            m = off + offs < ROW_SIZE
            x = tl.load(input_ptr + base + off + offs, mask=m, other=0.0).to(tl.float32)
            q = _round_even(x / safe) + azp_i.to(tl.float32)
            q = tl.minimum(tl.maximum(q, -128.0), 127.0)
            tl.store(output_ptr + base + off + offs, q.to(tl.int8), mask=m)


# ---------------------------------------------------------------------------
# Static quantization: pointwise with a per-tensor scale (and azp).
# ---------------------------------------------------------------------------


@triton.jit
def _static_sym_kernel(input_ptr, scale_ptr, output_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr)
    q = _round_even(x / s)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _static_asym_kernel(
    input_ptr, scale_ptr, azp_ptr, output_ptr, numel, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr)
    a = tl.load(azp_ptr).to(tl.float32)
    q = _round_even(x / s) + a
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)


def _pick_num_warps(block):
    if block <= 1024:
        return 4
    if block <= 4096:
        return 8
    return 16


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _run_dynamic(input, sym):
    row_size = input.shape[-1]
    rows = input.numel() // row_size
    device = input.device

    output = torch.empty(input.shape, dtype=torch.int8, device=device)
    scale_out = torch.empty((rows, 1), dtype=torch.float32, device=device)

    block = _next_pow2(row_size)
    # Inline (whole row in registers, one HBM read) is best for rows that fit
    # a modest block. Wide rows with many programs (e.g. 2048x5120 -> BLOCK
    # 8192, 37% masked lanes) starve occupancy; route them through the
    # register-light two-pass chunked loop. A single wide row also prefers the
    # loop with a small chunk and many warps (on-target sweep: 1x13824 loop
    # C512_w16 = 8.1us vs inline 10.5us).
    if row_size <= 4096:
        inline = True
        CHUNK, lnw = 1024, _pick_num_warps(block)
    elif rows == 1:
        inline = False
        CHUNK, lnw = 512, 16
    else:
        inline = False
        CHUNK, lnw = 1024, 8

    if inline:
        if sym:
            _dyn_sym_kernel[(rows,)](
                input,
                output,
                scale_out,
                ROW_SIZE=row_size,
                BLOCK=block,
                CHUNK=1024,
                INLINE=True,
                num_warps=lnw,
            )
            return output, scale_out, None
        azp_out = torch.empty((rows, 1), dtype=torch.int32, device=device)
        _dyn_asym_kernel[(rows,)](
            input,
            output,
            scale_out,
            azp_out,
            ROW_SIZE=row_size,
            BLOCK=block,
            CHUNK=1024,
            INLINE=True,
            num_warps=lnw,
        )
        return output, scale_out, azp_out

    if sym:
        _dyn_sym_kernel[(rows,)](
            input,
            output,
            scale_out,
            ROW_SIZE=row_size,
            BLOCK=1024,
            CHUNK=CHUNK,
            INLINE=False,
            num_warps=lnw,
        )
        return output, scale_out, None
    azp_out = torch.empty((rows, 1), dtype=torch.int32, device=device)
    _dyn_asym_kernel[(rows,)](
        input,
        output,
        scale_out,
        azp_out,
        ROW_SIZE=row_size,
        BLOCK=1024,
        CHUNK=CHUNK,
        INLINE=False,
        num_warps=lnw,
    )
    return output, scale_out, azp_out


def _run_static(input, scale, azp, sym):
    numel = input.numel()
    device = input.device
    output = torch.empty(input.shape, dtype=torch.int8, device=device)
    # On-target sweep: BLOCK=256/num_warps=2 wins for large tensors, while a
    # tiny workload (1x13824, 27 blocks) prefers 512 elements with 8 warps.
    if numel <= 16384:
        BLOCK, nw = 512, 8
    else:
        BLOCK, nw = 256, 2
    grid = (triton.cdiv(numel, BLOCK),)
    if sym:
        _static_sym_kernel[grid](input, scale, output, numel, BLOCK=BLOCK, num_warps=nw)
        return output, scale, None
    _static_asym_kernel[grid](
        input, scale, azp, output, numel, BLOCK=BLOCK, num_warps=nw
    )
    return output, scale, azp


def scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
):
    if hasattr(symmetric, "item"):
        sym = bool(symmetric.item())
    else:
        sym = bool(symmetric)

    if scale is None:
        return _run_dynamic(input, sym)
    return _run_static(input, scale, azp, sym)
