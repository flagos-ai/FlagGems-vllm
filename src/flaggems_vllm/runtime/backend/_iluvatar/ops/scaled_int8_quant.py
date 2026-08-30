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

try:
    from triton.language.extra import libdevice as _tld

    @triton.jit
    def _round_even(x):
        return _tld.nearbyint(x)

except Exception:
    # Fallback: floor-based round-half-to-even (bitwise identical to torch.round)
    @triton.jit
    def _round_even(x):
        f = tl.math.floor(x)
        d = x - f
        f_odd = (f - 2.0 * tl.math.floor(f * 0.5)) == 1.0
        r = tl.where(
            d < 0.5, f, tl.where(d > 0.5, f + 1.0, tl.where(f_odd, f + 1.0, f))
        )
        return r


@triton.jit
def _clamp_i8(x):
    return tl.maximum(tl.minimum(x, 127.0), -128.0).to(tl.int8)


# ---------------------------------------------------------------------------
# Static quantization: scale (and optionally azp) are given.  Pure pointwise.
# Flat 1-D grid over the whole tensor.  EVEN skips boundary masks when
# numel % BLOCK_SIZE == 0.
# ---------------------------------------------------------------------------
@triton.jit
def _static_quant_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    azp_ptr,
    numel,
    SYMMETRIC: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    scale = tl.load(scale_ptr)

    if EVEN:
        src = tl.load(input_ptr + offs).to(tl.float32)
        if SYMMETRIC:
            q = _round_even(src / scale)
        else:
            azp = tl.load(azp_ptr)
            q = _round_even(src / scale) + azp.to(tl.float32)
        tl.store(output_ptr + offs, _clamp_i8(q))
    else:
        mask = offs < numel
        src = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if SYMMETRIC:
            q = _round_even(src / scale)
        else:
            azp = tl.load(azp_ptr)
            q = _round_even(src / scale) + azp.to(tl.float32)
        tl.store(output_ptr + offs, _clamp_i8(q), mask=mask)


# ---------------------------------------------------------------------------
# Dynamic quantization, symmetric: per-row absmax, scale = absmax/127,
# out = clamp(round(src * (127/absmax)), -128, 127).
# SINGLE keeps the whole row in registers; LOOP re-reads the row in chunks.
# EVEN skips boundary masks when the row exactly fills the tile.
# ---------------------------------------------------------------------------
@triton.jit
def _dyn_sym_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    azp_out_ptr,
    hidden,
    SINGLE: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * hidden

    if SINGLE:
        offs = tl.arange(0, BLOCK_SIZE)
        if EVEN:
            src = tl.load(input_ptr + row_offset + offs).to(tl.float32)
            row_absmax = tl.max(tl.abs(src))
            scale = row_absmax / 127.0
            inv = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)
            tl.store(scale_out_ptr + pid, scale)
            dst = _clamp_i8(_round_even(src * inv))
            tl.store(output_ptr + row_offset + offs, dst)
        else:
            mask = offs < hidden
            src = tl.load(input_ptr + row_offset + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            row_absmax = tl.max(tl.abs(src))
            scale = row_absmax / 127.0
            inv = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)
            tl.store(scale_out_ptr + pid, scale)
            dst = _clamp_i8(_round_even(src * inv))
            tl.store(output_ptr + row_offset + offs, dst, mask=mask)
    else:
        if EVEN:
            row_absmax = 0.0
            for start in range(0, hidden, BLOCK_SIZE):
                offs = start + tl.arange(0, BLOCK_SIZE)
                src = tl.load(input_ptr + row_offset + offs).to(tl.float32)
                row_absmax = tl.maximum(row_absmax, tl.max(tl.abs(src)))
            scale = row_absmax / 127.0
            inv = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)
            tl.store(scale_out_ptr + pid, scale)
            for start in range(0, hidden, BLOCK_SIZE):
                offs = start + tl.arange(0, BLOCK_SIZE)
                src = tl.load(input_ptr + row_offset + offs).to(tl.float32)
                dst = _clamp_i8(_round_even(src * inv))
                tl.store(output_ptr + row_offset + offs, dst)
        else:
            row_absmax = 0.0
            for start in range(0, hidden, BLOCK_SIZE):
                offs = start + tl.arange(0, BLOCK_SIZE)
                mask = offs < hidden
                src = tl.load(input_ptr + row_offset + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                row_absmax = tl.maximum(row_absmax, tl.max(tl.abs(src)))
            scale = row_absmax / 127.0
            inv = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)
            tl.store(scale_out_ptr + pid, scale)
            for start in range(0, hidden, BLOCK_SIZE):
                offs = start + tl.arange(0, BLOCK_SIZE)
                mask = offs < hidden
                src = tl.load(input_ptr + row_offset + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                dst = _clamp_i8(_round_even(src * inv))
                tl.store(output_ptr + row_offset + offs, dst, mask=mask)


# ---------------------------------------------------------------------------
# Dynamic quantization, asymmetric: per-row min/max, scale=(max-min)/255,
# azp = clamp(round(-128 - min/scale)) as int32, out = clamp(round(src/scale)+azp).
# ---------------------------------------------------------------------------
@triton.jit
def _dyn_asy_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    azp_out_ptr,
    hidden,
    SINGLE: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * hidden

    if SINGLE:
        offs = tl.arange(0, BLOCK_SIZE)
        if EVEN:
            src = tl.load(input_ptr + row_offset + offs).to(tl.float32)
            row_max = tl.max(src)
            row_min = tl.min(src)
            scale = (row_max - row_min) / 255.0
            azp = _round_even(-128.0 - row_min / scale)
            azp = tl.maximum(tl.minimum(azp, 2147483647.0), -2147483648.0).to(tl.int32)
            tl.store(scale_out_ptr + pid, scale)
            tl.store(azp_out_ptr + pid, azp)
            q = _round_even(src / scale) + azp.to(tl.float32)
            tl.store(output_ptr + row_offset + offs, _clamp_i8(q))
        else:
            mask = offs < hidden
            src = tl.load(input_ptr + row_offset + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            row_max = tl.max(tl.where(mask, src, -1e30))
            row_min = tl.min(tl.where(mask, src, 1e30))
            scale = (row_max - row_min) / 255.0
            azp = _round_even(-128.0 - row_min / scale)
            azp = tl.maximum(tl.minimum(azp, 2147483647.0), -2147483648.0).to(tl.int32)
            tl.store(scale_out_ptr + pid, scale)
            tl.store(azp_out_ptr + pid, azp)
            q = _round_even(src / scale) + azp.to(tl.float32)
            tl.store(output_ptr + row_offset + offs, _clamp_i8(q), mask=mask)
    else:
        if EVEN:
            row_max = -1e30
            row_min = 1e30
            for start in range(0, hidden, BLOCK_SIZE):
                offs = start + tl.arange(0, BLOCK_SIZE)
                src = tl.load(input_ptr + row_offset + offs).to(tl.float32)
                row_max = tl.maximum(row_max, tl.max(src))
                row_min = tl.minimum(row_min, tl.min(src))
            scale = (row_max - row_min) / 255.0
            azp = _round_even(-128.0 - row_min / scale)
            azp = tl.maximum(tl.minimum(azp, 2147483647.0), -2147483648.0).to(tl.int32)
            tl.store(scale_out_ptr + pid, scale)
            tl.store(azp_out_ptr + pid, azp)
            for start in range(0, hidden, BLOCK_SIZE):
                offs = start + tl.arange(0, BLOCK_SIZE)
                src = tl.load(input_ptr + row_offset + offs).to(tl.float32)
                q = _round_even(src / scale) + azp.to(tl.float32)
                tl.store(output_ptr + row_offset + offs, _clamp_i8(q))
        else:
            row_max = -1e30
            row_min = 1e30
            for start in range(0, hidden, BLOCK_SIZE):
                offs = start + tl.arange(0, BLOCK_SIZE)
                mask = offs < hidden
                src = tl.load(input_ptr + row_offset + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                row_max = tl.maximum(row_max, tl.max(tl.where(mask, src, -1e30)))
                row_min = tl.minimum(row_min, tl.min(tl.where(mask, src, 1e30)))
            scale = (row_max - row_min) / 255.0
            azp = _round_even(-128.0 - row_min / scale)
            azp = tl.maximum(tl.minimum(azp, 2147483647.0), -2147483648.0).to(tl.int32)
            tl.store(scale_out_ptr + pid, scale)
            tl.store(azp_out_ptr + pid, azp)
            for start in range(0, hidden, BLOCK_SIZE):
                offs = start + tl.arange(0, BLOCK_SIZE)
                mask = offs < hidden
                src = tl.load(input_ptr + row_offset + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                q = _round_even(src / scale) + azp.to(tl.float32)
                tl.store(output_ptr + row_offset + offs, _clamp_i8(q), mask=mask)


_LOOP_BLOCK = 1024


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


_dummy_azp_cache = {}


def _dummy_azp(dev):
    t = _dummy_azp_cache.get(dev)
    if t is None:
        t = torch.empty(1, dtype=torch.int32, device=dev)
        _dummy_azp_cache[dev] = t
    return t


def _dyn_launch(kernel, input, out, scale_out, azp_out, num_rows, hidden):
    """Dispatch the dynamic kernel: register-tile SINGLE path when the row is
    small (hidden <= 4096) or there are few rows (<= 128) with hidden <= 16384;
    otherwise a two-loop path with 1024-lane tiles and 8 warps."""
    if hidden <= 4096 or (num_rows <= 128 and hidden <= 16384):
        blk = _next_pow2(hidden)
        if blk < 256:
            blk = 256
        nw = (
            64
            if blk >= 16384
            else (
                32 if blk >= 8192 else (16 if blk >= 2048 else (8 if blk >= 512 else 4))
            )
        )
        kernel[(num_rows,)](
            input,
            out,
            scale_out,
            azp_out,
            hidden,
            SINGLE=True,
            EVEN=(hidden == blk),
            BLOCK_SIZE=blk,
            num_warps=nw,
        )
    else:
        kernel[(num_rows,)](
            input,
            out,
            scale_out,
            azp_out,
            hidden,
            SINGLE=False,
            EVEN=(hidden % _LOOP_BLOCK == 0),
            BLOCK_SIZE=_LOOP_BLOCK,
            num_warps=8,
        )


def scaled_int8_quant(input, scale, azp, symmetric):
    if isinstance(symmetric, bool):
        sym = symmetric
    elif hasattr(symmetric, "item"):
        sym = bool(symmetric.item())
    else:
        sym = bool(symmetric)

    hidden = input.shape[-1]
    numel = input.numel()
    num_rows = numel // hidden
    dev = input.device

    out = torch.empty(numel, dtype=torch.int8, device=dev)

    if scale is None:
        # ---- dynamic: compute per-row stats, then quantize ----
        scale_out = torch.empty((num_rows, 1), dtype=torch.float32, device=dev)
        if sym:
            _dyn_launch(
                _dyn_sym_kernel,
                input,
                out,
                scale_out,
                _dummy_azp(dev),
                num_rows,
                hidden,
            )
            return out.view(input.shape), scale_out, None
        else:
            azp_out = torch.empty((num_rows, 1), dtype=torch.int32, device=dev)
            _dyn_launch(
                _dyn_asy_kernel, input, out, scale_out, azp_out, num_rows, hidden
            )
            return out.view(input.shape), scale_out, azp_out

    # ---- static: scale (and maybe azp) given ----
    azp_arg = azp if azp is not None else _dummy_azp(dev)
    if numel < 262144:
        blk, nw = 2048, 8
    else:
        blk, nw = 1024, 8
    grid = (triton.cdiv(numel, blk),)
    _static_quant_kernel[grid](
        input,
        out,
        scale,
        azp_arg,
        numel,
        SYMMETRIC=sym,
        EVEN=(numel % blk == 0),
        BLOCK_SIZE=blk,
        num_warps=nw,
    )
    return out.view(input.shape), scale, azp
