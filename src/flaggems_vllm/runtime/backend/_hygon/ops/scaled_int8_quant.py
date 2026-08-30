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

I8_MIN_VAL = tl.constexpr(-128.0)
I8_MAX_VAL = tl.constexpr(127.0)
I32_MIN_VAL = tl.constexpr(-2147483648.0)
I32_MAX_VAL = tl.constexpr(2147483647.0)


def _static_launch_cfg(hidden_size):
    # Re-tuned with the reciprocal-multiply + EVEN kernel on the timing shapes.
    if hidden_size == 1024 or hidden_size == 5120:
        return 4096, 4
    return 2048, 4


@triton.jit
def _static_quant_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    azp_ptr,
    numel,
    SYMMETRIC: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    if EVEN:
        x = tl.load(input_ptr + offs).to(tl.float32)
    else:
        mask = offs < numel
        x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    scale = tl.load(scale_ptr)
    inv_s = 1.0 / scale

    r = tldevice.nearbyint(x * inv_s)
    if not SYMMETRIC:
        azp = tl.load(azp_ptr).to(tl.float32)
        r = r + azp
    r = tl.clamp(r, I8_MIN_VAL, I8_MAX_VAL)
    if EVEN:
        tl.store(output_ptr + offs, r.to(tl.int8))
    else:
        tl.store(output_ptr + offs, r.to(tl.int8), mask=mask)


@triton.jit
def _dynamic_quant_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    azp_out_ptr,
    hidden_size,
    SYMMETRIC: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * hidden_size
    offs = tl.arange(0, BLOCK)

    if EVEN:
        x = tl.load(input_ptr + row_start + offs).to(tl.float32)
    else:
        mask = offs < hidden_size
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)

    if SYMMETRIC:
        absmax = tl.max(tl.abs(x))
        scale = absmax / 127.0
        inv_s = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
        tl.store(scale_out_ptr + pid, scale)
        r = tldevice.nearbyint(x * inv_s)
    else:
        if EVEN:
            row_max = tl.max(x)
            row_min = tl.min(x)
        else:
            xm = tl.where(mask, x, -1e30)
            xp = tl.where(mask, x, 1e30)
            row_max = tl.max(xm)
            row_min = tl.min(xp)
        scale = (row_max - row_min) / 255.0
        azp = tl.clamp(
            tldevice.nearbyint(-128.0 - row_min / scale),
            I32_MIN_VAL,
            I32_MAX_VAL,
        ).to(tl.int32)
        tl.store(scale_out_ptr + pid, scale)
        tl.store(azp_out_ptr + pid, azp)
        r = tldevice.nearbyint(x / scale) + azp.to(tl.float32)

    r = tl.clamp(r, I8_MIN_VAL, I8_MAX_VAL)
    if EVEN:
        tl.store(output_ptr + row_start + offs, r.to(tl.int8))
    else:
        tl.store(output_ptr + row_start + offs, r.to(tl.int8), mask=mask)


def scaled_int8_quant(input, scale, azp, symmetric):
    symmetric = bool(symmetric)
    hidden_size = input.shape[-1]
    numel = input.numel()
    num_tokens = numel // hidden_size

    if scale is not None:
        output = torch.empty_like(input, dtype=torch.int8)
        # When SYMMETRIC the kernel never dereferences azp_ptr, so the scale
        # tensor (a valid 1-element buffer) can stand in instead of a dummy alloc.
        azp_buf = azp if azp is not None else scale
        static_block, static_warps = _static_launch_cfg(hidden_size)
        even = numel % static_block == 0
        grid = (triton.cdiv(numel, static_block),)
        _static_quant_kernel[grid](
            input,
            output,
            scale,
            azp_buf,
            numel,
            SYMMETRIC=symmetric,
            BLOCK=static_block,
            EVEN=even,
            num_warps=static_warps,
        )
        return output, scale, azp

    output = torch.empty_like(input, dtype=torch.int8)
    scale_out = torch.empty((num_tokens, 1), device=input.device, dtype=torch.float32)

    if symmetric:
        # azp_out_ptr is unused in the SYMMETRIC path; reuse scale_out as the
        # valid-but-never-written pointer to avoid a dummy allocation.
        azp_buf = scale_out
        azp_out = None
    else:
        azp_buf = torch.empty((num_tokens, 1), device=input.device, dtype=torch.int32)
        azp_out = azp_buf

    block = 1
    while block < hidden_size:
        block *= 2
    num_warps = 8 if block > 8192 else 4
    even = hidden_size % block == 0

    _dynamic_quant_kernel[(num_tokens,)](
        input,
        output,
        scale_out,
        azp_buf,
        hidden_size,
        SYMMETRIC=symmetric,
        BLOCK=block,
        EVEN=even,
        num_warps=num_warps,
    )
    return output, scale_out, azp_out
