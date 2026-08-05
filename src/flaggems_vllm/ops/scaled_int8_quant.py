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

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry

I8_MIN_VAL = tl.constexpr(-128.0)
I8_MAX_VAL = tl.constexpr(127.0)
I32_MIN_VAL = tl.constexpr(-2147483648.0)
I32_MAX_VAL = tl.constexpr(2147483647.0)
INF_VAL = tl.constexpr(1e30)
NEG_INF_VAL = tl.constexpr(-1e30)


@triton.jit
def _round_i8_sat(x):
    return tl.clamp(tldevice.nearbyint(x), I8_MIN_VAL, I8_MAX_VAL).to(tl.int8)


@triton.jit
def _round_i32_sat(x):
    return tl.clamp(tldevice.nearbyint(x), I32_MIN_VAL, I32_MAX_VAL).to(tl.int32)


@triton.jit
def _saturate_i32_to_i8(x):
    return tl.clamp(x, I8_MIN_VAL, I8_MAX_VAL).to(tl.int8)


# ── static: 2-D grid when few tokens, 1-D otherwise ──────────────────────────


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("scaled_int8_quant_static"),
    key=["hidden_size"],
)
@triton.jit
def _static_int8_quant_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    azp_ptr,
    hidden_size,
    SYMMETRIC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    chunk_start = pid_n * BLOCK_SIZE
    if chunk_start >= hidden_size:
        return

    row_offset = pid_m * hidden_size

    scale = tl.load(scale_ptr)
    inv_s = 1.0 / scale

    if not SYMMETRIC:
        azp = tl.load(azp_ptr)

    in_blk = tl.make_block_ptr(
        base=input_ptr + row_offset,
        shape=(hidden_size,),
        strides=(1,),
        offsets=(chunk_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    src = tl.load(in_blk, boundary_check=(0,), padding_option="zero").to(tl.float32)

    if SYMMETRIC:
        dst = _round_i8_sat(src * inv_s)
    else:
        dst = _saturate_i32_to_i8(_round_i32_sat(src * inv_s) + azp)

    out_blk = tl.make_block_ptr(
        base=output_ptr + row_offset,
        shape=(hidden_size,),
        strides=(1,),
        offsets=(chunk_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    tl.store(out_blk, dst, boundary_check=(0,))


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("scaled_int8_quant_static"),
    key=["hidden_size"],
)
@triton.jit
def _static_int8_quant_kernel_1d(
    input_ptr,
    output_ptr,
    scale_ptr,
    azp_ptr,
    hidden_size,
    SYMMETRIC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * hidden_size

    scale = tl.load(scale_ptr)
    inv_s = 1.0 / scale

    if not SYMMETRIC:
        azp = tl.load(azp_ptr)

    for start in range(0, hidden_size, BLOCK_SIZE):
        in_blk = tl.make_block_ptr(
            base=input_ptr + row_offset,
            shape=(hidden_size,),
            strides=(1,),
            offsets=(start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        src = tl.load(in_blk, boundary_check=(0,), padding_option="zero").to(tl.float32)

        if SYMMETRIC:
            dst = _round_i8_sat(src * inv_s)
        else:
            dst = _saturate_i32_to_i8(_round_i32_sat(src * inv_s) + azp)

        out_blk = tl.make_block_ptr(
            base=output_ptr + row_offset,
            shape=(hidden_size,),
            strides=(1,),
            offsets=(start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        tl.store(out_blk, dst, boundary_check=(0,))


# ── dynamic: single-kernel 1-D, block-pointer loading ────────────────────────


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("scaled_int8_quant_dynamic"),
    key=["hidden_size"],
)
@triton.jit
def _dynamic_int8_quant_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    azp_out_ptr,
    hidden_size,
    SYMMETRIC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * hidden_size

    if SYMMETRIC:
        row_absmax = 0.0
        for start in range(0, hidden_size, BLOCK_SIZE):
            in_blk = tl.make_block_ptr(
                base=input_ptr + row_offset,
                shape=(hidden_size,),
                strides=(1,),
                offsets=(start,),
                block_shape=(BLOCK_SIZE,),
                order=(0,),
            )
            src = tl.load(in_blk, boundary_check=(0,), padding_option="zero").to(
                tl.float32
            )
            chunk_absmax = tl.max(tl.abs(src))
            row_absmax = tl.maximum(row_absmax, chunk_absmax)

        scale = row_absmax / 127.0
        inv_s = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)

        tl.store(scale_out_ptr + pid, scale)

        for start in range(0, hidden_size, BLOCK_SIZE):
            in_blk = tl.make_block_ptr(
                base=input_ptr + row_offset,
                shape=(hidden_size,),
                strides=(1,),
                offsets=(start,),
                block_shape=(BLOCK_SIZE,),
                order=(0,),
            )
            src = tl.load(in_blk, boundary_check=(0,), padding_option="zero").to(
                tl.float32
            )
            dst = _round_i8_sat(src * inv_s)
            out_blk = tl.make_block_ptr(
                base=output_ptr + row_offset,
                shape=(hidden_size,),
                strides=(1,),
                offsets=(start,),
                block_shape=(BLOCK_SIZE,),
                order=(0,),
            )
            tl.store(out_blk, dst, boundary_check=(0,))

    else:
        row_min = INF_VAL
        row_max = NEG_INF_VAL
        for start in range(0, hidden_size, BLOCK_SIZE):
            in_blk = tl.make_block_ptr(
                base=input_ptr + row_offset,
                shape=(hidden_size,),
                strides=(1,),
                offsets=(start,),
                block_shape=(BLOCK_SIZE,),
                order=(0,),
            )
            src = tl.load(in_blk, boundary_check=(0,), padding_option="zero").to(
                tl.float32
            )
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < hidden_size
            row_max = tl.maximum(row_max, tl.max(tl.where(mask, src, NEG_INF_VAL)))
            row_min = tl.minimum(row_min, tl.min(tl.where(mask, src, INF_VAL)))

        scale = (row_max - row_min) / 255.0
        inv_s = 1.0 / scale
        azp = _round_i32_sat(-128.0 - row_min * inv_s)

        tl.store(scale_out_ptr + pid, scale)
        tl.store(azp_out_ptr + pid, azp)

        for start in range(0, hidden_size, BLOCK_SIZE):
            in_blk = tl.make_block_ptr(
                base=input_ptr + row_offset,
                shape=(hidden_size,),
                strides=(1,),
                offsets=(start,),
                block_shape=(BLOCK_SIZE,),
                order=(0,),
            )
            src = tl.load(in_blk, boundary_check=(0,), padding_option="zero").to(
                tl.float32
            )
            dst = _saturate_i32_to_i8(_round_i32_sat(src * inv_s) + azp)
            out_blk = tl.make_block_ptr(
                base=output_ptr + row_offset,
                shape=(hidden_size,),
                strides=(1,),
                offsets=(start,),
                block_shape=(BLOCK_SIZE,),
                order=(0,),
            )
            tl.store(out_blk, dst, boundary_check=(0,))


# ── host dispatch ────────────────────────────────────────────────────────────

# When the token count is below this threshold we use a 2-D grid so each
# block handles a single chunk — this spreads the work across more SMs when
# there are few rows. When there are many rows, the 1-D grid (one block per
# row) already saturates the GPU, and keeping the inner loop avoids grid
# launch overhead.
_2D_GRID_TOKEN_THRESHOLD = 256


def scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    input_2d = input.reshape(-1, input.shape[-1])
    num_tokens, hidden_size = input_2d.shape

    if scale is not None:
        output = torch.empty_like(input_2d, dtype=torch.int8)
        azp_or_dummy = (
            azp
            if azp is not None
            else torch.empty(1, dtype=torch.int32, device=input.device)
        )

        if num_tokens < _2D_GRID_TOKEN_THRESHOLD:
            max_chunks = triton.cdiv(hidden_size, 256)
            grid = (num_tokens, max_chunks)
            _static_int8_quant_kernel[grid](
                input_2d,
                output,
                scale,
                azp_or_dummy,
                hidden_size,
                SYMMETRIC=symmetric,
            )
        else:
            grid = (num_tokens,)
            _static_int8_quant_kernel_1d[grid](
                input_2d,
                output,
                scale,
                azp_or_dummy,
                hidden_size,
                SYMMETRIC=symmetric,
            )
        return output.view(input.shape), scale, azp

    output = torch.empty_like(input_2d, dtype=torch.int8)
    input_scales = torch.empty(
        (num_tokens, 1), device=input.device, dtype=torch.float32
    )
    if symmetric:
        input_azp = None
        azp_out_or_dummy = torch.empty(
            (num_tokens, 1), device=input.device, dtype=torch.int32
        )
    else:
        input_azp = torch.empty((num_tokens, 1), device=input.device, dtype=torch.int32)
        azp_out_or_dummy = input_azp

    _dynamic_int8_quant_kernel[(num_tokens,)](
        input_2d,
        output,
        input_scales,
        azp_out_or_dummy,
        hidden_size,
        SYMMETRIC=symmetric,
    )
    return output.view(input.shape), input_scales, input_azp
