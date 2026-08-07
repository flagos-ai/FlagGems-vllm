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
    configs=runtime.get_tuned_config("scaled_int8_quant_dynamic_single"),
    key=["hidden_size", "num_tokens"],
)
@triton.heuristics(
    values={"ROW_BLOCK": lambda args: triton.next_power_of_2(args["hidden_size"])}
)
@triton.jit
def _dynamic_int8_quant_kernel_single_pass(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    hidden_size,
    num_tokens,
    ROW_BLOCK: tl.constexpr,
    PLACEHOLDER_UNUSED: tl.constexpr = 1,
):
    """Symmetric dynamic quant that reads the whole row once.

    The full row is loaded into registers, so the absmax pass and the
    quantize pass share a single global-memory read. ROW_BLOCK is set to
    next_power_of_2(hidden_size) via @triton.heuristics, so the tuned
    candidates only act through num_warps/num_stages.
    """
    pid = tl.program_id(0)
    row_offset = pid.to(tl.int64) * hidden_size
    offsets = tl.arange(0, ROW_BLOCK)
    mask = offsets < hidden_size
    src = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    row_absmax = tl.max(tl.abs(src))
    scale = row_absmax / 127.0
    inv_s = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)
    tl.store(scale_out_ptr + pid, scale)
    dst = _round_i8_sat(src * inv_s)
    tl.store(output_ptr + row_offset + offsets, dst, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("scaled_int8_quant_dynamic_main_tail"),
    key=["hidden_size", "num_tokens"],
)
@triton.jit
def _dynamic_int8_quant_kernel_main_tail(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    hidden_size,
    num_tokens,
    MAIN_BLOCK: tl.constexpr,
    TAIL_BLOCK: tl.constexpr,
    PLACEHOLDER_UNUSED: tl.constexpr = 1,
):
    """Symmetric dynamic quant for rows that fit in one main pow2 block plus
    one masked tail block (MAIN_BLOCK + TAIL_BLOCK >= hidden_size).

    Single global read: both chunks stay in registers across the absmax and
    quantize steps. The main block is loaded without masking, which avoids
    the dead-lane waste of padding the whole row to a single power of two
    (e.g. hidden_size 5120 -> MAIN 4096 + TAIL 1024 instead of a masked 8192).
    """
    pid = tl.program_id(0)
    row_offset = pid.to(tl.int64) * hidden_size
    main_offs = tl.arange(0, MAIN_BLOCK)
    tail_offs = MAIN_BLOCK + tl.arange(0, TAIL_BLOCK)
    tail_mask = tail_offs < hidden_size
    src_main = tl.load(input_ptr + row_offset + main_offs).to(tl.float32)
    src_tail = tl.load(
        input_ptr + row_offset + tail_offs, mask=tail_mask, other=0.0
    ).to(tl.float32)
    row_absmax = tl.maximum(tl.max(tl.abs(src_main)), tl.max(tl.abs(src_tail)))
    scale = row_absmax / 127.0
    inv_s = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)
    tl.store(scale_out_ptr + pid, scale)
    tl.store(output_ptr + row_offset + main_offs, _round_i8_sat(src_main * inv_s))
    tl.store(
        output_ptr + row_offset + tail_offs,
        _round_i8_sat(src_tail * inv_s),
        mask=tail_mask,
    )


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("scaled_int8_quant_dynamic_single"),
    key=["hidden_size", "num_tokens"],
)
@triton.heuristics(
    values={"ROW_BLOCK": lambda args: triton.next_power_of_2(args["hidden_size"])}
)
@triton.jit
def _dynamic_int8_quant_kernel_azp_single_pass(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    azp_out_ptr,
    hidden_size,
    num_tokens,
    ROW_BLOCK: tl.constexpr,
    PLACEHOLDER_UNUSED: tl.constexpr = 1,
):
    """Asymmetric dynamic quant, single global read.

    Same structure as the symmetric single-pass kernel: the whole row stays
    in registers while min/max are reduced, then quantized in place.
    """
    pid = tl.program_id(0)
    row_offset = pid.to(tl.int64) * hidden_size
    offsets = tl.arange(0, ROW_BLOCK)
    mask = offsets < hidden_size
    src = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    row_max = tl.max(tl.where(mask, src, NEG_INF_VAL))
    row_min = tl.min(tl.where(mask, src, INF_VAL))
    span = row_max - row_min
    scale = span / 255.0
    # constant row (span == 0): inv_s is inf, and the downstream
    # round(-min*inf) saturates to int32 min/max exactly like the two-loop
    # reference kernel (matches vLLM CUDA behavior for constant rows).
    inv_s = 1.0 / scale
    azp = _round_i32_sat(-128.0 - row_min * inv_s)
    tl.store(scale_out_ptr + pid, scale)
    tl.store(azp_out_ptr + pid, azp)
    dst = _saturate_i32_to_i8(_round_i32_sat(src * inv_s) + azp)
    tl.store(output_ptr + row_offset + offsets, dst, mask=mask)


def _decompose_main_tail(hidden_size):
    """Largest-pow2 main block + pow2 tail block covering hidden_size.

    Returns (MAIN_BLOCK, TAIL_BLOCK) with MAIN_BLOCK + TAIL_BLOCK >= hidden_size
    and MAIN_BLOCK <= hidden_size, or None when the row is an exact power of
    two (plain single-pass is already mask-free) or the tail would be too
    large to keep register pressure bounded.
    """
    main = 1 << (hidden_size.bit_length() - 1)  # largest pow2 <= hidden_size
    rem = hidden_size - main
    if rem == 0:
        return None
    tail = triton.next_power_of_2(rem)
    if tail > 4096:
        return None
    return main, tail


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

# Dynamic symmetric rows up to this size are quantized with a single global
# read (whole row held in registers). Above it the two-loop kernel is used.
_SINGLE_PASS_MAX_HIDDEN = 4096

# Non-pow2 rows in (single-pass max, this] go through the main+tail kernel
# (unmasked pow2 main block + masked pow2 tail). Pow2 rows up to
# _SINGLE_PASS_MAX_HIDDEN_WIDE use the plain single-pass kernel.
_MAIN_TAIL_MAX_HIDDEN = 16384
_SINGLE_PASS_MAX_HIDDEN_WIDE = 16384


def _token_bucket(num_tokens):
    """Bucket token counts so the autotune cache stays small on decode-style
    workloads where num_tokens changes every step."""
    if num_tokens <= 8:
        return num_tokens
    if num_tokens <= 128:
        return 16 * ((num_tokens + 15) // 16)
    if num_tokens <= 1024:
        return 128 * ((num_tokens + 127) // 128)
    return 1024 * ((num_tokens + 1023) // 1024)


def scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    input_2d = input.reshape(-1, input.shape[-1])
    num_tokens, hidden_size = input_2d.shape
    token_bucket = _token_bucket(num_tokens)

    if scale is not None:
        if not symmetric and azp is None:
            raise ValueError("azp must be provided for asymmetric static quantization")
        output = torch.empty_like(input_2d, dtype=torch.int8)
        azp_or_dummy = azp if azp is not None else scale  # unused in this path

        if num_tokens < _2D_GRID_TOKEN_THRESHOLD:
            grid = lambda META: (  # noqa: E731
                num_tokens,
                triton.cdiv(hidden_size, META["BLOCK_SIZE"]),
            )
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
        azp_out_or_dummy = input_scales  # pointer unused in this path
    else:
        input_azp = torch.empty((num_tokens, 1), device=input.device, dtype=torch.int32)
        azp_out_or_dummy = input_azp

    if symmetric:
        # Single global read paths. The whole row stays in registers across
        # the absmax and quantize steps, halving DRAM traffic vs the generic
        # two-loop kernel. Single-pass pads the row to one pow2 block (cheap
        # when hidden_size is a power of two or nearly so); main+tail splits
        # it into an unmasked pow2 main block plus a masked pow2 tail, which
        # wins for sizes like 5120 = 4096 + 1024.
        main_tail = (
            _decompose_main_tail(hidden_size)
            if _SINGLE_PASS_MAX_HIDDEN < hidden_size <= _MAIN_TAIL_MAX_HIDDEN
            else None
        )
        if hidden_size <= _SINGLE_PASS_MAX_HIDDEN:
            _dynamic_int8_quant_kernel_single_pass[(num_tokens,)](
                input_2d,
                output,
                input_scales,
                hidden_size,
                token_bucket,
            )
        elif main_tail is not None:
            main_block, tail_block = main_tail
            _dynamic_int8_quant_kernel_main_tail[(num_tokens,)](
                input_2d,
                output,
                input_scales,
                hidden_size,
                token_bucket,
                MAIN_BLOCK=main_block,
                TAIL_BLOCK=tail_block,
            )
        elif hidden_size <= _SINGLE_PASS_MAX_HIDDEN_WIDE:
            _dynamic_int8_quant_kernel_single_pass[(num_tokens,)](
                input_2d,
                output,
                input_scales,
                hidden_size,
                token_bucket,
            )
        else:
            _dynamic_int8_quant_kernel[(num_tokens,)](
                input_2d,
                output,
                input_scales,
                azp_out_or_dummy,
                hidden_size,
                SYMMETRIC=symmetric,
            )
    else:
        if hidden_size <= _SINGLE_PASS_MAX_HIDDEN:
            _dynamic_int8_quant_kernel_azp_single_pass[(num_tokens,)](
                input_2d,
                output,
                input_scales,
                input_azp,
                hidden_size,
                token_bucket,
            )
        else:
            _dynamic_int8_quant_kernel[(num_tokens,)](
                input_2d,
                output,
                input_scales,
                azp_out_or_dummy,
                hidden_size,
                SYMMETRIC=symmetric,
            )
    return output.view(input.shape), input_scales, input_azp
