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

"""thead PPU specialization of scaled_int8_quant.

Reuses the generic dynamic-quantization kernels (single-pass / main+tail /
two-loop) and replaces only the static path with a flat elementwise kernel:
one block handles BLOCK_SIZE contiguous elements over the whole tensor, with
no row bookkeeping. Measured on PPU-ZW810E vs the generic row-structured
static kernels: ~15% faster at 64x8192, ~1.6x at 64x13824 (which previously
mis-picked an oversized BLOCK_SIZE), equal elsewhere.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.scaled_int8_quant import (
    _MAIN_TAIL_MAX_HIDDEN,
    _SINGLE_PASS_MAX_HIDDEN,
    _SINGLE_PASS_MAX_HIDDEN_WIDE,
    _decompose_main_tail,
    _dynamic_int8_quant_kernel,
    _dynamic_int8_quant_kernel_azp_single_pass,
    _dynamic_int8_quant_kernel_main_tail,
    _dynamic_int8_quant_kernel_single_pass,
    _round_i8_sat,
    _round_i32_sat,
    _saturate_i32_to_i8,
    _token_bucket,
)
from flaggems_vllm.utils import libentry


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("scaled_int8_quant_static_flat"),
    key=["numel"],
)
@triton.jit
def _static_int8_quant_kernel_flat(
    input_ptr,
    output_ptr,
    scale_ptr,
    azp_ptr,
    numel,
    SYMMETRIC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Static quant as flat elementwise over numel (no row structure).

    One block handles BLOCK_SIZE contiguous elements; the scalar scale/azp
    are loaded once per block.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    scale = tl.load(scale_ptr)
    inv_s = 1.0 / scale

    if not SYMMETRIC:
        azp = tl.load(azp_ptr)

    src = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if SYMMETRIC:
        dst = _round_i8_sat(src * inv_s)
    else:
        dst = _saturate_i32_to_i8(_round_i32_sat(src * inv_s) + azp)

    tl.store(output_ptr + offsets, dst, mask=mask)


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

        numel = input_2d.numel()
        grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)  # noqa: E731
        _static_int8_quant_kernel_flat[grid](
            input_2d,
            output,
            scale,
            azp_or_dummy,
            numel,
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
