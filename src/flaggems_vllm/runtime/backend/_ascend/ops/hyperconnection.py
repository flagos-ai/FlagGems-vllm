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

"""Ascend-optimized Qwen4 HyperConnection inject-combine.

[KernelGen] Auto-generated and tuned for Ascend910.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_inject_combine_kernel(
    injection_logits_ptr,
    block_output_ptr,
    residual_ptr,
    output_ptr,
    stride_logits_row,
    stride_block_row,
    stride_residual_row,
    stride_output_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_size
    block_output = tl.load(
        block_output_ptr + row * stride_block_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    logits_base = injection_logits_ptr + row * stride_logits_row
    residual_base = residual_ptr + row * stride_residual_row
    output_base = output_ptr + row * stride_output_row
    for branch in tl.static_range(0, hc_count):
        logits = tl.load(logits_base + branch).to(tl.float32)
        injection_weight = 2.0 * tl.sigmoid(logits / hc_count)
        branch_offsets = branch * hidden_size + offsets
        residual = tl.load(
            residual_base + branch_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            output_base + branch_offsets,
            residual + block_output * injection_weight,
            mask=mask,
        )


def qwen4_hc_inject_combine(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if injection_logits.shape != (*block_output.shape[:-1], hc_count):
        raise ValueError("Qwen4 HC injection logits have an invalid shape")
    if hc_count <= 0:
        raise ValueError("Qwen4 HC injection requires a positive HC count")
    if residual.shape != (
        *block_output.shape[:-1],
        hc_count * block_output.shape[-1],
    ):
        raise ValueError("Qwen4 HC residual and block output shapes are incompatible")
    if any(
        t.device.type in ("cpu", "meta")
        or t.dtype not in (torch.bfloat16, torch.float16)
        for t in (injection_logits, block_output, residual)
    ):
        raise RuntimeError(
            "Qwen4 HC injection received an unsupported accelerator layout"
        )
    if len({t.device for t in (injection_logits, block_output, residual)}) != 1:
        raise RuntimeError("Qwen4 HC injection requires same-device tensors")

    output = torch.empty_like(residual)
    if not residual.numel():
        return output

    hidden_size = block_output.shape[-1]
    rows = block_output.numel() // hidden_size

    # Adaptive tile size for Ascend910
    if rows >= 16:
        block_h = triton.next_power_of_2(hidden_size)
        if block_h > 4096:
            block_h = 4096
    elif rows >= 4:
        block_h = 512 if hidden_size >= 512 else 256
    else:
        block_h = 256 if hidden_size >= 256 else 128

    num_warps = 4

    _hc_inject_combine_kernel[(rows, triton.cdiv(hidden_size, block_h))](
        injection_logits,
        block_output,
        residual,
        output,
        injection_logits.stride(-2),
        block_output.stride(-2),
        residual.stride(-2),
        output.stride(-2),
        hidden_size=hidden_size,
        hc_count=hc_count,
        block_h=block_h,
        num_warps=num_warps,
    )
    return output
