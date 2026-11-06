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

"""MetaX-optimized Qwen4 HyperConnection inject-combine.

[KernelGen] Auto-generated and tuned for MetaX C-series.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.qwen4.hc_combine import _combine


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
    branch = tl.program_id(1)
    offsets = tl.program_id(2) * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_size
    branch_offsets = branch * hidden_size + offsets
    logits = tl.load(injection_logits_ptr + row * stride_logits_row + branch).to(
        tl.float32
    )
    injection_weight = 2.0 * tl.sigmoid(logits / hc_count)
    block_output = tl.load(
        block_output_ptr + row * stride_block_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * stride_residual_row + branch_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + branch_offsets,
        residual + block_output * injection_weight,
        mask=mask,
    )


@triton.jit
def _hc_inject_combine_fused_kernel(
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
    offsets = tl.program_id(1) * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_size
    block_output = tl.load(
        block_output_ptr + row * stride_block_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    for branch in tl.static_range(0, hc_count):
        logits = tl.load(injection_logits_ptr + row * stride_logits_row + branch).to(
            tl.float32
        )
        injection_weight = 2.0 * tl.sigmoid(logits / hc_count)
        branch_offsets = branch * hidden_size + offsets
        residual = tl.load(
            residual_ptr + row * stride_residual_row + branch_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            output_ptr + row * stride_output_row + branch_offsets,
            residual + block_output * injection_weight,
            mask=mask,
        )


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    return _combine(residual, block_output, injection_logits, hc_count)


def qwen4_hc_inject_combine(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    return _combine(residual, block_output, injection_logits, hc_count, legacy=True)
