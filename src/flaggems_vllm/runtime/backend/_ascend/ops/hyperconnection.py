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

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.qwen4.hc_combine import _combine, _hc_combine_kernel

logger = logging.getLogger(__name__)

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# A uniform D tile permits a concrete grid, avoiding a metadata merge per call.
# Derive it from the canonical configs; retain dynamic grids for mixed tiles.
_BLOCK_D_VALUES = {
    config.kwargs["BLOCK_D"] for config in runtime.get_tuned_config("hc_combine")
}
_GRID_BLOCK_D = next(iter(_BLOCK_D_VALUES)) if len(_BLOCK_D_VALUES) == 1 else None


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


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    # Keep metadata validation in this public entry to avoid a second host call.
    logger.debug("GEMS HC_COMBINE")
    if isinstance(hc_count, bool) or not isinstance(hc_count, int) or hc_count <= 0:
        raise ValueError("hc_combine requires a positive integer hc_count")
    shape = residual.shape
    if len(shape) != 2:
        raise ValueError("hc_combine requires rank-2 residual")
    n, width = shape
    if width % hc_count:
        raise ValueError("hc_combine residual width must be divisible by hc_count")
    d = width // hc_count
    if block_output.shape != (n, d):
        raise ValueError("hc_combine block_output shape does not match residual")
    device = residual.device
    if device != block_output.device:
        raise ValueError("hc_combine requires same-device tensors")
    if residual.dtype not in _DTYPES or block_output.dtype not in _DTYPES:
        raise TypeError("hc_combine supports float16, bfloat16 and float32 inputs")
    sr, ir = residual.stride()
    sb, ib = block_output.stride()
    if ir != 1 or ib != 1:
        raise ValueError("hc_combine requires unit inner stride")
    has_injection = injection_logits is not None
    si = 0
    if has_injection:
        if injection_logits.shape != (n, hc_count):
            raise ValueError(
                "hc_combine injection_logits shape does not match residual"
            )
        if injection_logits.device != device:
            raise ValueError("hc_combine requires same-device tensors")
        if injection_logits.dtype not in _DTYPES:
            raise TypeError("hc_combine supports float16, bfloat16 and float32 inputs")
        si, ii = injection_logits.stride()
        if ii != 1:
            raise ValueError("hc_combine requires unit inner stride")
    if device.type in ("cpu", "meta"):
        raise NotImplementedError("hc_combine requires accelerator tensors")
    out = residual.new_empty(shape)
    if n and width:
        grid = (
            (n, (d + _GRID_BLOCK_D - 1) // _GRID_BLOCK_D)
            if _GRID_BLOCK_D is not None
            else lambda meta: (n, (d + meta["BLOCK_D"] - 1) // meta["BLOCK_D"])
        )
        _hc_combine_kernel[grid](
            R=residual,
            B=block_output,
            I=injection_logits,
            O=out,
            SHAPE=(n, d, hc_count, sr, sb, si, has_injection),
        )
    return out


def qwen4_hc_inject_combine(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    return _combine(residual, block_output, injection_logits, hc_count, legacy=True)
