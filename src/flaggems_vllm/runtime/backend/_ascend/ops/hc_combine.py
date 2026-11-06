# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Canonical ascend HC combine entry point."""

from __future__ import annotations

import logging

import torch

from flaggems_vllm import runtime
from flaggems_vllm.ops.qwen4.hc_combine import _hc_combine_kernel

logger = logging.getLogger(__name__)

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK_D_VALUES = {
    config.kwargs["BLOCK_D"] for config in runtime.get_tuned_config("hc_combine")
}
_GRID_BLOCK_D = next(iter(_BLOCK_D_VALUES)) if len(_BLOCK_D_VALUES) == 1 else None


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    # Keep metadata validation in this public entry to avoid a second host call.
    logger.debug("GEMS_ASCEND HC_COMBINE")
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
