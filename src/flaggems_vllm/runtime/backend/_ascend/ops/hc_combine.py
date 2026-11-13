# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Canonical ascend HC combine entry point."""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.qwen4.hc_combine import (
    _hc_combine_kernel,
    _hc_combine_vectorized,
)
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils.libentry import major_version, minor_version

logger = logging.getLogger(__name__)

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CONFIGS = runtime.get_tuned_config("hc_combine")
_FIXED_CONFIG = (
    _CONFIGS[0] if len(_CONFIGS) == 1 and not _CONFIGS[0].kwargs["LOOP"] else None
)
_COMPILED_LAUNCH_HAS_CONSTEXPRS = major_version == 3 and 3 <= minor_version <= 6


@libentry()
@triton.jit(do_not_specialize=["D", "SR", "SB", "SI"])
def _hc_combine_runtime_shape(
    R,
    B,
    I,
    O,
    D,
    SR,
    SB,
    SI,
    HC: tl.constexpr,
    HAS_INJECTION: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    block = tl.load(B + row * SB + d, d < D, other=0).to(tl.float32)
    _hc_combine_vectorized(R, I, O, block, row, d, D, SR, SI, HC, HAS_INJECTION)


def _launch_hc_combine(
    residual, block_output, injection_logits, out, n, d, hc_count, sr, sb, si
):
    has_injection = injection_logits is not None
    if _FIXED_CONFIG is None:
        _hc_combine_kernel[lambda meta: (n, triton.cdiv(d, meta["BLOCK_D"]))](
            R=residual,
            B=block_output,
            I=injection_logits,
            O=out,
            SHAPE=(n, d, hc_count, sr, sb, si, has_injection),
        )
    else:
        block_d = _FIXED_CONFIG.kwargs["BLOCK_D"]
        grid = (n, triton.cdiv(d, block_d), 1)
        pointers = (residual, block_output, injection_logits, out)
        dimensions = (d, sr, sb, si)
        constants = [hc_count, has_injection, block_d]
        kernel = _hc_combine_runtime_shape
        if _COMPILED_LAUNCH_HAS_CONSTEXPRS:
            # This fixed launch has no tuner or hooks. Reuse LibEntry's cache
            # and key, skipping its general argument binder on warm calls.
            device = runtime.torch_device_fn.current_device()
            entry = kernel.kernel_cache[device].get(
                kernel.key(pointers, dimensions, constants)
            )
            if entry is not None:
                entry[0][grid](*pointers, *dimensions, *constants)
                return
        kernel[grid](
            *pointers,
            *dimensions,
            *constants,
            num_warps=_FIXED_CONFIG.num_warps,
            num_stages=_FIXED_CONFIG.num_stages,
        )


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
    if not n or not width:
        return out
    _launch_hc_combine(
        residual, block_output, injection_logits, out, n, d, hc_count, sr, sb, si
    )
    return out
