# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Pure Qwen4 HC combine. No normalization, input mutation, or PDL."""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_combine"),
    # SHAPE contains N, D, HC, row strides, and the optional-injection branch.
    # String conversion keeps this full tuple compatible with persistent keys.
    key=["SHAPE"],
    strategy=str,
    warmup=5,
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_combine_kernel(
    R,
    B,
    I,
    O,
    SHAPE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOOP: tl.constexpr,
):
    D: tl.constexpr = SHAPE[1]
    HC: tl.constexpr = SHAPE[2]
    SR: tl.constexpr = SHAPE[3]
    SB: tl.constexpr = SHAPE[4]
    SI: tl.constexpr = SHAPE[5]
    HAS_INJECTION: tl.constexpr = SHAPE[6]
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    block = tl.load(B + row * SB + d, d < D, other=0).to(tl.float32)
    if LOOP:
        for h in tl.static_range(HC):
            res = tl.load(R + row * SR + h * D + d, d < D, other=0).to(tl.float32)
            if HAS_INJECTION:
                inj = tl.load(I + row * SI + h).to(tl.float32)
                value = block * (2.0 * tl.sigmoid(inj / HC))
            else:
                value = block
            tl.store(O + row * HC * D + h * D + d, res + value, d < D)
    else:
        h = tl.arange(0, triton.next_power_of_2(HC))
        offsets = h[:, None] * D + d[None, :]
        mask = (h[:, None] < HC) & (d[None, :] < D)
        res = tl.load(R + row * SR + offsets, mask, other=0).to(tl.float32)
        if HAS_INJECTION:
            inj = tl.load(I + row * SI + h, h < HC, other=0).to(tl.float32)
            value = block[None, :] * (2.0 * tl.sigmoid(inj / HC))[:, None]
        else:
            value = block[None, :]
        tl.store(O + row * HC * D + offsets, res + value, mask)


def _combine(residual, block_output, injection_logits, hc_count, *, legacy=False):
    if isinstance(hc_count, bool) or not isinstance(hc_count, int) or hc_count <= 0:
        raise ValueError("hc_combine requires a positive integer hc_count")
    shape = residual.shape
    rank = len(shape)
    if rank < 2 or (not legacy and rank != 2):
        raise ValueError("hc_combine requires rank-2 residual")
    width = shape[-1]
    if width % hc_count:
        raise ValueError("hc_combine residual width must be divisible by hc_count")
    d = width // hc_count
    if block_output.shape != (*shape[:-1], d):
        raise ValueError("hc_combine block_output shape does not match residual")
    device = residual.device
    if device != block_output.device:
        raise ValueError("hc_combine requires same-device tensors")
    if residual.dtype not in _FLOAT_DTYPES or block_output.dtype not in _FLOAT_DTYPES:
        raise TypeError("hc_combine supports float16, bfloat16 and float32 inputs")
    rs, bs = residual.stride(), block_output.stride()
    if rs[-1] != 1 or bs[-1] != 1:
        raise ValueError("hc_combine requires unit inner stride")
    has_injection = injection_logits is not None
    si = 0
    if has_injection:
        if injection_logits.shape != (*shape[:-1], hc_count):
            raise ValueError(
                "hc_combine injection_logits shape does not match residual"
            )
        if injection_logits.device != device:
            raise ValueError("hc_combine requires same-device tensors")
        if injection_logits.dtype not in _FLOAT_DTYPES:
            raise TypeError("hc_combine supports float16, bfloat16 and float32 inputs")
        istride = injection_logits.stride()
        if istride[-1] != 1:
            raise ValueError("hc_combine requires unit inner stride")
        si = istride[-2]
    if rank > 2:
        # The legacy API permits flattenable leading dimensions. Check these
        # only on that path; canonical inputs always have exactly two axes.
        tensors = (
            (residual, block_output)
            if not has_injection
            else (residual, block_output, injection_logits)
        )
        for tensor in tensors:
            strides = tensor.stride()
            if any(
                strides[i] != tensor.shape[i + 1] * strides[i + 1]
                for i in range(rank - 2)
            ):
                raise ValueError("hc_combine leading dimensions must be flattenable")
    if device.type in ("cpu", "meta"):
        raise NotImplementedError("hc_combine requires accelerator tensors")
    output = residual.new_empty(shape)
    if residual.numel() == 0:
        return output
    n = shape[0] if rank == 2 else residual.numel() // width
    _hc_combine_kernel[lambda meta: (n, (d + meta["BLOCK_D"] - 1) // meta["BLOCK_D"])](
        R=residual,
        B=block_output,
        I=injection_logits,
        O=output,
        SHAPE=(n, d, hc_count, rs[-2], bs[-2], si, has_injection),
    )
    return output


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    """Return residual + HC-broadcast block with optional FP32 sigmoid gating.

    Inputs independently support FP16/BF16/FP32, unit inner stride, and any
    nonnegative row stride (including packed, overlapping, and expanded rows).
    The fresh contiguous output has residual's shape and dtype.
    """
    logger.debug("GEMS HC_COMBINE")
    return _combine(residual, block_output, injection_logits, hc_count)
