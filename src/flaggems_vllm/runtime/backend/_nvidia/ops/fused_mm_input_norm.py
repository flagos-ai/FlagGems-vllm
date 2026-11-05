# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-channel affine normalization of contiguous vision patches.

The caller owns input preparation, output storage, and cross-call/graph lifetime.
The fixed BLOCK1024/four-warp launch preserves the selected implementation;
this migration does not retune its arithmetic or reduction schedule.
"""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fused_mm_input_norm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    numel,
    L,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 1D grid over the flattened (N, C, L) tensor. Each program processes
    # BLOCK contiguous elements; the channel index is recovered from the
    # flat offset via `(offs // L) % C`. int32 offsets: int64 division is
    # ~2x slower here (measured on GB200); the wrapper asserts
    # numel < 2**31 so the flat index never overflows.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    c = (offs // L) % C

    x = tl.load(
        x_ptr + offs,
        mask=mask,
        other=0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    w = tl.load(w_ptr + c, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + c, mask=mask, other=0).to(tl.float32)
    tl.store(
        y_ptr + offs,
        x * w + b,
        mask=mask,
        eviction_policy="evict_first",
    )


def fused_mm_input_norm(inputs, outputs, weight, bias):
    """Write FP32-computed channel affine values into the caller's output.

    Inputs and outputs are contiguous ``[N, C, L]`` GPU tensors. Inputs may be
    UINT8, FP16, BF16 or FP32; outputs may be FP16, BF16 or FP32. Weight and bias
    are contiguous FP32 vectors of length C. No input is modified. Outputs may
    alias inputs only when the full contiguous view and dtype are identical;
    parameters must not overlap the output. Empty inputs launch no kernel.
    """
    logger.debug("GEMS_NVIDIA FUSED_MM_INPUT_NORM")
    tensors = (inputs, outputs, weight, bias)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("normalization inputs and buffers must be tensors")
    if inputs.ndim != 3 or outputs.shape != inputs.shape:
        raise ValueError("normalization input/output must share shape [N, C, L]")
    n, channels, length = inputs.shape
    if channels <= 0 or length <= 0 or n * channels * length >= 2**31:
        raise ValueError(
            "positive channel/patch widths and fewer than 2**31 elements required"
        )
    if (
        inputs.device.type != "cuda"
        or torch.version.hip is not None
        or any(tensor.device != inputs.device for tensor in tensors)
    ):
        raise NotImplementedError("normalization requires one NVIDIA CUDA device")
    if inputs.dtype not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("unsupported normalization input dtype")
    if outputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("unsupported normalization output dtype")
    if (
        weight.shape != (channels,)
        or bias.shape != (channels,)
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
    ):
        raise ValueError("normalization weight/bias must be FP32 vectors of length C")
    if any(
        t.layout != torch.strided
        or not t.is_contiguous()
        or t.is_conj()
        or t.is_neg()
        or t.requires_grad
        for t in tensors
    ):
        raise ValueError("contiguous strided inference tensors required")
    if any(torch._C._overlaps(outputs, param) for param in (weight, bias)):
        raise ValueError("normalization output must not overlap parameters")
    if torch._C._overlaps(outputs, inputs) and (
        outputs.data_ptr() != inputs.data_ptr() or outputs.dtype != inputs.dtype
    ):
        raise ValueError("normalization input/output alias must be an identical view")
    if inputs.numel():
        with torch.cuda.device(inputs.device):
            _fused_mm_input_norm_kernel[(triton.cdiv(inputs.numel(), 1024),)](
                inputs,
                outputs,
                weight,
                bias,
                inputs.numel(),
                length,
                C=channels,
                BLOCK=1024,
                num_warps=4,
            )
    return outputs


__all__ = ["fused_mm_input_norm"]
