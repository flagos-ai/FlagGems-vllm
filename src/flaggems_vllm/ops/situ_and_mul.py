# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BF16 SiTU: FP32 bounded gate/sigmoid and optional bounded up projection."""
import math

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import tl_extra_shim

_tanh = tl_extra_shim.tanh
_exp2 = tl_extra_shim.exp2


@triton.jit
def _situ_and_mul_kernel(
    GATE_UP,
    ACTIVATED,
    ELEMENTS: tl.constexpr,
    I: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // I
    col = offsets % I
    gate = tl.load(GATE_UP + row * (2 * I) + col, offsets < ELEMENTS, 0).to(tl.float32)
    up = tl.load(GATE_UP + row * (2 * I) + I + col, offsets < ELEMENTS, 0).to(
        tl.float32
    )
    # Separate expressions preserve the materialization order of the reference.
    sigmoid = 1.0 / (1.0 + _exp2(-gate * 1.4426950408889634))
    gate = BETA * _tanh(gate / BETA) * sigmoid
    if LINEAR_BETA > 0:
        up = LINEAR_BETA * _tanh(up / LINEAR_BETA)
    # Match the registered FP32 multiply's exact-zero sign before the BF16
    # boundary. The +0 FMA does not merge any earlier SiTU arithmetic stage.
    result = tl.fma(gate, up, 0.0).to(tl.bfloat16)
    tl.store(ACTIVATED + offsets, result, offsets < ELEMENTS)


def situ_and_mul(
    out: torch.Tensor,
    input: torch.Tensor,
    beta: float = 1.0,
    linear_beta: float = -1.0,
) -> None:
    """Write SiTU(input[:,:I]) * bounded(input[:,I:]) into a separate BF16 out.

    The sensitive arithmetic and sigmoid are FP32; the final product uses the
    accepted positive-zero FMA convention before BF16 materialization.
    Negative linear_beta leaves the up branch unbounded. This implementation
    is qualified on MetaX; it does not fall back to Torch computation.
    """
    if runtime.device.vendor_name != "metax":
        raise NotImplementedError("This SiTU implementation is qualified on MetaX")
    if input.ndim != 2 or input.shape[1] == 0 or input.shape[1] % 2:
        raise ValueError("SiTU input must be [M, 2*I] with I > 0")
    if input.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise NotImplementedError("SiTU currently requires BF16 input and output")
    if (
        not input.is_cuda
        or out.device != input.device
        or not input.is_contiguous()
        or not out.is_contiguous()
        or tuple(out.shape) != (input.shape[0], input.shape[1] // 2)
    ):
        raise ValueError("SiTU requires matching contiguous input/output on one device")
    if not isinstance(beta, (int, float)) or not isinstance(linear_beta, (int, float)):
        raise TypeError("SiTU parameters must be Python scalars")
    if input.numel() >= 2**31:
        raise NotImplementedError(
            "SiTU input exceeds the supported address-index range"
        )
    if not math.isfinite(beta) or beta <= 0 or not math.isfinite(linear_beta):
        raise ValueError("SiTU beta must be positive and both parameters finite")
    if out.numel() == 0:
        return
    if out.untyped_storage().data_ptr() == input.untyped_storage().data_ptr():
        raise ValueError("SiTU output must not share storage with the input")
    intermediate = input.shape[1] // 2
    # Retain the accepted MetaX pointwise geometry; no global tuner changes.
    _situ_and_mul_kernel[(triton.cdiv(out.numel(), 256),)](
        input,
        out,
        out.numel(),
        intermediate,
        beta,
        linear_beta,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
