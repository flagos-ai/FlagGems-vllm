# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Value normalization and residual update for Hy4 Flash Engram."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hy4_flash_engram_lookup_norm import (
    _ENGRAM_DTYPES,
    _FP32_EPS,
    _check_engram_tensors,
)
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(configs=runtime.get_tuned_config("hy4_flash_engram_row"), key=["N"])
@triton.jit
def _engram_residual(
    Hidden,
    Value,
    Gate,
    Gamma,
    Scales,
    Valid,
    Out,
    H: tl.constexpr,
    N: tl.constexpr,
    HS0: tl.constexpr,
    HS1: tl.constexpr,
    HS2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    in_dim = col < N
    valid = tl.load(Valid + row)
    value = tl.load(Value + row * N + col, in_dim & valid, other=0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(value * value, 0) / N + _FP32_EPS)
    gamma = tl.load(Gamma + col, in_dim, other=0).to(tl.float32)
    signal = (gamma * value) * rms
    for channel in range(H):
        g = tl.load(Gate + row * H + channel).to(tl.float32)
        scales = tl.load(Scales + channel * N + col, in_dim, other=0).to(tl.float32)
        result = ((g + scales) * signal).to(Hidden.dtype.element_ty)
        residual = tl.where(valid, result.to(tl.float32), 0)
        h = tl.load(Hidden + row * HS0 + channel * HS1 + col * HS2, in_dim, other=0).to(
            tl.float32
        )
        tl.store(Out + (row * H + channel) * N + col, h + residual, in_dim)


def hy4_flash_engram_residual(hidden, value, gate, gamma, scales, valid_rows):
    """Apply a gated, normalized Engram value to [tokens, channels, hidden]."""
    _check_engram_tensors(hidden, value, gate, gamma, scales, valid_rows)
    if hidden.ndim != 3 or hidden.dtype not in _ENGRAM_DTYPES:
        raise ValueError(
            "Engram hidden state must be floating [tokens, channels, hidden]"
        )
    m, h, n = hidden.shape
    if h == 0 or not 0 < n <= 8192:
        raise ValueError("Invalid Engram residual channel count or hidden width")
    inputs = (
        (value, (m, n)),
        (gate, (m, h)),
        (gamma, (n,)),
        (scales, (h, n)),
    )
    if (
        any(
            x.shape != shape or x.dtype not in _ENGRAM_DTYPES or not x.is_contiguous()
            for x, shape in inputs
        )
        or valid_rows.shape != (m,)
        or valid_rows.dtype != torch.bool
        or not valid_rows.is_contiguous()
    ):
        raise ValueError("Invalid Engram residual input or row mask")
    out = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    if m:
        _engram_residual[(m,)](
            hidden,
            value,
            gate,
            gamma,
            scales,
            valid_rows,
            out,
            h,
            n,
            *hidden.stride(),
            triton.next_power_of_2(n),
            enable_fp_fusion=False,
        )
    return out
