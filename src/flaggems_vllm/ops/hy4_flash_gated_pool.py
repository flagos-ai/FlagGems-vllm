# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Pool IndexPool rows with per-channel softmax weights and FP32 RMS normalization."""

import torch
import triton
import triton.language as tl

from .hy4_flash_fwht_quant_fp8 import _check_tensor


@triton.jit
def _flash_gated_pool(V, G, A, W, Output, R: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    r = tl.arange(0, R)
    d = tl.arange(0, 128)
    off = row * R * 128 + r[:, None] * 128 + d[None, :]
    logits = tl.load(G + off).to(tl.float32) + tl.load(
        A + r[:, None] * 128 + d[None, :]
    ).to(tl.float32)
    exp = tl.exp(logits - tl.max(logits, 0)[None, :])
    prob = exp / tl.sum(exp, 0)[None, :]
    pooled = tl.sum(tl.load(V + off).to(tl.float32) * prob, 0)
    rms = tl.rsqrt(tl.sum(pooled * pooled, 0) / 128 + EPS)
    weight = tl.load(W + d).to(tl.float32)
    tl.store(Output + row * 128 + d, pooled * rms * weight)


def hy4_flash_gated_pool(values, gates, ape, norm_weight, eps):
    """Softmax across ratio, weighted pool and FP32 RMSNorm over 128 columns."""
    for x, name in [
        (values, "values"),
        (gates, "gates"),
        (ape, "ape"),
        (norm_weight, "norm_weight"),
    ]:
        _check_tensor(x, name)
        if x.device != values.device:
            raise ValueError("All IndexPool tensors must share a device")
    if values.ndim != 3 or values.shape[-1] != 128 or gates.shape != values.shape:
        raise ValueError("values and gates must have equal [rows,ratio,128] shape")
    n, ratio, _ = values.shape
    if (
        ratio not in (2, 4, 8, 16)
        or ape.shape != (ratio, 128)
        or norm_weight.shape != (128,)
    ):
        raise ValueError("Invalid Flash gated-pool shape")
    if eps <= 0:
        raise ValueError("eps must be positive")
    out = torch.empty((n, 128), dtype=torch.float32, device=values.device)
    if n:
        _flash_gated_pool[(n,)](
            values,
            gates,
            ape,
            norm_weight,
            out,
            ratio,
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
