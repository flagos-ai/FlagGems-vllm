# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Apply RoPE to the trailing dimensions of 128-channel IndexPool rows."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime

from .hy4_flash_fwht_quant_fp8 import _check_tensor

if runtime.device.vendor_name == "nvidia":
    from triton.language.extra.cuda import libdevice


@triton.jit
def _flash_rotate(
    X,
    P,
    INV,
    Output,
    H: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    R: tl.constexpr,
    THETA: tl.constexpr,
    AXES: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    half: tl.constexpr = R // 2
    j = tl.maximum(d - (128 - R), 0) % half
    axis = tl.full((128,), 0, tl.int32)
    if AXES:
        if INTERLEAVED:
            axis = tl.where((j % 3 == 1) & (j < S1 * 3), 1, axis)
            axis = tl.where((j % 3 == 2) & (j < S2 * 3), 2, axis)
        else:
            axis = tl.where(j >= S0, 1, axis)
            axis = tl.where(j >= S0 + S1, 2, axis)
    pos = tl.load(P + axis * PS0 + token * PS1).to(tl.float32)
    inverse = tl.load(INV + j)
    angle = pos * inverse
    base = (token * H + head) * 128
    left = tl.load(X + base + 128 - R + 2 * j).to(tl.float32)
    right = tl.load(X + base + 128 - R + 2 * j + 1).to(tl.float32)
    c = libdevice.cos(angle)
    s = libdevice.sin(angle)
    rotated = tl.where(d < 128 - half, left * c - right * s, right * c + left * s)
    original = tl.load(X + base + d)
    tl.store(Output + base + d, tl.where(d < 128 - R, original, rotated))


def hy4_flash_rotate_indexer(x, positions, rope_dim, parameters, inverse_freq):
    """Rotate using the model's precomputed contiguous FP32 inverse frequencies."""
    _check_tensor(x, "x")
    if rope_dim == 0:
        return x
    if (
        x.ndim not in (2, 3)
        or x.shape[-1] != 128
        or rope_dim % 2
        or not 0 < rope_dim <= 128
    ):
        raise ValueError(
            "Flash RoPE requires [tokens,(heads),128] and an even rotary dimension"
        )
    if positions.device != x.device or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("RoPE positions must be integer and share the input device")
    if parameters.get("rope_type", "default") != "default":
        raise ValueError("Flash IndexPool requires default RoPE")
    axes = positions.ndim == 2
    section = parameters.get("mrope_section", (0, 0, 0))
    if axes and (
        positions.shape[0] != 3 or len(section) != 3 or sum(section) * 2 != rope_dim
    ):
        raise ValueError("Invalid three-axis MRoPE sections")
    if positions.ndim not in (1, 2) or positions.shape[-1] != x.shape[0]:
        raise ValueError("RoPE position count must match the input token count")
    if (
        not inverse_freq.is_contiguous()
        or inverse_freq.shape != (rope_dim // 2,)
        or inverse_freq.dtype != torch.float32
        or inverse_freq.device != x.device
    ):
        raise ValueError("Invalid Flash rotary coefficients")
    out = torch.empty_like(x)
    if x.shape[0]:
        _flash_rotate[(x.shape[0], x.shape[1] if x.ndim == 3 else 1)](
            x,
            positions,
            inverse_freq,
            out,
            x.shape[1] if x.ndim == 3 else 1,
            positions.stride(0) if axes else 0,
            positions.stride(-1),
            rope_dim,
            parameters.get("rope_theta", 10000.0),
            axes,
            *section,
            parameters.get("mrope_interleaved", True),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
