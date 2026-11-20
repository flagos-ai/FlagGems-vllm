# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Dequantize and normalize already looked-up Hy4 Flash Engram rows."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

_ENGRAM_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_FP32_EPS = tl.constexpr(1.1920928955078125e-07)


def _check_engram_tensors(anchor, *tensors):
    if anchor.device.type != "cuda" or runtime.device.vendor_name != "nvidia":
        raise NotImplementedError("Hy4 Flash Engram requires NVIDIA CUDA tensors")
    if any(x.device != anchor.device for x in tensors):
        raise ValueError("Engram tensors must share a device")


@libentry()
@libtuner(configs=runtime.get_tuned_config("hy4_flash_engram_row"), key=["N"])
@triton.jit
def _lookup_norm(
    Rows,
    Scale,
    Gamma,
    Hashes,
    Out,
    H: tl.constexpr,
    N: tl.constexpr,
    HASH_S0: tl.constexpr,
    HASH_S1: tl.constexpr,
    FP8: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    token, head = i // H, i % H
    col = tl.arange(0, BLOCK)
    in_dim = col < N
    valid = tl.load(Hashes + token * HASH_S0 + head * HASH_S1) >= 0
    x = tl.load(Rows + i * N + col, in_dim & valid, other=0.0).to(tl.float32)
    if FP8:
        scale = tl.load(Scale + i, valid, other=0).to(tl.float32)
        x = x * scale
    rms = tl.rsqrt(tl.sum(x * x, 0) / N + _FP32_EPS)
    gamma = tl.load(Gamma + head * N + col, in_dim, other=0).to(tl.float32)
    tl.store(Out + i * N + col, (x * rms) * gamma, in_dim)


def hy4_flash_engram_lookup_norm(rows, row_scale, gamma, local_hashes, out_dtype):
    """Normalize [tokens, heads, width] rows; negative hashes mask invalid rows.

    BF16 rows require no scale. FP8 E4M3FN rows require one BF16 scale per row.
    The per-head gamma has shape [heads, width], and the hashes may be strided.
    """
    _check_engram_tensors(
        rows, gamma, local_hashes, *(() if row_scale is None else (row_scale,))
    )
    if rows.ndim != 3 or not rows.is_contiguous():
        raise ValueError("Engram rows must be contiguous [tokens, heads, width]")
    m, h, n = rows.shape
    if not 0 < n <= 8192 or h == 0:
        raise ValueError("Engram normalization width must be in [1, 8192]")
    if (
        gamma.shape != (h, n)
        or gamma.dtype not in _ENGRAM_DTYPES
        or not gamma.is_contiguous()
        or local_hashes.shape != (m, h)
        or local_hashes.dtype not in (torch.int32, torch.int64)
        or out_dtype not in _ENGRAM_DTYPES
    ):
        raise ValueError("Invalid Engram lookup, gamma or output contract")
    fp8 = rows.dtype == torch.float8_e4m3fn
    if fp8:
        if (
            row_scale is None
            or row_scale.shape != (m, h)
            or row_scale.dtype != torch.bfloat16
            or not row_scale.is_contiguous()
        ):
            raise ValueError(
                "Engram FP8 row scales must be contiguous BF16 [tokens, heads]"
            )
    elif rows.dtype != torch.bfloat16 or row_scale is not None:
        raise ValueError("Engram unquantized rows must be BF16 without scales")

    out = torch.empty(rows.shape, dtype=out_dtype, device=rows.device)
    if m:
        _lookup_norm[(m * h,)](
            rows,
            row_scale if fp8 else rows,
            gamma,
            local_hashes,
            out,
            h,
            n,
            *local_hashes.stride(),
            fp8,
            triton.next_power_of_2(n),
            enable_fp_fusion=False,
        )
    return out
