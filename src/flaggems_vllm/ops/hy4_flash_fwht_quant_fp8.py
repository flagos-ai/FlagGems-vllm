# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Hadamard-transform 128-channel rows and quantize with FP32 power-of-two scales."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.fp8_storage import encode_e4m3fn
from flaggems_vllm.ops.kpool import _hadamard128


@triton.jit
def _flash_fwht(X, Q, S):
    row = tl.program_id(0)
    d = tl.arange(0, 128)
    x = tl.load(X + row * 128 + d).to(tl.float32)
    x = _hadamard128(x)
    any_nan = tl.sum((x != x).to(tl.int32), 0) != 0
    a = tl.where(any_nan, float("nan"), tl.maximum(tl.max(tl.abs(x), 0), 1e-4))
    scale = tl.exp2(tl.ceil(tl.log2(a / 448.0)))
    y = x / scale
    y = tl.where(y != y, y, tl.minimum(tl.maximum(y, -448.0), 448.0))
    tl.store(Q + row * 128 + d, encode_e4m3fn(y))
    tl.store(S + row, scale)


def _check_tensor(x, name):
    if runtime.device.vendor_name != "nvidia" or x.device.type != "cuda":
        raise NotImplementedError("Hy4 Flash IndexPool is validated on NVIDIA CUDA")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"{name} must have FP32, FP16 or BF16 dtype")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def hy4_flash_fwht_quant_fp8(vectors):
    """[rows,128] -> (E4M3FN [rows,128], FP32 [rows]); no BF16 FWHT rounding."""
    _check_tensor(vectors, "vectors")
    if vectors.ndim != 2 or vectors.shape[1] != 128:
        raise ValueError("Flash FWHT requires [rows,128]")
    q = torch.empty(vectors.shape, dtype=torch.float8_e4m3fn, device=vectors.device)
    scale = torch.empty((vectors.shape[0],), dtype=torch.float32, device=vectors.device)
    if vectors.shape[0]:
        # Fixed native JIT: avoid the known LibEntry cache-hit incompatibility
        # in the pinned stock Triton 3.7 environment. One warp covers 128 lanes.
        _flash_fwht[(vectors.shape[0],)](
            vectors, q.view(torch.uint8), scale, num_warps=1, enable_fp_fusion=False
        )
    return q, scale
