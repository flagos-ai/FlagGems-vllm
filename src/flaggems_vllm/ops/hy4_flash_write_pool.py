# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Write active E4M3FN rows and FP32 scales into paged IndexPool cache storage."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime


@triton.jit
def _flash_write(
    C,
    T,
    OWN,
    GROUP,
    ACTIVE,
    Q,
    S,
    PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    TS0: tl.constexpr,
    TS1: tl.constexpr,
    NOWNER: tl.constexpr,
    NCOL: tl.constexpr,
):
    row = tl.program_id(0)
    active = tl.load(ACTIVE + row)
    if active:
        owner = tl.load(OWN + row).to(tl.int64)
        owner = tl.where(owner < 0, owner + NOWNER, owner)
        group = tl.load(GROUP + row).to(tl.int64)
        column = tl.minimum(tl.maximum(group // PAGE, 0), NCOL - 1)
        page = tl.load(T + owner * TS0 + column * TS1).to(tl.int64)
        slot = group % PAGE
        d = tl.arange(0, 128)
        byte = tl.load(Q + row * 128 + d)
        tl.store(C + page * PAGE_STRIDE + slot * 128 + d, byte)
        bits = tl.load(S + row).to(tl.uint32, bitcast=True)
        b = tl.arange(0, 4)
        tl.store(
            C + page * PAGE_STRIDE + PAGE * 128 + slot * 4 + b,
            (bits >> (b * 8)).to(tl.uint8),
        )


def hy4_flash_write_pool(cache, table, owners, groups, active, quant, scales):
    """Update active rows in place. Live owner/group pairs must map to valid, distinct slots."""
    if runtime.device.vendor_name != "nvidia" or cache.device.type != "cuda":
        raise NotImplementedError("Flash pool writes require NVIDIA CUDA")
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[-1] != 132
        or cache.stride(2) != 1
        or cache.stride(1) != 132
        or cache.stride(0) < cache.shape[1] * 132
    ):
        raise ValueError(
            f"Invalid Flash byte-page layout: dtype={cache.dtype}, shape={tuple(cache.shape)}, strides={cache.stride()}"
        )
    n = groups.numel()
    if (
        quant.shape != (n, 128)
        or quant.dtype != torch.float8_e4m3fn
        or not quant.is_contiguous()
        or scales.shape != (n,)
        or scales.dtype != torch.float32
        or not scales.is_contiguous()
    ):
        raise ValueError("Invalid FP8 quantized pool data")
    if (
        table.ndim != 2
        or (n and (not table.shape[0] or not table.shape[1] or not cache.shape[1]))
        or any(
            x.dtype not in (torch.int32, torch.int64) for x in (table, owners, groups)
        )
        or active.dtype != torch.bool
        or any(
            x.shape != (n,) or not x.is_contiguous() for x in (owners, groups, active)
        )
    ):
        raise ValueError("Invalid pool write metadata")
    if any(
        x.device != cache.device for x in (table, owners, groups, active, quant, scales)
    ):
        raise ValueError("Pool write tensors must share a device")
    if n:
        _flash_write[(n,)](
            cache,
            table,
            owners,
            groups,
            active,
            quant.view(torch.uint8),
            scales,
            cache.shape[1],
            cache.stride(0),
            *table.stride(),
            *table.shape,
            num_warps=4,
        )
