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
    Slots,
    Q,
    S,
    PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    NSLOT: tl.constexpr,
):
    row = tl.program_id(0)
    dst = tl.load(Slots + row).to(tl.int64)
    if (dst >= 0) & (dst < NSLOT):
        page = dst // PAGE
        slot = dst % PAGE
        d = tl.arange(0, 128)
        byte = tl.load(Q + row * 128 + d)
        tl.store(C + page * PAGE_STRIDE + slot * 128 + d, byte)
        bits = tl.load(S + row).to(tl.uint32, bitcast=True)
        b = tl.arange(0, 4)
        tl.store(
            C + page * PAGE_STRIDE + PAGE * 128 + slot * 4 + b,
            (bits >> (b * 8)).to(tl.uint8),
        )


def hy4_flash_write_pool(cache, slots, quant, scales):
    """Update distinct slots in place; negative or out-of-bounds slots do not write.

    Slots encode page * page_size + row. Each byte page stores an E4M3FN
    plane followed by its FP32 scale plane.
    """
    if runtime.device.vendor_name != "nvidia" or cache.device.type != "cuda":
        raise NotImplementedError("Flash pool writes require NVIDIA CUDA")
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[1] == 0
        or cache.shape[-1] != 132
        or cache.stride(2) != 1
        or cache.stride(1) != 132
        or cache.stride(0) < cache.shape[1] * 132
    ):
        raise ValueError(
            f"Invalid Flash byte-page layout: dtype={cache.dtype}, shape={tuple(cache.shape)}, strides={cache.stride()}"
        )
    n = slots.numel()
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
        slots.shape != (n,)
        or slots.dtype not in (torch.int32, torch.int64)
        or not slots.is_contiguous()
    ):
        raise ValueError("Invalid pool write metadata")
    if any(x.device != cache.device for x in (slots, quant, scales)):
        raise ValueError("Pool write tensors must share a device")
    if n:
        _flash_write[(n,)](
            cache,
            slots,
            quant.view(torch.uint8),
            scales,
            cache.shape[1],
            cache.stride(0),
            cache.shape[0] * cache.shape[1],
            num_warps=4,
        )
