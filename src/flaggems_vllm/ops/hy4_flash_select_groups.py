# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Select the highest-scoring live IndexPool groups and return ascending group IDs."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.cooperative_topk import (
    _key_greater,
    _masked_histogram,
    _ordered_key,
    _radix_digit,
)
from flaggems_vllm.utils import libentry


@triton.jit
def _flash_select(
    Logits,
    U,
    Output,
    S0: tl.constexpr,
    S1: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    upper = tl.load(U + row)
    value = tl.load(Logits + row * S0 + col * S1, col < W, other=-float("inf"))
    indices = tl.where(
        (col < W) & (col < upper) & (value != -float("inf")), col, 2147483647
    )
    indices = tl.sort(indices, descending=False)
    tl.store(
        Output + row * K + col, tl.where(indices == 2147483647, -1, indices), col < K
    )


@libentry()
@triton.jit
def _flash_select_single_cta(
    Logits,
    U,
    Output,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Exact radix threshold + ascending-ID compaction in one CTA. Reuse the
    # library's float ordering, including canonical NaNs and signed zero.
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, BLOCK)
    upper = tl.minimum(tl.maximum(tl.load(U + row), 0), WIDTH)
    x = tl.load(Logits + row * STRIDE + j, j < WIDTH, other=-float("inf"))
    valid = (j < upper) & (j < WIDTH) & (x != -float("inf"))
    count = tl.sum(valid.to(tl.int32), 0)
    selected = valid
    if count > K:
        keys = _ordered_key(x)
        prefix = tl.full((), 0, tl.uint32)
        remaining = tl.full((), K, tl.int32)
        candidates = valid
        for shift in tl.static_range(24, -1, -8):
            digits = ((keys >> shift) & 255).to(tl.int32)
            histogram = _masked_histogram(digits, candidates, 256)
            digit, remaining = _radix_digit(histogram, remaining)
            prefix = prefix | (digit << shift)
            candidates = candidates & (digits == digit)
        greater = valid & _key_greater(keys, prefix)
        equal = valid & (keys == prefix)
        rank = tl.cumsum(equal.to(tl.int32), 0)
        selected = greater | (equal & (rank <= remaining))
    offset = tl.cumsum(selected.to(tl.int32), 0) - 1
    tl.store(Output + row * K + offset, j, selected)
    nselected = tl.sum(selected.to(tl.int32), 0)
    tail = nselected + tl.arange(0, triton.next_power_of_2(K))
    tl.store(Output + row * K + tail, -1, tail < K)


def hy4_flash_select_groups(logits, upper, budget):
    """Exact score selection, ascending IDs, -1 padding; inputs remain read-only."""
    if runtime.device.vendor_name != "nvidia" or logits.device.type != "cuda":
        raise NotImplementedError("Flash selector requires NVIDIA CUDA")
    if (
        logits.ndim != 2
        or logits.dtype != torch.float32
        or any(s < 0 for s in logits.stride())
    ):
        raise ValueError("Flash selector requires strided FP32 matrix input")
    if (
        not isinstance(budget, int)
        or not 1 <= budget <= 2048
        or logits.shape[0] > 8192
        or logits.shape[1] >= 2**31
    ):
        raise NotImplementedError(
            "Flash selector supports rows<=8192, budget1..2048, int32 indices"
        )
    if (
        upper.shape != (logits.shape[0],)
        or upper.device != logits.device
        or upper.dtype not in (torch.int32, torch.int64)
        or not upper.is_contiguous()
    ):
        raise ValueError("Invalid group upper bounds")
    n, width = logits.shape
    out = torch.empty((n, budget), dtype=torch.int32, device=logits.device)
    if not n:
        return out
    if width <= budget:
        _flash_select[(n,)](
            logits,
            upper,
            out,
            *logits.stride(),
            width,
            budget,
            triton.next_power_of_2(budget),
            num_warps=4,
        )
    elif width <= 8192 and n < 1024 and logits.stride(1) == 1:
        _flash_select_single_cta[(n,)](
            logits,
            upper,
            out,
            logits.stride(0),
            width,
            budget,
            triton.next_power_of_2(width),
            num_warps=4,
        )
    else:
        from flaggems_vllm.ops._hy4_flash_radix import select_wide

        select_wide(logits, upper, out)
    return out
