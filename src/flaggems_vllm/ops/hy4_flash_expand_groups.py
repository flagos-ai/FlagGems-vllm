# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Expand history groups and the live current group into packed token IDs."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime


@triton.jit
def _flash_expand(
    G,
    P,
    A,
    Output,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    BUDGET: tl.constexpr,
    R: tl.constexpr,
    ACTIVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    pos = tl.load(P + row)
    upper = pos // R
    live = pos >= 0
    if ACTIVE:
        live = live & tl.load(A + row)
    group_index = j // R
    group = tl.load(G + row * GS0 + group_index * GS1, group_index < BUDGET, other=-1)
    history = j < BUDGET * R
    value = tl.where(history, group * R + j % R, upper * R + j % R)
    valid = tl.where(history, (group >= 0) & (group < upper), value <= pos)
    valid = valid & live & (j < BUDGET * R + R)
    total: tl.constexpr = BUDGET * R + R
    in_range = j < total
    nv = tl.sum(valid.to(tl.int32), 0)
    good = tl.cumsum(valid.to(tl.int32), 0) - 1
    bad = nv + tl.cumsum((in_range & ~valid).to(tl.int32), 0) - 1
    dst = tl.where(valid, good, bad)
    tl.store(Output + row * total + dst, tl.where(valid, value, -1), in_range)


def hy4_flash_expand_groups(group_ids, positions, ratio, active=None):
    if (
        group_ids.ndim != 2
        or group_ids.dtype not in (torch.int32, torch.int64)
        or ratio not in (2, 4, 8, 16)
    ):
        raise ValueError("Invalid Flash group expansion inputs")
    if runtime.device.vendor_name not in ("nvidia", "metax") or group_ids.device.type != "cuda":
        raise NotImplementedError("Flash group expansion requires NVIDIA CUDA")
    n, budget = group_ids.shape
    if (
        positions.dtype not in (torch.int32, torch.int64)
        or positions.shape != (n,)
        or positions.device != group_ids.device
        or not positions.is_contiguous()
    ):
        raise ValueError("Invalid group positions")
    if active is not None and (
        active.shape != (n,)
        or active.dtype != torch.bool
        or active.device != group_ids.device
        or not active.is_contiguous()
    ):
        raise ValueError("Invalid group activation mask")
    out = torch.empty(
        (n, budget * ratio + ratio), dtype=torch.int32, device=group_ids.device
    )
    if n:
        _flash_expand[(n,)](
            group_ids,
            positions,
            active,
            out,
            *group_ids.stride(),
            budget,
            ratio,
            active is not None,
            triton.next_power_of_2(out.shape[1]),
            num_warps=4,
        )
    return out
