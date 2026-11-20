# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Read group members from two buffers according to a prepared gather plan."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime


@triton.jit
def _gather(
    Keys,
    Tail,
    KeyRows,
    TailSlots,
    Out,
    R: tl.constexpr,
    RING: tl.constexpr,
    NKEY: tl.constexpr,
    NTAIL: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
):
    group = tl.program_id(0)
    member = tl.arange(0, R)
    channel = tl.arange(0, 128)
    key = tl.load(KeyRows + group * R + member).to(tl.int64)
    tail = tl.load(TailSlots + group * R + member).to(tl.int64)
    key_live = (key >= 0) & (key < NKEY)
    tail_live = (tail >= 0) & (tail < NTAIL)
    new = tl.load(
        Keys + key[:, None] * K0 + channel[None, :] * K1,
        key_live[:, None],
        0,
    ).to(tl.float32)
    old = tl.load(
        Tail
        + (tail // RING)[:, None] * T0
        + (tail % RING)[:, None] * T1
        + channel[None, :] * T2,
        tail_live[:, None],
        0,
    ).to(tl.float32)
    tl.store(
        Out + (group * R + member[:, None]) * 128 + channel[None, :],
        tl.where(key_live[:, None], new, old),
    )


def hy4_flash_gather_groups(keys, tail, key_rows, tail_slots):
    """Gather [groups, members, 128] rows from keys or tail.

    A nonnegative key row takes precedence over a tail slot for that member.
    Tail slots encode block * ring_size + row. Out-of-bounds/negative sources
    are treated as absent, and members with no source are zero-filled.
    """
    if keys.device.type != "cuda" or runtime.device.vendor_name != "nvidia":
        raise NotImplementedError("Flash group gather requires NVIDIA CUDA")
    if (
        keys.ndim != 2
        or keys.shape[1] != 128
        or keys.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or tail.ndim != 3
        or tail.shape[1] == 0
        or tail.shape[2] != 131
        or tail.dtype != torch.float32
    ):
        raise ValueError("Invalid Flash group source buffers")
    if (
        key_rows.ndim != 2
        or key_rows.shape[1] not in (2, 4, 8, 16)
        or tail_slots.shape != key_rows.shape
        or any(
            x.dtype not in (torch.int32, torch.int64) or not x.is_contiguous()
            for x in (key_rows, tail_slots)
        )
    ):
        raise ValueError("Invalid Flash group source mapping")
    if any(x.device != keys.device for x in (tail, key_rows, tail_slots)):
        raise ValueError("Flash group source tensors must share a device")
    groups, ratio = key_rows.shape
    out = torch.empty((groups, ratio, 128), dtype=keys.dtype, device=keys.device)
    if groups:
        _gather[(groups,)](
            keys,
            tail,
            key_rows,
            tail_slots,
            out,
            ratio,
            tail.shape[1],
            keys.shape[0],
            tail.shape[0] * tail.shape[1],
            *keys.stride(),
            *tail.stride(),
        )
    return out
