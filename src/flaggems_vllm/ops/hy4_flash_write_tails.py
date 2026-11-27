# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Copy key and rotary-position rows to explicitly mapped tail slots."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime


@triton.jit
def _write_tails(
    Keys,
    Positions,
    Tail,
    KeyRows,
    Slots,
    RING: tl.constexpr,
    NKEY: tl.constexpr,
    NPOS: tl.constexpr,
    NTAIL: tl.constexpr,
    KEY_ROW: tl.constexpr,
    KEY_COL: tl.constexpr,
    POS_PLANE: tl.constexpr,
    POS_TOKEN: tl.constexpr,
    THREE: tl.constexpr,
    TAIL_BLOCK: tl.constexpr,
    TAIL_ROW: tl.constexpr,
    TAIL_COL: tl.constexpr,
):
    row = tl.program_id(0)
    source = tl.load(KeyRows + row).to(tl.int64)
    slot = tl.load(Slots + row).to(tl.int64)
    live = (
        (source >= 0) & (source < NKEY) & (source < NPOS) & (slot >= 0) & (slot < NTAIL)
    )
    channel = tl.arange(0, 256)
    key = tl.load(
        Keys + source * KEY_ROW + channel * KEY_COL, live & (channel < 128), 0
    )
    if THREE:
        axis = tl.maximum(channel - 128, 0)
        position = tl.load(
            Positions + axis * POS_PLANE + source * POS_TOKEN,
            live & (channel >= 128) & (channel < 131),
            0,
        )
    else:
        position = tl.load(Positions + source * POS_TOKEN, live, 0)
    tl.store(
        Tail
        + (slot // RING) * TAIL_BLOCK
        + (slot % RING) * TAIL_ROW
        + channel * TAIL_COL,
        tl.where(channel < 128, key.to(tl.float32), position.to(tl.float32)),
        live & (channel < 131),
    )


def hy4_flash_write_tails(keys, positions, tail, key_rows, slots):
    """Copy mapped source rows into distinct tail slots in place.

    Slots encode block * ring_size + row. Negative or out-of-bounds entries
    are ignored. The tail may be strided, but must not overlap itself or the
    source tensors. Repeated live destination slots are not supported.
    """
    if keys.device.type != "cuda" or runtime.device.vendor_name != "nvidia":
        raise NotImplementedError("Flash tail writes require NVIDIA CUDA")
    if (
        keys.ndim != 2
        or keys.shape[1] != 128
        or keys.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or tail.ndim != 3
        or tail.shape[1] == 0
        or tail.shape[2] != 131
        or tail.dtype != torch.float32
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.ndim not in (1, 2)
        or (positions.ndim == 2 and positions.shape[0] != 3)
    ):
        raise ValueError("Invalid Flash tail buffers")
    if (
        key_rows.ndim != 1
        or slots.shape != key_rows.shape
        or any(
            x.dtype not in (torch.int32, torch.int64) or not x.is_contiguous()
            for x in (key_rows, slots)
        )
    ):
        raise ValueError("Invalid Flash tail write metadata")
    if any(x.device != keys.device for x in (positions, tail, key_rows, slots)):
        raise ValueError("Flash tail tensors must share a device")
    if key_rows.numel():
        _write_tails[(key_rows.numel(),)](
            keys,
            positions,
            tail,
            key_rows,
            slots,
            tail.shape[1],
            keys.shape[0],
            positions.shape[-1],
            tail.shape[0] * tail.shape[1],
            *keys.stride(),
            positions.stride(0) if positions.ndim == 2 else 0,
            positions.stride(-1),
            positions.ndim == 2,
            *tail.stride(),
        )
