# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retain new keys and rotary positions in an IndexPool tail ring."""

import triton
import triton.language as tl


@triton.jit
def _write_tails(
    Keys,
    Positions,
    Tail,
    Table,
    Starts,
    Lengths,
    RING: tl.constexpr,
    KEY_ROW: tl.constexpr,
    KEY_COL: tl.constexpr,
    POS_PLANE: tl.constexpr,
    POS_TOKEN: tl.constexpr,
    THREE: tl.constexpr,
    TAIL_BLOCK: tl.constexpr,
    TAIL_ROW: tl.constexpr,
    TABLE_ROW: tl.constexpr,
):
    request, offset = tl.program_id(0), tl.program_id(1)
    start = tl.load(Starts + request)
    end = tl.load(Starts + request + 1)
    length = tl.load(Lengths + request)
    context = length - (end - start)
    token = length - RING + offset
    live = (end > start) & (context >= 0) & (token >= context)
    source = start + token - context
    block = tl.load(Table + request * TABLE_ROW, live, 0).to(tl.int64)
    channel = tl.arange(0, 256)
    key = tl.load(
        Keys + source * KEY_ROW + channel * KEY_COL,
        live & (channel < 128),
        0,
    ).to(tl.float32)
    if THREE:
        axis = tl.maximum(channel - 128, 0)
        position = tl.load(
            Positions + axis * POS_PLANE + source * POS_TOKEN,
            live & (channel >= 128) & (channel < 131),
            0,
        ).to(tl.float32)
    else:
        position = tl.load(Positions + source * POS_TOKEN, live, 0).to(tl.float32)
    tl.store(
        Tail + block * TAIL_BLOCK + (token % RING) * TAIL_ROW + channel,
        tl.where(channel < 128, key, position),
        live & (channel < 131),
    )


def hy4_flash_write_tails(keys, positions, tail, table, starts, lengths):
    if lengths.numel():
        _write_tails[(lengths.numel(), tail.shape[1])](
            keys,
            positions,
            tail,
            table,
            starts,
            lengths,
            tail.shape[1],
            keys.stride(0),
            keys.stride(1),
            positions.stride(0) if positions.ndim == 2 else 0,
            positions.stride(-1),
            positions.ndim == 2,
            tail.stride(0),
            tail.stride(1),
            table.stride(0),
        )
