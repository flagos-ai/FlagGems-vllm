# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather live compressed groups from previous carry and incoming keys."""

import torch
import triton
import triton.language as tl


@triton.jit
def _group_prefix(
    Starts, Lengths, Ends, BATCH: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr
):
    request = tl.arange(0, BLOCK)
    start = tl.load(Starts + request, request < BATCH, 0)
    end = tl.load(Starts + request + 1, request < BATCH, 0)
    length = tl.load(Lengths + request, request < BATCH, 0)
    context = length - (end - start)
    live = (request < BATCH) & (end > start) & (context >= 0)
    count = tl.where(live, length // R - tl.maximum(context, 0) // R, 0)
    tl.store(Ends + request, tl.cumsum(count), request < BATCH)


@triton.jit
def _gather_groups(
    Keys,
    Positions,
    Tail,
    Table,
    Starts,
    Lengths,
    Ends,
    Raw,
    XYZ,
    Requests,
    Groups,
    Live,
    BATCH: tl.constexpr,
    TOKENS: tl.constexpr,
    SLOTS: tl.constexpr,
    R: tl.constexpr,
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
    slot = tl.program_id(0)
    lo, hi = 0, BATCH
    while lo < hi:
        mid = (lo + hi) // 2
        end_count = tl.load(Ends + mid, mid < BATCH, 2147483647)
        left = slot < end_count
        hi = tl.where(left, mid, hi)
        lo = tl.where(left, lo, mid + 1)
    live = lo < BATCH
    request = tl.minimum(lo, BATCH - 1)
    start = tl.load(Starts + request)
    end = tl.load(Starts + request + 1)
    length = tl.load(Lengths + request)
    context = tl.maximum(length - (end - start), 0)
    previous = tl.load(Ends + request - 1, request > 0, 0)
    group = context // R + slot - previous
    first = group * R
    block = tl.load(Table + request * TABLE_ROW, live, 0).to(tl.int64)
    member = first + tl.arange(0, R)
    channel = tl.arange(0, 128)
    old = member < context
    source = start + member - context
    retained = tl.load(
        Tail
        + block * TAIL_BLOCK
        + (member % RING)[:, None] * TAIL_ROW
        + channel[None, :],
        live & old[:, None],
        0,
    )
    current = tl.load(
        Keys + source[:, None] * KEY_ROW + channel[None, :] * KEY_COL,
        live & ~old[:, None] & (source[:, None] >= 0) & (source[:, None] < TOKENS),
        0,
    )
    tl.store(
        Raw + slot * R * 128 + tl.arange(0, R)[:, None] * 128 + channel[None, :],
        tl.where(old[:, None], retained, current),
    )
    axis = tl.arange(0, 4)
    retained_xyz = tl.load(
        Tail + block * TAIL_BLOCK + (first % RING) * TAIL_ROW + 128 + axis,
        live & (first < context) & (axis < 3),
        0,
    ).to(tl.int64)
    source_first = start + first - context
    if THREE:
        position = tl.load(
            Positions + axis * POS_PLANE + source_first * POS_TOKEN,
            live & (first >= context) & (axis < 3),
            0,
        ).to(tl.int64)
    else:
        position = tl.load(
            Positions + source_first * POS_TOKEN,
            live & (first >= context),
            0,
        ).to(tl.int64)
    coordinates = tl.where(first < context, retained_xyz, position)
    tl.store(XYZ + axis * SLOTS + slot, tl.where(live, coordinates, 0), axis < 3)
    tl.store(Requests + slot, request)
    tl.store(Groups + slot, group)
    tl.store(Live + slot, live)


@triton.jit
def _query_metadata(
    Starts,
    Lengths,
    Owner,
    Logical,
    Upper,
    Live,
    BATCH: tl.constexpr,
    TOKENS: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), BATCH, tl.int32)
    for _ in range((BATCH + 1).bit_length()):
        mid = (lo + hi) // 2
        end = tl.load(Starts + mid + 1, mid < BATCH, 2147483647)
        left = token < end
        hi = tl.where(left, mid, hi)
        lo = tl.where(left, lo, mid + 1)
    request = tl.minimum(lo, BATCH - 1)
    start = tl.load(Starts + request)
    end = tl.load(Starts + request + 1)
    length = tl.load(Lengths + request)
    context = length - (end - start)
    live = (lo < BATCH) & (end > start) & (context >= 0)
    logical = tl.maximum(context, 0).to(tl.int64) + token - start
    tl.store(Owner + token, request, token < TOKENS)
    tl.store(Logical + token, logical, token < TOKENS)
    tl.store(Upper + token, tl.where(live, logical // R, 0), token < TOKENS)
    tl.store(Live + token, live, token < TOKENS)


def hy4_flash_prepare_groups(
    keys, positions, tail, table, starts, lengths, tokens, ratio
):
    if ratio not in (2, 4, 8, 16):
        raise ValueError("Unsupported IndexPool group ratio")
    batch, device = lengths.numel(), keys.device
    if tokens and not batch:
        raise ValueError("IndexPool keys require live request metadata")
    slots = (tokens + batch * (ratio - 1)) // ratio
    prefix = torch.empty(batch, dtype=torch.int32, device=device)
    raw = torch.empty((slots, ratio, 128), dtype=keys.dtype, device=device)
    coordinates = torch.empty((3, slots), dtype=torch.int64, device=device)
    requests = torch.empty(slots, dtype=torch.int32, device=device)
    groups = torch.empty(slots, dtype=torch.int64, device=device)
    group_live = torch.empty(slots, dtype=torch.bool, device=device)
    owner = torch.empty(tokens, dtype=torch.int64, device=device)
    logical = torch.empty(tokens, dtype=torch.int64, device=device)
    upper = torch.empty(tokens, dtype=torch.int32, device=device)
    token_live = torch.empty(tokens, dtype=torch.bool, device=device)
    if batch:
        _group_prefix[(1,)](
            starts, lengths, prefix, batch, ratio, triton.next_power_of_2(batch)
        )
    if slots:
        _gather_groups[(slots,)](
            keys,
            positions,
            tail,
            table,
            starts,
            lengths,
            prefix,
            raw,
            coordinates,
            requests,
            groups,
            group_live,
            batch,
            tokens,
            slots,
            ratio,
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
    if tokens:
        _query_metadata[(triton.cdiv(tokens, 256),)](
            starts,
            lengths,
            owner,
            logical,
            upper,
            token_live,
            batch,
            tokens,
            ratio,
            256,
        )
    group_positions = coordinates if positions.ndim == 2 else coordinates[0]
    return (
        raw,
        group_positions,
        requests,
        groups,
        group_live,
        owner,
        logical,
        upper,
        token_live,
    )
