# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused sailSHMEM communication, projection, and epilogue on PPU."""

import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language import core

from flaggems_vllm.ops.mrope import _axis_for_pair


@core.extern
def _put_tile(dest, source, size, ready, epoch, peer, _semantic=None):
    u64 = tl.uint64
    return core.extern_elementwise(
        "sailshmem",
        "",
        [dest, source, size, ready, epoch, peer],
        {(u64, u64, u64, u64, u64, tl.int32): ("mega_put_signal", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def _broadcast_tile(
    dest, source, size, ready, epoch, rank, tile, world, _semantic=None
):
    u64 = tl.uint64
    i32 = tl.int32
    return core.extern_elementwise(
        "sailshmem",
        "",
        [dest, source, size, ready, epoch, rank, tile, world],
        {(u64, u64, u64, u64, u64, i32, i32, i32): ("mega_broadcast", i32)},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def _wait(signal, epoch, _semantic=None):
    return core.extern_elementwise(
        "sailshmem",
        "",
        [signal, epoch],
        {(tl.uint64, tl.uint64): ("mega_wait", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def _signal(signal, epoch, peer, _semantic=None):
    return core.extern_elementwise(
        "sailshmem",
        "",
        [signal, epoch, peer],
        {(tl.uint64, tl.uint64, tl.int32): ("mega_signal", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def _publish_tile(
    hidden,
    gathered,
    ready,
    entered,
    tile,
    epoch,
    rank,
    WORLD: tl.constexpr,
    COMM_TILES: tl.constexpr,
    COMM_JOBS: tl.constexpr,
    LOCAL_TOKENS: tl.constexpr,
    COMM_M: tl.constexpr,
    HIDDEN: tl.constexpr,
    TILE_ELEMS: tl.constexpr,
):
    if tile == 0:
        # The first CTA assigned on each rank announces entry. That rank has
        # finished the preceding launch and no longer reads the old payload.
        for step in range(1, WORLD):
            peer = (rank + step) % WORLD
            _signal((entered + rank).to(tl.uint64), epoch, peer)
    for step in range(1, WORLD):
        peer = (rank + step) % WORLD
        _wait((entered + peer).to(tl.uint64), epoch)

    for tile_index in range(tile, COMM_TILES, COMM_JOBS):
        row = rank.to(tl.int64) * COMM_TILES + tile_index
        source = (hidden + tile_index.to(tl.int64) * TILE_ELEMS).to(tl.uint64)
        if LOCAL_TOKENS % COMM_M:
            start = tile_index.to(tl.int64) * COMM_M
            rows = tl.minimum(LOCAL_TOKENS - start, COMM_M)
            target_row = rank.to(tl.int64) * LOCAL_TOKENS + start
            target = (gathered + target_row * HIDDEN).to(tl.uint64)
            size = rows.to(tl.uint64) * (HIDDEN * 2)
        else:
            target = (gathered + row * TILE_ELEMS).to(tl.uint64)
            size = tl.full((), TILE_ELEMS * 2, tl.uint64)
        signal = (ready + row).to(tl.uint64)
        if WORLD == 4 or WORLD == 8 or WORLD == 16:
            _broadcast_tile(
                target,
                source,
                size,
                signal,
                epoch,
                rank,
                tile_index,
                tl.full((), WORLD, tl.int32),
            )
        else:
            for step in range(1, WORLD):
                peer = (rank + step) % WORLD
                _put_tile(target, source, size, signal, epoch, peer)


@triton.jit
def _project_head(
    source,
    weight,
    head_tile,
    rows,
    HIDDEN: tl.constexpr,
    HEAD_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    HAS_ROW_TAIL: tl.constexpr,
):
    a_tile = tl.make_block_ptr(
        base=source,
        shape=(rows, HIDDEN),
        strides=(HIDDEN, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_tile = tl.make_block_ptr(
        base=weight,
        shape=(HIDDEN, HEAD_TILES * 256),
        strides=(1, HIDDEN),
        offsets=(0, head_tile * 256),
        block_shape=(BLOCK_K, 256),
        order=(0, 1),
    )
    acc = tl.zeros((BLOCK_M, 256), tl.float32)
    for _ in tl.range(0, HIDDEN // BLOCK_K, num_stages=PIPE_STAGES):
        if HAS_ROW_TAIL:
            a = tle.load(
                a_tile, boundary_check=(0,), padding_option="zero", is_async=True
            )
        else:
            a = tle.load(a_tile, is_async=True)
        b = tle.load(b_tile, is_async=True)
        acc = tl.dot(a, b, acc)
        a_tile = tl.advance(a_tile, (0, BLOCK_K))
        b_tile = tl.advance(b_tile, (BLOCK_K, 0))
    return acc


@triton.jit
def _norm_rope(
    raw,
    cols,
    token,
    row_valid,
    positions,
    norm,
    rope,
    eps,
    POS_AXIS: tl.constexpr,
    POS_TOKEN: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    H_MASK: tl.constexpr,
    W_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    x = raw.to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, 1) / 256.0 + eps)
    weight = tl.load(norm + cols).to(tl.float32)
    normalized = (x * scale[:, None] * (1.0 + weight[None, :])).to(tl.bfloat16)

    # Load a contiguous cos/sin row per axis and reuse it for matching positions.
    position_ptr = positions + token * POS_TOKEN
    zero = 0 if row_valid is not None else None
    position_t = tl.load(position_ptr, row_valid, zero)
    position_h = tl.load(position_ptr + POS_AXIS, row_valid, zero)
    position_w = tl.load(position_ptr + 2 * POS_AXIS, row_valid, zero)
    pair = tl.arange(0, 64)
    if H_MASK is None:
        axis = _axis_for_pair(
            pair % 32, None, 0, SECTION_H, SECTION_W, True, False, 256
        )
        use_h = axis == 1
        use_w = axis == 2
    else:
        shift = (pair % 32).to(tl.uint32)
        use_h = ((tl.full((), H_MASK, tl.uint32) >> shift) & 1) != 0
        use_w = ((tl.full((), W_MASK, tl.uint32) >> shift) & 1) != 0
    phase_t = rope + position_t[:, None] * ROPE_STRIDE + pair[None, :]
    phase_valid = row_valid[:, None] if row_valid is not None else None
    coefficients = tl.load(phase_t, phase_valid, zero)
    if SECTION_H:
        phase_h = rope + position_h[:, None] * ROPE_STRIDE + pair[None, :]
        distinct_h = (position_h != position_t)[:, None]
        select_h = distinct_h & use_h[None, :]
        coefficients = tl.where(select_h, tl.load(phase_h, distinct_h, 0), coefficients)
    if SECTION_W:
        phase_w = rope + position_w[:, None] * ROPE_STRIDE + pair[None, :]
        distinct_w = (position_w != position_t)[:, None]
        select_w = distinct_w & use_w[None, :]
        coefficients = tl.where(select_w, tl.load(phase_w, distinct_w, 0), coefficients)
    cos, sin = tl.split(coefficients.reshape(BLOCK_M, 2, 32).permute(0, 2, 1))
    cos = cos.to(tl.float32)
    sin = sin.to(tl.float32)
    low128, high128 = tl.split(normalized.reshape(BLOCK_M, 2, 128).permute(0, 2, 1))
    low64, high64 = tl.split(low128.reshape(BLOCK_M, 2, 64).permute(0, 2, 1))
    # Only the rotary prefix needs FP32 arithmetic after BF16 normalization.
    prefix = low64.reshape(BLOCK_M, 2, 32).permute(0, 2, 1)
    first, second = tl.split(prefix)
    first = first.to(tl.float32)
    second = second.to(tl.float32)
    first_rotated = (first * cos - second * sin).to(tl.bfloat16)
    second_rotated = (second * cos + first * sin).to(tl.bfloat16)
    low64 = tl.join(first_rotated, second_rotated).permute(0, 2, 1).reshape(BLOCK_M, 64)
    low128 = tl.join(low64, high64).permute(0, 2, 1).reshape(BLOCK_M, 128)
    return tl.join(low128, high128).permute(0, 2, 1).reshape(BLOCK_M, 256)


@triton.jit
def _cache_offset(
    slot,
    head,
    CACHE_BLOCK: tl.constexpr,
    CACHE_HEAD: tl.constexpr,
    CACHE_TOKEN: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
):
    slot = slot.to(tl.int64)
    return (
        (slot // CACHE_BLOCK_SIZE) * CACHE_BLOCK
        + head.to(tl.int64) * CACHE_HEAD
        + (slot % CACHE_BLOCK_SIZE) * CACHE_TOKEN
    )


@triton.jit
def _store_head(
    acc,
    token,
    row_valid,
    head_tile,
    query,
    key,
    value,
    gate,
    q_norm,
    k_norm,
    positions,
    rope,
    cache,
    slots,
    eps,
    POS_AXIS: tl.constexpr,
    POS_TOKEN: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_HEAD: tl.constexpr,
    CACHE_TOKEN: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    CACHE_CAPACITY: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    H_MASK: tl.constexpr,
    W_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    cols = tl.arange(0, 256)
    output_valid = row_valid[:, None] if row_valid is not None else None
    # Preserve the BF16 rounding at the projection boundary.
    raw = acc.to(tl.bfloat16)
    if head_tile < 2 * Q_HEADS:
        head = head_tile // 2
        out_offset = (token[:, None] * Q_HEADS + head) * 256 + cols[None, :]
        if head_tile % 2:
            tl.store(gate + out_offset, raw, output_valid)
        else:
            result = _norm_rope(
                raw,
                cols,
                token,
                row_valid,
                positions,
                q_norm,
                rope,
                eps,
                POS_AXIS,
                POS_TOKEN,
                ROPE_STRIDE,
                SECTION_H,
                SECTION_W,
                H_MASK,
                W_MASK,
                BLOCK_M,
            )
            tl.store(query + out_offset, result, output_valid)
    else:
        is_key = head_tile < 2 * Q_HEADS + KV_HEADS
        head = (head_tile - 2 * Q_HEADS) % KV_HEADS
        out_offset = (token[:, None] * KV_HEADS + head) * 256 + cols[None, :]
        no_slot = -1 if row_valid is not None else None
        slot = tl.load(slots + token * SLOT_STRIDE, row_valid, no_slot)
        valid = (slot >= 0) & (slot < CACHE_CAPACITY)
        cache_offset = _cache_offset(
            slot, head, CACHE_BLOCK, CACHE_HEAD, CACHE_TOKEN, CACHE_BLOCK_SIZE
        )
        target = cache + cache_offset[:, None] + cols[None, :]
        if is_key:
            result = _norm_rope(
                raw,
                cols,
                token,
                row_valid,
                positions,
                k_norm,
                rope,
                eps,
                POS_AXIS,
                POS_TOKEN,
                ROPE_STRIDE,
                SECTION_H,
                SECTION_W,
                H_MASK,
                W_MASK,
                BLOCK_M,
            )
            tl.store(key + out_offset, result, output_valid)
            tl.store(target, result, valid[:, None])
        else:
            tl.store(value + out_offset, raw, output_valid)
            tl.store(target + 256, raw, valid[:, None])


@triton.jit(do_not_specialize=["symmetric", "rank", "eps"])
def megapreattention_kernel(
    hidden,
    weight,
    positions,
    q_norm,
    k_norm,
    rope,
    cache,
    slots,
    query,
    key,
    value,
    gate,
    tickets,
    symmetric,
    rank,
    eps,
    POS_AXIS: tl.constexpr,
    POS_TOKEN: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_HEAD: tl.constexpr,
    CACHE_TOKEN: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    CACHE_CAPACITY: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    H_MASK: tl.constexpr,
    W_MASK: tl.constexpr,
    WORLD: tl.constexpr,
    LOCAL_TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    COMM_M: tl.constexpr,
    COMM_JOBS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    TILE_ELEMS: tl.constexpr = BLOCK_M * HIDDEN
    LOCAL_TILES: tl.constexpr = (LOCAL_TOKENS + BLOCK_M - 1) // BLOCK_M
    HEAD_TILES: tl.constexpr = 2 * (Q_HEADS + KV_HEADS)
    TILES_PER_MESSAGE: tl.constexpr = COMM_M // BLOCK_M
    COMM_TILES: tl.constexpr = (LOCAL_TOKENS + COMM_M - 1) // COMM_M
    LOCAL_JOBS: tl.constexpr = LOCAL_TILES * HEAD_TILES
    TOTAL_JOBS: tl.constexpr = COMM_JOBS + WORLD * LOCAL_JOBS

    gathered = symmetric.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    ready = (gathered + WORLD * LOCAL_TOKENS * HIDDEN).to(tl.pointer_type(tl.uint64))
    entered = ready + WORLD * COMM_TILES

    # Each launch claims exactly TOTAL_JOBS tickets. The first scheduled
    # CTAs send data, so consumers never block the launch of a producer.
    ticket = tl.atomic_add(tickets, tl.full((), 1, tl.uint64), sem="relaxed")
    epoch = ticket // TOTAL_JOBS + tl.full((), 1, tl.uint64)
    ordinal = (ticket % TOTAL_JOBS).to(tl.int32)
    if ordinal < COMM_JOBS:
        _publish_tile(
            hidden,
            gathered,
            ready,
            entered,
            ordinal,
            epoch,
            rank,
            WORLD,
            COMM_TILES,
            COMM_JOBS,
            LOCAL_TOKENS,
            COMM_M,
            HIDDEN,
            COMM_M * HIDDEN,
        )
    else:
        job = ordinal - COMM_JOBS
        source_rank = (rank + job // LOCAL_JOBS) % WORLD
        local_job = job % LOCAL_JOBS
        if GROUP_M == 1:
            tile_in_rank = local_job // HEAD_TILES
            head_tile = local_job % HEAD_TILES
        else:
            # Compute adjacent row tiles with the same weight tile.
            row_group = local_job // (GROUP_M * HEAD_TILES)
            first_row = row_group * GROUP_M
            group_rows = tl.minimum(LOCAL_TILES - first_row, GROUP_M)
            within = local_job % (GROUP_M * HEAD_TILES)
            tile_in_rank = first_row + within % group_rows
            head_tile = within // group_rows
        global_tile = source_rank.to(tl.int64) * LOCAL_TILES + tile_in_rank
        if LOCAL_TOKENS % BLOCK_M:
            local_start = tile_in_rank.to(tl.int64) * BLOCK_M
            token_start = source_rank.to(tl.int64) * LOCAL_TOKENS + local_start
            rows = tl.minimum(LOCAL_TOKENS - local_start, BLOCK_M).to(tl.int32)
            row_valid = tl.arange(0, BLOCK_M) < rows
        else:
            token_start = global_tile * BLOCK_M
            rows = BLOCK_M
            row_valid = None
        if source_rank == rank:
            source = hidden + tile_in_rank.to(tl.int64) * TILE_ELEMS
        else:
            if LOCAL_TOKENS % COMM_M:
                message = (
                    source_rank.to(tl.int64) * COMM_TILES
                    + tile_in_rank // TILES_PER_MESSAGE
                )
            else:
                message = global_tile // TILES_PER_MESSAGE
            _wait((ready + message).to(tl.uint64), epoch)
            if LOCAL_TOKENS % BLOCK_M:
                source = gathered + token_start * HIDDEN
            else:
                source = gathered + global_tile * TILE_ELEMS
        acc = _project_head(
            source,
            weight,
            head_tile,
            rows,
            HIDDEN,
            HEAD_TILES,
            BLOCK_M,
            BLOCK_K,
            PIPE_STAGES,
            LOCAL_TOKENS % BLOCK_M != 0,
        )
        token = token_start + tl.arange(0, BLOCK_M)
        _store_head(
            acc,
            token,
            row_valid,
            head_tile,
            query,
            key,
            value,
            gate,
            q_norm,
            k_norm,
            positions,
            rope,
            cache,
            slots,
            eps,
            POS_AXIS,
            POS_TOKEN,
            ROPE_STRIDE,
            SLOT_STRIDE,
            CACHE_BLOCK,
            CACHE_HEAD,
            CACHE_TOKEN,
            CACHE_BLOCK_SIZE,
            CACHE_CAPACITY,
            Q_HEADS,
            KV_HEADS,
            SECTION_H,
            SECTION_W,
            H_MASK,
            W_MASK,
            BLOCK_M,
        )
