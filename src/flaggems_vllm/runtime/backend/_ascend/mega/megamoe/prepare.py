# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight packing and input preparation helpers."""
import triton
import triton.experimental.tle.language.raw as raw
import triton.language as tl

from . import bindings


@triton.jit
def _copy(X, Y, N: tl.constexpr, B: tl.constexpr):
    pos = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    tl.store(Y + pos, tl.load(X + pos, pos < N, 0), pos < N)


@triton.jit
def _zero(X, N: tl.constexpr):
    pos = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024)
    tl.store(X + pos, 0, pos < N)


@triton.jit
def _pack_gated_weights(
    Input,
    Output,
    N: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BN: tl.constexpr,
):
    for block in range(tl.program_id(0), tl.cdiv(N, 4096), tl.num_programs(0)):
        offset = block.to(tl.int64) * 4096 + tl.arange(0, 4096)
        h = offset // (2 * INTERMEDIATE)
        col = offset % (2 * INTERMEDIATE)
        original = col // (2 * BN) * BN + col % BN + col % (2 * BN) // BN * INTERMEDIATE
        value = tl.load(Input + h * (2 * INTERMEDIATE) + original, offset < N, 0)
        tl.store(Output + offset, value, offset < N)


@triton.jit
def _begin_counted(
    Epoch,
    Flags,
    Errors,
    Counts,
    WORLD: tl.constexpr,
    ERROR_WORDS: tl.constexpr,
    COUNT_WORDS: tl.constexpr,
):
    raw.call(bindings.begin, [Epoch, Flags + 2 * WORLD * 16, WORLD], output_indices=[])
    e = tl.arange(0, triton.next_power_of_2(ERROR_WORDS))
    tl.store(Errors + e, 0, e < ERROR_WORDS)
    c = tl.arange(0, triton.next_power_of_2(COUNT_WORDS))
    tl.store(Counts + c, 0, c < COUNT_WORDS)


@triton.jit
def _copy_matrix(
    X,
    Y,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    PAD_ROWS: tl.constexpr,
    PAD_COLS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    GATED: tl.constexpr = False,
):
    WIDE: tl.constexpr = (PAD_ROWS * PAD_COLS > 2**31) or (
        (ROWS - 1) * S0 + (COLS - 1) * S1 >= 2**31 or S0 >= 2**31 or S1 >= 2**31
    )
    # Wide integer division/gather needs more UB temporaries on 910B.
    BLOCK: tl.constexpr = 1024 if WIDE else 4096
    for chunk in range(4096 // BLOCK):
        block = tl.program_id(0)
        if WIDE:
            block = block.to(tl.int64)
        pos = block * 4096 + chunk * BLOCK + tl.arange(0, BLOCK)
        row, col = pos // PAD_COLS, pos % PAD_COLS
        if GATED:
            half = PAD_COLS // 2
            original_col = col % half + (col // half) * (COLS // 2)
            valid_col = col % half < COLS // 2
        else:
            original_col = col
            valid_col = col < COLS
        value = tl.load(
            X + row * S0 + original_col * S1,
            (row < ROWS) & valid_col & (pos < PAD_ROWS * PAD_COLS),
            0,
        )
        tl.store(Y + pos, value, pos < PAD_ROWS * PAD_COLS)


@triton.jit
def _prepare_strided_input(
    X,
    Source,
    actor,
    actors: tl.constexpr,
    T: tl.constexpr,
    LOGICAL_H: tl.constexpr,
    H: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
):
    if T > 0:
        WIDE: tl.constexpr = (
            T * H >= 2**31
            or X0 >= 2**31
            or X1 >= 2**31
            or (T - 1) * X0 + (LOGICAL_H - 1) * X1 >= 2**31
        )
        B: tl.constexpr = 256 if WIDE else 512
        for block in range(actor, tl.cdiv(T * H, B), actors):
            base = block.to(tl.int64) if WIDE else block
            copy_pos = base * B + tl.arange(0, B)
            copy_row, copy_col = copy_pos // H, copy_pos % H
            copy_value = tl.load(
                X + copy_row * X0 + copy_col * X1,
                (copy_row < T) & (copy_col < LOGICAL_H),
                0,
            )
            tl.store(Source + copy_pos, copy_value, copy_pos < T * H)


@triton.jit
def _prepare_routes(
    Ids,
    Probs,
    Tokens,
    Experts,
    Weights,
    Packets,
    ExpertCounts,
    ChunkCounts,
    actor,
    actors: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    E: tl.constexpr,
    C: tl.constexpr,
    WORLD: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    ROWS: tl.constexpr,
    RETURN_BLOCK: tl.constexpr,
    RETURN_CHUNKS: tl.constexpr,
):
    if T > 0:
        # Bound routing temporaries while retaining wide top-k support.
        blocks: tl.constexpr = tl.cdiv(T, ROWS)
        for work in range(actor, WORLD * blocks, actors):
            destination = work // blocks
            route_row = (work % blocks).to(tl.int64) * ROWS + tl.arange(0, ROWS)
            slot = tl.arange(0, KP)
            valid = (route_row < T)[:, None] & (slot < K)[None, :]
            ids = tl.load(Ids + route_row[:, None] * I0 + slot[None, :] * I1, valid, -1)
            hit = valid & (ids >= destination * E) & (ids < (destination + 1) * E)
            if Probs.dtype.element_ty != tl.int16:
                if P1 == 1 and (T <= 1 or P0 == K):
                    probs = tl.load(
                        Probs + route_row[:, None] * P0 + slot[None, :], hit, 0
                    )
                else:
                    # Flatten the strided gather before restoring the route layout. The A2
                    # lowering of a transposed 2-D FP32 gather can silently read wrong values.
                    flat = tl.arange(0, ROWS * KP)
                    prob_row = (work % blocks).to(tl.int64) * ROWS + flat // KP
                    offset = prob_row * P0 + (flat % KP) * P1
                    mask = tl.reshape(hit, (ROWS * KP,))
                    probs = tl.reshape(tl.load(Probs + offset, mask, 0), (ROWS, KP))
            hits = tl.sum(hit.to(tl.int32), 1)
            if RETURN_BLOCK:
                chunk = (work % blocks) * ROWS // RETURN_BLOCK
                tl.atomic_add(
                    ChunkCounts + destination * RETURN_CHUNKS + chunk,
                    tl.sum(hits, 0),
                    sem="relaxed",
                )
            needed = hits > 0
            tl.store(
                Tokens + destination.to(tl.int64) * C + route_row,
                tl.where(needed, route_row, -1),
                route_row < T,
            )
            dst = (destination.to(tl.int64) * C + route_row[:, None]) * KP + slot[
                None, :
            ]
            tl.store(Experts + dst, tl.where(hit, ids, -1), (route_row < T)[:, None])
            if Probs.dtype.element_ty != tl.int16:
                tl.store(Weights + dst, probs, (route_row < T)[:, None])
            if work % blocks == 0:
                tl.store(Packets + destination * 16, T)
            bins: tl.constexpr = triton.next_power_of_2(E + 1)
            values = tl.reshape(tl.where(hit, ids - destination * E, E), (ROWS * KP,))
            histogram = tl.histogram(values.to(tl.int32), bins)
            expert = tl.arange(0, bins)
            tl.atomic_add(
                ExpertCounts + destination * (tl.cdiv(E, 8) * 8) + expert,
                histogram,
                expert < E,
                sem="relaxed",
            )
    elif actor < WORLD:
        tl.store(Packets + actor * 16, 0)


@triton.jit
def _reset_progress(
    Progress,
    Requests,
    Assignments,
    actor,
    actors: tl.constexpr,
    PROGRESS_WORDS: tl.constexpr,
):
    REQUEST_WORDS: tl.constexpr = actors * 16
    RESET_WORDS: tl.constexpr = (
        PROGRESS_WORDS if PROGRESS_WORDS > REQUEST_WORDS else REQUEST_WORDS
    )
    for block in range(actor, tl.cdiv(RESET_WORDS, 256), actors):
        reset_pos = block * 256 + tl.arange(0, 256)
        tl.store(Progress + reset_pos, 0, reset_pos < PROGRESS_WORDS)
        tl.store(Requests + reset_pos, 0, reset_pos < REQUEST_WORDS)
        tl.store(Assignments + reset_pos, 0, reset_pos < REQUEST_WORDS)
