# SPDX-License-Identifier: Apache-2.0
"""Pool expansion for small power-of-two pools on Ascend."""

import triton
import triton.language as tl


@triton.jit
def _expand_pool_tile(
    ids,
    lengths,
    out,
    GROUPS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    history: tl.constexpr = GROUPS * POOL_SIZE
    width: tl.constexpr = history + POOL_SIZE - 1
    row_out = out + row * width
    lane = tl.arange(0, POOL_SIZE)
    if GROUPS:
        group = tile * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
        pool = tl.load(ids + row * ROW_STRIDE + group, group < GROUPS, other=-1)
        # Test before narrowing: a negative int64 sentinel can have positive low bits.
        # The input contract bounds all valid expanded tokens by INT32_MAX.
        base = tl.where(pool >= 0, pool.to(tl.int32), -1) * POOL_SIZE
        # A missing pool starts at -POOL_SIZE, so every lane clamps to -1.
        tokens = tl.maximum(base[:, None] + lane[None, :], -1)
        columns = group[:, None] * POOL_SIZE + lane[None, :]
        tl.store(row_out + columns, tokens, group[:, None] < GROUPS)
    if POOL_SIZE > 1:
        if tile == 0:
            _store_tail(lengths + row, row_out + history, lane, POOL_SIZE)


@triton.jit
def _store_tail(length_ptr, tail_ptr, lane, POOL_SIZE: tl.constexpr):
    # Only the remainder matters for complete pools. For an incomplete pool,
    # length - 1 is an emitted index and is bounded by the input contract.
    length = tl.load(length_ptr).to(tl.int32)
    count = length & (POOL_SIZE - 1)
    tl.store(
        tail_ptr + lane,
        tl.where(lane < count, (length - count + lane).to(tl.int32), -1),
        lane < POOL_SIZE - 1,
    )


@triton.jit
def _expand_wide_pool_rows(
    ids,
    lengths,
    out,
    ROWS: tl.constexpr,
    GROUPS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    tile = tl.program_id(1)
    start_group = tile * BLOCK_GROUPS
    group = start_group + tl.arange(0, BLOCK_GROUPS)
    columns = tl.arange(0, BLOCK_GROUPS * POOL_SIZE)
    lane = tl.arange(0, POOL_SIZE)
    history: tl.constexpr = GROUPS * POOL_SIZE
    width: tl.constexpr = history + POOL_SIZE - 1
    first = tl.program_id(0) * BLOCK_ROWS
    if ROWS % BLOCK_ROWS == 0:
        end = first + BLOCK_ROWS
    else:
        end = tl.minimum(first + BLOCK_ROWS, ROWS)
    for row in range(first, end):
        if GROUPS % BLOCK_GROUPS == 0:
            pool = tl.load(ids + row * ROW_STRIDE + group)
        else:
            pool = tl.load(ids + row * ROW_STRIDE + group, group < GROUPS, other=-1)
        # fp32 preserves the sign of every int32/int64 input. Derive the
        # token payload from the original integer, including values above 2**24.
        base = tl.where(pool.to(tl.float32) >= 0, pool.to(tl.int32) * POOL_SIZE, -1)
        if POOL_SIZE == 4:
            expanded = tl.gather(base.to(tl.float32, bitcast=True), columns >> 2, 0).to(
                tl.int32, bitcast=True
            )
            tokens = expanded | (columns & 3)
        else:
            tokens = tl.reshape(
                base[:, None] | lane[None, :], (BLOCK_GROUPS * POOL_SIZE,)
            )
        # Pool bases have zero lane bits; OR also preserves every bit of -1.
        row_out = out + row * width
        col = start_group * POOL_SIZE + columns
        if GROUPS % BLOCK_GROUPS == 0:
            tl.store(row_out + col, tokens)
        else:
            tl.store(row_out + col, tokens, col < history)
        if tile == 0:
            _store_tail(lengths + row, row_out + history, lane, POOL_SIZE)


def _launch_expand_pool_tiles(pool_ids, seq_lens, pool_size, out):
    rows, groups = pool_ids.shape
    if pool_size in (4, 16) and groups >= 32:
        block_groups = min(triton.next_power_of_2(groups), 2048 // pool_size)
        # Reuse lane vectors across rows while retaining at least 32 row tiles
        # once the batch is large enough.
        block_rows = min(8, max(1, rows // 32))
        grid = (triton.cdiv(rows, block_rows), triton.cdiv(groups, block_groups))
        _expand_wide_pool_rows[grid](
            pool_ids,
            seq_lens,
            out,
            rows,
            groups,
            pool_size,
            pool_ids.stride(0),
            block_groups,
            block_rows,
        )
        return
    block_groups = min(triton.next_power_of_2(max(groups, 1)), 2048 // pool_size)
    grid = (rows, max(triton.cdiv(groups, block_groups), 1))
    _expand_pool_tile[grid](
        pool_ids, seq_lens, out, groups, pool_size, pool_ids.stride(0), block_groups
    )
