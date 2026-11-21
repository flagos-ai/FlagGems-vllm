# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused BF16 mHC post, pre and RMS normalization on Ascend."""

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.runtime import driver


@triton.jit
def _mhc_load(ptr, mask):
    if mask is None:
        return tl.load(ptr)
    else:
        return tl.load(ptr, mask, other=0)


@triton.jit
def _mhc_post_tile(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    current_ptr,
    cube_input_ptr,
    stats_ptr,
    rows,
    columns,
    h_tile,
    tokens,
    cube_rows,
    HIDDEN: tl.constexpr,
    H_TILES: tl.constexpr,
    BF16_CUBE: tl.constexpr,
    ALIGNED_ROWS: tl.constexpr,
    write_current=True,
):
    live = None if ALIGNED_ROWS else rows[:, None] < tokens
    x = _mhc_load(x_ptr + rows[:, None] * HIDDEN + columns[None, :], live)
    row_base = residual_ptr + rows[:, None] * (4 * HIDDEN) + columns[None, :]
    r0 = _mhc_load(row_base, live).to(tl.float32)
    r1 = _mhc_load(row_base + HIDDEN, live).to(tl.float32)
    r2 = _mhc_load(row_base + 2 * HIDDEN, live).to(tl.float32)
    r3 = _mhc_load(row_base + 3 * HIDDEN, live).to(tl.float32)
    dest = tl.arange(0, 4)
    comb_offsets = rows[:, None] * 16 + dest[None, :]
    c0 = _mhc_load(comb_ptr + comb_offsets, live)
    c1 = _mhc_load(comb_ptr + comb_offsets + 4, live)
    c2 = _mhc_load(comb_ptr + comb_offsets + 8, live)
    c3 = _mhc_load(comb_ptr + comb_offsets + 12, live)
    post = _mhc_load(post_ptr + rows[:, None] * 4 + dest[None, :], live)
    value = r0[:, None, :] * c0[:, :, None] + r1[:, None, :] * c1[:, :, None]
    value += r2[:, None, :] * c2[:, :, None] + r3[:, None, :] * c3[:, :, None]
    value = (value + x[:, None, :].to(tl.float32) * post[:, :, None]).to(tl.bfloat16)
    offsets = (rows[:, None, None] * 4 + dest[None, :, None]) * HIDDEN + columns[
        None, None, :
    ]
    rounded = value.to(tl.float32)
    cube_mask = None if ALIGNED_ROWS else rows[:, None, None] < cube_rows
    if BF16_CUBE:
        tl.store(current_ptr + offsets, value, cube_mask)
    else:
        if write_current:
            tl.store(
                current_ptr + offsets, value, None if ALIGNED_ROWS else live[:, :, None]
            )
        tl.store(cube_input_ptr + offsets, rounded, cube_mask)
    squares = tl.sum(tl.sum(rounded * rounded, 2), 1)
    tl.store(
        stats_ptr + rows * H_TILES + h_tile,
        squares,
        None if ALIGNED_ROWS else rows < tokens,
    )


@triton.jit
def _mhc_project_tile(
    cube_input_ptr,
    fn_ptr,
    projection_ptr,
    rows,
    tokens,
    head_start,
    SPLIT_HEADS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BF16_CUBE: tl.constexpr,
    MASK_ROWS: tl.constexpr,
):
    width: tl.constexpr = 4 * HIDDEN
    chunks: tl.constexpr = width // BLOCK_K
    block_n: tl.constexpr = 16 if SPLIT_HEADS else 64 if BF16_CUBE else 32
    channels = head_start + tl.arange(0, block_n)
    lanes = tl.arange(0, BLOCK_K)
    acc = tl.full((BLOCK_M, block_n), 0.0, tl.float32)
    for chunk in range(chunks):
        k = lanes + chunk * BLOCK_K
        residual_offsets = rows[:, None] * width + k[None, :]
        weight_offsets = k[:, None] + channels[None, :] * width
        if BF16_CUBE:
            if MASK_ROWS:
                residual = tl.load(
                    cube_input_ptr + residual_offsets, rows[:, None] < tokens, other=0
                )
            else:
                residual = tl.load(cube_input_ptr + residual_offsets)
            weight = tl.load(fn_ptr + weight_offsets)
        else:
            residual = tl.load(
                cube_input_ptr + residual_offsets, rows[:, None] < tokens, other=0
            )
            if SPLIT_HEADS:
                weight = tl.load(fn_ptr + weight_offsets)
            else:
                weight = tl.load(
                    fn_ptr + weight_offsets, channels[None, :] < 24, other=0
                )
        if BF16_CUBE:
            acc = tl.dot(residual, weight, acc=acc)
        else:
            acc = tl.dot(residual, weight, acc=acc, input_precision="hf32")
    offsets = rows[:, None] * block_n + channels[None, :]
    if BF16_CUBE and not MASK_ROWS:
        tl.store(projection_ptr + offsets, acc)
    else:
        tl.store(projection_ptr + offsets, acc, rows[:, None] < tokens)


@triton.jit
def _projection_slice(
    projection_ptr,
    rows,
    channels,
    tokens,
    BF16_CUBE: tl.constexpr,
    ALIGNED_ROWS: tl.constexpr,
):
    block_n: tl.constexpr = 64 if BF16_CUBE else 32
    offsets = rows[:, None] * block_n + channels[None, :]
    mask = None if ALIGNED_ROWS else rows[:, None] < tokens
    result = _mhc_load(projection_ptr + offsets, mask)
    if BF16_CUBE:
        result += _mhc_load(projection_ptr + offsets + 32, mask)
    return result


@triton.jit
def _mhc_sinkhorn_round(comb, RECIPROCAL: tl.constexpr, COMPACT: tl.constexpr):
    if COMPACT:
        row_sum = tl.sum(comb, 1) + 1e-6
        if RECIPROCAL:
            comb = comb * (1.0 / row_sum)[:, None, :]
        else:
            comb = comb / row_sum[:, None, :]
        col_sum = tl.sum(comb, 0) + 1e-6
        if RECIPROCAL:
            return comb * (1.0 / col_sum)[None, :, :]
        return comb / col_sum[None, :, :]
    if RECIPROCAL:
        comb = comb * (1.0 / (tl.sum(comb, 2) + 1e-6))[:, :, None]
        return comb * (1.0 / (tl.sum(comb, 1) + 1e-6))[:, None, :]
    comb = comb / (tl.sum(comb, 2)[:, :, None] + 1e-6)
    return comb / (tl.sum(comb, 1)[:, None, :] + 1e-6)


@triton.jit
def _mhc_pre_norm_tile(
    current_ptr,
    stats_ptr,
    projection_ptr,
    scale_ptr,
    base_ptr,
    norm_ptr,
    prev_pre_ptr,
    new_pre_ptr,
    new_post_ptr,
    new_comb_ptr,
    output_ptr,
    rows,
    tokens,
    rms_eps,
    HIDDEN: tl.constexpr,
    H_TILES: tl.constexpr,
    STATS_TILE: tl.constexpr,
    SHIFTED: tl.constexpr,
    ROWS: tl.constexpr,
    H_PAD: tl.constexpr,
    BF16_CUBE: tl.constexpr,
    ALIGNED_ROWS: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    MERGE_PROJECTION: tl.constexpr,
    COMPACT_SINKHORN: tl.constexpr,
    EXACT_SINKHORN: tl.constexpr,
    OUTPUT: tl.constexpr,
    EARLY_SHIFTED_RMS: tl.constexpr = False,
):
    write_mixes: tl.constexpr = OUTPUT != "sinkhorn"
    write_comb: tl.constexpr = OUTPUT != "mix"
    compact: tl.constexpr = COMPACT_SINKHORN or OUTPUT == "sinkhorn"
    live = None if ALIGNED_ROWS else rows[:, None] < tokens
    parts = tl.arange(0, STATS_TILE)
    if ALIGNED_ROWS:
        stats_mask = None if STATS_TILE == H_TILES else parts[None, :] < H_TILES
    else:
        stats_mask = live & (parts[None, :] < H_TILES)
    squares = _mhc_load(
        stats_ptr + rows[:, None] * H_TILES + parts[None, :], stats_mask
    )
    inv_hc_rms = tl.rsqrt(tl.sum(squares, 1) * (1.0 / (4 * HIDDEN)) + rms_eps)
    mix_ids = tl.arange(0, 4)
    comb_ids = tl.arange(0, 16)
    if MERGE_PROJECTION:
        tl.static_assert(BF16_CUBE)
        columns = tl.arange(0, 32)
        offset = rows[:, None] * 64 + columns[None, :]
        logits = _mhc_load(projection_ptr + offset, live)
        logits += _mhc_load(projection_ptr + offset + 32, live)
        if write_mixes:
            pre = tle.dsa.extract_slice(logits, (0, 0), (ROWS, 4), (1, 1))
            post = tle.dsa.extract_slice(logits, (0, 4), (ROWS, 4), (1, 1))
        if write_comb:
            comb = tle.dsa.extract_slice(logits, (0, 8), (ROWS, 16), (1, 1))
    else:
        if write_mixes:
            pre = _projection_slice(
                projection_ptr, rows, mix_ids, tokens, BF16_CUBE, ALIGNED_ROWS
            )
            post = _projection_slice(
                projection_ptr,
                rows,
                mix_ids + 4,
                tokens,
                BF16_CUBE,
                ALIGNED_ROWS,
            )
        if write_comb:
            comb = _projection_slice(
                projection_ptr,
                rows,
                comb_ids + 8,
                tokens,
                BF16_CUBE,
                ALIGNED_ROWS,
            )
    if write_mixes:
        pre = (
            pre * inv_hc_rms[:, None] * tl.load(scale_ptr)
            + tl.load(base_ptr + mix_ids)[None, :]
        )
        post = (
            post * inv_hc_rms[:, None] * tl.load(scale_ptr + 1)
            + tl.load(base_ptr + 4 + mix_ids)[None, :]
        )
        pre = 1.0 / (1.0 + tl.exp(-pre)) + 1e-6
        post = 2.0 / (1.0 + tl.exp(-post))
    if write_comb:
        comb = (
            comb * inv_hc_rms[:, None] * tl.load(scale_ptr + 2)
            + tl.load(base_ptr + 8 + comb_ids)[None, :]
        )
        comb = tl.reshape(comb, (ROWS, 4, 4))
        if compact:
            comb = tl.trans(comb, (1, 2, 0))
        if compact:
            comb = tl.exp(comb - tl.max(comb, 1)[:, None, :])
            comb = comb / tl.sum(comb, 1)[:, None, :] + 1e-6
            comb = comb / (tl.sum(comb, 0)[None, :, :] + 1e-6)
        else:
            comb = tl.exp(comb - tl.max(comb, 2)[:, :, None])
            comb = comb / tl.sum(comb, 2)[:, :, None] + 1e-6
            comb = comb / (tl.sum(comb, 1)[:, None, :] + 1e-6)
        if EXACT_SINKHORN:
            for _ in range(6):
                comb = _mhc_sinkhorn_round(comb, RECIPROCAL, compact)
            previous = comb
            comb = _mhc_sinkhorn_round(comb, RECIPROCAL, compact)
            comb = _mhc_sinkhorn_round(comb, RECIPROCAL, compact)
            # An exact two-round cycle can skip an even number of rounds.
            delta = tl.reshape(tl.abs(comb - previous), (ROWS * 16,))
            if tl.max(delta, 0) != 0.0:
                for _ in range(10):
                    comb = _mhc_sinkhorn_round(comb, RECIPROCAL, compact)
            comb = _mhc_sinkhorn_round(comb, RECIPROCAL, compact)
        else:
            for _ in range(19):
                comb = _mhc_sinkhorn_round(comb, RECIPROCAL, compact)
        if compact:
            comb = tl.trans(comb, (2, 0, 1))
    if write_mixes:
        tl.store(new_post_ptr + rows[:, None] * 4 + mix_ids[None, :], post, live)
    if write_comb:
        tl.store(
            new_comb_ptr + rows[:, None] * 16 + comb_ids[None, :],
            tl.reshape(comb, (ROWS, 16)),
            live,
        )
    if write_mixes:
        if SHIFTED:
            tl.store(new_pre_ptr + rows[:, None] * 4 + mix_ids[None, :], pre, live)
            if not EARLY_SHIFTED_RMS:
                pre = _mhc_load(
                    prev_pre_ptr + rows[:, None] * 4 + mix_ids[None, :], live
                )

        if not EARLY_SHIFTED_RMS:
            _mhc_apply_norm(
                current_ptr,
                norm_ptr,
                output_ptr,
                rows,
                tokens,
                rms_eps,
                pre,
                HIDDEN,
                H_PAD,
                ROWS,
                ALIGNED_ROWS,
            )


@triton.jit
def _mhc_apply_norm(
    current_ptr,
    norm_ptr,
    output_ptr,
    rows,
    tokens,
    rms_eps,
    pre,
    HIDDEN: tl.constexpr,
    H_PAD: tl.constexpr,
    ROWS: tl.constexpr,
    ALIGNED_ROWS: tl.constexpr,
):
    live = None if ALIGNED_ROWS else rows[:, None] < tokens
    mix_ids = tl.arange(0, 4)
    columns = tl.arange(0, H_PAD)
    column_mask = columns < HIDDEN
    if ALIGNED_ROWS:
        mask = None if H_PAD == HIDDEN else column_mask[None, :]
    else:
        mask = live & column_mask[None, :]
    mixed = tl.full((ROWS, H_PAD), 0.0, tl.float32)
    for lane in tl.static_range(4):
        coeff = tl.sum(tl.where(mix_ids[None, :] == lane, pre, 0.0), 1)
        current = _mhc_load(
            current_ptr + (rows[:, None] * 4 + lane) * HIDDEN + columns[None, :],
            mask,
        )
        mixed += coeff[:, None] * current.to(tl.float32)
    mixed = mixed.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(mixed * mixed, 1) * (1.0 / HIDDEN) + rms_eps)
    weight = _mhc_load(
        norm_ptr + columns, None if ALIGNED_ROWS and H_PAD == HIDDEN else column_mask
    ).to(tl.float32)
    y = mixed * inv_rms[:, None] * weight[None, :]
    tl.store(output_ptr + rows[:, None] * HIDDEN + columns[None, :], y, mask)


@triton.jit
def _mhc_post_range(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    current_ptr,
    cube_input_ptr,
    stats_ptr,
    first_row,
    start,
    end,
    step,
    tokens,
    cube_rows,
    HIDDEN: tl.constexpr,
    POST_ROWS: tl.constexpr,
    POST_H: tl.constexpr,
    BF16_CUBE: tl.constexpr,
    ALIGNED_ROWS: tl.constexpr,
    write_current=True,
):
    h_tiles: tl.constexpr = HIDDEN // POST_H
    for task in range(start, end, step):
        h_tile = task % h_tiles
        rows = first_row + task // h_tiles * POST_ROWS + tl.arange(0, POST_ROWS)
        columns = h_tile * POST_H + tl.arange(0, POST_H)
        _mhc_post_tile(
            x_ptr,
            residual_ptr,
            post_ptr,
            comb_ptr,
            current_ptr,
            cube_input_ptr,
            stats_ptr,
            rows,
            columns,
            h_tile,
            tokens,
            cube_rows,
            HIDDEN,
            h_tiles,
            BF16_CUBE,
            ALIGNED_ROWS,
            write_current,
        )


@triton.jit
def _mega_mhc_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    norm_ptr,
    prev_pre_ptr,
    current_ptr,
    new_post_ptr,
    new_comb_ptr,
    output_ptr,
    new_pre_ptr,
    stats_ptr,
    projection_ptr,
    cube_input_ptr,
    packed_fn_ptr,
    tokens,
    rms_eps,
    SPLIT_HEADS: tl.constexpr,
    DIRECT_PIPELINE: tl.constexpr,
    HIDDEN: tl.constexpr,
    H_PAD: tl.constexpr,
    SHIFTED: tl.constexpr,
    POST_ROWS: tl.constexpr,
    POST_H: tl.constexpr,
    STATS_TILE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PACK_K: tl.constexpr,
    OUT_ROWS: tl.constexpr,
    EXPLICIT_VECTOR: tl.constexpr,
    ALIGNED_ROWS: tl.constexpr,
    BALANCED_ROWS: tl.constexpr,
    BF16_CUBE: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    MERGE_PROJECTION: tl.constexpr,
    COMPACT_SINKHORN: tl.constexpr,
    EXACT_SINKHORN: tl.constexpr,
    SPLIT_VECTOR_TAIL: tl.constexpr,
    SHARE_RMS: tl.constexpr,
):
    pipelined: tl.constexpr = EXPLICIT_VECTOR and (BF16_CUBE or DIRECT_PIPELINE)
    early_rms: tl.constexpr = SHIFTED and not pipelined and EXPLICIT_VECTOR
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    head_start = 0
    if SPLIT_HEADS:
        # Each pair stages one copy of the token. The two 16-head projections
        # cover the independent pre/post and Sinkhorn outputs, respectively.
        group = worker
        cube_input_ptr += group * 4 * HIDDEN
        stats_ptr += group * (HIDDEN // POST_H)
        projection_ptr += group * 32
        head_start = group * 8
        worker = 0
        workers = 1
    if EXPLICIT_VECTOR:
        # Each Vector subcore processes complete tiles, including its own stores.
        sub = tle.dsa.ascend.sub_vec_id()
        local_vector_step: tl.constexpr = 2
    else:
        sub = 0
        local_vector_step: tl.constexpr = 1
    h_tiles: tl.constexpr = HIDDEN // POST_H
    if BF16_CUBE:
        width: tl.constexpr = 4 * HIDDEN
        pack_tiles: tl.constexpr = width // PACK_K
        # Each Cube reads only the weights prepared by its own Vector pair.
        # The first tile's Vector-to-Cube handoff also orders these MTE3 stores.
        packed_fn_ptr += worker * 64 * width
        for pack_task in range(sub, 8 * pack_tiles, local_vector_step):
            k = (pack_task % pack_tiles) * PACK_K + tl.arange(0, PACK_K)
            channels = (pack_task // pack_tiles) * 4 + tl.arange(0, 4)
            offsets = channels[:, None] * width + k[None, :]
            value = tl.load(fn_ptr + offsets, channels[:, None] < 24, other=0)
            # The two BF16 weight planes share one Cube dot and FP32 accumulation.
            high = value.to(tl.bfloat16)
            tl.store(packed_fn_ptr + offsets, high)
            tl.store(packed_fn_ptr + offsets + 32 * width, value - high.to(tl.float32))
        cube_rows = tl.cdiv(tokens, BLOCK_M) * BLOCK_M
        weights = packed_fn_ptr
    else:
        cube_rows = tokens
        weights = fn_ptr
    row_tiles = tl.cdiv(tokens, BLOCK_M)
    # A program owns its output rows through post, projection and normalization.
    # Pair-local handoffs therefore make progress independently on different streams.
    if BALANCED_ROWS:
        # Four-row boundaries keep post and output tiles within this core pair.
        units = tl.cdiv(tokens, 4)
        first_row_of_worker = (units * worker // workers) * 4
        end_row_of_worker = (units * (worker + 1) // workers) * 4
        first_tile = 0
        last_tile = tl.cdiv(end_row_of_worker - first_row_of_worker, BLOCK_M)
        tile_step = 1
        projection_end = tl.minimum(end_row_of_worker, tokens)
    else:
        first_tile, last_tile, tile_step = worker, row_tiles, workers
        first_row_of_worker = worker * BLOCK_M
        end_row_of_worker = cube_rows
        projection_end = tokens
    cube_tile = current_ptr if BF16_CUBE else cube_input_ptr
    if pipelined and first_tile < last_tile:
        first_rows = (
            tl.minimum(BLOCK_M, end_row_of_worker - first_row_of_worker)
            if BALANCED_ROWS
            else BLOCK_M
        )
        _mhc_post_range(
            x_ptr,
            residual_ptr,
            post_ptr,
            comb_ptr,
            current_ptr,
            cube_tile,
            stats_ptr,
            first_row_of_worker,
            sub,
            tl.cdiv(first_rows, POST_ROWS) * h_tiles,
            local_vector_step,
            tokens,
            cube_rows,
            HIDDEN,
            POST_ROWS,
            POST_H,
            BF16_CUBE,
            ALIGNED_ROWS,
        )
        tle.dsa.ascend.sync_block_set("vector", "cube", 2)
    for tile in range(first_tile, last_tile, tile_step):
        if BALANCED_ROWS:
            first_row = first_row_of_worker + tile * BLOCK_M
            active_rows = tl.minimum(BLOCK_M, end_row_of_worker - first_row)
        else:
            first_row = tile * BLOCK_M
            active_rows = BLOCK_M
        post_start, post_end, post_step = (
            sub,
            tl.cdiv(active_rows, POST_ROWS) * h_tiles,
            local_vector_step,
        )
        out_start, out_end, out_step = (
            sub,
            tl.cdiv(active_rows, OUT_ROWS),
            local_vector_step,
        )
        if not pipelined:
            _mhc_post_range(
                x_ptr,
                residual_ptr,
                post_ptr,
                comb_ptr,
                current_ptr,
                cube_tile,
                stats_ptr,
                first_row,
                post_start,
                post_end,
                post_step,
                tokens,
                cube_rows,
                HIDDEN,
                POST_ROWS,
                POST_H,
                BF16_CUBE,
                ALIGNED_ROWS,
                not SPLIT_HEADS or head_start == 0,
            )
            tle.dsa.ascend.sync_block_set("vector", "cube", 2)
        odd_slot = tile % 2 != 0 if BALANCED_ROWS else tile // workers % 2 != 0
        if pipelined and odd_slot:
            tle.dsa.ascend.sync_block_wait("vector", "cube", 5)
        else:
            tle.dsa.ascend.sync_block_wait("vector", "cube", 2)

        project_rows = first_row + tl.arange(0, BLOCK_M)
        _mhc_project_tile(
            cube_tile,
            weights,
            projection_ptr,
            project_rows,
            projection_end,
            head_start,
            SPLIT_HEADS,
            HIDDEN,
            BLOCK_M,
            BLOCK_K,
            BF16_CUBE,
            BALANCED_ROWS,
        )
        if pipelined and odd_slot:
            tle.dsa.ascend.sync_block_set("cube", "vector", 6)
        else:
            tle.dsa.ascend.sync_block_set("cube", "vector", 3)
        if pipelined:
            next_tile = tile + tile_step
            if next_tile < last_tile:
                if BALANCED_ROWS:
                    next_row = first_row_of_worker + next_tile * BLOCK_M
                    next_rows = tl.minimum(BLOCK_M, end_row_of_worker - next_row)
                else:
                    next_row = next_tile * BLOCK_M
                    next_rows = BLOCK_M
                _mhc_post_range(
                    x_ptr,
                    residual_ptr,
                    post_ptr,
                    comb_ptr,
                    current_ptr,
                    cube_tile,
                    stats_ptr,
                    next_row,
                    sub,
                    tl.cdiv(next_rows, POST_ROWS) * h_tiles,
                    local_vector_step,
                    tokens,
                    cube_rows,
                    HIDDEN,
                    POST_ROWS,
                    POST_H,
                    BF16_CUBE,
                    ALIGNED_ROWS,
                )
                # Alternate the event IDs since two token tiles can be in flight.
                if odd_slot:
                    tle.dsa.ascend.sync_block_set("vector", "cube", 2)
                else:
                    tle.dsa.ascend.sync_block_set("vector", "cube", 5)
        if early_rms and (not SPLIT_HEADS or head_start == 0):
            # Both Vector halves may read residual columns written by their partner.
            tle.dsa.ascend.sync_block_all("all_sub_vector", 0)
            # The shifted output depends only on the previous pre mix.
            lanes = tl.arange(0, 4)
            for rms_task in range(out_start, out_end, out_step):
                rms_rows = first_row + rms_task * OUT_ROWS + tl.arange(0, OUT_ROWS)
                live = rms_rows[:, None] < tokens
                pre = _mhc_load(
                    prev_pre_ptr + rms_rows[:, None] * 4 + lanes[None, :], live
                )
                _mhc_apply_norm(
                    current_ptr,
                    norm_ptr,
                    output_ptr,
                    rms_rows,
                    tokens,
                    rms_eps,
                    pre,
                    HIDDEN,
                    H_PAD,
                    OUT_ROWS,
                    False,
                )
        if pipelined and odd_slot:
            tle.dsa.ascend.sync_block_wait("cube", "vector", 6)
        else:
            tle.dsa.ascend.sync_block_wait("cube", "vector", 3)

        if SPLIT_HEADS:
            # Post used both Vector halves. One half now consumes this group's
            # projection and writes the outputs owned by the group.
            active_rows = 1 if sub == 0 else 0
        if SPLIT_VECTOR_TAIL:
            # Both subcores share the completed Cube projection. One can normalize
            # a wider Sinkhorn batch while the other applies the pre mix and RMS.
            output_end = tl.minimum(tokens, first_row + active_rows)
            if SHARE_RMS:
                if sub != 0:
                    for sink_task in range(tl.cdiv(active_rows, 8)):
                        out_rows = first_row + sink_task * 8 + tl.arange(0, 8)
                        _mhc_pre_norm_tile(
                            current_ptr,
                            stats_ptr,
                            projection_ptr,
                            scale_ptr,
                            base_ptr,
                            norm_ptr,
                            prev_pre_ptr,
                            new_pre_ptr,
                            new_post_ptr,
                            new_comb_ptr,
                            output_ptr,
                            out_rows,
                            output_end,
                            rms_eps,
                            HIDDEN,
                            h_tiles,
                            STATS_TILE,
                            SHIFTED,
                            8,
                            H_PAD,
                            BF16_CUBE,
                            False,
                            RECIPROCAL,
                            False,
                            False,
                            EXACT_SINKHORN,
                            "sinkhorn",
                        )
                mix_tasks = tl.cdiv(active_rows, 4)
                # The Sinkhorn subcore handles the last few RMS tiles after normalization.
                helper_start = mix_tasks - mix_tasks // 3
                mix_start = 0 if sub == 0 else helper_start
                mix_end = helper_start if sub == 0 else mix_tasks
                for mix_task in range(mix_start, mix_end):
                    out_rows = first_row + mix_task * 4 + tl.arange(0, 4)
                    _mhc_pre_norm_tile(
                        current_ptr,
                        stats_ptr,
                        projection_ptr,
                        scale_ptr,
                        base_ptr,
                        norm_ptr,
                        prev_pre_ptr,
                        new_pre_ptr,
                        new_post_ptr,
                        new_comb_ptr,
                        output_ptr,
                        out_rows,
                        output_end,
                        rms_eps,
                        HIDDEN,
                        h_tiles,
                        STATS_TILE,
                        SHIFTED,
                        4,
                        H_PAD,
                        BF16_CUBE,
                        ALIGNED_ROWS,
                        RECIPROCAL,
                        False,
                        False,
                        EXACT_SINKHORN,
                        "mix",
                    )
            else:
                # Masked loads of both projection planes need extra Vector buffer space.
                masked_bf16: tl.constexpr = BF16_CUBE and not ALIGNED_ROWS
                mix_rows: tl.constexpr = 1 if SPLIT_HEADS else 2 if masked_bf16 else 4
                sink_rows: tl.constexpr = 1 if SPLIT_HEADS else 4 if masked_bf16 else 8
                write_mixes = head_start == 0 if SPLIT_HEADS else sub == 0
                if write_mixes:
                    for mix_task in range(tl.cdiv(active_rows, mix_rows)):
                        out_rows = (
                            first_row + mix_task * mix_rows + tl.arange(0, mix_rows)
                        )
                        _mhc_pre_norm_tile(
                            current_ptr,
                            stats_ptr,
                            projection_ptr,
                            scale_ptr,
                            base_ptr,
                            norm_ptr,
                            prev_pre_ptr,
                            new_pre_ptr,
                            new_post_ptr,
                            new_comb_ptr,
                            output_ptr,
                            out_rows,
                            output_end,
                            rms_eps,
                            HIDDEN,
                            h_tiles,
                            STATS_TILE,
                            SHIFTED,
                            mix_rows,
                            H_PAD,
                            BF16_CUBE,
                            ALIGNED_ROWS,
                            RECIPROCAL,
                            False,
                            False,
                            EXACT_SINKHORN,
                            "mix",
                            early_rms,
                        )
                else:
                    for sink_task in range(tl.cdiv(active_rows, sink_rows)):
                        out_rows = (
                            first_row + sink_task * sink_rows + tl.arange(0, sink_rows)
                        )
                        _mhc_pre_norm_tile(
                            current_ptr,
                            stats_ptr,
                            projection_ptr,
                            scale_ptr,
                            base_ptr,
                            norm_ptr,
                            prev_pre_ptr,
                            new_pre_ptr,
                            new_post_ptr,
                            new_comb_ptr,
                            output_ptr,
                            out_rows,
                            output_end,
                            rms_eps,
                            HIDDEN,
                            h_tiles,
                            STATS_TILE,
                            SHIFTED,
                            sink_rows,
                            H_PAD,
                            BF16_CUBE,
                            False,
                            RECIPROCAL,
                            False,
                            False,
                            EXACT_SINKHORN,
                            "sinkhorn",
                        )
        else:
            for out_task in range(out_start, out_end, out_step):
                out_rows = first_row + out_task * OUT_ROWS + tl.arange(0, OUT_ROWS)
                _mhc_pre_norm_tile(
                    current_ptr,
                    stats_ptr,
                    projection_ptr,
                    scale_ptr,
                    base_ptr,
                    norm_ptr,
                    prev_pre_ptr,
                    new_pre_ptr,
                    new_post_ptr,
                    new_comb_ptr,
                    output_ptr,
                    out_rows,
                    tokens,
                    rms_eps,
                    HIDDEN,
                    h_tiles,
                    STATS_TILE,
                    SHIFTED,
                    OUT_ROWS,
                    H_PAD,
                    BF16_CUBE,
                    ALIGNED_ROWS,
                    RECIPROCAL,
                    MERGE_PROJECTION,
                    COMPACT_SINKHORN,
                    EXACT_SINKHORN,
                    "all",
                    early_rms,
                )


@dataclass(frozen=True, eq=False, repr=False)
class MegaMHCContext:
    """Prepared launch and scratch storage for one layer, shape and NPU stream.

    Reuse the context on its owning thread and stream. Keep it alive until
    queued work finishes. Capture and replay Graphs on the owning stream.
    """

    shape: tuple[int, int]
    device: torch.device
    rms_eps: float
    shifted: bool
    _weights: tuple[torch.Tensor, ...]
    _weight_ptrs: tuple[int, ...]
    _owner: int
    _stream: int
    _cube_rows: int | None
    _runner: Callable
    _launch_tail: tuple
    _input_alignment: tuple[tuple[int, int], ...]


def _validate_inputs(x, residual, post, comb, fn, scale, base, norm_weight, pre_mix):
    assert x.ndim == 2, "x must have shape [T, H]"
    tokens, hidden = x.shape
    assert (
        128 <= hidden <= 8192 and hidden % 128 == 0
    ), "H must align to 128 in [128, 8192]"
    assert residual.shape == (tokens, 4, hidden), "residual must have shape [T, 4, H]"
    assert post.shape in (
        (tokens, 4),
        (tokens, 4, 1),
    ), "post must have shape [T, 4] or [T, 4, 1]"
    assert comb.shape == (tokens, 4, 4), "comb must have shape [T, 4, 4]"
    assert fn.shape == (24, 4 * hidden), "fn must have shape [24, 4H]"
    assert scale.shape == (3,) and base.shape == (
        24,
    ), "scale and base must have shapes [3] and [24]"
    assert norm_weight.shape == (hidden,), "norm_weight must have shape [H]"
    assert pre_mix is None or pre_mix.shape == (
        tokens,
        4,
    ), "pre_mix must have shape [T, 4]"
    bf16 = (x, residual, norm_weight)
    fp32 = (
        (post, comb, fn, scale, base)
        if pre_mix is None
        else (post, comb, fn, scale, base, pre_mix)
    )
    assert all(
        t.dtype == torch.bfloat16 for t in bf16
    ), "x, residual and norm_weight must be BF16"
    assert all(
        t.dtype == torch.float32 for t in fp32
    ), "mixes and projection parameters must be FP32"
    assert x.device.type == "npu" and all(
        t.device == x.device and t.is_contiguous() for t in (*bf16, *fp32)
    ), "inputs must be contiguous and on the same NPU"


def _allocate_outputs(x, residual, post, comb, pre_mix, cube_rows):
    tokens, hidden = x.shape
    current = (
        torch.empty((cube_rows, 4, hidden), dtype=x.dtype, device=x.device)[:tokens]
        if cube_rows is not None
        else torch.empty_like(residual)
    )
    new_post = torch.empty_like(post)
    new_comb = torch.empty_like(comb)
    output = torch.empty_like(x)
    new_pre = None if pre_mix is None else torch.empty_like(pre_mix)
    return current, new_post, new_comb, output, new_pre


def _launch_config(tokens, hidden, shifted, cores, vector_cores):
    projection_k = 4 * hidden
    k_alignment = projection_k & -projection_k
    h_pad = triton.next_power_of_2(hidden)
    # Four RMS rows fit within the tuned 16K-element Vector tile budget.
    four_row_tail = 4 * h_pad <= 16384
    # Column masks need extra UB space in the split epilogue. Leave half
    # the tile budget free when the hidden dimension needs padding.
    split_tail_fits = four_row_tail and (hidden == h_pad or 8 * h_pad <= 16384)
    wide_single_token = tokens == 1 and hidden >= 4096 and hidden % 1024 == 0
    tiny_tile = tokens <= 4 and (hidden % 2048 == 0 or wide_single_token)
    explicit_vector = (tiny_tile or tokens > 4) and vector_cores == 2 * cores
    split_heads = wide_single_token and hidden > 4096 and explicit_vector
    tiny_tile = tiny_tile and explicit_vector
    short_aligned_tile = (
        explicit_vector
        and hidden >= 4096
        and 4 < tokens <= cores * 2
        and tokens % 16 == 8
    )
    small_pipeline_tile = explicit_vector and cores * 10 < tokens <= cores * 64
    medium_tile = explicit_vector and cores * 8 < tokens <= cores * 10
    if tiny_tile:
        block_m = 1 if tokens <= 2 else 4
    elif short_aligned_tile:
        block_m = 8
    elif small_pipeline_tile:
        rows_with_eight = 8 * triton.cdiv(tokens, 8 * cores)
        rows_with_sixteen = 16 * triton.cdiv(tokens, 16 * cores)
        # Use enough tiles to start a second wave, then limit the busiest core's work.
        block_m = (
            8 if tokens <= cores * 16 or rows_with_eight < rows_with_sixteen else 16
        )
    elif medium_tile:
        # A second wave of small tiles keeps the handoff inside each core pair.
        block_m = 8 if hidden >= 4096 else 32
    elif explicit_vector and tokens > cores * 64:
        rows_with_sixty_four = 64 * triton.cdiv(tokens, 64 * cores)
        rows_with_sixteen = 16 * triton.cdiv(tokens, 16 * cores)
        # Use smaller tiles when they substantially reduce the busiest core's work.
        block_m = 16 if rows_with_sixteen * 16 <= rows_with_sixty_four * 15 else 64
        if (
            split_tail_fits
            and block_m == 64
            and tokens >= cores * 256
            and tokens % 64 == 0
        ):
            # Shorter aligned tiles overlap the Vector tail with more Cube waves.
            block_m = 32
    else:
        block_m = 1 if tokens == 1 else 16 if tokens <= cores * 16 else 64
    # Use contiguous row groups when cyclic tiles leave an uneven last wave.
    rows_per_core = block_m * triton.cdiv(tokens, block_m * cores)
    balanced_rows_per_core = 4 * triton.cdiv(tokens, 4 * cores)
    balance_rows = (
        explicit_vector
        and four_row_tail
        and tokens > cores * 16
        and (tokens <= cores * 32 or tokens > cores * 64)
        and balanced_rows_per_core * 16 <= rows_per_core * 15
    )
    if balance_rows:
        block_m = 16 if tokens <= cores * 64 else 32
    if explicit_vector and split_tail_fits and cores * 32 < tokens <= cores * 64:
        # Keep both Vector subcores paired with each Cube tile in this range.
        balance_rows = True
        block_m = 32
    row_tiles = triton.cdiv(tokens, block_m)
    # A single wave of direct FP32 projection avoids packing a private copy of
    # both BF16 weight planes on every core for at most 16 rows per core.
    direct_wave = (
        explicit_vector and split_tail_fits and not shifted and tokens <= cores * 16
    )
    # The next Post tile can run on Vector while Cube consumes the previous
    # FP32 residual tile. Eight-row tiles give medium batches multiple waves.
    direct_pipeline = (
        explicit_vector
        and not shifted
        and hidden == 4096
        and cores * 8 < tokens <= cores * 24
    )
    bf16_cube = (row_tiles > cores or balance_rows) and not (
        direct_wave or direct_pipeline
    )
    if direct_pipeline:
        block_m = 8
        row_tiles = triton.cdiv(tokens, block_m)
    elif not bf16_cube:
        # A second wave just for a few rows delays the last core pair.
        block_m = (
            1
            if tokens <= cores
            else max(4, triton.next_power_of_2(triton.cdiv(tokens, cores)))
        )
        row_tiles = triton.cdiv(tokens, block_m)
    cube_rows = row_tiles * block_m if bf16_cube else tokens
    block_k = min(2048 if split_heads else 1024, k_alignment)
    post_h_limit = (
        2048
        if explicit_vector
        else 1024 if bf16_cube else 512 if tokens > cores else 256
    )
    post_h = min(512 if split_heads else post_h_limit, hidden & -hidden)
    post_rows = min(4, 2048 // post_h, triton.next_power_of_2(tokens))
    out_rows = 2 if four_row_tail and tokens > 1 else 1
    if explicit_vector and (bf16_cube or direct_pipeline) and four_row_tail:
        out_rows = 4
    post_rows = min(post_rows, block_m)
    out_rows = min(out_rows, block_m)
    h_tiles = hidden // post_h
    aligned_rows = tokens % block_m == 0
    if balance_rows:
        aligned_rows = tokens % 4 == 0
    workers = min(cores, row_tiles)
    split_vector_tail = (
        explicit_vector
        and split_tail_fits
        and (
            direct_pipeline
            or (direct_wave and block_m >= 4)
            or (bf16_cube and (medium_tile or tokens > cores * 16))
        )
    )
    if split_heads:
        workers = 2
        split_vector_tail = True
    options = dict(
        SPLIT_HEADS=split_heads,
        DIRECT_PIPELINE=direct_pipeline,
        HIDDEN=hidden,
        H_PAD=h_pad,
        SHIFTED=shifted,
        POST_ROWS=post_rows,
        POST_H=post_h,
        STATS_TILE=triton.next_power_of_2(h_tiles),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        PACK_K=min(4096, k_alignment),
        OUT_ROWS=out_rows,
        EXPLICIT_VECTOR=explicit_vector,
        ALIGNED_ROWS=aligned_rows,
        BALANCED_ROWS=balance_rows,
        BF16_CUBE=bf16_cube,
        RECIPROCAL=(
            (bf16_cube or direct_pipeline)
            and hidden >= 4096
            and not medium_tile
            and tokens <= cores * 32
        ),
        MERGE_PROJECTION=bf16_cube
        and hidden >= 4096
        and cores * 16 < tokens <= cores * 32,
        EXACT_SINKHORN=bf16_cube
        or split_vector_tail
        or tiny_tile
        or short_aligned_tile
        or (explicit_vector and tokens <= cores and hidden >= 4096),
        COMPACT_SINKHORN=bf16_cube
        and explicit_vector
        and out_rows == 4
        and tokens > cores * 64,
        SPLIT_VECTOR_TAIL=split_vector_tail,
        SHARE_RMS=split_vector_tail and tokens > cores * 32,
        enable_auto_bind_sub_block=False if explicit_vector else None,
        enable_sync_block_lock=True,
        disable_auto_inject_block_sync=True,
        enable_reduce_sum_strength=explicit_vector and tokens > cores * 8,
        enable_preload=True if medium_tile else None,
        use_bytecode=False,
    )
    return (workers,), cube_rows, options


def _prepare_launch(x, residual, post, comb, pre_mix):
    tokens, hidden = x.shape
    utils = driver.active.utils
    grid, cube_rows, options = _launch_config(
        tokens,
        hidden,
        pre_mix is not None,
        utils.get_aicore_num(),
        utils.get_aivector_core_num(),
    )
    bf16_cube = options["BF16_CUBE"]
    # Independent head groups each need one row of residuals and statistics.
    scratch_rows = cube_rows * (2 if options["SPLIT_HEADS"] else 1)
    output_rows = cube_rows if bf16_cube else None
    outputs = _allocate_outputs(x, residual, post, comb, pre_mix, output_rows)
    device = x.device
    projection_width = 64 if bf16_cube else 32
    stats = torch.empty(
        (2 if options["SPLIT_HEADS"] else tokens, hidden // options["POST_H"]),
        dtype=torch.float32,
        device=device,
    )
    projection = torch.empty(
        (scratch_rows, projection_width),
        dtype=torch.float32,
        device=device,
    )
    cube_input = (
        None
        if bf16_cube
        else torch.empty((scratch_rows, 4, hidden), dtype=torch.float32, device=device)
    )
    packed_fn = (
        torch.empty(
            (grid[0], projection_width, 4 * hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        if bf16_cube
        else None
    )
    return (
        grid,
        output_rows,
        outputs,
        (stats, projection, cube_input, packed_fn),
        options,
    )


def _validate_context(context, inputs, rms_eps):
    assert isinstance(context, MegaMHCContext), "context must be a MegaMHCContext"
    x = inputs[0]
    assert (
        x.shape == context.shape
        and x.device == context.device
        and rms_eps == context.rms_eps
        and (inputs[-1] is not None) == context.shifted
    ), "input shape, device, epsilon and mode must match the prepared context"
    assert threading.get_ident() == context._owner, "context belongs to another thread"
    npu = driver.active
    device_index = npu.get_current_device()
    assert device_index == context.device.index, "context belongs to another device"
    assert (
        npu.get_current_stream(device_index) == context._stream
    ), "context belongs to another stream"
    assert all(
        weight is fixed and weight.data_ptr() == pointer
        for weight, fixed, pointer in zip(
            inputs[4:8], context._weights, context._weight_ptrs
        )
    ), "weights or their storage changed; rebuild the context"
    assert all(
        inputs[slot].data_ptr() % alignment == 0
        for slot, alignment in context._input_alignment
    ), "tensor alignment must match the prepared kernel"


def create_mega_mhc_context(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    pre_mix: torch.Tensor | None = None,
    *,
    rms_eps: float = 1e-6,
) -> MegaMHCContext:
    """Prepare an exact-shape MegaMHC launch before inference or Graph capture."""
    inputs = (x, residual, post, comb, fn, scale, base, norm_weight, pre_mix)
    _validate_inputs(*inputs)
    assert math.isfinite(rms_eps) and rms_eps > 0, "rms_eps must be positive and finite"
    assert x.shape[0] > 0, "prepared MegaMHC requires at least one token"
    npu = driver.active
    device_index = npu.get_current_device()
    assert x.device.index == device_index, "input must be on the current NPU"
    assert (
        not torch.npu.is_current_stream_capturing()
    ), "prepare MegaMHC before Graph capture"
    assert (
        not _mega_mhc_kernel.pre_run_hooks
    ), "prepared MegaMHC requires no JIT pre-run hooks"

    grid, output_rows, outputs, workspace, options = _prepare_launch(
        x, residual, post, comb, pre_mix
    )
    tail = (*workspace, x.shape[0], rms_eps)
    args = (*inputs, *outputs, *tail)
    compiled = _mega_mhc_kernel.warmup(*args, grid=grid, **options)
    if hasattr(compiled, "result"):
        compiled = compiled.result()
    assert compiled is not None and tuple(compiled.src.signature) == tuple(
        _mega_mhc_kernel.arg_names
    ), "unsupported compiled MegaMHC launcher signature"
    # Reuse the compiled pointer alignments. Weights keep their addresses, and
    # outputs and scratch start at the allocator's aligned base address.
    input_alignment = tuple(
        (slot, alignment)
        for slot in (0, 1, 2, 3, 8)
        for name, alignment in compiled.src.attrs.get((slot,), ())
        if name == "tt.divisibility"
    )
    weights = inputs[4:8]
    constexprs = tuple(
        options[name] for name in _mega_mhc_kernel.arg_names[len(args) :]
    )
    return MegaMHCContext(
        shape=tuple(x.shape),
        device=x.device,
        rms_eps=rms_eps,
        shifted=pre_mix is not None,
        _weights=weights,
        _weight_ptrs=tuple(weight.data_ptr() for weight in weights),
        _owner=threading.get_ident(),
        _stream=npu.get_current_stream(device_index),
        _cube_rows=output_rows,
        _runner=compiled[grid + (1, 1)],
        _launch_tail=(*tail, *constexprs),
        _input_alignment=input_alignment,
    )


def mega_mhc(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    pre_mix: torch.Tensor | None = None,
    *,
    rms_eps: float = 1e-6,
    context: MegaMHCContext | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return updated residual, post/comb mixes, normalized input and optional new pre mix.

    The shifted path aggregates with the supplied pre mix and returns the next one.
    Both paths use four residual streams and twenty Sinkhorn iterations.
    The returned post mix uses the input post mix's shape. An optional context
    reuses the launch and scratch storage, with fresh outputs.
    """
    inputs = (x, residual, post, comb, fn, scale, base, norm_weight, pre_mix)
    _validate_inputs(*inputs)
    if context is None:
        assert (
            math.isfinite(rms_eps) and rms_eps > 0
        ), "rms_eps must be positive and finite"
    else:
        _validate_context(context, inputs, rms_eps)

    tokens = x.shape[0]
    if context is not None:
        outputs = _allocate_outputs(
            x, residual, post, comb, pre_mix, context._cube_rows
        )
        context._runner(
            *inputs, *outputs, *context._launch_tail, stream=context._stream
        )
    elif tokens:
        assert (
            driver.active.get_current_device() == x.device.index
        ), "Run MegaMHC on the input device"
        grid, _, outputs, workspace, options = _prepare_launch(
            x, residual, post, comb, pre_mix
        )
        _mega_mhc_kernel[grid](
            *inputs, *outputs, *workspace, tokens, rms_eps, **options
        )
    else:
        outputs = _allocate_outputs(x, residual, post, comb, pre_mix, None)

    return outputs[:-1] if pre_mix is None else outputs
