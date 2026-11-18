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

"""Ascend-optimized Qwen4 QSA prepare: indexer Q/K, raw ring, main attention.

The generic kernel fits all segments into one launch and rotates with
``tl.split``/``tl.join``.  The Ascend Triton backend cannot lower those ops
(``TritonToUnstructureIncubated`` fails on them), so this implementation keeps
the same arithmetic and runs one program per row instead, addressing the rotate
halves with pointer arithmetic.  Triton cannot slice a tensor, so a row is
handled as three segments: the two rotate halves and the untouched tail.

Layout facts this file relies on:
  * ``cos_sin_cache`` is ``[positions, 2 * PAIR]``: cos first, sin next.
  * A head rotates only its first ``2 * PAIR`` dims in NeoX pairs.
  * ``main_qkv`` packs ``q|gate`` per query head, then the K blocks, then V.
  * The main path rounds the rotate halves to bf16 before rotating (like the
    unfused reference); the indexer path rotates in fp32 and rounds at store.
  * The raw state cache is a per-request ring; a 12-column tail, when present,
    holds three int64 MRoPE positions per row.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

try:  # CANN vector ops: slice a register tile without tl.split/tl.join
    from triton.language.extra.cann.extension import extract_slice, insert_slice
except ImportError:  # pragma: no cover - triton builds without the extension
    extract_slice = None
    insert_slice = None

# The Ascend launcher folds the grid into a single coreDim, which must stay
# within the hardware limit, so launches are capped and the kernels stride.


@triton.jit
def _axis_positions(
    positions_ptr,
    stride_token,
    stride_axis,
    token,
    lane,
    IS_2D: tl.constexpr,
    IS_MROPE: tl.constexpr,
    SEC_H: tl.constexpr,
    SEC_W: tl.constexpr,
):
    """MRoPE interleaved axis map for ``lane`` pair indices."""

    pos_t = tl.load(positions_ptr + token * stride_token)
    if IS_2D:
        pos_h = tl.load(positions_ptr + token * stride_token + stride_axis)
        pos_w = tl.load(positions_ptr + token * stride_token + 2 * stride_axis)
    else:
        pos_h = pos_t
        pos_w = pos_t
    if IS_MROPE:
        axis = lane % 3
        if SEC_H > 0:
            is_h = (axis == 1) & (lane < 3 * SEC_H)
        else:
            is_h = axis == 1
        if SEC_W > 0:
            is_w = (axis == 2) & (lane < 3 * SEC_W)
        else:
            is_w = axis == 2
        return tl.where(is_h, pos_h, tl.where(is_w, pos_w, pos_t))
    return pos_t + lane * 0


@triton.jit
def _rope_table_kernel(
    positions_ptr,
    cos_ptr,
    out_ptr,
    stride_pos_token,
    stride_pos_axis,
    stride_cos_row,
    stride_out_token,
    num_tokens,
    PAIR: tl.constexpr,
    ROTARY: tl.constexpr,
    IS_MROPE: tl.constexpr,
    COS_ROWS: tl.constexpr,
    SEC_H: tl.constexpr,
    SEC_W: tl.constexpr,
    TOKENS_PER_PROG: tl.constexpr,
):
    """Expand the per-token cos|sin row so the hot kernels read it contiguously."""

    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lane = tl.arange(0, PAIR)
    dims = tl.arange(0, ROTARY)
    rows = lane[None, :]
    block = tl.arange(0, TOKENS_PER_PROG)

    for first in tl.range(pid * TOKENS_PER_PROG, num_tokens, nprog * TOKENS_PER_PROG):
        token = first + block
        token_ok = token < num_tokens
        safe_token = tl.minimum(token, num_tokens - 1)
        base_pos = positions_ptr + safe_token * stride_pos_token
        pos_t = tl.load(base_pos)
        pos_h = tl.load(base_pos + stride_pos_axis)
        pos_w = tl.load(base_pos + 2 * stride_pos_axis)
        pos_t = tl.minimum(tl.maximum(pos_t, 0), COS_ROWS - 1).to(tl.int64)
        pos_h = tl.minimum(tl.maximum(pos_h, 0), COS_ROWS - 1).to(tl.int64)
        pos_w = tl.minimum(tl.maximum(pos_w, 0), COS_ROWS - 1).to(tl.int64)
        cos_t = tl.load(cos_ptr + pos_t[:, None] * stride_cos_row + rows).to(tl.float32)
        sin_t = tl.load(cos_ptr + pos_t[:, None] * stride_cos_row + PAIR + rows).to(
            tl.float32
        )
        if IS_MROPE:
            axis = lane % 3
            if SEC_H > 0:
                is_h = (axis == 1) & (lane < 3 * SEC_H)
            else:
                is_h = axis == 1
            if SEC_W > 0:
                is_w = (axis == 2) & (lane < 3 * SEC_W)
            else:
                is_w = axis == 2
            cos_h = tl.load(cos_ptr + pos_h[:, None] * stride_cos_row + rows).to(
                tl.float32
            )
            cos_w = tl.load(cos_ptr + pos_w[:, None] * stride_cos_row + rows).to(
                tl.float32
            )
            sin_h = tl.load(cos_ptr + pos_h[:, None] * stride_cos_row + PAIR + rows).to(
                tl.float32
            )
            sin_w = tl.load(cos_ptr + pos_w[:, None] * stride_cos_row + PAIR + rows).to(
                tl.float32
            )
            cos_half = tl.where(
                is_h[None, :], cos_h, tl.where(is_w[None, :], cos_w, cos_t)
            )
            sin_half = tl.where(
                is_h[None, :], sin_h, tl.where(is_w[None, :], sin_w, sin_t)
            )
        else:
            cos_half = cos_t
            sin_half = sin_t
        cos = tl.reshape(
            tl.broadcast_to(
                tl.reshape(cos_half, (TOKENS_PER_PROG, 1, PAIR)),
                (TOKENS_PER_PROG, 2, PAIR),
            ),
            (TOKENS_PER_PROG, ROTARY),
        )
        sin = tl.reshape(
            tl.broadcast_to(
                tl.reshape(sin_half, (TOKENS_PER_PROG, 1, PAIR)),
                (TOKENS_PER_PROG, 2, PAIR),
            ),
            (TOKENS_PER_PROG, ROTARY),
        )
        base = out_ptr + safe_token[:, None] * stride_out_token
        store_ok = token_ok[:, None]
        tl.store(base + dims[None, :], cos, mask=store_ok)
        tl.store(base + ROTARY + dims[None, :], sin, mask=store_ok)


@triton.jit
def _indexer_q_kernel(
    q_ptr,
    rope_ptr,
    weight_ptr,
    out_ptr,
    stride_q_token,
    stride_q_dim,
    stride_rope_token,
    stride_out_token,
    eps,
    num_tokens,
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    PAIR: tl.constexpr,
    ROTARY: tl.constexpr,
):
    """One program per vector core; each iteration handles a whole token row."""

    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    row = tl.arange(0, NUM_HEADS * D)
    weight = tl.load(weight_ptr + tl.arange(0, D)).to(tl.float32) + 1.0

    for token in tl.range(pid, num_tokens, nprog):
        x = tl.load(q_ptr + token * stride_q_token + row * stride_q_dim).to(tl.float32)
        x = x.reshape(NUM_HEADS, D)
        ss = tl.sum(x * x, axis=1)
        rstd = tl.rsqrt(ss / D + eps)
        y = x * rstd[:, None] * weight[None, :]

        cos = tl.load(rope_ptr + token * stride_rope_token + tl.arange(0, ROTARY))
        sin = tl.load(
            rope_ptr + token * stride_rope_token + ROTARY + tl.arange(0, ROTARY)
        )

        lo = extract_slice(y, offsets=(0, 0), sizes=(NUM_HEADS, PAIR), strides=(1, 1))
        hi = extract_slice(
            y, offsets=(0, PAIR), sizes=(NUM_HEADS, PAIR), strides=(1, 1)
        )
        cat = tl.zeros((NUM_HEADS, ROTARY), dtype=tl.float32)
        cat = insert_slice(
            cat, -hi, offsets=(0, 0), sizes=(NUM_HEADS, PAIR), strides=(1, 1)
        )
        cat = insert_slice(
            cat, lo, offsets=(0, PAIR), sizes=(NUM_HEADS, PAIR), strides=(1, 1)
        )
        orig = extract_slice(
            y, offsets=(0, 0), sizes=(NUM_HEADS, ROTARY), strides=(1, 1)
        )
        y = insert_slice(
            y,
            cat * sin[None, :] + orig * cos[None, :],
            offsets=(0, 0),
            sizes=(NUM_HEADS, ROTARY),
            strides=(1, 1),
        )
        tl.store(
            out_ptr + token * stride_out_token + row,
            y.reshape(NUM_HEADS * D).to(tl.bfloat16),
        )


@triton.jit
def _compress_kernel(
    k_ptr,
    positions_ptr,
    state_ptr,
    state_tail_ptr,
    compressed_ptr,
    weight_ptr,
    cos_ptr,
    work_ptr,
    table_ptr,
    query_start_ptr,
    logical_ptr,
    compressed_slots_ptr,
    stride_k_token,
    stride_pos_token,
    stride_pos_axis,
    stride_state_block,
    stride_state_row,
    stride_state_dim,
    stride_tail_block,
    stride_tail_row,
    stride_tail_dim,
    stride_table_req,
    stride_comp_block,
    stride_comp_row,
    stride_cos_row,
    eps,
    num_work,
    num_state_blocks,
    num_tokens,
    num_requests,
    STATE_SIZE: tl.constexpr,
    COMP_PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    D: tl.constexpr,
    PAIR: tl.constexpr,
    ROTARY: tl.constexpr,
    MID: tl.constexpr,
    IS_2D: tl.constexpr,
    IS_MROPE: tl.constexpr,
    HAS_ROPE_POS: tl.constexpr,
    SEC_H: tl.constexpr,
    SEC_W: tl.constexpr,
    COS_ROWS: tl.constexpr,
):
    """Pool one completed compression group and store the normalized key.

    Indices are clamped and accesses masked instead of returning early: the
    Ascend backend mislowers a return inside a runtime conditional, and a
    masked access still carries its address.  Tiles stay narrow (PAIR-wide
    rotate halves, MID-wide tail) to keep the unified buffer small.
    """

    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for work in tl.range(pid, num_work, nprog):
        ref = tl.load(work_ptr + tl.minimum(work, num_work - 1) * 2)
        sub = tl.load(work_ptr + tl.minimum(work, num_work - 1) * 2 + 1)
        active = (work < num_work) & (ref >= 0)
        request = tl.minimum(tl.maximum(ref, 0), num_requests - 1)

        row_start = tl.load(query_start_ptr + request)
        row_end = tl.load(query_start_ptr + request + 1)
        query_len = row_end - row_start
        chunk_end = tl.load(logical_ptr + tl.maximum(row_end - 1, 0))
        chunk_start = chunk_end - query_len + 1
        num_groups = (chunk_end + 1) // RATIO - chunk_start // RATIO
        active = active & (sub < num_groups)

        first_boundary = ((chunk_start + RATIO) // RATIO) * RATIO - 1
        end_position = first_boundary + sub * RATIO
        boundary = row_start + end_position - chunk_start
        valid_boundary = active & (boundary >= row_start) & (boundary < row_end)
        safe_boundary = tl.minimum(tl.maximum(boundary, 0), num_tokens - 1)
        slot = tl.load(compressed_slots_ptr + safe_boundary)
        active = valid_boundary & (slot >= 0)
        safe_slot = tl.maximum(slot, 0)

        state_block = tl.load(table_ptr + request * stride_table_req)
        valid_state = active & (state_block >= 0) & (state_block < num_state_blocks)
        safe_block = tl.maximum(state_block, 0)
        ring_base_req = state_ptr + safe_block * stride_state_block

        lane = tl.arange(0, PAIR)
        mid_lane = tl.arange(0, MID)
        mid_off = ROTARY + mid_lane
        dims = tl.arange(0, D)

        first_position = end_position - (RATIO - 1)
        row_offset = row_start - chunk_start
        acc = tl.zeros((D,), dtype=tl.float32)
        for off in tl.static_range(0, RATIO):
            position = first_position + off
            in_chunk = position >= chunk_start
            raw_row = row_offset + position
            # ``raw_row >= row_start`` is the same test as ``in_chunk``.
            raw_ok = active & in_chunk & (raw_row < row_end)
            ring_row = tl.minimum(tl.maximum(position % STATE_SIZE, 0), STATE_SIZE - 1)
            ring_ok = active & (not in_chunk) & valid_state
            safe_raw = tl.minimum(tl.maximum(raw_row, 0), num_tokens - 1)

            raw_base = k_ptr + safe_raw * stride_k_token
            ring_base = ring_base_req + ring_row * stride_state_row
            acc += tl.load(raw_base + dims, mask=raw_ok, other=0.0).to(tl.float32)
            acc += tl.load(
                ring_base + dims * stride_state_dim, mask=ring_ok, other=0.0
            ).to(tl.float32)

        pooled = (acc / RATIO).to(tl.bfloat16).to(tl.float32)
        ss = tl.sum(pooled * pooled, axis=0)
        rms = tl.rsqrt(ss / D + eps)
        weight = tl.load(weight_ptr + dims).to(tl.float32) + 1.0
        y = (pooled * rms * weight).reshape(1, D)
        y_lo = extract_slice(y, (0, 0), (1, PAIR), (1, 1)).reshape(PAIR)
        y_hi = extract_slice(y, (0, PAIR), (1, PAIR), (1, 1)).reshape(PAIR)
        y_mid = extract_slice(y, (0, ROTARY), (1, MID), (1, 1)).reshape(MID)

        first_in_chunk = first_position >= chunk_start
        first_token = row_start + first_position - chunk_start
        safe_first = tl.minimum(tl.maximum(first_token, 0), num_tokens - 1)

        base_first = positions_ptr + safe_first * stride_pos_token
        chunk_t = tl.load(base_first).to(tl.int64)
        if HAS_ROPE_POS:
            ring_first = tl.minimum(
                tl.maximum(first_position % STATE_SIZE, 0), STATE_SIZE - 1
            )
            tail_base = (
                state_tail_ptr
                + safe_block * stride_tail_block
                + ring_first * stride_tail_row
            )
            ring_t = tl.load(tail_base)
            pos_t = tl.where(first_in_chunk, chunk_t, ring_t)
            if IS_MROPE:
                chunk_h = tl.load(base_first + stride_pos_axis).to(tl.int64)
                chunk_w = tl.load(base_first + 2 * stride_pos_axis).to(tl.int64)
                ring_h = tl.load(tail_base + stride_tail_dim)
                ring_w = tl.load(tail_base + 2 * stride_tail_dim)
                pos_h = tl.where(first_in_chunk, chunk_h, ring_h)
                pos_w = tl.where(first_in_chunk, chunk_w, ring_w)
            else:
                pos_h = pos_t
                pos_w = pos_t
        else:
            pos_t = first_position.to(tl.int64)
            pos_h = pos_t
            pos_w = pos_t
        pos_t = tl.minimum(tl.maximum(pos_t, 0), COS_ROWS - 1)
        pos_h = tl.minimum(tl.maximum(pos_h, 0), COS_ROWS - 1)
        pos_w = tl.minimum(tl.maximum(pos_w, 0), COS_ROWS - 1)

        cos_t = tl.load(cos_ptr + pos_t * stride_cos_row + lane).to(tl.float32)
        sin_t = tl.load(cos_ptr + pos_t * stride_cos_row + PAIR + lane).to(tl.float32)
        if IS_MROPE:
            axis = lane % 3
            is_h = (axis == 1) & (lane < 3 * SEC_H)
            is_w = (axis == 2) & (lane < 3 * SEC_W)
            cos_h = tl.load(cos_ptr + pos_h * stride_cos_row + lane).to(tl.float32)
            cos_w = tl.load(cos_ptr + pos_w * stride_cos_row + lane).to(tl.float32)
            sin_h = tl.load(cos_ptr + pos_h * stride_cos_row + PAIR + lane).to(
                tl.float32
            )
            sin_w = tl.load(cos_ptr + pos_w * stride_cos_row + PAIR + lane).to(
                tl.float32
            )
            cos = tl.where(is_h, cos_h, tl.where(is_w, cos_w, cos_t))
            sin = tl.where(is_h, sin_h, tl.where(is_w, sin_w, sin_t))
        else:
            cos = cos_t
            sin = sin_t

        dst = (
            compressed_ptr
            + (safe_slot // COMP_PAGE) * stride_comp_block
            + (safe_slot % COMP_PAGE) * stride_comp_row
        )
        tl.store(dst + lane, (y_lo * cos - y_hi * sin).to(tl.bfloat16), mask=active)
        tl.store(
            dst + PAIR + lane, (y_hi * cos + y_lo * sin).to(tl.bfloat16), mask=active
        )
        tl.store(dst + mid_off, y_mid.to(tl.bfloat16), mask=active)


@triton.jit
def _state_ring_kernel(
    k_ptr,
    positions_ptr,
    state_ptr,
    state_tail_ptr,
    slots_ptr,
    query_start_ptr,
    stride_k_token,
    stride_pos_token,
    stride_pos_axis,
    stride_state_block,
    stride_state_row,
    stride_state_dim,
    stride_tail_block,
    stride_tail_row,
    stride_tail_dim,
    num_requests,
    D: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    IS_2D: tl.constexpr,
    HAS_ROPE_POS: tl.constexpr,
):
    """Commit the retained suffix of each request into the raw key ring."""

    request = tl.program_id(0)
    row_start = tl.load(query_start_ptr + request)
    row_end = tl.load(query_start_ptr + request + 1)
    query_len = row_end - row_start
    rows = tl.minimum(query_len, STATE_SIZE)

    dims = tl.arange(0, D)
    d4 = tl.arange(0, 4)
    for offset in tl.range(0, rows):
        token = row_end - rows + offset
        slot = tl.load(slots_ptr + token)
        ok = slot >= 0
        block = tl.maximum(slot, 0) // STATE_SIZE
        ring = tl.maximum(slot, 0) % STATE_SIZE
        values = tl.load(k_ptr + token * stride_k_token + dims)
        tl.store(
            state_ptr
            + block * stride_state_block
            + ring * stride_state_row
            + dims * stride_state_dim,
            values,
            mask=(dims < D) & ok,
        )
        if HAS_ROPE_POS:
            pos_t = tl.load(positions_ptr + token * stride_pos_token)
            if IS_2D:
                pos_h = tl.load(
                    positions_ptr + token * stride_pos_token + stride_pos_axis
                )
                pos_w = tl.load(
                    positions_ptr + token * stride_pos_token + 2 * stride_pos_axis
                )
            else:
                pos_h = pos_t
                pos_w = pos_t
            coords = tl.where(d4 == 0, pos_t, tl.where(d4 == 1, pos_h, pos_w))
            tl.store(
                state_tail_ptr
                + block * stride_tail_block
                + ring * stride_tail_row
                + d4,
                coords,
                mask=(d4 < 3) & ok,
            )


@triton.jit
def _main_kernel(
    qkv_ptr,
    rope_ptr,
    q_weight_ptr,
    k_weight_ptr,
    q_out_ptr,
    gate_out_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    stride_qkv_token,
    stride_qkv_dim,
    stride_rope_token,
    stride_out_token,
    stride_cache_block,
    stride_cache_page,
    stride_cache_head,
    stride_cache_dim,
    eps,
    num_tokens,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    PAIR: tl.constexpr,
    ROTARY: tl.constexpr,
):
    """One program per vector core; each iteration handles a whole token row.

    The q|gate blocks are read as one contiguous span and split with DSA
    slices, so a token costs a few wide transfers instead of one transfer per
    head.
    """

    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    q_size: tl.constexpr = NUM_Q_HEADS * D
    kv_size: tl.constexpr = NUM_KV_HEADS * D
    d_dims = tl.arange(0, D)
    q_dims = tl.arange(0, q_size)
    kv_dims = tl.arange(0, kv_size)
    q_weight = tl.load(q_weight_ptr + d_dims).to(tl.float32) + 1.0
    k_weight = tl.load(k_weight_ptr + d_dims).to(tl.float32) + 1.0

    for token in tl.range(pid, num_tokens, nprog):
        base = qkv_ptr + token * stride_qkv_token
        cos = tl.load(rope_ptr + token * stride_rope_token + tl.arange(0, ROTARY))
        sin = tl.load(
            rope_ptr + token * stride_rope_token + ROTARY + tl.arange(0, ROTARY)
        )

        qg = tl.load(base + tl.arange(0, 2 * q_size) * stride_qkv_dim).to(tl.float32)
        qg = qg.reshape(NUM_Q_HEADS, 2 * D)
        q = extract_slice(qg, offsets=(0, 0), sizes=(NUM_Q_HEADS, D), strides=(1, 1))
        gate = extract_slice(qg, offsets=(0, D), sizes=(NUM_Q_HEADS, D), strides=(1, 1))
        ss = tl.sum(q * q, axis=1)
        rstd = tl.rsqrt(ss / D + eps)
        q = q * rstd[:, None] * q_weight[None, :]
        q_head = (
            extract_slice(
                q, offsets=(0, 0), sizes=(NUM_Q_HEADS, ROTARY), strides=(1, 1)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        q_lo = extract_slice(
            q_head, offsets=(0, 0), sizes=(NUM_Q_HEADS, PAIR), strides=(1, 1)
        )
        q_hi = extract_slice(
            q_head, offsets=(0, PAIR), sizes=(NUM_Q_HEADS, PAIR), strides=(1, 1)
        )
        q_cat = tl.zeros((NUM_Q_HEADS, ROTARY), dtype=tl.float32)
        q_cat = insert_slice(
            q_cat, -q_hi, offsets=(0, 0), sizes=(NUM_Q_HEADS, PAIR), strides=(1, 1)
        )
        q_cat = insert_slice(
            q_cat, q_lo, offsets=(0, PAIR), sizes=(NUM_Q_HEADS, PAIR), strides=(1, 1)
        )
        q = insert_slice(
            q,
            q_cat * sin[None, :] + q_head * cos[None, :],
            offsets=(0, 0),
            sizes=(NUM_Q_HEADS, ROTARY),
            strides=(1, 1),
        )
        tl.store(
            q_out_ptr + token * stride_out_token + q_dims,
            q.reshape(q_size).to(tl.bfloat16),
        )
        tl.store(
            gate_out_ptr + token * stride_out_token + q_dims,
            gate.reshape(q_size).to(tl.bfloat16),
        )

        k_off = base + 2 * q_size * stride_qkv_dim
        k = tl.load(k_off + kv_dims * stride_qkv_dim).to(tl.float32)
        k = k.reshape(NUM_KV_HEADS, D)
        ss = tl.sum(k * k, axis=1)
        rstd = tl.rsqrt(ss / D + eps)
        k = k * rstd[:, None] * k_weight[None, :]
        k_head = (
            extract_slice(
                k, offsets=(0, 0), sizes=(NUM_KV_HEADS, ROTARY), strides=(1, 1)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        k_lo = extract_slice(
            k_head, offsets=(0, 0), sizes=(NUM_KV_HEADS, PAIR), strides=(1, 1)
        )
        k_hi = extract_slice(
            k_head, offsets=(0, PAIR), sizes=(NUM_KV_HEADS, PAIR), strides=(1, 1)
        )
        k_cat = tl.zeros((NUM_KV_HEADS, ROTARY), dtype=tl.float32)
        k_cat = insert_slice(
            k_cat, -k_hi, offsets=(0, 0), sizes=(NUM_KV_HEADS, PAIR), strides=(1, 1)
        )
        k_cat = insert_slice(
            k_cat, k_lo, offsets=(0, PAIR), sizes=(NUM_KV_HEADS, PAIR), strides=(1, 1)
        )
        k = insert_slice(
            k,
            k_cat * sin[None, :] + k_head * cos[None, :],
            offsets=(0, 0),
            sizes=(NUM_KV_HEADS, ROTARY),
            strides=(1, 1),
        )
        v = tl.load(k_off + kv_size * stride_qkv_dim + kv_dims * stride_qkv_dim)
        v = v.reshape(NUM_KV_HEADS, D)

        slot = tl.load(slot_mapping_ptr + token)
        ok = slot >= 0
        safe = tl.maximum(slot, 0)
        dst = (
            kv_cache_ptr
            + (safe // PAGE_SIZE) * stride_cache_block
            + (safe % PAGE_SIZE) * stride_cache_page
        )
        kv_idx = tl.arange(0, NUM_KV_HEADS)
        kv_off = (
            dst
            + kv_idx[:, None] * stride_cache_head
            + d_dims[None, :] * stride_cache_dim
        )
        tl.store(kv_off, k.to(tl.bfloat16), mask=ok)
        tl.store(kv_off + D * stride_cache_dim, v, mask=ok)


# Tokens handled per prep-kernel iteration; the kernel is scalar bound.
_ROPE_TOKENS_PER_PROG = 8


def _vector_core_count():
    """Vector core count; the kernels launch one program per core."""

    try:
        props = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        cores = int(props.get("num_vectorcore", 0))
        if cores > 0:
            return cores
    except Exception:  # noqa: BLE001 - fall back to a conservative width
        pass
    return 48


def qsa_prepare(
    q,
    k,
    positions,
    cos_sin_cache,
    q_norm_weight,
    k_norm_weight,
    eps,
    q_out,
    state_cache,
    state_slots,
    state_block_table,
    query_start_loc,
    logical_positions,
    compressed_cache,
    compressed_slots,
    k_work_metadata,
    *,
    compress_ratio,
    mrope_section,
    rope_pos_offset,
    main_qkv,
    main_q_norm_weight,
    main_k_norm_weight,
    main_eps,
    main_kv_cache,
    main_slot_mapping,
    main_k_scale,
    main_v_scale,
):
    """Run QSA prepare for the indexer, the raw ring and the main attention."""

    fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    if q_out.dtype in fp8 or compressed_cache.dtype in fp8:
        raise NotImplementedError("Ascend qsa_prepare supports bf16 indexer buffers")
    if main_kv_cache.dtype in fp8:
        raise NotImplementedError("Ascend qsa_prepare supports a bf16 main KV cache")

    num_tokens = q.shape[0]
    num_requests = query_start_loc.numel() - 1
    num_heads = q_out.shape[1]
    head_dim = q_out.shape[2]
    pair = head_dim // 4
    rotary = 2 * pair
    num_kv_heads = main_kv_cache.shape[2]
    main_dim = main_kv_cache.shape[-1] // 2
    num_q_heads = main_qkv.shape[1] // (2 * main_dim) - num_kv_heads
    main_pair = main_dim // 8
    main_rotary = 2 * main_pair
    main_mid = main_dim - main_rotary
    main_mid_a = triton.next_power_of_2(main_mid // 2)
    main_mid_b = triton.next_power_of_2(main_mid - main_mid_a)
    assert (
        main_mid_a + main_mid_b == main_mid
    ), "main head tail must split into two power-of-two segments"

    is_2d = positions.ndim == 2
    is_mrope = mrope_section is not None
    sec_h = int(mrope_section[1]) if is_mrope else 0
    sec_w = int(mrope_section[2]) if is_mrope else 0
    has_rope_pos = state_cache.shape[-1] > head_dim
    state_size = state_cache.shape[1]
    state_keys = state_cache[..., :head_dim]
    if has_rope_pos:
        state_tail = state_cache[..., head_dim:].view(torch.int64)
        tail_block_stride = state_tail.stride(0)
        tail_row_stride = state_tail.stride(1)
        tail_dim_stride = state_tail.stride(3)
    else:
        state_tail = state_cache
        tail_block_stride = 0
        tail_row_stride = 0
        tail_dim_stride = 0

    main_q_out = torch.empty(
        (num_tokens, num_q_heads, main_dim), dtype=torch.bfloat16, device=q.device
    )
    main_gate_out = torch.empty_like(main_q_out)
    if num_tokens == 0:
        return main_q_out, main_gate_out

    stride_pos_axis = positions.stride(0) if is_2d else 0
    stride_pos_token = positions.stride(-1)

    if extract_slice is None:
        raise NotImplementedError(
            "Ascend qsa_prepare needs the Triton CANN vector ops extension"
        )
    cores = _vector_core_count()
    rope_table = torch.empty(
        (num_tokens, 2 * rotary), dtype=torch.float32, device=q.device
    )
    stride_rope_token = rope_table.stride(0)
    # Blocking amortises the kernel's scalar work, but a small batch pays for the
    # masks instead, so only wide steps use a block.
    rope_block = _ROPE_TOKENS_PER_PROG if num_tokens >= 1024 else 2
    _rope_table_kernel[(min(triton.cdiv(num_tokens, rope_block), cores),)](
        positions,
        cos_sin_cache,
        rope_table,
        stride_pos_token,
        stride_pos_axis,
        cos_sin_cache.stride(0),
        stride_rope_token,
        num_tokens,
        PAIR=pair,
        ROTARY=rotary,
        IS_MROPE=is_mrope,
        COS_ROWS=cos_sin_cache.shape[0],
        SEC_H=sec_h,
        SEC_W=sec_w,
        TOKENS_PER_PROG=rope_block,
        num_warps=1,
    )
    _indexer_q_kernel[(min(num_tokens, cores),)](
        q,
        rope_table,
        q_norm_weight,
        q_out,
        q.stride(0),
        q.stride(1),
        stride_rope_token,
        q_out.stride(0),
        float(eps),
        num_tokens,
        NUM_HEADS=num_heads,
        D=head_dim,
        PAIR=pair,
        ROTARY=rotary,
        num_warps=1,
    )

    num_work = k_work_metadata.shape[0]
    _compress_kernel[(min(num_work, cores),)](
        k,
        positions,
        state_keys,
        state_tail,
        compressed_cache,
        k_norm_weight,
        cos_sin_cache,
        k_work_metadata,
        state_block_table,
        query_start_loc,
        logical_positions,
        compressed_slots,
        k.stride(0),
        stride_pos_token,
        stride_pos_axis,
        state_cache.stride(0),
        state_cache.stride(1),
        state_cache.stride(3),
        tail_block_stride,
        tail_row_stride,
        tail_dim_stride,
        state_block_table.stride(0),
        compressed_cache.stride(0),
        compressed_cache.stride(1),
        cos_sin_cache.stride(0),
        float(eps),
        num_work,
        state_cache.shape[0],
        num_tokens,
        num_requests,
        STATE_SIZE=state_size,
        COMP_PAGE=compressed_cache.shape[1],
        RATIO=int(compress_ratio),
        D=head_dim,
        PAIR=pair,
        ROTARY=rotary,
        MID=head_dim - rotary,
        IS_2D=is_2d,
        IS_MROPE=is_mrope,
        HAS_ROPE_POS=has_rope_pos,
        SEC_H=sec_h,
        SEC_W=sec_w,
        COS_ROWS=cos_sin_cache.shape[0],
        num_warps=1,
    )

    _state_ring_kernel[(num_requests,)](
        k,
        positions,
        state_keys,
        state_tail,
        state_slots,
        query_start_loc,
        k.stride(0),
        stride_pos_token,
        stride_pos_axis,
        state_cache.stride(0),
        state_cache.stride(1),
        state_cache.stride(3),
        tail_block_stride,
        tail_row_stride,
        tail_dim_stride,
        num_requests,
        D=head_dim,
        STATE_SIZE=state_size,
        IS_2D=is_2d,
        HAS_ROPE_POS=has_rope_pos,
        num_warps=1,
    )

    _main_kernel[(min(num_tokens, cores),)](
        main_qkv,
        rope_table,
        main_q_norm_weight,
        main_k_norm_weight,
        main_q_out,
        main_gate_out,
        main_kv_cache,
        main_slot_mapping,
        main_qkv.stride(0),
        main_qkv.stride(1),
        stride_rope_token,
        main_q_out.stride(0),
        main_kv_cache.stride(0),
        main_kv_cache.stride(1),
        main_kv_cache.stride(2),
        main_kv_cache.stride(3),
        float(main_eps),
        num_tokens,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        PAGE_SIZE=main_kv_cache.shape[1],
        D=main_dim,
        PAIR=main_pair,
        ROTARY=main_rotary,
        num_warps=1,
    )
    return main_q_out, main_gate_out


__all__ = ["qsa_prepare"]
