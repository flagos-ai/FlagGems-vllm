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

"""GCU (enflame) top_k_per_row_prefill for DeepSeek V4 prefill-phase topk.

The generic implementation is a 4-step radix select that cannot run on GCU:

- its TLE path compiles DSM buffers and CTA-scope ``tl.atomic_add`` (atomics
  are banned on GCU, and the current toolchain rejects the zero-fill store
  pattern outright, see compiler_issues/vllm_top_k_per_row_prefill.md);
- its non-TLE path reuses global scratch across loop iterations
  (store -> reload through the same kernel, the known stale-L1 trap) and also
  relies on CTA-scope atomics; on narrow-band inputs it produced wrong
  results and finally aborted the device.

This rewrite follows the proven persistent_topk recipe (knowledge/perf_playbook):
single kernel launch, one program per row, atomic-free, no scratch buffers, no
zero-valued stores, 1D tiles only:

1. monotone int32 key ``b < 0 ? b ^ 0x7FFFFFFF : b`` from the fp32 bit pattern
   (signed i32 compare == float order), masked lanes neutralised to INT32_MIN;
2. overflow-safe binary search for ``T = min{x : count(key > x) < k}`` (the
   k-th largest key) over the full int32 range, carrying ``count(key > T)``
   out of the search. Two probes per pass are packed into a single ``tl.sum``
   (``(key > p1) * SCALE + (key > p2)``, ``SCALE = 2 * BLOCK``) — the
   dependent-reduction chain is the bottleneck on GCU (~1.8us per ``tl.sum``),
   so halving the pass count beats halving the per-pass work. The search
   exits early when a probe count equals ``k`` exactly (the top-k set is then
   already separated);
3. one selection scan writes the top-k columns with forward ``tl.cumsum``
   slot assignment: keys > T first, then the first ``k - count(key > T)``
   tied keys by ascending column (stable tie-break), into scan order.

Rows that fit in one tile keep the key tile resident in registers across the
whole search (no reload per pass); wider rows use the reload loop. Output is a
deterministic permutation of the stable top-k index set (the tests compare
sorted sets and run-to-run stability, not slot order).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# GCU launch limits (triton.backends.enflame.backend.GcuLauncher):
# grid.x <= 65535, grid.y <= 255, grid.z <= 255.
_GRID_X_MAX = 65535
_GRID_Y_MAX = 255

_NUM_WARPS = 8
# The tile-resident search is reduction-latency-bound: with BLOCK <= 8192 a
# single warp keeps tl.sum intra-warp (no cross-warp DSM barrier) and wins
# 1.46-1.81x over nw>=4 (measured on the core shapes); at BLOCK >= 16384 the
# resident tile no longer fits one warp's registers, and nw=4 is fastest.
# The reload (tiled) kernel is DMA-bound and keeps nw=8.
_RESIDENT_NW1_MAX_BLOCK = 8192
_RESIDENT_NW_BIG = 4
# Binary-search iterations: two packed probes per pass retain at most half of
# the bracket, so 32 passes + 2 cleanup rounds bound the search; extra rounds
# are skipped by the ``lo < hi`` guard.
_MAX_ITERS = 40
# Packed-count fields: two probe counts ride in one tl.sum as
# (key > p1) * SCALE + (key > p2) with SCALE = 2 * BLOCK, so each per-tile
# count (<= BLOCK) fits its field. The tile sum is <= 2 * BLOCK^2 + BLOCK,
# which stays below 2**31 for BLOCK <= 16384; for BLOCK = 32768 the extreme
# all-lanes-count case wraps int32, but the masked unpack (v >> SHIFT) & MASK
# recovers the true fields from the two's-complement wrap, so the packing is
# exact for every BLOCK <= 32768.
# Rows up to this width use the tile-resident kernel (key tile lives across
# the whole search); wider rows reload per pass. BLOCK is capped at 32768:
# the selection temporaries of a [65536] tile exceed the per-CTA DSM budget
# (OutOfResources: required 1316864 > hardware limit 917504).
_RESIDENT_MAX_VOCAB = 32768
_TILED_BLOCK = 32768


def _row_grid(m):
    """Spread ``m`` logical rows over a 3-D grid within GCU launch limits.

    The kernel recombines (pid_x, pid_y, pid_z) into a flat row id via
    ``grid_x``/``grid_xy`` runtime arguments, so any decomposition with
    ``grid_x * grid_y * grid_z >= m`` is correct; rows beyond ``m`` are
    masked out by the in-kernel ``row < num_rows`` guard.
    """
    if m <= _GRID_X_MAX:
        return (m, 1, 1)

    grid_y = triton.cdiv(m, _GRID_X_MAX)
    if grid_y <= _GRID_Y_MAX:
        return (triton.cdiv(m, grid_y), grid_y, 1)

    grid_z = triton.cdiv(m, _GRID_X_MAX * _GRID_Y_MAX)
    grid_y = triton.cdiv(m, _GRID_X_MAX * grid_z)
    return (triton.cdiv(m, grid_y * grid_z), grid_y, grid_z)


@triton.jit
def _monotone_key_i32(x):
    # Monotone map from the fp32 bit pattern to signed int32: larger key ==
    # larger float (total order for non-NaN inputs). Negative branch flips the
    # low 31 bits, non-negative branch keeps the bits; the sign bit separates
    # the two ranges, so plain signed compares order the floats.
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _gcu_topk_row_resident(
    logits_ptr,
    out_ptr,
    row_starts_ptr,
    row_ends_ptr,
    num_rows,
    grid_x,
    grid_xy,
    stride0,
    stride1,
    top_k,
    BLOCK: tl.constexpr,
    SHIFT: tl.constexpr,
    MAX_ITERS: tl.constexpr,
):
    SCALE: tl.constexpr = 1 << SHIFT
    MASK: tl.constexpr = SCALE - 1
    row = tl.program_id(0) + tl.program_id(1) * grid_x + tl.program_id(2) * grid_xy
    if row < num_rows:
        row_start = tl.load(row_starts_ptr + row)
        row_end = tl.load(row_ends_ptr + row)
        row_len = row_end - row_start
        row_base = logits_ptr + row.to(tl.int64) * stride0 + row_start
        out_base = out_ptr + row.to(tl.int64) * top_k
        lane = tl.arange(0, BLOCK)

        if row_len <= top_k:
            # Every valid element is selected; pad the rest with -1.
            n_chunks = tl.cdiv(top_k, BLOCK)
            for chunk in tl.range(0, n_chunks):
                pos = chunk * BLOCK + lane
                tl.store(out_base + pos, pos, mask=pos < row_len)
                tl.store(out_base + pos, -1, mask=(pos >= row_len) & (pos < top_k))
        else:
            # Load + neutralise once: the key tile stays resident across the
            # whole search, so each pass costs one packed tl.sum only.
            in_row = lane < row_len
            x = tl.load(row_base + lane * stride1, mask=in_row, other=0.0)
            key = _monotone_key_i32(x)
            key = tl.where(in_row, key, -2147483648)

            lo = -2147483648
            hi = 2147483647
            n_gt = 0
            for _iter in tl.range(0, MAX_ITERS):
                if lo < hi:
                    # Two overflow-safe floor midpoints per pass; both counts
                    # ride in one packed tl.sum (halves the dependent-reduction
                    # chain). Early exit on an exact top_k count.
                    p1 = (lo & hi) + ((lo ^ hi) >> 1)
                    p2 = (p1 & hi) + ((p1 ^ hi) >> 1)
                    if p2 > p1:
                        v = tl.sum(
                            (key > p1).to(tl.int32) * SCALE + (key > p2).to(tl.int32),
                            axis=0,
                        )
                        c1 = (v >> SHIFT) & MASK
                        c2 = v & MASK
                        if c2 >= top_k:
                            if c2 == top_k:
                                lo = p2
                                hi = p2
                                n_gt = c2
                            else:
                                lo = p2 + 1
                        elif c1 >= top_k:
                            if c1 == top_k:
                                lo = p1
                                hi = p1
                                n_gt = c1
                            else:
                                lo = p1 + 1
                                hi = p2
                                n_gt = c2
                        else:
                            hi = p1
                            n_gt = c1
                    else:
                        c1 = tl.sum((key > p1).to(tl.int32), axis=0)
                        if c1 == top_k:
                            lo = p1
                            hi = p1
                            n_gt = c1
                        elif c1 < top_k:
                            hi = p1
                            n_gt = c1
                        else:
                            lo = p1 + 1

            # Stable selection in one scan: keys > T fill slots [0, n_gt),
            # the first (top_k - n_gt) tied keys (ascending column) fill the
            # rest. Forward cumsum assigns the slots in scan order.
            need = top_k - n_gt
            lt = (key > hi).to(tl.int32)
            eq = ((key == hi) & in_row).to(tl.int32)
            lt_ex = tl.cumsum(lt, axis=0) - lt
            eq_ex = tl.cumsum(eq, axis=0) - eq
            take_lt = lt > 0
            take_eq = (eq > 0) & (eq_ex < need)
            take = take_lt | take_eq
            slot = tl.where(take_lt, lt_ex, n_gt + eq_ex)
            tl.store(out_base + slot, lane, mask=take)


@triton.jit
def _gcu_topk_row_tiled(
    logits_ptr,
    out_ptr,
    row_starts_ptr,
    row_ends_ptr,
    num_rows,
    grid_x,
    grid_xy,
    stride0,
    stride1,
    top_k,
    BLOCK: tl.constexpr,
    SHIFT: tl.constexpr,
    MAX_ITERS: tl.constexpr,
):
    SCALE: tl.constexpr = 1 << SHIFT
    MASK: tl.constexpr = SCALE - 1
    row = tl.program_id(0) + tl.program_id(1) * grid_x + tl.program_id(2) * grid_xy
    if row < num_rows:
        row_start = tl.load(row_starts_ptr + row)
        row_end = tl.load(row_ends_ptr + row)
        row_len = row_end - row_start
        row_base = logits_ptr + row.to(tl.int64) * stride0 + row_start
        out_base = out_ptr + row.to(tl.int64) * top_k
        lane = tl.arange(0, BLOCK)

        if row_len <= top_k:
            # Every valid element is selected; pad the rest with -1.
            n_chunks = tl.cdiv(top_k, BLOCK)
            for chunk in tl.range(0, n_chunks):
                pos = chunk * BLOCK + lane
                tl.store(out_base + pos, pos, mask=pos < row_len)
                tl.store(out_base + pos, -1, mask=(pos >= row_len) & (pos < top_k))
        else:
            n_tiles = tl.cdiv(row_len, BLOCK)

            # Binary search T = min{x : count(key > x) < top_k} over the full
            # int32 range; two probes per pass packed into one tl.sum per tile.
            lo = -2147483648
            hi = 2147483647
            n_gt = 0
            for _iter in tl.range(0, MAX_ITERS):
                if lo < hi:
                    p1 = (lo & hi) + ((lo ^ hi) >> 1)
                    p2 = (p1 & hi) + ((p1 ^ hi) >> 1)
                    if p2 > p1:
                        c1 = 0
                        c2 = 0
                        for tile in tl.range(0, n_tiles):
                            offs = tile * BLOCK + lane
                            in_row = offs < row_len
                            x = tl.load(
                                row_base + offs * stride1, mask=in_row, other=0.0
                            )
                            key = _monotone_key_i32(x)
                            key = tl.where(in_row, key, -2147483648)
                            v = tl.sum(
                                (key > p1).to(tl.int32) * SCALE
                                + (key > p2).to(tl.int32),
                                axis=0,
                            )
                            c1 += (v >> SHIFT) & MASK
                            c2 += v & MASK
                        if c2 >= top_k:
                            if c2 == top_k:
                                lo = p2
                                hi = p2
                                n_gt = c2
                            else:
                                lo = p2 + 1
                        elif c1 >= top_k:
                            if c1 == top_k:
                                lo = p1
                                hi = p1
                                n_gt = c1
                            else:
                                lo = p1 + 1
                                hi = p2
                                n_gt = c2
                        else:
                            hi = p1
                            n_gt = c1
                    else:
                        c1 = 0
                        for tile in tl.range(0, n_tiles):
                            offs = tile * BLOCK + lane
                            in_row = offs < row_len
                            x = tl.load(
                                row_base + offs * stride1, mask=in_row, other=0.0
                            )
                            key = _monotone_key_i32(x)
                            key = tl.where(in_row, key, -2147483648)
                            c1 += tl.sum((key > p1).to(tl.int32), axis=0)
                        if c1 == top_k:
                            lo = p1
                            hi = p1
                            n_gt = c1
                        elif c1 < top_k:
                            hi = p1
                            n_gt = c1
                        else:
                            lo = p1 + 1

            # Stable selection in one scan: keys > T fill slots [0, n_gt),
            # the first (top_k - n_gt) tied keys (ascending column) fill the
            # rest. Forward cumsum assigns the slots in scan order; found and
            # ties ride packed through a single tl.sum per tile.
            need = top_k - n_gt
            found = 0
            ties = 0
            for tile in tl.range(0, n_tiles):
                offs = tile * BLOCK + lane
                in_row = offs < row_len
                x = tl.load(row_base + offs * stride1, mask=in_row, other=0.0)
                key = _monotone_key_i32(x)
                key = tl.where(in_row, key, -2147483648)
                lt = (key > hi).to(tl.int32)
                eq = ((key == hi) & in_row).to(tl.int32)
                lt_ex = tl.cumsum(lt, axis=0) - lt
                eq_ex = tl.cumsum(eq, axis=0) - eq
                take_lt = lt > 0
                take_eq = (eq > 0) & ((ties + eq_ex) < need)
                take = take_lt | take_eq
                slot = tl.where(take_lt, found + lt_ex, n_gt + ties + eq_ex)
                tl.store(out_base + slot, offs, mask=take)
                v = tl.sum(lt * SCALE + eq, axis=0)
                found += (v >> SHIFT) & MASK
                ties += v & MASK


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for prefill phase of DeepSeek V4 sparse attention.

    Selects top_k indices from each row of logits. Only valid elements within
    [row_starts[i], row_ends[i]) are considered.

    Args:
        logits: [num_rows, vocab_size] float32 tensor.
        row_starts: [num_rows] int32 — start of valid range per row (inclusive).
        row_ends: [num_rows] int32 — end of valid range per row (exclusive).
        indices: [num_rows, top_k] int32 — output buffer, filled with 0-based
                 indices relative to row_starts[i]. Caller pre-allocates this.
        num_rows: number of rows.
        stride0: logits.stride(0), typically == vocab_size for contiguous tensor.
        stride1: logits.stride(1), typically == 1 for contiguous tensor.
        top_k: number of top elements per row.
    """
    logger.debug("GEMS TOP_K_PER_ROW_PREFILL (GCU)")

    if logits.dtype != torch.float32:
        raise NotImplementedError(
            f"top_k_per_row_prefill on GCU supports float32 logits, "
            f"got {logits.dtype}"
        )
    if indices.dtype != torch.int32:
        raise NotImplementedError(
            f"top_k_per_row_prefill on GCU supports int32 indices, "
            f"got {indices.dtype}"
        )

    assert num_rows == logits.shape[0]
    if num_rows == 0:
        return

    grid = _row_grid(num_rows)
    vocab_size = logits.shape[1]
    if vocab_size <= _RESIDENT_MAX_VOCAB:
        block = max(1024, triton.next_power_of_2(vocab_size))
        num_warps = 1 if block <= _RESIDENT_NW1_MAX_BLOCK else _RESIDENT_NW_BIG
        _gcu_topk_row_resident[grid](
            logits,
            indices,
            row_starts,
            row_ends,
            num_rows,
            grid[0],
            grid[0] * grid[1],
            stride0,
            stride1,
            top_k,
            BLOCK=block,
            SHIFT=(2 * block).bit_length() - 1,
            MAX_ITERS=_MAX_ITERS,
            num_warps=num_warps,
        )
    else:
        _gcu_topk_row_tiled[grid](
            logits,
            indices,
            row_starts,
            row_ends,
            num_rows,
            grid[0],
            grid[0] * grid[1],
            stride0,
            stride1,
            top_k,
            BLOCK=_TILED_BLOCK,
            SHIFT=(2 * _TILED_BLOCK).bit_length() - 1,
            MAX_ITERS=_MAX_ITERS,
            num_warps=_NUM_WARPS,
        )
