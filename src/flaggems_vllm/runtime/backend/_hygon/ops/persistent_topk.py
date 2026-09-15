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

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = True
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False

# Hygon: DCU max threads/block = 1024, but tle.cumsum/tl.cumsum over
# [BLOCK_SIZE] requires threads == BLOCK_SIZE with BLOCK_SIZE <= 512 on the
# HIP backend (num_warps=32 over [1024] or [512] is rejected with
# "Required: 2048 threads"). Use BLOCK_SIZE=512 with 16 warps, matching the
# proven per-row-CTA decode wrapper config. (Generic NVIDIA version used 1024.)
THREADS_PER_BLOCK = 512
RADIX = 256
# Toggle for the round-0 histogram zeroing fix in _radix_topk (see its
# comment). Set to False (before importing) to reproduce the stale-count
# rotation bug: a batch with a non-radix row between two radix rows.
HYGON_RADIX_ROTATION_FIX = tl.constexpr(True)
# Bounded spin limit for the barrier poll (metax persistent_topk pattern,
# see _barrier_with_atomic_add): the loop counter in the while condition
# makes the scalar-poll loop compile/run correctly on the stock hcu backend
# (no compiler fix needed); 1<<30 far exceeds any legitimate wait.
BARRIER_SPIN_LIMIT = tl.constexpr(1 << 30)
MEDIUN_HIST_BYTES = 2 * (RADIX + 128) * 4  # 3072
MEDIUM_SCALARS_BYTES = 5 * 4  # 20
MEDIUM_HEADER_SIZE = (MEDIUN_HIST_BYTES + MEDIUM_SCALARS_BYTES + 127) & (~127)  # 3200
# Hygon adaptation: MAX_BUFFERED_ITEMS reduced 4096 -> 2048 so the medium
# layout (hist 2048 + 2*2048 buffers + scalars = 6149 u32) fits the 8192 u32
# shared_ordered budget (DCU smem/block = 64KB, alloc requires pow2).
# Buffer overflow only drops elements inside the 2048-bin threshold bucket
# (same approximation class as the generic version, just a tighter cap;
# capacity >= TOPK always holds since TOPK <= 2048).
MAX_BUFFERED_ITEMS = 2048
SMEM_MEDIUM = MEDIUM_HEADER_SIZE + 2 * MAX_BUFFERED_ITEMS * 4  # 19584
# Medium/radix dispatch boundary: rows with seq_len <= MEDIUM_THRESHOLD use
# the single-CTA medium path (_histogram_256_topk); larger rows use the
# cooperative radix path. Raised 32768 -> 49152 in v0.5 after the medium
# path's Phase 1 switched to a 512-bin tl.histogram (~5x faster) plus
# rank-select: medium stays below the radix path's ~200us fixed barrier cost
# up to ~50K elements (measured 2026-09-02).
MEDIUM_THRESHOLD = 49152
DECODE_BINS = 2048
HIST2048_THRESHOLD = 8192
FIXED_SMEM_LARGE = ((RADIX + RADIX + 5) * 4 + 15) & (~15)  # 2080
# Medium path with bin replay needs up to 32768 uint32 of shared_ordered.
# SMEM_MEDIUM_QUAD_C = 32768 → 128KB. Must fit within max_smem_per_block.
MEDIUM_REPLAY_SMEM = 32768 * 4  # 131072 bytes
# Medium/decode paths reuse shared_ordered's smem (mirroring vLLM's single
# extern __shared__ buffer). SMEM_MEDIUM // 4 = 4896 uint32s cover both the
# medium layout (6149) and the decode layout (8192).
SMEM_MEDIUM_QUAD = SMEM_MEDIUM // 4  # 4896
# Hygon: hard cap on the radix chunk size (u32). shared_ordered is a single
# 8192-u32 (32KB) allocation whose tail [7676, 8192) hosts the radix small
# buffers, so a chunk is capped at 4096 u32 to leave them room.
# v0.3 experiment (2026-08-31): radix small buffers are independent
# tle.gpu.alloc calls (not carved from shared_ordered's tail), so the whole
# 8192-u32 allocation can serve the chunk. chunk=8192 -> ctas_per_group=32
# -> num_groups=2 (2 rows in parallel) instead of num_groups=1 (serial rows).
# v0.3 experiment (2026-08-31): radix re-reads from global (no smem chunk),
# so CHUNK_SIZE is not smem-bound. Larger chunk -> fewer CTAs/group -> more
# groups (rows in parallel). 16384 -> ctas_per_group=16 -> num_groups=5.
# v0.3 C-plan: with radix fully global (re-read pass), CHUNK_SIZE no longer
# consumes smem at all, so the cap is raised to allow target_ctas_per_group=8:
# chunk=32768 -> ctas_per_group=8 -> num_groups ~19-20 (2 row iterations
# for 32 rows) with headroom = ctas_per_group.
SMEM_MEDIUM_QUAD_C_HY = 32768
# Decode path layout constants (see histogram_2048_topk in persistent_topk.cuh)
DECODE_SBASE = 8192 - 8  # 8184
DECODE_RHIST = RADIX + 128  # 384
DECODE_BOFF = 2 * DECODE_RHIST  # 768
DECODE_DBUF = (DECODE_SBASE - DECODE_BOFF) // 2  # 3708

logger = logging.getLogger(__name__)


@triton.jit
def _convert_to_uint32_v2(x):
    bits = x.to(tl.uint32, bitcast=True)
    return tl.where((bits & 0x80000000) != 0, ~bits, (bits | 0x80000000))


@triton.jit
def _convert_to_uint8(x):
    # FP16 high 8 bits -> 256-bin coarse key (order-preserving).
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    key = tl.where((bits & 0x8000) != 0, ~bits, bits | 0x8000)
    return (key >> 8).to(tl.int32)


@triton.jit
def _decode_bin(x):
    # FP16 high 11 bits -> 2048-bin decode key (order-preserving).
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    key = tl.where((bits & 0x8000) != 0, ~bits, bits | 0x8000)
    return (key >> 5).to(tl.int32)


@triton.jit
def _decode_bin512(x):
    # FP16 high 9 bits -> 512-bin coarse key (order-preserving). Used by the
    # medium path's Phase 1: a 512-bin tl.histogram costs ~5x less than the
    # 2048-bin per-element smem atomics (probe 2026-09-02), and the coarser
    # threshold bucket is resolved by rank-select / radix refinement anyway.
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    key = tl.where((bits & 0x8000) != 0, ~bits, bits | 0x8000)
    return (key >> 7).to(tl.int32)


@triton.jit
def _barrier_with_atomic_add(
    arrival_counter_ptr,
    zeros,
    lane,
    thresold,
):
    tl.atomic_add(
        arrival_counter_ptr + zeros,
        1,
        mask=lane == 0,
        sem="release",
        scope="gpu",
    )
    # Hygon: the generic version polled with sem="acquire"; on the HIP
    # backend the combination release-inc + acquire-poll deadlocks (spin
    # loop never observes the counter). Polling with relaxed atomics works
    # in the single-barrier case. The cooperative radix path is enabled with
    # the FlagTree hcu compiler fix (read-only scalar atomics bypass the LDS
    # staging). WITHOUT the compiler fix the scalar poll deadlocks, and the
    # vector-poll workaround (tl.max) only works in simple kernels — in this
    # kernel it also hangs (LDS-slot collision class, devlog 2026-08-24).
    arrival_counter = tl.atomic_add(
        arrival_counter_ptr,
        0,
        sem=None,
        scope="gpu",
    )
    # Hygon workaround (metax bounded-spin pattern, 2026-08-25): the plain
    # unbounded scalar-while poll triggers the needLdsStaging lowering and
    # hangs on the stock hcu backend (cross-CU). Adding a monotonically
    # increasing loop counter to the while condition fixes the lowering
    # (validated 3/3 on the stock compiler, exact counter). The limit is
    # far larger than any legitimate arrival wait.
    it = tl.zeros((), dtype=tl.int32)
    while (arrival_counter < thresold) & (it < BARRIER_SPIN_LIMIT):
        arrival_counter = tl.atomic_add(
            arrival_counter_ptr,
            0,
            sem=None,
            scope="gpu",
        )
        it += 1


# Extracted from persistent_topk.cuh in https://github.com/vllm-project/vllm
@triton.jit
def _radix_topk(
    row_input,
    row_output,
    seq_len,
    my_chunk_start,
    CHUNK_SIZE: tl.constexpr,
    local_histogram_ptr,
    suffix_sum_ptr,
    shared_scalars_ptr,
    shared_ordered_ptr,
    g_histogram_ptr,
    g_state_ptr,
    cta_in_group,
    ctas_per_group,
    barrier_phase,
    iter_idx,
    TOPK: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    RADIX: tl.constexpr = 256

    my_chunk_end = my_chunk_start + CHUNK_SIZE
    my_chunk_end = min(my_chunk_end, seq_len)
    actual_chunk_size = my_chunk_end - my_chunk_start if my_chunk_start < seq_len else 0
    lane = tl.arange(0, BLOCK_SIZE)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.uint32)
    zeros_2d = tl.zeros([BLOCK_SIZE, VEC_SIZE], dtype=tl.uint32)
    vec = tl.arange(0, VEC_SIZE)

    # -- Stage 1: (v0.3 experiment) global re-read, no smem chunk cache --
    # Chunk ownership (CHUNK_SIZE per CTA) is decoupled from smem: we no longer
    # buffer the chunk in shared_ordered, each radix pass re-reads + converts
    # on the fly (like torch mbtopk). Frees the 32KB chunk from smem budget so
    # CHUNK_SIZE can grow -> fewer CTAs per group -> more groups (row-parallel).
    n_vec_full = actual_chunk_size // (BLOCK_SIZE * VEC_SIZE)
    rem_tiles = (actual_chunk_size - n_vec_full * BLOCK_SIZE * VEC_SIZE) // BLOCK_SIZE
    rem_elems = actual_chunk_size % BLOCK_SIZE

    # -- Init radix select state --
    tl.store(shared_scalars_ptr + zeros, 0, mask=lane == 0)  # prefix
    tl.store(shared_scalars_ptr + 1 + zeros, TOPK, mask=lane == 0)  # remaining_k
    tl.debug_barrier()

    # Hygon fix: zero the round-0 global histogram buffer before the initial
    # barrier. The generic 3-buffer rotation only zeroes the NEXT round's
    # buffer, so the chain breaks when a non-radix (medium/decode) row sits
    # between radix rows: the round-0 buffer of the next radix row then
    # still holds stale counts from two radix rows earlier, corrupting the
    # histogram -> wrong top-k (and possibly overflowing the output counter
    # -> illegal memory access).
    if HYGON_RADIX_ROTATION_FIX and cta_in_group == 0:
        tl.store(
            g_histogram_ptr + ((iter_idx * 4) % 3) * RADIX + lane,
            0,
            mask=lane < RADIX,
        )

    # -- Initial barrier --
    _barrier_with_atomic_add(
        g_state_ptr + 2,
        zeros,
        lane,
        (barrier_phase + 1) * ctas_per_group,
    )
    barrier_phase += 1
    # tl.debug_barrier()

    if cta_in_group == 0:
        tl.store(g_state_ptr + 3 + zeros, 0, mask=lane == 0)  # output_counter
    # -- Stage 2: 4 rounds of radix select --
    for round_idx in tl.static_range(0, 4):
        global_round = iter_idx * 4 + round_idx
        shift_bits = 24 - round_idx * 8
        prefix = tl.load(shared_scalars_ptr)
        remaining_k = tl.load(shared_scalars_ptr + 1)

        # current_hist inited zero in host-side or pre iter of group
        current_hist_ptr = g_histogram_ptr + (global_round % 3) * RADIX
        next_hist_ptr = g_histogram_ptr + ((global_round + 1) % 3) * RADIX

        tl.store(local_histogram_ptr + lane, 0, mask=lane < RADIX)
        tl.debug_barrier()

        # TODO: no vec load from smem
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
            offs = base[:, None] + vec[None, :]
            x = tl.load(row_input + my_chunk_start + offs)
            ordered = _convert_to_uint32_v2(x)
            mask = 0 if round_idx == 0 else ((~0) << (32 - round_idx * 8))
            match = (ordered & mask) == prefix
            bucket = (ordered >> shift_bits) & 0xFF
            tl.atomic_add(
                local_histogram_ptr + bucket,
                1,
                mask=match,
                sem="relaxed",
                scope="cta",
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
            x = tl.load(row_input + my_chunk_start + offs)
            ordered = _convert_to_uint32_v2(x)
            mask = 0 if round_idx == 0 else ((~0) << (32 - round_idx * 8))
            match = (ordered & mask) == prefix
            bucket = (ordered >> shift_bits) & 0xFF
            tl.atomic_add(
                local_histogram_ptr + bucket,
                1,
                mask=match,
                sem="relaxed",
                scope="cta",
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(
                row_input + my_chunk_start + offs, mask=in_range, other=float("-inf")
            )
            ordered = _convert_to_uint32_v2(x)
            mask = 0 if round_idx == 0 else ((~0) << (32 - round_idx * 8))
            match = (ordered & mask) == prefix
            bucket = (ordered >> shift_bits) & 0xFF
            tl.atomic_add(
                local_histogram_ptr + bucket,
                1,
                mask=match & in_range,
                sem="relaxed",
                scope="cta",
            )
        tl.debug_barrier()

        counts = tl.load(local_histogram_ptr + lane, mask=lane < RADIX)
        tl.atomic_add(
            current_hist_ptr + lane,
            counts,
            mask=(counts > 0) & (lane < RADIX),
            sem="relaxed",
            scope="gpu",
        )
        if cta_in_group == 0:
            tl.store(next_hist_ptr + lane, 0, mask=lane < RADIX)
        _barrier_with_atomic_add(
            g_state_ptr + 2,
            zeros,
            lane,
            (barrier_phase + 1) * ctas_per_group,
        )
        barrier_phase += 1
        # tl.debug_barrier()
        g_counts = tl.load(current_hist_ptr + lane, mask=lane < RADIX, other=0)
        tl.store(suffix_sum_ptr + lane, g_counts, mask=lane < RADIX)
        tl.debug_barrier()

        for t in tl.static_range(0, 8):
            val = tl.load(suffix_sum_ptr + lane, mask=lane < RADIX)
            other_offs = lane + (1 << t)
            tmp = tl.load(suffix_sum_ptr + other_offs, mask=other_offs < RADIX, other=0)
            val += tmp
            tl.debug_barrier()
            tl.store(suffix_sum_ptr + lane, val, mask=lane < RADIX)
            tl.debug_barrier()

        tl.store(shared_scalars_ptr + 2 + zeros, 0, mask=lane == 0)  # threshold_bin
        tl.store(
            shared_scalars_ptr + 3 + zeros, remaining_k, mask=lane == 0
        )  # next_remaining_k
        tl.debug_barrier()

        count_ge = tl.load(suffix_sum_ptr + lane, mask=lane < RADIX, other=0)
        count_gt = tl.load(suffix_sum_ptr + lane + 1, mask=(lane + 1) < RADIX, other=0)
        threshold_mask = (
            (count_ge >= remaining_k) & (count_gt < remaining_k) & (lane < RADIX)
        )
        tl.store(shared_scalars_ptr + 2 + zeros, lane, mask=threshold_mask)
        tl.store(
            shared_scalars_ptr + 3 + zeros, remaining_k - count_gt, mask=threshold_mask
        )
        tl.debug_barrier()

        threshold_bin = tl.load(shared_scalars_ptr + 2 + zeros, mask=lane == 0, other=0)
        new_prefix = prefix | (threshold_bin << shift_bits)
        tl.store(shared_scalars_ptr + zeros, new_prefix, mask=lane == 0)
        next_remaining_k = tl.load(
            shared_scalars_ptr + 3 + zeros, mask=lane == 0, other=0
        )
        tl.store(shared_scalars_ptr + 1 + zeros, next_remaining_k, mask=lane == 0)
        tl.debug_barrier()
    # end 4 radix rounds

    # -- Count local > pivot elements --
    ordered_pivot = tl.load(shared_scalars_ptr)
    my_gt_count = tl.full((BLOCK_SIZE,), 0, dtype=tl.uint32)
    # TODO: no vec load from smem
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_input + my_chunk_start + offs)
        ordered = _convert_to_uint32_v2(x)
        gt_mask = ordered > ordered_pivot
        my_gt_count += tl.sum(gt_mask.to(tl.uint32), axis=-1)
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        x = tl.load(row_input + my_chunk_start + offs)
        ordered = _convert_to_uint32_v2(x)
        gt_mask = ordered > ordered_pivot
        my_gt_count += gt_mask.to(tl.uint32)
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        x = tl.load(
            row_input + my_chunk_start + offs, mask=in_range, other=float("-inf")
        )
        ordered = _convert_to_uint32_v2(x)
        gt_mask = (ordered > ordered_pivot) & in_range
        my_gt_count += gt_mask.to(tl.uint32)
    tl.debug_barrier()
    local_gt_count = tl.sum(my_gt_count)

    # -- Stage 3: Collect top-k indices --
    tl.store(local_histogram_ptr + zeros, 0, mask=lane == 0)
    gt_pos = tl.atomic_add(
        g_state_ptr + 3 + zeros,
        local_gt_count,
        mask=(lane == 0) & (local_gt_count > 0),
        sem="relaxed",
        scope="gpu",
    )
    tl.store(local_histogram_ptr + 1 + zeros, gt_pos, mask=lane == 0)
    tl.debug_barrier()
    gt_pos = tl.load(local_histogram_ptr + 1 + zeros)

    # TODO: no vec load from smem
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_input + my_chunk_start + offs)
        ordered = _convert_to_uint32_v2(x)
        gt_mask = ordered > ordered_pivot
        local_pos = tl.atomic_add(
            local_histogram_ptr + zeros_2d,
            1,
            mask=gt_mask,
            sem="relaxed",
            scope="cta",
        )
        tl.store(
            row_output + gt_pos[:, None] + local_pos,
            my_chunk_start + offs,
            mask=gt_mask,
        )
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        x = tl.load(row_input + my_chunk_start + offs)
        ordered = _convert_to_uint32_v2(x)
        gt_mask = ordered > ordered_pivot
        local_pos = tl.atomic_add(
            local_histogram_ptr + zeros,
            1,
            mask=gt_mask,
            sem="relaxed",
            scope="cta",
        )
        tl.store(row_output + gt_pos + local_pos, my_chunk_start + offs, mask=gt_mask)
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        x = tl.load(
            row_input + my_chunk_start + offs, mask=in_range, other=float("-inf")
        )
        ordered = _convert_to_uint32_v2(x)
        gt_mask = (ordered > ordered_pivot) & in_range
        local_pos = tl.atomic_add(
            local_histogram_ptr + zeros,
            1,
            mask=gt_mask,
            sem="relaxed",
            scope="cta",
        )
        tl.store(row_output + gt_pos + local_pos, my_chunk_start + offs, mask=gt_mask)

    _barrier_with_atomic_add(
        g_state_ptr + 2,
        zeros,
        lane,
        (barrier_phase + 1) * ctas_per_group,
    )
    barrier_phase += 1
    # tl.debug_barrier()

    # TODO: no vec load from smem
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_input + my_chunk_start + offs)
        ordered = _convert_to_uint32_v2(x)
        eq_mask = ordered == ordered_pivot
        eq_pos = tl.atomic_add(
            g_state_ptr + 3 + zeros_2d,
            1,
            mask=eq_mask,
            sem="relaxed",
            scope="gpu",
        )
        tl.store(
            row_output + eq_pos, my_chunk_start + offs, mask=eq_mask & (eq_pos < TOPK)
        )
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        x = tl.load(row_input + my_chunk_start + offs)
        ordered = _convert_to_uint32_v2(x)
        eq_mask = ordered == ordered_pivot
        eq_pos = tl.atomic_add(
            g_state_ptr + 3 + zeros,
            1,
            mask=eq_mask,
            sem="relaxed",
            scope="gpu",
        )
        tl.store(
            row_output + eq_pos, my_chunk_start + offs, mask=eq_mask & (eq_pos < TOPK)
        )
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        x = tl.load(
            row_input + my_chunk_start + offs, mask=in_range, other=float("-inf")
        )
        ordered = _convert_to_uint32_v2(x)
        eq_mask = (ordered == ordered_pivot) & in_range
        eq_pos = tl.atomic_add(
            g_state_ptr + 3 + zeros,
            1,
            mask=eq_mask,
            sem="relaxed",
            scope="gpu",
        )
        tl.store(
            row_output + eq_pos, my_chunk_start + offs, mask=eq_mask & (eq_pos < TOPK)
        )

    return barrier_phase

    # Medium path: 8K < seq_len <= 32K. Uses 2048-bin FP16-11bit histogram for


# Phase 1 (finer granularity → fewer elements in threshold bin), then 4-pass
# FP32 radix-256 refinement on the buffered threshold-bin elements.
@triton.jit
def _histogram_256_topk(
    row_input,
    row_output,
    seq_len,
    shared_ordered_ptr,
    buffered_ptr,
    shared_scalars_ptr,
    TOPK: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    RADIX: tl.constexpr = 256
    MEDIUM_BINS: tl.constexpr = 512
    MAX_BUFFERED_ITEMS: tl.constexpr = 2048
    BUF_TILES: tl.constexpr = (MAX_BUFFERED_ITEMS + BLOCK_SIZE - 1) // BLOCK_SIZE
    CLEAR_ROUNDS: tl.constexpr = MEDIUM_BINS // BLOCK_SIZE

    hist_ptr = shared_ordered_ptr
    hist0_ptr = shared_ordered_ptr
    hist1_ptr = shared_ordered_ptr + (RADIX + 128)
    # Hygon v0.3 (C-plan): the 2 x MAX_BUFFERED_ITEMS buffered_indices moved
    # out of smem into the per-row global buffered region (buffered_ptr).
    # medium_scalars stay in smem at [8, 13) of the shared scalars region.
    buffered_indices = buffered_ptr
    medium_scalars = shared_scalars_ptr + 8

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC_SIZE)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_2d = tl.zeros([BLOCK_SIZE, VEC_SIZE], dtype=tl.int32)

    n_vec_full = seq_len // (BLOCK_SIZE * VEC_SIZE)
    rem_tiles = (seq_len - n_vec_full * BLOCK_SIZE * VEC_SIZE) // BLOCK_SIZE
    rem_elems = seq_len % BLOCK_SIZE

    remaining_k = TOPK

    # -- Phase 1: 512-bin coarse histogram (FP16 high 9 bits) --
    # v0.5: per-tile tl.histogram(512) replaces the 2048-bin per-element smem
    # atomics (~5x faster at 512 threads: 53.6us vs 275.8us at sl=32768).
    # The coarser threshold bucket is resolved by rank-select or the 4-pass
    # radix fallback below.
    bins512 = tl.arange(0, MEDIUM_BINS)
    for r in tl.static_range(0, CLEAR_ROUNDS):
        tl.store(hist_ptr + r * BLOCK_SIZE + lane, 0)
    tl.debug_barrier()
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_input + offs)
        bin_flat = _decode_bin512(x).reshape(BLOCK_SIZE * VEC_SIZE)
        hist_t = tl.histogram(bin_flat, MEDIUM_BINS)
        tl.atomic_add(hist_ptr + bins512, hist_t, sem="relaxed", scope="cta")
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        x = tl.load(row_input + offs)
        bin = _decode_bin512(x)
        hist_t = tl.histogram(bin, MEDIUM_BINS)
        tl.atomic_add(hist_ptr + bins512, hist_t, sem="relaxed", scope="cta")
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        x = tl.load(row_input + offs, mask=in_range, other=float("-inf"))
        bin = _decode_bin512(x)
        tl.atomic_add(hist_ptr + bin, 1, mask=in_range, sem="relaxed", scope="cta")
    tl.debug_barrier()

    # -- Streaming threshold search via tle.cumsum --
    THRESHOLD_ROUNDS: tl.constexpr = MEDIUM_BINS // BLOCK_SIZE
    threshold_found = tl.full((), False, dtype=tl.int1)
    last_value = tl.zeros((), dtype=tl.int32)
    cutoff = seq_len - TOPK
    for round_idx in tl.static_range(0, THRESHOLD_ROUNDS):
        if not threshold_found:
            round_bins = round_idx * BLOCK_SIZE + lane
            round_counts = tl.load(hist_ptr + round_bins).to(tl.int32)
            ps, round_total = tle.cumsum(round_counts, axis=0, reverse=False)
            ps = ps + last_value
            cum_total = last_value + round_total
            nps = ps + round_counts
            thr_mask = (ps <= cutoff) & (nps > cutoff)
            tl.store(medium_scalars + 1 + zeros, round_bins, mask=thr_mask)
            tl.store(
                medium_scalars + 2 + zeros, (seq_len - nps).to(tl.int32), mask=thr_mask
            )
            tl.store(medium_scalars + 0 + zeros, 0, mask=thr_mask)
            threshold_found = tl.reduce_or(thr_mask, axis=0)
            last_value = cum_total
    tl.debug_barrier()

    threshold_bin = tl.load(medium_scalars + 1)
    count_above = tl.load(medium_scalars + 2).to(tl.int32)
    remaining_k = TOPK - count_above

    # -- Early return: all top-K above threshold --
    if remaining_k <= 0:
        tl.store(medium_scalars + 0 + zeros, 0, mask=lane == 0)
        tl.debug_barrier()
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
            offs = base[:, None] + vec[None, :]
            x = tl.load(row_input + offs)
            bin = _decode_bin512(x)
            above = (bin > threshold_bin).reshape(BLOCK_SIZE, VEC_SIZE)
            out_pos = tl.atomic_add(
                medium_scalars + 0 + zeros_2d, 1, mask=above, sem="relaxed", scope="cta"
            )
            tl.store(row_output + out_pos, offs, mask=above)
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
            x = tl.load(row_input + offs)
            above = _decode_bin512(x) > threshold_bin
            out_pos = tl.atomic_add(
                medium_scalars + 0 + zeros, 1, mask=above, sem="relaxed", scope="cta"
            )
            tl.store(row_output + out_pos, offs.to(tl.int32), mask=above)
        if rem_elems > 0:
            offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(row_input + offs, mask=in_range, other=float("-inf"))
            above = in_range & (_decode_bin512(x) > threshold_bin)
            out_pos = tl.atomic_add(
                medium_scalars + 0 + zeros, 1, mask=above, sem="relaxed", scope="cta"
            )
            tl.store(row_output + out_pos, offs.to(tl.int32), mask=above)
        tl.debug_barrier()
        return

    # -- Filter: output > threshold, buffer == threshold --
    # v0.5: the next-hist build moved out of this pass (into the 4-pass radix
    # fallback entry): it cost ~20us and is only needed when the buffered set
    # overflows the rank-select path.
    tl.store(medium_scalars + 0 + zeros, 0, mask=lane == 0)
    tl.store(medium_scalars + 2 + zeros, 0, mask=lane == 0)
    tl.debug_barrier()
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_input + offs)
        bin = _decode_bin512(x)
        above = (bin > threshold_bin).reshape(BLOCK_SIZE, VEC_SIZE)
        equal = (bin == threshold_bin).reshape(BLOCK_SIZE, VEC_SIZE)
        out_pos = tl.atomic_add(
            medium_scalars + 0 + zeros_2d, 1, mask=above, sem="relaxed", scope="cta"
        )
        tl.store(row_output + out_pos, offs, mask=above)
        buf_pos = tl.atomic_add(
            medium_scalars + 2 + zeros_2d, 1, mask=equal, sem="relaxed", scope="cta"
        )
        in_buf = equal & (buf_pos < MAX_BUFFERED_ITEMS)
        tl.store(buffered_indices + buf_pos, offs, mask=in_buf)
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        x = tl.load(row_input + offs)
        bin = _decode_bin512(x)
        above = bin > threshold_bin
        equal = bin == threshold_bin
        out_pos = tl.atomic_add(
            medium_scalars + 0 + zeros, 1, mask=above, sem="relaxed", scope="cta"
        )
        tl.store(row_output + out_pos, offs.to(tl.int32), mask=above)
        buf_pos = tl.atomic_add(
            medium_scalars + 2 + zeros, 1, mask=equal, sem="relaxed", scope="cta"
        )
        in_buf = equal & (buf_pos < MAX_BUFFERED_ITEMS)
        tl.store(buffered_indices + buf_pos, offs.to(tl.int32), mask=in_buf)
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        x = tl.load(row_input + offs, mask=in_range, other=float("-inf"))
        bin = _decode_bin512(x)
        above = in_range & (bin > threshold_bin)
        equal = in_range & (bin == threshold_bin)
        out_pos = tl.atomic_add(
            medium_scalars + 0 + zeros, 1, mask=above, sem="relaxed", scope="cta"
        )
        tl.store(row_output + out_pos, offs.to(tl.int32), mask=above)
        buf_pos = tl.atomic_add(
            medium_scalars + 2 + zeros, 1, mask=equal, sem="relaxed", scope="cta"
        )
        in_buf = equal & (buf_pos < MAX_BUFFERED_ITEMS)
        tl.store(buffered_indices + buf_pos, offs.to(tl.int32), mask=in_buf)
    tl.debug_barrier()

    # -- Short circuit: if all buffered elements fit, output directly --
    raw_buf0 = tl.load(medium_scalars + 2)
    num_buffered = tl.minimum(raw_buf0, MAX_BUFFERED_ITEMS)
    if num_buffered <= remaining_k and remaining_k > 0:
        out_base = tl.load(medium_scalars + 0)
        for b in tl.range(0, BUF_TILES):
            offs_b = b * BLOCK_SIZE + lane
            valid = offs_b < num_buffered
            idx = tl.load(buffered_indices + offs_b, mask=valid, other=0)
            tl.store(row_output + out_base + offs_b, idx, mask=valid)
        tl.debug_barrier()
        return

    # v0.5: rank-based direct selection for small buffered sets (same fix as
    # the decode path; the 4-pass radix refinement below costs ~42us of
    # barrier overhead on DCU regardless of the buffered count). Keys are
    # staged in the hist0 region, which is dead after the threshold search.
    RANK_SELECT_THRESHOLD: tl.constexpr = 256
    if num_buffered <= RANK_SELECT_THRESHOLD:
        out_base = tl.load(medium_scalars + 0)
        num_rank_tiles = tl.cdiv(num_buffered, BLOCK_SIZE)
        for st in tl.range(0, num_rank_tiles):
            offs = st * BLOCK_SIZE + lane
            valid = offs < num_buffered
            idx = tl.load(buffered_indices + offs, mask=valid, other=0)
            logit_val = tl.load(row_input + idx, mask=valid, other=float("-inf"))
            tl.store(hist0_ptr + offs, _convert_to_uint32_v2(logit_val), mask=valid)
        tl.debug_barrier()
        for st in tl.range(0, num_rank_tiles):
            offs = st * BLOCK_SIZE + lane
            valid = offs < num_buffered
            idx = tl.load(buffered_indices + offs, mask=valid, other=0)
            my_key = tl.load(hist0_ptr + offs, mask=valid, other=0)
            rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
            for j in tl.range(0, num_buffered):
                other_key = tl.load(hist0_ptr + j)
                better = (my_key < other_key) | ((my_key == other_key) & (offs < j))
                rank += (valid & better).to(tl.int32)
            take = valid & (rank < remaining_k)
            tl.store(row_output + out_base + rank, idx, mask=take)
        tl.debug_barrier()
        return

    # -- 4-pass radix refinement --
    # v0.5: build the round-0 refine histogram here (from the buffered
    # elements) instead of during Phase 2; only the fallback path needs it.
    tl.store(hist0_ptr + lane, 0, mask=lane < (RADIX + 1))
    tl.debug_barrier()
    for b in tl.range(0, BUF_TILES):
        offs_b = b * BLOCK_SIZE + lane
        valid_b = offs_b < num_buffered
        idx = tl.load(buffered_indices + offs_b, mask=valid_b, other=0)
        logit_val = tl.load(row_input + idx, mask=valid_b, other=float("-inf"))
        fp32_bits = _convert_to_uint32_v2(logit_val)
        next_bin = (fp32_bits >> 24) & 0xFF
        tl.atomic_add(hist0_ptr + next_bin, 1, mask=valid_b, sem="relaxed", scope="cta")
    tl.debug_barrier()

    for pass_idx in tl.static_range(0, 4):
        if remaining_k > 0:
            src_buffer = pass_idx % 2
            dst_buffer = src_buffer ^ 1
            bit_offset: tl.constexpr = 24 - pass_idx * 8
            raw_buffered = tl.load(medium_scalars + 2 + src_buffer)
            num_buffered = tl.minimum(raw_buffered, MAX_BUFFERED_ITEMS)

            for st in tl.static_range(0, 8):
                stride = 1 << st
                sb = hist0_ptr if (st & 1) == 0 else hist1_ptr
                db = hist1_ptr if (st & 1) == 0 else hist0_ptr
                val = tl.load(sb + lane, mask=lane < RADIX, other=0)
                tmp = tl.load(sb + lane + stride, mask=(lane + stride) < RADIX, other=0)
                tl.store(db + lane, val + tmp, mask=lane < RADIX)
                tl.debug_barrier()

            count_ge = tl.load(hist0_ptr + lane, mask=lane < RADIX, other=0).to(
                tl.int32
            )
            count_gt = tl.load(
                hist0_ptr + lane + 1, mask=(lane + 1) < RADIX, other=0
            ).to(tl.int32)
            thr_mask = (
                (count_ge > remaining_k) & (count_gt <= remaining_k) & (lane < RADIX)
            )
            tl.store(medium_scalars + 1 + zeros, lane, mask=thr_mask)
            tl.store(medium_scalars + 2 + dst_buffer + zeros, 0, mask=thr_mask)
            tl.store(medium_scalars + 4 + zeros, remaining_k - count_gt, mask=thr_mask)
            tl.debug_barrier()

            threshold_bin = tl.load(medium_scalars + 1)
            remaining_k = remaining_k - tl.load(
                hist0_ptr + threshold_bin + 1, mask=(threshold_bin + 1) < RADIX, other=0
            ).to(tl.int32)

            if remaining_k == 0:
                for b in tl.range(0, BUF_TILES):
                    offs_b = b * BLOCK_SIZE + lane
                    valid_b = offs_b < num_buffered
                    buf_idx = tl.load(
                        buffered_indices + src_buffer * MAX_BUFFERED_ITEMS + offs_b,
                        mask=valid_b,
                        other=0,
                    )
                    logit_val = tl.load(
                        row_input + buf_idx, mask=valid_b, other=float("-inf")
                    )
                    bin = (_convert_to_uint32_v2(logit_val) >> bit_offset) & 0xFF
                    above = valid_b & (bin > threshold_bin)
                    out_pos = tl.atomic_add(
                        medium_scalars + 0 + zeros,
                        1,
                        mask=above,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(row_output + out_pos, buf_idx, mask=above)
                tl.debug_barrier()
                remaining_k = tl.full((), -1, dtype=tl.int32)

            if remaining_k > 0:
                tl.store(hist0_ptr + lane, 0, mask=lane < (RADIX + 1))
                tl.debug_barrier()
                curent_buf = buffered_indices + src_buffer * MAX_BUFFERED_ITEMS
                next_buf = buffered_indices + dst_buffer * MAX_BUFFERED_ITEMS
                for b in tl.range(0, BUF_TILES):
                    offs_b = b * BLOCK_SIZE + lane
                    valid_b = offs_b < num_buffered
                    buf_idx = tl.load(curent_buf + offs_b, mask=valid_b, other=0)
                    logit_val = tl.load(
                        row_input + buf_idx, mask=valid_b, other=float("-inf")
                    )
                    fp32_bits = _convert_to_uint32_v2(logit_val)
                    bin = (fp32_bits >> bit_offset) & 0xFF
                    above = valid_b & (bin > threshold_bin)
                    equal = valid_b & (bin == threshold_bin)
                    out_pos = tl.atomic_add(
                        medium_scalars + 0 + zeros,
                        1,
                        mask=above,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(row_output + out_pos, buf_idx, mask=above)
                    if pass_idx == 3:
                        slot = tl.atomic_add(
                            medium_scalars + 4 + zeros,
                            -1,
                            mask=equal,
                            sem="relaxed",
                            scope="cta",
                        ).to(tl.int32)
                        take = equal & (slot > 0)
                        tl.store(row_output + (TOPK - slot), buf_idx, mask=take)
                    else:
                        buffer_pos = tl.atomic_add(
                            medium_scalars + 2 + dst_buffer + zeros,
                            1,
                            mask=equal,
                            sem="relaxed",
                            scope="cta",
                        )
                        in_buf = equal & (buffer_pos < MAX_BUFFERED_ITEMS)
                        tl.store(next_buf + buffer_pos, buf_idx, mask=in_buf)
                        next_bin = (fp32_bits >> (bit_offset - 8)) & 0xFF
                        tl.atomic_add(
                            hist0_ptr + next_bin,
                            1,
                            mask=in_buf,
                            sem="relaxed",
                            scope="cta",
                        )
                tl.debug_barrier()


# histogram_2048_topk — production xiao implementation


@triton.jit
def _histogram_2048_topk(
    row_input,
    row_output,
    seq_len,
    shared_ordered_ptr,
    buffered_ptr,
    shared_scalars_ptr,
    TOPK: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    DECODE_BINS_C: tl.constexpr = 2048
    RADIX_C: tl.constexpr = 256
    RHIST_C: tl.constexpr = RADIX_C + 128  # 384
    # Hygon v0.3 (C-plan): the 2 x DBUF_C element buffers moved out of smem
    # into the per-row global buffered region (buffered_ptr). smem keeps only
    # the 2048-bin histogram [0, 2048) (shared_ordered) and the 8 scalars
    # [0, 8) (shared_scalars), letting per-CTA smem drop below 21.8KB so 3
    # CTAs/SM fit (occupancy 1 -> 3).
    DECODE_SBASE_C: tl.constexpr = 0
    DBUF_C: tl.constexpr = 3708
    NUM_BUF_TILES: tl.constexpr = (DBUF_C + BLOCK_SIZE - 1) // BLOCK_SIZE

    hist_ptr = shared_ordered_ptr
    buf0_ptr = buffered_ptr
    buf1_ptr = buffered_ptr + DBUF_C
    scalars_ptr = shared_scalars_ptr + DECODE_SBASE_C  # 8 scalars at [0, 8)
    # Hygon: the generic version cached per-element bins in smem at
    # [8192, 8192+seq_len) for Phase 2 replay, but that pushes shared_ordered
    # to 16384 u32 = 64KB which, together with the tle.cumsum scratch, exceeds
    # the DCU 64KB smem/block limit. Phase 2 instead re-reads logits from
    # global memory (seq_len <= 8192, cheap).

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC_SIZE)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_2d = tl.zeros([BLOCK_SIZE, VEC_SIZE], dtype=tl.int32)
    # Hygon: DCU caps threads/block at 1024; a tl.arange(0, 2048) would
    # require 2048 threads on the HIP backend, so clear the 2048-bin
    # histogram in two BLOCK_SIZE rounds instead.
    CLEAR_ROUNDS_D: tl.constexpr = DECODE_BINS_C // BLOCK_SIZE  # 2

    n_vec_full = seq_len // (BLOCK_SIZE * VEC_SIZE)
    rem_tiles = (seq_len - n_vec_full * BLOCK_SIZE * VEC_SIZE) // BLOCK_SIZE
    rem_elems = seq_len % BLOCK_SIZE

    # scalar slot enum: sTHR=0, sOUT=1, sREF=2, sFIN=3, sBUF0=4, sBUF1=5
    tl.store(scalars_ptr + 0 + zeros, 0, mask=lane == 0)  # sTHR
    tl.store(scalars_ptr + 1 + zeros, 0, mask=lane == 0)  # sOUT
    tl.store(scalars_ptr + 2 + zeros, 0, mask=lane == 0)  # sREF
    tl.store(scalars_ptr + 3 + zeros, 0, mask=lane == 0)  # sFIN
    tl.store(scalars_ptr + 4 + zeros, 0, mask=lane == 0)  # sBUF0
    tl.store(scalars_ptr + 5 + zeros, 0, mask=lane == 0)  # sBUF1

    # -- Phase 1: build 2048-bin histogram --
    for cr in tl.static_range(0, CLEAR_ROUNDS_D):
        tl.store(hist_ptr + cr * BLOCK_SIZE + lane, 0)
    tl.debug_barrier()
    # Vectorized tiles: 4 elements per thread per load (vLLM float4 pattern)
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_input + offs)
        bin = _decode_bin(x)
        bin_flat = bin.reshape(BLOCK_SIZE * VEC_SIZE)
        tl.atomic_add(hist_ptr + bin_flat, 1, sem="relaxed", scope="cta")
    # Scalar tiles: tail elements that don't fill a full vec tile
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        x = tl.load(row_input + offs)
        bin = _decode_bin(x)
        tl.atomic_add(hist_ptr + bin, 1, sem="relaxed", scope="cta")
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        x = tl.load(row_input + offs, mask=in_range, other=float("-inf"))
        bin = _decode_bin(x)
        tl.atomic_add(hist_ptr + bin, 1, mask=in_range, sem="relaxed", scope="cta")
    tl.debug_barrier()

    # -- Streaming suffix sum via tle.cumsum on [BLOCK_SIZE] --
    THRESHOLD_ROUNDS: tl.constexpr = DECODE_BINS_C // BLOCK_SIZE
    threshold_found = tl.full((), False, dtype=tl.int1)
    last_value = tl.zeros((), dtype=tl.int32)
    cutoff = seq_len - TOPK
    for round_idx in tl.static_range(0, THRESHOLD_ROUNDS):
        if not threshold_found:
            round_bins = round_idx * BLOCK_SIZE + lane
            round_counts = tl.load(hist_ptr + round_bins).to(tl.int32)
            ps, round_total = tle.cumsum(round_counts, axis=0, reverse=False)
            ps = ps + last_value
            cum_total = last_value + round_total
            nps = ps + round_counts
            thr_mask = (ps <= cutoff) & (nps > cutoff)
            tl.store(scalars_ptr + 0 + zeros, round_bins, mask=thr_mask)
            tl.store(
                scalars_ptr + 2 + zeros, (seq_len - nps).to(tl.int32), mask=thr_mask
            )
            threshold_found = tl.reduce_or(thr_mask, axis=0)
            last_value = cum_total
    tl.debug_barrier()
    threshold_bin = tl.load(scalars_ptr + 0)
    count_above = tl.load(scalars_ptr + 2).to(tl.int32)
    remaining_k = TOPK - count_above

    # -- Phase 2: re-read logits, output above / buffer equal --
    # Vectorized tiles (same pattern as Phase 1)
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_input + offs)
        bin_flat = _decode_bin(x).reshape(BLOCK_SIZE * VEC_SIZE)
        above = (bin_flat > threshold_bin).reshape(BLOCK_SIZE, VEC_SIZE)
        equal = (bin_flat == threshold_bin).reshape(BLOCK_SIZE, VEC_SIZE)
        out_pos = tl.atomic_add(
            scalars_ptr + 1 + zeros_2d, 1, mask=above, sem="relaxed", scope="cta"
        )
        tl.store(row_output + out_pos, offs, mask=above)
        buf_pos = tl.atomic_add(
            scalars_ptr + 4 + zeros_2d, 1, mask=equal, sem="relaxed", scope="cta"
        )
        in_buf = equal & (buf_pos < DBUF_C)
        tl.store(buf0_ptr + buf_pos, offs, mask=in_buf)
    # Scalar tiles
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        x = tl.load(row_input + offs)
        bin = _decode_bin(x)
        above = bin > threshold_bin
        equal = bin == threshold_bin
        out_pos = tl.atomic_add(
            scalars_ptr + 1 + zeros, 1, mask=above, sem="relaxed", scope="cta"
        )
        tl.store(row_output + out_pos, offs, mask=above)
        buf_pos = tl.atomic_add(
            scalars_ptr + 4 + zeros, 1, mask=equal, sem="relaxed", scope="cta"
        )
        in_buf = equal & (buf_pos < DBUF_C)
        tl.store(buf0_ptr + buf_pos, offs, mask=in_buf)
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        x = tl.load(row_input + offs, mask=in_range, other=float("-inf"))
        bin = _decode_bin(x)
        above = (bin > threshold_bin) & in_range
        equal = (bin == threshold_bin) & in_range
        out_pos = tl.atomic_add(
            scalars_ptr + 1 + zeros, 1, mask=above, sem="relaxed", scope="cta"
        )
        tl.store(row_output + out_pos, offs, mask=above)
        buf_pos = tl.atomic_add(
            scalars_ptr + 4 + zeros, 1, mask=equal, sem="relaxed", scope="cta"
        )
        in_buf = equal & (buf_pos < DBUF_C)
        tl.store(buf0_ptr + buf_pos, offs, mask=in_buf)
    tl.debug_barrier()

    # -- If buffered <= remaining_k: output all buffered, return --
    raw_buf0 = tl.load(scalars_ptr + 4)
    num_buffered = tl.minimum(raw_buf0, DBUF_C)
    if num_buffered <= remaining_k:
        out_base = tl.load(scalars_ptr + 1)  # sOUT
        for st in tl.static_range(0, NUM_BUF_TILES):
            offs = st * BLOCK_SIZE + lane
            valid = offs < num_buffered
            idx = tl.load(buf0_ptr + offs, mask=valid, other=0)
            tl.store(row_output + out_base + offs, idx, mask=valid)
        tl.debug_barrier()
        return

    # v0.5: the 4-pass radix refinement below costs ~42us on DCU even for a
    # handful of buffered elements (64+ barriers per pass dominate), which
    # was the real decode bottleneck. For small buffered sets (the common
    # case: the 2048-bin threshold bucket holds ~10-30 elements), rank each
    # element against all others directly (O(n^2) scalar loop) and scatter.
    RANK_SELECT_THRESHOLD: tl.constexpr = 256
    if num_buffered <= RANK_SELECT_THRESHOLD:
        out_base = tl.load(scalars_ptr + 1)  # sOUT
        # Stage buffered order-preserving keys into smem once (hist is dead
        # after the threshold search, so reuse its region): the scalar j-loop
        # below reads LDS instead of paying ~500ns per global scalar load.
        num_rank_tiles = tl.cdiv(num_buffered, BLOCK_SIZE)
        for st in tl.range(0, num_rank_tiles):
            offs = st * BLOCK_SIZE + lane
            valid = offs < num_buffered
            idx = tl.load(buf0_ptr + offs, mask=valid, other=0)
            logit_val = tl.load(row_input + idx, mask=valid, other=float("-inf"))
            tl.store(hist_ptr + offs, _convert_to_uint32_v2(logit_val), mask=valid)
        tl.debug_barrier()
        for st in tl.range(0, num_rank_tiles):
            offs = st * BLOCK_SIZE + lane
            valid = offs < num_buffered
            idx = tl.load(buf0_ptr + offs, mask=valid, other=0)
            my_key = tl.load(hist_ptr + offs, mask=valid, other=0)
            rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
            for j in tl.range(0, num_buffered):
                other_key = tl.load(hist_ptr + j)
                better = (my_key < other_key) | ((my_key == other_key) & (offs < j))
                rank += (valid & better).to(tl.int32)
            take = valid & (rank < remaining_k)
            tl.store(row_output + out_base + rank, idx, mask=take)
        tl.debug_barrier()
        return

    # -- Phase 3: deferred 4-pass radix refinement on buffered elements --
    refine0_ptr = hist_ptr  # refine[0] at [0, RHIST_C)
    # Build initial refine histogram from FP32 MSB of buffered elements
    tl.store(refine0_ptr + lane, 0, mask=lane < RHIST_C)
    tl.debug_barrier()
    for st in tl.static_range(0, NUM_BUF_TILES):
        offs = st * BLOCK_SIZE + lane
        valid = offs < num_buffered
        idx = tl.load(buf0_ptr + offs, mask=valid, other=0)
        logit_val = tl.load(row_input + idx, mask=valid, other=float("-inf"))
        fp32_bits = _convert_to_uint32_v2(logit_val)
        next_bin = (fp32_bits >> 24) & 0xFF
        tl.atomic_add(refine0_ptr + next_bin, 1, mask=valid, sem="relaxed", scope="cta")
    tl.debug_barrier()

    for pass_idx in tl.static_range(0, 4):
        if remaining_k > 0:
            src_buf_ptr = buf0_ptr if (pass_idx % 2) == 0 else buf1_ptr
            dst_buf_ptr = buf1_ptr if (pass_idx % 2) == 0 else buf0_ptr
            buf_count_src_idx: tl.constexpr = 4 + (pass_idx % 2)
            buf_count_dst_idx: tl.constexpr = 4 + ((pass_idx % 2) ^ 1)
            bit_offset: tl.constexpr = 24 - pass_idx * 8

            # Suffix sum on refine histogram (8-step in-place)
            for st in tl.static_range(0, 8):
                val = tl.load(refine0_ptr + lane, mask=lane < RADIX_C, other=0)
                other_offs = lane + (1 << st)
                tmp = tl.load(
                    refine0_ptr + other_offs, mask=other_offs < RADIX_C, other=0
                )
                val = val + tmp
                tl.debug_barrier()
                tl.store(refine0_ptr + lane, val, mask=lane < RADIX_C)
                tl.debug_barrier()

            # Find threshold
            count_ge = tl.load(refine0_ptr + lane, mask=lane < RADIX_C, other=0).to(
                tl.int32
            )
            count_gt = tl.load(
                refine0_ptr + lane + 1, mask=(lane + 1) < RADIX_C, other=0
            ).to(tl.int32)
            threshold_mask = (
                (count_ge > remaining_k) & (count_gt <= remaining_k) & (lane < RADIX_C)
            )
            tl.store(scalars_ptr + 2 + zeros, 0, mask=lane == 0)  # sREF (threshold)
            tl.store(scalars_ptr + buf_count_dst_idx + zeros, 0, mask=lane == 0)
            tl.debug_barrier()
            tl.store(scalars_ptr + 2 + zeros, lane, mask=threshold_mask)
            tl.store(
                scalars_ptr + 3 + zeros, remaining_k - count_gt, mask=threshold_mask
            )  # sFIN
            tl.debug_barrier()

            ref_thr = tl.load(scalars_ptr + 2)
            count_gt_val = tl.load(
                refine0_ptr + ref_thr + 1, mask=(ref_thr + 1) < RADIX_C, other=0
            ).to(tl.int32)
            remaining_k = remaining_k - count_gt_val
            raw_buffered = tl.load(scalars_ptr + buf_count_src_idx)
            num_buf = tl.minimum(raw_buffered, DBUF_C)

            if remaining_k == 0:
                for st in tl.static_range(0, NUM_BUF_TILES):
                    offs = st * BLOCK_SIZE + lane
                    valid = offs < num_buf
                    idx = tl.load(src_buf_ptr + offs, mask=valid, other=0)
                    logit_val = tl.load(
                        row_input + idx, mask=valid, other=float("-inf")
                    )
                    fp32_bits = _convert_to_uint32_v2(logit_val)
                    bin = (fp32_bits >> bit_offset) & 0xFF
                    above = valid & (bin > ref_thr)
                    out_pos = tl.atomic_add(
                        scalars_ptr + 1 + zeros,
                        1,
                        mask=above,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(row_output + out_pos, idx, mask=above)
                tl.debug_barrier()
                remaining_k = tl.full((), -1, dtype=tl.int32)
            if remaining_k > 0:
                tl.store(refine0_ptr + lane, 0, mask=lane < RHIST_C)
                tl.debug_barrier()
                for st in tl.static_range(0, NUM_BUF_TILES):
                    offs = st * BLOCK_SIZE + lane
                    valid = offs < num_buf
                    idx = tl.load(src_buf_ptr + offs, mask=valid, other=0)
                    logit_val = tl.load(
                        row_input + idx, mask=valid, other=float("-inf")
                    )
                    fp32_bits = _convert_to_uint32_v2(logit_val)
                    bin = (fp32_bits >> bit_offset) & 0xFF
                    above = valid & (bin > ref_thr)
                    equal = valid & (bin == ref_thr)
                    out_pos = tl.atomic_add(
                        scalars_ptr + 1 + zeros,
                        1,
                        mask=above,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(row_output + out_pos, idx, mask=above)
                    if pass_idx == 3:
                        slot = tl.atomic_add(
                            scalars_ptr + 3 + zeros,
                            -1,
                            mask=equal,
                            sem="relaxed",
                            scope="cta",
                        ).to(tl.int32)
                        take = equal & (slot > 0)
                        tl.store(row_output + (TOPK - slot), idx, mask=take)
                    else:
                        buf_pos = tl.atomic_add(
                            scalars_ptr + buf_count_dst_idx + zeros,
                            1,
                            mask=equal,
                            sem="relaxed",
                            scope="cta",
                        )
                        in_buf = equal & (buf_pos < DBUF_C)
                        tl.store(dst_buf_ptr + buf_pos, idx, mask=in_buf)
                        next_bin = (fp32_bits >> (bit_offset - 8)) & 0xFF
                        tl.atomic_add(
                            refine0_ptr + next_bin,
                            1,
                            mask=in_buf,
                            sem="relaxed",
                            scope="cta",
                        )
                tl.debug_barrier()


@triton.jit
def persistent_topk_kernel(
    logits_ptr,
    output_ptr,
    lengths_ptr,
    num_rows,
    stride,
    TOPK: tl.constexpr,
    max_seq_len,
    CHUNK_SIZE: tl.constexpr,
    ctas_per_group,
    num_groups,
    g_histogram_ptr,
    g_state_ptr,
    g_buffered_ptr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    MEDIUM_THRESHOLD: tl.constexpr = 49152
    RADIX: tl.constexpr = 256
    HIST2048_THRESHOLD: tl.constexpr = 8192
    # Hygon v0.3 (C-plan): per-row stride of the global buffered region.
    # Chosen as max(2 x DBUF_C decode, 2 x MAX_BUFFERED_ITEMS medium) so both
    # the decode and the medium path can use the same per-row slot.
    BUFFERED_STRIDE: tl.constexpr = 7416

    pid = tl.program_id(0)
    group_id = pid // ctas_per_group
    cta_in_group = pid % ctas_per_group
    if pid >= num_groups * ctas_per_group:
        return  # TODO: remove
    if cta_in_group != 0 and max_seq_len <= MEDIUM_THRESHOLD:
        return
    # shared_ordered is reused (via offset pointers) by the medium and
    # decode paths, mirroring vLLM's single extern __shared__ buffer.
    # Hygon (DCU, gfx936): smem/block = 64KB and tle.gpu.alloc requires pow2.
    # tle.cumsum on [BLOCK_SIZE] additionally consumes 2*BLOCK_SIZE u32 of
    # hidden scratch. v0.3 (C-plan): decode/medium buffered buffers moved to
    # the per-row global region, so shared_ordered shrinks 8192 -> 4096 u32
    # (16KB) and per-CTA smem drops to ~22KB -> occupancy 2 (160 resident
    # CTAs instead of 80).
    # Layout within shared_ordered (2048 u32):
    #   [0, 2048)                   decode/medium hist (2048 bins)
    # Layout within shared_scalars (32 u32):
    #   [0, 8)                      decode scalars
    #   [8, 13)                     medium scalars
    #   [16, 20)                    radix scalars (kept separate from
    #                               decode/medium; all CTAs use them)
    # Notes vs the generic NVIDIA version:
    #   * decode bins replay buffer removed: Phase 2 re-reads logits from
    #     global memory (seq_len <= 8192, cheap) to stay within 64KB;
    #   * radix small buffers (local_histogram, suffix_sum, scalars) were
    #     carved from the tail instead of separate tle.gpu.alloc calls.
    SMEM_MEDIUM_QUAD_C: tl.constexpr = 2048
    # v0.3: radix re-reads from global (no smem chunk cache); v0.3 C-plan +
    # occupancy-3 step: shared_ordered holds only the hist (2048 u32);
    # scalars live in the shared_scalars region (independent of CHUNK_SIZE).
    ordered_size: tl.constexpr = SMEM_MEDIUM_QUAD_C
    shared_ordered = tle.gpu.alloc(
        [ordered_size],
        dtype=tl.uint32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    local_histogram = tle.gpu.alloc(
        [RADIX],
        dtype=tl.uint32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    suffix_sum = tle.gpu.alloc(
        [RADIX],
        dtype=tl.uint32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    shared_scalars = tle.gpu.alloc(
        [32],
        dtype=tl.uint32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    shared_ordered_ptr = tle.gpu.local_ptr(shared_ordered, (0,))
    local_histogram_ptr = tle.gpu.local_ptr(local_histogram, (0,))
    suffix_sum_ptr = tle.gpu.local_ptr(suffix_sum, (0,))
    shared_scalars_ptr = tle.gpu.local_ptr(shared_scalars, (0,))
    decode_scalars_ptr = shared_scalars_ptr + 0
    medium_scalars_ptr = shared_scalars_ptr + 8
    radix_scalars_ptr = shared_scalars_ptr + 16

    g_histogram_ptr += group_id * 3 * RADIX
    g_state_ptr += group_id * 4
    barrier_phase = tl.zeros((), dtype=tl.uint32)
    total_iters = tl.cdiv(num_rows, num_groups)
    for i in tl.range(total_iters):
        row_idx = group_id + i * num_groups
        if row_idx < num_rows:
            seq_len = tl.load(lengths_ptr + row_idx)
            row_output = output_ptr + row_idx * TOPK
            row_in = tl.multiple_of(logits_ptr + row_idx * stride, VEC_SIZE * 4)
            # Hygon v0.3 (C-plan): per-row slot in the global buffered region.
            # Rows are processed by a single group at a time (row_idx =
            # group_id + i * num_groups), so indexing by row_idx gives each
            # row its own slot with no cross-CTA aliasing.
            buffered_ptr = g_buffered_ptr + row_idx * BUFFERED_STRIDE
            # Hygon v0.1: single-CTA paths for every row; the cooperative
            # radix path stays disabled (see the v0.1 devlog for the
            # cross-CTA barrier/TLE analysis).
            if seq_len <= MEDIUM_THRESHOLD:
                if cta_in_group == 0:
                    if seq_len <= TOPK:
                        num_tiles: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
                        lane = tl.arange(0, BLOCK_SIZE)
                        for tile_idx in tl.static_range(0, num_tiles):
                            pos = tile_idx * BLOCK_SIZE + lane
                            take_row = pos < seq_len
                            tl.store(
                                row_output + pos,
                                pos.to(tl.int32),
                                mask=take_row,
                            )
                            take_pad = (pos >= seq_len) & (pos < TOPK)
                            tl.store(row_output + pos, -1, mask=take_pad)
                    elif seq_len <= HIST2048_THRESHOLD:
                        _histogram_2048_topk(
                            row_in,
                            row_output,
                            seq_len,
                            shared_ordered_ptr,
                            buffered_ptr,
                            decode_scalars_ptr,
                            TOPK,
                            VEC_SIZE,
                            BLOCK_SIZE,
                        )
                    else:
                        _histogram_256_topk(
                            row_in,
                            row_output,
                            seq_len,
                            shared_ordered_ptr,
                            buffered_ptr,
                            medium_scalars_ptr,
                            TOPK,
                            VEC_SIZE,
                            BLOCK_SIZE,
                        )
            else:
                my_chunk_start = cta_in_group * CHUNK_SIZE
                barrier_phase = _radix_topk(
                    row_in,
                    row_output,
                    seq_len,
                    my_chunk_start,
                    CHUNK_SIZE,
                    local_histogram_ptr,
                    suffix_sum_ptr,
                    radix_scalars_ptr,
                    shared_ordered_ptr,
                    g_histogram_ptr,
                    g_state_ptr,
                    cta_in_group,
                    ctas_per_group,
                    barrier_phase,
                    i,
                    TOPK,
                    VEC_SIZE,
                    BLOCK_SIZE,
                )
    return


# ════════════════════════════════════════════════════════════════════════════
# Per-Row CTA top-k path — per-row single-CTA kernel, dispatched when
# num_rows > 32. Ported from persistent_topk.py: 4-step radix
# (2048/2048/2048/1024 bins) with tle.cumsum streaming threshold search and
# _per_row_cta_final_radix_select. Uses inverted sortable-key convention
# compared to vLLM-style paths above.
# ════════════════════════════════════════════════════════════════════════════

PER_ROW_CTA_SIGN_BIT = tl.constexpr(-(1 << 31))


@triton.jit
def _per_row_cta_float_to_sortable(val):
    bits = val.to(tl.int32, bitcast=True)
    sign_ext = bits >> 31
    mask = sign_ext | tl.full(bits.shape, PER_ROW_CTA_SIGN_BIT, dtype=tl.int32)
    return bits ^ mask


@triton.jit
def _per_row_cta_convert_to_sortable_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    sign_set = (bits & sign_mask) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFFFFFF, tl.uint32)
    return tl.where(sign_set, bits, inv)


@triton.jit
def _per_row_cta_convert_to_fp16_hi11(x):
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
    sign_set = (bits & sign_mask) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
    mapped = tl.where(sign_set, bits, inv)
    return (mapped >> 5).to(tl.int32)


@triton.jit
def _per_row_cta_distribute_to_bins(
    x,
    in_range,
    ones,
    step_idx: tl.constexpr,
    logit_pattern,
    hist_base_ptr,
):
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_MASK: tl.constexpr = 0x3FF
    key = _per_row_cta_convert_to_sortable_uint32(x)
    if step_idx == 0:
        digit = _per_row_cta_convert_to_fp16_hi11(x)
    elif step_idx == 1:
        digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
    elif step_idx == 2:
        digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
    else:
        digit = (key & RADIX10_MASK).to(tl.int32)

    if step_idx < 2:
        partial = in_range
    elif step_idx == 2:
        partial = in_range & (((key ^ logit_pattern) >> 21) == 0)
    else:
        partial = in_range & (((key ^ logit_pattern) >> 10) == 0)

    tl.atomic_add(
        hist_base_ptr + digit,
        ones,
        mask=partial,
        sem="relaxed",
        scope="cta",
    )


@triton.jit
def _per_row_cta_process_bins(
    x,
    in_range,
    found_ptrs,
    ones,
    offs,
    final_cnt_ptrs,
    step_idx: tl.constexpr,
    logit_pattern,
    threshold_bin_idx,
    write_directly,
    s_out_indices_ptr,
    hist_base_ptr,
    use_final,
    TOPK: tl.constexpr = 0,
    s_final_vals_ptr=None,
    s_out_logits_ptr=None,
    row_start=0,
    split_indices_ptr=None,
    USE_MULTI_BLOCKS: tl.constexpr = False,
    IS_MERGE_BLOCKS: tl.constexpr = False,
):
    FINAL_SORT_ITEMS: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_MASK: tl.constexpr = 0x3FF

    key = _per_row_cta_convert_to_sortable_uint32(x)
    if step_idx == 0:
        digit = _per_row_cta_convert_to_fp16_hi11(x)
    elif step_idx == 1:
        digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
    elif step_idx == 2:
        digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
    else:
        digit = (key & RADIX10_MASK).to(tl.int32)

    if step_idx < 2:
        partial = in_range
    elif step_idx == 2:
        partial = in_range & (((key ^ logit_pattern) >> 21) == 0)
    else:
        partial = in_range & (((key ^ logit_pattern) >> 10) == 0)

    take_lt = partial & (digit < threshold_bin_idx) & write_directly
    out_pos_lt = tl.atomic_add(
        found_ptrs, ones, mask=take_lt, sem="relaxed", scope="cta"
    )
    if IS_MERGE_BLOCKS:
        split_idx = tl.load(
            split_indices_ptr + offs, mask=take_lt & (out_pos_lt < TOPK)
        )
        tl.store(
            s_out_indices_ptr + out_pos_lt,
            split_idx,
            mask=take_lt & (out_pos_lt < TOPK),
        )
    elif USE_MULTI_BLOCKS:
        tl.store(
            s_out_indices_ptr + out_pos_lt,
            (offs + row_start).to(tl.int32),
            mask=take_lt & (out_pos_lt < TOPK),
        )
        tl.store(s_out_logits_ptr + out_pos_lt, x, mask=take_lt & (out_pos_lt < TOPK))
    else:
        tl.store(
            s_out_indices_ptr + out_pos_lt,
            offs.to(tl.int32),
            mask=take_lt & (out_pos_lt < TOPK),
        )

    if step_idx == 3:
        take_eq = partial & (digit == threshold_bin_idx)
        out_pos_eq = tl.atomic_add(
            hist_base_ptr + digit, ones, mask=take_eq, sem="relaxed", scope="cta"
        )
        if IS_MERGE_BLOCKS:
            split_idx = tl.load(
                split_indices_ptr + offs, mask=take_eq & (out_pos_eq < TOPK)
            )
            tl.store(
                s_out_indices_ptr + out_pos_eq,
                split_idx,
                mask=take_eq & (out_pos_eq < TOPK),
            )
        elif USE_MULTI_BLOCKS:
            tl.store(
                s_out_indices_ptr + out_pos_eq,
                (offs + row_start).to(tl.int32),
                mask=take_eq & (out_pos_eq < TOPK),
            )
            tl.store(
                s_out_logits_ptr + out_pos_eq, x, mask=take_eq & (out_pos_eq < TOPK)
            )
        else:
            tl.store(
                s_out_indices_ptr + out_pos_eq,
                offs.to(tl.int32),
                mask=take_eq & (out_pos_eq < TOPK),
            )
    elif use_final:
        take_eq_final = partial & (digit == threshold_bin_idx)
        final_pos = tl.atomic_add(
            final_cnt_ptrs, ones, mask=take_eq_final, sem="relaxed", scope="cta"
        )
        if IS_MERGE_BLOCKS:
            split_idx = tl.load(
                split_indices_ptr + offs,
                mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
            )
            tl.store(
                hist_base_ptr + final_pos,
                split_idx,
                mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
            )
        elif USE_MULTI_BLOCKS:
            tl.store(
                hist_base_ptr + final_pos,
                (offs + row_start).to(tl.int32),
                mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
            )
        else:
            tl.store(
                hist_base_ptr + final_pos,
                offs.to(tl.int32),
                mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
            )
        tl.store(
            hist_base_ptr + (FINAL_SORT_ITEMS + final_pos),
            x.to(tl.int32, bitcast=True),
            mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
        )


@triton.jit
def _per_row_cta_process_histogram_step(
    row_ptr,
    stride_xn,
    row_start,
    row_end,
    seq_len,
    step_idx: tl.constexpr,
    logit_pattern,
    threshold_bin_idx,
    s_step_thresholds_ptr,
    found_topk_values,
    hist_base_ptr,
    s_out_indices_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    assume_aligned,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_TLE: tl.constexpr,
):
    VEC: tl.constexpr = 4
    FINAL_SORT_ITEMS: tl.constexpr = 2048
    RADIX11_SIZE: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_SIZE: tl.constexpr = 1024

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    ones_vec_2d = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_vec_2d = tl.zeros([BLOCK_SIZE, VEC], dtype=tl.int32)

    threshold_rounds: tl.constexpr = (
        RADIX10_SIZE // BLOCK_SIZE if step_idx == 3 else RADIX11_SIZE // BLOCK_SIZE
    )
    for clear_round in tl.static_range(0, threshold_rounds):
        clear_bins = clear_round * BLOCK_SIZE + lane
        tl.store(hist_base_ptr + clear_bins, 0)
    tl.debug_barrier()

    if step_idx == 2:
        logit_pattern = (threshold_bin_idx.to(tl.uint32) & RADIX11_MASK) << 21
    elif step_idx == 3:
        logit_pattern |= (threshold_bin_idx.to(tl.uint32) & RADIX11_MASK) << 10

    n_tiles = tl.cdiv(seq_len, BLOCK_SIZE)
    n_vec_full = seq_len // (BLOCK_SIZE * VEC)
    rem_tiles = (seq_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE

    if assume_aligned:
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(row_ptr + offs)
            _per_row_cta_distribute_to_bins(
                x_vec,
                True,
                ones_vec_2d,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(row_ptr + offs)
            _per_row_cta_distribute_to_bins(
                x,
                True,
                ones,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
    elif stride_xn == 1:
        aligned_row_start = (row_start + VEC - 1) // VEC * VEC
        skip_elems = aligned_row_start - row_start
        row_len = row_end - aligned_row_start
        n_vec_full = row_len // (BLOCK_SIZE * VEC)
        rem_tiles = (row_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = row_len % BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(row_ptr + aligned_row_start + offs)
            _per_row_cta_distribute_to_bins(
                x_vec,
                True,
                ones_vec_2d,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(row_ptr + aligned_row_start + offs)
            _per_row_cta_distribute_to_bins(
                x,
                True,
                ones,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
        if skip_elems > 0:
            offs = lane
            in_range = lane < skip_elems
            x = tl.load(row_ptr + row_start + offs, mask=in_range, other=float("-inf"))
            _per_row_cta_distribute_to_bins(
                x,
                in_range,
                ones,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(
                row_ptr + aligned_row_start + offs, mask=in_range, other=float("-inf")
            )
            _per_row_cta_distribute_to_bins(
                x,
                in_range,
                ones,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
    else:
        row_len = row_end - row_start
        n_tiles = tl.cdiv(row_len, BLOCK_SIZE)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = offs < row_len
            x = tl.load(
                row_ptr + row_start + offs * stride_xn,
                mask=in_range,
                other=float("-inf"),
            )
            _per_row_cta_distribute_to_bins(
                x,
                in_range,
                ones,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
    last_value = tl.load(s_found_topk_values_ptr)
    tl.debug_barrier()

    threshold_bin_ptrs = s_threshold_bin_idx_ptr + zeros
    final_bin_size_ptrs = s_final_bin_size_ptr + zeros
    threshold_found = False
    for round_idx in tl.static_range(0, threshold_rounds):
        if not threshold_found:
            bins = round_idx * BLOCK_SIZE + lane
            counts = tl.load(hist_base_ptr + bins)
            if HAS_TLE:
                prefix_sum, counts_total = tle.cumsum(counts, axis=0, reverse=False)
            else:
                counts_total = tl.sum(counts)
                prefix_sum = counts_total - tl.cumsum(counts, axis=0, reverse=True)
            prefix_sum = prefix_sum + last_value
            total_sum = last_value + counts_total
            next_prefix_sum = prefix_sum + counts
            threshold_mask = (prefix_sum < TOPK) & (next_prefix_sum >= TOPK)
            threshold_bin = bins
            threshold_bin_size = next_prefix_sum - prefix_sum
            tl.store(hist_base_ptr + bins, prefix_sum)
            tl.store(threshold_bin_ptrs, threshold_bin, mask=threshold_mask)
            tl.store(final_bin_size_ptrs, threshold_bin_size, mask=threshold_mask)
            found_round = tl.reduce_or(threshold_mask, axis=0)
            threshold_found = found_round
            last_value = total_sum

    threshold_bin_idx = tl.load(s_threshold_bin_idx_ptr)
    final_bin_size = tl.load(s_final_bin_size_ptr)

    use_final = final_bin_size <= FINAL_SORT_ITEMS
    write_directly = ((step_idx == 0) & (final_bin_size <= FINAL_SORT_ITEMS)) | (
        step_idx >= 1
    )

    found_ptrs = s_found_topk_values_ptr + zeros
    final_cnt_ptrs = s_final_cnt_ptr + zeros
    if assume_aligned:
        found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(row_ptr + offs)
            _per_row_cta_process_bins(
                x_vec,
                True,
                found_ptrs_vec_2d,
                ones_vec_2d,
                offs,
                final_cnt_ptrs_vec_2d,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(row_ptr + offs)
            _per_row_cta_process_bins(
                x,
                True,
                found_ptrs,
                ones,
                offs,
                final_cnt_ptrs,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
    elif stride_xn == 1:
        aligned_row_start = (row_start + VEC - 1) // VEC * VEC
        skip_elems = aligned_row_start - row_start
        row_len = row_end - aligned_row_start
        n_vec_full = row_len // (BLOCK_SIZE * VEC)
        rem_tiles = (row_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = row_len % BLOCK_SIZE
        found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(row_ptr + aligned_row_start + offs)
            _per_row_cta_process_bins(
                x_vec,
                True,
                found_ptrs_vec_2d,
                ones_vec_2d,
                offs + skip_elems,
                final_cnt_ptrs_vec_2d,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(row_ptr + aligned_row_start + offs)
            _per_row_cta_process_bins(
                x,
                True,
                found_ptrs,
                ones,
                offs + skip_elems,
                final_cnt_ptrs,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
        if skip_elems > 0:
            offs = lane
            in_range = lane < skip_elems
            x = tl.load(row_ptr + row_start + offs, mask=in_range, other=float("-inf"))
            _per_row_cta_process_bins(
                x,
                in_range,
                found_ptrs,
                ones,
                offs,
                final_cnt_ptrs,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(
                row_ptr + aligned_row_start + offs, mask=in_range, other=float("-inf")
            )
            _per_row_cta_process_bins(
                x,
                in_range,
                found_ptrs,
                ones,
                offs + skip_elems,
                final_cnt_ptrs,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
    else:
        row_len = row_end - row_start
        n_tiles = tl.cdiv(row_len, BLOCK_SIZE)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = offs < row_len
            x = tl.load(
                row_ptr + row_start + offs * stride_xn,
                mask=in_range,
                other=float("-inf"),
            )
            _per_row_cta_process_bins(
                x,
                in_range,
                found_ptrs,
                ones,
                offs,
                final_cnt_ptrs,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
    tl.debug_barrier()
    return (
        final_bin_size > FINAL_SORT_ITEMS,
        logit_pattern.to(tl.int32),
        threshold_bin_idx,
    )


@triton.jit
def _per_row_cta_final_radix_select(
    hist_base_ptr,
    s_out_indices_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    s_radix_count_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FINAL_SORT_ITEMS: tl.constexpr,
    HAS_TLE: tl.constexpr,
):
    RADIX_BITS_FINAL: tl.constexpr = 8
    RADIX_SIZE_FINAL: tl.constexpr = 1 << RADIX_BITS_FINAL
    RADIX_MASK_FINAL: tl.constexpr = RADIX_SIZE_FINAL - 1
    DIGIT_START: tl.constexpr = 32 - RADIX_BITS_FINAL

    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    bins = tl.arange(0, RADIX_SIZE_FINAL)

    radix_count_vec_ptr = s_radix_count_ptr + bins
    base_idx = tl.load(s_found_topk_values_ptr)
    final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), FINAL_SORT_ITEMS)
    remain = tl.minimum(TOPK - base_idx, final_cnt)
    tl.debug_barrier()

    if remain > 0:
        desired = tl.zeros((), dtype=tl.uint32)
        desired_mask = tl.zeros((), dtype=tl.uint32)
        k_to_find = remain + 1

        for digit_pos in tl.static_range(DIGIT_START, -1, -RADIX_BITS_FINAL):
            tl.store(s_radix_count_ptr + lane, 0, mask=lane < RADIX_SIZE_FINAL)
            tl.debug_barrier()

            cnt_tiles = tl.cdiv(final_cnt, BLOCK_SIZE)
            for t in tl.range(0, cnt_tiles):
                pos = t * BLOCK_SIZE + lane
                valid = pos < final_cnt
                x_bits_i32 = tl.load(
                    hist_base_ptr + (FINAL_SORT_ITEMS + pos),
                    mask=valid,
                    other=0,
                )
                x = x_bits_i32.to(tl.float32, bitcast=True)
                key = _per_row_cta_convert_to_sortable_uint32(x)
                matches = (key & desired_mask) == desired
                digit = ((key >> digit_pos) & RADIX_MASK_FINAL).to(tl.int32)
                take = valid & matches
                tl.atomic_add(
                    s_radix_count_ptr + digit,
                    ones,
                    mask=take,
                    sem="relaxed",
                    scope="cta",
                )

            tl.debug_barrier()
            counts = tl.load(radix_count_vec_ptr)
            if HAS_TLE:
                prefix_sum, _ = tle.cumsum(counts, axis=0, reverse=False)
            else:
                prefix_sum = tl.sum(counts) - tl.cumsum(counts, axis=0, reverse=True)
            next_prefix_sum = prefix_sum + counts
            threshold_mask = (prefix_sum < k_to_find) & (next_prefix_sum >= k_to_find)
            threshold_init = tl.full((), RADIX_SIZE_FINAL, dtype=tl.int32)
            threshold_bin = tl.min(
                tl.where(threshold_mask, bins, threshold_init), axis=0
            ).to(tl.int32)
            threshold_bin = tl.where(
                threshold_bin == RADIX_SIZE_FINAL, RADIX_SIZE_FINAL - 1, threshold_bin
            )
            counts_lt = tl.max(
                tl.where(bins == threshold_bin, prefix_sum, 0), axis=0
            ).to(tl.int32)

            desired = desired | (threshold_bin.to(tl.uint32) << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX_MASK_FINAL, dtype=tl.uint32) << digit_pos
            )
            k_to_find = k_to_find - counts_lt

        thr_key = desired
        found_ptrs = s_found_topk_values_ptr + zeros
        cnt_tiles = tl.cdiv(final_cnt, BLOCK_SIZE)
        for t in tl.range(0, cnt_tiles):
            pos = t * BLOCK_SIZE + lane
            valid = pos < final_cnt
            idx = tl.load(hist_base_ptr + pos, mask=valid, other=0)
            x_bits_i32 = tl.load(
                hist_base_ptr + (FINAL_SORT_ITEMS + pos),
                mask=valid,
                other=0,
            )
            x = x_bits_i32.to(tl.float32, bitcast=True)
            key = _per_row_cta_convert_to_sortable_uint32(x)
            take_lt = valid & (key < thr_key)
            out_pos_gt = tl.atomic_add(
                found_ptrs,
                ones,
                mask=take_lt,
                sem="relaxed",
                scope="cta",
            )
            tl.store(
                s_out_indices_ptr + out_pos_gt,
                idx,
                mask=take_lt & (out_pos_gt < TOPK),
            )

        tl.debug_barrier()
        cur = tl.load(s_found_topk_values_ptr)
        if cur < TOPK:
            for t in tl.range(0, cnt_tiles):
                cur = tl.load(s_found_topk_values_ptr)
                if cur < TOPK:
                    pos = t * BLOCK_SIZE + lane
                    valid = pos < final_cnt
                    idx = tl.load(hist_base_ptr + pos, mask=valid, other=0)
                    x_bits_i32 = tl.load(
                        hist_base_ptr + (FINAL_SORT_ITEMS + pos),
                        mask=valid,
                        other=0,
                    )
                    x = x_bits_i32.to(tl.float32, bitcast=True)
                    key = _per_row_cta_convert_to_sortable_uint32(x)
                    take_eq = valid & (key == thr_key)
                    out_pos_eq = tl.atomic_add(
                        found_ptrs,
                        ones,
                        mask=take_eq,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        s_out_indices_ptr + out_pos_eq,
                        idx,
                        mask=take_eq & (out_pos_eq < TOPK),
                    )

    tl.debug_barrier()
    tl.store(s_found_topk_values_ptr, TOPK)


@triton.jit
def _per_row_cta_topk_selector(
    row_ptr,
    out_row,
    row_start,
    row_end,
    stride_xn,
    vocab_size,
    hist_base_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    s_step_thresholds_ptr,
    s_out_indices_ptr,
    s_radix_count_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
    HAS_TLE: tl.constexpr,
):
    FINAL_SORT_ITEMS: tl.constexpr = 2048

    assume_aligned = (
        (row_start == 0)
        & (row_end == vocab_size)
        & (stride_xn == 1)
        & ((vocab_size % BLOCK_SIZE) == 0)
    )
    if assume_aligned:
        tl.assume(row_start == 0)
        tl.assume(row_end == vocab_size)
        tl.assume(stride_xn == 1)
        vocab_size = tl.multiple_of(vocab_size, BLOCK_SIZE)
    elif stride_xn == 1:
        tl.assume(stride_xn == 1)

    lane = tl.arange(0, BLOCK_SIZE)
    row_len = row_end - row_start
    if row_len <= TOPK:
        chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
        for chunk_idx in tl.range(0, chunks):
            pos = chunk_idx * BLOCK_SIZE + lane
            take_row = pos < row_len
            tl.store(out_row + pos, (row_start + pos).to(tl.int32), mask=take_row)
            take_pad = (pos >= row_len) & (pos < TOPK)
            tl.store(out_row + pos, -1, mask=take_pad)
        return

    tl.store(s_final_cnt_ptr, 0)
    tl.store(s_found_topk_values_ptr, 0)

    logit_pattern = tl.zeros((), dtype=tl.uint32)
    continue_to_next_step = True
    logit_pattern = 0
    threshold_bin_idx = -1

    tl.debug_barrier()
    for step_idx in tl.static_range(0, 4):
        if continue_to_next_step:
            continue_to_next_step, logit_pattern, threshold_bin_idx = (
                _per_row_cta_process_histogram_step(
                    row_ptr,
                    stride_xn,
                    row_start,
                    row_end,
                    vocab_size,
                    step_idx,
                    logit_pattern,
                    threshold_bin_idx,
                    s_step_thresholds_ptr,
                    0,
                    hist_base_ptr,
                    s_out_indices_ptr,
                    s_final_cnt_ptr,
                    s_found_topk_values_ptr,
                    s_threshold_bin_idx_ptr,
                    s_final_bin_size_ptr,
                    assume_aligned=assume_aligned,
                    TOPK=TOPK,
                    BLOCK_SIZE=BLOCK_SIZE,
                    HAS_TLE=HAS_TLE,
                )
            )

    if not continue_to_next_step:
        if USE_RADIX_FINAL:
            _per_row_cta_final_radix_select(
                hist_base_ptr,
                s_out_indices_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                s_radix_count_ptr,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
                FINAL_SORT_ITEMS=FINAL_SORT_ITEMS,
                HAS_TLE=HAS_TLE,
            )
        else:
            base_idx = tl.load(s_found_topk_values_ptr)
            final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), FINAL_SORT_ITEMS)
            sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)
            for sort_chunk in tl.range(0, sort_chunks):
                pos = sort_chunk * BLOCK_SIZE + lane
                valid = pos < final_cnt
                logit_i_bits = tl.load(
                    hist_base_ptr + FINAL_SORT_ITEMS + pos,
                    mask=valid,
                    other=0,
                )
                logit_i = logit_i_bits.to(tl.float32, bitcast=True)
                out_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
                for j in tl.range(0, final_cnt):
                    logit_j_bits = tl.load(hist_base_ptr + FINAL_SORT_ITEMS + j)
                    logit_j = logit_j_bits.to(tl.float32, bitcast=True)
                    better = (logit_i < logit_j) | ((logit_i == logit_j) & (pos < j))
                    out_rank = out_rank + (valid & better).to(tl.int32)
                dst_pos = base_idx + out_rank
                take = valid & (dst_pos < TOPK)
                idx_i = tl.load(hist_base_ptr + pos, mask=take, other=0)
                tl.store(s_out_indices_ptr + dst_pos, idx_i, mask=take)
            tl.debug_barrier()
            tl.store(s_found_topk_values_ptr, TOPK)

    flush_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    for flush_chunk in tl.static_range(flush_chunks):
        pos = flush_chunk * BLOCK_SIZE + lane
        mask = pos < TOPK
        out_vals = tl.load(s_out_indices_ptr + pos, mask=mask, other=-1)
        tl.store(out_row + pos, out_vals, mask=mask)


@triton.jit
def _per_row_cta_topk_wrapper(
    x_ptr,
    out_ptr,
    seq_lens_ptr,
    next_n,
    stride_xm,
    stride_xn,
    vocab_size,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
):
    HIST_SIZE: tl.constexpr = 4096
    RADIX_BITS_FINAL: tl.constexpr = 8
    RADIX_SIZE_FINAL: tl.constexpr = 1 << RADIX_BITS_FINAL

    pid = tl.program_id(0)
    batch_id = pid // next_n
    batch_offset = pid % next_n
    seq_len = tl.load(seq_lens_ptr + batch_id)
    row_start = 0
    row_len = seq_len - next_n + batch_offset + 1
    row_end = row_len

    x_ptr += pid * stride_xm
    out_ptr += pid * TOPK

    s_histogram = tle.gpu.alloc(
        [HIST_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_out_indices = tle.gpu.alloc(
        [TOPKP],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_cnt = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_threshold_bin_idx = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_bin_size = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_found_topk_values = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_step_thresholds = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hist_base_ptr = tle.gpu.local_ptr(s_histogram, (0,))
    s_final_cnt_ptr = tle.gpu.local_ptr(s_final_cnt, (0,))
    s_threshold_bin_idx_ptr = tle.gpu.local_ptr(s_threshold_bin_idx, (0,))
    s_final_bin_size_ptr = tle.gpu.local_ptr(s_final_bin_size, (0,))
    s_found_topk_values_ptr = tle.gpu.local_ptr(s_found_topk_values, (0,))
    s_step_thresholds_ptr = tle.gpu.local_ptr(s_step_thresholds, (0,))
    s_out_indices_ptr = tle.gpu.local_ptr(s_out_indices, (0,))
    if USE_RADIX_FINAL:
        s_radix_counts = tle.gpu.alloc(
            [RADIX_SIZE_FINAL],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        s_radix_count_ptr = tle.gpu.local_ptr(s_radix_counts, (0,))
    else:
        s_radix_count_ptr = None

    _per_row_cta_topk_selector(
        x_ptr,
        out_ptr,
        row_start,
        row_end,
        stride_xn,
        vocab_size,
        hist_base_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        s_step_thresholds_ptr,
        s_out_indices_ptr,
        s_radix_count_ptr,
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=USE_RADIX_FINAL,
        HAS_TLE=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# mbtopk top-k path — torch-mbtopk-style multi-launch radix select for
# num_rows > 32 (large rows); the hygon counterpart of the generic per-row-CTA path.
# Replicates torch's mbtopk structure: each row split across blocks_per_row
# CTAs, 4 rounds of 8-bit radix select, cross-CTA coordination via launch
# boundaries (no in-kernel barrier). Higher intra-row parallelism than the
# single-CTA per-row-CTA path. Key convention (from _per_row_cta_convert_to_sortable_uint32):
#   larger value -> smaller key  =>  top-k largest = k smallest keys.
# Workspace (reused from the persistent region; the paths are mutually
# exclusive by num_rows):
#   g_hist      [num_rows, 256] u32   per-row round histogram (recycled)
#   g_state     [num_rows, 5]   u32   desired, mask, k_to_find,
#                                     lt_counter, eq_counter
# ════════════════════════════════════════════════════════════════════════════


@triton.jit
def _mbtopk_init(
    out_ptr,
    seq_lens_ptr,
    num_rows,
    g_state_ptr,
    g_hist_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_rows:
        return
    lane = tl.arange(0, BLOCK_SIZE)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.uint32)
    row = pid
    seq_len = tl.load(seq_lens_ptr + row)
    # g_state: desired=0, mask=0, k_to_find=min(TOPK, seq_len),
    #          lt_counter=0, eq_counter=0
    k_init = tl.minimum(TOPK, seq_len)
    tl.store(g_state_ptr + row * 5 + 0 + zeros, 0, mask=lane == 0)
    tl.store(g_state_ptr + row * 5 + 1 + zeros, 0, mask=lane == 0)
    tl.store(g_state_ptr + row * 5 + 2 + zeros, k_init, mask=lane == 0)
    tl.store(g_state_ptr + row * 5 + 3 + zeros, 0, mask=lane == 0)
    tl.store(g_state_ptr + row * 5 + 4 + zeros, 0, mask=lane == 0)
    # clear the round histogram (256 bins, BLOCK_SIZE >= 256 lanes)
    tl.store(g_hist_ptr + row * 256 + lane, 0, mask=lane < 256)
    # pre-fill output with -1 (rows with seq_len < TOPK keep the tail -1)
    num_pad_tiles: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    for t in tl.static_range(0, num_pad_tiles):
        pos = t * BLOCK_SIZE + lane
        pad = (pos >= seq_len) & (pos < TOPK)
        tl.store(out_ptr + row * TOPK + pos, -1, mask=pad)


@triton.jit
def _mbtopk_accumulate(
    row_ptr,
    seq_lens_ptr,
    num_rows,
    stride,
    g_state_ptr,
    g_hist_ptr,
    seg_len,
    blocks_per_row,
    bit_offset: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // blocks_per_row
    blk = pid % blocks_per_row
    if row >= num_rows:
        return
    seq_len = tl.load(seq_lens_ptr + row)
    seg_start = blk * seg_len
    if seg_start >= seq_len:
        return
    seg_end = tl.minimum(seg_start + seg_len, seq_len)
    row_ptr += row * stride

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC_SIZE)
    bins = tl.arange(0, 256)
    local_hist = tle.gpu.alloc(
        [256],
        dtype=tl.uint32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    local_hist_ptr = tle.gpu.local_ptr(local_hist, (0,))
    tl.store(local_hist_ptr + lane, 0, mask=lane < 256)
    tl.debug_barrier()

    desired = tl.load(g_state_ptr + row * 5 + 0)
    mask = tl.load(g_state_ptr + row * 5 + 1)

    n_vec_full = (seg_end - seg_start) // (BLOCK_SIZE * VEC_SIZE)
    rem_tiles = (
        (seg_end - seg_start) - n_vec_full * BLOCK_SIZE * VEC_SIZE
    ) // BLOCK_SIZE
    rem_elems = (seg_end - seg_start) % BLOCK_SIZE

    for t in tl.range(0, n_vec_full):
        base = seg_start + t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_ptr + offs)
        key = _per_row_cta_convert_to_sortable_uint32(x)
        match = (key & mask) == desired
        digit = ((key >> bit_offset) & 0xFF).to(tl.int32)
        if bit_offset == 24:
            # round 0: all elements match; one tl.histogram per tile is
            # faster than per-element atomics into smem.
            hist = tl.histogram(digit.reshape(BLOCK_SIZE * VEC_SIZE), 256)
            tl.atomic_add(local_hist_ptr + bins, hist, sem="relaxed", scope="cta")
        else:
            # narrowing rounds: most lanes are masked out, per-element
            # atomics (predicated off) are cheaper than a full histogram.
            tl.atomic_add(
                local_hist_ptr + digit, 1, mask=match, sem="relaxed", scope="cta"
            )
    for t in tl.range(0, rem_tiles):
        offs = seg_start + n_vec_full * BLOCK_SIZE * VEC_SIZE + t * BLOCK_SIZE + lane
        x = tl.load(row_ptr + offs)
        key = _per_row_cta_convert_to_sortable_uint32(x)
        match = (key & mask) == desired
        digit = ((key >> bit_offset) & 0xFF).to(tl.int32)
        tl.atomic_add(local_hist_ptr + digit, 1, mask=match, sem="relaxed", scope="cta")
    if rem_elems > 0:
        offs = (
            seg_start
            + n_vec_full * BLOCK_SIZE * VEC_SIZE
            + rem_tiles * BLOCK_SIZE
            + lane
        )
        in_range = (lane < rem_elems) & (offs < seq_len)
        x = tl.load(row_ptr + offs, mask=in_range, other=float("-inf"))
        key = _per_row_cta_convert_to_sortable_uint32(x)
        match = (key & mask) == desired
        digit = ((key >> bit_offset) & 0xFF).to(tl.int32)
        tl.atomic_add(
            local_hist_ptr + digit, 1, mask=match & in_range, sem="relaxed", scope="cta"
        )
    tl.debug_barrier()
    counts = tl.load(local_hist_ptr + lane, mask=lane < 256, other=0)
    tl.atomic_add(
        g_hist_ptr + row * 256 + lane,
        counts,
        mask=lane < 256,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def _mbtopk_threshold(
    seq_lens_ptr,
    num_rows,
    g_state_ptr,
    g_hist_ptr,
    bit_offset: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_rows:
        return
    row = pid
    lane = tl.arange(0, BLOCK_SIZE)
    seq_len = tl.load(seq_lens_ptr + row)
    k_to_find = tl.load(g_state_ptr + row * 5 + 2)
    counts = tl.load(g_hist_ptr + row * 256 + lane, mask=lane < 256, other=0)
    prefix = tl.cumsum(counts, axis=0)
    ge_mask = (prefix >= k_to_find) & (lane < 256)
    thr_bin = tl.minimum(tl.min(tl.where(ge_mask, lane, 256), axis=0), 255).to(tl.int32)
    count_below = tl.sum(tl.where(lane < thr_bin, counts, 0), axis=0)
    # update desired / mask / k_to_find
    desired = tl.load(g_state_ptr + row * 5 + 0)
    mask = tl.load(g_state_ptr + row * 5 + 1)
    bit_mask: tl.constexpr = 0xFF
    desired_new = desired | (thr_bin.to(tl.uint32) << bit_offset)
    mask_new = mask | (tl.full((), bit_mask, tl.uint32) << bit_offset)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.uint32)
    tl.store(g_state_ptr + row * 5 + 0 + zeros, desired_new, mask=lane == 0)
    tl.store(g_state_ptr + row * 5 + 1 + zeros, mask_new, mask=lane == 0)
    k_new = k_to_find - count_below
    tl.store(
        g_state_ptr + row * 5 + 2 + zeros,
        k_new,
        mask=lane == 0,
    )
    if bit_offset == 0:
        # last round: total_lt = count of key < thr_key = min(seq_len, TOPK)
        # - k_new. eq counter starts there; short rows (seq_len<TOPK) have
        # the [seq_len, TOPK) tail pre-filled with -1 by init.
        tl.store(
            g_state_ptr + row * 5 + 4 + zeros,
            tl.minimum(TOPK, seq_len) - k_new,
            mask=lane == 0,
        )
    # recycle histogram for the next round
    tl.store(g_hist_ptr + row * 256 + lane, 0, mask=lane < 256)


@triton.jit
def _mbtopk_gather(
    row_ptr,
    out_ptr,
    seq_lens_ptr,
    num_rows,
    stride,
    g_state_ptr,
    seg_len,
    blocks_per_row,
    TOPK: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // blocks_per_row
    blk = pid % blocks_per_row
    if row >= num_rows:
        return
    seq_len = tl.load(seq_lens_ptr + row)
    seg_start = blk * seg_len
    if seg_start >= seq_len:
        return
    seg_end = tl.minimum(seg_start + seg_len, seq_len)
    thr_key = tl.load(g_state_ptr + row * 5 + 0)
    # lt_counter (slot 3) starts at 0; eq_counter (slot 4) was set to
    # total_lt = min(seq_len, TOPK) - k_new by the last threshold round.
    lt_counter_ptr = g_state_ptr + row * 5 + 3
    eq_counter_ptr = g_state_ptr + row * 5 + 4
    row_ptr += row * stride

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, dtype=tl.int32)
    ones_2d = tl.full([BLOCK_SIZE, VEC_SIZE], 1, dtype=tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_2d = tl.zeros([BLOCK_SIZE, VEC_SIZE], dtype=tl.int32)

    n_vec_full = (seg_end - seg_start) // (BLOCK_SIZE * VEC_SIZE)
    rem_tiles = (
        (seg_end - seg_start) - n_vec_full * BLOCK_SIZE * VEC_SIZE
    ) // BLOCK_SIZE
    rem_elems = (seg_end - seg_start) % BLOCK_SIZE

    for t in tl.range(0, n_vec_full):
        base = seg_start + t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_ptr + offs)
        key = _per_row_cta_convert_to_sortable_uint32(x)
        idx = offs.to(tl.int32)
        lt = key < thr_key
        lt_pos = tl.atomic_add(
            lt_counter_ptr + zeros_2d, ones_2d, mask=lt, sem="relaxed", scope="gpu"
        )
        tl.store(out_ptr + row * TOPK + lt_pos, idx, mask=lt & (lt_pos < TOPK))
        eq = key == thr_key
        eq_pos = tl.atomic_add(
            eq_counter_ptr + zeros_2d, ones_2d, mask=eq, sem="relaxed", scope="gpu"
        )
        tl.store(out_ptr + row * TOPK + eq_pos, idx, mask=eq & (eq_pos < TOPK))
    for t in tl.range(0, rem_tiles):
        offs = seg_start + n_vec_full * BLOCK_SIZE * VEC_SIZE + t * BLOCK_SIZE + lane
        x = tl.load(row_ptr + offs)
        key = _per_row_cta_convert_to_sortable_uint32(x)
        idx = offs.to(tl.int32)
        lt = key < thr_key
        lt_pos = tl.atomic_add(
            lt_counter_ptr + zeros, ones, mask=lt, sem="relaxed", scope="gpu"
        )
        tl.store(out_ptr + row * TOPK + lt_pos, idx, mask=lt & (lt_pos < TOPK))
        eq = key == thr_key
        eq_pos = tl.atomic_add(
            eq_counter_ptr + zeros, ones, mask=eq, sem="relaxed", scope="gpu"
        )
        tl.store(out_ptr + row * TOPK + eq_pos, idx, mask=eq & (eq_pos < TOPK))
    if rem_elems > 0:
        offs = (
            seg_start
            + n_vec_full * BLOCK_SIZE * VEC_SIZE
            + rem_tiles * BLOCK_SIZE
            + lane
        )
        in_range = (lane < rem_elems) & (offs < seq_len)
        x = tl.load(row_ptr + offs, mask=in_range, other=float("-inf"))
        key = _per_row_cta_convert_to_sortable_uint32(x)
        idx = offs.to(tl.int32)
        lt = (key < thr_key) & in_range
        lt_pos = tl.atomic_add(
            lt_counter_ptr + zeros, ones, mask=lt, sem="relaxed", scope="gpu"
        )
        tl.store(out_ptr + row * TOPK + lt_pos, idx, mask=lt & (lt_pos < TOPK))
        eq = (key == thr_key) & in_range
        eq_pos = tl.atomic_add(
            eq_counter_ptr + zeros, ones, mask=eq, sem="relaxed", scope="gpu"
        )
        tl.store(out_ptr + row * TOPK + eq_pos, idx, mask=eq & (eq_pos < TOPK))


# Threshold: route to the mbtopk path only for large rows. Small rows
# (max_seq_len <= threshold) keep the single-CTA per-row-CTA path.
MBTOPK_THRESHOLD = 8192
# per-thread elements per block; larger -> fewer CTAs/row (more row
# parallelism), measured optimum for the 262144-vocab radix shapes is 32
# (seg_len=16384, bpr=16).
MBTOPK_ITEMS_PER_THREAD = 32


def _mbtopk_topk(
    logits: torch.Tensor,
    output: torch.Tensor,
    seq_lens: torch.Tensor,
    num_rows: int,
    stride: int,
    workspace: torch.Tensor,
    k: int,
) -> None:
    """mbtopk-style multi-launch top-k path (num_rows > 32, large rows).

    Each row is split into blocks_per_row CTAs; 4 rounds of 8-bit radix
    select coordinate via launch boundaries (no in-kernel barrier). Workspace
    is reused from the persistent region (paths are mutually exclusive by
    num_rows). Layout:
        g_hist    [num_rows, 256] u32
        g_state   [num_rows, 5]   u32  (desired, mask, k_to_find,
                                        lt_counter, eq_counter)
    """
    vec_size = 1
    if stride % 4 == 0:
        vec_size = 4
    elif stride % 2 == 0:
        vec_size = 2
    seg_target = 512 * MBTOPK_ITEMS_PER_THREAD
    bpr = max(1, (stride + seg_target - 1) // seg_target)
    seg_len = (stride + bpr - 1) // bpr
    seg_len = ((seg_len + vec_size - 1) // vec_size) * vec_size
    bpr = max(1, (stride + seg_len - 1) // seg_len)
    bpr = min(bpr, 256)  # prefix kernel scans up to blocks_per_row lanes
    seg_len = (stride + bpr - 1) // bpr
    seg_len = ((seg_len + vec_size - 1) // vec_size) * vec_size

    g_hist_bytes = num_rows * 256 * 4
    g_state_bytes = num_rows * 5 * 4
    need = g_hist_bytes + g_state_bytes
    assert workspace.size(0) >= need, (
        f"persistent_topk(_mbtopk): workspace needs >= {need} bytes, got "
        f"{workspace.size(0)}"
    )
    g_hist = workspace[:g_hist_bytes].view(torch.uint32).view(num_rows, 256)
    g_state = (
        workspace[g_hist_bytes : g_hist_bytes + g_state_bytes]
        .view(torch.uint32)
        .view(num_rows, 5)
    )

    grid = num_rows * bpr
    _mbtopk_init[(num_rows,)](
        output,
        seq_lens,
        num_rows,
        g_state,
        g_hist,
        TOPK=k,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    for bit_offset in (24, 16, 8, 0):
        _mbtopk_accumulate[(grid,)](
            logits,
            seq_lens,
            num_rows,
            stride,
            g_state,
            g_hist,
            seg_len,
            bpr,
            bit_offset=bit_offset,
            VEC_SIZE=vec_size,
            BLOCK_SIZE=512,
            num_warps=8,
        )
        _mbtopk_threshold[(num_rows,)](
            seq_lens,
            num_rows,
            g_state,
            g_hist,
            bit_offset=bit_offset,
            TOPK=k,
            BLOCK_SIZE=256,
            num_warps=4,
        )
    _mbtopk_gather[(grid,)](
        logits,
        output,
        seq_lens,
        num_rows,
        stride,
        g_state,
        seg_len,
        bpr,
        TOPK=k,
        VEC_SIZE=vec_size,
        BLOCK_SIZE=512,
        num_warps=8,
    )


def persistent_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    k: int = 512,
    max_seq_len: int | None = None,
) -> None:
    """vLLM-compatible persistent topk decode.

    Args:
        logits:  [num_rows, stride] float32.
        lengths: [num_rows] int32, or [B, next_n] int32 for MTP.
        output:  [num_rows, k] int32 — pre-allocated output buffer.
        workspace: uint8 buffer (required). Used for internal
                   scratch if provided. Enables CUDAGraph compatibility
                   by avoiding internal torch.zeros allocation.
        k:       number of top elements to select. Must be 512/1024/2048.
        max_seq_len: global max seq_len across all rows.
    """
    assert logits.is_cuda, "persistent_topk: logits must be CUDA tensor"
    assert lengths.is_cuda, "persistent_topk: lengths must be CUDA tensor"
    assert output.is_cuda, "persistent_topk: output must be CUDA tensor"
    assert logits.dtype == torch.float32, "persistent_topk: only float32 supported"
    assert lengths.dtype == torch.int32, "persistent_topk: lengths must be int32"
    assert output.dtype == torch.int32, "persistent_topk: output must be int32"
    assert logits.dim() == 2, "persistent_topk: logits must be 2D"
    assert lengths.dim() in (1, 2), "persistent_topk: lengths must be 1D or 2D"
    assert lengths.is_contiguous(), "persistent_topk: lengths must be contiguous"
    assert output.dim() == 2, "persistent_topk: output must be 2D"

    num_rows = logits.size(0)
    stride = logits.stride(0)
    seq_lens = lengths.reshape(-1) if lengths.dim() == 2 else lengths
    assert (
        seq_lens.numel() == num_rows
    ), f"persistent_topk: lengths size mismatch: {seq_lens.numel()} vs {num_rows}"
    assert (
        output.size(0) == num_rows and output.size(1) == k
    ), f"persistent_topk: output size mismatch: ({output.size(0)}, {output.size(1)}) vs ({num_rows}, {k})"
    assert k in (
        512,
        1024,
        2048,
    ), f"persistent_topk supports k=512, k=1024, or k=2048, got k={k}"
    actual_max = max_seq_len if max_seq_len is not None else logits.shape[1]
    max_seq_len = min(logits.shape[1], actual_max)

    if num_rows > 32:
        if max_seq_len > MBTOPK_THRESHOLD:
            _mbtopk_topk(logits, output, seq_lens, num_rows, stride, workspace, k)
        else:
            _per_row_cta_topk_wrapper[(num_rows,)](
                logits,
                output,
                seq_lens,
                1,  # next_n
                stride,  # stride_xm
                1,  # stride_xn
                stride,  # vocab_size
                TOPK=k,
                TOPKP=max(k, 2048),
                BLOCK_SIZE=512,
                USE_RADIX_FINAL=True,
                num_warps=8,
            )
        return

    device = logits.device
    device_props = torch.cuda.get_device_properties(device.index)
    num_sms = device_props.multi_processor_count
    # v0.3: radix re-reads from global (no smem chunk cache), so CHUNK_SIZE is
    # no longer smem-bound. It now only partitions one row among CTAs: larger
    # chunk -> fewer CTAs per group -> more groups -> more rows in parallel.
    # Keep intra-row parallelism >= 8 CTAs/row; the rest goes to row-parallelism.
    vec_size = 1
    if stride % 4 == 0:
        vec_size = 4
    elif stride % 2 == 0:
        vec_size = 2
    target_ctas_per_group = 8
    max_chunk_elements = triton.next_power_of_2(
        (stride + target_ctas_per_group - 1) // target_ctas_per_group
    )
    max_chunk_elements = min(max_chunk_elements, SMEM_MEDIUM_QUAD_C_HY)
    min_chunk = vec_size * THREADS_PER_BLOCK

    ctas_per_group = (stride + max_chunk_elements - 1) // max_chunk_elements
    chunk_size = (stride + ctas_per_group - 1) // ctas_per_group
    chunk_size = ((chunk_size + vec_size - 1) // vec_size) * vec_size
    chunk_size = triton.next_power_of_2(chunk_size)
    chunk_size = min(max_chunk_elements, chunk_size)
    while chunk_size < min_chunk:
        max_chunk_elements = max_chunk_elements >> 1
        if max_chunk_elements < min_chunk:
            chunk_size = min_chunk
            break
        ctas_per_group = (stride + max_chunk_elements - 1) // max_chunk_elements
        chunk_size = (stride + ctas_per_group - 1) // ctas_per_group
        chunk_size = ((chunk_size + vec_size - 1) // vec_size) * vec_size
        chunk_size = triton.next_power_of_2(chunk_size)
        chunk_size = min(max_chunk_elements, chunk_size)

    # Hygon v0.3 (occupancy-3 step): per-CTA smem = 18560B (shared_ordered
    # 2048 u32 = 8KB + local_histogram/suffix_sum ~2KB + shared_scalars 128B
    # + cumsum hidden scratch 4KB + alignment). 3 x 18560 = 55680 <= 64KB and
    # 3 x 512 = 1536 threads <= 2560/SM and 3 x 512 x 118 VGPR = 181248 <=
    # 196608 regs/SM -> occupancy 3 (240 resident CTAs).
    occupancy = 3

    needs_cooperative = max_seq_len > MEDIUM_THRESHOLD
    if not needs_cooperative:
        ctas_per_group = 1
    hw_resident_cap = num_sms * occupancy
    max_resident_ctas = hw_resident_cap
    if needs_cooperative:
        # Hygon v0.3 (C-plan): headroom was num_sms (80), which drove
        # max_resident_ctas down from 160 to 80. With occupancy=2 the smem is
        # no longer the binding constraint, so headroom = one group's CTAs
        # (or smaller), letting max_resident_ctas approach 160 and doubling
        # the number of groups (rows in parallel).
        headroom = ctas_per_group if occupancy > 1 else 1
        if max_resident_ctas >= headroom + ctas_per_group:
            max_resident_ctas -= headroom
    num_groups = min(max_resident_ctas // ctas_per_group, num_rows)
    num_groups = max(1, num_groups)
    total_ctas = num_groups * ctas_per_group

    if needs_cooperative and total_ctas > hw_resident_cap:
        assert 0, "too many chunk"
    # RadixRowState layout:
    #     uint32_t histogram[3][256];
    #     uint32_t remaining_k;
    #     uint32_t prefix;
    #     int arrival_counter;
    #     int output_counter;
    histogram_bytes = RADIX * 3 * 4
    radix_row_state_bytes = histogram_bytes + 4 * 4
    # Hygon v0.3 (C-plan): per-row global buffered region (u32 per row).
    # max(2 x DBUF_C decode, 2 x MAX_BUFFERED_ITEMS medium) as in the kernel.
    BUFFERED_STRIDE = 7416
    buffered_bytes = num_rows * BUFFERED_STRIDE * 4
    assert workspace.size(0) >= num_groups * radix_row_state_bytes + buffered_bytes, (
        f"persistent_topk: workspace needs >= "
        f"{num_groups * radix_row_state_bytes + buffered_bytes} bytes, got "
        f"{workspace.size(0)}"
    )
    workspace[: (num_groups * radix_row_state_bytes)] = 0
    g_histogram_size = num_groups * histogram_bytes
    g_state_size = num_groups * 4 * 4
    g_histogram = (
        workspace[:g_histogram_size].view(torch.uint32).view(num_groups, 3, RADIX)
    )
    g_state = (
        workspace[g_histogram_size : g_histogram_size + g_state_size]
        .view(torch.int32)
        .view(num_groups, 4)
    )
    g_buffered = (
        workspace[
            g_histogram_size
            + g_state_size : g_histogram_size
            + g_state_size
            + buffered_bytes
        ]
        .view(torch.uint32)
        .view(num_rows, BUFFERED_STRIDE)
    )

    persistent_topk_kernel[(total_ctas,)](
        logits,
        output,
        seq_lens,
        num_rows,
        stride,
        k,
        max_seq_len,
        chunk_size,
        ctas_per_group,
        num_groups,
        g_histogram,
        g_state,
        g_buffered,
        VEC_SIZE=vec_size,
        BLOCK_SIZE=THREADS_PER_BLOCK,
        num_warps=THREADS_PER_BLOCK // 64,
    )
