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
import triton.experimental.tle.language as tle
import triton.language as tl

# MetaX FlagTree (0.6.0+metax, official main build) TLE support status:
# - Available: gpu.alloc + gpu.local_ptr (2c: full-view pointer + indices
#   sub-view), but they must be called inside compile-time static unrolling
#   (tl.static_range): mctle.local_pointers inside dynamic loops (scf.for)
#   crash ConvertTritonGPUToLLVM.
# - Not available: the `other` argument of smem loads is ignored (2b defect,
#   use mask instead).
# - Fixed by FlagTree V2.patch: smem atomics (tt.atomic_rmw local-pointer
#   verifier).
# Therefore: the large path (_radix_topk) keeps its ordered buffer in TLE
# smem (32KB cap); histogram accumulation uses block-level tl.histogram;
# the medium/decode paths and suffix/scalars keep global scratch.

THREADS_PER_BLOCK = 1024
RADIX = 256
# v0.2 T4: radix of the cooperative global path (11/11/10-bit rounds).
GLOBAL_RADIX = 2048
MEDIUN_HIST_BYTES = 2 * (RADIX + 128) * 4  # 3072
MEDIUM_SCALARS_BYTES = 5 * 4  # 20
MEDIUM_HEADER_SIZE = (MEDIUN_HIST_BYTES + MEDIUM_SCALARS_BYTES + 127) & (~127)  # 3200
MAX_BUFFERED_ITEMS = 4096
SMEM_MEDIUM = MEDIUM_HEADER_SIZE + 2 * MAX_BUFFERED_ITEMS * 4  # 35968
RADIX_THRESHOLD = 32768
DECODE_BINS = 2048
HIST2048_THRESHOLD = 8192
FIXED_SMEM_LARGE = ((RADIX + RADIX + 5) * 4 + 15) & (~15)  # 2080
# Medium path with bin replay needs up to 32768 uint32 of shared_ordered.
# SMEM_MEDIUM_QUAD_C = 32768 → 128KB. Must fit within max_smem_per_block.
MEDIUM_REPLAY_SMEM = 32768 * 4  # 131072 bytes
# Medium/decode paths reuse shared_ordered's smem (mirroring vLLM's single
# extern __shared__ buffer). SMEM_MEDIUM // 4 = 8992 uint32s cover both the
# medium layout (8992) and the decode layout (8192).
SMEM_MEDIUM_QUAD = SMEM_MEDIUM // 4  # 8992
# Decode path layout constants (see histogram_2048_topk in persistent_topk.cuh)
DECODE_SBASE = 8192 - 8  # 8184
DECODE_RHIST = RADIX + 128  # 384
DECODE_BOFF = 2 * DECODE_RHIST  # 768
DECODE_DBUF = (DECODE_SBASE - DECODE_BOFF) // 2  # 3708

# V0 per-CTA scratch layout (global memory, uint32) —
# replaces the NVIDIA version's tle.gpu.alloc shared-memory buffers:
#   [0, 256)             local_histogram
#   [256, 512)           suffix_sum
#   [512, 516)           shared_scalars
#   [516, 516+ordered)   shared_ordered (ordered = max(CHUNK_SIZE, 16384))
#   [516+16384)          medium/decode layout reuses the shared_ordered region
# The per-CTA stride is computed per launch (see scratch_stride in the host).
SCRATCH_HIST = tl.constexpr(0)
# v0.2 T4: coop radix is 11/11/10 bits -> the per-CTA suffix table needs 2048
# slots; it reuses the retired local-histogram area [0, 2048).
SCRATCH_SUFFIX = tl.constexpr(0)
SCRATCH_SCALARS = tl.constexpr(2048)
SCRATCH_ORDERED = tl.constexpr(2064)
SCRATCH_ORDERED_QUAD = tl.constexpr(
    16384
)  # decode-path bins buffer cap (8192 bins + 8192 layout)
# T2.2c (borrowed from hygon v0.3): chunk cap for the global large path.
# Global scratch does not consume smem, so the chunk can be enlarged to
# reduce ctas_per_group (more multi-row parallelism; hygon: >=8 CTAs/group).
RADIX_GLOBAL_CHUNK = tl.constexpr(32768)
# T2.2: the large path (_radix_topk) keeps its ordered buffer in TLE smem,
# capped at 8192 uint32 (32KB; 16384 exceeds the 64KB smem limit -- measured
# Required 69632 > 65536).
RADIX_SMEM_CHUNK = tl.constexpr(4096)

# MetaX C550 hardware limit: at most 512 threads per block (warp_size=64,
# num_warps<=8). Threads = num_warps * 64; BLOCK_SIZE is a tensor dimension
# and is decoupled from the thread count.
MAX_WARPS = 8
# Bounded-spin cap (see _barrier_with_atomic_add): 1<<30 far exceeds any
# legal wait count.
BARRIER_SPIN_LIMIT = tl.constexpr(1 << 30)

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
    # TODO: every thread query, no following debug_barrier needed
    arrival_counter = tl.atomic_add(
        arrival_counter_ptr,
        0,
        sem="acquire",
        scope="gpu",
    )
    # The MetaX FlagTree backend miscompiles/hangs on unbounded while-spin
    # (the loop never exits even when the condition is met; reproduced with
    # 2-16 CTAs). Use a bounded spin with an iteration counter instead: adding
    # the monotonically increasing `it` to the condition compiles correctly and
    # cross-CTA atomic visibility takes effect immediately.
    # BARRIER_SPIN_LIMIT is huge (~2^30), far beyond any real wait.
    it = tl.zeros((), dtype=tl.int32)
    while (arrival_counter < thresold) & (it < BARRIER_SPIN_LIMIT):
        arrival_counter = tl.atomic_add(
            arrival_counter_ptr,
            0,
            sem="acquire",
            scope="gpu",
        )
        it += 1


# Extracted from persistent_topk.cuh in https://github.com/vllm-project/vllm
@triton.jit
def _radix_topk(
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
    scratch_ptr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCRATCH_STRIDE: tl.constexpr,
):
    RADIX_THRESHOLD: tl.constexpr = 32768
    RADIX: tl.constexpr = 256
    HIST2048_THRESHOLD: tl.constexpr = 8192
    # T2.2: ordered buffer moved to TLE smem (32KB cap). All tile accesses are
    # statically unrolled (tl.static_range): local_pointers inside dynamic loops
    # cannot be lowered (427 assertion). Partial tiles are handled by a uniform
    # valid mask (replacing rem_tiles/rem_elems).
    NTILE: tl.constexpr = CHUNK_SIZE // (BLOCK_SIZE * VEC_SIZE)

    pid = tl.program_id(0)
    group_id = pid // ctas_per_group
    cta_in_group = pid % ctas_per_group
    if pid >= num_groups * ctas_per_group:
        return  # TODO: remove
    if cta_in_group != 0 and max_seq_len <= RADIX_THRESHOLD:
        return
    scratch_base = scratch_ptr + pid * SCRATCH_STRIDE
    suffix_sum_ptr = scratch_base + SCRATCH_SUFFIX
    shared_scalars_ptr = scratch_base + SCRATCH_SCALARS
    shared_ordered_ptr = scratch_base + SCRATCH_ORDERED

    g_histogram_ptr += group_id * 3 * RADIX
    g_state_ptr += group_id * 4
    barrier_phase = tl.zeros((), dtype=tl.uint32)
    total_iters = tl.cdiv(num_rows, num_groups)

    # T2.2: alloc must stay outside the row loop -- TLE alloc misbehaves inside
    # a helper called repeatedly from a loop, and buffered_tensor cannot be
    # passed across functions, so the whole row loop lives in this function.
    ordered_buf = tle.gpu.alloc(
        [CHUNK_SIZE],
        dtype=tl.uint32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )

    for i in tl.range(total_iters):
        row_idx = group_id + i * num_groups
        if row_idx < num_rows:
            seq_len = tl.load(lengths_ptr + row_idx)
            row_output = output_ptr + row_idx * TOPK
            row_in = tl.multiple_of(logits_ptr + row_idx * stride, VEC_SIZE * 4)
            if seq_len <= RADIX_THRESHOLD:
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
                            TOPK,
                            VEC_SIZE,
                            BLOCK_SIZE,
                        )
            else:
                my_chunk_start = cta_in_group * CHUNK_SIZE
                my_chunk_end = my_chunk_start + CHUNK_SIZE
                my_chunk_end = min(my_chunk_end, seq_len)
                actual_chunk_size = (
                    my_chunk_end - my_chunk_start if my_chunk_start < seq_len else 0
                )
                lane = tl.arange(0, BLOCK_SIZE)
                zeros = tl.zeros([BLOCK_SIZE], dtype=tl.uint32)

                # -- Stage 1: Load chunk to smem as ordered uint32 --
                # T2.1-2: build the round-0 histogram (top-8 bits) during the load/convert
                # pass, saving the first Stage-2 smem sweep.
                flat = tl.arange(0, BLOCK_SIZE * VEC_SIZE)
                hist_acc0 = tl.zeros([RADIX], dtype=tl.int32)
                for t in tl.static_range(0, NTILE):
                    offs = t * BLOCK_SIZE * VEC_SIZE + flat
                    valid = offs < actual_chunk_size
                    x = tl.load(
                        row_in + my_chunk_start + offs, mask=valid, other=float("-inf")
                    )
                    bits = _convert_to_uint32_v2(x)
                    tl.store(tle.gpu.local_ptr(ordered_buf, (offs,)), bits, mask=valid)
                    hist_acc0 += tl.histogram(
                        (bits >> 24).to(tl.int32), RADIX, mask=valid
                    )
                tl.debug_barrier()

                # -- Init radix select state --
                tl.store(shared_scalars_ptr + zeros, 0, mask=lane == 0)  # prefix
                tl.store(
                    shared_scalars_ptr + 1 + zeros, TOPK, mask=lane == 0
                )  # remaining_k
                tl.debug_barrier()

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
                    tl.store(
                        g_state_ptr + 3 + zeros, 0, mask=lane == 0
                    )  # output_counter

                # MetaX fix: the 3-buffer rotation with "clear next" breaks when a small
                # row (no radix rounds, skips the barrier) appears in the group -- a buffer
                # is reused before being cleared, and stale counts corrupt the histogram
                # (wrong threshold, lost elements). Clear all 3 histogram buffers + barrier
                # at row start; the in-round "clear next" pipeline is then safe to reuse.
                if cta_in_group == 0:
                    for buf in tl.static_range(0, 3):
                        tl.store(
                            g_histogram_ptr + buf * RADIX + lane, 0, mask=lane < RADIX
                        )
                _barrier_with_atomic_add(
                    g_state_ptr + 2,
                    zeros,
                    lane,
                    (barrier_phase + 1) * ctas_per_group,
                )
                barrier_phase += 1

                # -- Stage 2: 4 rounds of radix select --
                for round_idx in tl.static_range(0, 4):
                    global_round = i * 4 + round_idx
                    shift_bits = 24 - round_idx * 8
                    prefix = tl.load(shared_scalars_ptr)
                    remaining_k = tl.load(shared_scalars_ptr + 1)

                    # current_hist inited zero in host-side or pre iter of group
                    current_hist_ptr = g_histogram_ptr + (global_round % 3) * RADIX
                    next_hist_ptr = g_histogram_ptr + ((global_round + 1) % 3) * RADIX

                    # MetaX optimization: use a block-level tl.histogram (smem-based internally)
                    # instead of one global atomic per element (16384 atomics/round -> 0); 1D
                    # flat loads avoid layout conversions. T2.1-2: the round-0 histogram is
                    # already built in Stage 1 (hist_acc0) and reused directly.
                    if round_idx == 0:
                        counts = hist_acc0
                    else:
                        hist_acc = tl.zeros([RADIX], dtype=tl.int32)
                        mask = 0 if round_idx == 0 else ((~0) << (32 - round_idx * 8))
                        for t in tl.static_range(0, NTILE):
                            offs = t * BLOCK_SIZE * VEC_SIZE + flat
                            valid = offs < actual_chunk_size
                            ordered = tl.load(
                                tle.gpu.local_ptr(ordered_buf, (offs,)),
                                mask=valid,
                                other=0,
                            )
                            match = (ordered & mask) == prefix
                            bucket = ((ordered >> shift_bits) & 0xFF).to(tl.int32)
                            hist_acc += tl.histogram(bucket, RADIX, mask=match & valid)
                        counts = hist_acc

                    tl.atomic_add(
                        current_hist_ptr + tl.arange(0, RADIX),
                        counts,
                        mask=counts > 0,
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

                    # MetaX optimization: suffix-sum via a block-level tl.cumsum (one
                    # cross-thread scan) instead of the 8-step static loop (2 debug_barriers
                    # per step, 16 barriers/round, 64 per row); metax's block scan
                    # (maca.barrier internally) is much faster than per-step barriers.
                    g_counts = tl.load(
                        current_hist_ptr + lane, mask=lane < RADIX, other=0
                    )
                    suffix_vals = g_counts.to(tl.int32)
                    suffix_sum_val = tl.cumsum(suffix_vals, axis=0, reverse=True)
                    tl.store(suffix_sum_ptr + lane, suffix_sum_val, mask=lane < RADIX)
                    tl.debug_barrier()

                    tl.store(
                        shared_scalars_ptr + 2 + zeros, 0, mask=lane == 0
                    )  # threshold_bin
                    tl.store(
                        shared_scalars_ptr + 3 + zeros, remaining_k, mask=lane == 0
                    )  # next_remaining_k
                    tl.debug_barrier()

                    count_ge = tl.load(
                        suffix_sum_ptr + lane, mask=lane < RADIX, other=0
                    )
                    count_gt = tl.load(
                        suffix_sum_ptr + lane + 1, mask=(lane + 1) < RADIX, other=0
                    )
                    threshold_mask = (
                        (count_ge >= remaining_k)
                        & (count_gt < remaining_k)
                        & (lane < RADIX)
                    )
                    tl.store(shared_scalars_ptr + 2 + zeros, lane, mask=threshold_mask)
                    tl.store(
                        shared_scalars_ptr + 3 + zeros,
                        remaining_k - count_gt,
                        mask=threshold_mask,
                    )
                    tl.debug_barrier()

                    threshold_bin = tl.load(
                        shared_scalars_ptr + 2 + zeros, mask=lane == 0, other=0
                    )
                    new_prefix = prefix | (threshold_bin << shift_bits)
                    tl.store(shared_scalars_ptr + zeros, new_prefix, mask=lane == 0)
                    next_remaining_k = tl.load(
                        shared_scalars_ptr + 3 + zeros, mask=lane == 0, other=0
                    )
                    tl.store(
                        shared_scalars_ptr + 1 + zeros, next_remaining_k, mask=lane == 0
                    )
                    tl.debug_barrier()
                # end 4 radix rounds

                # -- Count local > pivot elements --
                ordered_pivot = tl.load(shared_scalars_ptr)
                # -- Stage 3: Collect top-k indices --
                # MetaX fix: high-concurrency atomic RMW to the same address on metax
                # returns duplicate values (measured ~half duplicates above ~65K RMWs),
                # causing position collisions and lost elements. Instead: count per-thread
                # elements in registers, do one atomic range reservation, then write with a
                # register cursor. Only non-overlapping positions matter (unordered output).
                # T2.1: merge gt/eq counting into a single pass.
                # T2.2: flat 1D static unrolling (smem ordered). Accumulate per-element
                # counts across tiles, one atomic reservation per element (4096/CTA,
                # equivalent to the original [BLOCK,VEC] structure).
                flat_zeros = tl.zeros([BLOCK_SIZE * VEC_SIZE], dtype=tl.uint32)
                gt_cnt = tl.zeros([BLOCK_SIZE * VEC_SIZE], dtype=tl.int32)
                eq_cnt = tl.zeros([BLOCK_SIZE * VEC_SIZE], dtype=tl.int32)
                for t in tl.static_range(0, NTILE):
                    offs = t * BLOCK_SIZE * VEC_SIZE + flat
                    valid = offs < actual_chunk_size
                    ordered = tl.load(
                        tle.gpu.local_ptr(ordered_buf, (offs,)), mask=valid, other=0
                    )
                    gt_cnt += ((ordered > ordered_pivot) & valid).to(tl.int32)
                    eq_cnt += ((ordered == ordered_pivot) & valid).to(tl.int32)
                gt_base = tl.atomic_add(
                    g_state_ptr + 3 + flat_zeros,
                    gt_cnt,
                    mask=gt_cnt > 0,
                    sem="relaxed",
                    scope="gpu",
                )
                cursor = tl.zeros([BLOCK_SIZE * VEC_SIZE], dtype=tl.int32)
                for t in tl.static_range(0, NTILE):
                    offs = t * BLOCK_SIZE * VEC_SIZE + flat
                    valid = offs < actual_chunk_size
                    ordered = tl.load(
                        tle.gpu.local_ptr(ordered_buf, (offs,)), mask=valid, other=0
                    )
                    gt_mask = (ordered > ordered_pivot) & valid
                    tl.store(
                        row_output + gt_base + cursor,
                        my_chunk_start + offs,
                        mask=gt_mask,
                    )
                    cursor += gt_mask.to(tl.int32)

                _barrier_with_atomic_add(
                    g_state_ptr + 2,
                    zeros,
                    lane,
                    (barrier_phase + 1) * ctas_per_group,
                )
                barrier_phase += 1
                # tl.debug_barrier()

                # -- eq-collect: single reservation + cursor write (counts from merged pass) --
                eq_base = tl.atomic_add(
                    g_state_ptr + 3 + flat_zeros,
                    eq_cnt,
                    mask=eq_cnt > 0,
                    sem="relaxed",
                    scope="gpu",
                )
                cursor = tl.zeros([BLOCK_SIZE * VEC_SIZE], dtype=tl.int32)
                for t in tl.static_range(0, NTILE):
                    offs = t * BLOCK_SIZE * VEC_SIZE + flat
                    valid = offs < actual_chunk_size
                    ordered = tl.load(
                        tle.gpu.local_ptr(ordered_buf, (offs,)), mask=valid, other=0
                    )
                    eq_mask = (ordered == ordered_pivot) & valid
                    pos = eq_base + cursor
                    tl.store(
                        row_output + pos,
                        my_chunk_start + offs,
                        mask=eq_mask & (pos < TOPK),
                    )
                    cursor += eq_mask.to(tl.int32)

    return

    # Medium path: 8K < seq_len <= 32K. Uses 2048-bin FP16-11bit histogram for


# Phase 1 (finer granularity → fewer elements in threshold bin), then 4-pass
# FP32 radix-256 refinement on the buffered threshold-bin elements.
@triton.jit
def _histogram_256_topk(
    row_input,
    row_output,
    seq_len,
    shared_ordered_ptr,
    TOPK: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    RADIX: tl.constexpr = 256
    DECODE_BINS: tl.constexpr = 2048
    MAX_BUFFERED_ITEMS: tl.constexpr = 4096
    BUF_TILES: tl.constexpr = (MAX_BUFFERED_ITEMS + BLOCK_SIZE - 1) // BLOCK_SIZE
    CLEAR_ROUNDS: tl.constexpr = DECODE_BINS // BLOCK_SIZE

    hist_ptr = shared_ordered_ptr
    hist0_ptr = shared_ordered_ptr
    hist1_ptr = shared_ordered_ptr + (RADIX + 128)
    buffered_indices = shared_ordered_ptr + DECODE_BINS
    medium_scalars = shared_ordered_ptr + DECODE_BINS + 2 * MAX_BUFFERED_ITEMS

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC_SIZE)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_2d = tl.zeros([BLOCK_SIZE, VEC_SIZE], dtype=tl.int32)

    n_vec_full = seq_len // (BLOCK_SIZE * VEC_SIZE)
    rem_tiles = (seq_len - n_vec_full * BLOCK_SIZE * VEC_SIZE) // BLOCK_SIZE
    rem_elems = seq_len % BLOCK_SIZE

    remaining_k = TOPK

    # -- Phase 1: 2048-bin histogram (FP16 high 11 bits) --
    for r in tl.static_range(0, CLEAR_ROUNDS):
        tl.store(hist_ptr + r * BLOCK_SIZE + lane, 0)
    tl.debug_barrier()
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_input + offs)
        bin_flat = _decode_bin(x).reshape(BLOCK_SIZE * VEC_SIZE)
        tl.atomic_add(hist_ptr + bin_flat, 1, sem="relaxed", scope="gpu")
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        x = tl.load(row_input + offs)
        bin = _decode_bin(x)
        tl.atomic_add(hist_ptr + bin, 1, sem="relaxed", scope="gpu")
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        x = tl.load(row_input + offs, mask=in_range, other=float("-inf"))
        bin = _decode_bin(x)
        tl.atomic_add(hist_ptr + bin, 1, mask=in_range, sem="relaxed", scope="gpu")
    tl.debug_barrier()

    # -- Streaming threshold search via block-wide cumsum --
    THRESHOLD_ROUNDS: tl.constexpr = DECODE_BINS // BLOCK_SIZE
    threshold_found = tl.full((), False, dtype=tl.int1)
    last_value = tl.zeros((), dtype=tl.int32)
    cutoff = seq_len - TOPK
    for round_idx in tl.static_range(0, THRESHOLD_ROUNDS):
        if not threshold_found:
            round_bins = round_idx * BLOCK_SIZE + lane
            round_counts = tl.load(hist_ptr + round_bins).to(tl.int32)
            round_total = tl.sum(round_counts)
            ps = round_total - tl.cumsum(round_counts, axis=0, reverse=True)
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
            bin = _decode_bin(x)
            above = (bin > threshold_bin).reshape(BLOCK_SIZE, VEC_SIZE)
            out_pos = tl.atomic_add(
                medium_scalars + 0 + zeros_2d, 1, mask=above, sem="relaxed", scope="gpu"
            )
            tl.store(row_output + out_pos, offs, mask=above)
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
            x = tl.load(row_input + offs)
            above = _decode_bin(x) > threshold_bin
            out_pos = tl.atomic_add(
                medium_scalars + 0 + zeros, 1, mask=above, sem="relaxed", scope="gpu"
            )
            tl.store(row_output + out_pos, offs.to(tl.int32), mask=above)
        if rem_elems > 0:
            offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(row_input + offs, mask=in_range, other=float("-inf"))
            above = in_range & (_decode_bin(x) > threshold_bin)
            out_pos = tl.atomic_add(
                medium_scalars + 0 + zeros, 1, mask=above, sem="relaxed", scope="gpu"
            )
            tl.store(row_output + out_pos, offs.to(tl.int32), mask=above)
        tl.debug_barrier()
        return

    # -- Filter: output > threshold, buffer == threshold + build next histogram --
    tl.store(medium_scalars + 0 + zeros, 0, mask=lane == 0)
    tl.store(medium_scalars + 2 + zeros, 0, mask=lane == 0)
    tl.store(hist0_ptr + lane, 0, mask=lane < (RADIX + 1))
    tl.debug_barrier()
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_input + offs)
        bin = _decode_bin(x)
        above = (bin > threshold_bin).reshape(BLOCK_SIZE, VEC_SIZE)
        equal = (bin == threshold_bin).reshape(BLOCK_SIZE, VEC_SIZE)
        out_pos = tl.atomic_add(
            medium_scalars + 0 + zeros_2d, 1, mask=above, sem="relaxed", scope="gpu"
        )
        tl.store(row_output + out_pos, offs, mask=above)
        buf_pos = tl.atomic_add(
            medium_scalars + 2 + zeros_2d, 1, mask=equal, sem="relaxed", scope="gpu"
        )
        in_buf = equal & (buf_pos < MAX_BUFFERED_ITEMS)
        tl.store(buffered_indices + buf_pos, offs, mask=in_buf)
        fp32_bits = _convert_to_uint32_v2(x)
        next_bin_flat = ((fp32_bits >> 24) & 0xFF).reshape(BLOCK_SIZE * VEC_SIZE)
        tl.atomic_add(
            hist0_ptr + next_bin_flat,
            1,
            mask=in_buf.reshape(BLOCK_SIZE * VEC_SIZE),
            sem="relaxed",
            scope="gpu",
        )
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        x = tl.load(row_input + offs)
        bin = _decode_bin(x)
        above = bin > threshold_bin
        equal = bin == threshold_bin
        out_pos = tl.atomic_add(
            medium_scalars + 0 + zeros, 1, mask=above, sem="relaxed", scope="gpu"
        )
        tl.store(row_output + out_pos, offs.to(tl.int32), mask=above)
        buf_pos = tl.atomic_add(
            medium_scalars + 2 + zeros, 1, mask=equal, sem="relaxed", scope="gpu"
        )
        in_buf = equal & (buf_pos < MAX_BUFFERED_ITEMS)
        tl.store(buffered_indices + buf_pos, offs.to(tl.int32), mask=in_buf)
        fp32_bits = _convert_to_uint32_v2(x)
        next_bin = (fp32_bits >> 24) & 0xFF
        tl.atomic_add(hist0_ptr + next_bin, 1, mask=in_buf, sem="relaxed", scope="gpu")
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        x = tl.load(row_input + offs, mask=in_range, other=float("-inf"))
        bin = _decode_bin(x)
        above = in_range & (bin > threshold_bin)
        equal = in_range & (bin == threshold_bin)
        out_pos = tl.atomic_add(
            medium_scalars + 0 + zeros, 1, mask=above, sem="relaxed", scope="gpu"
        )
        tl.store(row_output + out_pos, offs.to(tl.int32), mask=above)
        buf_pos = tl.atomic_add(
            medium_scalars + 2 + zeros, 1, mask=equal, sem="relaxed", scope="gpu"
        )
        in_buf = equal & (buf_pos < MAX_BUFFERED_ITEMS)
        tl.store(buffered_indices + buf_pos, offs.to(tl.int32), mask=in_buf)
        fp32_bits = _convert_to_uint32_v2(x)
        next_bin = (fp32_bits >> 24) & 0xFF
        tl.atomic_add(hist0_ptr + next_bin, 1, mask=in_buf, sem="relaxed", scope="gpu")
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

    # -- 4-pass radix refinement --
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
                        scope="gpu",
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
                        scope="gpu",
                    )
                    tl.store(row_output + out_pos, buf_idx, mask=above)
                    if pass_idx == 3:
                        slot = tl.atomic_add(
                            medium_scalars + 4 + zeros,
                            -1,
                            mask=equal,
                            sem="relaxed",
                            scope="gpu",
                        ).to(tl.int32)
                        take = equal & (slot > 0)
                        tl.store(row_output + (TOPK - slot), buf_idx, mask=take)
                    else:
                        buffer_pos = tl.atomic_add(
                            medium_scalars + 2 + dst_buffer + zeros,
                            1,
                            mask=equal,
                            sem="relaxed",
                            scope="gpu",
                        )
                        in_buf = equal & (buffer_pos < MAX_BUFFERED_ITEMS)
                        tl.store(next_buf + buffer_pos, buf_idx, mask=in_buf)
                        next_bin = (fp32_bits >> (bit_offset - 8)) & 0xFF
                        tl.atomic_add(
                            hist0_ptr + next_bin,
                            1,
                            mask=in_buf,
                            sem="relaxed",
                            scope="gpu",
                        )
                tl.debug_barrier()


# histogram_2048_topk — production xiao implementation


@triton.jit
def _histogram_2048_topk(
    row_input,
    row_output,
    seq_len,
    shared_ordered_ptr,
    TOPK: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    DECODE_BINS_C: tl.constexpr = 2048
    RADIX_C: tl.constexpr = 256
    RHIST_C: tl.constexpr = RADIX_C + 128  # 384
    BOFF_C: tl.constexpr = 2 * RHIST_C  # 768
    DECODE_SBASE_C: tl.constexpr = 8184
    DBUF_C: tl.constexpr = (DECODE_SBASE_C - BOFF_C) // 2  # 3708
    NUM_BUF_TILES: tl.constexpr = (DBUF_C + BLOCK_SIZE - 1) // BLOCK_SIZE  # 4

    hist_ptr = shared_ordered_ptr
    buf0_ptr = shared_ordered_ptr + BOFF_C
    buf1_ptr = buf0_ptr + DBUF_C
    scalars_ptr = shared_ordered_ptr + DECODE_SBASE_C  # 8 scalars at [8184, 8192)
    # Bins stored at [8192, 8192+seq_len) for Phase 2 replay — no overlap with
    # histogram/buffers/scalars, avoids re-loading logits in Phase 2.
    BINS_OFF: tl.constexpr = 8192
    bins_store_ptr = shared_ordered_ptr + BINS_OFF

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC_SIZE)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_2d = tl.zeros([BLOCK_SIZE, VEC_SIZE], dtype=tl.int32)
    bins_2048 = tl.arange(0, DECODE_BINS_C)

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

    # -- Phase 1: build 2048-bin histogram + store bins for Phase 2 replay --
    tl.store(hist_ptr + bins_2048, 0)
    tl.debug_barrier()
    # Vectorized tiles: 4 elements per thread per load (vLLM float4 pattern)
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_input + offs)
        bin = _decode_bin(x)
        bin_flat = bin.reshape(BLOCK_SIZE * VEC_SIZE)
        tl.atomic_add(hist_ptr + bin_flat, 1, sem="relaxed", scope="gpu")
        tl.store(bins_store_ptr + offs.reshape(BLOCK_SIZE * VEC_SIZE), bin_flat)
    # Scalar tiles: tail elements that don't fill a full vec tile
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        x = tl.load(row_input + offs)
        bin = _decode_bin(x)
        tl.atomic_add(hist_ptr + bin, 1, sem="relaxed", scope="gpu")
        tl.store(bins_store_ptr + offs, bin)
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        x = tl.load(row_input + offs, mask=in_range, other=float("-inf"))
        bin = _decode_bin(x)
        tl.atomic_add(hist_ptr + bin, 1, mask=in_range, sem="relaxed", scope="gpu")
        tl.store(bins_store_ptr + offs, bin, mask=in_range)
    tl.debug_barrier()

    # -- Streaming suffix sum via block-wide cumsum on [BLOCK_SIZE] --
    THRESHOLD_ROUNDS: tl.constexpr = DECODE_BINS_C // BLOCK_SIZE
    threshold_found = tl.full((), False, dtype=tl.int1)
    last_value = tl.zeros((), dtype=tl.int32)
    cutoff = seq_len - TOPK
    for round_idx in tl.static_range(0, THRESHOLD_ROUNDS):
        if not threshold_found:
            round_bins = round_idx * BLOCK_SIZE + lane
            round_counts = tl.load(hist_ptr + round_bins).to(tl.int32)
            round_total = tl.sum(round_counts)
            ps = round_total - tl.cumsum(round_counts, axis=0, reverse=True)
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

    # -- Phase 2: replay bins from smem, output above/buffer equal --
    # Vectorized tiles (same pattern as Phase 1)
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        offs_flat = offs.reshape(BLOCK_SIZE * VEC_SIZE)
        bin_flat = tl.load(bins_store_ptr + offs_flat)
        above = (bin_flat > threshold_bin).reshape(BLOCK_SIZE, VEC_SIZE)
        equal = (bin_flat == threshold_bin).reshape(BLOCK_SIZE, VEC_SIZE)
        out_pos = tl.atomic_add(
            scalars_ptr + 1 + zeros_2d, 1, mask=above, sem="relaxed", scope="gpu"
        )
        tl.store(row_output + out_pos, offs, mask=above)
        buf_pos = tl.atomic_add(
            scalars_ptr + 4 + zeros_2d, 1, mask=equal, sem="relaxed", scope="gpu"
        )
        in_buf = equal & (buf_pos < DBUF_C)
        tl.store(buf0_ptr + buf_pos, offs, mask=in_buf)
    # Scalar tiles
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        bin = tl.load(bins_store_ptr + offs)
        above = bin > threshold_bin
        equal = bin == threshold_bin
        out_pos = tl.atomic_add(
            scalars_ptr + 1 + zeros, 1, mask=above, sem="relaxed", scope="gpu"
        )
        tl.store(row_output + out_pos, offs, mask=above)
        buf_pos = tl.atomic_add(
            scalars_ptr + 4 + zeros, 1, mask=equal, sem="relaxed", scope="gpu"
        )
        in_buf = equal & (buf_pos < DBUF_C)
        tl.store(buf0_ptr + buf_pos, offs, mask=in_buf)
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        bin = tl.load(bins_store_ptr + offs, mask=in_range, other=0)
        above = (bin > threshold_bin) & in_range
        equal = (bin == threshold_bin) & in_range
        out_pos = tl.atomic_add(
            scalars_ptr + 1 + zeros, 1, mask=above, sem="relaxed", scope="gpu"
        )
        tl.store(row_output + out_pos, offs, mask=above)
        buf_pos = tl.atomic_add(
            scalars_ptr + 4 + zeros, 1, mask=equal, sem="relaxed", scope="gpu"
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
        tl.atomic_add(refine0_ptr + next_bin, 1, mask=valid, sem="relaxed", scope="gpu")
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
                        scope="gpu",
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
                        scope="gpu",
                    )
                    tl.store(row_output + out_pos, idx, mask=above)
                    if pass_idx == 3:
                        slot = tl.atomic_add(
                            scalars_ptr + 3 + zeros,
                            -1,
                            mask=equal,
                            sem="relaxed",
                            scope="gpu",
                        ).to(tl.int32)
                        take = equal & (slot > 0)
                        tl.store(row_output + (TOPK - slot), idx, mask=take)
                    else:
                        buf_pos = tl.atomic_add(
                            scalars_ptr + buf_count_dst_idx + zeros,
                            1,
                            mask=equal,
                            sem="relaxed",
                            scope="gpu",
                        )
                        in_buf = equal & (buf_pos < DBUF_C)
                        tl.store(dst_buf_ptr + buf_pos, idx, mask=in_buf)
                        next_bin = (fp32_bits >> (bit_offset - 8)) & 0xFF
                        tl.atomic_add(
                            refine0_ptr + next_bin,
                            1,
                            mask=in_buf,
                            sem="relaxed",
                            scope="gpu",
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
    scratch_ptr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCRATCH_STRIDE: tl.constexpr,
):
    _radix_topk(
        logits_ptr,
        output_ptr,
        lengths_ptr,
        num_rows,
        stride,
        TOPK,
        max_seq_len,
        CHUNK_SIZE,
        ctas_per_group,
        num_groups,
        g_histogram_ptr,
        g_state_ptr,
        scratch_ptr,
        VEC_SIZE,
        BLOCK_SIZE,
        SCRATCH_STRIDE,
    )


@triton.jit
def _radix_topk_global(
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
    scratch_ptr,
    cta_in_group,
    ctas_per_group,
    barrier_phase,
    iter_idx,
    TOPK: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # v0.2 T4: 3 rounds of 11/11/10 bits (radix 2048; an 11-bit digit needs
    # 2048 bins) instead of 4 x 8-bit (radix 256): one fewer global barrier
    # and one fewer full chunk scan.
    RADIX: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_MASK: tl.constexpr = 0x3FF

    my_chunk_end = my_chunk_start + CHUNK_SIZE
    my_chunk_end = min(my_chunk_end, seq_len)
    actual_chunk_size = my_chunk_end - my_chunk_start if my_chunk_start < seq_len else 0
    lane = tl.arange(0, BLOCK_SIZE)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.uint32)
    zeros_2d = tl.zeros([BLOCK_SIZE, VEC_SIZE], dtype=tl.uint32)
    vec = tl.arange(0, VEC_SIZE)
    lane_hist = tl.arange(0, RADIX)
    zeros_hist = tl.zeros([RADIX], dtype=tl.int32)

    # v0.2 T4b: the round-0 histogram (11-bit digit) is accumulated while the
    # chunk is loaded/converted below, so the smem histogram is zeroed once
    # here instead of at the start of round 0. Saves one full scan of the
    # ordered buffer per row.
    tl.store(local_histogram_ptr + lane_hist, 0, mask=lane_hist < RADIX)
    tl.debug_barrier()

    # -- Stage 1: Load chunk to shared memory as ordered uint32 --
    # TODO: remove rem_tiles, rem_elems
    n_vec_full = actual_chunk_size // (BLOCK_SIZE * VEC_SIZE)
    rem_tiles = (actual_chunk_size - n_vec_full * BLOCK_SIZE * VEC_SIZE) // BLOCK_SIZE
    rem_elems = actual_chunk_size % BLOCK_SIZE
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_input + my_chunk_start + offs)
        bits = _convert_to_uint32_v2(x)
        tl.store(shared_ordered_ptr + offs, bits)
        tl.atomic_add(
            local_histogram_ptr + ((bits >> 21) & RADIX11_MASK).to(tl.int32),
            1,
            sem="relaxed",
            scope="cta",
        )
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        x = tl.load(row_input + my_chunk_start + offs)
        bits = _convert_to_uint32_v2(x)
        tl.store(shared_ordered_ptr + offs, bits)
        tl.atomic_add(
            local_histogram_ptr + ((bits >> 21) & RADIX11_MASK).to(tl.int32),
            1,
            sem="relaxed",
            scope="cta",
        )
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        x = tl.load(
            row_input + my_chunk_start + offs, mask=in_range, other=float("-inf")
        )
        bits = _convert_to_uint32_v2(x)
        tl.store(shared_ordered_ptr + offs, bits, mask=in_range)
        tl.atomic_add(
            local_histogram_ptr + ((bits >> 21) & RADIX11_MASK).to(tl.int32),
            1,
            mask=in_range,
            sem="relaxed",
            scope="cta",
        )
    tl.debug_barrier()

    # -- Init radix select state --
    tl.store(shared_scalars_ptr + zeros, 0, mask=lane == 0)  # prefix
    tl.store(shared_scalars_ptr + 1 + zeros, TOPK, mask=lane == 0)  # remaining_k
    tl.debug_barrier()

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

    # MetaX fix: the 3-buffer rotation with "clear next" breaks when a small
    # row (no radix rounds, skips the barrier) appears in the group -- a buffer
    # is reused before being cleared, and stale counts corrupt the histogram
    # (wrong threshold, lost elements). Clear all 3 histogram buffers + barrier
    # at row start; the in-round "clear next" pipeline is then safe to reuse.
    if cta_in_group == 0:
        lane_hist0 = tl.arange(0, RADIX)
        for buf in tl.static_range(0, 3):
            tl.store(
                g_histogram_ptr + buf * RADIX + lane_hist0,
                0,
                mask=lane_hist0 < RADIX,
            )
    _barrier_with_atomic_add(
        g_state_ptr + 2,
        zeros,
        lane,
        (barrier_phase + 1) * ctas_per_group,
    )
    barrier_phase += 1

    # -- Stage 2: 3 rounds of radix select (11/11/10 bits) --
    for round_idx in tl.static_range(0, 3):
        global_round = iter_idx * 3 + round_idx
        shift_bits = 21 if round_idx == 0 else (10 if round_idx == 1 else 0)
        bucket_mask = RADIX11_MASK if round_idx < 2 else RADIX10_MASK
        prefix = tl.load(shared_scalars_ptr)
        remaining_k = tl.load(shared_scalars_ptr + 1)

        # current_hist inited zero in host-side or pre iter of group
        current_hist_ptr = g_histogram_ptr + (global_round % 3) * RADIX
        next_hist_ptr = g_histogram_ptr + ((global_round + 1) % 3) * RADIX

        # v0.1 optimization: per-element smem atomics instead of register
        # accumulation via tl.histogram. Measured on C550 (coop pattern,
        # 4 rounds x 32768 elements, BLOCK=1024): 0.070ms vs 0.203ms (2.9x).
        flat = tl.arange(0, BLOCK_SIZE * VEC_SIZE)
        if round_idx != 0:
            tl.store(local_histogram_ptr + lane_hist, 0, mask=lane_hist < RADIX)
            tl.debug_barrier()
            for t in tl.range(0, n_vec_full):
                offs = t * BLOCK_SIZE * VEC_SIZE + flat
                ordered = tl.load(shared_ordered_ptr + offs)
                if round_idx == 0:
                    match = tl.full(ordered.shape, True, tl.int1)
                elif round_idx == 1:
                    match = ((ordered ^ prefix) >> 21) == 0
                else:
                    match = ((ordered ^ prefix) >> 10) == 0
                bucket = ((ordered >> shift_bits) & bucket_mask).to(tl.int32)
                tl.atomic_add(
                    local_histogram_ptr + bucket,
                    1,
                    mask=match,
                    sem="relaxed",
                    scope="cta",
                )
            for t in tl.range(0, rem_tiles):
                offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
                ordered = tl.load(shared_ordered_ptr + offs)
                if round_idx == 0:
                    match = tl.full(ordered.shape, True, tl.int1)
                elif round_idx == 1:
                    match = ((ordered ^ prefix) >> 21) == 0
                else:
                    match = ((ordered ^ prefix) >> 10) == 0
                bucket = ((ordered >> shift_bits) & bucket_mask).to(tl.int32)
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
                ordered = tl.load(shared_ordered_ptr + offs, mask=in_range, other=0)
                if round_idx == 0:
                    match = tl.full(ordered.shape, True, tl.int1)
                elif round_idx == 1:
                    match = ((ordered ^ prefix) >> 21) == 0
                else:
                    match = ((ordered ^ prefix) >> 10) == 0
                bucket = ((ordered >> shift_bits) & bucket_mask).to(tl.int32)
                tl.atomic_add(
                    local_histogram_ptr + bucket,
                    1,
                    mask=match & in_range,
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()

        counts = tl.load(
            local_histogram_ptr + lane_hist, mask=lane_hist < RADIX, other=0
        )
        tl.atomic_add(
            current_hist_ptr + tl.arange(0, RADIX),
            counts,
            mask=counts > 0,
            sem="relaxed",
            scope="gpu",
        )

        if cta_in_group == 0:
            tl.store(next_hist_ptr + lane_hist, 0, mask=lane_hist < RADIX)

        _barrier_with_atomic_add(
            g_state_ptr + 2,
            zeros,
            lane,
            (barrier_phase + 1) * ctas_per_group,
        )
        barrier_phase += 1
        # tl.debug_barrier()

        # MetaX optimization: suffix-sum via a block-level tl.cumsum (one
        # cross-thread scan) instead of the 8-step static loop (2 debug_barriers
        # per step, 16 barriers/round, 64 per row); metax's block scan
        # (maca.barrier internally) is much faster than per-step barriers.
        g_counts = tl.load(
            current_hist_ptr + lane_hist, mask=lane_hist < RADIX, other=0
        )
        suffix_vals = g_counts.to(tl.int32)
        suffix_sum_val = tl.cumsum(suffix_vals, axis=0, reverse=True)
        tl.store(suffix_sum_ptr + lane_hist, suffix_sum_val, mask=lane_hist < RADIX)
        tl.debug_barrier()

        tl.store(shared_scalars_ptr + 2 + zeros, 0, mask=lane == 0)  # threshold_bin
        tl.store(
            shared_scalars_ptr + 3 + zeros, remaining_k, mask=lane == 0
        )  # next_remaining_k
        tl.debug_barrier()

        count_ge = tl.load(suffix_sum_ptr + lane_hist, mask=lane_hist < RADIX, other=0)
        count_gt = tl.load(
            suffix_sum_ptr + lane_hist + 1,
            mask=(lane_hist + 1) < RADIX,
            other=0,
        )
        threshold_mask = (
            (count_ge >= remaining_k) & (count_gt < remaining_k) & (lane_hist < RADIX)
        )
        tl.store(shared_scalars_ptr + 2 + zeros_hist, lane_hist, mask=threshold_mask)
        tl.store(
            shared_scalars_ptr + 3 + zeros_hist,
            remaining_k - count_gt,
            mask=threshold_mask,
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
    # end 3 radix rounds

    # -- Count local > pivot elements --
    ordered_pivot = tl.load(shared_scalars_ptr)
    # -- Stage 3: Collect top-k indices --
    # MetaX fix: high-concurrency atomic RMW to the same address on metax
    # returns duplicate values (measured ~half duplicates above ~65K RMWs),
    # causing position collisions and lost elements. Instead: count per-thread
    # register cursor. 2D and 1D parts reserve separately; only non-overlapping
    # positions matter (unordered output).
    # T2.1: merge gt/eq counting into one pass (collect 4 reads -> 3 reads).
    gt_cnt_2d = tl.zeros([BLOCK_SIZE, VEC_SIZE], dtype=tl.int32)
    eq_cnt_2d = tl.zeros([BLOCK_SIZE, VEC_SIZE], dtype=tl.int32)
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        ordered = tl.load(shared_ordered_ptr + offs)
        gt_cnt_2d += (ordered > ordered_pivot).to(tl.int32)
        eq_cnt_2d += (ordered == ordered_pivot).to(tl.int32)
    gt_cnt_1d = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    eq_cnt_1d = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        ordered = tl.load(shared_ordered_ptr + offs)
        gt_cnt_1d += (ordered > ordered_pivot).to(tl.int32)
        eq_cnt_1d += (ordered == ordered_pivot).to(tl.int32)
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        ordered = tl.load(shared_ordered_ptr + offs, mask=in_range, other=0)
        gt_cnt_1d += ((ordered > ordered_pivot) & in_range).to(tl.int32)
        eq_cnt_1d += ((ordered == ordered_pivot) & in_range).to(tl.int32)
    gt_base_2d = tl.atomic_add(
        g_state_ptr + 3 + zeros_2d,
        gt_cnt_2d,
        mask=gt_cnt_2d > 0,
        sem="relaxed",
        scope="gpu",
    )
    gt_base_1d = tl.atomic_add(
        g_state_ptr + 3 + zeros,
        gt_cnt_1d,
        mask=gt_cnt_1d > 0,
        sem="relaxed",
        scope="gpu",
    )
    _barrier_with_atomic_add(
        g_state_ptr + 2,
        zeros,
        lane,
        (barrier_phase + 1) * ctas_per_group,
    )
    barrier_phase += 1
    # tl.debug_barrier()

    # -- eq-collect: single reservation + cursor write (counts from merged pass) --
    eq_base_2d = tl.atomic_add(
        g_state_ptr + 3 + zeros_2d,
        eq_cnt_2d,
        mask=eq_cnt_2d > 0,
        sem="relaxed",
        scope="gpu",
    )
    eq_base_1d = tl.atomic_add(
        g_state_ptr + 3 + zeros,
        eq_cnt_1d,
        mask=eq_cnt_1d > 0,
        sem="relaxed",
        scope="gpu",
    )
    # v0.1: gt and eq are collected in a single pass. The gt reservation
    # happened before the barrier, so every gt position is globally below
    # every eq position (the k padding rule still holds); the eq reservation
    # is done here, then one scan writes both sets.
    cursor_gt_2d = tl.zeros([BLOCK_SIZE, VEC_SIZE], dtype=tl.int32)
    cursor_eq_2d = tl.zeros([BLOCK_SIZE, VEC_SIZE], dtype=tl.int32)
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        ordered = tl.load(shared_ordered_ptr + offs)
        gt_mask = ordered > ordered_pivot
        eq_mask = ordered == ordered_pivot
        tl.store(
            row_output + gt_base_2d + cursor_gt_2d,
            my_chunk_start + offs,
            mask=gt_mask,
        )
        cursor_gt_2d += gt_mask.to(tl.int32)
        pos = eq_base_2d + cursor_eq_2d
        tl.store(row_output + pos, my_chunk_start + offs, mask=eq_mask & (pos < TOPK))
        cursor_eq_2d += eq_mask.to(tl.int32)
    cursor_gt_1d = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    cursor_eq_1d = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        ordered = tl.load(shared_ordered_ptr + offs)
        gt_mask = ordered > ordered_pivot
        eq_mask = ordered == ordered_pivot
        tl.store(
            row_output + gt_base_1d + cursor_gt_1d, my_chunk_start + offs, mask=gt_mask
        )
        cursor_gt_1d += gt_mask.to(tl.int32)
        pos = eq_base_1d + cursor_eq_1d
        tl.store(row_output + pos, my_chunk_start + offs, mask=eq_mask & (pos < TOPK))
        cursor_eq_1d += eq_mask.to(tl.int32)
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        ordered = tl.load(shared_ordered_ptr + offs, mask=in_range, other=0)
        gt_mask = (ordered > ordered_pivot) & in_range
        eq_mask = (ordered == ordered_pivot) & in_range
        tl.store(
            row_output + gt_base_1d + cursor_gt_1d, my_chunk_start + offs, mask=gt_mask
        )
        cursor_gt_1d += gt_mask.to(tl.int32)
        pos = eq_base_1d + cursor_eq_1d
        tl.store(row_output + pos, my_chunk_start + offs, mask=eq_mask & (pos < TOPK))
        cursor_eq_1d += eq_mask.to(tl.int32)

    return barrier_phase

    # Medium path: 8K < seq_len <= 32K. Uses 2048-bin FP16-11bit histogram for


# Phase 1 (finer granularity → fewer elements in threshold bin), then 4-pass
# FP32 radix-256 refinement on the buffered threshold-bin elements.


@triton.jit
def persistent_topk_kernel_global(
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
    scratch_ptr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCRATCH_STRIDE: tl.constexpr,
):
    RADIX_THRESHOLD: tl.constexpr = 32768
    # v0.2 T4: local histogram radix matches _radix_topk_global (11/11/10).
    RADIX: tl.constexpr = 2048
    HIST2048_THRESHOLD: tl.constexpr = 8192

    pid = tl.program_id(0)
    group_id = pid // ctas_per_group
    cta_in_group = pid % ctas_per_group
    if pid >= num_groups * ctas_per_group:
        return  # TODO: remove
    if cta_in_group != 0 and max_seq_len <= RADIX_THRESHOLD:
        return
    # MetaX version: global scratch replaces the NVIDIA version's
    # tle.gpu.alloc shared-memory buffers. One segment per CTA; see the
    # SCRATCH_* constants for the layout:
    #   [0, 2048) suffix_sum, [2048, 2052) scalars, [2064, 2064+ordered)
    #   shared_ordered. v0.1: the radix histogram lives in smem (per-element
    #   atomics); v0.2 T4: suffix_sum grew to 2048 bins (11/11/10 radix).
    scratch_base = scratch_ptr + pid * SCRATCH_STRIDE
    suffix_sum_ptr = scratch_base + SCRATCH_SUFFIX
    shared_scalars_ptr = scratch_base + SCRATCH_SCALARS
    shared_ordered_ptr = scratch_base + SCRATCH_ORDERED
    s_histogram = tle.gpu.alloc(
        [RADIX],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    local_histogram_ptr = tle.gpu.local_ptr(s_histogram, (0,))

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
            if seq_len <= RADIX_THRESHOLD:
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
                            TOPK,
                            VEC_SIZE,
                            BLOCK_SIZE,
                        )
            else:
                my_chunk_start = cta_in_group * CHUNK_SIZE
                barrier_phase = _radix_topk_global(
                    row_in,
                    row_output,
                    seq_len,
                    my_chunk_start,
                    CHUNK_SIZE,
                    local_histogram_ptr,
                    suffix_sum_ptr,
                    shared_scalars_ptr,
                    shared_ordered_ptr,
                    g_histogram_ptr,
                    g_state_ptr,
                    scratch_ptr,
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
# per-row (per_row_cta) top-k path — single-CTA kernel, dispatched when num_rows > 32 or
# when the cooperative path is not expected to win (see persistent_topk host:
# small seq, or num_rows * 16384 >= max_seq_len).
# Ported from persistent_topk.py: 4-step radix (2048/2048/2048/1024 bins)
# with tle.cumsum streaming threshold search and _per_row_cta_final_radix_select.
# Uses inverted sortable-key convention compared to vLLM-style paths above.
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
    # v0.2: step 0 only needs the 11-bit hi digit; skip the full sortable-key
    # conversion (it is only used by steps >= 1).
    if step_idx == 0:
        digit = _per_row_cta_convert_to_fp16_hi11(x)
        partial = in_range
    else:
        key = _per_row_cta_convert_to_sortable_uint32(x)
        if step_idx == 1:
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
    JOINT=False,
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

    # v0.2: step 0 only needs the 11-bit hi digit; skip the full sortable-key
    # conversion (it is only used by steps >= 1).
    if step_idx == 0:
        digit = _per_row_cta_convert_to_fp16_hi11(x)
        partial = in_range
    else:
        key = _per_row_cta_convert_to_sortable_uint32(x)
        if step_idx == 1:
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

    if JOINT:
        # v0.2: single reservation for all candidates (digit <= threshold);
        # the final selection picks the top-k from this joint buffer, removing
        # the second per-element predicated atomic.
        take_j = partial & (digit <= threshold_bin_idx)
        out_pos_j = tl.atomic_add(
            found_ptrs, ones, mask=take_j, sem="relaxed", scope="cta"
        )
        tl.store(
            hist_base_ptr + out_pos_j,
            offs.to(tl.int32),
            mask=take_j & (out_pos_j < FINAL_SORT_ITEMS),
        )
        # keys are gathered from row_ptr + idx by the final selection
    else:
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
            tl.store(
                s_out_logits_ptr + out_pos_lt, x, mask=take_lt & (out_pos_lt < TOPK)
            )
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
            # keys are gathered from row_ptr + idx by the final selection


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
    USE_RADIX_FINAL: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    VEC: tl.constexpr = 4
    FINAL_SORT_ITEMS: tl.constexpr = 2048
    HIST_SIZE: tl.constexpr = 4096
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
            # The MetaX backend requires a mask on smem tensor loads (otherwise the
            # vec-inference assertion fires); bins < HIST_SIZE is always true and only
            # satisfies the backend lowering.
            counts = tl.load(hist_base_ptr + bins, mask=bins < HIST_SIZE, other=0)
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

    # v0.2 joint-candidate path: step 0 reserves one joint buffer holding all
    # candidates (digit <= threshold) when it fits FINAL_SORT_ITEMS.
    joint_ok = False
    if USE_RADIX_FINAL:
        joint_ok = (
            (step_idx == 0)
            & write_directly
            & use_final
            & ((TOPK + final_bin_size) <= FINAL_SORT_ITEMS)
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
                joint_ok,
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
                joint_ok,
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
                joint_ok,
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
                joint_ok,
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
                joint_ok,
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
                joint_ok,
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
                joint_ok,
                TOPK=TOPK,
            )
    tl.debug_barrier()
    if joint_ok:
        jcnt = tl.load(s_found_topk_values_ptr)
        tl.store(s_final_cnt_ptr + zeros, jcnt, mask=lane == 0)
        tl.store(s_found_topk_values_ptr + zeros, 0, mask=lane == 0)

    return (
        (final_bin_size > FINAL_SORT_ITEMS) & (joint_ok == 0),
        logit_pattern.to(tl.int32),
        threshold_bin_idx,
    )


@triton.jit
def _per_row_cta_final_radix_select(
    row_ptr,
    hist_base_ptr,
    s_out_indices_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    s_radix_count_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FINAL_SORT_ITEMS: tl.constexpr,
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
                idx_g = tl.load(hist_base_ptr + pos, mask=valid, other=0)
                x = tl.load(row_ptr + idx_g, mask=valid, other=float("-inf"))
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
            # The MetaX backend requires a mask on smem tensor loads (otherwise the
            # vec-inference assertion fires); bins < RADIX_SIZE_FINAL is always true and
            # only satisfies the backend lowering.
            counts = tl.load(radix_count_vec_ptr, mask=bins < RADIX_SIZE_FINAL, other=0)
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
            x = tl.load(row_ptr + idx, mask=valid, other=float("-inf"))
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
                    x = tl.load(row_ptr + idx, mask=valid, other=float("-inf"))
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
                    USE_RADIX_FINAL=USE_RADIX_FINAL,
                    TOPK=TOPK,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            )

    if not continue_to_next_step:
        if USE_RADIX_FINAL:
            _per_row_cta_final_radix_select(
                row_ptr,
                hist_base_ptr,
                s_out_indices_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                s_radix_count_ptr,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
                FINAL_SORT_ITEMS=FINAL_SORT_ITEMS,
            )
        else:
            base_idx = tl.load(s_found_topk_values_ptr)
            final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), FINAL_SORT_ITEMS)
            sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)
            for sort_chunk in tl.range(0, sort_chunks):
                pos = sort_chunk * BLOCK_SIZE + lane
                valid = pos < final_cnt
                idx_i_all = tl.load(hist_base_ptr + pos, mask=valid, other=0)
                logit_i = tl.load(
                    row_ptr + idx_i_all, mask=valid, other=float("-inf")
                ).to(tl.float32)
                out_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
                for j in tl.range(0, final_cnt):
                    idx_j = tl.load(hist_base_ptr + j)
                    logit_j = tl.load(row_ptr + idx_j).to(tl.float32)
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

    # MetaX: per-row single CTA with no cross-CTA sync, so all buffers live in
    # smem (aligned with the upstream _per_row_cta_* implementation; TLE smem
    # atomics fixed by FlagTree V2.patch).
    # Layout: s_histogram (holds the FINAL_SORT_ITEMS=2048 logit-bit store) +
    # s_out_indices + 5 scalars + s_radix_counts.
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
    )


# Cooperative global path: chunk-size selection.
# v0.2 T4 re-calibration (2026-09-22) after the coop radix became 3 rounds of
# 11/11/10 bits (radix 2048, one barrier less per row): per-CTA pass cost is
# about T(chunk) ~= 32us + 2.0us per 1000 elements and the machine still runs
# ~104 barrier CTAs concurrently, so total cost = waves * T(chunk) with
# waves = ceil(num_rows * cpg / 104), cpg = ceil(stride / chunk).
# Verified: 2 rows -> chunk 8192 (0.066ms), 7 -> 32768 (0.107ms),
# 13 -> 32768 (0.132ms), 14/16/18 -> 65536 (0.171/0.178/0.185ms, one wave of
# 56/64/72 CTAs), 20 rows -> per-row better (coop 0.191 vs 0.189ms).
_GLOBAL_CHUNK_CANDIDATES = (4096, 8192, 16384, 32768, 65536)
_GLOBAL_BARRIER_CAPACITY = 104
_GLOBAL_FIXED_US = 32.0
_GLOBAL_PER_KILO_US = 2.0
# per-row (per_row_cta) kernel cost (v0.2 T1e measured, joint single-atomic
# collect + key gather): one CTA per row, all rows in parallel, ~0.72us per
# 1000 elements (seq=262144 -> ~188us, flat for num_rows <= 104).
_PER_ROW_CTA_US_PER_KILO = 0.72


def _coop_cost_us(num_rows, stride):
    """Estimated cooperative large-path latency in us (see _pick_global_chunk).

    Uses the same stride-based geometry as the host (ctas_per_group is derived
    from the row stride, not from max_seq_len).
    """
    best = float("inf")
    for chunk in _GLOBAL_CHUNK_CANDIDATES:
        cpg = (stride + chunk - 1) // chunk
        waves = (
            num_rows * cpg + _GLOBAL_BARRIER_CAPACITY - 1
        ) // _GLOBAL_BARRIER_CAPACITY
        best = min(
            best, waves * (_GLOBAL_FIXED_US + _GLOBAL_PER_KILO_US * chunk / 1000.0)
        )
    return best


def _pick_global_chunk(num_rows, stride):
    best_chunk, best_cost = _GLOBAL_CHUNK_CANDIDATES[-1], float("inf")
    for chunk in _GLOBAL_CHUNK_CANDIDATES:
        cpg = (stride + chunk - 1) // chunk
        waves = (
            num_rows * cpg + _GLOBAL_BARRIER_CAPACITY - 1
        ) // _GLOBAL_BARRIER_CAPACITY
        cost = waves * (_GLOBAL_FIXED_US + _GLOBAL_PER_KILO_US * chunk / 1000.0)
        if cost < best_cost:
            best_cost, best_chunk = cost, chunk
    return best_chunk


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

    # per-row (per_row_cta) vs cooperative dispatch, cost-model driven (C550, k=512):
    #   - max_seq_len <= 32768: per-row wins for all row counts (measured:
    #     per-row 0.043-0.052ms vs cooperative 0.078-0.105ms);
    #   - otherwise compare the cooperative estimate (waves x T(chunk), see
    #     _coop_cost_us) against the per-row estimate (one CTA per row, all
    #     rows in parallel).
    # v0.2 T4 re-calibration (2026-09-22, same-process A/B, seq=262144,
    # 3-round coop): coop wins for 2-19 rows (18 rows: 0.185 vs 0.188;
    # 19 rows: 0.188 vs 0.189) and per-row wins from 20 rows (0.191 vs
    # 0.189). Hence the hard cap below (coop limited to num_rows <= 18).
    per_row_ok = max_seq_len <= RADIX_THRESHOLD or (
        num_rows >= 2
        and _coop_cost_us(num_rows, stride)
        > _PER_ROW_CTA_US_PER_KILO * max_seq_len / 1000.0
    )
    if num_rows > 18 or per_row_ok:
        # per-row (per_row_cta): one CTA per row, buffers in smem (upstream _per_row_cta_* style)
        # v0.1 tuning on C550 (seq=262144, k=512, measured median):
        #   BLOCK_SIZE=2048 is ~7-11% faster while the grid fits in one
        #   residency wave (num_rows <= 104): 19/24/32/40/64/96 rows
        #   0.242/0.242/0.244/0.247/0.255/0.262 vs 0.260/0.261/0.265/0.270/0.281/0.296.
        #   From 108 rows on, BLOCK_SIZE=512 wins (108/128/512 rows:
        #   0.394/0.407/1.097 vs 0.466/0.478/1.206) because the larger block
        #   reduces resident CTAs and adds a second wave.
        #   USE_RADIX_FINAL=False is consistently ~2% faster than True.
        # v0.2 T3 block sweep (2026-09-22, seq=262144, medians): BLOCK=512
        # saturates at ~212 resident CTAs, BLOCK=2048 at ~104, so the winner
        # is a wave-quantization see-saw (t2048 ~ ceil(n/104)*0.20ms,
        # t512 ~ ceil(n/212)*0.35ms):
        #   n <= 104       -> 2048 (1 wave, 0.218 vs 0.332)
        #   105-212        -> 512  (108/128/192/200/208: 0.35-0.38 vs 0.37-0.41)
        #   213-310        -> 2048 (216/224/256/288/300: 0.57-0.60 vs 0.60-0.67)
        #   311-460        -> 512  (320/352/384/400/432/448: 0.71-0.96 vs 0.77-0.97)
        #   n >= 461       -> 2048 (480/496/512: 0.98-0.99 vs 0.99-1.02)
        if num_rows <= 104 or 213 <= num_rows <= 310 or num_rows >= 461:
            per_row_cta_block_size = 2048
        else:
            per_row_cta_block_size = 512
        # v0.1: radix-final wins with BLOCK=2048 (0.237 vs 0.241 at 19 rows,
        # 0.251 vs 0.254 at 64, 0.257 vs 0.260 at 96); insertion-rank sort wins
        # with BLOCK=512 (0.407 vs 0.412 at 128, 0.695 vs 0.709 at 256).
        # v0.2 experiment: radix-final for all sizes (joint candidate path
        # only applies with radix-final; large candidate sets make the
        # insertion sort O(cnt^2) expensive).
        per_row_cta_radix_final = True
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
            BLOCK_SIZE=per_row_cta_block_size,
            USE_RADIX_FINAL=per_row_cta_radix_final,
            num_warps=MAX_WARPS,
        )
        return

    device = logits.device
    device_props = torch.cuda.get_device_properties(device.index)
    num_sms = device_props.multi_processor_count
    # Cooperative path (only 1 row + large seq, or large seq with few rows):
    # single row uses the smem version (faster sweep, 0.955 at 1 row); >=2 rows
    # use the global version (the smem version's 64-CTA-per-row barrier cost
    # kills multi-row parallelism; measured 0.41 at 2 rows).
    use_smem = num_rows <= 1
    if use_smem:
        available_for_ordered = int(RADIX_SMEM_CHUNK)
    else:
        # Row-count-dependent chunk (see _pick_global_chunk): the fixed 16K
        # chunk was 15-48% off the measured optimum for 2/3 and 7-13 rows.
        available_for_ordered = _pick_global_chunk(num_rows, stride)
    max_chunk_elements = available_for_ordered
    vec_size = 1
    if stride % 4 == 0:
        vec_size = 4
    elif stride % 2 == 0:
        vec_size = 2

    max_chunk_elements = (max_chunk_elements // vec_size) * vec_size
    min_chunk = vec_size * THREADS_PER_BLOCK
    max_chunk_elements = max(max_chunk_elements, min_chunk)
    max_chunk_elements = triton.next_power_of_2(max_chunk_elements)

    ctas_per_group = (stride + max_chunk_elements - 1) // max_chunk_elements
    chunk_size = (stride + ctas_per_group - 1) // ctas_per_group
    chunk_size = ((chunk_size + vec_size - 1) // vec_size) * vec_size
    chunk_size = triton.next_power_of_2(chunk_size)
    chunk_size = min(max_chunk_elements, chunk_size)
    while chunk_size > available_for_ordered:
        max_chunk_elements = max_chunk_elements >> 1
        if max_chunk_elements < min_chunk:
            chunk_size = min_chunk
            assert chunk_size <= available_for_ordered
            break
        ctas_per_group = (stride + max_chunk_elements - 1) // max_chunk_elements
        chunk_size = (stride + ctas_per_group - 1) // ctas_per_group
        chunk_size = ((chunk_size + vec_size - 1) // vec_size) * vec_size
        chunk_size = triton.next_power_of_2(chunk_size)
        chunk_size = min(max_chunk_elements, chunk_size)

    # Per-CTA scratch must cover the ordered buffer (chunk, large path) and the
    # decode/medium layout (SCRATCH_ORDERED_QUAD), whichever is larger.
    scratch_stride = int(SCRATCH_ORDERED) + max(int(SCRATCH_ORDERED_QUAD), chunk_size)

    # MetaX C550: 2048 threads per SM (measured max_threads_per_multi_processor),
    # 512 threads per CTA (num_warps=8 x warp_size=64) -> 4 resident CTAs/SM.
    # Use the real residency (not the conservative 1) so multiple rows run in
    # parallel (wave scheduling), greatly reducing multi-row latency; the
    # cooperative barrier only requires CTAs within a group to be co-resident,
    # and with num_rows <= 32, total_ctas <= 304 < 4x104, so everything is
    # co-resident with no deadlock risk.
    threads_per_sm = getattr(device_props, "max_threads_per_multi_processor", 2048)
    metax_threads_per_cta = MAX_WARPS * 64
    thread_occupancy = max(1, threads_per_sm // metax_threads_per_cta)
    if use_smem:
        # smem ordered (32KB/CTA) plus histogram overhead limits residency per SM
        smem_per_cta = available_for_ordered * 4 + 4096
        smem_occupancy = max(1, 65536 // smem_per_cta)
        occupancy = min(thread_occupancy, smem_occupancy)
    else:
        occupancy = thread_occupancy

    needs_cooperative = max_seq_len > RADIX_THRESHOLD
    if not needs_cooperative:
        ctas_per_group = 1
    hw_resident_cap = num_sms * occupancy
    max_resident_ctas = hw_resident_cap
    if needs_cooperative:
        headroom = num_sms if occupancy > 1 else 1
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
    histogram_bytes = GLOBAL_RADIX * 3 * 4
    g_state_stride = 4
    radix_row_state_bytes = histogram_bytes + g_state_stride * 4
    assert workspace.size(0) >= num_groups * radix_row_state_bytes
    workspace[: (num_groups * radix_row_state_bytes)] = 0
    g_histogram_size = num_groups * histogram_bytes
    g_state_size = num_groups * g_state_stride * 4
    g_histogram = (
        workspace[:g_histogram_size]
        .view(torch.uint32)
        .view(num_groups, 3, GLOBAL_RADIX)
    )
    g_state = (
        workspace[g_histogram_size : g_histogram_size + g_state_size]
        .view(torch.int32)
        .view(num_groups, g_state_stride)
    )

    # Global scratch: scratch_stride uint32 per CTA (replaces NVIDIA smem)
    scratch = torch.empty(
        total_ctas * scratch_stride,
        dtype=torch.uint32,
        device=logits.device,
    )

    if use_smem:
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
            scratch,
            VEC_SIZE=vec_size,
            BLOCK_SIZE=THREADS_PER_BLOCK,
            SCRATCH_STRIDE=scratch_stride,
            num_warps=MAX_WARPS,
        )
    else:
        persistent_topk_kernel_global[(total_ctas,)](
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
            scratch,
            VEC_SIZE=vec_size,
            BLOCK_SIZE=THREADS_PER_BLOCK,
            SCRATCH_STRIDE=scratch_stride,
            num_warps=MAX_WARPS,
        )
