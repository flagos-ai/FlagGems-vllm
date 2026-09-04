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

"""Sampled-threshold prefill top-K for Moore Threads.

The generic operator reads each row TWICE per refinement step: once to build a
2048-bin histogram so the top-K threshold can be found, and again to compact the
elements at or above it. Measured on S5000 at (64, 129280):

    Pass A  build histogram      75.2 us
    Pass B  re-read and compact 131.8 us
    final select                 10.2 us
    operator                    217.2 us      vLLM 141.6   -> 0.651

Pass A exists only to find a threshold. This estimates that threshold from
1/SSTRIDE of the row instead and spends the saving on a deliberately loose
threshold, so the single remaining pass collects several times top_k candidates
rather than exactly top_k.

Measured by driving the generic _process_bins at a loose threshold:

    rank target   trigger   pass    sample + pass + final   speedup
      2 x top_k    1.6%    135.7            146.8            0.965
      4 x top_k    3.2%    139.0            150.2            0.943
      8 x top_k    6.3%    141.6            152.7            0.927

It pays because compaction barely tracks the trigger rate -- four times the hits
cost 4% more -- so the atomics are per-lane issue overhead, not per-hit traffic,
and trading a looser threshold for a whole scan is close to free.

Two parameters decide whether that theoretical win survives contact with a real
sample, and the first version got both wrong (0.383 against the generic 0.652):
the sample must be large enough that the estimate's error is small compared with
the acceptance window, and the target must sit in the MIDDLE of that window
rather than against its edge. See SSTRIDE and TARGET_RANK below.

Correctness does not depend on the estimate being good. A sample can under- or
over-shoot, so the row's collected count is checked against [TOPK,
NUM_FINAL_ITEMS] and a miss redoes the threshold exactly, from a full histogram,
before compacting again. That fallback costs about what the generic operator
costs, so a bad estimate is slow, never wrong.
"""

import math
import os

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.top_k_per_row_prefill import (
    NUM_BINS,
    NUM_FILNAL_ITEMS,
    NUM_THREADS_PER_BLOCK,
    SORTING_ALGORITHM_THRESHOLD,
    _extract_bin_idx,
    _final_select_radix,
    _num_warps,
    _use_radix_final_for_prefill,
    tle_top_k_per_row_prefill,
)
from flaggems_vllm.ops.top_k_per_row_prefill import (
    top_k_per_row_prefill as _generic_prefill,
)
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


# One element in SSTRIDE feeds the estimate.
#
# 64 was the first choice and it was wrong: it leaves only ~2000 samples, of
# which ~32 land in the tail being estimated, for a relative standard deviation
# near 18%. The collected count then sits 95% inside [1324, 2772] against a
# buffer of 2048, so the retry fired almost every row and the operator ran
# sample + collect + full histogram + collect again -- 357 us predicted against
# 373 measured, worse than the generic two passes it replaced.
#
# 8 gives ~16000 samples and ~256 in the tail, cutting the deviation to ~6%,
# while the sample pass still costs 75.2/8 = 9.4 us against the 75.2 it removes.
SSTRIDE = int(os.environ.get("FLAGGEMS_MTT_PREFILL_SSTRIDE", "8"))

# Below this the sampled path loses. Measured on S5000, generic -> sampled:
#
#     vocab   8193   0.875 -> 0.706   loses
#     vocab  16385   0.765 -> 0.835   wins
#     vocab 129280   0.652 -> 0.869   wins
#
# A short row is dominated by fixed cost, so adding a sample pass and a 2048-bin
# cumsum on top costs more than the scan they remove. The crossover sits between
# 8193 and 16385; 16384 is the safe side of it.
MIN_SPAN = int(os.environ.get("FLAGGEMS_MTT_PREFILL_MIN_SPAN", "16384"))

# Bins used for the SAMPLE histogram only; the collection pass and the exact
# retry still work in the operator's full NUM_BINS space.
#
# MEASURED, and coarsening loses. The threshold scan is a cumsum over this many
# bins, and narrowing it looked like a way to buy the last few us to reach 0.9.
# At (64, 129280) it is catastrophic instead:
#
#     SBINS  2048 -> 0.856    512 -> 0.379    256 -> 0.368
#
# A coarse bin spans NUM_BINS/SAMPLE_BINS fine ones, so the count overshoots by
# up to one coarse bin's population. At top_k=1024 the acceptance window is only
# NFINAL/top_k = 2x wide and the target sits at the 1.1% quantile, deep in the
# tail where one coarse bin is enough to clear the buffer -- so the ~211 us retry
# fires on every row and eats the saving many times over. Same failure mode as
# SSTRIDE=64 had, reached through a different parameter.
#
# The scan was not worth attacking anyway. Timing this kernel at four round
# widths and solving for the model (R rounds of 2048/R cost R*a + 2048*b) gives
# a = 1.5 us per round from three independent points, and a conditional
# single-round variant then pins b, putting the whole 2048-wide scan at ~4.1 us
# -- 2.6% of the operator, not the ~10 us assumed here earlier. Even deleting it
# outright would only reach 0.924.
#
# It does help where the window is wide: at top_k=512 (4x window, 6.25% quantile)
# SBINS=256 gains about 5% once the baseline drift in that 37 us shape is
# discounted. Too small a gain, on too noisy a shape, to justify a rule fitted to
# two points -- so the default stays at the full width.
SAMPLE_BINS = int(os.environ.get("FLAGGEMS_MTT_PREFILL_SBINS", str(NUM_BINS)))


@triton.jit
def _sampled_prefill(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
    SSTRIDE: tl.constexpr,
    TARGET_RANK: tl.constexpr,
    SBINS: tl.constexpr,
    SSHIFT: tl.constexpr,
    NBINS: tl.constexpr,
    NFINAL: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    span = row_end - row_start
    # Base at the row's valid start, so every offset below is already in the
    # caller's convention: indices relative to row_starts[row_id].
    base = logits_ptr + row_id * stride0 + row_start * stride1
    out = out_indices_ptr + row_id * TOPK

    hist = tle.gpu.alloc(
        [NBINS], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    fin = tle.gpu.alloc(
        [NFINAL], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    oidx = tle.gpu.alloc(
        [TOPKP], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    ccnt = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    cfound = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hp = tle.gpu.local_ptr(hist, (0,))
    fp = tle.gpu.local_ptr(fin, (0,))
    op = tle.gpu.local_ptr(oidx, (0,))
    cp = tle.gpu.local_ptr(ccnt, (0,))
    fvp = tle.gpu.local_ptr(cfound, (0,))

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC)
    bins = tl.arange(0, NBINS)
    one1 = tl.full([BLOCK_SIZE], 1, tl.int32)
    one2 = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)

    # ---- pass 1: histogram of every SSTRIDE-th element -------------------
    # Only the first SBINS entries are used here, so only those need clearing.
    for z in tl.range(0, SBINS, BLOCK_SIZE):
        tl.store(hp + z + lane, 0)
    tl.debug_barrier()

    n_s = span // SSTRIDE
    for t in tl.range(0, tl.cdiv(n_s, BLOCK_SIZE)):
        i = (t * BLOCK_SIZE + lane) * SSTRIDE
        m = i < span
        b, _ = _extract_bin_idx(tl.load(base + i * stride1, mask=m, other=0.0),
                                m, 0, STEP=0)
        tl.atomic_add(hp + (b >> SSHIFT), one1, mask=m, sem="relaxed",
                      scope="cta")
    tl.debug_barrier()

    # A lower bin is a larger value, so the prefix count over bins is the count
    # of the largest elements. TARGET_RANK aims at the GEOMETRIC MIDPOINT of the
    # acceptance window [TOPK, NFINAL] rather than its upper edge, which is what
    # the first version did -- aiming at the edge meant half the sampling error
    # pushed the count straight out of the window and into the retry.
    # One wide scan, deliberately, where the generic operator loops in
    # BLOCK_SIZE-wide rounds with tle.cumsum, a carried total and an early exit.
    # That structure is not free: each round also pays a carry add, a threshold
    # mask, two masked stores and a reduce_or. The generic operator needs all of
    # it, because every round must yield threshold_bin_idx to prefix the next
    # refinement step and final_bin_size to decide whether to take one -- and it
    # splits elements three ways. This kernel needs a single cut and nothing
    # else, so the bookkeeping has no reader and the scan can be one shot.
    #
    # Measured, transcribing the generic loop into this kernel verbatim
    # (tools/mtt_scan_rounds.py), against this one wide scan:
    #
    #                    one 2048   tle x512   tle x1024
    #     (64,129280)      157.1      156.8      157.6
    #     (4,16385)         38.2       43.0       40.5
    #     (16,65536)        63.7       65.2       65.6
    #
    # tle.cumsum is the right primitive for rounds -- it returns
    # (prefix, total), and it beats tl.cumsum rounds by 3-6% on two of the three
    # shapes -- but rounds themselves lose here whichever primitive runs them.
    #
    # Cross-checked by wiring the generic loop into this operator for real and
    # running the acceptance benchmark: (64, 129280) came out 0.903 against the
    # probe's predicted 0.903 and this kernel's 0.902, the five run-stable
    # shapes moved by at most 0.4% and their geomean by 0.001, and correctness
    # held at 20/20. Two independent timing frameworks, one answer. The change
    # is not worth taking, and the reproducer is the `scan-tle-experiment`
    # branch.
    sbins = tl.arange(0, SBINS)
    cum = tl.cumsum(tl.load(hp + sbins), axis=0)
    target = TARGET_RANK // SSTRIDE + 1
    thr_c = tl.min(tl.where(cum >= target, sbins, SBINS - 1), axis=0)
    # Map back to the full bin space taking the WHOLE boundary coarse bin: at
    # SSHIFT=0 this is the exact threshold, and coarser settings deliberately
    # over-collect rather than under-collect, because falling short of TOPK
    # forces the expensive exact retry while overshooting only wastes buffer.
    thr = (thr_c + 1) << SSHIFT

    # ---- pass 2: collect everything below the threshold -------------------
    # Two attempts. The first uses the sampled threshold; if the count lands
    # outside [TOPK, NFINAL] the estimate was bad, and the retry derives the
    # threshold exactly from a full histogram, which is what the generic
    # operator does anyway.
    for attempt in tl.static_range(0, 2):
        redo = attempt == 1
        if (attempt == 0) or (tl.load(cp) < TOPK) or (tl.load(cp) > NFINAL):
            if redo:
                for z in tl.range(0, NBINS, BLOCK_SIZE):
                    tl.store(hp + z + lane, 0)
                tl.debug_barrier()
                for t in tl.range(0, tl.cdiv(span, BLOCK_SIZE)):
                    i = t * BLOCK_SIZE + lane
                    m = i < span
                    b, _ = _extract_bin_idx(
                        tl.load(base + i * stride1, mask=m, other=0.0), m, 0,
                        STEP=0,
                    )
                    tl.atomic_add(hp + b, one1, mask=m, sem="relaxed",
                                  scope="cta")
                tl.debug_barrier()
                cum2 = tl.cumsum(tl.load(hp + bins), axis=0)
                thr = tl.min(tl.where(cum2 >= TOPK, bins, NBINS - 1), axis=0) + 1

            # hist doubles as the candidate index buffer from here on
            for z in tl.range(0, NBINS, BLOCK_SIZE):
                tl.store(hp + z + lane, 0)
            tl.store(cp, 0)
            tl.store(fvp, 0)
            tl.debug_barrier()

            n_vec = span // (BLOCK_SIZE * VEC)
            for t in tl.range(0, n_vec):
                offs = (t * BLOCK_SIZE * VEC + lane * VEC)[:, None] + vec[None, :]
                x = tl.load(base + offs * stride1)
                b, _ = _extract_bin_idx(x, True, 0, STEP=0)
                # Cast explicitly: b is uint32 and thr int32, and leaving that
                # promotion implicit is what silently selected every element in
                # an earlier version of this kernel.
                take = b.to(tl.int32) < thr
                pos = tl.atomic_add(cp + tl.zeros([BLOCK_SIZE, VEC], tl.int32),
                                    one2, mask=take, sem="relaxed", scope="cta")
                keep = take & (pos < NFINAL)
                tl.store(hp + pos, offs.to(tl.int32), mask=keep)
            tail = n_vec * BLOCK_SIZE * VEC
            for t in tl.range(0, tl.cdiv(span - tail, BLOCK_SIZE)):
                i = tail + t * BLOCK_SIZE + lane
                m = i < span
                x = tl.load(base + i * stride1, mask=m, other=0.0)
                b, _ = _extract_bin_idx(x, m, 0, STEP=0)
                take = m & (b.to(tl.int32) < thr)
                pos = tl.atomic_add(cp + tl.zeros([BLOCK_SIZE], tl.int32),
                                    one1, mask=take, sem="relaxed", scope="cta")
                keep = take & (pos < NFINAL)
                tl.store(hp + pos, i.to(tl.int32), mask=keep)
            tl.debug_barrier()

    # ---- re-read the candidate values -------------------------------------
    # The collection loop keeps only the INDEX. Its two scattered shared-memory
    # stores per hit were the largest remaining cost; re-reading the values
    # afterwards, from global, in one short fully parallel pass is cheaper.
    #
    # Measured on S5000 at (64, 129280), kernel truncated after collection so
    # the three costs separate (tools/mtt_store_split.py in the working branch):
    #
    #     store value + index   150.5 us
    #     store index only      141.9        <- this
    #     store neither         119.8        <- the two stores cost 30.7
    #
    # Dropping the value store returns about half of that 30.7 and the gather
    # spends ~7 of it back, for a net 8.6. Whole operator 166.0 -> 157.2 us
    # against vLLM's 142.4, i.e. 0.858 -> 0.906, confirmed independently by the
    # benchmark at 0.906.
    #
    # A third variant, replacing the atomic with a deterministic pos, came out
    # 18 us SLOWER and bounds nothing: modulo positions are scattered duplicate
    # addresses, while the atomic's are dense, so it measured a different store
    # pattern rather than the absence of an atomic. Recorded so the experiment
    # is not repeated.
    c_have = tl.minimum(tl.load(cp), NFINAL)
    for t in tl.range(0, tl.cdiv(NFINAL, BLOCK_SIZE)):
        j = t * BLOCK_SIZE + lane
        m = j < c_have
        gi = tl.load(hp + j, mask=m, other=0)
        tl.store(fp + j, tl.load(base + gi * stride1, mask=m, other=0.0), mask=m)
    tl.debug_barrier()

    # ---- select TOPK out of the candidates --------------------------------
    _final_select_radix(
        hp, fp, cp, fvp, op, None,
        TOPK=TOPK, BLOCK_SIZE=BLOCK_SIZE, MULTIPLE_BLOCKS_PER_ROW=False,
    )
    tl.debug_barrier()

    n_have = tl.minimum(tl.load(cp), TOPK)
    for z in tl.range(0, TOPK, BLOCK_SIZE):
        o = z + lane
        m = o < TOPK
        v = tl.load(op + o, mask=m & (o < n_have), other=-1)
        tl.store(out + o, tl.where(o < n_have, v, -1), mask=m)


# A wider block buys warps per SM, which is what a small num_rows cannot get from
# the grid: prefill launches grid=(num_rows,), so a handful of rows leaves most of
# the device idle however much work each row carries.
#
# Measured here, and the crossover lands exactly on the SM count with no
# misclassification across ten points at two vocabularies:
#
#     rows        2     4     8    16    32    48    60  |    61    64   128
#     v=8193   1.08  1.14  1.13  1.14  1.21  1.14  1.13  |  0.68  0.65  0.52
#     v=129280 1.50  1.57  1.53  1.52  1.52  1.52  1.48  |  0.83  0.82  0.63
#
# The mechanism fixes the boundary: the wide block takes 32 warps and this SM
# tops out at 32, so it holds ONE program per SM where the narrow block holds
# two, and past the SM count the grid needs a second wave. That 32->64 step costs
# exactly 1.91x, and the independently measured concurrent capacity at
# BLOCK_SIZE=512 was ~120, i.e. the same 32-warp ceiling seen from the other side.
#
# This lives in the vendor override rather than the generic dispatcher on
# purpose. Both the threshold and the reasoning behind it come from this part's
# occupancy structure, and whether they carry to a device with a different warp
# ceiling is unmeasured.
_WIDE_BLOCK = 1024
_WIDE_MAX_ROWS = None


def _wide_max_rows():
    """Largest num_rows for which the wide block still fits in one wave."""
    global _WIDE_MAX_ROWS
    if _WIDE_MAX_ROWS is None:
        override = os.environ.get("FLAGGEMS_MTT_PREFILL_WIDE_MAX_ROWS")
        if override is not None:
            _WIDE_MAX_ROWS = int(override)
        else:
            try:
                props = runtime.torch_device_fn.get_device_properties(0)
                warp = getattr(props, "warp_size", 32) or 32
                sm = getattr(props, "multi_processor_count", 0)
                # num_warps here is BLOCK_SIZE // 32, so a part whose warp is not
                # 32 lanes would ask for a different thread count than the one
                # this crossover was measured at.
                _WIDE_MAX_ROWS = sm if (warp == 32 and sm) else 0
            except Exception:  # noqa: BLE001 - detection must not break dispatch
                _WIDE_MAX_ROWS = 0
    return _WIDE_MAX_ROWS


def _generic_at_block(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k, block
):
    """The generic TLE dispatch, at a chosen BLOCK_SIZE.

    Mirrors top_k_per_row_prefill's own TLE branch, including its split between
    insertion-sort and radix-final rows, and reuses its kernel unchanged.
    """
    vocab_size = logits.shape[1]
    topkp = triton.next_power_of_2(top_k)
    use_radix_final = _use_radix_final_for_prefill(vocab_size)
    n_insert = 0 if use_radix_final else min(num_rows, SORTING_ALGORITHM_THRESHOLD)
    nw = _num_warps(block)

    if n_insert > 0:
        tle_top_k_per_row_prefill[(n_insert,)](
            logits, indices, row_starts, row_ends, stride0, stride1, vocab_size,
            TOPK=top_k, TOPKP=topkp, BLOCK_SIZE=block,
            USE_RADIX_FINAL=False, ROW_OFFSET=0, num_warps=nw,
        )
    if num_rows > n_insert:
        tle_top_k_per_row_prefill[(num_rows - n_insert,)](
            logits, indices, row_starts, row_ends, stride0, stride1, vocab_size,
            TOPK=top_k, TOPKP=topkp, BLOCK_SIZE=block,
            USE_RADIX_FINAL=True, ROW_OFFSET=n_insert, num_warps=nw,
        )


def _can_sample(num_rows, vocab_size, stride1, top_k):
    """Route to the sampled kernel only where it was measured to pay."""
    if not HAS_TLE:
        return False
    if stride1 != 1:
        return False
    # The candidate buffer is the generic op's, so the margin has to fit in it.
    if top_k <= 0 or NUM_FILNAL_ITEMS // top_k < 2:
        return False
    return vocab_size >= MIN_SPAN


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for DeepSeek V4 prefill, with a sampled threshold.

    Falls back by *calling* the generic implementation, not by claiming to.
    """
    vocab_size = logits.shape[1]
    if not _can_sample(num_rows, vocab_size, stride1, top_k):
        # Short rows skip sampling, but a small num_rows can still be starved of
        # warps, and widening the block is the fix for that.
        if HAS_TLE and 0 < num_rows <= _wide_max_rows():
            return _generic_at_block(
                logits, row_starts, row_ends, indices, num_rows, stride0,
                stride1, top_k, _WIDE_BLOCK,
            )
        return _generic_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )

    # Midpoint of [top_k, NUM_FILNAL_ITEMS] in log space: maximum room on both
    # sides for the sample's error, instead of hugging one edge.
    target_rank = int(math.sqrt(top_k * NUM_FILNAL_ITEMS))
    # One row per program, so a grid smaller than the device leaves SMs idle and
    # the kernel is latency-bound rather than bandwidth-bound: at 4 rows x 129280
    # it moves ~18 GB/s against the 645 this card reaches on the same access
    # pattern. Widening puts twice the threads on each row.
    #
    # Same gate as the non-sampled path above, which the sampled path simply
    # never called. Measured wide/narrow ratio at span 129280, top_k 1024:
    #
    #     rows      4     32     60     64     96
    #           0.718  0.733  0.794  1.246  1.138
    #
    # The cliff is exactly at the SM count -- 1024 threads is 32 warps against a
    # 32-warp SM ceiling, so capacity falls to one program per SM and 64 rows
    # needs a second wave. Across span at 32 rows the gain grows monotonically,
    # 0.966 at 16384 to 0.737 at 129280, and never inverts, so no span condition
    # is added: the sampled path already requires span >= MIN_SPAN, and 16384 is
    # its weakest point.
    #
    # This moves no benchmark shape -- (4, 16385) is below the span where
    # widening matters and (64, 129280) is four rows past the cliff -- but is
    # worth 1.24x to 1.42x against vLLM for num_rows 4..60 at large
    # vocabularies, which core_shapes.yaml does not sample.
    block = _WIDE_BLOCK if 0 < num_rows <= _wide_max_rows() else NUM_THREADS_PER_BLOCK
    _sampled_prefill[(num_rows,)](
        logits,
        indices,
        row_starts,
        row_ends,
        stride0,
        stride1,
        TOPK=top_k,
        TOPKP=triton.next_power_of_2(top_k),
        BLOCK_SIZE=block,
        VEC=4,
        SSTRIDE=SSTRIDE,
        TARGET_RANK=target_rank,
        SBINS=SAMPLE_BINS,
        SSHIFT=(NUM_BINS // SAMPLE_BINS).bit_length() - 1,
        NBINS=NUM_BINS,
        NFINAL=NUM_FILNAL_ITEMS,
        num_warps=_num_warps(block),
    )
