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


# Sample every SSTRIDE-th element for the threshold estimate. 8 leaves ~256
# samples in the top-k tail (~6% error); 64 left ~32 and missed the
# [top_k, 2048] window on almost every row.
SSTRIDE = int(os.environ.get("FLAGGEMS_MTT_PREFILL_SSTRIDE", "8"))

# Shorter rows are fixed-cost dominated and lose with sampling (S5000: vocab
# 8193 0.875 -> 0.706, vocab 16385 0.765 -> 0.835).
MIN_SPAN = int(os.environ.get("FLAGGEMS_MTT_PREFILL_MIN_SPAN", "16384"))


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
        [NBINS],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    fin = tle.gpu.alloc(
        [NFINAL],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    oidx = tle.gpu.alloc(
        [TOPKP],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    ccnt = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    cfound = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
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
    for z in tl.range(0, NBINS, BLOCK_SIZE):
        tl.store(hp + z + lane, 0)
    tl.debug_barrier()

    n_s = span // SSTRIDE
    for t in tl.range(0, tl.cdiv(n_s, BLOCK_SIZE)):
        i = (t * BLOCK_SIZE + lane) * SSTRIDE
        m = i < span
        b, _ = _extract_bin_idx(
            tl.load(base + i * stride1, mask=m, other=0.0), m, 0, STEP=0
        )
        tl.atomic_add(hp + b, one1, mask=m, sem="relaxed", scope="cta")
    tl.debug_barrier()

    # Lower bins hold larger values, so the prefix sum counts the largest
    # elements. One wide scan instead of the generic op's tle.cumsum rounds:
    # a single cut has no use for the per-round bookkeeping.
    cum = tl.cumsum(tl.load(hp + bins), axis=0)
    target = TARGET_RANK // SSTRIDE + 1
    thr_c = tl.min(tl.where(cum >= target, bins, NBINS - 1), axis=0)
    thr = thr_c + 1

    # ---- pass 2: collect everything below the threshold -------------------
    # A count outside [TOPK, NFINAL] means a bad estimate; the retry recomputes
    # the threshold exactly from a full histogram.
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
                        tl.load(base + i * stride1, mask=m, other=0.0),
                        m,
                        0,
                        STEP=0,
                    )
                    tl.atomic_add(hp + b, one1, mask=m, sem="relaxed", scope="cta")
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
                # Explicit cast: implicit uint32/int32 promotion selects everything.
                take = b.to(tl.int32) < thr
                pos = tl.atomic_add(
                    cp + tl.zeros([BLOCK_SIZE, VEC], tl.int32),
                    one2,
                    mask=take,
                    sem="relaxed",
                    scope="cta",
                )
                keep = take & (pos < NFINAL)
                tl.store(hp + pos, offs.to(tl.int32), mask=keep)
            tail = n_vec * BLOCK_SIZE * VEC
            for t in tl.range(0, tl.cdiv(span - tail, BLOCK_SIZE)):
                i = tail + t * BLOCK_SIZE + lane
                m = i < span
                x = tl.load(base + i * stride1, mask=m, other=0.0)
                b, _ = _extract_bin_idx(x, m, 0, STEP=0)
                take = m & (b.to(tl.int32) < thr)
                pos = tl.atomic_add(
                    cp + tl.zeros([BLOCK_SIZE], tl.int32),
                    one1,
                    mask=take,
                    sem="relaxed",
                    scope="cta",
                )
                keep = take & (pos < NFINAL)
                tl.store(hp + pos, i.to(tl.int32), mask=keep)
            tl.debug_barrier()

    # ---- re-read the candidate values -------------------------------------
    # Collection keeps only the index; one read-back pass here is cheaper than
    # a second scattered smem store per hit (0.858 -> 0.906 at (64, 129280)).
    c_have = tl.minimum(tl.load(cp), NFINAL)
    for t in tl.range(0, tl.cdiv(NFINAL, BLOCK_SIZE)):
        j = t * BLOCK_SIZE + lane
        m = j < c_have
        gi = tl.load(hp + j, mask=m, other=0)
        tl.store(fp + j, tl.load(base + gi * stride1, mask=m, other=0.0), mask=m)
    tl.debug_barrier()

    # ---- select TOPK out of the candidates --------------------------------
    _final_select_radix(
        hp,
        fp,
        cp,
        fvp,
        op,
        None,
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
        MULTIPLE_BLOCKS_PER_ROW=False,
    )
    tl.debug_barrier()

    n_have = tl.minimum(tl.load(cp), TOPK)
    for z in tl.range(0, TOPK, BLOCK_SIZE):
        o = z + lane
        m = o < TOPK
        v = tl.load(op + o, mask=m & (o < n_have), other=-1)
        tl.store(out + o, tl.where(o < n_have, v, -1), mask=m)


# Prefill launches one program per row, so few rows leave SMs idle. A 1024-wide
# block is 32 warps, the SM ceiling: it doubles threads per row but halves SM
# capacity, so widen only while num_rows <= SM count (S5000: wins at 60 rows,
# loses at 61).
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
                # The crossover was measured on 32-lane warps only.
                _WIDE_MAX_ROWS = sm if (warp == 32 and sm) else 0
            except Exception:  # noqa: BLE001 - detection must not break dispatch
                _WIDE_MAX_ROWS = 0
    return _WIDE_MAX_ROWS


def _generic_at_block(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k, block
):
    """Generic TLE dispatch at a chosen BLOCK_SIZE."""
    vocab_size = logits.shape[1]
    topkp = triton.next_power_of_2(top_k)
    use_radix_final = _use_radix_final_for_prefill(vocab_size)
    n_insert = 0 if use_radix_final else min(num_rows, SORTING_ALGORITHM_THRESHOLD)
    nw = _num_warps(block)

    if n_insert > 0:
        tle_top_k_per_row_prefill[(n_insert,)](
            logits,
            indices,
            row_starts,
            row_ends,
            stride0,
            stride1,
            vocab_size,
            TOPK=top_k,
            TOPKP=topkp,
            BLOCK_SIZE=block,
            USE_RADIX_FINAL=False,
            ROW_OFFSET=0,
            num_warps=nw,
        )
    if num_rows > n_insert:
        tle_top_k_per_row_prefill[(num_rows - n_insert,)](
            logits,
            indices,
            row_starts,
            row_ends,
            stride0,
            stride1,
            vocab_size,
            TOPK=top_k,
            TOPKP=topkp,
            BLOCK_SIZE=block,
            USE_RADIX_FINAL=True,
            ROW_OFFSET=n_insert,
            num_warps=nw,
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
    """Top-K per row for DeepSeek V4 prefill with a sampled threshold."""
    vocab_size = logits.shape[1]
    if not _can_sample(num_rows, vocab_size, stride1, top_k):
        # Rows too short to sample can still use the wide block.
        if HAS_TLE and 0 < num_rows <= _wide_max_rows():
            return _generic_at_block(
                logits,
                row_starts,
                row_ends,
                indices,
                num_rows,
                stride0,
                stride1,
                top_k,
                _WIDE_BLOCK,
            )
        return _generic_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )

    # Midpoint of [top_k, NUM_FILNAL_ITEMS] in log space: maximum room on both
    # sides for the sample's error, instead of hugging one edge.
    target_rank = int(math.sqrt(top_k * NUM_FILNAL_ITEMS))
    # Same wide-block gate as the non-sampled path; see _WIDE_BLOCK.
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
        NBINS=NUM_BINS,
        NFINAL=NUM_FILNAL_ITEMS,
        num_warps=_num_warps(block),
    )
