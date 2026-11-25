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

"""Triton top_k_per_row_decode for DeepSeek V4 decode-phase topk selection.

GCU (enflame) rewrite of the FlagTree radix-select implementation, itself
restructured for the measured GCU cost model (probe-verified on gcu410,
triton 3.6.0 + flagtree 0.7.0):

* a single CTA streams a 262144-float row at only ~16.7 GB/s (BLOCK=8192
  masked loads), so the previous "one CTA scans the whole row once per
  bit-probe" search needed 33 x 1 MB = 2.3 ms per row;
* ``tl.sum`` reductions serialize (independent sums do not pipeline, probe
  2026-11-11), each costing ~1.8 us of latency, so the search must keep its
  dependent-probe chain on a *register-resident* tile, not reload the row;
* the aggregate read bandwidth saturates near ~490 GB/s, so wide per-row
  grids do not help a row-serial algorithm.

The rewrite is a two-level top-K-of-top-K pipeline (4 kernels, no atomics,
no histogram, deterministic):

* ``_topk_chunk_threshold_kernel`` (num_warps=8) splits each row into
  CHUNK-element tiles, one CTA per (row, chunk). Each CTA loads its chunk
  once into registers and runs a ternary value search — two pivots counted
  by a single packed ``tl.sum`` per round, the key interval narrowing to a
  third — for the chunk-local TOPK-th largest monotone key. The search
  exits early once a pivot's count hits TOPK exactly: then the elements
  above the pivot are precisely the chunk's top-K, so the selected set is
  final (worst case still bounded by the round cap, e.g. all-equal rows).
* ``_topk_chunk_select_kernel`` (num_warps=1, where tl.cumsum compiles)
  re-scans the chunk and stably compacts the chunk's top-K candidates —
  elements with key >= local threshold, ascending original index, capped at
  TOPK — into dense (key, index) slots, padded with key 0 / index -1.
* ``_topk_merge_threshold_kernel`` (num_warps=8) runs the same bit search
  over the num_chunks*TOPK candidates (one register-resident tile up to
  16384 wide — a [16384] masked load + tl.sum is correct and cheaper per
  dependent probe than two [8192] sums on this stack — reload loop
  otherwise) to find the global TOPK-th largest key T.
* ``_topk_merge_select_kernel`` (num_warps=1) selects the final answer from
  the candidates: strict-greater-than-T keys keep stable cumsum positions,
  threshold-equal keys fill the remaining slots in ascending original index,
  matching the stable-argsort reference on ties and staying bit-identical
  across repeated invocations.

Because the global top-K of a row is always contained in the union of the
per-chunk stable top-Ks, and threshold-equal candidates are emitted in
globally ascending original-index order (chunk-major, ascending within each
chunk), the selected index set equals the stable-argsort reference.

Measured do_bench (kernel mode, warmup 1000ms/iter 100ms), 2026-11-17 round
vs the 2026-11-11 4-kernel rewrite on gcu410: all 11 core shapes -29..-32%
(rows=1 173.4 -> 123.2 us, rows=496 24.71 -> 17.40 ms), via the ternary
packed-probe search (K1, ~19 effective rounds vs 28 greedy probes), the
single-tile merge search ([16384] load + tl.sum is correct on this stack),
wide-chunk dispatch for large grids, and packed (gt, eq) count
accumulation in the cumsum compaction loops.
"""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)

# Per-chunk scan width for the threshold search (num_warps=8). 8192 for
# small grids (the knowledge-sanctioned masked-load + tl.sum width; the
# per-CTA dependent-probe chain is the wall there, so narrower chunks with
# more CTAs win). 16384 once the grid is large enough to saturate the
# device (>= ~2 waves): halves the CTA count at a mildly wider per-round
# packed sum (2.52us vs 2.08us, probed), a net ~25% K1 win — the chain is
# shared by more elements. [16384] load + tl.sum is correct on this stack
# (probed 2026-11-17; the old >=16384 corruption precedent predates it).
CHUNK_NARROW = 8192
CHUNK_WIDE = 16384
# Switch to the wide chunk once the narrow grid would run >= ~2 waves
# (device holds ~48-52 concurrent CTAs).
WIDE_CHUNK_MIN_CTAS = 96
# Compaction / final-select scan width (num_warps=1): tl.cumsum compiles and
# is bit-correct up to 2048 lanes on this stack; 4096 fails with the
# memref_ext.slice_start properties error. Wider cumsum tiles were probed
# (2026-11-17): cost scales linearly with lanes, so widening is a loss.
SELECT_BLOCK = 2048
# Candidate-merge scan width (num_warps=8); single resident tile cap.
MERGE_BLOCK = 8192


@triton.jit
def _convert_to_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    sign_set = (bits & sign_mask) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFFFFFF, tl.uint32)
    return tl.where(sign_set, bits, inv)


@triton.jit
def _rank_key(x):
    # Ascending ranking key: larger float => larger key, so the top-k values
    # are exactly the k largest keys. This is the bitwise complement of the
    # seed's descending monotone map (_convert_to_uint32).
    return ~_convert_to_uint32(x)


@triton.jit
def _row_len(seq_lens_ptr, next_n, row_id, vocab):
    batch_id = row_id // next_n
    batch_offset = row_id % next_n
    seq_len = tl.load(seq_lens_ptr + batch_id)
    # Guard against malformed seq_lens: the valid prefix never exceeds the
    # logical row width (reading past it would leave the allocation).
    return tl.minimum(seq_len - next_n + batch_offset + 1, vocab)


@libentry()
@triton.jit
def _topk_chunk_threshold_kernel(
    logits_ptr,
    seq_lens_ptr,
    next_n,
    stride0,
    stride1,
    thr_ptr,
    gtc_ptr,
    vocab,
    n_chunk,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = pid // n_chunk
    chunk = pid % n_chunk
    row_len = _row_len(seq_lens_ptr, next_n, row_id, vocab)
    thr_ptr += row_id * n_chunk + chunk
    gtc_ptr += row_id * n_chunk + chunk
    start = chunk * CHUNK
    if (row_len <= TOPK) | (start >= row_len):
        tl.store(thr_ptr, 0)
        tl.store(gtc_ptr, 0)
        return

    lane = tl.arange(0, CHUNK)
    offs = start + lane
    in_range = offs < row_len
    x = tl.load(logits_ptr + row_id * stride0 + offs * stride1, mask=in_range)
    key = _rank_key(x)
    # Every search pivot is >= 1 (third = max(w // 3, 1), lo >= 0), so
    # mapping out-of-range lanes to key 0 excludes them from every count
    # exactly like the previous per-probe `& in_range`.
    key = tl.where(in_range, key, 0)

    # Ternary value search for the largest monotone key T with
    # count(key >= T) >= TOPK — the chunk-local TOPK-th largest key.
    # One packed tl.sum per round counts two independent pivots
    # (cnt1 + cnt2 * (2*CHUNK); each count <= CHUNK < 2*CHUNK, so the
    # fields never collide — a 2^14 scale with CHUNK=2^14 overflows at
    # cnt == CHUNK on all-equal/banded rows), narrowing the key interval
    # to a third: ~21 effective rounds vs ~28 dependent probes for
    # the one-bit-per-probe greedy form on this data.
    # Exits (all data-adaptive, worst case bounded by the 26-round cap):
    # * exact hit cnt == TOPK at a pivot p: the elements > p are then
    #   exactly TOPK, so the selected set is final with threshold p — the
    #   select kernel's budget logic lands the identical set whether or
    #   not p equals the bit-exact TOPK-th key;
    # * interval closure lo == hi: T = lo by the loop invariant
    #   (cnt(>= lo) >= TOPK and every probe above hi has been rejected).
    SCALE: tl.constexpr = 2 * CHUNK
    one = tl.full((), 1, tl.uint32)
    lo = tl.zeros((), dtype=tl.uint32)
    hi = tl.full((), 0xFFFFFFFF, tl.uint32)
    threshold = tl.zeros((), dtype=tl.uint32)
    done = tl.zeros((), dtype=tl.int32)
    for i in tl.range(0, 26):
        if done == 0:
            w = hi - lo
            third = tl.maximum(w // 3, one)
            p1 = lo + third
            p2 = tl.minimum(p1 + third, hi)
            packed = tl.sum(
                (key >= p1).to(tl.int32) + (key >= p2).to(tl.int32) * SCALE
            )
            c2 = packed // SCALE
            c1 = packed - c2 * SCALE
            in2 = c2 >= TOPK
            in1 = c1 >= TOPK
            new_lo = tl.where(in2, p2, tl.where(in1, p1, lo))
            new_hi = tl.where(in2, hi, tl.where(in1, p2 - 1, p1 - 1))
            exact = (c1 == TOPK) | (c2 == TOPK)
            closed = new_lo == new_hi
            lo = new_lo
            hi = new_hi
            threshold = tl.where(
                exact | closed,
                tl.where(c2 == TOPK, p2, tl.where(c1 == TOPK, p1, new_lo)),
                threshold,
            )
            done = (exact | closed).to(tl.int32)
    # Strictly-greater count: lets the select kernel place threshold-equal
    # elements behind the strict-greater ones without a second scan.
    gt_cnt = tl.sum(((key > threshold) & in_range).to(tl.int32))
    tl.store(thr_ptr, threshold.to(tl.int32, bitcast=True))
    tl.store(gtc_ptr, gt_cnt)


@libentry()
@triton.jit
def _topk_chunk_select_kernel(
    logits_ptr,
    seq_lens_ptr,
    next_n,
    stride0,
    stride1,
    thr_ptr,
    gtc_ptr,
    cand_key_ptr,
    cand_idx_ptr,
    vocab,
    n_chunk,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = pid // n_chunk
    chunk = pid % n_chunk
    row_len = _row_len(seq_lens_ptr, next_n, row_id, vocab)
    slot = row_id * (n_chunk * TOPK) + chunk * TOPK
    cand_key_ptr += slot
    cand_idx_ptr += slot
    lane = tl.arange(0, BLOCK)
    pad_chunks: tl.constexpr = (TOPK + BLOCK - 1) // BLOCK

    empty = (row_len <= TOPK) | (chunk * CHUNK >= row_len)
    if empty:
        for c in tl.range(0, pad_chunks):
            pos = c * BLOCK + lane
            pm = pos < TOPK
            tl.store(cand_key_ptr + pos, 0, mask=pm)
            tl.store(cand_idx_ptr + pos, -1, mask=pm)
        return

    thr = tl.load(thr_ptr + row_id * n_chunk + chunk).to(tl.uint32, bitcast=True)
    gt_total = tl.load(gtc_ptr + row_id * n_chunk + chunk)
    # Tie-aware compaction: keep every strict-greater element plus the
    # (TOPK - gt_total) lowest-original-index threshold-equal elements, and
    # write them in ascending original-index order — the merge-select kernel
    # relies on this order to break global ties by lowest original index.
    budget = TOPK - gt_total
    gt_before = tl.zeros((), dtype=tl.int32)
    eq_before = tl.zeros((), dtype=tl.int32)
    tiles: tl.constexpr = CHUNK // BLOCK
    ACC_SCALE: tl.constexpr = 16384  # per-tile counts < 2048 < 2^14
    for t in tl.range(0, tiles):
        offs = chunk * CHUNK + t * BLOCK + lane
        in_range = offs < row_len
        x = tl.load(logits_ptr + row_id * stride0 + offs * stride1, mask=in_range)
        key = _rank_key(x)
        gt = in_range & (key > thr)
        eq = in_range & (key == thr)
        c_gt = tl.cumsum(gt.to(tl.int32), axis=0)
        c_eq = tl.cumsum(eq.to(tl.int32), axis=0)
        eq_rank = eq_before + c_eq - 1
        # Position = (strict-greater candidates before) + (kept equals
        # before); kept equals are the lowest-index ones, so their count is
        # capped by the tie budget. Slots past the kept set are padded with
        # key 0 (minimum monotone key) / index -1 so the candidate array is
        # always fully defined.
        pos = tl.where(
            gt,
            gt_before + c_gt - 1 + tl.minimum(eq_before + c_eq, budget),
            gt_before + c_gt + tl.minimum(tl.maximum(eq_rank, 0), budget),
        )
        keep = gt | (eq & (eq_rank < budget))
        tl.store(cand_key_ptr + pos, key.to(tl.int32, bitcast=True), mask=keep)
        tl.store(cand_idx_ptr + pos, offs.to(tl.int32), mask=keep)
        # One packed sum carries both per-tile totals (independent tl.sum
        # calls serialize on this backend; +cnt_eq * 2^14 unpacks exactly).
        packed = tl.sum(gt.to(tl.int32) + eq.to(tl.int32) * ACC_SCALE)
        gt_before += packed & 0x3FFF
        eq_before += packed >> 14
    total = gt_before + tl.minimum(eq_before, budget)
    for c in tl.range(0, pad_chunks):
        pos = c * BLOCK + lane
        pm = (pos >= total) & (pos < TOPK)
        tl.store(cand_key_ptr + pos, 0, mask=pm)
        tl.store(cand_idx_ptr + pos, -1, mask=pm)


@libentry()
@triton.jit
def _topk_merge_threshold_kernel(
    cand_key_ptr,
    seq_lens_ptr,
    next_n,
    thr_ptr,
    vocab,
    n_cand,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    RESIDENT: tl.constexpr,
    EVEN: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_len = _row_len(seq_lens_ptr, next_n, row_id, vocab)
    thr_ptr += row_id
    if row_len <= TOPK:
        tl.store(thr_ptr, 0)
        return

    lane = tl.arange(0, BLOCK)
    base = row_id * n_cand
    one = tl.full((), 1, tl.uint32)
    # Same greedy MSB-first search over the per-chunk candidates; padded
    # slots carry key 0 and never count for any candidate >= 1. The
    # candidate set fits one register-resident tile on this stack (probed:
    # [16384] masked/unmasked load + tl.sum correct, 2.18us chain step vs
    # 2x1.84us for the previous two-tile form), so the search runs 1 sum
    # per probe instead of 2.
    threshold = tl.zeros((), dtype=tl.uint32)
    done = tl.zeros((), dtype=tl.int32)
    if RESIDENT:
        if EVEN:
            k0 = tl.load(cand_key_ptr + base + lane).to(tl.uint32, bitcast=True)
            for i in tl.range(0, 32):
                if done == 0:
                    bit = 31 - i
                    cand = threshold | (one << bit)
                    cnt = tl.sum((k0 >= cand).to(tl.int32))
                    threshold = tl.where(cnt >= TOPK, cand, threshold)
                    done = (cnt == TOPK).to(tl.int32)
        else:
            m = lane < n_cand
            k0 = tl.load(cand_key_ptr + base + lane, mask=m).to(
                tl.uint32, bitcast=True
            )
            for i in tl.range(0, 32):
                if done == 0:
                    bit = 31 - i
                    cand = threshold | (one << bit)
                    cnt = tl.sum(((k0 >= cand) & m).to(tl.int32))
                    threshold = tl.where(cnt >= TOPK, cand, threshold)
                    done = (cnt == TOPK).to(tl.int32)
    else:
        for i in tl.range(0, 32):
            if done == 0:
                bit = 31 - i
                cand = threshold | (one << bit)
                cnt = tl.zeros((), dtype=tl.int32)
                for t in tl.range(0, tl.cdiv(n_cand, BLOCK)):
                    offs = t * BLOCK + lane
                    in_range = offs < n_cand
                    k = tl.load(cand_key_ptr + base + offs, mask=in_range).to(
                        tl.uint32, bitcast=True
                    )
                    cnt += tl.sum(((k >= cand) & in_range).to(tl.int32))
                threshold = tl.where(cnt >= TOPK, cand, threshold)
                done = (cnt == TOPK).to(tl.int32)
    tl.store(thr_ptr, threshold.to(tl.int32, bitcast=True))


@libentry()
@triton.jit
def _topk_merge_select_kernel(
    cand_key_ptr,
    cand_idx_ptr,
    seq_lens_ptr,
    next_n,
    thr_ptr,
    out_indices_ptr,
    eq_scratch_ptr,
    vocab,
    n_cand,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_len = _row_len(seq_lens_ptr, next_n, row_id, vocab)
    out_indices_ptr += row_id * TOPK
    eq_scratch_ptr += row_id * TOPK
    thr_ptr += row_id
    lane = tl.arange(0, BLOCK)

    if row_len <= TOPK:
        # Whole row fits: emit every valid column in order and pad with -1.
        pad_chunks: tl.constexpr = (TOPK + BLOCK - 1) // BLOCK
        for chunk in tl.range(0, pad_chunks):
            pos = chunk * BLOCK + lane
            take_row = pos < row_len
            tl.store(out_indices_ptr + pos, pos.to(tl.int32), mask=take_row)
            take_pad = (pos >= row_len) & (pos < TOPK)
            tl.store(out_indices_ptr + pos, -1, mask=take_pad)
        return

    threshold = tl.load(thr_ptr).to(tl.uint32, bitcast=True)
    base = row_id * n_cand
    gt_count = tl.zeros((), dtype=tl.int32)
    eq_count = tl.zeros((), dtype=tl.int32)
    ACC_SCALE: tl.constexpr = 16384  # per-tile counts < 2048 < 2^14
    for t in tl.range(0, tl.cdiv(n_cand, BLOCK)):
        offs = t * BLOCK + lane
        in_range = offs < n_cand
        k = tl.load(cand_key_ptr + base + offs, mask=in_range).to(
            tl.uint32, bitcast=True
        )
        idx = tl.load(cand_idx_ptr + base + offs, mask=in_range)
        gt = in_range & (k > threshold)
        eq = in_range & (k == threshold)
        # Candidates are stored chunk-major with ascending original index
        # inside each chunk — globally ascending among threshold-equal keys —
        # so the cumsum fill picks the lowest original indices on ties.
        pos_gt = gt_count + tl.cumsum(gt.to(tl.int32), axis=0) - 1
        tl.store(out_indices_ptr + pos_gt, idx, mask=gt)
        pos_eq = eq_count + tl.cumsum(eq.to(tl.int32), axis=0) - 1
        tl.store(eq_scratch_ptr + pos_eq, idx, mask=eq & (pos_eq < TOPK))
        # One packed sum carries both running totals (independent tl.sum
        # calls serialize on this backend).
        packed = tl.sum(gt.to(tl.int32) + eq.to(tl.int32) * ACC_SCALE)
        gt_count += packed & 0x3FFF
        eq_count += packed >> 14
    # Emit the needed threshold-equal candidates behind the strict-greater
    # ones. gt_count < TOPK and gt_count + eq_count >= TOPK hold by the
    # maximality of the threshold, so exactly TOPK slots get written.
    eq_chunks: tl.constexpr = (TOPK + BLOCK - 1) // BLOCK
    for chunk in tl.range(0, eq_chunks):
        idx = chunk * BLOCK + lane
        valid = idx < (TOPK - gt_count)
        eq_idx = tl.load(eq_scratch_ptr + idx, mask=valid)
        eq_idx = tl.where(valid, eq_idx, 0)
        tl.store(out_indices_ptr + gt_count + idx, eq_idx, mask=valid)


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for decode phase of DeepSeek V4.

    Selects top_k indices per row of ``logits`` using a two-level
    top-K-of-top-K search: per-chunk bit searches prune each row to
    num_chunks*top_k candidates, then a merge search picks the global
    threshold and a stable cumsum select emits the final indices. Only valid
    elements within [0, row_len) are considered; ties at the threshold value
    resolve to the lowest original indices.

    Args:
        logits: [num_rows, vocab_size] float32 tensor.
        next_n: number of next tokens (MTP); rows belong to batch
            ``row // next_n`` with valid prefix ``seq_lens[b] - next_n +
            (row % next_n) + 1``.
        seq_lens: [B,] int32 valid prefix lengths.
        indices: [num_rows, top_k] int32 output buffer.
        num_rows: must equal logits.shape[0].
        stride0: logits.stride(0).
        stride1: logits.stride(1).
        top_k: number of top elements to select.
    """
    logger.debug("GEMS_ENFLAME TOP_K_PER_ROW_DECODE")

    assert num_rows == logits.shape[0]
    device = logits.device
    vocab = logits.shape[1]
    # Chunk-width dispatch: small grids stay on the narrow 8192 chunk (the
    # per-CTA probe chain is the wall; more CTAs shorten the critical path),
    # large grids switch to 16384 (halved CTA count amortizes the chain).
    if num_rows * triton.cdiv(vocab, CHUNK_NARROW) >= WIDE_CHUNK_MIN_CTAS:
        chunk = CHUNK_WIDE
    else:
        chunk = CHUNK_NARROW
    num_chunks = triton.cdiv(vocab, chunk)
    n_cand = num_chunks * top_k

    chunk_thr = torch.empty((num_rows, num_chunks), device=device, dtype=torch.int32)
    chunk_gtc = torch.empty((num_rows, num_chunks), device=device, dtype=torch.int32)
    cand_key = torch.empty((num_rows, n_cand), device=device, dtype=torch.int32)
    cand_idx = torch.empty((num_rows, n_cand), device=device, dtype=torch.int32)
    thresholds = torch.empty((num_rows,), device=device, dtype=torch.int32)
    eq_scratch = torch.empty((num_rows, top_k), device=device, dtype=torch.int32)

    _topk_chunk_threshold_kernel[(num_rows * num_chunks,)](
        logits,
        seq_lens,
        next_n,
        stride0,
        stride1,
        chunk_thr,
        chunk_gtc,
        vocab,
        num_chunks,
        TOPK=top_k,
        CHUNK=chunk,
        num_warps=8,
    )
    _topk_chunk_select_kernel[(num_rows * num_chunks,)](
        logits,
        seq_lens,
        next_n,
        stride0,
        stride1,
        chunk_thr,
        chunk_gtc,
        cand_key,
        cand_idx,
        vocab,
        num_chunks,
        TOPK=top_k,
        CHUNK=chunk,
        BLOCK=SELECT_BLOCK,
        num_warps=1,
    )
    # Merge-phase search: single register-resident tile up to 16384
    # candidates (probed correct + faster than the old two-tile form);
    # reload loop beyond that.
    if n_cand <= 2 * MERGE_BLOCK:
        merge_block = triton.next_power_of_2(n_cand)
        merge_even = n_cand == merge_block
        merge_resident = True
    else:
        merge_block = MERGE_BLOCK
        merge_even = False
        merge_resident = False
    _topk_merge_threshold_kernel[(num_rows,)](
        cand_key,
        seq_lens,
        next_n,
        thresholds,
        vocab,
        n_cand,
        TOPK=top_k,
        BLOCK=merge_block,
        RESIDENT=merge_resident,
        EVEN=merge_even,
        num_warps=8,
    )
    _topk_merge_select_kernel[(num_rows,)](
        cand_key,
        cand_idx,
        seq_lens,
        next_n,
        thresholds,
        indices,
        eq_scratch,
        vocab,
        n_cand,
        TOPK=top_k,
        BLOCK=SELECT_BLOCK,
        num_warps=1,
    )
