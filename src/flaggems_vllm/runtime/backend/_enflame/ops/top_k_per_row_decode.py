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
  once into registers and runs the same greedy MSB-first bit search for the
  chunk-local TOPK-th largest monotone key. The search exits early once
  ``count(key >= prefix) == TOPK`` exactly: then the elements at or above
  the prefix are precisely the chunk's top-K, so the remaining bits cannot
  change the selected set (worst case still 32 probes, e.g. all-equal rows).
* ``_topk_chunk_select_kernel`` (num_warps=1, where tl.cumsum compiles)
  re-scans the chunk and stably compacts the chunk's top-K candidates —
  elements with key >= local threshold, ascending original index, capped at
  TOPK — into dense (key, index) slots, padded with key 0 / index -1.
* ``_topk_merge_threshold_kernel`` (num_warps=8) runs the same bit search
  over the num_chunks*TOPK candidates (register-resident for <= 2 tiles,
  reload loop otherwise) to find the global TOPK-th largest key T.
* ``_topk_merge_select_kernel`` (num_warps=1) selects the final answer from
  the candidates: strict-greater-than-T keys keep stable cumsum positions,
  threshold-equal keys fill the remaining slots in ascending original index,
  matching the stable-argsort reference on ties and staying bit-identical
  across repeated invocations.

Because the global top-K of a row is always contained in the union of the
per-chunk stable top-Ks, and threshold-equal candidates are emitted in
globally ascending original-index order (chunk-major, ascending within each
chunk), the selected index set equals the stable-argsort reference.

Measured do_bench (kernel mode, warmup 1000ms/iter 100ms) vs the previous
2-kernel rewrite: rows=1 2771 -> ~180 us (~15x), rows=496 46.1 -> ~21 ms
(~2.2x); the rows>=16 shapes gain 2-5x.
"""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)

# Per-chunk scan width for the threshold search (num_warps=8): 8192 is the
# knowledge-sanctioned cap for masked-load + tl.sum combinations on GCU
# (>=16384 silently corrupts) and equals the 32KB-per-CTA 1D slice limit.
CHUNK = 8192
# Compaction / final-select scan width (num_warps=1): tl.cumsum compiles and
# is bit-correct up to 2048 lanes on this stack; 4096 fails with the
# memref_ext.slice_start properties error.
SELECT_BLOCK = 2048
# Candidate-merge scan width (num_warps=8); at most two resident tiles.
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

    one = tl.full((), 1, tl.uint32)
    # Greedy MSB-first search for the largest monotone key T with
    # count(key >= T) >= TOPK — the TOPK-th largest key of the chunk.
    # Early exit: once count(key >= prefix) == TOPK exactly, the elements at
    # or above the prefix are exactly the chunk's top-K, so the selected set
    # is already final and the remaining bits are irrelevant.
    threshold = tl.zeros((), dtype=tl.uint32)
    done = tl.zeros((), dtype=tl.int32)
    for i in tl.range(0, 32):
        if done == 0:
            bit = 31 - i
            cand = threshold | (one << bit)
            cnt = tl.sum(((key >= cand) & in_range).to(tl.int32))
            threshold = tl.where(cnt >= TOPK, cand, threshold)
            done = (cnt == TOPK).to(tl.int32)
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
        gt_before += tl.sum(gt.to(tl.int32))
        eq_before += tl.sum(eq.to(tl.int32))
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
    # slots carry key 0 and never count for any candidate >= 1.
    threshold = tl.zeros((), dtype=tl.uint32)
    done = tl.zeros((), dtype=tl.int32)
    if RESIDENT:
        m0 = lane < n_cand
        k0 = tl.load(cand_key_ptr + base + lane, mask=m0).to(tl.uint32, bitcast=True)
        m1 = (BLOCK + lane) < n_cand
        k1 = tl.load(cand_key_ptr + base + BLOCK + lane, mask=m1).to(
            tl.uint32, bitcast=True
        )
        for i in tl.range(0, 32):
            if done == 0:
                bit = 31 - i
                cand = threshold | (one << bit)
                cnt = tl.sum(((k0 >= cand) & m0).to(tl.int32)) + tl.sum(
                    ((k1 >= cand) & m1).to(tl.int32)
                )
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
        gt_count += tl.sum(gt.to(tl.int32))
        eq_count += tl.sum(eq.to(tl.int32))
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
    logger.debug("GEMS TOP_K_PER_ROW_DECODE")

    assert num_rows == logits.shape[0]
    device = logits.device
    vocab = logits.shape[1]
    num_chunks = triton.cdiv(vocab, CHUNK)
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
        CHUNK=CHUNK,
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
        CHUNK=CHUNK,
        BLOCK=SELECT_BLOCK,
        num_warps=1,
    )
    _topk_merge_threshold_kernel[(num_rows,)](
        cand_key,
        seq_lens,
        next_n,
        thresholds,
        vocab,
        n_cand,
        TOPK=top_k,
        BLOCK=MERGE_BLOCK,
        RESIDENT=n_cand <= 2 * MERGE_BLOCK,
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
