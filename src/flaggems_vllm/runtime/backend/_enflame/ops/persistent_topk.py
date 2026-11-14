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

"""vLLM-compatible persistent topk decode (GCU / enflame rewrite).

The seed implementation (ported from vLLM's persistent_topk.cuh: persistent
cooperative radix kernel with cross-CTA spin barriers, plus sampled /
per-CTA-per-row histogram paths) cannot compile on the current flagtree
toolchain:

* its ``_barrier_with_atomic_add`` void jit helper contains a ``while`` spin,
  and calling a void jit function with a ``while`` body aborts the in-process
  compiler with an ``mlir::ResultRange`` index assertion
  (compiler_issues/vllm_persistent_topk.md);
* its tle.gpu.alloc histograms / masked atomics / ``ptr + zeros`` scalar-slot
  stores lower through ``memref_ext.memset_start`` ops carrying
  ``operandSegmentSizes`` properties that ``gcu-compiler-opt`` rejects
  (compiler_issues/vllm_top_k_per_row_decode.md);
* the cross-CTA spin barrier also requires all CTAs of a group to be
  co-resident, which is a launch-geometry hazard the GCU runtime does not
  guarantee the way CUDA cooperative launch does.

This copy instead selects the top-k per row with an exact value threshold,
following the proven top_k_per_row_decode rewrite (enflame, commit 362fa91):

* ``_topk_threshold_search_kernel`` (num_warps=1) greedily finds the largest
  monotone uint32 key T with count(key >= T) >= TOPK — the TOPK-th largest
  key — via up to 32 reduction probes (bit-by-bit MSB-first search). No
  atomics, no histogram, no shared-memory buffers. Rows that fit one tile
  load and rank their data once and keep it register-resident across the
  whole probe chain (a dependent tl.sum costs ~1.8us on GCU and serializes,
  so reloading per probe adds ~30%), and the resident tile width — and with
  it the per-sum reduction latency — is picked per row from pow2 buckets
  (2048/4096/BLOCK; measured ~0.75us/sum at 2048 lanes vs ~1.8us at 8192);
  longer rows re-stream per probe at the full BLOCK width. The search exits
  early once count(key >= prefix) == TOPK exactly — the selected set is then
  already final (worst case still 32 probes, e.g. all-equal rows).
* ``_topk_select_kernel`` (num_warps=1, where tl.cumsum is bit-correct) makes
  one scan in ascending original-index order: strict-greater elements are
  written directly at stable cumsum positions, threshold-equal elements are
  collected (capped at TOPK) and emitted behind them, so ties resolve to the
  lowest original indices — deterministic and repeat-safe.

Both kernels are launched with grid=(num_rows,), one CTA per row: no cross-CTA
synchronization, so the persistent/cooperative machinery is unnecessary.
``workspace`` is used for the per-row threshold and equal-tie scratch when it
is large enough (keeps the CUDAGraph-friendly no-allocation contract),
otherwise small scratch tensors are allocated.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Wide-tile scan width for the threshold search's reload path (long rows) and
# the widest resident arm: 8192 is the knowledge-sanctioned cap for
# masked-load + tl.sum combinations on GCU (>=16384 silently corrupts); wider
# tiles amortize the per-iteration DMA setup cost.
THRESHOLD_SEARCH_BLOCK = 8192
# Selection scan width (num_warps=1): tl.cumsum compiles and is bit-correct
# up to 2048 lanes on this stack; 4096 fails with the memref_ext.slice_start
# properties error.
SELECT_BLOCK = 2048


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
    # seed's descending monotone map (_convert_to_uint32_v2).
    return ~_convert_to_uint32(x)


@triton.jit
def _row_len(seq_lens_ptr, next_n, row_id, max_len):
    # lengths may be [num_rows] (next_n == 1) or [B, next_n] (MTP). In the
    # 2D case the flattened layout stores lengths[b, j] at b*next_n + j and
    # the per-batch base length is column 0 (vLLM MTP semantics, mirrored by
    # the reference: row end = lengths[b, 0] - next_n + j + 1).
    batch_id = row_id // next_n
    batch_offset = row_id % next_n
    seq_len = tl.load(seq_lens_ptr + batch_id * next_n)
    length = seq_len - next_n + batch_offset + 1
    return tl.minimum(tl.maximum(length, 0), max_len)


@triton.jit
def _greedy_threshold_1bit(key, in_range, TOPK: tl.constexpr):
    # Greedy MSB-first search for the largest monotone key T with
    # count(key >= T) >= TOPK — the TOPK-th largest key of the tile.
    # NOTE: multi-bit pair probing (test prefix|11, fall back to prefix|10)
    # is NOT equivalent: it cannot represent the "high bit 0, low bit 1"
    # outcome, so every pair would need a third probe (expected 2.25 > 2
    # probes/pair) — verified by simulation, do not revisit.
    one = tl.full((), 1, tl.uint32)
    threshold = tl.zeros((), dtype=tl.uint32)
    done = tl.zeros((), dtype=tl.int32)
    for i in tl.range(0, 32):
        if done == 0:
            bit = 31 - i
            cand = threshold | (one << bit)
            cnt = tl.sum(((key >= cand) & in_range).to(tl.int32))
            threshold = tl.where(cnt >= TOPK, cand, threshold)
            done = (cnt == TOPK).to(tl.int32)
    return threshold


@triton.jit
def _topk_threshold_search_kernel(
    logits_ptr,
    seq_lens_ptr,
    next_n,
    max_len,
    stride0,
    stride1,
    thr_ptr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_len = _row_len(seq_lens_ptr, next_n, row_id, max_len)
    logits_ptr += row_id * stride0
    thr_ptr += row_id

    if row_len <= TOPK:
        tl.store(thr_ptr, 0)
        return

    lane = tl.arange(0, BLOCK)
    one = tl.full((), 1, tl.uint32)
    # count(key >= T + 1) < TOPK holds by maximality of the greedy threshold
    # T, so every element strictly greater than T belongs to the answer.
    # Early exit: once count(key >= prefix) == TOPK exactly, the elements at
    # or above the prefix are precisely the row's top-K, so the remaining
    # bits cannot change the selected set (worst case still 32 probes, e.g.
    # all-equal rows) — probe-verified pattern from top_k_per_row_decode.
    threshold = tl.zeros((), dtype=tl.uint32)
    done = tl.zeros((), dtype=tl.int32)
    if tl.cdiv(row_len, BLOCK) == 1:
        # Register-resident fast path (row fits one tile): load and rank the
        # row once, then run the whole dependent-probe chain without
        # re-fetching — a dependent tl.sum costs ~1.8us of latency on GCU and
        # serializes, so the reload would add ~30% per probe. The reduction
        # latency also scales with tile width, so short rows use narrower
        # resident tiles (measured: ~0.75us/sum at 2048 lanes vs ~1.8us at
        # 8192); all arms share the num_warps=1 launch that minimizes it.
        if row_len <= 2048:
            lane_a = tl.arange(0, 2048)
            mask_a = lane_a < row_len
            key_a = _rank_key(tl.load(logits_ptr + lane_a * stride1, mask=mask_a))
            threshold = _greedy_threshold_1bit(key_a, mask_a, TOPK)
        elif row_len <= 4096:
            lane_b = tl.arange(0, 4096)
            mask_b = lane_b < row_len
            key_b = _rank_key(tl.load(logits_ptr + lane_b * stride1, mask=mask_b))
            threshold = _greedy_threshold_1bit(key_b, mask_b, TOPK)
        else:
            mask_c = lane < row_len
            key_c = _rank_key(tl.load(logits_ptr + lane * stride1, mask=mask_c))
            threshold = _greedy_threshold_1bit(key_c, mask_c, TOPK)
    else:
        # Long rows re-stream per probe (the row exceeds any resident tile).
        for i in tl.range(0, 32):
            if done == 0:
                bit = 31 - i
                cand = threshold | (one << bit)
                cnt = tl.zeros((), dtype=tl.int32)
                for t in tl.range(0, tl.cdiv(row_len, BLOCK)):
                    offs = t * BLOCK + lane
                    in_range = offs < row_len
                    x = tl.load(logits_ptr + offs * stride1, mask=in_range)
                    key = _rank_key(x)
                    cnt += tl.sum(((key >= cand) & in_range).to(tl.int32))
                threshold = tl.where(cnt >= TOPK, cand, threshold)
                done = (cnt == TOPK).to(tl.int32)
    tl.store(thr_ptr, threshold.to(tl.int32, bitcast=True))


@triton.jit
def _topk_select_kernel(
    logits_ptr,
    seq_lens_ptr,
    next_n,
    max_len,
    stride0,
    stride1,
    thr_ptr,
    out_indices_ptr,
    eq_scratch_ptr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_len = _row_len(seq_lens_ptr, next_n, row_id, max_len)
    logits_ptr += row_id * stride0
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
    n_tiles = tl.cdiv(row_len, BLOCK)
    gt_count = tl.zeros((), dtype=tl.int32)
    eq_count = tl.zeros((), dtype=tl.int32)
    for t in tl.range(0, n_tiles):
        offs = t * BLOCK + lane
        in_range = offs < row_len
        x = tl.load(logits_ptr + offs * stride1, mask=in_range)
        key = _rank_key(x)
        gt = in_range & (key > threshold)
        eq = in_range & (key == threshold)
        # Stable positions: strict-greater elements take the leading slots
        # in scan order (ascending original index); threshold-equal elements
        # are collected behind them, capped at TOPK.
        pos_gt = gt_count + tl.cumsum(gt.to(tl.int32), axis=0) - 1
        tl.store(out_indices_ptr + pos_gt, offs.to(tl.int32), mask=gt)
        pos_eq = eq_count + tl.cumsum(eq.to(tl.int32), axis=0) - 1
        tl.store(
            eq_scratch_ptr + pos_eq,
            offs.to(tl.int32),
            mask=eq & (pos_eq < TOPK),
        )
        gt_count += tl.sum(gt.to(tl.int32))
        eq_count += tl.sum(eq.to(tl.int32))
    # Emit the needed threshold-equal elements — the ones with the lowest
    # original indices (collected first) — behind the strict-greater ones.
    # gt_count < TOPK and eq_count >= TOPK - gt_count both hold by the
    # maximality of the threshold, so exactly TOPK slots get written.
    eq_chunks: tl.constexpr = (TOPK + BLOCK - 1) // BLOCK
    for chunk in tl.range(0, eq_chunks):
        idx = chunk * BLOCK + lane
        valid = idx < (TOPK - gt_count)
        eq_idx = tl.load(eq_scratch_ptr + idx, mask=valid)
        eq_idx = tl.where(valid, eq_idx, 0)
        tl.store(out_indices_ptr + gt_count + idx, eq_idx, mask=valid)


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
    if num_rows == 0:
        return
    seq_lens = lengths.reshape(-1) if lengths.dim() == 2 else lengths
    next_n = lengths.size(1) if lengths.dim() == 2 else 1
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
    max_len = min(logits.shape[1], actual_max)

    device = logits.device
    scratch_bytes = num_rows * (4 + 4 * k)
    if workspace.numel() >= scratch_bytes:
        thresholds = workspace[: num_rows * 4].view(torch.int32)
        eq_scratch = (
            workspace[num_rows * 4 : scratch_bytes].view(torch.int32).view(num_rows, k)
        )
    else:
        thresholds = torch.empty((num_rows,), device=device, dtype=torch.int32)
        eq_scratch = torch.empty((num_rows, k), device=device, dtype=torch.int32)

    stride0 = logits.stride(0)
    stride1 = logits.stride(1)
    _topk_threshold_search_kernel[(num_rows,)](
        logits,
        seq_lens,
        next_n,
        max_len,
        stride0,
        stride1,
        thresholds,
        TOPK=k,
        BLOCK=THRESHOLD_SEARCH_BLOCK,
        num_warps=1,
    )
    _topk_select_kernel[(num_rows,)](
        logits,
        seq_lens,
        next_n,
        max_len,
        stride0,
        stride1,
        thresholds,
        output,
        eq_scratch,
        TOPK=k,
        BLOCK=SELECT_BLOCK,
        num_warps=1,
    )
