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

"""Ascend persistent_topk -- two route families, selected at runtime.

This file has no version-tagged paths; routes are named after the algorithm
they implement (see the per-section banners further down):

  radix_hist              histogram route.  Multi-block partial histograms
                          atomic-accumulated into a per-row workspace
                          histogram, 3 narrowing radix steps (11/11/10 bits),
                          then prefix + rank placement of the guaranteed and
                          final classes.  Correct on every shape, but each row
                          is streamed several times, so it is the slow route
                          on long rows.

  segmerge                segment-sort route, short rows.  Each 4096-element
                          segment is sorted by the device sort op, which emits
                          TOPK proposals, and one CTA per row merges the (at
                          most 4) proposal streams.  Needs the FlagTree TLE
                          Ascend custom ops and is gated on `k == 512`.

  segmerge_persistent     segment-sort route, long rows, persistent form.
                          A fixed 40-CTA grid walks the (row, segment) task
                          space, then a fan-in merge tree with at most 4 ways
                          per level reduces the proposal streams per row.

  segmerge_persistent_half
                          the same persistent route for k > 1024, where each
                          4096-element segment is sorted as two halves and the
                          two half-proposal streams are merged back into the
                          segment slot (the full-segment device sort path needs
                          a 144 KB UB workspace, which does not fit the 192 KB
                          per-AI-Core ABI budget).

Selection and fallback live in `persistent_topk()`.  The host side performs no
torch compute / copy / conversion ops: only metadata reads (shape/stride/dtype),
views and kernel launches.  All per-row decisions (trivial vs wide, MTP length
adjustment, prefix/threshold combination, batching indices) happen in kernels.

Layout per launch batch of rows of the radix_hist route (b rows fit the
caller-provided workspace):
    hist[b*_HIST_SLOTS]  t1[b] f1[b] t2[b] f2[b] t3[b] f3[b]
    cnt[b*2G] base[b*2G] found[b*2]
where t_k = threshold digit of step k, f_k = cumulative found after step k,
cnt/base/found = per-block per-class counts, prefix bases and totals used by
the two light write kernels."""

import os
import warnings

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils.triton_version_utils import has_triton_tle

_BLOCK = 1024
_G = 40
_HIST_SLOTS = 4096  # max per-row histogram slots (real range [0, 2048) + dummy region)


@triton.jit
def _v1_convert_to_trt_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    sign_set = (bits & sign_mask) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFFFFFF, tl.uint32)
    return tl.where(sign_set, bits, inv)


@triton.jit
def _radix_hist_clear(ptr, N, BLOCK: tl.constexpr):
    """Zero N int32 slots (grid = ceil(N / BLOCK))."""
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    offs = pid * BLOCK + lane
    tl.store(ptr + offs, tl.zeros([BLOCK], dtype=tl.int32), mask=offs < N)


@triton.jit
def _radix_hist_trivial_all(
    lengths_ptr, out_ptr, stride, next_n, TOPK: tl.constexpr, BLOCK: tl.constexpr
):
    """Rows with effective length <= TOPK are written directly (grid=num_rows)."""
    row = tl.program_id(0)
    raw = tl.load(lengths_ptr + row)
    if next_n > 1:
        N = raw - next_n + (row % next_n) + 1
    else:
        N = raw
    if N > TOPK:
        return  # wide row: handled by the multi-block path
    lane = tl.arange(0, BLOCK)
    row_out = out_ptr + row * TOPK
    n_chunks: tl.constexpr = (TOPK + BLOCK - 1) // BLOCK
    for c in tl.static_range(0, n_chunks):
        pos = c * BLOCK + lane
        tl.store(row_out + pos, pos, mask=pos < N)
        tl.store(row_out + pos, -1, mask=(pos >= N) & (pos < TOPK))


@triton.jit
def _radix_hist_hist(
    x_ptr,
    lengths_ptr,
    hist_ptr,
    p1_ptr,
    p2_ptr,
    stride,
    next_n,
    TOPK,
    STEP: tl.constexpr,
    SHIFT: tl.constexpr,
    BIN_MASK: tl.constexpr,
    NUM_BINS: tl.constexpr,
    PREF_SHIFT: tl.constexpr,
    PREF_MASK: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Multi-block partial histogram, atomic-accumulated into per-row hist.

    STEP controls the prefix: 1 -> no prefix; 2 -> prefix = t1[row] (p1);
    3 -> prefix = (t1 << 11) | t2 (combined from p1/p2). Rows with effective
    length <= TOPK return immediately (already written by _radix_hist_trivial_all).

    The per-block scan length is computed in-kernel from the row's effective
    length (``cblk = ceil(N / G)``), so short rows are not scanned up to the
    padded stride.
    """
    pid = tl.program_id(0)
    chunk = tl.program_id(1)
    raw = tl.load(lengths_ptr + pid)
    if next_n > 1:
        N = raw - next_n + (pid % next_n) + 1
    else:
        N = raw
    if N <= TOPK:
        return
    if STEP == 1:
        PREFIX = tl.zeros((), dtype=tl.int32)
    elif STEP == 2:
        PREFIX = tl.load(p1_ptr + pid)
    else:
        PREFIX = ((tl.load(p1_ptr + pid) & 0x7FF) << 11) | (
            tl.load(p2_ptr + pid) & 0x7FF
        )
    lane = tl.arange(0, BLOCK)
    row_in = x_ptr + pid * stride
    hist = tl.zeros([NUM_BINS], dtype=tl.int32)
    cblk = (N + G - 1) // G  # per-block span inside the row (runtime)
    n_tiles = tl.cdiv(cblk, BLOCK)
    for t in tl.range(0, n_tiles):
        base_in_chunk = t * BLOCK + lane
        in_chunk = base_in_chunk < cblk
        offs = chunk * cblk + base_in_chunk
        m = in_chunk & (offs < N)
        x = tl.load(row_in + offs, mask=m, other=float("-inf"))
        key = _v1_convert_to_trt_uint32(x)
        digit = ((key >> SHIFT) & BIN_MASK).to(tl.int32)
        if STEP == 1:
            # dummy bin sits at NUM_BINS//2: outside the real range [0, NUM_BINS//2)
            bin_ = tl.where(m, digit, NUM_BINS // 2)
        else:
            pdig = ((key >> PREF_SHIFT) & PREF_MASK).to(tl.int32)
            match = m & (pdig == PREFIX)
            bin_ = tl.where(match, digit, NUM_BINS // 2 + (pdig & (NUM_BINS // 2 - 1)))
        hist += tl.histogram(bin_, NUM_BINS)
    bins_all = tl.arange(0, NUM_BINS)
    tl.atomic_add(
        hist_ptr + pid * NUM_BINS + bins_all, hist, sem="relaxed", scope="gpu"
    )


@triton.jit
def _radix_hist_threshold(
    hist_ptr,
    t_ptr,
    f_ptr,
    prev_f_ptr,
    STEP: tl.constexpr,
    NUM_BINS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-row threshold from the combined histogram.

    STEP 1: base = 0. STEP 2/3: base = cumulative found of the previous step
    (prev_f[row]). Writes t_k[row] = threshold digit and
    f_k[row] = base + count(bins < t_k) (cumulative found).
    """
    pid = tl.program_id(0)
    if STEP == 1:
        base = tl.zeros((), dtype=tl.int32)
    else:
        base = tl.load(prev_f_ptr + pid)
    bins_all = tl.arange(0, NUM_BINS)
    hist = tl.load(hist_ptr + pid * NUM_BINS + bins_all)
    ps = tl.cumsum(hist, axis=0) - hist
    nps = ps + hist
    thr_mask = (base + ps < TOPK) & (base + nps >= TOPK)
    t = tl.sum(tl.where(thr_mask, bins_all, 0))
    cnt = tl.sum(tl.where(bins_all < t, hist, 0))
    tl.store(t_ptr + pid, t)
    tl.store(f_ptr + pid, base + cnt)


@triton.jit
def _radix_hist_count(
    x_ptr,
    lengths_ptr,
    cnt_ptr,
    t1_ptr,
    t2_ptr,
    t3_ptr,
    stride,
    next_n,
    TOPK,
    BLOCK: tl.constexpr,
    G: tl.constexpr,
):
    """Per-block counts of guaranteed (q1|q2|q3) and final (key == pattern)."""
    pid = tl.program_id(0)
    chunk = tl.program_id(1)
    raw = tl.load(lengths_ptr + pid)
    if next_n > 1:
        N = raw - next_n + (pid % next_n) + 1
    else:
        N = raw
    if N <= TOPK:
        return
    t1 = tl.load(t1_ptr + pid)
    t2 = tl.load(t2_ptr + pid)
    t3 = tl.load(t3_ptr + pid)
    p21 = ((t1 & 0x7FF) << 11) | (t2 & 0x7FF)
    pattern = (p21 << 10) | (t3 & 0x3FF)
    lane = tl.arange(0, BLOCK)
    row_in = x_ptr + pid * stride
    cg = tl.zeros((), dtype=tl.int32)
    cf = tl.zeros((), dtype=tl.int32)
    cblk = (N + G - 1) // G  # per-block span inside the row (runtime)
    n_tiles = tl.cdiv(cblk, BLOCK)
    for t in tl.range(0, n_tiles):
        base_in_chunk = t * BLOCK + lane
        in_chunk = base_in_chunk < cblk
        offs = chunk * cblk + base_in_chunk
        m = in_chunk & (offs < N)
        x = tl.load(row_in + offs, mask=m, other=float("-inf"))
        key = _v1_convert_to_trt_uint32(x)
        d1 = ((key >> 21) & 0x7FF).to(tl.int32)
        d2 = ((key >> 10) & 0x7FF).to(tl.int32)
        d3 = (key & 0x3FF).to(tl.int32)
        q1 = m & (d1 < t1)
        q2 = m & (d1 == t1) & (d2 < t2)
        q3 = m & (((key >> 10) & 0x3FFFFF).to(tl.int32) == p21) & (d3 < t3)
        qf = m & (key.to(tl.int32) == pattern)
        cg += tl.sum((q1 | q2 | q3).to(tl.int32))
        cf += tl.sum(qf.to(tl.int32))
    tl.store(cnt_ptr + pid * G * 2 + chunk * 2 + 0, cg)
    tl.store(cnt_ptr + pid * G * 2 + chunk * 2 + 1, cf)


@triton.jit
def _radix_hist_prefix(
    cnt_ptr, base_ptr, found_ptr, G: tl.constexpr, BLOCK: tl.constexpr
):
    """Per-block per-class prefix bases + class totals."""
    pid = tl.program_id(0)
    gs = tl.arange(0, G)
    counts = tl.load(cnt_ptr + pid * G * 2 + gs[:, None] * 2 + tl.arange(0, 2)[None, :])
    ps = tl.cumsum(counts, axis=0) - counts  # [G, 2] exclusive prefix per class
    tl.store(base_ptr + pid * G * 2 + gs[:, None] * 2 + tl.arange(0, 2)[None, :], ps)
    totals = tl.sum(counts, axis=0)  # [2] = (found3, final_total)
    tl.store(found_ptr + pid * 2 + tl.arange(0, 2), totals)


@triton.jit
def _radix_hist_write_guar(
    x_ptr,
    lengths_ptr,
    out_ptr,
    base_ptr,
    t1_ptr,
    t2_ptr,
    t3_ptr,
    stride,
    next_n,
    TOPK,
    BLOCK: tl.constexpr,
    G: tl.constexpr,
):
    """Write guaranteed (q1|q2|q3) elements at [block_base_guar + rank]."""
    pid = tl.program_id(0)
    chunk = tl.program_id(1)
    raw = tl.load(lengths_ptr + pid)
    if next_n > 1:
        N = raw - next_n + (pid % next_n) + 1
    else:
        N = raw
    if N <= TOPK:
        return
    t1 = tl.load(t1_ptr + pid)
    t2 = tl.load(t2_ptr + pid)
    t3 = tl.load(t3_ptr + pid)
    lane = tl.arange(0, BLOCK)
    row_in = x_ptr + pid * stride
    row_out = out_ptr + pid * TOPK
    p21 = ((t1 & 0x7FF) << 11) | (t2 & 0x7FF)
    bg = tl.load(base_ptr + pid * G * 2 + chunk * 2 + 0)
    cblk = (N + G - 1) // G  # per-block span inside the row (runtime)
    n_tiles = tl.cdiv(cblk, BLOCK)
    n_out = tl.zeros((), dtype=tl.int32)
    for t in tl.range(0, n_tiles):
        base_in_chunk = t * BLOCK + lane
        in_chunk = base_in_chunk < cblk
        offs = chunk * cblk + base_in_chunk
        m = in_chunk & (offs < N)
        x = tl.load(row_in + offs, mask=m, other=float("-inf"))
        key = _v1_convert_to_trt_uint32(x)
        d1 = ((key >> 21) & 0x7FF).to(tl.int32)
        d2 = ((key >> 10) & 0x7FF).to(tl.int32)
        d3 = (key & 0x3FF).to(tl.int32)
        q1 = m & (d1 < t1)
        q2 = m & (d1 == t1) & (d2 < t2)
        q3 = m & (((key >> 10) & 0x3FFFFF).to(tl.int32) == p21) & (d3 < t3)
        qg = (q1 | q2 | q3).to(tl.int32)
        lg = tl.cumsum(qg, axis=0) - qg
        pg = bg + n_out + lg
        tl.store(row_out + pg, offs, mask=(qg != 0) & (pg < TOPK))
        n_out += tl.sum(qg)


@triton.jit
def _radix_hist_write_final(
    x_ptr,
    lengths_ptr,
    out_ptr,
    base_ptr,
    found_ptr,
    t1_ptr,
    t2_ptr,
    t3_ptr,
    stride,
    next_n,
    TOPK,
    BLOCK: tl.constexpr,
    G: tl.constexpr,
):
    """Write final (key == pattern) at [found3 + block_base_final + rank], capped."""
    pid = tl.program_id(0)
    chunk = tl.program_id(1)
    raw = tl.load(lengths_ptr + pid)
    if next_n > 1:
        N = raw - next_n + (pid % next_n) + 1
    else:
        N = raw
    if N <= TOPK:
        return
    t1 = tl.load(t1_ptr + pid)
    t2 = tl.load(t2_ptr + pid)
    t3 = tl.load(t3_ptr + pid)
    p21 = ((t1 & 0x7FF) << 11) | (t2 & 0x7FF)
    pattern = (p21 << 10) | (t3 & 0x3FF)
    lane = tl.arange(0, BLOCK)
    row_in = x_ptr + pid * stride
    row_out = out_ptr + pid * TOPK
    bf = tl.load(base_ptr + pid * G * 2 + chunk * 2 + 1)
    found3 = tl.load(found_ptr + pid * 2 + 0)
    cblk = (N + G - 1) // G  # per-block span inside the row (runtime)
    n_tiles = tl.cdiv(cblk, BLOCK)
    n_fin = tl.zeros((), dtype=tl.int32)
    for t in tl.range(0, n_tiles):
        base_in_chunk = t * BLOCK + lane
        in_chunk = base_in_chunk < cblk
        offs = chunk * cblk + base_in_chunk
        m = in_chunk & (offs < N)
        x = tl.load(row_in + offs, mask=m, other=float("-inf"))
        key = _v1_convert_to_trt_uint32(x)
        qf = m & (key.to(tl.int32) == pattern)
        lf = tl.cumsum(qf.to(tl.int32), axis=0) - qf.to(tl.int32)
        pf = found3 + bf + n_fin + lf
        tl.store(row_out + pf, offs, mask=qf & (pf < TOPK))
        n_fin += tl.sum(qf.to(tl.int32))


# ---------------------------------------------------------------------------
# route `segmerge`: per-segment native sort + k-way merge (FlagTree TLE
# Ascend custom ops).  One CTA per (row, segment) task, one CTA per row merges
#
# Uses the Ascend custom ops added by FlagTree PR #1065 (sort_1d_pack /
# merge_exhaust_sort4 / unpack_sort). The row is cut into fixed 4096-element
# segments; every segment is sorted in its own CTA (grid = rows * segs) through
# the native vector sort path, and one CTA per row merges the segment
# proposals with repeated exhaustion merges before unpacking values/indices.
#
# Requirements (all probed at runtime, with automatic fallback to `radix_hist`):
#   * FlagTree >= 66f2efb87 (triton_v3.5.x) providing
#     triton.experimental.tle.language.dsa.ascend.custom_ops
#   * bishengir >= 1.2.0 (ships with CANN 9.1) for the custom-op
#     `operandSegmentSizes` encoding
#
# Scope of `segmerge`: k == 512 and rows padded to at most 4 segments
# (max_seq_len <= 16384). Everything else keeps using `radix_hist`.
# ---------------------------------------------------------------------------

_SEGMERGE_SEG = 4096
_SEGMERGE_MAX_SEGS = 4
_SEGMERGE_K = 512
_SEGMERGE_ENV = "FLAGGEMS_ASCEND_PTK_PATH"

# The Ascend custom ops (sort_1d_pack / merge_exhaust_sort4 / unpack_sort) live
# in FlagTree's TLE branch; gate the import the same way the other TLE paths in
# this repo do so the module stays importable on stock Triton builds.
if has_triton_tle(3, 5, 0):
    try:
        import triton.experimental.tle as tle

        HAS_TLE_PTK = True
    except ImportError:
        tle = None
        HAS_TLE_PTK = False
else:
    tle = None
    HAS_TLE_PTK = False

# triton.experimental.tle language-dsa binding is imported lazily; the module is
# absent on Triton builds without the Ascend custom-op support.
_SEGMERGE_STATE = {"probed": False, "usable": False}
_PSEGMERGE_STATE = {"usable": True}  # disabled once `segmerge_persistent` fails

# sort_impl codes accepted by sort_1d_pack (Ascend custom_ops registry).
_PSEGMERGE_SORT_BASE = 0
_PSEGMERGE_SORT_S4096_K129_512 = 1
_PSEGMERGE_SORT_S4096_K1_128_K2048 = 2


def _psegmerge_sort_tmp_size(seg_len: int, topk: int, sort_impl: int) -> int:
    """UB workspace (fp32 elements) that sort_1d_pack needs for this path.

    Sized exactly rather than generously: over-sizing makes the backend spill UB
    to GM (device link then fails on an undefined `malloc`), while under-sizing
    fails with `ub overflow, requires ... bits while ... available` -- and a
    single k=2048 segment really does walk the 192 KB UB budget to the edge
    (1575680 of 1572864 bits).  Both variants were measured on device:

      S4096_K129_512 (k <= 1024): 4x1024 candidate runs + 1024-element chunk
        scratch + a 4*topk merge output.
      S4096_K1_128_K2048 (k > 1024): 9*CEIL_FACTOR(seg_len,32) floats, measured
        (36864 floats for a 4096 segment == 147456 of the 192 KB ABI UB).
    """
    n = (seg_len + 31) // 32 * 32
    if sort_impl == _PSEGMERGE_SORT_S4096_K129_512:
        return 4 * topk * 2 + 1024 * 4 + 4 * topk * 2
    if sort_impl == _PSEGMERGE_SORT_S4096_K1_128_K2048:
        # measured on device with this exact value; the kernel does not compile
        # with it today, see the note on _PSEGMERGE_KS.
        return 9 * n
    return seg_len * 4


def _segmerge_probe() -> bool:
    """Return True when the Ascend custom-op bindings are importable.

    Never raises: any import problem (older FlagTree, non-Ascend build) simply
    disables the `segmerge` routes.
    """
    if not _SEGMERGE_STATE["probed"]:
        _SEGMERGE_STATE["probed"] = True
        usable = False
        if HAS_TLE_PTK:
            try:  # pragma: no cover - depends on the installed FlagTree build
                from triton.experimental.tle.language.dsa.ascend.custom_ops import (  # noqa: F401
                    SORT_IMPL_S4096_K129_512,
                )

                usable = True
            except Exception:
                usable = False
        _SEGMERGE_STATE["usable"] = usable
    return _SEGMERGE_STATE["usable"]


def _segmerge_sort_tmp_size(seg_len: int, sort_run_len: int, sort_impl: int) -> int:
    """UB workspace (in fp32 elements) required by sort_1d_pack.

    Mirrors the sizing rule published with the custom ops (see FlagTree
    python/tutorials/tle/custom/test_custom_ops.py, `_sort_tmp_size`).
    """
    if sort_impl == 0:  # SORT_IMPL_BASE
        return seg_len * 4
    if sort_impl == 2:  # SORT_IMPL_S4096_K1_128_K2048
        props_ab = seg_len * 2
        group_buf = 4 * 512 * 2
        return 2 * props_ab + 2 * group_buf + (2 * group_buf + 8)
    # SORT_IMPL_S4096_K129_512
    candidates = 4 * sort_run_len * 2
    chunk_tmp = 1024 * 4
    merge_out = 4 * sort_run_len * 2
    return candidates + chunk_tmp + merge_out


@triton.jit
def _segmerge_seg_sort(
    x_ptr,
    lengths_ptr,
    props_ptr,
    stride,
    next_n,
    TOPK: tl.constexpr,
    SEGS: tl.constexpr,
    SEG: tl.constexpr,
    TMP_SZ: tl.constexpr,
    SORT_IMPL: tl.constexpr,
):
    """One CTA per (row, segment): sort the segment, emit TOPK proposals.

    Lanes beyond the row's effective length load -inf, so the caller's stride
    and max_seq_len need no padding. Trivial rows are written by
    _radix_hist_trivial_all and skipped here.
    """
    pid = tl.program_id(0)
    seg = pid % SEGS
    row = pid // SEGS
    raw = tl.load(lengths_ptr + row)
    if next_n > 1:
        N = raw - next_n + (row % next_n) + 1
    else:
        N = raw
    if N <= TOPK:
        return
    seg_off = seg * SEG
    lane = tl.arange(0, SEG)
    offs = seg_off + lane
    m = offs < N
    x = tl.load(x_ptr + row * stride + offs, mask=m, other=float("-inf"))
    src_ub = tle.dsa.alloc([SEG], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.to_buffer(x, space=tle.dsa.ascend.UB, bind_buffer=src_ub)
    tmp = tl.zeros([TMP_SZ], dtype=tl.float32)
    props = tl.zeros([2 * TOPK], dtype=tl.float32)
    props = tle.dsa.ascend.raw(
        "sort_1d_pack",
        tle.dsa.to_tensor(src_ub),
        tmp,
        True,
        TOPK,
        seg_off,
        SORT_IMPL,
        out=props,
    )
    tl.store(props_ptr + pid * 2 * TOPK + tl.arange(0, 2 * TOPK), props)


@triton.jit
def _segmerge_merge_unpack(
    lengths_ptr,
    props_ptr,
    out_ptr,
    next_n,
    TOPK: tl.constexpr,
    SEGS: tl.constexpr,
):
    """One CTA per row: merge the SEGS proposal streams, emit TOPK indices.

    merge_exhaust_sort4 stops as soon as one way runs out and reports the
    per-way consumed counts; repeated rounds therefore append consecutive
    sorted prefixes until TOPK proposals are collected.
    """
    pid = tl.program_id(0)
    raw = tl.load(lengths_ptr + pid)
    if next_n > 1:
        N = raw - next_n + (pid % next_n) + 1
    else:
        N = raw
    if N <= TOPK:
        return
    base = props_ptr + pid * SEGS * 2 * TOPK
    in_ub = tle.dsa.alloc(
        [4 * 2 * TOPK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    for w in tl.static_range(0, 4):
        if w < SEGS:
            view = tle.dsa.subview(in_ub, [w * 2 * TOPK], [2 * TOPK], [1])
            tle.dsa.copy(base + w * 2 * TOPK + tl.arange(0, 2 * TOPK), view, [2 * TOPK])
    l0 = TOPK
    l1 = TOPK if SEGS > 1 else 0
    l2 = TOPK if SEGS > 2 else 0
    l3 = TOPK if SEGS > 3 else 0
    c0 = tl.zeros((), dtype=tl.int32)
    c1 = tl.zeros((), dtype=tl.int32)
    c2 = tl.zeros((), dtype=tl.int32)
    c3 = tl.zeros((), dtype=tl.int32)
    got = tl.zeros((), dtype=tl.int32)
    cons = tl.zeros([4], dtype=tl.int32)
    idx4 = tl.arange(0, 4)
    for _ in tl.static_range(0, 4):
        if got < TOPK:
            # The op only emits the 2*TOPK float prefix of its output (an exhaustion
            # merge over ways of TOPK cannot yield more), so the span is sized 2*TOPK.
            # The previous 8*TOPK span cost an extra 64 KB of UB at k=2048 and pushed
            # this kernel over the 192 KB budget (measured: 2*TOPK compiles and merges
            # correctly).
            out_t = tl.zeros([2 * TOPK], dtype=tl.float32)
            out_t, cons = tle.dsa.ascend.raw(
                "merge_exhaust_sort4",
                tle.dsa.to_tensor(in_ub),
                SEGS,
                c0,
                TOPK + c1,
                2 * TOPK + c2,
                3 * TOPK + c3,
                l0,
                l1,
                l2,
                l3,
                out=[out_t, cons],
            )
            cons0 = tl.sum(tl.where(idx4 == 0, cons, 0))
            cons1 = tl.sum(tl.where(idx4 == 1, cons, 0))
            cons2 = tl.sum(tl.where(idx4 == 2, cons, 0))
            cons3 = tl.sum(tl.where(idx4 == 3, cons, 0))
            take = tl.minimum(cons0 + cons1 + cons2 + cons3, TOPK - got)
            c0 += cons0
            c1 += cons1
            c2 += cons2
            c3 += cons3
            l0 -= cons0
            l1 -= cons1
            l2 -= cons2
            l3 -= cons3
            oo = tl.arange(0, 2 * TOPK)
            tl.store(base + got * 2 + oo, out_t, mask=oo < take * 2)
            got += take
    s_ub = tle.dsa.alloc([2 * TOPK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(base + tl.arange(0, 2 * TOPK), s_ub, [2 * TOPK])
    dval = tl.zeros([TOPK], dtype=tl.float32)
    didx = tl.zeros([TOPK], dtype=tl.int32)
    dval, didx = tle.dsa.ascend.raw(
        "unpack_sort", tle.dsa.to_tensor(s_ub), TOPK, out=[dval, didx]
    )
    tl.store(out_ptr + pid * TOPK + tl.arange(0, TOPK), didx)


def _persistent_topk_segmerge(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    k: int,
    max_seq_len: int,
    next_n: int,
) -> None:
    """Run the `segmerge` (segment sort + k-way merge) route for one batch of wide rows."""
    num_rows = logits.size(0)
    stride = logits.stride(0)
    segs = (max_seq_len + _SEGMERGE_SEG - 1) // _SEGMERGE_SEG
    props = torch.empty(
        (num_rows, segs, 2 * k), device=logits.device, dtype=torch.float32
    )
    tmp_sz = _segmerge_sort_tmp_size(_SEGMERGE_SEG, k, 1)
    _segmerge_seg_sort[(num_rows * segs,)](
        logits,
        seq_lens,
        props,
        stride,
        next_n,
        TOPK=k,
        SEGS=segs,
        SEG=_SEGMERGE_SEG,
        TMP_SZ=tmp_sz,
        SORT_IMPL=1,
    )
    _segmerge_merge_unpack[(num_rows,)](
        seq_lens, props, output, next_n, TOPK=k, SEGS=segs
    )


# ---------------------------------------------------------------------------
# route `segmerge_persistent`: persistent-form segment sort + multi-level merge
#
# Every kernel keeps a FIXED grid equal to the physical core count (40) and
# strides inside the core (`for task in range(pid, total, GRID)`), i.e. the
# persistent form:
#   kernel A          : stride over (row, segment) tasks, sort each 4096 segment
#                       with the native sort_1d_pack op -> TOPK proposals each
#   kernel B (levels) : stride over (row, group) tasks, 4-way exhaustion merge
#                       per group, shrinking segments -> groups -> ... -> 1
#   kernel B (final)  : stride over rows, merge the remaining ways, unpack indices
#
# Levels are separated by kernel boundaries (the do_bench_npu acceptance metric
# counts device kernel time only), which keeps every kernel in the persistent
# form without needing a cross-core barrier.
#
# Every proposal buffer carries a companion int32 "proposal count" (nv) buffer:
# nv[t] is the number of valid proposals in slot t (TOPK after a segment sort,
# 0 for a skipped slot, `got` after a merge).  The merge op is always called with
# per-way lengths taken from nv, so unwritten slack in a level buffer is never
# handed to the op as data.  Feeding uninitialised buffer slack into
# merge_exhaust_sort4 makes the vector core fault (507035) on this stack.
# ---------------------------------------------------------------------------

_PSEGMERGE_GRID = 40
_PSEGMERGE_FANOUT = 4
_PSEGMERGE_PROP_BUDGET = 32768  # max rows * segments kept in the proposal buffer
# k values the FULL-segment `segmerge_persistent` route can serve.  Both device
# sort ops were verified on device for every k they advertise (k=512/1024 via
# S4096_K129_512, k=2048 via S4096_K1_128_K2048), but a full 4096 segment at
# k=2048 needs 9*CEIL_FACTOR(4096,32) = 36864 floats of UB workspace (144 KiB,
# 147456 B) plus the 16 KiB src and 16 KiB seg_out buffers live in the same
# kernel, and that kernel does not compile inside the 192 KB UB ABI budget at any
# tmp size (bisected: 36864 down to 2048 floats all fail with `ub overflow`).
# k=2048 therefore takes the half-segment route below; k <= 1024 runs here.
_PSEGMERGE_KS = (512, 1024)
# Largest k served by the persistent route.  k > 1024 uses the half-segment route
# (`segmerge_persistent_half`), which fits because half a segment needs
# 9*CEIL_FACTOR(2048,32) = 18432 floats of UB workspace instead of 36864.
_PSEGMERGE_SORT_MAX_K = 2048
# Working-set budget, in bytes, for the host-side proposal buffers of the
# persistent routes.  Each route materialises `rows * per_row` bytes of GM
# scratch (the per_row formulas live in `_psegmerge_applicable` and
# `_psegmerge_half_applicable`), so this budget -- not the 192 KB per-AI-Core UB
# allowance, which does not depend on the row count at all -- is what limits how
# many rows stay on the fast route.  k > 1024 pays a large multiple of the
# k <= 1024 per-row cost, because every 4096 segment is sorted as two halves and
# each half still emits 2*k-wide proposals.
# 2 GiB covers 512 rows x 262144 at every supported k (that shape costs
# 3.7 MB/row at k=2048 => 582 rows fit); the previous 512 MB cap sent
# k=2048 shapes past ~146 rows to `radix_hist`, which is 25-55x slower there.
_PSEGMERGE_BYTE_BUDGET = 2 << 30


def _psegmerge_sort_impl(k: int) -> int:
    """Device sort path for this k (see CUSTOM_OP_USAGE.md '排序路径')."""
    if k <= 1024:
        return _PSEGMERGE_SORT_S4096_K129_512
    return _PSEGMERGE_SORT_S4096_K1_128_K2048


@triton.jit
def _psegmerge_seg_sort(
    x_ptr,
    lengths_ptr,
    props_ptr,
    nv_ptr,
    stride,
    next_n,
    total_tasks,
    nsegs,
    TOPK: tl.constexpr,
    SEG: tl.constexpr,
    GRID: tl.constexpr,
    TMP_SZ: tl.constexpr,
    SORT_IMPL: tl.constexpr,
):
    """Strided segment sort: one core loops over (row, segment) tasks.

    ``nv_ptr[t]`` publishes how many proposals slot ``t`` really holds: TOPK for
    a sorted segment of a wide row, 0 for every slot of a row whose effective
    length is <= TOPK (those rows are written directly by the trivial path).
    """
    pid = tl.program_id(0)
    src_ub = tle.dsa.alloc([SEG], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    seg_out = tle.dsa.alloc(
        [2 * TOPK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    tmp = tl.zeros([TMP_SZ], dtype=tl.float32)
    for t in tl.range(pid, total_tasks, GRID):
        row = t // nsegs
        seg = t % nsegs
        raw_len = tl.load(lengths_ptr + row)
        n_eff = tl.where(next_n > 1, raw_len - next_n + (row % next_n) + 1, raw_len)
        if n_eff > TOPK:
            seg_off = seg * SEG
            offs = seg_off + tl.arange(0, SEG)
            valid = offs < n_eff
            xv = tl.load(x_ptr + row * stride + offs, mask=valid, other=float("-inf"))
            tle.dsa.to_buffer(xv, space=tle.dsa.ascend.UB, bind_buffer=src_ub)
            tl.debug_barrier()
            props = tle.dsa.ascend.raw(
                "sort_1d_pack",
                tle.dsa.to_tensor(src_ub),
                tmp,
                True,
                TOPK,
                seg_off,
                SORT_IMPL,
                out=tle.dsa.to_tensor(seg_out),
            )
            tl.store(props_ptr + t * 2 * TOPK + tl.arange(0, 2 * TOPK), props)
            tl.store(nv_ptr + t, TOPK + tl.zeros((), dtype=tl.int32))
        else:
            tl.store(nv_ptr + t, tl.zeros((), dtype=tl.int32))


@triton.jit
def _psegmerge_half_seg_sort(
    x_ptr,
    lengths_ptr,
    props_ptr,
    nv_ptr,
    stride,
    next_n,
    total_tasks,
    halves,
    TOPK: tl.constexpr,
    SEG: tl.constexpr,
    HALF: tl.constexpr,
    GRID: tl.constexpr,
    TMP_SZ: tl.constexpr,
    SORT_IMPL: tl.constexpr,
):
    """Strided HALF-segment sort: one core loops over (row, half) tasks.

    The full-segment route for k > 1024 (``S4096_K1_128_K2048``) needs
    ``9 * CEIL_FACTOR(SEG, 32)`` floats of UB workspace and does not compile
    inside this kernel, so each 4096-element segment is fed to the sort op as
    two HALF-element halves.  A half still yields TOPK proposals, so the global
    top-TOPK of the segment is a 2-way merge of the two halves' proposals.

    Exactly one raw call per loop body: a second ``sort_1d_pack`` call inside the
    same ``tl.range`` body makes the Ascend backend crash, which is why the half
    (not the segment) is the task here.
    """
    pid = tl.program_id(0)
    src_ub = tle.dsa.alloc([HALF], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    seg_out = tle.dsa.alloc(
        [2 * TOPK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    tmp = tl.zeros([TMP_SZ], dtype=tl.float32)
    for t in tl.range(pid, total_tasks, GRID):
        row = t // halves
        half = t % halves
        raw_len = tl.load(lengths_ptr + row)
        n_eff = tl.where(next_n > 1, raw_len - next_n + (row % next_n) + 1, raw_len)
        seg_off = half * HALF
        # A half that covers any real data must still produce proposals: gating
        # on `n_eff > seg_off + TOPK` silently dropped the whole half whenever it
        # held fewer than TOPK elements, and nothing else writes those rows, so
        # the merge lost its smallest-ranked winner (measured: the k=2048 result
        # was exactly the true top-K shifted by one).
        if n_eff > seg_off:
            offs = seg_off + tl.arange(0, HALF)
            valid = offs < n_eff
            xv = tl.load(x_ptr + row * stride + offs, mask=valid, other=float("-inf"))
            tle.dsa.to_buffer(xv, space=tle.dsa.ascend.UB, bind_buffer=src_ub)
            tl.debug_barrier()
            props = tle.dsa.ascend.raw(
                "sort_1d_pack",
                tle.dsa.to_tensor(src_ub),
                tmp,
                True,
                TOPK,
                seg_off,
                SORT_IMPL,
                out=tle.dsa.to_tensor(seg_out),
            )
            tl.store(props_ptr + t * 2 * TOPK + tl.arange(0, 2 * TOPK), props)
            tl.store(nv_ptr + t, TOPK + tl.zeros((), dtype=tl.int32))
        else:
            tl.store(nv_ptr + t, tl.zeros((), dtype=tl.int32))


@triton.jit
def _psegmerge_half_merge(
    src_ptr,
    nv_src_ptr,
    src_slots,
    dst_ptr,
    nv_dst_ptr,
    dst_slots,
    total_tasks,
    TOPK: tl.constexpr,
    GRID: tl.constexpr,
):
    """Pairwise merge of a row's proposal slots: (2i, 2i+1) -> slot i.

    A row of the half-proposal buffer holds ``2*segs`` slots -- the two halves of
    every segment -- so the row is reduced by repeatedly pairing adjacent slots.
    Each step stays a two-way merge, which is what the per-AI-Core buffer budget
    allows at k=2048 (a single ``2*segs``-way merge needs 1575168 bits while only
    1572864 are available).

    The previous form indexed a task as one whole row but merged only two ways
    unconditionally, so everything past the first segment's halves was silently
    dropped: measured 1000/2048 overlap at k=2048 with segs=2, missing exactly
    the 1048 true top-K indices that live in the second segment.
    """
    pid = tl.program_id(0)
    in_ub = tle.dsa.alloc(
        [2 * 2 * TOPK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    out_ub = tle.dsa.alloc(
        [2 * 2 * TOPK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    cons_ub = tle.dsa.alloc([4], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
    io = tl.arange(0, 2 * 2 * TOPK)
    idx4 = tl.arange(0, 4)
    for t in tl.range(pid, total_tasks, GRID):
        row = t // dst_slots
        i = t % dst_slots
        a = 2 * i
        b = 2 * i + 1
        s_base = src_ptr + row * src_slots * 2 * TOPK
        n_base = nv_src_ptr + row * src_slots
        xv = tl.load(s_base + a * 2 * TOPK + io)
        tle.dsa.to_buffer(xv, space=tle.dsa.ascend.UB, bind_buffer=in_ub)
        tl.debug_barrier()
        l0 = tl.load(n_base + a)
        l1 = tl.where(b < src_slots, tl.load(n_base + b), 0)
        # One exhaustion merge stops as soon as a single way runs out, so it is
        # repeated with advancing cursors; the store is bounded so a round can
        # never write past the destination slot (2*TOPK floats).
        c0 = tl.zeros((), dtype=tl.int32)
        c1 = tl.zeros((), dtype=tl.int32)
        got = tl.zeros((), dtype=tl.int32)
        cons = tl.zeros([4], dtype=tl.int32)
        for _ in tl.static_range(0, 4):
            if got < TOPK and l0 + l1 > 0:
                merged, cons = tle.dsa.ascend.raw(
                    "merge_exhaust_sort4",
                    tle.dsa.to_tensor(in_ub),
                    2,
                    c0,
                    TOPK + c1,
                    0,
                    0,
                    l0,
                    l1,
                    0,
                    0,
                    out=[tle.dsa.to_tensor(out_ub), tle.dsa.to_tensor(cons_ub)],
                )
                cons0 = tl.sum(tl.where(idx4 == 0, cons, 0))
                cons1 = tl.sum(tl.where(idx4 == 1, cons, 0))
                take = tl.minimum(cons0 + cons1, TOPK - got)
                c0 += cons0
                c1 += cons1
                l0 -= cons0
                l1 -= cons1
                tl.store(
                    dst_ptr + (row * dst_slots + i) * 2 * TOPK + got * 2 + io,
                    merged,
                    mask=io < take * 2,
                )
                got += take
        tl.store(nv_dst_ptr + row * dst_slots + i, got)


@triton.jit
def _psegmerge_merge_level(
    src_ptr,
    nv_src_ptr,
    dst_ptr,
    nv_dst_ptr,
    nsrc,
    ngrp,
    total_tasks,
    TOPK: tl.constexpr,
    GRID: tl.constexpr,
):
    """One strided CTA per (row, group): 4-way exhaustion merge into TOPK slots.

    Way ``w`` of the group lives at proposal offset ``w * TOPK`` of ``src`` and
    holds ``nv_src[row * nsrc + w]`` valid proposals; only that prefix is ever
    described to the op.
    """
    pid = tl.program_id(0)
    in_ub = tle.dsa.alloc(
        [4 * 2 * TOPK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    for t in tl.range(pid, total_tasks, GRID):
        row = t // ngrp
        grp = t % ngrp
        first = grp * 4
        ways = tl.minimum(4, nsrc - first)
        base = src_ptr + row * nsrc * 2 * TOPK
        nv_base = nv_src_ptr + row * nsrc
        for w in tl.static_range(0, 4):
            if w < ways:
                view = tle.dsa.subview(in_ub, [w * 2 * TOPK], [2 * TOPK], [1])
                tle.dsa.copy(
                    base + (first + w) * 2 * TOPK + tl.arange(0, 2 * TOPK),
                    view,
                    [2 * TOPK],
                )
        tl.debug_barrier()
        wi = tl.minimum(first, nsrc - 1)
        l0 = tl.load(nv_base + wi)
        wi = tl.minimum(first + 1, nsrc - 1)
        l1 = tl.where(first + 1 < nsrc, tl.load(nv_base + wi), 0)
        wi = tl.minimum(first + 2, nsrc - 1)
        l2 = tl.where(first + 2 < nsrc, tl.load(nv_base + wi), 0)
        wi = tl.minimum(first + 3, nsrc - 1)
        l3 = tl.where(first + 3 < nsrc, tl.load(nv_base + wi), 0)
        c0 = tl.zeros((), dtype=tl.int32)
        c1 = tl.zeros((), dtype=tl.int32)
        c2 = tl.zeros((), dtype=tl.int32)
        c3 = tl.zeros((), dtype=tl.int32)
        got = tl.zeros((), dtype=tl.int32)
        cons = tl.zeros([4], dtype=tl.int32)
        idx4 = tl.arange(0, 4)
        dst_base = dst_ptr + (row * ngrp + grp) * 2 * TOPK
        for _ in tl.static_range(0, 4):
            if got < TOPK and l0 + l1 + l2 + l3 > 0:
                out_t = tl.zeros([2 * TOPK], dtype=tl.float32)
                out_t, cons = tle.dsa.ascend.raw(
                    "merge_exhaust_sort4",
                    tle.dsa.to_tensor(in_ub),
                    ways,
                    c0,
                    TOPK + c1,
                    2 * TOPK + c2,
                    3 * TOPK + c3,
                    l0,
                    l1,
                    l2,
                    l3,
                    out=[out_t, cons],
                )
                cons0 = tl.sum(tl.where(idx4 == 0, cons, 0))
                cons1 = tl.sum(tl.where(idx4 == 1, cons, 0))
                cons2 = tl.sum(tl.where(idx4 == 2, cons, 0))
                cons3 = tl.sum(tl.where(idx4 == 3, cons, 0))
                take = tl.minimum(cons0 + cons1 + cons2 + cons3, TOPK - got)
                c0 += cons0
                c1 += cons1
                c2 += cons2
                c3 += cons3
                l0 -= cons0
                l1 -= cons1
                l2 -= cons2
                l3 -= cons3
                oo = tl.arange(0, 2 * TOPK)
                tl.store(dst_base + got * 2 + oo, out_t, mask=oo < take * 2)
                got += take
        tl.store(nv_dst_ptr + t, got)


@triton.jit
def _psegmerge_merge_final(
    src_ptr,
    nv_src_ptr,
    scratch_ptr,
    out_ptr,
    nsrc,
    num_rows,
    TOPK: tl.constexpr,
    GRID: tl.constexpr,
):
    """Last level: strided over rows, merge the remaining ways and unpack indices.

    Rows whose ways are all empty (effective length <= TOPK, already written by
    the trivial path) are skipped: ``got`` stays 0 and nothing is stored.
    """
    pid = tl.program_id(0)
    # Only FANOUT ways are resident at a time: with TOPK == 2048 the previous
    # "all nsrc ways at once" form needed 4*2*TOPK + 8*TOPK floats and did not
    # fit the UB budget (measured ub overflow).  Ways are merged in groups of
    # four and each group's result is written back over its first way, so the
    # way count shrinks 9 -> 3 -> 1 while the buffer stays at 4 ways.
    in_ub = tle.dsa.alloc(
        [4 * 2 * TOPK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    oo = tl.arange(0, 2 * TOPK)
    for row in tl.range(pid, num_rows, GRID):
        row_base = src_ptr + row * nsrc * 2 * TOPK
        nv_base = nv_src_ptr + row * nsrc
        ng = (nsrc + 4 - 1) // 4
        for g in tl.range(0, ng):
            first = g * 4
            same = first == 0
            for w in tl.static_range(0, 4):
                if w < nsrc - first:
                    view = tle.dsa.subview(in_ub, [w * 2 * TOPK], [2 * TOPK], [1])
                    tle.dsa.copy(
                        row_base + (first + w) * 2 * TOPK + tl.arange(0, 2 * TOPK),
                        view,
                        [2 * TOPK],
                    )
            l0 = tl.where(first + 0 < nsrc, tl.load(nv_base + first + 0), 0)
            l1 = tl.where(first + 1 < nsrc, tl.load(nv_base + first + 1), 0)
            l2 = tl.where(first + 2 < nsrc, tl.load(nv_base + first + 2), 0)
            l3 = tl.where(first + 3 < nsrc, tl.load(nv_base + first + 3), 0)
            c0 = tl.zeros((), dtype=tl.int32)
            c1 = tl.zeros((), dtype=tl.int32)
            c2 = tl.zeros((), dtype=tl.int32)
            c3 = tl.zeros((), dtype=tl.int32)
            got = tl.zeros((), dtype=tl.int32)
            cons = tl.zeros([4], dtype=tl.int32)
            idx4 = tl.arange(0, 4)
            for _ in tl.static_range(0, 4):
                if got < TOPK and l0 + l1 + l2 + l3 > 0:
                    out_t = tl.zeros([2 * TOPK], dtype=tl.float32)
                    out_t, cons = tle.dsa.ascend.raw(
                        "merge_exhaust_sort4",
                        tle.dsa.to_tensor(in_ub),
                        4,
                        c0,
                        TOPK + c1,
                        2 * TOPK + c2,
                        3 * TOPK + c3,
                        l0,
                        l1,
                        l2,
                        l3,
                        out=[out_t, cons],
                    )
                    cons0 = tl.sum(tl.where(idx4 == 0, cons, 0))
                    cons1 = tl.sum(tl.where(idx4 == 1, cons, 0))
                    cons2 = tl.sum(tl.where(idx4 == 2, cons, 0))
                    cons3 = tl.sum(tl.where(idx4 == 3, cons, 0))
                    take = tl.minimum(cons0 + cons1 + cons2 + cons3, TOPK - got)
                    c0 += cons0
                    c1 += cons1
                    c2 += cons2
                    c3 += cons3
                    l0 -= cons0
                    l1 -= cons1
                    l2 -= cons2
                    l3 -= cons3
                    tl.store(
                        row_base + (first * 2 + got * 2) + oo,
                        out_t,
                        mask=oo < take * 2,
                    )
                    got += take
            if same:
                tl.store(nv_base, got)
            else:
                tl.store(nv_base + first, got)
        # final pass over the (at most FANOUT) merged groups, then unpack
        first = 0
        for w in tl.static_range(0, 4):
            if w < ng:
                view = tle.dsa.subview(in_ub, [w * 2 * TOPK], [2 * TOPK], [1])
                tle.dsa.copy(
                    row_base + (first + w) * 2 * TOPK + tl.arange(0, 2 * TOPK),
                    view,
                    [2 * TOPK],
                )
        l0 = tl.load(nv_base + 0)
        l1 = tl.where(ng > 1, tl.load(nv_base + 1), 0)
        l2 = tl.where(ng > 2, tl.load(nv_base + 2), 0)
        l3 = tl.where(ng > 3, tl.load(nv_base + 3), 0)
        c0 = tl.zeros((), dtype=tl.int32)
        c1 = tl.zeros((), dtype=tl.int32)
        c2 = tl.zeros((), dtype=tl.int32)
        c3 = tl.zeros((), dtype=tl.int32)
        got = tl.zeros((), dtype=tl.int32)
        cons = tl.zeros([4], dtype=tl.int32)
        idx4 = tl.arange(0, 4)
        row_scratch = scratch_ptr + row * 2 * TOPK
        for _ in tl.static_range(0, 4):
            if got < TOPK and l0 + l1 + l2 + l3 > 0:
                out_t = tl.zeros([2 * TOPK], dtype=tl.float32)
                out_t, cons = tle.dsa.ascend.raw(
                    "merge_exhaust_sort4",
                    tle.dsa.to_tensor(in_ub),
                    ng,
                    c0,
                    TOPK + c1,
                    2 * TOPK + c2,
                    3 * TOPK + c3,
                    l0,
                    l1,
                    l2,
                    l3,
                    out=[out_t, cons],
                )
                cons0 = tl.sum(tl.where(idx4 == 0, cons, 0))
                cons1 = tl.sum(tl.where(idx4 == 1, cons, 0))
                cons2 = tl.sum(tl.where(idx4 == 2, cons, 0))
                cons3 = tl.sum(tl.where(idx4 == 3, cons, 0))
                take = tl.minimum(cons0 + cons1 + cons2 + cons3, TOPK - got)
                c0 += cons0
                c1 += cons1
                c2 += cons2
                c3 += cons3
                l0 -= cons0
                l1 -= cons1
                l2 -= cons2
                l3 -= cons3
                tl.store(row_scratch + got * 2 + oo, out_t, mask=oo < take * 2)
                got += take
        if got > 0:
            acc = tle.dsa.subview(in_ub, [0], [2 * TOPK], [1])
            tle.dsa.copy(row_scratch + tl.arange(0, 2 * TOPK), acc, [2 * TOPK])
            tl.debug_barrier()
            dval = tl.zeros([TOPK], dtype=tl.float32)
            didx = tl.zeros([TOPK], dtype=tl.int32)
            dval, didx = tle.dsa.ascend.raw(
                "unpack_sort", tle.dsa.to_tensor(acc), TOPK, out=[dval, didx]
            )
            pos = tl.arange(0, TOPK)
            tl.store(out_ptr + row * TOPK + pos, tl.where(pos < got, didx, -1))


def _psegmerge_applicable(num_rows: int, max_seq_len: int, k: int) -> bool:
    """`segmerge_persistent` covers the k values the device sort ops offer, long
    rows, and shapes whose materialised proposal set fits the byte budget."""
    if k not in _PSEGMERGE_KS:
        return False
    segs = (max_seq_len + _SEGMERGE_SEG - 1) // _SEGMERGE_SEG
    if segs <= _SEGMERGE_MAX_SEGS:
        return False  # short rows stay on the `segmerge` route
    groups = (segs + _PSEGMERGE_FANOUT - 1) // _PSEGMERGE_FANOUT
    # proposals + ping-pong level buffers + their int32 nv companions, plus the
    # (num_rows, 2*k) scratch the tail merge uses to stage one row's result
    per_row = (segs + 2 * groups) * 2 * k * 4 + (segs + 2 * groups) * 4 + 2 * k * 4
    return (
        num_rows * segs <= _PSEGMERGE_PROP_BUDGET
        and num_rows * per_row <= _PSEGMERGE_BYTE_BUDGET
    )


def _psegmerge_half_applicable(num_rows: int, max_seq_len: int, k: int) -> bool:
    """May the k > 1024 half-segment route serve this shape?

    Costs one extra short-lived buffer (``props``) plus the count buffer; the
    merged result reuses the caller-sized ``props`` slot, so the byte budget has
    to cover both.  Also needs at least two full segments per row, since the
    route splits each 4096 segment into halves.
    """
    if k <= _SEGMERGE_K or k > _PSEGMERGE_SORT_MAX_K:
        return False
    segs = (max_seq_len + _SEGMERGE_SEG - 1) // _SEGMERGE_SEG
    if segs <= _SEGMERGE_MAX_SEGS:
        return False  # short rows keep the established `radix_hist` behaviour
    groups = (segs + _PSEGMERGE_FANOUT - 1) // _PSEGMERGE_FANOUT
    per_row_half = 2 * segs * 2 * k * 4 + 2 * segs * 4
    per_row_merge = (segs + 2 * groups) * 2 * k * 4 + (segs + 2 * groups) * 4
    # one (num_rows, 2*k) scratch is allocated once and shared by both loops
    per_row_scratch = 2 * k * 4
    return (
        num_rows * (per_row_half + per_row_merge + per_row_scratch)
        <= _PSEGMERGE_BYTE_BUDGET
    )


def _persistent_topk_segmerge_persistent_half(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    k: int,
    max_seq_len: int,
    next_n: int,
) -> None:
    """`segmerge_persistent_half`: for k > 1024, sort each 4096 segment as two
    halves, then merge the two HALF proposal streams back into the segment slot.

    The full-segment sort route for k > 1024 needs 144 KB of UB workspace (measured
    on device, and a minimal single-CTA probe of the same sort overflows too),
    while the half route needs 72 KB and only ever writes 2*k floats per sort.
    One extra merge launch buys back the k > 1024 coverage that otherwise runs
    the `radix_hist` route at 0.7-5% of the torch.topk baseline.
    """
    num_rows = logits.size(0)
    stride = logits.stride(0)
    segs = (max_seq_len + _SEGMERGE_SEG - 1) // _SEGMERGE_SEG
    halves = 2 * segs
    half_len = _SEGMERGE_SEG // 2
    props = torch.empty(
        (num_rows, halves, 2 * k), device=logits.device, dtype=torch.float32
    )
    nv_half = torch.empty((num_rows, halves), device=logits.device, dtype=torch.int32)
    first_groups = (segs + _PSEGMERGE_FANOUT - 1) // _PSEGMERGE_FANOUT
    buf_a = torch.empty(
        (num_rows, first_groups, 2 * k), device=logits.device, dtype=torch.float32
    )
    buf_b = torch.empty(
        (num_rows, first_groups, 2 * k), device=logits.device, dtype=torch.float32
    )
    nv_a = torch.empty(
        (num_rows, first_groups), device=logits.device, dtype=torch.int32
    )
    nv_b = torch.empty(
        (num_rows, first_groups), device=logits.device, dtype=torch.int32
    )
    nv_seg = torch.empty((num_rows, segs), device=logits.device, dtype=torch.int32)
    seg_props = torch.empty(
        (num_rows, segs, 2 * k), device=logits.device, dtype=torch.float32
    )
    scratch = torch.empty((num_rows, 2 * k), device=logits.device, dtype=torch.float32)
    tmp_sz = _psegmerge_sort_tmp_size(half_len, k, _PSEGMERGE_SORT_S4096_K1_128_K2048)
    _psegmerge_half_seg_sort[(_PSEGMERGE_GRID,)](
        logits,
        seq_lens,
        props,
        nv_half,
        stride,
        next_n,
        num_rows * halves,
        halves,
        TOPK=k,
        SEG=_SEGMERGE_SEG,
        HALF=half_len,
        GRID=_PSEGMERGE_GRID,
        TMP_SZ=tmp_sz,
        SORT_IMPL=_PSEGMERGE_SORT_S4096_K1_128_K2048,
    )
    # the merge reads both halves of a row from props and writes the segment
    # slots into a separate buffer (a strided view of props would have the
    # neighbouring halves overwrite each other).
    # Reduce the row's 2*segs half slots to one slot (the segment layout the
    # rest of the pipeline expects) by pairing adjacent slots, repeatedly.  One
    # launch per level; every merge stays two-way.
    cur, nv_cur, cur_slots = props, nv_half, halves
    while cur_slots > segs:
        nxt = (cur_slots + 1) // 2
        if nxt >= cur_slots:
            break
        # ping-pong through the caller-sized buffers, writing the final level
        # straight into seg_props
        if nxt == segs:
            dst, nv_dst = seg_props, nv_seg
        else:
            dst, nv_dst = (buf_a, nv_a) if cur is props else (buf_b, nv_b)
        _psegmerge_half_merge[(_PSEGMERGE_GRID,)](
            cur,
            nv_cur,
            cur_slots,
            dst,
            nv_dst,
            nxt,
            num_rows * nxt,
            TOPK=k,
            GRID=_PSEGMERGE_GRID,
        )
        cur, nv_cur, cur_slots = dst, nv_dst, nxt
    cur, nv_cur, nsrc, use_a = seg_props, nv_seg, segs, True
    while nsrc > _PSEGMERGE_FANOUT:
        ngrp = (nsrc + _PSEGMERGE_FANOUT - 1) // _PSEGMERGE_FANOUT
        dst = buf_a if use_a else buf_b
        nv_dst = nv_a if use_a else nv_b
        _psegmerge_merge_level[(_PSEGMERGE_GRID,)](
            cur,
            nv_cur,
            dst,
            nv_dst,
            nsrc,
            ngrp,
            num_rows * ngrp,
            TOPK=k,
            GRID=_PSEGMERGE_GRID,
        )
        cur, nv_cur, nsrc, use_a = dst, nv_dst, ngrp, not use_a
    _psegmerge_merge_final[(_PSEGMERGE_GRID,)](
        cur, nv_cur, scratch, output, nsrc, num_rows, TOPK=k, GRID=_PSEGMERGE_GRID
    )


def _persistent_topk_segmerge_persistent(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    k: int,
    max_seq_len: int,
    next_n: int,
) -> None:
    """Run the persistent (fixed grid + in-core striding) path for long rows."""
    num_rows = logits.size(0)
    stride = logits.stride(0)
    segs = (max_seq_len + _SEGMERGE_SEG - 1) // _SEGMERGE_SEG
    props = torch.empty(
        (num_rows, segs, 2 * k), device=logits.device, dtype=torch.float32
    )
    nv_props = torch.empty((num_rows, segs), device=logits.device, dtype=torch.int32)
    first_groups = (segs + _PSEGMERGE_FANOUT - 1) // _PSEGMERGE_FANOUT
    buf_a = torch.empty(
        (num_rows, first_groups, 2 * k), device=logits.device, dtype=torch.float32
    )
    buf_b = torch.empty(
        (num_rows, first_groups, 2 * k), device=logits.device, dtype=torch.float32
    )
    nv_a = torch.empty(
        (num_rows, first_groups), device=logits.device, dtype=torch.int32
    )
    nv_b = torch.empty(
        (num_rows, first_groups), device=logits.device, dtype=torch.int32
    )
    scratch = torch.empty((num_rows, 2 * k), device=logits.device, dtype=torch.float32)
    sort_impl = _psegmerge_sort_impl(k)
    tmp_sz = _psegmerge_sort_tmp_size(_SEGMERGE_SEG, k, sort_impl)
    _psegmerge_seg_sort[(_PSEGMERGE_GRID,)](
        logits,
        seq_lens,
        props,
        nv_props,
        stride,
        next_n,
        num_rows * segs,
        segs,
        TOPK=k,
        SEG=_SEGMERGE_SEG,
        GRID=_PSEGMERGE_GRID,
        TMP_SZ=tmp_sz,
        SORT_IMPL=sort_impl,
    )
    cur, nv_cur, nsrc, use_a = props, nv_props, segs, True
    while nsrc > _PSEGMERGE_FANOUT:
        ngrp = (nsrc + _PSEGMERGE_FANOUT - 1) // _PSEGMERGE_FANOUT
        dst = buf_a if use_a else buf_b
        nv_dst = nv_a if use_a else nv_b
        _psegmerge_merge_level[(_PSEGMERGE_GRID,)](
            cur,
            nv_cur,
            dst,
            nv_dst,
            nsrc,
            ngrp,
            num_rows * ngrp,
            TOPK=k,
            GRID=_PSEGMERGE_GRID,
        )
        cur, nv_cur, nsrc, use_a = dst, nv_dst, ngrp, not use_a
    _psegmerge_merge_final[(_PSEGMERGE_GRID,)](
        cur, nv_cur, scratch, output, nsrc, num_rows, TOPK=k, GRID=_PSEGMERGE_GRID
    )


def persistent_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    k: int = 512,
    max_seq_len: int | None = None,
) -> None:
    # --- metadata-only host: asserts, views, kernel launches ---
    assert logits.device.type == "npu", "persistent_topk: logits must be NPU tensor"
    assert lengths.device.type == "npu", "persistent_topk: lengths must be NPU tensor"
    assert output.device.type == "npu", "persistent_topk: output must be NPU tensor"
    assert logits.dtype == torch.float32, "persistent_topk: only float32 supported"
    assert lengths.dtype == torch.int32, "persistent_topk: lengths must be int32"
    assert output.dtype == torch.int32, "persistent_topk: output must be int32"
    assert logits.dim() == 2, "persistent_topk: logits must be 2D"
    assert lengths.dim() in (1, 2), "persistent_topk: lengths must be 1D or 2D"
    assert lengths.is_contiguous(), "persistent_topk: lengths must be contiguous"
    assert output.dim() == 2, "persistent_topk: output must be 2D"
    assert k in (512, 1024, 2048), f"persistent_topk supports k=512/1024/2048, got {k}"

    num_rows = logits.size(0)
    stride = logits.stride(0)
    next_n = 1 if lengths.dim() == 1 else lengths.size(1)
    seq_lens = lengths if lengths.dim() == 1 else lengths.view(-1)
    assert seq_lens.numel() == num_rows
    assert output.size(0) == num_rows and output.size(1) == k

    # trivial rows (effective length <= k): direct write, decided in-kernel
    _radix_hist_trivial_all[(num_rows,)](
        seq_lens, output, stride, next_n, TOPK=k, BLOCK=_BLOCK
    )

    # --- route `segmerge`: segment sort + k-way merge, short rows ---
    # max_seq_len may exceed the physical row width; clamp so segments cover real data only
    msl = min(max_seq_len or stride, logits.size(1))
    segs = (msl + _SEGMERGE_SEG - 1) // _SEGMERGE_SEG
    # FLAGGEMS_ASCEND_PTK_PATH = auto (default) | segmerge | radix
    route_mode = os.environ.get(_SEGMERGE_ENV, "auto").lower()
    # --- route `segmerge_persistent`: persistent segment sort + fan-in merge ---
    if (
        route_mode != "radix"
        and _PSEGMERGE_STATE["usable"]
        and _psegmerge_applicable(num_rows, msl, k)
        and _segmerge_probe()
    ):
        try:
            _persistent_topk_segmerge_persistent(
                logits, seq_lens, output, k, msl, next_n
            )
            return
        except Exception as exc:  # compile/launch failure -> permanent fallback
            _PSEGMERGE_STATE["usable"] = False
            if route_mode == "segmerge":
                raise
            warnings.warn(
                "[flaggems_vllm] Ascend persistent_topk route "
                "[segment-sort + merge, persistent] disabled "
                f"({type(exc).__name__}: {exc}); using the radix-histogram route",
                stacklevel=2,
            )
    # --- route `segmerge_persistent_half`: k > 1024, half a segment per sort task ---
    if (
        route_mode != "radix"
        and _PSEGMERGE_STATE["usable"]
        and _psegmerge_half_applicable(num_rows, msl, k)
        and _segmerge_probe()
    ):
        try:
            _persistent_topk_segmerge_persistent_half(
                logits, seq_lens, output, k, msl, next_n
            )
            return
        except Exception as exc:  # compile/launch failure -> permanent fallback
            _PSEGMERGE_STATE["usable"] = False
            if route_mode == "segmerge":
                raise
            warnings.warn(
                "[flaggems_vllm] Ascend persistent_topk route "
                "[segment-sort + merge, persistent, half-segment] disabled "
                f"({type(exc).__name__}: {exc}); using the radix-histogram route",
                stacklevel=2,
            )
    if (
        route_mode != "radix"
        and k == _SEGMERGE_K
        and segs <= _SEGMERGE_MAX_SEGS
        and _segmerge_probe()
    ):
        try:
            _persistent_topk_segmerge(logits, seq_lens, output, k, msl, next_n)
            return
        except Exception as exc:  # compile/launch failure -> permanent fallback
            _SEGMERGE_STATE["usable"] = False
            if route_mode == "segmerge":
                raise
            warnings.warn(
                "[flaggems_vllm] Ascend persistent_topk route "
                "[segment-sort + merge] disabled "
                f"({type(exc).__name__}: {exc}); using the radix-histogram route",
                stacklevel=2,
            )

    # --- route `radix_hist`: multi-block radix histogram over all rows ---
    # (kernels skip trivial rows)
    G = _G

    # workspace layout (reinterpret view: no data movement)
    ws = workspace.view(torch.int32)
    per_row_slots = _HIST_SLOTS + 6 + 4 * G + 2
    Bmax = max(1, ws.numel() // per_row_slots)
    hist_r = ws.narrow(0, 0, Bmax * _HIST_SLOTS)
    t1_r = ws.narrow(0, Bmax * _HIST_SLOTS, Bmax)
    f1_r = ws.narrow(0, Bmax * _HIST_SLOTS + Bmax, Bmax)
    t2_r = ws.narrow(0, Bmax * _HIST_SLOTS + 2 * Bmax, Bmax)
    f2_r = ws.narrow(0, Bmax * _HIST_SLOTS + 3 * Bmax, Bmax)
    t3_r = ws.narrow(0, Bmax * _HIST_SLOTS + 4 * Bmax, Bmax)
    f3_r = ws.narrow(0, Bmax * _HIST_SLOTS + 5 * Bmax, Bmax)
    o = Bmax * (_HIST_SLOTS + 6)
    cnt_r = ws.narrow(0, o, Bmax * G * 2).view(Bmax, G * 2)
    o += Bmax * G * 2
    base_r = ws.narrow(0, o, Bmax * G * 2).view(Bmax, G * 2)
    o += Bmax * G * 2
    found_r = ws.narrow(0, o, Bmax * 2).view(Bmax, 2)

    for b_start in range(0, num_rows, Bmax):
        b = min(Bmax, num_rows - b_start)
        hist_b = hist_r[: b * _HIST_SLOTS]
        t1b = t1_r[:b]
        f1b = f1_r[:b]
        t2b = t2_r[:b]
        f2b = f2_r[:b]
        t3b = t3_r[:b]
        f3b = f3_r[:b]
        cnt_b = cnt_r[:b]
        base_b = base_r[:b]
        found_b = found_r[:b]
        hist_slots = b * _HIST_SLOTS

        # step 1: hist (top-11 bits) -> threshold t1 / cumulative f1
        _radix_hist_clear[((hist_slots + _BLOCK - 1) // _BLOCK,)](
            hist_b, hist_slots, BLOCK=_BLOCK
        )
        _radix_hist_hist[(b, G)](
            logits,
            seq_lens,
            hist_b,
            t1b,
            t2b,
            stride,
            next_n,
            k,
            STEP=1,
            SHIFT=21,
            BIN_MASK=0x7FF,
            NUM_BINS=4096,
            PREF_SHIFT=0,
            PREF_MASK=0,
            G=G,
            BLOCK=_BLOCK,
        )
        _radix_hist_threshold[(b,)](
            hist_b, t1b, f1b, t1b, STEP=1, NUM_BINS=4096, TOPK=k, BLOCK=_BLOCK
        )

        # step 2: hist (top11 == t1, next 11 bits) -> t2 / f2 (base = f1)
        _radix_hist_clear[((hist_slots + _BLOCK - 1) // _BLOCK,)](
            hist_b, hist_slots, BLOCK=_BLOCK
        )
        _radix_hist_hist[(b, G)](
            logits,
            seq_lens,
            hist_b,
            t1b,
            t2b,
            stride,
            next_n,
            k,
            STEP=2,
            SHIFT=10,
            BIN_MASK=0x7FF,
            NUM_BINS=4096,
            PREF_SHIFT=21,
            PREF_MASK=0x7FF,
            G=G,
            BLOCK=_BLOCK,
        )
        _radix_hist_threshold[(b,)](
            hist_b, t2b, f2b, f1b, STEP=2, NUM_BINS=4096, TOPK=k, BLOCK=_BLOCK
        )

        # step 3: hist (top21 == p21, low 10 bits) -> t3 / f3 (base = f2)
        _radix_hist_clear[((hist_slots + _BLOCK - 1) // _BLOCK,)](
            hist_b, hist_slots, BLOCK=_BLOCK
        )
        _radix_hist_hist[(b, G)](
            logits,
            seq_lens,
            hist_b,
            t1b,
            t2b,
            stride,
            next_n,
            k,
            STEP=3,
            SHIFT=0,
            BIN_MASK=0x3FF,
            NUM_BINS=2048,
            PREF_SHIFT=10,
            PREF_MASK=0x3FFFFF,
            G=G,
            BLOCK=_BLOCK,
        )
        _radix_hist_threshold[(b,)](
            hist_b, t3b, f3b, f2b, STEP=3, NUM_BINS=2048, TOPK=k, BLOCK=_BLOCK
        )

        # output: count (guaranteed/final) -> prefix -> two light writes
        _radix_hist_count[(b, G)](
            logits, seq_lens, cnt_b, t1b, t2b, t3b, stride, next_n, k, BLOCK=_BLOCK, G=G
        )
        _radix_hist_prefix[(b,)](cnt_b, base_b, found_b, G=G, BLOCK=_BLOCK)
        _radix_hist_write_guar[(b, G)](
            logits,
            seq_lens,
            output,
            base_b,
            t1b,
            t2b,
            t3b,
            stride,
            next_n,
            k,
            BLOCK=_BLOCK,
            G=G,
        )
        _radix_hist_write_final[(b, G)](
            logits,
            seq_lens,
            output,
            base_b,
            found_b,
            t1b,
            t2b,
            t3b,
            stride,
            next_n,
            k,
            BLOCK=_BLOCK,
            G=G,
        )
