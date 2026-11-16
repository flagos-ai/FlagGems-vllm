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
"""MetaX block top-k for select_candidate_blocks_cooperative.

A self-contained, pure-Triton radix-select top-k over the per-row block scores,
written for the MetaX C550. It replaces persistent_topk on the metax backend:
the metax persistent_topk cooperative path relies on TLE shared-memory atomics
(tt.atomic_rmw on addrspace-3) that the metax FlagTree build does not compile,
so this op selects candidates with no TLE shared-memory alloc or shared atomics
-- one CTA per row, tl.histogram-based radix select, global stores only.

Contract matches persistent_topk: for each row it writes the ids of the k
largest finite block scores into out (-1 padded, unsorted). The score kernel has
already pinned the newest block to +inf and set dropped blocks to -inf, which
are excluded here.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mono_key(x):
    # Order-preserving fp32 -> uint32 key (NaN and +inf sort highest).
    ui = x.to(tl.uint32, bitcast=True)
    sign = ui >> 31
    flip = tl.where(
        sign == 0,
        tl.full(x.shape, 0x80000000, tl.uint32),
        tl.full(x.shape, 0xFFFFFFFF, tl.uint32),
    )
    return ui ^ flip


@triton.jit
def _block_topk_kernel(
    scores_ptr,
    lengths_ptr,
    out_ptr,
    stride_row,
    k,
    num_tiles,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(lengths_ptr + row)
    base = row * stride_row
    NEG = float("-inf")
    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint32)
    # Rows with fewer than k finite blocks keep them all and pad with -1.
    total_valid = 0
    for t in tl.range(0, num_tiles):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        m = offs < length
        x = tl.load(scores_ptr + base + offs, mask=m, other=NEG)
        total_valid += tl.sum((m & (x != NEG)).to(tl.int32))
    k_eff = tl.minimum(k, total_valid)
    prefix = tl.zeros((), dtype=tl.uint32)
    k_rem = k_eff
    # Radix select: 4 passes of 8 bits, MSB first, to find the k-th largest key.
    for p in tl.static_range(0, 4):
        shift = 24 - 8 * p
        hist = tl.zeros((256,), dtype=tl.int32)
        corr = 0
        for t in tl.range(0, num_tiles):
            offs = t * BLOCK + tl.arange(0, BLOCK)
            m = offs < length
            x = tl.load(scores_ptr + base + offs, mask=m, other=NEG)
            sel = m & (x != NEG)
            key = _mono_key(x)
            if p == 0:
                inmatch = sel
            else:
                inmatch = sel & ((key >> (shift + 8)) == (prefix >> (shift + 8)))
            digit = ((key >> shift) & 0xFF).to(tl.int32)
            # Dump non-participants into bin 0, then subtract their count back.
            digit = tl.where(inmatch, digit, 0)
            hist += tl.histogram(digit, 256)
            corr += tl.sum((~inmatch).to(tl.int32))
        hist = hist - tl.where(bins == 0, corr, 0)
        suffix_incl = tl.sum(hist) - tl.cumsum(hist) + hist
        selb = tl.max(tl.where(suffix_incl >= k_rem, bins, -1))
        hist_sel = tl.sum(tl.where(bins == selb, hist, 0))
        suf_sel = tl.sum(tl.where(bins == selb, suffix_incl, 0))
        counts_above = suf_sel - hist_sel
        prefix = prefix | (selb.to(tl.uint32) << shift)
        k_rem = k_rem - counts_above
    # Gather: all ids with key > threshold, then fill with key == threshold.
    pos = 0
    for t in tl.range(0, num_tiles):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        m = offs < length
        x = tl.load(scores_ptr + base + offs, mask=m, other=NEG)
        sel = m & (x != NEG)
        key = _mono_key(x)
        gt = sel & (key > prefix)
        gti = gt.to(tl.int32)
        wp = pos + tl.cumsum(gti) - gti
        tl.store(out_ptr + row * k + wp, offs.to(tl.int32), mask=gt & (wp < k))
        pos += tl.sum(gti)
    need = k_eff - pos
    taken = 0
    for t in tl.range(0, num_tiles):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        m = offs < length
        x = tl.load(scores_ptr + base + offs, mask=m, other=NEG)
        sel = m & (x != NEG)
        key = _mono_key(x)
        eq = sel & (key == prefix)
        eqi = eq.to(tl.int32)
        rank = taken + tl.cumsum(eqi) - eqi
        take = eq & (rank < need)
        wp = pos + rank
        tl.store(out_ptr + row * k + wp, offs.to(tl.int32), mask=take & (wp < k))
        taken += tl.sum(eqi)


# One CTA per row streams the row in BLOCK-wide tiles; 2048 keeps the histogram
# reductions in registers.
_BLOCK = 2048


def block_topk_metax(scores, lengths, out, k, workspace):
    # scores [rows, nblocks] fp32, lengths [rows] int32, out [rows, k] int32 (-1).
    import torch  # noqa: F401  (workspace.view dtype)

    rows, nblocks = scores.shape
    num_tiles = (nblocks + _BLOCK - 1) // _BLOCK
    # Thin batch x large nblocks: one CTA per row leaves most SMs idle, so split
    # each row across G cooperating CTAs (global-memory histogram + gather, no
    # TLE). Elsewhere the single-CTA path avoids the cooperative launch overhead.
    if num_tiles >= 8 and rows <= 32:
        G = max(1, min(max(1, 64 // rows), num_tiles))
        _block_topk_coop(scores, lengths, out, k, G, workspace)
    else:
        _block_topk_kernel[(rows,)](
            scores, lengths, out, scores.stride(0), k, num_tiles, BLOCK=_BLOCK
        )
    return out


# ---------------------------------------------------------------------------
# Cooperative multi-CTA path: for thin batch x large nblocks, one CTA per row
# cannot fill the SMs, so split each row across G CTAs that cooperate through
# global memory (atomic histogram + atomic gather positions) with kernel
# boundaries as the grid barrier. No TLE shared-memory alloc or shared atomics.
# ---------------------------------------------------------------------------


@triton.jit
def _coop_hist_kernel(scores_ptr, lengths_ptr, prefix_ptr, hist_ptr,
                      stride_row, num_tiles, G, p: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    g = tl.program_id(1)
    length = tl.load(lengths_ptr + row)
    base = row * stride_row
    NEG = float("-inf")
    shift = 24 - 8 * p
    prefix = tl.load(prefix_ptr + row).to(tl.uint32, bitcast=True)
    bins = tl.arange(0, 256)
    lh = tl.zeros((256,), tl.int32)
    corr = 0
    ntloc = tl.cdiv(tl.maximum(num_tiles - g, 0), G)
    for i in tl.range(0, ntloc):
        t = g + i * G
        offs = t * BLOCK + tl.arange(0, BLOCK)
        m = offs < length
        x = tl.load(scores_ptr + base + offs, mask=m, other=NEG)
        sel = m & (x != NEG)
        key = _mono_key(x)
        if p == 0:
            inmatch = sel
        else:
            inmatch = sel & ((key >> (shift + 8)) == (prefix >> (shift + 8)))
        digit = ((key >> shift) & 0xFF).to(tl.int32)
        digit = tl.where(inmatch, digit, 0)
        lh += tl.histogram(digit, 256)
        corr += tl.sum((~inmatch).to(tl.int32))
    lh = lh - tl.where(bins == 0, corr, 0)
    tl.atomic_add(hist_ptr + row * 256 + bins, lh, sem="relaxed", scope="gpu")


@triton.jit
def _coop_reduce_kernel(hist_ptr, prefix_ptr, krem_ptr, keff_ptr, k, p: tl.constexpr):
    row = tl.program_id(0)
    bins = tl.arange(0, 256)
    hist = tl.load(hist_ptr + row * 256 + bins)
    shift = 24 - 8 * p
    if p == 0:
        keff = tl.minimum(k, tl.sum(hist))
        k_rem = keff
        tl.store(keff_ptr + row, keff)
    else:
        k_rem = tl.load(krem_ptr + row)
    prefix = tl.load(prefix_ptr + row).to(tl.uint32, bitcast=True)
    suffix_incl = tl.sum(hist) - tl.cumsum(hist) + hist
    selb = tl.max(tl.where(suffix_incl >= k_rem, bins, -1))
    hist_sel = tl.sum(tl.where(bins == selb, hist, 0))
    suf_sel = tl.sum(tl.where(bins == selb, suffix_incl, 0))
    counts_above = suf_sel - hist_sel
    prefix = prefix | (selb.to(tl.uint32) << shift)
    k_rem = k_rem - counts_above
    tl.store(prefix_ptr + row, prefix.to(tl.int32, bitcast=True))
    tl.store(krem_ptr + row, k_rem)
    tl.store(hist_ptr + row * 256 + bins, tl.zeros((256,), tl.int32))


@triton.jit
def _coop_gather_gt_kernel(scores_ptr, lengths_ptr, prefix_ptr, out_ptr,
                           posgt_ptr, stride_row, num_tiles, G, k,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    g = tl.program_id(1)
    length = tl.load(lengths_ptr + row)
    base = row * stride_row
    NEG = float("-inf")
    prefix = tl.load(prefix_ptr + row).to(tl.uint32, bitcast=True)
    ntloc = tl.cdiv(tl.maximum(num_tiles - g, 0), G)
    for i in tl.range(0, ntloc):
        t = g + i * G
        offs = t * BLOCK + tl.arange(0, BLOCK)
        m = offs < length
        x = tl.load(scores_ptr + base + offs, mask=m, other=NEG)
        key = _mono_key(x)
        gt = (m & (x != NEG)) & (key > prefix)
        gti = gt.to(tl.int32)
        bg = tl.atomic_add(posgt_ptr + row, tl.sum(gti), sem="relaxed", scope="gpu")
        wp = bg + tl.cumsum(gti) - gti
        tl.store(out_ptr + row * k + wp, offs.to(tl.int32), mask=gt & (wp < k))


@triton.jit
def _coop_gather_eq_kernel(scores_ptr, lengths_ptr, prefix_ptr, keff_ptr,
                           posgt_ptr, out_ptr, poseq_ptr, stride_row, num_tiles,
                           G, k, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    g = tl.program_id(1)
    length = tl.load(lengths_ptr + row)
    base = row * stride_row
    NEG = float("-inf")
    prefix = tl.load(prefix_ptr + row).to(tl.uint32, bitcast=True)
    gt_total = tl.load(posgt_ptr + row)
    need_eq = tl.load(keff_ptr + row) - gt_total
    ntloc = tl.cdiv(tl.maximum(num_tiles - g, 0), G)
    for i in tl.range(0, ntloc):
        t = g + i * G
        offs = t * BLOCK + tl.arange(0, BLOCK)
        m = offs < length
        x = tl.load(scores_ptr + base + offs, mask=m, other=NEG)
        key = _mono_key(x)
        eq = (m & (x != NEG)) & (key == prefix)
        eqi = eq.to(tl.int32)
        be = tl.atomic_add(poseq_ptr + row, tl.sum(eqi), sem="relaxed", scope="gpu")
        er = be + tl.cumsum(eqi) - eqi
        take = eq & (er < need_eq)
        wp = gt_total + er
        tl.store(out_ptr + row * k + wp, offs.to(tl.int32), mask=take & (wp < k))


@triton.jit
def _coop_init_kernel(hist_ptr, prefix_ptr, posgt_ptr, poseq_ptr):
    row = tl.program_id(0)
    tl.store(hist_ptr + row * 256 + tl.arange(0, 256), tl.zeros((256,), tl.int32))
    tl.store(prefix_ptr + row, 0)
    tl.store(posgt_ptr + row, 0)
    tl.store(poseq_ptr + row, 0)


def _block_topk_coop(scores, lengths, out, k, G, workspace):
    # Scratch is carved from the caller workspace (a uint8 buffer) and zeroed by
    # a Triton kernel -- no torch allocation or in-place op inside the call, so
    # the path is CUDA-graph safe and mode-agnostic.
    rows, nblocks = scores.shape
    num_tiles = (nblocks + _BLOCK - 1) // _BLOCK
    w = workspace.view(dtype=torch.int32)
    o = 0
    hist = w[o:o + rows * 256].view(rows, 256); o += rows * 256
    prefix = w[o:o + rows]; o += rows
    krem = w[o:o + rows]; o += rows
    keff = w[o:o + rows]; o += rows
    posgt = w[o:o + rows]; o += rows
    poseq = w[o:o + rows]; o += rows
    _coop_init_kernel[(rows,)](hist, prefix, posgt, poseq)
    sr = scores.stride(0)
    for p in range(4):
        _coop_hist_kernel[(rows, G)](
            scores, lengths, prefix, hist, sr, num_tiles, G, p, BLOCK=_BLOCK
        )
        _coop_reduce_kernel[(rows,)](hist, prefix, krem, keff, k, p)
    _coop_gather_gt_kernel[(rows, G)](
        scores, lengths, prefix, out, posgt, sr, num_tiles, G, k, BLOCK=_BLOCK
    )
    _coop_gather_eq_kernel[(rows, G)](
        scores, lengths, prefix, keff, posgt, out, poseq, sr, num_tiles, G, k, BLOCK=_BLOCK
    )
    return out
