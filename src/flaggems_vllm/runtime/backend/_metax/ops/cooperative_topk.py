# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Exact cooperative top-k selection for MetaX C550.

Narrow rows use a single CTA. Wide rows select a bounded number of candidates
per tile and record the largest omitted key. A second launch merges the
candidates and revisits tiles only when the omitted keys can change the result.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.cooperative_topk import (
    _cooperative_topk_impl,
    _ordered_key,
    _radix_digit,
)
from flaggems_vllm.utils import libentry, libtuner

try:
    from triton.experimental.tle.language import gpu
except ImportError:
    # The vendor Triton in MACA 3.9 exposes the same operations under mctle.
    try:
        from triton.language.extra.maca.mctle import gpu
    except ImportError:
        gpu = None

_HAS_SMEM = all(hasattr(gpu, name) for name in ("alloc", "local_ptr", "smem"))


@triton.jit
def _select_tile(
    logits,
    output,
    length,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    K: tl.constexpr,
    BINS: tl.constexpr,
    CAP: tl.constexpr,
    FULL: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    bins = tl.arange(0, BINS)
    values = tl.load(logits + i, i < WIDTH, other=0.0)
    valid = tl.full((BLOCK,), True, tl.int1) if FULL else i < length
    keys = _ordered_key(values)
    if WIDTH == BLOCK + 1:
        tail = _ordered_key(tl.load(logits + BLOCK, length > BLOCK, other=0.0))
    scaled = tl.minimum(tl.maximum(values * (BINS / 4.0) + BINS / 2, 0.0), BINS - 1)
    bucket = tl.where(keys == 0xFFFFFFFF, BINS - 1, scaled.to(tl.int32))

    # Keep independent shared allocations for the histogram and candidate pairs.
    histogram = gpu.alloc(
        [BINS], dtype=tl.int32, layout=None, scope=gpu.smem, nv_mma_shared_layout=False
    )
    compact_buf = gpu.alloc(
        [CAP], dtype=tl.uint64, layout=None, scope=gpu.smem, nv_mma_shared_layout=False
    )
    counter = gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=gpu.smem, nv_mma_shared_layout=False
    )
    tl.store(gpu.local_ptr(histogram, (bins,)), 0)
    tl.store(gpu.local_ptr(counter, (tl.arange(0, 1),)), 0)
    tl.debug_barrier()
    tl.atomic_add(
        gpu.local_ptr(histogram, (bucket,)), 1, mask=valid, sem="relaxed", scope="cta"
    )
    tl.debug_barrier()
    hist = tl.load(gpu.local_ptr(histogram, (bins,)))
    suffix = tl.cumsum(hist, reverse=True)
    hits = (suffix >= K) & (suffix - hist < K)
    # Only the boundary bin satisfies hits. The high fields fit because
    # above < K <= 2048. For a spilled bin only count > CAP matters.
    info = tl.sum(
        tl.where(
            hits, (bins << 19) | ((suffix - hist) << 8) | tl.minimum(hist, CAP + 1), 0
        )
    )
    digit = info >> 19
    above = (info >> 8) & 2047
    count = info & 255
    needed = K - above
    boundary = valid & (bucket == digit)
    if count <= CAP:
        greater = valid & (bucket > digit)
        # The low half counts winners and the high half counts candidates.
        # Both fit in 16 bits for every row in this kernel.
        reservation = tl.atomic_add(
            gpu.local_ptr(counter, (tl.full((BLOCK,), 0, tl.int32),)),
            tl.where(greater, 1, 65536),
            mask=greater | boundary,
            sem="relaxed",
            scope="cta",
        )
        tl.store(output + (reservation & 65535), i, greater)
        slot = tl.where(boundary, reservation >> 16, 0)
        pair = (keys.to(tl.uint64) << 32) | i.to(tl.uint32)
        tl.store(gpu.local_ptr(compact_buf, (slot,)), pair, boundary)
        tl.debug_barrier()
        j = tl.arange(0, CAP)
        pair = tl.load(gpu.local_ptr(compact_buf, (j,)))
        compact = tl.where(j < count, (pair >> 32).to(tl.uint32), 0)
        idx = pair.to(tl.int32)
        # Valid keys are nonzero. Subtracting one lets an equally valued
        # predecessor rank ahead of this candidate. Empty slots rank last.
        rank = tl.sum(
            (
                compact[None, :]
                > compact[:, None] - (j[:, None] > j[None, :]).to(tl.uint32)
            ).to(tl.int32),
            1,
        )
        if WIDTH == BLOCK + 1:
            # Only the lowest-ranked winner can be displaced by the peeled tail.
            replace = (length > BLOCK) & (rank == needed - 1) & (tail > compact)
            idx = tl.where(replace, BLOCK, idx)
        tl.store(output + above + rank, idx, (j < count) & (rank < needed))
    else:
        # Search the ordered FP32 keys when a dense or saturated bucket
        # exceeds the bounded shared buffer. Each step discards half the key
        # range and lands on an actual key. All-equal buckets exit immediately.
        low = tl.min(tl.where(boundary, keys, 0xFFFFFFFF))
        high = tl.max(tl.where(boundary, keys, 0))
        while low < high:
            delta = high - low
            mid = low + (delta >> 1) + (delta & 1)
            upper = boundary & (keys >= mid)
            n_upper = tl.sum(upper.to(tl.int32))
            if n_upper >= needed:
                low = tl.min(tl.where(upper, keys, 0xFFFFFFFF))
                if n_upper == needed:
                    high = low
            else:
                high = tl.max(tl.where(boundary & (keys < mid), keys, 0))
        threshold = low
        greater = valid & (keys > threshold)
        ties = valid & (keys == threshold)
        n_greater = tl.sum(greater.to(tl.int32))
        # BLOCK <= 8192, so the two inclusive ranks cannot carry into each other.
        ranks = tl.cumsum(greater.to(tl.int32) | (ties.to(tl.int32) << 16))
        tie_rank = ranks >> 16
        pos = tl.where(greater, (ranks & 65535) - 1, n_greater + tie_rank - 1)
        if WIDTH == BLOCK + 1:
            replace_tie = (length > BLOCK) & (pos == K - 1) & (tail > threshold)
            i = tl.where(replace_tie, BLOCK, i)
        tl.store(output + pos, i, greater | (ties & (tie_rank <= K - n_greater)))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_c550_row"),
    key=["WIDTH", "K", "ROWS"],
    strategy=["default"] * 3,
)
@triton.jit
def _select_row(
    Logits,
    Lengths,
    Output,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    BASE_BINS: tl.constexpr,
    BASE_CAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
    output = Output + row * K
    logits = Logits + row * STRIDE
    MAIN: tl.constexpr = BLOCK // 2 if WIDTH > K and WIDTH == BLOCK // 2 + 1 else BLOCK
    BINS: tl.constexpr = BASE_BINS if MAIN <= 4096 else 2 * BASE_BINS
    CAP: tl.constexpr = BASE_CAP if K == 512 else 2 * BASE_CAP
    if WIDTH <= K or length <= K:
        j = tl.arange(0, K)
        tl.store(output + j, tl.where(j < length, j, -1))
    elif length >= MAIN:
        _select_tile(logits, output, length, WIDTH, MAIN, K, BINS, CAP, True)
    else:
        _select_tile(logits, output, length, WIDTH, MAIN, K, BINS, CAP, False)


@lru_cache(maxsize=None)
def _is_c550(device):
    return runtime.torch_device_fn.get_device_name(device) == "MetaX C550"


@triton.jit
def _linear_bin(value, key, BINS: tl.constexpr):
    scaled = tl.minimum(tl.maximum(value * (BINS / 8.0) + BINS / 2.0, 0.0), BINS - 1.0)
    return tl.where(key == 0xFFFFFFFF, BINS - 1, scaled.to(tl.int32))


@triton.jit
def _boundary_bin(hist, rank, BINS: tl.constexpr):
    bins = tl.arange(0, BINS)
    suffix = tl.cumsum(hist, reverse=True)
    # Each tile and candidate pool contains fewer than 2**16 entries, so
    # the bin and number of strictly higher entries fit in one reduction.
    packed = tl.max(tl.where(suffix >= rank, (bins << 16) | (suffix - hist), 0))
    return packed >> 16, packed & 65535


@triton.jit
def _screen_tiles(
    X,
    Lengths,
    Scratch,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    TILE: tl.constexpr,
    SLOTS: tl.constexpr,
    BINS: tl.constexpr,
):
    p = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    n = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
    i = p * TILE + tl.arange(0, TILE)
    valid = i < n
    if WIDTH >= PARTS * TILE:
        x = tl.load(X + row * STRIDE + i)
    else:
        x = tl.load(X + row * STRIDE + i, i < WIDTH, 0.0)
    keys = _ordered_key(x)
    # Only the screening decision uses this quantization. Save the largest
    # omitted exact key so the merging CTA can certify its answer per tile.
    if WIDTH <= 131072:
        coarse = _linear_bin(x, keys, BINS)
    else:
        finite = valid & (keys > 0x007FFFFF) & (keys < 0xFF800000)
        low = tl.min(tl.where(finite, x, float("inf")))
        high = tl.max(tl.where(finite, x, -float("inf")))
        scale = (BINS - 4.0) / tl.maximum(high - low, 1e-20)
        coarse = (
            tl.minimum(tl.maximum((x - low) * scale, 0.0), BINS - 4.0).to(tl.int32) + 1
        )
        coarse = tl.where(keys == 0x007FFFFF, 0, coarse)
        coarse = tl.where(keys == 0xFF800000, BINS - 2, coarse)
        coarse = tl.where(keys == 0xFFFFFFFF, BINS - 1, coarse)
    hist = tl.histogram(coarse, BINS, mask=valid)
    # A short peeled tail avoids doubling the candidate vector for e.g. 65539.
    EXTRA: tl.constexpr = max(0, WIDTH - PARTS * TILE)
    capacity = tl.where(p == PARTS - 1, SLOTS - EXTRA, SLOTS)
    quota = tl.minimum(capacity, tl.maximum(n - p * TILE, 0))
    cut, above = _boundary_bin(hist, quota, BINS)
    greater = valid & (coarse > cut)
    equal = valid & (coarse == cut)
    ranks = tl.cumsum(greater.to(tl.int32) | (equal.to(tl.int32) << 16))
    eq_rank = ranks >> 16
    keep = greater | (equal & (eq_rank <= quota - above))
    slot = tl.where(greater, (ranks & 65535) - 1, above + eq_rank - 1)
    base = Scratch + row * PARTS * (2 * SLOTS + 1 + (WIDTH > 131072))
    dst = base + p * SLOTS
    tl.store(dst + slot, keys, keep)
    tl.store(dst + PARTS * SLOTS + slot, i, keep)
    j = tl.arange(0, SLOTS)
    tl.store(dst + j, 0, (j >= quota) & (j < capacity))
    if EXTRA:
        if p == PARTS - 1:
            t = tl.arange(0, triton.next_power_of_2(EXTRA))
            col = PARTS * TILE + t
            tail_key = _ordered_key(tl.load(X + row * STRIDE + col, col < n, 0.0))
            tl.store(dst + SLOTS - EXTRA + t, tl.where(col < n, tail_key, 0), t < EXTRA)
            tl.store(dst + PARTS * SLOTS + SLOTS - EXTRA + t, col, t < EXTRA)
    tl.store(base + 2 * PARTS * SLOTS + p, tl.max(tl.where(valid & ~keep, keys, 0)))
    if WIDTH > 131072:
        # Each full wide tile contains at least K values. Its minimum gives
        # a lower bound on the global cutoff, even for ordered inputs.
        full_min = tl.where(n >= (p + 1) * TILE, tl.min(keys), 0)
        tl.store(base + (2 * SLOTS + 1) * PARTS + p, full_min)


@triton.jit
def _write_candidates(
    keys,
    valid,
    indices,
    Out,
    Tmp,
    K: tl.constexpr,
    BINS: tl.constexpr,
    CAP: tl.constexpr,
    FIXED: tl.constexpr,
):
    if FIXED:
        raw = tl.where(keys >= 0x80000000, keys ^ 0x80000000, ~keys)
        v = raw.to(tl.float32, bitcast=True)
        buckets = _linear_bin(v, keys, BINS)
    else:
        # Choose the bucket width from the exponent of the finite key span.
        finite = valid & (keys > 0x007FFFFF) & (keys < 0xFF800000)
        lo = tl.min(tl.where(finite, keys, 0xFFFFFFFF))
        hi = tl.max(tl.where(finite, keys, 0))
        span_bits = (hi - lo).to(tl.float32).to(tl.uint32, bitcast=True)
        exponent = ((span_bits >> 23) & 255).to(tl.int32) - 127
        shift = tl.maximum(0, exponent + 2 - BINS.bit_length())
        buckets = tl.minimum((keys - lo) >> shift, BINS - 3).to(tl.int32) + 1
        buckets = tl.where(keys == 0x007FFFFF, 0, buckets)
        buckets = tl.where(keys >= 0xFF800000, BINS - 1, buckets)
    hist = tl.histogram(buckets, BINS, mask=valid)
    cut, above = _boundary_bin(hist, K, BINS)
    need = K - above
    eq = valid & (buckets == cut)
    count = tl.sum(eq.to(tl.int32))
    if count < need:
        # Fewer than K real candidates. Zero is below every valid FP32 key.
        threshold = tl.full((), 0, tl.uint32)
    elif count <= CAP:
        greater = valid & (buckets > cut)
        ranks = tl.cumsum(greater.to(tl.int32) | (eq.to(tl.int32) << 16))
        tl.store(Out + (ranks & 65535) - 1, indices, greater)
        slot = (ranks >> 16) - 1
        tl.store(Tmp + slot, keys, eq)
        tl.store(Tmp + CAP + slot, indices, eq)
        tl.debug_barrier()
        j = tl.arange(0, CAP)
        compact = tl.load(Tmp + j, j < count, 0).to(tl.uint32)
        index = tl.load(Tmp + CAP + j, j < count, 0).to(tl.uint32)
        ordered = tl.sort((compact.to(tl.uint64) << 32) | index, descending=True)
        threshold = tl.max(tl.where(j == need - 1, ordered >> 32, 0)).to(tl.uint32)
        tl.store(Out + above + j, ordered.to(tl.int32), j < need)
    else:
        threshold = tl.full((), 0, tl.uint32)
        for shift2 in tl.static_range(24, -1, -8):
            digits = ((keys >> shift2) & 255).to(tl.int32)
            digit, need = _radix_digit(tl.histogram(digits, 256, mask=eq), need)
            threshold |= digit << shift2
            eq &= digits == digit
        greater = valid & (keys > threshold)
        equal = valid & (keys == threshold)
        ranks = tl.cumsum(greater.to(tl.int32) | (equal.to(tl.int32) << 16))
        eq_rank = ranks >> 16
        pos = tl.where(greater, (ranks & 65535) - 1, K - need + eq_rank - 1)
        tl.store(Out + pos, indices, greater | (equal & (eq_rank <= need)))
    return threshold


@triton.jit
def _recover_tiles(
    X,
    Out,
    Scratch,
    Work,
    n,
    lower,
    K: tl.constexpr,
    PARTS: tl.constexpr,
    TILE: tl.constexpr,
    SLOTS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    EXTRA: tl.constexpr,
):
    # Add back the originals of exactly those tiles whose omitted keys can
    # exceed the tentative kth. Retained items of all remaining tiles suffice.
    p = tl.arange(0, triton.next_power_of_2(PARTS))
    rescan = (
        tl.load(Scratch + 2 * PARTS * SLOTS + p, p < PARTS, 0).to(tl.uint32) > lower
    )
    tl.store(Work + tl.cumsum(rescan.to(tl.int32)) - 1, p, rescan)
    tl.debug_barrier()
    tiles = tl.sum(rescan.to(tl.int32))
    slot = tl.arange(0, POOL_SIZE)
    valid = slot < PARTS * SLOTS
    omitted = tl.load(Scratch + 2 * PARTS * SLOTS + slot // SLOTS, valid, 0).to(
        tl.uint32
    )
    valid &= (omitted <= lower) | (slot >= PARTS * SLOTS - EXTRA)
    keys = tl.load(Scratch + slot, valid, 0).to(tl.uint32)
    valid &= keys != 0
    prefix = tl.full((), 0, tl.uint32)
    mask = tl.full((), 0, tl.uint32)
    left = K
    lane = tl.arange(0, TILE)
    for shift in tl.static_range(24, -1, -8):
        digits = ((keys >> shift) & 255).to(tl.int32)
        hist = tl.histogram(digits, 256, mask=valid & ((keys & mask) == prefix))
        for tile_idx in range(tiles):
            col = tl.load(Work + tile_idx) * TILE + lane
            tile_keys = _ordered_key(tl.load(X + col, col < n, 0.0))
            tile_digits = ((tile_keys >> shift) & 255).to(tl.int32)
            hist += tl.histogram(
                tile_digits, 256, mask=(col < n) & ((tile_keys & mask) == prefix)
            )
        digit, left = _radix_digit(hist, left)
        prefix |= digit << shift
        mask |= tl.full((), 255, tl.uint32) << shift
    gt = valid & (keys > prefix)
    eq = valid & (keys == prefix)
    er = tl.cumsum(eq.to(tl.int32))
    pos = tl.where(gt, tl.cumsum(gt.to(tl.int32)) - 1, K - left + er - 1)
    idx = tl.load(Scratch + PARTS * SLOTS + slot, valid, 0)
    tl.store(Out + pos, idx, gt | (eq & (er <= left)))
    greater = tl.sum(gt.to(tl.int32))
    equal = tl.sum(eq.to(tl.int32))
    for tile_idx in range(tiles):
        col = tl.load(Work + tile_idx) * TILE + lane
        tile_keys = _ordered_key(tl.load(X + col, col < n, 0.0))
        tile_greater = (col < n) & (tile_keys > prefix)
        tile_equal = (col < n) & (tile_keys == prefix)
        erank = equal + tl.cumsum(tile_equal.to(tl.int32))
        dst = tl.where(
            tile_greater,
            greater + tl.cumsum(tile_greater.to(tl.int32)) - 1,
            K - left + erank - 1,
        )
        tl.store(Out + dst, col, tile_greater | (tile_equal & (erank <= left)))
        greater += tl.sum(tile_greater.to(tl.int32))
        equal += tl.sum(tile_equal.to(tl.int32))


@triton.jit
def _merge_tiles(
    X,
    Lengths,
    Output,
    Scratch,
    Workspace,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    PARTS: tl.constexpr,
    TILE: tl.constexpr,
    SLOTS: tl.constexpr,
    R: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
    out = Output + row * K
    if n <= K:
        short = tl.arange(0, K)
        tl.store(out + short, tl.where(short < n, short, -1))
    else:
        s = Scratch + row * PARTS * (2 * SLOTS + 1 + (WIDTH > 131072))
        tmp = Workspace.to(tl.pointer_type(tl.int32)) + row * 4096
        j = tl.arange(0, R)
        keys = tl.load(s + j, j < PARTS * SLOTS, 0).to(tl.uint32)
        valid = keys != 0
        indices = tl.load(s + PARTS * SLOTS + j, j < PARTS * SLOTS, 0)
        threshold = _write_candidates(
            keys,
            valid,
            indices,
            out,
            tmp,
            K,
            256 if WIDTH <= 16385 else 512,
            128,
            WIDTH <= 131072,
        )
        p = tl.arange(0, triton.next_power_of_2(PARTS))
        omitted = tl.max(tl.load(s + 2 * PARTS * SLOTS + p, p < PARTS, 0).to(tl.uint32))
        if omitted > threshold:
            # All provisional writes must finish before this CTA reuses the
            # workspace and replaces the row's output.
            tl.debug_barrier()
            if WIDTH > 131072:
                tl.static_assert(TILE >= K)
                bound = tl.max(
                    tl.load(s + (2 * SLOTS + 1) * PARTS + p, p < PARTS, 0).to(tl.uint32)
                )
                # Keep omitted keys equal to a stronger tile-derived bound:
                # the candidate pool alone may contain too few of those ties.
                threshold = tl.maximum(threshold, tl.maximum(bound, 1) - 1)
            _recover_tiles(
                X + row * STRIDE,
                out,
                s,
                tmp,
                n,
                threshold,
                K,
                PARTS,
                TILE,
                SLOTS,
                R,
                max(0, WIDTH - PARTS * TILE),
            )


def _wide_topk(logits, lengths, output, workspace, k, max_seq_len):
    rows, width = logits.shape
    tile = 1024
    if width > 131072:
        tile = 2048 if rows <= 4 else 4096
    parts, remainder = divmod(width, tile)
    if remainder > 4:
        parts += 1
    target = (2 if width <= 131072 else 4) * k
    slots = max(8, triton.next_power_of_2(triton.cdiv(target, parts)))
    scratch = torch.empty(
        (rows, parts * (2 * slots + 1 + (width > 131072))),
        device=logits.device,
        dtype=torch.int32,
    )
    _screen_tiles[(parts, rows)](
        logits,
        lengths,
        scratch,
        width,
        logits.stride(0),
        parts,
        tile,
        slots,
        64 if width <= 16385 or width > 131072 else 256,
        num_warps=4,
        num_stages=1,
    )
    _merge_tiles[(rows,)](
        logits,
        lengths,
        output,
        scratch,
        workspace,
        width,
        logits.stride(0),
        k,
        parts,
        tile,
        slots,
        triton.next_power_of_2(parts * slots),
        num_warps=8,
        num_stages=1,
    )


def cooperative_topk(logits, lengths, output, workspace, k, max_seq_len):
    c550 = logits.device.type == "cuda" and _is_c550(logits.device)
    return _cooperative_topk_impl(
        logits,
        lengths,
        output,
        workspace,
        k,
        max_seq_len,
        small_kernel=_select_row if c550 and _HAS_SMEM else None,
        wide_kernel=_wide_topk if c550 else None,
        wide_max_width=1048576,
    )
