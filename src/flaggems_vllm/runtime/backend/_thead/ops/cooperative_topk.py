# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""PPU coarse selection with exact FP32 refinement.

The narrow path uses a coarse bucket followed by exact integer-key selection.
Medium rows compact candidates in one CTA and finish them in a second launch.
Wide-row compaction ranks two masks with one packed prefix scan.
Kernel boundaries synchronize the four stages. Each partition uses immutable
prefix offsets, so every autotuned stage can safely be repeated. A coarse
bucket only narrows the search; canonical FP32 keys decide the exact result.
"""
import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.cooperative_topk import _cooperative_topk_impl, _ordered_key
from flaggems_vllm.utils import libentry, libtuner


@triton.jit
def _thead_mask_ranks(mask, BLOCK: tl.constexpr):
    # PPU uses 32 lanes per warp; scan within each warp then prefix the groups.
    if BLOCK >= 32:
        bits = tl.reshape(mask.to(tl.int32), (BLOCK // 32, 32))
        counts = tl.sum(bits, 1)
        bases = tl.cumsum(counts) - counts
        return tl.reshape(tl.cumsum(bits, 1) + bases[:, None], (BLOCK,))
    else:
        return tl.cumsum(mask.to(tl.int32))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_thead_small"),
    key=["WIDTH", "K", "ROWS"],
    strategy=["default"] * 3,
)
@triton.jit
def _cooperative_topk_thead_small(
    Logits,
    Lengths,
    Output,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    BINS: tl.constexpr,
):
    TAIL: tl.constexpr = WIDTH >= 1024 and WIDTH == BLOCK // 2 + 1
    MAIN: tl.constexpr = BLOCK // 2 if TAIL else BLOCK
    row = tl.program_id(0).to(tl.int64)
    n = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
    if n <= K:
        j = tl.arange(0, K)
        tl.store(Output + row * K + j, tl.where(j < n, j, -1))
    else:
        i = tl.arange(0, MAIN)
        valid = i < n
        v = tl.load(Logits + row * STRIDE + i, valid, other=0.0)
        key = _ordered_key(v)
        bucket = tl.minimum(
            tl.maximum(v * (BINS / 16.0) + BINS / 2.0, 0.0), BINS - 1.0
        ).to(tl.int32)
        bucket = tl.where(key == 0xFFFFFFFF, BINS - 1, bucket)
        b = tl.arange(0, BINS)
        hist = tl.histogram(bucket, BINS, mask=valid)
        if TAIL:
            tv = tl.load(Logits + row * STRIDE + MAIN, n > MAIN, other=0.0)
            tk = _ordered_key(tv)
            tb = tl.minimum(
                tl.maximum(tv * (BINS / 16.0) + BINS / 2.0, 0.0), BINS - 1.0
            ).to(tl.int32)
            tb = tl.where(tk == 0xFFFFFFFF, BINS - 1, tb)
            hist += tl.where((b == tb) & (n > MAIN), 1, 0)
        suffix = tl.cumsum(hist, reverse=True)
        d = tl.max(tl.where(suffix >= K, b, 0), 0)
        above = valid & (bucket > d)
        candidates = valid & (bucket == d)
        remain = K - tl.sum(above.to(tl.int32))
        lo = tl.min(tl.where(candidates, key, 0xFFFFFFFF), 0)
        hi = tl.max(tl.where(candidates, key, 0), 0)
        if TAIL:
            ta = (n > MAIN) & (tb > d)
            te = (n > MAIN) & (tb == d)
            remain -= ta.to(tl.int32)
            lo = tl.minimum(lo, tl.where(te, tk, 0xFFFFFFFF))
            hi = tl.maximum(hi, tl.where(te, tk, 0))
        exact = tl.full((), False, tl.int1)
        while (lo < hi) & ~exact:
            mid = lo + ((hi - lo) >> 1)
            count = tl.sum((candidates & (key > mid)).to(tl.int32))
            if TAIL:
                count += (te & (tk > mid)).to(tl.int32)
            exact = count == remain
            lo = tl.where(count >= remain, mid + 1, lo)
            hi = tl.where(count < remain, mid, hi)
        if exact:
            selected = above | (candidates & (key >= lo))
            if TAIL:
                ts = ta | (te & (tk >= lo))
        else:
            gt = candidates & (key > lo)
            eq = candidates & (key == lo)
            ng = tl.sum(gt.to(tl.int32))
            if TAIL:
                ng += (te & (tk > lo)).to(tl.int32)
            er = _thead_mask_ranks(eq, MAIN)
            selected = above | gt | (eq & (er <= remain - ng))
            if TAIL:
                ts = ta | (
                    te
                    & (
                        (tk > lo)
                        | ((tk == lo) & (tl.sum(eq.to(tl.int32)) < remain - ng))
                    )
                )
        pos = _thead_mask_ranks(selected, MAIN) - 1
        tl.store(Output + row * K + pos, i, selected)
        if TAIL:
            tl.store(Output + row * K + tl.sum(selected.to(tl.int32)), MAIN, ts)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_thead_medium"),
    key=["WIDTH", "K", "ROWS"],
    strategy=["default"] * 3,
)
@triton.jit
def _cooperative_topk_thead_medium(
    Logits,
    Lengths,
    Output,
    State,
    Keys,
    Indices,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    n = tl.minimum(tl.maximum(tl.load(Lengths + r), 0), WIDTH)
    if n <= K:
        j = tl.arange(0, K)
        tl.store(Output + r * K + j, tl.where(j < n, j, -1))
        tl.store(State + r * 3, 0)
        tl.store(State + r * 3 + 1, 0)
        tl.store(State + r * 3 + 2, 0)
    else:
        i = tl.arange(0, BLOCK)
        valid = i < n
        vals = tl.load(Logits + r * STRIDE + i, valid, other=0.0)
        q = _cooperative_topk_thead_bucket(vals)
        h = tl.histogram(q, 256, mask=valid)
        b = tl.arange(0, 256)
        suffix = tl.cumsum(h, reverse=True)
        d = tl.max(tl.where(suffix >= K, b, 0), 0)
        above = valid & (q > d)
        equal = valid & (q == d)
        na = tl.sum(above.to(tl.int32))
        ne = tl.sum(equal.to(tl.int32))
        tl.store(State + r * 3, d)
        tl.store(State + r * 3 + 1, na)
        tl.store(State + r * 3 + 2, ne)
        ar, er = _thead_packed_ranks(above, equal, BLOCK)
        tl.store(Output + r * K + ar - 1, i, above)
        if ne <= CAPACITY:
            tl.store(Keys + r * CAPACITY + er - 1, _ordered_key(vals), equal)
            tl.store(Indices + r * CAPACITY + er - 1, i, equal)
        elif ne == K - na:
            tl.store(Output + r * K + na + er - 1, i, equal)


@triton.jit
def _thead_packed_ranks(above, equal, BLOCK: tl.constexpr):
    # Each tile contributes fewer than 65536 items, so the above count cannot
    # carry into the upper 16-bit candidate count. One scan ranks both masks.
    tl.static_assert(BLOCK < 65536)
    packed = above.to(tl.uint32) + (equal.to(tl.uint32) << 16)
    ranks = tl.cumsum(packed)
    return (ranks & 65535).to(tl.int32), (ranks >> 16).to(tl.int32)


@triton.jit
def _cooperative_topk_thead_bucket(v):
    bits = v.to(tl.uint32, bitcast=True)
    q = tl.minimum(tl.maximum(v * 32.0 + 128.0, 0.0), 255.0).to(tl.int32)
    return tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 255, q)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_thead_histogram"),
    key=["WIDTH", "ROWS", "PARTS"],
    strategy=["default"] * 3,
)
@triton.jit
def _cooperative_topk_thead_histogram(
    Logits,
    Lengths,
    Hist,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64)
    r = tl.program_id(1).to(tl.int64)
    n = tl.minimum(tl.maximum(tl.load(Lengths + r), 0), WIDTH)
    chunk: tl.constexpr = triton.cdiv(WIDTH, PARTS)
    end = tl.minimum((p + 1) * chunk, n)
    h = tl.full((256,), 0, tl.int32)
    for start in range(p * chunk, end, BLOCK):
        i = start + tl.arange(0, BLOCK)
        valid = i < end
        v = tl.load(Logits + r * STRIDE + i, valid, other=0.0)
        h += tl.histogram(_cooperative_topk_thead_bucket(v), 256, mask=valid)
    tl.store(Hist + (r * PARTS + p) * 256 + tl.arange(0, 256), h)


# Fixed per-row reduction: geometry follows the histogram/state protocol.
# Exempt from autotuning, like the shared radix threshold kernel.
@libentry()
@triton.jit
def _cooperative_topk_thead_threshold(
    Hist, State, K: tl.constexpr, PARTS: tl.constexpr
):
    r = tl.program_id(0).to(tl.int64)
    p = tl.arange(0, PARTS)
    b = tl.arange(0, 256)
    h = tl.load(Hist + (r * PARTS + p[:, None]) * 256 + b[None, :])
    total = tl.sum(h, 0)
    suffix = tl.cumsum(total, reverse=True)
    d = tl.max(tl.where(suffix >= K, b, 0), 0)
    above = tl.sum(tl.where(b[None, :] > d, h, 0), 1)
    equal = tl.load(Hist + (r * PARTS + p) * 256 + d)
    base = State + r * (2 * PARTS + 3)
    tl.store(base, d)
    tl.store(base + 1, tl.sum(above))
    tl.store(base + 2, tl.sum(equal))
    tl.store(base + 3 + p, tl.cumsum(above) - above)
    tl.store(base + 3 + PARTS + p, tl.cumsum(equal) - equal)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_thead_compact"),
    key=["WIDTH", "K", "ROWS", "PARTS", "CAPACITY"],
    strategy=["default"] * 5,
)
@triton.jit
def _cooperative_topk_thead_compact(
    Logits,
    Lengths,
    Output,
    State,
    Keys,
    Indices,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    PARTS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64)
    r = tl.program_id(1).to(tl.int64)
    n = tl.minimum(tl.maximum(tl.load(Lengths + r), 0), WIDTH)
    if n <= K:
        if p == 0:
            j = tl.arange(0, K)
            tl.store(Output + r * K + j, tl.where(j < n, j, -1))
    else:
        base = State + r * (2 * PARTS + 3)
        d = tl.load(base)
        na = tl.load(base + 1)
        ne = tl.load(base + 2)
        ao = tl.load(base + 3 + p)
        eo = tl.load(base + 3 + PARTS + p)
        chunk: tl.constexpr = triton.cdiv(WIDTH, PARTS)
        end = tl.minimum((p + 1) * chunk, n)
        for start in range(p * chunk, end, BLOCK):
            i = start + tl.arange(0, BLOCK)
            valid = i < end
            v = tl.load(Logits + r * STRIDE + i, valid, other=0.0)
            q = _cooperative_topk_thead_bucket(v)
            a = valid & (q > d)
            e = valid & (q == d)
            ar, er = _thead_packed_ranks(a, e, BLOCK)
            tl.store(Output + r * K + ao + ar - 1, i, a)
            ao += tl.sum(a.to(tl.int32))
            if ne <= CAPACITY:
                tl.store(Keys + r * CAPACITY + eo + er - 1, _ordered_key(v), e)
                tl.store(Indices + r * CAPACITY + eo + er - 1, i, e)
                eo += tl.sum(e.to(tl.int32))
            elif ne == K - na:
                tl.store(Output + r * K + na + eo + er - 1, i, e)
                eo += tl.sum(e.to(tl.int32))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_thead_finish"),
    key=["WIDTH", "K", "ROWS", "PARTS", "CAPACITY"],
    strategy=["default"] * 5,
)
@triton.jit
def _cooperative_topk_thead_finish(
    Logits,
    Lengths,
    Output,
    State,
    Keys,
    Indices,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    PARTS: tl.constexpr,
    CAPACITY: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    part = tl.program_id(0)
    r = tl.program_id(1).to(tl.int64)
    n = tl.minimum(tl.maximum(tl.load(Lengths + r), 0), WIDTH)
    base = State + r * (2 * PARTS + 3)
    d = tl.load(base)
    na = tl.load(base + 1)
    ne = tl.load(base + 2)
    remain = K - na
    if n > K:
        if ne <= CAPACITY:
            q = part * TILE + tl.arange(0, TILE)
            j = tl.arange(0, CAPACITY)
            valid_q = q < ne
            query = tl.load(Keys + r * CAPACITY + q, valid_q, other=0)
            other = tl.load(Keys + r * CAPACITY + j, j < ne, other=0)
            higher = (other[None, :] > query[:, None]) | (
                (other[None, :] == query[:, None]) & (j[None, :] < q[:, None])
            )
            rank = tl.sum(((j[None, :] < ne) & higher).to(tl.int32), 1)
            index = tl.load(Indices + r * CAPACITY + q, valid_q, other=0)
            tl.store(Output + r * K + na + rank, index, valid_q & (rank < remain))
        elif (ne != remain) & (part == 0):
            # Capacity overflow: exact integer-key refinement over the original
            # row. Never truncate candidates, including all-equal/extreme data.
            lo = tl.full((), 4294967295, tl.uint32)
            hi = tl.full((), 0, tl.uint32)
            for start in range(0, n, BLOCK):
                i = start + tl.arange(0, BLOCK)
                v = tl.load(Logits + r * STRIDE + i, i < n, other=0.0)
                valid = (i < n) & (_cooperative_topk_thead_bucket(v) == d)
                key = _ordered_key(v)
                lo = tl.minimum(lo, tl.min(tl.where(valid, key, 4294967295), 0))
                hi = tl.maximum(hi, tl.max(tl.where(valid, key, 0), 0))
            exact = tl.full((), False, tl.int1)
            while (lo < hi) & ~exact:
                mid = lo + ((hi - lo) >> 1)
                cnt = tl.full((), 0, tl.int32)
                for start in range(0, n, BLOCK):
                    i = start + tl.arange(0, BLOCK)
                    v = tl.load(Logits + r * STRIDE + i, i < n, other=0.0)
                    valid = (i < n) & (_cooperative_topk_thead_bucket(v) == d)
                    key = _ordered_key(v)
                    cnt += tl.sum((valid & (key > mid)).to(tl.int32))
                exact = cnt == remain
                lo = tl.where(cnt >= remain, mid + 1, lo)
                hi = tl.where(cnt < remain, mid, hi)
            ng = tl.full((), 0, tl.int32)
            for start in range(0, n, BLOCK):
                i = start + tl.arange(0, BLOCK)
                v = tl.load(Logits + r * STRIDE + i, i < n, other=0.0)
                valid = (i < n) & (_cooperative_topk_thead_bucket(v) == d)
                key = _ordered_key(v)
                ng += tl.sum((valid & (key > lo)).to(tl.int32))
            pos = na
            ties = tl.full((), 0, tl.int32)
            for start in range(0, n, BLOCK):
                i = start + tl.arange(0, BLOCK)
                v = tl.load(Logits + r * STRIDE + i, i < n, other=0.0)
                valid = (i < n) & (_cooperative_topk_thead_bucket(v) == d)
                key = _ordered_key(v)
                gt = valid & (key > lo)
                eq = valid & (key == lo)
                er = tl.cumsum(eq.to(tl.int32))
                sel = tl.where(
                    exact, valid & (key >= lo), gt | (eq & (ties + er <= remain - ng))
                )
                rank = tl.cumsum(sel.to(tl.int32))
                tl.store(Output + r * K + pos + rank - 1, i, sel)
                ties += tl.sum(eq.to(tl.int32))
                pos += tl.sum(sel.to(tl.int32))


def _cooperative_topk_medium(logits, lengths, output, workspace, k, max_seq_len):
    rows, width = logits.shape
    capacity = 256
    # No histogram survives the first CTA. These disjoint, no-copy views use
    # only 2060 bytes/row of the caller's 16 KiB/row workspace.
    scratch = workspace.view(-1)[: rows * (3 + 2 * capacity) * 4].view(torch.int32)
    state = scratch[: rows * 3]
    keys = scratch[rows * 3 : rows * (3 + capacity)].view(torch.uint32)
    indices = scratch[rows * (3 + capacity) :]
    _cooperative_topk_thead_medium[(rows,)](
        logits,
        lengths,
        output,
        state,
        keys,
        indices,
        width,
        logits.stride(0),
        k,
        rows,
        BLOCK=triton.next_power_of_2(width),
        CAPACITY=capacity,
    )
    # PARTS=0 selects three state scalars per row; no partition offsets are
    # needed after one CTA has compacted the complete row.
    _cooperative_topk_thead_finish[
        lambda meta: (triton.cdiv(capacity, meta["TILE"]), rows)
    ](
        logits,
        lengths,
        output,
        state,
        keys,
        indices,
        width,
        logits.stride(0),
        k,
        0,
        capacity,
        rows,
    )


def _cooperative_topk_wide(logits, lengths, output, workspace, k, max_seq_len):
    rows, width = logits.shape
    if width <= 16384 and rows <= 16:
        return _cooperative_topk_medium(
            logits, lengths, output, workspace, k, max_seq_len
        )
    # Measured on PPU-ZW810E: short rows need fewer CTAs, while a small
    # batch of long rows needs all 64 SMs to avoid long per-partition loops.
    if rows > 16:
        parts = 8
    elif width <= 16384:
        parts = 16
    elif width < 262144:
        parts = 32
    elif rows == 1 and width >= 524288:
        # Double the CTAs for long single rows to halve sequential tile scans.
        parts = 128
    else:
        parts = 64
    capacity = 256
    hist = torch.empty((rows, parts, 256), dtype=torch.int32, device=logits.device)
    state = torch.empty((rows, 2 * parts + 3), dtype=torch.int32, device=logits.device)
    keys = torch.empty((rows, capacity), dtype=torch.uint32, device=logits.device)
    indices = torch.empty((rows, capacity), dtype=torch.int32, device=logits.device)
    _cooperative_topk_thead_histogram[(parts, rows)](
        logits, lengths, hist, width, logits.stride(0), parts, rows
    )
    _cooperative_topk_thead_threshold[(rows,)](
        hist, state, k, parts, num_warps=4, num_stages=1
    )
    _cooperative_topk_thead_compact[(parts, rows)](
        logits,
        lengths,
        output,
        state,
        keys,
        indices,
        width,
        logits.stride(0),
        k,
        parts,
        rows,
        CAPACITY=capacity,
    )
    _cooperative_topk_thead_finish[
        lambda meta: (triton.cdiv(capacity, meta["TILE"]), rows)
    ](
        logits,
        lengths,
        output,
        state,
        keys,
        indices,
        width,
        logits.stride(0),
        k,
        parts,
        capacity,
        rows,
    )


def cooperative_topk(logits, lengths, output, workspace, k, max_seq_len):
    return _cooperative_topk_impl(
        logits,
        lengths,
        output,
        workspace,
        k,
        max_seq_len,
        small_kernel=_cooperative_topk_thead_small,
        wide_kernel=_cooperative_topk_wide,
        wide_max_width=1048576,
    )
