# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Hygon coarse selection with exact FP32 refinement.

The narrow path keeps FP32 keys in registers. The 4K/512 fast path checks a
boundary interval and sorts compacted candidates in shared memory; other
cases select a coarse bucket and refine its exact integer-key threshold.
Power-of-two-plus-one widths use a separate scalar tail. Medium K=1024 rows
use uint16 suffix counts and three stages; other wide rows use four stages.
Kernel boundaries synchronize the four stages. Each partition uses immutable
prefix offsets, so every autotuned stage can safely be repeated. A coarse
bucket only narrows the search; canonical FP32 keys decide the exact result.
Histogram and compaction use the same partition rule: when the floor-sized
chunk is 1024-aligned, only the final partition owns the small remainder.
That remainder uses a separate tiny tile rather than an extra full tile in
all partitions. Otherwise the original ceiling-sized partitioning is retained.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.cooperative_topk import _cooperative_topk_impl, _ordered_key
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle
    except ImportError:
        tle = None
else:
    tle = None

_HAS_TLE = bool(
    tle is not None
    and hasattr(tle, "gpu")
    and hasattr(tle.gpu, "alloc")
    and hasattr(tle.gpu, "local_ptr")
    and hasattr(tle.gpu, "smem")
)


@triton.jit
def _mask_ranks(mask, BLOCK: tl.constexpr):
    if BLOCK >= 64:
        bits = tl.reshape(mask.to(tl.int32), (BLOCK // 64, 64))
        counts = tl.sum(bits, 1)
        bases = tl.cumsum(counts) - counts
        return tl.reshape(tl.cumsum(bits, 1) + bases[:, None], (BLOCK,))
    else:
        return tl.cumsum(mask.to(tl.int32))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_hygon_small"),
    key=["WIDTH", "K", "ROWS"],
    strategy=["default"] * 3,
)
@triton.jit
def _cooperative_topk_hygon_small(
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
            er = _mask_ranks(eq, MAIN)
            selected = above | gt | (eq & (er <= remain - ng))
            if TAIL:
                ts = ta | (
                    te
                    & (
                        (tk > lo)
                        | ((tk == lo) & (tl.sum(eq.to(tl.int32)) < remain - ng))
                    )
                )
        pos = _mask_ranks(selected, MAIN) - 1
        tl.store(Output + row * K + pos, i, selected)
        if TAIL:
            tl.store(Output + row * K + tl.sum(selected.to(tl.int32)), MAIN, ts)


@triton.jit
def _packed_ranks(above, equal, BLOCK: tl.constexpr):
    # Each chunk has at most BLOCK <= 4096 elements. The low 16-bit above
    # count cannot carry into the high 16-bit candidate count.
    tl.static_assert(BLOCK < 65536)
    if BLOCK < 64:
        return tl.cumsum(above.to(tl.int32)), tl.cumsum(equal.to(tl.int32))
    else:
        packed = above.to(tl.uint32) + (equal.to(tl.uint32) << 16)
        groups = tl.reshape(packed, (BLOCK // 64, 64))
        counts = tl.sum(groups, 1)
        bases = tl.cumsum(counts) - counts
        ranks = tl.reshape(tl.cumsum(groups, 1) + bases[:, None], (BLOCK,))
        return (ranks & 65535).to(tl.int32), (ranks >> 16).to(tl.int32)


@triton.jit
def _bracket_packed_ranks(above, equal, BLOCK: tl.constexpr):
    # Preserve the measured 4K kernel's four contiguous values per thread.
    # Top-k output order is unspecified: compact in logical thread-major order,
    # performing the two register-local scans before the 512-thread prefix.
    tl.static_assert(BLOCK % 2048 == 0)
    tl.static_assert(BLOCK < 65536)
    packed = above.to(tl.uint32) + (equal.to(tl.uint32) << 16)
    groups = tl.reshape(packed, (BLOCK // 2048, 512, 4))
    local_counts = tl.sum(groups, 2)
    counts = tl.sum(local_counts, 0)
    base = tl.cumsum(counts) - counts
    rep_base = tl.cumsum(local_counts, 0) - local_counts
    ranks = tl.reshape(
        tl.cumsum(groups, 2) + rep_base[:, :, None] + base[None, :, None],
        (BLOCK,),
    )
    return (ranks & 65535).to(tl.int32), (ranks >> 16).to(tl.int32)


# Four monotone counts prove that a boundary interval contains the kth key.
# The fixed intervals are only a speculative fast path for 4K/512 rows:
# unbracketed rows and capacity overflow use exact FP32-key refinement.
if _HAS_TLE:

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("cooperative_topk_hygon_bracket_small"),
        key=["WIDTH", "K", "ROWS"],
        strategy=["default"] * 3,
    )
    @triton.jit
    def _cooperative_topk_hygon_bracket_small(
        Logits,
        Lengths,
        Output,
        WIDTH: tl.constexpr,
        STRIDE: tl.constexpr,
        K: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK: tl.constexpr,
        LOW: tl.constexpr,
        HIGH: tl.constexpr,
        CAP: tl.constexpr,
    ):
        tl.static_assert(WIDTH < 65536)
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
            p0 = _ordered_key(tl.full((), LOW, tl.float32))
            p1 = _ordered_key(tl.full((), LOW + (HIGH - LOW) / 3.0, tl.float32))
            p2 = _ordered_key(tl.full((), LOW + 2.0 * (HIGH - LOW) / 3.0, tl.float32))
            p3 = _ordered_key(tl.full((), HIGH, tl.float32))
            counted = (
                (valid & (key > p0)).to(tl.uint64)
                | ((valid & (key > p1)).to(tl.uint64) << 16)
                | ((valid & (key > p2)).to(tl.uint64) << 32)
                | ((valid & (key > p3)).to(tl.uint64) << 48)
            )
            packed = tl.sum(counted)
            if TAIL:
                tv = tl.load(Logits + row * STRIDE + MAIN, n > MAIN, other=0.0)
                tk = _ordered_key(tv)
                tvalid = n > MAIN
                packed += (
                    (tvalid & (tk > p0)).to(tl.uint64)
                    | ((tvalid & (tk > p1)).to(tl.uint64) << 16)
                    | ((tvalid & (tk > p2)).to(tl.uint64) << 32)
                    | ((tvalid & (tk > p3)).to(tl.uint64) << 48)
                )
            c0 = (packed & 65535).to(tl.int32)
            c1 = ((packed >> 16) & 65535).to(tl.int32)
            c2 = ((packed >> 32) & 65535).to(tl.int32)
            c3 = ((packed >> 48) & 65535).to(tl.int32)
            bracket = (c3 < K) & (c0 >= K)
            lk = tl.where(c2 >= K, p2, tl.where(c1 >= K, p1, p0))
            hk = tl.where(c2 >= K, p3, tl.where(c1 >= K, p2, p1))
            nlower = tl.where(c2 >= K, c2, tl.where(c1 >= K, c1, c0))
            nupper = tl.where(c2 >= K, c3, tl.where(c1 >= K, c2, c1))
            above = valid & (key > hk) & bracket
            candidates = valid & tl.where(bracket, (key > lk) & (key <= hk), True)
            remain = K - tl.where(bracket, nupper, 0)
            if TAIL:
                ta = tvalid & (tk > hk) & bracket
                te = tvalid & tl.where(bracket, (tk > lk) & (tk <= hk), True)
            ne = nlower - nupper
            if bracket and (ne <= CAP):
                storage = tle.gpu.alloc(
                    [CAP],
                    tl.uint64,
                    layout=None,
                    scope=tle.gpu.smem,
                    nv_mma_shared_layout=False,
                )
                ptr = tle.gpu.local_ptr(storage, (0,))
                ar, er = _bracket_packed_ranks(above, candidates, MAIN)
                er -= 1
                tl.store(Output + row * K + ar - 1, i, above)
                if TAIL:
                    tl.store(Output + row * K + nupper - 1, MAIN, ta)
                word = (key.to(tl.uint64) << 32) | (0xFFFFFFFF - i.to(tl.uint32)).to(
                    tl.uint64
                )
                tl.store(ptr + er, word, candidates)
                if TAIL:
                    tw = (tk.to(tl.uint64) << 32) | tl.full(
                        (), 0xFFFFFFFF - MAIN, tl.uint64
                    )
                    tl.store(ptr + ne - 1, tw, te)
                tl.debug_barrier()
                q = tl.arange(0, CAP)
                words = tl.load(ptr + q, q < ne, other=0)
                ordered = tl.sort(words, descending=True)
                idx = (0xFFFFFFFF - (ordered & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
                tl.store(Output + row * K + K - remain + q, idx, q < remain)
                selected = above
                if TAIL:
                    ts = ta
            else:
                lo = tl.min(tl.where(candidates, key, 0xFFFFFFFF), 0)
                hi = tl.max(tl.where(candidates, key, 0), 0)
                if TAIL:
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
                    er = _mask_ranks(eq, MAIN)
                    selected = above | gt | (eq & (er <= remain - ng))
                    if TAIL:
                        ts = ta | (
                            te
                            & (
                                (tk > lo)
                                | ((tk == lo) & (tl.sum(eq.to(tl.int32)) < remain - ng))
                            )
                        )
            if not (bracket and (ne <= CAP)):
                pos = _mask_ranks(selected, MAIN) - 1
                tl.store(Output + row * K + pos, i, selected)
                if TAIL:
                    tl.store(Output + row * K + tl.sum(selected.to(tl.int32)), MAIN, ts)


@triton.jit
def _cooperative_topk_hygon_bucket(v):
    bits = v.to(tl.uint32, bitcast=True)
    q = tl.minimum(tl.maximum(v * 32.0 + 128.0, 0.0), 255.0).to(tl.int32)
    return tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 255, q)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_hygon_histogram"),
    key=["WIDTH", "ROWS", "PARTS"],
    strategy=["default"] * 3,
)
@triton.jit
def _cooperative_topk_hygon_histogram(
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
    SPLIT: tl.constexpr = (WIDTH % PARTS != 0) and ((WIDTH // PARTS) % 1024 == 0)
    chunk: tl.constexpr = WIDTH // PARTS if SPLIT else triton.cdiv(WIDTH, PARTS)
    end = tl.minimum((p + 1) * chunk, n)
    h = tl.full((256,), 0, tl.int32)
    for start in range(p * chunk, end, BLOCK):
        i = start + tl.arange(0, BLOCK)
        valid = i < end
        v = tl.load(Logits + r * STRIDE + i, valid, other=0.0)
        h += tl.histogram(_cooperative_topk_hygon_bucket(v), 256, mask=valid)
    if SPLIT:
        if (p == PARTS - 1) & (n > chunk * PARTS):
            ti = chunk * PARTS + tl.arange(0, triton.next_power_of_2(WIDTH % PARTS))
            tv = tl.load(Logits + r * STRIDE + ti, ti < n, other=0.0)
            h += tl.histogram(_cooperative_topk_hygon_bucket(tv), 256, mask=ti < n)
    tl.store(Hist + (r * PARTS + p) * 256 + tl.arange(0, 256), h)


# Fixed per-row reduction: geometry follows the histogram/state protocol.
# Exempt from autotuning, like the shared radix threshold kernel.
@libentry()
@triton.jit
def _cooperative_topk_hygon_threshold(
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


@triton.jit
def _compact_chunk(
    Logits,
    Output,
    Keys,
    Indices,
    r,
    STRIDE: tl.constexpr,
    start,
    end,
    d,
    na,
    ne,
    ao,
    eo,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = start + tl.arange(0, BLOCK)
    valid = i < end
    v = tl.load(Logits + r * STRIDE + i, valid, other=0.0)
    q = _cooperative_topk_hygon_bucket(v)
    a = valid & (q > d)
    e = valid & (q == d)
    ar, er = _packed_ranks(a, e, BLOCK)
    tl.store(Output + r * K + ao + ar - 1, i, a)
    ao += tl.sum(a.to(tl.int32))
    if ne <= CAPACITY:
        tl.store(Keys + r * CAPACITY + eo + er - 1, _ordered_key(v), e)
        tl.store(Indices + r * CAPACITY + eo + er - 1, i, e)
        eo += tl.sum(e.to(tl.int32))
    elif ne == K - na:
        tl.store(Output + r * K + na + eo + er - 1, i, e)
        eo += tl.sum(e.to(tl.int32))
    return ao, eo


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_hygon_suffix_histogram"),
    key=["WIDTH", "ROWS", "PARTS"],
    strategy=["default"] * 3,
)
@triton.jit
def _cooperative_topk_hygon_suffix_histogram(
    Logits,
    Lengths,
    Hist,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.static_assert(PARTS == 16 and 8192 < WIDTH and WIDTH <= 16384)
    tl.static_assert(triton.cdiv(WIDTH, PARTS) < 65536)
    p = tl.program_id(0).to(tl.int64)
    r = tl.program_id(1).to(tl.int64)
    n = tl.minimum(tl.maximum(tl.load(Lengths + r), 0), WIDTH)
    SPLIT: tl.constexpr = (WIDTH % PARTS != 0) and ((WIDTH // PARTS) % 1024 == 0)
    chunk: tl.constexpr = WIDTH // PARTS if SPLIT else triton.cdiv(WIDTH, PARTS)
    end = tl.minimum((p + 1) * chunk, n)
    h = tl.full((256,), 0, tl.int32)
    for start in range(p * chunk, end, BLOCK):
        i = start + tl.arange(0, BLOCK)
        valid = i < end
        v = tl.load(Logits + r * STRIDE + i, valid, other=0.0)
        h += tl.histogram(_cooperative_topk_hygon_bucket(v), 256, mask=valid)
    if SPLIT:
        if (p == PARTS - 1) & (n > chunk * PARTS):
            ti = chunk * PARTS + tl.arange(0, triton.next_power_of_2(WIDTH % PARTS))
            tv = tl.load(Logits + r * STRIDE + ti, ti < n, other=0.0)
            h += tl.histogram(_cooperative_topk_hygon_bucket(tv), 256, mask=ti < n)
    tl.store(
        Hist + (r * PARTS + p) * 256 + tl.arange(0, 256), tl.cumsum(h, reverse=True)
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_hygon_suffix_compact"),
    key=["WIDTH", "K", "ROWS", "PARTS", "CAPACITY"],
    strategy=["default"] * 5,
)
@triton.jit
def _cooperative_topk_hygon_suffix_compact(
    Logits,
    Lengths,
    Output,
    Hist,
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
    tl.static_assert(PARTS == 16 and 8192 < WIDTH and WIDTH <= 16384)
    tl.static_assert(triton.cdiv(WIDTH, PARTS) < 65536)
    p = tl.program_id(0).to(tl.int64)
    r = tl.program_id(1).to(tl.int64)
    n = tl.minimum(tl.maximum(tl.load(Lengths + r), 0), WIDTH)
    if n <= K:
        if p == 0:
            j = tl.arange(0, K)
            tl.store(Output + r * K + j, tl.where(j < n, j, -1))
            tl.store(State + r * 3, 0)
            tl.store(State + r * 3 + 1, 0)
            tl.store(State + r * 3 + 2, 0)
    else:
        base = State + r * 3
        pp = tl.arange(0, PARTS)
        b = tl.arange(0, 256)
        suffix = tl.sum(
            tl.load(Hist + (r * PARTS + pp[:, None]) * 256 + b[None, :]).to(tl.int32), 0
        )
        d = tl.max(tl.where(suffix >= K, b, 0), 0)
        above = tl.load(Hist + (r * PARTS + pp) * 256 + d + 1, d < 255, other=0).to(
            tl.int32
        )
        equal = tl.load(Hist + (r * PARTS + pp) * 256 + d).to(tl.int32) - above
        na = tl.sum(above)
        ne = tl.sum(equal)
        ao = tl.sum(tl.where(pp < p, above, 0))
        eo = tl.sum(tl.where(pp < p, equal, 0))
        if p == 0:
            tl.store(base, d)
            tl.store(base + 1, na)
            tl.store(base + 2, ne)
        SPLIT: tl.constexpr = (WIDTH % PARTS != 0) and ((WIDTH // PARTS) % 1024 == 0)
        chunk: tl.constexpr = WIDTH // PARTS if SPLIT else triton.cdiv(WIDTH, PARTS)
        end = tl.minimum((p + 1) * chunk, n)
        for start in range(p * chunk, end, BLOCK):
            ao, eo = _compact_chunk(
                Logits,
                Output,
                Keys,
                Indices,
                r,
                STRIDE,
                start,
                end,
                d,
                na,
                ne,
                ao,
                eo,
                K,
                CAPACITY,
                BLOCK,
            )
        if SPLIT:
            if (p == PARTS - 1) & (n > chunk * PARTS):
                ao, eo = _compact_chunk(
                    Logits,
                    Output,
                    Keys,
                    Indices,
                    r,
                    STRIDE,
                    chunk * PARTS,
                    n,
                    d,
                    na,
                    ne,
                    ao,
                    eo,
                    K,
                    CAPACITY,
                    triton.next_power_of_2(WIDTH % PARTS),
                )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_hygon_compact"),
    key=["WIDTH", "K", "ROWS", "PARTS", "CAPACITY"],
    strategy=["default"] * 5,
)
@triton.jit
def _cooperative_topk_hygon_compact(
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
        SPLIT: tl.constexpr = (WIDTH % PARTS != 0) and ((WIDTH // PARTS) % 1024 == 0)
        chunk: tl.constexpr = WIDTH // PARTS if SPLIT else triton.cdiv(WIDTH, PARTS)
        end = tl.minimum((p + 1) * chunk, n)
        for start in range(p * chunk, end, BLOCK):
            ao, eo = _compact_chunk(
                Logits,
                Output,
                Keys,
                Indices,
                r,
                STRIDE,
                start,
                end,
                d,
                na,
                ne,
                ao,
                eo,
                K,
                CAPACITY,
                BLOCK,
            )
        if SPLIT:
            if (p == PARTS - 1) & (n > chunk * PARTS):
                ao, eo = _compact_chunk(
                    Logits,
                    Output,
                    Keys,
                    Indices,
                    r,
                    STRIDE,
                    chunk * PARTS,
                    n,
                    d,
                    na,
                    ne,
                    ao,
                    eo,
                    K,
                    CAPACITY,
                    triton.next_power_of_2(WIDTH % PARTS),
                )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_hygon_finish"),
    key=["WIDTH", "K", "ROWS", "PARTS", "CAPACITY"],
    strategy=["default"] * 5,
)
@triton.jit
def _cooperative_topk_hygon_finish(
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
            if part * TILE < ne:
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
                valid = (i < n) & (_cooperative_topk_hygon_bucket(v) == d)
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
                    valid = (i < n) & (_cooperative_topk_hygon_bucket(v) == d)
                    key = _ordered_key(v)
                    cnt += tl.sum((valid & (key > mid)).to(tl.int32))
                exact = cnt == remain
                lo = tl.where(cnt >= remain, mid + 1, lo)
                hi = tl.where(cnt < remain, mid, hi)
            ng = tl.full((), 0, tl.int32)
            for start in range(0, n, BLOCK):
                i = start + tl.arange(0, BLOCK)
                v = tl.load(Logits + r * STRIDE + i, i < n, other=0.0)
                valid = (i < n) & (_cooperative_topk_hygon_bucket(v) == d)
                key = _ordered_key(v)
                ng += tl.sum((valid & (key > lo)).to(tl.int32))
            pos = na
            ties = tl.full((), 0, tl.int32)
            for start in range(0, n, BLOCK):
                i = start + tl.arange(0, BLOCK)
                v = tl.load(Logits + r * STRIDE + i, i < n, other=0.0)
                valid = (i < n) & (_cooperative_topk_hygon_bucket(v) == d)
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


# BW1000 event measurements selected one schedule for each row regime.
# Keep the schedules in the standard config/tuner mechanism without letting
# data from correctness tests retune a borderline performance case.
_cooperative_topk_hygon_medium_histogram = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cooperative_topk_hygon_medium_histogram"),
        key=["WIDTH", "ROWS", "PARTS"],
        strategy=["default"] * 3,
    )(_cooperative_topk_hygon_histogram.jit_function)
)
_cooperative_topk_hygon_medium_compact = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cooperative_topk_hygon_medium_compact"),
        key=["WIDTH", "K", "ROWS", "PARTS", "CAPACITY"],
        strategy=["default"] * 5,
    )(_cooperative_topk_hygon_compact.jit_function)
)
_cooperative_topk_hygon_medium_finish = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cooperative_topk_hygon_medium_finish"),
        key=["WIDTH", "K", "ROWS", "PARTS", "CAPACITY"],
        strategy=["default"] * 5,
    )(_cooperative_topk_hygon_finish.jit_function)
)


# Four queries per CTA keep the 128-candidate reduction inside one wave64.
_cooperative_topk_hygon_suffix_finish = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cooperative_topk_hygon_suffix_finish"),
        key=["WIDTH", "K", "ROWS", "PARTS", "CAPACITY"],
        strategy=["default"] * 5,
    )(_cooperative_topk_hygon_finish.jit_function)
)


def _cooperative_topk_wide(logits, lengths, output, workspace, k, max_seq_len):
    rows, width = logits.shape
    # The partition count exposes enough CTAs for each batch regime while
    # keeping the threshold reduction bounded. The caller's workspace ABI
    # remains unchanged; these independent arrays are private to this call.
    if rows > 16:
        parts = 8
    elif width <= 16384:
        parts = 16
    elif width < 262144:
        parts = 32
    else:
        parts = 128
    # Narrow K=1024 rows need a smaller boundary bucket. Reducing the
    # pairwise refinement matrix preserves the exact overflow path below.
    capacity = 128 if k == 1024 and width <= 32768 else 256
    use_suffix = rows <= 16 and k == 1024 and 8192 < width <= 16384
    # Larger grids need the smaller workgroups to retain CTA occupancy.
    # In particular, crossing 16384 -> 16385 must not double the workgroup.
    use_medium = width <= 32768 or rows > 16
    histogram_kernel = (
        _cooperative_topk_hygon_suffix_histogram
        if use_suffix
        else (
            _cooperative_topk_hygon_medium_histogram
            if use_medium
            else _cooperative_topk_hygon_histogram
        )
    )
    compact_kernel = (
        _cooperative_topk_hygon_medium_compact
        if use_medium
        else _cooperative_topk_hygon_compact
    )
    finish_kernel = (
        _cooperative_topk_hygon_suffix_finish
        if use_suffix
        else (
            _cooperative_topk_hygon_medium_finish
            if use_medium
            else _cooperative_topk_hygon_finish
        )
    )
    # Each medium partition contributes at most 1024 suffix counts.
    hist = torch.empty(
        (rows, parts, 256),
        dtype=torch.uint16 if use_suffix else torch.int32,
        device=logits.device,
    )
    if use_suffix:
        # The fused path only needs d/above/equal per row. Reuse a bounded,
        # disjoint region of the validated caller workspace without a copy.
        state = workspace.view(-1)[: rows * 12].view(torch.int32)
    else:
        state = torch.empty(
            (rows, 2 * parts + 3), dtype=torch.int32, device=logits.device
        )
    keys = torch.empty((rows, capacity), dtype=torch.uint32, device=logits.device)
    indices = torch.empty((rows, capacity), dtype=torch.int32, device=logits.device)
    histogram_kernel[(parts, rows)](
        logits, lengths, hist, width, logits.stride(0), parts, rows
    )
    if use_suffix:
        # Each CTA derives its own offsets from the completed histogram.
        # Only partition zero writes the state consumed by the next kernel.
        _cooperative_topk_hygon_suffix_compact[(parts, rows)](
            logits,
            lengths,
            output,
            hist,
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
    else:
        _cooperative_topk_hygon_threshold[(rows,)](
            hist, state, k, parts, num_warps=1 if parts <= 16 else 4, num_stages=1
        )
        compact_kernel[(parts, rows)](
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
    finish_kernel[lambda meta: (triton.cdiv(capacity, meta["TILE"]), rows)](
        logits,
        lengths,
        output,
        state,
        keys,
        indices,
        width,
        logits.stride(0),
        k,
        0 if use_suffix else parts,
        capacity,
        rows,
    )


def cooperative_topk(logits, lengths, output, workspace, k, max_seq_len):
    # Read shape only after establishing that it is a rank-two tensor;
    # the shared entry remains responsible for all argument validation.
    use_bracket = (
        _HAS_TLE
        and isinstance(logits, torch.Tensor)
        and logits.ndim == 2
        and logits.shape[1] in (4096, 4097)
        and isinstance(k, int)
        and k == 512
    )
    return _cooperative_topk_impl(
        logits,
        lengths,
        output,
        workspace,
        k,
        max_seq_len,
        small_kernel=(
            _cooperative_topk_hygon_bracket_small
            if use_bracket
            else _cooperative_topk_hygon_small
        ),
        wide_kernel=_cooperative_topk_wide,
        wide_max_width=1048576,
    )
