# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Moore Threads coarse selection with exact compacted-bucket refinement."""

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

HAS_TLE = bool(
    tle is not None
    and hasattr(tle, "gpu")
    and hasattr(tle.gpu, "alloc")
    and hasattr(tle.gpu, "local_ptr")
    and hasattr(tle.gpu, "smem")
    and hasattr(tle, "cumsum")
)


@triton.jit
def _cooperative_topk_bucket(x, BINS: tl.constexpr):
    digit = tl.minimum(tl.maximum(x * (BINS / 16.0) + BINS / 2.0, 0.0), BINS - 1.0).to(
        tl.int32
    )
    bits = x.to(tl.uint32, bitcast=True)
    return tl.where((bits & 0x7FFFFFFF) > 0x7F800000, BINS - 1, digit)


if HAS_TLE:

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("cooperative_topk_mthreads_compact"),
        key=["WIDTH", "K", "ROWS"],
        strategy=["default"] * 3,
    )
    @triton.jit
    def _cooperative_topk_compact(
        Logits,
        Lengths,
        Output,
        WIDTH: tl.constexpr,
        STRIDE: tl.constexpr,
        K: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK: tl.constexpr,
        TILE: tl.constexpr,
        BINS: tl.constexpr,
    ):
        CAP: tl.constexpr = 64
        tail_bits: tl.constexpr = (WIDTH % TILE - 1) if WIDTH % TILE else 0
        tail_1: tl.constexpr = tail_bits | (tail_bits >> 1)
        tail_2: tl.constexpr = tail_1 | (tail_1 >> 2)
        tail_4: tl.constexpr = tail_2 | (tail_2 >> 4)
        tail_8: tl.constexpr = tail_4 | (tail_4 >> 8)
        TAIL: tl.constexpr = tail_8 + 1
        row = tl.program_id(0).to(tl.int64)
        length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
        if length <= K:
            short = tl.arange(0, K)
            tl.store(Output + row * K + short, tl.where(short < length, short, -1))
        else:
            lanes = tl.arange(0, TILE)
            bins = tl.arange(0, BINS)
            hist = tle.gpu.alloc(
                [BINS],
                tl.int32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            hist_ptr = tle.gpu.local_ptr(hist, (0,))
            compact = tle.gpu.alloc(
                [CAP],
                tl.uint32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            compact_ptr = tle.gpu.local_ptr(compact, (0,))
            tl.store(hist_ptr + bins, 0)
            tl.debug_barrier()
            for start in range(0, tl.minimum(length, (WIDTH // TILE) * TILE), TILE):
                offsets = start + lanes
                valid = offsets < length
                values = tl.load(Logits + row * STRIDE + offsets, valid, other=0.0)
                digit = _cooperative_topk_bucket(values, BINS)
                tl.atomic_add(
                    hist_ptr + digit, 1, mask=valid, sem="relaxed", scope="cta"
                )
            if WIDTH % TILE != 0:
                h_tail_i = (WIDTH // TILE) * TILE + tl.arange(0, TAIL)
                h_tail_valid = h_tail_i < length
                h_tail_values = tl.load(
                    Logits + row * STRIDE + h_tail_i, h_tail_valid, other=0.0
                )
                h_tail_digit = _cooperative_topk_bucket(h_tail_values, BINS)
                tl.atomic_add(
                    hist_ptr + h_tail_digit,
                    1,
                    mask=h_tail_valid,
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()
            counts = tl.load(hist_ptr + bins)
            group_ids = tl.arange(0, BINS // 32)
            inner_ids = tl.arange(0, 32)
            group_counts = tl.sum(counts.reshape((BINS // 32, 32)), 1)
            group_suffix = tl.cumsum(group_counts, reverse=True)
            group_digit = tl.max(tl.where(group_suffix >= K, group_ids, 0), 0)
            above_groups = tl.sum(tl.where(group_ids > group_digit, group_counts, 0), 0)
            inside = tl.load(hist_ptr + group_digit * 32 + inner_ids)
            inner_suffix = tl.cumsum(inside, reverse=True)
            inner_digit = tl.max(
                tl.where(inner_suffix >= K - above_groups, inner_ids, 0), 0
            )
            coarse_digit = group_digit * 32 + inner_digit
            above_count = above_groups + tl.sum(
                tl.where(inner_ids > inner_digit, inside, 0), 0
            )
            equal_count = tl.sum(tl.where(inner_ids == inner_digit, inside, 0), 0)
            remaining = K - above_count
            counters = tle.gpu.alloc(
                [2],
                tl.int32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            counter_ptr = tle.gpu.local_ptr(counters, (0,))
            tl.store(counter_ptr + tl.arange(0, 2), 0)
            tl.debug_barrier()
            zeros = tl.full((TILE,), 0, tl.int32)
            for start in range(0, tl.minimum(length, (WIDTH // TILE) * TILE), TILE):
                offsets = start + lanes
                valid = offsets < length
                values = tl.load(Logits + row * STRIDE + offsets, valid, other=0.0)
                digit = _cooperative_topk_bucket(values, BINS)
                above = valid & (digit > coarse_digit)
                equal = valid & (digit == coarse_digit) & (equal_count <= CAP)
                above_slots = tl.atomic_add(
                    counter_ptr + zeros, 1, mask=above, sem="relaxed", scope="cta"
                )
                equal_slots = tl.atomic_add(
                    counter_ptr + 1 + zeros, 1, mask=equal, sem="relaxed", scope="cta"
                )
                tl.store(Output + row * K + above_slots, offsets, above)
                tl.store(compact_ptr + equal_slots, offsets.to(tl.uint32), equal)
            if WIDTH % TILE != 0:
                s_tail_i = (WIDTH // TILE) * TILE + tl.arange(0, TAIL)
                s_tail_valid = s_tail_i < length
                s_tail_values = tl.load(
                    Logits + row * STRIDE + s_tail_i, s_tail_valid, other=0.0
                )
                s_tail_digit = _cooperative_topk_bucket(s_tail_values, BINS)
                s_tail_above = s_tail_valid & (s_tail_digit > coarse_digit)
                s_tail_equal = (
                    s_tail_valid & (s_tail_digit == coarse_digit) & (equal_count <= CAP)
                )
                s_tail_zeros = tl.full((TAIL,), 0, tl.int32)
                s_tail_aslot = tl.atomic_add(
                    counter_ptr + s_tail_zeros,
                    1,
                    mask=s_tail_above,
                    sem="relaxed",
                    scope="cta",
                )
                s_tail_eslot = tl.atomic_add(
                    counter_ptr + 1 + s_tail_zeros,
                    1,
                    mask=s_tail_equal,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(Output + row * K + s_tail_aslot, s_tail_i, s_tail_above)
                tl.store(
                    compact_ptr + s_tail_eslot, s_tail_i.to(tl.uint32), s_tail_equal
                )
            tl.debug_barrier()
            if equal_count <= CAP:
                pi = tl.arange(0, 64)
                pv = pi < equal_count
                pidx = tl.load(compact_ptr + pi, pv, other=0)
                pk = _ordered_key(tl.load(Logits + row * STRIDE + pidx, pv, other=0.0))
                outranks = pv[None, :] & (
                    (pk[None, :] > pk[:, None])
                    | ((pk[None, :] == pk[:, None]) & (pi[None, :] < pi[:, None]))
                )
                ranks = tl.sum(outranks.to(tl.int32), 1)
                tl.store(
                    Output + row * K + above_count + ranks,
                    pidx,
                    pv & (ranks < remaining),
                )
            else:
                # Exact overflow path: refine the coarse bucket without ever
                # truncating it, then emit its selected values in streaming tiles.
                refine_bins = tl.arange(0, 256)
                prefix = tl.full((), 0, tl.uint32)
                for shift in tl.static_range(24, -1, -8):
                    tl.store(hist_ptr + refine_bins, 0)
                    tl.debug_barrier()
                    for start in range(0, length, TILE):
                        offsets = start + lanes
                        valid = offsets < length
                        values = tl.load(
                            Logits + row * STRIDE + offsets, valid, other=0.0
                        )
                        keys = _ordered_key(values)
                        active = valid & (
                            _cooperative_topk_bucket(values, BINS) == coarse_digit
                        )
                        if shift < 24:
                            active &= (keys >> (shift + 8)) == (prefix >> (shift + 8))
                        digit = ((keys >> shift) & 255).to(tl.int32)
                        tl.atomic_add(
                            hist_ptr + digit, 1, mask=active, sem="relaxed", scope="cta"
                        )
                    tl.debug_barrier()
                    refine_counts = tl.load(hist_ptr + refine_bins)
                    refine_suffix = tl.cumsum(refine_counts, reverse=True)
                    refine_digit = tl.max(
                        tl.where(refine_suffix >= remaining, refine_bins, 0), 0
                    )
                    remaining -= tl.sum(
                        tl.where(refine_bins > refine_digit, refine_counts, 0), 0
                    )
                    prefix |= refine_digit.to(tl.uint32) << shift
                    tl.debug_barrier()
                greater_offset = above_count
                tie_offset = tl.full((), 0, tl.int32)
                for start in range(0, length, TILE):
                    offsets = start + lanes
                    valid = offsets < length
                    values = tl.load(Logits + row * STRIDE + offsets, valid, other=0.0)
                    keys = _ordered_key(values)
                    active = valid & (
                        _cooperative_topk_bucket(values, BINS) == coarse_digit
                    )
                    greater = active & (keys > prefix)
                    equal = active & (keys == prefix)
                    greater_rank = tl.cumsum(greater.to(tl.int32))
                    tie_rank = tie_offset + tl.cumsum(equal.to(tl.int32))
                    pos = tl.where(
                        greater,
                        greater_offset + greater_rank - 1,
                        K - remaining + tie_rank - 1,
                    )
                    take = greater | (equal & (tie_rank <= remaining))
                    tl.store(Output + row * K + pos, offsets, take)
                    greater_offset += tl.sum(greater.to(tl.int32), 0)
                    tie_offset += tl.sum(equal.to(tl.int32), 0)

    @triton.jit
    def _cooperative_topk_wide_bucket(values):
        bits = values.to(tl.uint32, bitcast=True)
        mag = bits & 2147483647
        q = tl.minimum(tl.maximum(values * 32.0 + 128.0, 0.0), 255.0).to(tl.int32)
        return tl.where(mag > 2139095040, 255, q)

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("cooperative_topk_mthreads_wide_histogram"),
        key=["WIDTH", "ROWS", "PARTS"],
        strategy=["default", "default", "default"],
    )
    @triton.jit
    def _cooperative_topk_wide_histogram(
        Logits,
        Lengths,
        Histograms,
        WIDTH: tl.constexpr,
        STRIDE: tl.constexpr,
        PARTS: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        p = tl.program_id(0).to(tl.int64)
        r = tl.program_id(1).to(tl.int64)
        n = tl.minimum(tl.maximum(tl.load(Lengths + r), 0), WIDTH)
        shared = tle.gpu.alloc(
            [256],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        bins = tl.arange(0, 256)
        hist = tle.gpu.local_ptr(shared, (bins,))
        tl.store(hist, 0)
        tl.debug_barrier()
        chunk: tl.constexpr = triton.cdiv(WIDTH, PARTS)
        end = tl.minimum((p + 1) * chunk, n)
        for start in range(p * chunk, end, BLOCK):
            i = start + tl.arange(0, BLOCK)
            valid = i < end
            v = tl.load(Logits + r * STRIDE + i, valid, other=0.0)
            q = _cooperative_topk_wide_bucket(v)
            ptr = tle.gpu.local_ptr(shared, (q,))
            tl.atomic_add(ptr, 1, mask=valid, sem="relaxed", scope="cta")
        tl.debug_barrier()
        count = tl.load(hist)
        tl.store(Histograms + (r * PARTS + p) * 256 + bins, count)

    @triton.jit
    def _sum_matches(digit1, above1, equal1, digit2, above2, equal2):
        return digit1 + digit2, above1 + above2, equal1 + equal2

    # Fixed reduction geometry follows the per-row histogram/state layout.
    @libentry()
    @triton.jit
    def _cooperative_topk_wide_threshold(
        Histograms, State, K: tl.constexpr, PARTS: tl.constexpr
    ):
        r = tl.program_id(0).to(tl.int64)
        p = tl.arange(0, PARTS)
        b = tl.arange(0, 256)
        h = tl.load(Histograms + (r * PARTS + p[:, None]) * 256 + b[None, :])
        total = tl.sum(h, 0)
        suffix = tl.cumsum(total, reverse=True)
        hit = (suffix >= K) & (suffix - total < K)
        digit, na, ne = tl.reduce(
            (
                tl.where(hit, b, 0),
                tl.where(hit, suffix - total, 0),
                tl.where(hit, total, 0),
            ),
            0,
            _sum_matches,
        )
        tl.store(State + r * (2 * PARTS + 3), digit)
        tl.store(State + r * (2 * PARTS + 3) + 1, na)
        tl.store(State + r * (2 * PARTS + 3) + 2, ne)
        tl.store(State + r * (2 * PARTS + 3) + 3, 0)
        tl.store(State + r * (2 * PARTS + 3) + 4, 0)

    # Fixed reduction geometry follows the per-row histogram/state layout.
    @libentry()
    @triton.jit
    def _cooperative_topk_wide_threshold_sparse(
        Histograms, State, K: tl.constexpr, PARTS: tl.constexpr
    ):
        r = tl.program_id(0).to(tl.int64)
        p = tl.arange(0, PARTS)
        b = tl.arange(0, 256)
        h = tl.load(Histograms + (r * PARTS + p[:, None]) * 256 + b[None, :])
        total = tl.sum(h, 0)
        suffix = tl.cumsum(total, reverse=True)
        d = tl.max(tl.where(suffix >= K, b, 0), 0)
        a = tl.sum(tl.where(b[None, :] > d, h, 0), 1)
        e = tl.load(Histograms + (r * PARTS + p) * 256 + d)
        base = State + r * (4 * PARTS + 3)
        tl.store(base, d)
        tl.store(base + 1, tl.sum(a))
        tl.store(base + 2, tl.sum(e))
        tl.store(base + 3 + p, tl.cumsum(a) - a)
        tl.store(base + 3 + PARTS + p, tl.cumsum(e) - e)
        tl.store(base + 3 + 2 * PARTS + p, 0)
        tl.store(base + 3 + 3 * PARTS + p, 0)

    # Fixed measured launch: this kernel consumes counters initialized by the
    # threshold kernel and cannot be independently repeated during autotuning.
    @libentry()
    @triton.jit
    def _cooperative_topk_wide_compact_sparse(
        Logits,
        Lengths,
        Output,
        State,
        CandidateKeys,
        CandidateIndices,
        WIDTH: tl.constexpr,
        STRIDE: tl.constexpr,
        K: tl.constexpr,
        PARTS: tl.constexpr,
        BLOCK: tl.constexpr,
        CAPACITY: tl.constexpr,
    ):
        p = tl.program_id(0).to(tl.int64)
        sub = tl.program_id(1)
        r = tl.program_id(2).to(tl.int64)
        n = tl.minimum(tl.maximum(tl.load(Lengths + r), 0), WIDTH)
        if n <= K:
            if (p == 0) & (sub == 0):
                i = tl.arange(0, K)
                tl.store(Output + r * K + i, tl.where(i < n, i, -1))
        else:
            base = State + r * (4 * PARTS + 3)
            d = tl.load(base)
            na = tl.load(base + 1)
            ne = tl.load(base + 2)
            chunk: tl.constexpr = triton.cdiv(WIDTH, PARTS)
            i = p * chunk + sub * BLOCK + tl.arange(0, BLOCK)
            valid = i < tl.minimum((p + 1) * chunk, n)
            v = tl.load(Logits + r * STRIDE + i, valid, other=0.0)
            q = _cooperative_topk_wide_bucket(v)
            a = valid & (q > d)
            e = valid & (q == d)
            zeros = tl.full((BLOCK,), 0, tl.int32)
            ao = tl.load(base + 3 + p) + tl.atomic_add(
                base + 3 + 2 * PARTS + p + zeros, 1, mask=a, sem="relaxed"
            )
            tl.store(Output + r * K + ao, i, a)
            if ne <= CAPACITY:
                eo = tl.load(base + 3 + PARTS + p) + tl.atomic_add(
                    base + 3 + 3 * PARTS + p + zeros, 1, mask=e, sem="relaxed"
                )
                tl.store(CandidateKeys + r * CAPACITY + eo, _ordered_key(v), e)
                tl.store(CandidateIndices + r * CAPACITY + eo, i, e)
            elif ne == K - na:
                eo = tl.load(base + 3 + PARTS + p) + tl.atomic_add(
                    base + 3 + 3 * PARTS + p + zeros, 1, mask=e, sem="relaxed"
                )
                tl.store(Output + r * K + na + eo, i, e)

    # Fixed measured launch: this kernel consumes counters initialized by the
    # threshold kernel and cannot be independently repeated during autotuning.
    @libentry()
    @triton.jit
    def _cooperative_topk_wide_compact(
        Logits,
        Lengths,
        Output,
        State,
        CandidateKeys,
        CandidateIndices,
        WIDTH: tl.constexpr,
        STRIDE: tl.constexpr,
        K: tl.constexpr,
        PARTS: tl.constexpr,
        BLOCK: tl.constexpr,
        CAPACITY: tl.constexpr,
    ):
        p = tl.program_id(0)
        r = tl.program_id(1).to(tl.int64)
        n = tl.minimum(tl.maximum(tl.load(Lengths + r), 0), WIDTH)
        if n <= K:
            if p == 0:
                i = tl.arange(0, K)
                tl.store(Output + r * K + i, tl.where(i < n, i, -1))
        else:
            base = State + r * (2 * PARTS + 3)
            d = tl.load(base)
            na = tl.load(base + 1)
            ne = tl.load(base + 2)
            i = p * BLOCK + tl.arange(0, BLOCK)
            valid = i < n
            v = tl.load(Logits + r * STRIDE + i, valid, other=0.0)
            q = _cooperative_topk_wide_bucket(v)
            a = valid & (q > d)
            e = valid & (q == d)
            ar, ac = tle.cumsum(a.to(tl.int32), axis=0)
            er, ec = tle.cumsum(e.to(tl.int32), axis=0)
            ao = tl.atomic_add(base + 3, ac, sem="relaxed")
            eo = tl.atomic_add(base + 4, ec, sem="relaxed")
            tl.store(Output + r * K + ao + ar, i, a)
            if ne <= CAPACITY:
                tl.store(CandidateKeys + r * CAPACITY + eo + er, _ordered_key(v), e)
                tl.store(CandidateIndices + r * CAPACITY + eo + er, i, e)
            elif ne == K - na:
                tl.store(Output + r * K + na + eo + er, i, e)

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("cooperative_topk_mthreads_wide_finish"),
        key=["WIDTH", "K", "ROWS", "PARTS"],
        strategy=["default", "default", "default", "default"],
    )
    @triton.jit
    def _cooperative_topk_wide_finish(
        Logits,
        Lengths,
        Output,
        State,
        CandidateKeys,
        CandidateIndices,
        WIDTH: tl.constexpr,
        STRIDE: tl.constexpr,
        K: tl.constexpr,
        PARTS: tl.constexpr,
        ROWS: tl.constexpr,
        CAPACITY: tl.constexpr,
        BLOCK: tl.constexpr,
        STATE_STRIDE: tl.constexpr = 0,
        TILE: tl.constexpr = 8,
    ):
        part = tl.program_id(0)
        r = tl.program_id(1).to(tl.int64)
        n = tl.minimum(tl.maximum(tl.load(Lengths + r), 0), WIDTH)
        base = State + r * (STATE_STRIDE if STATE_STRIDE else 2 * PARTS + 3)
        d = tl.load(base)
        na = tl.load(base + 1)
        ne = tl.load(base + 2)
        remain = K - na
        if n > K:
            if ne <= CAPACITY:
                q = part * TILE + tl.arange(0, TILE)
                j = tl.arange(0, CAPACITY)
                valid_q = q < ne
                query = tl.load(CandidateKeys + r * CAPACITY + q, valid_q, other=0)
                other = tl.load(CandidateKeys + r * CAPACITY + j, j < ne, other=0)
                higher = (other[None, :] > query[:, None]) | (
                    other[None, :] == query[:, None]
                ) & (j[None, :] < q[:, None])
                rank = tl.sum(((j[None, :] < ne) & higher).to(tl.int32), 1)
                index = tl.load(CandidateIndices + r * CAPACITY + q, valid_q, other=0)
                tl.store(Output + r * K + na + rank, index, valid_q & (rank < remain))
            elif (ne != remain) & (part == 0):
                lo = tl.full((), 4294967295, tl.uint32)
                hi = tl.full((), 0, tl.uint32)
                for start in range(0, n, BLOCK):
                    i = start + tl.arange(0, BLOCK)
                    v = tl.load(Logits + r * STRIDE + i, i < n, other=0.0)
                    valid = (i < n) & (_cooperative_topk_wide_bucket(v) == d)
                    key = _ordered_key(v)
                    lo = tl.minimum(lo, tl.min(tl.where(valid, key, 4294967295), 0))
                    hi = tl.maximum(hi, tl.max(tl.where(valid, key, 0), 0))
                exact = tl.full((), False, tl.int1)
                while (lo < hi) & ~exact:
                    mid = lo + (hi - lo >> 1)
                    cnt = tl.full((), 0, tl.int32)
                    for start in range(0, n, BLOCK):
                        i = start + tl.arange(0, BLOCK)
                        v = tl.load(Logits + r * STRIDE + i, i < n, other=0.0)
                        valid = (i < n) & (_cooperative_topk_wide_bucket(v) == d)
                        key = _ordered_key(v)
                        cnt += tl.sum((valid & (key > mid)).to(tl.int32))
                    exact = cnt == remain
                    lo = tl.where(cnt >= remain, mid + 1, lo)
                    hi = tl.where(cnt < remain, mid, hi)
                ng = tl.full((), 0, tl.int32)
                for start in range(0, n, BLOCK):
                    i = start + tl.arange(0, BLOCK)
                    v = tl.load(Logits + r * STRIDE + i, i < n, other=0.0)
                    valid = (i < n) & (_cooperative_topk_wide_bucket(v) == d)
                    key = _ordered_key(v)
                    ng += tl.sum((valid & (key > lo)).to(tl.int32))
                pos = na
                ties = tl.full((), 0, tl.int32)
                for start in range(0, n, BLOCK):
                    i = start + tl.arange(0, BLOCK)
                    v = tl.load(Logits + r * STRIDE + i, i < n, other=0.0)
                    valid = (i < n) & (_cooperative_topk_wide_bucket(v) == d)
                    key = _ordered_key(v)
                    gt = valid & (key > lo)
                    eq = valid & (key == lo)
                    er = tl.cumsum(eq.to(tl.int32))
                    sel = tl.where(
                        exact, valid & (key >= lo), gt | eq & (ties + er <= remain - ng)
                    )
                    rank = tl.cumsum(sel.to(tl.int32))
                    tl.store(Output + r * K + pos + rank - 1, i, sel)
                    ties += tl.sum(eq.to(tl.int32))
                    pos += tl.sum(sel.to(tl.int32))

    def _cooperative_topk_wide(logits, lengths, output, workspace, k, max_seq_len):
        rows, width = logits.shape
        sparse = width >= 262144
        parts = 32 if sparse else 64
        capacity = 256
        state_stride = 4 * parts + 3 if sparse else 2 * parts + 3
        histogram = torch.empty(
            (rows, parts, 256), device=logits.device, dtype=torch.int32
        )
        state = torch.empty(
            (rows, state_stride), device=logits.device, dtype=torch.int32
        )
        candidates = torch.empty(
            (rows, capacity), device=logits.device, dtype=torch.uint32
        )
        indices = torch.empty((rows, capacity), device=logits.device, dtype=torch.int32)
        options = {"llc_options": "-mtgpu-opt-level=1"}
        _cooperative_topk_wide_histogram[parts, rows](
            logits, lengths, histogram, width, logits.stride(0), parts, rows
        )
        if sparse:
            _cooperative_topk_wide_threshold_sparse[rows,](
                histogram, state, k, parts, num_warps=16, **options
            )
            block = 4096
            _cooperative_topk_wide_compact_sparse[
                parts, triton.cdiv(triton.cdiv(width, parts), block), rows
            ](
                logits,
                lengths,
                output,
                state,
                candidates,
                indices,
                width,
                logits.stride(0),
                k,
                parts,
                block,
                capacity,
                num_warps=8,
                **options,
            )
        else:
            _cooperative_topk_wide_threshold[rows,](
                histogram, state, k, parts, num_warps=8, **options
            )
            block = 512
            _cooperative_topk_wide_compact[triton.cdiv(width, block), rows](
                logits,
                lengths,
                output,
                state,
                candidates,
                indices,
                width,
                logits.stride(0),
                k,
                parts,
                block,
                capacity,
                num_warps=4,
                **options,
            )
        _cooperative_topk_wide_finish[triton.cdiv(capacity, 8), rows](
            logits,
            lengths,
            output,
            state,
            candidates,
            indices,
            width,
            logits.stride(0),
            k,
            parts,
            rows,
            capacity,
            1024,
            STATE_STRIDE=state_stride,
        )

else:
    _cooperative_topk_compact = None
    _cooperative_topk_wide = None


def cooperative_topk(logits, lengths, output, workspace, k, max_seq_len):
    return _cooperative_topk_impl(
        logits,
        lengths,
        output,
        workspace,
        k,
        max_seq_len,
        small_kernel=_cooperative_topk_compact,
        small_max_width=32768 if HAS_TLE else 16384,
        small_max_rows=64 if HAS_TLE else 16,
        wide_kernel=_cooperative_topk_wide,
        wide_max_width=1048576,
    )
