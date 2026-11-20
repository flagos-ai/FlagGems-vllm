# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl
from triton.experimental.tle import language as tle

from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_native import (
    lma_wait,
    v32_copy_page,
    vector_exp2,
)
from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_scheduling import (
    mixed_v32_span,
)


@triton.jit
def observed_query_bounds(q_lower, q_upper, q_rope, query_scale, SCALE: tl.constexpr):
    query_valid = (query_scale >= 2.0**-20) & (query_scale <= 2.0**-6)
    q_scale = tl.where(query_valid, query_scale, 0.0) * SCALE
    q_bound = tl.max(
        tl.sum(tl.abs(q_lower.to(tl.float32)) + tl.abs(q_upper.to(tl.float32)), 1)
        * q_scale,
        0,
    )
    qr_bound = tl.max(tl.sum(tl.abs(q_rope.to(tl.float32)), 1) * q_scale, 0)
    return q_scale, query_valid, q_bound, qr_bound


@triton.jit
def observed_sentinel_ok(
    KV,
    Indices,
    row,
    start,
    end,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # Zero probability cannot mask a nonfinite placeholder value in PV.
    sentinel = tl.load(KV + tl.arange(0, 512)).to(tl.float32)
    sentinel_ok = tl.min((tl.abs(sentinel) <= 448.0).to(tl.int32), 0) != 0
    if not sentinel_ok:
        offsets = start + tl.arange(0, triton.next_power_of_2(CHUNK))
        within_split = offsets < end
        idx = tl.load(Indices + row * TOPK + offsets, within_split, 0)
        selected = (idx >= 0) & (idx < TOKENS)
        sentinel_ok = tl.min((~within_split | selected).to(tl.int32), 0) != 0
    return sentinel_ok


@triton.jit
def observed_score_error(q_bound, qr_bound, k_max, rope_max):
    # Include FP32 accumulation and scaling in the logit error bound.
    gamma: tl.constexpr = (580.0 * 2.0**-24) / (1.0 - 580.0 * 2.0**-24)
    return gamma * (448.0 * q_bound * tl.max(k_max, 0) + qr_bound * rope_max)


@triton.jit
def preloaded_active(
    Skip, Cutoff, Again, split_index, OBSERVED: tl.constexpr, MODE: tl.constexpr
):
    if MODE == 1:
        return tl.load(Cutoff + split_index) > 0.0
    elif MODE == 2:
        return tl.load(Again + split_index)
    else:
        return True if OBSERVED else not tl.load(Skip + split_index)


@triton.jit
def observed_eligible(
    sentinel_ok, query_valid, q_bound, qr_bound, k_max, rope_max, weighted, total
):
    score_error = observed_score_error(q_bound, qr_bound, k_max, rope_max)
    energy = tl.sum(weighted, 1) * (1.00001 / 16384.0)
    return (
        sentinel_ok
        & (tl.sum((~query_valid).to(tl.int32), 0) == 0)
        & (1.001 * score_error <= 8.0e-5)
        & (tl.max(energy - 0.00105 * total, 0) <= 0.0)
    )


@triton.jit
def produce(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    LSE,
    Skip,
    QL,
    QU,
    QRope,
    KL,
    KU,
    KRope,
    PH,
    PL,
    Inv,
    Full,
    Empty,
    UpperEmpty,
    Local,
    Norm,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    OBSERVED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(2)
    split_index = row * SPLITS + split
    run = True if OBSERVED else not tl.load(Skip + split_index)
    if run:
        start = 0 if SPLITS == 1 else split * CHUNK
        end = tl.minimum(start + CHUNK, TOPK)
        pages = (
            tl.cdiv(CHUNK, BN)
            if SPLITS * CHUNK == TOPK
            else tl.cdiv(tl.maximum(end - start, 0), BN)
        )
        heads = tl.arange(0, 64)
        dims = tl.arange(0, 256)
        rope_dims = tl.arange(0, 64)
        cols = tl.arange(0, BN)
        qr_offset = (row * 64 + heads[:, None]) * 64 + rope_dims[None, :]
        q_offset = (row * 64 + heads[:, None]) * 512 + dims[None, :]
        if OBSERVED:
            q_lower = tl.load(Q + q_offset)
            q_upper = tl.load(Q + q_offset + 256)
            q_rope = tl.load(QR + qr_offset)
            tl.store(tle.gpu.local_ptr(QL), q_lower)
            tl.store(tle.gpu.local_ptr(QU), q_upper)
            tl.store(tle.gpu.local_ptr(QRope), q_rope)
            query_scale = tl.load(QS + row * 64 + heads)
            q_scale, query_valid, q_bound, qr_bound = observed_query_bounds(
                q_lower, q_upper, q_rope, query_scale, SCALE
            )
            sentinel_ok = observed_sentinel_ok(
                KV, Indices, row, start, end, TOKENS, TOPK, CHUNK
            )
            k_max = tl.full((BN,), 0.0, tl.float32)
            kr_max = tl.full((BN, 64), 0.0, tl.float32)
            p_sum = tl.full((64, BN), 0.0, tl.float32)
            weighted = tl.full((64, BN), 0.0, tl.float32)
        else:
            tle.gpu.copy(Q + q_offset, QL, (64, 256))
            tle.gpu.copy(Q + q_offset + 256, QU, (64, 256))
            tle.gpu.copy(QR + qr_offset, QRope, (64, 64))
            q_scale = tl.load(QS + row * 64 + heads) * SCALE
            # A separate check has already bounded the scores for this path.
            q_scale *= 1.4426950408889634
            p_sum = tl.full((64, BN), 0.0, tl.float32)
        for page in range(pages):
            slot = page & 1
            phase = (page // 2) & 1
            # The consumer owns this slot until both PV halves finish.
            tle.gpu.barrier_wait(Empty[slot], phaseIdx=phase)
            offsets = start + page * BN + cols
            index_ptr = Indices + row * TOPK + offsets
            if SPLITS * CHUNK == TOPK and CHUNK % BN == 0:
                idx = tl.load(index_ptr)
            else:
                idx = tl.load(index_ptr, offsets < end, -1)
            valid = (idx >= 0) & (idx < TOKENS)
            safe = tl.where(valid, idx, 0).to(tl.int64)
            kv = KV + safe[:, None] * 512 + dims[None, :]
            rope = KR + safe[:, None] * 64 + rope_dims[None, :]
            tle.gpu.copy(kv, KL.slot(slot), (BN, 256))
            if OBSERVED:
                k_scale = tl.where(valid, tl.load(KS + safe), 0.0)
                scale_valid = (k_scale >= 2.0**-20) & (k_scale <= 2.0**-9)
                k_scale = tl.where(scale_valid, k_scale, 0.0)
                k_max = tl.maximum(k_max, k_scale)
                key_rope = tl.load(rope, valid[:, None], 0.0)
                kr_max = tl.maximum(
                    kr_max, tl.abs(key_rope.to(tl.float32)) * k_scale[:, None]
                )
                tl.store(tle.gpu.local_ptr(KRope.slot(slot)), key_rope)
            else:
                tle.gpu.copy(rope, KRope, (BN, 64))
                k_scale = tl.where(valid, tl.load(KS + safe), 0.0)
            if OBSERVED:
                tle.gpu.barrier_wait(UpperEmpty, phaseIdx=slot)
                tle.gpu.copy(kv + 256, KU, (BN, 256))
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(Local, phaseIdx=slot)
                tle.gpu.barrier_wait(Local, phaseIdx=slot)
                scores = tle.gpu.wgmma(QL, KL.slot(slot), trans_b=True)
                scores = tle.gpu.wgmma(
                    QRope, KRope.slot(slot), acc=scores, trans_b=True
                )
                scores = tle.gpu.wgmma(QU, KU, trans_b=True, acc=scores)
                scores = tle.gpu.wgmma_wait(0, scores)
            else:
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(Local, phaseIdx=0)
                tle.gpu.barrier_wait(Local, phaseIdx=0)
                scores = tle.gpu.wgmma(QL, KL.slot(slot), trans_b=True)
                scores = tle.gpu.wgmma(QRope, KRope, acc=scores, trans_b=True)
                scores = tle.gpu.wgmma_wait(0, scores)
                # The other role may still be using the previous upper half.
                tle.gpu.barrier_wait(UpperEmpty, phaseIdx=slot)
                tle.gpu.copy(kv + 256, KU, (BN, 256))
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(Local, phaseIdx=1)
                tle.gpu.barrier_wait(Local, phaseIdx=1)
                scores = tle.gpu.wgmma(QU, KU, trans_b=True, acc=scores)
                scores = tle.gpu.wgmma_wait(0, scores)
            scores *= q_scale[:, None] * k_scale[None, :]
            if OBSERVED:
                logit_valid = tl.abs(scores) <= 0.625
                accepted = logit_valid & scale_valid[None, :]
            else:
                accepted = True
            exponent = tl.where(valid[None, :] & accepted, scores, float("-inf"))
            if OBSERVED:
                exponent *= 1.4426950408889634
            p = vector_exp2(exponent, 64, BN)
            p_sum += p
            scaled = p * (k_scale[None, :] * 16384.0)
            if OBSERVED:
                weighted += tl.where(valid[None, :] & ~accepted, float("inf"), scaled)
            high = scaled.to(tl.float8e4nv)
            low = (scaled - high.to(tl.float32)).to(tl.float8e4nv)
            tl.store(tle.gpu.local_ptr(PH), high)
            tl.store(tle.gpu.local_ptr(PL), low)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Full[slot], phaseIdx=phase)
        total = tl.sum(p_sum, 1)
        if OBSERVED:
            eligible = observed_eligible(
                sentinel_ok,
                query_valid,
                q_bound,
                qr_bound,
                k_max,
                tl.max(tl.max(kr_max, 0), 0),
                weighted,
                total,
            )
            tl.store(Skip + split_index, ~eligible)
        if SPLITS == 1:
            inverse = tl.div_rn(1.0 / 16384.0, tl.where(total > 0.0, total, 1.0))
            tl.store(
                LSE + row * 64 + heads,
                tl.where(total > 0.0, tl.log(total), float("inf")),
            )
            tl.store(tle.gpu.local_ptr(Inv), inverse)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Norm, phaseIdx=0)
        else:
            # The caller passes its split stats in place of LSE.
            stats = (split_index * 64 + heads) * 2
            tl.store(LSE + stats, tl.where(total > 0.0, 0.0, float("-inf")))
            tl.store(LSE + stats + 1, total)


@triton.jit
def pipeline_plan(
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BN: tl.constexpr,
    FULL_ROWS: tl.constexpr,
):
    if FULL_ROWS:
        # Long programs occupy the first wave. The short programs fill the tail.
        slot = tl.program_id(0).to(tl.int64)
        row, full, start, end = mixed_v32_span(slot, TOPK, SPLITS, CHUNK, FULL_ROWS)
        pages = tl.where(full, TOPK // BN, CHUNK // BN)
    else:
        row = tl.program_id(0).to(tl.int64)
        split = tl.program_id(2)
        slot = row * SPLITS + split
        full = SPLITS == 1
        start = 0 if SPLITS == 1 else split * CHUNK
        end = tl.minimum(start + CHUNK, TOPK)
        pages = (
            tl.cdiv(CHUNK, BN)
            if SPLITS * CHUNK == TOPK
            else tl.cdiv(tl.maximum(end - start, 0), BN)
        )
    return slot, row, full, start, end, pages


@triton.jit
def qualify_v32_scales(
    KV,
    KS,
    Indices,
    Cutoff,
    Skip,
    Again,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    FULL_ROWS: tl.constexpr,
):
    if FULL_ROWS:
        slot, row, _, start, end, _ = pipeline_plan(TOPK, SPLITS, CHUNK, 128, FULL_ROWS)
    else:
        row = tl.program_id(0).to(tl.int64)
        split = tl.program_id(1)
        slot = row * SPLITS + split
        start = split * CHUNK
        end = tl.minimum(start + CHUNK, TOPK)
    span: tl.constexpr = TOPK if FULL_ROWS else CHUNK
    pos = start + tl.arange(0, triton.next_power_of_2(span))
    idx = tl.load(Indices + row * TOPK + pos, pos < end, -1)
    valid = (idx >= 0) & (idx < TOKENS)
    safe = tl.where(valid, idx, 0)
    scale = tl.load(KS + safe, valid, 0.0)
    allowed = (~valid) | ((scale >= 2.0**-20) & (scale <= 2.0**-9))
    scales_ok = tl.min(allowed.to(tl.int32), 0) != 0
    dev = tl.where(valid, scale - 0.0010499, 0.0)
    excess = tl.sum(tl.maximum(dev, 0.0), 0)
    slack = tl.sum(tl.maximum(-dev, 0.0), 0)
    # A cap c gives a weighted-scale certificate when exp(2c) * excess <= slack.
    # Keep room for exponent and reduction rounding as well as the final PV margin.
    cutoff = tl.where(
        excess * (1.00001 * 3.4903429574618414) <= slack,
        0.625,
        tl.where(
            excess * (1.00001 * 2.718281828459045) <= slack,
            0.5,
            tl.where(excess * (1.00001 * 1.6487212707001282) <= slack, 0.25, 0.0),
        ),
    )
    safe_placeholder = observed_sentinel_ok(
        KV, Indices, row, start, end, TOKENS, TOPK, span
    )
    cutoff = tl.where(scales_ok & safe_placeholder, cutoff, 0.0)
    tl.store(Cutoff + slot, cutoff)
    tl.store(Skip + slot, cutoff == 0.0)
    tl.store(Again + slot, cutoff == 0.0)


@triton.jit
def consume(
    Y,
    TailY,
    Skip,
    KL,
    KU,
    PH,
    PL,
    Inv,
    Full,
    Empty,
    UpperEmpty,
    Norm,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BN: tl.constexpr,
    OBSERVED: tl.constexpr,
    Cutoff,
    Again,
    SCALE_MODE: tl.constexpr,
    FULL_ROWS: tl.constexpr,
    STATIC_SPLITS: tl.constexpr,
    HighP=None,
    PEEL_LAST: tl.constexpr = False,
):
    # Both mixed schedules get constant page counts. Only one runs per CTA.
    for part in tl.static_range(2 if STATIC_SPLITS else 1):
        if not STATIC_SPLITS or (tl.program_id(0) < FULL_ROWS) == (part == 0):
            if STATIC_SPLITS:
                split_index = tl.program_id(0).to(tl.int64)
                full_row: tl.constexpr = part == 0
                if full_row:
                    row = split_index
                    start: tl.constexpr = 0
                    end: tl.constexpr = TOPK
                    pages: tl.constexpr = TOPK // BN
                else:
                    tail_slot = split_index - FULL_ROWS
                    row = FULL_ROWS + tail_slot // SPLITS
                    start = (tail_slot % SPLITS) * CHUNK
                    end = start + CHUNK
                    pages: tl.constexpr = CHUNK // BN
            else:
                split_index, row, full_row, start, end, pages = pipeline_plan(
                    TOPK, SPLITS, CHUNK, BN, FULL_ROWS
                )
            if preloaded_active(Skip, Cutoff, Again, split_index, OBSERVED, SCALE_MODE):
                heads = tl.arange(0, 64)
                dims = tl.arange(0, 256)
                y_l = tl.full((64, 256), 0.0, tl.float32)
                y_u = tl.full((64, 256), 0.0, tl.float32)
                for is_last in tl.static_range(2 if PEEL_LAST else 1):
                    first = tl.maximum(pages - 1, 0) if is_last else 0
                    stop = (
                        pages if is_last or not PEEL_LAST else tl.maximum(pages - 1, 0)
                    )
                    for page in range(first, stop):
                        slot = page & 1
                        phase = (page // 2) & 1
                        lower = KL.slot(slot)
                        if HighP is not None:
                            tle.gpu.barrier_wait(HighP[slot], phaseIdx=phase)
                            y_l = tle.gpu.wgmma(PH, lower, acc=y_l)
                            y_u = tle.gpu.wgmma(PH, KU, acc=y_u)
                            y_l = tle.gpu.wgmma_wait(0, y_l)
                            y_u = tle.gpu.wgmma_wait(0, y_u)
                            tle.gpu.barrier_wait(Full[slot], phaseIdx=phase)
                            y_l = tle.gpu.wgmma(PL, lower, acc=y_l)
                            y_u = tle.gpu.wgmma(PL, KU, acc=y_u)
                        else:
                            tle.gpu.barrier_wait(Full[slot], phaseIdx=phase)
                            y_l = tle.gpu.wgmma(PH, lower, acc=y_l)
                            y_l = tle.gpu.wgmma(PL, lower, acc=y_l)
                            y_u = tle.gpu.wgmma(PH, KU, acc=y_u)
                            y_u = tle.gpu.wgmma(PL, KU, acc=y_u)
                        y_l = tle.gpu.wgmma_wait(0, y_l)
                        y_u = tle.gpu.wgmma_wait(0, y_u)
                        # The loader only needs the release signals when more pages follow.
                        if not PEEL_LAST or not is_last:
                            tle.gpu.barrier_arrive(UpperEmpty, phaseIdx=slot)
                            tle.gpu.barrier_arrive(Empty[slot], phaseIdx=phase)
                if full_row:
                    y_offset = (row * 64 + heads[:, None]) * 512 + dims[None, :]
                    tle.gpu.barrier_wait(Norm, phaseIdx=0)
                    inv = tl.load(tle.gpu.local_ptr(Inv), volatile=True)
                    tl.store(Y + y_offset, y_l * inv[:, None])
                    tl.store(Y + y_offset + 256, y_u * inv[:, None])
                else:
                    y_offset = (
                        (split_index - FULL_ROWS) * 64 + heads[:, None]
                    ) * 512 + dims[None, :]
                    partial = TailY if FULL_ROWS else Y
                    # Partial outputs use the same gain as the full normalization.
                    tl.store(partial + y_offset, y_l * (1.0 / 16384.0))
                    tl.store(partial + y_offset + 256, y_u * (1.0 / 16384.0))


@triton.jit
def scale_bound_terms(scale, valid, factor):
    limit: tl.constexpr = 0.00105
    scale_ok = (scale >= 2.0**-20) & (scale <= 2.0**-9)
    value = tl.where(valid, scale, limit)
    excess = tl.maximum(value - limit, 0.0)
    weighted = tl.fma(excess, factor - 1.0, value)
    return tl.where(~valid | scale_ok, weighted, float("inf"))


@triton.jit
def load_pages(
    KV,
    KR,
    KS,
    Indices,
    Skip,
    KL,
    KU,
    KRope,
    Cache,
    Present,
    RopeBound,
    Peak,
    Empty,
    UpperEmpty,
    LowReady,
    HighReady,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BN: tl.constexpr,
    OBSERVED: tl.constexpr,
    PEAKS: tl.constexpr,
    Cutoff,
    Again,
    SCALE_MODE: tl.constexpr,
    FULL_ROWS: tl.constexpr,
    STATIC_SPLITS: tl.constexpr,
    DIRECT_COPY: tl.constexpr,
    ScaleOK,
    ScaleReady,
    SMALL_SCALE_CHECK: tl.constexpr,
    LARGE_SCALE_CHECK: tl.constexpr,
    SINGLE_ROPE: tl.constexpr,
    RopeEmpty,
):
    PREFETCH_UPPER: tl.constexpr = DIRECT_COPY == 2 and (
        LARGE_SCALE_CHECK or SMALL_SCALE_CHECK
    )
    # Both mixed schedules get constant page counts. Only one runs per CTA.
    for part in tl.static_range(2 if STATIC_SPLITS else 1):
        if not STATIC_SPLITS or (tl.program_id(0) < FULL_ROWS) == (part == 0):
            if STATIC_SPLITS:
                split_index = tl.program_id(0).to(tl.int64)
                full_row: tl.constexpr = part == 0
                if full_row:
                    row = split_index
                    start: tl.constexpr = 0
                    end: tl.constexpr = TOPK
                    pages: tl.constexpr = TOPK // BN
                else:
                    tail_slot = split_index - FULL_ROWS
                    row = FULL_ROWS + tail_slot // SPLITS
                    start = (tail_slot % SPLITS) * CHUNK
                    end = start + CHUNK
                    pages: tl.constexpr = CHUNK // BN
            else:
                split_index, row, full_row, start, end, pages = pipeline_plan(
                    TOPK, SPLITS, CHUNK, BN, FULL_ROWS
                )
            if preloaded_active(Skip, Cutoff, Again, split_index, OBSERVED, SCALE_MODE):
                cols = tl.arange(0, BN)
                dims = tl.arange(0, 256)
                rd = tl.arange(0, 64)
                if SMALL_SCALE_CHECK:
                    scale_bound = tl.full((BN,), 0.0, tl.float32)
                if LARGE_SCALE_CHECK:
                    scale_sum = tl.full((BN,), 0.0, tl.float32)
                    excess_sum = tl.full((BN,), 0.0, tl.float32)
                for page in range(pages):
                    slot = page & 1
                    phase = (page // 2) & 1
                    rope_tile = KRope if SINGLE_ROPE else KRope.slot(slot)
                    tle.gpu.barrier_wait(Empty[slot], phaseIdx=phase)
                    pos = start + page * BN + cols
                    if FULL_ROWS or (
                        OBSERVED and SPLITS * CHUNK == TOPK and CHUNK % BN == 0
                    ):
                        idx = tl.load(Indices + row * TOPK + pos)
                    else:
                        idx = tl.load(Indices + row * TOPK + pos, pos < end, -1)
                    valid = (idx >= 0) & (idx < TOKENS)
                    safe = tl.where(valid, idx, 0).to(tl.int64)
                    if SMALL_SCALE_CHECK:
                        # Fetch the scale before scheduling the two bulk copies.
                        raw_scale = tl.load(KS + safe)
                    kv = KV + safe[:, None] * 512 + dims[None, :]
                    rope = KR + safe[:, None] * 64 + rd[None, :]
                    if DIRECT_COPY:
                        tl.static_assert(
                            BN == 128 and TOPK % BN == 0 and CHUNK % BN == 0
                        )
                        zero = tl.full((), 0, tl.int32)
                        low_addr = tle.gpu.local_ptr(KL.slot(slot), (zero, zero))
                        ids = (Indices + row * TOPK + start + page * BN).to(tl.uint64)
                        if DIRECT_COPY == 2:
                            rope_addr = (
                                tle.gpu.local_ptr(rope_tile, (zero, zero))
                                .to(tl.uint64)
                                .to(tl.uint32)
                            )
                            rope_ptr = KR.to(tl.uint64)
                        else:
                            rope_addr = tl.full((), 0, tl.uint32)
                            rope_ptr = tl.full((), 0, tl.uint64)
                        v32_copy_page(
                            low_addr.to(tl.uint64).to(tl.uint32),
                            KV.to(tl.uint64),
                            ids,
                            TOKENS,
                            rope_addr,
                            rope_ptr,
                        )
                    else:
                        tle.gpu.copy(kv, KL.slot(slot), (BN, 256))
                    if not SMALL_SCALE_CHECK:
                        raw_scale = tl.load(KS + safe)
                    if LARGE_SCALE_CHECK:
                        scale_ok = (raw_scale >= 2.0**-20) & (raw_scale <= 2.0**-9)
                        marked = tl.where(scale_ok, raw_scale, float("inf"))
                        ks = tl.where(valid, marked, -1.0)
                    else:
                        ks = tl.where(valid, raw_scale, 0.0)
                    if SMALL_SCALE_CHECK:
                        # Covers |logit| <= 0.15625 and the relative polynomial error.
                        scale_bound += scale_bound_terms(ks, valid, 1.3672)
                    tl.store(tle.gpu.local_ptr(Cache.slot(slot)), ks)
                    if OBSERVED and SCALE_MODE != 1:
                        tl.store(
                            tle.gpu.local_ptr(Present.slot(slot)), valid.to(tl.int8)
                        )
                    if DIRECT_COPY != 2:
                        if SINGLE_ROPE:
                            tle.gpu.barrier_wait(RopeEmpty, phaseIdx=slot)
                        tle.gpu.copy(rope, rope_tile, (BN, 64))
                    if PEAKS:
                        peak = tl.load(Peak + safe, valid, 0.0)
                        tl.store(tle.gpu.local_ptr(RopeBound.slot(slot)), peak)
                    if PREFETCH_UPPER:
                        upper_data = tl.load(
                            kv + 256, cache_modifier=".ca", volatile=True
                        )
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(LowReady[slot], phaseIdx=phase)
                    tle.gpu.barrier_wait(UpperEmpty, phaseIdx=slot)
                    if DIRECT_COPY == 1:
                        high_addr = tle.gpu.local_ptr(KU, (zero, zero))
                        v32_copy_page(
                            high_addr.to(tl.uint64).to(tl.uint32),
                            (KV + 256).to(tl.uint64),
                            ids,
                            TOKENS,
                            rope_addr,
                            rope_ptr,
                        )
                    elif PREFETCH_UPPER:
                        tl.store(tle.gpu.local_ptr(KU), upper_data)
                    else:
                        tle.gpu.copy(kv + 256, KU, (BN, 256))
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(HighReady, phaseIdx=slot)
                    if LARGE_SCALE_CHECK:
                        scale_terms = tl.where(valid, ks, 0.00105)
                        scale_sum += scale_terms
                        excess_sum += tl.maximum(scale_terms - 0.00105, 0.0)
                if SMALL_SCALE_CHECK:
                    allowed = tl.sum(scale_bound, 0) * 1.0002 <= pages * BN * 0.00105
                    addr = tle.gpu.local_ptr(ScaleOK, (tl.full((), 0, tl.int32),))
                    tl.store(addr, allowed.to(tl.int32))
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(ScaleReady, phaseIdx=0)
                if LARGE_SCALE_CHECK:
                    base_ptr = tle.gpu.local_ptr(ScaleOK, (tl.full((), 0, tl.int32),))
                    extra_ptr = tle.gpu.local_ptr(ScaleOK, (tl.full((), 1, tl.int32),))
                    tl.store(base_ptr, tl.sum(scale_sum, 0))
                    tl.store(extra_ptr, tl.sum(excess_sum, 0))
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(ScaleReady, phaseIdx=0)


@triton.jit
def fold_probability_pairs(values, BN: tl.constexpr):
    # The elements differing in token bit 3 are in the same SQMMA lane.
    view = tl.reshape(values, (64, BN // 32, 2, 1, 2, 8))
    lanes = tl.permute(view, (0, 1, 3, 5, 2, 4))
    low_group, high_group = tl.split(lanes)
    return tl.reshape(low_group + high_group, (64, BN // 2))


@triton.jit
def produce_preloaded(
    Q,
    QR,
    KV,
    QS,
    Indices,
    LSE,
    TailStats,
    Skip,
    QL,
    QU,
    QRope,
    KL,
    KU,
    KRope,
    PH,
    PL,
    Inv,
    Full,
    Local,
    Norm,
    Cache,
    Present,
    RopeBound,
    LowReady,
    HighReady,
    HighP,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    OBSERVED: tl.constexpr,
    PEAKS: tl.constexpr,
    Cutoff,
    Again,
    SCALE_MODE: tl.constexpr,
    FULL_ROWS: tl.constexpr,
    STATIC_SPLITS: tl.constexpr,
    CERTIFIED_POLY: tl.constexpr,
    ScaleOK,
    ScaleReady,
    UpperEmpty,
    Coeff,
    BadRows,
    BadCount,
    SMALL_SCALE_CHECK: tl.constexpr,
    LARGE_SCALE_CHECK: tl.constexpr,
    LOCAL_QUERY_NORM: tl.constexpr,
    TailDenominators,
    HAS_TAIL_DENOMINATORS: tl.constexpr,
    STATIC_SQ64: tl.constexpr,
    SINGLE_ROPE: tl.constexpr,
    RopeEmpty,
):
    # Both mixed schedules get constant page counts. Only one runs per CTA.
    for part in tl.static_range(2 if STATIC_SPLITS else 1):
        if not STATIC_SPLITS or (tl.program_id(0) < FULL_ROWS) == (part == 0):
            if STATIC_SPLITS:
                split_index = tl.program_id(0).to(tl.int64)
                full_row: tl.constexpr = part == 0
                if full_row:
                    row = split_index
                    start: tl.constexpr = 0
                    end: tl.constexpr = TOPK
                    pages: tl.constexpr = TOPK // BN
                else:
                    tail_slot = split_index - FULL_ROWS
                    row = FULL_ROWS + tail_slot // SPLITS
                    start = (tail_slot % SPLITS) * CHUNK
                    end = start + CHUNK
                    pages: tl.constexpr = CHUNK // BN
            else:
                split_index, row, full_row, start, end, pages = pipeline_plan(
                    TOPK, SPLITS, CHUNK, BN, FULL_ROWS
                )
            if preloaded_active(Skip, Cutoff, Again, split_index, OBSERVED, SCALE_MODE):
                SUM_COLS: tl.constexpr = BN // 2 if FULL_ROWS else BN
                DEFER_TAIL_SUM: tl.constexpr = HAS_TAIL_DENOMINATORS and part == 1
                if DEFER_TAIL_SUM:
                    tl.static_assert(SMALL_SCALE_CHECK and SUM_COLS == 64)

                DEFER_DENOMINATOR: tl.constexpr = SMALL_SCALE_CHECK or LARGE_SCALE_CHECK
                # Keep the original exponent for the short branch of the static split plan.
                observed_poly: tl.constexpr = (
                    TOPK <= 2048
                    and SCALE_MODE != 2
                    and (not STATIC_SPLITS or part == 0)
                )
                DECOUPLE_LOGIT_CHECK: tl.constexpr = SMALL_SCALE_CHECK and observed_poly
                logit_cap = tl.load(Cutoff + split_index) if SCALE_MODE == 1 else 0.625
                heads = tl.arange(0, 64)
                dims = tl.arange(0, 256)
                rope_dims = tl.arange(0, 64)
                qr_offset = (row * 64 + heads[:, None]) * 64 + rope_dims[None, :]
                q_offset = (row * 64 + heads[:, None]) * 512 + dims[None, :]
                if OBSERVED:
                    q_lower = tl.load(Q + q_offset)
                    q_upper = tl.load(Q + q_offset + 256)
                    q_rope = tl.load(QR + qr_offset)
                    tl.store(tle.gpu.local_ptr(QL), q_lower)
                    tl.store(tle.gpu.local_ptr(QU), q_upper)
                    tl.store(tle.gpu.local_ptr(QRope), q_rope)
                    query_scale = tl.load(QS + row * 64 + heads)
                    q_scale, query_valid, q_bound, qr_bound = observed_query_bounds(
                        q_lower, q_upper, q_rope, query_scale, SCALE
                    )
                    k_max = tl.full((BN,), 0.0, tl.float32)
                    kr_max = tl.full((BN,), 0.0, tl.float32)
                    p_sum = tl.full((64, SUM_COLS), 0.0, tl.float32)
                    if SCALE_MODE != 1 and not SMALL_SCALE_CHECK:
                        weighted = tl.full((64, SUM_COLS), 0.0, tl.float32)
                else:
                    query_scale = tl.load(QS + row * 64 + heads)
                    if LOCAL_QUERY_NORM:
                        k_max = tl.load(Coeff)
                        use_local_norm = k_max <= 1024.0
                        lower = tl.load(Q + q_offset)
                        upper = tl.load(Q + q_offset + 256)
                        rope = tl.load(QR + qr_offset)
                        tl.store(tle.gpu.local_ptr(QL), lower)
                        tl.store(tle.gpu.local_ptr(QU), upper)
                        tl.store(tle.gpu.local_ptr(QRope), rope)
                    else:
                        tle.gpu.copy(Q + q_offset, QL, (64, 256))
                        tle.gpu.copy(Q + q_offset + 256, QU, (64, 256))
                        tle.gpu.copy(QR + qr_offset, QRope, (64, 64))
                    q_scale = query_scale * SCALE
                    # A separate check has already bounded the scores for this path.
                    if not CERTIFIED_POLY:
                        q_scale *= 1.4426950408889634
                    p_sum = tl.full((64, SUM_COLS), 0.0, tl.float32)
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(Local, phaseIdx=0)
                tle.gpu.barrier_wait(Local, phaseIdx=0)
                if LARGE_SCALE_CHECK:
                    if LOCAL_QUERY_NORM:
                        norm_ok = tl.full((), True, tl.int1)
                        factor = tl.full((), 0.0, tl.float32)
                        if use_local_norm:
                            lower_fp = lower.to(tl.float32)
                            upper_fp = upper.to(tl.float32)
                            rope_fp = rope.to(tl.float32)
                            content_square = tl.sum(
                                tl.fma(lower_fp, lower_fp, upper_fp * upper_fp), 1
                            )
                            sq = (content_square + tl.sum(rope_fp * rope_fp, 1)) * (
                                query_scale * query_scale
                            )
                            valid_q = (
                                (sq >= 0.0)
                                & (query_scale >= 2.0**-20)
                                & (query_scale <= 2.0**-6)
                            )
                            q_max = tl.max(tl.where(valid_q, sq, float("inf")), 0)
                            cap = tl.sqrt(q_max) * tl.sqrt(k_max) * (1.001 * SCALE)
                            norm_ok = (cap <= 0.625) & (q_max <= 1024.0)
                            factor = tl.where(norm_ok, 1.00001 * tl.exp(2.0 * cap), 0.0)
                    else:
                        norm_ok = tl.full((), True, tl.int1)
                        factor = tl.load(Coeff + row)
                for page in range(pages):
                    slot = page & 1
                    phase = (page // 2) & 1
                    rope_tile = KRope if SINGLE_ROPE else KRope.slot(slot)
                    tle.gpu.barrier_wait(LowReady[slot], phaseIdx=phase)
                    ks = tl.load(tle.gpu.local_ptr(Cache.slot(slot)), volatile=True)
                    if LARGE_SCALE_CHECK:
                        scale_ok = (ks >= 2.0**-20) & (ks <= 2.0**-9)
                        ks = tl.where(scale_ok, ks, 0.0)
                    if OBSERVED:
                        if SCALE_MODE == 1:
                            valid = ks > 0.0
                            k_scale = ks
                        else:
                            valid = (
                                tl.load(
                                    tle.gpu.local_ptr(Present.slot(slot)), volatile=True
                                )
                                != 0
                            )
                            scale_valid = (ks >= 2.0**-20) & (ks <= 2.0**-9)
                            k_scale = tl.where(scale_valid, ks, 0.0)
                        k_max = tl.maximum(k_max, k_scale)
                    else:
                        k_scale = ks
                    scores = tle.gpu.wgmma(QL, KL.slot(slot), trans_b=True)
                    scores = tle.gpu.wgmma(QRope, rope_tile, acc=scores, trans_b=True)
                    if OBSERVED:
                        if PEAKS:
                            peak = tl.load(
                                tle.gpu.local_ptr(RopeBound.slot(slot)), volatile=True
                            )
                            if SCALE_MODE != 1:
                                peak = tl.where(scale_valid, peak, 0.0)
                        else:
                            key_rope = tl.load(
                                tle.gpu.local_ptr(rope_tile), volatile=True
                            )
                            if SCALE_MODE == 1:
                                # Unsigned magnitude bits preserve the order for finite FP16/BF16.
                                magnitude_bits = (
                                    key_rope.to(tl.uint16, bitcast=True) & 0x7FFF
                                )
                                peak_bits = tl.max(magnitude_bits, 1).to(tl.uint16)
                                magnitude = peak_bits.to(
                                    key_rope.dtype, bitcast=True
                                ).to(tl.float32)
                                magnitude = tl.where(
                                    magnitude < float("inf"), magnitude, float("inf")
                                )
                                peak = tl.where(valid, magnitude * k_scale, 0.0)
                            else:
                                magnitude = tl.where(
                                    valid[:, None], tl.abs(key_rope.to(tl.float32)), 0.0
                                )
                                peak = tl.max(magnitude, 1) * k_scale
                        kr_max = tl.maximum(kr_max, peak)
                    scores = tle.gpu.wgmma_wait(0, scores)
                    if SINGLE_ROPE:
                        tle.gpu.barrier_arrive(RopeEmpty, phaseIdx=slot)
                    tle.gpu.barrier_wait(HighReady, phaseIdx=slot)
                    scores = tle.gpu.wgmma(QU, KU, trans_b=True, acc=scores)
                    scores = tle.gpu.wgmma_wait(0, scores)
                    if OBSERVED:
                        scores *= q_scale[:, None] * k_scale[None, :]
                        if not DECOUPLE_LOGIT_CHECK:
                            # The observed path rejects logits outside the polynomial interval.
                            # The second pass keeps the original exponent and acceptance logic.
                            if observed_poly:
                                accepted = tl.abs(scores) <= tl.minimum(
                                    logit_cap, 0.1015625
                                )
                            else:
                                if SMALL_SCALE_CHECK:
                                    accepted = tl.abs(scores) <= tl.minimum(
                                        logit_cap, 0.15625
                                    )
                                else:
                                    accepted = tl.abs(scores) <= logit_cap
                            if SCALE_MODE != 1:
                                accepted &= scale_valid[None, :]
                        if observed_poly:
                            p = tl.fma(
                                scores,
                                tl.fma(0.5004297892252604, scores, 1.0012893676757812),
                                0.9999994458436655,
                            )
                            if DECOUPLE_LOGIT_CHECK:
                                p = tl.where(valid[None, :], p, 0.0)
                            else:
                                p = tl.where(valid[None, :] & accepted, p, 0.0)
                        else:
                            exponent = tl.where(
                                valid[None, :] & accepted, scores, float("-inf")
                            )
                            p = vector_exp2(exponent * 1.4426950408889634, 64, BN)
                    else:
                        # The norm certificate covers [-0.625, 0.625].
                        if CERTIFIED_POLY:
                            scores *= q_scale[:, None] * ks[None, :]
                            p = tl.fma(
                                scores,
                                tl.fma(
                                    scores,
                                    tl.fma(
                                        scores,
                                        tl.fma(
                                            scores,
                                            0.0424871223799997,
                                            0.17077562984900999,
                                        ),
                                        0.49987949041777147,
                                    ),
                                    0.9995974299900329,
                                ),
                                1.0000026187636524,
                            )
                            p = tl.where(ks[None, :] > 0.0, p, 0.0)
                        else:
                            exponent = tl.where(
                                ks[None, :] > 0.0,
                                scores * (q_scale[:, None] * ks[None, :]),
                                float("-inf"),
                            )
                            p = vector_exp2(exponent, 64, BN)
                    if not DEFER_DENOMINATOR:
                        if SCALE_MODE == 1:
                            p_terms = tl.where(
                                valid[None, :] & ~accepted, float("inf"), p
                            )
                        else:
                            p_terms = p
                        p_sum += (
                            fold_probability_pairs(p_terms, BN)
                            if FULL_ROWS
                            else p_terms
                        )
                    scaled = p * (k_scale[None, :] * 16384.0)
                    if OBSERVED and SCALE_MODE != 1 and not SMALL_SCALE_CHECK:
                        weighted_terms = tl.where(
                            valid[None, :] & ~accepted, float("inf"), scaled
                        )
                        weighted += (
                            fold_probability_pairs(weighted_terms, BN)
                            if FULL_ROWS
                            else weighted_terms
                        )
                    if DECOUPLE_LOGIT_CHECK:
                        # Rejected logits set an infinite denominator and replay.
                        accepted = scale_valid[None, :] & (
                            tl.abs(scores) <= tl.minimum(logit_cap, 0.1015625)
                        )
                    high = scaled.to(tl.float8e4nv)
                    tl.store(tle.gpu.local_ptr(PH), high)
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(HighP[slot], phaseIdx=phase)
                    low = (scaled - high.to(tl.float32)).to(tl.float8e4nv)
                    tl.store(tle.gpu.local_ptr(PL), low)
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(Full[slot], phaseIdx=phase)
                    # Keep these reductions off the probability publication path.
                    if DEFER_DENOMINATOR:
                        terms = p
                        if SMALL_SCALE_CHECK:
                            terms = tl.where(
                                valid[None, :] & ~accepted, float("inf"), p
                            )
                        p_sum += (
                            fold_probability_pairs(terms, BN) if FULL_ROWS else terms
                        )
                if not DEFER_TAIL_SUM:
                    total = tl.sum(p_sum, 1)
                if SMALL_SCALE_CHECK:
                    if full_row:
                        inverse = tl.div_rn(
                            1.0 / 16384.0, tl.where(total > 0.0, total, 1.0)
                        )
                        tl.store(tle.gpu.local_ptr(Inv), inverse)
                        lma_wait(tl.full((), 0, tl.int32))
                        tle.gpu.barrier_arrive(Norm, phaseIdx=0)
                if OBSERVED and SCALE_MODE != 1:
                    sentinel_ok = observed_sentinel_ok(
                        KV,
                        Indices,
                        row,
                        start,
                        end,
                        TOKENS,
                        TOPK,
                        TOPK if FULL_ROWS else CHUNK,
                    )
                if SMALL_SCALE_CHECK:
                    tle.gpu.barrier_wait(ScaleReady, phaseIdx=0)
                    addr = tle.gpu.local_ptr(ScaleOK, (tl.full((), 0, tl.int32),))
                    scale_ok = tl.load(addr, volatile=True) != 0
                    if DEFER_TAIL_SUM:
                        # The next kernel checks the complete probability sum.
                        denominator_ok = scale_ok
                    else:
                        p_max = tl.max(total, 0)
                        denominator_ok = (p_max < float("inf")) & (
                            scale_ok | (p_max == 0.0)
                        )
                    error = observed_score_error(
                        q_bound, qr_bound, k_max, tl.max(kr_max, 0)
                    )
                    eligible = (
                        sentinel_ok
                        & (tl.min(query_valid.to(tl.int32), 0) != 0)
                        & (1.001 * error <= 8.0e-5)
                        & denominator_ok
                    )
                    tl.store(Skip + split_index, ~eligible)
                elif OBSERVED:
                    if SCALE_MODE == 1:
                        error = observed_score_error(
                            q_bound, qr_bound, k_max, tl.max(kr_max, 0)
                        )
                        eligible = (
                            (tl.max(total, 0) < float("inf"))
                            & (tl.min(query_valid.to(tl.int32), 0) != 0)
                            & (1.001 * error <= 8.0e-5)
                        )
                        tl.store(Again + split_index, ~eligible)
                    else:
                        eligible = observed_eligible(
                            sentinel_ok,
                            query_valid,
                            q_bound,
                            qr_bound,
                            k_max,
                            tl.max(kr_max, 0),
                            weighted,
                            total,
                        )
                    tl.store(Skip + split_index, ~eligible)
                if full_row:
                    if not SMALL_SCALE_CHECK:
                        inverse = tl.div_rn(
                            1.0 / 16384.0, tl.where(total > 0.0, total, 1.0)
                        )
                        tl.store(tle.gpu.local_ptr(Inv), inverse)
                        lma_wait(tl.full((), 0, tl.int32))
                        tle.gpu.barrier_arrive(Norm, phaseIdx=0)
                    tl.store(
                        LSE + row * 64 + heads,
                        tl.where(total > 0.0, tl.log(total), float("inf")),
                    )
                else:
                    if DEFER_TAIL_SUM:
                        columns = tl.arange(0, 4)
                        row_base = (split_index - FULL_ROWS) * 64 + heads
                        offset = row_base[:, None] * 4 + columns[None, :]
                        group_sum = tl.sum(tl.reshape(p_sum, (64, 4, 16)), 2)
                        tl.store(TailDenominators + offset, group_sum)
                    else:
                        stats = (split_index - FULL_ROWS) * 128 + heads * 2
                        split_stats = TailStats if FULL_ROWS else LSE
                        tl.store(
                            split_stats + stats,
                            tl.where(total > 0.0, 0.0, float("-inf")),
                        )
                        tl.store(split_stats + stats + 1, total)
                if LARGE_SCALE_CHECK:
                    tle.gpu.barrier_wait(ScaleReady, phaseIdx=0)
                    base_ptr = tle.gpu.local_ptr(ScaleOK, (tl.full((), 0, tl.int32),))
                    extra_ptr = tle.gpu.local_ptr(ScaleOK, (tl.full((), 1, tl.int32),))
                    bias = tl.load(base_ptr, volatile=True)
                    excess = tl.load(extra_ptr, volatile=True)
                    weighted_scale = tl.fma(excess, factor - 1.0, bias)
                    if not norm_ok or (
                        factor > 0.0
                        and not (weighted_scale * 1.00021 <= TOPK * 0.00105)
                    ):
                        # PV also reads Skip on entry. Wait for its last release before changing it.
                        tle.gpu.barrier_wait(UpperEmpty, phaseIdx=pages & 1)
                        tl.store(Skip + split_index, True)
                        position = tl.atomic_add(BadCount, 1, sem="relaxed")
                        tl.store(BadRows + position, row)


@triton.jit
def sparse_fp8_v32_pipeline(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Y,
    LSE,
    Skip,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    WP: tl.constexpr,
    WC: tl.constexpr,
    OBSERVED: tl.constexpr,
    Peak=None,
    PEAKS: tl.constexpr = False,
    PRELOAD: tl.constexpr = False,
    Cutoff=None,
    Again=None,
    SCALE_MODE: tl.constexpr = 0,
    TailY=None,
    TailStats=None,
    FULL_ROWS: tl.constexpr = 0,
    STATIC_SPLITS: tl.constexpr = False,
    DIRECT_COPY: tl.constexpr = 0,
    CERTIFIED_POLY: tl.constexpr = False,
    STATIC_SQ64: tl.constexpr = False,
    Coeff=None,
    BadRows=None,
    BadCount=None,
    LOCAL_QUERY_NORM: tl.constexpr = False,
    TailDenominators=None,
    LOADER_SCALE_CHECK: tl.constexpr = False,
):
    SMALL_SCALE_CHECK: tl.constexpr = (
        (LOADER_SCALE_CHECK or (STATIC_SQ64 and STATIC_SPLITS))
        and OBSERVED
        and SCALE_MODE == 0
        and TOPK == 2048
    )
    LARGE_SCALE_CHECK: tl.constexpr = Coeff is not None
    SINGLE_ROPE: tl.constexpr = (
        STATIC_SQ64 and STATIC_SPLITS and PRELOAD and DIRECT_COPY == 0
    )
    HAS_TAIL_DENOMINATORS: tl.constexpr = TailDenominators is not None
    if HAS_TAIL_DENOMINATORS:
        tl.static_assert(SMALL_SCALE_CHECK and FULL_ROWS > 0)
    if LOCAL_QUERY_NORM:
        tl.static_assert(LARGE_SCALE_CHECK)
    if LARGE_SCALE_CHECK:
        tl.static_assert(PRELOAD and not OBSERVED and SPLITS == 1 and TOPK == 2048)
    dtype: tl.constexpr = QR.dtype.element_ty
    ql = tle.gpu.alloc((64, 256), tl.float8e4nv, nv_mma_shared_layout=True)
    qu = tle.gpu.alloc((64, 256), tl.float8e4nv, nv_mma_shared_layout=True)
    qr = tle.gpu.alloc((64, 64), dtype, nv_mma_shared_layout=True)
    kl = tle.gpu.alloc((2, BN, 256), tl.float8e4nv, nv_mma_shared_layout=True)
    ku = tle.gpu.alloc((BN, 256), tl.float8e4nv, nv_mma_shared_layout=True)
    rope_shape: tl.constexpr = (
        (2, BN, 64) if (OBSERVED or PRELOAD) and not SINGLE_ROPE else (BN, 64)
    )
    kr = tle.gpu.alloc(rope_shape, dtype, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
    inv = tle.gpu.alloc((64,), tl.float32, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barriers(2, arrive_count=WP)
    empty = tle.gpu.alloc_barriers(2, arrive_count=WC, init=tle.gpu.READY)
    upper_empty = tle.gpu.alloc_barrier(arrive_count=WC, init=tle.gpu.READY)
    local = tle.gpu.alloc_barrier(arrive_count=WP)
    norm = tle.gpu.alloc_barrier(arrive_count=WP)
    if PRELOAD:
        cache = tle.gpu.alloc((2, BN), tl.float32, nv_mma_shared_layout=False)
        if OBSERVED and SCALE_MODE != 1:
            present = tle.gpu.alloc((2, BN), tl.int8, nv_mma_shared_layout=False)
        else:
            present = cache
        if PEAKS:
            rope_bound = tle.gpu.alloc((2, BN), tl.float32, nv_mma_shared_layout=False)
        else:
            rope_bound = cache
        rope_empty = (
            tle.gpu.alloc_barrier(arrive_count=WP, init=tle.gpu.READY)
            if SINGLE_ROPE
            else local
        )
        low_ready = tle.gpu.alloc_barriers(2, arrive_count=4)
        high_ready = tle.gpu.alloc_barrier(arrive_count=4)
        high_p = tle.gpu.alloc_barriers(2, arrive_count=WP)
        if SMALL_SCALE_CHECK:
            scale_ok = tle.gpu.alloc((1,), tl.int32, nv_mma_shared_layout=False)
            scale_ready = tle.gpu.alloc_barrier(arrive_count=4)
        elif LARGE_SCALE_CHECK:
            scale_ok = tle.gpu.alloc((2,), tl.float32, nv_mma_shared_layout=False)
            scale_ready = tle.gpu.alloc_barrier(arrive_count=4)
        else:
            scale_ok = cache
            scale_ready = local
        tl.debug_barrier()
        tle.gpu.warp_specialize(
            [
                (
                    produce_preloaded,
                    (
                        Q,
                        QR,
                        KV,
                        QS,
                        Indices,
                        LSE,
                        TailStats if FULL_ROWS else LSE,
                        Skip,
                        ql,
                        qu,
                        qr,
                        kl,
                        ku,
                        kr,
                        ph,
                        pl,
                        inv,
                        full,
                        local,
                        norm,
                        cache,
                        present,
                        rope_bound,
                        low_ready,
                        high_ready,
                        high_p,
                        TOKENS,
                        TOPK,
                        SPLITS,
                        CHUNK,
                        SCALE,
                        BN,
                        OBSERVED,
                        PEAKS,
                        Cutoff if SCALE_MODE != 0 else Skip,
                        Again if SCALE_MODE != 0 else Skip,
                        SCALE_MODE,
                        FULL_ROWS,
                        STATIC_SPLITS,
                        CERTIFIED_POLY,
                        scale_ok,
                        scale_ready,
                        upper_empty,
                        Coeff if LARGE_SCALE_CHECK else Skip,
                        BadRows if LARGE_SCALE_CHECK else Skip,
                        BadCount if LARGE_SCALE_CHECK else Skip,
                        SMALL_SCALE_CHECK,
                        LARGE_SCALE_CHECK,
                        LOCAL_QUERY_NORM,
                        TailDenominators if HAS_TAIL_DENOMINATORS else Skip,
                        HAS_TAIL_DENOMINATORS,
                        STATIC_SQ64,
                        SINGLE_ROPE,
                        rope_empty,
                    ),
                ),
                (
                    consume,
                    (
                        Y,
                        TailY if FULL_ROWS else Y,
                        Skip,
                        kl,
                        ku,
                        ph,
                        pl,
                        inv,
                        full,
                        empty,
                        upper_empty,
                        norm,
                        TOPK,
                        SPLITS,
                        CHUNK,
                        BN,
                        OBSERVED,
                        Cutoff if SCALE_MODE != 0 else Skip,
                        Again if SCALE_MODE != 0 else Skip,
                        SCALE_MODE,
                        FULL_ROWS,
                        STATIC_SPLITS,
                        high_p,
                        STATIC_SQ64 and STATIC_SPLITS,
                    ),
                ),
                (
                    load_pages,
                    (
                        KV,
                        KR,
                        KS,
                        Indices,
                        Skip,
                        kl,
                        ku,
                        kr,
                        cache,
                        present,
                        rope_bound,
                        Peak,
                        empty,
                        upper_empty,
                        low_ready,
                        high_ready,
                        TOKENS,
                        TOPK,
                        SPLITS,
                        CHUNK,
                        BN,
                        OBSERVED,
                        PEAKS,
                        Cutoff if SCALE_MODE != 0 else Skip,
                        Again if SCALE_MODE != 0 else Skip,
                        SCALE_MODE,
                        FULL_ROWS,
                        STATIC_SPLITS,
                        DIRECT_COPY,
                        scale_ok,
                        scale_ready,
                        SMALL_SCALE_CHECK,
                        LARGE_SCALE_CHECK,
                        SINGLE_ROPE,
                        rope_empty,
                    ),
                ),
            ],
            worker_num_warps=[WC, 4],
            worker_num_regs=[128, 64],
        )
    else:
        tl.debug_barrier()
        tle.gpu.warp_specialize(
            [
                (
                    produce,
                    (
                        Q,
                        QR,
                        KV,
                        KR,
                        QS,
                        KS,
                        Indices,
                        LSE,
                        Skip,
                        ql,
                        qu,
                        qr,
                        kl,
                        ku,
                        kr,
                        ph,
                        pl,
                        inv,
                        full,
                        empty,
                        upper_empty,
                        local,
                        norm,
                        TOKENS,
                        TOPK,
                        SPLITS,
                        CHUNK,
                        SCALE,
                        BN,
                        OBSERVED,
                    ),
                ),
                (
                    consume,
                    (
                        Y,
                        Y,
                        Skip,
                        kl,
                        ku,
                        ph,
                        pl,
                        inv,
                        full,
                        empty,
                        upper_empty,
                        norm,
                        TOPK,
                        SPLITS,
                        CHUNK,
                        BN,
                        OBSERVED,
                        Skip,
                        Skip,
                        0,
                        0,
                        False,
                    ),
                ),
            ],
            worker_num_warps=[WC],
            worker_num_regs=[128],
        )
