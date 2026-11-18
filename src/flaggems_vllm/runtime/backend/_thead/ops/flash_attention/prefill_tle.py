# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl
from triton.experimental.tle.language import gpu

from flaggems_vllm.runtime.backend._thead.ops.flash_attention import common
from flaggems_vllm.utils import libentry


@triton.jit
def fragment_coordinates(tile, lane):
    rows = ()
    columns = ()
    for component in tl.static_range(8):
        rows += (tile * 16 + lane // 4 + component // 4 * 8,)
        columns += (lane % 4 + component % 4 * 4,)
    return rows, columns


@triton.jit
def zero_fragments(COUNT: tl.constexpr):
    fragments = ()
    for index in tl.static_range(COUNT):
        fragments += (tl.full((128,), 0.0, tl.float32),)
    return fragments


@triton.jit
def load_query(Q, qstart, qlen, tile, qhead, lane, QS: tl.constexpr, QH: tl.constexpr):
    rows, columns = fragment_coordinates(tile, lane)
    values = ()
    for column in tl.static_range(16):
        for component in tl.static_range(8):
            offset = 2 * (lane % 4) + component % 2 + component % 4 // 2 * 8
            values += (
                tl.load(
                    Q
                    + (qstart + rows[component]).to(tl.int64) * QS
                    + qhead.to(tl.int64) * QH
                    + column * 16
                    + offset,
                    rows[component] < qlen,
                    0.0,
                ),
            )
    packed = ()
    for column in tl.static_range(16):
        for component in tl.static_range(4):
            packed += (
                common.pack_bf(
                    values[column * 8 + component * 2],
                    values[column * 8 + component * 2 + 1],
                ),
            )
    return packed


@triton.jit
def qk_fragments(query, buffer, token, slot, DTYPE: tl.constexpr):
    scores0 = zero_fragments(8)
    scores1 = zero_fragments(8)
    for column in tl.static_range(16):
        keys = common.ld_fragment(buffer, column * 16, token, False, ROW=slot * 32)
        scores0 = common.mma_packed(
            *query[column * 4 : column * 4 + 4], *keys, *scores0, DTYPE=DTYPE
        )
        keys = common.ld_fragment(buffer, column * 16, token, False, ROW=slot * 32 + 16)
        scores1 = common.mma_packed(
            *query[column * 4 : column * 4 + 4], *keys, *scores1, DTYPE=DTYPE
        )
    return scores0, scores1


@triton.jit
def probability_fragments(
    scores,
    rows,
    columns,
    start,
    qlen,
    klen,
    delta,
    range_ok,
    word,
    sum0,
    sum1,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    DTYPE: tl.constexpr,
    CHECK_RANGE: tl.constexpr,
):
    probabilities = ()
    for component in tl.static_range(8):
        valid = (rows[component] < qlen) & (start + columns[component] < klen)
        if CAUSAL:
            valid &= start + columns[component] <= rows[component] + delta
        if DTYPE == tl.float16 and CHECK_RANGE:
            range_ok &= (~valid) | (tl.abs(scores[component] * SCALE) <= 8.0)
        probabilities += (
            tl.exp2(
                tl.where(
                    valid,
                    scores[component] * (SCALE * 1.4426950408889634)
                    - (0.0 if DTYPE == tl.float16 else 32.0) * 1.4426950408889634,
                    -float("inf"),
                )
            ),
        )
    sum0 += probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3]
    sum1 += probabilities[4] + probabilities[5] + probabilities[6] + probabilities[7]
    converted = ()
    for component in tl.static_range(8):
        converted += (probabilities[component].to(DTYPE),)
    reordered = common.mma_reorder(*converted, word)
    packed = ()
    for component in tl.static_range(4):
        packed += (
            common.pack_bf(reordered[component * 2], reordered[component * 2 + 1]),
        )
    return packed, sum0, sum1, range_ok


@triton.jit
def pv_fragments(
    probability0, probability1, buffer, token, slot, outputs, DTYPE: tl.constexpr
):
    updated = ()
    for column in tl.static_range(16):
        accumulator = outputs[column * 8 : column * 8 + 8]
        values = common.ld_fragment(buffer, column * 16, token, True, ROW=slot * 32)
        accumulator = common.mma_packed(
            *probability0, *values, *accumulator, DTYPE=DTYPE
        )
        values = common.ld_fragment(
            buffer, column * 16, token, True, ROW=slot * 32 + 16
        )
        accumulator = common.mma_packed(
            *probability1, *values, *accumulator, DTYPE=DTYPE
        )
        updated += accumulator
    return updated


@triton.jit
def store_output(
    Out,
    outputs,
    rows,
    columns,
    qstart,
    qlen,
    qhead,
    inv0,
    inv1,
    OS: tl.constexpr,
    OH: tl.constexpr,
):
    for column in tl.static_range(16):
        for component in tl.static_range(8):
            inverse = inv0 if component < 4 else inv1
            tl.store(
                Out
                + (qstart + rows[component]).to(tl.int64) * OS
                + qhead.to(tl.int64) * OH
                + column * 16
                + columns[component],
                outputs[column * 8 + component] * inverse,
                rows[component] < qlen,
            )


@libentry()
@triton.jit
def async_prefill_attention_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    CUQ,
    CUK,
    USEDK,
    TABLE,
    ALIBI,
    TQ: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    OS: tl.constexpr,
    OH: tl.constexpr,
    TS: tl.constexpr,
    AS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    SOFTCAP: tl.constexpr,
    PAGED: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    PACKED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    REVERSE: tl.constexpr = False,
    SHORT_LIMIT: tl.constexpr = 0,
    ORDER: tl.constexpr = 0,
    WRITE_LSE: tl.constexpr = True,
    BATCHES: tl.constexpr = 1,
    FOLD: tl.constexpr = 4,
    KNORM=None,
    WORKLIST=None,
    CAPACITY: tl.constexpr = 0,
    FIXED_MODE: tl.constexpr = False,
    UNROLL: tl.constexpr = 1,
):
    tl.static_assert(D == 256 and BM == 64 and (FOLD == 4) and PAGED)
    tl.static_assert(
        Q.dtype.element_ty == tl.bfloat16 or Q.dtype.element_ty == tl.float16
    )
    tl.static_assert(KP == PAGE * KS and VP == PAGE * VS)
    index = tl.program_id(0)
    group = tl.program_id(2)
    if index >= tl.load(WORKLIST + 5 * CAPACITY):
        return
    qstart = tl.load(WORKLIST + index)
    qlen = tl.load(WORKLIST + CAPACITY + index)
    klen = tl.load(WORKLIST + 2 * CAPACITY + index)
    batch = tl.load(WORKLIST + 3 * CAPACITY + index)
    tile = tl.load(WORKLIST + 4 * CAPACITY + index)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    kh = group // (HQ // HK // 4)
    if tl.load(KNORM + batch * HK + kh) == 0:
        return
    if REVERSE:
        tile = tl.cdiv(qlen, 16) - 1 - tile
    thread = tl.arange(0, 128)
    lane = thread % 32
    warp = thread // 32
    p = lane // 4
    q = lane % 4
    qhead = group * 4 + warp
    word = tl.where(
        p % 4 == q,
        tl.where(
            lane >= 16,
            1006632960 if Q.dtype.element_ty == tl.float16 else 1065353216,
            15360 if Q.dtype.element_ty == tl.float16 else 16256,
        ),
        0,
    ).to(tl.uint32)
    rows, columns = fragment_coordinates(tile, lane)
    query = load_query(Q, qstart, qlen, tile, qhead, lane, QS, QH)
    outputs = zero_fragments(128)
    range_ok = tl.full((128,), True, tl.int1)
    sum0 = tl.full((128,), 0.0, tl.float32)
    sum1 = tl.full((128,), 0.0, tl.float32)
    if Q.dtype.element_ty == tl.float16:
        range_ok = tl.full((128,), True, tl.int1)
    hi = klen
    delta = klen - qlen
    if CAUSAL:
        hi = tl.minimum(klen, tl.maximum(0, (tile + 1) * 16 + delta))
    base = batch.to(tl.int64) * (TS * PAGE)
    kbuf = gpu.alloc(
        (256, 64),
        K.dtype.element_ty,
        layout=common.cube_layout(False),
        nv_mma_shared_layout=False,
    )
    vbuf = gpu.alloc(
        (256, 64),
        V.dtype.element_ty,
        layout=common.cube_layout(False),
        nv_mma_shared_layout=False,
    )
    if hi > 0:
        common.issue_copy(K + base * KS, kbuf, 0, klen, KS, 0, kh * KH)
        common.issue_copy(V + base * VS, vbuf, 0, klen, VS, 0, kh * VH)
        common.commit_copies()
    token = common.wait_copies() + tl.zeros((128,), tl.int32)
    for start in tl.range(0, hi, 32, loop_unroll_factor=UNROLL):
        slot = start // 32 % 2
        if start + 32 < hi:
            common.issue_copy(
                K + base * KS, kbuf, start + 32, klen, KS, (1 - slot) * 32, kh * KH
            )
            common.issue_copy(
                V + base * VS, vbuf, start + 32, klen, VS, (1 - slot) * 32, kh * VH
            )
            common.commit_copies()
        scores0, scores1 = qk_fragments(query, kbuf, token, slot, Q.dtype.element_ty)
        probability0, sum0, sum1, range_ok = probability_fragments(
            scores0,
            rows,
            columns,
            start,
            qlen,
            klen,
            delta,
            range_ok,
            word,
            sum0,
            sum1,
            SCALE,
            CAUSAL,
            Q.dtype.element_ty,
            True,
        )
        probability1, sum0, sum1, range_ok = probability_fragments(
            scores1,
            rows,
            columns,
            start + 16,
            qlen,
            klen,
            delta,
            range_ok,
            word,
            sum0,
            sum1,
            SCALE,
            CAUSAL,
            Q.dtype.element_ty,
            True,
        )
        outputs = pv_fragments(
            probability0, probability1, vbuf, token, slot, outputs, Q.dtype.element_ty
        )
        token = common.wait_copies() + tl.zeros((128,), tl.int32)
    if Q.dtype.element_ty == tl.float16:
        if tl.sum(range_ok.to(tl.int32), 0) != 128:
            tl.atomic_and(KNORM + batch * HK + kh, 0, sem="relaxed")
    den0 = tl.sum(sum0.reshape((4, 8, 4)), 2)
    den0 = tl.broadcast_to(den0[:, :, None], (4, 8, 4)).reshape((128,))
    inv0 = tl.div_rn(1.0, tl.where(den0 > 0.0, den0, 1.0))
    den1 = tl.sum(sum1.reshape((4, 8, 4)), 2)
    den1 = tl.broadcast_to(den1[:, :, None], (4, 8, 4)).reshape((128,))
    inv1 = tl.div_rn(1.0, tl.where(den1 > 0.0, den1, 1.0))
    store_output(Out, outputs, rows, columns, qstart, qlen, qhead, inv0, inv1, OS, OH)
    if WRITE_LSE:
        lse0 = tl.where(
            den0 > 0.0,
            (0.0 if Q.dtype.element_ty == tl.float16 else 32.0)
            + tl.log2(den0) * 0.6931471805599453,
            float("inf"),
        )
        tl.store(
            LSE + qhead.to(tl.int64) * TQ + (qstart + rows[0]).to(tl.int64),
            lse0,
            (rows[0] < qlen) & (thread % 4 == 0),
        )
        lse1 = tl.where(
            den1 > 0.0,
            (0.0 if Q.dtype.element_ty == tl.float16 else 32.0)
            + tl.log2(den1) * 0.6931471805599453,
            float("inf"),
        )
        tl.store(
            LSE + qhead.to(tl.int64) * TQ + (qstart + rows[4]).to(tl.int64),
            lse1,
            (rows[4] < qlen) & (thread % 4 == 0),
        )


@libentry()
@triton.jit
def async_small_prefill_attention_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    CUQ,
    CUK,
    USEDK,
    TABLE,
    ALIBI,
    TQ: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    OS: tl.constexpr,
    OH: tl.constexpr,
    TS: tl.constexpr,
    AS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    SOFTCAP: tl.constexpr,
    PAGED: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    PACKED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    REVERSE: tl.constexpr = False,
    SHORT_LIMIT: tl.constexpr = 0,
    ORDER: tl.constexpr = 0,
    WRITE_LSE: tl.constexpr = True,
    BATCHES: tl.constexpr = 1,
    FOLD: tl.constexpr = 4,
    KNORM=None,
    WORKLIST=None,
    CAPACITY: tl.constexpr = 0,
    FIXED_MODE: tl.constexpr = False,
    UNROLL: tl.constexpr = 1,
    SCORE_PEAK: tl.constexpr = False,
    ROW_SUM_GUARD: tl.constexpr = False,
):
    tl.static_assert(D == 256 and BM == 64 and (FOLD == 4) and PAGED)
    tl.static_assert(
        Q.dtype.element_ty == tl.bfloat16 or Q.dtype.element_ty == tl.float16
    )
    tl.static_assert(QH >= D and QS >= (HQ - 1) * QH + D)
    tl.static_assert(KP == PAGE * KS and VP == PAGE * VS)
    index = tl.program_id(0)
    group = tl.program_id(2)
    if index >= tl.load(WORKLIST + 5 * CAPACITY):
        return
    qstart = tl.load(WORKLIST + index)
    qlen = tl.load(WORKLIST + CAPACITY + index)
    klen = tl.load(WORKLIST + 2 * CAPACITY + index)
    batch = tl.load(WORKLIST + 3 * CAPACITY + index)
    tile = tl.load(WORKLIST + 4 * CAPACITY + index)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    kh = group // (HQ // HK // 4)
    if tl.load(KNORM + batch * HK + kh) == 0:
        return
    if REVERSE:
        tile = tl.cdiv(qlen, 16) - 1 - tile
    thread = tl.arange(0, 128)
    lane = thread % 32
    warp = thread // 32
    p = lane // 4
    q = lane % 4
    qhead = group * 4 + warp
    word = tl.where(
        p % 4 == q,
        tl.where(
            lane >= 16,
            1006632960 if Q.dtype.element_ty == tl.float16 else 1065353216,
            15360 if Q.dtype.element_ty == tl.float16 else 16256,
        ),
        0,
    ).to(tl.uint32)
    rows, columns = fragment_coordinates(tile, lane)
    kbuf = gpu.alloc(
        (256, 64),
        K.dtype.element_ty,
        layout=common.cube_layout(False),
        nv_mma_shared_layout=False,
    )
    vbuf = gpu.alloc(
        (256, 64),
        V.dtype.element_ty,
        layout=common.cube_layout(False),
        nv_mma_shared_layout=False,
    )
    query = common.load_q(Q, kbuf, qstart, qlen, tile, group, QS, QH)
    outputs = zero_fragments(128)
    range_ok = tl.full((128,), True, tl.int1)
    sum0 = tl.full((128,), 0.0, tl.float32)
    sum1 = tl.full((128,), 0.0, tl.float32)
    if Q.dtype.element_ty == tl.float16 and not ROW_SUM_GUARD:
        if SCORE_PEAK:
            range_peak = tl.full((128,), 0.0, tl.float32)
        else:
            range_ok = tl.full((128,), True, tl.int1)
    hi = klen
    delta = klen - qlen
    if CAUSAL:
        hi = tl.minimum(klen, tl.maximum(0, (tile + 1) * 16 + delta))
    base = batch.to(tl.int64) * (TS * PAGE)
    if hi > 0:
        common.issue_copy(K + base * KS, kbuf, 0, klen, KS, 0, kh * KH)
        common.issue_copy(V + base * VS, vbuf, 0, klen, VS, 0, kh * VH)
        common.commit_copies()
    token = common.wait_copies() + tl.zeros((128,), tl.int32)
    for start in tl.range(0, hi, 32, loop_unroll_factor=UNROLL):
        slot = start // 32 % 2
        if start + 32 < hi:
            common.issue_copy(
                K + base * KS, kbuf, start + 32, klen, KS, (1 - slot) * 32, kh * KH
            )
            common.issue_copy(
                V + base * VS, vbuf, start + 32, klen, VS, (1 - slot) * 32, kh * VH
            )
            common.commit_copies()
        scores0, scores1 = qk_fragments(query, kbuf, token, slot, Q.dtype.element_ty)
        if Q.dtype.element_ty == tl.float16 and SCORE_PEAK and not ROW_SUM_GUARD:
            bound0 = tl.maximum(
                tl.maximum(tl.abs(scores0[0]), tl.abs(scores0[1])),
                tl.maximum(tl.abs(scores0[2]), tl.abs(scores0[3])),
            )
            bound1 = tl.maximum(
                tl.maximum(tl.abs(scores0[4]), tl.abs(scores0[5])),
                tl.maximum(tl.abs(scores0[6]), tl.abs(scores0[7])),
            )
            bound2 = tl.maximum(
                tl.maximum(tl.abs(scores1[0]), tl.abs(scores1[1])),
                tl.maximum(tl.abs(scores1[2]), tl.abs(scores1[3])),
            )
            bound3 = tl.maximum(
                tl.maximum(tl.abs(scores1[4]), tl.abs(scores1[5])),
                tl.maximum(tl.abs(scores1[6]), tl.abs(scores1[7])),
            )
            range_peak = tl.maximum(
                range_peak,
                tl.maximum(tl.maximum(bound0, bound1), tl.maximum(bound2, bound3)),
            )
        probability0, sum0, sum1, range_ok = probability_fragments(
            scores0,
            rows,
            columns,
            start,
            qlen,
            klen,
            delta,
            range_ok,
            word,
            sum0,
            sum1,
            SCALE,
            CAUSAL,
            Q.dtype.element_ty,
            not SCORE_PEAK and not ROW_SUM_GUARD,
        )
        probability1, sum0, sum1, range_ok = probability_fragments(
            scores1,
            rows,
            columns,
            start + 16,
            qlen,
            klen,
            delta,
            range_ok,
            word,
            sum0,
            sum1,
            SCALE,
            CAUSAL,
            Q.dtype.element_ty,
            not SCORE_PEAK and not ROW_SUM_GUARD,
        )
        outputs = pv_fragments(
            probability0, probability1, vbuf, token, slot, outputs, Q.dtype.element_ty
        )
        token = common.wait_copies() + tl.zeros((128,), tl.int32)
    if Q.dtype.element_ty == tl.float16 and not ROW_SUM_GUARD:
        if SCORE_PEAK:
            unsafe = not (tl.max(range_peak, 0) * tl.abs(SCALE) <= 8.0)
        else:
            unsafe = tl.sum(range_ok.to(tl.int32), 0) != 128
        if unsafe:
            tl.atomic_and(KNORM + batch * HK + kh, 0, sem="relaxed")
    den0 = tl.sum(sum0.reshape((4, 8, 4)), 2)
    den0 = tl.broadcast_to(den0[:, :, None], (4, 8, 4)).reshape((128,))
    if ROW_SUM_GUARD:
        inv0 = 1.0 / tl.where(den0 > 0.0, den0, 1.0)
    else:
        inv0 = tl.div_rn(1.0, tl.where(den0 > 0.0, den0, 1.0))
    den1 = tl.sum(sum1.reshape((4, 8, 4)), 2)
    den1 = tl.broadcast_to(den1[:, :, None], (4, 8, 4)).reshape((128,))
    if ROW_SUM_GUARD:
        inv1 = 1.0 / tl.where(den1 > 0.0, den1, 1.0)
    else:
        inv1 = tl.div_rn(1.0, tl.where(den1 > 0.0, den1, 1.0))
    if Q.dtype.element_ty == tl.float16 and ROW_SUM_GUARD:
        count0 = tl.full((128,), klen, tl.int32)
        count1 = tl.full((128,), klen, tl.int32)
        if CAUSAL:
            count0 = tl.minimum(klen, tl.maximum(0, rows[0] + delta + 1))
            count1 = tl.minimum(klen, tl.maximum(0, rows[4] + delta + 1))
        safe0 = (rows[0] >= qlen) | ((den0 <= 32768.0) & (den0 >= count0 * 0.0078125))
        safe1 = (rows[4] >= qlen) | ((den1 <= 32768.0) & (den1 >= count1 * 0.0078125))
        tl.atomic_and(
            KNORM + batch * HK + kh + tl.zeros((128,), tl.int32),
            tl.zeros((128,), tl.int32),
            mask=~(safe0 & safe1),
            sem="relaxed",
        )
    store_output(Out, outputs, rows, columns, qstart, qlen, qhead, inv0, inv1, OS, OH)
    if WRITE_LSE:
        lse0 = tl.where(
            den0 > 0.0,
            (0.0 if Q.dtype.element_ty == tl.float16 else 32.0)
            + tl.log2(den0) * 0.6931471805599453,
            float("inf"),
        )
        tl.store(
            LSE + qhead.to(tl.int64) * TQ + (qstart + rows[0]).to(tl.int64),
            lse0,
            (rows[0] < qlen) & (thread % 4 == 0),
        )
        lse1 = tl.where(
            den1 > 0.0,
            (0.0 if Q.dtype.element_ty == tl.float16 else 32.0)
            + tl.log2(den1) * 0.6931471805599453,
            float("inf"),
        )
        tl.store(
            LSE + qhead.to(tl.int64) * TQ + (qstart + rows[4]).to(tl.int64),
            lse1,
            (rows[4] < qlen) & (thread % 4 == 0),
        )
