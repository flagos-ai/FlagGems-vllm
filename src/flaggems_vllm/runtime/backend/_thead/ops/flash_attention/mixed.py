# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime.backend._thead.ops.flash_attention import decode
from flaggems_vllm.utils import libentry, libtuner


@triton.jit
def combine_norms(
    QPART,
    KPART,
    FLAG,
    CUQ,
    USED,
    META,
    QW: tl.constexpr,
    KW: tl.constexpr,
    QB: tl.constexpr,
    KB: tl.constexpr,
    SCALE: tl.constexpr,
    HK: tl.constexpr,
    BATCHES: tl.constexpr,
    BB: tl.constexpr,
    QM: tl.constexpr,
    QT: tl.constexpr,
    CAP: tl.constexpr,
    SHORT_LIMIT: tl.constexpr = 64,
):
    group = tl.program_id(0)
    batch = group // HK
    qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
    klen = tl.load(USED + batch)
    if qlen <= SHORT_LIMIT and klen >= 8192:
        tl.store(FLAG + group, 0)
    else:
        qcols = tl.arange(0, QB)
        kcols = tl.arange(0, KB)
        qn = tl.max(tl.load(QPART + group * QW + qcols, qcols < QW, 0), 0)
        kn = tl.max(tl.load(KPART + group * KW + kcols, kcols < KW, 0), 0)
        ok = qn * kn * (SCALE * SCALE) <= 1008.0625
        tl.store(FLAG + group, ok.to(tl.int8))
    batch = group // HK
    if group % HK == 0:
        bs = tl.arange(0, BB)
        qs = tl.load(CUQ + bs, bs < BATCHES, 0)
        qe = tl.load(CUQ + bs + 1, bs < BATCHES, 0)
        counts = tl.cdiv(qe - qs, QM)
        before = tl.sum(tl.where(bs < batch, counts, 0), 0)
        total = tl.sum(counts, 0)
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        klen = tl.load(USED + batch)
        ts = tl.arange(0, QT)
        positions = before + ts
        valid = ts < tl.cdiv(qlen, QM)
        tl.store(META + positions, qstart, valid)
        tl.store(META + CAP + positions, qlen, valid)
        tl.store(META + 2 * CAP + positions, klen, valid)
        tl.store(META + 3 * CAP + positions, batch, valid)
        tl.store(META + 4 * CAP + positions, ts, valid)
        if group == 0:
            tl.store(META + 5 * CAP, total)


@triton.jit
def stage_norm(
    Q,
    K,
    V,
    CUQ,
    TABLE,
    USED,
    PK,
    PV,
    PT,
    QPART,
    KPART,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    QW: tl.constexpr,
    KW: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    SHORT: tl.constexpr,
    NORM_SHORT: tl.constexpr,
    EARLY: tl.constexpr,
):
    tile, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    qb = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qb
    klen = tl.load(USED + batch)
    if qlen <= NORM_SHORT and klen >= 8192:
        return
    ds = tl.arange(0, 256)
    if tile < KW:
        rows = tile * PAGE + tl.arange(0, PAGE)
        page = tl.load(
            TABLE + batch.to(tl.int64) * TS + tile, tile * PAGE < klen, 0
        ).to(tl.int64)
        kval = tl.load(
            K
            + page * KP
            + (rows[:, None] % PAGE).to(tl.int64) * KS
            + head * KH
            + ds[None, :],
            rows[:, None] < klen,
            0,
        )
        kf = kval.to(tl.float32)
        squares = tl.sum(kf * kf, 1)
        kmax = tl.max(tl.where(squares == squares, squares, float("inf")), 0)
        tl.store(KPART + (batch * HK + head) * KW + tile, kmax)
        copy = not (qlen <= SHORT and klen >= 8192)
        if EARLY:
            copy = copy & (tile * PAGE < klen)
        if copy:
            vval = tl.load(
                V
                + page * VP
                + (rows[:, None] % PAGE).to(tl.int64) * VS
                + head * VH
                + ds[None, :],
                rows[:, None] < klen,
                0,
            )
            dest = (
                (batch.to(tl.int64) * KW * PAGE + rows[:, None].to(tl.int64)) * HK * 256
                + head * 256
                + ds[None, :]
            )
            tl.store(PK + dest, kval)
            tl.store(PV + dest, vval)
            if head == 0:
                tl.store(PT + batch * KW + tile, batch * KW + tile)
    if tile < QW:
        flat = tile * 32 + tl.arange(0, 32)
        group: tl.constexpr = HQ // HK
        qi = flat // group
        qh = head * group + flat % group
        q = tl.load(
            Q + (qb + qi[:, None]).to(tl.int64) * QS + qh[:, None] * QH + ds[None, :],
            qi[:, None] < qlen,
            0,
        ).to(tl.float32)
        squares = tl.sum(q * q, 1)
        qmax = tl.max(tl.where(squares == squares, squares, float("inf")), 0)
        tl.store(QPART + (batch * HK + head) * QW + tile, qmax)


def prepare(
    q, k, v, cuq, table, used, maxq, scale, qm, short=64, norm_short=64, early=True
):
    batch, hk = (used.numel(), k.shape[-2])
    kw = max(1, table.shape[1])
    page = k.shape[1]
    qw = max(1, triton.cdiv(maxq * (q.shape[1] // hk), 32))
    pk = torch.empty((batch * kw, page, hk, 256), device=k.device, dtype=k.dtype)
    pv = torch.empty(pk.shape, device=pk.device, dtype=pk.dtype)
    pt = torch.empty((batch, kw), device=table.device, dtype=table.dtype)
    qp = torch.empty((batch, hk, qw), device=q.device, dtype=torch.float32)
    kp = torch.empty((batch, hk, kw), device=k.device, dtype=torch.float32)
    flags = torch.empty((batch, hk), device=q.device, dtype=torch.int8)
    cap = triton.cdiv(q.shape[0], qm) + batch - 1
    meta = torch.empty((5 * cap + 1,), device=q.device, dtype=torch.int32)
    stage_norm[max(qw, kw), batch, hk](
        q,
        k,
        v,
        cuq,
        table,
        used,
        pk,
        pv,
        pt,
        qp,
        kp,
        q.shape[1],
        hk,
        qw,
        kw,
        *q.stride()[:2],
        *k.stride()[:3],
        *v.stride()[:3],
        table.stride(0),
        page,
        short,
        norm_short,
        early,
        num_warps=4,
        num_stages=1,
    )
    combine_norms[batch * hk,](
        qp,
        kp,
        flags,
        cuq,
        used,
        meta,
        qw,
        kw,
        triton.next_power_of_2(qw),
        triton.next_power_of_2(kw),
        scale,
        hk,
        batch,
        triton.next_power_of_2(batch),
        qm,
        triton.next_power_of_2(max(1, triton.cdiv(maxq, qm))),
        cap,
        num_warps=4,
        num_stages=1,
        SHORT_LIMIT=norm_short,
    )
    return (pk, pv, pt, flags, meta)


@triton.jit
def tp2_mixed_prefill_step(
    q,
    K,
    V,
    TABLE,
    acc,
    m,
    row_sum,
    start,
    batch,
    kh,
    qi,
    klen,
    delta,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    CAUSAL: tl.constexpr,
    FULL: tl.constexpr,
    FIXED: tl.constexpr,
    VORDER: tl.constexpr,
):
    ns = tl.arange(0, BN)
    ds = tl.arange(0, BD)
    cols = start + ns
    tl.static_assert(KP == PAGE * KS)
    tl.static_assert(VP == PAGE * VS)
    logical = batch.to(tl.int64) * (TS * PAGE) + cols.to(tl.int64)
    ko = logical * KS
    vo = logical * VS
    if KP % 8 == 0 and KS % 8 == 0:
        ko = tl.multiple_of(ko, 8)
    if VP % 8 == 0 and VS % 8 == 0:
        vo = tl.multiple_of(vo, 8)
    if VORDER == 2:
        if FULL:
            v = tl.load(V + vo[:, None] + kh * VH + ds[None, :])
        else:
            v = tl.load(
                V + vo[:, None] + kh * VH + ds[None, :], cols[:, None] < klen, 0
            )
    if FULL:
        k = tl.load(K + ko[None, :] + kh * KH + ds[:, None])
    else:
        k = tl.load(K + ko[None, :] + kh * KH + ds[:, None], cols[None, :] < klen, 0)
    scores = tl.dot(q, k) * (SCALE * 1.4426950408889634)
    if VORDER == 1:
        if FULL:
            v = tl.load(V + vo[:, None] + kh * VH + ds[None, :])
        else:
            v = tl.load(
                V + vo[:, None] + kh * VH + ds[None, :], cols[:, None] < klen, 0
            )
    if not FULL:
        mask = cols[None, :] < klen
        if CAUSAL:
            mask = mask & (cols[None, :] <= qi[:, None] + delta)
        scores = tl.where(mask, scores, -float("inf"))
    if FIXED:
        new_m = tl.full((BM,), 32.0 * 1.4426950408889634, tl.float32)
        p = tl.exp2(scores - 32.0 * 1.4426950408889634)
        row_sum += tl.sum(p.reshape((BM, BN // 16, 16)), 1)
    else:
        new_m = tl.maximum(m, tl.max(scores, 1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        alpha = tl.exp2(m - safe_m)
        p = tl.exp2(scores - safe_m[:, None])
        row_sum = row_sum * alpha[:, None] + tl.sum(p.reshape((BM, BN // 16, 16)), 1)
        acc *= alpha[:, None]
    if VORDER == 0:
        if FULL:
            v = tl.load(V + vo[:, None] + kh * VH + ds[None, :])
        else:
            v = tl.load(
                V + vo[:, None] + kh * VH + ds[None, :], cols[:, None] < klen, 0
            )
    acc = tl.dot(p.to(v.dtype), v, acc)
    return (acc, new_m, row_sum)


@triton.jit
def attention_kernel(
    Q,
    K,
    V,
    OUT,
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
    FIXED_MODE: tl.constexpr = -1,
    VORDER: tl.constexpr = 0,
):
    tl.static_assert(PAGED and (PAGE == 16 or PAGE == 32 or PAGE == 64) and (D == BD))
    tl.static_assert(
        not PACKED and (not HAS_ALIBI) and (SOFTCAP == 0) and (WL == -1) and (WR == -1)
    )
    tl.static_assert(HQ // HK % FOLD == 0)
    HE: tl.constexpr = HQ // FOLD
    GROUP: tl.constexpr = FOLD
    QM: tl.constexpr = BM // GROUP
    tile, head = (tl.program_id(0), tl.program_id(2))
    if ORDER == 1:
        linear = head * tl.num_programs(0) + tile
        head = linear % HE
        tile = linear // HE
    if tile >= tl.load(WORKLIST + 5 * CAPACITY):
        return
    qstart = tl.load(WORKLIST + tile)
    qlen = tl.load(WORKLIST + CAPACITY + tile)
    klen = tl.load(WORKLIST + 2 * CAPACITY + tile)
    batch = tl.load(WORKLIST + 3 * CAPACITY + tile)
    tile = tl.load(WORKLIST + 4 * CAPACITY + tile)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    if REVERSE:
        tile = tl.cdiv(qlen, QM) - 1 - tile
    kh = head // (HQ // HK // FOLD)
    fixed_ok = (tl.load(KNORM + batch * HK + kh) != 0) & (
        Q.dtype.element_ty == tl.bfloat16
    )
    if FIXED_MODE >= 0:
        if fixed_ok != (FIXED_MODE == 1):
            return
    qi = tile * QM + tl.arange(0, BM) % QM
    qh = head * GROUP + tl.arange(0, BM) // QM
    ds = tl.arange(0, BD)
    qrow = qstart + qi[:, None]
    if (TQ - 1) * QS + (HQ - 1) * QH + D >= 2147483648:
        qrow = qrow.to(tl.int64)
    q = tl.load(Q + qrow * QS + qh[:, None] * QH + ds[None, :], qi[:, None] < qlen, 0)
    delta = klen - qlen
    hi = klen
    full_hi = klen // BN * BN
    if CAUSAL:
        hi = tl.minimum(klen, (tile + 1) * QM + delta)
        full_hi = tl.minimum(klen, tile * QM + delta + 1) // BN * BN
    full_hi = tl.maximum(full_hi, 0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BM, 16), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    if FIXED_MODE == 1 or (FIXED_MODE < 0 and fixed_ok):
        for start in range(0, full_hi, BN):
            acc, m, row_sum = tp2_mixed_prefill_step(
                q,
                K,
                V,
                TABLE,
                acc,
                m,
                row_sum,
                start,
                batch,
                kh,
                qi,
                klen,
                delta,
                KP,
                KS,
                KH,
                VP,
                VS,
                VH,
                TS,
                PAGE,
                SCALE,
                BM,
                BN,
                BD,
                CAUSAL,
                True,
                True,
                VORDER,
            )
        for start in range(full_hi, hi, BN):
            acc, m, row_sum = tp2_mixed_prefill_step(
                q,
                K,
                V,
                TABLE,
                acc,
                m,
                row_sum,
                start,
                batch,
                kh,
                qi,
                klen,
                delta,
                KP,
                KS,
                KH,
                VP,
                VS,
                VH,
                TS,
                PAGE,
                SCALE,
                BM,
                BN,
                BD,
                CAUSAL,
                False,
                True,
                VORDER,
            )
    else:
        for start in range(0, full_hi, BN):
            acc, m, row_sum = tp2_mixed_prefill_step(
                q,
                K,
                V,
                TABLE,
                acc,
                m,
                row_sum,
                start,
                batch,
                kh,
                qi,
                klen,
                delta,
                KP,
                KS,
                KH,
                VP,
                VS,
                VH,
                TS,
                PAGE,
                SCALE,
                BM,
                BN,
                BD,
                CAUSAL,
                True,
                False,
                VORDER,
            )
        for start in range(full_hi, hi, BN):
            acc, m, row_sum = tp2_mixed_prefill_step(
                q,
                K,
                V,
                TABLE,
                acc,
                m,
                row_sum,
                start,
                batch,
                kh,
                qi,
                klen,
                delta,
                KP,
                KS,
                KH,
                VP,
                VS,
                VH,
                TS,
                PAGE,
                SCALE,
                BM,
                BN,
                BD,
                CAUSAL,
                False,
                False,
                VORDER,
            )
    row_sum = tl.sum(row_sum, 1)
    inv = tl.div_rn(1.0, tl.where(row_sum > 0, row_sum, 1.0))
    result = acc * inv[:, None]
    orow = qstart + qi[:, None]
    if (TQ - 1) * OS + (HQ - 1) * OH + D >= 2147483648:
        orow = orow.to(tl.int64)
    tl.store(
        OUT + orow * OS + qh[:, None] * OH + ds[None, :], result, qi[:, None] < qlen
    )
    if WRITE_LSE:
        lse = tl.where(
            row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
        )
        tl.store(LSE + qh * TQ + qstart + qi, lse, qi < qlen)


@triton.jit
def tp2_mixed_medium_metadata(
    CUQ,
    USED,
    PREFIX,
    B: tl.constexpr,
    BB: tl.constexpr,
    QM: tl.constexpr,
    LIMIT: tl.constexpr,
):
    bs = tl.arange(0, BB)
    qb = tl.load(CUQ + bs, bs < B, 0)
    qe = tl.load(CUQ + bs + 1, bs < B, 0)
    length = tl.load(USED + bs, bs < B, 0)
    qlen = qe - qb
    count = tl.where(
        (bs < B) & (qlen > 1) & (qlen <= LIMIT) & (length >= 8192), tl.cdiv(qlen, QM), 0
    )
    ends = tl.cumsum(count, 0)
    tl.store(PREFIX + bs, ends, bs < B)
    tl.store(PREFIX + B, tl.sum(count, 0))


@triton.jit
def _step(
    q0,
    q1,
    K,
    V,
    TABLE,
    a0,
    a1,
    maximum,
    denom,
    start,
    length,
    batch,
    qi,
    qlen,
    KP: tl.constexpr,
    KS: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    TS: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
    FULL: tl.constexpr,
    FIXED: tl.constexpr,
    EARLY_K: tl.constexpr,
    EARLY_V: tl.constexpr,
):
    ns = tl.arange(0, BN)
    ps = tl.arange(0, BN // 32)
    dd = tl.arange(0, 128)
    if FULL:
        pages = tl.load(TABLE + batch * TS + start // 32 + ps)
    else:
        pages = tl.load(
            TABLE + batch * TS + start // 32 + ps, start + ps * 32 < length, 0
        )
    within = tl.arange(0, 32)
    ko = tl.multiple_of(
        (pages[:, None].to(tl.int64) * KP + within[None, :] * KS).reshape((BN,)), 8
    )
    vo = tl.multiple_of(
        (pages[:, None].to(tl.int64) * VP + within[None, :] * VS).reshape((BN,)), 8
    )
    if FULL:
        k0 = tl.load(K + dd[:, None] + ko[None, :], cache_modifier=".cg")
    else:
        k0 = tl.load(
            K + dd[:, None] + ko[None, :],
            start + ns[None, :] < length,
            0,
            cache_modifier=".cg",
        )
    if EARLY_K:
        if FULL:
            k1 = tl.load(K + 128 + dd[:, None] + ko[None, :], cache_modifier=".cg")
        else:
            k1 = tl.load(
                K + 128 + dd[:, None] + ko[None, :],
                start + ns[None, :] < length,
                0,
                cache_modifier=".cg",
            )
    scores = tl.dot(q0, k0)
    if not EARLY_K:
        if FULL:
            k1 = tl.load(K + 128 + dd[:, None] + ko[None, :], cache_modifier=".cg")
        else:
            k1 = tl.load(
                K + 128 + dd[:, None] + ko[None, :],
                start + ns[None, :] < length,
                0,
                cache_modifier=".cg",
            )
    scores = tl.dot(q1, k1, scores) * (SCALE * 1.4426950408889634)
    if not FULL:
        valid = (
            (start + ns[None, :] < length)
            & (start + ns[None, :] <= qi[:, None] + length - qlen)
            & (qi[:, None] < qlen)
        )
        scores = tl.where(valid, scores, -float("inf"))
    if FIXED:
        new_max = tl.full((BM,), 32.0 * 1.4426950408889634, tl.float32)
        prob = tl.exp2(scores - 32.0 * 1.4426950408889634)
        denom += tl.sum(prob, 1)
    else:
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        safe = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.exp2(maximum - safe)
        prob = tl.exp2(scores - safe[:, None])
        denom = denom * alpha + tl.sum(prob, 1)
        a0 *= alpha[:, None]
        a1 *= alpha[:, None]
    p = prob.to(V.dtype.element_ty)
    if FULL:
        v0 = tl.load(V + vo[:, None] + dd[None, :], cache_modifier=".cg")
    else:
        v0 = tl.load(
            V + vo[:, None] + dd[None, :],
            start + ns[:, None] < length,
            0,
            cache_modifier=".cg",
        )
    if EARLY_V:
        if FULL:
            v1 = tl.load(V + vo[:, None] + 128 + dd[None, :], cache_modifier=".cg")
        else:
            v1 = tl.load(
                V + vo[:, None] + 128 + dd[None, :],
                start + ns[:, None] < length,
                0,
                cache_modifier=".cg",
            )
    a0 = tl.dot(p, v0, a0)
    if not EARLY_V:
        if FULL:
            v1 = tl.load(V + vo[:, None] + 128 + dd[None, :], cache_modifier=".cg")
        else:
            v1 = tl.load(
                V + vo[:, None] + 128 + dd[None, :],
                start + ns[:, None] < length,
                0,
                cache_modifier=".cg",
            )
    a1 = tl.dot(p, v1, a1)
    return (a0, a1, new_max, denom)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_decode"),
    key=[
        "B",
        "BB",
        "QS",
        "QH",
        "KP",
        "KS",
        "VP",
        "VS",
        "TS",
        "S",
        "BN",
        "BM",
        "CHUNK",
        "WORKERS",
        "LIMIT",
        "PT",
        "ORDER",
        "FIXED_MODE",
        "HEAD_MAJOR",
        "EARLY_K",
        "EARLY_V",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def tp2_mixed_medium_producer(
    Q,
    K,
    V,
    CUQ,
    USED,
    TABLE,
    P,
    PL,
    PREFIX,
    FLAGS,
    B: tl.constexpr,
    BB: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    TS: tl.constexpr,
    SCALE: tl.constexpr,
    S: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
    CHUNK: tl.constexpr,
    WORKERS: tl.constexpr,
    LIMIT: tl.constexpr,
    PT: tl.constexpr,
    ORDER: tl.constexpr,
    FIXED_MODE: tl.constexpr,
    HEAD_MAJOR: tl.constexpr,
    EARLY_K: tl.constexpr,
    EARLY_V: tl.constexpr,
):
    bs = tl.arange(0, BB)
    ends = tl.load(PREFIX + bs, bs < B, 2147483647)
    tasks = tl.load(PREFIX + B)
    total = tasks * S
    linear = tl.program_id(0)
    if linear >= total:
        return
    if ORDER:
        task = linear % tl.maximum(tasks, 1)
        split = linear // tl.maximum(tasks, 1)
    else:
        task = linear // S
        split = linear % S
    batch = tl.sum((task >= ends).to(tl.int32), 0)
    eligible = (tl.load(FLAGS + batch) != 0) & (Q.dtype.element_ty == tl.bfloat16)
    if eligible != (FIXED_MODE == 1):
        return
    previous = tl.load(PREFIX + batch - 1, batch > 0, 0)
    tile = task - previous
    qb = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qb
    length = tl.load(USED + batch)
    rows = tl.arange(0, BM)
    dd = tl.arange(0, 128)
    if HEAD_MAJOR:
        qi = tile * (BM // 8) + rows % (BM // 8)
        heads = rows // (BM // 8)
    else:
        qi = tile * (BM // 8) + rows // 8
        heads = rows % 8
    q0 = tl.load(
        Q + (qb + qi[:, None]).to(tl.int64) * QS + heads[:, None] * QH + dd[None, :],
        qi[:, None] < qlen,
        0,
    )
    q1 = tl.load(
        Q
        + (qb + qi[:, None]).to(tl.int64) * QS
        + heads[:, None] * QH
        + 128
        + dd[None, :],
        qi[:, None] < qlen,
        0,
    )
    low = split * CHUNK
    high = tl.minimum(
        low + CHUNK, tl.minimum(length, (tile + 1) * (BM // 8) + length - qlen)
    )
    full = tl.maximum(
        0, tl.minimum(high, tile * (BM // 8) + length - qlen + 1) // BN * BN
    )
    a0 = tl.zeros((BM, 128), tl.float32)
    a1 = tl.zeros((BM, 128), tl.float32)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    for start in range(low, full, BN):
        a0, a1, maximum, denom = _step(
            q0,
            q1,
            K,
            V,
            TABLE,
            a0,
            a1,
            maximum,
            denom,
            start,
            length,
            batch,
            qi,
            qlen,
            KP,
            KS,
            VP,
            VS,
            TS,
            SCALE,
            BN,
            BM,
            True,
            FIXED_MODE == 1,
            EARLY_K,
            EARLY_V,
        )
    for start in range(tl.maximum(low, full), high, BN):
        a0, a1, maximum, denom = _step(
            q0,
            q1,
            K,
            V,
            TABLE,
            a0,
            a1,
            maximum,
            denom,
            start,
            length,
            batch,
            qi,
            qlen,
            KP,
            KS,
            VP,
            VS,
            TS,
            SCALE,
            BN,
            BM,
            False,
            FIXED_MODE == 1,
            EARLY_K,
            EARLY_V,
        )
    token = batch * LIMIT + qi
    off = (
        split.to(tl.int64) * PT * 8 * 256
        + token[:, None].to(tl.int64) * 8 * 256
        + heads[:, None] * 256
        + dd[None, :]
    )
    tl.store(P + off, a0, qi[:, None] < qlen)
    tl.store(P + off + 128, a1, qi[:, None] < qlen)
    base = split * 2 * 8 * PT + heads * PT + token
    tl.store(PL + base, maximum, qi < qlen)
    tl.store(PL + base + 8 * PT, denom, qi < qlen)


@triton.jit
def merge(
    P,
    PL,
    OUT,
    LSE,
    CUQ,
    PREFIX,
    B: tl.constexpr,
    BB: tl.constexpr,
    T: tl.constexpr,
    PT: tl.constexpr,
    LIMIT: tl.constexpr,
    QM: tl.constexpr,
    OS: tl.constexpr,
    OH: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    WORKERS: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    bs = tl.arange(0, BB)
    ends = tl.load(PREFIX + bs, bs < B, 2147483647)
    total = tl.load(PREFIX + B) * QM * 8 * 4
    for linear in range(tl.program_id(0), total, WORKERS):
        task = linear // (QM * 32)
        batch = tl.sum((task >= ends).to(tl.int32), 0)
        previous = tl.load(PREFIX + batch - 1, batch > 0, 0)
        tile = task - previous
        qi = tile * QM + linear // 32 % QM
        head = linear // 4 % 8
        dtile = linear % 4
        qb = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qb
        if qi < qlen:
            ptok = batch * LIMIT + qi
            ss = tl.arange(0, BS)
            ds = dtile * 64 + tl.arange(0, 64)
            maxima = tl.load(
                PL + ss * 2 * 8 * PT + head * PT + ptok, ss < S, -float("inf")
            )
            sums = tl.load(PL + (ss * 2 + 1) * 8 * PT + head * PT + ptok, ss < S, 0.0)
            maximum = tl.max(maxima, 0)
            maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
            weights = tl.exp2(maxima - maximum)
            denom = tl.sum(weights * sums, 0)
            values = tl.load(
                P
                + ss[:, None].to(tl.int64) * PT * 8 * 256
                + ptok * 8 * 256
                + head * 256
                + ds[None, :],
                ss[:, None] < S,
                0.0,
            )
            out = tl.sum(values * weights[:, None], 0) / tl.where(denom > 0, denom, 1.0)
            tl.store(OUT + (qb + qi).to(tl.int64) * OS + head * OH + ds, out)
            if WRITE_LSE:
                if dtile == 0:
                    tl.store(
                        LSE + head * T + qb + qi,
                        tl.where(
                            denom > 0,
                            maximum * 0.6931471805599453 + tl.log(denom),
                            float("inf"),
                        ),
                    )


def tp2_mixed_medium_launch(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    used,
    table,
    *,
    max_seqlen_k,
    flags,
    scale=0.0625,
    limit=128,
    splits=64,
    workers=512,
    order=False,
    bm=16,
    bn=64,
    head_major=False,
    early_k=False,
    early_v=False,
    write_lse=True,
):
    batch = cuq.numel() - 1
    pt = limit * batch
    prefix = torch.empty((batch + 1,), device=q.device, dtype=torch.int32)
    partial = torch.empty((splits, pt, 8, 256), device=q.device, dtype=torch.float32)
    stats = torch.empty((splits, 2, 8, pt), device=q.device, dtype=torch.float32)
    tp2_mixed_medium_metadata[1,](
        cuq,
        used,
        prefix,
        batch,
        triton.next_power_of_2(batch),
        bm // 8,
        limit,
        num_warps=4,
        num_stages=1,
    )
    for fixed_mode in (1, 0):
        tp2_mixed_medium_producer[splits * triton.cdiv(limit, bm // 8) * batch,](
            q,
            k,
            v,
            cuq,
            used,
            table,
            partial,
            stats,
            prefix,
            flags,
            batch,
            triton.next_power_of_2(batch),
            *q.stride()[:2],
            *k.stride()[:2],
            *v.stride()[:2],
            table.stride(0),
            scale,
            splits,
            bn,
            bm,
            triton.cdiv(triton.cdiv(max_seqlen_k, bn), splits) * bn,
            workers,
            limit,
            pt,
            order,
            fixed_mode,
            head_major,
            early_k,
            early_v,
        )
    merge[workers,](
        partial,
        stats,
        out,
        lse,
        cuq,
        prefix,
        batch,
        triton.next_power_of_2(batch),
        q.shape[0],
        pt,
        limit,
        bm // 8,
        *out.stride()[:2],
        splits,
        triton.next_power_of_2(splits),
        workers,
        write_lse,
        num_warps=4,
        num_stages=1,
    )
    decode.tp2_mixed_q1_launch(
        q,
        k,
        v,
        out,
        lse,
        cuq,
        used,
        table,
        max_seqlen_k=max_seqlen_k,
        scale=scale,
        write_lse=write_lse,
    )


@triton.jit
def tp2_mixed_short_metadata(
    CUQ,
    USED,
    PREFIX,
    B: tl.constexpr,
    BB: tl.constexpr,
    QM: tl.constexpr,
    LIMIT: tl.constexpr,
):
    bs = tl.arange(0, BB)
    qb = tl.load(CUQ + bs, bs < B, 0)
    qe = tl.load(CUQ + bs + 1, bs < B, 0)
    length = tl.load(USED + bs, bs < B, 0)
    qlen = qe - qb
    count = tl.where(
        (bs < B) & (qlen > 0) & (qlen <= LIMIT) & (length >= 8192), tl.cdiv(qlen, QM), 0
    )
    ends = tl.cumsum(count, 0)
    tl.store(PREFIX + bs, ends, bs < B)
    tl.store(PREFIX + B, tl.sum(count, 0))


@triton.jit
def tp2_mixed_short_step(
    q,
    K,
    V,
    TABLE,
    acc,
    maximum,
    denom,
    start,
    batch,
    qi,
    qlen,
    klen,
    delta,
    KP: tl.constexpr,
    KS: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    TS: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    FULL: tl.constexpr,
):
    ns = tl.arange(0, BN)
    cols = start + ns
    ds = tl.arange(0, 256)
    pages = tl.load(TABLE + batch * TS + cols // 32, cols < klen, 0).to(tl.int64)
    ko = tl.multiple_of(pages * KP + cols % 32 * KS, 256)
    vo = tl.multiple_of(pages * VP + cols % 32 * VS, 256)
    if FULL:
        k = tl.load(K + ds[:, None] + ko[None, :])
    else:
        k = tl.load(K + ds[:, None] + ko[None, :], cols[None, :] < klen, 0)
    scores = tl.dot(q, k) * (SCALE * 1.4426950408889634)
    if not FULL:
        scores = tl.where(
            (cols[None, :] < klen)
            & (cols[None, :] <= qi[:, None] + delta)
            & (qi[:, None] < qlen),
            scores,
            -float("inf"),
        )
    new_max = tl.maximum(maximum, tl.max(scores, 1))
    safe = tl.where(new_max == -float("inf"), 0.0, new_max)
    alpha = tl.exp2(maximum - safe)
    p = tl.exp2(scores - safe[:, None])
    denom = denom * alpha + tl.sum(p, 1)
    acc *= alpha[:, None]
    if FULL:
        v = tl.load(V + vo[:, None] + ds[None, :])
    else:
        v = tl.load(V + vo[:, None] + ds[None, :], cols[:, None] < klen, 0)
    acc = tl.dot(p.to(v.dtype), v, acc)
    return (acc, new_max, denom)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_decode"),
    key=[
        "B",
        "BB",
        "QS",
        "QH",
        "KP",
        "KS",
        "VP",
        "VS",
        "TS",
        "S",
        "BM",
        "BN",
        "CHUNK",
        "WORKERS",
        "LIMIT",
        "PT",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def tp2_mixed_short_producer(
    Q,
    K,
    V,
    CUQ,
    USED,
    TABLE,
    P,
    PL,
    PREFIX,
    B: tl.constexpr,
    BB: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    TS: tl.constexpr,
    SCALE: tl.constexpr,
    S: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    CHUNK: tl.constexpr,
    WORKERS: tl.constexpr,
    LIMIT: tl.constexpr,
    PT: tl.constexpr,
):
    bs = tl.arange(0, BB)
    ends = tl.load(PREFIX + bs, bs < B, 2147483647)
    total = tl.load(PREFIX + B) * S
    for linear in range(tl.program_id(0), total, WORKERS):
        task = linear // S
        split = linear % S
        batch = tl.sum((task >= ends).to(tl.int32), 0)
        previous = tl.load(PREFIX + batch - 1, batch > 0, 0)
        tile = task - previous
        qb = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qb
        klen = tl.load(USED + batch)
        rows = tl.arange(0, BM)
        ds = tl.arange(0, 256)
        qi = tile * (BM // 8) + rows // 8
        head = rows % 8
        q = tl.load(
            Q + (qb + qi[:, None]).to(tl.int64) * QS + head[:, None] * QH + ds[None, :],
            qi[:, None] < qlen,
            0,
        )
        low = split * CHUNK
        high = tl.minimum(
            low + CHUNK, tl.minimum(klen, (tile + 1) * (BM // 8) + klen - qlen)
        )
        full = tl.maximum(
            0, tl.minimum(high, tile * (BM // 8) + klen - qlen + 1) // BN * BN
        )
        acc = tl.zeros((BM, 256), tl.float32)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denom = tl.zeros((BM,), tl.float32)
        for start in range(low, full, BN):
            acc, maximum, denom = tp2_mixed_short_step(
                q,
                K,
                V,
                TABLE,
                acc,
                maximum,
                denom,
                start,
                batch,
                qi,
                qlen,
                klen,
                klen - qlen,
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BM,
                BN,
                True,
            )
        for start in range(tl.maximum(low, full), high, BN):
            acc, maximum, denom = tp2_mixed_short_step(
                q,
                K,
                V,
                TABLE,
                acc,
                maximum,
                denom,
                start,
                batch,
                qi,
                qlen,
                klen,
                klen - qlen,
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BM,
                BN,
                False,
            )
        token = batch * LIMIT + qi
        offset = (
            split.to(tl.int64) * PT * 8 * 256
            + token[:, None].to(tl.int64) * 8 * 256
            + head[:, None] * 256
            + ds[None, :]
        )
        tl.store(P + offset, acc, qi[:, None] < qlen)
        base = split * 2 * 8 * PT + head * PT + token
        tl.store(PL + base, maximum, qi < qlen)
        tl.store(PL + base + 8 * PT, denom, qi < qlen)


def tp2_mixed_short_launch(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    used,
    table,
    *,
    max_seqlen_k,
    scale=0.0625,
    limit=64,
    bm=16,
    bn=64,
    splits=64,
    workers=512,
    write_lse=True,
):
    batch = cuq.numel() - 1
    pt = limit * batch
    assert k.stride(0) % 256 == 0 and k.stride(1) % 256 == 0
    assert v.stride(0) % 256 == 0 and v.stride(1) % 256 == 0
    prefix = torch.empty((batch + 1,), device=q.device, dtype=torch.int32)
    partial = torch.empty((splits, pt, 8, 256), device=q.device, dtype=torch.float32)
    stats = torch.empty((splits, 2, 8, pt), device=q.device, dtype=torch.float32)
    tp2_mixed_short_metadata[1,](
        cuq,
        used,
        prefix,
        batch,
        triton.next_power_of_2(batch),
        bm // 8,
        limit,
        num_warps=4,
        num_stages=1,
    )
    tp2_mixed_short_producer[workers,](
        q,
        k,
        v,
        cuq,
        used,
        table,
        partial,
        stats,
        prefix,
        batch,
        triton.next_power_of_2(batch),
        *q.stride()[:2],
        *k.stride()[:2],
        *v.stride()[:2],
        table.stride(0),
        scale,
        splits,
        bm,
        bn,
        triton.cdiv(triton.cdiv(max_seqlen_k, bn), splits) * bn,
        workers,
        limit,
        pt,
    )
    decode.splitkv_merge_kernel[limit, 8, batch](
        partial,
        stats,
        out,
        lse,
        cuq,
        used,
        q.shape[0],
        pt,
        8,
        256,
        *out.stride()[:2],
        splits,
        triton.next_power_of_2(splits),
        256,
        limit,
        num_warps=4,
        num_stages=1,
    )


def supports(q, k, v):
    return (
        q.dtype == torch.bfloat16
        and q.shape[1:] == (16, 256)
        and (k.shape[1:] == (32, 2, 256))
        and all((x.stride(i) % 256 == 0 for x in (k, v) for i in (0, 1)))
    )


@triton.jit
def pack_flags(SRC, DST, B: tl.constexpr, HK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    batch, head = (i // HK, i % HK)
    value = tl.load(SRC + i, i < B * HK, 0)
    tl.store(DST + head * B + batch, value, i < B * HK)


def tp1_mixed_short_launch(
    q, k, v, out, lse, cuq, used, table, eligibility, scale, write_lse
):
    batch = used.numel()
    flags = torch.empty((2, batch), device=q.device, dtype=torch.int8)
    pack_flags[1,](
        eligibility,
        flags,
        batch,
        2,
        triton.next_power_of_2(batch * 2),
        num_warps=4,
        num_stages=1,
    )
    for kh in range(2):
        begin, end = (kh * 8, (kh + 1) * 8)
        tp2_mixed_medium_launch(
            q[:, begin:end, :],
            k[:, :, kh : kh + 1, :],
            v[:, :, kh : kh + 1, :],
            out[:, begin:end, :],
            lse[begin:end, :],
            cuq,
            used,
            table,
            max_seqlen_k=table.shape[1] * k.shape[1],
            flags=flags[kh],
            scale=scale,
            limit=64,
            splits=64,
            workers=512,
            order=False,
            bm=32,
            bn=64,
            head_major=True,
            early_k=True,
            early_v=True,
            write_lse=write_lse,
        )
