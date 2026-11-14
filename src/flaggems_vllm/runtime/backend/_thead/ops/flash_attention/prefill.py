# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def linear_metadata_page_norm(
    Q,
    K,
    CUQ,
    TABLE,
    USED,
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
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    B: tl.constexpr,
):
    tile, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    start = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - start
    klen = tl.load(USED + batch)
    if qlen <= 64 and klen >= 8192:
        return
    ds = tl.arange(0, 256)
    if tile < QW:
        qmax = 0.0
        group: tl.constexpr = HQ // HK
        if tile * B < qlen * group:
            flat = tile * B + tl.arange(0, B)
            qi, qh = (flat // group, head * group + flat % group)
            q = tl.load(
                Q
                + (start + qi[:, None]).to(tl.int64) * QS
                + qh[:, None] * QH
                + ds[None, :],
                qi[:, None] < qlen,
                0,
            ).to(tl.float32)
            squares = tl.sum(q * q, 1)
            qmax = tl.max(tl.where(squares == squares, squares, float("inf")), 0)
        tl.store(QPART + (batch * HK + head) * QW + tile, qmax)
    if tile < KW:
        kmax = 0.0
        if tile * PAGE < klen:
            page = tl.load(TABLE + batch * TS + tile).to(tl.int64)
            rows = tile * PAGE + tl.arange(0, PAGE)
            k = tl.load(
                K + page * KP + rows[:, None] % PAGE * KS + head * KH + ds[None, :],
                rows[:, None] < klen,
                0,
            ).to(tl.float32)
            squares = tl.sum(k * k, 1)
            kmax = tl.max(tl.where(squares == squares, squares, float("inf")), 0)
        tl.store(KPART + (batch * HK + head) * KW + tile, kmax)


@triton.jit
def linear_metadata_combine_norms(
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
):
    group = tl.program_id(0)
    batch = group // HK
    qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
    klen = tl.load(USED + batch)
    if qlen <= 64 and klen >= 8192:
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
def linear_stage_copy_pages(
    K,
    V,
    TABLE,
    USED,
    CUQ,
    PK,
    PV,
    PT,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    PP: tl.constexpr,
    PS: tl.constexpr,
    PH: tl.constexpr,
    TS: tl.constexpr,
    NP: tl.constexpr,
    PAGE: tl.constexpr,
    D: tl.constexpr,
    SHORT_LIMIT: tl.constexpr,
):
    slot, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    head64 = head.to(tl.int64)
    klen = tl.load(USED + batch)
    qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    logical = batch * NP + slot
    rows = slot * PAGE + tl.arange(0, PAGE)
    ds = tl.arange(0, D)
    page = tl.load(TABLE + batch.to(tl.int64) * TS + slot, slot * PAGE < klen, 0).to(
        tl.int64
    )
    offset = (rows % PAGE).to(tl.int64)
    kval = tl.load(
        K + page * KP + offset[:, None] * KS + head64 * KH + ds[None, :],
        rows[:, None] < klen,
        0,
    )
    vval = tl.load(
        V + page * VP + offset[:, None] * VS + head64 * VH + ds[None, :],
        rows[:, None] < klen,
        0,
    )
    dest = logical.to(tl.int64) * PP + offset[:, None] * PS + head64 * PH + ds[None, :]
    tl.store(PK + dest, kval)
    tl.store(PV + dest, vval)
    if head == 0:
        tl.store(PT + batch * NP + slot, logical)


def linear_stage_prepare(k, v, table, used, cuq, layout, short_limit):
    batch, hk, dim = (used.numel(), k.shape[-2], k.shape[-1])
    page, npages = (k.shape[1], max(1, table.shape[1]))
    assert dim == 256 and page in (16, 32)
    assert k.stride(-1) == v.stride(-1) == 1
    physical = batch * npages
    shape = (physical, page, hk, dim)
    if layout == "interleaved":
        strides = (page * hk * dim, hk * dim, dim, 1)
    elif layout == "head_major":
        strides = (page * dim, dim, physical * page * dim, 1)
    else:
        raise ValueError(layout)
    pk = torch.empty_strided(shape, strides, dtype=k.dtype, device=k.device)
    pv = torch.empty_strided(shape, strides, dtype=v.dtype, device=v.device)
    pt = torch.empty((batch, npages), dtype=table.dtype, device=table.device)
    linear_stage_copy_pages[npages, batch, hk](
        k,
        v,
        table,
        used,
        cuq,
        pk,
        pv,
        pt,
        *k.stride()[:3],
        *v.stride()[:3],
        *pk.stride()[:3],
        table.stride(0),
        npages,
        page,
        dim,
        short_limit,
        num_warps=4,
        num_stages=1,
    )
    return (pk, pv, pt)


@triton.jit
def linear_prefill_step(
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
    if FULL:
        k = tl.load(K + ko[None, :] + kh * KH + ds[:, None])
    else:
        k = tl.load(K + ko[None, :] + kh * KH + ds[:, None], cols[None, :] < klen, 0)
    scores = tl.dot(q, k) * (SCALE * 1.4426950408889634)
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
    if FULL:
        v = tl.load(V + vo[:, None] + kh * VH + ds[None, :])
    else:
        v = tl.load(V + vo[:, None] + kh * VH + ds[None, :], cols[:, None] < klen, 0)
    acc = tl.dot(p.to(v.dtype), v, acc)
    return (acc, new_m, row_sum)


@triton.jit
def linear_prefill_attention_kernel(
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
    qi = tile * QM + tl.arange(0, BM) % QM
    qh = head * GROUP + tl.arange(0, BM) // QM
    ds = tl.arange(0, BD)
    qrow = qstart + qi[:, None]
    if (TQ - 1) * QS + (HQ - 1) * QH + D >= 2147483648:
        qrow = qrow.to(tl.int64)
    q = tl.load(Q + qrow * QS + qh[:, None] * QH + ds[None, :], qi[:, None] < qlen, 0)
    kh = head // (HQ // HK // FOLD)
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
    fixed_ok = (tl.load(KNORM + batch * HK + kh) != 0) & (
        Q.dtype.element_ty == tl.bfloat16
    )
    if fixed_ok:
        for start in range(0, full_hi, BN):
            acc, m, row_sum = linear_prefill_step(
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
            )
        for start in range(full_hi, hi, BN):
            acc, m, row_sum = linear_prefill_step(
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
            )
    else:
        for start in range(0, full_hi, BN):
            acc, m, row_sum = linear_prefill_step(
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
            )
        for start in range(full_hi, hi, BN):
            acc, m, row_sum = linear_prefill_step(
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
            )
    row_sum = tl.sum(row_sum, 1)
    inv = tl.div_rn(1.0, tl.where(row_sum > 0, row_sum, 1.0))
    result = acc * inv[:, None]
    orow = qstart + qi[:, None]
    if (TQ - 1) * OS + (HQ - 1) * OH + D >= 2147483648:
        orow = orow.to(tl.int64)
    tl.store(
        Out + orow * OS + qh[:, None] * OH + ds[None, :], result, qi[:, None] < qlen
    )
    if WRITE_LSE:
        lse = tl.where(
            row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
        )
        tl.store(LSE + qh * TQ + qstart + qi, lse, qi < qlen)


@triton.jit
def bounded_prefill_page_norm(
    Q,
    K,
    CUQ,
    TABLE,
    USED,
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
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    B: tl.constexpr,
):
    tile, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    start = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - start
    klen = tl.load(USED + batch)
    ds = tl.arange(0, 256)
    if tile < QW:
        qmax = 0.0
        group: tl.constexpr = HQ // HK
        if tile * B < qlen * group:
            flat = tile * B + tl.arange(0, B)
            qi, qh = (flat // group, head * group + flat % group)
            q = tl.load(
                Q
                + (start + qi[:, None]).to(tl.int64) * QS
                + qh[:, None] * QH
                + ds[None, :],
                qi[:, None] < qlen,
                0,
            ).to(tl.float32)
            squares = tl.sum(q * q, 1)
            qmax = tl.max(tl.where(squares == squares, squares, float("inf")), 0)
        tl.store(QPART + (batch * HK + head) * QW + tile, qmax)
    if tile < KW:
        kmax = 0.0
        if tile * PAGE < klen:
            page = tl.load(TABLE + batch * TS + tile).to(tl.int64)
            rows = tile * PAGE + tl.arange(0, PAGE)
            k = tl.load(
                K + page * KP + rows[:, None] % PAGE * KS + head * KH + ds[None, :],
                rows[:, None] < klen,
                0,
            ).to(tl.float32)
            squares = tl.sum(k * k, 1)
            kmax = tl.max(tl.where(squares == squares, squares, float("inf")), 0)
        tl.store(KPART + (batch * HK + head) * KW + tile, kmax)


def prepare_flags(q, k, cuq, table, used, maxq, scale, qm):
    batch, hk = (used.numel(), k.shape[-2])
    kw = max(1, table.shape[1])
    qw = max(1, triton.cdiv(maxq * (q.shape[1] // hk), 32))
    kp = torch.empty((batch, hk, kw), device=k.device, dtype=torch.float32)
    qp = torch.empty((batch, hk, qw), device=q.device, dtype=torch.float32)
    flags = torch.empty((batch, hk), device=q.device, dtype=torch.int8)
    cap = triton.cdiv(q.shape[0], qm) + batch - 1
    meta = torch.empty((5 * cap + 1,), device=q.device, dtype=torch.int32)
    bounded_prefill_page_norm[max(qw, kw), batch, hk](
        q,
        k,
        cuq,
        table,
        used,
        qp,
        kp,
        q.shape[1],
        hk,
        qw,
        kw,
        *q.stride()[:2],
        *k.stride()[:3],
        table.stride(0),
        k.shape[1],
        32,
        num_warps=4,
        num_stages=1,
    )
    bounded_prefill_combine_norms[batch * hk,](
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
    )
    return (flags, meta)


@triton.jit
def bounded_prefill_combine_norms(
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
):
    group = tl.program_id(0)
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
def bounded_prefill_step(
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
):
    ns = tl.arange(0, BN)
    ds = tl.arange(0, BD)
    cols = start + ns
    if BN <= PAGE:
        page = tl.load(TABLE + batch * TS + start // PAGE).to(tl.int64)
        ko = page * KP + (start % PAGE + ns) * KS
        vo = page * VP + (start % PAGE + ns) * VS
    else:
        slots = tl.arange(0, BN // PAGE)
        if FULL:
            pages = tl.load(TABLE + batch * TS + start // PAGE + slots)
        else:
            pages = tl.load(
                TABLE + batch * TS + start // PAGE + slots,
                start + slots * PAGE < klen,
                0,
            )
        within = tl.arange(0, PAGE)
        ko = (pages[:, None].to(tl.int64) * KP + within[None, :] * KS).reshape(BN)
        vo = (pages[:, None].to(tl.int64) * VP + within[None, :] * VS).reshape(BN)
    if KP % 8 == 0 and KS % 8 == 0:
        ko = tl.multiple_of(ko, 8)
    if VP % 8 == 0 and VS % 8 == 0:
        vo = tl.multiple_of(vo, 8)
    if FULL:
        k = tl.load(K + ko[None, :] + kh * KH + ds[:, None])
    else:
        k = tl.load(K + ko[None, :] + kh * KH + ds[:, None], cols[None, :] < klen, 0)
    scores = tl.dot(q, k) * (SCALE * 1.4426950408889634)
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
    if FULL:
        v = tl.load(V + vo[:, None] + kh * VH + ds[None, :])
    else:
        v = tl.load(V + vo[:, None] + kh * VH + ds[None, :], cols[:, None] < klen, 0)
    acc = tl.dot(p.to(v.dtype), v, acc)
    return (acc, new_m, row_sum)


@triton.jit
def bounded_prefill_attention_kernel(
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
    qi = tile * QM + tl.arange(0, BM) % QM
    qh = head * GROUP + tl.arange(0, BM) // QM
    ds = tl.arange(0, BD)
    qrow = qstart + qi[:, None]
    if (TQ - 1) * QS + (HQ - 1) * QH + D >= 2147483648:
        qrow = qrow.to(tl.int64)
    q = tl.load(Q + qrow * QS + qh[:, None] * QH + ds[None, :], qi[:, None] < qlen, 0)
    kh = head // (HQ // HK // FOLD)
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
    fixed_ok = (tl.load(KNORM + batch * HK + kh) != 0) & (
        Q.dtype.element_ty == tl.bfloat16
    )
    if fixed_ok:
        for start in range(0, full_hi, BN):
            acc, m, row_sum = bounded_prefill_step(
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
            )
        for start in range(full_hi, hi, BN):
            acc, m, row_sum = bounded_prefill_step(
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
            )
    else:
        for start in range(0, full_hi, BN):
            acc, m, row_sum = bounded_prefill_step(
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
            )
        for start in range(full_hi, hi, BN):
            acc, m, row_sum = bounded_prefill_step(
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
            )
    row_sum = tl.sum(row_sum, 1)
    inv = tl.div_rn(1.0, tl.where(row_sum > 0, row_sum, 1.0))
    result = acc * inv[:, None]
    orow = qstart + qi[:, None]
    if (TQ - 1) * OS + (HQ - 1) * OH + D >= 2147483648:
        orow = orow.to(tl.int64)
    tl.store(
        Out + orow * OS + qh[:, None] * OH + ds[None, :], result, qi[:, None] < qlen
    )
    if WRITE_LSE:
        lse = tl.where(
            row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
        )
        tl.store(LSE + qh * TQ + qstart + qi, lse, qi < qlen)


def launch_bounded_prefill(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    write_lse=True,
):
    paged = table is not None
    packed = False
    hkv = k.shape[-2]
    batches = cuq.numel() - 1
    qm = plan.block_m // 4
    flags, meta = prepare_flags(q, k, cuq, table, usedk, maxq, scale, qm)
    grid = (triton.cdiv(q.shape[0], qm) + batches - 1, 1, q.shape[1] // 4)
    return bounded_prefill_attention_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        q.shape[0],
        q.shape[1],
        hkv,
        q.shape[2],
        *q.stride()[:2],
        k.stride(0) if paged else 0,
        k.stride(-3),
        k.stride(-2),
        v.stride(0) if paged else 0,
        v.stride(-3),
        v.stride(-2),
        *out.stride()[:2],
        table.stride(0) if paged else 0,
        alibi.stride(0) if alibi is not None and alibi.ndim == 2 else 0,
        k.shape[1] if paged else 1,
        scale,
        causal,
        *window,
        softcap,
        paged,
        alibi is not None,
        packed,
        plan.block_m,
        plan.block_n,
        max(64, triton.next_power_of_2(q.shape[2])),
        BATCHES=batches,
        FOLD=4,
        ORDER=0 if plan.family == "bounded_prefill_long" else 1,
        KNORM=flags,
        WORKLIST=meta,
        CAPACITY=(meta.numel() - 1) // 5,
        WRITE_LSE=write_lse,
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
    )


@triton.jit
def ordered_prefill_attention_kernel(
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
    qi = tile * QM + tl.arange(0, BM) % QM
    qh = head * GROUP + tl.arange(0, BM) // QM
    ds = tl.arange(0, BD)
    qrow = qstart + qi[:, None]
    if (TQ - 1) * QS + (HQ - 1) * QH + D >= 2147483648:
        qrow = qrow.to(tl.int64)
    q = tl.load(Q + qrow * QS + qh[:, None] * QH + ds[None, :], qi[:, None] < qlen, 0)
    kh = head // (HQ // HK // FOLD)
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
    fixed_ok = (tl.load(KNORM + batch * HK + kh) != 0) & (
        Q.dtype.element_ty == tl.bfloat16
    )
    if fixed_ok & (SCALE >= -0.0625) & (SCALE <= 0.0625):
        for start in range(0, full_hi, BN):
            acc, m, row_sum = bounded_prefill_step(
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
            )
        for start in range(full_hi, hi, BN):
            acc, m, row_sum = bounded_prefill_step(
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
            )
    else:
        for start in range(0, full_hi, BN):
            acc, m, row_sum = bounded_prefill_step(
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
            )
        for start in range(full_hi, hi, BN):
            acc, m, row_sum = bounded_prefill_step(
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
            )
    row_sum = tl.sum(row_sum, 1)
    inv = tl.div_rn(1.0, tl.where(row_sum > 0, row_sum, 1.0))
    result = acc * inv[:, None]
    orow = qstart + qi[:, None]
    if (TQ - 1) * OS + (HQ - 1) * OH + D >= 2147483648:
        orow = orow.to(tl.int64)
    tl.store(
        Out + orow * OS + qh[:, None] * OH + ds[None, :], result, qi[:, None] < qlen
    )
    if WRITE_LSE:
        lse = tl.where(
            row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
        )
        tl.store(LSE + qh * TQ + qstart + qi, lse, qi < qlen)


@triton.jit
def ordered_worklist_combine_prefill(
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
    INCLUDE_Q1: tl.constexpr = False,
):
    MIN_Q: tl.constexpr = 0 if INCLUDE_Q1 else 1
    group = tl.program_id(0)
    batch = group // HK
    qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
    if qlen > MIN_Q:
        qcols = tl.arange(0, QB)
        kcols = tl.arange(0, KB)
        qn = tl.max(tl.load(QPART + group * QW + qcols, qcols < QW, 0), 0)
        kn = tl.max(tl.load(KPART + group * KW + kcols, kcols < KW, 0), 0)
        ok = qn * kn * (SCALE * SCALE) <= 1008.0625
        tl.store(FLAG + group, ok.to(tl.int8))
    else:
        tl.store(FLAG + group, 0)
    batch = group // HK
    if group % HK == 0:
        bs = tl.arange(0, BB)
        qs = tl.load(CUQ + bs, bs < BATCHES, 0)
        qe = tl.load(CUQ + bs + 1, bs < BATCHES, 0)
        counts = tl.where(qe - qs > MIN_Q, tl.cdiv(qe - qs, QM), 0)
        total = tl.sum(counts, 0)
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        klen = tl.load(USED + batch)
        ts = tl.arange(0, QT)
        BN: tl.constexpr = 32
        tl.static_assert(QM == 16)
        all_k = tl.load(USED + bs, bs < BATCHES, 0)
        all_delta = all_k - (qe - qs)
        n = tl.cdiv(qlen, QM)
        actual_tile = n - 1 - ts
        hi = tl.minimum(klen, (actual_tile + 1) * QM + klen - qlen)
        costs = tl.maximum(0, tl.cdiv(hi, BN))
        cut = tl.maximum(costs[None, :] * BN - all_delta[:, None], 0) // QM
        greater = tl.where(
            all_k[:, None] > costs[None, :] * BN,
            tl.maximum(counts[:, None] - cut, 0),
            0,
        )
        cut_prev = tl.maximum((costs[None, :] - 1) * BN - all_delta[:, None], 0) // QM
        greater_prev = tl.where(
            all_k[:, None] > (costs[None, :] - 1) * BN,
            tl.maximum(counts[:, None] - cut_prev, 0),
            0,
        )
        greater_prev = tl.where(costs[None, :] == 0, counts[:, None], greater_prev)
        own_cut = tl.maximum(costs * BN - (klen - qlen), 0) // QM
        own_greater = tl.where(klen > costs * BN, tl.maximum(n - own_cut, 0), 0)
        positions = (
            tl.sum(greater, 0)
            + tl.sum(tl.where(bs[:, None] < batch, greater_prev - greater, 0), 0)
            + ts
            - own_greater
        )
        valid = (qlen > MIN_Q) & (ts < tl.cdiv(qlen, QM))
        tl.store(META + positions, qstart, valid)
        tl.store(META + CAP + positions, qlen, valid)
        tl.store(META + 2 * CAP + positions, klen, valid)
        tl.store(META + 3 * CAP + positions, batch, valid)
        tl.store(META + 4 * CAP + positions, ts, valid)
        if group == 0:
            tl.store(META + 5 * CAP, total)


@triton.jit
def page_norm_prefill(
    Q,
    K,
    CUQ,
    TABLE,
    USED,
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
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    B: tl.constexpr,
):
    tile, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    start = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - start
    if qlen <= 1:
        return
    klen = tl.load(USED + batch)
    ds = tl.arange(0, 256)
    if tile < QW:
        qmax = 0.0
        group: tl.constexpr = HQ // HK
        if tile * B < qlen * group:
            flat = tile * B + tl.arange(0, B)
            qi, qh = (flat // group, head * group + flat % group)
            q = tl.load(
                Q
                + (start + qi[:, None]).to(tl.int64) * QS
                + qh[:, None] * QH
                + ds[None, :],
                qi[:, None] < qlen,
                0,
            ).to(tl.float32)
            squares = tl.sum(q * q, 1)
            qmax = tl.max(tl.where(squares == squares, squares, float("inf")), 0)
        tl.store(QPART + (batch * HK + head) * QW + tile, qmax)
    if tile < KW:
        kmax = 0.0
        if tile * PAGE < klen:
            page = tl.load(TABLE + batch * TS + tile).to(tl.int64)
            rows = tile * PAGE + tl.arange(0, PAGE)
            k = tl.load(
                K + page * KP + rows[:, None] % PAGE * KS + head * KH + ds[None, :],
                rows[:, None] < klen,
                0,
            ).to(tl.float32)
            squares = tl.sum(k * k, 1)
            kmax = tl.max(tl.where(squares == squares, squares, float("inf")), 0)
        tl.store(KPART + (batch * HK + head) * KW + tile, kmax)


@triton.jit
def prefill_worklist_combine_prefill(
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
):
    group = tl.program_id(0)
    batch = group // HK
    qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
    if qlen > 1:
        qcols = tl.arange(0, QB)
        kcols = tl.arange(0, KB)
        qn = tl.max(tl.load(QPART + group * QW + qcols, qcols < QW, 0), 0)
        kn = tl.max(tl.load(KPART + group * KW + kcols, kcols < KW, 0), 0)
        ok = qn * kn * (SCALE * SCALE) <= 1008.0625
        tl.store(FLAG + group, ok.to(tl.int8))
    else:
        tl.store(FLAG + group, 0)
    batch = group // HK
    if group % HK == 0:
        bs = tl.arange(0, BB)
        qs = tl.load(CUQ + bs, bs < BATCHES, 0)
        qe = tl.load(CUQ + bs + 1, bs < BATCHES, 0)
        counts = tl.where(qe - qs > 1, tl.cdiv(qe - qs, QM), 0)
        before = tl.sum(tl.where(bs < batch, counts, 0), 0)
        total = tl.sum(counts, 0)
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        klen = tl.load(USED + batch)
        ts = tl.arange(0, QT)
        positions = before + ts
        valid = (qlen > 1) & (ts < tl.cdiv(qlen, QM))
        tl.store(META + positions, qstart, valid)
        tl.store(META + CAP + positions, qlen, valid)
        tl.store(META + 2 * CAP + positions, klen, valid)
        tl.store(META + 3 * CAP + positions, batch, valid)
        tl.store(META + 4 * CAP + positions, ts, valid)
        if group == 0:
            tl.store(META + 5 * CAP, total)


def short_metadata_prepare(
    q,
    k,
    cuq,
    table,
    used,
    maxq,
    scale,
    qm,
    prefill_only=True,
    ordered=False,
    norm_block=32,
    norm_warps=4,
):
    batch, hk = (used.numel(), k.shape[-2])
    kw = max(1, table.shape[1])
    qw = max(1, triton.cdiv(maxq * (q.shape[1] // hk), norm_block))
    kp = torch.empty((batch, hk, kw), device=k.device, dtype=torch.float32)
    qp = torch.empty((batch, hk, qw), device=q.device, dtype=torch.float32)
    flags = torch.empty((batch, hk), device=q.device, dtype=torch.int8)
    cap = triton.cdiv(q.shape[0], qm) + batch - 1
    meta = torch.empty((5 * cap + 1,), device=q.device, dtype=torch.int32)
    norm = page_norm_prefill if prefill_only else bounded_prefill_page_norm
    norm[max(qw, kw), batch, hk](
        q,
        k,
        cuq,
        table,
        used,
        qp,
        kp,
        q.shape[1],
        hk,
        qw,
        kw,
        *q.stride()[:2],
        *k.stride()[:3],
        table.stride(0),
        k.shape[1],
        norm_block,
        num_warps=norm_warps,
        num_stages=1,
    )
    if ordered:
        combine = ordered_worklist_combine_prefill
    else:
        combine = (
            prefill_worklist_combine_prefill
            if prefill_only
            else bounded_prefill_combine_norms
        )
    combine_options = dict(INCLUDE_Q1=not prefill_only) if ordered else {}
    combine[batch * hk,](
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
        **combine_options,
    )
    return (flags, meta)


@triton.jit
def async_stage_copy_pages(
    K,
    V,
    TABLE,
    USED,
    CUQ,
    PK,
    PV,
    PT,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    PP: tl.constexpr,
    PS: tl.constexpr,
    PH: tl.constexpr,
    TS: tl.constexpr,
    NP: tl.constexpr,
    PAGE: tl.constexpr,
    D: tl.constexpr,
    SHORT_LIMIT: tl.constexpr,
    PREFILL_ONLY: tl.constexpr = False,
):
    slot, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    head64 = head.to(tl.int64)
    klen = tl.load(USED + batch)
    qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
    if PREFILL_ONLY and qlen <= 1:
        return
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    logical = batch * NP + slot
    rows = slot * PAGE + tl.arange(0, PAGE)
    ds = tl.arange(0, D)
    page = tl.load(TABLE + batch.to(tl.int64) * TS + slot, slot * PAGE < klen, 0).to(
        tl.int64
    )
    offset = (rows % PAGE).to(tl.int64)
    kval = tl.load(
        K + page * KP + offset[:, None] * KS + head64 * KH + ds[None, :],
        rows[:, None] < klen,
        0,
    )
    vval = tl.load(
        V + page * VP + offset[:, None] * VS + head64 * VH + ds[None, :],
        rows[:, None] < klen,
        0,
    )
    dest = logical.to(tl.int64) * PP + offset[:, None] * PS + head64 * PH + ds[None, :]
    tl.store(PK + dest, kval)
    tl.store(PV + dest, vval)
    if head == 0:
        tl.store(PT + batch * NP + slot, logical)


def async_stage_prepare(
    k, v, table, used, cuq, layout, short_limit, prefill_only=False
):
    batch, hk, dim = (used.numel(), k.shape[-2], k.shape[-1])
    page, npages = (k.shape[1], max(1, table.shape[1]))
    assert dim == 256 and page in (16, 32)
    assert k.stride(-1) == v.stride(-1) == 1
    physical = batch * npages
    shape = (physical, page, hk, dim)
    if layout == "interleaved":
        strides = (page * hk * dim, hk * dim, dim, 1)
    elif layout == "head_major":
        strides = (page * dim, dim, physical * page * dim, 1)
    else:
        raise ValueError(layout)
    pk = torch.empty_strided(shape, strides, dtype=k.dtype, device=k.device)
    pv = torch.empty_strided(shape, strides, dtype=v.dtype, device=v.device)
    pt = torch.empty((batch, npages), dtype=table.dtype, device=table.device)
    async_stage_copy_pages[npages, batch, hk](
        k,
        v,
        table,
        used,
        cuq,
        pk,
        pv,
        pt,
        *k.stride()[:3],
        *v.stride()[:3],
        *pk.stride()[:3],
        table.stride(0),
        npages,
        page,
        dim,
        short_limit,
        PREFILL_ONLY=prefill_only,
        num_warps=4,
        num_stages=1,
    )
    return (pk, pv, pt)
