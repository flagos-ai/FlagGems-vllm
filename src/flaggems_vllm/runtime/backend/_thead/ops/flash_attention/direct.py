# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


def _prune_attention_tiles(configs, named_args, **kwargs):
    if not named_args["PACKED"]:
        return configs
    groups = named_args["HQ"] // named_args["HK"]
    choices = [config for config in configs if config.kwargs["BM"] >= groups]
    return choices or [
        triton.Config(
            {"BM": triton.next_power_of_2(max(16, groups)), "BN": 64},
            num_warps=4,
            num_stages=1,
        )
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_tiles"),
    key=[
        "TQ",
        "HQ",
        "HK",
        "D",
        "QS",
        "QH",
        "KP",
        "KS",
        "KH",
        "VP",
        "VS",
        "VH",
        "OS",
        "OH",
        "TS",
        "AS",
        "PAGE",
        "CAUSAL",
        "WL",
        "WR",
        "SOFTCAP",
        "PAGED",
        "HAS_ALIBI",
        "PACKED",
        "BD",
        "MAX_Q",
        "KV_CAPACITY",
        "BATCH_COUNT",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
    prune_configs_by={"early_config_prune": _prune_attention_tiles},
)
@triton.jit
def direct_attention_kernel(
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
    MAX_Q: tl.constexpr = 0,
    KV_CAPACITY: tl.constexpr = 0,
    BATCH_COUNT: tl.constexpr = 0,
):
    tile, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    if PAGED:
        kstart = 0
        klen = tl.load(USEDK + batch)
    else:
        kstart = tl.load(CUK + batch)
        klen = tl.load(CUK + batch + 1) - kstart
    groups: tl.constexpr = HQ // HK
    rows = tl.arange(0, BM)
    if PACKED:
        qi = tl.full((BM,), 0, tl.int32)
        qh = head * groups + rows
        kh = head
        valid_q = (rows < groups) & (qlen > 0)
        row_base = 0
        row_end = 1
    else:
        qi = tile * BM + rows
        qh = tl.full((BM,), head, tl.int32)
        kh = head // groups
        valid_q = qi < qlen
        row_base = tile * BM
        row_end = tl.minimum((tile + 1) * BM, qlen)
    if row_base >= qlen:
        return
    dims = tl.arange(0, BD)
    q = tl.load(
        Q + (qstart + qi[:, None]).to(tl.int64) * QS + qh[:, None] * QH + dims[None, :],
        valid_q[:, None] & (dims[None, :] < D),
        0,
    )
    low = 0
    high = klen
    if WL >= 0:
        low = tl.maximum(0, row_base + klen - qlen - WL)
        low = low // BN * BN
    if CAUSAL:
        high = tl.minimum(high, row_end + klen - qlen)
    if WR >= 0:
        high = tl.minimum(high, row_end + klen - qlen + WR)
    m = tl.full((BM,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    if HAS_ALIBI:
        slope = tl.load(ALIBI + batch * AS + qh, valid_q, 0)
    for start in range(low, high, BN):
        cols = start + tl.arange(0, BN)
        if PAGED:
            page = tl.load(TABLE + batch * TS + cols // PAGE, cols < klen, 0)
            ko = page.to(tl.int64) * KP + cols % PAGE * KS
            vo = page.to(tl.int64) * VP + cols % PAGE * VS
        else:
            ko = (kstart + cols).to(tl.int64) * KS
            vo = (kstart + cols).to(tl.int64) * VS
        k = tl.load(
            K + ko[None, :] + kh * KH + dims[:, None],
            (cols[None, :] < klen) & (dims[:, None] < D),
            0,
        )
        scores = tl.dot(q, k) * SCALE
        if SOFTCAP > 0:
            z = scores / SOFTCAP
            scores = SOFTCAP * (2.0 / (1.0 + tl.exp(2.0 * -z)) - 1.0)
        pos = qi[:, None] + klen - qlen
        if HAS_ALIBI:
            scores -= slope[:, None] * tl.abs(pos - cols[None, :])
        mask = valid_q[:, None] & (cols[None, :] < klen)
        if CAUSAL:
            mask &= cols[None, :] <= pos
        if WL >= 0:
            mask &= cols[None, :] >= pos - WL
        if WR >= 0:
            mask &= cols[None, :] <= pos + WR
        scores = tl.where(mask, scores * 1.4426950408889634, -float("inf"))
        new_m = tl.maximum(m, tl.max(scores, 1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        alpha = tl.exp2(m - safe_m)
        p = tl.exp2(scores - safe_m[:, None])
        row_sum = row_sum * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        v = tl.load(
            V + vo[:, None] + kh * VH + dims[None, :],
            (cols[:, None] < klen) & (dims[None, :] < D),
            0,
        )
        acc = tl.dot(p.to(v.dtype), v, acc)
        m = new_m
    result = acc / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
    lse = tl.where(
        row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
    )
    tl.store(
        Out
        + (qstart + qi[:, None]).to(tl.int64) * OS
        + qh[:, None] * OH
        + dims[None, :],
        result,
        valid_q[:, None] & (dims[None, :] < D),
    )
    tl.store(LSE + qh * TQ + qstart + qi, lse, valid_q)


def launch_direct(
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
):
    paged = table is not None
    packed = plan.family == "packed_decode"
    hkv = k.shape[-2]
    grid = lambda meta: (
        1 if packed else triton.cdiv(maxq, meta["BM"]),
        cuq.numel() - 1,
        hkv if packed else q.shape[1],
    )
    return direct_attention_kernel[grid](
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
        BD=max(64, triton.next_power_of_2(q.shape[2])),
        MAX_Q=maxq,
        KV_CAPACITY=table.shape[1] * k.shape[1] if paged else k.shape[0],
        BATCH_COUNT=cuq.numel() - 1,
    )


@triton.jit
def step(
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
    new_m = tl.maximum(m, tl.max(scores, 1))
    safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
    alpha = tl.exp2(m - safe_m)
    p = tl.exp2(scores - safe_m[:, None])
    row_sum = row_sum * alpha + tl.sum(p, 1)
    acc *= alpha[:, None]
    if FULL:
        v = tl.load(V + vo[:, None] + kh * VH + ds[None, :])
    else:
        v = tl.load(V + vo[:, None] + kh * VH + ds[None, :], cols[:, None] < klen, 0)
    acc = tl.dot(p.to(v.dtype), v, acc)
    return (acc, new_m, row_sum)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_tiles"),
    key=[
        "TQ",
        "HQ",
        "HK",
        "D",
        "QS",
        "QH",
        "KP",
        "KS",
        "KH",
        "VP",
        "VS",
        "VH",
        "OS",
        "OH",
        "TS",
        "AS",
        "PAGE",
        "CAUSAL",
        "WL",
        "WR",
        "SOFTCAP",
        "PAGED",
        "HAS_ALIBI",
        "PACKED",
        "BD",
        "REVERSE",
        "SHORT_LIMIT",
        "ORDER",
        "SINGLE",
        "MAX_Q",
        "KV_CAPACITY",
        "BATCH_COUNT",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
    prune_configs_by={"early_config_prune": _prune_attention_tiles},
)
@triton.jit
def paged_attention_kernel(
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
    SINGLE: tl.constexpr = False,
    MAX_Q: tl.constexpr = 0,
    KV_CAPACITY: tl.constexpr = 0,
    BATCH_COUNT: tl.constexpr = 0,
):
    tl.static_assert(PAGED and (PAGE == 16 or PAGE == 32 or PAGE == 64) and (D == BD))
    tl.static_assert(
        not PACKED and (not HAS_ALIBI) and (SOFTCAP == 0) and (WL == -1) and (WR == -1)
    )
    tile, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    if SINGLE:
        batch = 0
    if ORDER == 1:
        linear = head * (tl.cdiv(TQ, BM) if SINGLE else tl.num_programs(0)) + tile
        head = linear % HQ
        tile = linear // HQ
    elif ORDER:
        total_tiles = tl.num_programs(0)
        linear = head * total_tiles + tile
        group = linear // (ORDER * HQ)
        first_tile = group * ORDER
        group_size = tl.minimum(total_tiles - first_tile, ORDER)
        head = linear % (ORDER * HQ) // group_size
        tile = first_tile + linear % group_size
    if SINGLE:
        qstart = 0
        qlen = TQ
    else:
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
    klen = tl.load(USEDK + batch)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    if tile * BM >= qlen:
        return
    if REVERSE:
        tile = tl.cdiv(qlen, BM) - 1 - tile
    qi = tile * BM + tl.arange(0, BM)
    ds = tl.arange(0, BD)
    qrow = qstart + qi[:, None]
    if (TQ - 1) * QS + (HQ - 1) * QH + D >= 2147483648:
        qrow = qrow.to(tl.int64)
    q = tl.load(Q + qrow * QS + head * QH + ds[None, :], qi[:, None] < qlen, 0)
    kh = head // (HQ // HK)
    delta = klen - qlen
    hi = klen
    full_hi = klen // BN * BN
    if CAUSAL:
        hi = tl.minimum(klen, (tile + 1) * BM + delta)
        full_hi = tl.minimum(klen, tile * BM + delta + 1) // BN * BN
    full_hi = tl.maximum(full_hi, 0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    for start in range(0, full_hi, BN):
        acc, m, row_sum = step(
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
        )
    for start in range(full_hi, hi, BN):
        acc, m, row_sum = step(
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
        )
    if BN == 128:
        inv = 1.0 / tl.where(row_sum > 0, row_sum, 1.0)
        result = acc * inv[:, None]
    else:
        result = acc / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
    lse = tl.where(
        row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
    )
    orow = qstart + qi[:, None]
    if (TQ - 1) * OS + (HQ - 1) * OH + D >= 2147483648:
        orow = orow.to(tl.int64)
    tl.store(Out + orow * OS + head * OH + ds[None, :], result, qi[:, None] < qlen)
    tl.store(LSE + head * TQ + qstart + qi, lse, qi < qlen)


def paged_launch_paged(
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
    short_limit=0,
    *,
    kernel=paged_attention_kernel,
):
    paged = table is not None
    packed = False
    single = cuq.numel() == 2
    if single:
        maxq = q.shape[0]
    hkv = k.shape[-2]
    grid = lambda meta: (
        1 if packed else triton.cdiv(maxq, meta["BM"]),
        cuq.numel() - 1,
        hkv if packed else q.shape[1],
    )
    return kernel[grid](
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
        SINGLE=single,
        REVERSE=True,
        ORDER=1,
        SHORT_LIMIT=short_limit,
        BD=max(64, triton.next_power_of_2(q.shape[2])),
        MAX_Q=maxq,
        KV_CAPACITY=table.shape[1] * k.shape[1] if paged else k.shape[0],
        BATCH_COUNT=cuq.numel() - 1,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_compute"),
    key=[
        "TQ",
        "HQ",
        "HK",
        "D",
        "QS",
        "QH",
        "KP",
        "KS",
        "KH",
        "VP",
        "VS",
        "VH",
        "OS",
        "OH",
        "TS",
        "AS",
        "PAGE",
        "CAUSAL",
        "WL",
        "WR",
        "SOFTCAP",
        "PAGED",
        "HAS_ALIBI",
        "PACKED",
        "BM",
        "BN",
        "BD",
        "REVERSE",
        "SHORT_LIMIT",
        "ORDER",
        "BATCHES",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def paged_kernel(
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
    BATCHES: tl.constexpr = 1,
):
    tl.static_assert(PAGED and (PAGE == 16 or PAGE == 32 or PAGE == 64) and (D == BD))
    tl.static_assert(
        not PACKED and (not HAS_ALIBI) and (SOFTCAP == 0) and (WL == -1) and (WR == -1)
    )
    QM: tl.constexpr = BM
    tile, head = (tl.program_id(0), tl.program_id(2))
    if ORDER == 1:
        linear = head * tl.num_programs(0) + tile
        head = linear % HQ
        tile = linear // HQ
    batch = 0
    qstart = tl.load(CUQ)
    qend = tl.load(CUQ + 1)
    count = tl.cdiv(qend - qstart, QM)
    while batch < BATCHES - 1 and tile >= count:
        tile -= count
        batch += 1
        qstart = qend
        qend = tl.load(CUQ + batch + 1)
        count = tl.cdiv(qend - qstart, QM)
    if tile >= count:
        return
    qlen = qend - qstart
    klen = tl.load(USEDK + batch)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    if REVERSE:
        tile = tl.cdiv(qlen, QM) - 1 - tile
    qi = tile * BM + tl.arange(0, BM)
    ds = tl.arange(0, BD)
    qrow = qstart + qi[:, None]
    if (TQ - 1) * QS + (HQ - 1) * QH + D >= 2147483648:
        qrow = qrow.to(tl.int64)
    q = tl.load(Q + qrow * QS + head * QH + ds[None, :], qi[:, None] < qlen, 0)
    kh = head // (HQ // HK)
    delta = klen - qlen
    hi = klen
    full_hi = klen // BN * BN
    if CAUSAL:
        hi = tl.minimum(klen, (tile + 1) * BM + delta)
        full_hi = (tile * BM + delta + 1) // BN * BN
    full_hi = tl.maximum(full_hi, 0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    for start in range(0, full_hi, BN):
        acc, m, row_sum = step(
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
        )
    for start in range(full_hi, hi, BN):
        acc, m, row_sum = step(
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
        )
    inv = 1.0 / tl.where(row_sum > 0, row_sum, 1.0)
    result = acc * inv[:, None]
    lse = tl.where(
        row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
    )
    orow = qstart + qi[:, None]
    if (TQ - 1) * OS + (HQ - 1) * OH + D >= 2147483648:
        orow = orow.to(tl.int64)
    tl.store(Out + orow * OS + head * OH + ds[None, :], result, qi[:, None] < qlen)
    tl.store(LSE + head * TQ + qstart + qi, lse, qi < qlen)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_compute"),
    key=[
        "TQ",
        "HQ",
        "HK",
        "D",
        "QS",
        "QH",
        "KP",
        "KS",
        "KH",
        "VP",
        "VS",
        "VH",
        "OS",
        "OH",
        "TS",
        "AS",
        "PAGE",
        "CAUSAL",
        "WL",
        "WR",
        "SOFTCAP",
        "PAGED",
        "HAS_ALIBI",
        "PACKED",
        "BM",
        "BN",
        "BD",
        "REVERSE",
        "SHORT_LIMIT",
        "ORDER",
        "BATCHES",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def folded_kernel(
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
    BATCHES: tl.constexpr = 1,
):
    tl.static_assert(PAGED and (PAGE == 16 or PAGE == 32 or PAGE == 64) and (D == BD))
    tl.static_assert(
        not PACKED and (not HAS_ALIBI) and (SOFTCAP == 0) and (WL == -1) and (WR == -1)
    )
    GROUP: tl.constexpr = HQ // HK
    QM: tl.constexpr = BM // GROUP
    tile, head = (tl.program_id(0), tl.program_id(2))
    if ORDER == 1:
        linear = head * tl.num_programs(0) + tile
        head = linear % HK
        tile = linear // HK
    batch = 0
    qstart = tl.load(CUQ)
    qend = tl.load(CUQ + 1)
    count = tl.cdiv(qend - qstart, QM)
    while batch < BATCHES - 1 and tile >= count:
        tile -= count
        batch += 1
        qstart = qend
        qend = tl.load(CUQ + batch + 1)
        count = tl.cdiv(qend - qstart, QM)
    if tile >= count:
        return
    qlen = qend - qstart
    klen = tl.load(USEDK + batch)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    if REVERSE:
        tile = tl.cdiv(qlen, QM) - 1 - tile
    qi = tile * QM + tl.arange(0, BM) // GROUP
    qh = head * GROUP + tl.arange(0, BM) % GROUP
    ds = tl.arange(0, BD)
    qrow = qstart + qi[:, None]
    if (TQ - 1) * QS + (HQ - 1) * QH + D >= 2147483648:
        qrow = qrow.to(tl.int64)
    q = tl.load(Q + qrow * QS + qh[:, None] * QH + ds[None, :], qi[:, None] < qlen, 0)
    kh = head
    delta = klen - qlen
    hi = klen
    full_hi = klen // BN * BN
    if CAUSAL:
        hi = tl.minimum(klen, (tile + 1) * QM + delta)
        full_hi = tl.minimum(klen, tile * QM + delta + 1) // BN * BN
    full_hi = tl.maximum(full_hi, 0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    for start in range(0, full_hi, BN):
        acc, m, row_sum = step(
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
        )
    for start in range(full_hi, hi, BN):
        acc, m, row_sum = step(
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
        )
    result = acc / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
    lse = tl.where(
        row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
    )
    orow = qstart + qi[:, None]
    if (TQ - 1) * OS + (HQ - 1) * OH + D >= 2147483648:
        orow = orow.to(tl.int64)
    tl.store(
        Out + orow * OS + qh[:, None] * OH + ds[None, :], result, qi[:, None] < qlen
    )
    tl.store(LSE + qh * TQ + qstart + qi, lse, qi < qlen)


def launch_compact(
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
    short_limit=0,
):
    paged = table is not None
    packed = False
    hkv = k.shape[-2]
    batches = cuq.numel() - 1
    folded = plan.family == "compact_folded"
    group = q.shape[1] // hkv if folded else 1
    query_tile = plan.block_m // group
    grid = (triton.cdiv(q.shape[0], query_tile) + batches - 1, 1, q.shape[1] // group)
    kernel = folded_kernel if folded else paged_kernel
    return kernel[grid](
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
        REVERSE=not folded,
        ORDER=1,
        SHORT_LIMIT=short_limit,
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_compute"),
    key=[
        "TQ",
        "HQ",
        "HK",
        "D",
        "QS",
        "QH",
        "KP",
        "KS",
        "KH",
        "VP",
        "VS",
        "VH",
        "OS",
        "OH",
        "TS",
        "AS",
        "PAGE",
        "CAUSAL",
        "WL",
        "WR",
        "SOFTCAP",
        "PAGED",
        "HAS_ALIBI",
        "PACKED",
        "BM",
        "BN",
        "BD",
        "REVERSE",
        "SHORT_LIMIT",
        "ORDER",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def folded_attention_kernel(
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
):
    tl.static_assert(PAGED and (PAGE == 16 or PAGE == 32 or PAGE == 64) and (D == BD))
    tl.static_assert(
        not PACKED and (not HAS_ALIBI) and (SOFTCAP == 0) and (WL == -1) and (WR == -1)
    )
    tile, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    GROUP: tl.constexpr = HQ // HK
    QM: tl.constexpr = BM // GROUP
    if ORDER == 1:
        linear = head * tl.num_programs(0) + tile
        head = linear % HK
        tile = linear // HK
    elif ORDER:
        total_tiles = tl.num_programs(0)
        linear = head * total_tiles + tile
        group = linear // (ORDER * HK)
        first_tile = group * ORDER
        group_size = tl.minimum(total_tiles - first_tile, ORDER)
        head = linear % (ORDER * HK) // group_size
        tile = first_tile + linear % group_size
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    klen = tl.load(USEDK + batch)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    if REVERSE:
        tile = tl.num_programs(0) - 1 - tile
    if tile * QM >= qlen:
        return
    qi = tile * QM + tl.arange(0, BM) // GROUP
    qh = head * GROUP + tl.arange(0, BM) % GROUP
    ds = tl.arange(0, BD)
    qrow = qstart + qi[:, None]
    if (TQ - 1) * QS + (HQ - 1) * QH + D >= 2147483648:
        qrow = qrow.to(tl.int64)
    q = tl.load(Q + qrow * QS + qh[:, None] * QH + ds[None, :], qi[:, None] < qlen, 0)
    kh = head
    delta = klen - qlen
    hi = klen
    full_hi = klen // BN * BN
    if CAUSAL:
        hi = tl.minimum(klen, (tile + 1) * QM + delta)
        full_hi = tl.minimum(klen, tile * QM + delta + 1) // BN * BN
    full_hi = tl.maximum(full_hi, 0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    for start in range(0, full_hi, BN):
        acc, m, row_sum = step(
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
        )
    for start in range(full_hi, hi, BN):
        acc, m, row_sum = step(
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
        )
    result = acc / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
    lse = tl.where(
        row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
    )
    orow = qstart + qi[:, None]
    if (TQ - 1) * OS + (HQ - 1) * OH + D >= 2147483648:
        orow = orow.to(tl.int64)
    tl.store(
        Out + orow * OS + qh[:, None] * OH + ds[None, :], result, qi[:, None] < qlen
    )
    tl.store(LSE + qh * TQ + qstart + qi, lse, qi < qlen)


def launch_folded(
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
    short_limit=0,
):
    paged = table is not None
    packed = False
    single = cuq.numel() == 2
    if single:
        maxq = q.shape[0]
    hkv = k.shape[-2]
    grid = (triton.cdiv(maxq, plan.block_m) * (q.shape[1] // hkv), cuq.numel() - 1, hkv)
    return folded_attention_kernel[grid](
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
        REVERSE=False,
        ORDER=1,
        SHORT_LIMIT=short_limit,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_compute"),
    key=[
        "TQ",
        "HQ",
        "GROUPS",
        "QS",
        "QH",
        "KP",
        "KH",
        "VP",
        "VH",
        "OS",
        "OH",
        "TS",
        "WRITE_LSE",
        "BH",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def single_key_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    CUQ,
    USEDK,
    TABLE,
    TQ: tl.constexpr,
    HQ: tl.constexpr,
    GROUPS: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KP: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VH: tl.constexpr,
    OS: tl.constexpr,
    OH: tl.constexpr,
    TS: tl.constexpr,
    SCALE: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    BH: tl.constexpr,
):
    batch = tl.program_id(0)
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    dims = tl.arange(0, 256)
    start = tl.load(CUQ + batch)
    end = tl.load(CUQ + batch + 1)
    valid = (start < end) & (start < TQ) & (heads < HQ)
    has_key = tl.load(USEDK + batch) > 0
    page = tl.load(TABLE + batch * TS, (start < end) & has_key, 0).to(tl.int64)
    kh = (heads // GROUPS).to(tl.int64)
    q = tl.load(
        Q + start.to(tl.int64) * QS + heads[:, None].to(tl.int64) * QH + dims[None, :],
        valid[:, None],
        0,
    ).to(tl.float32)
    k = tl.load(
        K + page * KP + kh[:, None] * KH + dims[None, :], valid[:, None] & has_key, 0
    ).to(tl.float32)
    score = tl.sum(q * k, 1) * SCALE
    finite = (score == score) & (tl.abs(score) != float("inf"))
    value = tl.load(
        V + page * VP + kh[:, None] * VH + dims[None, :], valid[:, None] & has_key, 0
    )
    result = tl.where(finite[:, None], value, float("nan"))
    result = tl.where(
        score[:, None] == -float("inf"), value.to(tl.float32) * 0.0, result
    )
    result = tl.where(has_key, result, 0.0)
    tl.store(
        Out
        + start.to(tl.int64) * OS
        + heads[:, None].to(tl.int64) * OH
        + dims[None, :],
        result,
        valid[:, None],
    )
    if WRITE_LSE:
        lse = tl.where(has_key & finite, score, float("inf"))
        tl.store(LSE + heads.to(tl.int64) * TQ + start, lse, valid)


def launch_single_key(
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
    *,
    write_lse,
):
    return single_key_kernel[cuq.numel() - 1, triton.cdiv(q.shape[1], plan.block_m), 1](
        q,
        k,
        v,
        out,
        lse,
        cuq,
        usedk,
        table,
        q.shape[0],
        q.shape[1],
        q.shape[1] // k.shape[-2],
        *q.stride()[:2],
        k.stride(0),
        k.stride(-2),
        v.stride(0),
        v.stride(-2),
        *out.stride()[:2],
        table.stride(0),
        scale,
        write_lse,
        plan.block_m,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_compute"),
    key=[
        "TQ",
        "HQ",
        "HK",
        "D",
        "QS",
        "QH",
        "KP",
        "KS",
        "KH",
        "VP",
        "VS",
        "VH",
        "OS",
        "OH",
        "TS",
        "AS",
        "PAGE",
        "CAUSAL",
        "WL",
        "WR",
        "SOFTCAP",
        "PAGED",
        "HAS_ALIBI",
        "PACKED",
        "BM",
        "BN",
        "BD",
        "REVERSE",
        "SHORT_LIMIT",
        "ORDER",
        "WRITE_LSE",
        "SINGLE",
        "FOLD",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def single_prefill_attention_kernel(
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
    SINGLE: tl.constexpr = False,
    FOLD: tl.constexpr = 8,
):
    tl.static_assert(PAGED and (PAGE == 16 or PAGE == 32 or PAGE == 64) and (D == BD))
    tl.static_assert(
        not PACKED and (not HAS_ALIBI) and (SOFTCAP == 0) and (WL == -1) and (WR == -1)
    )
    tile, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    tl.static_assert(HQ // HK % FOLD == 0)
    HE: tl.constexpr = HQ // FOLD
    GROUP: tl.constexpr = FOLD
    QM: tl.constexpr = BM // GROUP
    if ORDER == 1:
        linear = head * tl.num_programs(0) + tile
        head = linear % HE
        tile = linear // HE
    elif ORDER:
        total_tiles = tl.num_programs(0)
        linear = head * total_tiles + tile
        group = linear // (ORDER * HE)
        first_tile = group * ORDER
        group_size = tl.minimum(total_tiles - first_tile, ORDER)
        head = linear % (ORDER * HE) // group_size
        tile = first_tile + linear % group_size
    if SINGLE:
        qstart = 0
        qlen = TQ
    else:
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
    klen = tl.load(USEDK + batch)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    if REVERSE:
        tile = tl.num_programs(0) - 1 - tile
    if tile * QM >= qlen:
        return
    qi = tile * QM + tl.arange(0, BM) // GROUP
    qh = head * GROUP + tl.arange(0, BM) % GROUP
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
        full_hi = (tile * QM + delta + 1) // BN * BN
    full_hi = tl.maximum(full_hi, 0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    for start in range(0, full_hi, BN):
        acc, m, row_sum = step(
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
        )
    if hi - full_hi <= 32:
        for start in range(full_hi, hi, 32):
            acc, m, row_sum = step(
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
                32,
                BD,
                CAUSAL,
                False,
            )
    else:
        for start in range(full_hi, hi, BN):
            acc, m, row_sum = step(
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
            )
    inv = 1.0 / tl.where(row_sum > 0, row_sum, 1.0)
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


def launch_single_prefill(
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
    single = cuq.numel() == 2
    if single:
        maxq = q.shape[0]
    hkv = k.shape[-2]
    grid = (triton.cdiv(maxq, plan.block_m) * 2, cuq.numel() - 1, q.shape[1] // 2)
    return single_prefill_attention_kernel[grid](
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
        SINGLE=single,
        REVERSE=True,
        ORDER=1,
        FOLD=2,
        WRITE_LSE=write_lse,
        SHORT_LIMIT=0,
    )


@triton.jit
def equal_body(
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
    SINGLE: tl.constexpr = False,
    EQUAL: tl.constexpr = False,
):
    tl.static_assert(PAGED and (PAGE == 16 or PAGE == 32 or PAGE == 64) and (D == BD))
    tl.static_assert(
        not PACKED and (not HAS_ALIBI) and (SOFTCAP == 0) and (WL == -1) and (WR == -1)
    )
    tile, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    if SINGLE:
        batch = 0
    if ORDER == 1:
        linear = head * (tl.cdiv(TQ, BM) if SINGLE else tl.num_programs(0)) + tile
        head = linear % HQ
        tile = linear // HQ
    elif ORDER:
        total_tiles = tl.num_programs(0)
        linear = head * total_tiles + tile
        group = linear // (ORDER * HQ)
        first_tile = group * ORDER
        group_size = tl.minimum(total_tiles - first_tile, ORDER)
        head = linear % (ORDER * HQ) // group_size
        tile = first_tile + linear % group_size
    if SINGLE:
        qstart = 0
        qlen = TQ
    else:
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
    klen = qlen if EQUAL else tl.load(USEDK + batch)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    if tile * BM >= qlen:
        return
    if REVERSE:
        tile = tl.cdiv(qlen, BM) - 1 - tile
    qi = tile * BM + tl.arange(0, BM)
    ds = tl.arange(0, BD)
    qrow = qstart + qi[:, None]
    if (TQ - 1) * QS + (HQ - 1) * QH + D >= 2147483648:
        qrow = qrow.to(tl.int64)
    q = tl.load(Q + qrow * QS + head * QH + ds[None, :], qi[:, None] < qlen, 0)
    kh = head // (HQ // HK)
    delta = klen - qlen
    hi = klen
    full_hi = klen // BN * BN
    if CAUSAL:
        hi = tl.minimum(klen, (tile + 1) * BM + delta)
        full_hi = tl.minimum(klen, tile * BM + delta + 1) // BN * BN
    full_hi = tl.maximum(full_hi, 0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    for start in range(0, full_hi, BN):
        acc, m, row_sum = step(
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
        )
    for start in range(full_hi, hi, BN):
        acc, m, row_sum = step(
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
        )
    result = acc / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
    lse = tl.where(
        row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
    )
    orow = qstart + qi[:, None]
    if (TQ - 1) * OS + (HQ - 1) * OH + D >= 2147483648:
        orow = orow.to(tl.int64)
    tl.store(Out + orow * OS + head * OH + ds[None, :], result, qi[:, None] < qlen)
    tl.store(LSE + head * TQ + qstart + qi, lse, qi < qlen)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_tiles"),
    key=[
        "TQ",
        "HQ",
        "HK",
        "D",
        "QS",
        "QH",
        "KP",
        "KS",
        "KH",
        "VP",
        "VS",
        "VH",
        "OS",
        "OH",
        "TS",
        "AS",
        "PAGE",
        "CAUSAL",
        "WL",
        "WR",
        "SOFTCAP",
        "PAGED",
        "HAS_ALIBI",
        "PACKED",
        "BD",
        "REVERSE",
        "SHORT_LIMIT",
        "ORDER",
        "SINGLE",
        "MAX_Q",
        "KV_CAPACITY",
        "BATCH_COUNT",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
    prune_configs_by={"early_config_prune": _prune_attention_tiles},
)
@triton.jit
def equal_length_attention_kernel(
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
    SINGLE: tl.constexpr = False,
    MAX_Q: tl.constexpr = 0,
    KV_CAPACITY: tl.constexpr = 0,
    BATCH_COUNT: tl.constexpr = 0,
):
    b = tl.program_id(1)
    nq = TQ if SINGLE else tl.load(CUQ + b + 1) - tl.load(CUQ + b)
    nk = tl.load(USEDK + b)
    if nq == nk:
        equal_body(
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
            TQ,
            HQ,
            HK,
            D,
            QS,
            QH,
            KP,
            KS,
            KH,
            VP,
            VS,
            VH,
            OS,
            OH,
            TS,
            AS,
            PAGE,
            SCALE,
            CAUSAL,
            WL,
            WR,
            SOFTCAP,
            PAGED,
            HAS_ALIBI,
            PACKED,
            BM,
            BN,
            BD,
            REVERSE,
            SHORT_LIMIT,
            ORDER,
            SINGLE,
            EQUAL=True,
        )
    else:
        equal_body(
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
            TQ,
            HQ,
            HK,
            D,
            QS,
            QH,
            KP,
            KS,
            KH,
            VP,
            VS,
            VH,
            OS,
            OH,
            TS,
            AS,
            PAGE,
            SCALE,
            CAUSAL,
            WL,
            WR,
            SOFTCAP,
            PAGED,
            HAS_ALIBI,
            PACKED,
            BM,
            BN,
            BD,
            REVERSE,
            SHORT_LIMIT,
            ORDER,
            SINGLE,
            EQUAL=False,
        )


launch_equal_length = partial(paged_launch_paged, kernel=equal_length_attention_kernel)
equal_length_launch_paged = launch_equal_length
