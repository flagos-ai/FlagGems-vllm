# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_decode"),
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
        "SPLITS",
        "PT",
        "PART_Q",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def splitkv_split_attention_kernel(
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
    SPLITS: tl.constexpr,
    PT: tl.constexpr,
    PART_Q: tl.constexpr,
):
    tile, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    split = tile % SPLITS
    tile = tile // SPLITS
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    if PAGED:
        kstart = 0
        klen = tl.load(USEDK + batch)
    else:
        kstart = tl.load(CUK + batch)
        klen = tl.load(CUK + batch + 1) - kstart
    groups: tl.constexpr = HQ // HK
    if PART_Q and (qlen > PART_Q or klen < 8192):
        return
    rows = tl.arange(0, BM)
    if PACKED:
        qi = tile * (BM // groups) + rows // groups
        qh = head * groups + rows % groups
        kh = head
        valid_q = qi < qlen
        row_base = tile * (BM // groups)
        row_end = tl.minimum((tile + 1) * (BM // groups), qlen)
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
    chunk = tl.cdiv(tl.cdiv(klen, BN), SPLITS) * BN
    low = tl.maximum(low, split * chunk)
    high = tl.minimum(high, (split + 1) * chunk)
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
    result = acc
    tl.store(
        Out
        + split.to(tl.int64) * PT * HQ * D
        + ((batch * PART_Q if PART_Q else qstart) + qi[:, None]).to(tl.int64) * OS
        + qh[:, None] * OH
        + dims[None, :],
        result,
        valid_q[:, None] & (dims[None, :] < D),
    )
    tl.store(
        LSE
        + split * 2 * HQ * PT
        + qh * PT
        + (batch * PART_Q if PART_Q else qstart)
        + qi,
        tl.where(row_sum > 0, m, -float("inf")),
        valid_q,
    )
    tl.store(
        LSE
        + (split * 2 + 1) * HQ * PT
        + qh * PT
        + (batch * PART_Q if PART_Q else qstart)
        + qi,
        row_sum,
        valid_q,
    )


def launch_split(
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
    splits=16,
    packed=True,
    short_limit=0,
):
    paged = table is not None
    pt = short_limit * (cuq.numel() - 1) if short_limit else q.shape[0]
    partial = torch.empty(
        (splits, pt, q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    partial_lse = torch.empty(
        (splits, 2, q.shape[1], pt), device=q.device, dtype=torch.float32
    )
    hkv = k.shape[-2]
    grid = (
        splits
        * triton.cdiv(
            short_limit or maxq,
            plan.block_m // (q.shape[1] // hkv) if packed else plan.block_m,
        ),
        cuq.numel() - 1,
        hkv if packed else q.shape[1],
    )
    splitkv_split_attention_kernel[grid](
        q,
        k,
        v,
        partial,
        partial_lse,
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
        *partial.stride()[1:3],
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
        splits,
        pt,
        short_limit,
    )
    splitkv_merge_kernel[
        short_limit or q.shape[0], q.shape[1], cuq.numel() - 1 if short_limit else 1
    ](
        partial,
        partial_lse,
        out,
        lse,
        cuq,
        usedk,
        q.shape[0],
        pt,
        q.shape[1],
        q.shape[2],
        *out.stride()[:2],
        splits,
        triton.next_power_of_2(splits),
        triton.next_power_of_2(q.shape[2]),
        num_warps=4,
        PART_Q=short_limit,
    )


@triton.jit
def splitkv_merge_kernel(
    P,
    PL,
    Out,
    LSE,
    CUQ,
    USED,
    T: tl.constexpr,
    PT: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    OS: tl.constexpr,
    OH: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    PART_Q: tl.constexpr,
):
    token, head, batch = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    if PART_Q:
        qb = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qb
        klen = tl.load(USED + batch)
        if token >= qlen or qlen > PART_Q or klen < 8192:
            return
        ptok = batch * PART_Q + token
    else:
        qb = 0
        ptok = token
    ss, ds = (tl.arange(0, BS), tl.arange(0, BD))
    ls = tl.load(PL + ss * 2 * H * PT + head * PT + ptok, ss < S, -float("inf"))
    sums = tl.load(PL + (ss * 2 + 1) * H * PT + head * PT + ptok, ss < S, 0)
    maximum = tl.max(ls, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weight = tl.exp2(ls - maximum)
    denom = tl.sum(weight * sums, 0)
    values = tl.load(
        P
        + ss[:, None].to(tl.int64) * PT * H * D
        + ptok * H * D
        + head * D
        + ds[None, :],
        (ss[:, None] < S) & (ds[None, :] < D),
        0,
    )
    result = tl.sum(values * weight[:, None], 0) / tl.where(denom > 0, denom, 1.0)
    tl.store(Out + (qb + token).to(tl.int64) * OS + head * OH + ds, result, ds < D)
    tl.store(
        LSE + head * T + qb + token,
        tl.where(denom > 0, maximum * 0.6931471805599453 + tl.log(denom), float("inf")),
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_decode"),
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
        "SPLITS",
        "PT",
        "PART_Q",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def q1_decode_split_attention_kernel(
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
    SPLITS: tl.constexpr,
    PT: tl.constexpr,
    PART_Q: tl.constexpr,
):
    tile, batch, head = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    split = tile % SPLITS
    tile = tile // SPLITS
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    if PAGED:
        kstart = 0
        klen = tl.load(USEDK + batch)
    else:
        kstart = tl.load(CUK + batch)
        klen = tl.load(CUK + batch + 1) - kstart
    groups: tl.constexpr = HQ // HK
    if PART_Q and qlen > PART_Q:
        return
    rows = tl.arange(0, BM)
    if PACKED:
        qi = tile * (BM // groups) + rows // groups
        qh = head * groups + rows % groups
        kh = head
        valid_q = qi < qlen
        row_base = tile * (BM // groups)
        row_end = tl.minimum((tile + 1) * (BM // groups), qlen)
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
    chunk = tl.cdiv(tl.cdiv(klen, BN), SPLITS) * BN
    low = tl.maximum(low, split * chunk)
    high = tl.minimum(high, (split + 1) * chunk)
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
    result = acc
    tl.store(
        Out
        + split.to(tl.int64) * PT * HQ * D
        + ((batch * PART_Q if PART_Q else qstart) + qi[:, None]).to(tl.int64) * OS
        + qh[:, None] * OH
        + dims[None, :],
        result,
        valid_q[:, None] & (dims[None, :] < D),
    )
    tl.store(
        LSE
        + split * 2 * HQ * PT
        + qh * PT
        + (batch * PART_Q if PART_Q else qstart)
        + qi,
        tl.where(row_sum > 0, m, -float("inf")),
        valid_q,
    )
    tl.store(
        LSE
        + (split * 2 + 1) * HQ * PT
        + qh * PT
        + (batch * PART_Q if PART_Q else qstart)
        + qi,
        row_sum,
        valid_q,
    )


@triton.jit
def q1_decode_merge_kernel(
    P,
    PL,
    Out,
    LSE,
    CUQ,
    USED,
    T: tl.constexpr,
    PT: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    OS: tl.constexpr,
    OH: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    PART_Q: tl.constexpr,
    WRITE_LSE: tl.constexpr = True,
):
    token, head, batch = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    if PART_Q:
        qb = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qb
        if token >= qlen or qlen > PART_Q:
            return
        ptok = batch * PART_Q + token
    else:
        qb = 0
        ptok = token
    ss, ds = (tl.arange(0, BS), tl.arange(0, BD))
    ls = tl.load(PL + ss * 2 * H * PT + head * PT + ptok, ss < S, -float("inf"))
    sums = tl.load(PL + (ss * 2 + 1) * H * PT + head * PT + ptok, ss < S, 0)
    maximum = tl.max(ls, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weight = tl.exp2(ls - maximum)
    denom = tl.sum(weight * sums, 0)
    values = tl.load(
        P
        + ss[:, None].to(tl.int64) * PT * H * D
        + ptok * H * D
        + head * D
        + ds[None, :],
        (ss[:, None] < S) & (ds[None, :] < D),
        0,
    )
    result = tl.sum(values * weight[:, None], 0) / tl.where(denom > 0, denom, 1.0)
    tl.store(Out + (qb + token).to(tl.int64) * OS + head * OH + ds, result, ds < D)
    if WRITE_LSE:
        tl.store(
            LSE + head * T + qb + token,
            tl.where(
                denom > 0, maximum * 0.6931471805599453 + tl.log(denom), float("inf")
            ),
        )


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
    KP: tl.constexpr,
    KS: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    TS: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    DK: tl.constexpr,
    VD: tl.constexpr,
    CACHE: tl.constexpr,
    FULL: tl.constexpr,
):
    ns = tl.arange(0, BN)
    ps = tl.arange(0, BN // 32)
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
    kd = tl.arange(0, DK)
    if FULL:
        k0 = tl.load(K + kd[:, None] + ko[None, :], cache_modifier=CACHE)
    else:
        k0 = tl.load(
            K + kd[:, None] + ko[None, :],
            start + ns[None, :] < length,
            0,
            cache_modifier=CACHE,
        )
    scores = tl.dot(q0, k0)
    if DK == 128:
        if FULL:
            k1 = tl.load(K + 128 + kd[:, None] + ko[None, :], cache_modifier=CACHE)
        else:
            k1 = tl.load(
                K + 128 + kd[:, None] + ko[None, :],
                start + ns[None, :] < length,
                0,
                cache_modifier=CACHE,
            )
        scores = tl.dot(q1, k1, scores)
    scores *= SCALE * 1.4426950408889634
    if not FULL:
        scores = tl.where(start + ns[None, :] < length, scores, -float("inf"))
    new_max = tl.maximum(maximum, tl.max(scores, 1))
    safe = tl.where(new_max == -float("inf"), 0.0, new_max)
    alpha = tl.exp2(maximum - safe)
    prob = tl.exp2(scores - safe[:, None])
    denom = denom * alpha + tl.sum(prob, 1)
    a0 *= alpha[:, None]
    if VD == 128:
        a1 *= alpha[:, None]
    p = prob.to(V.dtype.element_ty)
    vd = tl.arange(0, VD)
    if FULL:
        v0 = tl.load(V + vo[:, None] + vd[None, :], cache_modifier=CACHE)
    else:
        v0 = tl.load(
            V + vo[:, None] + vd[None, :],
            start + ns[:, None] < length,
            0,
            cache_modifier=CACHE,
        )
    a0 = tl.dot(p, v0, a0)
    if VD == 128:
        if FULL:
            v1 = tl.load(V + vo[:, None] + 128 + vd[None, :], cache_modifier=CACHE)
        else:
            v1 = tl.load(
                V + vo[:, None] + 128 + vd[None, :],
                start + ns[:, None] < length,
                0,
                cache_modifier=CACHE,
            )
        a1 = tl.dot(p, v1, a1)
    return (a0, a1, new_max, denom)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_decode"),
    key=[
        "B",
        "QS",
        "QH",
        "KP",
        "KS",
        "VP",
        "VS",
        "TS",
        "S",
        "BN",
        "CHUNK",
        "DK",
        "VD",
        "BATCH_FIRST",
        "CACHE",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def tp2_decode_producer(
    Q,
    K,
    V,
    CUQ,
    USED,
    TABLE,
    P,
    PL,
    B: tl.constexpr,
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
    CHUNK: tl.constexpr,
    DK: tl.constexpr,
    VD: tl.constexpr,
    BATCH_FIRST: tl.constexpr,
    CACHE: tl.constexpr,
):
    linear = tl.program_id(0)
    if BATCH_FIRST:
        batch = linear % B
        split = linear // B
    else:
        batch = linear // S
        split = linear % S
    qb = tl.load(CUQ + batch)
    qe = tl.load(CUQ + batch + 1)
    if qe == qb:
        return
    length = tl.load(USED + batch)
    hs = tl.arange(0, 16)
    kd = tl.arange(0, DK)
    q0 = tl.load(
        Q + qb.to(tl.int64) * QS + hs[:, None] * QH + kd[None, :], hs[:, None] < 8, 0
    )
    q1 = tl.zeros((16, DK), q0.dtype)
    if DK == 128:
        q1 = tl.load(
            Q + qb.to(tl.int64) * QS + hs[:, None] * QH + 128 + kd[None, :],
            hs[:, None] < 8,
            0,
        )
    a0 = tl.zeros((16, VD), tl.float32)
    a1 = tl.zeros((16, VD), tl.float32)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denom = tl.zeros((16,), tl.float32)
    low = split * CHUNK
    high = tl.minimum(low + CHUNK, length)
    full = high // BN * BN
    if low + CHUNK <= length:
        for offset in range(0, CHUNK, BN):
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
                low + offset,
                length,
                batch,
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BN,
                DK,
                VD,
                CACHE,
                True,
            )
    else:
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
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BN,
                DK,
                VD,
                CACHE,
                True,
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
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BN,
                DK,
                VD,
                CACHE,
                False,
            )
    vd = tl.arange(0, VD)
    off = (
        split.to(tl.int64) * B * 8 * 256
        + batch * 8 * 256
        + hs[:, None] * 256
        + vd[None, :]
    )
    tl.store(P + off, a0, hs[:, None] < 8)
    if VD == 128:
        tl.store(P + off + 128, a1, hs[:, None] < 8)
    base = split * 2 * 8 * B + hs * B + batch
    tl.store(PL + base, maximum, hs < 8)
    tl.store(PL + base + 8 * B, denom, hs < 8)


@triton.jit
def tp2_decode_merge(
    P,
    PL,
    output_ptr,
    LSE,
    CUQ,
    B: tl.constexpr,
    T: tl.constexpr,
    OS: tl.constexpr,
    OH: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    qb = tl.load(CUQ + batch)
    qe = tl.load(CUQ + batch + 1)
    if qe == qb:
        return
    ss = tl.arange(0, BS)
    ds = tile * BD + tl.arange(0, BD)
    maxima = tl.load(PL + ss * 2 * 8 * B + head * B + batch, ss < S, -float("inf"))
    sums = tl.load(PL + (ss * 2 + 1) * 8 * B + head * B + batch, ss < S, 0.0)
    maximum = tl.max(maxima, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp2(maxima - maximum)
    denominator = tl.sum(weights * sums, 0)
    values = tl.load(
        P
        + ss[:, None].to(tl.int64) * B * 8 * 256
        + batch * 8 * 256
        + head * 256
        + ds[None, :],
        (ss[:, None] < S) & (ds[None, :] < 256),
        0.0,
    )
    output = tl.sum(values * weights[:, None], 0) / tl.where(
        denominator > 0, denominator, 1.0
    )
    tl.store(output_ptr + qb.to(tl.int64) * OS + head * OH + ds, output, ds < 256)
    if WRITE_LSE:
        if tile == 0:
            tl.store(
                LSE + head * T + qb,
                tl.where(
                    denominator > 0,
                    maximum * 0.6931471805599453 + tl.log(denominator),
                    float("inf"),
                ),
            )


@triton.jit
def _balanced_chunk(
    q0,
    q1,
    K,
    V,
    TABLE,
    a0,
    a1,
    maximum,
    denom,
    low,
    length,
    batch,
    KP: tl.constexpr,
    KS: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    TS: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    DK: tl.constexpr,
    VD: tl.constexpr,
    CACHE: tl.constexpr,
    CHUNK: tl.constexpr,
):
    high = tl.minimum(low + CHUNK, length)
    full = high // BN * BN
    if low + CHUNK <= length:
        for offset in range(0, CHUNK, BN):
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
                low + offset,
                length,
                batch,
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BN,
                DK,
                VD,
                CACHE,
                True,
            )
    else:
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
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BN,
                DK,
                VD,
                CACHE,
                True,
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
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BN,
                DK,
                VD,
                CACHE,
                False,
            )
    return (a0, a1, maximum, denom)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_decode"),
    key=[
        "B",
        "QS",
        "QH",
        "KP",
        "KS",
        "VP",
        "VS",
        "TS",
        "S",
        "BN",
        "CHUNK",
        "DK",
        "VD",
        "BATCH_FIRST",
        "CACHE",
        "BALANCED",
        "MAX_TILES",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def _balanced_producer(
    Q,
    K,
    V,
    CUQ,
    USED,
    TABLE,
    P,
    PL,
    B: tl.constexpr,
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
    CHUNK: tl.constexpr,
    DK: tl.constexpr,
    VD: tl.constexpr,
    BATCH_FIRST: tl.constexpr,
    CACHE: tl.constexpr,
    BALANCED: tl.constexpr,
    MAX_TILES: tl.constexpr,
):
    linear = tl.program_id(0)
    if BATCH_FIRST:
        batch = linear % B
        split = linear // B
    else:
        batch = linear // S
        split = linear % S
    qb = tl.load(CUQ + batch)
    qe = tl.load(CUQ + batch + 1)
    if qe == qb:
        return
    length = tl.load(USED + batch)
    hs = tl.arange(0, 16)
    kd = tl.arange(0, DK)
    q0 = tl.load(
        Q + qb.to(tl.int64) * QS + hs[:, None] * QH + kd[None, :], hs[:, None] < 8, 0
    )
    q1 = tl.zeros((16, DK), q0.dtype)
    if DK == 128:
        q1 = tl.load(
            Q + qb.to(tl.int64) * QS + hs[:, None] * QH + 128 + kd[None, :],
            hs[:, None] < 8,
            0,
        )
    a0 = tl.zeros((16, VD), tl.float32)
    a1 = tl.zeros((16, VD), tl.float32)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denom = tl.zeros((16,), tl.float32)
    if BALANCED:
        tiles_base: tl.constexpr = MAX_TILES // S
        tiles_extra: tl.constexpr = MAX_TILES % S
        low = (split * tiles_base + tl.minimum(split, tiles_extra)) * BN
        if split < tiles_extra:
            a0, a1, maximum, denom = _balanced_chunk(
                q0,
                q1,
                K,
                V,
                TABLE,
                a0,
                a1,
                maximum,
                denom,
                low,
                length,
                batch,
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BN,
                DK,
                VD,
                CACHE,
                (tiles_base + 1) * BN,
            )
        else:
            a0, a1, maximum, denom = _balanced_chunk(
                q0,
                q1,
                K,
                V,
                TABLE,
                a0,
                a1,
                maximum,
                denom,
                low,
                length,
                batch,
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BN,
                DK,
                VD,
                CACHE,
                tiles_base * BN,
            )
    else:
        low = split * CHUNK
        a0, a1, maximum, denom = _balanced_chunk(
            q0,
            q1,
            K,
            V,
            TABLE,
            a0,
            a1,
            maximum,
            denom,
            low,
            length,
            batch,
            KP,
            KS,
            VP,
            VS,
            TS,
            SCALE,
            BN,
            DK,
            VD,
            CACHE,
            CHUNK,
        )
    vd = tl.arange(0, VD)
    off = (
        split.to(tl.int64) * B * 8 * 256
        + batch * 8 * 256
        + hs[:, None] * 256
        + vd[None, :]
    )
    tl.store(P + off, a0, hs[:, None] < 8)
    if VD == 128:
        tl.store(P + off + 128, a1, hs[:, None] < 8)
    base = split * 2 * 8 * B + hs * B + batch
    tl.store(PL + base, maximum, hs < 8)
    tl.store(PL + base + 8 * B, denom, hs < 8)


def tp2_decode_launch(
    q, k, v, out, lse, cuq, used, table, *, max_seqlen_k, scale=0.0625, write_lse=True
):
    batch = cuq.numel() - 1
    if batch == 1:
        dk, vd, batch_first, cache = (128, 256, False, "")
    elif batch in (5, 7, 8):
        dk, vd, batch_first, cache = (128, 128, True, ".cg")
    else:
        dk, vd, batch_first, cache = (256, 256, True, ".cg")
    splits, bn = (64, 64)
    partial = torch.empty((splits, batch, 8, 256), device=q.device, dtype=torch.float32)
    stats = torch.empty((splits, 2, 8, batch), device=q.device, dtype=torch.float32)
    producer = _balanced_producer if batch == 7 else tp2_decode_producer
    extra = (
        dict(BALANCED=True, MAX_TILES=triton.cdiv(max_seqlen_k, bn))
        if batch == 7
        else {}
    )
    producer[splits * batch,](
        q,
        k,
        v,
        cuq,
        used,
        table,
        partial,
        stats,
        batch,
        *q.stride()[:2],
        *k.stride()[:2],
        *v.stride()[:2],
        table.stride(0),
        scale,
        splits,
        bn,
        triton.cdiv(triton.cdiv(max_seqlen_k, bn), splits) * bn,
        dk,
        vd,
        batch_first,
        cache,
        **extra,
    )
    tp2_decode_merge[4, 8, batch](
        partial,
        stats,
        out,
        lse,
        cuq,
        batch,
        q.shape[0],
        *out.stride()[:2],
        splits,
        triton.next_power_of_2(splits),
        64,
        write_lse,
        num_warps=4,
        num_stages=1,
    )


def launch_tp2_decode(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    used,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    *,
    max_seqlen_k,
    write_lse=True,
):
    tp2_decode_launch(
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_attention_decode"),
    key=[
        "B",
        "QS",
        "QH",
        "KP",
        "KS",
        "VP",
        "VS",
        "TS",
        "S",
        "BN",
        "CHUNK",
        "DK",
        "VD",
        "BATCH_FIRST",
        "CACHE",
    ],
    strategy="default",
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def tp2_mixed_q1_producer(
    Q,
    K,
    V,
    CUQ,
    USED,
    TABLE,
    P,
    PL,
    B: tl.constexpr,
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
    CHUNK: tl.constexpr,
    DK: tl.constexpr,
    VD: tl.constexpr,
    BATCH_FIRST: tl.constexpr,
    CACHE: tl.constexpr,
):
    linear = tl.program_id(0)
    if BATCH_FIRST:
        batch = linear % B
        split = linear // B
    else:
        batch = linear // S
        split = linear % S
    qb = tl.load(CUQ + batch)
    qe = tl.load(CUQ + batch + 1)
    if qe - qb != 1:
        return
    length = tl.load(USED + batch)
    hs = tl.arange(0, 16)
    kd = tl.arange(0, DK)
    q0 = tl.load(
        Q + qb.to(tl.int64) * QS + hs[:, None] * QH + kd[None, :], hs[:, None] < 8, 0
    )
    q1 = tl.zeros((16, DK), q0.dtype)
    if DK == 128:
        q1 = tl.load(
            Q + qb.to(tl.int64) * QS + hs[:, None] * QH + 128 + kd[None, :],
            hs[:, None] < 8,
            0,
        )
    a0 = tl.zeros((16, VD), tl.float32)
    a1 = tl.zeros((16, VD), tl.float32)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denom = tl.zeros((16,), tl.float32)
    low = split * CHUNK
    high = tl.minimum(low + CHUNK, length)
    full = high // BN * BN
    if low + CHUNK <= length:
        for offset in range(0, CHUNK, BN):
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
                low + offset,
                length,
                batch,
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BN,
                DK,
                VD,
                CACHE,
                True,
            )
    else:
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
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BN,
                DK,
                VD,
                CACHE,
                True,
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
                KP,
                KS,
                VP,
                VS,
                TS,
                SCALE,
                BN,
                DK,
                VD,
                CACHE,
                False,
            )
    vd = tl.arange(0, VD)
    off = (
        split.to(tl.int64) * B * 8 * 256
        + batch * 8 * 256
        + hs[:, None] * 256
        + vd[None, :]
    )
    tl.store(P + off, a0, hs[:, None] < 8)
    if VD == 128:
        tl.store(P + off + 128, a1, hs[:, None] < 8)
    base = split * 2 * 8 * B + hs * B + batch
    tl.store(PL + base, maximum, hs < 8)
    tl.store(PL + base + 8 * B, denom, hs < 8)


@triton.jit
def tp2_mixed_q1_merge(
    P,
    PL,
    OUT,
    LSE,
    CUQ,
    B: tl.constexpr,
    T: tl.constexpr,
    OS: tl.constexpr,
    OH: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    qb = tl.load(CUQ + batch)
    qe = tl.load(CUQ + batch + 1)
    if qe - qb != 1:
        return
    ss = tl.arange(0, BS)
    ds = tile * BD + tl.arange(0, BD)
    maxima = tl.load(PL + ss * 2 * 8 * B + head * B + batch, ss < S, -float("inf"))
    sums = tl.load(PL + (ss * 2 + 1) * 8 * B + head * B + batch, ss < S, 0.0)
    maximum = tl.max(maxima, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp2(maxima - maximum)
    denominator = tl.sum(weights * sums, 0)
    values = tl.load(
        P
        + ss[:, None].to(tl.int64) * B * 8 * 256
        + batch * 8 * 256
        + head * 256
        + ds[None, :],
        (ss[:, None] < S) & (ds[None, :] < 256),
        0.0,
    )
    output = tl.sum(values * weights[:, None], 0) / tl.where(
        denominator > 0, denominator, 1.0
    )
    tl.store(OUT + qb.to(tl.int64) * OS + head * OH + ds, output, ds < 256)
    if WRITE_LSE:
        if tile == 0:
            tl.store(
                LSE + head * T + qb,
                tl.where(
                    denominator > 0,
                    maximum * 0.6931471805599453 + tl.log(denominator),
                    float("inf"),
                ),
            )


def tp2_mixed_q1_launch(
    q, k, v, out, lse, cuq, used, table, *, max_seqlen_k, scale=0.0625, write_lse=True
):
    batch = cuq.numel() - 1
    if batch == 1:
        dk, vd, batch_first, cache = (128, 256, False, "")
    elif batch in (5, 7, 8):
        dk, vd, batch_first, cache = (128, 128, True, ".cg")
    else:
        dk, vd, batch_first, cache = (256, 256, True, ".cg")
    splits, bn = (64, 64)
    partial = torch.empty((splits, batch, 8, 256), device=q.device, dtype=torch.float32)
    stats = torch.empty((splits, 2, 8, batch), device=q.device, dtype=torch.float32)
    tp2_mixed_q1_producer[splits * batch,](
        q,
        k,
        v,
        cuq,
        used,
        table,
        partial,
        stats,
        batch,
        *q.stride()[:2],
        *k.stride()[:2],
        *v.stride()[:2],
        table.stride(0),
        scale,
        splits,
        bn,
        triton.cdiv(triton.cdiv(max_seqlen_k, bn), splits) * bn,
        dk,
        vd,
        batch_first,
        cache,
    )
    tp2_mixed_q1_merge[4, 8, batch](
        partial,
        stats,
        out,
        lse,
        cuq,
        batch,
        q.shape[0],
        *out.stride()[:2],
        splits,
        64,
        64,
        write_lse,
        num_warps=4,
        num_stages=1,
    )
