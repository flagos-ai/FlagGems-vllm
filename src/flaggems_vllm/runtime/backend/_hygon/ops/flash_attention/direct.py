# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime.backend._hygon.ops.flash_attention.common import (
    ATTENTION_TUNING_KEY,
    kernel_arguments,
    scan_attention,
)
from flaggems_vllm.runtime.backend._hygon.ops.flash_attention.tuning import (
    AttentionEntry,
    prune_configs,
    source_version,
    tuning_strategy,
)
from flaggems_vllm.utils import libtuner


@AttentionEntry
@libtuner(
    configs=runtime.get_tuned_config("hygon_attention_direct"),
    key=ATTENTION_TUNING_KEY,
    strategy=tuning_strategy(ATTENTION_TUNING_KEY),
    prune_configs_by={"early_config_prune": prune_configs},
    cache_version=source_version(),
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit(do_not_specialize=["MAX_Q", "MAX_K"])
def attention_kernel(
    Q,
    K,
    V,
    O,
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
    MAX_Q=0,
    MAX_K=0,
    BATCHES: tl.constexpr = 1,
):
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
    result, lse, _ = scan_attention(
        q,
        K,
        V,
        TABLE,
        ALIBI,
        dims,
        qi,
        qh,
        valid_q,
        batch,
        kh,
        kstart,
        klen,
        qlen,
        low,
        high,
        BM,
        BN,
        BD,
        D,
        KP,
        KS,
        KH,
        VP,
        VS,
        VH,
        TS,
        AS,
        PAGE,
        SCALE,
        SOFTCAP,
        CAUSAL,
        WL,
        WR,
        PAGED,
        HAS_ALIBI,
    )
    tl.store(
        O + (qstart + qi[:, None]).to(tl.int64) * OS + qh[:, None] * OH + dims[None, :],
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
    maxk,
    scale,
    causal,
    window,
    softcap,
    packed=False,
    kernel=None,
    **options,
):
    hkv = k.shape[-2]
    grid = lambda meta: (
        1 if packed else triton.cdiv(maxq, meta["BM"]),
        cuq.numel() - 1,
        hkv if packed else q.shape[1],
    )
    if kernel is None:
        kernel = attention_kernel
    return kernel[grid](
        *kernel_arguments(
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
            scale,
            causal,
            window,
            softcap,
            packed,
        ),
        BD=max(64, triton.next_power_of_2(q.shape[2])),
        MAX_Q=maxq,
        MAX_K=maxk,
        BATCHES=cuq.numel() - 1,
        **options,
    )


@triton.jit
def paged_step(
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
    page = tl.load(TABLE + batch * TS + start // PAGE).to(tl.int64)
    ko = page * KP + (start % PAGE + ns) * KS
    vo = page * VP + (start % PAGE + ns) * VS
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
    return acc, new_m, row_sum


@AttentionEntry
@libtuner(
    configs=runtime.get_tuned_config("hygon_attention_paged"),
    key=ATTENTION_TUNING_KEY + ["SHORT_LIMIT"],
    strategy=tuning_strategy(ATTENTION_TUNING_KEY + ["SHORT_LIMIT"]),
    prune_configs_by={"early_config_prune": prune_configs},
    cache_version=source_version(),
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit(do_not_specialize=["MAX_Q", "MAX_K"])
def paged_attention_kernel(
    Q,
    K,
    V,
    O,
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
    MAX_Q=0,
    MAX_K=0,
    BATCHES: tl.constexpr = 1,
):
    tl.static_assert(PAGED and PAGE == 64 and D == BD and BN <= PAGE)
    tl.static_assert(
        not PACKED and not HAS_ALIBI and SOFTCAP == 0 and WL == -1 and WR == -1
    )
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if ORDER:
        total_tiles = tl.num_programs(0)
        linear = head * total_tiles + tile
        group = linear // (ORDER * HQ)
        first_tile = group * ORDER
        group_size = tl.minimum(total_tiles - first_tile, ORDER)
        head = (linear % (ORDER * HQ)) // group_size
        tile = first_tile + linear % group_size
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    klen = tl.load(USEDK + batch)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and klen >= 8192:
        return
    if tile * BM >= qlen:
        return
    if REVERSE:
        tile = tl.cdiv(qlen, BM) - 1 - tile
    qi = tile * BM + tl.arange(0, BM)
    ds = tl.arange(0, BD)
    q = tl.load(
        Q + (qstart + qi[:, None]).to(tl.int64) * QS + head * QH + ds[None, :],
        qi[:, None] < qlen,
        0,
    )
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
    for full in tl.static_range(2):
        lo = 0 if full == 0 else full_hi
        end = full_hi if full == 0 else hi
        for start in range(lo, end, BN):
            (acc, m, row_sum) = paged_step(
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
                full == 0,
            )
    result = acc / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
    lse = tl.where(
        row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
    )
    tl.store(
        O + (qstart + qi[:, None]).to(tl.int64) * OS + head * OH + ds[None, :],
        result,
        qi[:, None] < qlen,
    )
    tl.store(LSE + head * TQ + qstart + qi, lse, qi < qlen)
