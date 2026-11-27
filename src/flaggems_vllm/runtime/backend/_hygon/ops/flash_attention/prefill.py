# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime.backend._hygon.ops.flash_attention.common import (
    ATTENTION_TUNING_KEY,
)
from flaggems_vllm.runtime.backend._hygon.ops.flash_attention.tuning import (
    AttentionEntry,
    prune_configs,
    source_version,
    tuning_strategy,
)
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@triton.jit
def page_norm(
    K,
    TABLE,
    USED,
    PART,
    HK: tl.constexpr,
    WIDTH: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    TS: tl.constexpr,
):
    page_idx, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(USED + batch)
    page = tl.load(TABLE + batch * TS + page_idx, page_idx * 64 < length, 0)
    rows, dims = page_idx * 64 + tl.arange(0, 64), tl.arange(0, 256)
    values = tl.load(
        K
        + page.to(tl.int64) * KP
        + (rows[:, None] % 64) * KS
        + head * KH
        + dims[None, :],
        rows[:, None] < length,
        0,
    ).to(tl.float32)
    squared = tl.sum(values * values, 1)
    result = tl.max(tl.where(squared == squared, squared, float("inf")), 0)
    tl.store(PART + (batch * HK + head) * WIDTH + page_idx, result)


@libentry()
@triton.jit
def reduce_norm(PART, NORM, WIDTH: tl.constexpr, BW: tl.constexpr):
    group = tl.program_id(0)
    cols = tl.arange(0, BW)
    values = tl.load(PART + group * WIDTH + cols, cols < WIDTH, 0)
    tl.store(NORM + group, tl.max(values, 0))


def prepare_norms(k, table, used, maxk):
    batch, hk = used.numel(), k.shape[-2]
    width = max(1, triton.cdiv(maxk, 64))
    partial = torch.empty((batch, hk, width), dtype=torch.float32, device=k.device)
    result = torch.empty((batch, hk), dtype=torch.float32, device=k.device)
    page_norm[(width, batch, hk)](
        k,
        table,
        used,
        partial,
        hk,
        width,
        *k.stride()[:3],
        table.stride(0),
        num_warps=4,
    )
    reduce_norm[(batch * hk,)](
        partial, result, width, triton.next_power_of_2(width), num_warps=4
    )
    return result


@triton.jit
def step(
    q0,
    q1,
    K,
    V,
    TABLE,
    a0,
    a1,
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
    CAUSAL: tl.constexpr,
    FULL: tl.constexpr,
    FIXED: tl.constexpr,
):
    ns, ds = tl.arange(0, BN), tl.arange(0, 128)
    cols = start + ns
    page = tl.load(TABLE + batch * TS + start // PAGE).to(tl.int64)
    ko = page * KP + (start % PAGE + ns) * KS
    vo = page * VP + (start % PAGE + ns) * VS
    if FULL:
        k0 = tl.load(K + ko[None, :] + kh * KH + ds[:, None])
        k1 = tl.load(K + ko[None, :] + kh * KH + ds[:, None] + 128)
    else:
        k0 = tl.load(K + ko[None, :] + kh * KH + ds[:, None], cols[None, :] < klen, 0)
        k1 = tl.load(
            K + ko[None, :] + kh * KH + ds[:, None] + 128, cols[None, :] < klen, 0
        )
    s = tl.dot(q0, k0)
    s = tl.dot(q1, k1, s) * (SCALE * 1.4426950408889634)
    if not FULL:
        mask = cols[None, :] < klen
        if CAUSAL:
            mask &= cols[None, :] <= qi[:, None] + delta
        s = tl.where(mask, s, -float("inf"))
    if FIXED:
        nm = tl.full((BM,), 32.0 * 1.4426950408889634, tl.float32)
        p = tl.exp2(s - (32.0 * 1.4426950408889634))
        row_sum += tl.sum(p, 1)
    else:
        nm = tl.maximum(m, tl.max(s, 1))
        safe = tl.where(nm == -float("inf"), 0.0, nm)
        alpha = tl.exp2(m - safe)
        p = tl.exp2(s - safe[:, None])
        row_sum = row_sum * alpha + tl.sum(p, 1)
        a0 *= alpha[:, None]
        a1 *= alpha[:, None]
    if FULL:
        v0 = tl.load(V + vo[:, None] + kh * VH + ds[None, :])
    else:
        v0 = tl.load(V + vo[:, None] + kh * VH + ds[None, :], cols[:, None] < klen, 0)
    if FIXED and V.dtype.element_ty == tl.bfloat16:
        bits = p.to(tl.uint32, bitcast=True)
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        pb = rounded.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    else:
        pb = p.to(v0.dtype)
    a0 = tl.dot(pb, v0, a0)
    if FULL:
        v1 = tl.load(V + vo[:, None] + kh * VH + ds[None, :] + 128)
    else:
        v1 = tl.load(
            V + vo[:, None] + kh * VH + ds[None, :] + 128, cols[:, None] < klen, 0
        )
    a1 = tl.dot(pb, v1, a1)
    return a0, a1, nm, row_sum


@AttentionEntry
@libtuner(
    configs=runtime.get_tuned_config("hygon_attention_prefill"),
    key=ATTENTION_TUNING_KEY + ["SHORT_LIMIT"],
    strategy=tuning_strategy(ATTENTION_TUNING_KEY + ["SHORT_LIMIT"]),
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
    REVERSE: tl.constexpr = False,
    SHORT_LIMIT: tl.constexpr = 0,
    ORDER: tl.constexpr = 0,
    KNORM=None,
    MAX_Q=0,
    MAX_K=0,
    BATCHES: tl.constexpr = 1,
):
    tl.static_assert(PAGED and PAGE == 64 and D == 256 and BN <= PAGE)
    tl.static_assert(
        not PACKED and not HAS_ALIBI and SOFTCAP == 0 and WL == -1 and WR == -1
    )
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if ORDER:
        nt = tl.num_programs(0)
        linear = head * nt + tile
        group = linear // (ORDER * HQ)
        first = group * ORDER
        size = tl.minimum(nt - first, ORDER)
        head = (linear % (ORDER * HQ)) // size
        tile = first + linear % size
    qb = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qb
    klen = tl.load(USEDK + batch)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and klen >= 8192:
        return
    if tile * BM >= qlen:
        return
    if REVERSE:
        tile = tl.cdiv(qlen, BM) - 1 - tile
    qi = tile * BM + tl.arange(0, BM)
    ds = tl.arange(0, 128)
    qp = Q + (qb + qi[:, None]).to(tl.int64) * QS + head * QH + ds[None, :]
    q0 = tl.load(qp, qi[:, None] < qlen, 0)
    q1 = tl.load(qp + 128, qi[:, None] < qlen, 0)
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
    a0 = tl.zeros((BM, 128), tl.float32)
    a1 = tl.zeros((BM, 128), tl.float32)
    if Q.dtype.element_ty == tl.bfloat16:
        qf0, qf1 = q0.to(tl.float32), q1.to(tl.float32)
        squared = tl.sum(qf0 * qf0 + qf1 * qf1, 1)
        qnorm = tl.max(tl.where(squared == squared, squared, float("inf")), 0)
        knorm = tl.load(KNORM + batch * HK + kh)
        fixed_ok = (qnorm * knorm * (SCALE * SCALE) <= 1024.0) & (
            Q.dtype.element_ty == tl.bfloat16
        )
    else:
        fixed_ok = False
    for fixed in tl.static_range(2):
        if (fixed == 1 and fixed_ok) or (fixed == 0 and not fixed_ok):
            for full in tl.static_range(2):
                lo = 0 if full == 0 else full_hi
                end = full_hi if full == 0 else hi
                for start in range(lo, end, BN):
                    (a0, a1, m, row_sum) = step(
                        q0,
                        q1,
                        K,
                        V,
                        TABLE,
                        a0,
                        a1,
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
                        CAUSAL,
                        full == 0,
                        fixed == 1,
                    )
    denom = tl.where(row_sum > 0, row_sum, 1.0)
    op = O + (qb + qi[:, None]).to(tl.int64) * OS + head * OH + ds[None, :]
    tl.store(op, a0 / denom[:, None], qi[:, None] < qlen)
    tl.store(op + 128, a1 / denom[:, None], qi[:, None] < qlen)
    lse = tl.where(
        row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
    )
    tl.store(LSE + head * TQ + qb + qi, lse, qi < qlen)
