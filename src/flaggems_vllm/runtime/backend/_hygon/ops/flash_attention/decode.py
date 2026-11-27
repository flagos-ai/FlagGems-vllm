# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
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
from flaggems_vllm.utils import libentry, libtuner


@AttentionEntry
@libtuner(
    configs=runtime.get_tuned_config("hygon_attention_decode"),
    key=ATTENTION_TUNING_KEY + ["SPLITS", "PT", "PART_Q"],
    strategy=tuning_strategy(ATTENTION_TUNING_KEY + ["SPLITS", "PT", "PART_Q"]),
    prune_configs_by={"early_config_prune": prune_configs},
    cache_version=source_version(),
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit(do_not_specialize=["MAX_Q", "MAX_K"])
def split_attention_kernel(
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
    SPLITS: tl.constexpr,
    PT: tl.constexpr,
    PART_Q: tl.constexpr,
    MAX_Q=0,
    MAX_K=0,
    BATCHES: tl.constexpr = 1,
):
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
    result, lse, row_sum = scan_attention(
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
        O
        + split.to(tl.int64) * PT * HQ * D
        + ((batch * PART_Q if PART_Q else qstart) + qi[:, None]).to(tl.int64) * OS
        + qh[:, None] * OH
        + dims[None, :],
        result,
        valid_q[:, None] & (dims[None, :] < D),
    )
    tl.store(
        LSE + split * HQ * PT + qh * PT + (batch * PART_Q if PART_Q else qstart) + qi,
        tl.where(row_sum > 0, lse, -float("inf")),
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
    maxk,
    scale,
    causal,
    window,
    softcap,
    splits=16,
    packed=True,
    short_limit=0,
):
    pt = short_limit * (cuq.numel() - 1) if short_limit else q.shape[0]
    partial = torch.empty(
        (splits, pt, q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    partial_lse = torch.empty(
        (splits, q.shape[1], pt), device=q.device, dtype=torch.float32
    )
    hkv = k.shape[-2]
    grid = lambda meta: (
        splits
        * triton.cdiv(
            short_limit or maxq,
            meta["BM"] // (q.shape[1] // hkv) if packed else meta["BM"],
        ),
        cuq.numel() - 1,
        hkv if packed else q.shape[1],
    )
    split_attention_kernel[grid](
        *kernel_arguments(
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
            scale,
            causal,
            window,
            softcap,
            packed,
        ),
        BD=max(64, triton.next_power_of_2(q.shape[2])),
        SPLITS=splits,
        PT=pt,
        PART_Q=short_limit,
        MAX_Q=maxq,
        MAX_K=maxk,
        BATCHES=cuq.numel() - 1,
    )
    merge_kernel[
        (short_limit or q.shape[0], q.shape[1], cuq.numel() - 1 if short_limit else 1)
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


@libentry()
@triton.jit
def merge_kernel(
    P,
    PL,
    O,
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
    token, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
    ss, ds = tl.arange(0, BS), tl.arange(0, BD)
    ls = tl.load(PL + ss * H * PT + head * PT + ptok, ss < S, -float("inf"))
    maximum = tl.max(ls, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weight = tl.exp(ls - maximum)
    denom = tl.sum(weight, 0)
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
    tl.store(O + (qb + token).to(tl.int64) * OS + head * OH + ds, result, ds < D)
    tl.store(
        LSE + head * T + qb + token,
        tl.where(denom > 0, maximum + tl.log(denom), float("inf")),
    )
