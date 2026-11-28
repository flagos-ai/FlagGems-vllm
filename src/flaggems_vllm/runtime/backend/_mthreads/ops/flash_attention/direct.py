# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import (
    current_config,
    launch,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.ragged import (
    compact_tile_coords,
)

_BACKEND_OPT = (
    "enable_backend_opt"
    if tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 6)
    else "en_backend_opt"
)


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
    NUM_SPLITS: tl.constexpr = 1,
    Q_TILES: tl.constexpr = 1,
    HEAD_PACK: tl.constexpr = 1,
    COMPACT: tl.constexpr = False,
    BATCH: tl.constexpr = 1,
    BB: tl.constexpr = 1,
    NORM=None,
    NORM_TILES: tl.constexpr = 1,
):
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if NUM_SPLITS > 1:
        split_id = tile // Q_TILES
        tile = tile % Q_TILES
        OUT += split_id * TQ * HQ * D
        LSE += split_id * HQ * TQ
    if COMPACT:
        batch, tile, valid_task = compact_tile_coords(
            tile, CUQ, BATCH, BB, BM, HEAD_PACK
        )
        if not valid_task:
            return
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
        packed_row = tile * BM + rows
        qi = packed_row // HEAD_PACK
        qh = head * HEAD_PACK + packed_row % HEAD_PACK
        kh = head * HEAD_PACK // groups
        valid_q = qi < qlen
        row_base = tile * BM // HEAD_PACK
        row_end = tl.minimum(tl.cdiv((tile + 1) * BM, HEAD_PACK), qlen)
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
    if NUM_SPLITS > 1:
        # Partition the actual masked span, not a host-side maximum KV length.
        blocks = tl.cdiv(tl.maximum(high - low, 0), BN)
        blocks_per_split = tl.cdiv(blocks, NUM_SPLITS)
        split_start = low + split_id * blocks_per_split * BN
        high = tl.minimum(high, split_start + blocks_per_split * BN)
        low = split_start
    eligible = False
    fixed_m = 0.0
    if NORM is not None:
        qnorm = tl.max(tl.sum(q.to(tl.float32) * q.to(tl.float32), 1), 0)
        ni = tl.arange(0, triton.next_power_of_2(NORM_TILES))
        knorm = tl.max(
            tl.load(NORM + (batch * HK + kh) * NORM_TILES + ni, ni < NORM_TILES, 0), 0
        )
        bound = tl.sqrt(qnorm * knorm) * tl.abs(SCALE)
        eligible = (bound >= 0) & (bound <= 30)
        fixed_m = tl.ceil(bound * 1.4426950408889634) + 2.0
    m = tl.full((BM,), -float("inf"), tl.float32)
    m = tl.where(eligible, fixed_m, m)
    row_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    if HAS_ALIBI:
        slope = tl.load(ALIBI + batch * AS + qh, valid_q, 0)
    for start in range(low, high, BN):
        cols = start + tl.arange(0, BN)
        if PAGED:
            if BN <= PAGE and PAGE % BN == 0:
                page = tl.load(TABLE + batch * TS + start // PAGE)
            else:
                page = tl.load(TABLE + batch * TS + cols // PAGE, cols < klen, 0)
            # Cache tensors in the active workload exceed signed 32-bit offsets.
            ko = page.to(tl.int64) * KP + (cols % PAGE) * KS
            vo = page.to(tl.int64) * VP + (cols % PAGE) * VS
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
        if eligible:
            new_m = m
            p = tl.exp2(scores - m[:, None])
            row_sum += tl.sum(p, 1)
        else:
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
        row_sum > 0,
        (m + tl.log2(row_sum)) * 0.6931471805599453,
        -float("inf") if NUM_SPLITS > 1 else float("inf"),
    )
    tl.store(
        OUT
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
    norm=None,
    norm_tiles=1,
    maxk=None,
):
    paged = table is not None
    hkv = k.shape[-2]
    head_pack = (
        q.shape[1] // hkv
        if plan.family in ("packed_decode", "split_decode")
        else plan.head_pack
    )
    packed = head_pack > 1
    q_tiles = triton.cdiv(maxq * head_pack, plan.block_m)
    batch = cuq.numel() - 1
    compact_upper = triton.cdiv(q.shape[0] * head_pack, plan.block_m) + batch - 1
    compact = plan.family == "compact" or (
        plan.num_splits > 1
        and 1 < batch <= 128
        and q_tiles * batch > 1.5 * compact_upper
    )
    config = current_config()
    if config is not None and config.compact != -1:
        compact = bool(config.compact)
    if compact:
        q_tiles = compact_upper
    grid = (
        q_tiles * plan.num_splits,
        1 if compact else batch,
        q.shape[1] // head_pack,
    )
    return launch(
        attention_kernel,
        (grid),
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
        out.stride(-3),
        out.stride(-2),
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
        # Keep both MMA output dimensions in the validated S5000 range.
        max(64, triton.next_power_of_2(q.shape[2])),
        plan.num_splits,
        q_tiles,
        head_pack,
        compact,
        batch,
        # MUSA prefix scans require at least one full 32-lane warp.
        max(32, triton.next_power_of_2(batch)),
        NORM=norm,
        NORM_TILES=norm_tiles,
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
        **{
            _BACKEND_OPT: q.shape[2] == 256
            and (
                plan.head_pack == 8
                # Triton 3.6 needs this for the existing ungrouped Split-KV
                # route as well; Triton 3.2 keeps its original setting.
                or (_BACKEND_OPT == "enable_backend_opt" and plan.family == "split_kv")
            )
        },
    )
