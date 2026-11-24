# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.standard import bitonic_merge


class _Profile(NamedTuple):
    hidden: int
    experts: int
    topk: int
    scale: float
    route: int


_SQRT_SOFTPLUS = 0
_GROUPED_SIGMOID = 1
_SOFTMAX = 2
_PROFILES = {
    "dsv4_flash": _Profile(4096, 256, 6, 1.5, _SQRT_SOFTPLUS),
    "dsv32": _Profile(7168, 256, 8, 2.5, _GROUPED_SIGMOID),
    "qwen_h2048_e256_k8": _Profile(2048, 256, 8, 1.0, _SOFTMAX),
    "qwen_h4096_e512_k10": _Profile(4096, 512, 10, 1.0, _SOFTMAX),
}


@triton.jit
def _project_tile(
    X,
    W,
    rows,
    experts,
    T,
    split,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    UNROLL: tl.constexpr = None,
    ACTIVATION_ORDER: tl.constexpr = None,
):
    tl.static_assert(H % (SPLITS * BK) == 0)
    k = split * (H // SPLITS) + tl.arange(0, BK)
    input_rows = tl.minimum(rows, T - 1)
    if ACTIVATION_ORDER is not None:
        x_ptr = X + k[:, None] + input_rows[None, :] * H
    else:
        x_ptr = X + input_rows[:, None] * H + k[None, :]
    w_ptr = W + experts[:, None] * H + k[None, :]
    if TRANSPOSE:
        acc = tl.full((BN, BM), 0, tl.float32)
    else:
        acc = tl.full((BM, BN), 0, tl.float32)
    for _ in tl.range(H // (SPLITS * BK), loop_unroll_factor=UNROLL):
        if ACTIVATION_ORDER == "weight_first":
            b = tl.load(w_ptr)
            a = tl.load(x_ptr)
        else:
            a = tl.load(x_ptr)
            b = tl.load(w_ptr)
        if TRANSPOSE:
            if ACTIVATION_ORDER is None:
                a = tl.trans(a)
            acc = tl.dot(b, a, acc)
        else:
            acc = tl.dot(a, tl.trans(b), acc)
        x_ptr += BK
        w_ptr += BK
    if TRANSPOSE:
        acc = tl.trans(acc)
    return acc


@triton.jit
def _rank_key(score, expert):
    bits = score.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits >> 31) != 0, ~bits, bits ^ 0x80000000)
    return (ordered.to(tl.uint64) << 32) | (65535 - expert).to(tl.uint64)


@triton.jit
def _softmax_keys(logits, experts):
    # Order BF16 values as unsigned integers and break ties by expert ID.
    bits = logits.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    ordered = tl.where((bits >> 15) != 0, bits ^ 0xFFFF, bits ^ 0x8000)
    return (ordered << 16) | (65535 - experts[None, :]).to(tl.uint32)


@triton.jit
def _softmax_logits(keys):
    ordered = keys >> 16
    bits = tl.where((ordered & 0x8000) != 0, ordered ^ 0x8000, ordered ^ 0xFFFF)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def _store_routes(
    weights,
    keys,
    Weights,
    Indices,
    t,
    T,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    SOFTMAX: tl.constexpr,
):
    pad: tl.constexpr = triton.next_power_of_2(TOPK)
    slot = tl.arange(0, pad)
    valid_slot = slot[None, :] < TOPK
    if SOFTMAX:
        maximum = tl.max(tl.where(valid_slot, weights, -float("inf")), 1)
        weights = tl.exp(weights - maximum[:, None])
    denominator = tl.sum(tl.where(valid_slot, weights, 0.0), 1)
    out_offset = t[:, None] * TOPK + slot[None, :]
    out_mask = (t[:, None] < T) & valid_slot
    selected = 65535 - (keys & 65535).to(tl.int32)
    tl.store(Weights + out_offset, (weights / denominator[:, None]) * SCALE, out_mask)
    tl.store(Indices + out_offset, selected, out_mask)


@triton.jit
def _sqrt_softplus(logits):
    softplus = tl.maximum(logits, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(logits)))
    return tl.sqrt(tl.where(logits < -15.0, tl.exp(logits), softplus))


@triton.jit
def _route_epilogue(
    logits,
    Bias,
    Weights,
    Indices,
    t,
    T,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    KIND: tl.constexpr,
    ROWS: tl.constexpr,
):
    e = tl.arange(0, E)
    if KIND == 0:
        scores = _sqrt_softplus(logits)
        ranking = scores + tl.load(Bias + e)[None, :]
    else:
        logits = logits.to(tl.bfloat16).to(tl.float32)
        if KIND == 1:
            scores = tl.sigmoid(logits)
            ranking = scores + tl.load(Bias + e)[None, :]
            grouped = tl.reshape(ranking, (ROWS, 8, 32))
            group_ranking = tl.sum(tl.topk(grouped, 2), 2)
            group = tl.arange(0, 8)
            group_keys = tl.topk(_rank_key(group_ranking, group[None, :]), 4)
            group_ids = 65535 - (group_keys & 65535).to(tl.int32)
            group_bits = tl.sum(1 << group_ids, 1)
            keep = ((group_bits[:, None] >> (e[None, :] // 32)) & 1) != 0
            ranking = tl.where(keep, ranking, -float("inf"))
        else:
            ranking = logits

    pad: tl.constexpr = triton.next_power_of_2(TOPK)
    if KIND == 2:
        keys = tl.topk(_softmax_keys(ranking, e), pad)
        weights = _softmax_logits(keys)
    else:
        keys = tl.topk(_rank_key(ranking, e[None, :]), pad)
        weights = tl.gather(scores, 65535 - (keys & 65535).to(tl.int32), axis=1)
    _store_routes(weights, keys, Weights, Indices, t, T, TOPK, SCALE, KIND == 2)


@triton.jit
def _direct_megagate(
    X,
    W,
    Bias,
    Weights,
    Indices,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    KIND: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    experts = tl.arange(0, E)
    logits = _project_tile(X, W, rows, experts, T, 0, H, 1, BM, E, 64, False)
    _route_epilogue(logits, Bias, Weights, Indices, rows, T, E, TOPK, SCALE, KIND, BM)


@triton.jit
def _publish_partial(counter, WORKGROUPS: tl.constexpr, WARPS: tl.constexpr):
    # Hygon uses 64-lane waves. Publish once per wave so every wave releases
    # its own global stores before the last workgroup consumes the partials.
    lanes = tl.arange(0, WARPS * 64)
    leaders = lanes % 64 == 0
    tickets = tl.atomic_add(
        counter + tl.full((WARPS * 64,), 0, tl.int32),
        1,
        leaders,
        sem="acq_rel",
        scope="gpu",
    )
    last_ticket = tl.max(tl.where(leaders, tickets, -1), axis=0)
    return last_ticket == WORKGROUPS * WARPS - 1


@triton.jit
def _tiled_megagate(
    X,
    W,
    Bias,
    Partial,
    Arrivals,
    Weights,
    Indices,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    KIND: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLITS: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    ROWS: tl.constexpr,
    REDUCE_SPLITS: tl.constexpr,
    ACTIVE_ROWS: tl.constexpr,
    BK: tl.constexpr,
    SHORT_PROJECTION: tl.constexpr,
    WARPS: tl.constexpr,
):
    n_blocks: tl.constexpr = E // BN
    pid = tl.program_id(0)
    tile, expert_tile = pid // n_blocks, pid % n_blocks
    split = tl.program_id(1)
    local_rows = tl.arange(0, BM)
    rows = tile * ACTIVE_ROWS + tl.minimum(local_rows, ACTIVE_ROWS - 1)
    experts = expert_tile * BN + tl.arange(0, BN)
    if SHORT_PROJECTION:
        accum = _project_grouped(
            X, W, rows, experts, T, split, H, BM, BN, BK, SPLITS, False
        )
    else:
        accum = _project_tile(
            X, W, rows, experts, T, split, H, SPLITS, BM, BN, BK, TRANSPOSE
        )
    if REDUCE_SPLITS:
        offsets = (rows[:, None] * SPLITS + split) * E + experts[None, :]
    else:
        offsets = (split * T + rows[:, None]) * E + experts[None, :]
    mask = (rows[:, None] < T) & (local_rows[:, None] < ACTIVE_ROWS)
    tl.store(Partial + offsets, accum, mask)

    if _publish_partial(Arrivals + tile, n_blocks * SPLITS, WARPS):
        e = tl.arange(0, E)
        valid_rows = tl.minimum(T - tile * ACTIVE_ROWS, ACTIVE_ROWS)
        for start in range(tl.cdiv(valid_rows, ROWS)):
            t = tile * ACTIVE_ROWS + start * ROWS + tl.arange(0, ROWS)
            if REDUCE_SPLITS:
                # Keep one row's split fragments together and reduce them in parallel.
                kparts = tl.arange(0, SPLITS)
                offset = (t[:, None, None] * SPLITS + kparts[None, :, None]) * E
                partials = tl.load(
                    Partial + offset + e[None, None, :], t[:, None, None] < T, other=0
                )
                logits = tl.sum(partials, axis=1)
            else:
                offset = t[:, None] * E + e[None, :]
                logits = tl.load(Partial + offset, t[:, None] < T, other=0).to(
                    tl.float32
                )
                for kpart in tl.static_range(1, SPLITS):
                    logits += tl.load(
                        Partial + kpart * T * E + offset, t[:, None] < T, other=0
                    )
            _route_epilogue(
                logits, Bias, Weights, Indices, t, T, E, TOPK, SCALE, KIND, ROWS
            )
        tl.debug_barrier()
        tl.atomic_xchg(Arrivals + tile, 0, sem="release", scope="gpu")


@triton.jit
def _project_grouped(
    X,
    W,
    rows,
    experts,
    T,
    split,
    H: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    ALIGNED: tl.constexpr,
    CLAMP_TAIL_ROWS: tl.constexpr = False,
):
    # Grouped GEMM addressing follows FlagGems' Hygon mm projection.
    if ALIGNED:
        aligned_rows = tl.max_contiguous(tl.multiple_of(rows, BM), BM)
        aligned_experts = tl.max_contiguous(tl.multiple_of(experts, BN), BN)
    else:
        aligned_rows = rows
        aligned_experts = experts
    if CLAMP_TAIL_ROWS:
        # The output stores still mask the tail. Clamp the input rows once
        # to keep the GEMM loop's loads unmasked.
        aligned_rows = tl.minimum(rows, T - 1)
    k = split * (H // SPLITS) + tl.arange(0, BK)
    x_ptr = X + aligned_rows[:, None] * H + k[None, :]
    w_ptr = W + k[:, None] + aligned_experts[None, :] * H
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in tl.range(H // (SPLITS * BK), loop_unroll_factor=1):
        if CLAMP_TAIL_ROWS:
            x = tl.load(x_ptr + step * BK)
        else:
            x = tl.load(x_ptr + step * BK, rows[:, None] < T, other=0.0)
        w = tl.load(w_ptr + step * BK)
        acc = tl.dot(x, w, acc, allow_tf32=False)
    return acc


@triton.jit
def _store_qwen_local_topk(
    Partial,
    accum,
    experts,
    base,
    T,
    expert_tile,
    PAD: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = base + tl.arange(0, ROWS)
    keys = tl.topk(_softmax_keys(accum, experts), PAD)
    offsets = rows[:, None] * WIDTH + expert_tile * PAD + tl.arange(0, PAD)[None, :]
    tl.store(Partial + offsets, keys.to(tl.int32, bitcast=True), rows[:, None] < T)


@triton.jit
def _split_qwen_rows(accum, M: tl.constexpr, N: tl.constexpr):
    return tl.split(tl.permute(tl.reshape(accum, (2, M // 2, N)), (1, 2, 0)))


@triton.jit
def _store_qwen_topk_quarters(
    Partial,
    accum,
    experts,
    base,
    T,
    expert_tile,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PAD: tl.constexpr,
    WIDTH: tl.constexpr,
):
    first, second = _split_qwen_rows(accum, BM, BN)
    a0, a1 = _split_qwen_rows(first, BM // 2, BN)
    a2, a3 = _split_qwen_rows(second, BM // 2, BN)
    chunk: tl.constexpr = BM // 4
    _store_qwen_local_topk(
        Partial, a0, experts, base, T, expert_tile, PAD, WIDTH, chunk
    )
    _store_qwen_local_topk(
        Partial, a1, experts, base + chunk, T, expert_tile, PAD, WIDTH, chunk
    )
    _store_qwen_local_topk(
        Partial, a2, experts, base + 2 * chunk, T, expert_tile, PAD, WIDTH, chunk
    )
    _store_qwen_local_topk(
        Partial, a3, experts, base + 3 * chunk, T, expert_tile, PAD, WIDTH, chunk
    )


@triton.jit
def _merge_qwen_topk_pairs(
    staged, ROWS: tl.constexpr, PAIRS: tl.constexpr, PAD: tl.constexpr
):
    # Each local list is already descending. Reversing the second list
    # pairs each low-ranked key with a high-ranked key from the other tile.
    lists = tl.reshape(staged, (ROWS, PAIRS, 2, PAD))
    first, second = tl.split(tl.permute(lists, (0, 1, 3, 2)))
    return bitonic_merge(tl.maximum(first, tl.flip(second, -1)), descending=True)


@triton.jit
def _project_qwen_expert_pair(
    X,
    W,
    rows,
    experts,
    extra,
    T,
    has_extra,
    H: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    EXTRA_N: tl.constexpr,
    BK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    k = tl.arange(0, BK)
    input_rows = tl.minimum(rows, T - 1)
    if EXTRA_N == BN:
        # Lay out the activation loads in the right-hand dot operand's order.
        x_ptr = X + k[:, None] + input_rows[None, :] * H
    else:
        x_ptr = X + input_rows[:, None] * H + k[None, :]
    w_ptr = W + experts[:, None] * H + k[None, :]
    extra_ptr = W + extra[:, None] * H + k[None, :]
    left = tl.full((BN, BM), 0, tl.float32)
    right = tl.full((EXTRA_N, BM), 0, tl.float32)
    for _ in tl.range(H // BK, loop_unroll_factor=UNROLL):
        if EXTRA_N < BN:
            c = tl.load(extra_ptr, has_extra, 0)
            a = tl.trans(tl.load(x_ptr))
            b = tl.load(w_ptr)
            right = tl.dot(c, a, right)
            left = tl.dot(b, a, left)
        else:
            a = tl.load(x_ptr)
            b = tl.load(w_ptr)
            left = tl.dot(b, a, left)
            c = tl.load(extra_ptr)
            right = tl.dot(c, a, right)
        x_ptr += BK
        w_ptr += BK
        extra_ptr += BK
    return tl.trans(left), tl.trans(right)


@triton.jit
def _merge_qwen_sorted(first, second):
    return bitonic_merge(tl.maximum(first, tl.flip(second, -1)), descending=True)


@triton.jit
def _store_qwen_expert_pair(
    Partial,
    left,
    right,
    experts,
    extra,
    base,
    T,
    expert_tile,
    has_extra,
    GROUPS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAD: tl.constexpr,
    ROWS: tl.constexpr,
):
    if GROUPS == 2:
        _store_qwen_local_topk(
            Partial, left, experts, base, T, 2 * expert_tile, PAD, WIDTH, ROWS
        )
        _store_qwen_local_topk(
            Partial, right, extra, base, T, 2 * expert_tile + 1, PAD, WIDTH, ROWS
        )
    else:
        first = tl.topk(_softmax_keys(left, experts), PAD)
        second = tl.topk(_softmax_keys(right, extra), PAD)
        keys = _merge_qwen_sorted(first, tl.where(has_extra, second, 0))
        rows = base + tl.arange(0, ROWS)
        offset = rows[:, None] * WIDTH + expert_tile * PAD + tl.arange(0, PAD)[None, :]
        tl.store(Partial + offset, keys.to(tl.int32, bitcast=True), rows[:, None] < T)


@triton.jit
def _store_qwen_expert_pair_quarters(
    Partial,
    left,
    right,
    experts,
    extra,
    base,
    T,
    expert_tile,
    has_extra,
    BM: tl.constexpr,
    BN: tl.constexpr,
    EXTRA_N: tl.constexpr,
    GROUPS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAD: tl.constexpr,
):
    l0, l1 = _split_qwen_rows(left, BM, BN)
    r0, r1 = _split_qwen_rows(right, BM, EXTRA_N)
    l00, l01 = _split_qwen_rows(l0, BM // 2, BN)
    l10, l11 = _split_qwen_rows(l1, BM // 2, BN)
    r00, r01 = _split_qwen_rows(r0, BM // 2, EXTRA_N)
    r10, r11 = _split_qwen_rows(r1, BM // 2, EXTRA_N)
    chunk: tl.constexpr = BM // 4
    _store_qwen_expert_pair(
        Partial,
        l00,
        r00,
        experts,
        extra,
        base,
        T,
        expert_tile,
        has_extra,
        GROUPS,
        WIDTH,
        PAD,
        chunk,
    )
    _store_qwen_expert_pair(
        Partial,
        l01,
        r01,
        experts,
        extra,
        base + chunk,
        T,
        expert_tile,
        has_extra,
        GROUPS,
        WIDTH,
        PAD,
        chunk,
    )
    _store_qwen_expert_pair(
        Partial,
        l10,
        r10,
        experts,
        extra,
        base + 2 * chunk,
        T,
        expert_tile,
        has_extra,
        GROUPS,
        WIDTH,
        PAD,
        chunk,
    )
    _store_qwen_expert_pair(
        Partial,
        l11,
        r11,
        experts,
        extra,
        base + 3 * chunk,
        T,
        expert_tile,
        has_extra,
        GROUPS,
        WIDTH,
        PAD,
        chunk,
    )


@triton.jit
def _qwen_pruned_megagate(
    X,
    W,
    Partial,
    Arrivals,
    Weights,
    Indices,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    ROWS: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARPS: tl.constexpr,
    SPLIT_PRUNE_ROWS: tl.constexpr,
    MERGE_SORTED: tl.constexpr,
    UNROLL: tl.constexpr,
    EXPERT_GROUPS: tl.constexpr,
    ACTIVATION_ORDER: tl.constexpr,
):
    n_blocks: tl.constexpr = E // BN
    pad: tl.constexpr = triton.next_power_of_2(TOPK)
    width: tl.constexpr = n_blocks * pad
    workgroups: tl.constexpr = EXPERT_GROUPS or n_blocks
    pid = tl.program_id(0)
    if GROUP_M:
        first = pid // (GROUP_M * n_blocks) * GROUP_M
        span = tl.minimum(tl.cdiv(T, BM) - first, GROUP_M)
        inner = pid % (GROUP_M * n_blocks)
        tile, expert_tile = first + inner % span, inner // span
    else:
        tile, expert_tile = pid // workgroups, pid % workgroups
    rows = tile * BM + tl.arange(0, BM)
    if EXPERT_GROUPS:
        # Two workgroups each cover 256 experts. Three cover 192, 192, and 128.
        extra_n: tl.constexpr = BN if EXPERT_GROUPS == 2 else BN // 2
        expert_base = expert_tile * (BN + extra_n)
        experts = expert_base + tl.arange(0, BN)
        has_extra = expert_tile < 2
        extra = tl.where(has_extra, expert_base + BN, 0) + tl.arange(0, extra_n)
        left, right = _project_qwen_expert_pair(
            X, W, rows, experts, extra, T, has_extra, H, BM, BN, extra_n, BK, UNROLL
        )
        _store_qwen_expert_pair_quarters(
            Partial,
            left,
            right,
            experts,
            extra,
            tile * BM,
            T,
            expert_tile,
            has_extra,
            BM,
            BN,
            extra_n,
            EXPERT_GROUPS,
            width,
            pad,
        )
    else:
        experts = expert_tile * BN + tl.arange(0, BN)
        if GROUP_M:
            clamp_tail_rows: tl.constexpr = (
                E == 512 and BM == 64 and BN == 64 and GROUP_M == 2
            )
            accum = _project_grouped(
                X, W, rows, experts, T, 0, H, BM, BN, BK, 1, True, clamp_tail_rows
            )
        else:
            accum = _project_tile(
                X,
                W,
                rows,
                experts,
                T,
                0,
                H,
                1,
                BM,
                BN,
                BK,
                TRANSPOSE,
                UNROLL,
                ACTIVATION_ORDER,
            )

        # An expert below its tile's top-k cannot enter the global top-k.
        if SPLIT_PRUNE_ROWS:
            if BM == 256:
                first, second = _split_qwen_rows(accum, BM, BN)
                _store_qwen_topk_quarters(
                    Partial,
                    first,
                    experts,
                    tile * BM,
                    T,
                    expert_tile,
                    BM // 2,
                    BN,
                    pad,
                    width,
                )
                _store_qwen_topk_quarters(
                    Partial,
                    second,
                    experts,
                    tile * BM + BM // 2,
                    T,
                    expert_tile,
                    BM // 2,
                    BN,
                    pad,
                    width,
                )
            else:
                _store_qwen_topk_quarters(
                    Partial,
                    accum,
                    experts,
                    tile * BM,
                    T,
                    expert_tile,
                    BM,
                    BN,
                    pad,
                    width,
                )
        else:
            local_keys = tl.topk(_softmax_keys(accum, experts), pad)
            offset = (
                rows[:, None] * width + expert_tile * pad + tl.arange(0, pad)[None, :]
            )
            tl.store(
                Partial + offset,
                local_keys.to(tl.int32, bitcast=True),
                rows[:, None] < T,
            )
    if _publish_partial(Arrivals + tile, workgroups, WARPS):
        valid_rows = tl.minimum(T - tile * BM, BM)
        for start in range(tl.cdiv(valid_rows, ROWS)):
            t = tile * BM + start * ROWS + tl.arange(0, ROWS)
            if EXPERT_GROUPS == 3:
                offsets = t[:, None] * width + tl.arange(0, pad)[None, :]
                valid = t[:, None] < T
                first = tl.load(Partial + offsets, valid, 0).to(tl.uint32, bitcast=True)
                second = tl.load(Partial + offsets + pad, valid, 0).to(
                    tl.uint32, bitcast=True
                )
                third = tl.load(Partial + offsets + 2 * pad, valid, 0).to(
                    tl.uint32, bitcast=True
                )
                keys = _merge_qwen_sorted(_merge_qwen_sorted(first, second), third)
            else:
                load_offset = t[:, None] * width + tl.arange(0, width)[None, :]
                staged = tl.load(Partial + load_offset, t[:, None] < T, other=0)
                ranked = staged.to(tl.uint32, bitcast=True)
                if n_blocks == 4 and (
                    SPLIT_PRUNE_ROWS or MERGE_SORTED or EXPERT_GROUPS
                ):
                    pairs = _merge_qwen_topk_pairs(ranked, ROWS, 2, pad)
                    keys = tl.reshape(
                        _merge_qwen_topk_pairs(pairs, ROWS, 1, pad), (ROWS, pad)
                    )
                else:
                    keys = tl.topk(ranked, pad)
            _store_routes(
                _softmax_logits(keys), keys, Weights, Indices, t, T, TOPK, SCALE, True
            )
        tl.debug_barrier()
        tl.atomic_xchg(Arrivals + tile, 0, sem="release", scope="gpu")


@triton.jit
def _biased_pruned_megagate(
    X,
    W,
    Bias,
    Keys,
    Scores,
    GroupScores,
    Arrivals,
    Weights,
    Indices,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    ROWS: tl.constexpr,
    GROUPED: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARPS: tl.constexpr,
):
    n_blocks: tl.constexpr = E // BN
    pad: tl.constexpr = triton.next_power_of_2(TOPK)
    group_width: tl.constexpr = 32 if GROUPED else BN
    groups_per_tile: tl.constexpr = BN // group_width
    width: tl.constexpr = (E // group_width) * pad
    tl.static_assert(pad <= 16)

    pid = tl.program_id(0)
    if GROUP_M:
        first = pid // (GROUP_M * n_blocks) * GROUP_M
        span = tl.minimum(tl.cdiv(T, BM) - first, GROUP_M)
        inner = pid % (GROUP_M * n_blocks)
        tile, expert_tile = first + inner % span, inner // span
    else:
        tile, expert_tile = pid // n_blocks, pid % n_blocks
    rows = tile * BM + tl.arange(0, BM)
    experts = expert_tile * BN + tl.arange(0, BN)
    if GROUP_M:
        logits = _project_grouped(X, W, rows, experts, T, 0, H, BM, BN, BK, 1, True)
    else:
        logits = _project_tile(X, W, rows, experts, T, 0, H, 1, BM, BN, BK, TRANSPOSE)
    bias = tl.load(Bias + experts)[None, :]
    if GROUPED:
        tl.static_assert(E == 256)
        bf16 = logits.to(tl.bfloat16)
        ranking = tl.sigmoid(bf16.to(tl.float32)) + bias
        # Keep the full FP32 rank, the expert tie break, and the BF16 logit
        # in one key. The logit reconstructs the unbiased sigmoid score.
        ordered = _rank_key(ranking, experts[None, :]) & 0xFFFFFFFF00000000
        encoded_experts = (255 - experts[None, :]).to(tl.uint64) << 16
        encoded_logits = bf16.to(tl.uint16, bitcast=True).to(tl.uint64)
        packed = ordered | encoded_experts | encoded_logits
        local_keys = tl.topk(
            tl.reshape(packed, (BM, groups_per_tile, group_width)), pad
        )
        top_ranks = (local_keys >> 32).to(tl.uint32)
        bits = tl.where(
            (top_ranks & 0x80000000) != 0, top_ranks ^ 0x80000000, ~top_ranks
        )
        top_ranks_f32 = bits.to(tl.float32, bitcast=True)
        first_two = tl.arange(0, pad)[None, None, :] < 2
        group_scores = tl.sum(tl.where(first_two, top_ranks_f32, 0.0), 2)
        groups = expert_tile * groups_per_tile + tl.arange(0, groups_per_tile)
        tl.store(
            GroupScores + rows[:, None] * 8 + groups[None, :],
            group_scores,
            rows[:, None] < T,
        )
    else:
        scores = _sqrt_softplus(logits)
        ranking = scores + bias
        rank_keys = tl.reshape(
            _rank_key(ranking, experts[None, :]), (BM, groups_per_tile, group_width)
        )
        local_keys = tl.topk(rank_keys, pad)
        local_ids = 65535 - (local_keys & 65535).to(tl.int32)
        score_groups = tl.reshape(scores, (BM, groups_per_tile, group_width))
        local_scores = tl.gather(score_groups, local_ids % group_width, axis=2)

        # Four bits record the slot of the original unbiased FP32 score.
        slots = tl.arange(0, pad)[None, None, :].to(tl.uint64)
        local_keys = (
            (local_keys & 0xFFFFFFFF00000000) | ((local_keys & 65535) << 4) | slots
        )
    tile_width: tl.constexpr = groups_per_tile * pad
    offset = (
        rows[:, None] * width
        + expert_tile * tile_width
        + tl.arange(0, tile_width)[None, :]
    )
    tl.store(
        Keys + offset,
        tl.reshape(local_keys, (BM, tile_width)).to(tl.int64, bitcast=True),
        rows[:, None] < T,
    )
    if not GROUPED:
        tl.store(
            Scores + offset,
            tl.reshape(local_scores, (BM, tile_width)),
            rows[:, None] < T,
        )

    if _publish_partial(Arrivals + tile, n_blocks, WARPS):
        valid_rows = tl.minimum(T - tile * BM, BM)
        for start in range(tl.cdiv(valid_rows, ROWS)):
            t = tile * BM + start * ROWS + tl.arange(0, ROWS)
            staged_row = t[:, None] * width
            mask = t[:, None] < T
            slots_in_row = tl.arange(0, width)
            staged = tl.load(Keys + staged_row + slots_in_row[None, :], mask, other=0)
            staged_keys = staged.to(tl.uint64, bitcast=True)
            if GROUPED:
                group_ids = tl.arange(0, 8)
                staged_groups = tl.load(
                    GroupScores + t[:, None] * 8 + group_ids[None, :], mask, other=0
                )
                group_keys = tl.topk(_rank_key(staged_groups, group_ids[None, :]), 4)
                chosen = 65535 - (group_keys & 65535).to(tl.int32)
                group_bits = tl.sum(1 << chosen, 1)
                keep = (
                    (group_bits[:, None] >> (slots_in_row[None, :] // pad)) & 1
                ) != 0
                staged_keys = tl.where(keep, staged_keys, 0)
            selected = tl.topk(staged_keys, pad)
            if GROUPED:
                bf16_payload = (
                    (selected & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True)
                )
                weights = tl.sigmoid(bf16_payload.to(tl.float32))
                expert_keys = 65280 + ((selected >> 16) & 255)
            else:
                expert_keys = (selected >> 4) & 65535
                expert_ids = 65535 - expert_keys.to(tl.int32)
                column = expert_ids // group_width * pad + (selected & 15).to(tl.int32)
                weights = tl.load(Scores + staged_row + column, mask, other=0)
            _store_routes(
                weights, expert_keys, Weights, Indices, t, T, TOPK, SCALE, False
            )
        tl.debug_barrier()
        tl.atomic_xchg(Arrivals + tile, 0, sem="release", scope="gpu")


@triton.jit
def _init_arrivals(Arrivals, Count, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Arrivals + i, 0, i < Count)


class _Tile(NamedTuple):
    m: int
    n: int
    splits: int
    warps: int
    transpose: bool = False
    route_rows: int = 4
    prune: bool = False
    reduce_splits: bool = False
    active_rows: int | None = None
    block_k: int = 128
    stages: int = 1
    group_m: int = 0
    split_prune_rows: bool = False
    merge_sorted: bool = False
    projection_unroll: int | None = None
    expert_groups: int = 0
    activation_order: str | None = None

    @property
    def rows_per_tile(self) -> int:
        return self.active_rows or self.m


def _grouped_tile(m, n, route_rows=16, *, block_k=64):
    return _Tile(
        m,
        n,
        1,
        4,
        route_rows=route_rows,
        prune=True,
        block_k=block_k,
        stages=2,
        group_m=8,
    )


def _qwen_split_tile(n, splits, rows):
    return _Tile(
        16,
        n,
        splits,
        4,
        route_rows=rows,
        reduce_splits=True,
        active_rows=rows,
        block_k=64,
        stages=2,
        group_m=1,
    )


def _select_short_qwen_tile(tokens, experts):
    if experts == 512:
        if tokens <= 24:
            return _qwen_split_tile(128, 8, 2 if tokens <= 8 else 4)
        if tokens <= 48:
            return _qwen_split_tile(128, 4, 4)
        if tokens < 88:
            return _qwen_split_tile(128, 4, 8)
        block_m = 16 if tokens < 144 else 32
        return _grouped_tile(block_m, 64, block_m, block_k=128)

    if tokens <= 24:
        return _qwen_split_tile(64, 8, 4)
    if tokens <= 64:
        return _qwen_split_tile(64, 8 if tokens <= 32 else 4, 8)
    if tokens <= 128:
        return _qwen_split_tile(128, 8 if tokens <= 112 else 4, 8)
    block_m = 16 if tokens < 320 else 32
    return _grouped_tile(block_m, 64, block_m, block_k=128)


def _select_long_qwen_tile(tokens):
    if tokens <= 1280:
        # Four expert blocks per token tile fit in one wave of workgroups here.
        return _Tile(
            64,
            128,
            1,
            4,
            True,
            16,
            prune=True,
            block_k=64,
            stages=2,
            split_prune_rows=True,
            merge_sorted=True,
            projection_unroll=2,
            activation_order="activation_first",
        )
    if tokens < 1536:
        return _grouped_tile(32, 128, 32)._replace(group_m=16, merge_sorted=True)
    if 3072 <= tokens < 4096:
        return _grouped_tile(64, 64, 32)._replace(group_m=2)
    if tokens <= 16384:
        block_m, warps = (64, 4) if tokens < 3072 else (128, 8)
        # Fewer expert workgroups avoid the measured occupancy cliffs.
        expert_groups = 0
        if 5120 < tokens <= 6784:
            expert_groups = 3
        elif 7680 < tokens <= 10240 or 15360 < tokens <= 16384:
            expert_groups = 2
        elif 12800 < tokens <= 15360:
            block_m = 256
        activation_order = None
        if 1920 <= tokens <= 2496:
            activation_order = "activation_first"
        elif 4096 <= tokens <= 5120 or block_m == 256:
            activation_order = "weight_first"
        return _Tile(
            block_m,
            128,
            1,
            warps,
            True,
            min(block_m // 4, 32),
            prune=True,
            block_k=32,
            stages=2,
            split_prune_rows=True,
            projection_unroll=(
                2
                if expert_groups != 2 and (block_m == 128 or 1920 <= tokens <= 2496)
                else None
            ),
            expert_groups=expert_groups,
            activation_order=activation_order,
        )
    return _grouped_tile(64, 128, 32)


def _select_tile(tokens, profile):
    if profile.route == _SQRT_SOFTPLUS:
        if 32 <= tokens < 64:
            return _Tile(16, 64, 8, 4, True, 8, reduce_splits=True, active_rows=8)
        if 112 <= tokens <= 128:
            return _Tile(
                16,
                128,
                4,
                4,
                route_rows=8,
                reduce_splits=True,
                active_rows=8,
                block_k=64,
                stages=2,
                group_m=1,
            )
        if 64 <= tokens < 112:
            return _Tile(16, 128, 4, 4, True, 8, reduce_splits=True, active_rows=8)
        if tokens >= 8192:
            return _grouped_tile(64, 64, 64)._replace(group_m=4)
        if tokens >= 4096:
            return _grouped_tile(128, 64, 32)
        if tokens >= 3072:
            return _grouped_tile(64, 64, 64)
        if tokens >= 1536:
            return _grouped_tile(64, 64, 32)
        if tokens >= 768:
            return _grouped_tile(32, 64, block_k=128)
        if tokens >= 256:
            return _Tile(32, 64, 1, 4, True, 16, prune=True)
    if (
        profile.route == _SOFTMAX
        and tokens < 768
        and (tokens > 8 or profile.experts == 512)
    ):
        return _select_short_qwen_tile(tokens, profile.experts)
    if tokens <= 128:
        splits = 8 if profile.route == _GROUPED_SIGMOID else 16
        return _Tile(16, 32, splits, 2)
    if profile.route == _SOFTMAX:
        if profile.experts == 256:
            route_rows = 32 if 3072 <= tokens <= 6144 else 64
            return _grouped_tile(64, 64, route_rows)._replace(merge_sorted=True)
        return _select_long_qwen_tile(tokens)
    if profile.route == _GROUPED_SIGMOID:
        if tokens >= 7680:
            return _grouped_tile(64, 128, 64)
        if tokens >= 4096:
            return _Tile(64, 64, 1, 4, True, 16, prune=True)
        if tokens >= 1920:
            return _grouped_tile(64, 64, 64)
        if tokens >= 768:
            return _grouped_tile(32, 64)
        return _Tile(32, 128, 2, 4, True, 16)
    return _Tile(16, 128, 2, 4, True)


class MegaGateContext:
    """Reusable workspace for one device and shape.

    Prepare and warm up before graph capture. Use a separate context for each
    overlapping stream or graph. Replays that share a context must be serialized.
    Keep the context alive until all graphs captured with it are destroyed.
    """

    def __init__(self, profile: str = "dsv4_flash"):
        if profile not in _PROFILES:
            raise NotImplementedError(f"Unsupported Hygon MegaGate profile: {profile}")
        self._profile = profile
        self._workspace = None

    @property
    def profile(self) -> str:
        return self._profile

    def prepare(self, x: torch.Tensor) -> None:
        """Allocate the workspace before graph capture."""
        if self._workspace is not None:
            raise RuntimeError(
                "Use a new MegaGate context for a different shape or graph"
            )
        config = _PROFILES[self.profile]
        if x.ndim != 2 or x.shape[1] != config.hidden or x.device.type != "cuda":
            raise ValueError(
                "Prepare MegaGate with the input shape and destination device"
            )
        tokens = x.shape[0]
        tile = _select_tile(tokens, config)
        ntiles = triton.cdiv(tokens, tile.rows_per_tile)
        scores = None
        group_scores = None
        if tile.prune:
            grouped = config.route == _GROUPED_SIGMOID
            group_width = 32 if grouped else tile.n
            width = config.experts // group_width * triton.next_power_of_2(config.topk)
            dtype = torch.int32 if config.route == _SOFTMAX else torch.int64
            partial = torch.empty((1, tokens, width), dtype=dtype, device=x.device)
            if config.route == _SQRT_SOFTPLUS:
                scores = torch.empty(
                    (tokens, width), dtype=torch.float32, device=x.device
                )
            if grouped:
                group_scores = torch.empty(
                    (tokens, 8), dtype=torch.float32, device=x.device
                )
        else:
            dtype = (
                torch.bfloat16
                if tile.splits == 1 and config.route != _SQRT_SOFTPLUS
                else torch.float32
            )
            partial = torch.empty(
                (tile.splits, tokens, config.experts), dtype=dtype, device=x.device
            )
        arrivals = torch.empty((ntiles,), dtype=torch.int32, device=x.device)
        if ntiles:
            _init_arrivals[(triton.cdiv(ntiles, 128),)](arrivals, ntiles, 128)
        self._workspace = (tile, partial, arrivals, scores, group_scores)


def _launch(x, weight, bias, config, scale, context):
    tokens = x.shape[0]
    shape = (tokens, config.topk)
    weights = torch.empty(shape, dtype=torch.float32, device=x.device)
    index_dtype = torch.int64 if config.route == _SQRT_SOFTPLUS else torch.int32
    indices = torch.empty(shape, dtype=index_dtype, device=x.device)
    if not tokens:
        return weights, indices

    launch_options = dict(num_stages=1, matrix_instr_nonkdim=16, kpack=2)
    if context is None:
        bm = 16
        _direct_megagate[(triton.cdiv(tokens, bm),)](
            x,
            weight,
            bias,
            weights,
            indices,
            tokens,
            config.hidden,
            config.experts,
            config.topk,
            scale,
            config.route,
            bm,
            num_warps=4,
            **launch_options,
        )
    else:
        if context._workspace is None:
            raise ValueError("Prepare the MegaGate context before calling megagate")
        tile, partial, arrivals, scores, group_scores = context._workspace
        if partial.shape[1] != tokens or partial.device != x.device:
            raise ValueError("MegaGate context must match the input shape and device")
        grid = (
            triton.cdiv(tokens, tile.rows_per_tile)
            * (tile.expert_groups or config.experts // tile.n),
            tile.splits,
        )
        if scores is not None or group_scores is not None:
            _biased_pruned_megagate[(grid[0],)](
                x,
                weight,
                bias,
                partial,
                scores,
                group_scores,
                arrivals,
                weights,
                indices,
                tokens,
                config.hidden,
                config.experts,
                config.topk,
                scale,
                tile.m,
                tile.n,
                tile.transpose,
                tile.route_rows,
                config.route == _GROUPED_SIGMOID,
                tile.block_k,
                tile.group_m,
                tile.warps,
                num_warps=tile.warps,
                **dict(launch_options, num_stages=tile.stages, num_ldmatrixes=0),
            )
            return weights, indices
        if tile.prune:
            _qwen_pruned_megagate[(grid[0],)](
                x,
                weight,
                partial,
                arrivals,
                weights,
                indices,
                tokens,
                config.hidden,
                config.experts,
                config.topk,
                scale,
                tile.m,
                tile.n,
                tile.transpose,
                tile.route_rows,
                tile.block_k,
                tile.group_m,
                tile.warps,
                tile.split_prune_rows,
                tile.merge_sorted,
                tile.projection_unroll,
                tile.expert_groups,
                tile.activation_order,
                num_warps=tile.warps,
                **dict(launch_options, num_stages=tile.stages, num_ldmatrixes=0),
            )
            return weights, indices
        _tiled_megagate[grid](
            x,
            weight,
            bias,
            partial,
            arrivals,
            weights,
            indices,
            tokens,
            config.hidden,
            config.experts,
            config.topk,
            scale,
            config.route,
            tile.m,
            tile.n,
            tile.splits,
            tile.transpose,
            tile.route_rows,
            tile.reduce_splits,
            tile.rows_per_tile,
            tile.block_k,
            bool(tile.group_m),
            tile.warps,
            num_warps=tile.warps,
            **dict(launch_options, num_stages=tile.stages, num_ldmatrixes=0),
        )
    return weights, indices


def megagate(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None = None,
    *,
    profile: str = "dsv4_flash",
    routed_scaling_factor: float | None = None,
    context: MegaGateContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the normalized MoE weights and expert indices for one Hygon profile.

    Activations and gate weights are contiguous BF16 tensors. One Triton
    kernel runs on the current stream. A prepared context enables tiled
    cooperative execution and can be captured after warmup.
    """
    if profile not in _PROFILES:
        raise NotImplementedError(f"Unsupported Hygon MegaGate profile: {profile}")
    config = _PROFILES[profile]
    if (
        not torch.version.hip
        or x.device.type != "cuda"
        or x.device != gate_weight.device
        or x.dtype != torch.bfloat16
        or gate_weight.dtype != torch.bfloat16
        or x.ndim != 2
        or x.shape[1] != config.hidden
        or gate_weight.shape != (config.experts, config.hidden)
        or not x.is_contiguous()
        or not gate_weight.is_contiguous()
    ):
        raise ValueError(
            "Expected contiguous BF16 gate tensors on the same Hygon device"
        )
    if config.route == _SOFTMAX:
        if correction_bias is not None:
            raise ValueError("The Qwen MegaGate profiles do not take correction bias")
    elif (
        correction_bias is None
        or correction_bias.device != x.device
        or correction_bias.dtype != torch.float32
        or correction_bias.shape != (config.experts,)
        or not correction_bias.is_contiguous()
    ):
        raise ValueError("Expected contiguous FP32 correction bias on the gate device")
    if x.numel() >= 2**31:
        raise ValueError("MegaGate input exceeds the 32-bit index range")
    scale = config.scale if routed_scaling_factor is None else routed_scaling_factor
    if context is not None and context.profile != profile:
        raise ValueError(
            "MegaGate context profile does not match the requested profile"
        )
    return _launch(x, gate_weight, correction_bias, config, scale, context)
