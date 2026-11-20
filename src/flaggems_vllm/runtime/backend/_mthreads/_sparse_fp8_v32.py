# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.experimental.tle import language as tle
from triton.runtime.errors import OutOfResources

from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_native import (
    V32_DIRECT_COPY_IR,
    V32_DIRECT_COPY_TWO_LANE_IR,
    VECTOR_EXP2_IR,
    lma_wait,
    native_library,
)
from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_scheduling import (
    collect_v32_tail_denominator,
    mixed_v32_span,
)


@triton.jit
def v32_norm_bound(q_max, k_max, excess, slack, SCALE: tl.constexpr):
    # Cauchy-Schwarz, padded for the floating-point norm reductions.
    score_cap = tl.sqrt(q_max) * tl.sqrt(k_max) * (1.001 * SCALE)
    return (
        (score_cap <= 0.625)
        & (q_max <= 1024.0)
        & (k_max <= 1024.0)
        & (excess * (1.00001 * tl.exp(2.0 * score_cap)) <= slack)
    )


@triton.jit
def v32_norm_eligible(
    q_square,
    Indices,
    KS,
    KeyNorm,
    row,
    split,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    CHECK_SENTINEL: tl.constexpr,
    GlobalNorm=None,
):
    check_begin = split * CHUNK
    check_offsets = check_begin + tl.arange(0, triton.next_power_of_2(CHUNK))
    chosen = tl.load(
        Indices + row * TOPK + check_offsets,
        check_offsets < tl.minimum(check_begin + CHUNK, TOPK),
        -1,
    )
    check_valid = (chosen >= 0) & (chosen < TOKENS)
    safe = tl.where(check_valid, chosen, 0)
    key_scale = tl.load(KS + safe, check_valid, 0.0)
    q_max = tl.max(q_square, 0)
    # Under the score cap, exp(x) is between exp(-cap) and exp(cap).
    # For theta, sum exp(x) * (k_scale - theta) <= 0 certifies
    # a bound on the weighted mean scale without tracking weights in the loop.
    # Limit the probability-weighted mean scale to keep the PV error below
    # the output's absolute tolerance after two E4M3 roundings.
    scale_limit: tl.constexpr = 0.00105
    deviation = tl.where(check_valid, key_scale - scale_limit, 0.0)
    excess = tl.sum(tl.maximum(deviation, 0.0), 0)
    slack = tl.sum(tl.maximum(-deviation, 0.0), 0)
    scale_ok = (tl.max(key_scale, 0) <= 2.0**-9) & (
        tl.min(tl.where(check_valid, key_scale, 1.0), 0) >= 2.0**-20
    )
    eligible = tl.full((), False, tl.int1)
    if GlobalNorm is not None:
        # A proof over all keys also covers any selected subset and key zero.
        eligible = scale_ok & v32_norm_bound(
            q_max, tl.load(GlobalNorm), excess, slack, SCALE
        )
    if GlobalNorm is None or (scale_ok & ~eligible):
        key_square = tl.load(KeyNorm + safe, check_valid, 0.0)
        eligible = scale_ok & v32_norm_bound(
            q_max, tl.max(key_square, 0), excess, slack, SCALE
        )
        if CHECK_SENTINEL:
            # The unmasked cache copies use key zero for invalid indices.
            if not (tl.load(KeyNorm) < float("inf")):
                eligible &= tl.min(check_valid.to(tl.int32), 0) != 0
    return eligible


@triton.jit
def v32_centered_eligible(
    q_data,
    qr_data,
    qscale,
    Indices,
    KS,
    KeyNorm,
    row,
    split,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    CHECK_SENTINEL: tl.constexpr,
):
    # The squared norm includes the physical FP32 scales.
    q_square = (
        tl.sum(q_data.to(tl.float32) * q_data.to(tl.float32), 1)
        + tl.sum(qr_data.to(tl.float32) * qr_data.to(tl.float32), 1)
    ) * (qscale * qscale)
    valid_scale = (qscale >= 2.0**-20) & (qscale <= 2.0**-6)
    q_square = tl.where(valid_scale & (q_square >= 0.0), q_square, float("inf"))
    return v32_norm_eligible(
        q_square,
        Indices,
        KS,
        KeyNorm,
        row,
        split,
        TOKENS,
        TOPK,
        CHUNK,
        SCALE,
        CHECK_SENTINEL,
    )


@triton.jit
def prequalify_v32_local_query(
    Q,
    QR,
    QS,
    KS,
    Indices,
    Norms,
    Flags,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    BadRows,
    BadCount,
    GlobalNorm,
):
    row = tl.program_id(0).to(tl.int64)
    eligible = tl.load(GlobalNorm) <= 1024.0
    if not eligible:
        # With no usable global bound, check the selected keys before starting QK.
        head = tl.arange(0, 16)
        dim = tl.arange(0, 256)
        rope_dim = tl.arange(0, 64)
        q_max = tl.full((), 0.0, tl.float32)
        for group in range(4):
            h = row * 64 + group * 16 + head
            offset = h[:, None] * 512 + dim[None, :]
            lower = tl.load(Q + offset).to(tl.float32)
            upper = tl.load(Q + offset + 256).to(tl.float32)
            rope = tl.load(QR + h[:, None] * 64 + rope_dim[None, :]).to(tl.float32)
            gain = tl.load(QS + h)
            content_square = tl.sum(tl.fma(lower, lower, upper * upper), 1)
            sq = (content_square + tl.sum(rope * rope, 1)) * (gain * gain)
            valid = (sq >= 0.0) & (gain >= 2.0**-20) & (gain <= 2.0**-6)
            q_max = tl.maximum(q_max, tl.max(tl.where(valid, sq, float("inf")), 0))
        eligible = v32_norm_eligible(
            tl.full((1,), q_max, tl.float32),
            Indices,
            KS,
            Norms,
            row,
            0,
            TOKENS,
            TOPK,
            TOPK,
            SCALE,
            True,
            None,
        )
    tl.store(Flags + row, ~eligible)
    if not eligible:
        pos = tl.atomic_add(BadCount, 1, sem="relaxed")
        tl.store(BadRows + pos, row)


@triton.jit
def prequalify_v32_scale_rows(
    QueryMax,
    KS,
    Indices,
    Norms,
    Flags,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    GROUPS: tl.constexpr,
    GlobalNorm,
    BadRows,
    BadCount,
    Coeff,
):
    row = tl.program_id(0).to(tl.int64)
    sq = tl.load(QueryMax + row * GROUPS + tl.arange(0, GROUPS))
    qsq = tl.max(sq, 0)
    ksq = tl.load(GlobalNorm)
    cap = tl.sqrt(qsq) * tl.sqrt(ksq) * (1.001 * SCALE)
    use_global = (cap <= 0.625) & (qsq <= 1024.0) & (ksq <= 1024.0)
    eligible = use_global
    if not use_global:
        eligible = v32_norm_eligible(
            sq, Indices, KS, Norms, row, 0, TOKENS, TOPK, TOPK, SCALE, True, None
        )
    tl.store(Flags + row, ~eligible)
    tl.store(Coeff + row, tl.where(use_global, 1.00001 * tl.exp(2.0 * cap), 0.0))
    if not eligible:
        slot = tl.atomic_add(BadCount, 1, sem="relaxed")
        tl.store(BadRows + slot, row)


@triton.jit
def certify_v32_rows(
    QueryMax,
    KS,
    Indices,
    Norms,
    Flags,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    GROUPS: tl.constexpr,
    GlobalNorm=None,
    BadRows=None,
    BadCount=None,
):
    row = tl.program_id(0).to(tl.int64)
    q_square = tl.load(QueryMax + row * GROUPS + tl.arange(0, GROUPS))
    eligible = v32_norm_eligible(
        q_square,
        Indices,
        KS,
        Norms,
        row,
        0,
        TOKENS,
        TOPK,
        TOPK,
        SCALE,
        True,
        GlobalNorm,
    )
    tl.store(Flags + row, ~eligible)
    if BadCount is not None:
        if not eligible:
            position = tl.atomic_add(BadCount, 1, sem="relaxed")
            tl.store(BadRows + position, row)


@triton.jit
def _sparse_fp8_v32_partition(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Partial,
    Stats,
    Output,
    LSE,
    row,
    split,
    flag,
    partial_slot,
    start,
    end,
    full,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    KeyNorm,
    CENTERED: tl.constexpr,
    FULL_ROWS: tl.constexpr,
):
    H: tl.constexpr = 64
    N: tl.constexpr = 128
    h = tl.arange(0, H)
    n = tl.arange(0, N)
    dims = tl.arange(0, 512)
    rdims = tl.arange(0, 64)
    rope_dtype: tl.constexpr = QR.dtype.element_ty
    q = tle.gpu.alloc((H, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    qr = tle.gpu.alloc((H, 64), rope_dtype, nv_mma_shared_layout=True)
    sk = tle.gpu.alloc((N, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    skr = tle.gpu.alloc((N, 64), rope_dtype, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((H, N), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((H, N), tl.float8e4nv, nv_mma_shared_layout=True)
    q_data = tl.load(Q + (row * H + h[:, None]) * 512 + dims[None, :])
    qr_data = tl.load(QR + (row * H + h[:, None]) * 64 + rdims[None, :])
    qscale = tl.load(QS + row * H + h)
    if CENTERED:
        eligible = v32_centered_eligible(
            q_data,
            qr_data,
            qscale,
            Indices,
            KS,
            KeyNorm,
            row,
            split,
            TOKENS,
            TOPK,
            CHUNK,
            SCALE,
            False,
        )
        tl.store(flag, ~eligible)
    else:
        eligible = True
        q_l1 = tl.sum(tl.abs(q_data.to(tl.float32)), 1) * qscale
        qr_l1 = tl.sum(tl.abs(qr_data.to(tl.float32)), 1) * qscale
    if eligible:
        tl.store(tle.gpu.local_ptr(q), q_data)
        tl.store(tle.gpu.local_ptr(qr), qr_data)
        lma_wait(tl.full((), 0, tl.int32))
        tl.debug_barrier()
        accum = tl.full((H, 512), 0.0, tl.float32)
        if CENTERED:
            sums = tl.full((H, N), 0.0, tl.float32)
        else:
            total = tl.full((H,), 0.0, tl.float32)
            m = tl.full((H,), float("-inf"), tl.float32)
        needs_repair = tl.full((), False, tl.int1)
        for begin in range(start, end, N):
            offsets = begin + n
            idx = tl.load(Indices + row * TOPK + offsets, offsets < end, -1)
            valid = (offsets < end) & (idx >= 0) & (idx < TOKENS)
            idx_safe = tl.where(valid, idx, 0)
            tle.gpu.copy(
                KV + idx_safe[:, None].to(tl.int64) * 512 + dims[None, :],
                sk,
                (N, 512),
                mask=tl.broadcast_to(valid[:, None], (N, 512)),
            )
            rk = tl.load(
                KR + idx_safe[:, None].to(tl.int64) * 64 + rdims[None, :],
                valid[:, None],
                0.0,
            )
            ks = tl.load(KS + idx_safe, valid, 0)
            if not CENTERED:
                max_ks = tl.max(ks, 0)
                max_kr = tl.max(tl.max(tl.abs(rk.to(tl.float32)), 1) * ks, 0)
                # Apply the general kernel's QK error bound to each split.
                error_bound = q_l1 * (448.0 * max_ks) * SCALE * (
                    (512.0 * 2.0**-24) / (1.0 - 512.0 * 2.0**-24)
                ) + qr_l1 * max_kr * SCALE * (
                    (64.0 * 2.0**-24) / (1.0 - 64.0 * 2.0**-24)
                )
                needs_repair |= tl.max(error_bound, 0) > 1.0e-4
            tl.store(tle.gpu.local_ptr(skr), rk)
            lma_wait(tl.full((), 0, tl.int32))
            tl.debug_barrier()
            dot = tle.gpu.wgmma(q, sk, trans_b=True)
            dot = tle.gpu.wgmma_wait(0, dot)
            rope = tle.gpu.wgmma(qr, skr, trans_b=True)
            rope = tle.gpu.wgmma_wait(0, rope)
            scores = (dot + rope) * (qscale[:, None] * ks[None, :] * SCALE)
            if CENTERED:
                probability = tl.exp(tl.where(valid[None, :], scores, float("-inf")))
                sums += probability
                norm = probability * (ks[None, :] * 16384.0)
                high = norm.to(tl.float8e4nv)
                low = (norm - high.to(tl.float32)).to(tl.float8e4nv)
                tl.store(tle.gpu.local_ptr(ph), high)
                tl.store(tle.gpu.local_ptr(pl), low)
                lma_wait(tl.full((), 0, tl.int32))
                tl.debug_barrier()
                accum = tle.gpu.wgmma(ph, sk, acc=accum)
                accum = tle.gpu.wgmma(pl, sk, acc=accum)
                accum = tle.gpu.wgmma_wait(0, accum)
            else:
                scores = tl.where(valid[None, :], scores, float("-inf"))
                next_max = tl.maximum(m, tl.max(scores, 1))
                safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
                correction = tl.exp(m - safe_max)
                p = tl.exp(scores - safe_max[:, None])
                total = total * correction + tl.sum(p, 1)
                w = p * ks[None, :]
                pscale = tl.maximum(tl.max(w, 1) / 448.0, 1.0e-30)
                norm = w / pscale[:, None]
                hi = norm.to(tl.float8e4nv)
                lo = ((norm - hi.to(tl.float32)) * 16.0).to(tl.float8e4nv)
                tl.store(tle.gpu.local_ptr(ph), hi)
                tl.store(tle.gpu.local_ptr(pl), lo)
                lma_wait(tl.full((), 0, tl.int32))
                tl.debug_barrier()
                prod = tle.gpu.wgmma(ph, sk)
                prod = tle.gpu.wgmma_wait(0, prod)
                prod = tle.gpu.wgmma(pl, sk, acc=prod * 16.0)
                prod = tle.gpu.wgmma_wait(0, prod)
                accum = accum * correction[:, None] + prod * (
                    pscale[:, None] * (1.0 / 16.0)
                )
                m = next_max
            tl.debug_barrier()
        if CENTERED:
            total = tl.sum(sums, 1)
            accum *= 1.0 / 16384.0
            m = tl.where(total > 0.0, 0.0, float("-inf"))
        tl.store(flag, needs_repair)
        if full if FULL_ROWS else SPLITS == 1:
            inverse = tl.div_rn(1.0, tl.where(total > 0, total, 1.0))
            tl.store(
                Output + (row * H + h[:, None]) * 512 + dims[None, :],
                accum * inverse[:, None],
            )
            tl.store(
                LSE + row * H + h,
                tl.where(total > 0, m + tl.log(total), float("inf")),
            )
        else:
            base = partial_slot * H + h
            tl.store(Partial + base[:, None] * 512 + dims[None, :], accum)
            tl.store(Stats + base * 2, m)
            tl.store(Stats + base * 2 + 1, total)


@triton.jit
def sparse_fp8_v32(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Partial,
    Stats,
    Output,
    LSE,
    RepairFlags,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    KeyNorm,
    CENTERED: tl.constexpr,
    REPLAY: tl.constexpr,
    FULL_ROWS: tl.constexpr = 0,
    TailDenominators=None,
):
    pid = tl.program_id(0).to(tl.int64)
    split = tl.program_id(2)
    if FULL_ROWS:
        row, full, start, end = mixed_v32_span(pid, TOPK, SPLITS, CHUNK, FULL_ROWS)
        flag = RepairFlags + pid
        partial_slot = pid - FULL_ROWS
    else:
        row = pid
        start = split * CHUNK
        end = tl.minimum(start + CHUNK, TOPK)
        partial_slot = row * SPLITS + split
        flag = RepairFlags + partial_slot
    if REPLAY:
        active = tl.load(flag)
        if TailDenominators is not None:
            tl.static_assert(FULL_ROWS > 0 and not CENTERED)
            if not full and not active:
                active = not collect_v32_tail_denominator(
                    TailDenominators, Stats, partial_slot
                )
    else:
        active = True
    if active:
        _sparse_fp8_v32_partition(
            Q,
            QR,
            KV,
            KR,
            QS,
            KS,
            Indices,
            Partial,
            Stats,
            Output,
            LSE,
            row,
            split,
            flag,
            partial_slot,
            start,
            end,
            full if FULL_ROWS else None,
            TOKENS,
            TOPK,
            SPLITS,
            CHUNK,
            SCALE,
            KeyNorm,
            CENTERED,
            FULL_ROWS,
        )


@triton.jit
def compact_v32_stable(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Output,
    LSE,
    RepairFlags,
    BadRows,
    BadCount,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
):
    count = tl.load(BadCount)
    for position in range(tl.program_id(0), count, tl.num_programs(0)):
        row = tl.load(BadRows + position).to(tl.int64)
        _sparse_fp8_v32_partition(
            Q,
            QR,
            KV,
            KR,
            QS,
            KS,
            Indices,
            None,
            None,
            Output,
            LSE,
            row,
            0,
            RepairFlags + row,
            row,
            0,
            TOPK,
            None,
            TOKENS,
            TOPK,
            1,
            TOPK,
            SCALE,
            None,
            False,
            0,
        )


@triton.jit
def prepare_v32_key_norm(KV, KR, KS, KNorm, TOKENS: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0) * B + tl.arange(0, B)
    dims = tl.arange(0, 512)
    rdims = tl.arange(0, 64)
    k = tl.load(
        KV + token[:, None].to(tl.int64) * 512 + dims[None, :],
        token[:, None] < TOKENS,
        0.0,
    ).to(tl.float32)
    rope = tl.load(
        KR + token[:, None].to(tl.int64) * 64 + rdims[None, :],
        token[:, None] < TOKENS,
        0.0,
    ).to(tl.float32)
    scale = tl.load(KS + token, token < TOKENS, 0.0)
    squared = (tl.sum(k * k, 1) + tl.sum(rope * rope, 1)) * (scale * scale)
    squared = tl.where(squared >= 0.0, squared, float("inf"))
    tl.store(KNorm + token, squared, token < TOKENS)


@triton.jit
def prepare_v32_qk_norm(
    Q,
    QR,
    QS,
    KV,
    KR,
    KS,
    QueryMax,
    KeyNorm,
    TOKENS: tl.constexpr,
    B: tl.constexpr,
    GroupNorm=None,
):
    block = tl.program_id(0)
    is_key = block < TOKENS // B
    if is_key:
        content, rope, scale = KV, KR, KS
        group = block
    else:
        content, rope, scale = Q, QR, QS
        group = block - TOKENS // B
    row = group * B + tl.arange(0, B)
    offset = row[:, None].to(tl.int64)
    address = content + offset * 512 + tl.arange(0, 256)[None, :]
    lower = tl.load(address).to(tl.float32)
    upper = tl.load(address + 256).to(tl.float32)
    content_square = tl.sum(tl.fma(lower, lower, upper * upper), 1)
    rope_values = tl.load(rope + offset * 64 + tl.arange(0, 64)[None, :]).to(tl.float32)
    gain = tl.load(scale + row)
    square = (content_square + tl.sum(rope_values * rope_values, 1)) * (gain * gain)
    if is_key:
        square = tl.where(square >= 0.0, square, float("inf"))
        tl.store(KeyNorm + row, square)
        if GroupNorm is not None:
            tl.store(GroupNorm + group, tl.max(square, 0))
    else:
        # Encode any invalid head so the whole query fails the later certificate.
        valid = (square >= 0.0) & (gain >= 2.0**-20) & (gain <= 2.0**-6)
        tl.store(QueryMax + group, tl.max(tl.where(valid, square, float("inf")), 0))


@triton.jit
def reduce_v32_key_norm(
    GroupNorm, GlobalNorm, GROUPS: tl.constexpr, BLOCK: tl.constexpr, BadCount=None
):
    group = tl.arange(0, BLOCK)
    maximum = tl.max(tl.load(GroupNorm + group, group < GROUPS, 0.0), 0)
    tl.store(GlobalNorm, maximum)
    if BadCount is not None:
        tl.store(BadCount, 0)


@triton.jit
def prepare_v32_rope_peak(KR, KS, Peak, TOKENS: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0) * B + tl.arange(0, B)
    channel = tl.arange(0, 64)
    rope = tl.load(
        KR + token[:, None] * 64 + channel[None, :], token[:, None] < TOKENS, 0.0
    ).to(tl.float32)
    magnitude = tl.abs(rope)
    magnitude = tl.where(magnitude < float("inf"), magnitude, float("inf"))
    scale = tl.load(KS + token, token < TOKENS, 0.0)
    scaled = tl.max(magnitude, 1) * scale
    scaled = tl.where(scaled >= 0.0, scaled, float("inf"))
    tl.store(Peak + token, scaled, token < TOKENS)


@lru_cache(maxsize=64)
def _v32_direct_layout_matches(kernel, copy_mode):
    # The native copy inverts the MP31 shared-memory swizzle at (0, 0).
    # Validate the actual allocated tile encodings before launching it.
    ir = kernel.asm.get("ttgir", "")
    encodings = dict(
        re.findall(r"^(#[a-zA-Z0-9_]+) = (#ttg\.swizzled_shared<.+>)$", ir, re.M)
    )
    kv = ("vec = 16", "perPhase = 1", "maxPhase = 16")
    expected = [
        ("2x128x256xf8E4M3FN", (*kv, "order = [2, 1, 0]")),
        (
            "128x256xf8E4M3FN" if copy_mode == 1 else r"2x128x64x(?:bf16|f16)",
            (
                (*kv, "order = [1, 0]")
                if copy_mode == 1
                else ("vec = 8", "perPhase = 2", "maxPhase = 16", "order = [2, 1, 0]")
            ),
        ),
    ]
    for shape, fields in expected:
        allocations = re.findall(
            r"ttg\.local_alloc[^\n]*?!ttg\.memdesc<"
            + shape
            + r", (#[a-zA-Z0-9_]+), #smem, mutable>",
            ir,
        )
        if len(allocations) != 1:
            return False
        encoding = encodings.get(allocations[0], "")
        if not all(field in encoding for field in fields):
            return False
    return True


def _warmup_v32_pipeline(pipeline, args, options, copy_mode=0, scale_check_args=()):
    options = dict(options)
    certified_poly = options.pop("CERTIFIED_POLY", False)
    static_sq64 = options.pop("STATIC_SQ64", False)
    local_query_norm = options.pop("LOCAL_QUERY_NORM", False)
    tail_denominators = options.pop("TAIL_DENOMINATORS", None)
    loader_scale_check = options.pop("LOADER_SCALE_CHECK", False)
    trailing = (
        certified_poly,
        static_sq64,
        *(scale_check_args or (None, None, None)),
        local_query_norm,
        tail_denominators,
        loader_scale_check,
    )

    def bind(compiled, mode):
        launch = compiled[options["grid"]]

        def run(*launch_args):
            launch(
                *launch_args,
                mode,
                certified_poly,
                static_sq64,
                *(scale_check_args or (None, None, None)),
                local_query_norm,
                tail_denominators,
                loader_scale_check,
            )

        return run

    if copy_mode:
        target = triton.runtime.driver.active.get_current_target()
        inputs_aligned = args[2].data_ptr() % 16 == 0 and (
            copy_mode == 1 or args[3].data_ptr() % 16 == 0
        )
        if (
            target.backend == "musa"
            and str(target.arch) == "31"
            and torch.musa.get_device_capability(args[0].device) == (3, 1)
            and args[11] % 128 == 0
            and args[13] % 128 == 0
            and inputs_aligned
        ):
            try:
                copy_ir = (
                    V32_DIRECT_COPY_TWO_LANE_IR
                    if copy_mode == 2 and local_query_norm
                    else V32_DIRECT_COPY_IR
                )
                direct_options = dict(
                    options,
                    extern_libs={"native": native_library(VECTOR_EXP2_IR + copy_ir)},
                )
                compiled = pipeline.warmup(
                    *args, copy_mode, *trailing, **direct_options
                )
                if _v32_direct_layout_matches(compiled, copy_mode):
                    return bind(compiled, copy_mode)
            except (CompilationError, OutOfResources, RuntimeError) as error:
                logging.getLogger(__name__).debug(
                    "V32 direct copy unavailable: %s", error
                )
    return bind(pipeline.warmup(*args, 0, *trailing, **options), 0)


@triton.jit
def finalize_v32_tail_denominator(Denominators, Stats, Flags, FULL_ROWS: tl.constexpr):
    split = tl.program_id(0)
    flag_ptr = Flags + FULL_ROWS + split
    if not tl.load(flag_ptr):
        if not collect_v32_tail_denominator(Denominators, Stats, split):
            tl.store(flag_ptr, True)


def prepare_mixed_v32(
    inputs,
    output,
    lse,
    scale,
    repair,
    mixed_repair,
    combine,
    finish_mixed_kernel,
    *,
    loader_scale_check=None,
):
    rows, topk = inputs[0].shape[0], inputs[-1].shape[-1]
    if loader_scale_check is None:
        loader_scale_check = rows in (132, 193) and topk == 2048
    tokens = inputs[2].shape[0] * 64
    # Fill whole waves before distributing the final rows across short splits.
    full_rows = (rows // 60) * 60
    splits = {64: 8, 132: 4, 193: 4, 449: 2}[rows]
    tail_rows = rows - full_rows
    chunk = topk // splits
    flags = torch.empty(
        (full_rows + tail_rows * splits,), dtype=torch.bool, device=output.device
    )
    partial = torch.empty(
        (tail_rows, splits, 64, 512), dtype=torch.float32, device=output.device
    )
    stats = torch.empty(
        (tail_rows, splits, 64, 2), dtype=torch.float32, device=output.device
    )

    def prepare_replay(lo, hi, count, partial, stats, flags, mixed_rows=0):
        q, qr, kv, kr, qs, ks, indices = inputs
        local_inputs = (q[lo:hi], qr[lo:hi], kv, kr, qs[lo:hi], ks, indices[lo:hi])
        y, local_lse = output[lo:hi], lse[lo:hi]
        n = hi - lo
        chunk = topk // count
        args = (
            *local_inputs,
            partial,
            stats,
            y,
            local_lse,
            flags,
            tokens,
            topk,
            count,
            chunk,
            scale,
            None,
            False,
            True,
            mixed_rows,
            tail_denominators if mixed_rows else None,
        )
        grid = (flags.numel(), 1, 1) if mixed_rows else (n, 1, count)
        safe = sparse_fp8_v32.warmup(
            *args,
            grid=grid,
            num_warps=16,
            num_stages=1,
            enable_backend_opt=True,
            extern_libs={"native": native_library()},
        )[grid]
        repair_args = (
            *local_inputs,
            None,
            partial,
            stats,
            y,
            local_lse,
            None,
            n,
            64,
            1,
            topk,
            tokens,
            64,
            scale,
            False,
            False,
            count,
            chunk,
            32,
            64,
            flags,
            mixed_rows,
        )
        grid = (flags.numel(), 2, 1) if mixed_rows else (n, 2, count)
        exact = repair.warmup(*repair_args, grid=grid, num_warps=4, num_stages=1)[grid]
        return (safe, args), (exact, repair_args)

    try:
        from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_v32_pipeline import (
            qualify_v32_scales,
            sparse_fp8_v32_pipeline,
        )

        tail_denominators = None
        finalize_tail = None
        if topk == 2048 and (rows == 64 or loader_scale_check):
            tail_denominators = torch.empty(
                (tail_rows, splits, 64, 4), dtype=torch.float32, device=output.device
            )

        preload_args = (
            *inputs,
            output,
            lse,
            flags,
            tokens,
            topk,
            splits,
            chunk,
            scale,
            128,
            16,
            8,
            True,
            inputs[5],
            False,
            True,
        )
        grid = (flags.numel(), 1, 1)
        pipeline_options = dict(
            grid=grid,
            num_warps=16,
            num_stages=1,
            enable_backend_opt=True,
            STATIC_SQ64=rows == 64,
            TAIL_DENOMINATORS=tail_denominators,
            LOADER_SCALE_CHECK=loader_scale_check,
            extern_libs={"native": native_library(VECTOR_EXP2_IR)},
        )
        check = None
        cutoff = again = flags
        if rows != 64 and not loader_scale_check:
            cutoff = torch.empty_like(flags, dtype=torch.float32)
            again = torch.empty_like(flags)
            check_args = (
                inputs[2],
                inputs[5],
                inputs[6],
                cutoff,
                flags,
                again,
                tokens,
                topk,
                splits,
                chunk,
                full_rows,
            )
            check = qualify_v32_scales.warmup(*check_args, grid=grid, num_warps=4)[grid]

        def prepare_pipeline(static_splits):
            args = (*preload_args, cutoff, again)
            tail = (partial, stats, full_rows, static_splits)
            pipeline_args = (*args, 0 if check is None else 1, *tail)
            pipeline = _warmup_v32_pipeline(
                sparse_fp8_v32_pipeline, pipeline_args, pipeline_options
            )
            scale_precheck = None
            if check is not None:
                repeat_args = (*args, 2, *tail)
                repeat = _warmup_v32_pipeline(
                    sparse_fp8_v32_pipeline, repeat_args, pipeline_options
                )
                scale_precheck = check, check_args, repeat, repeat_args
            return pipeline, pipeline_args, scale_precheck

        try:
            pipeline, pipeline_args, scale_precheck = prepare_pipeline(True)
        except (CompilationError, OutOfResources, RuntimeError):
            if rows == 449 or tail_denominators is not None:
                # Use the uniform pipeline if static specialization is unavailable.
                raise
            pipeline, pipeline_args, scale_precheck = prepare_pipeline(False)
        if check is None and tail_denominators is not None:
            heads = 16 if rows == 193 else 8
            finish_args = (
                *inputs,
                partial,
                output,
                lse,
                flags,
                tail_denominators,
                tokens,
                topk,
                splits,
                scale,
                full_rows,
                heads,
            )
            finish_grid = (rows, 64 // heads, 1)
            try:
                finish = finish_mixed_kernel.warmup(
                    *finish_args, grid=finish_grid, num_warps=4, num_stages=1
                )[finish_grid]
            except (CompilationError, OutOfResources, RuntimeError) as error:
                logging.getLogger(__name__).debug(
                    "V32 combined finish unavailable: %s", error
                )
            else:

                def finish_mixed():
                    pipeline(*pipeline_args)
                    finish(*finish_args)

                return finish_mixed
        mixed_args = (
            *inputs,
            partial,
            stats,
            output,
            lse,
            flags,
            tail_denominators,
            tokens,
            topk,
            splits,
            chunk,
            scale,
            full_rows,
        )
        repair_grid = (flags.numel(), 2, 1)
        try:
            fused_repair = mixed_repair.warmup(
                *mixed_args, grid=repair_grid, num_warps=4, num_stages=1
            )[repair_grid]
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug(
                "V32 fused exact replay unavailable: %s", error
            )
            try:
                replays = prepare_replay(
                    0, rows, splits, partial, stats, flags, full_rows
                )
            except (CompilationError, OutOfResources, RuntimeError) as error:
                logging.getLogger(__name__).debug(
                    "V32 combined replay unavailable: %s", error
                )
                replays = (
                    *prepare_replay(0, full_rows, 1, None, None, flags[:full_rows]),
                    *prepare_replay(
                        full_rows, rows, splits, partial, stats, flags[full_rows:]
                    ),
                )
                if tail_denominators is not None:
                    finalize_args = (tail_denominators, stats, flags, full_rows)
                    tail_grid = (tail_rows * splits, 1, 1)
                    finalize_tail = finalize_v32_tail_denominator.warmup(
                        *finalize_args, grid=tail_grid, num_warps=4
                    )[tail_grid]
        else:
            replays = ((fused_repair, mixed_args),)
        merge_args = (
            partial,
            stats,
            None,
            output[full_rows:],
            lse[full_rows:],
            tail_rows,
            64,
            1,
            splits,
            False,
            splits,
            512,
        )
        grid = (tail_rows, 64, 1)
        merge = combine.warmup(*merge_args, grid=grid, num_warps=4)[grid]
    except (ImportError, CompilationError, OutOfResources, RuntimeError) as error:
        logging.getLogger(__name__).debug("V32 mixed splits unavailable: %s", error)
        if loader_scale_check:
            return prepare_mixed_v32(
                inputs,
                output,
                lse,
                scale,
                repair,
                mixed_repair,
                combine,
                finish_mixed_kernel,
                loader_scale_check=False,
            )
        return None

    def run():
        if scale_precheck is not None:
            check, check_args, repeat, repeat_args = scale_precheck
            check(*check_args)
        pipeline(*pipeline_args)
        if scale_precheck is not None:
            repeat(*repeat_args)
        if finalize_tail is not None:
            finalize_tail(*finalize_args)
        for replay, args in replays:
            replay(*args)
        merge(*merge_args)

    return run


def prepare_sparse_fp8_v32(
    inputs,
    partial,
    stats,
    output,
    lse,
    scale,
    splits,
    chunk,
    repair,
    repair_args,
    compact_repair,
):
    rows, topk = inputs[0].shape[0], inputs[-1].shape[-1]
    tokens = inputs[2].shape[0] * 64
    flags = torch.empty((rows, splits), dtype=torch.bool, device=output.device)
    grid = (rows, 1, splits)
    repair_grid = (rows, 2, splits)
    args = (
        *inputs,
        partial,
        stats,
        output,
        lse,
        flags,
        tokens,
        topk,
        splits,
        chunk,
        scale,
    )
    repair_args = (*repair_args, 32, 64, flags, 0)
    options = dict(
        grid=grid,
        num_warps=16,
        num_stages=1,
        enable_backend_opt=True,
        extern_libs={"native": native_library()},
    )
    centered = pipeline = prepare_peak = bound_reduce = None
    scale_precheck = compact_replay = None
    observed = 64 <= rows <= 512
    use_norms = rows > 512 and splits == 1 and tokens <= 1024 * rows
    if (observed or use_norms) and 0 < tokens <= (1 << 20) and scale <= 1.0:
        try:
            if use_norms:
                norms = torch.empty(
                    (tokens,), dtype=torch.float32, device=output.device
                )
            try:
                from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_v32_pipeline import (
                    qualify_v32_scales,
                    sparse_fp8_v32_pipeline,
                )

                base_pipeline_args = (
                    *inputs,
                    output if splits == 1 else partial,
                    lse if splits == 1 else stats,
                    flags,
                    tokens,
                    topk,
                    splits,
                    chunk,
                    scale,
                    128,
                    16,
                    8,
                    observed,
                )
                if use_norms:
                    norm_block = 16
                    groups = 64 // norm_block
                    query_max = torch.empty(
                        (rows, groups), dtype=torch.float32, device=output.device
                    )
                    norm_args = (
                        inputs[0],
                        inputs[1],
                        inputs[4],
                        inputs[2],
                        inputs[3],
                        inputs[5],
                        query_max,
                        norms,
                        tokens,
                        norm_block,
                    )
                    # Both inputs have a whole number of 64-row blocks.
                    norm_grid = ((tokens + rows * 64) // norm_block, 1, 1)
                    certify_args = (
                        query_max,
                        inputs[5],
                        inputs[6],
                        norms,
                        flags,
                        tokens,
                        topk,
                        scale,
                        groups,
                    )
                    key_groups = tokens // norm_block
                    group_norm = torch.empty(
                        (key_groups,), dtype=torch.float32, device=output.device
                    )
                    global_norm = torch.empty(
                        (), dtype=torch.float32, device=output.device
                    )
                    bound_grid = (1, 1, 1)
                    bound_args = (
                        group_norm,
                        global_norm,
                        key_groups,
                        triton.next_power_of_2(key_groups),
                    )
                    try:
                        prepare_norm = prepare_v32_qk_norm.warmup(
                            *norm_args, group_norm, grid=norm_grid, num_warps=4
                        )[norm_grid]
                        reduce_bound = reduce_v32_key_norm.warmup(
                            *bound_args, None, grid=bound_grid, num_warps=8
                        )[bound_grid]
                        certify = certify_v32_rows.warmup(
                            *certify_args,
                            global_norm,
                            None,
                            None,
                            grid=grid,
                            num_warps=4,
                        )[grid]
                    except (CompilationError, OutOfResources, RuntimeError) as error:
                        logging.getLogger(__name__).debug(
                            "V32 global norm check unavailable: %s", error
                        )
                        norm_args = (*norm_args, None)
                        certify_args = (*certify_args, None, None, None)
                        prepare_norm = prepare_v32_qk_norm.warmup(
                            *norm_args, grid=norm_grid, num_warps=4
                        )[norm_grid]
                        certify = certify_v32_rows.warmup(
                            *certify_args, grid=grid, num_warps=4
                        )[grid]
                    else:
                        norm_args = (*norm_args, group_norm)
                        certify_args = (*certify_args, global_norm, None, None)
                        bound_reduce = reduce_bound, (*bound_args, None)
                pipeline_options = dict(
                    options,
                    extern_libs={"native": native_library(VECTOR_EXP2_IR)},
                    CERTIFIED_POLY=use_norms and topk <= 2048,
                )
                use_peaks = observed and rows >= 384 and tokens <= 1024 * rows
                try:
                    if use_peaks:
                        peak = torch.empty(
                            (tokens,), dtype=torch.float32, device=output.device
                        )
                        peak_grid = (triton.cdiv(tokens, 16), 1, 1)
                        peak_args = (inputs[3], inputs[5], peak, tokens, 16)
                        prepare_peak = prepare_v32_rope_peak.warmup(
                            *peak_args, grid=peak_grid, num_warps=4
                        )[peak_grid]
                    else:
                        peak = inputs[5]
                    preload_args = (*base_pipeline_args, peak, use_peaks, True)
                    copy_mode = 2 if rows == 512 or use_norms else 0
                    if rows == 512 and observed and splits == 1 and topk == 2048:
                        loader_args = (
                            *preload_args,
                            flags,
                            flags,
                            0,
                            None,
                            None,
                            0,
                            False,
                        )
                        try:
                            pipeline = _warmup_v32_pipeline(
                                sparse_fp8_v32_pipeline,
                                loader_args,
                                dict(pipeline_options, LOADER_SCALE_CHECK=True),
                                copy_mode,
                            )
                        except (
                            CompilationError,
                            OutOfResources,
                            RuntimeError,
                        ) as error:
                            logging.getLogger(__name__).debug(
                                "V32 loader scale check unavailable: %s", error
                            )
                        else:
                            pipeline_args = loader_args
                    if pipeline is None and observed and splits == 1:
                        try:
                            cutoff = torch.empty(
                                (rows, splits),
                                dtype=torch.float32,
                                device=output.device,
                            )
                            again = torch.empty(
                                (rows, splits), dtype=torch.bool, device=output.device
                            )
                            check_grid = (rows, splits, 1)
                            check_args = (
                                inputs[2],
                                inputs[5],
                                inputs[6],
                                cutoff,
                                flags,
                                again,
                                tokens,
                                topk,
                                splits,
                                chunk,
                                0,
                            )
                            check = qualify_v32_scales.warmup(
                                *check_args, grid=check_grid, num_warps=4
                            )[check_grid]
                            fast_args = (
                                *preload_args,
                                cutoff,
                                again,
                                1,
                                None,
                                None,
                                0,
                                False,
                            )
                            again_args = (
                                *preload_args,
                                cutoff,
                                again,
                                2,
                                None,
                                None,
                                0,
                                False,
                            )
                            fast = _warmup_v32_pipeline(
                                sparse_fp8_v32_pipeline,
                                fast_args,
                                pipeline_options,
                                copy_mode,
                            )
                            observed_again = _warmup_v32_pipeline(
                                sparse_fp8_v32_pipeline,
                                again_args,
                                pipeline_options,
                                copy_mode,
                            )
                        except (
                            CompilationError,
                            OutOfResources,
                            RuntimeError,
                        ) as error:
                            logging.getLogger(__name__).debug(
                                "V32 scale precheck unavailable: %s", error
                            )
                        else:
                            scale_precheck = (
                                check,
                                check_args,
                                observed_again,
                                again_args,
                            )
                            pipeline, pipeline_args = fast, fast_args
                    if pipeline is None:
                        pipeline_args = (
                            *preload_args,
                            flags,
                            flags,
                            0,
                            None,
                            None,
                            0,
                            False,
                        )
                        pipeline = _warmup_v32_pipeline(
                            sparse_fp8_v32_pipeline,
                            pipeline_args,
                            pipeline_options,
                            copy_mode,
                        )
                except (CompilationError, OutOfResources, RuntimeError):
                    # Keep the two-partition pipeline available on resource-constrained compilers.
                    prepare_peak = None
                    pipeline_args = (
                        *base_pipeline_args,
                        inputs[5],
                        False,
                        False,
                        flags,
                        flags,
                        0,
                        None,
                        None,
                        0,
                        False,
                    )
                    pipeline = _warmup_v32_pipeline(
                        sparse_fp8_v32_pipeline, pipeline_args, pipeline_options
                    )
            except (
                ImportError,
                CompilationError,
                OutOfResources,
                RuntimeError,
            ) as error:
                logging.getLogger(__name__).debug("V32 pipeline unavailable: %s", error)
            if pipeline is None and use_norms and rows >= 512:
                norm_block = 16
                norm_args = (*inputs[2:4], inputs[5], norms, tokens, norm_block)
                norm_grid = (triton.cdiv(tokens, norm_block), 1, 1)
                prepare_norm = prepare_v32_key_norm.warmup(
                    *norm_args, grid=norm_grid, num_warps=4
                )[norm_grid]
                centered = sparse_fp8_v32.warmup(
                    *args, norms, True, False, 0, None, **options
                )[grid]
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug(
                "Centered V32 sparse FP8 tile unavailable: %s", error
            )
            centered = pipeline = None

    if pipeline is not None and bound_reduce is not None:
        bad_rows = torch.empty((rows,), dtype=torch.int32, device=output.device)
        bad_count = torch.empty((), dtype=torch.int32, device=output.device)
        queued_bound_args = (*bound_reduce[1][:-1], bad_count)
        queued_certify_args = (*certify_args[:-2], bad_rows, bad_count)
        replay_args = (
            *inputs,
            output,
            lse,
            flags,
            bad_rows,
            bad_count,
            tokens,
            topk,
            scale,
        )
        replay_grid = (120, 1, 1)
        exact_grid = (120, 2, 1)
        try:
            queued_reduce = reduce_v32_key_norm.warmup(
                *queued_bound_args, grid=bound_grid, num_warps=8
            )[bound_grid]
            queued_certify = certify_v32_rows.warmup(
                *queued_certify_args, grid=grid, num_warps=4
            )[grid]
            safe = compact_v32_stable.warmup(
                *replay_args, **dict(options, grid=replay_grid)
            )[replay_grid]
            # Four warps can trigger conflicting LLVM fences for FP16 RoPE.
            exact_warps = 8 if inputs[1].dtype == torch.float16 else 4
            exact = compact_repair.warmup(
                *replay_args, grid=exact_grid, num_warps=exact_warps, num_stages=1
            )[exact_grid]
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug(
                "V32 compact replay unavailable: %s", error
            )
        else:
            bound_reduce = queued_reduce, queued_bound_args
            certify, certify_args = queued_certify, queued_certify_args
            compact_replay = safe, exact, replay_args
            if topk == 2048:
                coeff = torch.empty((rows,), dtype=torch.float32, device=output.device)
                fused_certify_args = (*queued_certify_args, coeff)
                try:
                    fused_certify = prequalify_v32_scale_rows.warmup(
                        *fused_certify_args, grid=grid, num_warps=4
                    )[grid]
                    fused_pipeline = _warmup_v32_pipeline(
                        sparse_fp8_v32_pipeline,
                        pipeline_args,
                        pipeline_options,
                        copy_mode,
                        scale_check_args=(coeff, bad_rows, bad_count),
                    )
                except (CompilationError, OutOfResources, RuntimeError) as error:
                    logging.getLogger(__name__).debug(
                        "Fused V32 scale check unavailable: %s", error
                    )
                else:
                    pipeline = fused_pipeline
                    certify, certify_args = fused_certify, fused_certify_args
                    key_grid = (tokens // norm_block, 1, 1)
                    local_certify_args = (
                        inputs[0],
                        inputs[1],
                        inputs[4],
                        inputs[5],
                        inputs[6],
                        norms,
                        flags,
                        tokens,
                        topk,
                        scale,
                        bad_rows,
                        bad_count,
                        global_norm,
                    )
                    try:
                        key_norm = prepare_v32_qk_norm.warmup(
                            *norm_args, grid=key_grid, num_warps=4
                        )[key_grid]
                        local_certify = prequalify_v32_local_query.warmup(
                            *local_certify_args, grid=grid, num_warps=4
                        )[grid]
                        local_pipeline = _warmup_v32_pipeline(
                            sparse_fp8_v32_pipeline,
                            pipeline_args,
                            dict(pipeline_options, LOCAL_QUERY_NORM=True),
                            copy_mode,
                            scale_check_args=(global_norm, bad_rows, bad_count),
                        )
                    except (CompilationError, OutOfResources, RuntimeError) as error:
                        logging.getLogger(__name__).debug(
                            "Local V32 query norm unavailable: %s", error
                        )
                    else:
                        prepare_norm = key_norm
                        pipeline = local_pipeline
                        certify, certify_args = local_certify, local_certify_args

    if compact_replay is None:
        try:
            main = sparse_fp8_v32.warmup(
                *args,
                None,
                False,
                centered is not None or pipeline is not None,
                0,
                None,
                **options,
            )[grid]
            repair = repair.warmup(
                *repair_args, grid=repair_grid, num_warps=4, num_stages=1
            )[repair_grid]
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug(
                "V32 sparse FP8 tile unavailable: %s", error
            )
            return None

    def run():
        if pipeline is not None:
            if use_norms:
                prepare_norm(*norm_args)
                if bound_reduce is not None:
                    reduce_bound, bound_args = bound_reduce
                    reduce_bound(*bound_args)
                certify(*certify_args)
            elif prepare_peak is not None:
                prepare_peak(*peak_args)
            if scale_precheck is not None:
                check, check_args, observed_again, again_args = scale_precheck
                check(*check_args)
                pipeline(*pipeline_args)
                observed_again(*again_args)
            else:
                pipeline(*pipeline_args)
        elif centered is not None:
            prepare_norm(*norm_args)
            centered(*args, norms, True, False, 0, None)
        if compact_replay is None:
            main(
                *args,
                None,
                False,
                centered is not None or pipeline is not None,
                0,
                None,
            )
            repair(*repair_args)
        else:
            safe, exact, replay_args = compact_replay
            safe(*replay_args)
            exact(*replay_args)

    return run
