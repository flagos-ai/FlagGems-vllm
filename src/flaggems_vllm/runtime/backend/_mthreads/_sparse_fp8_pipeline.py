# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton._C.libtriton import mthreads as musa_compiler
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError
from triton.experimental.tle import language as tle
from triton.experimental.tle.language.gpu.mthreads.types import MusaSqmmaEncoding
from triton.runtime import driver
from triton.runtime.errors import OutOfResources
from triton.tools.tensor_descriptor import TensorDescriptor

from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_metadata import (
    BOUNDED_EXP_PV_BUDGET,
    FAST_BOUNDED_EXP,
    KV_OFFSET_TOKEN_LIMIT,
    KV_TOKEN_BYTES,
    PACKED_METADATA_TOKEN_LIMIT,
    cache_scale_bounds,
    encode_kv_byte_offset,
    gather_scales,
    query_bounds_global,
    reduce_scales,
)
from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_native import (
    QUERY_NORM_IR,
    VECTOR_EXP2_IR,
    bounded_copy16,
    bounded_copy_wait,
    exp2_pair,
    lma_wait,
    native_library,
    promote_fp8_four,
    vector_exp2,
)
from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_scheduling import (
    group_short_pages,
    prepare_persistent_combine,
)

# The mixed path reuses FP16 operands and crosses SQMMA accumulator layouts.
# Older FlagTree builds compile it but produce incorrect values.
_MIXED_PV_SUPPORTED = tl.constexpr(
    all(
        getattr(musa_compiler, feature, lambda: False)()
        for feature in (
            "supports_tle_sqmma_m16_m_split",
            "supports_tle_barrier_memory_ordering",
        )
    )
)

PV = tl.constexpr(MusaSqmmaEncoding([3, 1], [4, 4], [64, 128, 128]))
OROW = tl.constexpr(
    tle.gpu.SlicedEncoding(1, MusaSqmmaEncoding([3, 1], [4, 4], [64, 128, 128]))
)
OCOL = tl.constexpr(
    tle.gpu.SlicedEncoding(0, MusaSqmmaEncoding([3, 1], [4, 4], [64, 128, 128]))
)


@triton.jit
def consume(
    Output,
    PScale,
    ph,
    pl,
    sk,
    FinalInv,
    Full,
    PEmpty,
    NormReady,
    Active,
    KV,
    Locations,
    CopyReady,
    TOKENS: tl.constexpr,
    USE_BYTE_OFFSETS: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    DOUBLE_P: tl.constexpr,
    CONSUMER_COPY: tl.constexpr,
    PACKED_META: tl.constexpr,
    MetaInfo,
    Selected,
):
    if Active:
        for meta_kind in tl.static_range(2 if PACKED_META else 1):
            if not PACKED_META or (meta_kind == 1) == ((MetaInfo & 1) != 0):
                row = tl.program_id(0).to(tl.int64)
                LOCATION_BYTES: tl.constexpr = (
                    (meta_kind == 1) if PACKED_META else USE_BYTE_OFFSETS
                )
                LOCATION_PACKED: tl.constexpr = PACKED_META and meta_kind == 1
                LocationPtr = Selected if LOCATION_PACKED else Locations
                if CONSUMER_COPY:
                    origin = tl.inline_asm_elementwise(
                        "",
                        constraints="=R,0",
                        args=[tl.full((), 128, tl.int32)],
                        dtype=tl.int32,
                        is_pure=False,
                        pack=1,
                    )
                    copy_rows = tl.arange(128, 128 + BN) - origin
                    columns = tl.arange(0, 32) * 16
                    copy_layout: tl.constexpr = tle.gpu.BlockEncoding(
                        [1, 1], [8, 4], [16, 1], [1, 0]
                    )
                    rr = tle.gpu.set_layout(
                        tl.broadcast_to(copy_rows[:, None], (BN, 32)), copy_layout
                    )
                    cc = tle.gpu.set_layout(
                        tl.broadcast_to(columns[None, :], (BN, 32)), copy_layout
                    )
                    sentinel = (
                        tl.full((), TOKENS * KV_TOKEN_BYTES, tl.uint32).to(tl.int32)
                        if LOCATION_BYTES
                        else TOKENS
                    )
                    tle.gpu.barrier_wait(PEmpty[0], phaseIdx=0)
                    init_location = tl.load(
                        LocationPtr + row * K + rr, rr < K, sentinel, volatile=True
                    )
                    init_source = kv_copy_address(
                        KV, init_location, cc, TOKENS, LOCATION_BYTES, LOCATION_PACKED
                    )
                    init_destination = (
                        tle.gpu.local_ptr(sk.slot(0), (rr, cc))
                        .to(tl.uint64)
                        .to(tl.uint32)
                    )
                    bounded_copy16(
                        init_destination,
                        init_source,
                        KV.to(tl.uint64),
                        tl.full((), TOKENS * KV_TOKEN_BYTES, tl.uint64),
                    )
                    bounded_copy_wait(tl.full((), 0, tl.int32))
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(CopyReady[0], phaseIdx=0)
                accum = tle.gpu.set_layout(tl.full((64, 512), 0.0, tl.float32), PV)
                PAGES: tl.constexpr = triton.cdiv(K, BN)
                # Four pages use both slots at both barrier phases.
                PERIOD: tl.constexpr = 4 if CONSUMER_COPY and K == 8320 else 1
                for group in range(PAGES // PERIOD):
                    for step in tl.static_range(PERIOD):
                        page = group * PERIOD + step
                        phase_step = step if PERIOD == 4 else page
                        slot = phase_step & 1
                        if CONSUMER_COPY and page + 1 < tl.cdiv(K, BN):
                            next_slot = (phase_step + 1) & 1
                            next_phase = ((phase_step + 1) // 2) & 1
                            tle.gpu.barrier_wait(PEmpty[next_slot], phaseIdx=next_phase)
                            offsets = (page + 1) * BN + rr
                            next_location = tl.load(
                                LocationPtr + row * K + offsets,
                                offsets < K,
                                sentinel,
                                volatile=True,
                            )
                            next_source = kv_copy_address(
                                KV,
                                next_location,
                                cc,
                                TOKENS,
                                LOCATION_BYTES,
                                LOCATION_PACKED,
                            )
                            next_destination = (
                                tle.gpu.local_ptr(sk.slot(next_slot), (rr, cc))
                                .to(tl.uint64)
                                .to(tl.uint32)
                            )
                            bounded_copy16(
                                next_destination,
                                next_source,
                                KV.to(tl.uint64),
                                tl.full((), TOKENS * KV_TOKEN_BYTES, tl.uint64),
                            )
                            bounded_copy_wait(tl.full((), 0, tl.int32))
                            lma_wait(tl.full((), 0, tl.int32))
                            tle.gpu.barrier_arrive(
                                CopyReady[next_slot], phaseIdx=next_phase
                            )
                        if DOUBLE_P:
                            phase = (phase_step // 2) & 1
                            tle.gpu.barrier_wait(Full[slot], phaseIdx=phase)
                            high, low = ph.slot(slot), pl.slot(slot)
                        else:
                            tle.gpu.barrier_wait(Full, phaseIdx=slot)
                            high, low = ph, pl
                        values = sk.slot(slot)
                        accum = tle.gpu.wgmma(high, values, acc=accum)
                        accum = tle.gpu.wgmma(low, values, acc=accum)
                        accum = tle.gpu.wgmma_wait(0, accum)
                        if DOUBLE_P:
                            tle.gpu.barrier_arrive(PEmpty[slot], phaseIdx=phase)
                        else:
                            tle.gpu.barrier_arrive(PEmpty, phaseIdx=slot)
                if PERIOD == 4:
                    # Page 64 has no successor and starts a fresh barrier cycle.
                    tl.static_assert(PAGES == 65)
                    tle.gpu.barrier_wait(Full[0], phaseIdx=0)
                    accum = tle.gpu.wgmma(ph.slot(0), sk.slot(0), acc=accum)
                    accum = tle.gpu.wgmma(pl.slot(0), sk.slot(0), acc=accum)
                    accum = tle.gpu.wgmma_wait(0, accum)
                    tle.gpu.barrier_arrive(PEmpty[0], phaseIdx=0)
                oh = tl.inline_asm_elementwise(
                    "",
                    constraints="=R,0",
                    args=[tl.full((), 256, tl.int32)],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
                od = tl.inline_asm_elementwise(
                    "",
                    constraints="=R,0",
                    args=[tl.full((), 1024, tl.int32)],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
                h = tle.gpu.set_layout(tl.arange(256, 320) - oh, OROW)
                d = tle.gpu.set_layout(tl.arange(1024, 1536) - od, OCOL)
                tle.gpu.barrier_wait(NormReady, phaseIdx=0)
                inverse = tl.load(tle.gpu.local_ptr(FinalInv, (h,)), volatile=True)
                scale = tl.load(PScale + row * 64 + h, volatile=True)
                if K > 128 and K <= 1024:
                    # Keep the per-head factor normal at the lower certified scale.
                    factor = (scale * (2.0**32)) * inverse
                    result = (accum * factor[:, None]) * (2.0**-32)
                else:
                    result = accum * scale[:, None] * inverse[:, None]
                tl.store(Output + (row * 64 + h[:, None]) * 512 + d[None, :], result)


@triton.jit
def kv_copy_address(
    KV,
    location,
    dim,
    TOKENS: tl.constexpr,
    USE_BYTE_OFFSETS: tl.constexpr,
    PACKED_META: tl.constexpr = False,
):
    if PACKED_META:
        location = (location.to(tl.uint32) & 0xFFFFFE00).to(tl.int32)
    if USE_BYTE_OFFSETS:
        offset = location.to(tl.uint32).to(tl.uint64)
        if TOKENS * KV_TOKEN_BYTES >= 2**31:
            # Pointer indexing can lower the high bit as a signed offset.
            return KV.to(tl.uint64) + offset + dim.to(tl.uint64)
    else:
        offset = location.to(tl.int64) * KV_TOKEN_BYTES
    return (KV + offset + dim).to(tl.uint64)


@triton.jit
def bounded_exp_from_half(half_score):
    # For a half score in [-1, 1], the squared result has relative error below 2.7e-5.
    value = tl.fma(half_score, 0.001439274288713932, 0.008686820976436138)
    value = tl.fma(half_score, value, 0.04163501039147377)
    value = tl.fma(half_score, value, 0.1664888709783554)
    value = tl.fma(half_score, value, 0.5000063180923462)
    value = tl.fma(half_score, value, 1.000022292137146)
    value = tl.fma(half_score, value, 0.9999998211860657)
    return value * value


@triton.jit
def produce(
    Q,
    SharedQ,
    QReady,
    KV,
    QS,
    Locations,
    PScale,
    LSE,
    ph,
    pl,
    sk,
    FinalInv,
    Full,
    PEmpty,
    NormReady,
    Sync,
    SumParts,
    SumSync,
    CopySync,
    CopyReady,
    Selected,
    Active,
    UsePoly,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
    USE_BYTE_OFFSETS: tl.constexpr,
    DOUBLE_P: tl.constexpr,
    CONSUMER_COPY: tl.constexpr,
    PACKED_META: tl.constexpr,
    MetaInfo,
):
    if Active:
        row = tl.program_id(0).to(tl.int64)
        h = tl.arange(0, 64)
        col = tl.arange(0, BN)
        if DOUBLE_P:
            q = SharedQ
        else:
            q = tle.gpu.alloc((64, 512), tl.float8e4nv, nv_mma_shared_layout=True)
        qs = tl.load(QS + row * 64 + h)
        ds = tl.load(PScale + row * 64 + h)
        reciprocal = tl.div_rn(1.0, ds)
        if CONSUMER_COPY:
            cd = tl.arange(0, 32) * 16
            copy_layout: tl.constexpr = tle.gpu.BlockEncoding(
                [1, 1], [2, 16], [16, 1], [1, 0]
            )
        else:
            copy_origin = tl.inline_asm_elementwise(
                "",
                constraints="=R,0",
                args=[tl.full((), 128, tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            cr = tl.arange(128, 128 + BN) - copy_origin
            cd = tl.arange(0, 32) * 16
            copy_layout: tl.constexpr = tle.gpu.BlockEncoding(
                [1, 1], [2, 16], [16, 1], [1, 0]
            )
            kv_lane_rows: tl.constexpr = 8 if K > 128 else 2
            kv_layout: tl.constexpr = tle.gpu.BlockEncoding(
                [1, 1], [kv_lane_rows, 32 // kv_lane_rows], [16, 1], [1, 0]
            )
            crr = tle.gpu.set_layout(tl.broadcast_to(cr[:, None], (BN, 32)), kv_layout)
            cdd = tle.gpu.set_layout(tl.broadcast_to(cd[None, :], (BN, 32)), kv_layout)
        if isinstance(Q, tl.tensor_descriptor):
            tle.gpu.copy(
                Q, q, (64, 512), offsets=(row.to(tl.int32) * 64, 0), barrier=QReady
            )
        else:
            # Stage raw Q bytes without materializing the MMA-distributed tensor.
            q_origin = tl.inline_asm_elementwise(
                "",
                constraints="=R,0",
                args=[tl.full((), 4096, tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            q_rows = tl.arange(4096, 4160) - q_origin
            qrr = tle.gpu.set_layout(
                tl.broadcast_to(q_rows[:, None], (64, 32)), copy_layout
            )
            qcc = tle.gpu.set_layout(
                tl.broadcast_to(cd[None, :], (64, 32)), copy_layout
            )
            q_dst = tle.gpu.local_ptr(q, (qrr, qcc)).to(tl.uint64).to(tl.uint32)
            q_base = Q + row * 64 * 512
            q_src = q_base + qrr.to(tl.int64) * 512 + qcc
            bounded_copy16(
                q_dst,
                q_src.to(tl.uint64),
                q_base.to(tl.uint64),
                tl.full((), 64 * 512, tl.uint64),
            )
        if not CONSUMER_COPY:
            if USE_BYTE_OFFSETS:
                sentinel = tl.full((), TOKENS * KV_TOKEN_BYTES, tl.uint32).to(tl.int32)
            else:
                sentinel = TOKENS
            init_location = tl.load(
                Locations + row * K + crr, crr < K, sentinel, volatile=True
            )
            init_src = kv_copy_address(
                KV, init_location, cdd, TOKENS, USE_BYTE_OFFSETS, PACKED_META
            )
            init_dst0 = (
                tle.gpu.local_ptr(sk.slot(0), (crr, cdd)).to(tl.uint64).to(tl.uint32)
            )
            bounded_copy16(
                init_dst0,
                init_src,
                KV.to(tl.uint64),
                tl.full((), TOKENS * 512, tl.uint64),
            )
        bounded_copy_wait(tl.full((), 0, tl.int32))
        if isinstance(Q, tl.tensor_descriptor):
            tle.gpu.barrier_wait(QReady, phaseIdx=0)
        lma_wait(tl.full((), 0, tl.int32))
        tle.gpu.barrier_arrive(Sync, phaseIdx=0)
        tle.gpu.barrier_wait(Sync, phaseIdx=0)
        for meta_kind in tl.static_range(2 if PACKED_META else 1):
            if not PACKED_META or (meta_kind == 1) == ((MetaInfo & 1) != 0):
                # Select the exponent mode per query, outside its hot page loop.
                SPLIT_EXP: tl.constexpr = CONSUMER_COPY and K == 8320
                for mode in tl.static_range(2 if SPLIT_EXP else 1):
                    if not SPLIT_EXP or (mode == 1) == UsePoly:
                        if K > 1024:
                            total_matrix = tl.full((64, BN // 16, 8), 0.0, tl.float32)
                        else:
                            total_matrix = tl.full((64, BN), 0.0, tl.float32)
                        for page in range(tl.cdiv(K, BN)):
                            slot = page & 1
                            if CONSUMER_COPY:
                                tle.gpu.barrier_wait(
                                    CopyReady[slot], phaseIdx=(page // 2) & 1
                                )
                            offsets = page * BN + col
                            if PACKED_META:
                                word = tl.load(
                                    Selected + row * K + offsets, offsets < K, 0
                                ).to(tl.uint32)
                                if meta_kind == 1:
                                    ks = ((word & 255) << 23).to(
                                        tl.float32, bitcast=True
                                    )
                                else:
                                    ks = word.to(tl.float32, bitcast=True)
                            else:
                                ks = tl.load(
                                    Selected + row * K + offsets, offsets < K, 0.0
                                )
                            valid = (offsets < K) & (ks > 0.0)
                            dot = tle.gpu.wgmma(q, sk.slot(slot), trans_b=True)
                            dot = tle.gpu.wgmma_wait(0, dot).to(tl.float32)
                            if DOUBLE_P:
                                phase = (page // 2) & 1
                                high_tile, low_tile = ph.slot(slot), pl.slot(slot)
                            else:
                                tle.gpu.barrier_wait(PEmpty, phaseIdx=slot)
                                high_tile, low_tile = ph, pl
                            shifted = dot * qs[:, None] * ks[None, :] * SCALE
                            if K > 128 and (mode == 1 if SPLIT_EXP else UsePoly):
                                if K > 512:
                                    # Pair token bit 3 within each SQMMA thread's registers.
                                    pairs = tl.permute(
                                        tl.reshape(shifted * 0.5, (64, BN // 16, 2, 8)),
                                        (0, 1, 3, 2),
                                    )
                                    first, second = tl.split(pairs)
                                    packed = exp2_pair(first, second)
                                    low = packed.to(tl.uint32).to(
                                        tl.float32, bitcast=True
                                    )
                                    high = (
                                        (packed >> 32)
                                        .to(tl.uint32)
                                        .to(tl.float32, bitcast=True)
                                    )
                                    exp_value = tl.reshape(
                                        tl.permute(tl.join(low, high), (0, 1, 3, 2)),
                                        (64, BN),
                                    )
                                else:
                                    exp_value = bounded_exp_from_half(shifted * 0.5)
                            else:
                                exp_value = tl.exp(shifted)
                            probability = tl.where(valid[None, :], exp_value, 0.0)
                            if K > 1024:
                                # Token bit 3 is register-local; the summation depth is unchanged.
                                sum_pairs = tl.permute(
                                    tl.reshape(probability, (64, BN // 16, 2, 8)),
                                    (0, 1, 3, 2),
                                )
                                sum_first, sum_second = tl.split(sum_pairs)
                                total_matrix += sum_first + sum_second
                            else:
                                total_matrix += probability
                            normalized = probability * ks[None, :] * reciprocal[:, None]
                            high = normalized.to(tl.float8e4nv)
                            low = (normalized - high.to(tl.float32)).to(tl.float8e4nv)
                            phh = tl.broadcast_to(h[:, None], (64, BN))
                            phc = tl.broadcast_to(col[None, :], (64, BN))
                            tl.store(tle.gpu.local_ptr(high_tile, (phh, phc)), high)
                            tl.store(tle.gpu.local_ptr(low_tile, (phh, phc)), low)
                            lma_wait(tl.full((), 0, tl.int32))
                            # Full already orders all producer writes before the consumer's PV.
                            if DOUBLE_P:
                                tle.gpu.barrier_arrive(Full[slot], phaseIdx=phase)
                            else:
                                tle.gpu.barrier_arrive(Full, phaseIdx=slot)
                            if page + 1 == tl.cdiv(K, BN):
                                if K > 1024:
                                    partial = tl.sum(
                                        tl.reshape(total_matrix, (64, 4, 16)), 2
                                    )
                                else:
                                    partial = tl.sum(
                                        tl.reshape(total_matrix, (64, 4, 32)), 2
                                    )
                                # Preserve the partials' layout to avoid conversion scratch.
                                tl.store(tle.gpu.local_ptr(SumParts), partial)
                                lma_wait(tl.full((), 0, tl.int32))
                                # This barrier is used once, regardless of the number of tiles.
                                tle.gpu.barrier_arrive(SumSync, phaseIdx=0)
                                tle.gpu.barrier_wait(SumSync, phaseIdx=0)
                                s0 = tl.load(
                                    tle.gpu.local_ptr(
                                        SumParts, (h, tl.full((64,), 0, tl.int32))
                                    ),
                                    volatile=True,
                                )
                                s1 = tl.load(
                                    tle.gpu.local_ptr(
                                        SumParts, (h, tl.full((64,), 1, tl.int32))
                                    ),
                                    volatile=True,
                                )
                                s2 = tl.load(
                                    tle.gpu.local_ptr(
                                        SumParts, (h, tl.full((64,), 2, tl.int32))
                                    ),
                                    volatile=True,
                                )
                                s3 = tl.load(
                                    tle.gpu.local_ptr(
                                        SumParts, (h, tl.full((64,), 3, tl.int32))
                                    ),
                                    volatile=True,
                                )
                                total = (s0 + s1) + (s2 + s3)
                                inverse = tl.where(
                                    total > 0,
                                    1.0 / tl.where(total > 0, total, 1.0),
                                    0.0,
                                )
                                tl.store(tle.gpu.local_ptr(FinalInv, (h,)), inverse)
                                lse = tl.where(total > 0, tl.log(total), float("inf"))
                                tl.store(LSE + row * 64 + h, lse)
                                lma_wait(tl.full((), 0, tl.int32))
                                tle.gpu.barrier_arrive(NormReady, phaseIdx=0)
                            if not CONSUMER_COPY and page + 1 < tl.cdiv(K, BN):
                                ns = (page + 1) & 1
                                copy_offsets = (page + 1) * BN + crr
                                copy_location = tl.load(
                                    Locations + row * K + copy_offsets,
                                    copy_offsets < K,
                                    sentinel,
                                    volatile=True,
                                )
                                copy_src = kv_copy_address(
                                    KV,
                                    copy_location,
                                    cdd,
                                    TOKENS,
                                    USE_BYTE_OFFSETS,
                                    PACKED_META,
                                )
                                copy_dst0 = (
                                    tle.gpu.local_ptr(sk.slot(ns), (crr, cdd))
                                    .to(tl.uint64)
                                    .to(tl.uint32)
                                )
                                if DOUBLE_P:
                                    # Wait once before overwriting the KV and later P in this slot.
                                    tle.gpu.barrier_wait(
                                        PEmpty[ns], phaseIdx=((page + 1) // 2) & 1
                                    )
                                bounded_copy16(
                                    copy_dst0,
                                    copy_src,
                                    KV.to(tl.uint64),
                                    tl.full((), TOKENS * 512, tl.uint64),
                                )
                                bounded_copy_wait(tl.full((), 0, tl.int32))
                                lma_wait(tl.full((), 0, tl.int32))
                                tle.gpu.barrier_arrive(CopySync, phaseIdx=page & 1)
                                tle.gpu.barrier_wait(CopySync, phaseIdx=page & 1)


@triton.jit
def fused_fp8_guarded(
    Q,
    KV,
    QS,
    Locations,
    PScale,
    Fast,
    Output,
    LSE,
    Selected,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    USE_BYTE_OFFSETS: tl.constexpr,
    DOUBLE_P: tl.constexpr,
    PACKED_META: tl.constexpr,
):
    BN: tl.constexpr = 128
    consumer_copy: tl.constexpr = DOUBLE_P and K > 1024 and _MIXED_PV_SUPPORTED
    sk = tle.gpu.alloc((2, BN, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    if DOUBLE_P:
        ph = tle.gpu.alloc((2, 64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
        pl = tle.gpu.alloc((2, 64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
        q = tle.gpu.alloc((64, 512), tl.float8e4nv, nv_mma_shared_layout=True)
        # The final normalization uses this storage only after the last QK wait.
        inverse = tle.gpu.alloc((64,), tl.float32, alias=q, nv_mma_shared_layout=False)
        sp = tle.gpu.alloc(
            (64, 4),
            tl.float32,
            alias=q,
            alias_offset_bytes=256,
            nv_mma_shared_layout=False,
        )
        full = tle.gpu.alloc_barriers(2, arrive_count=16)
        pe = tle.gpu.alloc_barriers(2, arrive_count=16, init=tle.gpu.READY)
    else:
        ph = tle.gpu.alloc((64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
        pl = tle.gpu.alloc((64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
        # Only the double-buffered specialization uses this argument.
        q = sk
        inverse = tle.gpu.alloc((64,), tl.float32, nv_mma_shared_layout=False)
        sp = tle.gpu.alloc((64, 4), tl.float32, nv_mma_shared_layout=False)
        full = tle.gpu.alloc_barrier(arrive_count=16)
        pe = tle.gpu.alloc_barrier(arrive_count=16, init=tle.gpu.READY)
    q_ready = (
        tle.gpu.alloc_barrier(arrive_count=1, expect_bytes=64 * 512)
        if isinstance(Q, tl.tensor_descriptor)
        else full
    )
    norm_ready = tle.gpu.alloc_barrier(arrive_count=16)
    sync = tle.gpu.alloc_barrier(arrive_count=16)
    sb = tle.gpu.alloc_barrier(arrive_count=16)
    if consumer_copy:
        copy_ready = tle.gpu.alloc_barriers(2, arrive_count=16)
        cs = sync
    else:
        copy_ready = full
        cs = tle.gpu.alloc_barrier(arrive_count=16)
    mode = tl.load(Fast + tl.program_id(0), volatile=True)
    active = mode != 0
    use_poly = ((mode & 3) if PACKED_META else mode) == FAST_BOUNDED_EXP
    meta_info = mode >> 2 if PACKED_META else 0
    tl.debug_barrier()
    tle.gpu.warp_specialize(
        [
            (
                produce,
                (
                    Q,
                    q,
                    q_ready,
                    KV,
                    QS,
                    Locations,
                    PScale,
                    LSE,
                    ph,
                    pl,
                    sk,
                    inverse,
                    full,
                    pe,
                    norm_ready,
                    sync,
                    sp,
                    sb,
                    cs,
                    copy_ready,
                    Selected,
                    active,
                    use_poly,
                    K,
                    TOKENS,
                    BN,
                    SCALE,
                    USE_BYTE_OFFSETS,
                    DOUBLE_P,
                    consumer_copy,
                    PACKED_META,
                    meta_info,
                ),
            ),
            (
                consume,
                (
                    Output,
                    PScale,
                    ph,
                    pl,
                    sk,
                    inverse,
                    full,
                    pe,
                    norm_ready,
                    active,
                    KV,
                    Locations,
                    copy_ready,
                    TOKENS,
                    USE_BYTE_OFFSETS,
                    K,
                    BN,
                    DOUBLE_P,
                    consumer_copy,
                    PACKED_META,
                    meta_info,
                    Selected,
                ),
            ),
        ],
        worker_num_warps=[16],
        worker_num_regs=[128],
    )


@lru_cache(maxsize=64)
def compiled_pipeline(
    topk,
    tokens,
    scale,
    output_dtype,
    device_index,
    target,
    use_byte_offsets,
    use_tme_q=False,
    packed_meta=False,
):
    """Cache compiled code by metadata; never retain caller Tensor storage."""
    fn = fused_fp8_guarded
    signature = dict(
        zip(
            fn.arg_names[:9],
            [
                "*fp8e4nv",
                "*fp8e4nv",
                "*fp32",
                "*i32",
                "*fp32",
                "*i32",
                output_dtype,
                "*fp32",
                "*fp32",
            ],
        )
    )
    if packed_meta:
        signature["Selected"] = "*i32"
    attrs = {(i,): [["tt.divisibility", 16]] for i in range(9)}
    if use_tme_q:
        signature["Q"] = "tensordesc<fp8e4nv[64,512]>"
        attrs[(0,)] = [["musa.tme_tail_divisibility", 4]]
    constants = dict(
        K=topk,
        TOKENS=tokens,
        SCALE=scale,
        USE_BYTE_OFFSETS=use_byte_offsets,
        PACKED_META=packed_meta,
    )
    library = native_library()
    buffering = (
        (True,)
        if (use_tme_q or packed_meta)
        else ((True, False) if topk > 128 else (False,))
    )
    for double_p in buffering:
        try:
            return triton.compile(
                ASTSource(
                    fn,
                    signature,
                    dict(constants, DOUBLE_P=double_p),
                    attrs=attrs,
                ),
                target=target,
                options=dict(
                    num_warps=16,
                    num_stages=1,
                    maxnreg=128,
                    enable_backend_opt=True,
                    enable_llc_opt=True,
                    extern_libs={"native": library},
                ),
            )
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug(
                "Sparse FP8 pipeline compile failed (double_p=%s): %s", double_p, error
            )
    return None


def prepare_pipeline(q, kv, qs, ks, indices, output, lse, scale):
    rows = q.shape[0]
    topk = indices.shape[-1]
    tokens = kv.shape[0] * 64
    # Only selected tokens contribute to QK/PV; their maxima give the same
    # error guarantees without scanning a larger, mostly unused cache.
    selected_count = rows * topk if rows * topk < tokens else 0
    parts = triton.cdiv(selected_count or tokens, 1024)
    selected_bytes = 4 * rows * topk
    workspace_bytes = selected_bytes + rows * (64 * 4 + 8) + 4 * parts + 8
    workspace_limit = 8 * 1024**3
    if workspace_bytes > workspace_limit:
        return None
    if any(t.data_ptr() % 16 for t in (q, kv, qs, ks, indices, output, lse)):
        return None
    target = driver.active.get_current_target()
    dtype = "*bf16" if output.dtype == torch.bfloat16 else "*fp16"
    use_byte_offsets = (
        tokens < KV_OFFSET_TOKEN_LIMIT
        and workspace_bytes + selected_bytes <= workspace_limit
    )
    packed_meta = (
        topk == 8320
        and rows >= 128
        and tokens < PACKED_METADATA_TOKEN_LIMIT
        and use_byte_offsets
        and _MIXED_PV_SUPPORTED.value
        and driver.active.utils.get_device_properties(
            driver.active.get_current_device()
        )["max_shared_mem"]
        >= 192 * 1024
    )
    compile_args = (topk, tokens, scale, dtype, q.device.index, target)
    # Large K640 queries amortize descriptor expansion in the MUSA launcher.
    use_tme_q = topk == 640 and rows >= 4096 and _MIXED_PV_SUPPORTED.value
    programs, kernel = 0, None
    if topk == 128 and rows >= 128 and _MIXED_PV_SUPPORTED.value:
        from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_k128 import (
            compiled_kernel,
        )

        properties = driver.active.utils.get_device_properties(
            driver.active.get_current_device()
        )
        programs = min(rows, properties["multiprocessor_count"])
        kernel = compiled_kernel(*compile_args, use_byte_offsets, programs)
        if kernel is not None and kernel.metadata.shared > properties["max_shared_mem"]:
            kernel = None
    if kernel is None:
        programs = 0
        kernel = compiled_pipeline(
            *compile_args, use_byte_offsets, use_tme_q, packed_meta
        )
    if use_tme_q and kernel is None:
        use_tme_q = False
        kernel = compiled_pipeline(*compile_args, use_byte_offsets)
    if kernel is None:
        return None
    probability_scale = torch.empty((rows, 64), dtype=torch.float32, device=q.device)
    flags = torch.empty((rows,), dtype=torch.int32, device=q.device)
    bad_rows = torch.empty_like(flags)
    bad_count = torch.empty((1,), dtype=torch.int32, device=q.device)
    selected = torch.empty(
        (rows, topk),
        dtype=torch.int32 if packed_meta else torch.float32,
        device=q.device,
    )
    byte_offsets = selected if packed_meta else None
    cache_bits = None
    if packed_meta:
        try:
            cache_bits = torch.empty(
                (triton.cdiv(tokens, 32),), dtype=torch.int32, device=q.device
            )
        except torch.OutOfMemoryError:
            packed_meta = False
            use_byte_offsets = False
            byte_offsets = None
            selected = selected.view(torch.float32)
            kernel = compiled_pipeline(*compile_args, False, False, False)
            if kernel is None:
                return None
    if use_byte_offsets and not packed_meta:
        try:
            byte_offsets = torch.empty_like(indices)
        except torch.OutOfMemoryError:
            # Offset staging is optional when the caller has limited device memory.
            use_tme_q = False
            if programs:
                kernel = compiled_kernel(*compile_args, False, programs)
            if not programs or kernel is None:
                programs = 0
                kernel = compiled_pipeline(*compile_args, False)
            if kernel is None:
                return None
    a2 = (
        q,
        qs,
        None,
        probability_scale,
        flags,
        bad_rows,
        bad_count,
        topk,
        scale,
        topk == 640 and rows >= 4096,
    )
    bounds_options = dict(
        num_warps=16,
        grid=(rows,),
        extern_libs={"native": native_library(QUERY_NORM_IR)},
    )
    k2 = None
    if topk == 640 and rows >= 128 and _MIXED_PV_SUPPORTED.value:
        active_query_args = (*a2[:5], None, None, topk, scale)
        row_data = (indices, ks, selected, byte_offsets)
        row_args = (*active_query_args, True, *row_data, tokens, False)
        try:
            k2 = query_bounds_global.warmup(*row_args, **bounds_options)
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug("Sparse FP8 row bounds failed: %s", error)
    row_scales = k2 is not None
    if row_scales:
        a2 = row_args
    else:
        scales = torch.empty(
            (parts * (3 if packed_meta else 1),), dtype=torch.float32, device=q.device
        )
        global_scale = torch.empty(
            (2 if packed_meta else 1,), dtype=torch.float32, device=q.device
        )
        a0 = (
            ks,
            indices,
            selected,
            byte_offsets,
            scales,
            bad_count,
            tokens,
            selected_count,
            1024,
            packed_meta,
            cache_bits,
        )
        a1 = (scales, global_scale, parts, triton.next_power_of_2(parts), packed_meta)
        r0 = cache_scale_bounds.warmup(*a0, num_warps=4, grid=(parts,))[(parts, 1, 1)]
        r1 = reduce_scales.warmup(*a1, num_warps=4, grid=(1,))[(1, 1, 1)]
        a2 = (
            a2[:2] + (global_scale,) + a2[3:] + (None, None, None, None, 0, packed_meta)
        )
        k2 = query_bounds_global.warmup(*a2, **bounds_options)
    a3 = (
        indices,
        ks,
        selected,
        byte_offsets,
        rows,
        topk,
        tokens,
        1024,
        packed_meta,
        global_scale if packed_meta else None,
        cache_bits,
    )
    g3 = triton.cdiv(rows * topk, 1024)
    if selected_count or row_scales:
        r3 = None
    else:
        k3 = gather_scales.warmup(*a3, num_warps=4, grid=(g3,))
        r3 = k3[(g3, 1, 1)]
    r2 = k2[(rows, 1, 1)]
    r4 = kernel[(programs or rows, 1, 1)]
    locations = indices if packed_meta or byte_offsets is None else byte_offsets
    query_arg = (
        TensorDescriptor.from_tensor(q.view(-1, 512), [64, 512]) if use_tme_q else q
    )
    a4 = (query_arg, kv, qs, locations, probability_scale, flags, output, lse, selected)
    if programs:
        a4 += (rows,)

    def run_pipeline():
        if not row_scales:
            r0(*a0)
            r1(*a1)
        r2(*a2)
        if r3 is not None:
            r3(*a3)
        r4(*a4)

    if row_scales:
        return run_pipeline, flags, None
    return run_pipeline, bad_rows, bad_count


def prepare_short_pipeline(q, kv, qs, ks, indices, output, lse, scale, combine, repair):
    rows, topk = q.shape[0], indices.shape[-1]
    if any(t.data_ptr() % 16 for t in (q, kv, qs, ks, indices, output, lse)):
        return None
    fixed_pv = _MIXED_PV_SUPPORTED.value and (
        topk >= 256 or (topk == 128 and rows >= 128)
    )
    if (
        fixed_pv
        and rows == 16
        and topk == 8320
        and kv.shape[0] * 64 < KV_OFFSET_TOKEN_LIMIT
        and 2.0**-125 <= scale <= 2.0**16
    ):
        from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_short_pipeline import (
            prepare_pipeline as prepare_staged_pipeline,
        )

        staged = prepare_staged_pipeline(
            q, kv, qs, ks, indices, output, lse, scale, repair
        )
        if staged is not None:
            return staged
    pages = triton.cdiv(topk, 128)
    direct = _MIXED_PV_SUPPORTED.value and (
        64 < topk <= 128 and not (topk == 128 and rows >= 128)
    )
    if rows > 16 and not direct:
        return None
    persistent = direct or (fixed_pv and topk >= 512)
    single_split = persistent and rows == 16 and topk == 640
    can_prefetch = False
    if direct:
        heads, warps = (64, 16) if rows > 16 else (32, 8)
        splits = 1
        programs, groups = rows, None
    elif persistent:
        small_grid = topk <= 1024 and rows * pages < 40
        # Keep all K=640 pages in one CTA when query parallelism is sufficient.
        if topk == 640 and (rows <= 4 or single_split):
            heads, warps = 32, 16
        else:
            heads, warps = (32, 8) if small_grid else (64, 16)
        properties = driver.active.utils.get_device_properties(
            driver.active.get_current_device()
        )
        can_prefetch = (
            single_split
            and properties["max_shared_mem"] >= 192 * 1024
            and kv.shape[0] * 64 < KV_OFFSET_TOKEN_LIMIT
        )
        programs = min(properties["multiprocessor_count"], rows * pages)
        groups = group_short_pages(rows, pages, programs) if topk > 1024 else None
        if single_split:
            programs, splits, groups = rows, 1, None
        elif topk <= 1024 and 2 * rows <= programs < rows * pages:
            # Keep short fragments inside one query to avoid extra Q staging.
            splits = programs // rows
            programs = rows * splits
        elif groups is not None:
            group_rows, group_programs, prefix_rows, row_programs = groups
            tail_rows = group_rows - prefix_rows
            tail_programs = group_programs - prefix_rows * row_programs
            splits = max(row_programs, triton.cdiv(tail_programs, tail_rows) + 1)
        elif small_grid and programs % rows == 0 and programs > rows:
            splits = programs // rows
        else:
            splits = triton.cdiv(programs, rows) + 1
    elif fixed_pv:
        heads, warps = 32, 8
        splits = min(8, triton.cdiv(topk, 64), triton.cdiv(64, rows * 2))
    else:
        heads = 16 if topk <= 1024 else 64
        warps = 4 if heads == 16 else 16
        splits = (
            1
            if topk <= 1024
            else min(32, triton.cdiv(topk, 64), triton.cdiv(256, rows))
        )
    partial_flags = torch.empty(
        (rows, splits * (64 // heads)), dtype=torch.int32, device=q.device
    )
    if splits == 1:
        partial, stats = output, lse
    else:
        partial = torch.empty(
            (rows, splits, 64, 512), dtype=torch.float32, device=q.device
        )
        stats = torch.empty((rows, splits, 64, 2), dtype=torch.float32, device=q.device)
    arguments = (
        q,
        kv,
        qs,
        ks,
        indices,
        partial,
        stats,
        partial_flags,
        topk,
        kv.shape[0] * 64,
        scale,
        heads,
        splits,
    )
    vector_log2 = can_prefetch or (persistent and rows == 4 and topk == 640)
    library = native_library(VECTOR_EXP2_IR) if vector_log2 else native_library()
    if persistent:
        arguments = arguments[:8] + (
            rows,
            topk,
            kv.shape[0] * 64,
            scale,
            programs,
            splits,
            groups,
            heads,
            warps,
        )
        kernel_function = persistent_sparse_fp8
        grid = (programs, 64 // heads, 1)
    else:
        kernel_function = fixed_scale_sparse_fp8 if fixed_pv else short_sparse_fp8
        grid = (rows, 64 // heads, splits)
    # Single-split CTAs can repair their own head group before writing.
    repairs = (repair, None) if (direct and rows <= 16) or single_split else (None,)
    for direct_repair in repairs:
        kernel_args = (
            arguments + (direct_repair, can_prefetch) if persistent else arguments
        )
        try:
            kernel = kernel_function.warmup(
                *kernel_args,
                num_warps=warps,
                num_stages=1,
                enable_backend_opt=persistent or fixed_pv or topk > 1024,
                enable_llc_opt=persistent or fixed_pv or topk > 1024,
                extern_libs={"native": library},
                grid=grid,
            )
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug(
                "Sparse FP8 short path compile failed (fused_repair=%s): %s",
                direct_repair is not None,
                error,
            )
        else:
            break
    else:
        return None
    runner = kernel[grid]
    if persistent and splits > 1:
        combine_runner, combine_args, flags = prepare_persistent_combine(
            q,
            kv,
            qs,
            ks,
            indices,
            partial,
            stats,
            partial_flags,
            output,
            lse,
            scale,
            splits,
            programs,
            groups,
            repair,
            64 // heads,
        )
    elif splits > 1:
        flags = partial_flags
        combine_args = (
            partial,
            stats,
            None,
            output,
            lse,
            rows,
            64,
            1,
            splits,
            False,
            triton.next_power_of_2(splits),
            512,
        )
        combined = combine.warmup(
            *combine_args, num_warps=4, num_stages=1, grid=(rows, 64, 1)
        )
        combine_runner = combined[(rows, 64, 1)]
    else:
        flags = None if direct_repair is not None else partial_flags

    def run_short():
        runner(*kernel_args)
        if splits > 1:
            combine_runner(*combine_args)

    return run_short, flags


@triton.jit
def short_sparse_fp8(
    Q,
    KV,
    QS,
    KS,
    Indices,
    Output,
    LSE,
    Flags,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    BH: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(2)
    begin = tl.cdiv(K, 64) * split // SPLITS
    end = tl.cdiv(K, 64) * (split + 1) // SPLITS
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    d = tl.arange(0, 512)
    col = tl.arange(0, 64)
    qv = tl.load(Q + (row * 64 + h[:, None]) * 512 + d[None, :])
    qs = tl.load(QS + row * 64 + h)
    q_l1 = tl.sum(tl.abs(qv.to(tl.float32)), 1) * qs
    q = tle.gpu.alloc(
        (BH, 512), tl.float8e4nv, init_value=qv, nv_mma_shared_layout=True
    )
    sk = tle.gpu.alloc((64, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((BH, 64), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((BH, 64), tl.float8e4nv, nv_mma_shared_layout=True)
    lma_wait(tl.full((), 0, tl.int32))
    tl.debug_barrier()
    cr = tl.arange(128, 192) - 128
    cd = tl.arange(1024, 1056) * 16 - 16384
    crr = tl.broadcast_to(cr[:, None], (64, 32))
    cdd = tl.broadcast_to(cd[None, :], (64, 32))
    dst = tle.gpu.local_ptr(sk, (crr, cdd)).to(tl.uint64).to(tl.uint32)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.full((BH,), 0.0, tl.float32)
    running = tl.full((BH,), 0.0, tl.float32)
    accum = tl.full((BH, 512), 0.0, tl.float32)
    needs_repair = tl.full((), 0, tl.int32)
    for page in range(begin, end):
        ids_copy = tl.load(
            Indices + row * K + page * 64 + crr, page * 64 + crr < K, TOKENS
        )
        src = KV + ids_copy.to(tl.int64) * 512 + cdd
        bounded_copy16(
            dst,
            src.to(tl.uint64),
            KV.to(tl.uint64),
            tl.full((), TOKENS * 512, tl.uint64),
        )
        bounded_copy_wait(tl.full((), 0, tl.int32))
        lma_wait(tl.full((), 0, tl.int32))
        tl.debug_barrier()
        dot = tle.gpu.wgmma(q, sk, trans_b=True)
        dot = tle.gpu.wgmma_wait(0, dot)
        ids = tl.load(Indices + row * K + page * 64 + col, page * 64 + col < K, -1)
        valid = (page * 64 + col < K) & (ids >= 0) & (ids < TOKENS)
        ks = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        error = (
            q_l1
            * (448.0 * tl.max(ks, 0))
            * SCALE
            * ((512.0 * 2.0**-24) / (1.0 - 512.0 * 2.0**-24))
        )
        needs_repair = tl.maximum(
            needs_repair, (tl.max(error, 0) > 1.0e-4).to(tl.int32)
        )
        scores = tl.where(
            valid[None, :],
            dot * qs[:, None] * ks[None, :] * (SCALE * 1.4426950408889634),
            -float("inf"),
        )
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        safe = tl.where(new_max == -float("inf"), 0.0, new_max)
        p = tl.exp2(scores - safe[:, None])
        correction = tl.exp2(maximum - safe)
        denominator = denominator * correction + tl.sum(p, 1)
        weights = p * ks[None, :]
        ds = tl.maximum(tl.max(weights, 1) * (1.0 / 448.0), 1.0e-30)
        inv = 1.0 / ds
        norm = weights * inv[:, None]
        high = norm.to(tl.float8e4nv)
        low = ((norm - high.to(tl.float32)) * 16.0).to(tl.float8e4nv)
        tl.store(tle.gpu.local_ptr(ph), high)
        tl.store(tle.gpu.local_ptr(pl), low)
        lma_wait(tl.full((), 0, tl.int32))
        tl.debug_barrier()
        accum = tle.gpu.wgmma(ph, sk, acc=accum * (running * correction * inv)[:, None])
        accum = tle.gpu.wgmma_wait(0, accum)
        accum = tle.gpu.wgmma(pl, sk, acc=accum * 16.0)
        accum = tle.gpu.wgmma_wait(0, accum) * (1.0 / 16.0)
        tl.debug_barrier()
        running = ds
        maximum = new_max
    tl.store(
        Flags + (row * SPLITS + split) * (64 // BH) + tl.program_id(1), needs_repair
    )
    factor = running
    if SPLITS == 1:
        factor *= tl.div_rn(1.0, tl.where(denominator > 0, denominator, 1.0))
    base = (row * SPLITS + split) * 64 + h
    tl.store(Output + base[:, None] * 512 + d[None, :], accum * factor[:, None])
    logsum = maximum * 0.6931471805599453 + tl.log(
        tl.where(denominator > 0, denominator, 1.0)
    )
    if SPLITS == 1:
        tl.store(LSE + row * 64 + h, tl.where(denominator > 0, logsum, float("inf")))
    else:
        tl.store(LSE + base * 2, maximum * 0.6931471805599453)
        tl.store(LSE + base * 2 + 1, denominator)


@triton.jit
def fixed_scale_sparse_fp8(
    Q,
    KV,
    QS,
    KS,
    Indices,
    Output,
    LSE,
    Flags,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    BH: tl.constexpr,
    SPLITS: tl.constexpr,
):
    BN: tl.constexpr = 64
    V_TILE: tl.constexpr = 256
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(2)
    begin = tl.cdiv(K, BN) * split // SPLITS
    end = tl.cdiv(K, BN) * (split + 1) // SPLITS
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    d = tl.arange(0, 512)
    col = tl.arange(0, BN)
    qv = tl.load(Q + (row * 64 + h[:, None]) * 512 + d[None, :])
    qs = tl.load(QS + row * 64 + h)
    q_l1 = tl.sum(tl.abs(qv.to(tl.float32)), 1) * qs
    q = tle.gpu.alloc(
        (BH, 512), tl.float8e4nv, init_value=qv, nv_mma_shared_layout=True
    )
    sk = tle.gpu.alloc((BN, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((BH, BN), tl.float16, nv_mma_shared_layout=True)
    value = tle.gpu.alloc((BN, V_TILE), tl.float16, nv_mma_shared_layout=True)
    lma_wait(tl.full((), 0, tl.int32))
    tl.debug_barrier()
    cr = tl.arange(128, 128 + BN) - 128
    cd = tl.arange(1024, 1056) * 16 - 16384
    crr = tl.broadcast_to(cr[:, None], (BN, 32))
    cdd = tl.broadcast_to(cd[None, :], (BN, 32))
    dst = tle.gpu.local_ptr(sk, (crr, cdd)).to(tl.uint64).to(tl.uint32)
    denominator = tl.full((BH,), 0.0, tl.float32)
    running = tl.full((BH,), 0.0, tl.float32)
    if V_TILE == 256:
        accum0 = tl.full((BH, 256), 0.0, tl.float32)
        accum1 = tl.full((BH, 256), 0.0, tl.float32)
    else:
        accum = tl.full((BH, 512), 0.0, tl.float32)
    needs_repair = tl.full((), 0, tl.int32)
    PRE: tl.constexpr = triton.next_power_of_2(
        triton.cdiv(triton.cdiv(K, BN), SPLITS) * BN
    )
    pre_offsets = begin * BN + tl.arange(0, PRE)
    pre_ids = tl.load(
        Indices + row * K + pre_offsets,
        (pre_offsets < K) & (pre_offsets < end * BN),
        -1,
    )
    pre_valid = (
        (pre_offsets < K)
        & (pre_offsets < end * BN)
        & (pre_ids >= 0)
        & (pre_ids < TOKENS)
    )
    pre_scales = tl.load(KS + tl.where(pre_valid, pre_ids, 0), pre_valid, 0.0)
    fixed_scale = tl.maximum(tl.max(pre_scales, 0), 1e-30)
    minimum_scale = tl.min(tl.where(pre_valid, pre_scales, float("inf")), 0)
    original_score_error = (
        q_l1
        * (448.0 * tl.max(pre_scales, 0))
        * SCALE
        * (512.0 * 2.0 ** (-24) / (1.0 - 512.0 * 2.0 ** (-24)))
    )
    PART_TOKENS: tl.constexpr = triton.cdiv(triton.cdiv(K, BN), SPLITS) * BN
    ROUND_UNIT: tl.constexpr = 2.0 ** (-23)
    SUM_GAMMA: tl.constexpr = (
        PART_TOKENS * ROUND_UNIT / (1.0 - PART_TOKENS * ROUND_UNIT)
    )
    normalization_error = (
        2.0 ** (-11) + 2.0 * SUM_GAMMA + 2.0 ** (-19) + 4.0 * SPLITS * ROUND_UNIT
    ) / (1.0 - SUM_GAMMA)
    score_sensitivity = tl.exp(4.1 * original_score_error) - 1.0
    pv_error_bound = 448.0 * fixed_scale * (normalization_error + score_sensitivity)
    eligible = (
        tl.min(
            ((original_score_error <= 0.0001) & (pv_error_bound < 0.0018)).to(tl.int32),
            0,
        )
        != 0
    )
    eligible = eligible & (minimum_scale >= fixed_scale * (1.0 / 256.0))
    eligible = eligible & (0.5 * tl.max(pre_scales, 0) <= 0.0019)
    needs_repair = (~eligible).to(tl.int32)
    for page in range(begin, end):
        ids_copy = tl.load(
            Indices + row * K + page * BN + crr, page * BN + crr < K, TOKENS
        )
        src = KV + ids_copy.to(tl.int64) * 512 + cdd
        bounded_copy16(
            dst,
            src.to(tl.uint64),
            KV.to(tl.uint64),
            tl.full((), TOKENS * 512, tl.uint64),
        )
        bounded_copy_wait(tl.full((), 0, tl.int32))
        lma_wait(tl.full((), 0, tl.int32))
        tl.debug_barrier()
        dot = tle.gpu.wgmma(q, sk, trans_b=True)
        dot = tle.gpu.wgmma_wait(0, dot)
        ids = tl.load(Indices + row * K + page * BN + col, page * BN + col < K, -1)
        valid = (page * BN + col < K) & (ids >= 0) & (ids < TOKENS)
        ks = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        scores = tl.where(
            valid[None, :],
            dot * qs[:, None] * ks[None, :] * (SCALE * 1.4426950408889634),
            -float("inf"),
        )
        p = tl.where(valid[None, :], tl.exp2(tl.where(eligible, scores, 0.0)), 0.0)
        denominator += tl.sum(p, 1)
        ds = fixed_scale + tl.zeros((BH,), tl.float32)
        inv = 1.0 / ds
        tl.store(
            tle.gpu.local_ptr(ph), (p * (ks[None, :] * inv[:, None])).to(tl.float16)
        )
        vr = tl.arange(256, 256 + BN) - 256
        vc = tl.arange(4096, 4096 + V_TILE // 4) * 4 - 16384
        vrr = tl.broadcast_to(vr[:, None], (BN, V_TILE // 4))
        vcc = tl.broadcast_to(vc[None, :], (BN, V_TILE // 4))
        for half in tl.static_range(512 // V_TILE):
            raw = tl.load(
                tle.gpu.local_ptr(sk, (vrr, vcc + half * V_TILE)).to(
                    tl.pointer_type(tl.uint32, 3)
                ),
                volatile=True,
            )
            tl.store(
                tle.gpu.local_ptr(value, (vrr, vcc)).to(tl.pointer_type(tl.uint64, 3)),
                promote_fp8_four(raw),
            )
            lma_wait(tl.full((), 0, tl.int32))
            tl.debug_barrier()
            if V_TILE == 512:
                accum = tle.gpu.wgmma(ph, value, acc=accum)
                accum = tle.gpu.wgmma_wait(0, accum)
            elif half == 0:
                accum0 = tle.gpu.wgmma(ph, value, acc=accum0)
                accum0 = tle.gpu.wgmma_wait(0, accum0)
            else:
                accum1 = tle.gpu.wgmma(ph, value, acc=accum1)
                accum1 = tle.gpu.wgmma_wait(0, accum1)
            tl.debug_barrier()
        running = ds
    tl.store(
        Flags + (row * SPLITS + split) * (64 // BH) + tl.program_id(1), needs_repair
    )
    factor = running
    if SPLITS == 1:
        factor *= tl.div_rn(1.0, tl.where(denominator > 0, denominator, 1.0))
    base = (row * SPLITS + split) * 64 + h
    if V_TILE == 256:
        output_dims = tl.arange(0, 256)
        tl.store(
            Output + base[:, None] * 512 + output_dims[None, :],
            accum0 * factor[:, None],
        )
        tl.store(
            Output + base[:, None] * 512 + 256 + output_dims[None, :],
            accum1 * factor[:, None],
        )
    else:
        tl.store(Output + base[:, None] * 512 + d[None, :], accum * factor[:, None])
    logsum = tl.log(tl.where(denominator > 0, denominator, 1.0))
    if SPLITS == 1:
        tl.store(LSE + row * 64 + h, tl.where(denominator > 0, logsum, float("inf")))
    else:
        tl.store(LSE + base * 2, 0.0)
        tl.store(LSE + base * 2 + 1, denominator)


@triton.jit
def sparse_fp8_program_range(
    row,
    ROWS: tl.constexpr,
    PAGES: tl.constexpr,
    PROGRAMS: tl.constexpr,
    GROUPS: tl.constexpr,
):
    if GROUPS is None:
        first = ((row * PAGES + 1) * PROGRAMS - 1) // (ROWS * PAGES)
        last = ((row + 1) * PAGES * PROGRAMS - 1) // (ROWS * PAGES)
    else:
        GROUP_ROWS: tl.constexpr = GROUPS[0]
        GROUP_PROGRAMS: tl.constexpr = GROUPS[1]
        PREFIX_ROWS: tl.constexpr = GROUPS[2]
        ROW_PROGRAMS: tl.constexpr = GROUPS[3]
        PREFIX_PROGRAMS: tl.constexpr = PREFIX_ROWS * ROW_PROGRAMS
        TAIL_PROGRAMS: tl.constexpr = GROUP_PROGRAMS - PREFIX_PROGRAMS
        TAIL_PAGES: tl.constexpr = (GROUP_ROWS - PREFIX_ROWS) * PAGES
        group_start = (row // GROUP_ROWS) * GROUP_PROGRAMS
        local_row = row % GROUP_ROWS
        tail_row = local_row - PREFIX_ROWS
        first = group_start + tl.where(
            local_row < PREFIX_ROWS,
            local_row * ROW_PROGRAMS,
            PREFIX_PROGRAMS
            + ((tail_row * PAGES + 1) * TAIL_PROGRAMS - 1) // TAIL_PAGES,
        )
        last = group_start + tl.where(
            local_row < PREFIX_ROWS,
            (local_row + 1) * ROW_PROGRAMS - 1,
            PREFIX_PROGRAMS
            + ((tail_row + 1) * PAGES * TAIL_PROGRAMS - 1) // TAIL_PAGES,
        )
    return first, last


@triton.jit
def persistent_sparse_fp8(
    Q,
    KV,
    QS,
    KS,
    Indices,
    Partial,
    Stats,
    PartialFlags,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    SLOTS: tl.constexpr,
    GROUPS: tl.constexpr,
    HEADS: tl.constexpr,
    WARPS: tl.constexpr,
    REPAIR: tl.constexpr,
    PREFETCH: tl.constexpr = False,
):
    BN: tl.constexpr = 128
    PAGES: tl.constexpr = triton.cdiv(K, BN)
    PART_TOKENS: tl.constexpr = triton.cdiv(ROWS * PAGES, PROGRAMS) * BN
    PRE: tl.constexpr = triton.next_power_of_2(PART_TOKENS)
    CACHE_METADATA: tl.constexpr = PRE <= 4096 and TOKENS * KV_TOKEN_BYTES < 2**32
    # Keep the exponent multipliers normal and limit amplification of underflow
    # in the hoisted query factor.
    PREPARE_SCALES: tl.constexpr = (
        (K > 1024 or (ROWS == 16 and K == 640))
        and CACHE_METADATA
        and SCALE >= 2.0**-125
        and SCALE <= 2.0**16
    )
    if CACHE_METADATA:
        offset_cache = tle.gpu.alloc((PRE,), tl.int32, nv_mma_shared_layout=False)
        scale_cache = tle.gpu.alloc((PRE,), tl.float32, nv_mma_shared_layout=False)
    program = tl.program_id(0)
    ALIGNED: tl.constexpr = GROUPS is None and PROGRAMS % ROWS == 0
    if ALIGNED:
        QUERY_SPLITS: tl.constexpr = PROGRAMS // ROWS
        query = program // QUERY_SPLITS
        query_split = program % QUERY_SPLITS
    elif GROUPS is None:
        first = ROWS * PAGES * program // PROGRAMS
        stop = ROWS * PAGES * (program + 1) // PROGRAMS
    else:
        GROUP_ROWS: tl.constexpr = GROUPS[0]
        GROUP_PROGRAMS: tl.constexpr = GROUPS[1]
        PREFIX_ROWS: tl.constexpr = GROUPS[2]
        ROW_PROGRAMS: tl.constexpr = GROUPS[3]
        PREFIX_PROGRAMS: tl.constexpr = PREFIX_ROWS * ROW_PROGRAMS
        TAIL_PROGRAMS: tl.constexpr = GROUP_PROGRAMS - PREFIX_PROGRAMS
        TAIL_PAGES: tl.constexpr = (GROUP_ROWS - PREFIX_ROWS) * PAGES
        group_row = (program // GROUP_PROGRAMS) * GROUP_ROWS
        local_program = program % GROUP_PROGRAMS
        if local_program < PREFIX_PROGRAMS:
            query = group_row + local_program // ROW_PROGRAMS
            part = local_program % ROW_PROGRAMS
            first = query * PAGES + PAGES * part // ROW_PROGRAMS
            stop = query * PAGES + PAGES * (part + 1) // ROW_PROGRAMS
        else:
            base = (group_row + PREFIX_ROWS) * PAGES
            part = local_program - PREFIX_PROGRAMS
            first = base + TAIL_PAGES * part // TAIL_PROGRAMS
            stop = base + TAIL_PAGES * (part + 1) // TAIL_PROGRAMS
    head_group = tl.program_id(1) if HEADS < 64 else 0
    h = head_group * HEADS + tl.arange(0, HEADS)
    d = tl.arange(0, 512)
    col = tl.arange(0, BN)
    q = tle.gpu.alloc((HEADS, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    if PREFETCH:
        sk = tle.gpu.alloc((2, BN, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    else:
        sk = tle.gpu.alloc((BN, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((HEADS, BN), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((HEADS, BN), tl.float8e4nv, nv_mma_shared_layout=True)
    # Keep the copy indices separate from MMA indices to avoid layout reuse.
    origin = tl.inline_asm_elementwise(
        "",
        constraints="=R,0",
        args=[tl.full((), 128, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    copy_rows = tl.arange(128, 128 + BN) - origin
    cd = tl.arange(1024, 1056) * 16 - 16384
    copy_layout: tl.constexpr = tle.gpu.BlockEncoding(
        [1, 1], [2, 16], [WARPS, 1], [1, 0]
    )
    rr = tle.gpu.set_layout(tl.broadcast_to(copy_rows[:, None], (BN, 32)), copy_layout)
    cc = tle.gpu.set_layout(tl.broadcast_to(cd[None, :], (BN, 32)), copy_layout)
    if not PREFETCH:
        destination = tle.gpu.local_ptr(sk, (rr, cc)).to(tl.uint64).to(tl.uint32)
    first_request = query if ALIGNED else first // PAGES
    fragments = 1 if ALIGNED else (stop - 1) // PAGES - first_request + 1
    for fragment in range(fragments):
        request = first_request + fragment
        row = request.to(tl.int64)
        if ALIGNED:
            begin = query_split * PAGES // QUERY_SPLITS
            end = (query_split + 1) * PAGES // QUERY_SPLITS
            slot = query_split
        else:
            begin = tl.maximum(first - request * PAGES, 0)
            end = tl.minimum(stop - request * PAGES, PAGES)
            first_program, _ = sparse_fp8_program_range(
                request, ROWS, PAGES, PROGRAMS, GROUPS
            )
            slot = program - first_program
        qv = tl.load(Q + (row * 64 + h[:, None]) * 512 + d[None, :])
        qs = tl.load(QS + row * 64 + h)
        if K >= 4096 or (PREFETCH and ALIGNED and ROWS == 16 and K == 640):
            # Four finite FP8 magnitudes fit in FP16. Inflate the subtotal
            # before applying QS to cover rounding in both reductions.
            q_abs = tl.reshape(tl.abs(qv.to(tl.float16)), (HEADS, 128, 4))
            q_parts = tl.sum(q_abs, 2, dtype=tl.float16).to(tl.float32)
            q_l1 = tl.sum(q_parts, 1) * 1.00154 * qs
        else:
            q_l1 = tl.sum(tl.abs(qv.to(tl.float32)), 1) * qs
        tl.store(tle.gpu.local_ptr(q), qv)
        meta_pos = tl.arange(0, PRE)
        offsets = begin * BN + meta_pos
        ids = tl.load(
            Indices + row * K + offsets, (offsets < K) & (offsets < end * BN), -1
        )
        valid = (offsets < K) & (offsets < end * BN) & (ids >= 0) & (ids < TOKENS)
        scales = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        if CACHE_METADATA:
            # Reuse the guard's gathers instead of reloading them for each page.
            tl.store(
                tle.gpu.local_ptr(offset_cache, (meta_pos,)),
                encode_kv_byte_offset(ids, valid, TOKENS),
            )
            if not PREPARE_SCALES:
                tl.store(tle.gpu.local_ptr(scale_cache, (meta_pos,)), scales)
        maximum = tl.max(scales, 0)
        fixed_scale = tl.maximum(maximum, 1e-30)
        minimum = tl.min(tl.where(valid, scales, float("inf")), 0)
        score_error = (
            q_l1 * (448 * maximum) * SCALE * (512 * 2**-24 / (1 - 512 * 2**-24))
        )
        U: tl.constexpr = 2**-24
        GP: tl.constexpr = 2 * PART_TOKENS * U / (1 - 2 * PART_TOKENS * U)
        DS: tl.constexpr = triton.cdiv(PART_TOKENS, BN) + 8
        GD: tl.constexpr = DS * 2**-23 / (1 - DS * 2**-23)
        coefficient = (
            2**-9 + 1.25 * GP + GD + 4 * 2**-19 + 8 * U + 4 * SLOTS * 2**-23
        ) / (1 - GD)
        bound = q_l1 * (448 * fixed_scale) * SCALE * 1.0005 + 1e-5
        if PREPARE_SCALES:
            # Budget rounding in the reordered FP32 scale products and quotient.
            score_error += bound * (16 * U)
        floor = 2**-14 * tl.exp(tl.minimum(bound, 20.0)) * (1 + 4 * GP) / (1 - GD)
        pv_error = (
            448 * fixed_scale * (coefficient + floor + tl.exp(4.1 * score_error) - 1)
        )
        if K > 1024 or (ROWS == 16 and K == 640):
            use_poly = tl.min(((qs >= 0) & (bound <= 2.0)).to(tl.int32), 0) != 0
            # A row can mix exponential paths across its fragments.
            pv_error += 448 * fixed_scale * BOUNDED_EXP_PV_BUDGET
        else:
            use_poly = False
        if PREPARE_SCALES:
            scale = fixed_scale * (1.0 / 16.0)
            # Zero queries do not need a potentially large scale product.
            q_scaled = tl.where(q_l1 == 0.0, 0.0, qs) * scale
            VECTOR_LOG2: tl.constexpr = PREFETCH and ALIGNED and ROWS == 16 and K == 640
            # The vector exponential takes the native log2 argument directly.
            if VECTOR_LOG2:
                multiplier = SCALE * 1.4426950408889634
            else:
                multiplier = tl.where(use_poly, SCALE * 0.5, SCALE * 1.4426950408889634)
            exponent_scale = q_scaled * multiplier
        eligible = (
            tl.min(
                ((score_error <= 1e-4) & (pv_error < 0.0038)).to(tl.int32),
                0,
            )
            != 0
        )
        eligible = eligible & (minimum >= fixed_scale / 256) & (0.5 * maximum <= 0.0019)
        accumulator = tl.full((HEADS, 512), 0.0, tl.float32)
        # Defer the cross-warp row reduction until the fragment is complete.
        probability_sums = tl.full((HEADS, BN), 0.0, tl.float32)
        if PREPARE_SCALES:
            inverse = tl.div_rn(16.0, fixed_scale)
            tl.store(tle.gpu.local_ptr(scale_cache, (meta_pos,)), scales * inverse)
            scale = scale + tl.zeros((HEADS,), tl.float32)
        else:
            scale = fixed_scale / 16 + tl.zeros((HEADS,), tl.float32)
            inverse = 1.0 / scale
        lma_wait(tl.full((), 0, tl.int32))
        tl.debug_barrier()
        if PREFETCH:
            # Keep shared views local; older TLE cannot lower them across calls.
            init_location = tl.load(tle.gpu.local_ptr(offset_cache, (rr,)))
            init_source = kv_copy_address(KV, init_location, cc, TOKENS, True)
            init_destination = (
                tle.gpu.local_ptr(sk.slot(0), (rr, cc)).to(tl.uint64).to(tl.uint32)
            )
            bounded_copy16(
                init_destination,
                init_source,
                KV.to(tl.uint64),
                tl.full((), TOKENS * 512, tl.uint64),
            )
            bounded_copy_wait(tl.full((), 0, tl.int32))
            lma_wait(tl.full((), 0, tl.int32))
            tl.debug_barrier()
        for page in range(begin, end):
            if PREFETCH:
                pipe_slot = (page - begin) & 1
                has_next = page + 1 < end
                if has_next:
                    next_position = (page + 1 - begin) * BN + rr
                    next_location = tl.load(
                        tle.gpu.local_ptr(offset_cache, (next_position,))
                    )
                    next_source = kv_copy_address(KV, next_location, cc, TOKENS, True)
                    next_destination = (
                        tle.gpu.local_ptr(sk.slot(1 - pipe_slot), (rr, cc))
                        .to(tl.uint64)
                        .to(tl.uint32)
                    )
                    bounded_copy16(
                        next_destination,
                        next_source,
                        KV.to(tl.uint64),
                        tl.full((), TOKENS * 512, tl.uint64),
                    )
                current_kv = sk.slot(pipe_slot)
            else:
                if CACHE_METADATA:
                    copy_pos = (page - begin) * BN + rr
                    copy_location = tl.load(
                        tle.gpu.local_ptr(offset_cache, (copy_pos,))
                    )
                    source = kv_copy_address(KV, copy_location, cc, TOKENS, True)
                else:
                    copy_offsets = page * BN + rr
                    copy_ids = tl.load(
                        Indices + row * K + copy_offsets, copy_offsets < K, TOKENS
                    )
                    source = kv_copy_address(KV, copy_ids, cc, TOKENS, False)
                bounded_copy16(
                    destination,
                    source,
                    KV.to(tl.uint64),
                    tl.full((), TOKENS * 512, tl.uint64),
                )
                bounded_copy_wait(tl.full((), 0, tl.int32))
                lma_wait(tl.full((), 0, tl.int32))
                tl.debug_barrier()
                current_kv = sk
            scores = tle.gpu.wgmma(q, current_kv, trans_b=True)
            scores = tle.gpu.wgmma_wait(0, scores)
            if CACHE_METADATA:
                page_pos = (page - begin) * BN + col
                ks = tl.load(tle.gpu.local_ptr(scale_cache, (page_pos,)))
                if PREPARE_SCALES:
                    present = ks > 0.0
                else:
                    page_location = tl.load(
                        tle.gpu.local_ptr(offset_cache, (page_pos,))
                    )
                    present = page_location.to(tl.uint32) < TOKENS * KV_TOKEN_BYTES
            else:
                indices = tl.load(
                    Indices + row * K + page * BN + col, page * BN + col < K, -1
                )
                present = (page * BN + col < K) & (indices >= 0) & (indices < TOKENS)
                ks = tl.load(KS + tl.where(present, indices, 0), present, 0.0)
            if PREPARE_SCALES:
                scaled = scores * exponent_scale[:, None] * ks[None, :]
                scaled = tl.where(eligible, scaled, 0.0)
                if VECTOR_LOG2:
                    probability = vector_exp2(scaled, HEADS, BN)
                elif use_poly:
                    probability = bounded_exp_from_half(scaled)
                else:
                    probability = tl.exp2(scaled)
            else:
                scores = scores * qs[:, None] * ks[None, :]
                if use_poly:
                    probability = bounded_exp_from_half(
                        tl.where(eligible, scores * (SCALE * 0.5), 0.0)
                    )
                else:
                    scores = scores * (SCALE * 1.4426950408889634)
                    log2_score = tl.where(eligible, scores, 0.0)
                    if ROWS == 4 and K == 640 and HEADS == 32 and WARPS == 16:
                        probability = vector_exp2(log2_score, HEADS, BN)
                    else:
                        probability = tl.exp2(log2_score)
            p = tl.where(present[None, :], probability, 0.0)
            probability_sums += p
            if PREPARE_SCALES:
                normalized = p * ks[None, :]
            else:
                normalized = p * (ks[None, :] * inverse[:, None])
            high = normalized.to(tl.float8e4nv)
            low = (normalized - high.to(tl.float32)).to(tl.float8e4nv)
            tl.store(tle.gpu.local_ptr(ph), high)
            tl.store(tle.gpu.local_ptr(pl), low)
            lma_wait(tl.full((), 0, tl.int32))
            tl.debug_barrier()
            accumulator = tle.gpu.wgmma(ph, current_kv, acc=accumulator)
            accumulator = tle.gpu.wgmma(pl, current_kv, acc=accumulator)
            accumulator = tle.gpu.wgmma_wait(0, accumulator)
            if PREFETCH:
                if has_next:
                    bounded_copy_wait(tl.full((), 0, tl.int32))
                    lma_wait(tl.full((), 0, tl.int32))
            tl.debug_barrier()
        denominator = tl.sum(probability_sums, 1)
        base = (row * SLOTS + slot) * 64 + h
        if SLOTS == 1:
            if REPAIR is not None and not eligible:
                REPAIR(
                    Q,
                    None,
                    KV,
                    None,
                    QS,
                    KS,
                    Indices,
                    None,
                    None,
                    None,
                    Partial,
                    Stats,
                    None,
                    row,
                    64,
                    1,
                    K,
                    TOKENS,
                    0,
                    SCALE,
                    False,
                    False,
                    1,
                    triton.cdiv(K, 64) * 64,
                    HEADS,
                    32,
                )
            else:
                norm = scale / tl.where(denominator > 0, denominator, 1.0)
                tl.store(
                    Partial + base[:, None] * 512 + d[None, :],
                    accumulator * norm[:, None],
                )
                tl.store(
                    Stats + base,
                    tl.where(denominator > 0, tl.log(denominator), float("inf")),
                )
        else:
            tl.store(
                Partial + base[:, None] * 512 + d[None, :], accumulator * scale[:, None]
            )
            tl.store(Stats + base * 2, 0.0)
            tl.store(Stats + base * 2 + 1, denominator)
        flag_offset = (row * SLOTS + slot) * (64 // HEADS) + head_group
        if REPAIR is None:
            tl.store(PartialFlags + flag_offset, (~eligible).to(tl.int32))
        tl.debug_barrier()


@triton.jit
def combine_persistent_sparse_fp8(
    Partial,
    Stats,
    PartialFlags,
    Output,
    LSE,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BLOCK_FLAGS: tl.constexpr,
    GROUPS: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    COMBINE_HEADS: tl.constexpr,
    Flags,
    Q,
    KV,
    QS,
    KS,
    Indices,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    REPAIR: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    PAGES: tl.constexpr = triton.cdiv(K, 128)
    ALIGNED: tl.constexpr = GROUPS is None and PROGRAMS % ROWS == 0
    if ALIGNED:
        splits = tl.full((), PROGRAMS // ROWS, tl.int32)
    else:
        first_program, last_program = sparse_fp8_program_range(
            row, ROWS, PAGES, PROGRAMS, GROUPS
        )
        splits = last_program - first_program + 1
    part = tl.arange(0, BLOCK_FLAGS)
    flag_pos = row * SLOTS * HEAD_GROUPS + part
    flag_valid = part < splits * HEAD_GROUPS
    if REPAIR is not None:
        bad = tl.max(tl.load(PartialFlags + flag_pos, flag_valid, 0), 0)
    else:
        bad = 0
    if REPAIR is not None and bad != 0:
        REPAIR_HEADS: tl.constexpr = max(16, COMBINE_HEADS)
        if tile < 64 // REPAIR_HEADS:
            REPAIR(
                Q,
                None,
                KV,
                None,
                QS,
                KS,
                Indices,
                None,
                None,
                None,
                Output,
                LSE,
                None,
                row,
                64,
                1,
                K,
                TOKENS,
                0,
                SCALE,
                False,
                False,
                1,
                triton.cdiv(K, 64) * 64,
                REPAIR_HEADS,
                32,
            )
    else:
        HEADS: tl.constexpr = COMBINE_HEADS
        h = tile * HEADS + tl.arange(0, HEADS)
        d = tl.arange(0, 512)
        numerator = tl.full((HEADS, 512), 0.0, tl.float32)
        denominator = tl.full((HEADS,), 0.0, tl.float32)
        iterations = SLOTS if REPAIR is None else splits
        UNROLL: tl.constexpr = SLOTS if ALIGNED and SLOTS <= 6 else 1
        for split in tl.range(0, iterations, loop_unroll_factor=UNROLL):
            valid = split < splits if REPAIR is None else True
            partial = tl.load(
                Partial + ((row * SLOTS + split) * 64 + h[:, None]) * 512 + d[None, :],
                valid,
                0.0,
            )
            weight = tl.load(
                Stats + ((row * SLOTS + split) * 64 + h) * 2 + 1, valid, 0.0
            )
            numerator += partial
            denominator += weight
        inverse = 1.0 / tl.where(denominator > 0, denominator, 1.0)
        output = numerator * inverse[:, None]
        tl.store(Output + (row * 64 + h[:, None]) * 512 + d[None, :], output)
        tl.store(
            LSE + row * 64 + h,
            tl.where(denominator > 0, tl.log(denominator), float("inf")),
        )
        if REPAIR is None and tile == 0:
            bad = tl.max(tl.load(PartialFlags + flag_pos, flag_valid, 0), 0)
            tl.store(Flags + row, bad)
