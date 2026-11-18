# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.experimental.tle import language as tle
from triton.runtime import driver
from triton.runtime.errors import OutOfResources
from triton.runtime.jit import JITFunction

from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_metadata import (
    BOUNDED_EXP_PV_BUDGET,
    KV_TOKEN_BYTES,
    encode_kv_byte_offset,
)
from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_native import (
    QUERY_NORM_IR,
    bounded_copy16,
    bounded_copy_wait,
    lma_wait,
    native_library,
    query_abs4,
)
from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_pipeline import (
    OCOL,
    OROW,
    PV,
    bounded_exp_from_half,
    kv_copy_address,
    prepare_persistent_combine,
    sparse_fp8_program_range,
)
from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_scheduling import (
    group_short_pages,
)


@triton.jit
def staged_guard(
    Q,
    QS,
    KS,
    Indices,
    PartialFlags,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    SLOTS: tl.constexpr,
    GROUPS: tl.constexpr,
    HEADS: tl.constexpr,
    MetaOffset,
    MetaScale,
    Setup,
    Modes,
    First,
    Stop,
):
    BN: tl.constexpr = 128
    PAGES: tl.constexpr = triton.cdiv(K, BN)
    PART_TOKENS: tl.constexpr = triton.cdiv(ROWS * PAGES, PROGRAMS) * BN
    PRE: tl.constexpr = triton.next_power_of_2(PART_TOKENS)
    CACHE_METADATA: tl.constexpr = PRE <= 4096 and TOKENS * KV_TOKEN_BYTES < 2**32
    PREPARE_SCALES: tl.constexpr = (
        (K > 1024 or (ROWS == 16 and K == 640))
        and CACHE_METADATA
        and (SCALE >= 2.0 ** (-125))
        and (SCALE <= 2.0**16)
    )
    tl.static_assert(ROWS == 16 and K == 8320 and (HEADS == 64) and PREPARE_SCALES)
    program = tl.program_id(0)
    h = tl.arange(0, HEADS)
    first_request = First // PAGES
    fragments = (Stop - 1) // PAGES - first_request + 1
    for fragment in range(fragments):
        request = first_request + fragment
        row = request.to(tl.int64)
        begin = tl.maximum(First - request * PAGES, 0)
        end = tl.minimum(Stop - request * PAGES, PAGES)
        (first_program, _) = sparse_fp8_program_range(
            request, ROWS, PAGES, PROGRAMS, GROUPS
        )
        slot = program - first_program
        word_col = tl.arange(0, 128)
        words = tl.load(
            Q.to(tl.pointer_type(tl.uint32))
            + (row * 64 + h[:, None]) * 128
            + word_col[None, :]
        )
        qs = tl.load(QS + row * 64 + h)
        q_parts = query_abs4(words)
        # Four finite E4M3 magnitudes fit in FP16; retain the original L1 inflation.
        q_l1 = tl.sum(q_parts, 1) * 1.00154 * qs
        meta_pos = tl.arange(0, PRE)
        offsets = begin * BN + meta_pos
        ids = tl.load(
            Indices + row * K + offsets, (offsets < K) & (offsets < end * BN), -1
        )
        valid = (offsets < K) & (offsets < end * BN) & (ids >= 0) & (ids < TOKENS)
        scales = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        maximum = tl.max(scales, 0)
        fixed_scale = tl.maximum(maximum, 1e-30)
        minimum = tl.min(tl.where(valid, scales, float("inf")), 0)
        score_error = (
            q_l1 * (448 * maximum) * SCALE * (512 * 2 ** (-24) / (1 - 512 * 2 ** (-24)))
        )
        U: tl.constexpr = 2 ** (-24)
        GP: tl.constexpr = 2 * PART_TOKENS * U / (1 - 2 * PART_TOKENS * U)
        DS: tl.constexpr = triton.cdiv(PART_TOKENS, BN) + 8
        GD: tl.constexpr = DS * 2 ** (-23) / (1 - DS * 2 ** (-23))
        coefficient = (
            2 ** (-9) + 1.25 * GP + GD + 4 * 2 ** (-19) + 8 * U + 4 * SLOTS * 2 ** (-23)
        ) / (1 - GD)
        bound = q_l1 * (448 * fixed_scale) * SCALE * 1.0005 + 1e-05
        if PREPARE_SCALES:
            score_error += bound * (16 * U)
        floor = 2 ** (-14) * tl.exp(tl.minimum(bound, 20.0)) * (1 + 4 * GP) / (1 - GD)
        pv_error = (
            448 * fixed_scale * (coefficient + floor + tl.exp(4.1 * score_error) - 1)
        )
        if K > 1024 or (ROWS == 16 and K == 640):
            use_poly = tl.min(((qs >= 0) & (bound <= 2.0)).to(tl.int32), 0) != 0
            pv_error += 448 * fixed_scale * BOUNDED_EXP_PV_BUDGET
        else:
            use_poly = False
        if PREPARE_SCALES:
            scale = fixed_scale * (1.0 / 16.0)
            q_scaled = tl.where(q_l1 == 0.0, 0.0, qs) * scale
            multiplier = tl.where(use_poly, SCALE * 0.5, SCALE * 1.4426950408889634)
            exponent_scale = q_scaled * multiplier
        eligible = (
            tl.min(((score_error <= 0.0001) & (pv_error < 0.0038)).to(tl.int32), 0) != 0
        )
        eligible = eligible & (minimum >= fixed_scale / 256) & (0.5 * maximum <= 0.0019)
        inverse = tl.div_rn(16.0, fixed_scale)
        meta_base = (row * SLOTS + slot) * PRE
        tl.store(
            MetaOffset + meta_base + meta_pos, encode_kv_byte_offset(ids, valid, TOKENS)
        )
        tl.store(MetaScale + meta_base + meta_pos, scales * inverse)
        setup_base = ((row * SLOTS + slot) * 64 + h) * 2
        tl.store(Setup + setup_base, exponent_scale)
        tl.store(Setup + setup_base + 1, scale)
        tl.store(
            Modes + row * SLOTS + slot, tl.where(eligible, tl.where(use_poly, 2, 1), 0)
        )
        tl.store(PartialFlags + row * SLOTS + slot, (~eligible).to(tl.int32))


@triton.jit
def staged_produce(
    Q,
    MetaScale,
    Setup,
    Modes,
    Stats,
    First,
    Stop,
    q,
    sk,
    ph,
    pl,
    Full,
    Ready,
    Sync,
    SumParts,
    SumSync,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    PROGRAMS: tl.constexpr,
    SLOTS: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    G3: tl.constexpr,
):
    BN: tl.constexpr = 128
    PAGES: tl.constexpr = triton.cdiv(K, BN)
    PRE: tl.constexpr = triton.next_power_of_2(triton.cdiv(ROWS * PAGES, PROGRAMS) * BN)
    first_request = First // PAGES
    fragments = (Stop - 1) // PAGES - first_request + 1
    h = tl.arange(0, 64)
    col = tl.arange(0, BN)
    copy_layout: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [2, 16], [16, 1], [1, 0])
    q_origin = tl.inline_asm_elementwise(
        "",
        constraints="=R,0",
        args=[tl.full((), 4096, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    q_rows = tl.arange(4096, 4160) - q_origin
    qrr = tle.gpu.set_layout(tl.broadcast_to(q_rows[:, None], (64, 32)), copy_layout)
    qcc = tle.gpu.set_layout(
        tl.broadcast_to((tl.arange(0, 32) * 16)[None, :], (64, 32)), copy_layout
    )
    q_dst = tle.gpu.local_ptr(q, (qrr, qcc)).to(tl.uint64).to(tl.uint32)
    epoch = 0
    fragment_phase = 0
    for fragment in range(fragments):
        row = (first_request + fragment).to(tl.int64)
        begin = tl.maximum(First - row * PAGES, 0)
        end = tl.minimum(Stop - row * PAGES, PAGES)
        (first_program, _) = sparse_fp8_program_range(
            row, ROWS, PAGES, PROGRAMS, (G0, G1, G2, G3)
        )
        part = tl.program_id(0) - first_program
        base = row * SLOTS + part
        mode = tl.load(Modes + base)
        if mode != 0:
            q_base = Q + row * 64 * 512
            q_src = q_base + qrr * 512 + qcc
            bounded_copy16(
                q_dst,
                q_src.to(tl.uint64),
                q_base.to(tl.uint64),
                tl.full((), 64 * 512, tl.uint64),
            )
            exponent_scale = tl.load(Setup + (base * 64 + h) * 2)
            bounded_copy_wait(tl.full((), 0, tl.int32))
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Sync, phaseIdx=0)
            tle.gpu.barrier_wait(Sync, phaseIdx=0)
            probability_sums = tl.full((64, BN), 0.0, tl.float32)
            for page in range(begin, end):
                slot = epoch & 1
                phase = epoch // 2 & 1
                tle.gpu.barrier_wait(Ready[slot], phaseIdx=phase)
                ks = tl.load(MetaScale + base * PRE + (page - begin) * BN + col)
                present = ks > 0.0
                scores = tle.gpu.wgmma(q, sk.slot(slot), trans_b=True)
                scores = tle.gpu.wgmma_wait(0, scores)
                scaled = scores * exponent_scale[:, None] * ks[None, :]
                if mode == 2:
                    probability = bounded_exp_from_half(scaled)
                else:
                    probability = tl.exp2(scaled)
                probability = tl.where(present[None, :], probability, 0.0)
                probability_sums += probability
                normalized = probability * ks[None, :]
                high = normalized.to(tl.float8e4nv)
                low = (normalized - high.to(tl.float32)).to(tl.float8e4nv)
                phh = tl.broadcast_to(h[:, None], (64, BN))
                phc = tl.broadcast_to(col[None, :], (64, BN))
                tl.store(tle.gpu.local_ptr(ph.slot(slot), (phh, phc)), high)
                tl.store(tle.gpu.local_ptr(pl.slot(slot), (phh, phc)), low)
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(Full[slot], phaseIdx=phase)
                epoch += 1
            # The last QK wait releases Q storage for the denominator reduction.
            partial_sum = tl.sum(tl.reshape(probability_sums, (64, 4, 32)), 2)
            tl.store(tle.gpu.local_ptr(SumParts), partial_sum)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(SumSync, phaseIdx=fragment_phase)
            tle.gpu.barrier_wait(SumSync, phaseIdx=fragment_phase)
            s0 = tl.load(
                tle.gpu.local_ptr(SumParts, (h, tl.full((64,), 0, tl.int32))),
                volatile=True,
            )
            s1 = tl.load(
                tle.gpu.local_ptr(SumParts, (h, tl.full((64,), 1, tl.int32))),
                volatile=True,
            )
            s2 = tl.load(
                tle.gpu.local_ptr(SumParts, (h, tl.full((64,), 2, tl.int32))),
                volatile=True,
            )
            s3 = tl.load(
                tle.gpu.local_ptr(SumParts, (h, tl.full((64,), 3, tl.int32))),
                volatile=True,
            )
            denominator = s0 + s1 + (s2 + s3)
            tl.store(Stats + (base * 64 + h) * 2, 0.0)
            tl.store(Stats + (base * 64 + h) * 2 + 1, denominator)
            # Finish all alias reads before any producer overwrites Q.
            tle.gpu.barrier_arrive(Sync, phaseIdx=1)
            tle.gpu.barrier_wait(Sync, phaseIdx=1)
            fragment_phase ^= 1


@triton.jit
def staged_consume(
    KV,
    MetaOffset,
    Setup,
    Modes,
    Partial,
    First,
    Stop,
    sk,
    ph,
    pl,
    Full,
    Empty,
    Ready,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    PROGRAMS: tl.constexpr,
    SLOTS: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    G3: tl.constexpr,
):
    BN: tl.constexpr = 128
    PAGES: tl.constexpr = triton.cdiv(K, BN)
    PRE: tl.constexpr = triton.next_power_of_2(triton.cdiv(ROWS * PAGES, PROGRAMS) * BN)
    first_request = First // PAGES
    fragments = (Stop - 1) // PAGES - first_request + 1
    origin = tl.inline_asm_elementwise(
        "",
        constraints="=R,0",
        args=[tl.full((), 128, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    copy_rows = tl.arange(128, 128 + BN) - origin
    copy_layout: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [8, 4], [16, 1], [1, 0])
    rr = tle.gpu.set_layout(tl.broadcast_to(copy_rows[:, None], (BN, 32)), copy_layout)
    cc = tle.gpu.set_layout(
        tl.broadcast_to((tl.arange(0, 32) * 16)[None, :], (BN, 32)), copy_layout
    )
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
    epoch = 0
    for fragment in range(fragments):
        row = (first_request + fragment).to(tl.int64)
        begin = tl.maximum(First - row * PAGES, 0)
        end = tl.minimum(Stop - row * PAGES, PAGES)
        (first_program, _) = sparse_fp8_program_range(
            row, ROWS, PAGES, PROGRAMS, (G0, G1, G2, G3)
        )
        part = tl.program_id(0) - first_program
        base = row * SLOTS + part
        mode = tl.load(Modes + base)
        if mode != 0:
            first_slot = epoch & 1
            first_phase = epoch // 2 & 1
            tle.gpu.barrier_wait(Empty[first_slot], phaseIdx=first_phase)
            location = tl.load(MetaOffset + base * PRE + rr)
            source = kv_copy_address(KV, location, cc, TOKENS, True)
            destination = (
                tle.gpu.local_ptr(sk.slot(first_slot), (rr, cc))
                .to(tl.uint64)
                .to(tl.uint32)
            )
            bounded_copy16(
                destination,
                source,
                KV.to(tl.uint64),
                tl.full((), TOKENS * 512, tl.uint64),
            )
            bounded_copy_wait(tl.full((), 0, tl.int32))
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Ready[first_slot], phaseIdx=first_phase)
            accum = tle.gpu.set_layout(tl.full((64, 512), 0.0, tl.float32), PV)
            for page in range(begin, end):
                slot = epoch & 1
                phase = epoch // 2 & 1
                if page + 1 < end:
                    next_slot = epoch + 1 & 1
                    next_phase = (epoch + 1) // 2 & 1
                    tle.gpu.barrier_wait(Empty[next_slot], phaseIdx=next_phase)
                    next_location = tl.load(
                        MetaOffset + base * PRE + (page + 1 - begin) * BN + rr
                    )
                    next_source = kv_copy_address(KV, next_location, cc, TOKENS, True)
                    next_destination = (
                        tle.gpu.local_ptr(sk.slot(next_slot), (rr, cc))
                        .to(tl.uint64)
                        .to(tl.uint32)
                    )
                    bounded_copy16(
                        next_destination,
                        next_source,
                        KV.to(tl.uint64),
                        tl.full((), TOKENS * 512, tl.uint64),
                    )
                    bounded_copy_wait(tl.full((), 0, tl.int32))
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(Ready[next_slot], phaseIdx=next_phase)
                tle.gpu.barrier_wait(Full[slot], phaseIdx=phase)
                accum = tle.gpu.wgmma(ph.slot(slot), sk.slot(slot), acc=accum)
                accum = tle.gpu.wgmma(pl.slot(slot), sk.slot(slot), acc=accum)
                accum = tle.gpu.wgmma_wait(0, accum)
                tle.gpu.barrier_arrive(Empty[slot], phaseIdx=phase)
                epoch += 1
            scale = tl.load(Setup + (base * 64 + h) * 2 + 1)
            tl.store(
                Partial + (base * 64 + h[:, None]) * 512 + d[None, :],
                accum * scale[:, None],
            )


@triton.jit
def staged_main(
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
    MetaOffset,
    MetaScale,
    Setup,
    Modes,
):
    tl.static_assert(
        ROWS == 16 and K == 8320 and (HEADS == 64) and (GROUPS is not None)
    )
    tl.static_assert(WARPS == 16)
    PAGES: tl.constexpr = triton.cdiv(K, 128)
    G0: tl.constexpr = GROUPS[0]
    G1: tl.constexpr = GROUPS[1]
    G2: tl.constexpr = GROUPS[2]
    G3: tl.constexpr = GROUPS[3]
    PREFIX_PROGRAMS: tl.constexpr = G2 * G3
    TAIL_PROGRAMS: tl.constexpr = G1 - PREFIX_PROGRAMS
    TAIL_PAGES: tl.constexpr = (G0 - G2) * PAGES
    program = tl.program_id(0)
    group_row = program // G1 * G0
    local_program = program % G1
    if local_program < PREFIX_PROGRAMS:
        query = group_row + local_program // G3
        part = local_program % G3
        first = query * PAGES + PAGES * part // G3
        stop = query * PAGES + PAGES * (part + 1) // G3
    else:
        base = (group_row + G2) * PAGES
        part = local_program - PREFIX_PROGRAMS
        first = base + TAIL_PAGES * part // TAIL_PROGRAMS
        stop = base + TAIL_PAGES * (part + 1) // TAIL_PROGRAMS
    staged_guard(
        Q,
        QS,
        KS,
        Indices,
        PartialFlags,
        ROWS,
        K,
        TOKENS,
        SCALE,
        PROGRAMS,
        SLOTS,
        GROUPS,
        HEADS,
        MetaOffset,
        MetaScale,
        Setup,
        Modes,
        first,
        stop,
    )
    q = tle.gpu.alloc((64, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    sk = tle.gpu.alloc((2, 128, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((2, 64, 128), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((2, 64, 128), tl.float8e4nv, nv_mma_shared_layout=True)
    sums = tle.gpu.alloc((64, 4), tl.float32, alias=q, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barriers(2, arrive_count=16)
    empty = tle.gpu.alloc_barriers(2, arrive_count=16, init=tle.gpu.READY)
    ready = tle.gpu.alloc_barriers(2, arrive_count=16)
    sync = tle.gpu.alloc_barrier(arrive_count=16)
    sum_sync = tle.gpu.alloc_barrier(arrive_count=16)
    tl.debug_barrier()
    tle.gpu.warp_specialize(
        [
            (
                staged_produce,
                (
                    Q,
                    MetaScale,
                    Setup,
                    Modes,
                    Stats,
                    first,
                    stop,
                    q,
                    sk,
                    ph,
                    pl,
                    full,
                    ready,
                    sync,
                    sums,
                    sum_sync,
                    ROWS,
                    K,
                    PROGRAMS,
                    SLOTS,
                    G0,
                    G1,
                    G2,
                    G3,
                ),
            ),
            (
                staged_consume,
                (
                    KV,
                    MetaOffset,
                    Setup,
                    Modes,
                    Partial,
                    first,
                    stop,
                    sk,
                    ph,
                    pl,
                    full,
                    empty,
                    ready,
                    ROWS,
                    K,
                    TOKENS,
                    PROGRAMS,
                    SLOTS,
                    G0,
                    G1,
                    G2,
                    G3,
                ),
            ),
        ],
        worker_num_warps=[16],
        worker_num_regs=[128],
    )


def prepare_pipeline(
    q: torch.Tensor,
    kv: torch.Tensor,
    qs: torch.Tensor,
    ks: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    repair: JITFunction,
) -> Optional[Tuple[Callable[[], None], Optional[torch.Tensor]]]:
    rows, topk = q.shape[0], indices.shape[-1]
    pages = triton.cdiv(topk, 128)
    properties = driver.active.utils.get_device_properties(
        driver.active.get_current_device()
    )
    programs = min(properties["multiprocessor_count"], rows * pages)
    groups = group_short_pages(rows, pages, programs)
    if groups is None or properties["max_shared_mem"] < 192 * 1024:
        return None
    group_rows, group_programs, prefix_rows, row_programs = groups
    splits = max(
        row_programs,
        triton.cdiv(
            group_programs - prefix_rows * row_programs, group_rows - prefix_rows
        )
        + 1,
    )
    pre = triton.next_power_of_2(triton.cdiv(rows * pages, programs) * 128)
    # This optional specialization needs metadata beyond the ordinary short path.
    try:
        partial_flags = torch.empty((rows, splits), dtype=torch.int32, device=q.device)
        partial = torch.empty(
            (rows, splits, 64, 512), dtype=torch.float32, device=q.device
        )
        stats = torch.empty((rows, splits, 64, 2), dtype=torch.float32, device=q.device)
        meta_offset = torch.empty(
            (rows, splits, pre), dtype=torch.int32, device=q.device
        )
        meta_scale = torch.empty_like(meta_offset, dtype=torch.float32)
        setup = torch.empty((rows, splits, 64, 2), dtype=torch.float32, device=q.device)
        modes = torch.empty((rows, splits), dtype=torch.int32, device=q.device)
    except torch.OutOfMemoryError:
        return None
    kernel_args = (
        q,
        kv,
        qs,
        ks,
        indices,
        partial,
        stats,
        partial_flags,
        rows,
        topk,
        kv.shape[0] * 64,
        scale,
        programs,
        splits,
        groups,
        64,
        16,
        meta_offset,
        meta_scale,
        setup,
        modes,
    )
    grid = (programs, 1, 1)
    try:
        kernel = staged_main.warmup(
            *kernel_args,
            grid=grid,
            num_warps=16,
            num_stages=1,
            maxnreg=128,
            enable_backend_opt=True,
            enable_llc_opt=True,
            extern_libs={"native": native_library(QUERY_NORM_IR)},
        )
    except (CompilationError, OutOfResources, RuntimeError) as error:
        logging.getLogger(__name__).debug(
            "Sparse short pipeline unavailable: %s", error
        )
        return None
    runner = kernel[grid]
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
        1,
    )

    def run_short():
        runner(*kernel_args)
        combine_runner(*combine_args)

    return run_short, flags
