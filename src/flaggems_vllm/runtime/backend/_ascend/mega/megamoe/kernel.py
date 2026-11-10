# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Directed persistent MegaMoE device stages.

Cube/Vector rings use flags 0/1 for GEMM1, 2/3 for GEMM2, 4 for activation
ready and 5 for activation reuse. Progress is published only after output
DMA completion. The output dispatcher has one writer per consumer mailbox.
"""
import triton
import triton.experimental.tle as tle
import triton.experimental.tle.language.raw as raw
import triton.language as tl
from triton.language.extra.cann.extension import scope

from . import bindings
from .prepare import (
    _begin_counted,
    _prepare_routes,
    _prepare_strided_input,
    _reset_progress,
)


@triton.jit
def _read_epoch(Epoch):
    token = tl.zeros((1,), tl.int64)
    token = raw.call(bindings.read_epoch, [Epoch, token], output_indices=[1])
    return tle.dsa.extract_element(token, [tl.full((), 0, tl.int32)])


pipe = tle.dsa.ascend.PIPE


@triton.jit
def _queue_cube_worker(
    TaskFlags,
    TaskMeta,
    Packed,
    Ring,
    Activation,
    Errors,
    Control,
    W1,
    W2,
    Epoch,
    worker,
    H: tl.constexpr,
    I: tl.constexpr,
    COMPUTE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    POOL_ROWS: tl.constexpr,
    ERROR_BASE: tl.constexpr,
    FUSED_G1: tl.constexpr = False,
):
    ctl = Control + worker * 48
    raw.call(bindings.cube_read_epoch, [Epoch, ctl], output_indices=[])
    epoch = tl.load(ctl, volatile=True)
    m = tl.arange(0, BM)
    n0 = tl.arange(0, BN)
    k0 = tl.arange(0, BK)
    offset = m[:, None] * BN + n0[None, :]
    tle.dsa.ascend.sync_block_set("cube", "vector", 5, pipe.PIPE_MTE2, pipe.PIPE_MTE3)
    raw.call(bindings.wait_cube, [TaskMeta, epoch], output_indices=[])
    task_count = tl.load(TaskMeta + 1, volatile=True).to(tl.int32)
    for job in range(worker, task_count, COMPUTE):
        raw.call(bindings.acquire_cube, [TaskFlags, epoch, job, ctl], output_indices=[])
        expert = tl.load(ctl + 2, volatile=True).to(tl.int32)
        start = tl.load(ctl + 3, volatile=True).to(tl.int32)
        valid = tl.load(ctl, volatile=True).to(tl.int32)
        if valid < 0:
            tl.store(Errors + (ERROR_BASE + worker) * 16, 41)
        if valid > 0:
            acquired_start = tl.load(ctl + 1, volatile=True) + start
            input_ok = (acquired_start >= 0) & (acquired_start + BM <= POOL_ROWS)
            if not input_ok:
                tl.store(Errors + (ERROR_BASE + worker) * 16, 42)
            safe_start = tl.minimum(tl.maximum(acquired_start, 0), POOL_ROWS - BM)
            for column in range(I // BN):
                n = column * BN + n0
                if FUSED_G1:
                    nn = tl.arange(0, 2 * BN)
                    both = tl.full((BM, 2 * BN), 0, tl.float32)
                    for block in range(H // BK):
                        h = block * BK + k0
                        x = tl.load(Packed + (safe_start + m[:, None]) * H + h[None, :])
                        w = tl.load(
                            W1
                            + expert.to(tl.int64) * H * 2 * I
                            + h[:, None].to(tl.int64) * (2 * I)
                            + column * (2 * BN)
                            + nn[None, :]
                        )
                        both += tl.dot(x, w)
                else:
                    gate = tl.full((BM, BN), 0, tl.float32)
                    up = tl.full((BM, BN), 0, tl.float32)
                    for block in range(H // BK):
                        h = block * BK + k0
                        x = tl.load(Packed + (safe_start + m[:, None]) * H + h[None, :])
                        weight_ptr = (
                            W1
                            + expert.to(tl.int64) * H * 2 * I
                            + h[:, None].to(tl.int64) * (2 * I)
                            + n[None, :]
                        )
                        gate += tl.dot(x, tl.load(weight_ptr))
                        up += tl.dot(x, tl.load(weight_ptr + I))
                slot = column % 2
                address = Ring + (worker * 6 + slot * 2) * BM * BN
                tle.dsa.ascend.sync_block_wait(
                    "vector", "cube", slot, pipe.PIPE_MTE2, pipe.PIPE_FIX
                )
                if FUSED_G1:
                    tl.store(
                        address + m[:, None] * (2 * BN) + tl.arange(0, 2 * BN)[None, :],
                        both,
                    )
                else:
                    tl.store(address + offset, gate)
                    tl.store(address + BM * BN + offset, up)
                tle.dsa.ascend.sync_block_set(
                    "cube", "vector", slot, pipe.PIPE_FIX, pipe.PIPE_MTE2
                )
            tle.dsa.ascend.sync_block_wait(
                "vector", "cube", 4, pipe.PIPE_MTE3, pipe.PIPE_MTE2
            )
            for column in range(H // BN):
                n = column * BN + n0
                acc = tl.full((BM, BN), 0, tl.float32)
                for block in range(I // BK):
                    i = block * BK + k0
                    a = tl.load(
                        Activation
                        + worker.to(tl.int64) * BM * I
                        + m[:, None].to(tl.int64) * I
                        + i[None, :]
                    )
                    weight2 = tl.load(
                        W2
                        + expert.to(tl.int64) * I * H
                        + i[:, None].to(tl.int64) * H
                        + n[None, :]
                    )
                    acc += tl.dot(a, weight2)
                slot = column % 2
                address = Ring + (worker * 6 + 4 + slot) * BM * BN
                tle.dsa.ascend.sync_block_wait(
                    "vector", "cube", 2 + slot, pipe.PIPE_MTE2, pipe.PIPE_FIX
                )
                tl.store(address + offset, acc)
                tle.dsa.ascend.sync_block_set(
                    "cube", "vector", 2 + slot, pipe.PIPE_FIX, pipe.PIPE_MTE2
                )
            tle.dsa.ascend.sync_block_set(
                "cube", "vector", 5, pipe.PIPE_MTE2, pipe.PIPE_MTE3
            )
    for event_id in tl.static_range(4):
        tle.dsa.ascend.sync_block_wait(
            "vector", "cube", event_id, pipe.PIPE_MTE2, pipe.PIPE_FIX
        )


@triton.jit
def _split_queue_vector_worker(
    TaskFlags,
    TaskMeta,
    Headers,
    Ring,
    Activation,
    Errors,
    Control,
    Return0,
    epoch,
    worker,
    H: tl.constexpr,
    I: tl.constexpr,
    WORLD: tl.constexpr,
    COMPUTE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    POOL_ROWS: tl.constexpr,
    ROUTES: tl.constexpr,
    FUSED_G1: tl.constexpr = False,
    Progress=None,
    SOURCE_ROUTES: tl.constexpr = 0,
    ATOMIC_PARTIAL: tl.constexpr = False,
    TOPK: tl.constexpr = 1,
    NATIVE_ATOMIC: tl.constexpr = False,
    RETURN_BLOCK: tl.constexpr = 0,
    RETURN_CHUNKS: tl.constexpr = 0,
):
    ctl = Control + worker * 48 + 16 + tle.dsa.ascend.sub_vec_id() * 16
    m = tl.arange(0, BM)
    VN: tl.constexpr = BN // 2
    lane = tle.dsa.ascend.sub_vec_id() * VN
    n0 = tl.arange(0, VN)
    offset = m[:, None] * BN + lane + n0[None, :]
    for event_id in tl.static_range(4):
        tle.dsa.ascend.sync_block_set(
            "vector", "cube", event_id, pipe.PIPE_MTE2, pipe.PIPE_FIX
        )
    meta_token = tl.zeros((1,), tl.int64)
    meta_token = raw.call(
        bindings.wait_epoch, [TaskMeta, epoch, meta_token], output_indices=[2]
    )
    task_count = tl.load(TaskMeta + 1, volatile=True).to(tl.int32)
    for job in range(worker, task_count, COMPUTE):
        raw.call(
            bindings.acquire_vector, [TaskFlags, epoch, job, ctl], output_indices=[]
        )
        start = tl.load(ctl + 3, volatile=True).to(tl.int32)
        valid = tl.load(ctl, volatile=True).to(tl.int32)
        base = tl.load(ctl + 1, volatile=True)
        if valid < 0:
            tl.store(Errors + (WORLD + worker) * 16, 41)
        if valid > 0:
            tle.dsa.ascend.sync_block_wait(
                "cube", "vector", 5, pipe.PIPE_MTE2, pipe.PIPE_MTE3
            )
            for column in range(I // BN):
                slot = column % 2
                address = Ring + (worker * 6 + slot * 2) * BM * BN
                tle.dsa.ascend.sync_block_wait(
                    "cube", "vector", slot, pipe.PIPE_FIX, pipe.PIPE_MTE2
                )
                if FUSED_G1:
                    gate = tl.load(address + m[:, None] * (2 * BN) + lane + n0[None, :])
                    up = tl.load(
                        address + BN + m[:, None] * (2 * BN) + lane + n0[None, :]
                    )
                else:
                    gate = tl.load(address + offset)
                    up = tl.load(address + BM * BN + offset)
                value = gate / (1 + tl.exp(-gate)) * up
                n = column * BN + lane + n0
                tl.store(
                    Activation
                    + worker.to(tl.int64) * BM * I
                    + m[:, None].to(tl.int64) * I
                    + n[None, :],
                    value,
                )
                tle.dsa.ascend.sync_block_set(
                    "vector", "cube", slot, pipe.PIPE_MTE2, pipe.PIPE_FIX
                )
            tle.dsa.ascend.sync_block_set(
                "vector", "cube", 4, pipe.PIPE_MTE3, pipe.PIPE_MTE2
            )
            row_begin = base + start
            base_ok = (base >= 0) & (row_begin >= 0) & (row_begin + valid <= POOL_ROWS)
            safe_begin = tl.minimum(tl.maximum(row_begin, 0), POOL_ROWS - BM)
            row_mask = (m < valid) & base_ok
            route = tl.load(Headers + (safe_begin + m) * 8, row_mask, 0)
            bits = tl.load(Headers + (safe_begin + m) * 8 + 1, row_mask, 0)
            route_ok = (route >= 0) & (route < ROUTES)
            if ~base_ok | (tl.sum(((m < valid) & ~route_ok).to(tl.int32), 0) != 0):
                tl.store(Errors + (WORLD + worker) * 16, tl.where(base_ok, 23, 22))
            prob = bits.to(tl.float32, bitcast=True)
            for column in range(H // BN):
                slot = column % 2
                address = Ring + (worker * 6 + 4 + slot) * BM * BN
                tle.dsa.ascend.sync_block_wait(
                    "cube", "vector", 2 + slot, pipe.PIPE_FIX, pipe.PIPE_MTE2
                )
                acc = tl.load(address + offset)
                n = column * BN + lane + n0
                weighted = acc * prob[:, None]
                if ATOMIC_PARTIAL and not NATIVE_ATOMIC:
                    safe_route = tl.minimum(tl.maximum(route, 0), ROUTES - 1)
                    row = (safe_route // TOPK).to(tl.int64)
                    tl.atomic_add(
                        Return0 + row[:, None] * H + n[None, :],
                        weighted,
                        (row_mask & route_ok)[:, None],
                        sem="relaxed",
                    )
                else:
                    raw.call(
                        bindings.scatter_rows,
                        [
                            weighted,
                            route,
                            Return0,
                            ctl,
                            tl.where(base_ok, valid, 0),
                            H,
                            column * BN + lane,
                            ROUTES,
                            BM,
                            VN,
                            TOPK if ATOMIC_PARTIAL else 0,
                        ],
                        output_indices=[],
                    )
                    scatter_error = tl.load(ctl + 2, volatile=True)
                    if scatter_error != 0:
                        tl.store(Errors + (WORLD + worker) * 16, scatter_error)
                tle.dsa.ascend.sync_block_set(
                    "vector", "cube", 2 + slot, pipe.PIPE_MTE2, pipe.PIPE_FIX
                )
            if RETURN_BLOCK:
                raw.call(
                    bindings.advance_return_chunk,
                    [
                        route,
                        Progress,
                        valid,
                        SOURCE_ROUTES,
                        TOPK,
                        RETURN_BLOCK,
                        RETURN_CHUNKS,
                        (worker * 2 + tle.dsa.ascend.sub_vec_id()).to(tl.int32),
                        COMPUTE * 2,
                        WORLD,
                    ],
                    output_indices=[],
                )
    tle.dsa.ascend.sync_block_wait("cube", "vector", 5, pipe.PIPE_MTE2, pipe.PIPE_MTE3)


@triton.jit
def _queue_pack(
    TaskMeta,
    MetaSources,
    MergedPacked,
    MergedHeaders,
    MergedReady,
    LocalSlots,
    SendControl,
    PeerX,
    PeerTokens,
    PeerExperts,
    PeerWeights,
    PeerCounts,
    PeerReturns,
    PeerFlags,
    LocalFlags,
    Epoch,
    Ready,
    Ring,
    Activation,
    Errors,
    Control,
    W1,
    W2,
    Return,
    Output,
    Counts,
    X,
    Ids,
    Probs,
    PublishStatus,
    X0: tl.constexpr,
    X1: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    ERROR_WORDS: tl.constexpr,
    AIVS: tl.constexpr,
    PROBS_BF16: tl.constexpr,
    T: tl.constexpr,
    OUTPUT_T: tl.constexpr,
    LOGICAL_H: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    E: tl.constexpr,
    C: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    COMPUTE: tl.constexpr,
    DISPATCH_AIVS: tl.constexpr,
    SCRATCH_WORDS: tl.constexpr,
    NATIVE_REV: tl.constexpr,
    QueueFlags,
    Progress,
    OutputRequests,
    OutputAssignments,
    PeerExpertCounts,
    PeerChunkCounts,
    RETURN_BLOCK: tl.constexpr,
    RETURN_CHUNKS: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 128,
    BK: tl.constexpr = 64,
    FUSED_G1: tl.constexpr = False,
    ATOMIC_PARTIAL: tl.constexpr = False,
    NO_INPUT_PADDING: tl.constexpr = False,
    RM: tl.constexpr = 32,
    RN: tl.constexpr = 512,
    BATCH_ROWS: tl.constexpr = 8,
    NATIVE_ATOMIC: tl.constexpr = False,
):
    pid = tl.program_id(0)
    SLOT_TOPK: tl.constexpr = 1 if ATOMIC_PARTIAL else K
    LINEAR_INPUT: tl.constexpr = (
        LOGICAL_H == H and X1 == 1 and (OUTPUT_T <= 1 or X0 == H)
    )
    ROUTE_LIMIT: tl.constexpr = 128 // KP if KP <= 128 else 1
    ROUTE_ROWS: tl.constexpr = (
        triton.next_power_of_2(OUTPUT_T)
        if OUTPUT_T > 0 and OUTPUT_T < ROUTE_LIMIT
        else ROUTE_LIMIT
    )
    with scope(core_mode="vector"):
        prep_actor = pid * 2 + tle.dsa.ascend.sub_vec_id()
        if prep_actor == 0:
            _begin_counted(
                Epoch,
                LocalFlags,
                Errors,
                PeerExpertCounts[RANK],
                WORLD,
                ERROR_WORDS,
                WORLD * (tl.cdiv(E, 8) * 8),
            )
            if RETURN_BLOCK:
                i = tl.arange(0, triton.next_power_of_2(WORLD * RETURN_CHUNKS))
                tl.store(PeerChunkCounts[RANK] + i, 0, i < WORLD * RETURN_CHUNKS)
    # A previous consumer can still be reading the symmetric input.
    tle.dsa.ascend.sync_block_all("all", 9)
    with scope(core_mode="vector"):
        if RETURN_BLOCK:
            _reset_progress(
                Progress,
                OutputRequests,
                OutputAssignments,
                prep_actor,
                AIVS,
                WORLD * RETURN_CHUNKS * COMPUTE * 2 * 16,
            )
        if not LINEAR_INPUT:
            _prepare_strided_input(
                X, PeerX[RANK], prep_actor, AIVS, OUTPUT_T, LOGICAL_H, H, X0, X1
            )
        _prepare_routes(
            Ids,
            Probs,
            PeerTokens[RANK],
            PeerExperts[RANK],
            PeerWeights[RANK],
            PeerCounts[RANK],
            PeerExpertCounts[RANK],
            PeerChunkCounts[RANK],
            prep_actor,
            AIVS,
            OUTPUT_T,
            K,
            KP,
            E,
            C,
            WORLD,
            I0,
            I1,
            P0,
            P1,
            ROUTE_ROWS,
            RETURN_BLOCK,
            RETURN_CHUNKS,
        )
        # Reuse the dispatch arena for input DMA and slot clearing.
        scratch = tl.zeros((SCRATCH_WORDS,), tl.int32)
        if OUTPUT_T > 0 and Probs.dtype.element_ty == tl.int16:
            scratch = raw.call(
                bindings.prepare_weights16,
                [
                    Probs,
                    PeerExperts[RANK],
                    PeerWeights[RANK],
                    prep_actor,
                    AIVS,
                    WORLD,
                    OUTPUT_T,
                    C,
                    KP,
                    ROUTE_ROWS,
                    tl.full((), P0, tl.int64),
                    tl.full((), P1, tl.int64),
                    1 if PROBS_BF16 else 0,
                    scratch,
                ],
                output_indices=[13],
            )
        scratch = raw.call(
            bindings.prepare_memory,
            [
                X,
                PeerX[RANK],
                LocalSlots,
                prep_actor,
                AIVS,
                WORLD,
                tl.full((), OUTPUT_T * H if LINEAR_INPUT else 0, tl.int64),
                tl.full((), T * SLOT_TOPK * H, tl.int64),
                tl.full((), C * SLOT_TOPK * H, tl.int64),
                scratch,
            ],
            output_indices=[9],
        )
    # Finish source and private-state stores before any consumer starts.
    tle.dsa.ascend.sync_block_all("all", 10)
    dispatch_groups: tl.constexpr = DISPATCH_AIVS // 2
    source_shards: tl.constexpr = DISPATCH_AIVS // WORLD
    with scope(core_mode="vector"):
        epoch = _read_epoch(Epoch)
        for peer in tl.static_range(WORLD):
            if prep_actor == peer:
                token = tl.zeros((1,), tl.int64)
                token = raw.call(
                    bindings.signal,
                    [epoch, PeerFlags[peer] + RANK * 16, token],
                    output_indices=[2],
                )
                # Consume the raw signal's completion token.
                tl.store(PublishStatus + peer, tl.sum(token, 0))
        if pid < dispatch_groups:
            source_rank = prep_actor % WORLD
            for source in tl.static_range(WORLD):
                if source_rank == source:
                    token = tl.zeros((1,), tl.int64)
                    scratch, token = raw.call(
                        bindings.dispatch_inputs,
                        [
                            PeerX[source],
                            PeerTokens[source] + RANK * C,
                            PeerExperts[source] + RANK * C * KP,
                            PeerWeights[source] + RANK * C * KP,
                            PeerCounts[source] + RANK * 16,
                            PeerExpertCounts[source] + RANK * (tl.cdiv(E, 8) * 8),
                            LocalFlags + source * 16,
                            MergedPacked,
                            MergedHeaders,
                            Ready,
                            epoch,
                            source,
                            RANK,
                            WORLD,
                            C,
                            H,
                            E,
                            K,
                            KP,
                            BM,
                            MetaSources,
                            MergedReady,
                            TaskMeta,
                            BATCH_ROWS,
                            NO_INPUT_PADDING,
                            prep_actor // WORLD,
                            source_shards,
                            scratch,
                            token,
                        ],
                        output_indices=[27, 28],
                    )
                    if tl.sum(token, 0) != epoch:
                        tl.store(Errors + source * 16, 1)
    with scope(core_mode="vector"):
        if pid == dispatch_groups:
            if tle.dsa.ascend.sub_vec_id() == 0:
                plan_snapshot = tl.zeros(
                    (triton.next_power_of_2(E * WORLD * 32),), tl.int32
                )
                # A tensor<1xi32> raw-call output is scalarized by the
                # Ascend lowering pipeline; retain a DMA-sized scratch vector.
                TOTALS: tl.constexpr = triton.next_power_of_2(E) if E > 8 else 8
                plan_totals = tl.zeros((TOTALS,), tl.int32)
                plan_snapshot, plan_totals = raw.call(
                    bindings.plan_experts,
                    [
                        Ready,
                        MergedReady,
                        MetaSources,
                        TaskMeta,
                        Counts,
                        epoch,
                        WORLD,
                        E,
                        BM,
                        plan_snapshot,
                        plan_totals,
                    ],
                    output_indices=[9, 10],
                )
    with scope(core_mode="vector"):
        if pid == dispatch_groups:
            if tle.dsa.ascend.sub_vec_id() == 1:
                queue_scratch = tl.zeros(
                    (
                        triton.next_power_of_2(
                            triton.cdiv(2 * E, 8) * 8 + 32 * E * WORLD
                        ),
                    ),
                    tl.int32,
                )
                queue_scratch = raw.call(
                    bindings.schedule_ready,
                    [
                        Ready,
                        MergedReady,
                        TaskMeta,
                        QueueFlags,
                        epoch,
                        WORLD,
                        E,
                        BM,
                        queue_scratch,
                    ],
                    output_indices=[8],
                )
    if RETURN_BLOCK:
        with scope(core_mode="vector"):
            if pid == 0 and tle.dsa.ascend.sub_vec_id() == 1:
                out_scratch = tl.zeros((64,), tl.int32)
                out_scratch = raw.call(
                    bindings.schedule_output,
                    [
                        OutputRequests,
                        OutputAssignments,
                        epoch,
                        AIVS,
                        1,
                        WORLD * tl.cdiv(T, RM) * (H // RN),
                        16,
                        out_scratch,
                    ],
                    output_indices=[7],
                )
    worker_base: tl.constexpr = dispatch_groups + 1
    worker = (pid + COMPUTE - worker_base) % COMPUTE
    is_worker = pid != 0
    with scope(core_mode="cube"):
        if is_worker:
            _queue_cube_worker(
                QueueFlags,
                TaskMeta,
                MergedPacked,
                Ring,
                Activation,
                Errors,
                Control,
                W1,
                W2,
                Epoch,
                worker,
                H,
                I,
                COMPUTE,
                BM,
                BN,
                BK,
                WORLD * (C * K + E * BM),
                WORLD,
                FUSED_G1,
            )
    with scope(core_mode="vector"):
        if is_worker:
            _split_queue_vector_worker(
                QueueFlags,
                TaskMeta,
                MergedHeaders,
                Ring,
                Activation,
                Errors,
                Control,
                LocalSlots,
                epoch,
                worker,
                H,
                I,
                WORLD,
                COMPUTE,
                BM,
                BN,
                WORLD * (C * K + E * BM),
                WORLD * C * K,
                FUSED_G1,
                Progress,
                C * K,
                ATOMIC_PARTIAL,
                K,
                NATIVE_ATOMIC,
                RETURN_BLOCK,
                RETURN_CHUNKS,
            )
    if not RETURN_BLOCK:
        tle.dsa.ascend.sync_block_all("all", 13)
    with scope(core_mode="vector"):
        actor = pid * 2 + tle.dsa.ascend.sub_vec_id()
        if RETURN_BLOCK:
            sm = tl.arange(0, RM)
            sn = tl.arange(0, RN)
            nrows: tl.constexpr = tl.cdiv(T, RM)
            ncols: tl.constexpr = H // RN
            previous = tl.full((), -1, tl.int32)
            if actor != 1:
                sequence = tl.full((), 1, tl.int64)
                active = tl.full((), True, tl.int1)
                while active:
                    ticket = epoch << 32 | sequence
                    ack = tl.zeros((1,), tl.int64)
                    ack = raw.call(
                        bindings.signal,
                        [ticket, OutputRequests + actor * 16, ack],
                        output_indices=[2],
                    )
                    out_ctl = SendControl + actor * 16
                    raw.call(
                        bindings.wait_output_task,
                        [OutputAssignments + actor * 16, ticket, out_ctl],
                        output_indices=[],
                    )
                    first_task = tl.load(out_ctl, volatile=True).to(tl.int32)
                    end_task = tl.load(out_ctl + 1, volatile=True).to(tl.int32)
                    active = first_task < end_task
                    for assigned_task in range(first_task, end_task):
                        source = (assigned_task // (nrows * ncols)).to(tl.int32)
                        row_tile = assigned_task // ncols % nrows
                        chunk = row_tile * RM // RETURN_BLOCK
                        bucket = source * RETURN_CHUNKS + chunk
                        if bucket != previous:
                            for peer in tl.static_range(WORLD):
                                if source == peer:
                                    raw.call(
                                        bindings.wait_chunk_complete,
                                        [
                                            PeerChunkCounts[peer]
                                            + RANK * RETURN_CHUNKS
                                            + chunk,
                                            LocalFlags + peer * 16,
                                            Progress + bucket * COMPUTE * 2 * 16,
                                            Errors,
                                            epoch,
                                            WORLD,
                                            COMPUTE,
                                        ],
                                        output_indices=[],
                                    )
                            previous = bucket
                        row = row_tile * RM + sm
                        col = assigned_task % ncols * RN + sn
                        acc = tl.full((RM, RN), 0, tl.float32)
                        if ATOMIC_PARTIAL:
                            acc = tl.load(
                                LocalSlots
                                + (source * C + row[:, None].to(tl.int64)) * H
                                + col[None, :],
                                (row < T)[:, None],
                                0,
                            )
                        else:
                            for slot in tl.static_range(K):
                                acc += tl.load(
                                    LocalSlots
                                    + (
                                        (source * C + row[:, None].to(tl.int64)) * K
                                        + slot
                                    )
                                    * H
                                    + col[None, :],
                                    (row < T)[:, None],
                                    0,
                                )
                        for peer in tl.static_range(WORLD):
                            if source == peer:
                                tl.store(
                                    PeerReturns[peer]
                                    + (RANK * C + row[:, None].to(tl.int64)) * H
                                    + col[None, :],
                                    acc,
                                    (row < T)[:, None],
                                )
                    sequence += 1
        else:
            sm = tl.arange(0, RM)
            sn = tl.arange(0, RN)
            nrows: tl.constexpr = tl.cdiv(T, RM)
            ncols: tl.constexpr = H // RN
            for tile in range(actor, WORLD * nrows * ncols, AIVS):
                source = tile // (nrows * ncols)
                row = tile // ncols % nrows * RM + sm
                col = tile % ncols * RN + sn
                acc = tl.full((RM, RN), 0, tl.float32)
                if ATOMIC_PARTIAL:
                    acc = tl.load(
                        LocalSlots
                        + (source * C + row[:, None].to(tl.int64)) * H
                        + col[None, :],
                        (row < T)[:, None],
                        0,
                    )
                else:
                    for slot in tl.static_range(K):
                        acc += tl.load(
                            LocalSlots
                            + ((source * C + row[:, None].to(tl.int64)) * K + slot) * H
                            + col[None, :],
                            (row < T)[:, None],
                            0,
                        )
                for peer in tl.static_range(WORLD):
                    if source == peer:
                        tl.store(
                            PeerReturns[peer]
                            + (RANK * C + row[:, None].to(tl.int64)) * H
                            + col[None, :],
                            acc,
                            (row < T)[:, None],
                        )
    tle.dsa.ascend.sync_block_all("all", 12)
    with scope(core_mode="vector"):
        epoch = _read_epoch(Epoch)
        if pid == 0:
            if tle.dsa.ascend.sub_vec_id() == 0:
                for peer in tl.static_range(WORLD):
                    token = tl.zeros((1,), tl.int64)
                    token = raw.call(
                        bindings.signal,
                        [epoch, PeerFlags[peer] + (WORLD + RANK) * 16, token],
                        output_indices=[2],
                    )
                    tl.store(
                        Errors + (WORLD + COMPUTE + peer) * 16,
                        (tl.sum(token, 0) != epoch).to(tl.int64),
                    )
        completed = tl.zeros((1,), tl.int64)
        completed = raw.call(
            bindings.wait_many,
            [LocalFlags + WORLD * 16, epoch, WORLD, completed],
            output_indices=[3],
        )
        fault_idx = tl.arange(0, triton.next_power_of_2(WORLD + COMPUTE))
        faults = tl.load(Errors + fault_idx * 16, fault_idx < WORLD + COMPUTE, 0)
        failed = (tl.sum(completed, 0) != epoch) | (tl.sum(faults.to(tl.int32), 0) != 0)
        out_actor = pid * 2 + tle.dsa.ascend.sub_vec_id()
        if LOGICAL_H == H:
            out_row = tl.arange(0, 16)
            out_col = tl.arange(0, 256)
            for tile in range(out_actor, tl.cdiv(OUTPUT_T, 16) * (H // 256), AIVS):
                row = tile // (H // 256) * 16 + out_row
                h = tile % (H // 256) * 256 + out_col
                acc = tl.full((16, 256), 0, tl.float32)
                for slot in tl.static_range(WORLD):
                    ptr = Return + (slot * C + row[:, None].to(tl.int64)) * H
                    acc += tl.load(
                        ptr + h[None, :], ((row < OUTPUT_T) & ~failed)[:, None], 0
                    )
                tl.store(
                    Output + row[:, None].to(tl.int64) * H + h[None, :],
                    tl.where(failed, float("nan"), acc),
                    (row < OUTPUT_T)[:, None],
                )
        else:
            offset = tl.arange(0, 1024)
            for tile in range(out_actor, tl.cdiv(OUTPUT_T * LOGICAL_H, 1024), AIVS):
                pos = tile.to(tl.int64) * 1024 + offset
                flat_row, flat_col = pos // LOGICAL_H, pos % LOGICAL_H
                flat_acc = tl.full((1024,), 0, tl.float32)
                for slot in tl.static_range(WORLD):
                    ptr = Return + (slot * C + flat_row) * H + flat_col
                    flat_acc += tl.load(ptr, (pos < OUTPUT_T * LOGICAL_H) & ~failed, 0)
                tl.store(
                    Output + pos,
                    tl.where(failed, float("nan"), flat_acc),
                    pos < OUTPUT_T * LOGICAL_H,
                )
    tle.dsa.ascend.sync_block_all("all", 14)
    with scope(core_mode="vector"):
        if pid == 0:
            if tle.dsa.ascend.sub_vec_id() == 0:
                epoch = _read_epoch(Epoch)
                for peer in tl.static_range(WORLD):
                    token = tl.zeros((1,), tl.int64)
                    token = raw.call(
                        bindings.signal,
                        [epoch, PeerFlags[peer] + (2 * WORLD + RANK) * 16, token],
                        output_indices=[2],
                    )
                    tl.store(
                        Errors + (WORLD + COMPUTE + peer) * 16,
                        (tl.sum(token, 0) != epoch).to(tl.int64),
                    )
