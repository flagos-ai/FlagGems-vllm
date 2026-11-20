# Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0
"""Persistent one-page sparse MLA on the supported MThreads backend."""
import logging
from functools import lru_cache

import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError
from triton.experimental.tle import language as tle
from triton.runtime.errors import OutOfResources

from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_metadata import KV_TOKEN_BYTES
from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_native import (
    VECTOR_EXP2_IR,
    bounded_copy16,
    bounded_copy_wait,
    lma_wait,
    native_library,
    vector_exp2,
)
from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_pipeline import (
    OCOL,
    OROW,
    PV,
    kv_copy_address,
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
    Fast,
    Rows,
    PROGRAMS: tl.constexpr,
    QueryDone,
    TME_QUERY: tl.constexpr,
):
    # Each CTA owns complete queries; skipped queries do not advance phases.
    query_start = Rows * tl.program_id(0) // PROGRAMS
    query_stop = Rows * (tl.program_id(0) + 1) // PROGRAMS
    epoch = 0
    for request in range(query_start, query_stop):
        row = request.to(tl.int64)
        mode = tl.load(Fast + row, volatile=True)
        if mode != 0:
            accum = tle.gpu.set_layout(tl.full((64, 512), 0.0, tl.float32), PV)
            slot = epoch & 1
            tle.gpu.barrier_wait(Full, phaseIdx=slot)
            (high, low) = (ph, pl)
            values = sk.slot(slot)
            accum = tle.gpu.wgmma(high, values, acc=accum)
            accum = tle.gpu.wgmma(low, values, acc=accum)
            accum = tle.gpu.wgmma_wait(0, accum)
            tle.gpu.barrier_arrive(PEmpty, phaseIdx=slot)
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
            tle.gpu.barrier_wait(NormReady, phaseIdx=epoch & 1)
            inverse = tl.load(tle.gpu.local_ptr(FinalInv, (h,)), volatile=True)
            if TME_QUERY:
                result = accum * inverse[:, None]
            else:
                scale = tl.load(PScale + row * 64 + h, volatile=True)
                result = accum * scale[:, None] * inverse[:, None]
            tl.store(Output + (row * 64 + h[:, None]) * 512 + d[None, :], result)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(QueryDone, phaseIdx=epoch & 1)
            epoch += 1


@triton.jit
def produce(
    Q,
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
    SumSync,
    Selected,
    Fast,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    USE_BYTE_OFFSETS: tl.constexpr,
    Rows,
    PROGRAMS: tl.constexpr,
    QueryDone,
    QReady,
):
    q = tle.gpu.alloc((64, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    # Each CTA owns complete queries; skipped queries do not advance phases.
    query_start = Rows * tl.program_id(0) // PROGRAMS
    query_stop = Rows * (tl.program_id(0) + 1) // PROGRAMS
    epoch = 0
    for request in range(query_start, query_stop):
        row = request.to(tl.int64)
        mode = tl.load(Fast + row, volatile=True)
        if mode != 0:
            h = tl.arange(0, 64)
            col = tl.arange(0, 128)
            qs = tl.load(QS + row * 64 + h)
            ds = tl.load(PScale + row * 64 + h)
            reciprocal = tl.div_rn(1.0, ds)
            copy_origin = tl.inline_asm_elementwise(
                "",
                constraints="=R,0",
                args=[tl.full((), 128, tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            cr = tl.arange(128, 256) - copy_origin
            cd = tl.arange(0, 32) * 16
            copy_layout: tl.constexpr = tle.gpu.BlockEncoding(
                [1, 1], [2, 16], [16, 1], [1, 0]
            )
            kv_lane_rows: tl.constexpr = 8 if isinstance(Q, tl.tensor_descriptor) else 2
            kv_layout: tl.constexpr = tle.gpu.BlockEncoding(
                [1, 1], [kv_lane_rows, 32 // kv_lane_rows], [16, 1], [1, 0]
            )
            crr = tle.gpu.set_layout(tl.broadcast_to(cr[:, None], (128, 32)), kv_layout)
            cdd = tle.gpu.set_layout(tl.broadcast_to(cd[None, :], (128, 32)), kv_layout)
            if isinstance(Q, tl.tensor_descriptor):
                tle.gpu.copy(Q, q, (64, 512), offsets=(request * 64, 0), barrier=QReady)
            else:
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
            if USE_BYTE_OFFSETS:
                sentinel = tl.full((), TOKENS * KV_TOKEN_BYTES, tl.uint32).to(tl.int32)
            else:
                sentinel = TOKENS
            init_location = tl.load(
                Locations + row * 128 + crr, crr < 128, sentinel, volatile=True
            )
            init_src = kv_copy_address(KV, init_location, cdd, TOKENS, USE_BYTE_OFFSETS)
            init_dst0 = (
                tle.gpu.local_ptr(sk.slot(epoch & 1), (crr, cdd))
                .to(tl.uint64)
                .to(tl.uint32)
            )
            bounded_copy16(
                init_dst0,
                init_src,
                KV.to(tl.uint64),
                tl.full((), TOKENS * 512, tl.uint64),
            )
            bounded_copy_wait(tl.full((), 0, tl.int32))
            if isinstance(Q, tl.tensor_descriptor):
                tle.gpu.barrier_wait(QReady, phaseIdx=epoch & 1)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Sync, phaseIdx=epoch & 1)
            tle.gpu.barrier_wait(Sync, phaseIdx=epoch & 1)
            tle.gpu.barrier_wait(QueryDone, phaseIdx=epoch & 1)
            total_matrix = tl.full((64, 128), 0.0, tl.float32)
            slot = epoch & 1
            offsets = col
            ks = tl.load(Selected + row * 128 + offsets, offsets < 128, 0.0)
            valid = (offsets < 128) & (ks > 0.0)
            dot = tle.gpu.wgmma(q, sk.slot(slot), trans_b=True)
            dot = tle.gpu.wgmma_wait(0, dot).to(tl.float32)
            tle.gpu.barrier_wait(PEmpty, phaseIdx=slot)
            (high_tile, low_tile) = (ph, pl)
            shifted = dot * qs[:, None] * ks[None, :] * SCALE
            if isinstance(Q, tl.tensor_descriptor):
                exp_value = vector_exp2(shifted * 1.4426950408889634, 64, 128)
            else:
                exp_value = tl.exp(shifted)
            probability = tl.where(valid[None, :], exp_value, 0.0)
            total_matrix += probability
            normalized = probability * ks[None, :] * reciprocal[:, None]
            high = normalized.to(tl.float8e4nv)
            low = (normalized - high.to(tl.float32)).to(tl.float8e4nv)
            phh = tl.broadcast_to(h[:, None], (64, 128))
            phc = tl.broadcast_to(col[None, :], (64, 128))
            tl.store(tle.gpu.local_ptr(high_tile, (phh, phc)), high)
            tl.store(tle.gpu.local_ptr(low_tile, (phh, phc)), low)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Full, phaseIdx=slot)
            total = tl.sum(tl.sum(tl.reshape(total_matrix, (64, 4, 32)), 2), 1)
            inverse = tl.where(total > 0, 1.0 / tl.where(total > 0, total, 1.0), 0.0)
            if isinstance(Q, tl.tensor_descriptor):
                inverse *= ds
            tl.store(tle.gpu.local_ptr(FinalInv, (h,)), inverse)
            lse = tl.where(total > 0, tl.log(total), float("inf"))
            tl.store(LSE + row * 64 + h, lse)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(NormReady, phaseIdx=epoch & 1)
            # Complete all producer LSE scratch reads before the next query.
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(SumSync, phaseIdx=epoch & 1)
            tle.gpu.barrier_wait(SumSync, phaseIdx=epoch & 1)
            epoch += 1


@triton.jit
def persistent_k128(
    Q,
    KV,
    QS,
    Locations,
    PScale,
    Fast,
    Output,
    LSE,
    Selected,
    Rows,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    USE_BYTE_OFFSETS: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    sk = tle.gpu.alloc((2, 128, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((64, 128), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((64, 128), tl.float8e4nv, nv_mma_shared_layout=True)
    inverse = tle.gpu.alloc((64,), tl.float32, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barrier(arrive_count=16)
    pe = tle.gpu.alloc_barrier(arrive_count=16, init=tle.gpu.READY)
    norm_ready = tle.gpu.alloc_barrier(arrive_count=16)
    sync = tle.gpu.alloc_barrier(arrive_count=16)
    sb = tle.gpu.alloc_barrier(arrive_count=16)
    query_done = tle.gpu.alloc_barrier(arrive_count=16, init=tle.gpu.READY)
    q_ready = (
        tle.gpu.alloc_barrier(arrive_count=1, expect_bytes=64 * 512)
        if isinstance(Q, tl.tensor_descriptor)
        else full
    )
    tl.debug_barrier()
    tle.gpu.warp_specialize(
        [
            (
                produce,
                (
                    Q,
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
                    sb,
                    Selected,
                    Fast,
                    TOKENS,
                    SCALE,
                    USE_BYTE_OFFSETS,
                    Rows,
                    PROGRAMS,
                    query_done,
                    q_ready,
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
                    Fast,
                    Rows,
                    PROGRAMS,
                    query_done,
                    isinstance(Q, tl.tensor_descriptor),
                ),
            ),
        ],
        worker_num_warps=[16],
        worker_num_regs=[128],
    )


@lru_cache(maxsize=64)
def compiled_kernel(
    topk,
    tokens,
    scale,
    output_dtype,
    device_index,
    target,
    use_byte_offsets,
    programs,
    use_tme_q=False,
):
    assert topk == 128 and programs > 0
    fn = persistent_k128
    signature = dict(
        zip(
            fn.arg_names[:10],
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
                "i32",
            ],
        )
    )
    attrs = {(i,): [["tt.divisibility", 16]] for i in range(9)}
    if use_tme_q:
        signature["Q"] = "tensordesc<fp8e4nv[64,512]>"
        attrs[(0,)] = [["musa.tme_tail_divisibility", 4]]
    constants = dict(
        TOKENS=tokens, SCALE=scale, USE_BYTE_OFFSETS=use_byte_offsets, PROGRAMS=programs
    )
    try:
        return triton.compile(
            ASTSource(
                fn,
                signature,
                constants,
                attrs=attrs,
            ),
            target=target,
            options=dict(
                num_warps=16,
                num_stages=1,
                maxnreg=128,
                enable_backend_opt=True,
                enable_llc_opt=True,
                extern_libs={
                    "native": (
                        native_library(VECTOR_EXP2_IR)
                        if use_tme_q
                        else native_library()
                    )
                },
            ),
        )
    except (CompilationError, OutOfResources, RuntimeError) as error:
        logging.getLogger(__name__).debug(
            "Sparse FP8 K128 pipeline unavailable: %s", error
        )
        return None
