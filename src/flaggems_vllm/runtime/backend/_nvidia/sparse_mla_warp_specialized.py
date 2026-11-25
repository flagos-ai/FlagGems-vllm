# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.flash_mla_with_kvcache_model1 import tle_subslice
from flaggems_vllm.runtime.backend._nvidia.sparse_mla_tiles import (
    ACCUMULATOR_SCALE_FLOOR,
    QK_RECOMPUTE_THRESHOLD,
    publish_sparse_p128_fp8,
    quantize_p_fp8_before_ready,
    sink_normalization,
    sparse_fp8_publish_output,
    sparse_fp8_split_blocks,
    sparse_fp8_transpose_values,
    sparse_named_barriers,
)
from flaggems_vllm.utils import libentry, libtuner


@triton.jit
def sparse_fp8_producer(
    Q,
    QR,
    KV,
    KR,
    KS,
    Indices,
    Length,
    qb,
    qh,
    qrb,
    qrh,
    kp,
    kt,
    krp,
    krt,
    ksp,
    kst,
    ib,
    ik,
    sq,
    sr,
    sk,
    skr,
    scales,
    mask,
    cached_indices,
    qfull,
    kfull,
    kfree,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_ROPE: tl.constexpr = True,
    PRODUCER_WARPS: tl.constexpr = 4,
    CAN_ASYNC_NOPE: tl.constexpr = False,
    CAN_ASYNC_ROPE: tl.constexpr = False,
):
    batch = tl.program_id(0) // (H // 64)
    heads = (tl.program_id(0) % (H // 64)) * 64 + tl.arange(0, 64)
    dims = tl.arange(0, 512)
    ropes = tl.arange(0, 64)
    CACHE_INDICES: tl.constexpr = not HAS_ROPE and SPLITS == 1 and TOPK <= 1024
    ASYNC_KR: tl.constexpr = (
        HAS_ROPE and CAN_ASYNC_ROPE and CAN_ASYNC_NOPE and PRODUCER_WARPS == 8
    )
    if CACHE_INDICES:
        CACHE_SLOTS: tl.constexpr = triton.next_power_of_2(max(TOPK, 128)) // 128
        cache_length = (
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
        )
        cache_positions = (
            tl.arange(0, CACHE_SLOTS)[:, None] * 128 + tl.arange(0, 128)[None, :]
        )
        index_data = tl.load(
            Indices + batch * ib + cache_positions * ik,
            cache_positions < cache_length,
            -1,
        )
        tl.store(tle.gpu.local_ptr(cached_indices), index_data)
    else:
        pass
    query = tl.load(
        Q + batch * qb + heads[:, None] * qh + dims[None, :],
        eviction_policy="evict_first",
    )
    tl.store(tle.gpu.local_ptr(sq), query)
    if HAS_ROPE:
        query_rope = tl.load(
            QR + batch * qrb + heads[:, None] * qrh + ropes[None, :],
            eviction_policy="evict_first",
        )
        tl.store(tle.gpu.local_ptr(sr), query_rope)
    else:
        pass
    tle.gpu.barrier_arrive(qfull[0])
    length = (
        cache_length
        if CACHE_INDICES
        else (
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
        )
    )
    first_block, key_blocks = sparse_fp8_split_blocks(length, SPLITS)
    split_blocks = tl.cdiv(key_blocks, 2)
    split_end = tl.minimum(length, (first_block + key_blocks) * 64)
    for step in range(split_blocks):
        buf = 0
        meta_buf = step % 2
        positions = first_block * 64 + step * 128 + tl.arange(0, 128)
        if CACHE_INDICES:
            ids = tl.load(tle.gpu.local_ptr(cached_indices.slot(step)))
        else:
            ids = tl.load(
                Indices + batch * ib + positions * ik,
                positions < split_end,
                -1,
                eviction_policy="evict_first",
            )
        valid = (positions < split_end) & (ids >= 0) & (ids < N)
        ids = tl.where(valid, ids, 0).to(tl.uint64)
        pages, slots = (ids // 64, ids % 64)
        if CAN_ASYNC_NOPE:
            copy_rows = tl.broadcast_to(tl.arange(0, 128)[:, None], (128, 8))
            copy_columns = tl.broadcast_to((tl.arange(0, 8) * 16)[None, :], (128, 8))
            for key_half in tl.static_range(2):
                if step >= 1:
                    tle.gpu.barrier_wait(kfree[key_half])
                else:
                    pass
                for group in tl.static_range(4):
                    dst = tle.gpu.local_ptr(
                        tle_subslice(sk.slot(buf), [0, group * 128], [128, 128]),
                        (copy_rows, copy_columns),
                    )
                    src = (
                        KV
                        + pages[:, None] * kp
                        + slots[:, None] * kt
                        + group * 128
                        + copy_columns
                    )
                    # Keep the first K64 copies pending across the second free wait.
                    tl.inline_asm_elementwise(
                        "{ .reg .pred active; setp.ne.u32 active, $4, 0; "
                        "@active cp.async.cg.shared.global [$1], [$2], 16, $3; "
                        "mov.u32 $0, 0; }",
                        constraints="=r,r,l,r,r",
                        args=[
                            dst.to(tl.uint64).to(tl.uint32),
                            src.to(tl.uint64),
                            tl.where(valid[:, None], 16, 0),
                            ((copy_rows // 64) == key_half).to(tl.uint32),
                        ],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
                # Consumer 0 releases all RoPE rows with kfree[0].
                if key_half == 0 and ASYNC_KR:
                    rope_columns = tl.broadcast_to(
                        (tl.arange(0, 8) * 8)[None, :], (128, 8)
                    )
                    dst_r = tle.gpu.local_ptr(skr.slot(buf), (copy_rows, rope_columns))
                    src_r = (
                        KR + pages[:, None] * krp + slots[:, None] * krt + rope_columns
                    )
                    tl.inline_asm_elementwise(
                        "{ cp.async.cg.shared.global [$1], [$2], 16, $3; "
                        "mov.u32 $0, 0; }",
                        constraints="=r,r,l,r",
                        args=[
                            dst_r.to(tl.uint64).to(tl.uint32),
                            src_r.to(tl.uint64),
                            tl.where(valid[:, None], 16, 0).to(tl.uint32),
                        ],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
                else:
                    pass
        else:
            if step >= 1:
                tle.gpu.barrier_wait(kfree[0])
                tle.gpu.barrier_wait(kfree[1])
            else:
                pass
            columns = tl.arange(0, 128)
            for group in tl.static_range(4):
                content = tl.load(
                    KV
                    + pages[:, None] * kp
                    + slots[:, None] * kt
                    + group * 128
                    + columns[None, :],
                    valid[:, None],
                    0.0,
                )
                tl.store(
                    tle.gpu.local_ptr(
                        tle_subslice(sk.slot(buf), [0, group * 128], [128, 128])
                    ),
                    content,
                )
        if HAS_ROPE and not ASYNC_KR:
            rope = tl.load(
                KR + pages[:, None] * krp + slots[:, None] * krt + ropes[None, :],
                valid[:, None],
                0.0,
            )
            tl.store(tle.gpu.local_ptr(skr.slot(buf)), rope)
        else:
            pass
        kv_scale = tl.load(KS + pages * ksp + slots * kst, valid, 0.0)
        tl.store(tle.gpu.local_ptr(scales.slot(meta_buf)), kv_scale)
        if HAS_ROPE or TOPK > 512:
            tl.store(
                tle.gpu.local_ptr(mask.slot(meta_buf)),
                tl.where(valid, 0.0, -float("inf")),
            )
        else:
            pass
        if CAN_ASYNC_NOPE:
            tl.inline_asm_elementwise(
                "{ cp.async.commit_group; cp.async.wait_group 0; mov.u32 $0, $1; }",
                constraints="=r,r",
                args=[tl.arange(0, PRODUCER_WARPS * 32)],
                dtype=tl.uint32,
                is_pure=False,
                pack=1,
            )
        else:
            pass
        # Matrix transpose avoids byte stores and their shared-memory bank conflicts.
        tl.debug_barrier()
        tl.inline_asm_elementwise(
            "{ fence.proxy.async.shared::cta; mov.u32 $0, $1; }",
            constraints="=r,r",
            args=[tl.arange(0, 128)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tle.gpu.barrier_arrive(kfull[0])
        tle.gpu.barrier_arrive(kfull[1])


@triton.jit
def sparse_fp8_consumer0(
    QS,
    Length,
    Sink,
    Output,
    LSE,
    qsb,
    qsh,
    sq,
    sr,
    sk,
    skr,
    sv0,
    sp,
    alpha,
    scales,
    mask,
    factor,
    qfull,
    kfull,
    kfree,
    vfull,
    vfree,
    pfull,
    ofull,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    RepairFlags,
    HAS_ROPE: tl.constexpr = True,
    PREQUANTIZE_P: tl.constexpr = False,
):
    batch = tl.program_id(0) // (H // 64)
    heads = (tl.program_id(0) % (H // 64)) * 64 + tl.arange(0, 64)
    # Keep online softmax in base-2 units and fold row-constant scaling once.
    query_scale = tl.load(
        QS + batch * qsb + heads * qsh, eviction_policy="evict_first"
    ) * (SCALE * 1.4426950408889634)
    amplification = tl.max(tl.abs(query_scale), 0)
    key_repair = tl.full((128,), False, tl.int1)
    head_repair = tl.full((64,), False, tl.int1)
    maximum_weight_scale = tl.zeros((64,), tl.float32)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    maximum = tl.full((64,), -float("inf"), tl.float32)
    denominator = tl.zeros((64,), tl.float32)
    acc0 = tl.zeros((64, 128), tl.float32)
    previous_scale = tl.full((64,), 1.0, tl.float32)
    tle.gpu.barrier_wait(qfull[0])
    first_block, key_blocks = sparse_fp8_split_blocks(length, SPLITS)
    split_blocks = tl.cdiv(key_blocks, 2)
    for step in range(split_blocks):
        meta_buf = step % 2
        tle.gpu.barrier_wait(kfull[0])
        content_logits = tle.gpu.wgmma(
            sq, sk.slot(0), out_dtype=tl.float32, trans_b=True
        )
        if HAS_ROPE:
            logits = tle.gpu.wgmma(sr, skr.slot(0), content_logits, trans_b=True)
        else:
            logits = content_logits
        logits = tle.gpu.wgmma_wait(0, logits)
        tle.gpu.barrier_arrive(kfree[0])
        tle.gpu.barrier_arrive(kfree[1])
        kv_scale = tl.load(tle.gpu.local_ptr(scales.slot(meta_buf)))
        if HAS_ROPE or TOPK > 512:
            add_mask = tl.load(tle.gpu.local_ptr(mask.slot(meta_buf)))
        else:
            add_mask = tl.where(
                kv_scale.to(tl.uint32, bitcast=True) != 0,
                0.0,
                -float("inf"),
            )
        key_repair |= amplification * kv_scale > QK_RECOMPUTE_THRESHOLD
        scores = logits * query_scale[:, None] * kv_scale[None, :] + add_mask[None, :]
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp2(maximum - safe_maximum)
        probabilities = tl.exp2(scores - safe_maximum[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        weighted = probabilities * kv_scale[None, :]
        probability_scale = tl.max(weighted, 1) / 448.0
        next_weight_scale = tl.maximum(
            maximum_weight_scale * correction, probability_scale
        )
        probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
        head_repair |= next_weight_scale * ACCUMULATOR_SCALE_FLOOR > probability_scale
        if PREQUANTIZE_P:
            p = quantize_p_fp8_before_ready(weighted / probability_scale[:, None])
        else:
            p = weighted / probability_scale[:, None]
        beta = correction * previous_scale / probability_scale
        tle.gpu.barrier_wait(vfull[0])
        tle.gpu.barrier_wait(vfull[1])
        publish_sparse_p128_fp8(sp.slot(0), p, sp.slot(1), PREQUANTIZED=PREQUANTIZE_P)
        tl.store(tle.gpu.local_ptr(alpha.slot(0)), beta)
        tl.debug_barrier()
        tle.gpu.barrier_arrive(pfull[0])
        tle.gpu.barrier_arrive(pfull[1])
        acc0 *= beta[:, None]
        for half in tl.static_range(2):
            acc0 = tle.gpu.wgmma(
                sp.slot(half),
                tle_subslice(sv0.slot(half), [0, 0], [128, 64]),
                acc0,
                trans_b=True,
            )
        acc0 = tle.gpu.wgmma_wait(0, acc0)
        maximum = next_maximum
        maximum_weight_scale = next_weight_scale
        previous_scale = probability_scale
        tle.gpu.barrier_arrive(vfree[0])
    if SPLITS == 1:
        logsum = tl.where(
            denominator > 0,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
            float("inf"),
        )
        denominator_inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
        if HAS_SINK:
            sink = tl.load(Sink + heads)
            inverse = sink_normalization(denominator_inverse, sink, logsum)
        else:
            inverse = denominator_inverse
        inverse *= previous_scale
    else:
        # Keep partial values in physical units; merge applies the sink only once.
        inverse = previous_scale
    output_batch = batch if SPLITS == 1 else batch * SPLITS + tl.program_id(2)
    tl.store(tle.gpu.local_ptr(factor), inverse)
    tle.gpu.barrier_arrive(ofull[0], phaseIdx=0)
    sparse_fp8_publish_output(
        acc0,
        inverse,
        Output,
        output_batch,
        (tl.program_id(0) % (H // 64)),
        H,
        0,
        STREAMING=SPLITS == 1,
    )
    if SPLITS == 1:
        tl.store(LSE + batch * H + heads, logsum, eviction_policy="evict_first")
    else:
        stats = LSE + ((batch * SPLITS + tl.program_id(2)) * H + heads) * 2
        tl.store(stats, maximum * 0.6931471805599453)
        tl.store(stats + 1, denominator)
    needs_repair = (tl.max(key_repair.to(tl.int32), 0) != 0) | (
        tl.max(head_repair.to(tl.int32), 0) != 0
    )
    tl.store(
        RepairFlags
        + (batch * (H // 64) + (tl.program_id(0) % (H // 64))) * SPLITS
        + tl.program_id(2),
        needs_repair.to(tl.int32),
    )


@triton.jit
def sparse_fp8_consumer1(
    Length,
    Output,
    sk,
    sv0,
    sv1,
    sv_tail,
    sp,
    alpha,
    factor,
    kfull,
    kfree,
    vfull,
    vfree,
    pfull,
    ofull,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    SPLITS: tl.constexpr,
):
    batch = tl.program_id(0) // (H // 64)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    acc0 = tl.zeros((64, 128), tl.float32)
    acc1 = tl.zeros((64, 128), tl.float32)
    acc2 = tl.zeros((64, 128), tl.float32)
    first_block, key_blocks = sparse_fp8_split_blocks(length, SPLITS)
    split_blocks = tl.cdiv(key_blocks, 2)
    for step in range(split_blocks):
        buf = 0
        tle.gpu.barrier_wait(kfull[1])
        # Complete K64 transposes permit the next half-copy to start early.
        for outer in tl.static_range(2):
            for inner in tl.static_range(2):
                if inner == 0:
                    source = tle_subslice(
                        sk.slot(buf),
                        [outer * 64, 0],
                        [64, 128],
                    )
                    for warp_half in tl.static_range(2):
                        sparse_fp8_transpose_values(
                            source,
                            sv0.slot(outer),
                            0,
                            warp_half * 4,
                        )
                else:
                    for group in tl.static_range(1, 4):
                        source = tle_subslice(
                            sk.slot(buf),
                            [outer * 64, group * 128],
                            [64, 128],
                        )
                        for warp_half in tl.static_range(2):
                            if group == 3:
                                sparse_fp8_transpose_values(
                                    source,
                                    sv_tail.slot(outer),
                                    0,
                                    warp_half * 4,
                                )
                            else:
                                sparse_fp8_transpose_values(
                                    source,
                                    sv1.slot(outer),
                                    (group - 1) * 128,
                                    warp_half * 4,
                                )
                tl.inline_asm_elementwise(
                    "{ fence.proxy.async.shared::cta; mov.u32 $0, $1; }",
                    constraints="=r,r",
                    args=[tl.arange(0, 128)],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
                if inner == 0:
                    tle.gpu.barrier_arrive(vfull[outer])
                else:
                    tle.gpu.barrier_arrive(kfree[outer])
        for half in tl.static_range(2):
            tle.gpu.barrier_wait(pfull[half])
            if half == 0:
                correction = tl.load(tle.gpu.local_ptr(alpha.slot(half)))
                acc0 *= correction[:, None]
                acc1 *= correction[:, None]
                acc2 *= correction[:, None]
            else:
                pass
            acc0 = tle.gpu.wgmma(
                sp.slot(half),
                tle_subslice(sv1.slot(half), [0, 0], [128, 64]),
                acc0,
                trans_b=True,
            )
            acc1 = tle.gpu.wgmma(
                sp.slot(half),
                tle_subslice(sv1.slot(half), [128, 0], [128, 64]),
                acc1,
                trans_b=True,
            )
            acc2 = tle.gpu.wgmma(
                sp.slot(half),
                sv_tail.slot(half),
                acc2,
                trans_b=True,
            )
            if half == 1:
                acc0 = tle.gpu.wgmma_wait(0, acc0)
                acc1 = tle.gpu.wgmma_wait(0, acc1)
                acc2 = tle.gpu.wgmma_wait(0, acc2)
            else:
                pass
        tle.gpu.barrier_wait(vfree[0])
    tle.gpu.barrier_wait(ofull[0], phaseIdx=0)
    inverse = tl.load(tle.gpu.local_ptr(factor))
    output_batch = batch if SPLITS == 1 else batch * SPLITS + tl.program_id(2)
    sparse_fp8_publish_output(
        acc0,
        inverse,
        Output,
        output_batch,
        (tl.program_id(0) % (H // 64)),
        H,
        128,
        STREAMING=SPLITS == 1,
    )
    sparse_fp8_publish_output(
        acc1,
        inverse,
        Output,
        output_batch,
        (tl.program_id(0) % (H // 64)),
        H,
        256,
        STREAMING=SPLITS == 1,
    )

    sparse_fp8_publish_output(
        acc2,
        inverse,
        Output,
        output_batch,
        (tl.program_id(0) % (H // 64)),
        H,
        384,
        STREAMING=SPLITS == 1,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_fp8_warp_specialized"),
    key=[
        "B_BUCKET",
        "H",
        "TOPK",
        "SPLITS",
        "HAS_ROPE",
        "N",
        "CAN_ASYNC_NOPE",
        "CAN_ASYNC_ROPE",
    ],
)
@triton.jit
def sparse_fp8_warp_specialized(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Length,
    Sink,
    Output,
    LSE,
    qb: tl.constexpr,
    qh: tl.constexpr,
    qrb: tl.constexpr,
    qrh: tl.constexpr,
    kp: tl.constexpr,
    kt: tl.constexpr,
    krp: tl.constexpr,
    krt: tl.constexpr,
    qsb: tl.constexpr,
    qsh: tl.constexpr,
    ksp: tl.constexpr,
    kst: tl.constexpr,
    ib: tl.constexpr,
    ik: tl.constexpr,
    B_BUCKET: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    RepairFlags,
    SMALL_REGS: tl.constexpr,
    SMALL_WARPS: tl.constexpr,
    LARGE_REGS: tl.constexpr,
    LARGE_WARPS: tl.constexpr,
    HAS_ROPE: tl.constexpr = True,
    CAN_ASYNC_NOPE: tl.constexpr = False,
    CAN_ASYNC_ROPE: tl.constexpr = False,
):
    PRODUCER_WARPS: tl.constexpr = SMALL_WARPS if B_BUCKET == 0 else LARGE_WARPS
    PRODUCER_REGS: tl.constexpr = SMALL_REGS if B_BUCKET == 0 else LARGE_REGS
    # H800 has 50 MiB L2; large RoPE KV sets benefit from prepacking P.
    PREQUANTIZE_P: tl.constexpr = HAS_ROPE and N * (512 + 64 * 2 + 4) > 50 * 1024 * 1024
    COPY_REGS: tl.constexpr = (
        40
        if PREQUANTIZE_P and PRODUCER_WARPS == 8 and PRODUCER_REGS == 32
        else PRODUCER_REGS
    )
    sq = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
    sr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
    # Paired copies overlap NoPE buffer reuse with consumer compute.
    sk = tle.gpu.alloc([1, 128, 512], tl.float8e4nv, scope=tle.gpu.smem)
    skr = tle.gpu.alloc([1, 128, 64], tl.bfloat16, scope=tle.gpu.smem)
    sv0 = tle.gpu.alloc([2, 128, 64], tl.float8e4nv, scope=tle.gpu.smem)
    sv1 = tle.gpu.alloc([2, 256, 64], tl.float8e4nv, scope=tle.gpu.smem)
    sv_tail = tle.gpu.alloc([2, 128, 64], tl.float8e4nv, scope=tle.gpu.smem)
    sp = tle.gpu.alloc([2, 64, 64], tl.float8e4nv, scope=tle.gpu.smem)
    alpha = tle.gpu.alloc(
        [2, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    # Metadata survives early K reuse, including compiler-rematerialized loads.
    scales = tle.gpu.alloc(
        [2, 128], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    if HAS_ROPE or TOPK > 512:
        mask = tle.gpu.alloc(
            [2, 128], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
    else:
        # Preserve private helper arguments without unused mask storage.
        mask = scales
    CACHE_INDICES: tl.constexpr = not HAS_ROPE and SPLITS == 1 and TOPK <= 1024
    if CACHE_INDICES:
        CACHE_SLOTS: tl.constexpr = triton.next_power_of_2(max(TOPK, 128)) // 128
        cached_indices = tle.gpu.alloc(
            [CACHE_SLOTS, 128],
            tl.int32,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
    else:
        cached_indices = mask
    factor = tle.gpu.alloc(
        [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    # TLE maps these virtual IDs to physical IDs outside the WS reserved set.
    qfull = sparse_named_barriers(1, PRODUCER_WARPS * 32 + 128, 16)
    kfull = sparse_named_barriers(2, PRODUCER_WARPS * 32 + 128, 17)
    kfree = sparse_named_barriers(2, PRODUCER_WARPS * 32 + 256, 19)
    vfull = sparse_named_barriers(2, 256, 24)
    vfree = sparse_named_barriers(1, 256, 23)
    pfull = sparse_named_barriers(2, 256, 21)
    ofull = tle.gpu.alloc_barriers(1, arrive_count=1)
    tle.gpu.warp_specialize(
        [
            (
                sparse_fp8_consumer0,
                (
                    QS,
                    Length,
                    Sink,
                    Output,
                    LSE,
                    qsb,
                    qsh,
                    sq,
                    sr,
                    sk,
                    skr,
                    sv0,
                    sp,
                    alpha,
                    scales,
                    mask,
                    factor,
                    qfull,
                    kfull,
                    kfree,
                    vfull,
                    vfree,
                    pfull,
                    ofull,
                    H,
                    TOPK,
                    SCALE,
                    HAS_LENGTH,
                    SPLITS,
                    HAS_SINK,
                    RepairFlags,
                    HAS_ROPE,
                    PREQUANTIZE_P,
                ),
            ),
            (
                sparse_fp8_producer,
                (
                    Q,
                    QR,
                    KV,
                    KR,
                    KS,
                    Indices,
                    Length,
                    qb,
                    qh,
                    qrb,
                    qrh,
                    kp,
                    kt,
                    krp,
                    krt,
                    ksp,
                    kst,
                    ib,
                    ik,
                    sq,
                    sr,
                    sk,
                    skr,
                    scales,
                    mask,
                    cached_indices,
                    qfull,
                    kfull,
                    kfree,
                    H,
                    N,
                    TOPK,
                    HAS_LENGTH,
                    SPLITS,
                    HAS_ROPE,
                    PRODUCER_WARPS,
                    CAN_ASYNC_NOPE,
                    CAN_ASYNC_ROPE,
                ),
            ),
            (
                sparse_fp8_consumer1,
                (
                    Length,
                    Output,
                    sk,
                    sv0,
                    sv1,
                    sv_tail,
                    sp,
                    alpha,
                    factor,
                    kfull,
                    kfree,
                    vfull,
                    vfree,
                    pfull,
                    ofull,
                    H,
                    TOPK,
                    HAS_LENGTH,
                    SPLITS,
                ),
            ),
        ],
        [PRODUCER_WARPS, 4],
        [
            COPY_REGS,
            (240),
        ],
    )
