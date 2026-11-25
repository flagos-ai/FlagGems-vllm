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
from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import (
    publish_p_fp8_sw64_coupled_stmatrix,
    vtranspose_fp8_64x128_kperm,
)
from flaggems_vllm.runtime.backend._nvidia.sparse_mla_tiles import (
    ACCUMULATOR_SCALE_FLOOR,
    QK_RECOMPUTE_THRESHOLD,
    sparse_named_barriers,
    sparse_smem_subslice,
)
from flaggems_vllm.utils import libentry, libtuner


@triton.jit
def sparse_fp8_precise_scores(
    Q,
    KV,
    Indices,
    batch,
    heads,
    base,
    length,
    seed,
    qb: tl.constexpr,
    qh: tl.constexpr,
    kp: tl.constexpr,
    kt: tl.constexpr,
    ib: tl.constexpr,
    ik: tl.constexpr,
    N: tl.constexpr,
):
    positions = base + tl.arange(0, 64)
    ids = tl.load(Indices + batch * ib + positions * ik, positions < length, -1)
    valid = (positions < length) & (ids >= 0) & (ids < N)
    ids = tl.where(valid, ids, 0).to(tl.uint64)
    # Preserve the incoming MMA layout instead of allocating a shared conversion tile.
    scores = tl.inline_asm_elementwise(
        "mov.b32 $0, 0;",
        "=f,f",
        [seed],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )
    for feature in range(512):
        query = tl.load(Q + batch * qb + heads * qh + feature).to(tl.float32)
        key = tl.load(KV + (ids // 64) * kp + (ids % 64) * kt + feature, valid, 0.0).to(
            tl.float32
        )
        product = query[:, None] * key[None, :]
        scores = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [scores, product],
            dtype=tl.float32,
            is_pure=False,
            pack=1,
        )
    return scores


@triton.jit
def sparse_fp8_load_tile(
    KV,
    KR,
    KS,
    Indices,
    sk,
    skr,
    scales,
    masks,
    batch,
    base,
    length,
    slot: tl.constexpr,
    N: tl.constexpr,
    kp: tl.constexpr,
    kt: tl.constexpr,
    krp: tl.constexpr,
    krt: tl.constexpr,
    ksp: tl.constexpr,
    kst: tl.constexpr,
    ib: tl.constexpr,
    ik: tl.constexpr,
    CAN_ASYNC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_ROPE: tl.constexpr = True,
):
    keys = tl.arange(0, BLOCK_K)
    features = tl.arange(0, 512)
    ropes = tl.arange(0, 64)
    positions = base + keys
    ids = tl.load(Indices + batch * ib + positions * ik, positions < length, -1)
    valid = (positions < length) & (ids >= 0) & (ids < N)
    ids = tl.where(valid, ids, 0).to(tl.uint64)
    pages, slots = ids // 64, ids % 64
    cache = tl.load(
        KV + pages[:, None] * kp + slots[:, None] * kt + features[None, :],
        valid[:, None],
        0.0,
        volatile=not CAN_ASYNC,
    )
    if HAS_ROPE:
        rope = tl.load(
            KR + pages[:, None] * krp + slots[:, None] * krt + ropes[None, :],
            valid[:, None],
            0.0,
            volatile=not CAN_ASYNC,
        )
    scale = tl.load(KS + pages * ksp + slots * kst, valid, 0.0)
    tl.store(tle.gpu.local_ptr(sk.slot(slot)), cache)
    if HAS_ROPE:
        tl.store(tle.gpu.local_ptr(skr.slot(slot)), rope)
    tl.store(tle.gpu.local_ptr(scales.slot(slot)), scale)
    tl.store(tle.gpu.local_ptr(masks.slot(slot)), valid.to(tl.int32))


@triton.jit
def sparse_fp8_compact_leader(
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
    Stats,
    RepairFlags,
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
    sq,
    sr,
    sk,
    skr,
    scales,
    masks,
    sp,
    sv0,
    sv1,
    alpha,
    factor,
    full,
    empty,
    done,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    CAN_ASYNC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_ROPE: tl.constexpr = True,
):
    batch = tl.program_id(0)
    head_group = tl.program_id(1)
    heads = head_group * 64 + tl.arange(0, 64)
    split = tl.program_id(2)
    features = tl.arange(0, 512)
    ropes = tl.arange(0, 64)
    # Wider scores need more registers; move another 128 PV columns to the follower.
    LEADER_D: tl.constexpr = 128 if BLOCK_K == 128 else 256
    columns = tl.arange(0, LEADER_D)
    query = tl.load(
        Q + batch * qb + heads[:, None] * qh + features[None, :],
        volatile=not CAN_ASYNC,
    )
    if HAS_ROPE:
        query_rope = tl.load(
            QR + batch * qrb + heads[:, None] * qrh + ropes[None, :],
            volatile=not CAN_ASYNC,
        )
    tl.store(tle.gpu.local_ptr(sq), query)
    if HAS_ROPE:
        tl.store(tle.gpu.local_ptr(sr), query_rope)
    query_scale = tl.load(QS + batch * qsb + heads * qsh) * (SCALE * 1.4426950408889634)
    amplification = tl.max(tl.abs(query_scale), 0)
    needs_repair = tl.full((), False, tl.int1)
    maximum_weight_scale = tl.zeros((64,), tl.float32)
    maximum = tl.full((64,), -float("inf"), tl.float32)
    denominator = tl.zeros((64,), tl.float32)
    value = tl.zeros((64, LEADER_D), tl.float32)
    previous_scale = tl.full((64,), 1.0, tl.float32)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    blocks_per_split: tl.constexpr = triton.cdiv(TOPK, BLOCK_K * SPLITS)
    start = split * blocks_per_split * BLOCK_K
    stop = tl.minimum(start + blocks_per_split * BLOCK_K, length)
    block_count = tl.cdiv(tl.maximum(stop - start, 0), BLOCK_K)
    if BLOCK_K == 64 and block_count > 0:
        sparse_fp8_load_tile(
            KV,
            KR,
            KS,
            Indices,
            sk,
            skr,
            scales,
            masks,
            batch,
            start,
            length,
            0,
            N,
            kp,
            kt,
            krp,
            krt,
            ksp,
            kst,
            ib,
            ik,
            CAN_ASYNC=CAN_ASYNC,
            BLOCK_K=BLOCK_K,
            HAS_ROPE=HAS_ROPE,
        )
    else:
        pass
    SLOTS: tl.constexpr = 2 if BLOCK_K == 64 else 1
    for pair in range(tl.cdiv(block_count, SLOTS)):
        for slot in tl.static_range(SLOTS):
            step = pair * SLOTS + slot
            if step < block_count:
                if BLOCK_K == 128:
                    sparse_fp8_load_tile(
                        KV,
                        KR,
                        KS,
                        Indices,
                        sk,
                        skr,
                        scales,
                        masks,
                        batch,
                        start + step * BLOCK_K,
                        length,
                        slot,
                        N,
                        kp,
                        kt,
                        krp,
                        krt,
                        ksp,
                        kst,
                        ib,
                        ik,
                        CAN_ASYNC=CAN_ASYNC,
                        BLOCK_K=BLOCK_K,
                        HAS_ROPE=HAS_ROPE,
                    )
                logits = tle.gpu.wgmma(
                    sq, sk.slot(slot), out_dtype=tl.float32, trans_b=True
                )
                if HAS_ROPE:
                    logits = tle.gpu.wgmma(sr, skr.slot(slot), logits, trans_b=True)
                if BLOCK_K == 64 and step + 1 < block_count:
                    sparse_fp8_load_tile(
                        KV,
                        KR,
                        KS,
                        Indices,
                        sk,
                        skr,
                        scales,
                        masks,
                        batch,
                        start + (step + 1) * BLOCK_K,
                        length,
                        1 - slot,
                        N,
                        kp,
                        kt,
                        krp,
                        krt,
                        ksp,
                        kst,
                        ib,
                        ik,
                        CAN_ASYNC=CAN_ASYNC,
                        BLOCK_K=BLOCK_K,
                        HAS_ROPE=HAS_ROPE,
                    )
                else:
                    pass
                logits = tle.gpu.wgmma_wait(0, logits)
                kv_scale = tl.load(tle.gpu.local_ptr(scales.slot(slot)))
                valid = tl.load(tle.gpu.local_ptr(masks.slot(slot))) != 0
                needs_repair |= (
                    amplification * tl.max(kv_scale, 0) > QK_RECOMPUTE_THRESHOLD
                )
                logits = logits * query_scale[:, None] * kv_scale[None, :]
                logits = tl.where(valid[None, :], logits, -float("inf"))
                next_maximum = tl.maximum(maximum, tl.max(logits, 1))
                safe_maximum = tl.where(
                    next_maximum == -float("inf"), 0.0, next_maximum
                )
                correction = tl.exp2(maximum - safe_maximum)
                probabilities = tl.exp2(logits - safe_maximum[:, None])
                denominator = denominator * correction + tl.sum(probabilities, 1)
                weighted = probabilities * kv_scale[None, :]
                probability_scale = tl.max(weighted, 1) / 448.0
                maximum_weight_scale = tl.maximum(
                    maximum_weight_scale * correction, probability_scale
                )
                probability_scale = tl.where(
                    probability_scale > 0, probability_scale, 1.0
                )
                needs_repair |= (
                    tl.max(
                        (
                            maximum_weight_scale * ACCUMULATOR_SCALE_FLOOR
                            > probability_scale
                        ).to(tl.int32),
                        0,
                    )
                    != 0
                )
                probability_values = weighted / probability_scale[:, None]
                if BLOCK_K == 128:
                    tl.store(
                        tle.gpu.local_ptr(sp), probability_values.to(tl.float8e4nv)
                    )
                    vp = tle.gpu.local_ptr(sk.slot(slot))
                    left, right = vp.reshape(BLOCK_K, 2, 256).permute(0, 2, 1).split()
                    tl.store(tle.gpu.local_ptr(sv0), tl.trans(tl.load(left)))
                    tl.store(tle.gpu.local_ptr(sv1), tl.trans(tl.load(right)))
                else:
                    publish_p_fp8_sw64_coupled_stmatrix(sp, probability_values)
                    for tile in tl.static_range(4):
                        source = sparse_smem_subslice(
                            sk.slot(slot), [0, tile * 128], [64, 128]
                        )
                        if tile < 2:
                            vtranspose_fp8_64x128_kperm(source, sv0, tile * 128)
                        else:
                            vtranspose_fp8_64x128_kperm(source, sv1, (tile - 2) * 128)
                tl.inline_asm_elementwise(
                    "{ fence.proxy.async.shared::cta; mov.b32 $0, $1; }",
                    "=r,r",
                    [tl.arange(0, 128)],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
                tl.debug_barrier()
                correction = (correction * previous_scale) / probability_scale
                tl.store(tle.gpu.local_ptr(alpha), correction)
                tle.gpu.barrier_arrive(full[0])
                value *= correction[:, None]
                value = tle.gpu.wgmma(
                    sp,
                    sparse_smem_subslice(sv0, [0, 0], [LEADER_D, BLOCK_K]),
                    value,
                    trans_b=True,
                )
                value = tle.gpu.wgmma_wait(0, value)
                previous_scale = probability_scale
                maximum = next_maximum
                tle.gpu.barrier_wait(empty[0])
            else:
                pass
    output_batch = batch * SPLITS + split
    if SPLITS == 1:
        logsum = tl.where(
            denominator > 0,
            maximum * 0.6931471805599453 + tl.log(denominator),
            float("inf"),
        )
        inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
        if HAS_SINK:
            sink = tl.load(Sink + heads)
            inverse *= tl.where(
                sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
            )
        else:
            pass
        output_scale = inverse * previous_scale
    else:
        output_scale = previous_scale
    tl.store(tle.gpu.local_ptr(factor), output_scale)
    tle.gpu.barrier_arrive(done[0], phaseIdx=0)
    value *= output_scale[:, None]
    tl.store(
        Output + (output_batch * H + heads[:, None]) * 512 + columns[None, :], value
    )
    if SPLITS == 1:
        tl.store(Stats + batch * H + heads, logsum)
    else:
        stats = Stats + (output_batch * H + heads) * 2
        tl.store(stats, maximum * 0.6931471805599453)
        tl.store(stats + 1, denominator)
    tl.store(
        RepairFlags + (batch * (H // 64) + head_group) * SPLITS + split,
        needs_repair.to(tl.int32),
    )


@triton.jit
def sparse_fp8_compact_follower(
    Length,
    Output,
    sp,
    sv0,
    sv1,
    alpha,
    factor,
    full,
    empty,
    done,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    split = tl.program_id(2)
    heads = tl.program_id(1) * 64 + tl.arange(0, 64)
    columns = 256 + tl.arange(0, 256)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    blocks_per_split: tl.constexpr = triton.cdiv(TOPK, BLOCK_K * SPLITS)
    start = split * blocks_per_split * BLOCK_K
    stop = tl.minimum(start + blocks_per_split * BLOCK_K, length)
    value = tl.zeros((64, 256), tl.float32)
    if BLOCK_K == 128:
        extra = tl.zeros((64, 128), tl.float32)
    for base in range(start, stop, BLOCK_K):
        tle.gpu.barrier_wait(full[0])
        correction = tl.load(tle.gpu.local_ptr(alpha))
        value *= correction[:, None]
        if BLOCK_K == 128:
            extra *= correction[:, None]
            extra = tle.gpu.wgmma(
                sp,
                sparse_smem_subslice(sv0, [128, 0], [128, BLOCK_K]),
                extra,
                trans_b=True,
            )
        value = tle.gpu.wgmma(sp, sv1, value, trans_b=True)
        value = tle.gpu.wgmma_wait(0, value)
        if BLOCK_K == 128:
            extra = tle.gpu.wgmma_wait(0, extra)
        tle.gpu.barrier_arrive(empty[0])
    tle.gpu.barrier_wait(done[0], phaseIdx=0)
    output_scale = tl.load(tle.gpu.local_ptr(factor))
    tl.store(
        Output
        + ((batch * SPLITS + split) * H + heads[:, None]) * 512
        + columns[None, :],
        value * output_scale[:, None],
    )
    if BLOCK_K == 128:
        tl.store(
            Output
            + ((batch * SPLITS + split) * H + heads[:, None]) * 512
            + 128
            + tl.arange(0, 128)[None, :],
            extra * output_scale[:, None],
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_fp8_compact"),
    key=["B", "H", "TOPK", "SPLITS", "CAN_ASYNC", "HAS_ROPE"],
    use_cuda_graph=True,
)
@triton.jit
def sparse_fp8_compact(
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
    Stats,
    RepairFlags,
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
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    FOLLOWER_REGS: tl.constexpr,
    CAN_ASYNC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_ROPE: tl.constexpr = True,
):
    sq = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
    sr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
    # A double-buffered 128-key tile would exceed Hopper shared-memory capacity.
    SLOTS: tl.constexpr = 2 if BLOCK_K == 64 else 1
    sk = tle.gpu.alloc([SLOTS, BLOCK_K, 512], tl.float8e4nv, scope=tle.gpu.smem)
    skr = tle.gpu.alloc([SLOTS, BLOCK_K, 64], tl.bfloat16, scope=tle.gpu.smem)
    scales = tle.gpu.alloc(
        [SLOTS, BLOCK_K], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    masks = tle.gpu.alloc(
        [SLOTS, BLOCK_K], tl.int32, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    sp = tle.gpu.alloc([64, BLOCK_K], tl.float8e4nv, scope=tle.gpu.smem)
    sv0 = tle.gpu.alloc([256, BLOCK_K], tl.float8e4nv, scope=tle.gpu.smem)
    sv1 = tle.gpu.alloc([256, BLOCK_K], tl.float8e4nv, scope=tle.gpu.smem)
    alpha = tle.gpu.alloc(
        [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    factor = tle.gpu.alloc(
        [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    full = sparse_named_barriers(1, 256, 16)
    empty = sparse_named_barriers(1, 256, 17)
    done = tle.gpu.alloc_barriers(1, arrive_count=1)
    tle.gpu.warp_specialize(
        [
            (
                sparse_fp8_compact_leader,
                (
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
                    Stats,
                    RepairFlags,
                    qb,
                    qh,
                    qrb,
                    qrh,
                    kp,
                    kt,
                    krp,
                    krt,
                    qsb,
                    qsh,
                    ksp,
                    kst,
                    ib,
                    ik,
                    sq,
                    sr,
                    sk,
                    skr,
                    scales,
                    masks,
                    sp,
                    sv0,
                    sv1,
                    alpha,
                    factor,
                    full,
                    empty,
                    done,
                    H,
                    N,
                    TOPK,
                    SCALE,
                    SPLITS,
                    HAS_LENGTH,
                    HAS_SINK,
                    CAN_ASYNC,
                    BLOCK_K,
                    HAS_ROPE,
                ),
            ),
            (
                sparse_fp8_compact_follower,
                (
                    Length,
                    Output,
                    sp,
                    sv0,
                    sv1,
                    alpha,
                    factor,
                    full,
                    empty,
                    done,
                    H,
                    TOPK,
                    SPLITS,
                    HAS_LENGTH,
                    BLOCK_K,
                ),
            ),
        ],
        [4],
        [FOLLOWER_REGS],
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_fp8_tile"),
    key=["B", "H", "TOPK", "SPLITS", "CAN_ASYNC", "HAS_ROPE"],
    use_cuda_graph=True,
)
@triton.jit
def sparse_fp8_tile(
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
    Stats,
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
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAN_ASYNC: tl.constexpr,
    HAS_ROPE: tl.constexpr = True,
):
    tl.static_assert(TOPK <= 64 * SPLITS)
    batch = tl.program_id(0)
    head_group = tl.program_id(1)
    heads = head_group * 64 + tl.arange(0, 64)
    value_group = tl.program_id(2) % (512 // BLOCK_D)
    split = tl.program_id(2) // (512 // BLOCK_D)
    features = tl.arange(0, 512)
    ropes = tl.arange(0, 64)
    keys = tl.arange(0, BLOCK_K)
    columns = value_group * BLOCK_D + tl.arange(0, BLOCK_D)
    sq = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
    sr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
    sk = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
    skr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
    sp = tle.gpu.alloc([64, 64], tl.float8e4nv, scope=tle.gpu.smem)
    sv = tle.gpu.alloc([BLOCK_D, 64], tl.float8e4nv, scope=tle.gpu.smem)
    query = tl.load(
        Q + batch * qb + heads[:, None] * qh + features[None, :],
        volatile=not CAN_ASYNC,
    )
    if HAS_ROPE:
        query_rope = tl.load(
            QR + batch * qrb + heads[:, None] * qrh + ropes[None, :],
            volatile=not CAN_ASYNC,
        )
    tl.store(tle.gpu.local_ptr(sq), query)
    if HAS_ROPE:
        tl.store(tle.gpu.local_ptr(sr), query_rope)
    query_scale = tl.load(QS + batch * qsb + heads * qsh) * (SCALE * 1.4426950408889634)
    amplification = tl.max(tl.abs(query_scale), 0)
    maximum = tl.full((64,), -float("inf"), tl.float32)
    denominator = tl.zeros((64,), tl.float32)
    value = tl.zeros((64, BLOCK_D), tl.float32)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    blocks_per_split: tl.constexpr = triton.cdiv(TOPK, BLOCK_K * SPLITS)
    start = split * blocks_per_split * BLOCK_K
    stop = tl.minimum(start + blocks_per_split * BLOCK_K, length)
    for base in range(start, stop, BLOCK_K):
        positions = base + keys
        ids = tl.load(Indices + batch * ib + positions * ik, positions < length, -1)
        valid = (positions < length) & (ids >= 0) & (ids < N)
        ids = tl.where(valid, ids, 0).to(tl.uint64)
        pages, slots = ids // 64, ids % 64
        cache = tl.load(
            KV + pages[:, None] * kp + slots[:, None] * kt + features[None, :],
            valid[:, None],
            0.0,
            volatile=not CAN_ASYNC,
        )
        if HAS_ROPE:
            cache_rope = tl.load(
                KR + pages[:, None] * krp + slots[:, None] * krt + ropes[None, :],
                valid[:, None],
                0.0,
                volatile=not CAN_ASYNC,
            )
        tl.store(tle.gpu.local_ptr(sk), cache)
        if HAS_ROPE:
            tl.store(tle.gpu.local_ptr(skr), cache_rope)
        tl.debug_barrier()
        logits = tle.gpu.wgmma(sq, sk, out_dtype=tl.float32, trans_b=True)
        if HAS_ROPE:
            logits = tle.gpu.wgmma(sr, skr, logits, trans_b=True)
        logits = tle.gpu.wgmma_wait(0, logits)
        kv_scale = tl.load(KS + pages * ksp + slots * kst, valid, 0.0)
        if amplification * tl.max(kv_scale, 0) > QK_RECOMPUTE_THRESHOLD:
            exact = sparse_fp8_precise_scores(
                Q,
                KV,
                Indices,
                batch,
                heads,
                base,
                length,
                logits,
                qb,
                qh,
                kp,
                kt,
                ib,
                ik,
                N,
            )
            logits = exact
            if HAS_ROPE:
                rope_scores = tle.gpu.wgmma(sr, skr, out_dtype=tl.float32, trans_b=True)
                rope_scores = tle.gpu.wgmma_wait(0, rope_scores)
                logits = exact + rope_scores
        else:
            pass
        logits = logits * query_scale[:, None] * kv_scale[None, :]
        logits = tl.where(valid[None, :], logits, -float("inf"))
        next_maximum = tl.max(logits, 1)
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        probabilities = tl.exp2(logits - safe_maximum[:, None])
        denominator = tl.sum(probabilities, 1)
        weighted = probabilities * kv_scale[None, :]
        probability_scale = tl.max(weighted, 1) / 448.0
        probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
        publish_p_fp8_sw64_coupled_stmatrix(sp, weighted / probability_scale[:, None])
        for candidate_group in tl.static_range(512 // BLOCK_D):
            if value_group == candidate_group:
                for tile in tl.static_range(BLOCK_D // 128):
                    source = sparse_smem_subslice(
                        sk, [0, candidate_group * BLOCK_D + tile * 128], [64, 128]
                    )
                    vtranspose_fp8_64x128_kperm(source, sv, tile * 128)
            else:
                pass
        tl.inline_asm_elementwise(
            "{ fence.proxy.async.shared::cta; mov.b32 $0, $1; }",
            "=r,r",
            [tl.arange(0, 128)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.debug_barrier()
        value = tle.gpu.wgmma(sp, sv, out_dtype=tl.float32, trans_b=True)
        value = tle.gpu.wgmma_wait(0, value)
        value *= probability_scale[:, None]
        maximum = next_maximum
    output_batch = batch * SPLITS + split
    if SPLITS == 1:
        logsum = tl.where(
            denominator > 0,
            maximum * 0.6931471805599453 + tl.log(denominator),
            float("inf"),
        )
        inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
        if HAS_SINK:
            sink = tl.load(Sink + heads)
            inverse *= tl.where(
                sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
            )
        else:
            pass
        value *= inverse[:, None]
    else:
        pass
    tl.store(
        Output + (output_batch * H + heads[:, None]) * 512 + columns[None, :], value
    )
    if value_group == 0:
        if SPLITS == 1:
            tl.store(Stats + batch * H + heads, logsum)
        else:
            stats = Stats + (output_batch * H + heads) * 2
            tl.store(stats, maximum * 0.6931471805599453)
            tl.store(stats + 1, denominator)
    else:
        pass
