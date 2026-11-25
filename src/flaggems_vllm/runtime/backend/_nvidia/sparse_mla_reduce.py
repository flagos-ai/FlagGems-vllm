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
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime.backend._nvidia.sparse_mla_tiles import (
    QK_RECOMPUTE_THRESHOLD,
    sink_normalization,
    sparse_fp8_accumulate,
)
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@triton.jit
def sparse_fp8_empty(Output, LSE, ROWS: tl.constexpr):
    # Fixed write-only geometry; no attention tiling or reduction is performed.
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    tl.store(Output + offsets, 0.0, offsets < ROWS * 512)
    tl.store(LSE + offsets, float("inf"), offsets < ROWS)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_fp8_repair"),
    key=["H", "TOPK", "SPLITS", "HAS_ROPE", "REPAIR_GROUP"],
)
@triton.jit
def sparse_fp8_repair(
    Q,
    QRope,
    KV,
    KVRope,
    Indices,
    QScale,
    KVScale,
    Sink,
    Length,
    Output,
    LSE,
    Partial,
    Stats,
    stride_qb,
    stride_qh,
    stride_qrb,
    stride_qrh,
    stride_kvp,
    stride_kvt,
    stride_krp,
    stride_krt,
    stride_ib,
    stride_ik,
    stride_qsb,
    stride_qsh,
    stride_ksp,
    stride_kst,
    B,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    SM_SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    RepairFlags,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    REPAIR_SPLITS: tl.constexpr = 1,
    HAS_ROPE: tl.constexpr = True,
    REPAIR_GROUP: tl.constexpr = 256,
):
    # Coarsen single-split flag scan to keep empty repair <= 1 wave.
    GROUP: tl.constexpr = REPAIR_GROUP if SPLITS == 1 else 1
    base_batch = tl.program_id(0) * GROUP
    head_group = tl.program_id(1)
    split = tl.program_id(2)
    batches = base_batch + tl.arange(0, GROUP)
    if SPLITS == 1 and REPAIR_SPLITS == 1:
        group_flags = tl.load(
            RepairFlags + batches * (H // 64) + head_group,
            batches < B,
            0,
        )
        if tl.max((group_flags != 0).to(tl.int32), 0) == 0:
            return
        else:
            pass
    else:
        flag_splits = tl.arange(0, REPAIR_SPLITS)
        group_flags = tl.load(
            RepairFlags
            + (batches[:, None] * (H // 64) + head_group) * REPAIR_SPLITS
            + flag_splits[None, :],
            batches[:, None] < B,
            0,
        )
        active_rows = (tl.max(group_flags, 1) != 0) & (batches < B)
        if tl.max(active_rows.to(tl.int32), 0) == 0:
            return
        else:
            pass
    for offset in range(GROUP):
        batch = base_batch + offset
        if GROUP == 1:
            # The initial check already established this sole task is active.
            active = True
        elif SPLITS == 1 and REPAIR_SPLITS == 1:
            flag = tl.load(
                RepairFlags + batch * (H // 64) + head_group,
                batch < B,
                0,
            )
            active = (batch < B) & (flag != 0)
        else:
            flag_splits = tl.arange(0, REPAIR_SPLITS)
            # Do not retain the group flag vector across attention computation.
            flags = tl.load(
                RepairFlags
                + (batch * (H // 64) + head_group) * REPAIR_SPLITS
                + flag_splits,
                batch < B,
                0,
            )
            active = (batch < B) & (tl.max(flags, 0) != 0)
        if active:
            heads = head_group * BLOCK_H + tl.arange(0, BLOCK_H)
            dims = tl.arange(0, 256)
            keys = tl.arange(0, BLOCK_K)
            q_ptr = Q + batch * stride_qb + heads[:, None] * stride_qh + dims[None, :]
            query0 = tl.load(q_ptr, heads[:, None] < H, 0.0)
            query1 = tl.load(q_ptr + 256, heads[:, None] < H, 0.0)
            rope_dims = tl.arange(0, 64)
            if HAS_ROPE:
                query_rope = tl.load(
                    QRope
                    + batch * stride_qrb
                    + heads[:, None] * stride_qrh
                    + rope_dims[None, :],
                    heads[:, None] < H,
                    0,
                )
            else:
                pass
            query_scale = tl.load(
                QScale + batch * stride_qsb + heads * stride_qsh, heads < H, 0
            )
            amplification = tl.max(
                tl.abs(query_scale * (SM_SCALE * 1.4426950408889634)), 0
            )
            maximum = tl.full((BLOCK_H,), -float("inf"), tl.float32)
            denominator = tl.full((BLOCK_H,), 0, tl.float32)
            value0 = tl.full((BLOCK_H, 256), 0, tl.float32)
            value1 = tl.full((BLOCK_H, 256), 0, tl.float32)
            length = (
                tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
                if HAS_LENGTH
                else TOPK
            )
            blocks_per_split: tl.constexpr = triton.cdiv(TOPK, BLOCK_K * SPLITS)
            start = split * blocks_per_split * BLOCK_K
            stop = tl.minimum(start + blocks_per_split * BLOCK_K, length)
            for base in range(start, stop, BLOCK_K):
                positions = base + keys
                ids = tl.load(
                    Indices + batch * stride_ib + positions * stride_ik,
                    positions < length,
                    -1,
                )
                valid = (positions < length) & (ids >= 0) & (ids < N)
                ids = tl.where(valid, ids, 0).to(tl.int64)
                pages = ids // 64
                tokens = ids % 64
                kv_ptr = (
                    KV
                    + pages[None, :] * stride_kvp
                    + tokens[None, :] * stride_kvt
                    + dims[:, None]
                )
                key0 = tl.load(kv_ptr, valid[None, :], 0.0)
                key1 = tl.load(kv_ptr + 256, valid[None, :], 0.0)
                kv_scale = tl.load(
                    KVScale + pages * stride_ksp + tokens * stride_kst, valid, 0
                )
                if amplification * tl.max(kv_scale, 0) > QK_RECOMPUTE_THRESHOLD:
                    logits = tl.zeros((BLOCK_H, BLOCK_K), tl.float32)
                    pad = tl.arange(0, 32)
                    for feature in range(512):
                        query_column = tl.load(
                            Q + batch * stride_qb + heads * stride_qh + feature,
                            heads < H,
                            0.0,
                        )
                        key_column = tl.load(
                            KV + pages * stride_kvp + tokens * stride_kvt + feature,
                            valid,
                            0.0,
                        )
                        query_feature = tl.where(
                            pad[None, :] == 0, query_column[:, None], 0.0
                        ).to(tl.float8e4nv)
                        key_feature = tl.where(
                            pad[:, None] == 0, key_column[None, :], 0.0
                        ).to(tl.float8e4nv)
                        contribution = tl.dot(
                            query_feature, key_feature, out_dtype=tl.float32
                        )
                        logits = sparse_fp8_accumulate(logits, contribution)
                else:
                    logits = tl.dot(query0, key0, out_dtype=tl.float32)
                    logits = tl.dot(query1, key1, logits)
                if HAS_ROPE:
                    key_rope = tl.load(
                        KVRope
                        + pages[None, :] * stride_krp
                        + tokens[None, :] * stride_krt
                        + rope_dims[:, None],
                        valid[None, :],
                        0,
                    )
                    rope_logits = tl.dot(query_rope, key_rope, out_dtype=tl.float32)
                    logits += rope_logits
                else:
                    pass
                logits = logits * query_scale[:, None] * kv_scale[None, :] * SM_SCALE
                logits = tl.where(valid[None, :], logits, -float("inf"))
                next_maximum = tl.maximum(maximum, tl.max(logits, 1))
                safe_maximum = tl.where(
                    next_maximum == -float("inf"), 0.0, next_maximum
                )
                correction = tl.exp(maximum - safe_maximum)
                probabilities = tl.exp(logits - safe_maximum[:, None])
                denominator = denominator * correction + tl.sum(probabilities, 1)
                # Fold token-dependent V scales into P so PV still uses FP8 operands.
                weighted = probabilities * kv_scale[None, :]
                probability_scale = tl.max(weighted, 1) / 448.0
                probability_scale = tl.where(
                    probability_scale > 0, probability_scale, 1.0
                )
                probability_fp8 = (weighted / probability_scale[:, None]).to(
                    tl.float8e4nv
                )
                contribution0 = tl.dot(
                    probability_fp8, tl.trans(key0), out_dtype=tl.float32
                )
                contribution1 = tl.dot(
                    probability_fp8, tl.trans(key1), out_dtype=tl.float32
                )
                value0 = (
                    value0 * correction[:, None]
                    + contribution0 * probability_scale[:, None]
                )
                value1 = (
                    value1 * correction[:, None]
                    + contribution1 * probability_scale[:, None]
                )
                maximum = next_maximum
            if SPLITS == 1:
                logsum = tl.where(
                    denominator > 0, maximum + tl.log(denominator), float("inf")
                )
                denominator_inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
                if HAS_SINK:
                    sink = tl.load(Sink + heads, heads < H, 0)
                    inverse = sink_normalization(denominator_inverse, sink, logsum)
                else:
                    inverse = denominator_inverse
                output_ptr = Output + (batch * H + heads[:, None]) * 512 + dims[None, :]
                tl.store(output_ptr, value0 * inverse[:, None], heads[:, None] < H)
                tl.store(
                    output_ptr + 256, value1 * inverse[:, None], heads[:, None] < H
                )
                tl.store(LSE + batch * H + heads, logsum, heads < H)
            else:
                partial_ptr = (
                    Partial
                    + ((batch * SPLITS + split) * H + heads[:, None]) * 512
                    + dims[None, :]
                )
                tl.store(partial_ptr, value0, heads[:, None] < H)
                tl.store(partial_ptr + 256, value1, heads[:, None] < H)
                stats_ptr = Stats + ((batch * SPLITS + split) * H + heads) * 2
                tl.store(stats_ptr, maximum, heads < H)
                tl.store(stats_ptr + 1, denominator, heads < H)
        else:
            pass


@triton.jit
def sparse_fp8_fast_merge(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Length,
    Sink,
    Flags,
    Partial,
    Stats,
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
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    CHECK_REPAIR: tl.constexpr,
    HEAD_TILE: tl.constexpr = 4,
    HAS_ROPE: tl.constexpr = True,
):
    batch = tl.program_id(0)
    head_base = tl.program_id(1) * HEAD_TILE
    dims = tl.arange(0, 512)
    if CHECK_REPAIR:
        head_group = head_base // 64
        flag_base = Flags + (batch * (H // 64) + head_group) * SPLITS
        flag_acc = tl.full((), 0, tl.int32)
        for s in tl.static_range(SPLITS):
            flag_acc = flag_acc | tl.load(flag_base + s)
        need_repair = flag_acc != 0
    else:
        need_repair = False
    if need_repair:
        ropes = tl.arange(0, 64)
        length = (
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
        )
        for h_off in range(HEAD_TILE):
            head = head_base + h_off
            row = batch * H + head
            query = tl.load(Q + batch * qb + head * qh + dims).to(tl.float32)
            if HAS_ROPE:
                query_rope = tl.load(QR + batch * qrb + head * qrh + ropes).to(
                    tl.float32
                )
            else:
                pass
            query_scale = tl.load(QS + batch * qsb + head * qsh)
            maximum = -float("inf")
            denominator = 0.0
            values = tl.zeros((512,), tl.float32)
            for position in range(length):
                token = tl.load(Indices + batch * ib + position * ik)
                if (token >= 0) & (token < N):
                    physical_token = token.to(tl.uint64)
                    page, slot = physical_token // 64, physical_token % 64
                    key = tl.load(KV + page * kp + slot * kt + dims).to(tl.float32)
                    if HAS_ROPE:
                        key_rope = tl.load(KR + page * krp + slot * krt + ropes).to(
                            tl.float32
                        )
                    else:
                        pass
                    key_scale = tl.load(KS + page * ksp + slot * kst)
                    if HAS_ROPE:
                        rope_score = tl.sum(query_rope * key_rope, 0)
                    else:
                        rope_score = 0.0
                    score = (
                        (tl.sum(query * key, 0) + rope_score)
                        * query_scale
                        * key_scale
                        * (SCALE * 1.4426950408889634)
                    )
                    new_maximum = tl.maximum(maximum, score)
                    old_delta = tl.inline_asm_elementwise(
                        "sub.rn.f32 $0, $1, $2;",
                        "=f,f,f",
                        [maximum, new_maximum],
                        dtype=tl.float32,
                        is_pure=False,
                        pack=1,
                    )
                    score_delta = tl.inline_asm_elementwise(
                        "sub.rn.f32 $0, $1, $2;",
                        "=f,f,f",
                        [score, new_maximum],
                        dtype=tl.float32,
                        is_pure=False,
                        pack=1,
                    )
                    correction = tl.exp2(old_delta)
                    probability = tl.exp2(score_delta)
                    denominator = denominator * correction + probability
                    values = values * correction + probability * (key * key_scale)
                    maximum = new_maximum
                else:
                    pass
            logsum = tl.where(
                denominator > 0,
                (maximum + tl.log2(denominator)) * 0.6931471805599453,
                float("inf"),
            )
            inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
            if HAS_SINK:
                sink = tl.load(Sink + head)
                inverse *= tl.where(
                    sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
                )
            else:
                pass
            tl.store(Output + row * 512 + dims, (values * inverse).to(tl.bfloat16))
            tl.store(LSE + row, logsum)
    else:
        if SPLITS == 2 and HEAD_TILE % 2 == 0:
            base_s0 = (batch * 2) * H
            base_s1 = (batch * 2 + 1) * H
            for pair in tl.static_range(HEAD_TILE // 2):
                h0 = head_base + pair * 2
                h1 = h0 + 1
                st0_a = Stats + (base_s0 + h0) * 2
                st1_a = Stats + (base_s1 + h0) * 2
                st0_b = Stats + (base_s0 + h1) * 2
                st1_b = Stats + (base_s1 + h1) * 2
                m0_a = tl.load(st0_a)
                d0_a = tl.load(st0_a + 1)
                m1_a = tl.load(st1_a)
                d1_a = tl.load(st1_a + 1)
                m0_b = tl.load(st0_b)
                d0_b = tl.load(st0_b + 1)
                m1_b = tl.load(st1_b)
                d1_b = tl.load(st1_b + 1)

                p0_a = tl.load(
                    Partial + (base_s0 + h0) * 512 + dims,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                p1_a = tl.load(
                    Partial + (base_s1 + h0) * 512 + dims,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                p0_b = tl.load(
                    Partial + (base_s0 + h1) * 512 + dims,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                p1_b = tl.load(
                    Partial + (base_s1 + h1) * 512 + dims,
                    eviction_policy="evict_first",
                ).to(tl.float32)

                gm_a = tl.maximum(m0_a, m1_a)
                sm_a = tl.where(gm_a == -float("inf"), 0.0, gm_a)
                w0_a = tl.exp(m0_a - sm_a)
                w1_a = tl.exp(m1_a - sm_a)
                tot_a = d0_a * w0_a + d1_a * w1_a
                lse_a = tl.where(tot_a > 0.0, gm_a + tl.log(tot_a), float("inf"))
                denominator_inverse = tl.where(tot_a > 0.0, 1.0 / tot_a, 0.0)
                if HAS_SINK:
                    sk_a = tl.load(Sink + h0)
                    inv_a = sink_normalization(denominator_inverse, sk_a, lse_a)
                else:
                    inv_a = denominator_inverse
                s0_a = w0_a * inv_a
                s1_a = w1_a * inv_a

                gm_b = tl.maximum(m0_b, m1_b)
                sm_b = tl.where(gm_b == -float("inf"), 0.0, gm_b)
                w0_b = tl.exp(m0_b - sm_b)
                w1_b = tl.exp(m1_b - sm_b)
                tot_b = d0_b * w0_b + d1_b * w1_b
                lse_b = tl.where(tot_b > 0.0, gm_b + tl.log(tot_b), float("inf"))
                denominator_inverse = tl.where(tot_b > 0.0, 1.0 / tot_b, 0.0)
                if HAS_SINK:
                    sk_b = tl.load(Sink + h1)
                    inv_b = sink_normalization(denominator_inverse, sk_b, lse_b)
                else:
                    inv_b = denominator_inverse
                s0_b = w0_b * inv_b
                s1_b = w1_b * inv_b

                out_a = (p0_a * s0_a + p1_a * s1_a).to(tl.bfloat16)
                out_b = (p0_b * s0_b + p1_b * s1_b).to(tl.bfloat16)
                row_a = batch * H + h0
                row_b = batch * H + h1
                tl.store(
                    Output + row_a * 512 + dims,
                    out_a,
                    eviction_policy="evict_first",
                )
                tl.store(
                    Output + row_b * 512 + dims,
                    out_b,
                    eviction_policy="evict_first",
                )
                tl.store(LSE + row_a, lse_a)
                tl.store(LSE + row_b, lse_b)
        else:
            for h_off in tl.static_range(HEAD_TILE):
                head = head_base + h_off
                row = batch * H + head
                gmax = tl.full((), -float("inf"), tl.float32)
                for s in tl.static_range(SPLITS):
                    m_s = tl.load(Stats + ((batch * SPLITS + s) * H + head) * 2)
                    gmax = tl.maximum(gmax, m_s)
                safe_max = tl.where(gmax == -float("inf"), 0.0, gmax)
                total = tl.full((), 0.0, tl.float32)
                for s in tl.static_range(SPLITS):
                    st_ptr = Stats + ((batch * SPLITS + s) * H + head) * 2
                    m_s = tl.load(st_ptr)
                    d_s = tl.load(st_ptr + 1)
                    total += d_s * tl.exp(m_s - safe_max)
                logsum = tl.where(total > 0.0, gmax + tl.log(total), float("inf"))
                denominator_inverse = tl.where(total > 0.0, 1.0 / total, 0.0)
                if HAS_SINK:
                    sink = tl.load(Sink + head)
                    inv = sink_normalization(denominator_inverse, sink, logsum)
                else:
                    inv = denominator_inverse
                acc = tl.zeros((512,), tl.float32)
                for s in tl.static_range(SPLITS):
                    m_s = tl.load(Stats + ((batch * SPLITS + s) * H + head) * 2)
                    w_s = tl.exp(m_s - safe_max) * inv
                    p_s = tl.load(
                        Partial + ((batch * SPLITS + s) * H + head) * 512 + dims,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    acc += p_s * w_s
                tl.store(
                    Output + row * 512 + dims,
                    acc.to(tl.bfloat16),
                    eviction_policy="evict_first",
                )
                tl.store(LSE + row, logsum)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_fp8_short"),
    key=["B", "H", "N", "TOPK", "HAS_LENGTH", "HAS_SINK"],
    use_cuda_graph=True,
)
@triton.jit
def sparse_fp8_short(
    Q,
    KV,
    QS,
    KS,
    Indices,
    Length,
    Sink,
    Output,
    LSE,
    qb: tl.constexpr,
    qh: tl.constexpr,
    kp: tl.constexpr,
    kt: tl.constexpr,
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
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    VALUE_DIM: tl.constexpr,
):
    batch = tl.program_id(0)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    columns = tl.program_id(2) * VALUE_DIM + tl.arange(0, VALUE_DIM)
    features = tl.arange(0, 512)
    positions = tl.arange(0, BLOCK_K)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    ids = tl.load(Indices + batch * ib + positions * ik, positions < length, -1)
    valid = (positions < length) & (ids >= 0) & (ids < N)
    ids = tl.where(valid, ids, 0).to(tl.int64)
    pages, tokens = ids // 64, ids % 64
    query = tl.load(Q + batch * qb + heads[:, None] * qh + features[None, :])
    cache = tl.load(
        KV + pages[:, None] * kp + tokens[:, None] * kt + features[None, :],
        valid[:, None],
        0.0,
    )
    query_scale = tl.load(QS + batch * qsb + heads * qsh)
    cache_scale = tl.load(KS + pages * ksp + tokens * kst, valid, 0.0)
    # Merge FP8 QK partial accumulators in FP32 at each MMA K step.
    scores = tl.dot(
        query, tl.trans(cache), out_dtype=tl.float32, max_num_imprecise_acc=0
    )
    scores *= query_scale[:, None] * cache_scale[None, :] * SCALE
    scores = tl.where(valid[None, :], scores, -float("inf"))
    maximum = tl.max(scores, 1)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    probabilities = tl.exp(scores - safe_maximum[:, None])
    denominator = tl.sum(probabilities, 1)
    weighted = probabilities * cache_scale[None, :]
    probability_scale = tl.max(weighted, 1) / 448.0
    probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
    probability_fp8 = (weighted / probability_scale[:, None]).to(tl.float8e4nv)
    values = tl.load(
        KV + pages[:, None] * kp + tokens[:, None] * kt + columns[None, :],
        valid[:, None],
        0.0,
    )
    value = tl.dot(probability_fp8, values, out_dtype=tl.float32)
    logsum = tl.where(denominator > 0, maximum + tl.log(denominator), float("inf"))
    denominator_inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
    if HAS_SINK:
        sink = tl.load(Sink + heads)
        inverse = sink_normalization(denominator_inverse, sink, logsum)
    else:
        inverse = denominator_inverse
    value *= (probability_scale * inverse)[:, None]
    tl.store(Output + (batch * H + heads[:, None]) * 512 + columns[None, :], value)
    tl.store(LSE + batch * H + heads, logsum, tl.program_id(2) == 0)
