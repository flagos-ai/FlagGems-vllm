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


"""Complete Hopper FP8 sparse MLA implementation, including tiles and precision repair."""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.language.core as tlc

from flaggems_vllm import runtime
from flaggems_vllm.ops.flash_mla import HAS_TLE_FLASH_MLA as HAS_TLE
from flaggems_vllm.ops.flash_mla import _get_num_sms, tle
from flaggems_vllm.utils import libentry, libtuner

if HAS_TLE:
    from triton.experimental.tle.language.gpu import types as tle_types
else:
    tle_types = None

QK_RECOMPUTE_THRESHOLD = tl.constexpr(2.0**-14)

ACCUMULATOR_SCALE_FLOOR = tl.constexpr(2.0**-32)


@tlc.builtin
def sparse_smem_subslice(buf, offsets, shape, _semantic=None):
    offsets = [int(tlc._unwrap_if_constexpr(value)) for value in offsets]
    shape = [int(tlc._unwrap_if_constexpr(value)) for value in shape]
    view_type = tle_types.buffered_tensor_type(
        buf.dtype,
        shape,
        buf.type.storage,
        buf.type.layout,
        _semantic,
        alloc_shape=buf.type.alloc_shape,
    )
    handle = _semantic.builder.create_memdesc_subslice(
        view_type.to_ir(_semantic.builder), buf.handle, offsets
    )
    return tle_types.buffered_tensor(
        handle,
        buf.dtype,
        shape,
        buf.type.storage,
        buf.type.layout,
        _semantic,
        alloc_shape=buf.type.alloc_shape,
    )


@tlc.builtin
def sparse_named_barriers(count, threads, base, _semantic=None):
    barriers = tle.gpu.alloc_barriers(count, arrive_count=threads, _semantic=_semantic)
    # Lazy IDs are not unique across JIT helpers; reserve them before capture.
    base = int(tlc._unwrap_if_constexpr(base))
    barriers.named_base_id = base
    barriers.type.named_base_id = base
    return barriers


@libentry()
@triton.jit
def sparse_fp8_empty(Output, LSE, ROWS: tl.constexpr):
    # Fixed write-only geometry; no attention tiling or reduction is performed.
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    tl.store(Output + offsets, 0.0, offsets < ROWS * 512)
    tl.store(LSE + offsets, float("inf"), offsets < ROWS)


@triton.jit
def sparse_fp8_accumulate(logits, contribution):
    # Keep FP32 additions separate from tensor-core accumulation.
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [logits, contribution],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_fp8_repair"),
    key=["B", "H", "TOPK", "SPLITS"],
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
    B: tl.constexpr,
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
):
    batch = tl.program_id(0)
    flags = tl.load(
        RepairFlags
        + (batch * (H // 64) + tl.program_id(1)) * REPAIR_SPLITS
        + tl.arange(0, REPAIR_SPLITS)
    )
    if tl.max(flags, 0) == 0:
        return
    else:
        pass
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    split = tl.program_id(2)
    dims = tl.arange(0, 256)
    keys = tl.arange(0, BLOCK_K)
    q_ptr = Q + batch * stride_qb + heads[:, None] * stride_qh + dims[None, :]
    query0 = tl.load(q_ptr, heads[:, None] < H, 0.0)
    query1 = tl.load(q_ptr + 256, heads[:, None] < H, 0.0)
    rope_dims = tl.arange(0, 64)
    query_rope = tl.load(
        QRope + batch * stride_qrb + heads[:, None] * stride_qrh + rope_dims[None, :],
        heads[:, None] < H,
        0,
    )
    query_scale = tl.load(
        QScale + batch * stride_qsb + heads * stride_qsh, heads < H, 0
    )
    amplification = tl.max(tl.abs(query_scale * (SM_SCALE * 1.4426950408889634)), 0)
    maximum = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    denominator = tl.full((BLOCK_H,), 0, tl.float32)
    value0 = tl.full((BLOCK_H, 256), 0, tl.float32)
    value1 = tl.full((BLOCK_H, 256), 0, tl.float32)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    blocks_per_split: tl.constexpr = triton.cdiv(TOPK, BLOCK_K * SPLITS)
    start = split * blocks_per_split * BLOCK_K
    stop = tl.minimum(start + blocks_per_split * BLOCK_K, length)
    for base in range(start, stop, BLOCK_K):
        positions = base + keys
        ids = tl.load(
            Indices + batch * stride_ib + positions * stride_ik, positions < length, -1
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
        kv_scale = tl.load(KVScale + pages * stride_ksp + tokens * stride_kst, valid, 0)
        if amplification * tl.max(kv_scale, 0) > QK_RECOMPUTE_THRESHOLD:
            logits = tl.zeros((BLOCK_H, BLOCK_K), tl.float32)
            pad = tl.arange(0, 32)
            for feature in range(512):
                query_column = tl.load(
                    Q + batch * stride_qb + heads * stride_qh + feature, heads < H, 0.0
                )
                key_column = tl.load(
                    KV + pages * stride_kvp + tokens * stride_kvt + feature, valid, 0.0
                )
                query_feature = tl.where(
                    pad[None, :] == 0, query_column[:, None], 0.0
                ).to(tl.float8e4nv)
                key_feature = tl.where(pad[:, None] == 0, key_column[None, :], 0.0).to(
                    tl.float8e4nv
                )
                contribution = tl.dot(query_feature, key_feature, out_dtype=tl.float32)
                logits = sparse_fp8_accumulate(logits, contribution)
        else:
            logits = tl.dot(query0, key0, out_dtype=tl.float32)
            logits = tl.dot(query1, key1, logits)
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
        logits = logits * query_scale[:, None] * kv_scale[None, :] * SM_SCALE
        logits = tl.where(valid[None, :], logits, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(logits, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp(maximum - safe_maximum)
        probabilities = tl.exp(logits - safe_maximum[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        # Fold token-dependent V scales into P so PV still uses FP8 operands.
        weighted = probabilities * kv_scale[None, :]
        probability_scale = tl.max(weighted, 1) / 448.0
        probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
        probability_fp8 = (weighted / probability_scale[:, None]).to(tl.float8e4nv)
        contribution0 = tl.dot(probability_fp8, tl.trans(key0), out_dtype=tl.float32)
        contribution1 = tl.dot(probability_fp8, tl.trans(key1), out_dtype=tl.float32)
        value0 = (
            value0 * correction[:, None] + contribution0 * probability_scale[:, None]
        )
        value1 = (
            value1 * correction[:, None] + contribution1 * probability_scale[:, None]
        )
        maximum = next_maximum
    if SPLITS == 1:
        logsum = tl.where(denominator > 0, maximum + tl.log(denominator), float("inf"))
        inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
        if HAS_SINK:
            sink = tl.load(Sink + heads, heads < H, 0)
            inverse *= tl.where(
                sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
            )
        output_ptr = Output + (batch * H + heads[:, None]) * 512 + dims[None, :]
        tl.store(output_ptr, value0 * inverse[:, None], heads[:, None] < H)
        tl.store(output_ptr + 256, value1 * inverse[:, None], heads[:, None] < H)
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


@triton.jit
def sparse_fp8_merge(
    Partial,
    Stats,
    Sink,
    Output,
    LSE,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // H
    head = row % H
    split = tl.arange(0, SPLITS)
    dims = tl.arange(0, 512)
    stats_ptr = Stats + ((batch * SPLITS + split) * H + head) * 2
    maximum = tl.load(stats_ptr)
    denominator = tl.load(stats_ptr + 1)
    global_maximum = tl.max(maximum, 0)
    safe_maximum = tl.where(global_maximum == -float("inf"), 0.0, global_maximum)
    weights = tl.exp(maximum - safe_maximum)
    total = tl.sum(denominator * weights, 0)
    logsum = tl.where(total > 0, global_maximum + tl.log(total), float("inf"))
    inverse = tl.where(total > 0, 1.0 / total, 0.0)
    if HAS_SINK:
        sink = tl.load(Sink + head)
        inverse *= tl.where(
            sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
        )
    partial = tl.load(
        Partial + ((batch * SPLITS + split[:, None]) * H + head) * 512 + dims[None, :]
    )
    output = tl.sum(partial * weights[:, None], 0) * inverse
    tl.store(Output + row * 512 + dims, output)
    tl.store(LSE + row, logsum)


@triton.jit
def sparse_fp8_checked_merge(
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
):
    row = tl.program_id(0)
    batch, head = row // H, row % H
    split = tl.arange(0, SPLITS)
    flags = tl.load(Flags + (batch * (H // 64) + head // 64) * SPLITS + split)
    dims = tl.arange(0, 512)
    if tl.max(flags, 0) != 0:
        ropes = tl.arange(0, 64)
        query = tl.load(Q + batch * qb + head * qh + dims).to(tl.float32)
        query_rope = tl.load(QR + batch * qrb + head * qrh + ropes).to(tl.float32)
        query_scale = tl.load(QS + batch * qsb + head * qsh)
        length = (
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
        )
        maximum = -float("inf")
        denominator = 0.0
        values = tl.zeros((512,), tl.float32)
        for position in range(length):
            token = tl.load(Indices + batch * ib + position * ik)
            if (token >= 0) & (token < N):
                physical_token = token.to(tl.uint64)
                page, slot = physical_token // 64, physical_token % 64
                key = tl.load(KV + page * kp + slot * kt + dims).to(tl.float32)
                key_rope = tl.load(KR + page * krp + slot * krt + ropes).to(tl.float32)
                key_scale = tl.load(KS + page * ksp + slot * kst)
                score = (
                    (tl.sum(query * key, 0) + tl.sum(query_rope * key_rope, 0))
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
        tl.store(Output + row * 512 + dims, values * inverse)
        tl.store(LSE + row, logsum)
    elif SPLITS > 1:
        stats_ptr = Stats + ((batch * SPLITS + split) * H + head) * 2
        maxima = tl.load(stats_ptr)
        denominators = tl.load(stats_ptr + 1)
        global_maximum = tl.max(maxima, 0)
        safe_maximum = tl.where(global_maximum == -float("inf"), 0.0, global_maximum)
        weights = tl.exp(maxima - safe_maximum)
        total = tl.sum(denominators * weights, 0)
        logsum = tl.where(total > 0, global_maximum + tl.log(total), float("inf"))
        inverse = tl.where(total > 0, 1.0 / total, 0.0)
        if HAS_SINK:
            sink = tl.load(Sink + head)
            inverse *= tl.where(
                sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
            )
        else:
            pass
        partial = tl.load(
            Partial
            + ((batch * SPLITS + split[:, None]) * H + head) * 512
            + dims[None, :]
        )
        output = tl.sum(partial * weights[:, None], 0) * inverse
        tl.store(Output + row * 512 + dims, output)
        tl.store(LSE + row, logsum)
    else:
        pass


@triton.jit
def sparse_fp8_transpose_values(
    s_src,
    s_dst,
    dst_row: tl.constexpr,
):
    """Transpose with coupled K permutation and bank-distributed output rows."""
    carrier = tl.arange(0, 256).to(tl.uint32)
    src_base = tle.gpu.local_ptr(s_src, (0, 0))
    dst_base = tle.gpu.local_ptr(s_dst, (dst_row, 0))
    return tl.inline_asm_elementwise(
        asm=(
            "{\n"
            ".reg .b32 tid, lane, warp, src_row, tmp, tmp2;\n"
            ".reg .b32 src_log, src_phys, src_addr0, src_addr1;\n"
            ".reg .b32 dst_row_r, dst_col, dst_log0, dst_log1;\n"
            ".reg .b32 dst_phys0, dst_phys1, dst_addr0, dst_addr1;\n"
            ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
            ".reg .b32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
            "mov.u32 tid, %tid.x;\n"
            "and.b32 tid, tid, 255;\n"
            "and.b32 lane, tid, 31;\n"
            "shr.u32 warp, tid, 5;\n"
            # The coupled P permutation cancels the source-row bit permutation.
            "mov.u32 src_row, lane;\n"
            "shl.b32 src_log, src_row, 7;\n"
            "shl.b32 tmp, warp, 4;\n"
            "add.u32 src_log, src_log, tmp;\n"
            "shr.u32 tmp, src_log, 7;\n"
            "and.b32 tmp, tmp, 7;\n"
            "shl.b32 tmp, tmp, 4;\n"
            "xor.b32 src_phys, src_log, tmp;\n"
            "add.u32 src_addr0, $2, src_phys;\n"
            "add.u32 src_addr1, src_addr0, 4096;\n"
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
            "{a0, a1, a2, a3}, [src_addr0];\n"
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
            "{b0, b1, b2, b3}, [src_addr1];\n"
            "prmt.b32 c0, a0, a1, 0x6420;\n"
            "prmt.b32 c1, a0, a1, 0x7531;\n"
            "prmt.b32 c2, a2, a3, 0x6420;\n"
            "prmt.b32 c3, a2, a3, 0x7531;\n"
            "prmt.b32 d0, b0, b1, 0x6420;\n"
            "prmt.b32 d1, b0, b1, 0x7531;\n"
            "prmt.b32 d2, b2, b3, 0x6420;\n"
            "prmt.b32 d3, b2, b3, 0x7531;\n"
            # Consecutive rows distribute each matrix store over all shared banks.
            "and.b32 dst_row_r, lane, 15;\n"
            "shl.b32 tmp, warp, 4;\n"
            "add.u32 dst_row_r, dst_row_r, tmp;\n"
            "shr.u32 dst_col, lane, 4;\n"
            "and.b32 dst_col, dst_col, 1;\n"
            "shl.b32 dst_col, dst_col, 4;\n"
            "shl.b32 dst_log0, dst_row_r, 6;\n"
            "add.u32 dst_log0, dst_log0, dst_col;\n"
            "add.u32 dst_log1, dst_log0, 32;\n"
            "shr.u32 tmp, dst_log0, 7;\n"
            "and.b32 tmp, tmp, 3;\n"
            "shl.b32 tmp, tmp, 4;\n"
            "xor.b32 dst_phys0, dst_log0, tmp;\n"
            "shr.u32 tmp2, dst_log1, 7;\n"
            "and.b32 tmp2, tmp2, 3;\n"
            "shl.b32 tmp2, tmp2, 4;\n"
            "xor.b32 dst_phys1, dst_log1, tmp2;\n"
            "add.u32 dst_addr0, $3, dst_phys0;\n"
            "add.u32 dst_addr1, $3, dst_phys1;\n"
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "[dst_addr0], {c0, c1, c2, c3};\n"
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "[dst_addr1], {d0, d1, d2, d3};\n"
            "mov.u32 $0, $1;\n"
            "}"
        ),
        constraints="=r,r,r,r",
        args=[carrier, src_base, dst_base],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def sparse_named_wait_pair(barriers, slot):
    # Separate branches keep barrier IDs constant through the combine pass.
    if slot == 0:
        tle.gpu.barrier_wait(barriers[0])
    else:
        pass
    if slot == 1:
        tle.gpu.barrier_wait(barriers[1])
    else:
        pass


@triton.jit
def sparse_named_arrive_pair(barriers, slot):
    # Separate branches keep barrier IDs constant through the combine pass.
    if slot == 0:
        tle.gpu.barrier_arrive(barriers[0])
    else:
        pass
    if slot == 1:
        tle.gpu.barrier_arrive(barriers[1])
    else:
        pass


@triton.jit
def sparse_fp8_split_blocks(length, SPLITS: tl.constexpr):
    blocks = tl.cdiv(length, 64)
    if SPLITS == 1:
        first_block = 0
        split_blocks = blocks
    else:
        blocks_per_split = tl.cdiv(blocks, SPLITS)
        first_block = tl.program_id(2) * blocks_per_split
        split_blocks = tl.maximum(0, tl.minimum(blocks_per_split, blocks - first_block))
    return first_block, split_blocks


@triton.jit
def sparse_fp8_producer(
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
    sv0,
    sv1,
    sp,
    alpha,
    scales,
    mask,
    factor,
    qfull,
    kfull,
    kempty,
    pfull,
    ofull,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    SPLITS: tl.constexpr,
):
    batch = tl.program_id(0)
    heads = tl.program_id(1) * 64 + tl.arange(0, 64)
    dims = tl.arange(0, 512)
    ropes = tl.arange(0, 64)
    query = tl.load(Q + batch * qb + heads[:, None] * qh + dims[None, :])
    query_rope = tl.load(QR + batch * qrb + heads[:, None] * qrh + ropes[None, :])
    tl.store(tle.gpu.local_ptr(sq), query)
    tl.store(tle.gpu.local_ptr(sr), query_rope)
    tle.gpu.barrier_arrive(qfull[0])
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    first_block, split_blocks = sparse_fp8_split_blocks(length, SPLITS)
    for step in range(split_blocks):
        buf = step % 2
        if step >= 2:
            sparse_named_wait_pair(kempty, buf)
        else:
            pass
        positions = (first_block + step) * 64 + ropes
        ids = tl.load(Indices + batch * ib + positions * ik, positions < length, -1)
        valid = (positions < length) & (ids >= 0) & (ids < N)
        ids = tl.where(valid, ids, 0).to(tl.int64)
        (pages, slots) = (ids // 64, ids % 64)
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
                    sparse_smem_subslice(sk.slot(buf), [0, group * 128], [64, 128])
                ),
                content,
            )
        rope = tl.load(
            KR + pages[:, None] * krp + slots[:, None] * krt + ropes[None, :],
            valid[:, None],
            0.0,
        )
        tl.store(tle.gpu.local_ptr(skr.slot(buf)), rope)
        kv_scale = tl.load(KS + pages * ksp + slots * kst, valid, 0.0)
        tl.store(tle.gpu.local_ptr(scales.slot(buf)), kv_scale)
        tl.store(tle.gpu.local_ptr(mask.slot(buf)), tl.where(valid, 0.0, -float("inf")))
        # Matrix transpose avoids byte stores and their shared-memory bank conflicts.
        tl.debug_barrier()
        for group in tl.static_range(4):
            source = sparse_smem_subslice(sk.slot(buf), [0, group * 128], [64, 128])
            if group < 2:
                sparse_fp8_transpose_values(source, sv0.slot(buf), group * 128)
            else:
                sparse_fp8_transpose_values(source, sv1.slot(buf), (group - 2) * 128)
        tl.inline_asm_elementwise(
            "{ fence.proxy.async.shared::cta; mov.u32 $0, $1; }",
            constraints="=r,r",
            args=[tl.arange(0, 256)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        # Inline PTX stores are opaque to TLE's automatic publication barriers.
        tl.debug_barrier()
        sparse_named_arrive_pair(kfull, buf)


@triton.jit
def sparse_fp8_publish_output(
    acc, inverse, Output, batch, head_block, H: tl.constexpr, OFFSET: tl.constexpr
):
    if Output.dtype.element_ty == tl.float32:
        rows = head_block * 64 + tl.arange(0, 64)
        columns = tl.arange(0, 128)
        columns = (columns // 16) * 16 + (columns % 8) * 2 + (columns % 16) // 8
        tl.store(
            Output + (batch * H + rows[:, None]) * 512 + OFFSET + columns[None, :],
            acc * inverse[:, None],
        )
        return
    # Pair permuted accumulator columns into adjacent BF16 output elements.
    base_u64 = (Output + (batch * H + head_block * 64) * 512 + OFFSET).to(tl.uint64)
    tl.inline_asm_elementwise(
        asm=(
            "{\n"
            ".reg .b32 tid, row, col, offset, a, b, c, d;\n"
            ".reg .b64 addr;\n"
            "mov.u32 tid, %tid.x;\n"
            "and.b32 tid, tid, 127;\n"
            "shr.u32 row, tid, 5;\n"
            "shl.b32 row, row, 4;\n"
            "and.b32 col, tid, 31;\n"
            "shr.u32 col, col, 2;\n"
            "add.u32 row, row, col;\n"
            "and.b32 col, tid, 3;\n"
            "shl.b32 col, col, 3;\n"
            "shl.b32 row, row, 10;\n"
            "add.u32 offset, row, col;\n"
            "cvt.u64.u32 addr, offset;\n"
            "add.u64 addr, addr, $128;\n"
            "cvt.rn.bf16x2.f32 a, $68, $64;\n"
            "cvt.rn.bf16x2.f32 b, $69, $65;\n"
            "cvt.rn.bf16x2.f32 c, $70, $66;\n"
            "cvt.rn.bf16x2.f32 d, $71, $67;\n"
            "st.global.v2.b32 [addr+0], {a, b};\n"
            "st.global.v2.b32 [addr+8192], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $76, $72;\n"
            "cvt.rn.bf16x2.f32 b, $77, $73;\n"
            "cvt.rn.bf16x2.f32 c, $78, $74;\n"
            "cvt.rn.bf16x2.f32 d, $79, $75;\n"
            "st.global.v2.b32 [addr+32], {a, b};\n"
            "st.global.v2.b32 [addr+8224], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $84, $80;\n"
            "cvt.rn.bf16x2.f32 b, $85, $81;\n"
            "cvt.rn.bf16x2.f32 c, $86, $82;\n"
            "cvt.rn.bf16x2.f32 d, $87, $83;\n"
            "st.global.v2.b32 [addr+64], {a, b};\n"
            "st.global.v2.b32 [addr+8256], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $92, $88;\n"
            "cvt.rn.bf16x2.f32 b, $93, $89;\n"
            "cvt.rn.bf16x2.f32 c, $94, $90;\n"
            "cvt.rn.bf16x2.f32 d, $95, $91;\n"
            "st.global.v2.b32 [addr+96], {a, b};\n"
            "st.global.v2.b32 [addr+8288], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $100, $96;\n"
            "cvt.rn.bf16x2.f32 b, $101, $97;\n"
            "cvt.rn.bf16x2.f32 c, $102, $98;\n"
            "cvt.rn.bf16x2.f32 d, $103, $99;\n"
            "st.global.v2.b32 [addr+128], {a, b};\n"
            "st.global.v2.b32 [addr+8320], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $108, $104;\n"
            "cvt.rn.bf16x2.f32 b, $109, $105;\n"
            "cvt.rn.bf16x2.f32 c, $110, $106;\n"
            "cvt.rn.bf16x2.f32 d, $111, $107;\n"
            "st.global.v2.b32 [addr+160], {a, b};\n"
            "st.global.v2.b32 [addr+8352], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $116, $112;\n"
            "cvt.rn.bf16x2.f32 b, $117, $113;\n"
            "cvt.rn.bf16x2.f32 c, $118, $114;\n"
            "cvt.rn.bf16x2.f32 d, $119, $115;\n"
            "st.global.v2.b32 [addr+192], {a, b};\n"
            "st.global.v2.b32 [addr+8384], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $124, $120;\n"
            "cvt.rn.bf16x2.f32 b, $125, $121;\n"
            "cvt.rn.bf16x2.f32 c, $126, $122;\n"
            "cvt.rn.bf16x2.f32 d, $127, $123;\n"
            "st.global.v2.b32 [addr+224], {a, b};\n"
            "st.global.v2.b32 [addr+8416], {c, d};\n"
            "mov.u32 $0, 0;\n"
            "}\n"
        ),
        constraints=(
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,"
            "l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,"
            "l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,"
            "l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l"
        ),
        args=[acc * inverse[:, None], base_u64],
        dtype=tl.uint32,
        is_pure=False,
        pack=64,
    )


@triton.jit
def sparse_fp8_consumer0(
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
    sv0,
    sv1,
    sp,
    alpha,
    scales,
    mask,
    factor,
    qfull,
    kfull,
    kempty,
    pfull,
    ofull,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    RepairFlags,
):
    batch = tl.program_id(0)
    heads = tl.program_id(1) * 64 + tl.arange(0, 64)
    # Keep online softmax in base-2 units and fold row-constant scaling once.
    query_scale = tl.load(QS + batch * qsb + heads * qsh) * (SCALE * 1.4426950408889634)
    amplification = tl.max(tl.abs(query_scale), 0)
    needs_repair = tl.full((), False, tl.int1)
    maximum_weight_scale = tl.zeros((64,), tl.float32)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    maximum = tl.full((64,), -float("inf"), tl.float32)
    denominator = tl.zeros((64,), tl.float32)
    acc0 = tl.zeros((64, 128), tl.float32)
    acc1 = tl.zeros((64, 128), tl.float32)
    previous_scale = tl.full((64,), 1.0, tl.float32)
    tle.gpu.barrier_wait(qfull[0])
    first_block, split_blocks = sparse_fp8_split_blocks(length, SPLITS)
    for step in range(split_blocks):
        buf = step % 2
        sparse_named_wait_pair(kfull, buf)
        logits = tle.gpu.wgmma(sq, sk.slot(buf), out_dtype=tl.float32, trans_b=True)
        logits = tle.gpu.wgmma(sr, skr.slot(buf), logits, trans_b=True)
        logits = tle.gpu.wgmma_wait(0, logits)
        kv_scale = tl.load(tle.gpu.local_ptr(scales.slot(buf)))
        needs_repair |= amplification * tl.max(kv_scale, 0) > QK_RECOMPUTE_THRESHOLD
        add_mask = tl.load(tle.gpu.local_ptr(mask.slot(buf)))
        logits = logits * query_scale[:, None] * kv_scale[None, :] + add_mask[None, :]
        next_maximum = tl.maximum(maximum, tl.max(logits, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp2(maximum - safe_maximum)
        probabilities = tl.exp2(logits - safe_maximum[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        weighted = probabilities * kv_scale[None, :]
        probability_scale = tl.max(weighted, 1) / 448.0
        maximum_weight_scale = tl.maximum(
            maximum_weight_scale * correction, probability_scale
        )
        probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
        needs_repair |= (
            tl.max(
                (maximum_weight_scale * ACCUMULATOR_SCALE_FLOOR > probability_scale).to(
                    tl.int32
                ),
                0,
            )
            != 0
        )
        p = weighted / probability_scale[:, None]
        # P and V share the same K permutation, avoiding cross-lane P shuffles.
        publish_p_fp8_sw64_coupled_stmatrix(sp.slot(buf), p)
        # Keep PV in probability-scale units, requiring one rescale per tile.
        correction = correction * previous_scale / probability_scale
        tl.store(tle.gpu.local_ptr(alpha.slot(buf)), correction)
        sparse_named_arrive_pair(pfull, buf)
        acc0 *= correction[:, None]
        acc1 *= correction[:, None]
        acc0 = tle.gpu.wgmma(
            sp.slot(buf),
            sparse_smem_subslice(sv0.slot(buf), [0, 0], [128, 64]),
            acc0,
            trans_b=True,
        )
        acc1 = tle.gpu.wgmma(
            sp.slot(buf),
            sparse_smem_subslice(sv0.slot(buf), [128, 0], [128, 64]),
            acc1,
            trans_b=True,
        )
        acc0 = tle.gpu.wgmma_wait(0, acc0)
        acc1 = tle.gpu.wgmma_wait(0, acc1)
        previous_scale = probability_scale
        maximum = next_maximum
        sparse_named_arrive_pair(kempty, buf)
    if SPLITS == 1:
        logsum = tl.where(
            denominator > 0,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
            float("inf"),
        )
        inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
        if HAS_SINK:
            sink = tl.load(Sink + heads)
            inverse *= tl.where(
                sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
            )
        inverse *= previous_scale
    else:
        # Keep partial values in physical units; merge applies the sink only once.
        inverse = previous_scale
    output_batch = batch if SPLITS == 1 else batch * SPLITS + tl.program_id(2)
    tl.store(tle.gpu.local_ptr(factor), inverse)
    tle.gpu.barrier_arrive(ofull[0], phaseIdx=0)
    sparse_fp8_publish_output(
        acc0, inverse, Output, output_batch, tl.program_id(1), H, 0
    )
    sparse_fp8_publish_output(
        acc1, inverse, Output, output_batch, tl.program_id(1), H, 128
    )
    if SPLITS == 1:
        tl.store(LSE + batch * H + heads, logsum)
    else:
        stats = LSE + ((batch * SPLITS + tl.program_id(2)) * H + heads) * 2
        tl.store(stats, maximum * 0.6931471805599453)
        tl.store(stats + 1, denominator)
    tl.store(
        RepairFlags
        + (batch * (H // 64) + tl.program_id(1)) * SPLITS
        + tl.program_id(2),
        needs_repair.to(tl.int32),
    )


@triton.jit
def sparse_fp8_consumer1(
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
    sq,
    sr,
    sk,
    skr,
    sv0,
    sv1,
    sp,
    alpha,
    scales,
    mask,
    factor,
    qfull,
    kfull,
    kempty,
    pfull,
    ofull,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    SPLITS: tl.constexpr,
):
    batch = tl.program_id(0)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    acc0 = tl.zeros((64, 128), tl.float32)
    acc1 = tl.zeros((64, 128), tl.float32)
    first_block, split_blocks = sparse_fp8_split_blocks(length, SPLITS)
    for step in range(split_blocks):
        buf = step % 2
        sparse_named_wait_pair(pfull, buf)
        correction = tl.load(tle.gpu.local_ptr(alpha.slot(buf)))
        acc0 *= correction[:, None]
        acc1 *= correction[:, None]
        acc0 = tle.gpu.wgmma(
            sp.slot(buf),
            sparse_smem_subslice(sv1.slot(buf), [0, 0], [128, 64]),
            acc0,
            trans_b=True,
        )
        acc1 = tle.gpu.wgmma(
            sp.slot(buf),
            sparse_smem_subslice(sv1.slot(buf), [128, 0], [128, 64]),
            acc1,
            trans_b=True,
        )
        acc0 = tle.gpu.wgmma_wait(0, acc0)
        acc1 = tle.gpu.wgmma_wait(0, acc1)
        sparse_named_arrive_pair(kempty, buf)
    tle.gpu.barrier_wait(ofull[0], phaseIdx=0)
    inverse = tl.load(tle.gpu.local_ptr(factor))
    output_batch = batch if SPLITS == 1 else batch * SPLITS + tl.program_id(2)
    sparse_fp8_publish_output(
        acc0, inverse, Output, output_batch, tl.program_id(1), H, 256
    )
    sparse_fp8_publish_output(
        acc1, inverse, Output, output_batch, tl.program_id(1), H, 384
    )


if HAS_TLE:

    @triton.jit
    def publish_p_fp8_sw64_coupled_stmatrix(s_p, p):
        """CUDA-native P publication; V repack carries the matching K permutation."""
        base = tle.gpu.local_ptr(s_p, (0, 0))
        base_u32 = tl.inline_asm_elementwise(
            asm="mov.u32 $0, $1;",
            constraints="=r,r",
            args=[base],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15;\n"
                ".reg .b32 tid, warp_off, row_off, common, tmp, phys0, phys1, addr0, addr1;\n"
                ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 warp_off, tid, 96;\n"
                "shl.b32 warp_off, warp_off, 5;\n"
                "and.b32 row_off, tid, 15;\n"
                "shl.b32 row_off, row_off, 6;\n"
                "or.b32 common, warp_off, row_off;\n"
                "and.b32 tmp, tid, 16;\n"
                "or.b32 common, common, tmp;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h0, $33, $32;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h1, $35, $34;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h2, $37, $36;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h3, $39, $38;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h4, $41, $40;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h5, $43, $42;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h6, $45, $44;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h7, $47, $46;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h8, $49, $48;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h9, $51, $50;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h10, $53, $52;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h11, $55, $54;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h12, $57, $56;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h13, $59, $58;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h14, $61, $60;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h15, $63, $62;\n"
                "mov.b32 a0, {h0, h2};\n"
                "mov.b32 a1, {h1, h3};\n"
                "mov.b32 a2, {h4, h6};\n"
                "mov.b32 a3, {h5, h7};\n"
                "mov.b32 b0, {h8, h10};\n"
                "mov.b32 b1, {h9, h11};\n"
                "mov.b32 b2, {h12, h14};\n"
                "mov.b32 b3, {h13, h15};\n"
                "shr.u32 tmp, common, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 phys0, common, tmp;\n"
                "add.u32 common, common, 32;\n"
                "shr.u32 tmp, common, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 phys1, common, tmp;\n"
                "add.u32 addr0, $64, phys0;\n"
                "add.u32 addr1, $64, phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr0], {a0, a1, a2, a3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr1], {b0, b1, b2, b3};\n"
                "fence.proxy.async.shared::cta;\n"
                "mov.u32 $0, $64;\n"
                "}"
            ),
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",  # noqa: E501
            args=[p, base_u32],
            dtype=tl.uint32,
            is_pure=False,
            pack=32,
        )

    @triton.jit
    def vtranspose_fp8_64x128_kperm(
        s_src,
        s_dst,
        dst_row: tl.constexpr,
    ):
        """CUDA-authority SW128 -> SW64 FP8 transpose for one 64x128 tile."""
        carrier = tl.arange(0, 128).to(tl.uint32)
        src_base = tle.gpu.local_ptr(s_src, (0, 0))
        dst_base = tle.gpu.local_ptr(s_dst, (dst_row, 0))
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b32 tid, lane, warp, src_row, tmp, tmp2;\n"
                ".reg .b32 src_log, src_phys, src_addr0, src_addr1;\n"
                ".reg .b32 dst_row_r, dst_col, dst_log0, dst_log1;\n"
                ".reg .b32 dst_phys0, dst_phys1, dst_addr0, dst_addr1;\n"
                ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
                ".reg .b32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 tid, tid, 127;\n"
                "and.b32 lane, tid, 31;\n"
                "shr.u32 warp, tid, 5;\n"
                # CUDA's LDSM/STSM register order presents source-row bits as
                # [b1,b3,b2,b0] to TLE's logical SW64 view.  Apply the inverse
                # [b3,b1,b2,b0] mapping at the load boundary so PV observes
                # the same logical transpose as the tensor path.
                "and.b32 src_row, lane, 17;\n"
                "and.b32 tmp, lane, 8;\n"
                "shr.u32 tmp, tmp, 2;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 2;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 4;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                # Direct CUDA STSM presents P to TLE as dest <- source pi,
                # pi=[0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] per K16.
                # Load V from pi(dest) as well, preserving the dot product
                # while removing the publication-side cross-lane shuffle.
                "mov.u32 tmp2, src_row;\n"
                "and.b32 src_row, tmp2, 17;\n"
                "and.b32 tmp, tmp2, 4;\n"
                "shr.u32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, tmp2, 8;\n"
                "shr.u32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, tmp2, 2;\n"
                "shl.b32 tmp, tmp, 2;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "shl.b32 src_log, src_row, 7;\n"
                "shl.b32 tmp, warp, 4;\n"
                "add.u32 src_log, src_log, tmp;\n"
                "shr.u32 tmp, src_log, 7;\n"
                "and.b32 tmp, tmp, 7;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 src_phys, src_log, tmp;\n"
                "add.u32 src_addr0, $2, src_phys;\n"
                "add.u32 src_addr1, src_addr0, 4096;\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{a0, a1, a2, a3}, [src_addr0];\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{b0, b1, b2, b3}, [src_addr1];\n"
                "prmt.b32 c0, a0, a1, 0x6420;\n"
                "prmt.b32 c1, a0, a1, 0x7531;\n"
                "prmt.b32 c2, a2, a3, 0x6420;\n"
                "prmt.b32 c3, a2, a3, 0x7531;\n"
                "prmt.b32 d0, b0, b1, 0x6420;\n"
                "prmt.b32 d1, b0, b1, 0x7531;\n"
                "prmt.b32 d2, b2, b3, 0x6420;\n"
                "prmt.b32 d3, b2, b3, 0x7531;\n"
                "and.b32 dst_row_r, lane, 7;\n"
                "shl.b32 dst_row_r, dst_row_r, 1;\n"
                "shr.u32 tmp, lane, 3;\n"
                "and.b32 tmp, tmp, 1;\n"
                "add.u32 dst_row_r, dst_row_r, tmp;\n"
                "shl.b32 tmp, warp, 4;\n"
                "add.u32 dst_row_r, dst_row_r, tmp;\n"
                "shr.u32 dst_col, lane, 4;\n"
                "and.b32 dst_col, dst_col, 1;\n"
                "shl.b32 dst_col, dst_col, 4;\n"
                "shl.b32 dst_log0, dst_row_r, 6;\n"
                "add.u32 dst_log0, dst_log0, dst_col;\n"
                "add.u32 dst_log1, dst_log0, 32;\n"
                "shr.u32 tmp, dst_log0, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 dst_phys0, dst_log0, tmp;\n"
                "shr.u32 tmp2, dst_log1, 7;\n"
                "and.b32 tmp2, tmp2, 3;\n"
                "shl.b32 tmp2, tmp2, 4;\n"
                "xor.b32 dst_phys1, dst_log1, tmp2;\n"
                "add.u32 dst_addr0, $3, dst_phys0;\n"
                "add.u32 dst_addr1, $3, dst_phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr0], {c0, c1, c2, c3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr1], {d0, d1, d2, d3};\n"
                "add.u32 src_log, src_log, 64;\n"
                "shr.u32 tmp, src_log, 7;\n"
                "and.b32 tmp, tmp, 7;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 src_phys, src_log, tmp;\n"
                "add.u32 src_addr0, $2, src_phys;\n"
                "add.u32 src_addr1, src_addr0, 4096;\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{a0, a1, a2, a3}, [src_addr0];\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{b0, b1, b2, b3}, [src_addr1];\n"
                "prmt.b32 c0, a0, a1, 0x6420;\n"
                "prmt.b32 c1, a0, a1, 0x7531;\n"
                "prmt.b32 c2, a2, a3, 0x6420;\n"
                "prmt.b32 c3, a2, a3, 0x7531;\n"
                "prmt.b32 d0, b0, b1, 0x6420;\n"
                "prmt.b32 d1, b0, b1, 0x7531;\n"
                "prmt.b32 d2, b2, b3, 0x6420;\n"
                "prmt.b32 d3, b2, b3, 0x7531;\n"
                "add.u32 dst_log0, dst_log0, 4096;\n"
                "add.u32 dst_log1, dst_log1, 4096;\n"
                "shr.u32 tmp, dst_log0, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 dst_phys0, dst_log0, tmp;\n"
                "shr.u32 tmp2, dst_log1, 7;\n"
                "and.b32 tmp2, tmp2, 3;\n"
                "shl.b32 tmp2, tmp2, 4;\n"
                "xor.b32 dst_phys1, dst_log1, tmp2;\n"
                "add.u32 dst_addr0, $3, dst_phys0;\n"
                "add.u32 dst_addr1, $3, dst_phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr0], {c0, c1, c2, c3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr1], {d0, d1, d2, d3};\n"
                "mov.u32 $0, $1;\n"
                "}"
            ),
            constraints="=r,r,r,r",
            args=[carrier, src_base, dst_base],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

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
            key = tl.load(
                KV + (ids // 64) * kp + (ids % 64) * kt + feature, valid, 0.0
            ).to(tl.float32)
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
        rope = tl.load(
            KR + pages[:, None] * krp + slots[:, None] * krt + ropes[None, :],
            valid[:, None],
            0.0,
            volatile=not CAN_ASYNC,
        )
        scale = tl.load(KS + pages * ksp + slots * kst, valid, 0.0)
        tl.store(tle.gpu.local_ptr(sk.slot(slot)), cache)
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
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        TOPK: tl.constexpr,
        SCALE: tl.constexpr,
        SPLITS: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        HAS_SINK: tl.constexpr,
        CAN_ASYNC: tl.constexpr,
        BLOCK_K: tl.constexpr,
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
        query_rope = tl.load(
            QR + batch * qrb + heads[:, None] * qrh + ropes[None, :],
            volatile=not CAN_ASYNC,
        )
        tl.store(tle.gpu.local_ptr(sq), query)
        tl.store(tle.gpu.local_ptr(sr), query_rope)
        query_scale = tl.load(QS + batch * qsb + heads * qsh) * (
            SCALE * 1.4426950408889634
        )
        amplification = tl.max(tl.abs(query_scale), 0)
        needs_repair = tl.full((), False, tl.int1)
        maximum_weight_scale = tl.zeros((64,), tl.float32)
        maximum = tl.full((64,), -float("inf"), tl.float32)
        denominator = tl.zeros((64,), tl.float32)
        value = tl.zeros((64, LEADER_D), tl.float32)
        previous_scale = tl.full((64,), 1.0, tl.float32)
        length = (
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
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
                        )
                    logits = tle.gpu.wgmma(
                        sq, sk.slot(slot), out_dtype=tl.float32, trans_b=True
                    )
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
                        left, right = (
                            vp.reshape(BLOCK_K, 2, 256).permute(0, 2, 1).split()
                        )
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
                                vtranspose_fp8_64x128_kperm(
                                    source, sv1, (tile - 2) * 128
                                )
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
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
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
        key=["B", "H", "TOPK", "SPLITS", "CAN_ASYNC"],
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
                        B,
                        H,
                        N,
                        TOPK,
                        SCALE,
                        SPLITS,
                        HAS_LENGTH,
                        HAS_SINK,
                        CAN_ASYNC,
                        BLOCK_K,
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
        key=["B", "H", "TOPK", "SPLITS", "CAN_ASYNC"],
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
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAN_ASYNC: tl.constexpr,
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
        query_rope = tl.load(
            QR + batch * qrb + heads[:, None] * qrh + ropes[None, :],
            volatile=not CAN_ASYNC,
        )
        tl.store(tle.gpu.local_ptr(sq), query)
        tl.store(tle.gpu.local_ptr(sr), query_rope)
        query_scale = tl.load(QS + batch * qsb + heads * qsh) * (
            SCALE * 1.4426950408889634
        )
        amplification = tl.max(tl.abs(query_scale), 0)
        maximum = tl.full((64,), -float("inf"), tl.float32)
        denominator = tl.zeros((64,), tl.float32)
        value = tl.zeros((64, BLOCK_D), tl.float32)
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
            cache_rope = tl.load(
                KR + pages[:, None] * krp + slots[:, None] * krt + ropes[None, :],
                valid[:, None],
                0.0,
                volatile=not CAN_ASYNC,
            )
            tl.store(tle.gpu.local_ptr(sk), cache)
            tl.store(tle.gpu.local_ptr(skr), cache_rope)
            tl.debug_barrier()
            logits = tle.gpu.wgmma(sq, sk, out_dtype=tl.float32, trans_b=True)
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
            publish_p_fp8_sw64_coupled_stmatrix(
                sp, weighted / probability_scale[:, None]
            )
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

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("sparse_fp8_warp_specialized"),
        key=["B", "H", "TOPK", "SPLITS"],
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
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        TOPK: tl.constexpr,
        SCALE: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        SPLITS: tl.constexpr,
        HAS_SINK: tl.constexpr,
        RepairFlags,
        PRODUCER_REGS: tl.constexpr,
    ):
        sq = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
        sr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
        sk = tle.gpu.alloc([2, 64, 512], tl.float8e4nv, scope=tle.gpu.smem)
        skr = tle.gpu.alloc([2, 64, 64], tl.bfloat16, scope=tle.gpu.smem)
        sv0 = tle.gpu.alloc([2, 256, 64], tl.float8e4nv, scope=tle.gpu.smem)
        sv1 = tle.gpu.alloc([2, 256, 64], tl.float8e4nv, scope=tle.gpu.smem)
        sp = tle.gpu.alloc([2, 64, 64], tl.float8e4nv, scope=tle.gpu.smem)
        alpha = tle.gpu.alloc(
            [2, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        scales = tle.gpu.alloc(
            [2, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        mask = tle.gpu.alloc(
            [2, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        factor = tle.gpu.alloc(
            [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        # TLE maps these virtual IDs to physical IDs outside the WS reserved set.
        qfull = sparse_named_barriers(1, 384, 16)
        kfull = sparse_named_barriers(2, 384, 17)
        kempty = sparse_named_barriers(2, 512, 19)
        pfull = sparse_named_barriers(2, 256, 21)
        ofull = tle.gpu.alloc_barriers(1, arrive_count=1)
        tle.gpu.warp_specialize(
            [
                (
                    sparse_fp8_consumer0,
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
                        LSE,
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
                        sv0,
                        sv1,
                        sp,
                        alpha,
                        scales,
                        mask,
                        factor,
                        qfull,
                        kfull,
                        kempty,
                        pfull,
                        ofull,
                        H,
                        TOPK,
                        SCALE,
                        HAS_LENGTH,
                        SPLITS,
                        HAS_SINK,
                        RepairFlags,
                    ),
                ),
                (
                    sparse_fp8_producer,
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
                        LSE,
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
                        sv0,
                        sv1,
                        sp,
                        alpha,
                        scales,
                        mask,
                        factor,
                        qfull,
                        kfull,
                        kempty,
                        pfull,
                        ofull,
                        H,
                        N,
                        TOPK,
                        HAS_LENGTH,
                        SPLITS,
                    ),
                ),
                (
                    sparse_fp8_consumer1,
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
                        LSE,
                        sq,
                        sr,
                        sk,
                        skr,
                        sv0,
                        sv1,
                        sp,
                        alpha,
                        scales,
                        mask,
                        factor,
                        qfull,
                        kfull,
                        kempty,
                        pfull,
                        ofull,
                        H,
                        TOPK,
                        HAS_LENGTH,
                        SPLITS,
                    ),
                ),
            ],
            [8, 4],
            [PRODUCER_REGS, 168],
        )

else:
    sparse_fp8_precise_scores = None
    sparse_fp8_load_tile = None
    sparse_fp8_compact_leader = None
    sparse_fp8_compact_follower = None
    sparse_fp8_compact = None
    sparse_fp8_tile = None
    sparse_fp8_warp_specialized = None


def flash_mla_sparse_fwd_w8a8_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode using the separate per-token cache format of dense MLA.

    q_nope [B, 1, H, 512] and k_cache_lora [P, 64, 512] are FP8 e4m3fn;
    q_rope [B, 1, H, 64] and k_cache_rope [P, 64, 64] are BF16.
    Both NoPE and RoPE store values divided by the corresponding FP32 scale:
    q_scale [B, 1, H, 1] and k_scale [P, 64, 1]. This directly accepts
    quantize_q_ckv_per_token / quantize_k_ckv_per_token outputs from dense MLA.
    V is the dequantized 512-dimensional NoPE cache. H must be 64 or 128.

    indices [B, 1, topk] contains int32 physical token IDs (page * 64 + slot).
    Negative and out-of-range IDs are ignored. topk_length [B] optionally
    limits the number of entries read per request. attn_sink [H] affects
    output only. Inputs must be finite, with positive finite scales.

    Returns BF16 output [B, 1, H, 512] and natural-log FP32 LSE [B, H, 1].
    Empty attention produces zero output and +inf LSE. Forward only.
    Sensitive rows use FP32 recomputation inline, during merge, or in the
    existing separate repair kernel, depending on the selected path.
    Requires FlagTree cross-dtype WGMMA support (PR #1001).
    """
    if not HAS_TLE:
        raise NotImplementedError(
            "FP8 sparse MLA requires Hopper and FlagTree GPU extensions"
        )
    if q_nope.device.type != "cuda":
        raise NotImplementedError("FP8 sparse MLA requires NVIDIA Hopper CUDA")
    if torch.cuda.get_device_capability(q_nope.device)[0] != 9:
        raise NotImplementedError("FP8 sparse MLA is supported on Hopper")
    if q_nope.dtype != torch.float8_e4m3fn or k_cache_lora.dtype != torch.float8_e4m3fn:
        raise TypeError("NoPE tensors must have dtype float8_e4m3fn")
    if q_rope.dtype != torch.bfloat16 or k_cache_rope.dtype != torch.bfloat16:
        raise TypeError("RoPE tensors must have dtype bfloat16")
    if q_nope.ndim != 4 or k_cache_lora.ndim != 3 or indices.ndim != 3:
        raise ValueError("q_nope, cache and indices must have ranks 4, 3 and 3")
    batch, query_length, heads, dim = q_nope.shape
    pages = k_cache_lora.shape[0]
    topk = indices.shape[-1]
    if query_length != 1 or heads not in (64, 128) or dim != 512:
        raise NotImplementedError(
            "Requires one query, 64/128 heads and 512 NoPE dimensions"
        )
    if q_rope.shape != (batch, 1, heads, 64):
        raise ValueError("q_rope must have shape [batch, 1, heads, 64]")
    if k_cache_lora.shape != (pages, 64, 512) or k_cache_rope.shape != (pages, 64, 64):
        raise ValueError(
            "Caches must have page size 64 and NoPE/RoPE dimensions 512/64"
        )
    if indices.shape != (batch, 1, topk) or indices.dtype != torch.int32:
        raise ValueError("indices must be int32 [batch, 1, topk]")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("q_scale and k_scale must have dtype float32")
    if q_scale.shape != (batch, 1, heads, 1) or k_scale.shape != (pages, 64, 1):
        raise ValueError("Scale shapes must be [batch, 1, heads, 1] and [pages, 64, 1]")
    for tensor in (q_nope, q_rope, k_cache_lora, k_cache_rope):
        if tensor.stride(-1) != 1:
            raise ValueError("NoPE and RoPE must be contiguous in the last dimension")
    for tensor in (
        q_rope,
        k_cache_lora,
        k_cache_rope,
        indices,
        q_scale,
        k_scale,
        attn_sink,
        topk_length,
    ):
        if tensor is not None and tensor.device != q_nope.device:
            raise ValueError("All tensors must be on the same CUDA device")
    if attn_sink is not None and (
        attn_sink.shape != (heads,)
        or attn_sink.dtype != torch.float32
        or not attn_sink.is_contiguous()
    ):
        raise ValueError("attn_sink must be contiguous float32 [heads]")
    if topk_length is not None and (
        topk_length.shape != (batch,)
        or topk_length.dtype != torch.int32
        or not topk_length.is_contiguous()
    ):
        raise ValueError("topk_length must be contiguous int32 [batch]")
    output = torch.empty(
        (batch, 1, heads, 512), device=q_nope.device, dtype=torch.bfloat16
    )
    lse = torch.empty((batch, heads, 1), device=q_nope.device, dtype=torch.float32)
    if batch == 0:
        return output, lse
    if pages == 0 or topk == 0:
        rows = batch * heads
        sparse_fp8_empty[(triton.cdiv(rows * 512, 1024),)](
            output, lse, rows, num_warps=4
        )
        return output, lse
    softmax_scale = 576**-0.5 if softmax_scale is None else float(softmax_scale)
    use_partitioned = batch <= 16
    use_tile = False
    num_sms = _get_num_sms(q_nope.device)
    head_groups = batch * (heads // 64)
    desired_splits = max(1, num_sms // head_groups)
    keys_per_split = 64 if batch <= 16 else 256
    max_splits = min(desired_splits, 32, max(1, topk // keys_per_split))
    splits = 1 << (max_splits.bit_length() - 1)
    if use_partitioned:
        can_async_copy = all(
            tensor.data_ptr() % 16 == 0
            and tensor.stride(0) * tensor.element_size() % 16 == 0
            and tensor.stride(-2) * tensor.element_size() % 16 == 0
            for tensor in (q_nope, q_rope, k_cache_lora, k_cache_rope)
        )
        tile_splits = triton.next_power_of_2(max(1, triton.cdiv(topk, 64)))
        # Keep value tiling in its validated workset; short rows use the compact CTA.
        use_tile = topk >= 512 and head_groups <= 8 and tile_splits <= 32
        if use_tile:
            splits = tile_splits
        else:
            # Checked merge consumes partial statistics, including for empty/short rows.
            splits = max(splits, 2)
    repair_splits = splits
    flag_heads = heads // 64
    repair_flags = torch.empty(
        (batch, flag_heads, splits), device=q_nope.device, dtype=torch.int32
    )
    if splits > 1:
        partial = torch.empty(
            (batch, splits, heads, 512),
            device=q_nope.device,
            dtype=torch.float32,
        )
        stats = torch.empty(
            (batch, splits, heads, 2), device=q_nope.device, dtype=torch.float32
        )
    else:
        partial, stats = output, lse
    if use_tile:
        sparse_fp8_tile[
            lambda meta: (batch, heads // 64, splits * (512 // meta["BLOCK_D"]))
        ](
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            partial,
            stats,
            repair_flags,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            splits,
            topk_length is not None,
            attn_sink is not None,
            CAN_ASYNC=can_async_copy,
        )
    elif use_partitioned:
        sparse_fp8_compact[(batch, heads // 64, splits)](
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            partial,
            stats,
            repair_flags,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            splits,
            topk_length is not None,
            attn_sink is not None,
            CAN_ASYNC=can_async_copy,
        )
    else:
        sparse_fp8_warp_specialized[batch, heads // 64, splits](
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            partial,
            stats,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            topk_length is not None,
            splits,
            attn_sink is not None,
            repair_flags,
        )
    if not use_partitioned:
        sparse_fp8_repair[
            lambda meta: (batch, triton.cdiv(heads, meta["BLOCK_H"]), splits)
        ](
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            indices,
            q_scale,
            k_scale,
            attn_sink,
            topk_length,
            output,
            lse,
            partial,
            stats,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            indices.stride(0),
            indices.stride(2),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            splits,
            attn_sink is not None,
            topk_length is not None,
            repair_flags,
            REPAIR_SPLITS=repair_splits,
        )
    if use_partitioned and not use_tile:
        sparse_fp8_checked_merge[(batch * heads,)](
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            repair_flags,
            partial,
            stats,
            output,
            lse,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            heads,
            pages * 64,
            topk,
            softmax_scale,
            splits,
            topk_length is not None,
            attn_sink is not None,
            num_warps=4,
        )
    elif splits > 1:
        sparse_fp8_merge[(batch * heads,)](
            partial,
            stats,
            attn_sink,
            output,
            lse,
            heads,
            splits,
            attn_sink is not None,
            num_warps=4,
        )
    return output, lse
