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

import torch
import triton
import triton.language as tl

# GCU400 perf profile (measured on this op, L600):
#  - ~100 ns/CTA dispatch floor: the original one-program-per-(token,
#    head-or-kv) launch was fully dispatch-bound (19.7 GB/s effective,
#    latency linear in program count up to 16.9M programs).
#  - ~60 ns per 1-D memory op inside a program (DTE serialization); loads
#    co-issued before the first store overlap, but any store serializes
#    against later loads (conservative cross-pointer aliasing).
#  - staged reductions pipeline (~27 ns marginal once inputs are resident),
#    while load->reduce->store chains expose the full tree latency.
#  - 2-D tile memory access fails to compile (PassManager), and 2-D/3-D
#    register layout chains (reshape/split/join, axis!=0 reduce on tiles)
#    materialize through DSM ~3-6x slower. The kernel stays strictly 1-D.
#
# Structure: persistent grid capped at 48 programs (dual program/SP keeps
# 48-way concurrency while per-program DSM stays ~270 KB at U=32 f32
# staging; U=64 f32 exceeds the 896 KB/SP budget and falls to 24-way),
# each program strides over 32-row q groups and single-token kv items.
_GRID_TARGET = 48
_Q_GROUP = 16


@triton.jit
def fused_qnorm_rope_kv_insert_kernel(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps,
    cache_block_size: tl.constexpr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    kv_block_stride,
    num_tokens_insert: tl.constexpr,
    q_aligned: tl.constexpr,
):
    HEAD_DIM: tl.constexpr = 512
    NOPE_DIM: tl.constexpr = 448
    ROPE_DIM: tl.constexpr = 64
    HALF_ROPE_DIM: tl.constexpr = 32
    QUANT_BLOCK: tl.constexpr = 64
    NUM_QUANT_BLOCKS: tl.constexpr = NOPE_DIM // QUANT_BLOCK  # 7
    SCALE_BYTES_PER_TOKEN: tl.constexpr = NUM_QUANT_BLOCKS + 1  # 8 (7 real + 1 pad)
    TOKEN_DATA_BYTES: tl.constexpr = NOPE_DIM + 2 * ROPE_DIM  # 576
    FP8_MAX: tl.constexpr = 448.0
    NUM_ROWS: tl.constexpr = num_tokens * num_heads
    Q_GROUP: tl.constexpr = 16

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    offset = tl.arange(0, HEAD_DIM)
    mask_nope = offset < NOPE_DIM
    offset_half_rope = tl.arange(0, HALF_ROPE_DIM)
    offset_even = 2 * offset_half_rope
    offset_odd = offset_even + 1
    offset_quant = tl.arange(0, QUANT_BLOCK)

    if q_aligned:
        NUM_GROUPS: tl.constexpr = (NUM_ROWS + Q_GROUP - 1) // Q_GROUP
        for group in range(pid, NUM_GROUPS, num_programs):
            base = group * Q_GROUP
            # one token per group (Q_GROUP divides num_heads): hoist
            # position/cos/sin loads out of the row loop
            tok = base // num_heads
            position_id = tl.load(position_ids + tok).to(tl.int64)
            cs_base = cos_sin_cache + position_id * ROPE_DIM
            cos_blk = tl.load(cs_base + offset_half_rope)
            sin_blk = tl.load(cs_base + offset_half_rope + HALF_ROPE_DIM)
            r0 = tl.minimum(base + 0, NUM_ROWS - 1).to(tl.int64)
            qb0 = q + r0 * HEAD_DIM
            r1 = tl.minimum(base + 1, NUM_ROWS - 1).to(tl.int64)
            qb1 = q + r1 * HEAD_DIM
            r2 = tl.minimum(base + 2, NUM_ROWS - 1).to(tl.int64)
            qb2 = q + r2 * HEAD_DIM
            r3 = tl.minimum(base + 3, NUM_ROWS - 1).to(tl.int64)
            qb3 = q + r3 * HEAD_DIM
            r4 = tl.minimum(base + 4, NUM_ROWS - 1).to(tl.int64)
            qb4 = q + r4 * HEAD_DIM
            r5 = tl.minimum(base + 5, NUM_ROWS - 1).to(tl.int64)
            qb5 = q + r5 * HEAD_DIM
            r6 = tl.minimum(base + 6, NUM_ROWS - 1).to(tl.int64)
            qb6 = q + r6 * HEAD_DIM
            r7 = tl.minimum(base + 7, NUM_ROWS - 1).to(tl.int64)
            qb7 = q + r7 * HEAD_DIM
            r8 = tl.minimum(base + 8, NUM_ROWS - 1).to(tl.int64)
            qb8 = q + r8 * HEAD_DIM
            r9 = tl.minimum(base + 9, NUM_ROWS - 1).to(tl.int64)
            qb9 = q + r9 * HEAD_DIM
            r10 = tl.minimum(base + 10, NUM_ROWS - 1).to(tl.int64)
            qb10 = q + r10 * HEAD_DIM
            r11 = tl.minimum(base + 11, NUM_ROWS - 1).to(tl.int64)
            qb11 = q + r11 * HEAD_DIM
            r12 = tl.minimum(base + 12, NUM_ROWS - 1).to(tl.int64)
            qb12 = q + r12 * HEAD_DIM
            r13 = tl.minimum(base + 13, NUM_ROWS - 1).to(tl.int64)
            qb13 = q + r13 * HEAD_DIM
            r14 = tl.minimum(base + 14, NUM_ROWS - 1).to(tl.int64)
            qb14 = q + r14 * HEAD_DIM
            r15 = tl.minimum(base + 15, NUM_ROWS - 1).to(tl.int64)
            qb15 = q + r15 * HEAD_DIM
            x0 = tl.load(qb0 + offset).to(tl.float32)
            ev0 = tl.load(qb0 + NOPE_DIM + offset_even).to(tl.float32)
            od0 = tl.load(qb0 + NOPE_DIM + offset_odd).to(tl.float32)
            x1 = tl.load(qb1 + offset).to(tl.float32)
            ev1 = tl.load(qb1 + NOPE_DIM + offset_even).to(tl.float32)
            od1 = tl.load(qb1 + NOPE_DIM + offset_odd).to(tl.float32)
            x2 = tl.load(qb2 + offset).to(tl.float32)
            ev2 = tl.load(qb2 + NOPE_DIM + offset_even).to(tl.float32)
            od2 = tl.load(qb2 + NOPE_DIM + offset_odd).to(tl.float32)
            x3 = tl.load(qb3 + offset).to(tl.float32)
            ev3 = tl.load(qb3 + NOPE_DIM + offset_even).to(tl.float32)
            od3 = tl.load(qb3 + NOPE_DIM + offset_odd).to(tl.float32)
            x4 = tl.load(qb4 + offset).to(tl.float32)
            ev4 = tl.load(qb4 + NOPE_DIM + offset_even).to(tl.float32)
            od4 = tl.load(qb4 + NOPE_DIM + offset_odd).to(tl.float32)
            x5 = tl.load(qb5 + offset).to(tl.float32)
            ev5 = tl.load(qb5 + NOPE_DIM + offset_even).to(tl.float32)
            od5 = tl.load(qb5 + NOPE_DIM + offset_odd).to(tl.float32)
            x6 = tl.load(qb6 + offset).to(tl.float32)
            ev6 = tl.load(qb6 + NOPE_DIM + offset_even).to(tl.float32)
            od6 = tl.load(qb6 + NOPE_DIM + offset_odd).to(tl.float32)
            x7 = tl.load(qb7 + offset).to(tl.float32)
            ev7 = tl.load(qb7 + NOPE_DIM + offset_even).to(tl.float32)
            od7 = tl.load(qb7 + NOPE_DIM + offset_odd).to(tl.float32)
            x8 = tl.load(qb8 + offset).to(tl.float32)
            ev8 = tl.load(qb8 + NOPE_DIM + offset_even).to(tl.float32)
            od8 = tl.load(qb8 + NOPE_DIM + offset_odd).to(tl.float32)
            x9 = tl.load(qb9 + offset).to(tl.float32)
            ev9 = tl.load(qb9 + NOPE_DIM + offset_even).to(tl.float32)
            od9 = tl.load(qb9 + NOPE_DIM + offset_odd).to(tl.float32)
            x10 = tl.load(qb10 + offset).to(tl.float32)
            ev10 = tl.load(qb10 + NOPE_DIM + offset_even).to(tl.float32)
            od10 = tl.load(qb10 + NOPE_DIM + offset_odd).to(tl.float32)
            x11 = tl.load(qb11 + offset).to(tl.float32)
            ev11 = tl.load(qb11 + NOPE_DIM + offset_even).to(tl.float32)
            od11 = tl.load(qb11 + NOPE_DIM + offset_odd).to(tl.float32)
            x12 = tl.load(qb12 + offset).to(tl.float32)
            ev12 = tl.load(qb12 + NOPE_DIM + offset_even).to(tl.float32)
            od12 = tl.load(qb12 + NOPE_DIM + offset_odd).to(tl.float32)
            x13 = tl.load(qb13 + offset).to(tl.float32)
            ev13 = tl.load(qb13 + NOPE_DIM + offset_even).to(tl.float32)
            od13 = tl.load(qb13 + NOPE_DIM + offset_odd).to(tl.float32)
            x14 = tl.load(qb14 + offset).to(tl.float32)
            ev14 = tl.load(qb14 + NOPE_DIM + offset_even).to(tl.float32)
            od14 = tl.load(qb14 + NOPE_DIM + offset_odd).to(tl.float32)
            x15 = tl.load(qb15 + offset).to(tl.float32)
            ev15 = tl.load(qb15 + NOPE_DIM + offset_even).to(tl.float32)
            od15 = tl.load(qb15 + NOPE_DIM + offset_odd).to(tl.float32)
            var0 = tl.sum(x0 * x0) / HEAD_DIM
            rsq0 = tl.rsqrt(var0 + eps)
            var1 = tl.sum(x1 * x1) / HEAD_DIM
            rsq1 = tl.rsqrt(var1 + eps)
            var2 = tl.sum(x2 * x2) / HEAD_DIM
            rsq2 = tl.rsqrt(var2 + eps)
            var3 = tl.sum(x3 * x3) / HEAD_DIM
            rsq3 = tl.rsqrt(var3 + eps)
            var4 = tl.sum(x4 * x4) / HEAD_DIM
            rsq4 = tl.rsqrt(var4 + eps)
            var5 = tl.sum(x5 * x5) / HEAD_DIM
            rsq5 = tl.rsqrt(var5 + eps)
            var6 = tl.sum(x6 * x6) / HEAD_DIM
            rsq6 = tl.rsqrt(var6 + eps)
            var7 = tl.sum(x7 * x7) / HEAD_DIM
            rsq7 = tl.rsqrt(var7 + eps)
            var8 = tl.sum(x8 * x8) / HEAD_DIM
            rsq8 = tl.rsqrt(var8 + eps)
            var9 = tl.sum(x9 * x9) / HEAD_DIM
            rsq9 = tl.rsqrt(var9 + eps)
            var10 = tl.sum(x10 * x10) / HEAD_DIM
            rsq10 = tl.rsqrt(var10 + eps)
            var11 = tl.sum(x11 * x11) / HEAD_DIM
            rsq11 = tl.rsqrt(var11 + eps)
            var12 = tl.sum(x12 * x12) / HEAD_DIM
            rsq12 = tl.rsqrt(var12 + eps)
            var13 = tl.sum(x13 * x13) / HEAD_DIM
            rsq13 = tl.rsqrt(var13 + eps)
            var14 = tl.sum(x14 * x14) / HEAD_DIM
            rsq14 = tl.rsqrt(var14 + eps)
            var15 = tl.sum(x15 * x15) / HEAD_DIM
            rsq15 = tl.rsqrt(var15 + eps)
            if base + 0 < NUM_ROWS:
                tl.store(qb0 + offset, (x0 * rsq0).to(tl.bfloat16), mask=mask_nope)
                ne0 = (ev0 * rsq0 * cos_blk - od0 * rsq0 * sin_blk).to(tl.bfloat16)
                no0 = (ev0 * rsq0 * sin_blk + od0 * rsq0 * cos_blk).to(tl.bfloat16)
                tl.store(qb0 + NOPE_DIM + offset_even, ne0)
                tl.store(qb0 + NOPE_DIM + offset_odd, no0)
            if base + 1 < NUM_ROWS:
                tl.store(qb1 + offset, (x1 * rsq1).to(tl.bfloat16), mask=mask_nope)
                ne1 = (ev1 * rsq1 * cos_blk - od1 * rsq1 * sin_blk).to(tl.bfloat16)
                no1 = (ev1 * rsq1 * sin_blk + od1 * rsq1 * cos_blk).to(tl.bfloat16)
                tl.store(qb1 + NOPE_DIM + offset_even, ne1)
                tl.store(qb1 + NOPE_DIM + offset_odd, no1)
            if base + 2 < NUM_ROWS:
                tl.store(qb2 + offset, (x2 * rsq2).to(tl.bfloat16), mask=mask_nope)
                ne2 = (ev2 * rsq2 * cos_blk - od2 * rsq2 * sin_blk).to(tl.bfloat16)
                no2 = (ev2 * rsq2 * sin_blk + od2 * rsq2 * cos_blk).to(tl.bfloat16)
                tl.store(qb2 + NOPE_DIM + offset_even, ne2)
                tl.store(qb2 + NOPE_DIM + offset_odd, no2)
            if base + 3 < NUM_ROWS:
                tl.store(qb3 + offset, (x3 * rsq3).to(tl.bfloat16), mask=mask_nope)
                ne3 = (ev3 * rsq3 * cos_blk - od3 * rsq3 * sin_blk).to(tl.bfloat16)
                no3 = (ev3 * rsq3 * sin_blk + od3 * rsq3 * cos_blk).to(tl.bfloat16)
                tl.store(qb3 + NOPE_DIM + offset_even, ne3)
                tl.store(qb3 + NOPE_DIM + offset_odd, no3)
            if base + 4 < NUM_ROWS:
                tl.store(qb4 + offset, (x4 * rsq4).to(tl.bfloat16), mask=mask_nope)
                ne4 = (ev4 * rsq4 * cos_blk - od4 * rsq4 * sin_blk).to(tl.bfloat16)
                no4 = (ev4 * rsq4 * sin_blk + od4 * rsq4 * cos_blk).to(tl.bfloat16)
                tl.store(qb4 + NOPE_DIM + offset_even, ne4)
                tl.store(qb4 + NOPE_DIM + offset_odd, no4)
            if base + 5 < NUM_ROWS:
                tl.store(qb5 + offset, (x5 * rsq5).to(tl.bfloat16), mask=mask_nope)
                ne5 = (ev5 * rsq5 * cos_blk - od5 * rsq5 * sin_blk).to(tl.bfloat16)
                no5 = (ev5 * rsq5 * sin_blk + od5 * rsq5 * cos_blk).to(tl.bfloat16)
                tl.store(qb5 + NOPE_DIM + offset_even, ne5)
                tl.store(qb5 + NOPE_DIM + offset_odd, no5)
            if base + 6 < NUM_ROWS:
                tl.store(qb6 + offset, (x6 * rsq6).to(tl.bfloat16), mask=mask_nope)
                ne6 = (ev6 * rsq6 * cos_blk - od6 * rsq6 * sin_blk).to(tl.bfloat16)
                no6 = (ev6 * rsq6 * sin_blk + od6 * rsq6 * cos_blk).to(tl.bfloat16)
                tl.store(qb6 + NOPE_DIM + offset_even, ne6)
                tl.store(qb6 + NOPE_DIM + offset_odd, no6)
            if base + 7 < NUM_ROWS:
                tl.store(qb7 + offset, (x7 * rsq7).to(tl.bfloat16), mask=mask_nope)
                ne7 = (ev7 * rsq7 * cos_blk - od7 * rsq7 * sin_blk).to(tl.bfloat16)
                no7 = (ev7 * rsq7 * sin_blk + od7 * rsq7 * cos_blk).to(tl.bfloat16)
                tl.store(qb7 + NOPE_DIM + offset_even, ne7)
                tl.store(qb7 + NOPE_DIM + offset_odd, no7)
            if base + 8 < NUM_ROWS:
                tl.store(qb8 + offset, (x8 * rsq8).to(tl.bfloat16), mask=mask_nope)
                ne8 = (ev8 * rsq8 * cos_blk - od8 * rsq8 * sin_blk).to(tl.bfloat16)
                no8 = (ev8 * rsq8 * sin_blk + od8 * rsq8 * cos_blk).to(tl.bfloat16)
                tl.store(qb8 + NOPE_DIM + offset_even, ne8)
                tl.store(qb8 + NOPE_DIM + offset_odd, no8)
            if base + 9 < NUM_ROWS:
                tl.store(qb9 + offset, (x9 * rsq9).to(tl.bfloat16), mask=mask_nope)
                ne9 = (ev9 * rsq9 * cos_blk - od9 * rsq9 * sin_blk).to(tl.bfloat16)
                no9 = (ev9 * rsq9 * sin_blk + od9 * rsq9 * cos_blk).to(tl.bfloat16)
                tl.store(qb9 + NOPE_DIM + offset_even, ne9)
                tl.store(qb9 + NOPE_DIM + offset_odd, no9)
            if base + 10 < NUM_ROWS:
                tl.store(qb10 + offset, (x10 * rsq10).to(tl.bfloat16), mask=mask_nope)
                ne10 = (ev10 * rsq10 * cos_blk - od10 * rsq10 * sin_blk).to(tl.bfloat16)
                no10 = (ev10 * rsq10 * sin_blk + od10 * rsq10 * cos_blk).to(tl.bfloat16)
                tl.store(qb10 + NOPE_DIM + offset_even, ne10)
                tl.store(qb10 + NOPE_DIM + offset_odd, no10)
            if base + 11 < NUM_ROWS:
                tl.store(qb11 + offset, (x11 * rsq11).to(tl.bfloat16), mask=mask_nope)
                ne11 = (ev11 * rsq11 * cos_blk - od11 * rsq11 * sin_blk).to(tl.bfloat16)
                no11 = (ev11 * rsq11 * sin_blk + od11 * rsq11 * cos_blk).to(tl.bfloat16)
                tl.store(qb11 + NOPE_DIM + offset_even, ne11)
                tl.store(qb11 + NOPE_DIM + offset_odd, no11)
            if base + 12 < NUM_ROWS:
                tl.store(qb12 + offset, (x12 * rsq12).to(tl.bfloat16), mask=mask_nope)
                ne12 = (ev12 * rsq12 * cos_blk - od12 * rsq12 * sin_blk).to(tl.bfloat16)
                no12 = (ev12 * rsq12 * sin_blk + od12 * rsq12 * cos_blk).to(tl.bfloat16)
                tl.store(qb12 + NOPE_DIM + offset_even, ne12)
                tl.store(qb12 + NOPE_DIM + offset_odd, no12)
            if base + 13 < NUM_ROWS:
                tl.store(qb13 + offset, (x13 * rsq13).to(tl.bfloat16), mask=mask_nope)
                ne13 = (ev13 * rsq13 * cos_blk - od13 * rsq13 * sin_blk).to(tl.bfloat16)
                no13 = (ev13 * rsq13 * sin_blk + od13 * rsq13 * cos_blk).to(tl.bfloat16)
                tl.store(qb13 + NOPE_DIM + offset_even, ne13)
                tl.store(qb13 + NOPE_DIM + offset_odd, no13)
            if base + 14 < NUM_ROWS:
                tl.store(qb14 + offset, (x14 * rsq14).to(tl.bfloat16), mask=mask_nope)
                ne14 = (ev14 * rsq14 * cos_blk - od14 * rsq14 * sin_blk).to(tl.bfloat16)
                no14 = (ev14 * rsq14 * sin_blk + od14 * rsq14 * cos_blk).to(tl.bfloat16)
                tl.store(qb14 + NOPE_DIM + offset_even, ne14)
                tl.store(qb14 + NOPE_DIM + offset_odd, no14)
            if base + 15 < NUM_ROWS:
                tl.store(qb15 + offset, (x15 * rsq15).to(tl.bfloat16), mask=mask_nope)
                ne15 = (ev15 * rsq15 * cos_blk - od15 * rsq15 * sin_blk).to(tl.bfloat16)
                no15 = (ev15 * rsq15 * sin_blk + od15 * rsq15 * cos_blk).to(tl.bfloat16)
                tl.store(qb15 + NOPE_DIM + offset_even, ne15)
                tl.store(qb15 + NOPE_DIM + offset_odd, no15)
    else:
        for row in range(pid, NUM_ROWS, num_programs):
            row64 = row.to(tl.int64)
            q_base = q + row64 * HEAD_DIM
            x = tl.load(q_base + offset).to(tl.float32)
            ev = tl.load(q_base + NOPE_DIM + offset_even).to(tl.float32)
            od = tl.load(q_base + NOPE_DIM + offset_odd).to(tl.float32)
            tok = row64 // num_heads
            position_id = tl.load(position_ids + tok).to(tl.int64)
            cs_base = cos_sin_cache + position_id * ROPE_DIM
            cos_blk = tl.load(cs_base + offset_half_rope)
            sin_blk = tl.load(cs_base + offset_half_rope + HALF_ROPE_DIM)
            variance = tl.sum(x * x) / HEAD_DIM
            rsqrt = tl.rsqrt(variance + eps)
            tl.store(q_base + offset, (x * rsqrt).to(tl.bfloat16), mask=mask_nope)
            ne = (ev * rsqrt * cos_blk - od * rsqrt * sin_blk).to(tl.bfloat16)
            no = (ev * rsqrt * sin_blk + od * rsqrt * cos_blk).to(tl.bfloat16)
            tl.store(q_base + NOPE_DIM + offset_even, ne)
            tl.store(q_base + NOPE_DIM + offset_odd, no)

    # ---- stage 2: kv rope + UE8M0 fp8 quant + paged insert ----
    for token_idx in range(pid, num_tokens_insert, num_programs):
        token64 = token_idx.to(tl.int64)
        kv_base = kv + token64 * HEAD_DIM
        # all loads first: rope pairs, the 7 quant groups, slot, position,
        # cos/sin -- no store may sit between them (alias serialization)
        ev = tl.load(kv_base + NOPE_DIM + offset_even)
        od = tl.load(kv_base + NOPE_DIM + offset_odd)
        q0 = tl.load(kv_base + 0 * QUANT_BLOCK + offset_quant)
        q1 = tl.load(kv_base + 1 * QUANT_BLOCK + offset_quant)
        q2 = tl.load(kv_base + 2 * QUANT_BLOCK + offset_quant)
        q3 = tl.load(kv_base + 3 * QUANT_BLOCK + offset_quant)
        q4 = tl.load(kv_base + 4 * QUANT_BLOCK + offset_quant)
        q5 = tl.load(kv_base + 5 * QUANT_BLOCK + offset_quant)
        q6 = tl.load(kv_base + 6 * QUANT_BLOCK + offset_quant)
        position_id = tl.load(position_ids + token64).to(tl.int64)
        cs_base = cos_sin_cache + position_id * ROPE_DIM
        cos_blk = tl.load(cs_base + offset_half_rope)
        sin_blk = tl.load(cs_base + offset_half_rope + HALF_ROPE_DIM)
        slot_id = tl.load(slot_mapping + token64)
        # rope (f32 math on bf16 loads)
        evf = ev.to(tl.float32)
        odf = od.to(tl.float32)
        ne = (evf * cos_blk - odf * sin_blk).to(tl.bfloat16)
        no = (evf * sin_blk + odf * cos_blk).to(tl.bfloat16)
        if slot_id >= 0:
            slot64 = slot_id.to(tl.int64)
            block_idx = slot64 // cache_block_size
            pos_in_block = slot64 % cache_block_size
            # Fold every scalar offset term into single i64 expressions:
            # element offsets >= 2**31 wrap in i32 addptr chains on this
            # backend, and KV caches larger than 2 GiB are routine.
            token_base = block_idx * kv_block_stride + pos_in_block * TOKEN_DATA_BYTES
            token_fp8_ptr = k_cache + token_base
            token_bf16_ptr = (k_cache + token_base + NOPE_DIM).to(
                tl.pointer_type(tl.bfloat16)
            )
            token_scale_ptr = k_cache + (
                block_idx * kv_block_stride
                + cache_block_size * TOKEN_DATA_BYTES
                + pos_in_block * SCALE_BYTES_PER_TOKEN
            )
            tl.store(token_bf16_ptr + offset_even, ne)
            tl.store(token_bf16_ptr + offset_odd, no)
            xf0 = q0.to(tl.float32)
            amax0 = tl.maximum(tl.max(tl.abs(xf0), axis=0), 1e-4)
            raw_scale0 = amax0 / FP8_MAX
            scale_bits0 = raw_scale0.to(tl.int32, bitcast=True)
            ceil_log2_0 = (
                (scale_bits0 >> 23)
                - 127
                + tl.where((scale_bits0 & 0x7FFFFF) != 0, 1, 0)
            )
            inv_scale0 = ((127 - ceil_log2_0) << 23).to(tl.float32, bitcast=True)
            xf1 = q1.to(tl.float32)
            amax1 = tl.maximum(tl.max(tl.abs(xf1), axis=0), 1e-4)
            raw_scale1 = amax1 / FP8_MAX
            scale_bits1 = raw_scale1.to(tl.int32, bitcast=True)
            ceil_log2_1 = (
                (scale_bits1 >> 23)
                - 127
                + tl.where((scale_bits1 & 0x7FFFFF) != 0, 1, 0)
            )
            inv_scale1 = ((127 - ceil_log2_1) << 23).to(tl.float32, bitcast=True)
            xf2 = q2.to(tl.float32)
            amax2 = tl.maximum(tl.max(tl.abs(xf2), axis=0), 1e-4)
            raw_scale2 = amax2 / FP8_MAX
            scale_bits2 = raw_scale2.to(tl.int32, bitcast=True)
            ceil_log2_2 = (
                (scale_bits2 >> 23)
                - 127
                + tl.where((scale_bits2 & 0x7FFFFF) != 0, 1, 0)
            )
            inv_scale2 = ((127 - ceil_log2_2) << 23).to(tl.float32, bitcast=True)
            xf3 = q3.to(tl.float32)
            amax3 = tl.maximum(tl.max(tl.abs(xf3), axis=0), 1e-4)
            raw_scale3 = amax3 / FP8_MAX
            scale_bits3 = raw_scale3.to(tl.int32, bitcast=True)
            ceil_log2_3 = (
                (scale_bits3 >> 23)
                - 127
                + tl.where((scale_bits3 & 0x7FFFFF) != 0, 1, 0)
            )
            inv_scale3 = ((127 - ceil_log2_3) << 23).to(tl.float32, bitcast=True)
            xf4 = q4.to(tl.float32)
            amax4 = tl.maximum(tl.max(tl.abs(xf4), axis=0), 1e-4)
            raw_scale4 = amax4 / FP8_MAX
            scale_bits4 = raw_scale4.to(tl.int32, bitcast=True)
            ceil_log2_4 = (
                (scale_bits4 >> 23)
                - 127
                + tl.where((scale_bits4 & 0x7FFFFF) != 0, 1, 0)
            )
            inv_scale4 = ((127 - ceil_log2_4) << 23).to(tl.float32, bitcast=True)
            xf5 = q5.to(tl.float32)
            amax5 = tl.maximum(tl.max(tl.abs(xf5), axis=0), 1e-4)
            raw_scale5 = amax5 / FP8_MAX
            scale_bits5 = raw_scale5.to(tl.int32, bitcast=True)
            ceil_log2_5 = (
                (scale_bits5 >> 23)
                - 127
                + tl.where((scale_bits5 & 0x7FFFFF) != 0, 1, 0)
            )
            inv_scale5 = ((127 - ceil_log2_5) << 23).to(tl.float32, bitcast=True)
            xf6 = q6.to(tl.float32)
            amax6 = tl.maximum(tl.max(tl.abs(xf6), axis=0), 1e-4)
            raw_scale6 = amax6 / FP8_MAX
            scale_bits6 = raw_scale6.to(tl.int32, bitcast=True)
            ceil_log2_6 = (
                (scale_bits6 >> 23)
                - 127
                + tl.where((scale_bits6 & 0x7FFFFF) != 0, 1, 0)
            )
            inv_scale6 = ((127 - ceil_log2_6) << 23).to(tl.float32, bitcast=True)
            xs0 = xf0 * inv_scale0
            xc0 = tl.clamp(xs0, -FP8_MAX, FP8_MAX)
            xq0 = xc0.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            enc0 = tl.minimum(tl.maximum(ceil_log2_0 + 127, 0), 255)
            xs1 = xf1 * inv_scale1
            xc1 = tl.clamp(xs1, -FP8_MAX, FP8_MAX)
            xq1 = xc1.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            enc1 = tl.minimum(tl.maximum(ceil_log2_1 + 127, 0), 255)
            xs2 = xf2 * inv_scale2
            xc2 = tl.clamp(xs2, -FP8_MAX, FP8_MAX)
            xq2 = xc2.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            enc2 = tl.minimum(tl.maximum(ceil_log2_2 + 127, 0), 255)
            xs3 = xf3 * inv_scale3
            xc3 = tl.clamp(xs3, -FP8_MAX, FP8_MAX)
            xq3 = xc3.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            enc3 = tl.minimum(tl.maximum(ceil_log2_3 + 127, 0), 255)
            xs4 = xf4 * inv_scale4
            xc4 = tl.clamp(xs4, -FP8_MAX, FP8_MAX)
            xq4 = xc4.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            enc4 = tl.minimum(tl.maximum(ceil_log2_4 + 127, 0), 255)
            xs5 = xf5 * inv_scale5
            xc5 = tl.clamp(xs5, -FP8_MAX, FP8_MAX)
            xq5 = xc5.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            enc5 = tl.minimum(tl.maximum(ceil_log2_5 + 127, 0), 255)
            xs6 = xf6 * inv_scale6
            xc6 = tl.clamp(xs6, -FP8_MAX, FP8_MAX)
            xq6 = xc6.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            enc6 = tl.minimum(tl.maximum(ceil_log2_6 + 127, 0), 255)
            tl.store(token_fp8_ptr + 0 * QUANT_BLOCK + offset_quant, xq0)
            tl.store(token_fp8_ptr + 1 * QUANT_BLOCK + offset_quant, xq1)
            tl.store(token_fp8_ptr + 2 * QUANT_BLOCK + offset_quant, xq2)
            tl.store(token_fp8_ptr + 3 * QUANT_BLOCK + offset_quant, xq3)
            tl.store(token_fp8_ptr + 4 * QUANT_BLOCK + offset_quant, xq4)
            tl.store(token_fp8_ptr + 5 * QUANT_BLOCK + offset_quant, xq5)
            tl.store(token_fp8_ptr + 6 * QUANT_BLOCK + offset_quant, xq6)
            # pack the 8 scale bytes (7 encoded + 1 zero pad) into one
            # uint64 store; the scale row is 8-byte aligned by layout
            # (block strides are multiples of 576*8).
            scale_word = enc0.to(tl.uint64)
            scale_word |= enc1.to(tl.uint64) << 8
            scale_word |= enc2.to(tl.uint64) << 16
            scale_word |= enc3.to(tl.uint64) << 24
            scale_word |= enc4.to(tl.uint64) << 32
            scale_word |= enc5.to(tl.uint64) << 40
            scale_word |= enc6.to(tl.uint64) << 48
            tl.store(token_scale_ptr.to(tl.pointer_type(tl.uint64)), scale_word)


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    cache_block_size: int,
):
    """
    Horizontally-fused DeepseekV4-MLA: per-head RMSNorm + GPT-J RoPE for Q, and
    GPT-J RoPE + UE8M0 FP8 quant + paged cache insert for KV, all in one kernel
    launch.
    K Cache block layout (block_size=64 tokens):
    - First 64 * 576 = 36864 bytes: Token data
      - Each token: 448 bytes (fp8) + 128 bytes (bf16)
    - Next 64 * 8 = 512 bytes: Scales
      - Each token: 8 bytes (uint8 scales, 7 real + 1 padding)
    - Padded to multiple of 576

    Args:
        q: [num_tokens, num_heads, 512], bfloat16, in place
        kv: [num_tokens, 512], bfloat16, read-only
        k_cache: [num_blocks, block_bytes], uint8
        slot_mapping: [num_tokens_insert], i64 (i32 also accepted; narrowed
            to i32 for the kernel on the GCU)
        position_ids: [num_tokens], i64 (i32 also accepted; narrowed to i32
            for the kernel on the GCU)
        cos_sin_cache: [max_pos, 64], fp32
        eps: used in RMSNorm
        cache_block_size: tokens per paged-cache block
    """
    assert q.is_contiguous() and kv.is_contiguous()
    num_tokens, num_heads, head_dims = q.shape
    assert head_dims == 512
    assert kv.shape == (num_tokens, 512)
    assert q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    assert k_cache.dtype == torch.uint8
    assert slot_mapping.dim() == 1
    num_tokens_insert = slot_mapping.shape[0]
    assert num_tokens_insert <= num_tokens
    assert slot_mapping.dtype in (torch.int64, torch.int32)
    assert position_ids.shape == (num_tokens,)
    assert position_ids.dtype in (torch.int64, torch.int32)
    assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == 64
    assert cos_sin_cache.dtype == torch.float32

    # torch_gcu silently narrows int64 tensors to 4-byte physical slots
    # ("GCU not support Long use Int replace"); a Triton kernel reading them
    # at the claimed int64 element width gets interleaved garbage slot ids /
    # positions and then writes out of bounds (device abort). Normalize to
    # real int32 tensors on the host -- slot ids and positions live well
    # below 2**31 and the ATen cast is value-exact.
    if slot_mapping.dtype == torch.int64:
        slot_mapping = slot_mapping.to(torch.int32)
    if position_ids.dtype == torch.int64:
        position_ids = position_ids.to(torch.int32)

    total_work = num_tokens * num_heads + num_tokens_insert
    grid = (min(_GRID_TARGET, total_work),)
    fused_qnorm_rope_kv_insert_kernel[grid](
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        cache_block_size,
        num_tokens,
        num_heads,
        k_cache.stride(0),
        num_tokens_insert,
        num_heads % _Q_GROUP == 0,
        num_warps=1,
        num_stages=2,
    )
