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

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import HAS_TLE

__all__ = ["flash_mla_sparse_fwd_w8a8"]

if not HAS_TLE:
    sparse_attention = None
else:
    import triton
    import triton.experimental.tle.language as tle
    import triton.language as tl
    import triton.language.core as tlc

    from flaggems_vllm import runtime
    from flaggems_vllm.ops.flash_mla import _get_num_sms as get_num_sms
    from flaggems_vllm.ops.flash_mla_with_kvcache_model1 import (
        _cache_views as cache_views,
    )
    from flaggems_vllm.ops.flash_mla_with_kvcache_model1 import tle_subslice
    from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import (
        cp_async_cg_16,
        cp_async_cg_16_pred,
        cp_async_wait_all,
        fence_proxy_async_shared,
        publish_p_fp8_sw64_asm,
        publish_p_fp8_sw64_constraints,
        publish_p_fp8_sw64_coupled_stmatrix,
        vtranspose_fp8_64x128_coupled_asm,
    )
    from flaggems_vllm.utils import libentry, libtuner
    from flaggems_vllm.utils.libentry import LibEntry

    ACCUMULATOR_SCALE_FLOOR = tl.constexpr(2.0**-32)

    QK_RECOMPUTE_THRESHOLD = tl.constexpr(2.0**-14)

    @tlc.builtin
    def sparse_named_barriers(count, threads, base, _semantic=None):
        barriers = tle.gpu.alloc_barriers(
            count, arrive_count=threads, _semantic=_semantic
        )
        # Lazy IDs are not unique across JIT helpers; reserve them before capture.
        base = int(tlc._unwrap_if_constexpr(base))
        barriers.named_base_id = base
        barriers.type.named_base_id = base
        return barriers

    @triton.jit
    def sink_normalization(inverse, sink, logsum):
        return inverse * tl.where(
            sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
        )

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

    @triton.jit
    def sparse_fp8_sub(lhs, rhs):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [lhs, rhs],
            dtype=tl.float32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def sparse_fp8_transpose_values(
        s_src,
        s_dst,
        dst_row: tl.constexpr,
        warp_offset: tl.constexpr,
    ):
        """Transpose with coupled K permutation and bank-distributed output rows."""
        carrier = tl.arange(0, 128).to(tl.uint32)
        src_base = tle.gpu.local_ptr(s_src, (0, 0))
        dst_base = tle.gpu.local_ptr(s_dst, (dst_row, 0))
        return tl.inline_asm_elementwise(
            asm=vtranspose_fp8_64x128_coupled_asm(),
            constraints="=r,r,r,r,r",
            args=[carrier, src_base, dst_base, tl.full((), warp_offset, tl.uint32)],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def sparse_fp8_split_blocks(length, SPLITS: tl.constexpr):
        blocks = tl.cdiv(length, 64)
        if SPLITS == 1:
            first_block = 0
            split_blocks = blocks
        else:
            blocks_per_split = tl.cdiv(blocks, SPLITS)
            first_block = tl.program_id(2) * blocks_per_split
            split_blocks = tl.maximum(
                0, tl.minimum(blocks_per_split, blocks - first_block)
            )
        return first_block, split_blocks

    @triton.constexpr_function
    def sparse_fp8_publish_output_asm(streaming: bool) -> str:
        st_op = "st.global.cs.v2.b32" if streaming else "st.global.v2.b32"
        lines = [
            "{",
            ".reg .b32 tid, row, col, offset, a, b, c, d;",
            ".reg .b64 addr;",
            "mov.u32 tid, %tid.x;",
            "and.b32 tid, tid, 127;",
            "shr.u32 row, tid, 5;",
            "shl.b32 row, row, 4;",
            "and.b32 col, tid, 31;",
            "shr.u32 col, col, 2;",
            "add.u32 row, row, col;",
            "and.b32 col, tid, 3;",
            "shl.b32 col, col, 3;",
            "shl.b32 row, row, 10;",
            "add.u32 offset, row, col;",
            "cvt.u64.u32 addr, offset;",
            "add.u64 addr, addr, $128;",
        ]
        for group in range(8):
            k = 64 + group * 8
            for j, reg in enumerate(("a", "b", "c", "d")):
                lines.append(f"cvt.rn.bf16x2.f32 {reg}, ${k + 4 + j}, ${k + j};")
            lines.append(f"{st_op} [addr+{group * 32}], {{a, b}};")
            lines.append(f"{st_op} [addr+{8192 + group * 32}], {{c, d}};")
        lines.extend(["mov.u32 $0, 0;", "}"])
        return "\n".join(lines) + "\n"

    SPARSE_FP8_PUBLISH_CONSTRAINTS = tl.constexpr(
        ",".join(["=r"] * 64 + ["f"] * 64 + ["l"] * 64)
    )

    @triton.jit
    def sparse_fp8_publish_output(
        acc,
        inverse,
        Output,
        batch,
        head_block,
        H: tl.constexpr,
        OFFSET: tl.constexpr,
        STREAMING: tl.constexpr = False,
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
            asm=sparse_fp8_publish_output_asm(STREAMING),
            constraints=SPARSE_FP8_PUBLISH_CONSTRAINTS,
            args=[acc * inverse[:, None], base_u64],
            dtype=tl.uint32,
            is_pure=False,
            pack=64,
        )

    @triton.constexpr_function
    def p128_publication_asm(asm, prequantized):
        input_registers = 8 if prequantized else 32
        first_base = 64 + input_registers * 2
        halves = []
        for half in range(2):
            block = asm.replace(
                f"${32 + input_registers}", f"${first_base + half * 64}"
            )
            for operand in range(31 + input_registers, 31, -1):
                block = block.replace(
                    f"${operand}", f"${operand + 32 + half * input_registers}"
                )
            halves.append(block)
        return "".join(halves)

    @triton.jit
    def publish_sparse_p128_fp8(s_p, p, s_p1, PREQUANTIZED: tl.constexpr = False):
        base = tle.gpu.local_ptr(s_p, (0, 0))
        base_u32 = tl.inline_asm_elementwise(
            "mov.u32 $0, $1;", "=r,r", [base], dtype=tl.uint32, is_pure=True, pack=1
        )
        base1 = tle.gpu.local_ptr(s_p1, (0, 0))
        return tl.inline_asm_elementwise(
            p128_publication_asm(publish_p_fp8_sw64_asm(PREQUANTIZED), PREQUANTIZED),
            publish_p_fp8_sw64_constraints(2, PREQUANTIZED),
            [p, base_u32, base1.to(tl.uint64).to(tl.uint32)],
            dtype=tl.uint32,
            is_pure=False,
            pack=64,
        )

    @triton.jit
    def quantize_p_fp8_before_ready(p):
        return tl.inline_asm_elementwise(
            "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; "
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi}; }",
            "=r,f,f,f,f",
            [p],
            dtype=tl.float8e4nv,
            is_pure=False,
            pack=4,
        )

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
                copy_columns = tl.broadcast_to(
                    (tl.arange(0, 8) * 16)[None, :], (128, 8)
                )
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
                        cp_async_cg_16_pred(
                            dst,
                            src,
                            tl.where(valid[:, None], 16, 0),
                            ((copy_rows // 64) == key_half).to(tl.uint32),
                        )
                    # Consumer 0 releases all RoPE rows with kfree[0].
                    if key_half == 0 and ASYNC_KR:
                        rope_columns = tl.broadcast_to(
                            (tl.arange(0, 8) * 8)[None, :], (128, 8)
                        )
                        dst_r = tle.gpu.local_ptr(
                            skr.slot(buf), (copy_rows, rope_columns)
                        )
                        src_r = (
                            KR
                            + pages[:, None] * krp
                            + slots[:, None] * krt
                            + rope_columns
                        )
                        cp_async_cg_16(
                            dst_r,
                            src_r,
                            tl.where(valid[:, None], 16, 0).to(tl.uint32),
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
                cp_async_wait_all(tl.arange(0, PRODUCER_WARPS * 32).to(tl.uint32))
            else:
                pass
            # Matrix transpose avoids byte stores and their shared-memory bank conflicts.
            tl.debug_barrier()
            fence_proxy_async_shared(tl.arange(0, 128))
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
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
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
            scores = (
                logits * query_scale[:, None] * kv_scale[None, :] + add_mask[None, :]
            )
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
            head_repair |= (
                next_weight_scale * ACCUMULATOR_SCALE_FLOOR > probability_scale
            )
            if PREQUANTIZE_P:
                p = quantize_p_fp8_before_ready(weighted / probability_scale[:, None])
            else:
                p = weighted / probability_scale[:, None]
            beta = correction * previous_scale / probability_scale
            tle.gpu.barrier_wait(vfull[0])
            tle.gpu.barrier_wait(vfull[1])
            publish_sparse_p128_fp8(
                sp.slot(0), p, sp.slot(1), PREQUANTIZED=PREQUANTIZE_P
            )
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
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
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
                    fence_proxy_async_shared(tl.arange(0, 128))
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
        PREQUANTIZE_P: tl.constexpr = (
            HAS_ROPE and N * (512 + 64 * 2 + 4) > 50 * 1024 * 1024
        )
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
                q_ptr = (
                    Q + batch * stride_qb + heads[:, None] * stride_qh + dims[None, :]
                )
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
                    logits = (
                        logits * query_scale[:, None] * kv_scale[None, :] * SM_SCALE
                    )
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
                    denominator_inverse = tl.where(
                        denominator > 0, 1.0 / denominator, 0.0
                    )
                    if HAS_SINK:
                        sink = tl.load(Sink + heads, heads < H, 0)
                        inverse = sink_normalization(denominator_inverse, sink, logsum)
                    else:
                        inverse = denominator_inverse
                    output_ptr = (
                        Output + (batch * H + heads[:, None]) * 512 + dims[None, :]
                    )
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
                        old_delta = sparse_fp8_sub(maximum, new_maximum)
                        score_delta = sparse_fp8_sub(score, new_maximum)
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

    @triton.jit
    def split_block_range(split, NUM_BLOCKS, SPLITS: tl.constexpr):
        if SPLITS == 1:
            return 0, NUM_BLOCKS
        base = NUM_BLOCKS // SPLITS
        rem = NUM_BLOCKS - base * SPLITS
        first = split * base + tl.minimum(split, rem)
        count = base + tl.where(split < rem, 1, 0)
        return first, count

    @triton.jit
    def model1_block_info(
        block,
        split,
        BLOCKS: tl.constexpr,
        S: tl.constexpr,
        PREFIX: tl.constexpr,
        TAIL: tl.constexpr,
        TAIL_SPLITS: tl.constexpr,
    ):
        if TAIL_SPLITS:
            # Only the final, underfilled SM wave shares each query across CTAs.
            index = tl.maximum(block - PREFIX, 0)
            start, count = split_block_range(index % TAIL_SPLITS, BLOCKS, TAIL_SPLITS)
            tail = block >= PREFIX
            return (
                tl.where(tail, PREFIX + index // TAIL_SPLITS, block),
                tl.where(tail, block + TAIL, block),
                tl.where(tail, start, 0),
                tl.where(tail, count, BLOCKS),
            )
        else:
            start, count = split_block_range(split, BLOCKS, S)
            return block, block * S + split, start, count

    @triton.jit
    def scaled_bytes(quantized, ratios, R: tl.constexpr):
        factors = tl.broadcast_to(ratios[:, :, None], (R, 8, 64)).reshape(R, 512)
        return (quantized.to(tl.float32) * factors).to(tl.float8e4nv)

    @triton.jit
    def normalize_query(Q, QS, Norm, Magnitude, H: tl.constexpr):
        b, tile = tl.program_id(0), tl.program_id(1)
        rows = b * H + tile * 4 + tl.arange(0, 4)
        g, d = tl.arange(0, 8), tl.arange(0, 512)
        scales = tl.load(QS + rows[:, None] * 7 + g[None, :], g[None, :] < 7, 0.0)
        mx = tl.max(scales, 1)
        mn = tl.min(tl.where(scales > 0.0, scales, float("inf")), 1)
        q = tl.load(Q + rows[:, None] * 448 + d[None, :], d[None, :] < 448, 0.0)
        qnorm = scaled_bytes(q, scales / tl.where(mx > 0.0, mx, 1.0)[:, None], 4)
        tl.store(Norm + rows[:, None] * 448 + d[None, :], qnorm, d[None, :] < 448)
        tl.store(Magnitude + rows, tl.where(mx > 4.0 * mn, -mx, mx))

    @triton.constexpr_function
    def rescale_fp8_registers(tmp="tmp"):
        return "".join(
            f"mov.b32 {{xlo,xhi}},{reg}; "
            f"cvt.rn.f16x2.e4m3x2 {tmp},xlo; cvt.rn.f16x2.e4m3x2 u,xhi; "
            f"mul.f16x2 {tmp},{tmp},fac; mul.f16x2 u,u,fac; "
            f"cvt.rn.satfinite.e4m3x2.f16x2 xlo,{tmp}; cvt.rn.satfinite.e4m3x2.f16x2 xhi,u; "
            f"mov.b32 {reg},{{xlo,xhi}};\n"
            for reg in ("a", "b", "c", "d")
        )

    @triton.constexpr_function
    def ptx_exponent_minmax(e0, e1, lo, hi, ex):
        lines = [f"and.b32 {hi},{e0},255; mov.u32 {lo},{hi};"]
        for g in range(1, 7):
            word = e0 if g < 4 else e1
            lines.append(
                f"bfe.u32 {ex},{word},{(g % 4) * 8},8; "
                f"min.u32 {lo},{lo},{ex}; max.u32 {hi},{hi},{ex};"
            )
        return "\n".join(lines) + "\n"

    @triton.constexpr_function
    def ptx_encode_e8m0_scale(hi, ok_pred, out_reg, first_pred="first"):
        return (
            f"shl.b32 {out_reg},{hi},23;\n"
            f"setp.eq.u32 {first_pred},{hi},0;\n"
            f"@{first_pred} mov.u32 {out_reg},0x00400000;\n"
            f"@!{ok_pred} mov.u32 {out_reg},0xbf800000;\n"
        )

    @triton.constexpr_function
    def ptx_rescale_factor(delta, fac, sub_op="sub.u32"):
        return (
            f"min.u32 {delta},{delta},15;\n"
            f"{sub_op} {fac},15,{delta};\n"
            f"shl.b32 {fac},{fac},10;\n"
            f"prmt.b32 {fac},{fac},{fac},0x1010;\n"
        )

    @triton.constexpr_function
    def gather_topk128_asm(page_stride):
        # Four lanes copy a 64-byte feature group. Each group of four lanes
        # follows two indices, 64 tokens apart, and keeps both addresses live.
        asm = """
        {
        .reg .b32 tid, lane, warp, row, chunk, offset, swizzle, base;
        .reg .b32 id<2>, exp_lo<2>, exp_hi<2>, largest<2>, smallest<2>, size<2>;
        .reg .b32 tmp, u, exponent, slot, delta, fac, kdst, rdst, flag, a, b, c, d;
        .reg .b64 idx_ptr, src<2>, page<2>, meta, byte_offset;
        .reg .b16 xlo, xhi;
        .reg .pred valid<2>, first, skip, wide;
        mov.u32 tid, %tid.x;
        and.b32 tid, tid, 255;
        and.b32 lane, tid, 31;
        shr.u32 warp, tid, 5;
        and.b32 chunk, lane, 3;
        shr.u32 row, lane, 2;
        mad.lo.u32 row, warp, 8, row;
        mul.wide.u32 idx_ptr, row, 4;
        add.u64 idx_ptr, idx_ptr, $3;
        ld.global.L1::evict_first.s32 id0, [idx_ptr];
        ld.global.L1::evict_first.s32 id1, [idx_ptr+256];
        mov.u32 flag, 0;
        and.b32 swizzle, row, 7;
        shl.b32 swizzle, swizzle, 4;
        shl.b32 offset, chunk, 4;
        xor.b32 base, swizzle, offset;
        mad.lo.u32 base, row, 128, base;
        """
        for half in range(2):
            asm += f"""
            setp.lt.u32 valid{half}, id{half}, $7;
            @!valid{half} mov.u32 id{half}, 0;
            selp.u32 size{half}, 16, 0, valid{half};
            shr.u32 exponent, id{half}, 6;
            mul.wide.u32 page{half}, exponent, {page_stride};
            add.u64 page{half}, page{half}, $2;
            and.b32 slot, id{half}, 63;
            mad.wide.u32 meta, slot, 8, page{half};
            ld.global.v2.b32 {{exp_lo{half},exp_hi{half}}}, [meta+36864];
            mad.wide.u32 src{half}, slot, 576, page{half};
            cvt.u64.u32 byte_offset, offset;
            add.u64 src{half}, src{half}, byte_offset;
            """
        for half in range(2):
            asm += ptx_exponent_minmax(
                f"exp_lo{half}",
                f"exp_hi{half}",
                f"smallest{half}",
                f"largest{half}",
                "exponent",
            )
            asm += f"""
            sub.u32 delta, largest{half}, smallest{half};
            setp.gt.and.u32 wide, delta, 2, valid{half};
            @wide mov.u32 flag, 1;
            """
            asm += ptx_encode_e8m0_scale(f"largest{half}", f"valid{half}", "exponent")
            asm += f"""
            setp.eq.u32 first, chunk, 0;
            mad.lo.u32 tmp, row, 4, $6;
            @first st.shared.b32 [tmp+{half * 256}], exponent;
            """
        # A panel holds 64 rows by 128 FP8 channels per half. The last
        # transfer also brings in the two 64-byte pieces of BF16 RoPE.
        for panel in range(4):
            for group in range(2 * panel, min(2 * panel + 2, 7)):
                asm += f"xor.b32 exponent,base,{(group % 2) * 64}; add.u32 kdst,$4,exponent;\n"
                for half in range(2):
                    dest = half * 32768 + panel * 8192
                    asm += f"cp.async.cg.shared.global [kdst+{dest}],[src{half}+{group * 64}],16,size{half};\n"
            if panel == 3:
                for piece in range(2):
                    asm += f"xor.b32 exponent,base,{piece * 64}; add.u32 rdst,$5,exponent;\n"
                    for half in range(2):
                        asm += (
                            f"cp.async.cg.shared.global [rdst+{half * 8192}],"
                            f"[src{half}+{448 + piece * 64}],16,size{half};\n"
                        )
            asm += "cp.async.commit_group;\n"
        for panel in range(4):
            asm += f"cp.async.wait_group {3 - panel};\n"
            for group in range(2 * panel, min(2 * panel + 2, 7)):
                asm += f"xor.b32 exponent,base,{(group % 2) * 64}; add.u32 kdst,$4,exponent;\n"
                word = "exp_lo" if group < 4 else "exp_hi"
                for half in range(2):
                    dest = half * 32768 + panel * 8192
                    label = f"END_GROUP_{group}_{half}"
                    asm += f"""
                    bfe.u32 exponent, {word}{half}, {(group % 4) * 8}, 8;
                    setp.eq.u32 skip, exponent, largest{half};
                    @skip bra {label};
                    sub.u32 delta, largest{half}, exponent;
                    """
                    asm += ptx_rescale_factor("delta", "fac")
                    asm += f"ld.shared.v4.b32 {{a,b,c,d}}, [kdst+{dest}];\n"
                    asm += rescale_fp8_registers()
                    asm += f"st.shared.v4.b32 [kdst+{dest}],{{a,b,c,d}}; {label}:\n"
            # Each of the eight producer warps signals after its writes to both halves.
            asm += f"""
            bar.warp.sync 0xffffffff;
            fence.proxy.async.shared::cta;
            setp.eq.u32 first, lane, 0;
            @first mbarrier.arrive.shared::cta.b64 _, [${8 + panel}];
            """
        return asm + "mov.b32 $0,flag; }"

    @triton.constexpr_function
    def gather_asm(block, page_stride, panel_rows, warps, prefetched):
        # A subgroup of two or four lanes owns one selected row. Every lane keeps
        # that row's address and exponent word through both copying and rescaling.
        rows = block // warps
        lanes = 32 // rows
        asm = f"""
    {{
    .reg .pred ok, first, diff, any_diff, more, lower, wide, skip;
    .reg .b32 tid,lane,warp,row,chunk,pos,id,page,slot,e0,e1,hi,lo,ex,sz,flag;
    .reg .b32 off,sw,kbase,rbase,kdst,rdst,sdst,tmp,u,group,word,shift,delta,fac;
    .reg .b32 a,b,c,d;
    .reg .b16 xlo,xhi;
    .reg .b64 address,ptr,src,pg;
    mov.u32 tid,%tid.x;
    and.b32 tid,tid,{warps*32-1};
    shr.u32 warp,tid,5;
    and.b32 lane,tid,31;
    shr.u32 row,lane,{lanes.bit_length()-1};
    mad.lo.u32 row,warp,{rows},row;
    and.b32 chunk,lane,{lanes-1};
    add.u32 pos,$8,row;
    setp.lt.u32 ok,pos,$9;
    mov.u32 id,-1;
    """
        if prefetched:
            asm += "mad.lo.u32 tmp,pos,4,$10; @ok ld.shared.s32 id,[tmp];\n"
        else:
            asm += "mul.wide.u32 address,pos,4; add.u64 ptr,$3,address; @ok ld.global.L1::evict_first.s32 id,[ptr];\n"
        asm += f"""
    setp.lt.u32 ok,id,$7;
    @!ok mov.u32 id,0;
    shr.u32 page,id,6;
    and.b32 slot,id,63;
    mul.wide.u32 address,page,{page_stride};
    add.u64 pg,$2,address;
    """
        if prefetched:
            asm += "mad.lo.u32 tmp,pos,8,$11; mov.b32 e0,0; mov.b32 e1,0; @ok ld.shared.v2.b32 {e0,e1},[tmp];\n"
        else:
            asm += "mul.wide.u32 address,slot,8; add.u64 ptr,pg,address; ld.global.v2.b32 {e0,e1},[ptr+36864];\n"
        asm += """
    mul.wide.u32 address,slot,576;
    add.u64 src,pg,address;
    shl.b32 off,chunk,4;
    cvt.u64.u32 address,off;
    add.u64 src,src,address;
    selp.u32 sz,16,0,ok;
    and.b32 sw,row,7;
    shl.b32 sw,sw,4;
    xor.b32 off,off,sw;
    mad.lo.u32 tmp,row,128,$4;
    add.u32 kbase,tmp,off;
    mad.lo.u32 tmp,row,128,$5;
    add.u32 rbase,tmp,off;
    """
        copies = []
        for g in range(9):
            group_copy = ""
            for seg in range(4 // lanes):
                column_offset = (g % 2) * 64 + seg * lanes * 16
                adr = g * 64 + seg * lanes * 16
                if g < 7:
                    group_copy += f"xor.b32 kdst,kbase,{column_offset};\n"
                    group_copy += f"cp.async.cg.shared.global [kdst+{(g//2)*panel_rows*128}],[src+{adr}],16,sz;\n"
                else:
                    column_offset = (g - 7) * 64 + seg * lanes * 16
                    group_copy += (
                        f"xor.b32 rdst,rbase,{column_offset};\n"
                        f"cp.async.cg.shared.global [rdst],[src+{adr}],16,sz;\n"
                    )
            copies.append(group_copy)
        metadata = ""
        if prefetched:
            metadata += "shr.u32 hi,e1,24;\n"
        else:
            metadata += ptx_exponent_minmax("e0", "e1", "lo", "hi", "ex")
        metadata += ptx_encode_e8m0_scale("hi", "ok", "tmp")
        metadata += """
    setp.eq.u32 first,chunk,0;
    mad.lo.u32 sdst,row,4,$6;
    @first st.shared.b32 [sdst],tmp;
    """
        if prefetched:
            metadata += (
                "mov.u32 flag,0; mul.lo.u32 word,hi,0x01010101; setp.ne.u32 diff,e0,word; "
                "setp.ne.or.u32 diff,e1,word,diff; and.pred diff,diff,ok;\n"
            )
        else:
            metadata += (
                "sub.u32 delta,hi,lo; setp.gt.and.u32 wide,delta,2,ok; "
                "selp.u32 flag,1,0,wide; setp.ne.and.u32 diff,delta,0,ok;\n"
            )
        asm += "".join(copies) + "cp.async.commit_group;\n" + metadata
        asm += """
    vote.sync.any.pred any_diff,diff,0xffffffff;
    cp.async.wait_group 0;
    @!any_diff bra DONE;
    mov.u32 group,0;
    RESCALE:
    setp.lt.u32 lower,group,4;
    selp.u32 word,e0,e1,lower;
    and.b32 shift,group,3;
    shl.b32 shift,shift,3;
    bfe.u32 ex,word,shift,8;
    setp.eq.u32 skip,hi,ex;
    @skip bra NEXT;
    sub.u32 delta,hi,ex;
    """
        asm += ptx_rescale_factor("delta", "fac")
        asm += """
    and.b32 tmp,group,1;
    shl.b32 tmp,tmp,6;
    xor.b32 kdst,kbase,tmp;
    shr.u32 tmp,group,1;
    """
        asm += f"mad.lo.u32 kdst,tmp,{panel_rows*128},kdst;\n"
        for seg in range(4 // lanes):
            asm += f"xor.b32 rdst,kdst,{seg*lanes*16};\nld.shared.v4.b32 {{a,b,c,d}},[rdst];\n"
            asm += rescale_fp8_registers()
            asm += "st.shared.v4.b32 [rdst],{a,b,c,d};\n"
        return (
            asm
            + """
    NEXT:
    add.u32 group,group,1;
    setp.lt.u32 more,group,7;
    @more bra RESCALE;
    DONE:
    mov.b32 $0,flag;
    }
    """
        )

    @triton.jit
    def gather_tile(
        K,
        Idx,
        sk,
        skr,
        ks,
        b,
        first_pos,
        N,
        T: tl.constexpr,
        BK: tl.constexpr,
        STRIDE: tl.constexpr,
        id_address,
        exponent_address,
        PREFETCHED: tl.constexpr,
    ):
        return tl.inline_asm_elementwise(
            gather_asm(BK, STRIDE, BK, 8, PREFETCHED),
            "=r,r,l,l,r,r,r,r,r,r,r,r",
            [
                tl.arange(0, 256),
                K.to(tl.uint64),
                (Idx + b.to(tl.int64) * T).to(tl.uint64),
                tle.gpu.local_ptr(sk, (0, 0)).to(tl.uint64).to(tl.uint32),
                tle.gpu.local_ptr(skr, (0, 0)).to(tl.uint64).to(tl.uint32),
                tle.gpu.local_ptr(ks, (0,)).to(tl.uint64).to(tl.uint32),
                N,
                first_pos,
                tl.full((), T, tl.uint32),
                id_address,
                exponent_address,
            ],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def producer(
        K,
        Idx,
        Flags,
        RepairState,
        sk,
        skr,
        ks,
        kfull,
        kfree,
        N,
        T: tl.constexpr,
        STRIDE: tl.constexpr,
        S: tl.constexpr,
        BK: tl.constexpr,
        BUFS: tl.constexpr,
        PREFIX: tl.constexpr,
        TAIL: tl.constexpr,
        TAIL_SPLITS: tl.constexpr,
    ):
        b, _, first, count = model1_block_info(
            tl.program_id(0).to(tl.int64),
            tl.program_id(2),
            tl.cdiv(T, BK),
            S,
            PREFIX,
            TAIL,
            TAIL_SPLITS,
        )
        # Resolve the two dependent global reads for all short-list tiles up front.
        PREFETCHED: tl.constexpr = T > 128 and T <= 640 and BK == 64 and S == 1
        if PREFETCHED:
            width: tl.constexpr = triton.next_power_of_2(T)
            saved_ids = tle.gpu.alloc(
                [width], tl.int32, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            saved_words = tle.gpu.alloc(
                [width], tl.int64, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            pos = tl.arange(0, width)
            token = tl.load(
                Idx + b.to(tl.int64) * T + pos,
                (pos < T) & (pos >= first * BK) & (pos < (first + count) * BK),
                -1,
            )
            valid_token = (token >= 0) & (token < N)
            selected = tl.where(valid_token, token, 0)
            page = (selected // 64).to(tl.int64) * STRIDE
            word = tl.load(
                K.to(tl.pointer_type(tl.int64)) + page // 8 + 4608 + selected % 64,
                valid_token,
                0,
            )
            maximum = (word & 255).to(tl.uint32)
            minimum = maximum
            for group in tl.static_range(1, 7):
                exponent = ((word >> (8 * group)) & 255).to(tl.uint32)
                maximum = tl.maximum(maximum, exponent)
                minimum = tl.minimum(minimum, exponent)
            repair = tl.max((valid_token & (maximum - minimum > 2)).to(tl.int32), 0)
            packed_word = (word.to(tl.uint64) & 0x00FFFFFFFFFFFFFF) | (
                maximum.to(tl.uint64) << 56
            )
            tl.store(tle.gpu.local_ptr(saved_ids), token)
            tl.store(
                tle.gpu.local_ptr(saved_words), packed_word.to(tl.int64, bitcast=True)
            )
            tl.debug_barrier()
            id_address = tle.gpu.local_ptr(saved_ids, (0,)).to(tl.uint64).to(tl.uint32)
            exponent_address = (
                tle.gpu.local_ptr(saved_words, (0,)).to(tl.uint64).to(tl.uint32)
            )
        else:
            repair = tl.full((), 0, tl.int32)
            id_address = 0
            exponent_address = 0
        for step in range(count):
            buf, cycle = step % BUFS, step // BUFS
            if BUFS == 1:
                if step > 0:
                    tle.gpu.barrier_wait(kfree[0])
                else:
                    pass
            else:
                tle.gpu.barrier_wait(kfree[buf], phaseIdx=cycle)
            flag = gather_tile(
                K,
                Idx,
                sk.slot(buf),
                skr.slot(step % 4 if BUFS == 2 else 0),
                ks.slot(buf),
                b,
                (first + step) * BK,
                N,
                T,
                BK,
                STRIDE,
                id_address,
                exponent_address,
                PREFETCHED,
            )
            repair |= tl.max(flag, 0).to(tl.int32)
            fence_proxy_async_shared(tl.arange(0, 256))
            tl.debug_barrier()
            if BUFS == 1:
                tle.gpu.barrier_arrive(kfull[0])
            else:
                tle.gpu.barrier_arrive(kfull[buf], phaseIdx=cycle)
        tl.atomic_or(Flags + b, 1, mask=repair != 0, sem="relaxed")
        tl.atomic_or(RepairState, 1, mask=repair != 0, sem="relaxed")

    @triton.jit
    def leader(
        Output,
        L,
        sq,
        sr,
        qs,
        sk,
        skr,
        ks,
        sv64,
        sp,
        alpha,
        factor,
        qfull,
        kfull,
        kfree,
        pfull,
        vfull,
        vfree,
        ofull,
        SCALE: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        S: tl.constexpr,
        BK: tl.constexpr,
        BUFS: tl.constexpr,
        RING: tl.constexpr,
        PREFIX: tl.constexpr,
        TAIL: tl.constexpr,
        TAIL_SPLITS: tl.constexpr,
    ):
        hb = tl.program_id(1)
        b, output_row, _, count = model1_block_info(
            tl.program_id(0).to(tl.int64),
            tl.program_id(2),
            tl.cdiv(T, BK),
            S,
            PREFIX,
            TAIL,
            TAIL_SPLITS,
        )
        rows = (output_row) * H + hb * 64 + tl.arange(0, 64)
        maximum = tl.full((64,), -float("inf"), tl.float32)
        denom = tl.full((64,), 0.0, tl.float32)
        prev = tl.full((64,), 1.0, tl.float32)
        acc = tl.full((64, 64), 0.0, tl.float32)
        tail = tl.full((64, 64), 0.0, tl.float32)
        tle.gpu.barrier_wait(qfull[0])
        qscale = tl.load(tle.gpu.local_ptr(qs))
        for step in range(count):
            buf, cycle = step % BUFS, step // BUFS
            if BUFS == 1:
                tle.gpu.barrier_wait(kfull[0])
            else:
                tle.gpu.barrier_wait(kfull[buf], phaseIdx=cycle)
            logits = tle.gpu.wgmma(
                tle_subslice(sq, [0, 0], [64, 256]),
                tle_subslice(sk.slot(buf), [0, 0], [BK, 256]),
                out_dtype=tl.float32,
                trans_b=True,
            )
            logits = tle.gpu.wgmma(
                tle_subslice(sq, [0, 256], [64, 128]),
                tle_subslice(sk.slot(buf), [0, 256], [BK, 128]),
                logits,
                trans_b=True,
            )
            logits = tle.gpu.wgmma(
                tle_subslice(sq, [0, 384], [64, 64]),
                tle_subslice(sk.slot(buf), [0, 384], [BK, 64]),
                logits,
                trans_b=True,
            )
            rope = tle.gpu.wgmma(
                sr, skr.slot(step % RING), out_dtype=tl.float32, trans_b=True
            )
            logits = tle.gpu.wgmma_wait(0, logits)
            rope = tle.gpu.wgmma_wait(0, rope)
            sc = tl.load(tle.gpu.local_ptr(ks.slot(buf)))
            if RING > BUFS:
                tle.gpu.barrier_arrive(kfree[buf], phaseIdx=cycle)
            else:
                pass
            scores = tl.where(
                sc[None, :] >= 0.0,
                (logits * qscale[:, None] * sc[None, :] + rope)
                * (SCALE * 1.4426950408889634),
                -float("inf"),
            )
            maximum_new = tl.maximum(maximum, tl.max(scores, 1))
            safe = tl.where(maximum_new == -float("inf"), 0.0, maximum_new)
            corr = tl.exp2(maximum - safe)
            p = tl.exp2(scores - safe[:, None])
            denom = denom * corr + tl.sum(p, 1)
            maximum = maximum_new
            weighted = p * tl.maximum(sc, 0.0)[None, :]
            w = tl.max(weighted, 1) / 448.0
            w = tl.where(w > 0.0, w, 1.0)
            normalized = weighted / w[:, None]
            beta = prev * corr / w
            prev = w
            if BK == 64:
                publish_p_fp8_sw64_coupled_stmatrix(sp.slot(buf), normalized)
            else:
                publish_sparse_p128_fp8(sp.slot(0), normalized, sp.slot(1))
            tl.store(tle.gpu.local_ptr(alpha.slot(buf)), beta)
            tle.gpu.barrier_arrive(pfull[0])
            tail = tle.gpu.wgmma(
                p.to(tl.bfloat16), skr.slot(step % RING), tail * corr[:, None]
            )
            if RING == BUFS:
                tail = tle.gpu.wgmma_wait(0, tail)
                if BUFS == 1:
                    tle.gpu.barrier_arrive(kfree[0])
                else:
                    tle.gpu.barrier_arrive(kfree[buf], phaseIdx=cycle)
            else:
                pass
            tle.gpu.barrier_wait(vfull[0])
            acc *= beta[:, None]
            for half in tl.static_range(BK // 64):
                vbuf = buf * (BK // 64) + half
                acc = tle.gpu.wgmma(
                    sp.slot(vbuf),
                    sv64.slot(vbuf),
                    acc,
                    trans_b=True,
                )
            acc = tle.gpu.wgmma_wait(0, acc)
            if RING > BUFS:
                tail = tle.gpu.wgmma_wait(0, tail)
            else:
                pass
            if BUFS == 1 and tl.cdiv(T, BK) > S:
                tle.gpu.barrier_wait(vfree[0])
            else:
                pass
        inv = tl.where(denom > 0.0, 1.0 / denom, 0.0)
        f = inv * prev
        tl.store(tle.gpu.local_ptr(factor), f)
        tle.gpu.barrier_arrive(ofull[0])
        d = tl.arange(0, 64)
        cols = (d // 16) * 16 + (d % 8) * 2 + (d % 16) // 8
        tl.store(
            Output + rows[:, None] * 512 + 384 + cols[None, :],
            acc * f[:, None],
            cache_modifier=".cs" if S == 1 else "",
        )
        tl.store(
            Output + rows[:, None] * 512 + 448 + tl.arange(0, 64)[None, :],
            tail * inv[:, None],
            cache_modifier=".cs" if S == 1 else "",
        )
        tl.store(
            L + rows,
            tl.where(
                denom > 0.0,
                maximum * 0.6931471805599453 + tl.log(denom),
                float("inf") if S == 1 else -float("inf"),
            ),
        )

    @triton.jit
    def follower(
        Q,
        QR,
        QS,
        Flags,
        RepairState,
        sq,
        sr,
        qs,
        qfull,
        Output,
        sk,
        sv,
        sv128,
        sv64,
        sp,
        alpha,
        factor,
        kfull,
        kfree,
        pfull,
        vfull,
        vfree,
        ofull,
        T: tl.constexpr,
        H: tl.constexpr,
        S: tl.constexpr,
        BK: tl.constexpr,
        BUFS: tl.constexpr,
        PREFIX: tl.constexpr,
        TAIL: tl.constexpr,
        TAIL_SPLITS: tl.constexpr,
    ):
        hb = tl.program_id(1)
        b, output_row, _, count = model1_block_info(
            tl.program_id(0).to(tl.int64),
            tl.program_id(2),
            tl.cdiv(T, BK),
            S,
            PREFIX,
            TAIL,
            TAIL_SPLITS,
        )
        h = (b * H + tl.program_id(1) * 64) + tl.arange(0, 64)
        d = tl.arange(0, 512)
        if S > 1 and T > 640:
            signed_scale = tl.load(QS + h)
            mx = tl.abs(signed_scale)
            qrepair = tl.max((signed_scale < 0.0).to(tl.int32), 0)
            tl.store(
                tle.gpu.local_ptr(sq),
                tl.load(
                    Q + h[:, None] * 448 + d[None, :],
                    d[None, :] < 448,
                    0.0,
                    eviction_policy="evict_first",
                ),
            )
        else:
            g = tl.arange(0, 8)
            scales = tl.load(QS + h[:, None] * 7 + g[None, :], g[None, :] < 7, 0.0)
            mx = tl.max(scales, 1)
            mn = tl.min(tl.where(scales > 0.0, scales, float("inf")), 1)
            qrepair = tl.max((mx > 4.0 * mn).to(tl.int32), 0)
            if T == 128 and S == 1:
                # Stage the short list's full query from a single load of its group scales.
                content = tl.load(
                    Q + h[:, None] * 448 + d[None, :],
                    d[None, :] < 448,
                    0.0,
                    eviction_policy="evict_first",
                )
                ratios = scales / tl.where(mx > 0.0, mx, 1.0)[:, None]
                normalized = scaled_bytes(content, ratios, 64)
                tl.store(tle.gpu.local_ptr(sq), normalized)
            else:
                cols128 = tl.arange(0, 128)
                inv = 1.0 / tl.where(mx > 0.0, mx, 1.0)
                for pair in tl.static_range(4):
                    dims = pair * 128 + cols128
                    qpart = tl.load(
                        Q + h[:, None] * 448 + dims[None, :],
                        dims[None, :] < 448,
                        0.0,
                        eviction_policy="evict_first",
                    )
                    qleft = tl.load(QS + h * 7 + 2 * pair) * inv
                    if pair < 3:
                        qright = tl.load(QS + h * 7 + 2 * pair + 1) * inv
                    else:
                        qright = 0.0
                    ratio = tl.where(
                        cols128[None, :] < 64, qleft[:, None], qright[:, None]
                    )
                    qnorm = (qpart.to(tl.float32) * ratio).to(tl.float8e4nv)
                    tl.store(
                        tle.gpu.local_ptr(tle_subslice(sq, [0, pair * 128], [64, 128])),
                        qnorm,
                    )
        tl.store(tle.gpu.local_ptr(qs), mx)
        tl.store(
            tle.gpu.local_ptr(sr),
            tl.load(
                QR + h[:, None] * 64 + tl.arange(0, 64)[None, :],
                eviction_policy="evict_first",
            ),
        )
        tle.gpu.barrier_arrive(qfull[0])
        tl.atomic_or(Flags + b, 1, mask=qrepair != 0, sem="relaxed")
        tl.atomic_or(RepairState, 1, mask=qrepair != 0, sem="relaxed")
        a1 = tl.full((64, 128), 0.0, tl.float32)
        a = tl.full((64, 128), 0.0, tl.float32)
        c = tl.full((64, 128), 0.0, tl.float32)
        for step in range(count):
            buf, cycle = step % BUFS, step // BUFS
            if BUFS == 1:
                tle.gpu.barrier_wait(kfull[0])
            else:
                tle.gpu.barrier_wait(kfull[buf], phaseIdx=cycle)
            for half in tl.static_range(BK // 64):
                for group in tl.static_range(4):
                    src = tle_subslice(
                        sk.slot(buf), [half * 64, group * 128], [64, 128]
                    )
                    for warp_half in tl.static_range(1 if group == 3 else 2):
                        sparse_fp8_transpose_values(
                            src,
                            (sv if group < 2 else sv128 if group == 2 else sv64).slot(
                                buf * (BK // 64) + half
                            ),
                            group * 128 if group < 2 else 0,
                            warp_half * 4,
                        )
            fence_proxy_async_shared(tl.arange(0, 128))
            tle.gpu.barrier_arrive(vfull[0])
            if BUFS == 1:
                tle.gpu.barrier_arrive(kfree[0])
            else:
                tle.gpu.barrier_arrive(kfree[buf], phaseIdx=cycle)
            tle.gpu.barrier_wait(pfull[0])
            beta = tl.load(tle.gpu.local_ptr(alpha.slot(buf)))
            a *= beta[:, None]
            a1 *= beta[:, None]
            c *= beta[:, None]
            for half in tl.static_range(BK // 64):
                vbuf = buf * (BK // 64) + half
                a = tle.gpu.wgmma(
                    sp.slot(vbuf),
                    tle_subslice(sv.slot(vbuf), [0, 0], [128, 64]),
                    a,
                    trans_b=True,
                )
                a1 = tle.gpu.wgmma(
                    sp.slot(vbuf),
                    tle_subslice(sv.slot(vbuf), [128, 0], [128, 64]),
                    a1,
                    trans_b=True,
                )
                c = tle.gpu.wgmma(
                    sp.slot(vbuf),
                    sv128.slot(vbuf),
                    c,
                    trans_b=True,
                )
            a1 = tle.gpu.wgmma_wait(0, a1)
            a = tle.gpu.wgmma_wait(0, a)
            c = tle.gpu.wgmma_wait(0, c)
            if BUFS == 1 and tl.cdiv(T, BK) > S:
                tle.gpu.barrier_wait(vfree[0])
            else:
                pass
        tle.gpu.barrier_wait(ofull[0])
        f = tl.load(tle.gpu.local_ptr(factor))
        sparse_fp8_publish_output(a, f, Output, output_row, hb, H, 0, S == 1)
        sparse_fp8_publish_output(a1, f, Output, output_row, hb, H, 128, S == 1)
        sparse_fp8_publish_output(c, f, Output, output_row, hb, H, 256, S == 1)

    @triton.jit
    def model1_fp8(
        Q,
        QR,
        QS,
        E,
        Idx,
        Output,
        L,
        Flags,
        RepairState,
        N,
        T: tl.constexpr,
        H: tl.constexpr,
        STRIDE: tl.constexpr,
        SCALE: tl.constexpr,
        S: tl.constexpr,
        BK: tl.constexpr,
        BUFS: tl.constexpr,
        PREFIX: tl.constexpr,
        TAIL: tl.constexpr,
        TAIL_SPLITS: tl.constexpr,
    ):
        K = E.to(tl.pointer_type(tl.float8e4nv))
        sq = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
        sr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
        qs = tle.gpu.alloc(
            [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        sk = tle.gpu.alloc([BUFS, BK, 512], tl.float8e4nv, scope=tle.gpu.smem)
        RING: tl.constexpr = 4 if BUFS == 2 else BUFS
        skr = tle.gpu.alloc([RING, BK, 64], tl.bfloat16, scope=tle.gpu.smem)
        ks = tle.gpu.alloc(
            [BUFS, BK], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        sv = tle.gpu.alloc(
            [BUFS * (BK // 64), 256, 64], tl.float8e4nv, scope=tle.gpu.smem
        )
        sv128 = tle.gpu.alloc(
            [BUFS * (BK // 64), 128, 64], tl.float8e4nv, scope=tle.gpu.smem
        )
        sv64 = tle.gpu.alloc(
            [BUFS * (BK // 64), 64, 64], tl.float8e4nv, scope=tle.gpu.smem
        )
        sp = tle.gpu.alloc(
            [BUFS * (BK // 64), 64, 64], tl.float8e4nv, scope=tle.gpu.smem
        )
        alpha = tle.gpu.alloc(
            [BUFS, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        factor = tle.gpu.alloc(
            [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        qfull = sparse_named_barriers(1, 256, 16)
        if BUFS == 1:
            kfull = sparse_named_barriers(1, 512, 18)
            kfree = sparse_named_barriers(1, 512, 19)
        else:
            kfull = tle.gpu.alloc_barriers(BUFS, arrive_count=1)
            kfree = tle.gpu.alloc_barriers(BUFS, arrive_count=2, init=tle.gpu.READY)
        vfull = sparse_named_barriers(1, 256, 20)
        pfull = sparse_named_barriers(1, 256, 21)
        vfree = sparse_named_barriers(1, 256, 23)
        ofull = sparse_named_barriers(1, 256, 17)
        tle.gpu.warp_specialize(
            [
                (
                    leader,
                    (
                        Output,
                        L,
                        sq,
                        sr,
                        qs,
                        sk,
                        skr,
                        ks,
                        sv64,
                        sp,
                        alpha,
                        factor,
                        qfull,
                        kfull,
                        kfree,
                        pfull,
                        vfull,
                        vfree,
                        ofull,
                        SCALE,
                        T,
                        H,
                        S,
                        BK,
                        BUFS,
                        RING,
                        PREFIX,
                        TAIL,
                        TAIL_SPLITS,
                    ),
                ),
                (
                    producer,
                    (
                        K,
                        Idx,
                        Flags,
                        RepairState,
                        sk,
                        skr,
                        ks,
                        kfull,
                        kfree,
                        N,
                        T,
                        STRIDE,
                        S,
                        BK,
                        BUFS,
                        PREFIX,
                        TAIL,
                        TAIL_SPLITS,
                    ),
                ),
                (
                    follower,
                    (
                        Q,
                        QR,
                        QS,
                        Flags,
                        RepairState,
                        sq,
                        sr,
                        qs,
                        qfull,
                        Output,
                        sk,
                        sv,
                        sv128,
                        sv64,
                        sp,
                        alpha,
                        factor,
                        kfull,
                        kfree,
                        pfull,
                        vfull,
                        vfree,
                        ofull,
                        T,
                        H,
                        S,
                        BK,
                        BUFS,
                        PREFIX,
                        TAIL,
                        TAIL_SPLITS,
                    ),
                ),
            ],
            [8, 4],
            [40 if S > 1 and BK == 64 else 48, 248],
        )

    @triton.jit
    def repair_heads(
        Q,
        QR,
        QS,
        K,
        E,
        KR,
        Idx,
        Output,
        L,
        b,
        group,
        N,
        T: tl.constexpr,
        H: tl.constexpr,
        STRIDE: tl.constexpr,
        SCALE: tl.constexpr,
        OUTPUT_TILES: tl.constexpr = 1,
    ):
        rows = b.to(tl.int64) * H + group * 16 + tl.arange(0, 16)
        d = tl.arange(0, 512)
        q = tl.load(Q + rows[:, None] * 448 + d[None, :], d[None, :] < 448, 0.0)
        qs = tl.load(QS + rows[:, None] * 7 + d[None, :] // 64, d[None, :] < 448, 0.0)
        qr = tl.load(
            QR + rows[:, None] * 64 + (d % 64)[None, :], d[None, :] >= 448, 0.0
        )
        qbf = tl.where(d[None, :] < 448, q.to(tl.float32) * qs, qr).to(tl.bfloat16)
        m = tl.full((16,), -float("inf"), tl.float32)
        denominator = tl.full((16,), 0.0, tl.float32)
        acc = tl.full((16, 512), 0.0, tl.float32)
        for it in range(tl.cdiv(T, 32)):
            pos = it * 32 + tl.arange(0, 32)
            ids = tl.load(
                Idx + b.to(tl.int64) * T + pos,
                pos < T,
                -1,
                eviction_policy="evict_first",
            )
            valid = (ids >= 0) & (ids < N)
            ids = tl.where(valid, ids, 0)
            page = (ids >> 6).to(tl.int64) * STRIDE
            slot = ids & 63
            base = page + slot * 576
            e = tl.load(E + (page + 36864 + slot * 8)[:, None] + (d // 64)[None, :]).to(
                tl.int32
            )
            ks = tl.exp2((e - 127).to(tl.float32))
            raw = tl.load(
                K + base[:, None] + d[None, :], valid[:, None] & (d[None, :] < 448), 0.0
            )
            r = tl.load(
                KR + (base // 2 + 224)[:, None] + (d % 64)[None, :],
                valid[:, None] & (d[None, :] >= 448),
                0.0,
            )
            kv = tl.where(d[None, :] < 448, raw.to(tl.float32) * ks, r).to(tl.bfloat16)
            scores = tl.where(
                valid[None, :],
                tl.dot(qbf, tl.trans(kv)) * (SCALE * 1.4426950408889634),
                -float("inf"),
            )
            nm = tl.maximum(m, tl.max(scores, 1))
            safe = tl.where(nm == -float("inf"), 0.0, nm)
            p = tl.exp2(scores - safe[:, None])
            corr = tl.exp2(m - safe)
            acc = tl.dot(p.to(tl.bfloat16), kv, acc * corr[:, None])
            denominator = denominator * corr + tl.sum(p, 1)
            m = nm
        tl.store(
            Output + rows[:, None] * 512 + d[None, :],
            acc * tl.where(denominator > 0.0, 1.0 / denominator, 0.0)[:, None],
            mask=(OUTPUT_TILES == 1)
            | (d[None, :] // (512 // OUTPUT_TILES) == tl.program_id(2)),
        )
        tl.store(
            L + rows,
            tl.where(
                denominator > 0.0,
                m * 0.6931471805599453 + tl.log(denominator),
                float("inf"),
            ),
            mask=(OUTPUT_TILES == 1) | (tl.program_id(2) == 0),
        )

    @triton.jit
    def model1_repair(
        Q,
        QR,
        QS,
        K,
        E,
        KR,
        Idx,
        Output,
        L,
        Meta0,
        Meta1,
        Meta2,
        B,
        N,
        T: tl.constexpr,
        H: tl.constexpr,
        STRIDE: tl.constexpr,
        SCALE: tl.constexpr,
        PACKED: tl.constexpr = False,
    ):
        if PACKED:
            kmax = tl.load(Meta2).to(tl.float32, bitcast=True)
            wide_cache = tl.load(Meta2 + 1)
        elif tl.load(Meta1) == 0:
            return
        for b in range(tl.program_id(0), B, tl.num_programs(0)):
            if PACKED:
                qs = tl.load(Meta0 + b * (H // 16) + tl.arange(0, H // 16))
                amplification = tl.max(tl.abs(qs), 0) * tl.abs(
                    SCALE * 1.4426950408889634
                )
                unsafe = tl.min(qs, 0) < 0.0
                if wide_cache != 0 or amplification * kmax > QK_RECOMPUTE_THRESHOLD:
                    for step in range(tl.cdiv(T, 256)):
                        pos = step * 256 + tl.arange(0, 256)
                        idx = tl.load(Idx + b.to(tl.int64) * T + pos, pos < T, -1)
                        valid = (idx >= 0) & (idx < N)
                        ks = tl.load(Meta1 + tl.where(valid, idx, 0), valid, 0.0)
                        unsafe |= (tl.min(ks, 0) < 0.0) | (
                            amplification * tl.max(tl.abs(ks), 0)
                            > QK_RECOMPUTE_THRESHOLD
                        )
            else:
                unsafe = tl.load(Meta0 + b) != 0
            if unsafe:
                for group in range(H // 16):
                    repair_heads(
                        Q,
                        QR,
                        QS,
                        K,
                        E,
                        KR,
                        Idx,
                        Output,
                        L,
                        b,
                        group,
                        N,
                        T,
                        H,
                        STRIDE,
                        SCALE,
                    )

    @triton.jit
    def merge(
        o_part,
        lse_part,
        out,
        lse,
        H: tl.constexpr,
        SPLITS: tl.constexpr,
        D: tl.constexpr,
    ):
        ss = tl.arange(0, triton.next_power_of_2(SPLITS))
        d = tl.program_id(2) * D + tl.arange(0, D)
        batch = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1)
        rows = (batch * SPLITS + ss) * H + h
        stats = tl.load(lse_part + rows, ss < SPLITS, -float("inf"))
        stats = tl.where(stats == float("inf"), -float("inf"), stats)
        m = tl.max(stats, 0)
        empty = m == -float("inf")
        w = tl.exp(stats - tl.where(empty, 0.0, m))
        denom = tl.sum(w, 0)
        v = tl.load(
            o_part + rows[:, None] * 512 + d[None, :],
            ss[:, None] < SPLITS,
            0.0,
            eviction_policy="evict_first",
        )
        merged_value = tl.sum(v.to(tl.float32) * w[:, None], 0) / tl.where(
            empty, 1.0, denom
        )
        tl.store(out + (batch * H + h) * 512 + d, merged_value.to(tl.bfloat16))
        tl.store(
            lse + batch * H + h,
            tl.where(empty, float("inf"), m + tl.log(denom)),
            mask=tl.program_id(2) == 0,
        )

    @triton.jit
    def model1_pack_rows(
        Content,
        Rope,
        Scales,
        Unified,
        Magnitude,
        Bounds,
        Global,
        N: tl.constexpr,
        STRIDE: tl.constexpr,
        CACHE: tl.constexpr,
        ROWS: tl.constexpr,
    ):
        row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        d = tl.arange(0, 512)
        groups = tl.arange(0, 8)
        rd = tl.arange(0, 64)
        if CACHE:
            base = (row // 64).to(tl.int64) * STRIDE
            payload = base + row % 64 * 576
            half = payload // 2 + 224
            exponent = tl.load(
                Scales + (base + 36864 + (row % 64) * 8)[:, None] + groups[None, :],
                row[:, None] < N,
                0,
            ).to(tl.int32)
            scales = tl.where(
                groups[None, :] < 7,
                tl.maximum(exponent << 23, 0x00400000).to(tl.float32, bitcast=True),
                0.0,
            )
        else:
            payload = row.to(tl.int64) * 448
            half = row.to(tl.int64) * 64
            scales = tl.load(
                Scales + row[:, None] * 7 + groups[None, :], groups[None, :] < 7, 0.0
            )
        valid = row < N
        data = tl.load(
            Content + payload[:, None] + d[None, :],
            valid[:, None] & (d[None, :] < 448),
            0.0,
        )
        rope = tl.load(Rope + half[:, None] + rd[None, :], valid[:, None], 0.0).to(
            tl.float32
        )
        largest = tl.max(scales, 1)
        smallest = tl.min(tl.where(groups[None, :] < 7, scales, float("inf")), 1)
        rope_scale = tl.max(tl.abs(rope), 1) * (1.0 / 448.0)
        # Rounding the tail scale up to a power of two preserves the original
        # FP8 content when the seven E8M0 groups share a scale.
        bits = rope_scale.to(tl.uint32, bitcast=True)
        rope_unit = ((bits + 0x007FFFFF) & 0x7F800000).to(tl.float32, bitcast=True)
        unit = tl.maximum(largest, rope_unit)
        inv = 1.0 / tl.where(unit > 0.0, unit, 1.0)
        ratio = tl.broadcast_to(
            (scales * inv[:, None])[:, :, None], (ROWS, 8, 64)
        ).reshape(ROWS, 512)
        tail = tl.broadcast_to(
            (rope * inv[:, None])[:, None, :], (ROWS, 8, 64)
        ).reshape(ROWS, 512)
        norm = tl.where(d[None, :] < 448, data.to(tl.float32) * ratio, tail)
        tl.store(
            Unified + row.to(tl.int64)[:, None] * 512 + d[None, :],
            norm.to(tl.float8e4nv),
            valid[:, None],
        )
        tl.store(Magnitude + row, unit, valid)
        wide = (
            (largest > 4.0 * smallest)
            | (rope_scale > 4.0 * smallest)
            | ((rope_scale > 0.0) & (rope_scale * 32.0 < unit))
        )
        if CACHE:
            tl.store(Bounds + row, tl.where(wide, -unit, unit), valid)
            tl.atomic_max(
                Global,
                tl.max(tl.where(valid, unit, 0.0), 0).to(tl.int32, bitcast=True),
                sem="relaxed",
            )
            if tl.max((valid & wide).to(tl.int32), 0):
                tl.atomic_or(Global + 1, 1, sem="relaxed")
        else:
            maximum = tl.max(unit, 0)
            tl.store(
                Bounds + tl.program_id(0),
                tl.where(tl.max(wide.to(tl.int32), 0) != 0, -maximum, maximum),
            )

    @tlc.builtin
    def model1_barrier_address(slot, _semantic=None):
        """Expose a TLE barrier slot to the producer's per-warp PTX arrival."""
        ptr_type = tl.pointer_type(tl.int64, 3)
        zero = _semantic.to_tensor(0)
        handle = _semantic.builder.create_local_pointers(
            ptr_type.to_ir(_semantic.builder), slot.handle, zero.handle
        )
        return tl.tensor(handle.get_result(0), ptr_type)

    @triton.jit
    def model1_load_key_pair(
        Cache,
        Idx,
        repair_flags,
        key_tiles,
        key_rope,
        key_scales,
        transposed_values,
        panel_barriers,
        values_ready,
        N: tl.constexpr,
        STRIDE: tl.constexpr,
        OUTPUT_TILES: tl.constexpr,
    ):
        batch = tl.program_id(0)
        # Eight producer warps gather both 64-key halves. Each 128-channel
        # panel can be consumed as soon as its final dequantizing stores finish.
        lane = tl.arange(0, 256)
        key_flags = tl.inline_asm_elementwise(
            gather_topk128_asm(STRIDE),
            "=r,r,l,l,r,r,r,r,r,r,r,r",
            [
                lane,
                Cache.to(tl.uint64),
                (Idx + batch.to(tl.int64) * 128).to(tl.uint64),
                tle.gpu.local_ptr(key_tiles.slot(0), (0, 0))
                .to(tl.uint64)
                .to(tl.uint32),
                tle.gpu.local_ptr(key_rope.slot(0), (0, 0)).to(tl.uint64).to(tl.uint32),
                tle.gpu.local_ptr(key_scales.slot(0), (0,)).to(tl.uint64).to(tl.uint32),
                N,
                model1_barrier_address(panel_barriers[0]).to(tl.uint64).to(tl.uint32),
                model1_barrier_address(panel_barriers[1]).to(tl.uint64).to(tl.uint32),
                model1_barrier_address(panel_barriers[2]).to(tl.uint64).to(tl.uint32),
                model1_barrier_address(panel_barriers[3]).to(tl.uint64).to(tl.uint32),
            ],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
        fence_proxy_async_shared(key_flags)
        tl.debug_barrier()
        half = tl.inline_asm_elementwise(
            "{.reg .u32 t; mov.u32 t,%tid.x; shr.u32 t,t,7; and.b32 $0,t,1;}",
            "=r,r",
            [lane],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        # Each group of four warps transposes one half for the later PV product.
        key_address = (
            tle.gpu.local_ptr(key_tiles.slot(0), (0, 0)).to(tl.uint64).to(tl.uint32)
            + half * 32768
        )
        value_address = (
            tle.gpu.local_ptr(transposed_values.slot(0), (0, 0))
            .to(tl.uint64)
            .to(tl.uint32)
            + half * 32768
        )
        for group in tl.static_range(4):
            if OUTPUT_TILES == 1 or tl.program_id(2) == group:
                for warp_half in tl.static_range(1 if group == 3 else 2):
                    tl.inline_asm_elementwise(
                        vtranspose_fp8_64x128_coupled_asm(),
                        "=r,r,r,r,r",
                        [
                            lane,
                            key_address + group * 8192,
                            value_address + group * 8192,
                            warp_half * 4,
                        ],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
        fence_proxy_async_shared(lane)
        tl.store(tle.gpu.local_ptr(repair_flags, (0,)), tl.max(key_flags, 0))
        tle.gpu.barrier_arrive(values_ready[0])

    @triton.constexpr_function
    def model1_store64_asm(permuted):
        # Pair adjacent BF16 output elements held by a lane. The content
        # accumulator has the column permutation introduced by the V transpose.
        code = """{
        .reg .b32 tid,row,col,ofs,a,b,c,d; .reg .b64 addr;
        mov.u32 tid,%tid.x; and.b32 tid,tid,127;
        shr.u32 row,tid,5; shl.b32 row,row,4;
        and.b32 col,tid,31; shr.u32 col,col,2; add.u32 row,row,col;
        and.b32 col,tid,3;
        """
        code += "shl.b32 col,col,3;" if permuted else "shl.b32 col,col,2;"
        code += (
            "mad.lo.u32 ofs,row,1024,col; cvt.u64.u32 addr,ofs; add.u64 addr,addr,$64;"
        )
        if permuted:
            for group in range(4):
                k = 32 + group * 8
                for j, reg in enumerate(("a", "b", "c", "d")):
                    code += f"cvt.rn.bf16x2.f32 {reg}, ${k+4+j}, ${k+j};"
                code += (
                    f"st.global.cs.v2.b32 [addr+{group*32}],{{a,b}}; "
                    f"st.global.cs.v2.b32 [addr+{8192+group*32}],{{c,d}};"
                )
        else:
            for group in range(8):
                k = 32 + group * 4
                code += f"cvt.rn.bf16x2.f32 a,${k+1},${k}; cvt.rn.bf16x2.f32 b,${k+3},${k+2};"
                code += f"st.global.cs.b32 [addr+{group*16}],a; st.global.cs.b32 [addr+{8192+group*16}],b;"
        return code + "mov.u32 $0,0; }"

    MODEL1_STORE64_CONSTRAINTS = tl.constexpr(
        ",".join(["=r"] * 32 + ["f"] * 32 + ["l"] * 32)
    )

    @triton.jit
    def model1_store64(
        value, factor, out, first, start: tl.constexpr, permuted: tl.constexpr
    ):
        base = (out + first * 512 + start).to(tl.uint64)
        tl.inline_asm_elementwise(
            model1_store64_asm(permuted),
            MODEL1_STORE64_CONSTRAINTS,
            [value * factor[:, None], base],
            dtype=tl.uint32,
            is_pure=False,
            pack=32,
        )

    @triton.jit
    def model1_value_band(
        fp8_prob,
        transposed_values,
        Out,
        batch,
        head_block,
        inverse,
        scale,
        H: tl.constexpr,
        start: tl.constexpr,
        width: tl.constexpr,
    ):
        value = tle.gpu.wgmma(
            fp8_prob.slot(0),
            tle_subslice(transposed_values.slot(0), [start, 0], [width, 64]),
            trans_b=True,
        )
        value = tle.gpu.wgmma(
            fp8_prob.slot(1),
            tle_subslice(transposed_values.slot(1), [start, 0], [width, 64]),
            value,
            trans_b=True,
        )
        value = tle.gpu.wgmma_wait(0, value)
        if width == 128:
            sparse_fp8_publish_output(
                value, inverse * scale, Out, batch, head_block, H, start, True
            )
        else:
            model1_store64(
                value, inverse * scale, Out, batch * H + head_block * 64, start, True
            )

    @triton.jit
    def model1_reciprocal(x):
        return tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0,$1;",
            "=f,f",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def model1_compute_key_half(
        Q,
        QR,
        QS,
        Out,
        L,
        repair_flags,
        query_tile,
        query_rope,
        query_scale,
        key_tiles,
        key_rope,
        key_scales,
        transposed_values,
        fp8_prob,
        bf16_prob,
        row_maxima,
        row_sums,
        panel_barriers,
        query_ready,
        max_ready,
        values_ready,
        output_ready,
        HALF: tl.constexpr,
        H: tl.constexpr,
        SCALE: tl.constexpr,
        OUTPUT_TILES: tl.constexpr,
    ):
        batch = tl.program_id(0).to(tl.int64)
        head_block = tl.program_id(1)
        head_rows = HALF * 32 + tl.arange(0, 32)
        heads = batch * H + head_block * 64 + head_rows
        col = tl.arange(0, 512)
        group = tl.arange(0, 8)
        q = tl.load(Q + heads[:, None] * 448 + col[None, :], col[None, :] < 448, 0.0)
        scales = tl.load(
            QS + heads[:, None] * 7 + group[None, :], group[None, :] < 7, 0.0
        )
        unit = tl.max(scales, 1)
        # Only exact power-of-two gains of one, one half, or one quarter
        # use the fast FP16 rescale. Other rows are recomputed before return.
        scale_delta = unit[:, None].to(tl.uint32, bitcast=True) - scales.to(
            tl.uint32, bitcast=True
        )
        needs_repair = tl.max(
            (
                (group[None, :] < 7)
                & (((scale_delta & 0x007FFFFF) != 0) | (scale_delta > 0x01000000))
            ).to(tl.int32),
            1,
        )
        needs_repair |= (unit < 4.70197740328915e-38).to(tl.int32)
        flag = tle.gpu.local_ptr(repair_flags, (1 + HALF,))
        tl.store(flag, tl.max(needs_repair, 0))
        gain = (
            ((15 - tl.minimum(scale_delta >> 23, 2)) << 10)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
        )
        normalized = (
            q.to(tl.float16)
            * tl.broadcast_to(gain[:, :, None], (32, 8, 64)).reshape(32, 512)
        ).to(tl.float8e4nv)
        tl.store(
            tle.gpu.local_ptr(tle_subslice(query_tile, [HALF * 32, 0], [32, 512])),
            normalized,
        )
        tl.store(tle.gpu.local_ptr(query_scale, (head_rows,)), unit)
        cd = tl.arange(0, 64)
        tl.store(
            tle.gpu.local_ptr(tle_subslice(query_rope, [HALF * 32, 0], [32, 64])),
            tl.load(QR + heads[:, None] * 64 + cd[None, :]),
        )
        tle.gpu.barrier_wait(query_ready[0])
        key = key_tiles.slot(HALF)
        tle.gpu.barrier_wait(panel_barriers[0], phaseIdx=0)
        score = tle.gpu.wgmma(
            tle_subslice(query_tile, [0, 0], [64, 128]),
            tle_subslice(key, [0, 0], [64, 128]),
            trans_b=True,
        )
        tle.gpu.barrier_wait(panel_barriers[1], phaseIdx=0)
        score = tle.gpu.wgmma(
            tle_subslice(query_tile, [0, 128], [64, 128]),
            tle_subslice(key, [0, 128], [64, 128]),
            score,
            trans_b=True,
        )
        tle.gpu.barrier_wait(panel_barriers[2], phaseIdx=0)
        score = tle.gpu.wgmma(
            tle_subslice(query_tile, [0, 256], [64, 128]),
            tle_subslice(key, [0, 256], [64, 128]),
            score,
            trans_b=True,
        )
        tle.gpu.barrier_wait(panel_barriers[3], phaseIdx=0)
        score = tle.gpu.wgmma(
            tle_subslice(query_tile, [0, 384], [64, 64]),
            tle_subslice(key, [0, 384], [64, 64]),
            score,
            trans_b=True,
        )
        rope = tle.gpu.wgmma(query_rope, key_rope.slot(HALF), trans_b=True)
        score = tle.gpu.wgmma_wait(0, score)
        rope = tle.gpu.wgmma_wait(0, rope)
        key_unit = tl.load(tle.gpu.local_ptr(key_scales.slot(HALF)))
        query_unit = tl.load(tle.gpu.local_ptr(query_scale))
        logits = tl.where(
            key_unit[None, :] >= 0.0,
            (score * query_unit[:, None] * key_unit[None, :] + rope)
            * (SCALE * 1.4426950408889634),
            -float("inf"),
        )
        half_max = tl.max(logits, 1)
        tl.store(tle.gpu.local_ptr(row_maxima.slot(HALF)), half_max)
        tle.gpu.barrier_wait(max_ready[0])
        row_max = tl.maximum(
            half_max, tl.load(tle.gpu.local_ptr(row_maxima.slot(1 - HALF)))
        )
        p = tl.exp2(logits - tl.where(row_max == -float("inf"), 0.0, row_max)[:, None])
        tl.store(tle.gpu.local_ptr(row_sums.slot(HALF)), tl.sum(p, 1))
        k0 = tl.load(tle.gpu.local_ptr(key_scales.slot(0)))
        k1 = tl.load(tle.gpu.local_ptr(key_scales.slot(1)))
        peak = tl.max(tl.maximum(k0, k1), 0)
        floor = tl.min(
            tl.minimum(
                tl.where(k0 > 0.0, k0, float("inf")),
                tl.where(k1 > 0.0, k1, float("inf")),
            ),
            0,
        )
        if HALF == 0:
            # Retain the BF16 reference path for wide scales and large logits.
            large_scores = (
                tl.max(query_unit, 0) * peak * tl.abs(SCALE * 1.4426950408889634)
                > QK_RECOMPUTE_THRESHOLD
            )
            tiny_scale = (peak > 0.0) & (peak < (448.0 * 1.17549435e-38))
            tl.store(flag, 1, mask=(peak > 4.0 * floor) | large_scores | tiny_scale)
        scale = peak * (1.0 / 448.0)
        scale = tl.where(scale >= 1.17549435e-38, scale, 1.0)
        publish_p_fp8_sw64_coupled_stmatrix(
            fp8_prob.slot(HALF),
            p * (tl.maximum(key_unit, 0.0)[None, :] * model1_reciprocal(scale)),
        )
        tl.store(tle.gpu.local_ptr(bf16_prob.slot(HALF)), p.to(tl.bfloat16))
        tle.gpu.barrier_wait(values_ready[0])
        denominator = tl.load(tle.gpu.local_ptr(row_sums.slot(0))) + tl.load(
            tle.gpu.local_ptr(row_sums.slot(1))
        )
        inverse = model1_reciprocal(tl.where(denominator > 0.0, denominator, 1.0))
        rows = batch * H + head_block * 64 + tl.arange(0, 64)
        if OUTPUT_TILES == 1:
            model1_value_band(
                fp8_prob,
                transposed_values,
                Out,
                batch,
                head_block,
                inverse,
                scale,
                H,
                HALF * 256,
                128,
            )
            model1_value_band(
                fp8_prob,
                transposed_values,
                Out,
                batch,
                head_block,
                inverse,
                scale,
                H,
                HALF * 256 + 128,
                64 if HALF == 1 else 128,
            )
        else:
            for band in tl.static_range(4):
                if tl.program_id(2) == band:
                    if band < 3 or HALF == 0:
                        model1_value_band(
                            fp8_prob,
                            transposed_values,
                            Out,
                            batch,
                            head_block,
                            inverse,
                            scale,
                            H,
                            band * 128 + HALF * 64,
                            64,
                        )
        if HALF == 1:
            if OUTPUT_TILES == 1 or tl.program_id(2) == 3:
                tail = tle.gpu.wgmma(bf16_prob.slot(0), key_rope.slot(0))
                tail = tle.gpu.wgmma(bf16_prob.slot(1), key_rope.slot(1), tail)
                tail = tle.gpu.wgmma_wait(0, tail)
                model1_store64(
                    tail, inverse, Out, batch * H + head_block * 64, 448, False
                )
        else:
            if OUTPUT_TILES == 1 or tl.program_id(2) == 0:
                tl.store(
                    L + rows,
                    tl.where(
                        denominator > 0.0,
                        row_max * 0.6931471805599453 + tl.log(denominator),
                        float("inf"),
                    ),
                )
        tle.gpu.barrier_wait(output_ready[0])

    @triton.jit
    def model1_key_pair(
        Q,
        QR,
        QS,
        Cache,
        Idx,
        Out,
        L,
        H: tl.constexpr,
        N: tl.constexpr,
        STRIDE: tl.constexpr,
        SCALE: tl.constexpr,
        OUTPUT_TILES: tl.constexpr,
        KEY_SPLITS: tl.constexpr,
    ):
        if KEY_SPLITS > 1:
            split_offset = (
                tl.program_id(0).to(tl.int64) * (KEY_SPLITS - 1)
                + tl.program_id(2) // OUTPUT_TILES
            )
            Idx = Idx + split_offset * 128
            Out = Out + split_offset * H * 512
            L = L + split_offset * H
        query_tile = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
        query_rope = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
        query_scale = tle.gpu.alloc(
            [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        key_tiles = tle.gpu.alloc([2, 64, 512], tl.float8e4nv, scope=tle.gpu.smem)
        key_rope = tle.gpu.alloc([2, 64, 64], tl.bfloat16, scope=tle.gpu.smem)
        key_scales = tle.gpu.alloc(
            [2, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        transposed_values = tle.gpu.alloc(
            [2, 512, 64], tl.float8e4nv, scope=tle.gpu.smem
        )
        fp8_prob = tle.gpu.alloc([2, 64, 64], tl.float8e4nv, scope=tle.gpu.smem)
        bf16_prob = tle.gpu.alloc([2, 64, 64], tl.bfloat16, scope=tle.gpu.smem)
        row_maxima = tle.gpu.alloc(
            [2, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        row_sums = tle.gpu.alloc(
            [2, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        repair_flags = tle.gpu.alloc(
            [4], tl.int32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        panel_barriers = tle.gpu.alloc_barriers(4, arrive_count=8)
        # Two 128-thread consumers exchange Q and row maxima. Together
        # with the producer they publish PV operands before the output phase.
        query_ready = sparse_named_barriers(1, 256, 16)
        max_ready = sparse_named_barriers(1, 256, 17)
        values_ready = sparse_named_barriers(1, 512, 18)
        output_ready = sparse_named_barriers(1, 256, 19)
        consumers = (
            Q,
            QR,
            QS,
            Out,
            L,
            repair_flags,
            query_tile,
            query_rope,
            query_scale,
            key_tiles,
            key_rope,
            key_scales,
            transposed_values,
            fp8_prob,
            bf16_prob,
            row_maxima,
            row_sums,
            panel_barriers,
            query_ready,
            max_ready,
            values_ready,
            output_ready,
        )
        constants = (
            H,
            SCALE,
            OUTPUT_TILES,
        )
        tle.gpu.warp_specialize(
            [
                (model1_compute_key_half, consumers + (0,) + constants),
                (model1_compute_key_half, consumers + (1,) + constants),
                (
                    model1_load_key_pair,
                    (
                        Cache,
                        Idx,
                        repair_flags,
                        key_tiles,
                        key_rope,
                        key_scales,
                        transposed_values,
                        panel_barriers,
                        values_ready,
                        N,
                        STRIDE,
                        OUTPUT_TILES,
                    ),
                ),
            ],
            [4, 8],
            [160, 64],
        )
        # Both consumers have finished writing these columns. The default
        # partition repairs only this CTA's output slice when a scale was unsafe.
        risk = (
            tl.load(tle.gpu.local_ptr(repair_flags, (0,)))
            | tl.load(tle.gpu.local_ptr(repair_flags, (1,)))
            | tl.load(tle.gpu.local_ptr(repair_flags, (2,)))
        )
        if risk:
            batch = tl.program_id(0).to(tl.int64)
            head_block = tl.program_id(1)
            for group in range(4):
                repair_heads(
                    Q,
                    QR,
                    QS,
                    Cache.to(tl.pointer_type(tl.float8e4nv)),
                    Cache.to(tl.pointer_type(tl.uint8)),
                    Cache.to(tl.pointer_type(tl.bfloat16)),
                    Idx,
                    Out,
                    L,
                    batch,
                    head_block * 4 + group,
                    N,
                    128,
                    H,
                    STRIDE,
                    SCALE,
                    OUTPUT_TILES,
                )

    def model1_attention(
        q: torch.Tensor,
        q_rope: torch.Tensor,
        q_scale: torch.Tensor,
        packed_cache: torch.Tensor,
        indices: torch.Tensor,
        *,
        tokens: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, heads, width = q.shape
        pages = packed_cache.shape[0]
        topk = indices.shape[-1]
        if batch == 0 or heads not in (64, 128) or width != 448 or topk == 0:
            raise ValueError(
                "Model1 requires nonempty queries, 64/128 heads and 448 FP8 dims"
            )
        if q.dtype != torch.float8_e4m3fn or not q.is_contiguous():
            raise ValueError("Query content must be contiguous E4M3")
        if (
            q_rope.shape != (batch, 1, heads, 64)
            or q_rope.dtype != torch.bfloat16
            or not q_rope.is_contiguous()
        ):
            raise ValueError("Query RoPE must be contiguous BF16 [batch, heads, 64]")
        if (
            q_scale.shape != (batch, 1, heads, 7)
            or q_scale.dtype != torch.float32
            or not q_scale.is_contiguous()
        ):
            raise ValueError("Query scales must be contiguous FP32 [batch, heads, 7]")
        if (
            packed_cache.shape != (pages, 64, 1, 584)
            or packed_cache.dtype != torch.uint8
        ):
            raise ValueError("Expected native V4 cache pages [pages, 64, 1, 584]")
        if (
            packed_cache.stride()[1:] != (584, 584, 1)
            or packed_cache.stride(0) % 16 != 0
            or packed_cache.data_ptr() % 16 != 0
            or packed_cache.stride(0) < 64 * 584
        ):
            raise ValueError(
                "V4 cache page data must be 16B-aligned and contiguous within a page"
            )
        if (
            indices.shape != (batch, 1, topk)
            or indices.dtype != torch.int32
            or not indices.is_contiguous()
        ):
            raise ValueError("Expected contiguous int32 indices [batch, 1, topk]")
        if not 0 < tokens <= pages * 64:
            raise ValueError("Token count is outside the allocated cache")
        if q.device.type != "cuda" or any(
            quantized_values.device != q.device
            for quantized_values in (q_rope, q_scale, packed_cache, indices)
        ):
            raise ValueError("Model1 tensors must share a CUDA device")

        if topk == 128 or (topk == 640 and batch <= 16):
            # Split a short query over output bands; a 640-key query
            # instead supplies enough CTAs by splitting its selected keys.
            out = q_rope.new_empty((batch, 1, heads, 512))
            lse = q_scale.new_empty((batch, heads, 1))
            key_splits = topk // 128
            output_tiles = 4 if batch <= 16 and key_splits == 1 else 1
            part = (
                out
                if key_splits == 1
                else out.new_empty((batch, key_splits, heads, 512))
            )
            stats = (
                lse if key_splits == 1 else lse.new_empty((batch, key_splits, heads))
            )
            model1_key_pair[(batch, heads // 64, output_tiles * key_splits)](
                q,
                q_rope,
                q_scale,
                packed_cache,
                indices,
                part,
                stats,
                heads,
                tokens,
                packed_cache.stride(0),
                scale,
                output_tiles,
                key_splits,
                num_warps=4,
                ptx_options="--register-usage-level=6",
                num_stages=1,
            )
            if key_splits > 1:
                merge[(batch, heads, 1)](
                    part, stats, out, lse, heads, key_splits, 512, num_warps=2
                )
            return out, lse

        work = batch * (heads // 64)
        q, q_rope, q_scale = q[:, 0], q_rope[:, 0], q_scale[:, 0]
        fp8_keys, exponent_words, bf16_keys, page_stride, _, _ = cache_views(
            packed_cache
        )
        if topk >= 512 and tokens == pages * 64 and batch * topk >= 8 * tokens:
            # Amortize the conversion over repeated cache visits in this call.
            query = q.new_empty((batch, 1, heads, 512))
            cache = q.new_empty((pages, 64, 512))
            qs = q_scale.new_empty((batch, 1, heads, 1))
            ks = q_scale.new_empty((pages, 64, 1))
            query_bounds = q_scale.new_empty((batch, heads // 16))
            key_bounds = q_scale.new_empty((tokens,))
            key_range = indices.new_zeros((2,))
            model1_pack_rows[(batch * heads // 16,)](
                q,
                q_rope,
                q_scale,
                query,
                qs,
                query_bounds,
                key_range,
                batch * heads,
                0,
                False,
                16,
                num_warps=8,
            )
            model1_pack_rows[(triton.cdiv(tokens, 32),)](
                fp8_keys,
                bf16_keys,
                packed_cache,
                cache,
                ks,
                key_bounds,
                key_range,
                tokens,
                page_stride,
                True,
                32,
                num_warps=8,
            )
            out, lse = sparse_attention(
                query,
                q_rope[:, None, :, :0],
                cache,
                q_rope.new_empty((pages, 64, 0)),
                qs,
                ks,
                indices,
                scale,
            )
            model1_repair[(min(batch, 256),)](
                q,
                q_rope,
                q_scale,
                fp8_keys,
                packed_cache,
                bf16_keys,
                indices,
                out,
                lse,
                query_bounds,
                key_bounds,
                key_range,
                batch,
                tokens,
                topk,
                heads,
                page_stride,
                scale,
                PACKED=True,
                num_warps=8,
                num_stages=1,
            )
            return out, lse
        sm = get_num_sms(q.device)
        sm_limit = sm // work
        block = 64
        num_blocks = triton.cdiv(topk, block)
        target = max(1, sm_limit)
        if num_blocks >= 32 and sm < work < 4 * sm and 0 < work % sm <= sm // 4:
            splits = 2
        else:
            splits = (
                1
                if topk <= 128
                else triton.cdiv(num_blocks, triton.cdiv(num_blocks, target))
            )
        # Bound FP8 MMA accumulation before the FP32 split reduction.
        splits = max(splits, triton.cdiv(topk, 4096))
        wave = sm // (heads // 64)
        remainder = batch % wave
        tail_splits = 0
        if (
            128 < topk <= 640
            and splits == 1
            and wave < batch <= 8 * wave
            and 0 < remainder < wave // 2
        ):
            tail_splits = min(5, wave // remainder, num_blocks)
        if tail_splits > 1:
            prefix = batch - remainder
            grid_batch = prefix + remainder * tail_splits
            part = q_rope.new_empty((batch + remainder * tail_splits, heads, 512))
            stats = q_scale.new_empty((batch + remainder * tail_splits, heads))
            out, lse = part[:batch], stats[:batch]
        else:
            prefix, remainder, tail_splits = 0, 0, 0
            grid_batch = batch
            out = torch.empty(
                (batch, heads, 512), dtype=torch.bfloat16, device=q.device
            )
            lse = torch.empty((batch, heads), dtype=torch.float32, device=q.device)
            part = out if splits == 1 else out.new_empty((batch, splits, heads, 512))
            stats = lse if splits == 1 else lse.new_empty((batch, splits, heads))
        repair_flags = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
        flags, repair_state = repair_flags[1:], repair_flags[:1]
        exponent_bytes = exponent_words.view(torch.uint8)
        if splits > 1 and topk > 640:
            query = torch.empty_like(q)
            query_magnitude = q_scale.new_empty((batch, heads))
            normalize_query[(batch, heads // 4)](
                q, q_scale, query, query_magnitude, heads, num_warps=4
            )
        else:
            query, query_magnitude = q, q_scale
        model1_fp8[(grid_batch, heads // 64, splits)](
            query,
            q_rope,
            query_magnitude,
            exponent_bytes,
            indices,
            part,
            stats,
            flags,
            repair_state,
            tokens,
            topk,
            heads,
            page_stride,
            scale,
            splits,
            block,
            2,
            prefix,
            remainder,
            tail_splits,
            ptx_options="--register-usage-level=2",
            num_warps=4,
            num_stages=1,
        )
        if tail_splits:
            merge[(remainder, heads, 1)](
                part[batch:],
                stats[batch:],
                out[prefix:],
                lse[prefix:],
                heads,
                tail_splits,
                512,
                num_warps=2,
            )
        elif splits > 1:
            d = 128 if batch <= 4 else 512
            merge[(batch, heads, 512 // d)](
                part,
                stats,
                out,
                lse,
                heads,
                splits,
                d,
                num_warps=2,
            )
        model1_repair[(min(batch, 64),)](
            q,
            q_rope,
            q_scale,
            fp8_keys,
            exponent_bytes,
            bf16_keys,
            indices,
            out,
            lse,
            flags,
            repair_state,
            repair_state,
            batch,
            tokens,
            topk,
            heads,
            page_stride,
            scale,
            num_warps=8,
            num_stages=1,
        )
        return out[:, None], lse[:, :, None]

    SPARSE_LAUNCH_CACHE_LIMIT = 128

    SPARSE_LAUNCH_CACHE: dict[
        tuple, tuple[Callable[..., None], tuple[int | bool, ...]]
    ] = {}

    def launch_sparse_entry(
        entry: LibEntry,
        grid: tuple[int, ...] | Callable[[dict[str, object]], tuple[int, ...]],
        *arguments: torch.Tensor | int | float | bool | None,
        **launch_kwargs: int | bool,
    ) -> None:
        # Retain dynamic FlagTune hooks; ordinary entries tune once per exact ABI.
        if entry._has_flagtune_tuner:
            entry[grid](*arguments, **launch_kwargs)
            return
        signature = tuple(
            (
                (argument.dtype, argument.data_ptr() % entry.divisibility == 0)
                if isinstance(argument, torch.Tensor)
                else (type(argument), argument)
            )
            for argument in arguments
        )
        static_grid = None if callable(grid) else grid
        key = (
            entry,
            torch.cuda.current_device(),
            static_grid,
            signature,
            tuple(launch_kwargs.items()),
        )
        prepared = SPARSE_LAUNCH_CACHE.get(key)
        if prepared is None:
            kernel, constants = entry[grid](*arguments, **launch_kwargs)
            trailing = tuple(
                (
                    launch_kwargs[param.name]
                    if param.name in launch_kwargs
                    else constants[param.name]
                )
                for param in entry.jit_function.params[len(arguments) :]
            )
            if callable(grid):
                resolved_grid = grid(
                    {
                        **dict(zip(entry.arg_names, arguments)),
                        **launch_kwargs,
                        **constants,
                    }
                )
            else:
                resolved_grid = grid
            while len(SPARSE_LAUNCH_CACHE) >= SPARSE_LAUNCH_CACHE_LIMIT:
                SPARSE_LAUNCH_CACHE.clear()
            # Retain launch metadata without holding input or output Tensor storage.
            SPARSE_LAUNCH_CACHE[key] = (kernel[resolved_grid], trailing)
            return
        runner, trailing = prepared
        runner(*arguments, *trailing)

    def sparse_attention(
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_cache_lora: torch.Tensor,
        k_cache_rope: Optional[torch.Tensor],
        q_scale: torch.Tensor,
        k_scale: Optional[torch.Tensor],
        indices: torch.Tensor,
        softmax_scale: Optional[float] = None,
        attn_sink: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        *,
        cache_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            q_nope.device.type != "cuda"
            or torch.cuda.get_device_capability(q_nope.device)[0] != 9
        ):
            raise NotImplementedError("FP8 sparse MLA requires NVIDIA Hopper CUDA")
        if k_cache_lora.dtype == torch.uint8:
            if k_cache_rope is not None or k_scale is not None:
                raise ValueError(
                    "Native V4 cache pages already contain RoPE and scales"
                )
            if attn_sink is not None or topk_length is not None:
                raise NotImplementedError("Native V4 uses the full supplied index row")
            if (
                q_nope.ndim != 4
                or q_nope.shape[1] != 1
                or q_rope.ndim != 4
                or q_scale.ndim != 4
            ):
                raise ValueError(
                    "Expected query components with singleton query length"
                )
            if q_rope.shape[1] != 1 or q_scale.shape[1] != 1:
                raise ValueError("Expected singleton RoPE and scale query length")
            return model1_attention(
                q_nope,
                q_rope,
                q_scale,
                k_cache_lora,
                indices,
                tokens=(
                    k_cache_lora.shape[0] * 64 if cache_tokens is None else cache_tokens
                ),
                scale=512**-0.5 if softmax_scale is None else float(softmax_scale),
            )
        if cache_tokens is not None or k_cache_rope is None or k_scale is None:
            raise ValueError("Separate V32 pages require explicit RoPE and scales")
        if (
            q_nope.dtype != torch.float8_e4m3fn
            or k_cache_lora.dtype != torch.float8_e4m3fn
        ):
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
        if (
            q_rope.ndim != 4
            or q_rope.shape[:3] != (batch, 1, heads)
            or q_rope.shape[-1] not in (0, 64)
        ):
            raise ValueError("q_rope must have shape [batch, 1, heads, 0 or 64]")
        rope_dim = q_rope.shape[-1]
        has_rope = rope_dim != 0
        if k_cache_lora.shape != (pages, 64, 512) or k_cache_rope.shape != (
            pages,
            64,
            rope_dim,
        ):
            raise ValueError(
                "Caches require page size 64 and the matching query RoPE dimension"
            )
        if indices.shape != (batch, 1, topk) or indices.dtype != torch.int32:
            raise ValueError("indices must be int32 [batch, 1, topk]")
        if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
            raise TypeError("q_scale and k_scale must have dtype float32")
        if q_scale.shape != (batch, 1, heads, 1) or k_scale.shape != (pages, 64, 1):
            raise ValueError(
                "Scale shapes must be [batch, 1, heads, 1] and [pages, 64, 1]"
            )
        for tensor in (q_nope, q_rope, k_cache_lora, k_cache_rope):
            if tensor.stride(-1) != 1:
                raise ValueError(
                    "NoPE and RoPE must be contiguous in the last dimension"
                )
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
        softmax_scale = (
            (512 + rope_dim) ** -0.5 if softmax_scale is None else float(softmax_scale)
        )
        num_sms = get_num_sms(q_nope.device)
        head_groups = batch * (heads // 64)
        desired_splits = max(1, num_sms // head_groups)
        keys_per_split = 64 if batch <= 16 else 256
        max_splits = min(desired_splits, 32, max(1, topk // keys_per_split))
        splits = 1 << (max_splits.bit_length() - 1)
        # Bound FP8 tensor-core accumulation; split outputs are merged in FP32.
        max_accumulation_keys = 4096
        splits = max(
            splits, triton.next_power_of_2(triton.cdiv(topk, max_accumulation_keys))
        )
        flag_heads = heads // 64
        repair_flags = torch.empty(
            (batch, flag_heads, splits), device=q_nope.device, dtype=torch.int32
        )
        if splits > 1:
            partial = torch.empty(
                (batch, splits, heads, 512),
                device=q_nope.device,
                dtype=torch.bfloat16,
            )
            stats = torch.empty(
                (batch, splits, heads, 2), device=q_nope.device, dtype=torch.float32
            )
        else:
            partial, stats = output, lse
        common_tensors = (
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
        )
        common_strides = (
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
        )
        b_bucket = 0 if head_groups * splits <= 128 else 1
        launch_sparse_entry(
            sparse_fp8_warp_specialized,
            (batch * (heads // 64), 1, splits),
            *common_tensors,
            partial,
            stats,
            *common_strides,
            b_bucket,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            topk_length is not None,
            splits,
            attn_sink is not None,
            repair_flags,
            HAS_ROPE=has_rope,
            CAN_ASYNC_ROPE=(
                has_rope
                and k_cache_rope.data_ptr() % 16 == 0
                and (k_cache_rope.stride(0) * 2) % 16 == 0
                and (k_cache_rope.stride(1) * 2) % 16 == 0
            ),
            CAN_ASYNC_NOPE=(
                k_cache_lora.data_ptr() % 16 == 0
                and k_cache_lora.stride(0) % 16 == 0
                and k_cache_lora.stride(1) % 16 == 0
            ),
        )
        if splits == 1:
            launch_sparse_entry(
                sparse_fp8_repair,
                lambda meta: (
                    triton.cdiv(batch, meta["REPAIR_GROUP"]),
                    triton.cdiv(heads, meta["BLOCK_H"]),
                    splits,
                ),
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
                *common_strides[:8],
                *common_strides[12:],
                *common_strides[8:12],
                batch,
                heads,
                pages * 64,
                topk,
                softmax_scale,
                splits,
                attn_sink is not None,
                topk_length is not None,
                repair_flags,
                REPAIR_SPLITS=splits,
                HAS_ROPE=has_rope,
            )
        else:
            head_tile = 2 if splits == 2 else 4
            sparse_fp8_fast_merge[(batch, heads // head_tile)](
                *common_tensors,
                repair_flags,
                partial,
                stats,
                output,
                lse,
                *common_strides,
                heads,
                pages * 64,
                topk,
                softmax_scale,
                splits,
                topk_length is not None,
                attn_sink is not None,
                CHECK_REPAIR=True,
                HEAD_TILE=head_tile,
                HAS_ROPE=has_rope,
                num_warps=2,
            )
        return output, lse


def flash_mla_sparse_fwd_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: Optional[torch.Tensor],
    q_scale: torch.Tensor,
    k_scale: Optional[torch.Tensor],
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    *,
    cache_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode using the separate per-token cache format of dense MLA.

    q_nope [B, 1, H, 512] and k_cache_lora [P, 64, 512] are FP8 e4m3fn;
    q_rope [B, 1, H, R] and k_cache_rope [P, 64, R] are BF16.
    R is 0 for inputs without RoPE, or 64 for the RoPE component.
    Both NoPE and RoPE store values divided by the corresponding FP32 scale:
    q_scale [B, 1, H, 1] and k_scale [P, 64, 1]. This directly accepts
    quantize_q_ckv_per_token / quantize_k_ckv_per_token outputs from dense MLA.
    V is the dequantized 512-dimensional NoPE cache. H must be 64 or 128.

    indices [B, 1, topk] contains int32 physical token IDs (page * 64 + slot).
    Negative and out-of-range IDs are ignored. topk_length [B] optionally
    limits the number of entries read per request. attn_sink [H] affects
    output only. Inputs must be finite, with positive finite scales.

    Native V4 takes 448 contiguous FP8 query channels, seven FP32 scales and
    64 unscaled BF16 RoPE channels. Pass the uint8 [P,64,1,584] page view
    as k_cache_lora with k_cache_rope and k_scale set to None. cache_tokens
    optionally excludes padding in the last page.

    Returns BF16 output [B, 1, H, 512] and natural-log FP32 LSE [B, H, 1].
    Empty attention produces zero output and +inf LSE. Forward only.
    Sensitive rows use FP32 recomputation inline, during merge, or in the
    existing separate repair kernel, depending on the selected path.
    Requires FlagTree cross-dtype WGMMA support (PR #1001).
    """
    if sparse_attention is None:
        raise NotImplementedError(
            "FP8 sparse MLA requires Hopper and FlagTree GPU extensions"
        )

    return sparse_attention(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        indices,
        softmax_scale,
        attn_sink,
        topk_length,
        cache_tokens=cache_tokens,
    )
