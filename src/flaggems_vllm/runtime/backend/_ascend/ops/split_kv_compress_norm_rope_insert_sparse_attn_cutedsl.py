# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._ascend.ops.per_token_group_quant_fp8 import (
    _f32_to_fp8_e4m3fn,
)

# Ascend flattens all grid dimensions into a 16-bit coreDim.
_MAX_GRID = 65535


@triton.jit
def _compress_c128_strided(
    State,
    Request,
    Position,
    Slot,
    Table,
    Work,
    N: tl.constexpr,
    N_REQ: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    N_PAGES: tl.constexpr,
    STATE_PAGE: tl.constexpr,
    STATE_ROW: tl.constexpr,
    STATE_COL: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    WORK_ROW: tl.constexpr,
    WORK_COL: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_GATHER: tl.constexpr,
    COMPRESS_KERNEL: tl.constexpr,
):
    tiles: tl.constexpr = 512 // BLOCK_D
    first_row = tl.program_id(0).to(tl.int64) // tiles
    # Whole rows per wave preserve the feature tile selected by the inlined
    # kernel's program_id. Advance only the tensors indexed by token.
    for row in range(first_row, N, tl.num_programs(0) // tiles):
        base = row - first_row
        COMPRESS_KERNEL(
            State,
            Request + base * REQ_STRIDE,
            Position + base * POS_STRIDE,
            Slot + base * SLOT_STRIDE,
            Table,
            Work + base * WORK_ROW,
            N,
            N_REQ,
            TABLE_WIDTH,
            N_PAGES,
            STATE_PAGE,
            STATE_ROW,
            STATE_COL,
            REQ_STRIDE,
            POS_STRIDE,
            SLOT_STRIDE,
            TABLE_ROW,
            TABLE_COL,
            WORK_ROW,
            WORK_COL,
            BLOCK_D,
            PAGE_GATHER,
        )


@triton.jit
def _encode_group(x, FP8_MAX: tl.constexpr):
    if FP8_MAX < 448.0:
        x = tl.minimum(tl.maximum(x, 0.0 - FP8_MAX), FP8_MAX)
    else:
        x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    return _f32_to_fp8_e4m3fn(x)


@triton.jit
def _write_bf16(ptr, index, values):
    bits = values.to(tl.float32).to(tl.int32, bitcast=True)
    tl.store(ptr + index * 2, ((bits >> 16) & 255).to(tl.uint8))
    tl.store(ptr + index * 2 + 1, ((bits >> 24) & 255).to(tl.uint8))


@triton.jit
def _store_c128_kernel(
    Work,
    Position,
    Slot,
    Weight,
    Rotary,
    Cache,
    KvSlot,
    Scale,
    eps,
    N: tl.constexpr,
    WORK_ROW: tl.constexpr,
    WORK_COL: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    W_STRIDE: tl.constexpr,
    ROTARY_ROW: tl.constexpr,
    ROTARY_COL: tl.constexpr,
    ROTARY_LEN: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    N_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    FULL: tl.constexpr,
    FP8: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    for item in range(tl.program_id(0).to(tl.int64), N * 8, tl.num_programs(0)):
        token = item // 8
        group = (item % 8).to(tl.int32)
        slot = tl.load(Slot + token * SLOT_STRIDE)
        position = tl.load(Position + token * POS_STRIDE)
        dst = tl.load(KvSlot + token * KV_STRIDE)
        rotary_pos = position - 127
        active = (slot >= 0) & (position >= 127) & ((position & 127) == 127)
        active &= (dst >= 0) & (dst < N_PAGES * PAGE_SIZE) & (rotary_pos < ROTARY_LEN)
        if active:
            work = Work + token * WORK_ROW
            row = tl.load(work + tl.arange(0, 512).to(tl.int64) * WORK_COL)
            rrms = tl.rsqrt(tl.sum(row * row, 0) * (1.0 / 512.0) + eps)
            dst_row = dst % PAGE_SIZE
            page = Cache + (dst // PAGE_SIZE) * PAGE_BYTES
            payload = page + dst_row * ROW_BYTES
            scale_row = page + PAGE_SIZE * ROW_BYTES + dst_row * SCALE_DIM
            scale_inv = 1.0
            if HAS_SCALE:
                scale_inv = 1.0 / tl.load(Scale)
            if group == 7:
                pair = tl.arange(0, 32)
                even_col = (448 + pair * 2).to(tl.int64)
                odd_col = even_col + 1
                even = tl.load(work + even_col * WORK_COL) * rrms
                odd = tl.load(work + odd_col * WORK_COL) * rrms
                even *= tl.load(Weight + even_col * W_STRIDE).to(tl.float32)
                odd *= tl.load(Weight + odd_col * W_STRIDE).to(tl.float32)
                cs = Rotary + rotary_pos * ROTARY_ROW + pair.to(tl.int64) * ROTARY_COL
                cosine = tl.load(cs)
                sine = tl.load(cs + 32 * ROTARY_COL)
                real = (even * cosine - odd * sine).to(tl.bfloat16)
                imag = (even * sine + odd * cosine).to(tl.bfloat16)
                if FULL and FP8:
                    tl.store(
                        payload + even_col,
                        _encode_group(real.to(tl.float32) * scale_inv, FP8_MAX),
                    )
                    tl.store(
                        payload + odd_col,
                        _encode_group(imag.to(tl.float32) * scale_inv, FP8_MAX),
                    )
                else:
                    if FULL:
                        tail = payload + 896
                    else:
                        tail = payload + 448
                        tl.store(scale_row + 7, 0)
                    _write_bf16(tail, pair * 2, real)
                    _write_bf16(tail, pair * 2 + 1, imag)
            else:
                col = group * 64 + tl.arange(0, 64)
                x = tl.load(work + col.to(tl.int64) * WORK_COL) * rrms
                x *= tl.load(Weight + col.to(tl.int64) * W_STRIDE).to(tl.float32)
                rounded = x.to(tl.bfloat16)
                if FULL and not FP8:
                    _write_bf16(payload, col, rounded)
                else:
                    group_f32 = rounded.to(tl.float32)
                    if FULL:
                        scaled = group_f32 * scale_inv
                    else:
                        raw_scale = tl.maximum(
                            tl.max(tl.abs(group_f32), 0) / FP8_MAX, 1e-4
                        )
                        bits = raw_scale.to(tl.int32, bitcast=True)
                        code = ((bits + 0x7FFFFF) >> 23) & 255
                        inv = ((254 - code) << 23).to(tl.float32, bitcast=True)
                        scaled = group_f32 * inv
                        tl.store(scale_row + group, code.to(tl.uint8))
                    tl.store(payload + col, _encode_group(scaled, FP8_MAX))
