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

"""Register-level GEMMs for sparse expert batches."""

import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._thead.fused.marlin.formats import (
    EXPERT_ID_MASK,
    SHORT_TILE_FLAG,
    W8_HALF_TILE_BYTES,
    W8_TILE_BYTES,
    decode_fp8_pair,
    decode_int8_fragment_bf16,
    decode_w4_fragment_pair,
    load_activation_fragment_x2,
    load_matrix_x4,
    load_w8_packed_scales,
    mma_bf16_m16n8,
    mma_bf16_square,
    mx_scale_fragment,
    mx_scale_source,
    prefetch_square,
    prefetch_w4_square,
    prefetch_w8_tile,
    tle_async,
    w8_scale_fragment,
    wait_w8_tile,
)
from flaggems_vllm.utils import libentry

# Packed conversion instructions are adapted from FlagTree's PPU backend.
# Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


@triton.jit
def square_accumulate(
    weight_smem,
    activation_smem,
    weight_bytes,
    activations,
    scales,
    QN,
    reduction_size: tl.constexpr,
    row_tile,
    expert,
    n0,
    warp,
    SE,
    SG,
    SN,
    FP8: tl.constexpr,
    FAST: tl.constexpr,
    initial_dep=None,
    SCALE_PAIR: tl.constexpr = False,
):
    if initial_dep is None:
        copy_token = tl.full((256,), 0, tl.int32)
        copy_token = prefetch_square(
            weight_smem,
            activation_smem,
            weight_bytes,
            activations,
            QN,
            reduction_size,
            row_tile,
            0,
            copy_token,
        )
    else:
        copy_token = initial_dep
    c0 = tl.full((256,), 0.0, tl.float32)
    c1 = c0
    c2 = c0
    c3 = c0
    c4 = c0
    c5 = c0
    c6 = c0
    c7 = c0
    if SCALE_PAIR:
        tl.static_assert(FP8 and reduction_size == 256)
        scale0, scale1 = load_w8_packed_scales(scales + expert * SE + n0 * SN, SN, FAST)
    unroll: tl.constexpr = 2 if FP8 and reduction_size % 256 == 0 else 1
    for k_tile in tl.range(reduction_size // 128, loop_unroll_factor=unroll):
        if not SCALE_PAIR:
            scale0 = tl.load(scales + expert * SE + k_tile * SG + n0 * SN)
            scale1 = tl.load(scales + expert * SE + k_tile * SG + (n0 + 8) * SN)
        if k_tile + 1 < reduction_size // 128:
            copy_token = prefetch_square(
                weight_smem + (k_tile + 1) % 2 * W8_TILE_BYTES,
                activation_smem + (k_tile + 1) % 2 * 4096,
                weight_bytes,
                activations,
                QN,
                reduction_size,
                row_tile,
                k_tile + 1,
                copy_token,
            )
            copy_token = wait_w8_tile(copy_token, True)
        else:
            copy_token = wait_w8_tile(copy_token, False)
        s0 = w8_scale_fragment(scale0, k_tile, SCALE_PAIR, FP8 and FAST)
        s1 = w8_scale_fragment(scale1, k_tile, SCALE_PAIR, FP8 and FAST)
        qhere = weight_smem + k_tile % 2 * W8_TILE_BYTES
        ahere = activation_smem + k_tile % 2 * 4096
        for group in tl.static_range(4):
            (r0, r1, r2, r3) = load_matrix_x4(
                qhere + (warp // 4) * W8_HALF_TILE_BYTES,
                (warp % 4) * 16,
                64,
                group * 32,
                copy_token,
            )
            for part in tl.static_range(2):
                q0 = r0 if part == 0 else r1
                q8 = r2 if part == 0 else r3
                sub = group * 2 + part
                if FP8:
                    a0, a1, a2, a3 = decode_fp8_pair(q0, q8, s0, s1, FAST)
                    (b0, b1, b2, b3) = load_matrix_x4(
                        ahere + sub // 4 * 2048, 0, 16, sub % 4 * 32, copy_token
                    )
                else:
                    (b0, b1, b2, b3) = load_matrix_x4(
                        ahere + sub // 4 * 2048, 0, 16, sub % 4 * 32, copy_token
                    )
                    (a0, a1) = decode_int8_fragment_bf16(q0, s0, s0)
                    (a2, a3) = decode_int8_fragment_bf16(q8, s1, s1)
                (c0, c1, c2, c3, c4, c5, c6, c7) = mma_bf16_square(
                    a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7
                )
        tl.debug_barrier()
    return (c0, c1, c2, c3, c4, c5, c6, c7)


@libentry()
@triton.jit
def marlin_moe_gemm_grouped_square_kernel(
    activations,
    weight_bytes,
    output_tile,
    scales,
    router_weights,
    route_ids,
    block_experts,
    padded_count,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    VALID,
    QE,
    QN,
    SE,
    SG,
    SN,
    CM,
    CN,
    MUL: tl.constexpr,
    scale_flags,
    FP8: tl.constexpr,
    USE32: tl.constexpr = False,
    PREFETCH_BEFORE_SCALE: tl.constexpr = False,
    SCALE_PAIR: tl.constexpr = False,
    PREFETCH_ROUTES: tl.constexpr = False,
    PACK_WIDTH: tl.constexpr = False,
):
    tl.static_assert(num_columns % 128 == 0 and reduction_size % 128 == 0)
    pid = tl.program_id(0)
    row_tile = pid // (num_columns // 128)
    col_tile = pid % (num_columns // 128)
    if row_tile * 16 >= tl.load(padded_count):
        return
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    expert_word = tl.load(block_experts + row_tile)
    if PACK_WIDTH:
        expert = (expert_word & EXPERT_ID_MASK).to(offset_type)
        short_tile = (expert_word & SHORT_TILE_FLAG) != 0
    else:
        expert = expert_word.to(offset_type)
    thread = tl.arange(0, 256)
    lane = thread % 32
    warp = thread // 32
    lane_group = lane // 4
    lane_in_group = lane % 4
    n0 = col_tile * 128 + warp * 16 + lane_group
    qmem = tle_async.gpu.alloc(
        [32768],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    amem = tle_async.gpu.alloc(
        [2 * 16 * 128 * 2],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    weight_smem = tle_async.gpu.local_ptr(qmem, (0,)).to(tl.int64).to(tl.int32)
    activation_smem = tle_async.gpu.local_ptr(amem, (0,)).to(tl.int64).to(tl.int32)
    weight_base = weight_bytes + expert * QE + col_tile * 128 * QN
    inner_n = warp * 16 + lane_group
    if not PACK_WIDTH:
        short_tile = tl.load(route_ids + row_tile * 16 + 8) >= VALID
    if short_tile:
        if PREFETCH_BEFORE_SCALE:
            copy_token = tl.full((256,), 0, tl.int32)
            copy_token = prefetch_w8_tile(
                weight_smem,
                activation_smem,
                weight_base,
                activations,
                QN,
                reduction_size,
                row_tile * 2,
                0,
                copy_token,
                True,
            )
        else:
            copy_token = None
        if PREFETCH_ROUTES:
            early0 = tl.load(route_ids + row_tile * 16 + 2 * lane_in_group)
            early1 = tl.load(route_ids + row_tile * 16 + 2 * lane_in_group + 1)
            if MUL:
                w0 = tl.load(
                    router_weights + early0.to(offset_type), early0 < VALID, other=0
                ).to(tl.float32)
                w1 = tl.load(
                    router_weights + early1.to(offset_type), early1 < VALID, other=0
                ).to(tl.float32)
        if FP8:
            can_fold = (
                tl.load(scale_flags + expert) != 0 if scale_flags is not None else False
            )
            tile_inputs = (
                weight_smem,
                activation_smem,
                weight_base,
                activations,
                scales,
                QN,
            )
            tile_coordinates = (row_tile * 2, expert, n0, inner_n, SE, SG, SN)
            if can_fold:
                (c0, c1, c2, c3) = w8_narrow_accumulate(
                    *tile_inputs,
                    reduction_size,
                    *tile_coordinates,
                    True,
                    True,
                    IS_SPLIT_Q=True,
                    initial_dep=copy_token,
                    SCALE_PAIR=SCALE_PAIR,
                )
            else:
                (c0, c1, c2, c3) = w8_narrow_accumulate(
                    *tile_inputs,
                    reduction_size,
                    *tile_coordinates,
                    True,
                    False,
                    IS_SPLIT_Q=True,
                    initial_dep=copy_token,
                    SCALE_PAIR=SCALE_PAIR,
                )
        else:
            (c0, c1, c2, c3) = w8_narrow_accumulate(
                weight_smem,
                activation_smem,
                weight_base,
                activations,
                scales,
                QN,
                reduction_size,
                row_tile * 2,
                expert,
                n0,
                inner_n,
                SE,
                SG,
                SN,
                False,
                False,
                IS_SPLIT_Q=True,
            )
        if PREFETCH_ROUTES:
            token0 = early0.to(offset_type)
            token1 = early1.to(offset_type)
        else:
            token0 = tl.load(route_ids + row_tile * 16 + 2 * lane_in_group).to(
                offset_type
            )
            token1 = tl.load(route_ids + row_tile * 16 + 2 * lane_in_group + 1).to(
                offset_type
            )
        if MUL:
            if not PREFETCH_ROUTES:
                w0 = tl.load(router_weights + token0, token0 < VALID, other=0).to(
                    tl.float32
                )
                w1 = tl.load(router_weights + token1, token1 < VALID, other=0).to(
                    tl.float32
                )
        else:
            w0 = 1.0
            w1 = 1.0
        c0 *= w0
        c2 *= w0
        c1 *= w1
        c3 *= w1
        tl.store(
            output_tile + token0 * CM + n0 * CN,
            c0.to(output_tile.dtype.element_ty),
            token0 < VALID,
        )
        tl.store(
            output_tile + token1 * CM + n0 * CN,
            c1.to(output_tile.dtype.element_ty),
            token1 < VALID,
        )
        tl.store(
            output_tile + token0 * CM + (n0 + 8) * CN,
            c2.to(output_tile.dtype.element_ty),
            token0 < VALID,
        )
        tl.store(
            output_tile + token1 * CM + (n0 + 8) * CN,
            c3.to(output_tile.dtype.element_ty),
            token1 < VALID,
        )
        return
    if PREFETCH_BEFORE_SCALE:
        copy_token = tl.full((256,), 0, tl.int32)
        copy_token = prefetch_square(
            weight_smem,
            activation_smem,
            weight_base,
            activations,
            QN,
            reduction_size,
            row_tile,
            0,
            copy_token,
        )
    else:
        copy_token = None
    if PREFETCH_ROUTES:
        tokens = ()
        weights = ()
        for j in tl.static_range(4):
            token = tl.load(route_ids + row_tile * 16 + lane_in_group + 4 * j)
            tokens += (token,)
            if MUL:
                weight = tl.load(
                    router_weights + token.to(offset_type), token < VALID, other=0
                ).to(tl.float32)
                weights += (weight,)
    if FP8:
        can_fold = (
            tl.load(scale_flags + expert) != 0 if scale_flags is not None else False
        )
        tile_inputs = (
            weight_smem,
            activation_smem,
            weight_base,
            activations,
            scales,
            QN,
        )
        tile_coordinates = (row_tile, expert, n0, warp, SE, SG, SN)
        if can_fold:
            c = square_accumulate(
                *tile_inputs,
                reduction_size,
                *tile_coordinates,
                True,
                True,
                initial_dep=copy_token,
                SCALE_PAIR=SCALE_PAIR,
            )
        else:
            c = square_accumulate(
                *tile_inputs,
                reduction_size,
                *tile_coordinates,
                True,
                False,
                initial_dep=copy_token,
                SCALE_PAIR=SCALE_PAIR,
            )
    else:
        c = square_accumulate(
            weight_smem,
            activation_smem,
            weight_base,
            activations,
            scales,
            QN,
            reduction_size,
            row_tile,
            expert,
            n0,
            warp,
            SE,
            SG,
            SN,
            False,
            False,
        )
    for j in tl.static_range(4):
        if PREFETCH_ROUTES:
            token = tokens[j].to(offset_type)
        else:
            token = tl.load(route_ids + row_tile * 16 + lane_in_group + 4 * j).to(
                offset_type
            )
        v0 = c[j]
        v1 = c[j + 4]
        if MUL:
            if PREFETCH_ROUTES:
                w = weights[j]
            else:
                w = tl.load(router_weights + token, token < VALID, other=0).to(
                    tl.float32
                )
            v0 *= w
            v1 *= w
        tl.store(
            output_tile + token * CM + n0 * CN,
            v0.to(output_tile.dtype.element_ty),
            token < VALID,
        )
        tl.store(
            output_tile + token * CM + (n0 + 8) * CN,
            v1.to(output_tile.dtype.element_ty),
            token < VALID,
        )


@libentry()
@triton.jit
def raw_w4_square_kernel(
    activations,
    weight_bytes,
    output_tile,
    scales,
    router_weights,
    route_ids,
    block_experts,
    padded_count,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    VALID,
    QE,
    QN,
    SE,
    SG,
    SN,
    CM,
    CN,
    MUL: tl.constexpr,
    MX: tl.constexpr,
    PACKED_SCALE: tl.constexpr,
    ROWS: tl.constexpr = 16,
    USE32: tl.constexpr = False,
):
    tl.static_assert(num_columns % 64 == 0 and reduction_size % 128 == 0)
    tl.static_assert(ROWS == 8 or ROWS == 16)
    pid = tl.program_id(0)
    row_tile = pid // (num_columns // 64)
    col_tile = pid % (num_columns // 64)
    if row_tile * ROWS >= tl.load(padded_count):
        return
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    expert = tl.load(block_experts + row_tile).to(offset_type)
    thread = tl.arange(0, 128)
    lane = thread % 32
    warp = thread // 32
    lane_group = lane // 4
    lane_in_group = lane % 4
    n0 = col_tile * 64 + warp * 16 + lane_group
    qm = tle_async.gpu.alloc(
        [2 * 64 * 64],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    am = tle_async.gpu.alloc(
        [ROWS * 512],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    weight_smem = tle_async.gpu.local_ptr(qm, (0,)).to(tl.int64).to(tl.int32)
    activation_smem = tle_async.gpu.local_ptr(am, (0,)).to(tl.int64).to(tl.int32)
    weight_base = weight_bytes + expert * QE + col_tile * 64 * QN
    copy_token = tl.full((128,), 0, tl.int32)
    copy_token = prefetch_w4_square(
        weight_smem,
        activation_smem,
        weight_base,
        activations,
        QN,
        reduction_size,
        row_tile,
        0,
        copy_token,
        ROWS,
    )
    c0 = tl.full((128,), 0.0, tl.float32)
    c1 = c0
    c2 = c0
    c3 = c0
    c4 = c0
    c5 = c0
    c6 = c0
    c7 = c0
    for k_tile in range(reduction_size // 128):
        if MX:
            scale_ptr = scales + expert * SE + (k_tile * 4) * SG + n0 * SN
            src0 = mx_scale_source(scale_ptr, PACKED_SCALE)
            src1 = mx_scale_source(scale_ptr + 8 * SN, PACKED_SCALE)
        if not MX and ROWS == 16:
            scale0 = tl.load(scales + expert * SE + k_tile * SG + n0 * SN)
            scale1 = tl.load(scales + expert * SE + k_tile * SG + (n0 + 8) * SN)
        if k_tile + 1 < reduction_size // 128:
            copy_token = prefetch_w4_square(
                weight_smem + ((k_tile + 1) % 2) * 4096,
                activation_smem + ((k_tile + 1) % 2) * (ROWS * 256),
                weight_base,
                activations,
                QN,
                reduction_size,
                row_tile,
                k_tile + 1,
                copy_token,
                ROWS,
            )
            copy_token = wait_w8_tile(copy_token, True)
        else:
            copy_token = wait_w8_tile(copy_token, False)
        qhere = weight_smem + (k_tile % 2) * 4096
        ahere = activation_smem + (k_tile % 2) * (ROWS * 256)
        if not MX:
            if ROWS == 8:
                scale0 = tl.load(scales + expert * SE + k_tile * SG + n0 * SN)
                scale1 = tl.load(scales + expert * SE + k_tile * SG + (n0 + 8) * SN)
            s0 = scale0.to(tl.uint16, bitcast=True).to(tl.uint32)
            s1 = scale1.to(tl.uint16, bitcast=True).to(tl.uint32)
            s0 = s0 | (s0 << 16)
            s1 = s1 | (s1 << 16)
        for group in tl.static_range(2):
            r0, r1, r2, r3 = load_matrix_x4(
                qhere, warp * 16, 64, group * 32, copy_token
            )
            for chunk in tl.static_range(2):
                if MX:
                    s0 = mx_scale_fragment(src0, SG, group * 2 + chunk, PACKED_SCALE)
                    s1 = mx_scale_fragment(src1, SG, group * 2 + chunk, PACKED_SCALE)
                q0 = r0 if chunk == 0 else r1
                q8 = r2 if chunk == 0 else r3
                for part in tl.static_range(2):
                    a0, a1, a2, a3 = decode_w4_fragment_pair(q0, q8, s0, s1, part, MX)
                    sub = group * 4 + chunk * 2 + part
                    if ROWS == 8:
                        b0, b1 = load_activation_fragment_x2(
                            ahere + (sub // 4) * 1024, (sub % 4) * 32, copy_token
                        )
                        c0, c1, c2, c3 = mma_bf16_m16n8(
                            a0, a2, a1, a3, b0, b1, c0, c1, c2, c3
                        )
                    else:
                        b0, b1, b2, b3 = load_matrix_x4(
                            ahere + (sub // 4) * 2048, 0, 16, (sub % 4) * 32, copy_token
                        )
                        c0, c1, c2, c3, c4, c5, c6, c7 = mma_bf16_square(
                            a0,
                            a1,
                            a2,
                            a3,
                            b0,
                            b1,
                            b2,
                            b3,
                            c0,
                            c1,
                            c2,
                            c3,
                            c4,
                            c5,
                            c6,
                            c7,
                        )
        tl.debug_barrier()
    if ROWS == 8:
        token0 = tl.load(route_ids + row_tile * 8 + 2 * lane_in_group).to(offset_type)
        token1 = tl.load(route_ids + row_tile * 8 + 2 * lane_in_group + 1).to(
            offset_type
        )
        if MUL:
            w0 = tl.load(router_weights + token0, token0 < VALID, other=0).to(
                tl.float32
            )
            w1 = tl.load(router_weights + token1, token1 < VALID, other=0).to(
                tl.float32
            )
            c0 *= w0
            c2 *= w0
            c1 *= w1
            c3 *= w1
        tl.store(
            output_tile + token0 * CM + n0 * CN,
            c0.to(output_tile.dtype.element_ty),
            token0 < VALID,
        )
        tl.store(
            output_tile + token1 * CM + n0 * CN,
            c1.to(output_tile.dtype.element_ty),
            token1 < VALID,
        )
        tl.store(
            output_tile + token0 * CM + (n0 + 8) * CN,
            c2.to(output_tile.dtype.element_ty),
            token0 < VALID,
        )
        tl.store(
            output_tile + token1 * CM + (n0 + 8) * CN,
            c3.to(output_tile.dtype.element_ty),
            token1 < VALID,
        )
    else:
        c = (c0, c1, c2, c3, c4, c5, c6, c7)
        for j in tl.static_range(4):
            token = tl.load(route_ids + row_tile * 16 + lane_in_group + 4 * j).to(
                offset_type
            )
            v0 = c[j]
            v1 = c[j + 4]
            if MUL:
                w = tl.load(router_weights + token, token < VALID, other=0).to(
                    tl.float32
                )
                v0 *= w
                v1 *= w
            tl.store(
                output_tile + token * CM + n0 * CN,
                v0.to(output_tile.dtype.element_ty),
                token < VALID,
            )
            tl.store(
                output_tile + token * CM + (n0 + 8) * CN,
                v1.to(output_tile.dtype.element_ty),
                token < VALID,
            )


@triton.jit
def w8_narrow_accumulate(
    shared,
    ashared,
    qbase,
    activations,
    scales,
    QN,
    reduction_size: tl.constexpr,
    row_tile,
    expert,
    n0,
    inner_n,
    SE,
    SG,
    SN,
    FP8: tl.constexpr,
    FAST: tl.constexpr,
    IS_SPLIT_Q: tl.constexpr = False,
    initial_dep=None,
    SCALE_PAIR: tl.constexpr = False,
):
    # Two K=128 buffers overlap AIU transfer and eight 16-K MMAs.
    if initial_dep is None:
        copy_token = tl.full((256,), 0, tl.int32)
        copy_token = prefetch_w8_tile(
            shared,
            ashared,
            qbase,
            activations,
            QN,
            reduction_size,
            row_tile,
            0,
            copy_token,
            IS_SPLIT_Q,
        )
    else:
        copy_token = initial_dep
    c0 = tl.full((256,), 0.0, tl.float32)
    c1 = c0
    c2 = c0
    c3 = c0
    if SCALE_PAIR:
        tl.static_assert(FP8 and reduction_size == 256)
        scale0, scale1 = load_w8_packed_scales(scales + expert * SE + n0 * SN, SN, FAST)
    unroll: tl.constexpr = 2 if FP8 and reduction_size % 256 == 0 else 1
    for k_tile in tl.range(reduction_size // 128, loop_unroll_factor=unroll):
        if not SCALE_PAIR:
            scale0 = tl.load(scales + expert * SE + k_tile * SG + n0 * SN)
            scale1 = tl.load(scales + expert * SE + k_tile * SG + (n0 + 8) * SN)
        if k_tile + 1 < reduction_size // 128:
            copy_token = prefetch_w8_tile(
                shared + (k_tile + 1) % 2 * W8_TILE_BYTES,
                ashared + (k_tile + 1) % 2 * 2048,
                qbase,
                activations,
                QN,
                reduction_size,
                row_tile,
                k_tile + 1,
                copy_token,
                IS_SPLIT_Q,
            )
            copy_token = wait_w8_tile(copy_token, True)
        else:
            copy_token = wait_w8_tile(copy_token, False)
        s0 = w8_scale_fragment(scale0, k_tile, SCALE_PAIR, FP8 and FAST)
        s1 = w8_scale_fragment(scale1, k_tile, SCALE_PAIR, FP8 and FAST)
        here = shared + k_tile % 2 * W8_TILE_BYTES
        ahere = ashared + k_tile % 2 * 2048
        warp = inner_n // 16
        for group in tl.static_range(4):
            if IS_SPLIT_Q:
                (r0, r1, r2, r3) = load_matrix_x4(
                    here + (warp // 4) * W8_HALF_TILE_BYTES,
                    (warp % 4) * 16,
                    64,
                    group * 32,
                    copy_token,
                )
            else:
                (r0, r1, r2, r3) = load_matrix_x4(
                    here, warp * 16, 128, group * 32, copy_token
                )
            for part in tl.static_range(2):
                q0 = r0 if part == 0 else r1
                q8 = r2 if part == 0 else r3
                sub = group * 2 + part
                if FP8:
                    if FAST:
                        (a0, a2, a1, a3) = decode_fp8_pair(q0, q8, s0, s1, True)
                    else:
                        (a0, a2, a1, a3) = decode_fp8_pair(q0, q8, s0, s1, False)
                    (b0, b1) = load_activation_fragment_x2(
                        ahere + sub // 4 * 1024, sub % 4 * 32, copy_token
                    )
                else:
                    (b0, b1) = load_activation_fragment_x2(
                        ahere + sub // 4 * 1024, sub % 4 * 32, copy_token
                    )
                    (a0, a2) = decode_int8_fragment_bf16(q0, s0, s0)
                    (a1, a3) = decode_int8_fragment_bf16(q8, s1, s1)
                (c0, c1, c2, c3) = mma_bf16_m16n8(
                    a0, a1, a2, a3, b0, b1, c0, c1, c2, c3
                )
        tl.debug_barrier()
    return (c0, c1, c2, c3)


@libentry()
@triton.jit
def marlin_moe_gemm_grouped_narrow_kernel(
    activations,
    weight_bytes,
    output_tile,
    scales,
    router_weights,
    route_ids,
    block_experts,
    padded_count,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    VALID,
    QE,
    QN,
    SE,
    SG,
    SN,
    CM,
    CN,
    MUL: tl.constexpr,
    scale_flags,
    FP8: tl.constexpr,
    USE32: tl.constexpr = False,
    PREFETCH_BEFORE_SCALE: tl.constexpr = False,
    SCALE_PAIR: tl.constexpr = False,
):
    # Each warp computes 16 output channels for eight routed rows.
    tl.static_assert(num_columns % 128 == 0 and reduction_size % 128 == 0)
    pid = tl.program_id(0)
    row_tile = pid // (num_columns // 128)
    col_tile = pid % (num_columns // 128)
    if row_tile * 8 >= tl.load(padded_count):
        return
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    expert = tl.load(block_experts + row_tile).to(offset_type)
    thread = tl.arange(0, 256)
    lane = thread % 32
    warp = thread // 32
    lane_group = lane // 4
    lane_in_group = lane % 4
    inner_n = warp * 16 + lane_group
    n0 = col_tile * 128 + inner_n
    smem = tle_async.gpu.alloc(
        [32768],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    act = tle_async.gpu.alloc(
        [4096],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    shared = tle_async.gpu.local_ptr(smem, (0,)).to(tl.int64).to(tl.int32)
    ashared = tle_async.gpu.local_ptr(act, (0,)).to(tl.int64).to(tl.int32)
    qbase = weight_bytes + expert * QE + col_tile * 128 * QN
    if FP8:
        if PREFETCH_BEFORE_SCALE:
            copy_token = tl.full((256,), 0, tl.int32)
            copy_token = prefetch_w8_tile(
                shared,
                ashared,
                qbase,
                activations,
                QN,
                reduction_size,
                row_tile,
                0,
                copy_token,
            )
        else:
            copy_token = None
        can_fold = (
            tl.load(scale_flags + expert) != 0 if scale_flags is not None else False
        )
        tile_inputs = (shared, ashared, qbase, activations, scales, QN)
        tile_coordinates = (row_tile, expert, n0, inner_n, SE, SG, SN)
        if can_fold:
            (c0, c1, c2, c3) = w8_narrow_accumulate(
                *tile_inputs,
                reduction_size,
                *tile_coordinates,
                True,
                True,
                initial_dep=copy_token,
                SCALE_PAIR=SCALE_PAIR,
            )
        else:
            (c0, c1, c2, c3) = w8_narrow_accumulate(
                *tile_inputs,
                reduction_size,
                *tile_coordinates,
                True,
                False,
                initial_dep=copy_token,
                SCALE_PAIR=SCALE_PAIR,
            )
    else:
        c0, c1, c2, c3 = w8_narrow_accumulate(
            shared,
            ashared,
            qbase,
            activations,
            scales,
            QN,
            reduction_size,
            row_tile,
            expert,
            n0,
            inner_n,
            SE,
            SG,
            SN,
            False,
            False,
        )
    token0 = tl.load(route_ids + row_tile * 8 + 2 * lane_in_group).to(offset_type)
    token1 = tl.load(route_ids + row_tile * 8 + 2 * lane_in_group + 1).to(offset_type)
    if MUL:
        w0 = tl.load(router_weights + token0, token0 < VALID, other=0).to(tl.float32)
        w1 = tl.load(router_weights + token1, token1 < VALID, other=0).to(tl.float32)
        c0 *= w0
        c2 *= w0
        c1 *= w1
        c3 *= w1
    tl.store(
        output_tile + token0 * CM + n0 * CN,
        c0.to(output_tile.dtype.element_ty),
        token0 < VALID,
    )
    tl.store(
        output_tile + token1 * CM + n0 * CN,
        c1.to(output_tile.dtype.element_ty),
        token1 < VALID,
    )
    tl.store(
        output_tile + token0 * CM + (n0 + 8) * CN,
        c2.to(output_tile.dtype.element_ty),
        token0 < VALID,
    )
    tl.store(
        output_tile + token1 * CM + (n0 + 8) * CN,
        c3.to(output_tile.dtype.element_ty),
        token1 < VALID,
    )
