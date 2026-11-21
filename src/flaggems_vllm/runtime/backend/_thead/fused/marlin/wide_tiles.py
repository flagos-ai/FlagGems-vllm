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

"""Wide quantized expert GEMM and its shared-memory epilogue."""

import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._thead.fused.marlin.formats import (
    _TL_QUANT_TYPE_FP4_E2M1,
    _TL_QUANT_TYPE_UINT4B8,
    _TL_QUANT_TYPE_UINT8B128,
    W8_HALF_TILE_BYTES,
    W8_TILE_BYTES,
    commit_w8_tile,
    copy_w8_tile_async,
    decode_fp8_pair,
    decode_int8_fragment_bf16,
    decode_w4_fragment_pair,
    load_matrix_x4,
    mma_bf16_square,
    mx_scale_fragment,
    mx_scale_source,
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
def prefetch_quantized_wide_tile(
    weight_smem,
    activation_smem,
    weight_bytes,
    activations,
    QN,
    reduction_size: tl.constexpr,
    row_tile,
    k_tile,
    copy_token,
    WEIGHT_K_BYTES: tl.constexpr,
):
    activation_base = (
        activations.to(tl.pointer_type(tl.uint8))
        + row_tile.to(tl.int64) * 64 * reduction_size * 2
    )
    copy_token = copy_w8_tile_async(
        activation_smem,
        activation_base,
        64,
        reduction_size * 2,
        128,
        k_tile * 256,
        copy_token,
    )
    copy_token = copy_w8_tile_async(
        activation_smem + W8_HALF_TILE_BYTES,
        activation_base,
        64,
        reduction_size * 2,
        128,
        k_tile * 256 + 128,
        copy_token,
    )
    copy_token = copy_w8_tile_async(
        weight_smem,
        weight_bytes,
        128,
        QN,
        WEIGHT_K_BYTES,
        k_tile * WEIGHT_K_BYTES,
        copy_token,
    )
    return commit_w8_tile(copy_token)


@triton.jit
def accumulate_quantized_wide_tile(
    weight_smem,
    activation_smem,
    weight_base,
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
    QTYPE: tl.constexpr,
    PACKED_SCALE: tl.constexpr,
    FAST: tl.constexpr,
    LIVE: tl.constexpr,
):
    MX: tl.constexpr = QTYPE == _TL_QUANT_TYPE_FP4_E2M1
    W4: tl.constexpr = MX or QTYPE == _TL_QUANT_TYPE_UINT4B8
    WEIGHT_K_BYTES: tl.constexpr = 64 if W4 else 128
    copy_token = tl.full((256,), 0, tl.int32)
    copy_token = prefetch_quantized_wide_tile(
        weight_smem,
        activation_smem,
        weight_base,
        activations,
        QN,
        reduction_size,
        row_tile,
        0,
        copy_token,
        WEIGHT_K_BYTES,
    )
    zero = tl.full((256,), 0.0, tl.float32)
    acc = ()
    for j in tl.static_range(LIVE // 2):
        acc += (zero,)
    for k_tile in range(reduction_size // 128):
        if not MX:
            scale0 = tl.load(scales + expert * SE + k_tile * SG + n0 * SN)
            scale1 = tl.load(scales + expert * SE + k_tile * SG + (n0 + 8) * SN)
        else:
            scale_ptr = scales + expert * SE + (k_tile * 4) * SG + n0 * SN
            src0 = mx_scale_source(scale_ptr, PACKED_SCALE)
            src1 = mx_scale_source(scale_ptr + 8 * SN, PACKED_SCALE)
        if k_tile + 1 < reduction_size // 128:
            next_offset = (k_tile + 1) % 2 * W8_TILE_BYTES
            copy_token = prefetch_quantized_wide_tile(
                weight_smem + (k_tile + 1) % 2 * 128 * WEIGHT_K_BYTES,
                activation_smem + next_offset,
                weight_base,
                activations,
                QN,
                reduction_size,
                row_tile,
                k_tile + 1,
                copy_token,
                WEIGHT_K_BYTES,
            )
            copy_token = wait_w8_tile(copy_token, True)
        else:
            copy_token = wait_w8_tile(copy_token, False)
        if not MX:
            s0 = w8_scale_fragment(scale0, k_tile, False, FAST)
            s1 = w8_scale_fragment(scale1, k_tile, False, FAST)
        current_offset = k_tile % 2 * W8_TILE_BYTES
        qhere = weight_smem + k_tile % 2 * 128 * WEIGHT_K_BYTES
        ahere = activation_smem + current_offset
        for group in tl.static_range(WEIGHT_K_BYTES // 32):
            r0, r1, r2, r3 = load_matrix_x4(
                qhere, warp * 16, 128, group * 32, copy_token
            )
            for part in tl.static_range(2):
                q0 = r0 if part == 0 else r1
                q8 = r2 if part == 0 else r3
                if MX:
                    s0 = mx_scale_fragment(src0, SG, group * 2 + part, PACKED_SCALE)
                    s1 = mx_scale_fragment(src1, SG, group * 2 + part, PACKED_SCALE)
                for half in tl.static_range(2 if W4 else 1):
                    if W4:
                        sub = group * 4 + part * 2 + half
                        a0, a1, a2, a3 = decode_w4_fragment_pair(
                            q0, q8, s0, s1, half, MX
                        )
                    else:
                        sub = group * 2 + part
                        if QTYPE == _TL_QUANT_TYPE_UINT8B128:
                            a0, a1 = decode_int8_fragment_bf16(q0, s0, s0)
                            a2, a3 = decode_int8_fragment_bf16(q8, s1, s1)
                        else:
                            a0, a1, a2, a3 = decode_fp8_pair(q0, q8, s0, s1, FAST)
                    # Reuse the decoded fragment for each live group of 16 routes.
                    next_acc = ()
                    for chunk in tl.static_range(LIVE // 16):
                        b0, b1, b2, b3 = load_matrix_x4(
                            ahere + sub // 4 * W8_HALF_TILE_BYTES,
                            chunk * 16,
                            64,
                            sub % 4 * 32,
                            copy_token,
                        )
                        next_acc += mma_bf16_square(
                            a0,
                            a1,
                            a2,
                            a3,
                            b0,
                            b1,
                            b2,
                            b3,
                            acc[chunk * 8],
                            acc[chunk * 8 + 1],
                            acc[chunk * 8 + 2],
                            acc[chunk * 8 + 3],
                            acc[chunk * 8 + 4],
                            acc[chunk * 8 + 5],
                            acc[chunk * 8 + 6],
                            acc[chunk * 8 + 7],
                        )
                    acc = next_acc
        tl.debug_barrier()
    return acc


@triton.jit
def project_quantized_wide_tile(
    weight_smem,
    activation_smem,
    weight_base,
    activations,
    output_tile,
    scales,
    router_weights,
    route_ids,
    scale_flags,
    QTYPE: tl.constexpr,
    PACKED_SCALE: tl.constexpr,
    reduction_size: tl.constexpr,
    QN,
    SE,
    SG,
    SN,
    CM,
    CN,
    VALID,
    MUL: tl.constexpr,
    LIVE: tl.constexpr,
    offset_type: tl.constexpr,
    expert,
    n0,
    warp,
    row_tile,
    lane_in_group,
    amem,
    col_tile,
):
    can_fold = tl.load(scale_flags + expert) != 0 if scale_flags is not None else False
    tile_inputs = (weight_smem, activation_smem, weight_base, activations, scales, QN)
    tile_coordinates = (row_tile, expert, n0, warp, SE, SG, SN)
    if can_fold:
        acc = accumulate_quantized_wide_tile(
            *tile_inputs,
            reduction_size,
            *tile_coordinates,
            QTYPE,
            PACKED_SCALE,
            True,
            LIVE,
        )
    else:
        acc = accumulate_quantized_wide_tile(
            *tile_inputs,
            reduction_size,
            *tile_coordinates,
            QTYPE,
            PACKED_SCALE,
            False,
            LIVE,
        )
    if reduction_size == 256:
        for chunk in tl.static_range(LIVE // 16):
            for j in tl.static_range(4):
                row = chunk * 16 + lane_in_group + 4 * j
                col = n0 % 128
                # Separate lanes that write the same column into distinct banks.
                bank = (row % 4) * 8
                tl.store(
                    tle_async.gpu.local_ptr(amem, (row * 128 + (col ^ bank),)),
                    acc[chunk * 8 + j],
                )
                tl.store(
                    tle_async.gpu.local_ptr(amem, (row * 128 + ((col + 8) ^ bank),)),
                    acc[chunk * 8 + j + 4],
                )
        tl.debug_barrier()
        rows = tl.arange(0, 64)
        cols = tl.arange(0, 128)
        bank = (rows[:, None] % 4) * 8
        indices = rows[:, None] * 128 + (cols[None, :] ^ bank)
        acc_output = tl.load(
            tle_async.gpu.local_ptr(amem, (indices,)), rows[:, None] < LIVE, other=0
        )
        tokens = tl.load(route_ids + row_tile * 64 + rows).to(offset_type)
        valid = tokens < VALID
        if MUL:
            weights = tl.load(router_weights + tokens, valid, other=0).to(tl.float32)
            acc_output *= weights[:, None]
        tl.store(
            output_tile + tokens[:, None] * CM + (col_tile * 128 + cols[None, :]) * CN,
            acc_output.to(output_tile.dtype.element_ty),
            valid[:, None],
        )
    else:
        for chunk in tl.static_range(LIVE // 16):
            for j in tl.static_range(4):
                token = tl.load(
                    route_ids + row_tile * 64 + chunk * 16 + lane_in_group + 4 * j
                ).to(offset_type)
                v0 = acc[chunk * 8 + j]
                v1 = acc[chunk * 8 + j + 4]
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


@libentry()
@triton.jit
def quantized_wide_grouped_kernel(
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
    QTYPE: tl.constexpr,
    PACKED_SCALE: tl.constexpr,
    USE32: tl.constexpr,
):
    pid = tl.program_id(0)
    row_tile = pid // (num_columns // 128)
    col_tile = pid % (num_columns // 128)
    if row_tile * 64 >= tl.load(padded_count):
        return
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    expert = tl.load(block_experts + row_tile).to(offset_type)
    thread = tl.arange(0, 256)
    lane = thread % 32
    warp = thread // 32
    n0 = col_tile * 128 + warp * 16 + lane // 4
    lane_in_group = lane % 4
    W4: tl.constexpr = (
        QTYPE == _TL_QUANT_TYPE_UINT4B8 or QTYPE == _TL_QUANT_TYPE_FP4_E2M1
    )
    qmem = tle_async.gpu.alloc(
        [W8_TILE_BYTES if W4 else 2 * W8_TILE_BYTES],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    if reduction_size == 256:
        # Reuse the activation buffer to assemble contiguous FP32 output rows.
        amem = tle_async.gpu.alloc(
            [64 * 128],
            tl.float32,
            layout=None,
            scope=tle_async.gpu.smem,
            nv_mma_shared_layout=False,
        )
    else:
        amem = tle_async.gpu.alloc(
            [32768],
            tl.uint8,
            layout=None,
            scope=tle_async.gpu.smem,
            nv_mma_shared_layout=False,
        )
    weight_smem = tle_async.gpu.local_ptr(qmem, (0,)).to(tl.int64).to(tl.int32)
    activation_smem = tle_async.gpu.local_ptr(amem, (0,)).to(tl.int64).to(tl.int32)
    weight_base = weight_bytes + expert * QE + col_tile * 128 * QN
    # Padding is contiguous at the end of each expert tile.
    if tl.load(route_ids + row_tile * 64 + 48) >= VALID:
        project_quantized_wide_tile(
            weight_smem,
            activation_smem,
            weight_base,
            activations,
            output_tile,
            scales,
            router_weights,
            route_ids,
            scale_flags,
            QTYPE,
            PACKED_SCALE,
            reduction_size,
            QN,
            SE,
            SG,
            SN,
            CM,
            CN,
            VALID,
            MUL,
            48,
            offset_type,
            expert,
            n0,
            warp,
            row_tile,
            lane_in_group,
            amem,
            col_tile,
        )
    else:
        project_quantized_wide_tile(
            weight_smem,
            activation_smem,
            weight_base,
            activations,
            output_tile,
            scales,
            router_weights,
            route_ids,
            scale_flags,
            QTYPE,
            PACKED_SCALE,
            reduction_size,
            QN,
            SE,
            SG,
            SN,
            CM,
            CN,
            VALID,
            MUL,
            64,
            offset_type,
            expert,
            n0,
            warp,
            row_tile,
            lane_in_group,
            amem,
            col_tile,
        )
