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


"""Routing, activation staging and quantized GEMMs for T-Head Marlin MoE."""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from flag_gems import zeros

from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_no_tle
from flaggems_vllm.runtime.backend._thead.fused.marlin_fragments import (
    _TL_QUANT_TYPE_FP4_E2M1,
    _TL_QUANT_TYPE_FP8_E4M3,
    _TL_QUANT_TYPE_UINT4B8,
    _TL_QUANT_TYPE_UINT8B128,
    byte_permute,
    e8m0_to_bf16_bits,
    scale_int8_pair,
    tle_async,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin_tiles import (
    SHORT_TILE_FLAG,
    store_k16,
    store_k32,
)
from flaggems_vllm.utils import libentry

LARGE_SCALE_WARP_THRESHOLD = 8 * 1024

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
def decode_int8_scaled_bf16(q, scale):
    value = tl.inline_asm_elementwise(
        (
            "{ .reg .b32 q, l<3>, h<3>, v<2>;\n"
            "ppu.xor.b32 q,$2,0x80808080;\n"
            "ppu.prmt.b32 l0,q,0x43,0x4140;\n"
            "ppu.prmt.b32 h0,q,0x43,0x4342;\n"
            "ppu.and.b32 l1,l0,0xff7fff7f;\n"
            "ppu.and.b32 h1,h0,0xff7fff7f;\n"
            "ppu.and.b32 l2,l0,0xff80ff80;\n"
            "ppu.and.b32 h2,h0,0xff80ff80;\n"
            "ppu.sub.bf16x2 v0,l1,l2;\n"
            "ppu.sub.bf16x2 v1,h1,h2;\n"
            "ppu.mov.b32 $0,v0;\n"
            "ppu.mov.b32 $1,v1;\n"
            "}\n"
        ),
        "=r,=r,r",
        [q],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )
    return (value * scale).to(tl.bfloat16)


@triton.jit
def decode_fp8_scaled_bf16_normalized(q, scale):
    # Normalize the exact BF16 representation before applying the original scale.
    return tl.inline_asm_elementwise(
        (
            "{ .reg .b32 mag,inc,mask,hi,lo,tmp,v0,v1,bias;\nppu.and.b32 mag,$2,0x7f7f7f7f;"
            "\nppu.add.u32 inc,mag,0x01010101;\nppu.prmt.b32 mask,inc,0,0xba98;"
            "\nppu.lop3.b32 hi,$2,mask,0x80808080,0xe4;\nppu.shr.b32 tmp,$2,4;"
            "\nppu.lop3.b32 hi,hi,tmp,0x07070707,0xf8;\nppu.shl.b32 lo,$2,4;"
            "\nppu.and.b32 lo,lo,0xf0f0f0f0;\nppu.prmt.b32 v0,lo,hi,0x5140;"
            "\nppu.prmt.b32 v1,lo,hi,0x7362;\nppu.mov.b32 bias,0x7b807b80;"
            " ppu.mul.rn.bf16x2 v0,v0,bias; ppu.mul.rn.bf16x2 v1,v1,bias;"
            "\nppu.mul.rn.bf16x2 v0,v0,$3; ppu.mul.rn.bf16x2 v1,v1,$4; ppu.mov.b32 $0,v0;"
            " ppu.mov.b32 $1,v1; }"
        ),
        "=r,=r,r,r,r",
        [q, scale],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )


@triton.jit
def decode_e4m3(q, scale, compute_type: tl.constexpr):
    if compute_type == tl.bfloat16:
        # Preserve E4M3 subnormals by embedding the bits in BF16. Its exponent
        # bias is corrected by an exact power-of-two factor in the scale.
        bits = ((q & 128) << 8) | ((q & 127) << 4)
        # Set the BF16 NaN exponent without a floating-point select per weight.
        bits = bits | (-(((q & 127) + 1) >> 7) & 0x7F80)
        tiny = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        exponent_bias = 2.0**120
    else:
        # E4M3 bits embedded in FP16 have exponent bias 15 rather than 7.
        bits = ((q & 128) << 8) | ((q & 127) << 7)
        bits = bits | (-(((q & 127) + 1) >> 7) & 0x7C00)
        tiny = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        exponent_bias = 256.0
    # Sparse routes avoid scanning scales for experts they never read.
    folded = scale * exponent_bias
    overflow = (tl.abs(folded) == float("inf")) & (tl.abs(scale) != float("inf"))
    folded = tl.where(overflow, scale, folded)
    value = tl.where(overflow, tiny * exponent_bias, tiny)
    return (value * folded).to(compute_type)


@triton.jit
def decode_e2m1(q, scale, compute_type: tl.constexpr):
    bits = ((q & 8) << 12) | ((q & 7) << 9)
    tiny = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return (tiny * (2.0**14) * scale).to(compute_type)


@triton.jit
def decode_raw_w4_bf16(b, scale, MX: tl.constexpr):
    # Both formats expand the low and high nibbles into BF16 pairs.
    fill: tl.constexpr = "0" if MX else "0x43"
    bias: tl.constexpr = "0x7e807e80" if MX else "0x43084308"
    unpack: tl.constexpr = (
        "{ .reg .b32 lo,hi,l<2>,h<2>,bias;"
        "ppu.mov.b32 bias," + bias + ";"
        "ppu.and.b32 lo,$4,0x0f0f0f0f; ppu.shr.b32 hi,$4,4; ppu.and.b32 hi,hi,0x0f0f0f0f;"
        "ppu.prmt.b32 l0,lo,"
        + fill
        + ",0x4140; ppu.prmt.b32 l1,lo,"
        + fill
        + ",0x4342;"
        "ppu.prmt.b32 h0,hi,"
        + fill
        + ",0x4140; ppu.prmt.b32 h1,hi,"
        + fill
        + ",0x4342;"
    )
    if MX:
        # E2M1 needs exponent correction before multiplication by the input scales.
        convert: tl.constexpr = (
            "ppu.mul.lo.u32 l0,l0,0x1040; ppu.and.b32 l0,l0,0x81c081c0; ppu.mul.rn.bf16x2 l0,l0,bias;"
            "ppu.mul.lo.u32 l1,l1,0x1040; ppu.and.b32 l1,l1,0x81c081c0; ppu.mul.rn.bf16x2 l1,l1,bias;"
            "ppu.mul.lo.u32 h0,h0,0x1040; ppu.and.b32 h0,h0,0x81c081c0; ppu.mul.rn.bf16x2 h0,h0,bias;"
            "ppu.mul.lo.u32 h1,h1,0x1040; ppu.and.b32 h1,h1,0x81c081c0; ppu.mul.rn.bf16x2 h1,h1,bias;"
        )
        constraints: tl.constexpr = "=r,=r,=r,=r,r"
        operands = (b,)
    else:
        convert: tl.constexpr = (
            "ppu.sub.bf16x2 l0,l0,bias; ppu.sub.bf16x2 l1,l1,bias;"
            "ppu.sub.bf16x2 h0,h0,bias; ppu.sub.bf16x2 h1,h1,bias;"
            "ppu.mul.rn.bf16x2 l0,l0,$5; ppu.mul.rn.bf16x2 l1,l1,$6;"
            "ppu.mul.rn.bf16x2 h0,h0,$5; ppu.mul.rn.bf16x2 h1,h1,$6;"
        )
        constraints: tl.constexpr = "=r,=r,=r,=r,r,r,r"
        operands = (b, scale)
    lo, hi = tl.inline_asm_elementwise(
        unpack
        + convert
        + "ppu.mov.b32 $0,l0; ppu.mov.b32 $1,l1; ppu.mov.b32 $2,h0; ppu.mov.b32 $3,h1; }",
        constraints,
        operands,
        dtype=(tl.bfloat16, tl.bfloat16),
        is_pure=True,
        pack=4,
    )
    if MX:
        lo, hi = (lo * scale).to(tl.bfloat16), (hi * scale).to(tl.bfloat16)
    return lo, hi


@triton.jit
def dequant_int4(
    b,
    s_ptr,
    expert,
    k_base,
    ns,
    se,
    sg,
    sn,
    num_columns: tl.constexpr,
    compute_type: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    scale = tl.load(
        s_ptr + expert * se + (k_base // GROUP_SIZE) * sg + ns * sn,
        ns < num_columns,
        other=0,
    )[None, :]
    if compute_type == tl.bfloat16 and s_ptr.dtype.element_ty == tl.bfloat16:
        lo, hi = decode_raw_w4_bf16(b, scale, False)
        q = tl.reshape(
            tl.permute(tl.join(lo, hi), (0, 2, 1)), (b.shape[0] * 2, b.shape[1])
        )
        return q
    q = tl.join(b & 15, (b >> 4) & 15)
    q = tl.reshape(tl.permute(q, (0, 2, 1)), (b.shape[0] * 2, b.shape[1]))
    return ((q.to(compute_type) - 8.0) * scale).to(compute_type)


@triton.jit
def dequant_int8(
    b,
    s_ptr,
    expert,
    k_base,
    ns,
    se,
    sg,
    sn,
    num_columns: tl.constexpr,
    compute_type: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PAIR_K: tl.constexpr = False,
):
    tl.static_assert(b.shape[0] == 128 or b.shape[0] == 64)
    group = k_base // GROUP_SIZE
    scale = tl.load(
        s_ptr + expert * se + group * sg + ns * sn,
        mask=ns < num_columns,
        other=0.0,
    )[None, :]
    if PAIR_K and compute_type == tl.bfloat16:
        # Each pair uses the same group scale; form its register fragment along K.
        byte0, byte1 = tl.split(
            tl.reshape(tl.trans(b), (b.shape[1], b.shape[0] // 2, 2))
        )
        codes = byte0.to(tl.uint32) | (byte1.to(tl.uint32) << 8)
        bits = tl.trans(scale).to(tl.uint16, bitcast=True).to(tl.uint32)
        scales = bits | (bits << 16)
        values = scale_int8_pair(byte_permute(codes, 0x43, 0x4140), scales)
        even = values.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        odd = (values >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        return tl.trans(tl.reshape(tl.join(even, odd), (b.shape[1], b.shape[0])))
    elif compute_type == tl.bfloat16 and (b.shape[1] <= 128 or b.shape[0] == 64):
        return decode_int8_scaled_bf16(b.to(tl.uint8), scale)
    else:
        quant = b.to(tl.int32) - 128
        return (quant.to(compute_type) * scale).to(compute_type)


@triton.jit
def dequant_mxfp4(
    b,
    s_ptr,
    expert,
    k_base,
    ns,
    se,
    sg,
    sn,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    compute_type: tl.constexpr,
):
    groups = k_base // 32 + tl.arange(0, b.shape[0] * 2 // 32)
    byte = tl.load(
        s_ptr + expert * se + groups[:, None] * sg + ns[None, :] * sn,
        (ns[None, :] < num_columns) & (groups[:, None] * 32 < reduction_size),
        other=127,
    ).to(tl.uint32)
    if compute_type == tl.bfloat16:
        sbits = e8m0_to_bf16_bits(byte)
        scale = sbits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        scale = tl.reshape(
            tl.broadcast_to(scale[:, None, :], (b.shape[0] * 2 // 32, 16, b.shape[1])),
            (b.shape[0], b.shape[1]),
        )
        lo, hi = decode_raw_w4_bf16(b, scale, True)
        return tl.reshape(
            tl.permute(tl.join(lo, hi), (0, 2, 1)), (b.shape[0] * 2, b.shape[1])
        )
    bits = tl.where(byte == 0, 0x00400000, byte << 23)
    bits = tl.where(byte == 255, 0x7FC00000, bits)
    scale = bits.to(tl.float32, bitcast=True)
    scale = tl.reshape(
        tl.broadcast_to(scale[:, None, :], (b.shape[0] * 2 // 32, 32, b.shape[1])),
        (b.shape[0] * 2, b.shape[1]),
    )
    q = tl.join(b & 15, (b >> 4) & 15)
    q = tl.reshape(tl.permute(q, (0, 2, 1)), (b.shape[0] * 2, b.shape[1])).to(tl.uint32)
    return decode_e2m1(q, scale, compute_type)


@triton.jit
def dequant_fp8(
    b,
    s_ptr,
    expert,
    k_base,
    ns,
    se,
    sg,
    sn,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    compute_type: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    tl.static_assert(b.shape[0] == 128 or b.shape[0] == 64)
    if GROUP_SIZE == -1 or GROUP_SIZE >= 128:
        group = 0 if GROUP_SIZE == -1 else k_base // GROUP_SIZE
        scale_ptrs = s_ptr + expert * se + group * sg + ns * sn
        if num_columns % ns.shape[0] == 0:
            scale = tl.load(scale_ptrs)[None, :]
        else:
            scale = tl.load(scale_ptrs, mask=ns < num_columns, other=1.0)[None, :]
    else:
        groups = k_base // GROUP_SIZE + tl.arange(0, 128 // GROUP_SIZE)
        scale_ptrs = s_ptr + expert * se + groups[:, None] * sg + ns[None, :] * sn
        if num_columns % ns.shape[0] == 0 and reduction_size % 128 == 0:
            scale = tl.load(scale_ptrs)
        else:
            scale = tl.load(
                scale_ptrs,
                mask=(ns[None, :] < num_columns)
                & (groups[:, None] * GROUP_SIZE < reduction_size),
                other=1.0,
            )
        scale = tl.broadcast_to(
            scale[:, None, :], (128 // GROUP_SIZE, GROUP_SIZE, ns.shape[0])
        )
        scale = tl.reshape(scale, (128, ns.shape[0]))
    if compute_type == tl.bfloat16 and s_ptr.dtype.element_ty == tl.bfloat16:
        decoded = decode_fp8_scaled_bf16_normalized(b.to(tl.uint8), scale)
    else:
        decoded = decode_e4m3(b.to(tl.uint32), scale, compute_type)
    if reduction_size % 128 == 0:
        return decoded
    # Channel scales can be nonfinite; padded bytes must never introduce NaN.
    return tl.where(
        (k_base + tl.arange(0, b.shape[0]))[:, None] < reduction_size, decoded, 0.0
    ).to(compute_type)


@triton.jit
def dequant_weight(
    b,
    s_ptr,
    expert,
    k_base,
    ns,
    se,
    sg,
    sn,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    compute_type: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    PAIR_INT8_K: tl.constexpr = False,
):
    tile = (b, s_ptr, expert, k_base, ns, se, sg, sn, num_columns)
    if QUANT_TYPE == _TL_QUANT_TYPE_UINT4B8:
        return dequant_int4(*tile, compute_type, GROUP_SIZE)
    elif QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128:
        return dequant_int8(*tile, compute_type, GROUP_SIZE, PAIR_K=PAIR_INT8_K)
    elif QUANT_TYPE == _TL_QUANT_TYPE_FP4_E2M1:
        return dequant_mxfp4(*tile, reduction_size, compute_type)
    else:
        return dequant_fp8(*tile, reduction_size, compute_type, GROUP_SIZE)


@triton.jit
def is_fp8_scale_fold_overflow(scale):
    tl.static_assert(scale.dtype == tl.bfloat16)
    # Only finite magnitudes >= 256 overflow the exact 2**120 fold.
    magnitude = scale.to(tl.uint16, bitcast=True).to(tl.uint32) & 0x7FFF
    return (magnitude - 0x4380).to(tl.uint32) < 0x3C00


@triton.jit
def write_fp8_expert_scale_safety(
    scales,
    Safe,
    expert,
    num_columns: tl.constexpr,
    G: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK).to(tl.int64)
    n, lane_group = i // G, i % G
    scale = tl.load(
        scales + expert * SE + n * SN + lane_group * SG, i < num_columns * G, other=1
    )
    bad = is_fp8_scale_fold_overflow(scale) & (i < num_columns * G)
    tl.store(Safe + expert, tl.sum(bad.to(tl.int32)) == 0)


@triton.jit
def w8_small_routes_kernel(
    ids_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    post_ptr,
    num_routes: tl.constexpr,
    block_m: tl.constexpr,
    block_routes: tl.constexpr,
):
    routes = tl.arange(0, block_routes)
    ids = tl.load(ids_ptr + routes, routes < num_routes, other=-1)
    same_expert = (
        (ids[:, None] == ids[None, :])
        & (routes[:, None] < num_routes)
        & (routes[None, :] < num_routes)
    )
    route_rank = tl.sum(
        (same_expert & (routes[None, :] < routes[:, None])).to(tl.int32), 1
    )
    expert_count = tl.sum(same_expert.to(tl.int32), 1)
    first_route = (route_rank == 0) & (routes < num_routes)
    padded_count = tl.where(first_route, tl.cdiv(expert_count, block_m) * block_m, 0)
    starts = tl.cumsum(padded_count) - padded_count
    first_route_index = tl.min(tl.where(same_expert, routes[None, :], block_routes), 1)
    first_route_index = tl.where(routes < num_routes, first_route_index, 0)
    expert_start = tl.gather(starts, first_route_index, axis=0)
    tl.store(sorted_ids_ptr + expert_start + route_rank, routes, routes < num_routes)
    slots = tl.arange(0, block_routes)
    tl.store(
        sorted_ids_ptr + starts[:, None] + slots[None, :],
        num_routes,
        first_route[:, None]
        & (slots[None, :] >= expert_count[:, None])
        & (slots[None, :] < padded_count[:, None]),
    )
    tl.store(
        expert_ids_ptr + starts[:, None] // block_m + slots[None, :],
        ids[:, None],
        first_route[:, None]
        & (slots[None, :] < tl.cdiv(expert_count[:, None], block_m)),
    )
    tl.store(post_ptr, tl.sum(padded_count))


@triton.jit
def init_route_workspace(
    sorted_routes, ranks, num_routes, capacity, num_experts, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(sorted_routes + offsets, num_routes, offsets < capacity)
    tl.store(ranks + offsets, 0, offsets < num_experts)


@triton.jit
def w8_one_cta_routes(
    route_ids,
    sorted_routes,
    block_experts,
    padded_count,
    expert_ranks,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BM: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    PACK_WIDTH: tl.constexpr = False,
):
    route = tl.arange(0, BLOCK_ROUTES)
    is_valid = route < NUM_ROUTES
    expert = tl.load(route_ids + route, is_valid, other=0).to(tl.int32)
    count = tl.histogram(expert, BLOCK_EXPERTS, mask=is_valid)
    padded = tl.cdiv(count, BM) * BM
    start = tl.cumsum(padded, 0) - padded
    rank = tl.atomic_add(expert_ranks + expert, 1, mask=is_valid, sem="relaxed")
    if PACK_WIDTH:
        tl.static_assert(BM == 16)
        offset = tl.gather(start, expert, 0) + rank
        valid_left = tl.gather(count, expert, 0) - rank
        # Record when the 16-row tile only needs its first eight rows.
        expert_word = expert | tl.where(valid_left <= 8, SHORT_TILE_FLAG, 0)
    else:
        offset = tl.gather(start, expert, 0) + rank
        expert_word = expert
    tl.store(sorted_routes + offset, route, is_valid)
    tl.store(block_experts + offset // BM, expert_word, is_valid & (rank % BM == 0))
    tl.store(padded_count, tl.sum(padded, 0))


@libentry()
@triton.jit
def fp8_safety_and_routes(
    gate_scales,
    down_scales,
    gate_scale_flags,
    down_scale_flags,
    route_ids,
    sorted_routes,
    block_experts,
    padded_count,
    expert_ranks,
    NUM_EXPERTS: tl.constexpr,
    N1: tl.constexpr,
    G1: tl.constexpr,
    N2: tl.constexpr,
    G2: tl.constexpr,
    SE1: tl.constexpr,
    SN1: tl.constexpr,
    SG1: tl.constexpr,
    SE2: tl.constexpr,
    SN2: tl.constexpr,
    SG2: tl.constexpr,
    BLOCK1: tl.constexpr,
    BLOCK2: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    BM: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    PACK_WIDTH: tl.constexpr = False,
):
    if tl.program_id(0) == 0:
        # Routing never reads other CTAs' flags; GEMMs consume both after this launch.
        w8_one_cta_routes(
            route_ids,
            sorted_routes,
            block_experts,
            padded_count,
            expert_ranks,
            NUM_ROUTES,
            NUM_EXPERTS,
            BM,
            BLOCK_ROUTES,
            BLOCK_EXPERTS,
            PACK_WIDTH=PACK_WIDTH,
        )
        return
    safety_pid = tl.program_id(0) - 1
    if NUM_ROUTES <= NUM_EXPERTS:
        # Only the first route for each expert writes its scale flags.
        route = safety_pid
        expert = tl.load(route_ids + route).to(tl.int64)
        prior = tl.arange(0, BLOCK_ROUTES)
        prior_ids = tl.load(route_ids + prior, prior < route, other=-1).to(tl.int64)
        is_first = (
            (expert >= 0)
            & (expert < NUM_EXPERTS)
            & (tl.sum(prior_ids == expert, 0) == 0)
        )
        if is_first:
            write_fp8_expert_scale_safety(
                gate_scales, gate_scale_flags, expert, N1, G1, SE1, SN1, SG1, BLOCK1
            )
            write_fp8_expert_scale_safety(
                down_scales, down_scale_flags, expert, N2, G2, SE2, SN2, SG2, BLOCK2
            )
    else:
        # An expert with no routed tokens has no GEMM consuming its flags.
        expert = safety_pid.to(tl.int64)
        row = tl.arange(0, BLOCK_ROUTES)
        routed = tl.load(route_ids + row, row < NUM_ROUTES, other=-1)
        is_active = tl.sum((routed == expert).to(tl.int32), 0) != 0
        if is_active:
            write_fp8_expert_scale_safety(
                gate_scales, gate_scale_flags, expert, N1, G1, SE1, SN1, SG1, BLOCK1
            )
            write_fp8_expert_scale_safety(
                down_scales, down_scale_flags, expert, N2, G2, SE2, SN2, SG2, BLOCK2
            )


def align_grouped_tokens(
    topk_ids: torch.Tensor,
    block_m: int,
    num_experts: int,
    fp8_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    fp8_safety: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    is_one_cta: bool = False,
    packed_width: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if is_one_cta:
        routes = topk_ids.numel()
        max_padded_routes = min(routes * block_m, routes + num_experts * (block_m - 1))
        sorted_ids = torch.empty(
            (max_padded_routes,), device=topk_ids.device, dtype=torch.int32
        )
        experts = torch.empty(
            (triton.cdiv(max_padded_routes, block_m),),
            device=topk_ids.device,
            dtype=torch.int32,
        )
        padded = torch.empty((1,), device=topk_ids.device, dtype=torch.int32)
        rank = torch.empty((num_experts,), device=topk_ids.device, dtype=torch.int32)
        # Stream launch order makes the global writes visible before scatter and atomics.
        init_route_workspace[(triton.cdiv(max(max_padded_routes, num_experts), 256),)](
            sorted_ids, rank, routes, max_padded_routes, num_experts, 256
        )
        warps = 8 if routes <= 128 else 16
        if fp8_scales is None:
            w8_one_cta_routes[(1,)](
                topk_ids,
                sorted_ids,
                experts,
                padded,
                rank,
                routes,
                num_experts,
                block_m,
                triton.next_power_of_2(max(routes, warps * 32)),
                triton.next_power_of_2(num_experts),
                num_warps=warps,
                num_stages=1,
            )
        else:
            s1, s2 = fp8_scales
            b1 = triton.next_power_of_2(s1.size(1) * s1.size(2))
            b2 = triton.next_power_of_2(s2.size(1) * s2.size(2))
            warps = max(warps, 16 if max(b1, b2) >= LARGE_SCALE_WARP_THRESHOLD else 4)
            safety_ctas = routes if routes <= num_experts else num_experts
            fp8_safety_and_routes[(safety_ctas + 1,)](
                s1,
                s2,
                *fp8_safety,
                topk_ids,
                sorted_ids,
                experts,
                padded,
                rank,
                num_experts,
                s1.size(1),
                s1.size(2),
                s2.size(1),
                s2.size(2),
                *s1.stride(),
                *s2.stride(),
                b1,
                b2,
                routes,
                block_m,
                triton.next_power_of_2(max(routes, warps * 32)),
                triton.next_power_of_2(num_experts),
                PACK_WIDTH=packed_width,
                num_warps=warps,
                num_stages=1,
            )
        return sorted_ids, experts, padded
    workspace = (
        zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device),
        zeros(
            (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
        ),
    )
    aligned = moe_align_block_size_no_tle(
        topk_ids, block_m, num_experts, workspace=workspace
    )
    if fp8_scales is not None:
        s1, s2 = fp8_scales
        b1 = triton.next_power_of_2(s1.size(1) * s1.size(2))
        b2 = triton.next_power_of_2(s2.size(1) * s2.size(2))
        check_routed_fp8_scales[(num_experts,)](
            s1,
            s2,
            *fp8_safety,
            workspace[0],
            *s1.shape[1:],
            *s2.shape[1:],
            *s1.stride(),
            *s2.stride(),
            b1,
            b2,
            num_warps=16 if max(b1, b2) >= LARGE_SCALE_WARP_THRESHOLD else 4,
        )
    return aligned


@libentry()
@triton.jit
def check_routed_fp8_scales(
    gate_scales,
    down_scales,
    gate_scale_flags,
    down_scale_flags,
    Prefix,
    N1: tl.constexpr,
    G1: tl.constexpr,
    N2: tl.constexpr,
    G2: tl.constexpr,
    SE1: tl.constexpr,
    SN1: tl.constexpr,
    SG1: tl.constexpr,
    SE2: tl.constexpr,
    SN2: tl.constexpr,
    SG2: tl.constexpr,
    BLOCK1: tl.constexpr,
    BLOCK2: tl.constexpr,
):
    expert = tl.program_id(0).to(tl.int64)
    if tl.load(Prefix + expert) != tl.load(Prefix + expert + 1):
        write_fp8_expert_scale_safety(
            gate_scales, gate_scale_flags, expert, N1, G1, SE1, SN1, SG1, BLOCK1
        )
        write_fp8_expert_scale_safety(
            down_scales, down_scale_flags, expert, N2, G2, SE2, SN2, SG2, BLOCK2
        )


@triton.jit
def w4_pair_gemm(
    activations,
    b_block_ptr,
    scales,
    expert,
    ns,
    se,
    sg,
    sn,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    EM: tl.constexpr,
    PID_M,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    STAGES: tl.constexpr,
    QTYPE: tl.constexpr,
):
    # Even and odd K values occupy separate contiguous EM x K/2 planes.
    # The logical width and physical row pitch must agree for these loads.
    ae = tl.make_block_ptr(
        activations,
        shape=(EM, reduction_size // 2),
        strides=(reduction_size // 2, 1),
        offsets=(PID_M * BM, 0),
        block_shape=(BM, BK // 2),
        order=(1, 0),
    )
    ao = tl.make_block_ptr(
        activations + EM * (reduction_size // 2),
        shape=(EM, reduction_size // 2),
        strides=(reduction_size // 2, 1),
        offsets=(PID_M * BM, 0),
        block_shape=(BM, BK // 2),
        order=(1, 0),
    )
    acc = tl.zeros((BN, BM), tl.float32)
    for tile in tl.range(0, tl.cdiv(reduction_size, BK), num_stages=STAGES):
        a0 = tle_async.load(
            ae, boundary_check=(0, 1), padding_option="zero", is_async=True
        )
        a1 = tle_async.load(
            ao, boundary_check=(0, 1), padding_option="zero", is_async=True
        )
        if num_columns % BN == 0 and reduction_size % BK == 0:
            q = tle_async.load(b_block_ptr, is_async=True)
        else:
            q = tle_async.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
        if QTYPE == _TL_QUANT_TYPE_UINT4B8:
            scale = tl.load(
                scales + expert * se + (tile * BK // GROUP_SIZE) * sg + ns * sn,
                ns < num_columns,
                other=0,
            )[None, :]
            lo, hi = decode_raw_w4_bf16(q, scale, False)
        else:
            gs = tile * BK // 32 + tl.arange(0, BK // 32)
            byte = tl.load(
                scales + expert * se + gs[:, None] * sg + ns[None, :] * sn,
                (ns[None, :] < num_columns) & (gs[:, None] * 32 < reduction_size),
                other=127,
            ).to(tl.uint32)
            bits = e8m0_to_bf16_bits(byte)
            scale = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
            scale = tl.reshape(
                tl.broadcast_to(scale[:, None, :], (BK // 32, 16, BN)),
                (BK // 2, BN),
            )
            lo, hi = decode_raw_w4_bf16(q, scale, True)
        acc = tl.dot(tl.trans(lo), tl.trans(a0), acc)
        acc = tl.dot(tl.trans(hi), tl.trans(a1), acc)
        ae = tl.advance(ae, (0, BK // 2))
        ao = tl.advance(ao, (0, BK // 2))
        b_block_ptr = tl.advance(b_block_ptr, (BK // 2, 0))
    return acc


@triton.jit
def quantized_gemm_tile(
    a_block_ptr,
    b_block_ptr,
    b_scale_ptr,
    expert,
    offs_n,
    stride_bse,
    stride_bsg,
    stride_bsn,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    compute_type: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr = 128,
):
    """Load, decode and multiply one stream of quantized weight tiles."""
    IS_8BIT: tl.constexpr = (
        QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128 or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
    )
    BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for k_tile in tl.range(
        0, tl.cdiv(reduction_size, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES
    ):
        activation = tle_async.load(
            a_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
            is_async=True,
        )
        if reduction_size % BLOCK_SIZE_K == 0 and num_columns % BLOCK_SIZE_N == 0:
            b_quantized = tle_async.load(b_block_ptr, is_async=True)
        else:
            b_quantized = tle_async.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
        weight = dequant_weight(
            b_quantized,
            b_scale_ptr,
            expert,
            k_tile * BLOCK_SIZE_K,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            compute_type,
            GROUP_SIZE_K,
            QUANT_TYPE,
        )
        acc = tl.dot(tl.trans(weight), tl.trans(activation), acc=acc)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K_WEIGHT, 0))
    return acc


@libentry()
@triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
def marlin_moe_gemm_direct_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    A_ROUTE_DIVISOR: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
):
    # One program computes one routed row and one output-N tile.  A
    # 16-row activation tile is used because PPU AIU requires a regular
    # 2D tile; boundary padding leaves only row zero valid.
    BLOCK_SIZE_M: tl.constexpr = 16
    BLOCK_SIZE_K: tl.constexpr = 128
    IS_8BIT: tl.constexpr = (
        QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128 or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
    )
    BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2

    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = tl.cdiv(num_columns, BLOCK_SIZE_N)
    route = pid // num_pid_n
    pid_n = pid % num_pid_n
    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    a_row = route // A_ROUTE_DIVISOR
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + a_row * stride_am,
        shape=(1, reduction_size),
        strides=(stride_am, stride_ak),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + expert * stride_be,
        shape=(
            (reduction_size if IS_8BIT else tl.cdiv(reduction_size, 2)),
            num_columns,
        ),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
        order=(0, 1),
    )
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    acc = quantized_gemm_tile(
        a_block_ptr,
        b_block_ptr,
        b_scale_ptr,
        expert,
        offs_n,
        stride_bse,
        stride_bsg,
        stride_bsn,
        num_columns,
        reduction_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        GROUP_SIZE_K,
        PIPELINE_STAGES,
        compute_type,
        QUANT_TYPE,
    )

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
        acc *= routed_weight

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_rows = route + offs_m[None, :]
    c_cols = offs_n[:, None]
    c_ptrs = c_ptr + c_rows * stride_cm + c_cols * stride_cn
    tl.store(
        c_ptrs,
        acc.to(compute_type),
        mask=(offs_m[None, :] == 0) & (offs_n[:, None] < num_columns),
    )


@libentry()
@triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
def marlin_moe_matvec_direct_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    A_ROUTE_DIVISOR: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    SPLIT_STRIDE: tl.constexpr = 0,
):
    BLOCK_SIZE_K: tl.constexpr = 128
    IS_8BIT: tl.constexpr = (
        QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128 or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
    )
    BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2

    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = tl.cdiv(num_columns, BLOCK_SIZE_N)
    route = pid // num_pid_n
    pid_n = pid % num_pid_n
    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    a_row = route // A_ROUTE_DIVISOR
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + expert * stride_be,
        shape=(
            (reduction_size if IS_8BIT else tl.cdiv(reduction_size, 2)),
            num_columns,
        ),
        strides=(stride_bk, stride_bn),
        offsets=(
            (
                (
                    tl.program_id(1)
                    * tl.cdiv(tl.cdiv(reduction_size, 128), SPLIT_K)
                    * 128
                )
                if SPLIT_K > 1
                else 0
            ),
            pid_n * BLOCK_SIZE_N,
        ),
        block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
        order=(0, 1),
    )
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    if SPLIT_K > 1:
        chunk: tl.constexpr = tl.cdiv(tl.cdiv(reduction_size, BLOCK_SIZE_K), SPLIT_K)
        begin = tl.program_id(1) * chunk
        end = tl.minimum(begin + chunk, tl.cdiv(reduction_size, BLOCK_SIZE_K))
    else:
        begin = 0
        end = tl.cdiv(reduction_size, BLOCK_SIZE_K)
    for k_tile in tl.range(begin, end, num_stages=PIPELINE_STAGES):
        a = tl.load(
            a_ptr + a_row * stride_am + (k_tile * BLOCK_SIZE_K + offs_k) * stride_ak,
            mask=k_tile * BLOCK_SIZE_K + offs_k < reduction_size,
            other=0.0,
        )
        if reduction_size % BLOCK_SIZE_K == 0 and num_columns % BLOCK_SIZE_N == 0:
            b_quantized = tle_async.load(b_block_ptr, is_async=True)
        else:
            b_quantized = tle_async.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
        b = dequant_weight(
            b_quantized,
            b_scale_ptr,
            expert,
            k_tile * BLOCK_SIZE_K,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            compute_type,
            GROUP_SIZE_K,
            QUANT_TYPE,
            PAIR_INT8_K=True,
        )
        acc += tl.sum(a[:, None].to(tl.float32) * b.to(tl.float32), axis=0)
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K_WEIGHT, 0))

    if MUL_ROUTED_WEIGHT and SPLIT_K == 1:
        acc *= tl.load(topk_weights_ptr + route).to(tl.float32)
    tl.store(
        c_ptr
        + (tl.program_id(1) * SPLIT_STRIDE if SPLIT_K > 1 else 0)
        + route * stride_cm
        + offs_n * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=offs_n < num_columns,
    )


@libentry()
@triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
def marlin_moe_gemm_reduce_direct_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    TOP_K: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
):
    """Compute GEMM2 and reduce all routed experts into one token row."""
    BLOCK_SIZE_M: tl.constexpr = 16
    BLOCK_SIZE_K: tl.constexpr = 128
    IS_8BIT: tl.constexpr = (
        QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128 or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
    )
    BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2

    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = tl.cdiv(num_columns, BLOCK_SIZE_N)
    token = pid // num_pid_n
    pid_n = pid % num_pid_n
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for topk_index in tl.range(0, TOP_K):
        route = token * TOP_K + topk_index
        expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr + route * stride_am,
            shape=(1, reduction_size),
            strides=(stride_am, stride_ak),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(
                (reduction_size if IS_8BIT else tl.cdiv(reduction_size, 2)),
                num_columns,
            ),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
            order=(0, 1),
        )

        route_acc = quantized_gemm_tile(
            a_block_ptr,
            b_block_ptr,
            b_scale_ptr,
            expert,
            pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N),
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_K,
            PIPELINE_STAGES,
            compute_type,
            QUANT_TYPE,
        )
        if MUL_ROUTED_WEIGHT:
            routed_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
            route_acc *= routed_weight
        acc += route_acc

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (token + offs_m[None, :]) * stride_cm + offs_n[:, None] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(compute_type),
        mask=(offs_m[None, :] == 0) & (offs_n[:, None] < num_columns),
    )


@libentry()
@triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
def marlin_moe_gemm_silu_direct_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    A_ROUTE_DIVISOR: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
):
    BLOCK_SIZE_M: tl.constexpr = 16
    BLOCK_SIZE_K: tl.constexpr = 128
    IS_8BIT: tl.constexpr = (
        QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128 or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
    )
    BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2

    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = tl.cdiv(num_columns, BLOCK_SIZE_N)
    route = pid // num_pid_n
    pid_n = pid % num_pid_n
    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    a_row = route // A_ROUTE_DIVISOR
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + a_row * stride_am,
        shape=(1, reduction_size),
        strides=(stride_am, stride_ak),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_gate_block_ptr = tl.make_block_ptr(
        base=b_ptr + expert * stride_be,
        shape=(
            (reduction_size if IS_8BIT else tl.cdiv(reduction_size, 2)),
            2 * num_columns,
        ),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
        order=(0, 1),
    )
    b_up_block_ptr = tl.make_block_ptr(
        base=b_ptr + expert * stride_be,
        shape=(
            (reduction_size if IS_8BIT else tl.cdiv(reduction_size, 2)),
            2 * num_columns,
        ),
        strides=(stride_bk, stride_bn),
        offsets=(0, num_columns + pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
        order=(0, 1),
    )
    acc_gate = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for k_tile in tl.range(
        0, tl.cdiv(reduction_size, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES
    ):
        a = tle_async.load(
            a_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
            is_async=True,
        )
        b_gate_quantized = tle_async.load(
            b_gate_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
            is_async=True,
        )
        b_up_quantized = tle_async.load(
            b_up_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
            is_async=True,
        )

        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        gate_b = dequant_weight(
            b_gate_quantized,
            b_scale_ptr,
            expert,
            k_tile * BLOCK_SIZE_K,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            compute_type,
            GROUP_SIZE_K,
            QUANT_TYPE,
        )
        up_b = dequant_weight(
            b_up_quantized,
            b_scale_ptr + num_columns * stride_bsn,
            expert,
            k_tile * BLOCK_SIZE_K,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            compute_type,
            GROUP_SIZE_K,
            QUANT_TYPE,
        )
        a_trans = tl.trans(a)
        acc_gate = tl.dot(tl.trans(gate_b), a_trans, acc=acc_gate)
        acc_up = tl.dot(tl.trans(up_b), a_trans, acc=acc_up)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_gate_block_ptr = tl.advance(b_gate_block_ptr, (BLOCK_SIZE_K_WEIGHT, 0))
        b_up_block_ptr = tl.advance(b_up_block_ptr, (BLOCK_SIZE_K_WEIGHT, 0))

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
        acc_gate *= routed_weight
        acc_up *= routed_weight

    acc = (acc_gate * tl.sigmoid(acc_gate)) * acc_up
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (route + offs_m[None, :]) * stride_cm + offs_n[:, None] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(compute_type),
        mask=(offs_m[None, :] == 0) & (offs_n[:, None] < num_columns),
    )


@triton.jit
def routed_activation_offset(
    rows,
    columns,
    EM: tl.constexpr,
    WIDTH: tl.constexpr,
    DEINTERLEAVE_K: tl.constexpr,
    PERMUTE_K16: tl.constexpr,
    PERMUTE_K32: tl.constexpr,
):
    rows = rows[:, None].to(tl.int64)
    if PERMUTE_K32:
        tl.static_assert(WIDTH % 32 == 0 and not PERMUTE_K16 and not DEINTERLEAVE_K)
        columns = store_k32(columns)
    elif PERMUTE_K16:
        tl.static_assert(WIDTH % 16 == 0 and not DEINTERLEAVE_K)
        columns = store_k16(columns)
    elif DEINTERLEAVE_K:
        return (
            rows * (WIDTH // 2)
            + columns[None, :] // 2
            + (columns[None, :] % 2).to(tl.int64) * EM * (WIDTH // 2)
        )
    return rows * WIDTH + columns[None, :]


@triton.jit
def stage_routed_activations_kernel(
    a_ptr,
    routed_a_ptr,
    sorted_token_ids_ptr,
    num_tokens_post_padded_ptr,
    EM: tl.constexpr,
    reduction_size: tl.constexpr,
    num_valid_tokens,
    stride_am,
    stride_ak,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DEINTERLEAVE_K: tl.constexpr = False,
    PERMUTE_K16: tl.constexpr = False,
    PERMUTE_K32: tl.constexpr = False,
    SKIP_INACTIVE: tl.constexpr = False,
):
    """Gather expert-sorted rows into a contiguous AIU input matrix."""
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if SKIP_INACTIVE:
        # Raw W8 only consumes complete tiles below the padded route count.
        if tl.program_id(0) * BLOCK_SIZE_M >= num_tokens_post_padded:
            return
    offs_m = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.program_id(1) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    route_mask = (offs_m < EM) & (offs_m < num_tokens_post_padded)
    routed_token = tl.load(
        sorted_token_ids_ptr + offs_m,
        mask=route_mask,
        other=num_valid_tokens,
    ).to(tl.int64)
    valid = route_mask[:, None] & (routed_token[:, None] < num_valid_tokens)
    activation = tl.load(
        a_ptr
        + (routed_token[:, None] // top_k) * stride_am
        + offs_k[None, :] * stride_ak,
        mask=valid & (offs_k[None, :] < reduction_size),
        other=0.0,
    )
    store_offset = routed_activation_offset(
        offs_m, offs_k, EM, reduction_size, DEINTERLEAVE_K, PERMUTE_K16, PERMUTE_K32
    )
    tl.store(
        routed_a_ptr + store_offset,
        activation,
        mask=(offs_m[:, None] < EM) & (offs_k[None, :] < reduction_size),
    )


@triton.jit
def silu_and_stage_routed_kernel(
    intermediate1_ptr,
    routed_intermediate2_ptr,
    sorted_token_ids_ptr,
    num_tokens_post_padded_ptr,
    EM: tl.constexpr,
    num_columns: tl.constexpr,
    num_valid_tokens,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    DEINTERLEAVE_K: tl.constexpr = False,
    PERMUTE_K16: tl.constexpr = False,
    PERMUTE_K32: tl.constexpr = False,
    SKIP_INACTIVE: tl.constexpr = False,
    SKIP_UNUSED_UPPER_ROWS: tl.constexpr = False,
    PAIRED_DOWN: tl.constexpr = False,
):
    """Fuse SwiGLU with the expert-sorted layout required by GEMM2.

    A separate SiLU call followed by ``_stage_routed_activations_kernel``
    would add one launch and a full intermediate write/read cycle.
    """
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    start = tl.program_id(0) * BLOCK_SIZE_M
    if SKIP_INACTIVE:
        # Raw W8 only consumes complete tiles below the padded route count.
        if start >= num_tokens_post_padded:
            return
    if SKIP_UNUSED_UPPER_ROWS:
        tl.static_assert(BLOCK_SIZE_M == 8)
        if start % 16 == 8:
            # The down GEMM bypasses this subtile if its first row is padding.
            first = tl.load(
                sorted_token_ids_ptr + start,
                (start < EM) & (start < num_tokens_post_padded),
                other=num_valid_tokens,
            )
            if first >= num_valid_tokens:
                return
    offs_m = start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    route_mask = (offs_m < EM) & (offs_m < num_tokens_post_padded)
    routed_token = tl.load(
        sorted_token_ids_ptr + offs_m,
        mask=route_mask,
        other=num_valid_tokens,
    ).to(tl.int64)
    valid = (
        route_mask[:, None]
        & (routed_token[:, None] < num_valid_tokens)
        & (offs_n[None, :] < num_columns)
    )
    row_base = routed_token[:, None] * (2 * num_columns)
    gate = tl.load(
        intermediate1_ptr + row_base + offs_n[None, :],
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        intermediate1_ptr + row_base + num_columns + offs_n[None, :],
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    output = (gate * tl.sigmoid(gate)) * up
    store_offset = routed_activation_offset(
        offs_m, offs_n, EM, num_columns, DEINTERLEAVE_K, PERMUTE_K16, PERMUTE_K32
    )
    if PERMUTE_K16 and PAIRED_DOWN:
        tl.static_assert(BLOCK_SIZE_M == 8)
        if SKIP_UNUSED_UPPER_ROWS:
            ninth = start // 16 * 16 + 8
            short = (
                tl.load(
                    sorted_token_ids_ptr + ninth, ninth < EM, other=num_valid_tokens
                )
                >= num_valid_tokens
            )
        else:
            short = True
        if short:
            # Interleave two K16 fragments as rows 0..7 and 8..15 of a tile.
            k = store_k16(offs_n)
            row = (offs_m.to(tl.int64) // 8 * 16 + offs_m % 8)[:, None] + (
                (k // 16) % 2 * 8
            )[None, :]
            column = k // 32 * 16 + k % 16
            store_offset = row * (num_columns // 2) + column[None, :]
    tl.store(
        routed_intermediate2_ptr + store_offset,
        output,
        mask=(offs_m[:, None] < EM) & (offs_n[None, :] < num_columns),
    )


@libentry()
@triton.jit(do_not_specialize_on_alignment=["routed_a_ptr", "b_ptr", "c_ptr"])
def marlin_moe_gemm_grouped_kernel(
    routed_a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    EM: tl.constexpr,
    num_valid_tokens,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    compute_type: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
):
    """Expert-grouped quantized GEMM using contiguous TLE AIU transfers."""
    IS_8BIT: tl.constexpr = (
        QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128 or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
    )
    BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(num_columns, BLOCK_SIZE_N)
    pid = tl.program_id(0)
    pid_m, pid_n = tl.swizzle2d(
        pid // num_pid_n, pid % num_pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M
    )

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    routed_token = tl.load(
        sorted_token_ids_ptr + offs_m,
        mask=offs_m < EM,
        other=num_valid_tokens,
    ).to(tl.int64)
    token_mask = (offs_m < EM) & (routed_token < num_valid_tokens)

    a_block_ptr = tl.make_block_ptr(
        base=routed_a_ptr,
        shape=(EM, reduction_size),
        strides=(reduction_size, 1),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + expert * stride_be,
        shape=(
            (reduction_size if IS_8BIT else tl.cdiv(reduction_size, 2)),
            num_columns,
        ),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
        order=(0, 1),
    )
    if not IS_8BIT and compute_type == tl.bfloat16 and BLOCK_SIZE_M <= 32:
        acc = w4_pair_gemm(
            routed_a_ptr,
            b_block_ptr,
            b_scale_ptr,
            expert,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            EM,
            pid_m,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_K,
            PIPELINE_STAGES,
            QUANT_TYPE,
        )
    else:
        acc = quantized_gemm_tile(
            a_block_ptr,
            b_block_ptr,
            b_scale_ptr,
            expert,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_K,
            PIPELINE_STAGES,
            compute_type,
            QUANT_TYPE,
            BLOCK_SIZE_K,
        )

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(
            topk_weights_ptr + routed_token,
            mask=token_mask,
            other=0.0,
        )
        acc *= routed_weight[None, :]

    c_ptrs = c_ptr + routed_token[None, :] * stride_cm + offs_n[:, None] * stride_cn
    if num_columns % BLOCK_SIZE_N == 0:
        tl.store(c_ptrs, acc.to(compute_type), mask=token_mask[None, :])
    else:
        tl.store(
            c_ptrs,
            acc.to(compute_type),
            mask=token_mask[None, :] & (offs_n[:, None] < num_columns),
        )


@libentry()
@triton.jit
def reduce_split_silu_kernel(
    partials,
    output_tile,
    router_weights,
    num_columns: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    MUL: tl.constexpr,
    BN: tl.constexpr,
):
    r = tl.program_id(0)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    gate = tl.full((BN,), 0, tl.float32)
    up = tl.full((BN,), 0, tl.float32)
    for part in tl.static_range(SPLIT_COUNT):
        gate += tl.load(
            partials + part * NUM_ROUTES * 2 * num_columns + r * 2 * num_columns + ns,
            ns < num_columns,
            other=0,
        )
        up += tl.load(
            partials
            + part * NUM_ROUTES * 2 * num_columns
            + r * 2 * num_columns
            + num_columns
            + ns,
            ns < num_columns,
            other=0,
        )
    if MUL:
        w = tl.load(router_weights + r)
        gate *= w
        up *= w
    # Match the original GEMM1 rounding before computing SiLU.
    gate = gate.to(output_tile.dtype.element_ty).to(tl.float32)
    up = up.to(output_tile.dtype.element_ty).to(tl.float32)
    result = tl.fdiv(gate, 1.0 + tl.exp(-gate)) * up
    tl.store(output_tile + r * num_columns + ns, result, ns < num_columns)


@libentry()
@triton.jit
def stage_bf16_activation_pairs_kernel(
    activations,
    Out,
    SortedTokenIds,
    NumTokensPostPadded,
    EM: tl.constexpr,
    K_PAIRS: tl.constexpr,
    num_valid_tokens,
    stride_am_pairs,
    TOP_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SKIP_UNUSED_UPPER_ROWS: tl.constexpr = False,
    PERMUTE_K32: tl.constexpr = False,
):
    padded = tl.load(NumTokensPostPadded)
    start = tl.program_id(0) * BLOCK_SIZE_M
    if start >= padded:
        return
    m = (start + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    k = tl.program_id(1) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    live = (m < EM) & (m < padded)
    token = tl.load(SortedTokenIds + m, live, num_valid_tokens).to(tl.int64)
    valid = live & (token < num_valid_tokens)
    a = activations.to(tl.pointer_type(tl.uint32))
    out = Out.to(tl.pointer_type(tl.uint32))
    pairs = tl.load(
        a + (token[:, None] // TOP_K) * stride_am_pairs + k[None, :],
        valid[:, None] & (k[None, :] < K_PAIRS),
        other=0,
    )
    # Permute whole BF16 pairs before the contiguous store.
    if PERMUTE_K32:
        grouped = tl.reshape(pairs, (BLOCK_SIZE_M, BLOCK_SIZE_K // 16, 4, 4))
        pairs = tl.reshape(
            tl.permute(grouped, (0, 1, 3, 2)), (BLOCK_SIZE_M, BLOCK_SIZE_K)
        )
    else:
        grouped = tl.reshape(pairs, (BLOCK_SIZE_M, BLOCK_SIZE_K // 8, 2, 2, 2))
        pairs = tl.reshape(
            tl.permute(grouped, (0, 1, 4, 2, 3)), (BLOCK_SIZE_M, BLOCK_SIZE_K)
        )
    write_rows = m < EM
    if SKIP_UNUSED_UPPER_ROWS:
        tl.static_assert(BLOCK_SIZE_M == 16)
        # The raw 16-row GEMM reads only eight rows when the ninth slot is padding.
        upper = start + 8
        upper_id = tl.load(
            SortedTokenIds + upper, (upper < EM) & (upper < padded), num_valid_tokens
        )
        write_rows &= (m < upper) | (upper_id < num_valid_tokens)
    tl.store(
        out + m[:, None] * K_PAIRS + k[None, :],
        pairs,
        write_rows[:, None] & (k[None, :] < K_PAIRS),
    )


@libentry()
@triton.jit
def tiny_fp8_down_reduce(
    Activation,
    Weight,
    Scale,
    route_ids,
    Router,
    Out,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    TOPK: tl.constexpr,
    SE,
    SN,
    SG,
    BN: tl.constexpr,
    MUL: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, reduction_size)
    total = tl.zeros((BN,), tl.float32)
    for j in tl.static_range(TOPK):
        route = token * TOPK + j
        expert = tl.load(route_ids + route).to(tl.int32)
        activation = tl.load(Activation + route * reduction_size + ks).to(tl.float32)
        quantized = tl.load(
            Weight
            + expert * (num_columns * reduction_size)
            + col[:, None] * reduction_size
            + ks[None, :]
        )
        scale = tl.load(
            Scale + expert * SE + col[:, None] * SN + (ks[None, :] // 128) * SG
        )
        weight = decode_fp8_scaled_bf16_normalized(quantized, scale)
        route_sum = tl.sum(activation[None, :] * weight.to(tl.float32), axis=1)
        if MUL:
            route_weight = tl.load(Router + route).to(tl.float32)
        else:
            route_weight = 1.0
        # Keep the BF16 boundary and route order of GEMM2 followed by moe_sum.
        total += (route_sum * route_weight).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + token * num_columns + col, total)
