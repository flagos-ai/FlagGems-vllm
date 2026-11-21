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

"""Quantized formats, register decoding and PPU tile instructions."""

import triton
import triton.language as tl

from flaggems_vllm.ops.fused_marlin_moe import (
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
)

try:
    from triton.experimental.tle import language as tle_async
except ImportError:
    tle_async = None


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
W8_TILE_BYTES = tl.constexpr(128 * 128)
W8_HALF_TILE_BYTES = tl.constexpr(64 * 128)
EXPERT_ID_MASK = tl.constexpr(0x3FFFFFFF)
SHORT_TILE_FLAG = tl.constexpr(0x40000000)
DENSE_BLOCK_ROWS = 64
_PPU_DENSE_MIN_TOKENS = 4096
_TL_QUANT_TYPE_UINT4B8 = tl.constexpr(QUANT_TYPE_UINT4B8)
_TL_QUANT_TYPE_UINT8B128 = tl.constexpr(QUANT_TYPE_UINT8B128)
_TL_QUANT_TYPE_FP4_E2M1 = tl.constexpr(QUANT_TYPE_FP4_E2M1)
_TL_QUANT_TYPE_FP8_E4M3 = tl.constexpr(QUANT_TYPE_FP8_E4M3)


@triton.jit
def decode_int8_scaled_bf16(q, scale):
    return tl.inline_asm_elementwise(
        "{ .reg .b32 q, l<3>, h<3>, v<2>;\n"
        "ppu.xor.b32 q,$2,0x80808080;\n"
        "ppu.prmt.b32 l0,q,0x43,0x4140;\n"
        "ppu.prmt.b32 h0,q,0x43,0x4342;\n"
        "ppu.and.b32 l1,l0,0xff7fff7f; ppu.and.b32 h1,h0,0xff7fff7f;\n"
        "ppu.and.b32 l2,l0,0xff80ff80; ppu.and.b32 h2,h0,0xff80ff80;\n"
        "ppu.sub.bf16x2 v0,l1,l2; ppu.sub.bf16x2 v1,h1,h2;\n"
        "ppu.mul.rn.bf16x2 v0,v0,$3; ppu.mul.rn.bf16x2 v1,v1,$4;\n"
        "ppu.mov.b32 $0,v0; ppu.mov.b32 $1,v1; }",
        "=r,=r,r,r,r",
        [q, scale],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )


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
def decode_raw_int4_bf16(b, scale):
    return tl.inline_asm_elementwise(
        "{ .reg .b32 lo,hi,l<2>,h<2>,bias;\n"
        "ppu.mov.b32 bias,0x43084308;\n"
        "ppu.and.b32 lo,$4,0x0f0f0f0f; ppu.shr.b32 hi,$4,4; ppu.and.b32 hi,hi,0x0f0f0f0f;\n"
        "ppu.prmt.b32 l0,lo,0x43,0x4140; ppu.prmt.b32 l1,lo,0x43,0x4342;\n"
        "ppu.prmt.b32 h0,hi,0x43,0x4140; ppu.prmt.b32 h1,hi,0x43,0x4342;\n"
        "ppu.sub.bf16x2 l0,l0,bias; ppu.sub.bf16x2 l1,l1,bias;\n"
        "ppu.sub.bf16x2 h0,h0,bias; ppu.sub.bf16x2 h1,h1,bias;\n"
        "ppu.mul.rn.bf16x2 l0,l0,$5; ppu.mul.rn.bf16x2 l1,l1,$6;\n"
        "ppu.mul.rn.bf16x2 h0,h0,$5; ppu.mul.rn.bf16x2 h1,h1,$6;\n"
        "ppu.mov.b32 $0,l0; ppu.mov.b32 $1,l1; ppu.mov.b32 $2,h0; ppu.mov.b32 $3,h1; }",
        "=r,=r,=r,=r,r,r,r",
        [b, scale],
        dtype=(tl.bfloat16, tl.bfloat16),
        is_pure=True,
        pack=4,
    )


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
        lo, hi = decode_raw_int4_bf16(b, scale)
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
):
    tl.static_assert(b.shape[0] == 128 or b.shape[0] == 64)
    group = k_base // GROUP_SIZE
    scale = tl.load(
        s_ptr + expert * se + group * sg + ns * sn,
        mask=ns < num_columns,
        other=0.0,
    )[None, :]
    if compute_type == tl.bfloat16 and (b.shape[1] <= 128 or b.shape[0] == 64):
        return decode_int8_scaled_bf16(b.to(tl.uint8), scale)
    else:
        quant = b.to(tl.int32) - 128
        return (quant.to(compute_type) * scale).to(compute_type)


@triton.jit
def decode_raw_mxfp4_bf16(b, scale):
    return tl.inline_asm_elementwise(
        (
            (
                "{ .reg .b32 lo,hi,l<2>,h<2>,m,sgn,bias;\n\nppu.mov.b32 bias,0x7e807e80;"
                "\n\nppu.and.b32 lo,$4,0x0f0f0f0f;\n ppu.shr.b32 hi,$4,4;"
                "\n ppu.and.b32 hi,hi,0x0f0f0f0f;\n\nppu.prmt.b32 l0,lo,0,0x4140;"
                "\n ppu.prmt.b32 l1,lo,0,0x4342;\n\nppu.prmt.b32 h0,hi,0,0x4140;"
                "\n ppu.prmt.b32 h1,hi,0,0x4342;\n\nppu.mul.lo.u32 l0,l0,0x1040;"
                " ppu.and.b32 l0,l0,0x81c081c0;\n\nppu.mul.rn.bf16x2 l0,l0,bias;"
                "\n ppu.mul.rn.bf16x2 l0,l0,$5;\n\nppu.mul.lo.u32 l1,l1,0x1040;"
                " ppu.and.b32 l1,l1,0x81c081c0;\n\nppu.mul.rn.bf16x2 l1,l1,bias;"
                "\n ppu.mul.rn.bf16x2 l1,l1,$6;\n\nppu.mul.lo.u32 h0,h0,0x1040;"
                " ppu.and.b32 h0,h0,0x81c081c0;\n\nppu.mul.rn.bf16x2 h0,h0,bias;"
                "\n ppu.mul.rn.bf16x2 h0,h0,$5;\n\nppu.mul.lo.u32 h1,h1,0x1040;"
                " ppu.and.b32 h1,h1,0x81c081c0;\n\nppu.mul.rn.bf16x2 h1,h1,bias;"
                "\n ppu.mul.rn.bf16x2 h1,h1,$6;\n\nppu.mov.b32 $0,l0;\n ppu.mov.b32 $1,l1;"
                "\n ppu.mov.b32 $2,h0;\n ppu.mov.b32 $3,h1;\n }"
            )
        ),
        "=r,=r,=r,=r,r,r,r",
        [b, scale],
        dtype=(tl.bfloat16, tl.bfloat16),
        is_pure=True,
        pack=4,
    )


@triton.jit
def e8m0_to_bf16_bits(byte):
    bits = tl.where(byte == 0, 0x0040, byte << 7)
    return tl.where(byte == 255, 0x7FC0, bits)


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
        lo, hi = decode_raw_mxfp4_bf16(b, scale)
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
):
    if QUANT_TYPE == _TL_QUANT_TYPE_UINT4B8:
        return dequant_int4(
            b,
            s_ptr,
            expert,
            k_base,
            ns,
            se,
            sg,
            sn,
            num_columns,
            compute_type,
            GROUP_SIZE,
        )
    elif QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128:
        return dequant_int8(
            b,
            s_ptr,
            expert,
            k_base,
            ns,
            se,
            sg,
            sn,
            num_columns,
            compute_type,
            GROUP_SIZE,
        )
    elif QUANT_TYPE == _TL_QUANT_TYPE_FP4_E2M1:
        return dequant_mxfp4(
            b,
            s_ptr,
            expert,
            k_base,
            ns,
            se,
            sg,
            sn,
            num_columns,
            reduction_size,
            compute_type,
        )
    else:
        return dequant_fp8(
            b,
            s_ptr,
            expert,
            k_base,
            ns,
            se,
            sg,
            sn,
            num_columns,
            reduction_size,
            compute_type,
            GROUP_SIZE,
        )


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
def decode_int8_fragment_bf16(q, s0, s1):
    return tl.inline_asm_elementwise(
        (
            (
                "{ .reg .b32 l<3>, h<3>, v<2>;\n\nppu.prmt.b32 l0,$2,0x43,0x4140;"
                "\nppu.prmt.b32 h0,$2,0x43,0x4342;\nppu.and.b32 l1,l0,0xff7fff7f;"
                " ppu.and.b32 h1,h0,0xff7fff7f;\nppu.lop3.b32 l2,l0,0xff80ff80,0x00800080,0x6a;"
                " ppu.lop3.b32 h2,h0,0xff80ff80,0x00800080,0x6a;\nppu.sub.bf16x2 v0,l1,l2;"
                " ppu.sub.bf16x2 v1,h1,h2;\nppu.mul.rn.bf16x2 v0,v0,$3;"
                " ppu.mul.rn.bf16x2 v1,v1,$4;\nppu.mov.b32 $0,v0; ppu.mov.b32 $1,v1; }"
            )
        ),
        "=r,=r,r,r,r",
        [q, s0, s1],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


PAIR_PERMUTE = (
    "{ .reg .b32 mag_0,inc_0,mask_0,hi_0,lo_0,tmp_0,v0_0,v1_0,bias_0;"
    " .reg .b32 mag_1,inc_1,mask_1,hi_1,lo_1,tmp_1,v0_1,v1_1,bias_1;"
    " ppu.and.b32 mag_0,$4,0x7f7f7f7f;"
    " ppu.and.b32 mag_1,$5,0x7f7f7f7f;"
    " ppu.add.u32 inc_0,mag_0,0x01010101;"
    " ppu.add.u32 inc_1,mag_1,0x01010101;"
    " ppu.prmt.b32 mask_0,inc_0,0,0xba98;"
    " ppu.prmt.b32 mask_1,inc_1,0,0xba98;"
    " ppu.lop3.b32 hi_0,$4,mask_0,0x80808080,0xe4;"
    " ppu.lop3.b32 hi_1,$5,mask_1,0x80808080,0xe4;"
    " ppu.shr.b32 tmp_0,$4,4;"
    " ppu.shr.b32 tmp_1,$5,4;"
    " ppu.lop3.b32 hi_0,hi_0,tmp_0,0x07070707,0xf8;"
    " ppu.lop3.b32 hi_1,hi_1,tmp_1,0x07070707,0xf8;"
    " ppu.shl.b32 lo_0,$4,4;"
    " ppu.shl.b32 lo_1,$5,4;"
    " ppu.and.b32 lo_0,lo_0,0xf0f0f0f0;"
    " ppu.and.b32 lo_1,lo_1,0xf0f0f0f0;"
    " ppu.prmt.b32 v0_0,lo_0,hi_0,0x5140;"
    " ppu.prmt.b32 v0_1,lo_1,hi_1,0x5140;"
    " ppu.prmt.b32 v1_0,lo_0,hi_0,0x7362;"
    " ppu.prmt.b32 v1_1,lo_1,hi_1,0x7362;"
)
PAIR_EXPONENT_BIAS = (
    " ppu.mov.b32 bias_0,0x7b807b80;"
    " ppu.mov.b32 bias_1,0x7b807b80;"
    " ppu.mul.rn.bf16x2 v0_0,v0_0,bias_0;"
    " ppu.mul.rn.bf16x2 v0_1,v0_1,bias_1;"
    " ppu.mul.rn.bf16x2 v1_0,v1_0,bias_0;"
    " ppu.mul.rn.bf16x2 v1_1,v1_1,bias_1;"
)
PAIR_MUL_STORE = (
    " ppu.mul.rn.bf16x2 v0_0,v0_0,$6;"
    " ppu.mul.rn.bf16x2 v0_1,v0_1,$7;"
    " ppu.mul.rn.bf16x2 v1_0,v1_0,$6;"
    " ppu.mul.rn.bf16x2 v1_1,v1_1,$7;"
    " ppu.mov.b32 $0,v0_0;"
    " ppu.mov.b32 $2,v0_1;"
    " ppu.mov.b32 $1,v1_0;"
    " ppu.mov.b32 $3,v1_1;"
    " }"
)
FOLDED_PAIR_ASM = tl.constexpr(PAIR_PERMUTE + PAIR_MUL_STORE)
ROBUST_PAIR_ASM = tl.constexpr(PAIR_PERMUTE + PAIR_EXPONENT_BIAS + PAIR_MUL_STORE)


@triton.jit
def decode_fp8_pair(q0, q1, s0, s1, FAST: tl.constexpr):
    return tl.inline_asm_elementwise(
        FOLDED_PAIR_ASM if FAST else ROBUST_PAIR_ASM,
        "=r,=r,=r,=r,r,r,r,r",
        [q0, q1, s0, s1],
        dtype=(tl.uint32,) * 4,
        is_pure=True,
        pack=1,
    )


@triton.jit
def fold_fp8_scale_fragment(s):
    return tl.inline_asm_elementwise(
        (
            "{ .reg .b32 bias; ppu.mov.b32 bias,0x7b807b80; ppu.mul.rn.bf16x2 $0,$1,bias; }"
        ),
        "=r,r",
        [s],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def load_w8_packed_scales(ptr, stride_n, FOLD: tl.constexpr):
    words = ptr.to(tl.pointer_type(tl.uint32))
    first = tl.load(words)
    second = tl.load(words + 4 * stride_n)
    if FOLD:
        first = fold_fp8_scale_fragment(first)
        second = fold_fp8_scale_fragment(second)
    return first, second


@triton.jit
def w8_scale_fragment(scale, k_tile, PACKED: tl.constexpr, FOLD: tl.constexpr):
    if PACKED:
        bits = (scale >> (16 * k_tile)) & 0xFFFF
    else:
        bits = scale.to(tl.uint16, bitcast=True).to(tl.uint32)
    fragment = bits | (bits << 16)
    if FOLD and not PACKED:
        fragment = fold_fp8_scale_fragment(fragment)
    return fragment


@triton.jit
def mma_bf16_m16n8(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
    return tl.inline_asm_elementwise(
        (
            "ppu.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};"
        ),
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        [a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def copy_w8_tile_async(
    dst, src, num_rows: tl.constexpr, KS, KT: tl.constexpr, start, copy_token
):
    # AIU copies are collective within the first warp.
    return tl.inline_asm_elementwise(
        (
            "{ .reg .u32 tid; .reg .pred p; ppu.mov.u32 tid,%tid.x; ppu.setp.lt.u32 p,tid,32;"
            " @p ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.zfill.2d.b8 "
            "[$1],[$2],{1,$3,$4,0,0,$5},{1,$3,$6,1}; ppu.mov.b32 $0,$7; }"
        ),
        "=r,r,l,r,r,r,r,r,~{memory}",
        [dst, src, num_rows, KS, start, KT, copy_token],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def prefetch_w8_tile(
    weight_smem,
    activation_smem,
    weight_bytes,
    activations,
    QN,
    reduction_size: tl.constexpr,
    row_tile,
    k_tile,
    copy_token,
    IS_SPLIT_Q: tl.constexpr = False,
):
    activation_base = (
        activations.to(tl.pointer_type(tl.uint8))
        + row_tile.to(tl.int64) * 8 * reduction_size * 2
    )
    copy_token = copy_w8_tile_async(
        activation_smem,
        activation_base,
        8,
        reduction_size * 2,
        128,
        k_tile * 256,
        copy_token,
    )
    copy_token = copy_w8_tile_async(
        activation_smem + 1024,
        activation_base,
        8,
        reduction_size * 2,
        128,
        k_tile * 256 + 128,
        copy_token,
    )
    if IS_SPLIT_Q:
        # Split the two warp groups into independent 8 KiB swizzled blocks.
        copy_token = copy_w8_tile_async(
            weight_smem, weight_bytes, 64, QN, 128, k_tile * 128, copy_token
        )
        copy_token = copy_w8_tile_async(
            weight_smem + W8_HALF_TILE_BYTES,
            weight_bytes + 64 * QN,
            64,
            QN,
            128,
            k_tile * 128,
            copy_token,
        )
    else:
        copy_token = copy_w8_tile_async(
            weight_smem, weight_bytes, 128, QN, 128, k_tile * 128, copy_token
        )
    return commit_w8_tile(copy_token)


@triton.jit
def commit_w8_tile(copy_token):
    return tl.inline_asm_elementwise(
        ("ppu.cp.async.commit_group; ppu.mov.b32 $0,$1;"),
        "=r,r,~{memory}",
        [copy_token],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def wait_w8_tile(copy_token, KEEP: tl.constexpr):
    if KEEP:
        return tl.inline_asm_elementwise(
            ("ppu.cp.async.wait_group 1; bar.sync 0; ppu.mov.b32 $0,$1;"),
            "=r,r,~{memory}",
            [copy_token],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        return tl.inline_asm_elementwise(
            ("ppu.cp.async.wait_group 0; bar.sync 0; ppu.mov.b32 $0,$1;"),
            "=r,r,~{memory}",
            [copy_token],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def mma_bf16_square(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7):
    return tl.inline_asm_elementwise(
        (
            "ppu.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 "
            "{$0,$1,$2,$3,$4,$5,$6,$7},{$8,$9,$10,$11},{$12,$13,$14,$15},{$16,$17,$18,$19,$20"
            ",$21,$22,$23};"
        ),
        "=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,r,r,r,r,f,f,f,f,f,f,f,f",
        [a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7],
        dtype=(tl.float32,) * 8,
        is_pure=False,
        pack=1,
    )


@triton.jit
def prefetch_square(
    weight_smem,
    activation_smem,
    weight_bytes,
    activations,
    QN,
    reduction_size: tl.constexpr,
    row_tile,
    k_tile,
    copy_token,
):
    activation_base = (
        activations.to(tl.pointer_type(tl.uint8))
        + row_tile.to(tl.int64) * 16 * reduction_size * 2
    )
    copy_token = copy_w8_tile_async(
        activation_smem,
        activation_base,
        16,
        reduction_size * 2,
        128,
        k_tile * 256,
        copy_token,
    )
    copy_token = copy_w8_tile_async(
        activation_smem + 2048,
        activation_base,
        16,
        reduction_size * 2,
        128,
        k_tile * 256 + 128,
        copy_token,
    )
    copy_token = copy_w8_tile_async(
        weight_smem, weight_bytes, 64, QN, 128, k_tile * 128, copy_token
    )
    copy_token = copy_w8_tile_async(
        weight_smem + W8_HALF_TILE_BYTES,
        weight_bytes + 64 * QN,
        64,
        QN,
        128,
        k_tile * 128,
        copy_token,
    )
    return commit_w8_tile(copy_token)


@triton.jit
def load_matrix_x4(shared, row, CW: tl.constexpr, channel, copy_token):
    # PPU x4 has row0-low, row0-high, row8-low, row8-high register order.
    return tl.inline_asm_elementwise(
        (
            "ppu.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 "
            "{$0,$1,$2,$3},[$4],{$5,$6,$7,$8,$9,$10};"
        ),
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,~{memory}",
        [shared, 0, row, 1, CW, 1, channel, copy_token],
        dtype=(tl.uint32,) * 4,
        is_pure=False,
        pack=1,
    )


@triton.jit
def decode_i4_register(q, s0, s1):
    return tl.inline_asm_elementwise(
        (
            "{ .reg .b32 mag,shifted,v0,v1,bias; ppu.mov.b32 bias,0x43084308;"
            " ppu.shr.b32 shifted,$2,4; ppu.prmt.b32 mag,$2,shifted,0x5140;"
            " ppu.and.b32 mag,mag,0x0f0f0f0f; ppu.prmt.b32 v0,mag,0x43,0x4140;"
            " ppu.prmt.b32 v1,mag,0x43,0x4342; ppu.sub.bf16x2 v0,v0,bias;"
            " ppu.sub.bf16x2 v1,v1,bias; ppu.mul.rn.bf16x2 v0,v0,$3;"
            " ppu.mul.rn.bf16x2 v1,v1,$4; ppu.mov.b32 $0,v0; ppu.mov.b32 $1,v1; }"
        ),
        "=r,=r,r,r,r",
        [q, s0, s1],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def decode_mx_register(q, s0, s1):
    return tl.inline_asm_elementwise(
        (
            "{ .reg .b32 ctrl,low,high,sign,v0,v1; ppu.and.b32 ctrl,$2,0x7777;"
            " ppu.prmt.b32 low,0xc0800000,0xc0804000,ctrl;"
            " ppu.prmt.b32 high,0x3f3f3f00,0x40404040,ctrl;"
            " ppu.prmt.b32 sign,0x80808080,0x80808080,$2; ppu.shl.b32 sign,sign,1;"
            " ppu.and.b32 sign,sign,0x80808080; ppu.or.b32 high,high,sign;"
            " ppu.prmt.b32 v0,low,high,0x5140; ppu.prmt.b32 v1,low,high,0x7362;"
            " ppu.mul.rn.bf16x2 v0,v0,$3; ppu.mul.rn.bf16x2 v1,v1,$4; ppu.mov.b32 $0,v0;"
            " ppu.mov.b32 $1,v1; }"
        ),
        "=r,=r,r,r,r",
        [q, s0, s1],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def store_k32(k):
    # FP4 has eight K values per loaded word; its permutation is self-inverse.
    return (k & ~31) | (k & 1) | ((k & 6) << 2) | ((k & 24) >> 2)


@triton.jit
def mx_scale_source(ptr, PACKED_SCALE: tl.constexpr):
    if PACKED_SCALE:
        return tl.load(ptr.to(tl.pointer_type(tl.uint32)))
    else:
        return ptr


@triton.jit
def mx_scale_fragment(
    source, stride_group, group: tl.constexpr, PACKED_SCALE: tl.constexpr
):
    if PACKED_SCALE:
        byte = (source >> (group * 8)) & 255
    else:
        byte = tl.load(source + group * stride_group).to(tl.uint32)
    bits = e8m0_to_bf16_bits(byte)
    return bits | (bits << 16)


@triton.jit
def decode_w4_fragment_pair(q0, q8, s0, s1, half: tl.constexpr, MX: tl.constexpr):
    lo, hi = q0 >> (half * 16), q8 >> (half * 16)
    if MX:
        a0, a1 = decode_mx_register(lo, s0, s0)
        a2, a3 = decode_mx_register(hi, s1, s1)
    else:
        a0, a1 = decode_i4_register(lo, s0, s0)
        a2, a3 = decode_i4_register(hi, s1, s1)
    return a0, a1, a2, a3


@triton.jit
def prefetch_w4_square(
    weight_smem,
    activation_smem,
    weight_bytes,
    activations,
    QN,
    reduction_size: tl.constexpr,
    row_tile,
    k_tile,
    copy_token,
    ROWS: tl.constexpr,
):
    copy_token = copy_w8_tile_async(
        weight_smem, weight_bytes, 64, QN, 64, k_tile * 64, copy_token
    )
    activation_base = (
        activations.to(tl.pointer_type(tl.uint8))
        + row_tile.to(tl.int64) * ROWS * reduction_size * 2
    )
    copy_token = copy_w8_tile_async(
        activation_smem,
        activation_base,
        ROWS,
        reduction_size * 2,
        128,
        k_tile * 256,
        copy_token,
    )
    copy_token = copy_w8_tile_async(
        activation_smem + ROWS * 128,
        activation_base,
        ROWS,
        reduction_size * 2,
        128,
        k_tile * 256 + 128,
        copy_token,
    )
    return commit_w8_tile(copy_token)


@triton.jit
def store_k16(k):
    # The permutation stays inside a scale group and avoids weight repacking.
    return (k & ~15) | (k & 1) | ((k & 2) << 2) | ((k & 12) >> 1)


@triton.jit
def load_activation_fragment_x2(shared, channel: tl.constexpr, copy_token):
    # Load the packed fragment directly from the AIU shared-memory swizzle.
    return tl.inline_asm_elementwise(
        "ppu.ldmatrix.sync.aligned.m8n8.x2.swzl.shared.b16 {$0,$1},[$2],{$3,$4,$5,$6,$7,$8};",
        "=r,=r,r,r,r,r,r,r,r,r,~{memory}",
        [shared, 0, 0, 1, 8, 1, channel, copy_token],
        dtype=(tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
