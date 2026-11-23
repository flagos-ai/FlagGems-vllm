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

"""T-Head PPU specialization for quantized fused Marlin MoE."""

from enum import Enum
from functools import lru_cache
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from flag_gems import zeros
from torch.utils.weak import WeakTensorKeyDictionary

from flaggems_vllm.ops.fused_marlin_moe import (
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
)
from flaggems_vllm.ops.fused_marlin_moe import (
    fused_marlin_moe as generic_fused_marlin_moe,
)
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_no_tle,
    moe_align_block_size_singleton,
    moe_align_block_size_small_grouped,
)
from flaggems_vllm.ops.moe_sum import moe_sum
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.utils import libentry

try:
    from triton.experimental.tle import language as tle_async
except ImportError:
    tle_async = None

_SCALE_CACHE = WeakTensorKeyDictionary()


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


LARGE_SCALE_WARP_THRESHOLD = 8 * 1024


@libentry()
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


@libentry()
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


_PPU_DIRECT_ROUTE_LIMIT = 32
MAX_COMPACT_SCALE_ELEMENTS = 16 * 1024
_PPU_WIDE_SHARED_BYTES = 64 << 10
_PPU_DENSE_MAX_DECODE_BYTES = 1 << 30


class ScaleLayout(NamedTuple):
    # Kernel argument order: expert, quantization group, output channel.
    strides: Tuple[int, int, int]
    max_offset: int


class QuantizedStage(NamedTuple):
    weight: torch.Tensor
    scale: torch.Tensor
    scale_safe: Optional[torch.Tensor]
    layout: ScaleLayout


def select_direct_block_n(n: int) -> int:
    if n <= 32:
        return 32
    if n < 512:
        return 64
    return 128


def select_grouped_config(
    num_rows: int,
    reduction_size: int,
    num_columns: int,
    block_m: int,
    quant_type_id: int,
) -> Tuple[int, int, int, int]:
    if quant_type_id == QUANT_TYPE_UINT8B128 and block_m <= 32:
        # Sparse INT8 expert batches spill with 256-wide decoded tiles. A
        # 128-wide tile keeps both GEMMs resident across the upstream shapes.
        return 128, 1, 4, 3
    if quant_type_id == QUANT_TYPE_FP8_E4M3 and block_m <= 32:
        # Sparse FP8 expert batches need smaller decoded tiles to avoid spills.
        return (
            128 if reduction_size > num_columns else 64,
            1 if num_rows < 512 else 8,
            4,
            3,
        )
    if 256 <= num_rows <= 512 and reduction_size > num_columns:
        return 128, 1, 4, 3
    if num_rows < 512:
        return 256, 1, 8, 3
    if num_rows >= 4096:
        return 128, 8, 4, 3
    if num_rows == 2048 and reduction_size > num_columns:
        return 256, 8, 8, 2
    return 256, 8, 8, 3


def select_grouped_block_m(num_rows: int, num_experts: int, top_k: int) -> int:
    routes = num_rows * top_k
    if routes <= 16 * num_experts:
        return 16
    if routes <= 32 * num_experts:
        return 32
    return 64


def use_direct_route(
    num_rows: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
) -> bool:
    routes = num_rows * top_k
    if num_experts <= 8 and routes >= num_experts:
        return False
    max_output_n = max(hidden_size, 2 * intermediate_size)
    block_n = select_direct_block_n(max_output_n)
    n_tiles = triton.cdiv(max_output_n, block_n)
    max_routes_by_grid = 65535 // n_tiles
    return routes <= min(_PPU_DIRECT_ROUTE_LIMIT, max_routes_by_grid)


class MoEShape(NamedTuple):
    tokens: int
    hidden: int
    intermediate: int
    experts: int
    top_k: int


def validate_w8(
    shape: MoEShape,
    activation_dtype: torch.dtype,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    group_size: int,
    quant_type_id: int,
) -> None:
    _, k, n, e, _ = shape
    if w1.shape != (e, 2 * n, k) or w2.shape != (e, k, n):
        raise ValueError("8-bit weight shapes do not match activations")
    allowed_group_sizes = (
        (-1, 32, 64, 128) if quant_type_id == QUANT_TYPE_FP8_E4M3 else (128,)
    )
    if group_size not in allowed_group_sizes:
        raise NotImplementedError(
            f"quant_type_id={quant_type_id} does not support group_size="
            f"{group_size}"
        )
    if group_size != -1 and (k % group_size or n % group_size):
        raise ValueError("Input dimensions must be divisible by group_size")
    g1 = 1 if group_size == -1 else k // group_size
    g2 = 1 if group_size == -1 else n // group_size
    if w1_scale.shape != (e, 2 * n, g1) or w2_scale.shape != (e, k, g2):
        raise ValueError("8-bit scale shapes must match the selected group size")
    if quant_type_id == QUANT_TYPE_FP8_E4M3:
        fp8_dtype = getattr(torch, "float8_e4m3fn", torch.uint8)
        if w1.dtype not in (torch.uint8, fp8_dtype) or w2.dtype not in (
            torch.uint8,
            fp8_dtype,
        ):
            raise NotImplementedError("Expected output-major E4M3FN weights")
        if w1_scale.dtype not in (torch.float16, torch.bfloat16, torch.float32) or (
            w2_scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise NotImplementedError("FP8 scales must be FP16, BF16 or FP32")
    elif w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise NotImplementedError("INT8 weights must use output-major uint8")
    elif w1_scale.dtype != activation_dtype or w2_scale.dtype != activation_dtype:
        raise NotImplementedError("INT8 scales must match activation dtype")


def validate_w4(
    shape: MoEShape,
    activation_dtype: torch.dtype,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    group_size: int,
    quant_type_id: int,
) -> None:
    _, k, n, e, _ = shape
    if w1.shape != (e, 2 * n, k // 2) or w2.shape != (e, k, n // 2):
        raise ValueError("4-bit weight shapes do not match activations")
    if w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise NotImplementedError("Expected output-major packed uint8 weights")
    expected_group_size = 128 if quant_type_id == QUANT_TYPE_UINT4B8 else 32
    if group_size != expected_group_size:
        raise NotImplementedError(
            f"quant_type_id={quant_type_id} requires group_size="
            f"{expected_group_size}"
        )
    if k % 128 or n % 128:
        raise ValueError(
            "4-bit hidden_size and intermediate_size must be multiples of 128"
        )
    if w1_scale.shape != (e, 2 * n, k // group_size) or w2_scale.shape != (
        e,
        k,
        n // group_size,
    ):
        raise ValueError("4-bit scale shapes do not match the selected group size")
    if quant_type_id == QUANT_TYPE_UINT4B8:
        if w1_scale.dtype != activation_dtype or w2_scale.dtype != activation_dtype:
            raise NotImplementedError("INT4 scales must match activation dtype")
    else:
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", torch.uint8)
        if w1_scale.dtype not in (torch.uint8, e8m0_dtype) or w2_scale.dtype not in (
            torch.uint8,
            e8m0_dtype,
        ):
            raise NotImplementedError("MXFP4 scales must use E8M0 bytes")


def validate_output(
    hidden_states: torch.Tensor,
    output: Optional[torch.Tensor],
    inplace: bool,
    live_inputs: Tuple[torch.Tensor, ...],
) -> None:
    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")
    if output is not None:
        if (
            output.shape != hidden_states.shape
            or output.dtype != hidden_states.dtype
            or output.device != hidden_states.device
        ):
            raise ValueError("Output shape/dtype/device must match hidden_states")
        if not output.is_contiguous() or output.requires_grad:
            raise ValueError("Output must be contiguous and inference-only")
    # Weights, scales and routing remain live during GEMM2.
    target = hidden_states if inplace else output
    if target is not None and target.numel():
        for tensor in live_inputs:
            if (
                tensor.numel()
                and target.untyped_storage().data_ptr()
                == tensor.untyped_storage().data_ptr()
            ):
                raise ValueError("Output must not alias weights, scales or routing")


def validate_inputs(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: Optional[torch.Tensor],
    inplace: bool,
    group_size: int,
    quant_type_id: int,
) -> MoEShape:
    if any(tensor.layout != torch.strided for tensor in (w1, w2, w1_scale, w2_scale)):
        raise NotImplementedError(
            "PPU Marlin weights and scales require strided tensors"
        )
    if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or topk_ids.ndim != 2:
        raise ValueError("Expected activations/routing rank 2 and weights rank 3")
    if not w1.is_contiguous() or not w2.is_contiguous():
        raise NotImplementedError(
            "PPU Marlin requires contiguous output-major weights with unit K stride"
        )
    m, k = hidden_states.shape
    e, n2, _ = w1.shape
    n = n2 // 2
    if e <= 0 or n2 % 2 or min(k, n) <= 0 or k % 32 or n % 32:
        raise ValueError(
            "Positive K/N multiples of 32 and paired gate/up weights required"
        )
    if (
        topk_weights.shape != topk_ids.shape
        or topk_ids.shape[0] != m
        or not 1 <= topk_ids.shape[1] <= e
    ):
        raise ValueError("Routing must have shape [M, topk], 1 <= topk <= E")
    shape = MoEShape(m, k, n, e, topk_ids.shape[1])
    validate_format = (
        validate_w8
        if quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3)
        else validate_w4
    )
    validate_format(
        shape,
        hidden_states.dtype,
        w1,
        w2,
        w1_scale,
        w2_scale,
        group_size,
        quant_type_id,
    )
    if any(
        scale.size(-1) > 1 and scale.stride(-1) != 1 for scale in (w1_scale, w2_scale)
    ):
        raise NotImplementedError(
            "PPU Marlin scales require unit stride along the group dimension"
        )
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("PPU fused Marlin MoE requires FP16 or BF16")
    if topk_ids.dtype not in (torch.int32, torch.int64) or topk_weights.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise NotImplementedError("Unsupported routing dtype")
    tensors = (hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids)
    if hidden_states.device.type != "cuda" or any(
        tensor.device != hidden_states.device for tensor in tensors
    ):
        raise ValueError("All tensors must reside on the same PPU device")
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("PPU fused Marlin MoE is inference-only")
    if any(
        not tensor.is_contiguous() for tensor in (hidden_states, topk_weights, topk_ids)
    ):
        raise NotImplementedError("Activations and routing must be contiguous")
    validate_output(
        hidden_states,
        output,
        inplace,
        (w1, w2, w1_scale, w2_scale, topk_weights, topk_ids),
    )
    return shape


def scale_layout(scale: torch.Tensor, is_mxfp4: bool) -> ScaleLayout:
    shape, strides = scale.shape, scale.stride()
    version = scale._version if is_mxfp4 and not scale.is_inference() else None
    if version is not None:
        cached = _SCALE_CACHE.get(scale)
        # Rebinding .data can change shape or strides without bumping the version.
        if cached is not None and cached[:3] == (version, shape, strides):
            return cached[3]

    se, sn, sg = strides
    layout = ScaleLayout(
        (se, sg, sn),
        sum((size - 1) * stride for size, stride in zip(shape, strides)),
    )
    if version is not None:
        _SCALE_CACHE[scale] = (version, shape, strides, layout)
    return layout


def prepare_quantized_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    needs_expert_flags: bool,
    is_mxfp4: bool,
) -> Tuple[QuantizedStage, QuantizedStage]:
    b1 = w1.view(torch.uint8).transpose(1, 2)
    b2 = w2.view(torch.uint8).transpose(1, 2)
    s1 = w1_scale.view(torch.uint8) if is_mxfp4 else w1_scale
    s2 = w2_scale.view(torch.uint8) if is_mxfp4 else w2_scale
    if needs_expert_flags:
        safe1 = torch.empty((w1.size(0),), device=w1.device, dtype=torch.int32)
        safe2 = torch.empty_like(safe1)
    else:
        safe1, safe2 = None, None
    return (
        QuantizedStage(b1, s1, safe1, scale_layout(w1_scale, is_mxfp4)),
        QuantizedStage(b2, s2, safe2, scale_layout(w2_scale, is_mxfp4)),
    )


class MarlinCall(NamedTuple):
    shape: MoEShape
    hidden_states: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    out: torch.Tensor
    group_size: int
    quant_type_id: int
    apply_router_weight_on_input: bool
    compute_type: tl.dtype


class SparsePlan(NamedTuple):
    packed_fp8: bool
    wide_fp8_gate_shape: bool
    use_wide_tile: bool
    raw_w8: bool
    raw_w4: bool
    direct: bool
    is_tiny_int8_matvec: bool
    small_fp8_gate: bool
    decode_fp8_gate: bool
    is_joint_safety: bool


def raw_tile_modes(call: MarlinCall, use_wide_tile: bool) -> Tuple[bool, bool]:
    tokens, hidden, intermediate, experts, top_k = call.shape
    if (
        call.compute_type != tl.bfloat16
        or hidden % 128
        or intermediate % 128
        or not (tokens * top_k <= 16 * experts or use_wide_tile)
    ):
        return False, False
    bf16_group = (
        call.group_size == 128
        and call.w1_scale.dtype == call.w2_scale.dtype == torch.bfloat16
    )
    raw_w8 = bf16_group and call.quant_type_id in (
        QUANT_TYPE_UINT8B128,
        QUANT_TYPE_FP8_E4M3,
    )
    raw_w4 = (bf16_group and call.quant_type_id == QUANT_TYPE_UINT4B8) or (
        call.group_size == 32 and call.quant_type_id == QUANT_TYPE_FP4_E2M1
    )
    return raw_w8, raw_w4


def direct_schedule(
    call: MarlinCall,
    workspace_limit: int,
    packed_fp8: bool,
    wide_fp8_gate_shape: bool,
    raw_w8: bool,
    raw_w4: bool,
) -> Tuple[bool, bool, bool, bool]:
    tokens, hidden, intermediate, experts, top_k = call.shape
    routes = tokens * top_k
    # Sparse raw tiles pad to eight rows; account for both staged activations.
    padded_routes = routes + 7 * experts
    sparse_bytes = (
        4 * (experts + 1) ** 2
        + 16 * padded_routes
        + 2 * padded_routes * (hidden + intermediate)
        + 2 * tokens * top_k * (hidden + 2 * intermediate)
        + 8
    )
    sparse_raw = raw_w4 or (raw_w8 and call.quant_type_id == QUANT_TYPE_UINT8B128)
    direct = use_direct_route(
        tokens,
        experts,
        top_k,
        hidden,
        intermediate,
    ) and not (sparse_raw and sparse_bytes <= workspace_limit)
    # A one-token gate/up grid may not fill the PPU without split K.
    is_tiny_int8_matvec = (
        raw_w8
        and call.quant_type_id == QUANT_TYPE_UINT8B128
        and tokens == 1
        and hidden >= 1024
        and hidden % 1024 == 0
        and 2 * hidden < (1 << 17)
        and 2 * intermediate <= 1024
        and top_k * triton.cdiv(2 * intermediate, 128) < 32
    )
    direct = direct or is_tiny_int8_matvec
    if (
        packed_fp8
        and experts > 8
        and routes <= 160
        and max(hidden, 2 * intermediate) <= 4096
        and not raw_w8
    ):
        direct = True
    # Reuse routed weights once the grouped projection can fill the PPU.
    is_raw_fp8_sparse = (
        raw_w8
        and packed_fp8
        and routes <= min(32, experts // 2)
        and tokens * top_k * triton.cdiv(2 * intermediate, 128) >= 64
        and sparse_bytes <= workspace_limit
    )
    small_fp8_gate = wide_fp8_gate_shape and tokens in (1, 2, 4)
    decode_fp8_gate = (
        wide_fp8_gate_shape
        and raw_w8
        and 4 < tokens <= 32
        and 32 <= routes <= 192
        # Denser routes reuse expert weights in the grouped projection.
        and 4 * routes <= 3 * experts
    )
    direct = direct and (not is_raw_fp8_sparse or (small_fp8_gate and tokens == 4))
    direct = direct or decode_fp8_gate
    return direct, is_tiny_int8_matvec, small_fp8_gate, decode_fp8_gate


def select_sparse_plan(call: MarlinCall, workspace_limit: int) -> SparsePlan:
    tokens, hidden, intermediate, experts, top_k = call.shape
    routes = tokens * top_k
    packed_fp8 = (
        call.quant_type_id == QUANT_TYPE_FP8_E4M3
        and call.hidden_states.dtype
        == call.w1_scale.dtype
        == call.w2_scale.dtype
        == torch.bfloat16
    )
    wide_gate_shape = hidden == 4096 and intermediate == 256 and 4 <= top_k <= 8
    wide_fp8_gate_shape = packed_fp8 and call.group_size == 128 and wide_gate_shape
    # A 64-row tile reuses each decoded weight fragment across three or four MMAs.
    use_wide_tile = (
        wide_gate_shape
        and call.hidden_states.dtype == torch.bfloat16
        and 1024 <= tokens < _PPU_DENSE_MIN_TOKENS
        and 32 * experts < routes <= 64 * experts
        and torch.cuda.get_device_properties(
            call.hidden_states.device
        ).shared_memory_per_block_optin
        >= _PPU_WIDE_SHARED_BYTES
    )
    raw_w8, raw_w4 = raw_tile_modes(call, use_wide_tile)
    direct, is_tiny_int8_matvec, small_fp8_gate, decode_fp8_gate = direct_schedule(
        call, workspace_limit, packed_fp8, wide_fp8_gate_shape, raw_w8, raw_w4
    )
    is_joint_safety = (
        raw_w8
        and packed_fp8
        and not direct
        and tokens < _PPU_DENSE_MIN_TOKENS
        and max(
            call.w1_scale.size(1) * call.w1_scale.size(2),
            call.w2_scale.size(1) * call.w2_scale.size(2),
        )
        <= MAX_COMPACT_SCALE_ELEMENTS
    )
    return SparsePlan(
        packed_fp8,
        wide_fp8_gate_shape,
        use_wide_tile,
        raw_w8,
        raw_w4,
        direct,
        is_tiny_int8_matvec,
        small_fp8_gate,
        decode_fp8_gate,
        is_joint_safety,
    )


def eligible_for_dense(call: MarlinCall) -> bool:
    return (
        call.shape.tokens >= _PPU_DENSE_MIN_TOKENS
        and call.shape.tokens * call.shape.top_k > 32 * call.shape.experts
        and call.hidden_states.dtype == torch.bfloat16
        and (
            (
                call.group_size == 128
                and call.w1_scale.dtype == call.w2_scale.dtype == torch.bfloat16
            )
            or call.quant_type_id == QUANT_TYPE_FP4_E2M1
        )
        and max(call.w1.numel(), call.w2.numel())
        * (4 if call.quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1) else 2)
        <= _PPU_DENSE_MAX_DECODE_BYTES
    )


def direct_split_count(call: MarlinCall, plan: SparsePlan) -> int:
    gemv_split_k = 1
    if (
        (plan.packed_fp8 or plan.is_tiny_int8_matvec)
        and call.group_size == 128
        and call.shape.hidden >= 1024
        and call.shape.hidden % 1024 == 0
        and 2 * call.shape.intermediate <= 1024
    ):
        routes = call.shape.tokens * call.shape.top_k
        # Fill the compute units for very sparse routes; the 33--64 route
        # band already has enough column tiles without another reduction.
        if plan.small_fp8_gate:
            # The wide gate tiles need more CTAs at low token counts.
            gemv_split_k = 16 // call.shape.tokens
        elif plan.decode_fp8_gate:
            gemv_split_k = 2 if routes <= 64 else 8
        else:
            gemv_split_k = (
                min(8, 64 // triton.next_power_of_2(routes)) if routes <= 64 else 4
            )
    return gemv_split_k


MIN_OVERLAP_TOKENS = 8192


@libentry()
@triton.jit
def decode_bank(
    weight_bytes,
    scales,
    decoded,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    QE: tl.constexpr,
    QN: tl.constexpr,
    QK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    QTYPE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    FP8: tl.constexpr = QTYPE == _TL_QUANT_TYPE_FP8_E4M3
    W4: tl.constexpr = (
        QTYPE == _TL_QUANT_TYPE_UINT4B8 or QTYPE == _TL_QUANT_TYPE_FP4_E2M1
    )
    expert = tl.program_id(2).to(tl.int64)
    ns = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    k_base = (tl.program_id(1) * BK).to(tl.int64)
    ks = k_base + tl.arange(0, BK)
    mask = (ns[:, None] < num_columns) & (ks[None, :] < reduction_size)
    if W4:
        kps = k_base // 2 + tl.arange(0, BK // 2)
        q = tl.load(
            weight_bytes + expert * QE + ns[:, None] * QN + kps[None, :] * QK,
            (ns[:, None] < num_columns) & (kps[None, :] * 2 < reduction_size),
            other=0,
        )
        if QTYPE == _TL_QUANT_TYPE_UINT4B8:
            scale = tl.load(
                scales + expert * SE + ns * SN + (k_base // 128) * SG,
                ns < num_columns,
                other=0,
            )[:, None]
            lo, hi = decode_raw_int4_bf16(q, scale)
        else:
            gs = k_base // 32 + tl.arange(0, BK // 32)
            byte = tl.load(
                scales + expert * SE + ns[:, None] * SN + gs[None, :] * SG,
                (ns[:, None] < num_columns) & (gs[None, :] * 32 < reduction_size),
                other=127,
            ).to(tl.uint32)
            bits = e8m0_to_bf16_bits(byte)
            scale = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
            scale = tl.reshape(
                tl.broadcast_to(scale[:, :, None], (BN, BK // 32, 16)), (BN, BK // 2)
            )
            lo, hi = decode_raw_mxfp4_bf16(q, scale)
        value = tl.reshape(tl.join(lo, hi), (BN, BK))
    else:
        q = tl.load(
            weight_bytes + expert * QE + ns[:, None] * QN + ks[None, :] * QK,
            mask,
            other=0 if FP8 else 128,
        )
        scale = tl.load(
            scales + expert * SE + ns * SN + (k_base // 128) * SG,
            ns < num_columns,
            other=0,
        )[:, None]
        if FP8:
            # Normalize the FP8 value before multiplying by the original scale.
            value = decode_fp8_scaled_bf16_normalized(q, scale)
        else:
            value = decode_int8_scaled_bf16(q, scale)
    tl.store(
        decoded
        + expert * num_columns * reduction_size
        + ns[:, None]
        + ks[None, :] * num_columns,
        value,
        mask,
    )


@triton.jit
def dense_w8_bf16_tile(
    activations,
    B,
    output_tile,
    router_weights,
    route_ids,
    block_experts,
    row_tile,
    col_tile,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    EM: tl.constexpr,
    VALID,
    AM,
    AK,
    BE,
    BKSTR,
    BNSTR,
    CM,
    CN,
    TOPK: tl.constexpr,
    MUL: tl.constexpr,
    USE32: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    STAGES: tl.constexpr,
    DIRECT_B: tl.constexpr = False,
    FIRST_EXPERT: tl.constexpr = 0,
):
    offsets = row_tile * BM + tl.arange(0, BM)
    tokens = tl.load(route_ids + offsets, offsets < EM, other=VALID)
    expert = tl.load(block_experts + row_tile) - FIRST_EXPERT
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    tokens = tokens.to(offset_type)
    expert = expert.to(offset_type)
    mask = tokens < VALID
    ks = tl.arange(0, BK)
    abase = activations + (tokens[:, None] // TOPK) * AM + ks[None, :] * AK
    bp = tl.make_block_ptr(
        base=B + expert * BE,
        shape=(reduction_size, num_columns),
        strides=(BKSTR, BNSTR),
        offsets=(0, col_tile * BN),
        block_shape=(BK, BN),
        order=(1, 0),
    )
    acc = tl.zeros((BM, BN), tl.float32)
    for k_tile in tl.range(0, tl.cdiv(reduction_size, BK), num_stages=STAGES):
        a = tl.load(
            abase,
            mask=mask[:, None] & (k_tile * BK + ks[None, :] < reduction_size),
            other=0.0,
        )
        if DIRECT_B:
            cols = col_tile * BN + tl.arange(0, BN)
            b = tl.load(
                B
                + expert * BE
                + (k_tile * BK + ks[:, None]) * BKSTR
                + cols[None, :] * BNSTR
            )
        elif num_columns % BN == 0 and reduction_size % BK == 0:
            b = tle_async.load(bp, is_async=True)
        else:
            b = tle_async.load(
                bp, boundary_check=(0, 1), padding_option="zero", is_async=True
            )
        acc = tl.dot(a, b, acc=acc)
        abase += BK * AK
        bp = tl.advance(bp, (BK, 0))
    if MUL:
        w = tl.load(router_weights + tokens, mask, other=0).to(tl.float32)
    else:
        w = tl.full((BM,), 1.0, tl.float32)
    acc *= w[:, None]
    ns = col_tile * BN + tl.arange(0, BN)
    tl.store(
        output_tile + tokens[:, None] * CM + ns[None, :] * CN,
        acc.to(output_tile.dtype.element_ty),
        mask[:, None] & (ns[None, :] < num_columns),
    )


@libentry()
@triton.jit
def dense_w8_bf16_kernel(
    activations,
    B,
    output_tile,
    router_weights,
    route_ids,
    block_experts,
    padded_count,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    EM: tl.constexpr,
    VALID,
    AM,
    AK,
    BE,
    BKSTR,
    BNSTR,
    CM,
    CN,
    TOPK: tl.constexpr,
    MUL: tl.constexpr,
    USE32: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    STAGES: tl.constexpr,
    DIRECT_B: tl.constexpr = False,
    REVERSE_M: tl.constexpr = False,
):
    """Gather routed activations and load a decoded BF16 bank."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(EM, BM)
    num_n = tl.cdiv(num_columns, BN)
    row_tile, col_tile = tl.swizzle2d(pid // num_n, pid % num_n, num_m, num_n, GM)
    live_rows = tl.load(padded_count)
    if row_tile * BM >= live_rows:
        return
    if REVERSE_M:
        row_tile = live_rows // BM - 1 - row_tile
    dense_w8_bf16_tile(
        activations,
        B,
        output_tile,
        router_weights,
        route_ids,
        block_experts,
        row_tile,
        col_tile,
        num_columns,
        reduction_size,
        EM,
        VALID,
        AM,
        AK,
        BE,
        BKSTR,
        BNSTR,
        CM,
        CN,
        TOPK,
        MUL,
        USE32,
        BM,
        BN,
        BK,
        STAGES,
        DIRECT_B,
    )


@libentry()
@triton.jit
def dense_w8_bf16_expert_window(
    activations,
    B,
    output_tile,
    router_weights,
    route_ids,
    block_experts,
    Prefix,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    EM: tl.constexpr,
    VALID,
    AM,
    AK,
    BE,
    BKSTR,
    BNSTR,
    CM,
    CN,
    TOPK: tl.constexpr,
    MUL: tl.constexpr,
    USE32: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    STAGES: tl.constexpr,
    FIRST_EXPERT: tl.constexpr,
    LAST_EXPERT: tl.constexpr,
):
    begin = tl.load(Prefix + FIRST_EXPERT)
    end = tl.load(Prefix + LAST_EXPERT)
    num_m = (end - begin) // BM
    num_n = tl.cdiv(num_columns, BN)
    # Reuse CTAs for overflow tiles when routes concentrate in one window.
    for pid in range(tl.program_id(0), num_m * num_n, tl.num_programs(0)):
        row_tile, col_tile = tl.swizzle2d(pid // num_n, pid % num_n, num_m, num_n, GM)
        row_tile = end // BM - 1 - row_tile
        dense_w8_bf16_tile(
            activations,
            B,
            output_tile,
            router_weights,
            route_ids,
            block_experts,
            row_tile,
            col_tile,
            num_columns,
            reduction_size,
            EM,
            VALID,
            AM,
            AK,
            BE,
            BKSTR,
            BNSTR,
            CM,
            CN,
            TOPK,
            MUL,
            USE32,
            BM,
            BN,
            BK,
            STAGES,
            False,
            FIRST_EXPERT,
        )


@libentry()
@triton.jit
def dense_align_count_histogram(
    route_ids,
    Count,
    Prefix,
    sorted_routes,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EM: tl.constexpr,
    RB: tl.constexpr,
    ID_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    expert = tl.arange(0, NUM_EXPERTS)
    row = pid * RB + tl.arange(0, RB)
    valid = row < NUM_ROUTES
    ids = tl.load(route_ids + row, valid, other=0).to(tl.int32)
    val = tl.histogram(ids, NUM_EXPERTS, mask=valid)
    tl.store(Count + (pid + 1) * NUM_EXPERTS + expert, val)
    if pid == 0:
        tl.store(Prefix, 0)
        tl.store(Count + expert, 0)
    o = pid * ID_BLOCK + tl.arange(0, ID_BLOCK)
    tl.store(sorted_routes + o, NUM_ROUTES, o < EM)


def align_dense_fp8_histogram(
    ids: torch.Tensor, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from flaggems_vllm.ops.moe_align_block_size import (
        moe_align_block_size_stage2_vec,
        moe_align_block_size_stage3,
        moe_align_block_size_stage4,
    )

    r = ids.numel()
    e = num_experts
    em = r + (DENSE_BLOCK_ROWS - 1) * e
    mb = triton.cdiv(em, 64)
    rb = triton.next_power_of_2(triton.cdiv(r, e))
    sorted_ids = torch.empty((em,), device=ids.device, dtype=torch.int32)
    experts = torch.empty((mb,), device=ids.device, dtype=torch.int32)
    post = torch.empty((1,), device=ids.device, dtype=torch.int32)
    count = torch.empty((e + 1, e), device=ids.device, dtype=torch.int32)
    prefix = torch.empty((e + 1,), device=ids.device, dtype=torch.int32)
    dense_align_count_histogram[(e,)](
        ids,
        count,
        prefix,
        sorted_ids,
        r,
        e,
        em,
        rb,
        triton.next_power_of_2(triton.cdiv(em, e)),
        num_warps=4,
    )
    moe_align_block_size_stage2_vec[(e,)](count, e)
    moe_align_block_size_stage3[(1,)](post, count, prefix, e, e, 64)

    moe_align_block_size_stage4[(e,)](
        ids, sorted_ids, experts, count, prefix, e, 64, r, rb
    )
    return sorted_ids, experts, post, prefix


@lru_cache(maxsize=16)
def eager_decode_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device, priority=-1)


def decode_dense_bank(
    weight: torch.Tensor,
    scales: torch.Tensor,
    destination: torch.Tensor,
    quant_type_id: int,
) -> None:
    experts, width, packed_k = weight.shape
    reduction = (
        packed_k * 2
        if quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1)
        else packed_k
    )
    columns = 128 if quant_type_id == QUANT_TYPE_FP8_E4M3 and width >= 128 else 64
    decode_bank[(triton.cdiv(width, columns), triton.cdiv(reduction, 128), experts)](
        weight.view(torch.uint8),
        scales,
        destination,
        width,
        reduction,
        *weight.stride(),
        *scales.stride(),
        quant_type_id,
        columns,
        128,
        num_warps=8 if columns == 128 else 4,
        num_stages=1,
    )


def run_fp8_ordered_overlap(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    router: torch.Tensor,
    output: torch.Tensor,
    ids: torch.Tensor,
    input_mul: bool,
    banks: torch.Tensor,
    gateup: torch.Tensor,
    middle: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    m, k = hidden.shape
    experts_count, n2, _ = w1.shape
    n = n2 // 2
    topk = ids.size(1)
    routes = m * topk
    half = experts_count // 2
    # Each bank holds one gate/up expert window or one down-projection column tile.
    gate_bank, up_bank = banks.chunk(2)
    gateup = gateup.view(routes, n2)
    down_tile = down.size(-1)

    main = torch.cuda.current_stream(hidden.device)
    # Eager callers can share a work queue; graph capture needs its own fork/join stream.
    if torch.cuda.is_current_stream_capturing():
        side = torch.cuda.Stream(device=hidden.device, priority=-1)
    else:
        side = eager_decode_stream(hidden.device)
    for operand in (banks, w1, w2, w1_scale, w2_scale, ids):
        operand.record_stream(side)

    first, second = slice(half, experts_count), slice(0, half)
    gate_weight, gate_scale = w1[:, :n], w1_scale[:, :n]
    up_weight, up_scale = w1[:, n:], w1_scale[:, n:]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        sorted_ids, experts, padded, prefix = align_dense_fp8_histogram(
            ids, experts_count
        )
    decode_dense_bank(
        gate_weight[first], gate_scale[first], gate_bank, QUANT_TYPE_FP8_E4M3
    )
    main.wait_stream(side)
    for metadata in (sorted_ids, experts, padded, prefix):
        metadata.record_stream(main)
    em = sorted_ids.numel()
    row_tiles = triton.cdiv(em, 64)

    def project_gate(
        bank: torch.Tensor, projected: torch.Tensor, window: slice
    ) -> None:
        dense_w8_bf16_expert_window[(triton.cdiv(row_tiles, 2),)](
            hidden,
            bank,
            projected,
            router,
            sorted_ids,
            experts,
            prefix,
            n,
            k,
            em,
            routes,
            *hidden.stride(),
            k * n,
            n,
            1,
            n2,
            1,
            topk,
            input_mul,
            True,
            64,
            256,
            32,
            8,
            4,
            num_warps=4,
            num_stages=4,
            FIRST_EXPERT=window.start,
            LAST_EXPERT=window.stop,
        )

    gate_output, up_output = gateup[:, :n], gateup[:, n:]
    for window in (first, second):
        side.wait_stream(main)
        project_gate(gate_bank, gate_output, window)
        with torch.cuda.stream(side):
            decode_dense_bank(
                up_weight[window], up_scale[window], up_bank, QUANT_TYPE_FP8_E4M3
            )
        main.wait_stream(side)
        side.wait_stream(main)
        project_gate(up_bank, up_output, window)
        with torch.cuda.stream(side):
            if window == first:
                decode_dense_bank(
                    gate_weight[second],
                    gate_scale[second],
                    gate_bank,
                    QUANT_TYPE_FP8_E4M3,
                )
            else:
                decode_dense_bank(
                    w2[:, :down_tile],
                    w2_scale[:, :down_tile],
                    gate_bank,
                    QUANT_TYPE_FP8_E4M3,
                )
        main.wait_stream(side)
    silu_and_mul_out(gate_output, up_output, middle)
    for col in range(0, k, down_tile):
        end = col + down_tile
        if end < k:
            side.wait_stream(main)
        dense_w8_bf16_kernel[(row_tiles * (down_tile // 256),)](
            middle,
            gate_bank,
            down,
            router,
            sorted_ids,
            experts,
            padded,
            down_tile,
            n,
            em,
            routes,
            *middle.stride(),
            down_tile * n,
            down_tile,
            1,
            down_tile,
            1,
            1,
            not input_mul,
            True,
            64,
            256,
            32,
            8,
            4,
            DIRECT_B=True,
            REVERSE_M=True,
            num_warps=4,
            num_stages=4,
        )
        moe_sum(down, output[:, col:end])
        if end < k:
            with torch.cuda.stream(side):
                decode_dense_bank(
                    w2[:, end : end + down_tile],
                    w2_scale[:, end : end + down_tile],
                    up_bank,
                    QUANT_TYPE_FP8_E4M3,
                )
            main.wait_stream(side)
            gate_bank, up_bank = up_bank, gate_bank
    return output


def project_dense_w8(
    activation: torch.Tensor,
    decoded: torch.Tensor,
    projected: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_ids: torch.Tensor,
    experts: torch.Tensor,
    padded: torch.Tensor,
    multiply_router: bool,
    projection_topk: int,
    config: Dict[str, int],
    is_down: bool,
) -> None:
    _, width, reduction = decoded.shape
    # Direct loads avoid staging aligned short-K down-projection tiles.
    direct_b = (
        is_down
        and width >= 1024
        and reduction == 256
        and width % config["BLOCK_SIZE_N"] == 0
        and reduction % config["BLOCK_SIZE_K"] == 0
    )
    dense_w8_bf16_kernel[
        (
            triton.cdiv(sorted_ids.numel(), config["BLOCK_SIZE_M"])
            * triton.cdiv(width, config["BLOCK_SIZE_N"]),
        )
    ](
        activation,
        decoded,
        projected,
        topk_weights,
        sorted_ids,
        experts,
        padded,
        width,
        reduction,
        sorted_ids.numel(),
        topk_weights.numel(),
        *activation.stride(),
        decoded.stride(0),
        decoded.stride(2),
        decoded.stride(1),
        projected.stride(-2),
        projected.stride(-1),
        projection_topk,
        multiply_router,
        config["USE_INT32_OFFSETS"],
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["num_stages"],
        DIRECT_B=direct_b,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


def run_dense_wna16(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool,
    *,
    gate_tile: int,
    down_tile: int,
    decode_elements: int,
    projection_width: int,
    workspace_limit: int,
) -> torch.Tensor:
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
        get_default_config,
        invoke_fused_moe_triton_kernel,
    )

    m, k = hidden_states.shape
    e, n2, _ = w1.shape
    n, topk = n2 // 2, topk_ids.shape[1]
    is_fp8 = quant_type_id == QUANT_TYPE_FP8_E4M3
    is_w4 = quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1)
    routes = m * topk
    scratch = torch.empty(
        decode_elements, device=hidden_states.device, dtype=torch.bfloat16
    )
    # Gate/up and down do not overlap in time; only the activation survives both GEMMs.
    cache = hidden_states.new_empty((routes * projection_width,))
    gateup = cache[: routes * n2].view(m, topk, n2)
    intermediate = hidden_states.new_empty((routes, n))
    down = cache[: routes * down_tile].view(m, topk, down_tile)
    if (
        is_fp8
        and m >= MIN_OVERLAP_TOKENS
        and e in (128, 256)
        and k == 4096
        and n == 256
        and 4 <= topk <= 8
        and gate_tile == n
        and down_tile <= k // 2
        and k % down_tile == 0
        and down_tile % 256 == 0
        and hidden_states.is_contiguous()
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
    ):
        workspace = (
            2 * decode_elements
            + 2 * routes * (projection_width + n)
            + 4 * (e + 1) ** 2
            + 16 * (routes + (DENSE_BLOCK_ROWS - 1) * e)
            + 8
        )
        if workspace <= workspace_limit:
            return run_fp8_ordered_overlap(
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_weights,
                output,
                topk_ids,
                apply_router_weight_on_input,
                scratch,
                gateup,
                intermediate,
                down,
            )
    if quant_type_id == QUANT_TYPE_FP4_E2M1:
        w1_scale, w2_scale = w1_scale.view(torch.uint8), w2_scale.view(torch.uint8)
    for index, (weight, scales) in enumerate(((w1, w1_scale), (w2, w2_scale))):
        first = index == 0
        output_size, reduction = weight.shape[1:]
        if is_w4:
            reduction *= 2
        config = get_default_config(
            max(m, _PPU_DENSE_MIN_TOKENS),
            e,
            output_size,
            reduction,
            topk,
            "bf16",
            gemm_stage="gemm1" if first else "gemm2",
        )
        if config["BLOCK_SIZE_M"] != DENSE_BLOCK_ROWS:
            raise NotImplementedError(
                "Dense Marlin workspace requires 64-row expert blocks"
            )
        if first:
            sorted_ids, experts, padded = align_grouped_tokens(
                topk_ids, config["BLOCK_SIZE_M"], e
            )
        tile = gate_tile if first else down_tile
        for col in range(0, output_size, tile):
            stop = min(col + tile, output_size)
            width = stop - col
            qw, ss = weight[:, col:stop], scales[:, col:stop]
            decode_dense_bank(qw, ss, scratch, quant_type_id)
            decoded = (
                scratch[: e * reduction * width]
                .view(e, reduction, width)
                .transpose(1, 2)
            )
            projected = gateup[:, :, col:stop] if first else down[:, :, :width]
            operands = (
                hidden_states if first else intermediate,
                decoded,
                projected,
            )
            routing = (topk_weights, sorted_ids, experts, padded)
            multiply_router = (
                apply_router_weight_on_input
                if first
                else not apply_router_weight_on_input
            )
            projection_topk = topk if first else 1
            if is_w4:
                invoke_fused_moe_triton_kernel(
                    *operands,
                    None,
                    None,
                    *routing,
                    multiply_router,
                    projection_topk,
                    config,
                    tl.bfloat16,
                )
            else:
                project_dense_w8(
                    *operands,
                    *routing,
                    multiply_router,
                    projection_topk,
                    config,
                    is_down=not first,
                )
            if not first:
                moe_sum(projected, output[:, col:stop])
        if first:
            flat = gateup.view(m * topk, n2)
            silu_and_mul_out(flat[:, :n], flat[:, n:], intermediate)
    return output


def run_dense_batches(
    call: MarlinCall, gate_limit: int, down_limit: int, workspace_limit: int
) -> None:
    gate_tile = min(2 * call.shape.intermediate, gate_limit)
    down_tile = min(call.shape.hidden, down_limit)
    decode_elements = call.shape.experts * max(
        gate_tile * call.shape.hidden, down_tile * call.shape.intermediate
    )
    projection_width = max(2 * call.shape.intermediate, down_tile)
    fixed_bytes = (
        2 * decode_elements
        + 4 * (call.shape.experts + 1) ** 2
        + 16 * (DENSE_BLOCK_ROWS - 1) * call.shape.experts
        + 8
    )
    bytes_per_token = call.shape.top_k * (
        2 * (projection_width + call.shape.intermediate) + 12
    )
    batch_tokens = min(
        call.shape.tokens, (workspace_limit - fixed_bytes) // bytes_per_token
    )
    if batch_tokens < 1:
        raise NotImplementedError("PPU dense MoE workspace exceeds its budget")
    if batch_tokens < call.shape.tokens and call.out is not call.hidden_states:
        # A chunk can overwrite activations needed by a later chunk.
        byte_offset = abs(call.out.data_ptr() - call.hidden_states.data_ptr())
        if 0 < byte_offset < call.hidden_states.nbytes:
            raise ValueError(
                "output must not partially overlap hidden_states in chunked dense MoE"
            )
    for start in range(0, call.shape.tokens, batch_tokens):
        end = min(start + batch_tokens, call.shape.tokens)
        run_dense_wna16(
            call.hidden_states[start:end],
            call.w1,
            call.w2,
            call.w1_scale,
            call.w2_scale,
            call.topk_weights[start:end],
            call.out[start:end],
            call.topk_ids[start:end],
            call.quant_type_id,
            call.apply_router_weight_on_input,
            gate_tile=gate_tile,
            down_tile=down_tile,
            decode_elements=decode_elements,
            projection_width=projection_width,
            workspace_limit=workspace_limit,
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
            lo, hi = decode_raw_int4_bf16(q, scale)
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
            lo, hi = decode_raw_mxfp4_bf16(q, scale)
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
        route_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
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

        for k_tile in tl.range(
            0, tl.cdiv(reduction_size, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES
        ):
            a = tle_async.load(
                a_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            b_quantized = tle_async.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )

            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
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
            )
            route_acc = tl.dot(tl.trans(b), tl.trans(a), acc=route_acc)

            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K_WEIGHT, 0))

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


@libentry()
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
    store_offset = offs_m[:, None].to(tl.int64) * reduction_size + offs_k[None, :]
    if PERMUTE_K32:
        tl.static_assert(
            reduction_size % 32 == 0 and not PERMUTE_K16 and not DEINTERLEAVE_K
        )
        store_offset = (
            offs_m[:, None].to(tl.int64) * reduction_size + store_k32(offs_k)[None, :]
        )
    elif PERMUTE_K16:
        tl.static_assert(reduction_size % 16 == 0 and not DEINTERLEAVE_K)
        store_offset = (
            offs_m[:, None].to(tl.int64) * reduction_size + store_k16(offs_k)[None, :]
        )
    elif DEINTERLEAVE_K:
        store_offset = (
            offs_m[:, None].to(tl.int64) * (reduction_size // 2)
            + offs_k[None, :] // 2
            + (offs_k[None, :] % 2).to(tl.int64) * EM * (reduction_size // 2)
        )
    tl.store(
        routed_a_ptr + store_offset,
        activation,
        mask=(offs_m[:, None] < EM) & (offs_k[None, :] < reduction_size),
    )


@libentry()
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
    store_offset = offs_m[:, None].to(tl.int64) * num_columns + offs_n[None, :]
    if PERMUTE_K32:
        tl.static_assert(
            num_columns % 32 == 0 and not PERMUTE_K16 and not DEINTERLEAVE_K
        )
        store_offset = (
            offs_m[:, None].to(tl.int64) * num_columns + store_k32(offs_n)[None, :]
        )
    elif PERMUTE_K16:
        tl.static_assert(num_columns % 16 == 0 and not DEINTERLEAVE_K)
        store_offset = (
            offs_m[:, None].to(tl.int64) * num_columns + store_k16(offs_n)[None, :]
        )
    elif DEINTERLEAVE_K:
        store_offset = (
            offs_m[:, None].to(tl.int64) * (num_columns // 2)
            + offs_n[None, :] // 2
            + (offs_n[None, :] % 2).to(tl.int64) * EM * (num_columns // 2)
        )
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


_PPU_W8_PAIR_ALIGN_MAX_ROUTES = 32
_PPU_SMALL_ALIGN_MAX_EXPERTS = 256
_PPU_SMALL_ALIGN_MAX_TOPK = 6


def invoke_marlin_moe_gemm_direct(
    activations: torch.Tensor,
    output_tile: torch.Tensor,
    stage: QuantizedStage,
    topk_weights: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    *,
    mul_routed_weight: bool,
    a_route_divisor: int,
    group_size: int,
    compute_type: tl.dtype,
    quant_type_id: int,
    use_matvec: bool = False,
    fuse_silu: bool = False,
    split_k: int = 1,
    pipeline_stages: Optional[int] = None,
    use_wide_gate: bool = False,
) -> None:
    if tle_async is None:
        raise RuntimeError("PPU fused Marlin MoE requires Triton TLE")

    B, B_scale = stage.weight, stage.scale
    stride_se, stride_sg, stride_sn = stage.layout.strides
    reduction_size = activations.size(1)
    num_columns = output_tile.size(-1) if fuse_silu else B.size(2)
    routes = topk_ids.numel()
    if fuse_silu:
        block_n = 32 if routes >= 12 else select_direct_block_n(num_columns)
        # Fill more of the 64 PPU compute units for sparse decode batches.
        if routes * triton.cdiv(num_columns, block_n) < 64:
            block_n = 32
    else:
        block_n = 64 if use_matvec else select_direct_block_n(num_columns)
        if (
            use_matvec
            and min(reduction_size, num_columns) <= 512
            and (
                quant_type_id == QUANT_TYPE_UINT8B128
                or (
                    quant_type_id == QUANT_TYPE_FP8_E4M3
                    and compute_type == tl.bfloat16
                    and B_scale.dtype == torch.bfloat16
                )
            )
        ):
            block_n = 32 if reduction_size > num_columns else 128
            four_token_gate = (
                activations.size(0) == 4
                and reduction_size == 4096
                and num_columns == 512
                and group_size == 128
                and 4 <= a_route_divisor <= 8
            )
            if (
                quant_type_id == QUANT_TYPE_FP8_E4M3
                and (routes <= 16 or four_token_gate or use_wide_gate)
                and reduction_size >= 1024
                and num_columns % 128 == 0
            ):
                # Use fewer CTAs for the long-K projection of very few routes.
                block_n = 128
    # Packed FP8 decode favors more long-K tiles and wider short-K tiles.
    # The PPU pipeline pass allocates num_stages - 1 loop buffers.  Three
    # scheduling stages therefore provide the two buffers required to overlap
    # the next AIU copy with the current tile's unpack/dequantize/dot work.
    pipeline_stages = pipeline_stages or (3 if reduction_size > 128 else 1)
    grid = (routes * triton.cdiv(num_columns, block_n),)

    kernel = (
        marlin_moe_gemm_silu_direct_kernel
        if fuse_silu
        else (
            marlin_moe_matvec_direct_kernel
            if use_matvec
            else marlin_moe_gemm_direct_kernel
        )
    )

    target = output_tile
    if split_k > 1:
        assert use_matvec and not fuse_silu
        target = torch.empty(
            (split_k, routes, num_columns),
            device=output_tile.device,
            dtype=torch.float32,
        )
        grid = (*grid, split_k)
        pipeline_stages = min(
            pipeline_stages, triton.cdiv(triton.cdiv(reduction_size, 128), split_k)
        )
    extra = (
        dict(SPLIT_K=split_k, SPLIT_STRIDE=routes * num_columns) if use_matvec else {}
    )
    kernel[grid](
        activations,
        B,
        target,
        B_scale,
        topk_weights,
        topk_ids,
        num_columns,
        reduction_size,
        activations.stride(0),
        activations.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        target.stride(-2),
        target.stride(-1),
        stride_se,
        stride_sg,
        stride_sn,
        A_ROUTE_DIVISOR=a_route_divisor,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        PIPELINE_STAGES=pipeline_stages,
        compute_type=compute_type,
        BLOCK_SIZE_N=block_n,
        num_stages=pipeline_stages,
        QUANT_TYPE=quant_type_id,
        **extra,
    )

    if split_k > 1:
        reduce_split_silu_kernel[(routes, triton.cdiv(num_columns // 2, 128))](
            target,
            output_tile,
            topk_weights,
            num_columns // 2,
            routes,
            split_k,
            mul_routed_weight,
            128,
            num_warps=4,
        )


def invoke_marlin_moe_gemm_reduce_direct(
    activations: torch.Tensor,
    output_tile: torch.Tensor,
    stage: QuantizedStage,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    mul_routed_weight: bool,
    group_size: int,
    compute_type: tl.dtype,
    quant_type_id: int,
) -> None:
    if tle_async is None:
        raise RuntimeError("PPU fused Marlin MoE requires Triton TLE")

    B, B_scale = stage.weight, stage.scale
    stride_se, stride_sg, stride_sn = stage.layout.strides
    reduction_size = activations.size(1)
    num_columns = B.size(2)
    top_k = topk_ids.size(1)
    tokens = activations.size(0) // top_k
    block_n = (64 if num_columns >= 1024 else 128) if tokens == 1 else 32
    pipeline_stages = 3 if reduction_size > 128 else 1
    grid = (tokens * triton.cdiv(num_columns, block_n),)

    marlin_moe_gemm_reduce_direct_kernel[grid](
        activations,
        B,
        output_tile,
        B_scale,
        topk_weights,
        topk_ids,
        num_columns,
        reduction_size,
        activations.stride(0),
        activations.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        output_tile.stride(0),
        output_tile.stride(1),
        stride_se,
        stride_sg,
        stride_sn,
        TOP_K=top_k,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        PIPELINE_STAGES=pipeline_stages,
        compute_type=compute_type,
        BLOCK_SIZE_N=block_n,
        num_stages=pipeline_stages,
        QUANT_TYPE=quant_type_id,
    )


def stage_grouped_activations(
    activations: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    em: int,
    num_valid_tokens: int,
    top_k: int,
    block_m: int,
    deinterleave_k: bool = False,
    permute_k16: bool = False,
    permute_k32: bool = False,
    should_skip_inactive: bool = False,
    use_packed_gather: bool = False,
) -> torch.Tensor:
    routed_a = torch.empty(
        (em, activations.size(1)), dtype=activations.dtype, device=activations.device
    )
    if (
        use_packed_gather
        and (permute_k16 or permute_k32)
        and activations.dtype == torch.bfloat16
        and activations.size(1) >= 1024
        and activations.size(1) % (32 if permute_k32 else 16) == 0
        and activations.stride(1) == 1
        and activations.stride(0) % 2 == 0
        and activations.data_ptr() % 4 == 0
    ):
        k_pairs = activations.size(1) // 2
        stage_bf16_activation_pairs_kernel[
            (triton.cdiv(em, 16), triton.cdiv(k_pairs, 256))
        ](
            activations,
            routed_a,
            sorted_token_ids,
            num_tokens_post_padded,
            em,
            k_pairs,
            num_valid_tokens,
            activations.stride(0) // 2,
            top_k,
            16,
            256,
            SKIP_UNUSED_UPPER_ROWS=block_m == 16 and not permute_k32,
            PERMUTE_K32=permute_k32,
            num_warps=8,
            num_stages=1,
        )
        return routed_a
    grid = (triton.cdiv(em, block_m), triton.cdiv(activations.size(1), 128))
    stage_routed_activations_kernel[grid](
        activations,
        routed_a,
        sorted_token_ids,
        num_tokens_post_padded,
        EM=em,
        reduction_size=activations.size(1),
        num_valid_tokens=num_valid_tokens,
        stride_am=activations.stride(0),
        stride_ak=activations.stride(1),
        top_k=top_k,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=128,
        DEINTERLEAVE_K=deinterleave_k,
        PERMUTE_K16=permute_k16,
        PERMUTE_K32=permute_k32,
        SKIP_INACTIVE=should_skip_inactive,
        num_warps=4,
        num_stages=1,
    )
    return routed_a


def silu_and_stage_grouped(
    intermediate1: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    em: int,
    num_valid_tokens: int,
    block_m: int,
    deinterleave_k: bool = False,
    permute_k16: bool = False,
    permute_k32: bool = False,
    should_skip_inactive: bool = False,
    skip_unused_upper_rows: bool = False,
) -> torch.Tensor:
    n = intermediate1.size(1) // 2
    routed = torch.empty(
        (em, n), dtype=intermediate1.dtype, device=intermediate1.device
    )
    stage_rows = 8 if skip_unused_upper_rows else block_m
    grid = (triton.cdiv(em, stage_rows), triton.cdiv(n, 128))
    silu_and_stage_routed_kernel[grid](
        intermediate1,
        routed,
        sorted_token_ids,
        num_tokens_post_padded,
        EM=em,
        num_columns=n,
        num_valid_tokens=num_valid_tokens,
        BLOCK_SIZE_M=stage_rows,
        BLOCK_SIZE_N=128,
        DEINTERLEAVE_K=deinterleave_k,
        PERMUTE_K16=permute_k16,
        PERMUTE_K32=permute_k32,
        SKIP_INACTIVE=should_skip_inactive,
        SKIP_UNUSED_UPPER_ROWS=skip_unused_upper_rows,
        num_warps=4,
        num_stages=1,
    )
    return routed


def launch_raw_projection(
    routed_a: torch.Tensor,
    output_tile: torch.Tensor,
    stage: QuantizedStage,
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    mul_routed_weight: bool,
    block_m: int,
    quant_type_id: int,
    is_down_projection: bool = False,
    raw_w4: bool = False,
    packed_width: bool = False,
) -> None:
    B, B_scale = stage.weight, stage.scale
    em = sorted_token_ids.size(0)
    num_valid_tokens = output_tile.numel() // output_tile.size(-1)
    n = B.size(2)
    stride_se, stride_sg, stride_sn = stage.layout.strides
    use32 = False
    if raw_w4 or quant_type_id == QUANT_TYPE_FP8_E4M3:
        largest_offset = max(
            routed_a.nbytes - 1,
            sum((length - 1) * stride for length, stride in zip(B.shape, B.stride())),
            stage.layout.max_offset,
            num_valid_tokens * output_tile.stride(-2) + n * output_tile.stride(-1),
        )
        use32 = largest_offset < (1 << 31)
    packed_mx_scales = (
        raw_w4
        and quant_type_id == QUANT_TYPE_FP4_E2M1
        and stride_sg == 1
        and stride_se % 4 == 0
        and stride_sn % 4 == 0
        and B_scale.data_ptr() % 4 == 0
    )
    tile_args = (
        routed_a,
        B,
        output_tile,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        routed_a.size(1),
        num_valid_tokens,
        B.stride(0),
        B.stride(2),
        stride_se,
        stride_sg,
        stride_sn,
        output_tile.stride(-2),
        output_tile.stride(-1),
        mul_routed_weight,
    )
    if block_m == 64:
        quantized_wide_grouped_kernel[(triton.cdiv(em, 64) * triton.cdiv(n, 128),)](
            *tile_args,
            stage.scale_safe,
            quant_type_id,
            packed_mx_scales,
            use32,
            num_warps=8,
            num_stages=1,
        )
        return
    if raw_w4:
        raw_w4_square_kernel[(triton.cdiv(em, block_m) * triton.cdiv(n, 64),)](
            *tile_args,
            quant_type_id == QUANT_TYPE_FP4_E2M1,
            packed_mx_scales,
            ROWS=block_m,
            USE32=use32,
            num_warps=4,
            num_stages=1,
        )
        return
    # The first down-projection tile is shared by both FP8 scale paths.
    prefetch_before_scale = (
        quant_type_id == QUANT_TYPE_FP8_E4M3
        and is_down_projection
        and routed_a.size(1) == 256
        and n == 4096
    )
    # The two adjacent BF16 scale groups can be loaded and folded together.
    scale_pair = (
        prefetch_before_scale
        and B_scale.dtype == torch.bfloat16
        and stride_sg == 1
        and stride_se % 2 == 0
        and stride_sn % 2 == 0
        and B_scale.data_ptr() % 4 == 0
    )
    if block_m == 16:
        marlin_moe_gemm_grouped_square_kernel[
            (triton.cdiv(em, 16) * triton.cdiv(n, 128),)
        ](
            *tile_args,
            stage.scale_safe,
            quant_type_id == QUANT_TYPE_FP8_E4M3,
            PREFETCH_ROUTES=(
                is_down_projection
                and quant_type_id == QUANT_TYPE_FP8_E4M3
                and routed_a.size(1) == 256
                and num_valid_tokens >= (3 if prefetch_before_scale else 9) * B.size(0)
            ),
            USE32=use32,
            PREFETCH_BEFORE_SCALE=prefetch_before_scale,
            PACK_WIDTH=packed_width,
            SCALE_PAIR=scale_pair,
            num_warps=8,
            num_stages=1,
        )
        return
    assert block_m == 8
    marlin_moe_gemm_grouped_narrow_kernel[(triton.cdiv(em, 8) * triton.cdiv(n, 128),)](
        *tile_args,
        stage.scale_safe,
        quant_type_id == QUANT_TYPE_FP8_E4M3,
        USE32=use32,
        PREFETCH_BEFORE_SCALE=prefetch_before_scale,
        SCALE_PAIR=scale_pair,
        num_warps=8,
        num_stages=1,
    )


def invoke_marlin_moe_gemm_grouped(
    routed_a: torch.Tensor,
    output_tile: torch.Tensor,
    stage: QuantizedStage,
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    mul_routed_weight: bool,
    top_k: int,
    block_m: int,
    group_size: int,
    compute_type: tl.dtype,
    quant_type_id: int,
    is_down_projection: bool = False,
    raw_w8: bool = False,
    raw_w4: bool = False,
    packed_width: bool = False,
) -> None:
    if tle_async is None:
        raise RuntimeError("PPU fused Marlin MoE requires Triton TLE")
    B, B_scale = stage.weight, stage.scale
    stride_se, stride_sg, stride_sn = stage.layout.strides
    em = sorted_token_ids.size(0)
    num_valid_tokens = output_tile.numel() // output_tile.size(-1)
    n = B.size(2)
    if raw_w4 or raw_w8:
        launch_raw_projection(
            routed_a,
            output_tile,
            stage,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=mul_routed_weight,
            block_m=block_m,
            quant_type_id=quant_type_id,
            is_down_projection=is_down_projection,
            raw_w4=raw_w4,
            packed_width=packed_width,
        )
        return
    batch_m = (
        output_tile.size(0) if is_down_projection else output_tile.size(0) // top_k
    )
    block_n, group_m, num_warps, pipeline_stages = select_grouped_config(
        batch_m, routed_a.size(1), n, block_m, quant_type_id
    )
    # A 64-K tile bounds decoded-weight state; FP8 can then use a wider down tile.
    if quant_type_id == QUANT_TYPE_FP8_E4M3 and group_size == 128 and block_m <= 32:
        block_n = 128
    if (
        quant_type_id == QUANT_TYPE_FP8_E4M3
        and compute_type == tl.bfloat16
        and 1024 <= batch_m < _PPU_DENSE_MIN_TOKENS
        and group_size == 128
        and block_m == 64
        and is_down_projection
    ):
        block_n, num_warps = 128, 4
    stride_cm, stride_cn = output_tile.stride(-2), output_tile.stride(-1)
    grid = (triton.cdiv(em, block_m) * triton.cdiv(n, block_n),)
    marlin_moe_gemm_grouped_kernel[grid](
        routed_a,
        B,
        output_tile,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_columns=n,
        reduction_size=routed_a.size(1),
        EM=em,
        num_valid_tokens=num_valid_tokens,
        stride_be=B.stride(0),
        stride_bk=B.stride(1),
        stride_bn=B.stride(2),
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        stride_bse=stride_se,
        stride_bsg=stride_sg,
        stride_bsn=stride_sn,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=(
            64
            if (
                quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3)
                and group_size == 128
            )
            or (
                quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1)
                and block_m >= 64
                and batch_m >= 4096
            )
            else 128
        ),
        GROUP_SIZE_M=group_m,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        PIPELINE_STAGES=pipeline_stages,
        compute_type=compute_type,
        num_warps=num_warps,
        num_stages=pipeline_stages,
        QUANT_TYPE=quant_type_id,
    )


def project_direct_gate(
    call: MarlinCall,
    plan: SparsePlan,
    c1: Optional[torch.Tensor],
    c2: torch.Tensor,
    stage1: QuantizedStage,
    split_silu: bool,
    use_matvec: bool,
    gemv_split_k: int,
) -> None:
    stage1_topk_weights = (
        call.topk_weights if call.apply_router_weight_on_input else None
    )
    if split_silu:
        invoke_marlin_moe_gemm_direct(
            activations=call.hidden_states,
            output_tile=c1 if gemv_split_k == 1 else c2,
            stage=stage1,
            topk_weights=stage1_topk_weights,
            topk_ids=call.topk_ids,
            mul_routed_weight=call.apply_router_weight_on_input,
            a_route_divisor=call.shape.top_k,
            group_size=call.group_size,
            compute_type=call.compute_type,
            quant_type_id=call.quant_type_id,
            use_matvec=use_matvec,
            split_k=gemv_split_k,
            pipeline_stages=(
                2
                if plan.small_fp8_gate
                or (
                    plan.decode_fp8_gate
                    and (
                        gemv_split_k == 2 or call.shape.tokens * call.shape.top_k >= 128
                    )
                )
                else None
            ),
            use_wide_gate=plan.decode_fp8_gate,
        )
        if gemv_split_k == 1:
            silu_and_mul_out(
                c1[:, : call.shape.intermediate], c1[:, call.shape.intermediate :], c2
            )
    else:
        invoke_marlin_moe_gemm_direct(
            activations=call.hidden_states,
            output_tile=c2,
            stage=stage1,
            topk_weights=stage1_topk_weights,
            topk_ids=call.topk_ids,
            mul_routed_weight=call.apply_router_weight_on_input,
            a_route_divisor=call.shape.top_k,
            group_size=call.group_size,
            compute_type=call.compute_type,
            quant_type_id=call.quant_type_id,
            fuse_silu=True,
        )


def project_direct_down(
    call: MarlinCall,
    plan: SparsePlan,
    c2: torch.Tensor,
    c3: Optional[torch.Tensor],
    stage2: QuantizedStage,
    tiny_fp8_down: bool,
    reduced: bool,
    use_matvec: bool,
) -> None:
    stage2_topk_weights = (
        None if call.apply_router_weight_on_input else call.topk_weights
    )
    if tiny_fp8_down:
        block_n = 16
        tiny_fp8_down_reduce[(call.shape.tokens, call.shape.hidden // block_n)](
            c2,
            call.w2.view(torch.uint8),
            call.w2_scale,
            call.topk_ids,
            call.topk_weights,
            call.out,
            call.shape.hidden,
            call.shape.intermediate,
            call.shape.top_k,
            *call.w2_scale.stride(),
            block_n,
            not call.apply_router_weight_on_input,
            num_warps=4,
        )
    elif reduced:
        invoke_marlin_moe_gemm_reduce_direct(
            activations=c2,
            output_tile=call.out,
            stage=stage2,
            topk_weights=call.topk_weights,
            topk_ids=call.topk_ids,
            mul_routed_weight=not call.apply_router_weight_on_input,
            group_size=call.group_size,
            compute_type=call.compute_type,
            quant_type_id=call.quant_type_id,
        )
    else:
        invoke_marlin_moe_gemm_direct(
            activations=c2,
            output_tile=c3,
            stage=stage2,
            topk_weights=stage2_topk_weights,
            topk_ids=call.topk_ids,
            mul_routed_weight=not call.apply_router_weight_on_input,
            a_route_divisor=1,
            group_size=call.group_size,
            compute_type=call.compute_type,
            quant_type_id=call.quant_type_id,
            use_matvec=use_matvec,
            pipeline_stages=(
                1 if plan.small_fp8_gate else 2 if plan.decode_fp8_gate else None
            ),
        )


def run_direct(
    call: MarlinCall, plan: SparsePlan, stage1: QuantizedStage, stage2: QuantizedStage
) -> None:
    split_silu = call.shape.intermediate >= 1024 or call.quant_type_id in (
        QUANT_TYPE_UINT8B128,
        QUANT_TYPE_FP8_E4M3,
    )
    use_matvec = plan.packed_fp8 or (
        call.shape.tokens == 1
        and call.quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3)
    )
    gemv_split_k = direct_split_count(call, plan)
    tiny_fp8_down = (
        plan.small_fp8_gate
        and call.shape.tokens == 2
        and call.w2.is_contiguous()
        and call.w2_scale.is_contiguous()
        and call.w2.numel() < (1 << 31)
    )
    reduced = tiny_fp8_down or (call.shape.tokens <= 2 and not split_silu)
    c2 = torch.empty(
        (call.shape.tokens * call.shape.top_k, call.shape.intermediate),
        device=call.out.device,
        dtype=call.out.dtype,
    )
    c1 = (
        torch.empty(
            (call.shape.tokens * call.shape.top_k, 2 * call.shape.intermediate),
            device=call.out.device,
            dtype=call.out.dtype,
        )
        if split_silu and gemv_split_k == 1
        else None
    )
    c3 = (
        None
        if reduced
        else torch.empty(
            (call.shape.tokens, call.shape.top_k, call.shape.hidden),
            device=call.out.device,
            dtype=call.out.dtype,
        )
    )
    project_direct_gate(
        call, plan, c1, c2, stage1, split_silu, use_matvec, gemv_split_k
    )
    project_direct_down(call, plan, c2, c3, stage2, tiny_fp8_down, reduced, use_matvec)
    if not reduced:
        moe_sum(c3, call.out)


def group_routes(
    call: MarlinCall, plan: SparsePlan, stage1: QuantizedStage, stage2: QuantizedStage
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
    packed_width = False
    bm = select_grouped_block_m(call.shape.tokens, call.shape.experts, call.shape.top_k)
    if (plan.raw_w8 or plan.raw_w4) and plan.use_wide_tile:
        bm = 64
    elif plan.raw_w8:
        narrow_density = 5 if call.quant_type_id == QUANT_TYPE_FP8_E4M3 else 6
        if plan.wide_fp8_gate_shape:
            narrow_density = 3
        bm = (
            8
            if call.shape.tokens * call.shape.top_k
            <= narrow_density * call.shape.experts
            else 16
        )
    elif plan.raw_w4:
        bm = 8 if call.shape.tokens * call.shape.top_k <= 5 * call.shape.experts else 16
    if (
        (plan.raw_w4 or (plan.raw_w8 and call.quant_type_id == QUANT_TYPE_UINT8B128))
        and call.shape.tokens <= 2
        and call.shape.top_k <= _PPU_SMALL_ALIGN_MAX_TOPK
        and call.shape.tokens * call.shape.top_k
        <= call.shape.experts
        <= _PPU_SMALL_ALIGN_MAX_EXPERTS
    ):
        if call.shape.tokens == 1:
            # One token typically routes to distinct experts.
            sorted_ids, experts, padded = moe_align_block_size_singleton(
                call.topk_ids, bm
            )
        else:
            # Group routes so both tokens can share an expert tile.
            sorted_ids, experts, padded = moe_align_block_size_small_grouped(
                call.topk_ids, call.shape.experts, bm
            )
    elif (
        plan.raw_w8
        and call.quant_type_id == QUANT_TYPE_UINT8B128
        and call.shape.tokens * call.shape.top_k <= _PPU_W8_PAIR_ALIGN_MAX_ROUTES
    ):
        routes = call.shape.tokens * call.shape.top_k
        max_padded_routes = min(routes * bm, routes + call.shape.experts * (bm - 1))
        sorted_ids = torch.empty(
            (max_padded_routes,), device=call.topk_ids.device, dtype=torch.int32
        )
        experts = torch.empty((routes,), device=call.topk_ids.device, dtype=torch.int32)
        padded = torch.empty((1,), device=call.topk_ids.device, dtype=torch.int32)
        w8_small_routes_kernel[(1,)](
            call.topk_ids,
            sorted_ids,
            experts,
            padded,
            routes,
            bm,
            triton.next_power_of_2(max(routes, bm)),
            num_warps=4,
        )
    else:
        is_one_cta = (
            (plan.raw_w8 or plan.raw_w4)
            and call.shape.tokens * call.shape.top_k <= 4096
            and call.shape.experts <= 1024
        )
        if plan.is_joint_safety:
            packed_width = is_one_cta and bm == 16
            sorted_ids, experts, padded = align_grouped_tokens(
                call.topk_ids,
                bm,
                call.shape.experts,
                fp8_scales=(call.w1_scale, call.w2_scale),
                fp8_safety=(stage1.scale_safe, stage2.scale_safe),
                packed_width=packed_width,
                is_one_cta=is_one_cta,
            )
        else:
            sorted_ids, experts, padded = align_grouped_tokens(
                call.topk_ids,
                bm,
                call.shape.experts,
                is_one_cta=is_one_cta,
            )
    return sorted_ids, experts, padded, bm, packed_width


def run_grouped(
    call: MarlinCall, plan: SparsePlan, stage1: QuantizedStage, stage2: QuantizedStage
) -> None:
    stage1_topk_weights = (
        call.topk_weights if call.apply_router_weight_on_input else None
    )
    stage2_topk_weights = (
        None if call.apply_router_weight_on_input else call.topk_weights
    )
    sorted_ids, experts, padded, bm, packed_width = group_routes(
        call, plan, stage1, stage2
    )
    c3 = torch.empty(
        (call.shape.tokens, call.shape.top_k, call.shape.hidden),
        device=call.out.device,
        dtype=call.out.dtype,
    )
    c1 = torch.empty(
        (call.shape.tokens * call.shape.top_k, 2 * call.shape.intermediate),
        device=call.out.device,
        dtype=call.out.dtype,
    )
    staging = dict(
        em=sorted_ids.numel(),
        num_valid_tokens=call.shape.tokens * call.shape.top_k,
        block_m=bm,
        deinterleave_k=call.quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1)
        and call.compute_type == tl.bfloat16
        and bm <= 32
        and not plan.raw_w4,
        permute_k16=plan.raw_w8,
        permute_k32=plan.raw_w4,
        should_skip_inactive=plan.raw_w8
        or plan.raw_w4
        or call.quant_type_id == QUANT_TYPE_FP8_E4M3,
    )
    invoke_marlin_moe_gemm_grouped(
        routed_a=stage_grouped_activations(
            call.hidden_states,
            sorted_ids,
            padded,
            top_k=call.shape.top_k,
            use_packed_gather=plan.raw_w4
            or (plan.raw_w8 and (plan.packed_fp8 or plan.use_wide_tile)),
            **staging,
        ),
        output_tile=c1,
        stage=stage1,
        topk_weights=stage1_topk_weights,
        sorted_token_ids=sorted_ids,
        expert_ids=experts,
        num_tokens_post_padded=padded,
        mul_routed_weight=call.apply_router_weight_on_input,
        top_k=call.shape.top_k,
        block_m=bm,
        group_size=call.group_size,
        compute_type=call.compute_type,
        quant_type_id=call.quant_type_id,
        raw_w8=plan.raw_w8,
        raw_w4=plan.raw_w4,
        packed_width=packed_width,
    )
    routed_c2 = silu_and_stage_grouped(
        c1,
        sorted_ids,
        padded,
        **staging,
        skip_unused_upper_rows=(
            plan.raw_w8 and call.quant_type_id == QUANT_TYPE_FP8_E4M3 and bm == 16
        ),
    )
    invoke_marlin_moe_gemm_grouped(
        routed_a=routed_c2,
        output_tile=c3,
        stage=stage2,
        topk_weights=stage2_topk_weights,
        sorted_token_ids=sorted_ids,
        expert_ids=experts,
        num_tokens_post_padded=padded,
        mul_routed_weight=not call.apply_router_weight_on_input,
        top_k=1,
        block_m=bm,
        group_size=call.group_size,
        compute_type=call.compute_type,
        quant_type_id=call.quant_type_id,
        is_down_projection=True,
        raw_w8=plan.raw_w8,
        raw_w4=plan.raw_w4,
        packed_width=packed_width,
    )
    moe_sum(c3, call.out)


_PPU_DENSE_GATE_TILE = 256
_PPU_DENSE_DOWN_TILE = 1024
_PPU_DENSE_MAX_WORKSPACE_BYTES = 2 << 30
_PPU_QUANT_TYPES = (
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
)
_VLLM_QUANT_TYPE_FP8_E4M3 = 2814749767172868


def dispatch_marlin_call(call: MarlinCall) -> None:
    with torch.cuda.device(call.hidden_states.device):
        if eligible_for_dense(call):
            run_dense_batches(
                call,
                _PPU_DENSE_GATE_TILE,
                _PPU_DENSE_DOWN_TILE,
                _PPU_DENSE_MAX_WORKSPACE_BYTES,
            )
        else:
            plan = select_sparse_plan(call, _PPU_DENSE_MAX_WORKSPACE_BYTES)
            stage1, stage2 = prepare_quantized_weights(
                call.w1,
                call.w2,
                call.w1_scale,
                call.w2_scale,
                needs_expert_flags=plan.is_joint_safety,
                is_mxfp4=call.quant_type_id == QUANT_TYPE_FP4_E2M1,
            )
            if plan.direct:
                run_direct(call, plan, stage1, stage2)
            else:
                run_grouped(call, plan, stage1, stage2)


def run_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    apply_router_weight_on_input: bool = False,
    inplace: bool = False,
    output: Optional[torch.Tensor] = None,
    group_size: int = 128,
    quant_type_id: int = QUANT_TYPE_UINT4B8,
) -> torch.Tensor:
    """Forward SwiGLU MoE with output-major quantized weights.

    All weights and scales are read directly on every invocation. MXFP4
    E8M0 scales are decoded on every invocation without a persistent copy.
    Expert IDs must be in [0, E); expert parallel maps are not supported.
    """
    shape = validate_inputs(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        inplace,
        group_size,
        quant_type_id,
    )
    if tle_async is None:
        raise NotImplementedError(
            "PPU fused Marlin MoE requires a TLE-enabled FlagTree build"
        )
    out = hidden_states if inplace else output
    if out is None:
        out = torch.empty_like(hidden_states)
    if shape.tokens == 0:
        return out
    compute_type = tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16
    call = MarlinCall(
        shape,
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        out,
        group_size,
        quant_type_id,
        apply_router_weight_on_input,
        compute_type,
    )
    dispatch_marlin_call(call)
    return out


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Optional[Union[str, Enum]] = None,
    activation_func: Optional[Callable] = None,
    moe_sum: Optional[Callable] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    clamp_limit: Optional[float] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """PPU override for INT4, MXFP4, INT8 and FP8 fused Marlin MoE."""
    if quant_type_id == _VLLM_QUANT_TYPE_FP8_E4M3:
        quant_type_id = QUANT_TYPE_FP8_E4M3
    if quant_type_id not in _PPU_QUANT_TYPES:
        return generic_fused_marlin_moe(**locals())
    activation_str = getattr(
        activation, "value", getattr(activation, "name", activation)
    )
    if activation_str is not None and str(activation_str).lower() != "silu":
        raise NotImplementedError("PPU fused Marlin MoE supports only SiLU")
    if any(
        value is not None for value in (g_idx1, g_idx2, sort_indices1, sort_indices2)
    ):
        raise NotImplementedError("PPU fused Marlin MoE does not support act_order")
    if input_dtype is not None:
        raise NotImplementedError("PPU fused Marlin MoE does not support FP8 input")
    unsupported = (
        bias1,
        bias2,
        activation_func,
        moe_sum,
        expert_map,
        input_global_scale1,
        input_global_scale2,
        global_scale1,
        global_scale2,
        w1_zeros,
        w2_zeros,
        workspace,
        intermediate_cache13,
        intermediate_cache2,
        clamp_limit,
    )
    if any(value is not None for value in unsupported) or not is_k_full:
        raise NotImplementedError(
            "Unsupported PPU fused Marlin MoE option "
            "(bias/map/scaling/workspace/activation)"
        )
    if w1.ndim != 3 or global_num_experts not in (-1, w1.shape[0]):
        raise NotImplementedError("PPU fused Marlin MoE requires local weights")
    return run_marlin_moe(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=apply_router_weight_on_input,
        inplace=inplace,
        output=output,
        group_size=group_size,
        quant_type_id=quant_type_id,
    )


__all__ = ["fused_marlin_moe"]
