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

import triton
import triton.language as tl


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
def decode_fp8_scaled_bf16(q, scale):
    # Each byte produces its NaN exponent and sign before expanding into BF16.
    # No carries or borrows cross byte boundaries because magnitudes are <=127.
    # Folding a BF16 scale by an exact power of two adds no rounding when finite.
    folded = tl.inline_asm_elementwise(
        "{ .reg .b32 bias; ppu.mov.b32 bias,0x7b807b80;\n"
        "ppu.mul.rn.bf16x2 $0,$1,bias; }",
        "=r,r",
        [scale],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )
    return tl.inline_asm_elementwise(
        "{ .reg .b32 mag, flags, high, shifted, b<2>, special<2>, v<2>;\n"
        "ppu.and.b32 mag,$2,0x7f7f7f7f; ppu.add.u32 flags,mag,0x01010101; ppu.and.b32 flags,flags,0x80808080;\n"
        "ppu.shr.b32 shifted,flags,7; ppu.sub.u32 high,flags,shifted;\n"
        "ppu.lop3.b32 high,high,0x80808080,$2,0xf8;\n"
        "ppu.prmt.b32 special0,flags,high,0x5140; ppu.prmt.b32 special1,flags,high,0x7362;\n"
        "ppu.prmt.b32 b0,mag,0,0x4140; ppu.prmt.b32 b1,mag,0,0x4342;\n"
        "ppu.shl.b32 b0,b0,4; ppu.shl.b32 b1,b1,4;\n"
        "ppu.or.b32 v0,b0,special0; ppu.or.b32 v1,b1,special1;\n"
        "ppu.mul.rn.bf16x2 v0,v0,$3; ppu.mul.rn.bf16x2 v1,v1,$4;\n"
        "ppu.mov.b32 $0,v0; ppu.mov.b32 $1,v1; }",
        "=r,=r,r,r,r",
        [q, folded],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )


@triton.jit
def decode_fp8_scaled_bf16_normalized(q, scale):
    # All finite E4M3FN values are exact in both FP16 and BF16. Normalize in
    # packed FP16 before applying the original BF16 scale, so sparse routes
    # need neither a scale scan nor an overflow guard for every weight.
    value = tl.inline_asm_elementwise(
        "{ .reg .b32 a<2>, b<2>, m<2>, v<2>, bias;\n"
        "ppu.prmt.b32 a0,0,$2,0x5140; ppu.prmt.b32 a1,0,$2,0x7362;\n"
        "ppu.and.b32 b0,a0,0x7f007f00; ppu.and.b32 b1,a1,0x7f007f00;\n"
        "ppu.shr.b32 b0,b0,1; ppu.shr.b32 b1,b1,1;\n"
        "ppu.add.u32 m0,b0,0x00800080; ppu.add.u32 m1,b1,0x00800080;\n"
        "ppu.and.b32 m0,m0,0x40004000; ppu.and.b32 m1,m1,0x40004000;\n"
        "ppu.shr.b32 m0,m0,14; ppu.shr.b32 m1,m1,14;\n"
        "ppu.mul.lo.u32 m0,m0,0x7c00; ppu.mul.lo.u32 m1,m1,0x7c00;\n"
        "ppu.or.b32 b0,b0,m0; ppu.or.b32 b1,b1,m1;\n"
        "ppu.lop3.b32 v0,b0,0x80008000,a0,0xf8; ppu.lop3.b32 v1,b1,0x80008000,a1,0xf8;\n"
        "ppu.mov.b32 bias,0x5c005c00;\n"
        "ppu.mul.rn.f16x2 v0,v0,bias; ppu.mul.rn.f16x2 v1,v1,bias;\n"
        "ppu.mov.b32 $0,v0; ppu.mov.b32 $1,v1; }",
        "=r,=r,r",
        [q],
        dtype=tl.float16,
        is_pure=True,
        pack=4,
    )
    value = value.to(tl.bfloat16)
    return tl.inline_asm_elementwise(
        "ppu.mul.rn.bf16x2 $0,$1,$2;",
        "=r,r,r",
        [value, scale],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def decode_e4m3(q, scale, compute_type: tl.constexpr, FAST: tl.constexpr = False):
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
    if FAST is None:
        # Sparse routes avoid scanning scales for experts they never read.
        folded = scale * exponent_bias
        overflow = (tl.abs(folded) == float("inf")) & (tl.abs(scale) != float("inf"))
        folded = tl.where(overflow, scale, folded)
        value = tl.where(overflow, tiny * exponent_bias, tiny)
        return (value * folded).to(compute_type)
    else:
        value = tiny
        if not FAST:
            value = value * exponent_bias
        return (value * scale).to(compute_type)
