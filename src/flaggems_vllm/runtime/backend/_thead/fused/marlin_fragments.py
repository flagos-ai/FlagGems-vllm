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

"""Packed BF16 arithmetic for T-Head Marlin, lowered by FlagTree and the PPU SDK."""

import triton
import triton.language as tl
from triton.language import core as tl_core


@tl_core.extern
def packed_u32(symbol, operands, _semantic=None):
    # These scalar SDK functions lower through FlagTree to packed PPU operations.
    symbol = tl_core._unwrap_if_constexpr(symbol)
    words = [
        _semantic.cast(_semantic.to_tensor(operand), tl_core.uint32)
        for operand in operands
    ]
    signature = (tl_core.uint32,) * len(words)
    return tl_core.extern_elementwise(
        "",
        "",
        words,
        {signature: (symbol, tl_core.uint32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def byte_permute(a, b, control):
    return packed_u32("__ppukernel_prmt_u32", (a, b, control))


@triton.jit
def multiply_bf16_pairs(a, b):
    return packed_u32("__ppumath_hmul2_2bf16", (a, b))


@triton.jit
def subtract_bf16_pairs(a, b):
    return packed_u32("__ppumath_hsub2_2bf16", (a, b))


@triton.jit
def logic3(a, b, c, truth: tl.constexpr):
    return packed_u32("llvm.ppu.lop3", (a, b, c, truth))


@triton.jit
def scale_int8_pair(encoded, scale):
    bias = logic3(encoded, 0xFF80FF80, 0x00800080, 0x6A)
    return multiply_bf16_pairs(subtract_bf16_pairs(encoded & 0xFF7FFF7F, bias), scale)


@triton.jit
def decode_int8_fragments(q0, q8, s0, s8):
    low0 = byte_permute(q0, 0x43, 0x4140)
    low8 = byte_permute(q8, 0x43, 0x4140)
    high0 = byte_permute(q0, 0x43, 0x4342)
    high8 = byte_permute(q8, 0x43, 0x4342)
    a0 = scale_int8_pair(low0, s0)
    a2 = scale_int8_pair(low8, s8)
    a1 = scale_int8_pair(high0, s0)
    a3 = scale_int8_pair(high8, s8)
    return a0, a1, a2, a3


@triton.jit
def fp8_fragment_bits(q):
    magnitude = q & 0x7F7F7F7F
    nan_mask = byte_permute(magnitude + 0x01010101, 0, 0xBA98)
    high = logic3(q, nan_mask, 0x80808080, 0xE4)
    high = logic3(high, q >> 4, 0x07070707, 0xF8)
    low = (q << 4) & 0xF0F0F0F0
    return byte_permute(low, high, 0x5140), byte_permute(low, high, 0x7362)


@triton.jit
def decode_fp8_word(q, scale):
    low, high = fp8_fragment_bits(q)
    low = multiply_bf16_pairs(low, 0x7B807B80)
    high = multiply_bf16_pairs(high, 0x7B807B80)
    return multiply_bf16_pairs(low, scale), multiply_bf16_pairs(high, scale)


@triton.jit
def decode_fp8_pair(q0, q1, s0, s1, FAST: tl.constexpr):
    if not FAST:
        a0, a1 = decode_fp8_word(q0, s0)
        b0, b1 = decode_fp8_word(q1, s1)
        return a0, a1, b0, b1
    a0, a1 = fp8_fragment_bits(q0)
    b0, b1 = fp8_fragment_bits(q1)
    return (
        multiply_bf16_pairs(a0, s0),
        multiply_bf16_pairs(a1, s0),
        multiply_bf16_pairs(b0, s1),
        multiply_bf16_pairs(b1, s1),
    )


@triton.jit
def fold_fp8_scale_fragment(s):
    return multiply_bf16_pairs(s, 0x7B807B80)


@triton.jit
def decode_i4_register(q, s0, s1):
    codes = byte_permute(q, q >> 4, 0x5140) & 0x0F0F0F0F
    low = subtract_bf16_pairs(byte_permute(codes, 0x43, 0x4140), 0x43084308)
    high = subtract_bf16_pairs(byte_permute(codes, 0x43, 0x4342), 0x43084308)
    return multiply_bf16_pairs(low, s0), multiply_bf16_pairs(high, s1)


@triton.jit
def decode_mx_register(q, s0, s1):
    codes = q & 0x7777
    low_bytes = byte_permute(0xC0800000, 0xC0804000, codes)
    high_bytes = byte_permute(0x3F3F3F00, 0x40404040, codes)
    signs = byte_permute(0x80808080, 0x80808080, q)
    high_bytes |= (signs << 1) & 0x80808080
    return (
        multiply_bf16_pairs(byte_permute(low_bytes, high_bytes, 0x5140), s0),
        multiply_bf16_pairs(byte_permute(low_bytes, high_bytes, 0x7362), s1),
    )
