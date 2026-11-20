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

"""T-Head PPU specialization for fused Marlin MoE.

W4 and W8 read public byte-packed weight/scale storage on every call.
Selected BF16 paths use packed casts and bounded per-call dense decoding.
"""

from typing import Any, Callable, NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl
from flag_gems import zeros

try:
    from triton.experimental.tle import language as tle_async
except ImportError:
    tle_async = None

from flaggems_vllm.ops.fused_marlin_moe import (
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
)
from flaggems_vllm.ops.fused_marlin_moe import (
    fused_marlin_moe as _generic_fused_marlin_moe,
)
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_no_tle,
    moe_align_block_size_singleton,
    moe_align_block_size_small_grouped,
)
from flaggems_vllm.ops.moe_sum import moe_sum
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.utils import libentry

_PPU_DIRECT_ROUTE_LIMIT = 32
_PPU_W8_PAIR_ALIGN_MAX_ROUTES = 32
_PPU_SMALL_ALIGN_MAX_EXPERTS = 256
_PPU_SMALL_ALIGN_MAX_TOPK = 6
_PPU_DENSE_MIN_TOKENS = 4096
_PPU_DENSE_GATE_TILE = 256
_PPU_DENSE_DOWN_TILE = 1024
_PPU_WIDE_SHARED_BYTES = 64 << 10
_PPU_DENSE_MAX_DECODE_BYTES = 1 << 30
# Inputs and the caller-visible output are outside the temporary workspace budget.
_PPU_DENSE_MAX_WORKSPACE_BYTES = 2 << 30
_PPU_QUANT_TYPES = (
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
)
_TL_QUANT_TYPE_UINT4B8 = tl.constexpr(QUANT_TYPE_UINT4B8)
_TL_QUANT_TYPE_UINT8B128 = tl.constexpr(QUANT_TYPE_UINT8B128)
_TL_QUANT_TYPE_FP4_E2M1 = tl.constexpr(QUANT_TYPE_FP4_E2M1)
_TL_QUANT_TYPE_FP8_E4M3 = tl.constexpr(QUANT_TYPE_FP8_E4M3)
# vLLM ScalarType.float8_e4m3fn.id. The public FlagGems tag is the local id 2.
_VLLM_QUANT_TYPE_FP8_E4M3 = 2814749767172868


class _QuantizedStage(NamedTuple):
    weight: torch.Tensor
    scale: torch.Tensor
    scale_safe: Optional[torch.Tensor]


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
def _decode_e4m3(q, scale, compute_type: tl.constexpr):
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
def _decode_e2m1(q, scale, compute_type: tl.constexpr):
    bits = ((q & 8) << 12) | ((q & 7) << 9)
    tiny = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return (tiny * 16384.0 * scale).to(compute_type)


@triton.jit
def _decode_raw_int4_bf16(b, scale):
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
def _dequant_int4(
    b,
    s_ptr,
    expert,
    k_base,
    ns,
    se,
    sg,
    sn,
    N: tl.constexpr,
    compute_type: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    scale = tl.load(
        s_ptr + expert * se + (k_base // GROUP_SIZE) * sg + ns * sn, ns < N, other=0
    )[None, :]
    if compute_type == tl.bfloat16 and s_ptr.dtype.element_ty == tl.bfloat16:
        lo, hi = _decode_raw_int4_bf16(b, scale)
        q = tl.reshape(
            tl.permute(tl.join(lo, hi), (0, 2, 1)), (b.shape[0] * 2, b.shape[1])
        )
        return q
    q = tl.join(b & 15, (b >> 4) & 15)
    q = tl.reshape(tl.permute(q, (0, 2, 1)), (b.shape[0] * 2, b.shape[1]))
    return ((q.to(compute_type) - 8.0) * scale).to(compute_type)


@triton.jit
def _dequant_int8(
    b,
    s_ptr,
    expert,
    k_base,
    ns,
    se,
    sg,
    sn,
    N: tl.constexpr,
    compute_type: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    tl.static_assert(b.shape[0] == 128 or b.shape[0] == 64)
    group = k_base // GROUP_SIZE
    scale = tl.load(
        s_ptr + expert * se + group * sg + ns * sn,
        mask=ns < N,
        other=0.0,
    )[None, :]
    if compute_type == tl.bfloat16 and (b.shape[1] <= 128 or b.shape[0] == 64):
        return decode_int8_scaled_bf16(b.to(tl.uint8), scale)
    else:
        quant = b.to(tl.int32) - 128
        return (quant.to(compute_type) * scale).to(compute_type)


@triton.jit
def _decode_raw_mxfp4_bf16(b, scale):
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
def _dequant_mxfp4(
    b,
    s_ptr,
    expert,
    k_base,
    ns,
    se,
    sg,
    sn,
    N: tl.constexpr,
    K: tl.constexpr,
    compute_type: tl.constexpr,
):
    groups = k_base // 32 + tl.arange(0, b.shape[0] * 2 // 32)
    byte = tl.load(
        s_ptr + expert * se + groups[:, None] * sg + ns[None, :] * sn,
        (ns[None, :] < N) & (groups[:, None] * 32 < K),
        other=127,
    ).to(tl.uint32)
    if compute_type == tl.bfloat16:
        sbits = e8m0_to_bf16_bits(byte)
        scale = sbits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        scale = tl.reshape(
            tl.broadcast_to(scale[:, None, :], (b.shape[0] * 2 // 32, 16, b.shape[1])),
            (b.shape[0], b.shape[1]),
        )
        lo, hi = _decode_raw_mxfp4_bf16(b, scale)
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
    return _decode_e2m1(q, scale, compute_type)


@triton.jit
def _dequant_fp8(
    b,
    s_ptr,
    expert,
    k_base,
    ns,
    se,
    sg,
    sn,
    N: tl.constexpr,
    K: tl.constexpr,
    compute_type: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    tl.static_assert(b.shape[0] == 128 or b.shape[0] == 64)
    if GROUP_SIZE == -1 or GROUP_SIZE >= 128:
        group = 0 if GROUP_SIZE == -1 else k_base // GROUP_SIZE
        scale_ptrs = s_ptr + expert * se + group * sg + ns * sn
        if N % ns.shape[0] == 0:
            scale = tl.load(scale_ptrs)[None, :]
        else:
            scale = tl.load(scale_ptrs, mask=ns < N, other=1.0)[None, :]
    else:
        groups = k_base // GROUP_SIZE + tl.arange(0, 128 // GROUP_SIZE)
        scale_ptrs = s_ptr + expert * se + groups[:, None] * sg + ns[None, :] * sn
        if N % ns.shape[0] == 0 and K % 128 == 0:
            scale = tl.load(scale_ptrs)
        else:
            scale = tl.load(
                scale_ptrs,
                mask=(ns[None, :] < N) & (groups[:, None] * GROUP_SIZE < K),
                other=1.0,
            )
        scale = tl.broadcast_to(
            scale[:, None, :], (128 // GROUP_SIZE, GROUP_SIZE, ns.shape[0])
        )
        scale = tl.reshape(scale, (128, ns.shape[0]))
    if compute_type == tl.bfloat16 and s_ptr.dtype.element_ty == tl.bfloat16:
        decoded = decode_fp8_scaled_bf16_normalized(b.to(tl.uint8), scale)
    else:
        decoded = _decode_e4m3(b.to(tl.uint32), scale, compute_type)
    if K % 128 == 0:
        return decoded
    # Channel scales can be nonfinite; padded bytes must never introduce NaN.
    return tl.where((k_base + tl.arange(0, b.shape[0]))[:, None] < K, decoded, 0.0).to(
        compute_type
    )


@triton.jit
def _dequant_weight(
    b,
    s_ptr,
    expert,
    k_base,
    ns,
    se,
    sg,
    sn,
    N: tl.constexpr,
    K: tl.constexpr,
    compute_type: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
):
    if QUANT_TYPE == _TL_QUANT_TYPE_UINT4B8:
        return _dequant_int4(
            b, s_ptr, expert, k_base, ns, se, sg, sn, N, compute_type, GROUP_SIZE
        )
    elif QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128:
        return _dequant_int8(
            b, s_ptr, expert, k_base, ns, se, sg, sn, N, compute_type, GROUP_SIZE
        )
    elif QUANT_TYPE == _TL_QUANT_TYPE_FP4_E2M1:
        return _dequant_mxfp4(
            b, s_ptr, expert, k_base, ns, se, sg, sn, N, K, compute_type
        )
    else:
        return _dequant_fp8(
            b,
            s_ptr,
            expert,
            k_base,
            ns,
            se,
            sg,
            sn,
            N,
            K,
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
    S,
    Safe,
    expert,
    N: tl.constexpr,
    G: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK).to(tl.int64)
    n, g = i // G, i % G
    scale = tl.load(S + expert * SE + n * SN + g * SG, i < N * G, other=1)
    bad = is_fp8_scale_fold_overflow(scale) & (i < N * G)
    tl.store(Safe + expert, tl.sum(bad.to(tl.int32)) == 0)


@triton.jit
def _decode_int8_fragment_bf16(q, s0, s1):
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


@triton.jit
def paired_fp8_folded(q0, q1, s0, s1):
    return tl.inline_asm_elementwise(
        (
            "{ .reg .b32 mag_0,inc_0,mask_0,hi_0,lo_0,tmp_0,v0_0,v1_0;"
            " .reg .b32 mag_1,inc_1,mask_1,hi_1,lo_1,tmp_1,v0_1,v1_1;"
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
            " ppu.mul.rn.bf16x2 v0_0,v0_0,$6;"
            " ppu.mul.rn.bf16x2 v0_1,v0_1,$7;"
            " ppu.mul.rn.bf16x2 v1_0,v1_0,$6;"
            " ppu.mul.rn.bf16x2 v1_1,v1_1,$7;"
            " ppu.mov.b32 $0,v0_0;"
            " ppu.mov.b32 $2,v0_1;"
            " ppu.mov.b32 $1,v1_0;"
            " ppu.mov.b32 $3,v1_1;"
            " }"
        ),
        "=r,=r,=r,=r,r,r,r,r",
        [q0, q1, s0, s1],
        dtype=(tl.uint32,) * 4,
        is_pure=True,
        pack=1,
    )


@triton.jit
def paired_fp8_robust(q0, q1, s0, s1):
    return tl.inline_asm_elementwise(
        (
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
            " ppu.mov.b32 bias_0,0x7b807b80;"
            " ppu.mov.b32 bias_1,0x7b807b80;"
            " ppu.mul.rn.bf16x2 v0_0,v0_0,bias_0;"
            " ppu.mul.rn.bf16x2 v0_1,v0_1,bias_1;"
            " ppu.mul.rn.bf16x2 v1_0,v1_0,bias_0;"
            " ppu.mul.rn.bf16x2 v1_1,v1_1,bias_1;"
            " ppu.mul.rn.bf16x2 v0_0,v0_0,$6;"
            " ppu.mul.rn.bf16x2 v0_1,v0_1,$7;"
            " ppu.mul.rn.bf16x2 v1_0,v1_0,$6;"
            " ppu.mul.rn.bf16x2 v1_1,v1_1,$7;"
            " ppu.mov.b32 $0,v0_0;"
            " ppu.mov.b32 $2,v0_1;"
            " ppu.mov.b32 $1,v1_0;"
            " ppu.mov.b32 $3,v1_1;"
            " }"
        ),
        "=r,=r,=r,=r,r,r,r,r",
        [q0, q1, s0, s1],
        dtype=(tl.uint32,) * 4,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fold_fp8_scale_fragment(s):
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
        first = _fold_fp8_scale_fragment(first)
        second = _fold_fp8_scale_fragment(second)
    return first, second


@triton.jit
def w8_scale_fragment(scale, kg, PACKED: tl.constexpr, FOLD: tl.constexpr):
    if PACKED:
        bits = (scale >> (16 * kg)) & 0xFFFF
    else:
        bits = scale.to(tl.uint16, bitcast=True).to(tl.uint32)
    fragment = bits | (bits << 16)
    if FOLD and not PACKED:
        fragment = _fold_fp8_scale_fragment(fragment)
    return fragment


@triton.jit
def _mma_bf16_m16n8(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
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
def _copy_w8_tile_async(dst, src, M: tl.constexpr, KS, KT: tl.constexpr, start, dep):
    # AIU copies are collective within the first warp.
    return tl.inline_asm_elementwise(
        (
            "{ .reg .u32 tid; .reg .pred p; ppu.mov.u32 tid,%tid.x; ppu.setp.lt.u32 p,tid,32;"
            " @p ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.zfill.2d.b8 "
            "[$1],[$2],{1,$3,$4,0,0,$5},{1,$3,$6,1}; ppu.mov.b32 $0,$7; }"
        ),
        "=r,r,l,r,r,r,r,r,~{memory}",
        [dst, src, M, KS, start, KT, dep],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _prefetch_w8_tile(
    qs, aa, Q, A, QN, K: tl.constexpr, PM, kg, dep, IS_SPLIT_Q: tl.constexpr = False
):
    ab = A.to(tl.pointer_type(tl.uint8)) + PM * 8 * K * 2
    dep = _copy_w8_tile_async(aa, ab, 8, K * 2, 128, kg * 256, dep)
    dep = _copy_w8_tile_async(aa + 1024, ab, 8, K * 2, 128, kg * 256 + 128, dep)
    if IS_SPLIT_Q:
        # Split the two warp groups into independent 8 KiB swizzled blocks.
        dep = _copy_w8_tile_async(qs, Q, 64, QN, 128, kg * 128, dep)
        dep = _copy_w8_tile_async(qs + 8192, Q + 64 * QN, 64, QN, 128, kg * 128, dep)
    else:
        dep = _copy_w8_tile_async(qs, Q, 128, QN, 128, kg * 128, dep)
    return _commit_w8_tile(dep)


@triton.jit
def _commit_w8_tile(dep):
    return tl.inline_asm_elementwise(
        ("ppu.cp.async.commit_group; ppu.mov.b32 $0,$1;"),
        "=r,r,~{memory}",
        [dep],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_w8_tile(dep, KEEP: tl.constexpr):
    if KEEP:
        return tl.inline_asm_elementwise(
            ("ppu.cp.async.wait_group 1; bar.sync 0; ppu.mov.b32 $0,$1;"),
            "=r,r,~{memory}",
            [dep],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        return tl.inline_asm_elementwise(
            ("ppu.cp.async.wait_group 0; bar.sync 0; ppu.mov.b32 $0,$1;"),
            "=r,r,~{memory}",
            [dep],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _mma_bf16_square(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7):
    return tl.inline_asm_elementwise(
        (
            "ppu.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 "
            "{$0,$1,$2,$3,$4,$5,$6,$7},{$8,$9,$10,$11},{$12,$13,$14,$15},{$16,$17,$18,$19,$20"
            ",$21,$22,$23};"
        ),
        "=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,r,r,r,r,f,f,f,f,f,f,f,f",
        [a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7],
        dtype=(tl.float32,) * 8,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _prefetch_square(qs, aa, Q, A, QN, K: tl.constexpr, pm, kg, dep):
    ab = A.to(tl.pointer_type(tl.uint8)) + pm * 16 * K * 2
    dep = _copy_w8_tile_async(aa, ab, 16, K * 2, 128, kg * 256, dep)
    dep = _copy_w8_tile_async(aa + 2048, ab, 16, K * 2, 128, kg * 256 + 128, dep)
    dep = _copy_w8_tile_async(qs, Q, 64, QN, 128, kg * 128, dep)
    dep = _copy_w8_tile_async(qs + 8192, Q + 64 * QN, 64, QN, 128, kg * 128, dep)
    return _commit_w8_tile(dep)


@triton.jit
def _load_matrix_x4(shared, row, CW: tl.constexpr, channel, dep):
    # PPU x4 has row0-low, row0-high, row8-low, row8-high register order.
    return tl.inline_asm_elementwise(
        (
            "ppu.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 "
            "{$0,$1,$2,$3},[$4],{$5,$6,$7,$8,$9,$10};"
        ),
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,~{memory}",
        [shared, 0, row, 1, CW, 1, channel, dep],
        dtype=(tl.uint32,) * 4,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _square_accumulate(
    qs,
    aa,
    Q,
    A,
    S,
    QN,
    K: tl.constexpr,
    pm,
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
        dep = tl.full((256,), 0, tl.int32)
        dep = _prefetch_square(qs, aa, Q, A, QN, K, pm, 0, dep)
    else:
        dep = initial_dep
    c0 = tl.full((256,), 0.0, tl.float32)
    c1 = c0
    c2 = c0
    c3 = c0
    c4 = c0
    c5 = c0
    c6 = c0
    c7 = c0
    if SCALE_PAIR:
        tl.static_assert(FP8 and K == 256)
        scale0, scale1 = load_w8_packed_scales(S + expert * SE + n0 * SN, SN, FAST)
    unroll: tl.constexpr = 2 if FP8 and K % 256 == 0 else 1
    for kg in tl.range(K // 128, loop_unroll_factor=unroll):
        if not SCALE_PAIR:
            scale0 = tl.load(S + expert * SE + kg * SG + n0 * SN)
            scale1 = tl.load(S + expert * SE + kg * SG + (n0 + 8) * SN)
        if kg + 1 < K // 128:
            dep = _prefetch_square(
                qs + (kg + 1) % 2 * 16384,
                aa + (kg + 1) % 2 * 4096,
                Q,
                A,
                QN,
                K,
                pm,
                kg + 1,
                dep,
            )
            dep = _wait_w8_tile(dep, True)
        else:
            dep = _wait_w8_tile(dep, False)
        s0 = w8_scale_fragment(scale0, kg, SCALE_PAIR, FP8 and FAST)
        s1 = w8_scale_fragment(scale1, kg, SCALE_PAIR, FP8 and FAST)
        qhere = qs + kg % 2 * 16384
        ahere = aa + kg % 2 * 4096
        for group in tl.static_range(4):
            (r0, r1, r2, r3) = _load_matrix_x4(
                qhere + (warp // 4) * 8192, (warp % 4) * 16, 64, group * 32, dep
            )
            for part in tl.static_range(2):
                q0 = r0 if part == 0 else r1
                q8 = r2 if part == 0 else r3
                sub = group * 2 + part
                if FP8:
                    if FAST:
                        (a0, a1, a2, a3) = paired_fp8_folded(q0, q8, s0, s1)
                    else:
                        (a0, a1, a2, a3) = paired_fp8_robust(q0, q8, s0, s1)
                    (b0, b1, b2, b3) = _load_matrix_x4(
                        ahere + sub // 4 * 2048, 0, 16, sub % 4 * 32, dep
                    )
                else:
                    (b0, b1, b2, b3) = _load_matrix_x4(
                        ahere + sub // 4 * 2048, 0, 16, sub % 4 * 32, dep
                    )
                    (a0, a1) = _decode_int8_fragment_bf16(q0, s0, s0)
                    (a2, a3) = _decode_int8_fragment_bf16(q8, s1, s1)
                (c0, c1, c2, c3, c4, c5, c6, c7) = _mma_bf16_square(
                    a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7
                )
        tl.debug_barrier()
    return (c0, c1, c2, c3, c4, c5, c6, c7)


@libentry()
@triton.jit
def _marlin_moe_gemm_grouped_square_kernel(
    A,
    Q,
    C,
    S,
    TW,
    Ids,
    Experts,
    Post,
    N: tl.constexpr,
    K: tl.constexpr,
    VALID,
    QE,
    QN,
    SE,
    SG,
    SN,
    CM,
    CN,
    MUL: tl.constexpr,
    SSafe,
    FP8: tl.constexpr,
    USE32: tl.constexpr = False,
    PREFETCH_BEFORE_SCALE: tl.constexpr = False,
    SCALE_PAIR: tl.constexpr = False,
    PREFETCH_ROUTES: tl.constexpr = False,
    PACK_WIDTH: tl.constexpr = False,
):
    tl.static_assert(N % 128 == 0 and K % 128 == 0)
    pid = tl.program_id(0)
    pm = pid // (N // 128)
    pn = pid % (N // 128)
    if pm * 16 >= tl.load(Post):
        return
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    expert_word = tl.load(Experts + pm)
    if PACK_WIDTH:
        expert = (expert_word & 0x3FFFFFFF).to(offset_type)
        short_tile = (expert_word & 0x40000000) != 0
    else:
        expert = expert_word.to(offset_type)
    thread = tl.arange(0, 256)
    lane = thread % 32
    warp = thread // 32
    g = lane // 4
    t = lane % 4
    n0 = pn * 128 + warp * 16 + g
    qmem = tle_async.gpu.alloc(
        [32768],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    amem = tle_async.gpu.alloc(
        [8192],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    qs = tle_async.gpu.local_ptr(qmem, (0,)).to(tl.int64).to(tl.int32)
    aa = tle_async.gpu.local_ptr(amem, (0,)).to(tl.int64).to(tl.int32)
    qb = Q + expert * QE + pn * 128 * QN
    inner_n = warp * 16 + g
    if not PACK_WIDTH:
        short_tile = tl.load(Ids + pm * 16 + 8) >= VALID
    if short_tile:
        if PREFETCH_BEFORE_SCALE:
            dep = tl.full((256,), 0, tl.int32)
            dep = _prefetch_w8_tile(qs, aa, qb, A, QN, K, pm * 2, 0, dep, True)
        else:
            dep = None
        if PREFETCH_ROUTES:
            early0 = tl.load(Ids + pm * 16 + 2 * t)
            early1 = tl.load(Ids + pm * 16 + 2 * t + 1)
            if MUL:
                w0 = tl.load(TW + early0.to(offset_type), early0 < VALID, other=0).to(
                    tl.float32
                )
                w1 = tl.load(TW + early1.to(offset_type), early1 < VALID, other=0).to(
                    tl.float32
                )
        if FP8:
            can_fold = tl.load(SSafe + expert) != 0 if SSafe is not None else False
            if can_fold:
                (c0, c1, c2, c3) = _w8_narrow_accumulate(
                    qs,
                    aa,
                    qb,
                    A,
                    S,
                    QN,
                    K,
                    pm * 2,
                    expert,
                    n0,
                    inner_n,
                    SE,
                    SG,
                    SN,
                    True,
                    True,
                    IS_SPLIT_Q=True,
                    initial_dep=dep,
                    SCALE_PAIR=SCALE_PAIR,
                )
            else:
                (c0, c1, c2, c3) = _w8_narrow_accumulate(
                    qs,
                    aa,
                    qb,
                    A,
                    S,
                    QN,
                    K,
                    pm * 2,
                    expert,
                    n0,
                    inner_n,
                    SE,
                    SG,
                    SN,
                    True,
                    False,
                    IS_SPLIT_Q=True,
                    initial_dep=dep,
                    SCALE_PAIR=SCALE_PAIR,
                )
        else:
            (c0, c1, c2, c3) = _w8_narrow_accumulate(
                qs,
                aa,
                qb,
                A,
                S,
                QN,
                K,
                pm * 2,
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
            token0 = tl.load(Ids + pm * 16 + 2 * t).to(offset_type)
            token1 = tl.load(Ids + pm * 16 + 2 * t + 1).to(offset_type)
        if MUL:
            if not PREFETCH_ROUTES:
                w0 = tl.load(TW + token0, token0 < VALID, other=0).to(tl.float32)
                w1 = tl.load(TW + token1, token1 < VALID, other=0).to(tl.float32)
        else:
            w0 = 1.0
            w1 = 1.0
        c0 *= w0
        c2 *= w0
        c1 *= w1
        c3 *= w1
        tl.store(C + token0 * CM + n0 * CN, c0.to(C.dtype.element_ty), token0 < VALID)
        tl.store(C + token1 * CM + n0 * CN, c1.to(C.dtype.element_ty), token1 < VALID)
        tl.store(
            C + token0 * CM + (n0 + 8) * CN, c2.to(C.dtype.element_ty), token0 < VALID
        )
        tl.store(
            C + token1 * CM + (n0 + 8) * CN, c3.to(C.dtype.element_ty), token1 < VALID
        )
        return
    if PREFETCH_BEFORE_SCALE:
        dep = tl.full((256,), 0, tl.int32)
        dep = _prefetch_square(qs, aa, qb, A, QN, K, pm, 0, dep)
    else:
        dep = None
    if PREFETCH_ROUTES:
        tokens = ()
        weights = ()
        for j in tl.static_range(4):
            token = tl.load(Ids + pm * 16 + t + 4 * j)
            tokens += (token,)
            if MUL:
                weight = tl.load(TW + token.to(offset_type), token < VALID, other=0).to(
                    tl.float32
                )
                weights += (weight,)
    if FP8:
        can_fold = tl.load(SSafe + expert) != 0 if SSafe is not None else False
        if can_fold:
            c = _square_accumulate(
                qs,
                aa,
                qb,
                A,
                S,
                QN,
                K,
                pm,
                expert,
                n0,
                warp,
                SE,
                SG,
                SN,
                True,
                True,
                initial_dep=dep,
                SCALE_PAIR=SCALE_PAIR,
            )
        else:
            c = _square_accumulate(
                qs,
                aa,
                qb,
                A,
                S,
                QN,
                K,
                pm,
                expert,
                n0,
                warp,
                SE,
                SG,
                SN,
                True,
                False,
                initial_dep=dep,
                SCALE_PAIR=SCALE_PAIR,
            )
    else:
        c = _square_accumulate(
            qs, aa, qb, A, S, QN, K, pm, expert, n0, warp, SE, SG, SN, False, False
        )
    for j in tl.static_range(4):
        if PREFETCH_ROUTES:
            token = tokens[j].to(offset_type)
        else:
            token = tl.load(Ids + pm * 16 + t + 4 * j).to(offset_type)
        v0 = c[j]
        v1 = c[j + 4]
        if MUL:
            if PREFETCH_ROUTES:
                w = weights[j]
            else:
                w = tl.load(TW + token, token < VALID, other=0).to(tl.float32)
            v0 *= w
            v1 *= w
        tl.store(C + token * CM + n0 * CN, v0.to(C.dtype.element_ty), token < VALID)
        tl.store(
            C + token * CM + (n0 + 8) * CN, v1.to(C.dtype.element_ty), token < VALID
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
def _store_k32(k):
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
def _prefetch_w4_square(
    qs, aa, Q, A, QN, K: tl.constexpr, pm, kg, dep, ROWS: tl.constexpr
):
    dep = _copy_w8_tile_async(qs, Q, 64, QN, 64, kg * 64, dep)
    ab = A.to(tl.pointer_type(tl.uint8)) + pm * ROWS * K * 2
    dep = _copy_w8_tile_async(aa, ab, ROWS, K * 2, 128, kg * 256, dep)
    dep = _copy_w8_tile_async(
        aa + ROWS * 128, ab, ROWS, K * 2, 128, kg * 256 + 128, dep
    )
    return _commit_w8_tile(dep)


@libentry()
@triton.jit
def _raw_w4_square_kernel(
    A,
    Q,
    C,
    S,
    TW,
    Ids,
    Experts,
    Post,
    N: tl.constexpr,
    K: tl.constexpr,
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
    tl.static_assert(N % 64 == 0 and K % 128 == 0)
    tl.static_assert(ROWS == 8 or ROWS == 16)
    pid = tl.program_id(0)
    pm = pid // (N // 64)
    pn = pid % (N // 64)
    if pm * ROWS >= tl.load(Post):
        return
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    expert = tl.load(Experts + pm).to(offset_type)
    thread = tl.arange(0, 128)
    lane = thread % 32
    warp = thread // 32
    g = lane // 4
    t = lane % 4
    n0 = pn * 64 + warp * 16 + g
    qm = tle_async.gpu.alloc(
        [8192],
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
    qs = tle_async.gpu.local_ptr(qm, (0,)).to(tl.int64).to(tl.int32)
    aa = tle_async.gpu.local_ptr(am, (0,)).to(tl.int64).to(tl.int32)
    qb = Q + expert * QE + pn * 64 * QN
    dep = tl.full((128,), 0, tl.int32)
    dep = _prefetch_w4_square(qs, aa, qb, A, QN, K, pm, 0, dep, ROWS)
    c0 = tl.full((128,), 0.0, tl.float32)
    c1 = c0
    c2 = c0
    c3 = c0
    c4 = c0
    c5 = c0
    c6 = c0
    c7 = c0
    for kg in range(K // 128):
        if MX:
            scale_ptr = S + expert * SE + (kg * 4) * SG + n0 * SN
            src0 = mx_scale_source(scale_ptr, PACKED_SCALE)
            src1 = mx_scale_source(scale_ptr + 8 * SN, PACKED_SCALE)
        if not MX and ROWS == 16:
            scale0 = tl.load(S + expert * SE + kg * SG + n0 * SN)
            scale1 = tl.load(S + expert * SE + kg * SG + (n0 + 8) * SN)
        if kg + 1 < K // 128:
            dep = _prefetch_w4_square(
                qs + ((kg + 1) % 2) * 4096,
                aa + ((kg + 1) % 2) * (ROWS * 256),
                qb,
                A,
                QN,
                K,
                pm,
                kg + 1,
                dep,
                ROWS,
            )
            dep = _wait_w8_tile(dep, True)
        else:
            dep = _wait_w8_tile(dep, False)
        qhere = qs + (kg % 2) * 4096
        ahere = aa + (kg % 2) * (ROWS * 256)
        if not MX:
            if ROWS == 8:
                scale0 = tl.load(S + expert * SE + kg * SG + n0 * SN)
                scale1 = tl.load(S + expert * SE + kg * SG + (n0 + 8) * SN)
            s0 = scale0.to(tl.uint16, bitcast=True).to(tl.uint32)
            s1 = scale1.to(tl.uint16, bitcast=True).to(tl.uint32)
            s0 = s0 | (s0 << 16)
            s1 = s1 | (s1 << 16)
        for group in tl.static_range(2):
            r0, r1, r2, r3 = _load_matrix_x4(qhere, warp * 16, 64, group * 32, dep)
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
                        b0, b1 = _load_activation_fragment_x2(
                            ahere + (sub // 4) * 1024, (sub % 4) * 32, dep
                        )
                        c0, c1, c2, c3 = _mma_bf16_m16n8(
                            a0, a2, a1, a3, b0, b1, c0, c1, c2, c3
                        )
                    else:
                        b0, b1, b2, b3 = _load_matrix_x4(
                            ahere + (sub // 4) * 2048, 0, 16, (sub % 4) * 32, dep
                        )
                        c0, c1, c2, c3, c4, c5, c6, c7 = _mma_bf16_square(
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
        token0 = tl.load(Ids + pm * 8 + 2 * t).to(offset_type)
        token1 = tl.load(Ids + pm * 8 + 2 * t + 1).to(offset_type)
        if MUL:
            w0 = tl.load(TW + token0, token0 < VALID, other=0).to(tl.float32)
            w1 = tl.load(TW + token1, token1 < VALID, other=0).to(tl.float32)
            c0 *= w0
            c2 *= w0
            c1 *= w1
            c3 *= w1
        tl.store(C + token0 * CM + n0 * CN, c0.to(C.dtype.element_ty), token0 < VALID)
        tl.store(C + token1 * CM + n0 * CN, c1.to(C.dtype.element_ty), token1 < VALID)
        tl.store(
            C + token0 * CM + (n0 + 8) * CN, c2.to(C.dtype.element_ty), token0 < VALID
        )
        tl.store(
            C + token1 * CM + (n0 + 8) * CN, c3.to(C.dtype.element_ty), token1 < VALID
        )
    else:
        c = (c0, c1, c2, c3, c4, c5, c6, c7)
        for j in tl.static_range(4):
            token = tl.load(Ids + pm * 16 + t + 4 * j).to(offset_type)
            v0 = c[j]
            v1 = c[j + 4]
            if MUL:
                w = tl.load(TW + token, token < VALID, other=0).to(tl.float32)
                v0 *= w
                v1 *= w
            tl.store(C + token * CM + n0 * CN, v0.to(C.dtype.element_ty), token < VALID)
            tl.store(
                C + token * CM + (n0 + 8) * CN, v1.to(C.dtype.element_ty), token < VALID
            )


@triton.jit
def _store_k16(k):
    # The permutation stays inside a scale group and avoids weight repacking.
    return (k & ~15) | (k & 1) | ((k & 2) << 2) | ((k & 12) >> 1)


@triton.jit
def _load_activation_fragment_x2(shared, channel: tl.constexpr, dep):
    # Load the packed fragment directly from the AIU shared-memory swizzle.
    return tl.inline_asm_elementwise(
        "ppu.ldmatrix.sync.aligned.m8n8.x2.swzl.shared.b16 {$0,$1},[$2],{$3,$4,$5,$6,$7,$8};",
        "=r,=r,r,r,r,r,r,r,r,r,~{memory}",
        [shared, 0, 0, 1, 8, 1, channel, dep],
        dtype=(tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _w8_narrow_accumulate(
    shared,
    ashared,
    qbase,
    A,
    S,
    QN,
    K: tl.constexpr,
    pm,
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
        dep = tl.full((256,), 0, tl.int32)
        dep = _prefetch_w8_tile(
            shared, ashared, qbase, A, QN, K, pm, 0, dep, IS_SPLIT_Q
        )
    else:
        dep = initial_dep
    c0 = tl.full((256,), 0.0, tl.float32)
    c1 = c0
    c2 = c0
    c3 = c0
    if SCALE_PAIR:
        tl.static_assert(FP8 and K == 256)
        scale0, scale1 = load_w8_packed_scales(S + expert * SE + n0 * SN, SN, FAST)
    unroll: tl.constexpr = 2 if FP8 and K % 256 == 0 else 1
    for kg in tl.range(K // 128, loop_unroll_factor=unroll):
        if not SCALE_PAIR:
            scale0 = tl.load(S + expert * SE + kg * SG + n0 * SN)
            scale1 = tl.load(S + expert * SE + kg * SG + (n0 + 8) * SN)
        if kg + 1 < K // 128:
            dep = _prefetch_w8_tile(
                shared + (kg + 1) % 2 * 16384,
                ashared + (kg + 1) % 2 * 2048,
                qbase,
                A,
                QN,
                K,
                pm,
                kg + 1,
                dep,
                IS_SPLIT_Q,
            )
            dep = _wait_w8_tile(dep, True)
        else:
            dep = _wait_w8_tile(dep, False)
        s0 = w8_scale_fragment(scale0, kg, SCALE_PAIR, FP8 and FAST)
        s1 = w8_scale_fragment(scale1, kg, SCALE_PAIR, FP8 and FAST)
        here = shared + kg % 2 * 16384
        ahere = ashared + kg % 2 * 2048
        warp = inner_n // 16
        for group in tl.static_range(4):
            if IS_SPLIT_Q:
                (r0, r1, r2, r3) = _load_matrix_x4(
                    here + (warp // 4) * 8192, (warp % 4) * 16, 64, group * 32, dep
                )
            else:
                (r0, r1, r2, r3) = _load_matrix_x4(
                    here, warp * 16, 128, group * 32, dep
                )
            for part in tl.static_range(2):
                q0 = r0 if part == 0 else r1
                q8 = r2 if part == 0 else r3
                sub = group * 2 + part
                if FP8:
                    if FAST:
                        (a0, a2, a1, a3) = paired_fp8_folded(q0, q8, s0, s1)
                    else:
                        (a0, a2, a1, a3) = paired_fp8_robust(q0, q8, s0, s1)
                    (b0, b1) = _load_activation_fragment_x2(
                        ahere + sub // 4 * 1024, sub % 4 * 32, dep
                    )
                else:
                    (b0, b1) = _load_activation_fragment_x2(
                        ahere + sub // 4 * 1024, sub % 4 * 32, dep
                    )
                    (a0, a2) = _decode_int8_fragment_bf16(q0, s0, s0)
                    (a1, a3) = _decode_int8_fragment_bf16(q8, s1, s1)
                (c0, c1, c2, c3) = _mma_bf16_m16n8(
                    a0, a1, a2, a3, b0, b1, c0, c1, c2, c3
                )
        tl.debug_barrier()
    return (c0, c1, c2, c3)


@libentry()
@triton.jit
def _marlin_moe_gemm_grouped_narrow_kernel(
    A,
    Q,
    C,
    S,
    TW,
    Ids,
    Experts,
    Post,
    N: tl.constexpr,
    K: tl.constexpr,
    VALID,
    QE,
    QN,
    SE,
    SG,
    SN,
    CM,
    CN,
    MUL: tl.constexpr,
    SSafe,
    FP8: tl.constexpr,
    USE32: tl.constexpr = False,
    PREFETCH_BEFORE_SCALE: tl.constexpr = False,
    SCALE_PAIR: tl.constexpr = False,
):
    # Each warp computes 16 output channels for eight routed rows.
    tl.static_assert(N % 128 == 0 and K % 128 == 0)
    pid = tl.program_id(0)
    pm = pid // (N // 128)
    pn = pid % (N // 128)
    if pm * 8 >= tl.load(Post):
        return
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    expert = tl.load(Experts + pm).to(offset_type)
    thread = tl.arange(0, 256)
    lane = thread % 32
    warp = thread // 32
    g = lane // 4
    t = lane % 4
    inner_n = warp * 16 + g
    n0 = pn * 128 + inner_n
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
    qbase = Q + expert * QE + pn * 128 * QN
    if FP8:
        if PREFETCH_BEFORE_SCALE:
            dep = tl.full((256,), 0, tl.int32)
            dep = _prefetch_w8_tile(shared, ashared, qbase, A, QN, K, pm, 0, dep)
        else:
            dep = None
        can_fold = tl.load(SSafe + expert) != 0 if SSafe is not None else False
        if can_fold:
            c0, c1, c2, c3 = _w8_narrow_accumulate(
                shared,
                ashared,
                qbase,
                A,
                S,
                QN,
                K,
                pm,
                expert,
                n0,
                inner_n,
                SE,
                SG,
                SN,
                True,
                True,
                initial_dep=dep,
                SCALE_PAIR=SCALE_PAIR,
            )
        else:
            c0, c1, c2, c3 = _w8_narrow_accumulate(
                shared,
                ashared,
                qbase,
                A,
                S,
                QN,
                K,
                pm,
                expert,
                n0,
                inner_n,
                SE,
                SG,
                SN,
                True,
                False,
                initial_dep=dep,
                SCALE_PAIR=SCALE_PAIR,
            )
    else:
        c0, c1, c2, c3 = _w8_narrow_accumulate(
            shared,
            ashared,
            qbase,
            A,
            S,
            QN,
            K,
            pm,
            expert,
            n0,
            inner_n,
            SE,
            SG,
            SN,
            False,
            False,
        )
    token0 = tl.load(Ids + pm * 8 + 2 * t).to(offset_type)
    token1 = tl.load(Ids + pm * 8 + 2 * t + 1).to(offset_type)
    if MUL:
        w0 = tl.load(TW + token0, token0 < VALID, other=0).to(tl.float32)
        w1 = tl.load(TW + token1, token1 < VALID, other=0).to(tl.float32)
        c0 *= w0
        c2 *= w0
        c1 *= w1
        c3 *= w1
    tl.store(C + token0 * CM + n0 * CN, c0.to(C.dtype.element_ty), token0 < VALID)
    tl.store(C + token1 * CM + n0 * CN, c1.to(C.dtype.element_ty), token1 < VALID)
    tl.store(C + token0 * CM + (n0 + 8) * CN, c2.to(C.dtype.element_ty), token0 < VALID)
    tl.store(C + token1 * CM + (n0 + 8) * CN, c3.to(C.dtype.element_ty), token1 < VALID)


if tle_async is not None:

    @triton.jit
    def _w4_pair_gemm(
        A,
        b_block_ptr,
        S,
        expert,
        ns,
        se,
        sg,
        sn,
        N: tl.constexpr,
        K: tl.constexpr,
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
            A,
            shape=(EM, K // 2),
            strides=(K // 2, 1),
            offsets=(PID_M * BM, 0),
            block_shape=(BM, BK // 2),
            order=(1, 0),
        )
        ao = tl.make_block_ptr(
            A + EM * (K // 2),
            shape=(EM, K // 2),
            strides=(K // 2, 1),
            offsets=(PID_M * BM, 0),
            block_shape=(BM, BK // 2),
            order=(1, 0),
        )
        acc = tl.zeros((BN, BM), tl.float32)
        for tile in tl.range(0, tl.cdiv(K, BK), num_stages=STAGES):
            a0 = tle_async.load(
                ae, boundary_check=(0, 1), padding_option="zero", is_async=True
            )
            a1 = tle_async.load(
                ao, boundary_check=(0, 1), padding_option="zero", is_async=True
            )
            if N % BN == 0 and K % BK == 0:
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
                    S + expert * se + (tile * BK // GROUP_SIZE) * sg + ns * sn,
                    ns < N,
                    other=0,
                )[None, :]
                lo, hi = _decode_raw_int4_bf16(q, scale)
            else:
                gs = tile * BK // 32 + tl.arange(0, BK // 32)
                byte = tl.load(
                    S + expert * se + gs[:, None] * sg + ns[None, :] * sn,
                    (ns[None, :] < N) & (gs[:, None] * 32 < K),
                    other=127,
                ).to(tl.uint32)
                bits = e8m0_to_bf16_bits(byte)
                scale = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
                scale = tl.reshape(
                    tl.broadcast_to(scale[:, None, :], (BK // 32, 16, BN)),
                    (BK // 2, BN),
                )
                lo, hi = _decode_raw_mxfp4_bf16(q, scale)
            acc = tl.dot(tl.trans(lo), tl.trans(a0), acc)
            acc = tl.dot(tl.trans(hi), tl.trans(a1), acc)
            ae = tl.advance(ae, (0, BK // 2))
            ao = tl.advance(ao, (0, BK // 2))
            b_block_ptr = tl.advance(b_block_ptr, (BK // 2, 0))
        return acc

    @triton.jit
    def _quantized_gemm_tile(
        a_block_ptr,
        b_block_ptr,
        b_scale_ptr,
        expert,
        offs_n,
        stride_bse,
        stride_bsg,
        stride_bsn,
        N: tl.constexpr,
        K: tl.constexpr,
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
            QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128
            or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
        )
        BLOCK_SIZE_K_WEIGHT: tl.constexpr = (
            BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2
        )
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k_tile in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES):
            activation = tle_async.load(
                a_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            if K % BLOCK_SIZE_K == 0 and N % BLOCK_SIZE_N == 0:
                b_quantized = tle_async.load(b_block_ptr, is_async=True)
            else:
                b_quantized = tle_async.load(
                    b_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            weight = _dequant_weight(
                b_quantized,
                b_scale_ptr,
                expert,
                k_tile * BLOCK_SIZE_K,
                offs_n,
                stride_bse,
                stride_bsg,
                stride_bsn,
                N,
                K,
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
    def _marlin_moe_gemm_direct_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
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
            QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128
            or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
        )
        BLOCK_SIZE_K_WEIGHT: tl.constexpr = (
            BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2
        )

        pid = tl.program_id(0)
        num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
        route = pid // num_pid_n
        pid_n = pid % num_pid_n
        expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        a_row = route // A_ROUTE_DIVISOR
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr + a_row * stride_am,
            shape=(1, K),
            strides=(stride_am, stride_ak),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(
                (K if IS_8BIT else tl.cdiv(K, 2)),
                N,
            ),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
            order=(0, 1),
        )
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        acc = _quantized_gemm_tile(
            a_block_ptr,
            b_block_ptr,
            b_scale_ptr,
            expert,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            N,
            K,
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
            mask=(offs_m[None, :] == 0) & (offs_n[:, None] < N),
        )

    @libentry()
    @triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
    def _marlin_moe_matvec_direct_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
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
            QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128
            or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
        )
        BLOCK_SIZE_K_WEIGHT: tl.constexpr = (
            BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2
        )

        pid = tl.program_id(0)
        num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
        route = pid // num_pid_n
        pid_n = pid % num_pid_n
        expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        a_row = route // A_ROUTE_DIVISOR
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(
                (K if IS_8BIT else tl.cdiv(K, 2)),
                N,
            ),
            strides=(stride_bk, stride_bn),
            offsets=(
                (
                    (tl.program_id(1) * tl.cdiv(tl.cdiv(K, 128), SPLIT_K) * 128)
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
            chunk: tl.constexpr = tl.cdiv(tl.cdiv(K, BLOCK_SIZE_K), SPLIT_K)
            begin = tl.program_id(1) * chunk
            end = tl.minimum(begin + chunk, tl.cdiv(K, BLOCK_SIZE_K))
        else:
            begin = 0
            end = tl.cdiv(K, BLOCK_SIZE_K)
        for k_tile in tl.range(begin, end, num_stages=PIPELINE_STAGES):
            a = tl.load(
                a_ptr
                + a_row * stride_am
                + (k_tile * BLOCK_SIZE_K + offs_k) * stride_ak,
                mask=k_tile * BLOCK_SIZE_K + offs_k < K,
                other=0.0,
            )
            if K % BLOCK_SIZE_K == 0 and N % BLOCK_SIZE_N == 0:
                b_quantized = tle_async.load(b_block_ptr, is_async=True)
            else:
                b_quantized = tle_async.load(
                    b_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            b = _dequant_weight(
                b_quantized,
                b_scale_ptr,
                expert,
                k_tile * BLOCK_SIZE_K,
                offs_n,
                stride_bse,
                stride_bsg,
                stride_bsn,
                N,
                K,
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
            mask=offs_n < N,
        )

    @libentry()
    @triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
    def _marlin_moe_gemm_reduce_direct_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
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
            QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128
            or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
        )
        BLOCK_SIZE_K_WEIGHT: tl.constexpr = (
            BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2
        )

        pid = tl.program_id(0)
        num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
        token = pid // num_pid_n
        pid_n = pid % num_pid_n
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

        for topk_index in tl.range(0, TOP_K):
            route = token * TOP_K + topk_index
            expert = tl.load(topk_ids_ptr + route).to(tl.int64)
            route_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            a_block_ptr = tl.make_block_ptr(
                base=a_ptr + route * stride_am,
                shape=(1, K),
                strides=(stride_am, stride_ak),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                order=(1, 0),
            )
            b_block_ptr = tl.make_block_ptr(
                base=b_ptr + expert * stride_be,
                shape=(
                    (K if IS_8BIT else tl.cdiv(K, 2)),
                    N,
                ),
                strides=(stride_bk, stride_bn),
                offsets=(0, pid_n * BLOCK_SIZE_N),
                block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
                order=(0, 1),
            )

            for k_tile in tl.range(
                0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES
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
                b = _dequant_weight(
                    b_quantized,
                    b_scale_ptr,
                    expert,
                    k_tile * BLOCK_SIZE_K,
                    offs_n,
                    stride_bse,
                    stride_bsg,
                    stride_bsn,
                    N,
                    K,
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
        c_ptrs = (
            c_ptr + (token + offs_m[None, :]) * stride_cm + offs_n[:, None] * stride_cn
        )
        tl.store(
            c_ptrs,
            acc.to(compute_type),
            mask=(offs_m[None, :] == 0) & (offs_n[:, None] < N),
        )

    @libentry()
    @triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
    def _marlin_moe_gemm_silu_direct_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
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
            QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128
            or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
        )
        BLOCK_SIZE_K_WEIGHT: tl.constexpr = (
            BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2
        )

        pid = tl.program_id(0)
        num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
        route = pid // num_pid_n
        pid_n = pid % num_pid_n
        expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        a_row = route // A_ROUTE_DIVISOR
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr + a_row * stride_am,
            shape=(1, K),
            strides=(stride_am, stride_ak),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )
        b_gate_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(
                (K if IS_8BIT else tl.cdiv(K, 2)),
                2 * N,
            ),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
            order=(0, 1),
        )
        b_up_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(
                (K if IS_8BIT else tl.cdiv(K, 2)),
                2 * N,
            ),
            strides=(stride_bk, stride_bn),
            offsets=(0, N + pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
            order=(0, 1),
        )
        acc_gate = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

        for k_tile in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES):
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
            gate_b = _dequant_weight(
                b_gate_quantized,
                b_scale_ptr,
                expert,
                k_tile * BLOCK_SIZE_K,
                offs_n,
                stride_bse,
                stride_bsg,
                stride_bsn,
                N,
                K,
                compute_type,
                GROUP_SIZE_K,
                QUANT_TYPE,
            )
            up_b = _dequant_weight(
                b_up_quantized,
                b_scale_ptr + N * stride_bsn,
                expert,
                k_tile * BLOCK_SIZE_K,
                offs_n,
                stride_bse,
                stride_bsg,
                stride_bsn,
                N,
                K,
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
        c_ptrs = (
            c_ptr + (route + offs_m[None, :]) * stride_cm + offs_n[:, None] * stride_cn
        )
        tl.store(
            c_ptrs,
            acc.to(compute_type),
            mask=(offs_m[None, :] == 0) & (offs_n[:, None] < N),
        )

    @triton.jit
    def _stage_routed_activations_kernel(
        a_ptr,
        routed_a_ptr,
        sorted_token_ids_ptr,
        num_tokens_post_padded_ptr,
        EM: tl.constexpr,
        K: tl.constexpr,
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
            mask=valid & (offs_k[None, :] < K),
            other=0.0,
        )
        store_offset = offs_m[:, None] * K + offs_k[None, :]
        if PERMUTE_K32:
            tl.static_assert(K % 32 == 0 and not PERMUTE_K16 and not DEINTERLEAVE_K)
            store_offset = offs_m[:, None] * K + _store_k32(offs_k)[None, :]
        elif PERMUTE_K16:
            tl.static_assert(K % 16 == 0 and not DEINTERLEAVE_K)
            store_offset = offs_m[:, None] * K + _store_k16(offs_k)[None, :]
        elif DEINTERLEAVE_K:
            store_offset = (
                offs_m[:, None] * (K // 2)
                + offs_k[None, :] // 2
                + (offs_k[None, :] % 2) * EM * (K // 2)
            )
        tl.store(
            routed_a_ptr + store_offset,
            activation,
            mask=(offs_m[:, None] < EM) & (offs_k[None, :] < K),
        )

    @triton.jit
    def _silu_and_stage_routed_kernel(
        intermediate1_ptr,
        routed_intermediate2_ptr,
        sorted_token_ids_ptr,
        num_tokens_post_padded_ptr,
        EM: tl.constexpr,
        N: tl.constexpr,
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
            & (offs_n[None, :] < N)
        )
        row_base = routed_token[:, None] * (2 * N)
        gate = tl.load(
            intermediate1_ptr + row_base + offs_n[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            intermediate1_ptr + row_base + N + offs_n[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        output = (gate * tl.sigmoid(gate)) * up
        store_offset = offs_m[:, None] * N + offs_n[None, :]
        if PERMUTE_K32:
            tl.static_assert(N % 32 == 0 and not PERMUTE_K16 and not DEINTERLEAVE_K)
            store_offset = offs_m[:, None] * N + _store_k32(offs_n)[None, :]
        elif PERMUTE_K16:
            tl.static_assert(N % 16 == 0 and not DEINTERLEAVE_K)
            store_offset = offs_m[:, None] * N + _store_k16(offs_n)[None, :]
        elif DEINTERLEAVE_K:
            store_offset = (
                offs_m[:, None] * (N // 2)
                + offs_n[None, :] // 2
                + (offs_n[None, :] % 2) * EM * (N // 2)
            )
        tl.store(
            routed_intermediate2_ptr + store_offset,
            output,
            mask=(offs_m[:, None] < EM) & (offs_n[None, :] < N),
        )

    @libentry()
    @triton.jit(do_not_specialize_on_alignment=["routed_a_ptr", "b_ptr", "c_ptr"])
    def _marlin_moe_gemm_grouped_kernel(
        routed_a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
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
            QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128
            or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
        )
        BLOCK_SIZE_K_WEIGHT: tl.constexpr = (
            BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2
        )
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
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
            shape=(EM, K),
            strides=(K, 1),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(
                (K if IS_8BIT else tl.cdiv(K, 2)),
                N,
            ),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
            order=(0, 1),
        )
        if not IS_8BIT and compute_type == tl.bfloat16 and BLOCK_SIZE_M <= 32:
            acc = _w4_pair_gemm(
                routed_a_ptr,
                b_block_ptr,
                b_scale_ptr,
                expert,
                offs_n,
                stride_bse,
                stride_bsg,
                stride_bsn,
                N,
                K,
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
            acc = _quantized_gemm_tile(
                a_block_ptr,
                b_block_ptr,
                b_scale_ptr,
                expert,
                offs_n,
                stride_bse,
                stride_bsg,
                stride_bsn,
                N,
                K,
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
        if N % BLOCK_SIZE_N == 0:
            tl.store(c_ptrs, acc.to(compute_type), mask=token_mask[None, :])
        else:
            tl.store(
                c_ptrs,
                acc.to(compute_type),
                mask=token_mask[None, :] & (offs_n[:, None] < N),
            )


def _select_direct_block_n(n: int) -> int:
    if n <= 32:
        return 32
    if n < 512:
        return 64
    return 128


def _select_grouped_config(M: int, K: int, N: int, block_m: int, quant_type_id: int):
    if quant_type_id == QUANT_TYPE_UINT8B128 and block_m <= 32:
        # Sparse INT8 expert batches spill with 256-wide decoded tiles. A
        # 128-wide tile keeps both GEMMs resident across the upstream shapes.
        return 128, 1, 4, 3
    if quant_type_id == QUANT_TYPE_FP8_E4M3 and block_m <= 32:
        # Sparse FP8 expert batches need smaller decoded tiles to avoid spills.
        return (
            128 if K > N else 64,
            1 if M < 512 else 8,
            4,
            3,
        )
    if 256 <= M <= 512 and K > N:
        return 128, 1, 4, 3
    if M < 512:
        return 256, 1, 8, 3
    if M >= 4096:
        return 128, 8, 4, 3
    if M == 2048 and K > N:
        return 256, 8, 8, 2
    return 256, 8, 8, 3


def _select_grouped_block_m(M: int, E: int, top_k: int) -> int:
    routes = M * top_k
    if routes <= 16 * E:
        return 16
    if routes <= 32 * E:
        return 32
    return 64


def _use_direct_route(
    M: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
) -> bool:
    routes = M * top_k
    if num_experts <= 8 and routes >= num_experts:
        return False
    max_output_n = max(hidden_size, 2 * intermediate_size)
    block_n = _select_direct_block_n(max_output_n)
    n_tiles = triton.cdiv(max_output_n, block_n)
    max_routes_by_grid = 65535 // n_tiles
    return routes <= min(_PPU_DIRECT_ROUTE_LIMIT, max_routes_by_grid)


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
    # ponytail: cap pairwise ranks at 32 routes; larger calls use grouped alignment.
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
def w8_one_cta_routes(
    Ids,
    Sorted,
    Experts,
    Post,
    Rank,
    R: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    EM: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    BLOCK_SORTED: tl.constexpr,
    BLOCK_ZERO: tl.constexpr,
    PACK_WIDTH: tl.constexpr = False,
):
    sorted_offset = tl.arange(0, BLOCK_SORTED)
    tl.store(Sorted + sorted_offset, R, sorted_offset < EM)
    expert = tl.arange(0, BLOCK_ZERO)
    tl.store(Rank + expert, 0, expert < E)
    # All producers and consumers of this scratch belong to the same CTA.
    tl.debug_barrier()
    route = tl.arange(0, BLOCK_ROUTES)
    is_valid = route < R
    expert = tl.load(Ids + route, is_valid, other=0).to(tl.int32)
    count = tl.histogram(expert, BLOCK_EXPERTS, mask=is_valid)
    padded = tl.cdiv(count, BM) * BM
    start = tl.cumsum(padded, 0) - padded
    rank = tl.atomic_add(Rank + expert, 1, mask=is_valid, sem="relaxed")
    if PACK_WIDTH:
        tl.static_assert(BM == 16)
        offset = tl.gather(start, expert, 0) + rank
        valid_left = tl.gather(count, expert, 0) - rank
        # Record when the 16-row tile only needs its first eight rows.
        expert_word = expert | tl.where(valid_left <= 8, 0x40000000, 0)
    else:
        offset = tl.gather(start, expert, 0) + rank
        expert_word = expert
    tl.store(Sorted + offset, route, is_valid)
    tl.store(Experts + offset // BM, expert_word, is_valid & (rank % BM == 0))
    tl.store(Post, tl.sum(padded, 0))


@triton.jit
def fp8_safety_and_routes(
    S1,
    S2,
    F1,
    F2,
    Ids,
    Sorted,
    Experts,
    Post,
    Rank,
    E: tl.constexpr,
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
    R: tl.constexpr,
    BM: tl.constexpr,
    EM: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    BLOCK_SORTED: tl.constexpr,
    BLOCK_ZERO: tl.constexpr,
    PACK_WIDTH: tl.constexpr = False,
):
    if tl.program_id(0) == 0:
        # Routing never reads other CTAs' flags; GEMMs consume both after this launch.
        w8_one_cta_routes(
            Ids,
            Sorted,
            Experts,
            Post,
            Rank,
            R,
            E,
            BM,
            EM,
            BLOCK_ROUTES,
            BLOCK_EXPERTS,
            BLOCK_SORTED,
            BLOCK_ZERO,
            PACK_WIDTH=PACK_WIDTH,
        )
        return
    safety_pid = tl.program_id(0) - 1
    if R <= E:
        # Only the first route for each expert writes its scale flags.
        route = safety_pid
        expert = tl.load(Ids + route).to(tl.int64)
        prior = tl.arange(0, BLOCK_ROUTES)
        prior_ids = tl.load(Ids + prior, prior < route, other=-1).to(tl.int64)
        is_first = (expert >= 0) & (expert < E) & (tl.sum(prior_ids == expert, 0) == 0)
        if is_first:
            write_fp8_expert_scale_safety(S1, F1, expert, N1, G1, SE1, SN1, SG1, BLOCK1)
            write_fp8_expert_scale_safety(S2, F2, expert, N2, G2, SE2, SN2, SG2, BLOCK2)
    else:
        # An expert with no routed tokens has no GEMM consuming its flags.
        expert = safety_pid
        row = tl.arange(0, BLOCK_ROUTES)
        routed = tl.load(Ids + row, row < R, other=-1)
        is_active = tl.sum((routed == expert).to(tl.int32), 0) != 0
        if is_active:
            write_fp8_expert_scale_safety(S1, F1, expert, N1, G1, SE1, SN1, SG1, BLOCK1)
            write_fp8_expert_scale_safety(S2, F2, expert, N2, G2, SE2, SN2, SG2, BLOCK2)


def _align_grouped_tokens(
    topk_ids: torch.Tensor,
    block_m: int,
    num_experts: int,
    fp8_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    fp8_safety: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    is_one_cta: bool = False,
    packed_width: bool = False,
):
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
                max_padded_routes,
                triton.next_power_of_2(max(routes, warps * 32)),
                triton.next_power_of_2(num_experts),
                triton.next_power_of_2(max(max_padded_routes, warps * 32)),
                triton.next_power_of_2(max(num_experts, warps * 32)),
                num_warps=warps,
                num_stages=1,
            )
        else:
            s1, s2 = fp8_scales
            b1 = triton.next_power_of_2(s1.size(1) * s1.size(2))
            b2 = triton.next_power_of_2(s2.size(1) * s2.size(2))
            warps = max(warps, 16 if max(b1, b2) >= 8192 else 4)
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
                max_padded_routes,
                triton.next_power_of_2(max(routes, warps * 32)),
                triton.next_power_of_2(num_experts),
                triton.next_power_of_2(max(max_padded_routes, warps * 32)),
                triton.next_power_of_2(max(num_experts, warps * 32)),
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
            num_warps=16 if max(b1, b2) >= 8192 else 4,
        )
    return aligned


@triton.jit
def _reduce_split_silu_kernel(
    P,
    C,
    W,
    N: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    MUL: tl.constexpr,
    BN: tl.constexpr,
):
    r = tl.program_id(0)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    gate = tl.full((BN,), 0, tl.float32)
    up = tl.full((BN,), 0, tl.float32)
    for part in tl.static_range(S):
        gate += tl.load(P + part * R * 2 * N + r * 2 * N + ns, ns < N, other=0)
        up += tl.load(P + part * R * 2 * N + r * 2 * N + N + ns, ns < N, other=0)
    if MUL:
        w = tl.load(W + r)
        gate *= w
        up *= w
    # Match the original GEMM1 rounding before computing SiLU.
    gate = gate.to(C.dtype.element_ty).to(tl.float32)
    up = up.to(C.dtype.element_ty).to(tl.float32)
    result = tl.fdiv(gate, 1.0 + tl.exp(-gate)) * up
    tl.store(C + r * N + ns, result, ns < N)


def _invoke_marlin_moe_gemm_direct(
    A: torch.Tensor,
    C: torch.Tensor,
    stage: _QuantizedStage,
    topk_weights: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    *,
    mul_routed_weight: bool,
    a_route_divisor: int,
    group_size: int,
    compute_type,
    quant_type_id: int,
    use_matvec: bool = False,
    fuse_silu: bool = False,
    split_k: int = 1,
    pipeline_stages: Optional[int] = None,
    use_wide_gate: bool = False,
):
    if tle_async is None:
        raise RuntimeError("PPU fused Marlin MoE requires Triton TLE")

    B, B_scale = stage.weight, stage.scale
    K = A.size(1)
    N = C.size(-1) if fuse_silu else B.size(2)
    routes = topk_ids.numel()
    if fuse_silu:
        block_n = 32 if routes >= 12 else _select_direct_block_n(N)
        # Fill more of the 64 PPU compute units for sparse decode batches.
        if routes * triton.cdiv(N, block_n) < 64:
            block_n = 32
    else:
        block_n = 64 if use_matvec else _select_direct_block_n(N)
        if (
            use_matvec
            and min(K, N) <= 512
            and (
                quant_type_id == QUANT_TYPE_UINT8B128
                or (
                    quant_type_id == QUANT_TYPE_FP8_E4M3
                    and compute_type == tl.bfloat16
                    and B_scale.dtype == torch.bfloat16
                )
            )
        ):
            block_n = 32 if K > N else 128
            four_token_gate = (
                A.size(0) == 4
                and K == 4096
                and N == 512
                and group_size == 128
                and 4 <= a_route_divisor <= 8
            )
            if (
                quant_type_id == QUANT_TYPE_FP8_E4M3
                and (routes <= 16 or four_token_gate or use_wide_gate)
                and K >= 1024
                and N % 128 == 0
            ):
                # Use fewer CTAs for the long-K projection of very few routes.
                block_n = 128
    # Packed FP8 decode favors more long-K tiles and wider short-K tiles.
    # The PPU pipeline pass allocates ``num_stages - 1`` loop buffers.  Three
    # scheduling stages therefore provide the two buffers required to overlap
    # the next AIU copy with the current tile's unpack/dequantize/dot work.
    pipeline_stages = pipeline_stages or (3 if K > 128 else 1)
    grid = (routes * triton.cdiv(N, block_n),)

    kernel = (
        _marlin_moe_gemm_silu_direct_kernel
        if fuse_silu
        else (
            _marlin_moe_matvec_direct_kernel
            if use_matvec
            else _marlin_moe_gemm_direct_kernel
        )
    )

    target = C
    if split_k > 1:
        assert use_matvec and not fuse_silu
        target = torch.empty((split_k, routes, N), device=C.device, dtype=torch.float32)
        grid = (*grid, split_k)
        pipeline_stages = min(
            pipeline_stages, triton.cdiv(triton.cdiv(K, 128), split_k)
        )
    extra = dict(SPLIT_K=split_k, SPLIT_STRIDE=routes * N) if use_matvec else {}
    kernel[grid](
        A,
        B,
        target,
        B_scale,
        topk_weights,
        topk_ids,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        target.stride(-2),
        target.stride(-1),
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
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
        _reduce_split_silu_kernel[(routes, triton.cdiv(N // 2, 128))](
            target,
            C,
            topk_weights,
            N // 2,
            routes,
            split_k,
            mul_routed_weight,
            128,
            num_warps=4,
        )


def _invoke_marlin_moe_gemm_reduce_direct(
    A: torch.Tensor,
    C: torch.Tensor,
    stage: _QuantizedStage,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    mul_routed_weight: bool,
    group_size: int,
    compute_type,
    quant_type_id: int,
):
    if tle_async is None:
        raise RuntimeError("PPU fused Marlin MoE requires Triton TLE")

    B, B_scale = stage.weight, stage.scale
    K = A.size(1)
    N = B.size(2)
    top_k = topk_ids.size(1)
    tokens = A.size(0) // top_k
    block_n = (64 if N >= 1024 else 128) if tokens == 1 else 32
    pipeline_stages = 3 if K > 128 else 1
    grid = (tokens * triton.cdiv(N, block_n),)

    _marlin_moe_gemm_reduce_direct_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        topk_ids,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        TOP_K=top_k,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        PIPELINE_STAGES=pipeline_stages,
        compute_type=compute_type,
        BLOCK_SIZE_N=block_n,
        num_stages=pipeline_stages,
        QUANT_TYPE=quant_type_id,
    )


@triton.jit
def _stage_bf16_activation_pairs_kernel(
    A,
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
    a = A.to(tl.pointer_type(tl.uint32))
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


def _stage_grouped_activations(
    A: torch.Tensor,
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
    routed_a = torch.empty((em, A.size(1)), dtype=A.dtype, device=A.device)
    if (
        use_packed_gather
        and (permute_k16 or permute_k32)
        and A.dtype == torch.bfloat16
        and A.size(1) >= 1024
        and A.size(1) % (32 if permute_k32 else 16) == 0
        and A.stride(1) == 1
        and A.stride(0) % 2 == 0
        and A.data_ptr() % 4 == 0
    ):
        k_pairs = A.size(1) // 2
        _stage_bf16_activation_pairs_kernel[
            (triton.cdiv(em, 16), triton.cdiv(k_pairs, 256))
        ](
            A,
            routed_a,
            sorted_token_ids,
            num_tokens_post_padded,
            em,
            k_pairs,
            num_valid_tokens,
            A.stride(0) // 2,
            top_k,
            16,
            256,
            SKIP_UNUSED_UPPER_ROWS=block_m == 16 and not permute_k32,
            PERMUTE_K32=permute_k32,
            num_warps=8,
            num_stages=1,
        )
        return routed_a
    grid = (triton.cdiv(em, block_m), triton.cdiv(A.size(1), 128))
    _stage_routed_activations_kernel[grid](
        A,
        routed_a,
        sorted_token_ids,
        num_tokens_post_padded,
        EM=em,
        K=A.size(1),
        num_valid_tokens=num_valid_tokens,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
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


def _silu_and_stage_grouped(
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
    _silu_and_stage_routed_kernel[grid](
        intermediate1,
        routed,
        sorted_token_ids,
        num_tokens_post_padded,
        EM=em,
        N=n,
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


@triton.jit
def prefetch_quantized_wide_tile(
    qs, aa, Q, A, QN, K: tl.constexpr, pm, kg, dep, WEIGHT_K_BYTES: tl.constexpr
):
    ab = A.to(tl.pointer_type(tl.uint8)) + pm * 64 * K * 2
    dep = _copy_w8_tile_async(aa, ab, 64, K * 2, 128, kg * 256, dep)
    dep = _copy_w8_tile_async(aa + 8192, ab, 64, K * 2, 128, kg * 256 + 128, dep)
    dep = _copy_w8_tile_async(qs, Q, 128, QN, WEIGHT_K_BYTES, kg * WEIGHT_K_BYTES, dep)
    return _commit_w8_tile(dep)


@triton.jit
def accumulate_quantized_wide_tile(
    qs,
    aa,
    qb,
    A,
    S,
    QN,
    K: tl.constexpr,
    pm,
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
    dep = tl.full((256,), 0, tl.int32)
    dep = prefetch_quantized_wide_tile(qs, aa, qb, A, QN, K, pm, 0, dep, WEIGHT_K_BYTES)
    zero = tl.full((256,), 0.0, tl.float32)
    acc = ()
    for j in tl.static_range(LIVE // 2):
        acc += (zero,)
    for kg in range(K // 128):
        if not MX:
            scale0 = tl.load(S + expert * SE + kg * SG + n0 * SN)
            scale1 = tl.load(S + expert * SE + kg * SG + (n0 + 8) * SN)
        else:
            scale_ptr = S + expert * SE + (kg * 4) * SG + n0 * SN
            src0 = mx_scale_source(scale_ptr, PACKED_SCALE)
            src1 = mx_scale_source(scale_ptr + 8 * SN, PACKED_SCALE)
        if kg + 1 < K // 128:
            next_offset = (kg + 1) % 2 * 16384
            dep = prefetch_quantized_wide_tile(
                qs + (kg + 1) % 2 * 128 * WEIGHT_K_BYTES,
                aa + next_offset,
                qb,
                A,
                QN,
                K,
                pm,
                kg + 1,
                dep,
                WEIGHT_K_BYTES,
            )
            dep = _wait_w8_tile(dep, True)
        else:
            dep = _wait_w8_tile(dep, False)
        if not MX:
            s0 = w8_scale_fragment(scale0, kg, False, FAST)
            s1 = w8_scale_fragment(scale1, kg, False, FAST)
        current_offset = kg % 2 * 16384
        qhere = qs + kg % 2 * 128 * WEIGHT_K_BYTES
        ahere = aa + current_offset
        for group in tl.static_range(WEIGHT_K_BYTES // 32):
            r0, r1, r2, r3 = _load_matrix_x4(qhere, warp * 16, 128, group * 32, dep)
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
                            a0, a1 = _decode_int8_fragment_bf16(q0, s0, s0)
                            a2, a3 = _decode_int8_fragment_bf16(q8, s1, s1)
                        elif FAST:
                            a0, a1, a2, a3 = paired_fp8_folded(q0, q8, s0, s1)
                        else:
                            a0, a1, a2, a3 = paired_fp8_robust(q0, q8, s0, s1)
                    # Reuse the decoded fragment for each live group of 16 routes.
                    next_acc = ()
                    for chunk in tl.static_range(LIVE // 16):
                        b0, b1, b2, b3 = _load_matrix_x4(
                            ahere + sub // 4 * 8192, chunk * 16, 64, sub % 4 * 32, dep
                        )
                        next_acc += _mma_bf16_square(
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
    qs,
    aa,
    qb,
    A,
    C,
    S,
    TW,
    Ids,
    SSafe,
    QTYPE: tl.constexpr,
    PACKED_SCALE: tl.constexpr,
    K: tl.constexpr,
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
    pm,
    t,
    amem,
    pn,
):
    can_fold = tl.load(SSafe + expert) != 0 if SSafe is not None else False
    if can_fold:
        acc = accumulate_quantized_wide_tile(
            qs,
            aa,
            qb,
            A,
            S,
            QN,
            K,
            pm,
            expert,
            n0,
            warp,
            SE,
            SG,
            SN,
            QTYPE,
            PACKED_SCALE,
            True,
            LIVE,
        )
    else:
        acc = accumulate_quantized_wide_tile(
            qs,
            aa,
            qb,
            A,
            S,
            QN,
            K,
            pm,
            expert,
            n0,
            warp,
            SE,
            SG,
            SN,
            QTYPE,
            PACKED_SCALE,
            False,
            LIVE,
        )
    if K == 256:
        for chunk in tl.static_range(LIVE // 16):
            for j in tl.static_range(4):
                row = chunk * 16 + t + 4 * j
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
        tokens = tl.load(Ids + pm * 64 + rows).to(offset_type)
        valid = tokens < VALID
        if MUL:
            weights = tl.load(TW + tokens, valid, other=0).to(tl.float32)
            acc_output *= weights[:, None]
        tl.store(
            C + tokens[:, None] * CM + (pn * 128 + cols[None, :]) * CN,
            acc_output.to(C.dtype.element_ty),
            valid[:, None],
        )
    else:
        for chunk in tl.static_range(LIVE // 16):
            for j in tl.static_range(4):
                token = tl.load(Ids + pm * 64 + chunk * 16 + t + 4 * j).to(offset_type)
                v0 = acc[chunk * 8 + j]
                v1 = acc[chunk * 8 + j + 4]
                if MUL:
                    w = tl.load(TW + token, token < VALID, other=0).to(tl.float32)
                    v0 *= w
                    v1 *= w
                tl.store(
                    C + token * CM + n0 * CN, v0.to(C.dtype.element_ty), token < VALID
                )
                tl.store(
                    C + token * CM + (n0 + 8) * CN,
                    v1.to(C.dtype.element_ty),
                    token < VALID,
                )


@triton.jit
def quantized_wide_grouped_kernel(
    A,
    Q,
    C,
    S,
    TW,
    Ids,
    Experts,
    Post,
    N: tl.constexpr,
    K: tl.constexpr,
    VALID,
    QE,
    QN,
    SE,
    SG,
    SN,
    CM,
    CN,
    MUL: tl.constexpr,
    SSafe,
    QTYPE: tl.constexpr,
    PACKED_SCALE: tl.constexpr,
    USE32: tl.constexpr,
):
    pid = tl.program_id(0)
    pm = pid // (N // 128)
    pn = pid % (N // 128)
    if pm * 64 >= tl.load(Post):
        return
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    expert = tl.load(Experts + pm).to(offset_type)
    thread = tl.arange(0, 256)
    lane = thread % 32
    warp = thread // 32
    n0 = pn * 128 + warp * 16 + lane // 4
    t = lane % 4
    W4: tl.constexpr = (
        QTYPE == _TL_QUANT_TYPE_UINT4B8 or QTYPE == _TL_QUANT_TYPE_FP4_E2M1
    )
    qmem = tle_async.gpu.alloc(
        [16384 if W4 else 32768],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    if K == 256:
        # Reuse the activation buffer to assemble contiguous FP32 output rows.
        amem = tle_async.gpu.alloc(
            [8192],
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
    qs = tle_async.gpu.local_ptr(qmem, (0,)).to(tl.int64).to(tl.int32)
    aa = tle_async.gpu.local_ptr(amem, (0,)).to(tl.int64).to(tl.int32)
    qb = Q + expert * QE + pn * 128 * QN
    # Padding is contiguous at the end of each expert tile.
    if tl.load(Ids + pm * 64 + 48) >= VALID:
        project_quantized_wide_tile(
            qs,
            aa,
            qb,
            A,
            C,
            S,
            TW,
            Ids,
            SSafe,
            QTYPE,
            PACKED_SCALE,
            K,
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
            pm,
            t,
            amem,
            pn,
        )
    else:
        project_quantized_wide_tile(
            qs,
            aa,
            qb,
            A,
            C,
            S,
            TW,
            Ids,
            SSafe,
            QTYPE,
            PACKED_SCALE,
            K,
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
            pm,
            t,
            amem,
            pn,
        )


def _invoke_marlin_moe_gemm_grouped(
    routed_a: torch.Tensor,
    C: torch.Tensor,
    stage: _QuantizedStage,
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    mul_routed_weight: bool,
    top_k: int,
    block_m: int,
    group_size: int,
    compute_type,
    quant_type_id: int,
    is_down_projection: bool = False,
    raw_w8: bool = False,
    raw_w4: bool = False,
    packed_width: bool = False,
):
    if tle_async is None:
        raise RuntimeError("PPU fused Marlin MoE requires Triton TLE")
    B, B_scale = stage.weight, stage.scale
    em = sorted_token_ids.size(0)
    num_valid_tokens = C.numel() // C.size(-1)
    n = B.size(2)
    use32 = False
    if raw_w4 or (raw_w8 and quant_type_id == QUANT_TYPE_FP8_E4M3):
        largest_offset = max(
            sum((length - 1) * stride for length, stride in zip(B.shape, B.stride())),
            sum(
                (length - 1) * stride
                for length, stride in zip(B_scale.shape, B_scale.stride())
            ),
            num_valid_tokens * C.stride(-2) + n * C.stride(-1),
        )
        use32 = largest_offset < (1 << 31)
    packed_mx_scales = (
        raw_w4
        and quant_type_id == QUANT_TYPE_FP4_E2M1
        and B_scale.stride(1) == 1
        and B_scale.stride(0) % 4 == 0
        and B_scale.stride(2) % 4 == 0
        and B_scale.data_ptr() % 4 == 0
    )
    if (raw_w8 or raw_w4) and block_m == 64:
        quantized_wide_grouped_kernel[(triton.cdiv(em, 64) * triton.cdiv(n, 128),)](
            routed_a,
            B,
            C,
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
            B_scale.stride(0),
            B_scale.stride(1),
            B_scale.stride(2),
            C.stride(-2),
            C.stride(-1),
            mul_routed_weight,
            stage.scale_safe,
            quant_type_id,
            packed_mx_scales,
            use32,
            num_warps=8,
            num_stages=1,
        )
        return
    if raw_w4:
        _raw_w4_square_kernel[(triton.cdiv(em, block_m) * triton.cdiv(n, 64),)](
            routed_a,
            B,
            C,
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
            B_scale.stride(0),
            B_scale.stride(1),
            B_scale.stride(2),
            C.stride(-2),
            C.stride(-1),
            mul_routed_weight,
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
        and B_scale.stride(1) == 1
        and B_scale.stride(0) % 2 == 0
        and B_scale.stride(2) % 2 == 0
        and B_scale.data_ptr() % 4 == 0
    )
    if raw_w8 and block_m == 16:
        _marlin_moe_gemm_grouped_square_kernel[
            (triton.cdiv(em, 16) * triton.cdiv(n, 128),)
        ](
            routed_a,
            B,
            C,
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
            B_scale.stride(0),
            B_scale.stride(1),
            B_scale.stride(2),
            C.stride(-2),
            C.stride(-1),
            mul_routed_weight,
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
    if raw_w8 and block_m == 8:
        _marlin_moe_gemm_grouped_narrow_kernel[
            (triton.cdiv(em, 8) * triton.cdiv(n, 128),)
        ](
            routed_a,
            B,
            C,
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
            B_scale.stride(0),
            B_scale.stride(1),
            B_scale.stride(2),
            C.stride(-2),
            C.stride(-1),
            mul_routed_weight,
            stage.scale_safe,
            quant_type_id == QUANT_TYPE_FP8_E4M3,
            USE32=use32,
            PREFETCH_BEFORE_SCALE=prefetch_before_scale,
            SCALE_PAIR=scale_pair,
            num_warps=8,
            num_stages=1,
        )
        return
    batch_m = C.size(0) if is_down_projection else C.size(0) // top_k
    block_n, group_m, num_warps, pipeline_stages = _select_grouped_config(
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
    stride_cm, stride_cn = C.stride(-2), C.stride(-1)
    grid = (triton.cdiv(em, block_m) * triton.cdiv(n, block_n),)
    _marlin_moe_gemm_grouped_kernel[grid](
        routed_a,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N=n,
        K=routed_a.size(1),
        EM=em,
        num_valid_tokens=num_valid_tokens,
        stride_be=B.stride(0),
        stride_bk=B.stride(1),
        stride_bn=B.stride(2),
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        stride_bse=B_scale.stride(0),
        stride_bsg=B_scale.stride(1),
        stride_bsn=B_scale.stride(2),
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


def _validate_inputs(
    a,
    w1,
    w2,
    s1,
    s2,
    tw,
    ids,
    output,
    inplace,
    group_size,
    quant_type_id,
):
    if a.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or ids.ndim != 2:
        raise ValueError("Expected activations/routing rank 2 and weights rank 3")
    m, k = a.shape
    e, n2, _ = w1.shape
    n = n2 // 2
    if e <= 0 or n2 % 2 or min(k, n) <= 0 or k % 32 or n % 32:
        raise ValueError(
            "Positive K/N multiples of 32 and paired gate/up weights required"
        )
    if quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3):
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
        if s1.shape != (e, 2 * n, g1) or s2.shape != (e, k, g2):
            raise ValueError("8-bit scale shapes must match the selected group size")
        if quant_type_id == QUANT_TYPE_FP8_E4M3:
            fp8_dtype = getattr(torch, "float8_e4m3fn", torch.uint8)
            if w1.dtype not in (torch.uint8, fp8_dtype) or w2.dtype not in (
                torch.uint8,
                fp8_dtype,
            ):
                raise NotImplementedError("Expected output-major E4M3FN weights")
            if s1.dtype not in (torch.float16, torch.bfloat16, torch.float32) or (
                s2.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            ):
                raise NotImplementedError("FP8 scales must be FP16, BF16 or FP32")
        elif w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
            raise NotImplementedError("INT8 weights must use output-major uint8")
        elif s1.dtype != a.dtype or s2.dtype != a.dtype:
            raise NotImplementedError("INT8 scales must match activation dtype")
    else:
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
        if s1.shape != (e, 2 * n, k // group_size) or s2.shape != (
            e,
            k,
            n // group_size,
        ):
            raise ValueError("4-bit scale shapes do not match the selected group size")
        if quant_type_id == QUANT_TYPE_UINT4B8:
            if s1.dtype != a.dtype or s2.dtype != a.dtype:
                raise NotImplementedError("INT4 scales must match activation dtype")
        else:
            e8m0_dtype = getattr(torch, "float8_e8m0fnu", torch.uint8)
            if s1.dtype not in (torch.uint8, e8m0_dtype) or s2.dtype not in (
                torch.uint8,
                e8m0_dtype,
            ):
                raise NotImplementedError("MXFP4 scales must use E8M0 bytes")
    if tw.shape != ids.shape or ids.shape[0] != m or not 1 <= ids.shape[1] <= e:
        raise ValueError("Routing must have shape [M, topk], 1 <= topk <= E")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("PPU fused Marlin MoE requires FP16 or BF16")
    if ids.dtype not in (torch.int32, torch.int64) or tw.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise NotImplementedError("Unsupported routing dtype")
    tensors = (a, w1, w2, s1, s2, tw, ids)
    if a.device.type != "cuda" or any(t.device != a.device for t in tensors):
        raise ValueError("All tensors must reside on the same PPU device")
    if any(t.requires_grad for t in tensors):
        raise NotImplementedError("PPU fused Marlin MoE is inference-only")
    if any(not t.is_contiguous() for t in (a, tw, ids)):
        raise NotImplementedError("Activations and routing must be contiguous")
    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")
    if output is not None:
        if (
            output.shape != a.shape
            or output.dtype != a.dtype
            or output.device != a.device
        ):
            raise ValueError("Output shape/dtype/device must match hidden_states")
        if not output.is_contiguous() or output.requires_grad:
            raise ValueError("Output must be contiguous and inference-only")
    # Weights, scales and routing remain live during GEMM2.
    target = a if inplace else output
    if target is not None and target.numel():
        for tensor in (w1, w2, s1, s2, tw, ids):
            if (
                tensor.numel()
                and target.untyped_storage().data_ptr()
                == tensor.untyped_storage().data_ptr()
            ):
                raise ValueError("Output must not alias weights, scales or routing")
    return m, k, n, e, ids.shape[1]


@triton.jit
def check_routed_fp8_scales(
    S1,
    S2,
    F1,
    F2,
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
        write_fp8_expert_scale_safety(S1, F1, expert, N1, G1, SE1, SN1, SG1, BLOCK1)
        write_fp8_expert_scale_safety(S2, F2, expert, N2, G2, SE2, SN2, SG2, BLOCK2)


def _prepare_quantized_weights(
    w1,
    w2,
    w1_scale,
    w2_scale,
    quant_type_id,
    *,
    needs_expert_flags: bool,
):
    b1 = w1.view(torch.uint8).transpose(1, 2)
    b2 = w2.view(torch.uint8).transpose(1, 2)
    s1 = (
        w1_scale.view(torch.uint8) if quant_type_id == QUANT_TYPE_FP4_E2M1 else w1_scale
    ).transpose(1, 2)
    s2 = (
        w2_scale.view(torch.uint8) if quant_type_id == QUANT_TYPE_FP4_E2M1 else w2_scale
    ).transpose(1, 2)
    if needs_expert_flags:
        safe1 = torch.empty((w1.size(0),), device=w1.device, dtype=torch.int32)
        safe2 = torch.empty_like(safe1)
    else:
        safe1, safe2 = None, None
    return _QuantizedStage(b1, s1, safe1), _QuantizedStage(b2, s2, safe2)


@triton.jit
def decode_bank(
    Q,
    S,
    D,
    N: tl.constexpr,
    K: tl.constexpr,
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
    mask = (ns[:, None] < N) & (ks[None, :] < K)
    if W4:
        kps = k_base // 2 + tl.arange(0, BK // 2)
        q = tl.load(
            Q + expert * QE + ns[:, None] * QN + kps[None, :] * QK,
            (ns[:, None] < N) & (kps[None, :] * 2 < K),
            other=0,
        )
        if QTYPE == _TL_QUANT_TYPE_UINT4B8:
            scale = tl.load(
                S + expert * SE + ns * SN + (k_base // 128) * SG, ns < N, other=0
            )[:, None]
            lo, hi = _decode_raw_int4_bf16(q, scale)
        else:
            gs = k_base // 32 + tl.arange(0, BK // 32)
            byte = tl.load(
                S + expert * SE + ns[:, None] * SN + gs[None, :] * SG,
                (ns[:, None] < N) & (gs[None, :] * 32 < K),
                other=127,
            ).to(tl.uint32)
            bits = e8m0_to_bf16_bits(byte)
            scale = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
            scale = tl.reshape(
                tl.broadcast_to(scale[:, :, None], (BN, BK // 32, 16)), (BN, BK // 2)
            )
            lo, hi = _decode_raw_mxfp4_bf16(q, scale)
        value = tl.reshape(tl.join(lo, hi), (BN, BK))
    else:
        q = tl.load(
            Q + expert * QE + ns[:, None] * QN + ks[None, :] * QK,
            mask,
            other=0 if FP8 else 128,
        )
        scale = tl.load(
            S + expert * SE + ns * SN + (k_base // 128) * SG, ns < N, other=0
        )[:, None]
        if FP8:
            # Normalize the FP8 value before multiplying by the original scale.
            value = decode_fp8_scaled_bf16_normalized(q, scale)
        else:
            value = decode_int8_scaled_bf16(q, scale)
    tl.store(D + expert * N * K + ns[:, None] + ks[None, :] * N, value, mask)


@triton.jit
def _dense_w8_bf16_tile(
    A,
    B,
    C,
    TW,
    Ids,
    Experts,
    pm,
    pn,
    N: tl.constexpr,
    K: tl.constexpr,
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
    offsets = pm * BM + tl.arange(0, BM)
    tokens = tl.load(Ids + offsets, offsets < EM, other=VALID)
    expert = tl.load(Experts + pm) - FIRST_EXPERT
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    tokens = tokens.to(offset_type)
    expert = expert.to(offset_type)
    mask = tokens < VALID
    ks = tl.arange(0, BK)
    abase = A + (tokens[:, None] // TOPK) * AM + ks[None, :] * AK
    bp = tl.make_block_ptr(
        base=B + expert * BE,
        shape=(K, N),
        strides=(BKSTR, BNSTR),
        offsets=(0, pn * BN),
        block_shape=(BK, BN),
        order=(1, 0),
    )
    acc = tl.zeros((BM, BN), tl.float32)
    for kg in tl.range(0, tl.cdiv(K, BK), num_stages=STAGES):
        a = tl.load(abase, mask=mask[:, None] & (kg * BK + ks[None, :] < K), other=0.0)
        if DIRECT_B:
            cols = pn * BN + tl.arange(0, BN)
            b = tl.load(
                B
                + expert * BE
                + (kg * BK + ks[:, None]) * BKSTR
                + cols[None, :] * BNSTR
            )
        elif N % BN == 0 and K % BK == 0:
            b = tle_async.load(bp, is_async=True)
        else:
            b = tle_async.load(
                bp, boundary_check=(0, 1), padding_option="zero", is_async=True
            )
        acc = tl.dot(a, b, acc=acc)
        abase += BK * AK
        bp = tl.advance(bp, (BK, 0))
    if MUL:
        w = tl.load(TW + tokens, mask, other=0).to(tl.float32)
    else:
        w = tl.full((BM,), 1.0, tl.float32)
    acc *= w[:, None]
    ns = pn * BN + tl.arange(0, BN)
    tl.store(
        C + tokens[:, None] * CM + ns[None, :] * CN,
        acc.to(C.dtype.element_ty),
        mask[:, None] & (ns[None, :] < N),
    )


@triton.jit
def dense_w8_bf16_kernel(
    A,
    B,
    C,
    TW,
    Ids,
    Experts,
    Post,
    N: tl.constexpr,
    K: tl.constexpr,
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
    num_n = tl.cdiv(N, BN)
    pm, pn = tl.swizzle2d(pid // num_n, pid % num_n, num_m, num_n, GM)
    live_rows = tl.load(Post)
    if pm * BM >= live_rows:
        return
    if REVERSE_M:
        pm = live_rows // BM - 1 - pm
    _dense_w8_bf16_tile(
        A,
        B,
        C,
        TW,
        Ids,
        Experts,
        pm,
        pn,
        N,
        K,
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


@triton.jit
def dense_w8_bf16_expert_window(
    A,
    B,
    C,
    TW,
    Ids,
    Experts,
    Prefix,
    N: tl.constexpr,
    K: tl.constexpr,
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
    num_n = tl.cdiv(N, BN)
    # Reuse CTAs for overflow tiles when routes concentrate in one window.
    for pid in range(tl.program_id(0), num_m * num_n, tl.num_programs(0)):
        pm, pn = tl.swizzle2d(pid // num_n, pid % num_n, num_m, num_n, GM)
        pm = end // BM - 1 - pm
        _dense_w8_bf16_tile(
            A,
            B,
            C,
            TW,
            Ids,
            Experts,
            pm,
            pn,
            N,
            K,
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


@triton.jit
def dense_align_count_histogram(
    Ids,
    Count,
    Prefix,
    Sorted,
    R: tl.constexpr,
    E: tl.constexpr,
    EM: tl.constexpr,
    RB: tl.constexpr,
    ID_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    expert = tl.arange(0, E)
    row = pid * RB + tl.arange(0, RB)
    valid = row < R
    ids = tl.load(Ids + row, valid, other=0).to(tl.int32)
    val = tl.histogram(ids, E, mask=valid)
    tl.store(Count + (pid + 1) * E + expert, val)
    if pid == 0:
        tl.store(Prefix, 0)
        tl.store(Count + expert, 0)
    o = pid * ID_BLOCK + tl.arange(0, ID_BLOCK)
    tl.store(Sorted + o, R, o < EM)


def align_dense_fp8_histogram(ids, num_experts, split_at):
    from flaggems_vllm.ops.moe_align_block_size import (
        moe_align_block_size_stage2_vec,
        moe_align_block_size_stage3,
        moe_align_block_size_stage4,
    )

    r = ids.numel()
    e = num_experts
    em = r + 63 * e
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

    def scatter(first_expert, last_expert):
        moe_align_block_size_stage4[(e,)](
            ids,
            sorted_ids,
            experts,
            count,
            prefix,
            e,
            64,
            r,
            rb,
            FIRST_EXPERT=first_expert,
            LAST_EXPERT=last_expert,
        )

    scatter(split_at, e)

    def finish_routes():
        scatter(0, split_at)

    return sorted_ids, experts, post, prefix, finish_routes


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
    side = torch.cuda.Stream(device=hidden.device, priority=-1)
    for operand in (banks, w1, w2, w1_scale, w2_scale, ids):
        operand.record_stream(side)

    def decode(weight, scales, destination):
        count, width, reduction = weight.shape
        decode_bank[(width // 128, reduction // 128, count)](
            weight.view(torch.uint8),
            scales,
            destination,
            width,
            reduction,
            *weight.stride(),
            *scales.stride(),
            QUANT_TYPE_FP8_E4M3,
            128,
            128,
            num_warps=8,
            num_stages=1,
        )

    first, second = slice(half, experts_count), slice(0, half)
    gate_weight, gate_scale = w1[:, :n], w1_scale[:, :n]
    up_weight, up_scale = w1[:, n:], w1_scale[:, n:]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        sorted_ids, experts, padded, prefix, finish_routes = align_dense_fp8_histogram(
            ids, experts_count, half
        )
    decode(gate_weight[first], gate_scale[first], gate_bank)
    main.wait_stream(side)
    for metadata in (sorted_ids, experts, padded, prefix):
        metadata.record_stream(main)
    em = sorted_ids.numel()
    row_tiles = triton.cdiv(em, 64)

    def project_gate(bank, projected, window):
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
            if window == first:
                finish_routes()
            decode(up_weight[window], up_scale[window], up_bank)
        main.wait_stream(side)
        side.wait_stream(main)
        project_gate(up_bank, up_output, window)
        with torch.cuda.stream(side):
            if window == first:
                decode(gate_weight[second], gate_scale[second], gate_bank)
            else:
                decode(w2[:, :down_tile], w2_scale[:, :down_tile], gate_bank)
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
                decode(
                    w2[:, end : end + down_tile],
                    w2_scale[:, end : end + down_tile],
                    up_bank,
                )
            main.wait_stream(side)
            gate_bank, up_bank = up_bank, gate_bank
    return output


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
        and m >= 8192
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
            + 16 * (routes + 63 * e)
            + 8
        )
        if workspace <= _PPU_DENSE_MAX_WORKSPACE_BYTES:
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
    sorted_ids, experts, padded = _align_grouped_tokens(topk_ids, 64, e)
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
        tile = gate_tile if first else down_tile
        for col in range(0, output_size, tile):
            stop = min(col + tile, output_size)
            width = stop - col
            qw, ss = weight[:, col:stop], scales[:, col:stop]
            # Decode only the output columns consumed by this GEMM launch.
            decode_cols = 128 if is_fp8 and width >= 128 else 64
            decode_bank[
                (triton.cdiv(width, decode_cols), triton.cdiv(reduction, 128), e)
            ](
                qw.view(torch.uint8),
                (ss.view(torch.uint8) if quant_type_id == QUANT_TYPE_FP4_E2M1 else ss),
                scratch,
                width,
                reduction,
                *qw.stride(),
                *ss.stride(),
                quant_type_id,
                decode_cols,
                128,
                num_warps=8 if decode_cols == 128 else 4,
                num_stages=1,
            )
            decoded = (
                scratch[: e * reduction * width]
                .view(e, reduction, width)
                .transpose(1, 2)
            )
            projected = gateup[:, :, col:stop] if first else down[:, :, :width]
            if is_w4:
                invoke_fused_moe_triton_kernel(
                    hidden_states if first else intermediate,
                    decoded,
                    projected,
                    None,
                    None,
                    topk_weights,
                    sorted_ids,
                    experts,
                    padded,
                    (
                        apply_router_weight_on_input
                        if first
                        else not apply_router_weight_on_input
                    ),
                    topk if first else 1,
                    config,
                    tl.bfloat16,
                )
            else:
                # Direct loads avoid staging aligned short-K down-projection tiles.
                direct_b = (
                    not first
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
                    hidden_states if first else intermediate,
                    decoded,
                    projected,
                    topk_weights,
                    sorted_ids,
                    experts,
                    padded,
                    width,
                    reduction,
                    sorted_ids.numel(),
                    m * topk,
                    *(hidden_states.stride() if first else intermediate.stride()),
                    decoded.stride(0),
                    decoded.stride(2),
                    decoded.stride(1),
                    projected.stride(-2),
                    projected.stride(-1),
                    topk if first else 1,
                    (
                        apply_router_weight_on_input
                        if first
                        else not apply_router_weight_on_input
                    ),
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
            if not first:
                moe_sum(projected, output[:, col:stop])
        if first:
            flat = gateup.view(m * topk, n2)
            silu_and_mul_out(flat[:, :n], flat[:, n:], intermediate)
    return output


@triton.jit
def tiny_fp8_down_reduce(
    Activation,
    Weight,
    Scale,
    Ids,
    Router,
    Out,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    SE,
    SN,
    SG,
    BN: tl.constexpr,
    MUL: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, K)
    total = tl.zeros((BN,), tl.float32)
    for j in tl.static_range(TOPK):
        route = token * TOPK + j
        expert = tl.load(Ids + route).to(tl.int32)
        activation = tl.load(Activation + route * K + ks).to(tl.float32)
        quantized = tl.load(Weight + expert * (N * K) + col[:, None] * K + ks[None, :])
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
    tl.store(Out + token * N + col, total)


def _fused_marlin_moe_impl(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_ids,
    *,
    apply_router_weight_on_input=False,
    inplace=False,
    output=None,
    group_size=128,
    quant_type_id=QUANT_TYPE_UINT4B8,
):
    """Forward SwiGLU MoE with output-major quantized weights.

    All weights and scales are read directly on every invocation. MXFP4
    E8M0 scales are decoded on every invocation without a persistent copy.
    Expert IDs must be in [0, E); expert parallel maps are not supported.
    """
    m, k, n, e, topk = _validate_inputs(
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
    if m == 0:
        return out
    compute_type = tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16
    with torch.cuda.device(hidden_states.device):
        packed_fp8 = (
            quant_type_id == QUANT_TYPE_FP8_E4M3
            and hidden_states.dtype
            == w1_scale.dtype
            == w2_scale.dtype
            == torch.bfloat16
        )
        wide_gate_shape = k == 4096 and n == 256 and 4 <= topk <= 8
        wide_fp8_gate_shape = packed_fp8 and group_size == 128 and wide_gate_shape
        # A 64-row tile reuses each decoded weight fragment across three or four MMAs.
        use_wide_tile = (
            wide_gate_shape
            and hidden_states.dtype == torch.bfloat16
            and 1024 <= m < _PPU_DENSE_MIN_TOKENS
            and 32 * e < m * topk <= 64 * e
            and torch.cuda.get_device_properties(
                hidden_states.device
            ).shared_memory_per_block_optin
            >= _PPU_WIDE_SHARED_BYTES
        )
        raw_w8 = (
            quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3)
            and compute_type == tl.bfloat16
            and group_size == 128
            and w1_scale.dtype == w2_scale.dtype == torch.bfloat16
            and k % 128 == 0
            and n % 128 == 0
            and (m * topk <= 16 * e or use_wide_tile)
            and w1.is_contiguous()
            and w2.is_contiguous()
        )
        raw_w4 = (
            compute_type == tl.bfloat16
            and k % 128 == 0
            and n % 128 == 0
            and (m * topk <= 16 * e or use_wide_tile)
            and w1.is_contiguous()
            and w2.is_contiguous()
            and (
                (
                    quant_type_id == QUANT_TYPE_UINT4B8
                    and group_size == 128
                    and w1_scale.dtype == w2_scale.dtype == torch.bfloat16
                )
                or (quant_type_id == QUANT_TYPE_FP4_E2M1 and group_size == 32)
            )
        )
        # Sparse raw tiles pad to eight rows; account for both staged activations.
        padded_routes = m * topk + 7 * e
        sparse_bytes = (
            4 * (e + 1) ** 2
            + 16 * padded_routes
            + 2 * padded_routes * (k + n)
            + 2 * m * topk * (k + 2 * n)
            + 8
        )
        sparse_raw = raw_w4 or (raw_w8 and quant_type_id == QUANT_TYPE_UINT8B128)
        direct = _use_direct_route(m, e, topk, k, n) and not (
            sparse_raw and sparse_bytes <= _PPU_DENSE_MAX_WORKSPACE_BYTES
        )
        # A one-token gate/up grid may not fill the PPU without split K.
        is_tiny_int8_matvec = (
            raw_w8
            and quant_type_id == QUANT_TYPE_UINT8B128
            and m == 1
            and k >= 1024
            and k % 1024 == 0
            and 2 * k < (1 << 17)
            and 2 * n <= 1024
            and topk * triton.cdiv(2 * n, 128) < 32
        )
        direct = direct or is_tiny_int8_matvec
        if (
            packed_fp8
            and e > 8
            and m * topk <= 160
            and max(k, 2 * n) <= 4096
            and not raw_w8
        ):
            direct = True
        # Reuse routed weights once the grouped projection can fill the PPU.
        is_raw_fp8_sparse = (
            raw_w8
            and packed_fp8
            and m * topk <= min(32, e // 2)
            and m * topk * triton.cdiv(2 * n, 128) >= 64
            and sparse_bytes <= _PPU_DENSE_MAX_WORKSPACE_BYTES
        )
        small_fp8_gate = wide_fp8_gate_shape and m in (1, 2, 4)
        decode_fp8_gate = (
            wide_fp8_gate_shape
            and raw_w8
            and 4 < m <= 32
            and 32 <= m * topk <= 192
            # Denser routes reuse expert weights in the grouped projection.
            and 4 * m * topk <= 3 * e
        )
        direct = direct and (not is_raw_fp8_sparse or (small_fp8_gate and m == 4))
        direct = direct or decode_fp8_gate
        is_joint_safety = (
            raw_w8
            and packed_fp8
            and not direct
            and m < _PPU_DENSE_MIN_TOKENS
            and max(
                w1_scale.size(1) * w1_scale.size(2), w2_scale.size(1) * w2_scale.size(2)
            )
            <= 16384
        )
        if (
            m >= _PPU_DENSE_MIN_TOKENS
            and m * topk > 32 * e
            and hidden_states.dtype == torch.bfloat16
            and (
                (
                    group_size == 128
                    and w1_scale.dtype == w2_scale.dtype == torch.bfloat16
                )
                or quant_type_id == QUANT_TYPE_FP4_E2M1
            )
            and max(w1.numel(), w2.numel())
            * (4 if quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1) else 2)
            <= _PPU_DENSE_MAX_DECODE_BYTES
        ):
            gate_tile = min(2 * n, _PPU_DENSE_GATE_TILE)
            down_tile = min(k, _PPU_DENSE_DOWN_TILE)
            decode_elements = e * max(gate_tile * k, down_tile * n)
            projection_width = max(2 * n, down_tile)
            fixed_bytes = 2 * decode_elements + 4 * (e + 1) ** 2 + 16 * 63 * e + 8
            bytes_per_token = topk * (2 * (projection_width + n) + 12)
            batch_tokens = min(
                m, (_PPU_DENSE_MAX_WORKSPACE_BYTES - fixed_bytes) // bytes_per_token
            )
            if batch_tokens < 1:
                raise NotImplementedError("PPU dense MoE workspace exceeds its budget")
            if batch_tokens < m and output is not None:
                # A chunk can overwrite activations needed by a later chunk.
                byte_offset = abs(output.data_ptr() - hidden_states.data_ptr())
                if 0 < byte_offset < hidden_states.nbytes:
                    raise ValueError(
                        "output must not partially overlap hidden_states in chunked dense MoE"
                    )
            for start in range(0, m, batch_tokens):
                end = min(start + batch_tokens, m)
                run_dense_wna16(
                    hidden_states[start:end],
                    w1,
                    w2,
                    w1_scale,
                    w2_scale,
                    topk_weights[start:end],
                    out[start:end],
                    topk_ids[start:end],
                    quant_type_id,
                    apply_router_weight_on_input,
                    gate_tile=gate_tile,
                    down_tile=down_tile,
                    decode_elements=decode_elements,
                    projection_width=projection_width,
                )
            return out
        stage1, stage2 = _prepare_quantized_weights(
            w1,
            w2,
            w1_scale,
            w2_scale,
            quant_type_id,
            needs_expert_flags=is_joint_safety,
        )
        split_silu = direct and (
            n >= 1024 or quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3)
        )
        use_matvec = direct and (
            packed_fp8
            or (m == 1 and quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3))
        )
        gemv_split_k = 1
        if (
            (packed_fp8 or is_tiny_int8_matvec)
            and direct
            and group_size == 128
            and k >= 1024
            and k % 1024 == 0
            and 2 * n <= 1024
        ):
            routes = m * topk
            # Fill the compute units for very sparse routes; the 33--64 route
            # band already has enough column tiles without another reduction.
            if small_fp8_gate:
                # The wide gate tiles need more CTAs at low token counts.
                gemv_split_k = 16 // m
            elif decode_fp8_gate:
                gemv_split_k = 2 if routes <= 64 else 8
            else:
                gemv_split_k = (
                    min(8, 64 // triton.next_power_of_2(routes)) if routes <= 64 else 4
                )
        tiny_fp8_down = (
            small_fp8_gate
            and direct
            and m == 2
            and w2.is_contiguous()
            and w2_scale.is_contiguous()
            and w2.numel() < (1 << 31)
        )
        reduced = tiny_fp8_down or (direct and m <= 2 and not split_silu)
        stage1_topk_weights = topk_weights if apply_router_weight_on_input else None
        stage2_topk_weights = None if apply_router_weight_on_input else topk_weights
        c2 = (
            torch.empty((m * topk, n), device=out.device, dtype=out.dtype)
            if direct
            else None
        )
        c1 = (
            torch.empty((m * topk, 2 * n), device=out.device, dtype=out.dtype)
            if split_silu and gemv_split_k == 1
            else None
        )
        c3 = (
            None
            if reduced
            else torch.empty((m, topk, k), device=out.device, dtype=out.dtype)
        )
        if direct:
            if split_silu:
                _invoke_marlin_moe_gemm_direct(
                    A=hidden_states,
                    C=c1 if gemv_split_k == 1 else c2,
                    stage=stage1,
                    topk_weights=stage1_topk_weights,
                    topk_ids=topk_ids,
                    mul_routed_weight=apply_router_weight_on_input,
                    a_route_divisor=topk,
                    group_size=group_size,
                    compute_type=compute_type,
                    quant_type_id=quant_type_id,
                    use_matvec=use_matvec,
                    split_k=gemv_split_k,
                    pipeline_stages=(
                        2
                        if small_fp8_gate
                        or (decode_fp8_gate and (gemv_split_k == 2 or m * topk >= 128))
                        else None
                    ),
                    use_wide_gate=decode_fp8_gate,
                )
                if gemv_split_k == 1:
                    silu_and_mul_out(c1[:, :n], c1[:, n:], c2)
            else:
                _invoke_marlin_moe_gemm_direct(
                    A=hidden_states,
                    C=c2,
                    stage=stage1,
                    topk_weights=stage1_topk_weights,
                    topk_ids=topk_ids,
                    mul_routed_weight=apply_router_weight_on_input,
                    a_route_divisor=topk,
                    group_size=group_size,
                    compute_type=compute_type,
                    quant_type_id=quant_type_id,
                    fuse_silu=True,
                )
            if tiny_fp8_down:
                block_n = 16
                tiny_fp8_down_reduce[(m, k // block_n)](
                    c2,
                    w2.view(torch.uint8),
                    w2_scale,
                    topk_ids,
                    topk_weights,
                    out,
                    k,
                    n,
                    topk,
                    *w2_scale.stride(),
                    block_n,
                    not apply_router_weight_on_input,
                    num_warps=4,
                )
            elif reduced:
                _invoke_marlin_moe_gemm_reduce_direct(
                    A=c2,
                    C=out,
                    stage=stage2,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    mul_routed_weight=not apply_router_weight_on_input,
                    group_size=group_size,
                    compute_type=compute_type,
                    quant_type_id=quant_type_id,
                )
            else:
                _invoke_marlin_moe_gemm_direct(
                    A=c2,
                    C=c3,
                    stage=stage2,
                    topk_weights=stage2_topk_weights,
                    topk_ids=topk_ids,
                    mul_routed_weight=not apply_router_weight_on_input,
                    a_route_divisor=1,
                    group_size=group_size,
                    compute_type=compute_type,
                    quant_type_id=quant_type_id,
                    use_matvec=use_matvec,
                    pipeline_stages=(
                        1 if small_fp8_gate else 2 if decode_fp8_gate else None
                    ),
                )
        else:
            packed_width = False
            bm = _select_grouped_block_m(m, e, topk)
            if (raw_w8 or raw_w4) and use_wide_tile:
                bm = 64
            elif raw_w8:
                narrow_density = 5 if quant_type_id == QUANT_TYPE_FP8_E4M3 else 6
                if wide_fp8_gate_shape:
                    narrow_density = 3
                bm = 8 if m * topk <= narrow_density * e else 16
            elif raw_w4:
                bm = 8 if m * topk <= 5 * e else 16
            if (
                sparse_raw
                and m <= 2
                and topk <= _PPU_SMALL_ALIGN_MAX_TOPK
                and m * topk <= e <= _PPU_SMALL_ALIGN_MAX_EXPERTS
            ):
                if m == 1:
                    # One token typically routes to distinct experts.
                    sorted_ids, experts, padded = moe_align_block_size_singleton(
                        topk_ids, bm
                    )
                else:
                    # Group routes so both tokens can share an expert tile.
                    sorted_ids, experts, padded = moe_align_block_size_small_grouped(
                        topk_ids, e, bm
                    )
            elif (
                raw_w8
                and quant_type_id == QUANT_TYPE_UINT8B128
                and m * topk <= _PPU_W8_PAIR_ALIGN_MAX_ROUTES
            ):
                routes = m * topk
                max_padded_routes = min(routes * bm, routes + e * (bm - 1))
                sorted_ids = torch.empty(
                    (max_padded_routes,), device=topk_ids.device, dtype=torch.int32
                )
                experts = torch.empty(
                    (routes,), device=topk_ids.device, dtype=torch.int32
                )
                padded = torch.empty((1,), device=topk_ids.device, dtype=torch.int32)
                w8_small_routes_kernel[(1,)](
                    topk_ids,
                    sorted_ids,
                    experts,
                    padded,
                    routes,
                    bm,
                    triton.next_power_of_2(max(routes, bm)),
                    num_warps=4,
                )
            else:
                is_one_cta = (raw_w8 or raw_w4) and m * topk <= 4096 and e <= 1024
                if is_joint_safety:
                    packed_width = is_one_cta and bm == 16
                    sorted_ids, experts, padded = _align_grouped_tokens(
                        topk_ids,
                        bm,
                        e,
                        fp8_scales=(w1_scale, w2_scale),
                        fp8_safety=(stage1.scale_safe, stage2.scale_safe),
                        packed_width=packed_width,
                        is_one_cta=is_one_cta,
                    )
                else:
                    sorted_ids, experts, padded = _align_grouped_tokens(
                        topk_ids,
                        bm,
                        e,
                        is_one_cta=is_one_cta,
                    )
            c1 = torch.empty((m * topk, 2 * n), device=out.device, dtype=out.dtype)
            staging = dict(
                em=sorted_ids.numel(),
                num_valid_tokens=m * topk,
                block_m=bm,
                deinterleave_k=quant_type_id
                in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1)
                and compute_type == tl.bfloat16
                and bm <= 32
                and not raw_w4,
                permute_k16=raw_w8,
                permute_k32=raw_w4,
                should_skip_inactive=raw_w8
                or raw_w4
                or quant_type_id == QUANT_TYPE_FP8_E4M3,
            )
            _invoke_marlin_moe_gemm_grouped(
                routed_a=_stage_grouped_activations(
                    hidden_states,
                    sorted_ids,
                    padded,
                    top_k=topk,
                    use_packed_gather=raw_w4
                    or (raw_w8 and (packed_fp8 or use_wide_tile)),
                    **staging,
                ),
                C=c1,
                stage=stage1,
                topk_weights=stage1_topk_weights,
                sorted_token_ids=sorted_ids,
                expert_ids=experts,
                num_tokens_post_padded=padded,
                mul_routed_weight=apply_router_weight_on_input,
                top_k=topk,
                block_m=bm,
                group_size=group_size,
                compute_type=compute_type,
                quant_type_id=quant_type_id,
                raw_w8=raw_w8,
                raw_w4=raw_w4,
                packed_width=packed_width,
            )
            routed_c2 = _silu_and_stage_grouped(
                c1,
                sorted_ids,
                padded,
                **staging,
                skip_unused_upper_rows=(
                    raw_w8 and quant_type_id == QUANT_TYPE_FP8_E4M3 and bm == 16
                ),
            )
            _invoke_marlin_moe_gemm_grouped(
                routed_a=routed_c2,
                C=c3,
                stage=stage2,
                topk_weights=stage2_topk_weights,
                sorted_token_ids=sorted_ids,
                expert_ids=experts,
                num_tokens_post_padded=padded,
                mul_routed_weight=not apply_router_weight_on_input,
                top_k=1,
                block_m=bm,
                group_size=group_size,
                compute_type=compute_type,
                quant_type_id=quant_type_id,
                is_down_projection=True,
                raw_w8=raw_w8,
                raw_w4=raw_w4,
                packed_width=packed_width,
            )
        if not reduced:
            moe_sum(c3, out)
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
    activation: Any = None,
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
        return _generic_fused_marlin_moe(**locals())
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
    return _fused_marlin_moe_impl(
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
