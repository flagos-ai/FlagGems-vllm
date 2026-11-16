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

INT4/MXFP4 retain packed weights. W8A16 reads public byte/scale storage.
Selected BF16 paths use packed casts and bounded per-call dense decoding.
"""

from typing import Any, Callable, NamedTuple, Optional

import torch
import triton
import triton.language as tl
from flag_gems import zeros
from torch.utils.weak import WeakTensorKeyDictionary

try:
    from triton.experimental.tle import language as tle_async
except ImportError:
    tle_async = None

from flaggems_vllm.ops.fused_marlin_moe import (
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
    _stack_8,
)
from flaggems_vllm.ops.fused_marlin_moe import (
    fused_marlin_moe as _generic_fused_marlin_moe,
)
from flaggems_vllm.ops.fused_marlin_moe import w4a16_int4_pack
from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_no_tle
from flaggems_vllm.ops.moe_sum import moe_sum
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.utils import libentry

_PPU_DIRECT_ROUTE_LIMIT = 32
_PPU_DENSE_MIN_TOKENS = 4096
_PPU_DENSE_GATE_TILE = 256
_PPU_DENSE_DOWN_TILE = 1024
_PPU_DENSE_MAX_DECODE_BYTES = 1 << 30
# Inputs and the caller-visible output are outside the temporary workspace budget.
_PPU_DENSE_MAX_WORKSPACE_BYTES = 2 << 30
_SCALE_CACHE = WeakTensorKeyDictionary()
_PPU_FAST_VARIANTS = {
    QUANT_TYPE_UINT4B8: (False,),
    QUANT_TYPE_UINT8B128: (False,),
    QUANT_TYPE_FP4_E2M1: (False,),
    QUANT_TYPE_FP8_E4M3: (True, False),
}
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
def _decode_e4m3(q, scale, compute_type: tl.constexpr, FAST: tl.constexpr = False):
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


@triton.jit
def _decode_e2m1(q, scale, compute_type: tl.constexpr):
    bits = ((q & 8) << 12) | ((q & 7) << 9)
    tiny = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return (tiny * 16384.0 * scale).to(compute_type)


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
    tl.static_assert(b.shape[0] == 16, "INT4 packing uses 128-K tiles")
    group = k_base // GROUP_SIZE
    scale = tl.load(
        s_ptr + expert * se + group * sg + ns * sn,
        mask=ns < N,
        other=0.0,
    )[None, :]
    parts = (
        (((b >> 0) & 0xF).to(compute_type) - 8.0) * scale,
        (((b >> 16) & 0xF).to(compute_type) - 8.0) * scale,
        (((b >> 4) & 0xF).to(compute_type) - 8.0) * scale,
        (((b >> 20) & 0xF).to(compute_type) - 8.0) * scale,
        (((b >> 8) & 0xF).to(compute_type) - 8.0) * scale,
        (((b >> 24) & 0xF).to(compute_type) - 8.0) * scale,
        (((b >> 12) & 0xF).to(compute_type) - 8.0) * scale,
        (((b >> 28) & 0xF).to(compute_type) - 8.0) * scale,
    )
    return _stack_8(parts, 16, ns.shape[0])


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
    if compute_type == tl.bfloat16 and b.shape[1] <= 128:
        return decode_int8_scaled_bf16(b.to(tl.uint8), scale)
    else:
        quant = b.to(tl.int32) - 128
        return (quant.to(compute_type) * scale).to(compute_type)


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
    tl.static_assert(b.shape[0] == 16, "MXFP4 packing uses 128-K tiles")
    parts = tl.arange(0, 8)
    shifts = (parts // 2) * 4 + (parts % 2) * 16
    quant = (b[None, :, :] >> shifts[:, None, None]) & 15
    quant = tl.reshape(quant, (128, ns.shape[0]))
    groups = k_base // 32 + tl.arange(0, 4)
    scales = tl.load(
        s_ptr + expert * se + groups[:, None] * sg + ns[None, :] * sn,
        mask=(ns[None, :] < N) & (groups[:, None] * 32 < K),
        other=1.0,
    )
    scales = tl.broadcast_to(scales[:, None, :], (4, 32, ns.shape[0]))
    scales = tl.reshape(scales, (128, ns.shape[0]))
    return _decode_e2m1(quant, scales, compute_type)


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
    FAST: tl.constexpr = False,
):
    tl.static_assert(b.shape[0] == 128 or b.shape[0] == 64)
    PACKED: tl.constexpr = (
        FAST is True
        and compute_type == tl.bfloat16
        and s_ptr.dtype.element_ty == tl.bfloat16
    )
    quant = b.to(tl.uint32)
    if GROUP_SIZE == -1 or GROUP_SIZE >= 128:
        group = 0 if GROUP_SIZE == -1 else k_base // GROUP_SIZE
        scale_ptrs = s_ptr + expert * se + group * sg + ns * sn
        if N % ns.shape[0] == 0:
            scale = tl.load(scale_ptrs)[None, :]
        else:
            scale = tl.load(scale_ptrs, mask=ns < N, other=1.0)[None, :]
        if FAST and not PACKED:
            scale = scale * (2.0**120 if compute_type == tl.bfloat16 else 256.0)
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
        if FAST and not PACKED:
            scale = scale * (2.0**120 if compute_type == tl.bfloat16 else 256.0)
        scale = tl.broadcast_to(
            scale[:, None, :], (128 // GROUP_SIZE, GROUP_SIZE, ns.shape[0])
        )
        scale = tl.reshape(scale, (128, ns.shape[0]))
    if (
        FAST is None
        and compute_type == tl.bfloat16
        and s_ptr.dtype.element_ty == tl.bfloat16
    ):
        decoded = decode_fp8_scaled_bf16_normalized(b.to(tl.uint8), scale)
    elif PACKED:
        decoded = decode_fp8_scaled_bf16(b.to(tl.uint8), scale)
    else:
        decoded = _decode_e4m3(quant, scale, compute_type, FAST)
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
    FAST: tl.constexpr = False,
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
            FAST,
        )


@triton.jit
def _pack_e8m0_kernel(
    S,
    O,
    N: tl.constexpr,
    G: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = tl.program_id(1)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n, g = i % N, i // N
    byte = tl.load(S + e * SE + n * SN + g * SG, mask=i < G * N, other=127)
    byte = byte.to(tl.int32)
    bits = tl.where(byte == 0, 0x00400000, byte << 23)
    bits = tl.where(byte == 255, 0x7FC00000, bits)
    tl.store(O + e * G * N + i, bits.to(tl.float32, bitcast=True), mask=i < G * N)


def _pack_e8m0(s):
    try:
        version = s._version
    except RuntimeError:
        # Inference tensors can change without a version counter.
        version = None
    # Capture the conversion so replay reads the current scale storage.
    cacheable = version is not None and not torch.cuda.is_current_stream_capturing()
    if cacheable:
        cached = _SCALE_CACHE.get(s)
        if cached is not None and cached[0] == version:
            return cached[1]
    e, n, g = s.shape
    out = torch.empty((e, g, n), device=s.device, dtype=torch.float32)
    _pack_e8m0_kernel[(triton.cdiv(n * g, 256), e)](
        s.view(torch.uint8), out, n, g, *s.stride(), 256
    )
    if cacheable:
        _SCALE_CACHE[s] = (version, out)
    return out


@triton.jit
def _reduce_safety_kernel(
    Chunks, Safe, EXPERTS: tl.constexpr, CHUNKS: tl.constexpr, BLOCK: tl.constexpr
):
    expert = tl.program_id(0)
    # The last program reduces all chunk flags directly. It does not depend
    # on the other programs' writes, so no inter-CTA synchronization is needed.
    count = CHUNKS if expert < EXPERTS else EXPERTS * CHUNKS
    base = expert * CHUNKS if expert < EXPERTS else 0
    bad = tl.full((), 0, tl.int32)
    for start in range(0, tl.cdiv(count, BLOCK)):
        offsets = start * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(Chunks + base + offsets, mask=offsets < count, other=0)
        bad |= tl.sum(values, 0)
    tl.store(Safe + expert, bad == 0)


@triton.jit
def _check_fp8_scale_kernel(
    S,
    Flags,
    N: tl.constexpr,
    G: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = tl.program_id(1)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n, g = i // G, i % G
    valid = i < N * G
    scale = tl.load(S + e * SE + n * SN + g * SG, mask=valid, other=1.0).to(tl.float32)
    folded = scale * (2.0**120)
    bad = (tl.abs(folded) == float("inf")) & (tl.abs(scale) != float("inf")) & valid
    tl.store(Flags + e * CHUNKS + tl.program_id(0), tl.sum(bad.to(tl.int32), 0) != 0)


@triton.jit
def _compact_fp8_scale_safety_kernel(
    S,
    Safe,
    N: tl.constexpr,
    G: tl.constexpr,
    E: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ex = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, BLOCK).to(tl.int64)
    n, g = i // G, i % G
    scale = tl.load(S + ex * SE + n * SN + g * SG, i < N * G, other=1).to(tl.float32)
    folded = scale * (2.0**120)
    bad = (
        (tl.abs(folded) == float("inf")) & (tl.abs(scale) != float("inf")) & (i < N * G)
    )
    tl.store(Safe + ex, tl.sum(bad.to(tl.int32)) == 0)
    # Grouped GEMM reads expert flags; keep the dense global gate conservative.
    if ex == 0:
        tl.store(Safe + E, 0)


def _fp8_scale_safety(s, compact=False):
    e, n, g = s.shape
    if compact and n * g <= 16384:
        safe = torch.empty((e + 1,), device=s.device, dtype=torch.int32)
        block = triton.next_power_of_2(n * g)
        _compact_fp8_scale_safety_kernel[(e,)](
            s, safe, n, g, e, *s.stride(), block, num_warps=16 if block >= 8192 else 4
        )
        return safe
    count = triton.cdiv(n * g, 256)
    chunks = torch.empty((e, count), device=s.device, dtype=torch.int32)
    safe = torch.empty((e + 1,), device=s.device, dtype=torch.int32)
    _check_fp8_scale_kernel[(count, e)](s, chunks, n, g, *s.stride(), count, 256)
    _reduce_safety_kernel[(e + 1,)](chunks, safe, e, count, 512)
    return safe


if tle_async is not None:

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
        FAST: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr = 128,
    ):
        """Load, decode and multiply one stream of quantized weight tiles."""
        IS_8BIT: tl.constexpr = (
            QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128
            or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
        )
        BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else 16
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
                FAST,
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
        S_safe,
        NUM_EXPERTS: tl.constexpr,
        QUANT_TYPE: tl.constexpr,
        FAST: tl.constexpr,
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
        BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else 16

        if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 and FAST is False:
            is_safe = tl.load(S_safe + NUM_EXPERTS)
            if is_safe:
                return

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
                (K if IS_8BIT else tl.cdiv(K, 128) * BLOCK_SIZE_K_WEIGHT),
                N,
            ),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
            order=((0, 1) if IS_8BIT else (1, 0)),
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
            FAST,
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
        S_safe,
        NUM_EXPERTS: tl.constexpr,
        QUANT_TYPE: tl.constexpr,
        FAST: tl.constexpr,
        SPLIT_K: tl.constexpr = 1,
        SPLIT_STRIDE: tl.constexpr = 0,
    ):
        BLOCK_SIZE_K: tl.constexpr = 128
        IS_8BIT: tl.constexpr = (
            QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128
            or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
        )
        BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else 16

        if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 and FAST is False:
            is_safe = tl.load(S_safe + NUM_EXPERTS)
            if is_safe:
                return

        pid = tl.program_id(0)
        num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
        route = pid // num_pid_n
        pid_n = pid % num_pid_n
        expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        a_row = route // A_ROUTE_DIVISOR
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(
                (K if IS_8BIT else tl.cdiv(K, 128) * BLOCK_SIZE_K_WEIGHT),
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
            order=((0, 1) if IS_8BIT else (1, 0)),
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
                FAST,
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
        S_safe,
        NUM_EXPERTS: tl.constexpr,
        QUANT_TYPE: tl.constexpr,
        FAST: tl.constexpr,
    ):
        """Compute GEMM2 and reduce all routed experts into one token row."""
        BLOCK_SIZE_M: tl.constexpr = 16
        BLOCK_SIZE_K: tl.constexpr = 128
        IS_8BIT: tl.constexpr = (
            QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128
            or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
        )
        BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else 16

        if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 and FAST is False:
            is_safe = tl.load(S_safe + NUM_EXPERTS)
            if is_safe:
                return

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
                    (K if IS_8BIT else tl.cdiv(K, 128) * BLOCK_SIZE_K_WEIGHT),
                    N,
                ),
                strides=(stride_bk, stride_bn),
                offsets=(0, pid_n * BLOCK_SIZE_N),
                block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
                order=((0, 1) if IS_8BIT else (1, 0)),
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
                    FAST,
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
        S_safe,
        NUM_EXPERTS: tl.constexpr,
        QUANT_TYPE: tl.constexpr,
        FAST: tl.constexpr,
    ):
        BLOCK_SIZE_M: tl.constexpr = 16
        BLOCK_SIZE_K: tl.constexpr = 128
        IS_8BIT: tl.constexpr = (
            QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128
            or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
        )
        BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else 16

        if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 and FAST is False:
            is_safe = tl.load(S_safe + NUM_EXPERTS)
            if is_safe:
                return

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
                (K if IS_8BIT else tl.cdiv(K, 128) * BLOCK_SIZE_K_WEIGHT),
                2 * N,
            ),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
            order=((0, 1) if IS_8BIT else (1, 0)),
        )
        b_up_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(
                (K if IS_8BIT else tl.cdiv(K, 128) * BLOCK_SIZE_K_WEIGHT),
                2 * N,
            ),
            strides=(stride_bk, stride_bn),
            offsets=(0, N + pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
            order=((0, 1) if IS_8BIT else (1, 0)),
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
                FAST,
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
                FAST,
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
    ):
        """Gather expert-sorted rows into a contiguous AIU input matrix."""
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
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
        tl.store(
            routed_a_ptr + offs_m[:, None] * K + offs_k[None, :],
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
    ):
        """Fuse SwiGLU with the expert-sorted layout required by GEMM2.

        A separate SiLU call followed by ``_stage_routed_activations_kernel``
        would add one launch and a full intermediate write/read cycle.
        """
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        offs_m = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
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
        tl.store(
            routed_intermediate2_ptr + offs_m[:, None] * N + offs_n[None, :],
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
        S_safe,
        QUANT_TYPE: tl.constexpr,
        FAST: tl.constexpr,
    ):
        """Expert-grouped quantized GEMM using contiguous TLE AIU transfers."""
        IS_8BIT: tl.constexpr = (
            QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128
            or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
        )
        BLOCK_SIZE_K_WEIGHT: tl.constexpr = (
            BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 8
        )
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid = tl.program_id(0)
        if GROUP_SIZE_M == 1:
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

        expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 and FAST is False:
            is_safe = tl.load(S_safe + expert)
            if is_safe:
                return

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
                (K if IS_8BIT else tl.cdiv(K, 128) * BLOCK_SIZE_K_WEIGHT),
                N,
            ),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
            order=((0, 1) if IS_8BIT else (1, 0)),
        )
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
            FAST,
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


def _align_grouped_tokens(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    # Supply Triton-zeroed workspace to keep alignment independent of Torch kernels.
    workspace = (
        zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device),
        zeros(
            (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
        ),
    )
    return moe_align_block_size_no_tle(
        topk_ids, block_m, num_experts, workspace=workspace
    )


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
    silu_output: Optional[torch.Tensor] = None,
    split_k: int = 1,
):
    if tle_async is None:
        raise RuntimeError("PPU fused Marlin MoE requires Triton TLE")

    B, B_scale = stage.weight, stage.scale
    K = A.size(1)
    N = C.size(-1) if fuse_silu else B.size(2)
    routes = topk_ids.numel()
    stride_cm, stride_cn = C.stride(-2), C.stride(-1)
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
    # Packed FP8 decode favors more long-K tiles and wider short-K tiles.
    # The PPU pipeline pass allocates ``num_stages - 1`` loop buffers.  Three
    # scheduling stages therefore provide the two buffers required to overlap
    # the next AIU copy with the current tile's unpack/dequantize/dot work.
    pipeline_stages = 3 if K > 128 else 1
    grid = (routes * triton.cdiv(N, block_n),)

    fast_variants = (
        (None,)
        if quant_type_id == QUANT_TYPE_FP8_E4M3
        else _PPU_FAST_VARIANTS[quant_type_id]
    )
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
        assert use_matvec and silu_output is not None
        partials = torch.empty(
            (split_k, routes, N), device=C.device, dtype=torch.float32
        )
        target = partials
        grid = (*grid, split_k)
        pipeline_stages = min(
            pipeline_stages, triton.cdiv(triton.cdiv(K, 128), split_k)
        )
    extra = dict(SPLIT_K=split_k, SPLIT_STRIDE=routes * N) if use_matvec else {}
    for fast_variant in fast_variants:
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
            stride_cm,
            stride_cn,
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
            S_safe=stage.scale_safe,
            NUM_EXPERTS=B.size(0),
            QUANT_TYPE=quant_type_id,
            FAST=fast_variant,
            **extra,
        )

    if split_k > 1:
        _reduce_split_silu_kernel[(routes, triton.cdiv(N // 2, 128))](
            partials,
            silu_output,
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

    fast_variants = (
        (None,)
        if quant_type_id == QUANT_TYPE_FP8_E4M3
        else _PPU_FAST_VARIANTS[quant_type_id]
    )
    for fast_variant in fast_variants:
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
            S_safe=stage.scale_safe,
            NUM_EXPERTS=B.size(0),
            QUANT_TYPE=quant_type_id,
            FAST=fast_variant,
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
) -> torch.Tensor:
    routed_a = torch.empty((em, A.size(1)), dtype=A.dtype, device=A.device)
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
) -> torch.Tensor:
    n = intermediate1.size(1) // 2
    routed = torch.empty(
        (em, n), dtype=intermediate1.dtype, device=intermediate1.device
    )
    grid = (triton.cdiv(em, block_m), triton.cdiv(n, 128))
    _silu_and_stage_routed_kernel[grid](
        intermediate1,
        routed,
        sorted_token_ids,
        num_tokens_post_padded,
        EM=em,
        N=n,
        num_valid_tokens=num_valid_tokens,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=128,
        num_warps=4,
        num_stages=1,
    )
    return routed


def _invoke_marlin_moe_gemm_grouped(
    A: torch.Tensor,
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
    input_is_routed: bool = False,
):
    if tle_async is None:
        raise RuntimeError("PPU fused Marlin MoE requires Triton TLE")
    B, B_scale = stage.weight, stage.scale
    em = sorted_token_ids.size(0)
    num_valid_tokens = C.size(0) * C.size(1) if C.ndim == 3 else A.size(0) * top_k
    if input_is_routed:
        routed_a = A
    else:
        routed_a = _stage_grouped_activations(
            A,
            sorted_token_ids,
            num_tokens_post_padded,
            em=em,
            num_valid_tokens=num_valid_tokens,
            top_k=top_k,
            block_m=block_m,
        )

    n = B.size(2)
    batch_m = C.size(0) if C.ndim == 3 else A.size(0)
    block_n, group_m, num_warps, pipeline_stages = _select_grouped_config(
        batch_m, A.size(1), n, block_m, quant_type_id
    )
    # A 64-K tile bounds decoded-weight state; FP8 can then use a wider down tile.
    if quant_type_id == QUANT_TYPE_FP8_E4M3 and group_size == 128 and block_m <= 32:
        block_n = 128
    stride_cm, stride_cn = C.stride(-2), C.stride(-1)
    grid = (triton.cdiv(em, block_m) * triton.cdiv(n, block_n),)
    fast_variants = _PPU_FAST_VARIANTS[quant_type_id]
    for fast_variant in fast_variants:
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
            K=A.size(1),
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
                if quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3)
                and group_size == 128
                else 128
            ),
            GROUP_SIZE_M=group_m,
            GROUP_SIZE_K=group_size,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            PIPELINE_STAGES=pipeline_stages,
            compute_type=compute_type,
            num_warps=num_warps,
            num_stages=pipeline_stages,
            S_safe=stage.scale_safe,
            QUANT_TYPE=quant_type_id,
            FAST=fast_variant,
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
    # All GEMM1 reads finish before writing the output. Aliasing activations
    # is safe, but weights/scales/routing remain live during GEMM2.
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


def _prepare_quantized_weights(
    w1,
    w2,
    w1_scale,
    w2_scale,
    quant_type_id,
    group_size,
    *,
    direct=False,
    compact_safety=False,
):
    if quant_type_id == QUANT_TYPE_UINT4B8:
        b1, b2, s1, s2 = w4a16_int4_pack(
            w1, w2, w1_scale, w2_scale, cached=True, block_size_k=group_size
        )
    elif quant_type_id == QUANT_TYPE_FP4_E2M1:
        b1, b2, _, _ = w4a16_int4_pack(w1, w2, cached=True, block_size_k=128)
        s1, s2 = _pack_e8m0(w1_scale), _pack_e8m0(w2_scale)
    else:
        b1 = w1.view(torch.uint8).transpose(1, 2)
        b2 = w2.view(torch.uint8).transpose(1, 2)
        s1 = w1_scale.transpose(1, 2)
        s2 = w2_scale.transpose(1, 2)
    if quant_type_id == QUANT_TYPE_FP8_E4M3 and not direct:
        safe1, safe2 = _fp8_scale_safety(w1_scale, compact_safety), _fp8_scale_safety(
            w2_scale, compact_safety
        )
    else:
        safe1, safe2 = None, None
    return _QuantizedStage(b1, s1, safe1), _QuantizedStage(b2, s2, safe2)


@triton.jit
def decode_bank(
    Q,
    S,
    D,
    Safe,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    QE: tl.constexpr,
    QN: tl.constexpr,
    QK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    FP8: tl.constexpr,
    FAST: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    if FP8 and not FAST:
        if tl.load(Safe + E):
            return
    expert = tl.program_id(2).to(tl.int64)
    if FP8 and not FAST:
        if tl.load(Safe + expert):
            return
    ns = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    k_base = (tl.program_id(1) * BK).to(tl.int64)
    ks = (k_base + tl.arange(0, BK)).to(tl.int64)
    mask = (ns[:, None] < N) & (ks[None, :] < K)
    q = tl.load(
        Q + expert * QE + ns[:, None] * QN + ks[None, :] * QK,
        mask,
        other=0 if FP8 else 128,
    )
    scale = tl.load(S + expert * SE + ns * SN + (k_base // 128) * SG, ns < N, other=0)[
        :, None
    ]
    if FP8:
        if FAST:
            value = decode_fp8_scaled_bf16(q, scale)
        else:
            value = _decode_e4m3(q.to(tl.uint32), scale, tl.bfloat16, False)
    else:
        value = decode_int8_scaled_bf16(q, scale)
    tl.store(D + expert * N * K + ns[:, None] + ks[None, :] * N, value, mask)


def run_dense_w8a16(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    scale_safe1: Optional[torch.Tensor],
    scale_safe2: Optional[torch.Tensor],
    is_fp8: bool,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
        get_default_config,
        invoke_fused_moe_triton_kernel,
    )

    m, k = hidden_states.shape
    e, n2, _ = w1.shape
    n, topk = n2 // 2, topk_ids.shape[1]
    tile = _PPU_DENSE_GATE_TILE
    down_tile = _PPU_DENSE_DOWN_TILE
    scratch = torch.empty(
        (e * max(min(n2, tile) * k, min(k, down_tile) * n),),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    # Reuse projection storage after every gate/up value has fed SiLU.
    cache = torch.empty(
        (m * topk * max(n2, min(k, down_tile)),),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    gateup = cache[: m * topk * n2].view(m, topk, n2)
    intermediate = torch.empty(
        (m * topk, n), device=hidden_states.device, dtype=hidden_states.dtype
    )
    down = cache[: m * topk * min(k, down_tile)].view(m, topk, min(k, down_tile))
    sorted_ids, experts, padded = _align_grouped_tokens(topk_ids, 64, e)
    for index, (weight, scales, safe) in enumerate(
        ((w1, w1_scale, scale_safe1), (w2, w2_scale, scale_safe2))
    ):
        first = index == 0
        output_size, reduction = weight.shape[1:]
        config = get_default_config(
            max(m, _PPU_DENSE_MIN_TOKENS),
            e,
            output_size,
            reduction,
            topk,
            "bf16",
            gemm_stage="gemm1" if first else "gemm2",
        )
        tile = _PPU_DENSE_GATE_TILE if first else down_tile
        for col in range(0, output_size, tile):
            stop = min(col + tile, output_size)
            width = stop - col
            qw, ss = weight[:, col:stop], scales[:, col:stop]
            # Decode only the output columns consumed by this GEMM launch.
            for fast in (True, False) if is_fp8 else (True,):
                decode_bank[(triton.cdiv(width, 64), triton.cdiv(reduction, 128), e)](
                    qw.view(torch.uint8),
                    ss,
                    scratch,
                    safe,
                    width,
                    reduction,
                    e,
                    *qw.stride(),
                    *ss.stride(),
                    is_fp8,
                    fast,
                    64,
                    128,
                    num_warps=4,
                    num_stages=1,
                )
            decoded = (
                scratch[: e * reduction * width]
                .view(e, reduction, width)
                .transpose(1, 2)
            )
            projected = gateup[:, :, col:stop] if first else down[:, :, :width]
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
            if not first:
                moe_sum(projected, output[:, col:stop])
        if first:
            flat = gateup.view(m * topk, n2)
            silu_and_mul_out(flat[:, :n], flat[:, n:], intermediate)
    return output


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

    INT8/FP8 weights and scales are read directly on every invocation.
    Four-bit weights retain the existing cached packing behavior. MXFP4 scale
    conversions are cached only when the input has a version counter.
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
        direct = _use_direct_route(m, e, topk, k, n)
        if packed_fp8 and e > 8 and m * topk <= 160 and max(k, 2 * n) <= 4096:
            direct = True
        stage1, stage2 = _prepare_quantized_weights(
            w1,
            w2,
            w1_scale,
            w2_scale,
            quant_type_id,
            group_size,
            direct=direct,
            compact_safety=packed_fp8 and m < _PPU_DENSE_MIN_TOKENS,
        )
        if (
            m >= _PPU_DENSE_MIN_TOKENS
            and m * topk > 32 * e
            and group_size == 128
            and hidden_states.dtype == torch.bfloat16
            and w1_scale.dtype == w2_scale.dtype == torch.bfloat16
            and quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3)
            and max(w1.numel(), w2.numel()) * 2 <= _PPU_DENSE_MAX_DECODE_BYTES
        ):
            decode_bytes = (
                e
                * max(
                    min(2 * n, _PPU_DENSE_GATE_TILE) * k,
                    min(k, _PPU_DENSE_DOWN_TILE) * n,
                )
                * 2
            )
            fixed_bytes = decode_bytes + 4 * (e + 1) ** 2 + 16 * 63 * e + 8
            if quant_type_id == QUANT_TYPE_FP8_E4M3:
                fixed_bytes += 8 * (e + 1) + 4 * e * max(
                    triton.cdiv(2 * n * (k // group_size), 256),
                    triton.cdiv(k * (n // group_size), 256),
                )
            # Column tiles share projection storage; the activation survives both GEMMs.
            bytes_per_token = topk * (
                2 * (max(2 * n, min(k, _PPU_DENSE_DOWN_TILE)) + n) + 12
            )
            batch_tokens = min(
                m, (_PPU_DENSE_MAX_WORKSPACE_BYTES - fixed_bytes) // bytes_per_token
            )
            if batch_tokens < 1:
                raise NotImplementedError("PPU dense MoE workspace exceeds its budget")
            for start in range(0, m, batch_tokens):
                end = min(start + batch_tokens, m)
                run_dense_w8a16(
                    hidden_states[start:end],
                    w1,
                    w2,
                    w1_scale,
                    w2_scale,
                    topk_weights[start:end],
                    out[start:end],
                    topk_ids[start:end],
                    stage1.scale_safe,
                    stage2.scale_safe,
                    quant_type_id == QUANT_TYPE_FP8_E4M3,
                    apply_router_weight_on_input,
                )
            return out
        split_silu = direct and (
            n >= 1024 or quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3)
        )
        use_matvec = direct and (
            packed_fp8
            or (m == 1 and quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3))
        )
        gemv_split_k = 1
        if (
            packed_fp8
            and direct
            and group_size == 128
            and k >= 1024
            and k % 1024 == 0
            and 2 * n <= 1024
        ):
            routes = m * topk
            # Fill the compute units for very sparse routes; the 33--64 route
            # band already has enough column tiles without another reduction.
            gemv_split_k = (
                min(8, 64 // triton.next_power_of_2(routes)) if routes <= 64 else 4
            )
        reduced = direct and m <= 2 and not split_silu
        stage1_topk_weights = topk_weights if apply_router_weight_on_input else None
        stage2_topk_weights = None if apply_router_weight_on_input else topk_weights
        c2 = (
            torch.empty((m * topk, n), device=out.device, dtype=out.dtype)
            if direct
            else None
        )
        c1 = (
            torch.empty((m * topk, 2 * n), device=out.device, dtype=out.dtype)
            if split_silu
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
                    C=c1,
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
                    silu_output=c2 if gemv_split_k > 1 else None,
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
            if reduced:
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
                )
        else:
            bm = _select_grouped_block_m(m, e, topk)
            sorted_ids, experts, padded = _align_grouped_tokens(topk_ids, bm, e)
            c1 = torch.empty((m * topk, 2 * n), device=out.device, dtype=out.dtype)
            _invoke_marlin_moe_gemm_grouped(
                A=hidden_states,
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
            )
            routed_c2 = _silu_and_stage_grouped(
                c1,
                sorted_ids,
                padded,
                em=sorted_ids.numel(),
                num_valid_tokens=m * topk,
                block_m=bm,
            )
            _invoke_marlin_moe_gemm_grouped(
                A=routed_c2,
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
                input_is_routed=True,
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
    if quant_type_id not in _PPU_FAST_VARIANTS:
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
