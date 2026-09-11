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

The public operator covers W4A16 INT4, W4A16 MXFP4 and W8A16 FP8. All three
formats share the same routing, staging and GEMM kernels; weight decoding is
selected at compile time from ``quant_type_id``.

The grouped path consumes the same sorted routing layout as ``fused_moe``.
Its GEMM stays separate because Marlin weights are output-major INT32 tiles
that must be asynchronously loaded and decoded before each dot product.
"""

from typing import Any, Callable, NamedTuple, Optional

import torch
import triton
import triton.language as tl
from torch.utils.weak import WeakTensorKeyDictionary

try:
    from triton.experimental.tle import language as tle_async
except ImportError:
    tle_async = None

from flaggems_vllm.ops.fused_marlin_moe import (
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
    QUANT_TYPE_UINT4B8,
    _stack_8,
)
from flaggems_vllm.ops.fused_marlin_moe import (
    fused_marlin_moe as _generic_fused_marlin_moe,
)
from flaggems_vllm.ops.fused_marlin_moe import w4a16_int4_pack
from flaggems_vllm.ops.moe_sum import moe_sum
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.utils import libentry

_PPU_DIRECT_ROUTE_LIMIT = 32
_PACK_CACHE = WeakTensorKeyDictionary()
_SCALE_CACHE = WeakTensorKeyDictionary()
_FP8_SCALE_CACHE = WeakTensorKeyDictionary()
_PPU_FAST_VARIANTS = {
    QUANT_TYPE_UINT4B8: (False,),
    QUANT_TYPE_FP4_E2M1: (False,),
    QUANT_TYPE_FP8_E4M3: (True, False),
}
_TL_QUANT_TYPE_UINT4B8 = tl.constexpr(QUANT_TYPE_UINT4B8)
_TL_QUANT_TYPE_FP4_E2M1 = tl.constexpr(QUANT_TYPE_FP4_E2M1)
_TL_QUANT_TYPE_FP8_E4M3 = tl.constexpr(QUANT_TYPE_FP8_E4M3)


class _PackedStage(NamedTuple):
    weight: torch.Tensor
    scale: torch.Tensor
    weight_safe: torch.Tensor
    scale_safe: torch.Tensor


@triton.jit
def _decode_e2m1(q, scale, compute_type: tl.constexpr):
    bits = ((q & 8) << 12) | ((q & 7) << 9)
    tiny = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return (tiny * 16384.0 * scale).to(compute_type)


@triton.jit
def _decode_e4m3(q, scale, compute_type: tl.constexpr, FAST: tl.constexpr = False):
    if compute_type == tl.bfloat16:
        # Preserve E4M3 subnormals by embedding the bits in BF16. Its exponent
        # bias is corrected by an exact power-of-two factor in the scale.
        bits = ((q & 128) << 8) | ((q & 127) << 4)
        tiny = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        exponent_bias = 2.0**120
    else:
        # E4M3 bits embedded in FP16 have exponent bias 15 rather than 7.
        bits = ((q & 128) << 8) | ((q & 127) << 7)
        tiny = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        exponent_bias = 256.0
    value = tiny
    if not FAST:
        value = value * exponent_bias
        value = tl.where((q & 127) == 127, float("nan"), value)
    return (value * scale).to(compute_type)


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
    tl.static_assert(b.shape[0] == 32, "FP8 packing uses 128-K tiles")
    parts = tl.arange(0, 4)
    quant = (b[None, :, :] >> (parts[:, None, None] * 8)) & 255
    quant = tl.reshape(quant, (128, ns.shape[0]))
    if GROUP_SIZE == -1 or GROUP_SIZE >= 128:
        group = 0 if GROUP_SIZE == -1 else k_base // GROUP_SIZE
        scale_ptrs = s_ptr + expert * se + group * sg + ns * sn
        if N % ns.shape[0] == 0:
            scale = tl.load(scale_ptrs)[None, :]
        else:
            scale = tl.load(scale_ptrs, mask=ns < N, other=1.0)[None, :]
        if FAST:
            scale = scale * (2.0**120 if compute_type == tl.bfloat16 else 256.0)
    else:
        groups = k_base // GROUP_SIZE + tl.arange(0, 128 // GROUP_SIZE)
        scale_ptrs = (
            s_ptr + expert * se + groups[:, None] * sg + ns[None, :] * sn
        )
        if N % ns.shape[0] == 0 and K % 128 == 0:
            scale = tl.load(scale_ptrs)
        else:
            scale = tl.load(
                scale_ptrs,
                mask=(ns[None, :] < N) & (groups[:, None] * GROUP_SIZE < K),
                other=1.0,
            )
        if FAST:
            scale = scale * (2.0**120 if compute_type == tl.bfloat16 else 256.0)
        scale = tl.broadcast_to(
            scale[:, None, :], (128 // GROUP_SIZE, GROUP_SIZE, ns.shape[0])
        )
        scale = tl.reshape(scale, (128, ns.shape[0]))
    decoded = _decode_e4m3(quant, scale, compute_type, FAST)
    if K % 128 == 0:
        return decoded
    # Channel scales can be nonfinite; padded bytes must never introduce NaN.
    return tl.where((k_base + tl.arange(0, 128))[:, None] < K, decoded, 0.0).to(
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
        version = None
    cached = _SCALE_CACHE.get(s)
    if cached is not None and cached[0] == version:
        return cached[1]
    e, n, g = s.shape
    out = torch.empty((e, g, n), device=s.device, dtype=torch.float32)
    _pack_e8m0_kernel[(triton.cdiv(n * g, 256), e)](
        s.view(torch.uint8), out, n, g, *s.stride(), 256
    )
    if not torch.cuda.is_current_stream_capturing():
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
def _pack_fp8_kernel(
    W,
    P,
    Flags,
    N: tl.constexpr,
    K: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SK: tl.constexpr,
    KP: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = tl.program_id(1)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n, pk = idx % N, idx // N
    base = (pk // 32) * 128 + pk % 32
    packed = tl.full((BLOCK,), 0, tl.uint32)
    bad = tl.full((BLOCK,), 0, tl.int32)
    for i in tl.static_range(4):
        k = base + i * 32
        valid = (pk < KP) & (k < K)
        byte = tl.load(W + e * SE + n * SN + k * SK, mask=valid, other=0).to(tl.uint32)
        packed |= byte << (i * 8)
        bad |= ((byte & 127) == 127) & valid
    tl.store(P + e * KP * N + idx, packed, mask=pk < KP)
    tl.store(Flags + e * CHUNKS + tl.program_id(0), tl.sum(bad, 0) != 0)


def _pack_fp8_cache(w):
    try:
        version = w._version
    except RuntimeError:
        version = None
    cached = _PACK_CACHE.get(w)
    if cached is not None and cached[0] == version:
        return cached[1]
    e, n, k = w.shape
    kp = triton.cdiv(k, 128) * 32
    count = triton.cdiv(kp * n, 256)
    packed = torch.empty((e, kp, n), device=w.device, dtype=torch.int32)
    chunks = torch.empty((e, count), device=w.device, dtype=torch.int32)
    safe = torch.empty((e + 1,), device=w.device, dtype=torch.int32)
    _pack_fp8_kernel[(count, e)](
        w.view(torch.uint8), packed, chunks, n, k, *w.stride(), kp, count, 256
    )
    _reduce_safety_kernel[(e + 1,)](chunks, safe, e, count, 512)
    result = (packed, safe)
    if not torch.cuda.is_current_stream_capturing():
        _PACK_CACHE[w] = (version, result)
    return result


@triton.jit
def _pack_fp8_scale_kernel(
    S,
    Out,
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
    n, g = i % N, i // N
    valid = i < G * N
    scale = tl.load(S + e * SE + n * SN + g * SG, mask=valid, other=1.0).to(tl.float32)
    folded = scale * (2.0**120)
    # NaN/Inf scales remain semantically valid on the fast path. Use the
    # stronger BF16 exponent correction when deciding whether folding is safe.
    bad = (tl.abs(folded) == float("inf")) & (tl.abs(scale) != float("inf")) & valid
    tl.store(Out + e * G * N + i, scale, mask=valid)
    tl.store(Flags + e * CHUNKS + tl.program_id(0), tl.sum(bad.to(tl.int32), 0) != 0)


def _pack_fp8_scale_cache(s, input_size=None):
    try:
        version = s._version
    except RuntimeError:
        version = None
    cache_key = (version, input_size)
    cached = _FP8_SCALE_CACHE.get(s)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    e, n, g = s.shape
    count = triton.cdiv(n * g, 256)
    out_dtype = (
        s.dtype if input_size is not None and input_size > n else torch.float32
    )
    out = torch.empty((e, g, n), device=s.device, dtype=out_dtype)
    chunks = torch.empty((e, count), device=s.device, dtype=torch.int32)
    safe = torch.empty((e + 1,), device=s.device, dtype=torch.int32)
    _pack_fp8_scale_kernel[(count, e)](s, out, chunks, n, g, *s.stride(), count, 256)
    _reduce_safety_kernel[(e + 1,)](chunks, safe, e, count, 512)
    result = (out, safe)
    if not torch.cuda.is_current_stream_capturing():
        _FP8_SCALE_CACHE[s] = (cache_key, result)
    return result


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
    ):
        """Load, decode and multiply one stream of 128-K weight tiles."""
        BLOCK_SIZE_K: tl.constexpr = 128
        BLOCK_SIZE_K_PACK: tl.constexpr = (
            32 if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 else 16
        )
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k_tile in tl.range(
            0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES
        ):
            activation = tle_async.load(
                a_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            if K % BLOCK_SIZE_K == 0 and N % BLOCK_SIZE_N == 0:
                b_packed = tle_async.load(b_block_ptr, is_async=True)
            else:
                b_packed = tle_async.load(
                    b_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            weight = _dequant_weight(
                b_packed,
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
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K_PACK, 0))
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
        W_safe,
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
        BLOCK_SIZE_K_PACK: tl.constexpr = (
            32 if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 else 16
        )

        if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 and not FAST:
            is_safe = tl.load(W_safe + NUM_EXPERTS) & tl.load(S_safe + NUM_EXPERTS)
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
        # Packed B stores either four FP8 or eight 4-bit values per int32.
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(tl.cdiv(K, 128) * BLOCK_SIZE_K_PACK, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_PACK, BLOCK_SIZE_N),
            order=(1, 0),
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
        W_safe,
        S_safe,
        NUM_EXPERTS: tl.constexpr,
        QUANT_TYPE: tl.constexpr,
        FAST: tl.constexpr,
    ):
        BLOCK_SIZE_K: tl.constexpr = 128
        BLOCK_SIZE_K_PACK: tl.constexpr = (
            32 if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 else 16
        )

        if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 and not FAST:
            is_safe = tl.load(W_safe + NUM_EXPERTS) & tl.load(S_safe + NUM_EXPERTS)
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
            shape=(tl.cdiv(K, BLOCK_SIZE_K) * BLOCK_SIZE_K_PACK, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_PACK, BLOCK_SIZE_N),
            order=(1, 0),
        )
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

        for k_tile in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES):
            a = tl.load(
                a_ptr
                + a_row * stride_am
                + (k_tile * BLOCK_SIZE_K + offs_k) * stride_ak,
                mask=k_tile * BLOCK_SIZE_K + offs_k < K,
                other=0.0,
            )
            if K % BLOCK_SIZE_K == 0 and N % BLOCK_SIZE_N == 0:
                b_packed = tle_async.load(b_block_ptr, is_async=True)
            else:
                b_packed = tle_async.load(
                    b_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            b = _dequant_weight(
                b_packed,
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
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K_PACK, 0))

        if MUL_ROUTED_WEIGHT:
            acc *= tl.load(topk_weights_ptr + route).to(tl.float32)
        tl.store(
            c_ptr + route * stride_cm + offs_n * stride_cn,
            acc.to(compute_type),
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
        W_safe,
        S_safe,
        NUM_EXPERTS: tl.constexpr,
        QUANT_TYPE: tl.constexpr,
        FAST: tl.constexpr,
    ):
        """Compute GEMM2 and reduce all routed experts into one token row."""
        BLOCK_SIZE_M: tl.constexpr = 16
        BLOCK_SIZE_K: tl.constexpr = 128
        BLOCK_SIZE_K_PACK: tl.constexpr = (
            32 if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 else 16
        )

        if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 and not FAST:
            is_safe = tl.load(W_safe + NUM_EXPERTS) & tl.load(S_safe + NUM_EXPERTS)
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
                shape=(tl.cdiv(K, 128) * BLOCK_SIZE_K_PACK, N),
                strides=(stride_bk, stride_bn),
                offsets=(0, pid_n * BLOCK_SIZE_N),
                block_shape=(BLOCK_SIZE_K_PACK, BLOCK_SIZE_N),
                order=(1, 0),
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
                b_packed = tle_async.load(
                    b_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )

                offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                b = _dequant_weight(
                    b_packed,
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
                b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K_PACK, 0))

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
        W_safe,
        S_safe,
        NUM_EXPERTS: tl.constexpr,
        QUANT_TYPE: tl.constexpr,
        FAST: tl.constexpr,
    ):
        BLOCK_SIZE_M: tl.constexpr = 16
        BLOCK_SIZE_K: tl.constexpr = 128
        BLOCK_SIZE_K_PACK: tl.constexpr = (
            32 if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 else 16
        )

        if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 and not FAST:
            is_safe = tl.load(W_safe + NUM_EXPERTS) & tl.load(S_safe + NUM_EXPERTS)
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
            shape=(tl.cdiv(K, 128) * BLOCK_SIZE_K_PACK, 2 * N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_PACK, BLOCK_SIZE_N),
            order=(1, 0),
        )
        b_up_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(tl.cdiv(K, 128) * BLOCK_SIZE_K_PACK, 2 * N),
            strides=(stride_bk, stride_bn),
            offsets=(0, N + pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_PACK, BLOCK_SIZE_N),
            order=(1, 0),
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
            b_gate_packed = tle_async.load(
                b_gate_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            b_up_packed = tle_async.load(
                b_up_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )

            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            gate_b = _dequant_weight(
                b_gate_packed,
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
                b_up_packed,
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
            b_gate_block_ptr = tl.advance(b_gate_block_ptr, (BLOCK_SIZE_K_PACK, 0))
            b_up_block_ptr = tl.advance(b_up_block_ptr, (BLOCK_SIZE_K_PACK, 0))

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
        W_safe,
        S_safe,
        QUANT_TYPE: tl.constexpr,
        FAST: tl.constexpr,
    ):
        """Expert-grouped quantized GEMM using contiguous TLE AIU transfers."""
        BLOCK_SIZE_K_PACK: tl.constexpr = (
            BLOCK_SIZE_K // 4
            if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
            else BLOCK_SIZE_K // 8
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
        if QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3 and not FAST:
            is_safe = tl.load(W_safe + expert) & tl.load(S_safe + expert)
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
            shape=(tl.cdiv(K, 128) * BLOCK_SIZE_K_PACK, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_PACK, BLOCK_SIZE_N),
            order=(1, 0),
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
        )

        if MUL_ROUTED_WEIGHT:
            routed_weight = tl.load(
                topk_weights_ptr + routed_token,
                mask=token_mask,
                other=0.0,
            )
            acc *= routed_weight[None, :]

        c_ptrs = (
            c_ptr + routed_token[None, :] * stride_cm + offs_n[:, None] * stride_cn
        )
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


def _select_grouped_config(
    M: int, K: int, N: int, block_m: int, quant_type_id: int
):
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


def _align_grouped_tokens(
    topk_ids: torch.Tensor,
    block_m: int,
    num_experts: int,
):
    # FlagGems-vLLM is integrated into vLLM, and the T-Head CUDA extension's
    # alignment kernel avoids the high fixed cost of the generic Triton path.
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size as vllm_moe_align_block_size,
    )

    return vllm_moe_align_block_size(
        topk_ids,
        block_m,
        num_experts,
        expert_map=None,
        ignore_invalid_experts=True,
    )


def _invoke_marlin_moe_gemm_direct(
    A: torch.Tensor,
    C: torch.Tensor,
    packed: _PackedStage,
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
):
    if tle_async is None:
        raise RuntimeError("PPU fused Marlin MoE requires Triton TLE")

    B, B_scale = packed.weight, packed.scale
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
    # The PPU pipeline pass allocates ``num_stages - 1`` loop buffers.  Three
    # scheduling stages therefore provide the two buffers required to overlap
    # the next AIU copy with the current tile's unpack/dequantize/dot work.
    pipeline_stages = 3 if K > 128 else 1
    grid = (routes * triton.cdiv(N, block_n),)

    fast_variants = _PPU_FAST_VARIANTS[quant_type_id]
    kernel = (
        _marlin_moe_gemm_silu_direct_kernel
        if fuse_silu
        else (
            _marlin_moe_matvec_direct_kernel
            if use_matvec
            else _marlin_moe_gemm_direct_kernel
        )
    )
    for fast_variant in fast_variants:
        kernel[grid](
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
            W_safe=packed.weight_safe,
            S_safe=packed.scale_safe,
            NUM_EXPERTS=B.size(0),
            QUANT_TYPE=quant_type_id,
            FAST=fast_variant,
        )


def _invoke_marlin_moe_gemm_reduce_direct(
    A: torch.Tensor,
    C: torch.Tensor,
    packed: _PackedStage,
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

    B, B_scale = packed.weight, packed.scale
    K = A.size(1)
    N = B.size(2)
    top_k = topk_ids.size(1)
    tokens = A.size(0) // top_k
    block_n = (64 if N >= 1024 else 128) if tokens == 1 else 32
    pipeline_stages = 3 if K > 128 else 1
    grid = (tokens * triton.cdiv(N, block_n),)

    fast_variants = _PPU_FAST_VARIANTS[quant_type_id]
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
            W_safe=packed.weight_safe,
            S_safe=packed.scale_safe,
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
    packed: _PackedStage,
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
    B, B_scale = packed.weight, packed.scale
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
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=group_m,
            GROUP_SIZE_K=group_size,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            PIPELINE_STAGES=pipeline_stages,
            compute_type=compute_type,
            num_warps=num_warps,
            num_stages=pipeline_stages,
            W_safe=packed.weight_safe,
            S_safe=packed.scale_safe,
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
    if quant_type_id == QUANT_TYPE_FP8_E4M3:
        if w1.shape != (e, 2 * n, k) or w2.shape != (e, k, n):
            raise ValueError("FP8 weight shapes do not match activations")
        if group_size not in (-1, 32, 64, 128):
            raise NotImplementedError("FP8 group_size must be -1, 32, 64 or 128")
        if group_size != -1 and (k % group_size or n % group_size):
            raise ValueError("Input dimensions must be divisible by group_size")
        g1 = 1 if group_size == -1 else k // group_size
        g2 = 1 if group_size == -1 else n // group_size
        if s1.shape != (e, 2 * n, g1) or s2.shape != (e, k, g2):
            raise ValueError("FP8 scale shapes must match the selected group size")
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
        if k % group_size or n % group_size:
            raise ValueError("Input dimensions must be divisible by group_size")
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


def _pack_quantized_weights(w1, w2, w1_scale, w2_scale, quant_type_id, group_size):
    if quant_type_id == QUANT_TYPE_UINT4B8:
        b1, b2, s1, s2 = w4a16_int4_pack(
            w1,
            w2,
            w1_scale,
            w2_scale,
            cached=True,
            block_size_k=group_size,
        )
        safety = (b1, s1, b2, s2)
    elif quant_type_id == QUANT_TYPE_FP4_E2M1:
        b1, b2, _, _ = w4a16_int4_pack(
            w1,
            w2,
            cached=True,
            block_size_k=128,
        )
        s1, s2 = _pack_e8m0(w1_scale), _pack_e8m0(w2_scale)
        safety = (b1, s1, b2, s2)
    else:
        b1, b1_safe = _pack_fp8_cache(w1)
        b2, b2_safe = _pack_fp8_cache(w2)
        s1, s1_safe = _pack_fp8_scale_cache(w1_scale, w1.size(2))
        s2, s2_safe = _pack_fp8_scale_cache(w2_scale, w2.size(2))
        safety = (b1_safe, s1_safe, b2_safe, s2_safe)
    return (
        _PackedStage(b1, s1, safety[0], safety[1]),
        _PackedStage(b2, s2, safety[2], safety[3]),
    )


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

    Weight tensors created in torch.inference_mode are immutable after their
    first invocation. Warm up outside CUDA Graph capture to cache packing.
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
        stage1, stage2 = _pack_quantized_weights(
            w1, w2, w1_scale, w2_scale, quant_type_id, group_size
        )
        direct = _use_direct_route(m, e, topk, k, n)
        split_silu = direct and n >= 1024
        use_matvec = direct and m == 1 and quant_type_id == QUANT_TYPE_FP8_E4M3
        reduced = direct and m <= 2 and not split_silu
        stage1_topk_weights = topk_weights if apply_router_weight_on_input else None
        stage2_topk_weights = None if apply_router_weight_on_input else topk_weights
        c2 = torch.empty((m * topk, n), device=out.device, dtype=out.dtype)
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
                    packed=stage1,
                    topk_weights=stage1_topk_weights,
                    topk_ids=topk_ids,
                    mul_routed_weight=apply_router_weight_on_input,
                    a_route_divisor=topk,
                    group_size=group_size,
                    compute_type=compute_type,
                    quant_type_id=quant_type_id,
                    use_matvec=use_matvec,
                )
                silu_and_mul_out(c1[:, :n], c1[:, n:], c2)
            else:
                _invoke_marlin_moe_gemm_direct(
                    A=hidden_states,
                    C=c2,
                    packed=stage1,
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
                    packed=stage2,
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
                    packed=stage2,
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
                packed=stage1,
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
                packed=stage2,
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
    """PPU override for INT4, MXFP4 and FP8 fused Marlin MoE."""
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
