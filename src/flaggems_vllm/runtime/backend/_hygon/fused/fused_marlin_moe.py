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

"""Hygon SwiGLU MoE using output-major quantized weights (not Marlin repack).

Quantization tags (``quant_type_id``) are expert-major tags, not packed
Marlin IDs:

    0  uint4b8    GPTQ INT4, stored as w + 8, dequant subtracts 8
    1  uint8b128  GPTQ INT8, stored as w + 128, dequant subtracts 128
    2  FP8 E4M3FN, finite-only 4-bit exponent / 3-bit mantissa
    6  FP4 E2M1 (MXFP4), one E8M0 scale byte per 32 weights

Public weight layout -- plain row-major ``(out_features, in_features)``,
NOT the vLLM Marlin repack (no int32 tile + mma-fragment weight permutation,
no ``scale_perm`` scale shuffle):

    quant               w1                       w2
    INT4 (uint4b8)      (E, 2*I, K//2) uint8    (E, K, I//2) uint8
    INT8 (uint8b128)    (E, 2*I, K)    uint8    (E, K, I)    uint8
    FP8 (fp8_e4m3)      (E, 2*I, K)    E4M3FN   (E, K, I)    E4M3FN
    MXFP4 (fp4_e2m1)    (E, 2*I, K//2) uint8    (E, K, I//2) uint8

    w1_scale: (E, 2*I, K//group_size)  w2_scale: (E, K, I//group_size)

  - 4-bit formats pack two codes per byte along the reduction dim: k even
    in the LOW nibble, k odd in the HIGH nibble.
  - group_size is 128 for INT4/INT8 and 32 for MXFP4; FP8 accepts 128/64/32
    and -1 for channelwise scaling (one scale per output row).
  - Scales are un-permuted, one column per group along the reduction dim.
    INT4/INT8 scales use the activation dtype, FP8 scales FP16/BF16/FP32,
    MXFP4 scales E8M0 bytes (255 encodes NaN).
  - K = hidden_size, I = intermediate_size, E = num_experts.

Decoding uses portable Triton arithmetic, with no PTX, PPU TLE instructions,
or floating-point weight expansion. Expert IDs must be in [0, E). Nonfinite
values follow floating point arithmetic; backward, expert parallelism, and
Marlin int32 layouts are unsupported.
"""

from typing import Any, Callable, Optional

import torch
import triton
import triton.language as tl
from torch.utils.weak import WeakTensorKeyDictionary

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

QUANT_TYPE_UINT4B8 = 0
QUANT_TYPE_UINT8B128 = 1
QUANT_TYPE_FP8_E4M3 = 2
QUANT_TYPE_FP4_E2M1 = 6
_SUPPORTED_QUANT_TYPES = {0, 1, 2, 6}


# Weak keys avoid retaining model weights. Values contain no reference to the
# source tensor. A different stream repacks to avoid reading an unfinished copy.
_TRANSPOSE_CACHE = WeakTensorKeyDictionary()


@triton.jit
def _transpose_kernel(
    X,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SK: tl.constexpr,
    B: tl.constexpr,
    TRANSFORM: tl.constexpr,
):
    e = tl.program_id(2).to(tl.int64)
    n = (tl.program_id(0) * B + tl.arange(0, B)).to(tl.int64)
    k = (tl.program_id(1) * B + tl.arange(0, B)).to(tl.int64)
    values = tl.load(
        X
        + e * SE
        + n[:, None] * SN
        + (k[None, :] // 2 if TRANSFORM == 1 else k[None, :]) * SK,
        (n[:, None] < N) & (k[None, :] < K),
        other=0,
    )
    if TRANSFORM == 1:
        code = (values.to(tl.int32) >> (4 * (k[None, :] % 2))) & 15
        mag = code & 7
        integer = tl.where(mag < 4, mag, (2 + (mag & 1)) << ((mag >> 1) - 1))
        integer = tl.where(code < 8, integer, -integer)
        values = tl.where(code == 8, 128, integer).to(tl.uint8)
    elif TRANSFORM == 2:
        sb = values.to(tl.uint32)
        scale = (sb << 23).to(tl.float32, bitcast=True)
        scale = tl.where(sb == 0, 5.877471754111438e-39, scale)
        values = tl.where(sb == 255, float("nan"), scale * 0.5)
    tl.store(
        Y + e * N * K + k[None, :] * N + n[:, None],
        values,
        (n[:, None] < N) & (k[None, :] < K),
    )


def _cached_transpose(tensor, transform=0):
    try:
        version = tensor._version
    except RuntimeError:
        # Inference tensors have no version counter: never cache them.
        version = None
    stream = torch.cuda.current_stream(tensor.device).cuda_stream
    key = (
        transform,
        version,
        tensor.shape,
        tensor.stride(),
        tensor.dtype,
        tensor.device,
        tensor.data_ptr(),
        stream,
    )
    if version is not None:
        entry = _TRANSPOSE_CACHE.get(tensor)
        if entry is not None and entry[0] == key:
            return entry[1]
    source = tensor
    if tensor.dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e8m0fnu", None),
    ):
        source = tensor.view(torch.uint8)
    e, n, k = source.shape
    if transform == 1:
        k *= 2
    packed = torch.empty(
        (e, k, n),
        device=source.device,
        dtype=torch.bfloat16 if transform == 2 else source.dtype,
    )
    _transpose_kernel[(triton.cdiv(n, 32), triton.cdiv(k, 32), e)](
        source, packed, n, k, *source.stride(), 32, transform, num_warps=4
    )
    result = packed.transpose(1, 2)
    if version is not None and not torch.cuda.is_current_stream_capturing():
        _TRANSPOSE_CACHE[tensor] = (key, result)
    return result


@triton.jit
def _group_scale(
    S,
    expert,
    n,
    g,
    valid,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    Q: tl.constexpr,
    EXACT: tl.constexpr = False,
):
    """Load one scale per output column; the group is constant over the tile."""
    expert = tl.cast(expert, tl.int64)
    n = tl.cast(n, tl.int64)
    address = S + expert * SE + n * SN + tl.cast(g, tl.int64) * SG
    if EXACT:
        s = tl.load(address).to(tl.float32)
    else:
        s = tl.load(address, valid, other=0).to(tl.float32)
    if Q == 6:
        # E8M0 255 denotes NaN, including when the weight is zero.
        sb = s.to(tl.uint32)
        scale = (sb << 23).to(tl.float32, bitcast=True)
        scale = tl.where(sb == 0, 5.877471754111438e-39, scale)
        s = tl.where(sb == 255, float("nan"), scale)
    return s


@triton.jit
def _decode_scaled(
    W,
    s,
    expert,
    n,
    k,
    N: tl.constexpr,
    K: tl.constexpr,
    WE: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    Q: tl.constexpr,
    DTYPE: tl.constexpr,
    EXACT: tl.constexpr = False,
):
    """Decode a weight tile against an already loaded per-column scale."""
    # Cast before stride multiplication: multi-expert weights can exceed 2 GiB.
    expert = tl.cast(expert, tl.int64)
    n = tl.cast(n, tl.int64)
    packed_k = (k // 2 if Q == 0 or Q == 6 else k).to(tl.int64)
    if EXACT:
        # Whole tiles need no per element bound test.
        q = tl.load(W + expert * WE + n * WN + packed_k * WK).to(tl.int32)
    else:
        q = tl.load(
            W + expert * WE + n * WN + packed_k * WK, (n < N) & (k < K), other=0
        ).to(tl.int32)
    if Q == 0 or Q == 6:
        q = (q >> (4 * (k % 2))) & 15
    if Q == 0:
        v = (q - 8).to(tl.float32)
    elif Q == 1:
        v = (q - 128).to(tl.float32)
    elif Q == 7:
        signed = q.to(tl.int8).to(tl.float32)
        v = tl.where(q == 128, 0x80000000, signed.to(tl.uint32, bitcast=True)).to(
            tl.float32, bitcast=True
        )
    elif Q == 6:
        mag = q & 7
        bits = tl.where(
            mag < 2, mag * (126 << 23), (((mag >> 1) + 126) << 23) | ((mag & 1) << 22)
        )
        v = (bits | ((q & 8) << 28)).to(tl.float32, bitcast=True)
    else:
        bits = ((q & 127) << 20) | ((q & 128) << 24)
        v = bits.to(tl.float32, bitcast=True) * 1.329227995784916e36
        v = tl.where((q & 127) == 127, float("nan"), v)
    value = (v * s).to(DTYPE)
    if Q == 7 and DTYPE == tl.float16:
        # Preserve cached E2M1 negative zero through the FP16 conversion.
        value = (
            value.to(tl.uint16, bitcast=True) | ((q == 128).to(tl.uint16) << 15)
        ).to(DTYPE, bitcast=True)
    return value


@triton.jit
def _decode(
    W,
    S,
    expert,
    n,
    k,
    N: tl.constexpr,
    K: tl.constexpr,
    WE: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    Q: tl.constexpr,
    GROUP: tl.constexpr,
    DTYPE: tl.constexpr,
    EXACT: tl.constexpr = False,
):
    """Decode a weight tile, loading one scale per element."""
    g = tl.full(k.shape, 0, tl.int64) if GROUP == -1 else (k // GROUP).to(tl.int64)
    s = _group_scale(S, expert, n, g, (n < N) & (k < K), SE, SN, SG, Q, EXACT)
    return _decode_scaled(W, s, expert, n, k, N, K, WE, WN, WK, Q, DTYPE, EXACT)


@triton.jit
def _routes(
    Ids,
    Routes,
    Counts,
    R: tl.constexpr,
    CAP: tl.constexpr,
    BM: tl.constexpr,
    B: tl.constexpr,
):
    expert = tl.program_id(0)
    r = tl.arange(0, B)
    match = (r < R) & (tl.load(Ids + r, r < R, other=-1) == expert)
    rank = tl.cumsum(match.to(tl.int32)) - 1
    count = tl.sum(match.to(tl.int32))
    tl.store(Routes + expert * CAP + rank, r, match)
    tl.store(Counts + expert, tl.cdiv(count, BM))
    pad = count + tl.arange(0, BM)
    tl.store(Routes + expert * CAP + pad, R, pad < tl.cdiv(count, BM) * BM)


@triton.jit
def _prefix(Counts, Starts, E: tl.constexpr, B: tl.constexpr):
    e = tl.arange(0, B)
    count = tl.load(Counts + e, e < E, other=0)
    end = tl.cumsum(count)
    tl.store(Starts + e, end - count, e < E)
    tl.store(Starts + E, tl.sum(count))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hygon_marlin_gemm"),
    key=["N", "K", "R", "E", "Q", "FIRST", "DIRECT", "SPLIT_K", "BM"],
)
@triton.jit
def _gemm(
    A,
    W,
    S,
    Tw,
    Ids,
    Routes,
    Starts,
    C,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    CAP: tl.constexpr,
    WE: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    Q: tl.constexpr,
    GROUP: tl.constexpr,
    FIRST: tl.constexpr,
    ROUTER_INPUT: tl.constexpr,
    DIRECT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    block = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    if DIRECT:
        expert = tl.load(Ids + block)
        route = tl.where(tl.arange(0, BM) == 0, block, R)
    else:
        total = tl.load(Starts + E)
        if block >= total:
            return
        # Upper bound search also handles empty experts (repeated starts).
        lo = 0
        hi = E
        while lo < hi:
            mid = (lo + hi) // 2
            start = tl.load(Starts + mid)
            take = start <= block
            lo = tl.where(take, mid + 1, lo)
            hi = tl.where(take, hi, mid)
        expert = lo - 1
        start = tl.load(Starts + expert)
        row = (block - start) * BM + tl.arange(0, BM)
        route = tl.load(Routes + expert * CAP + row)
    route = route.to(tl.int64)
    valid = (route < R) & (expert >= 0) & (expert < E)
    dtype: tl.constexpr = A.dtype.element_ty
    acc = tl.zeros((BM, BN), tl.float32)
    if FIRST:
        up = tl.zeros((BM, BN), tl.float32)
    split = tl.program_id(2)
    tiles = tl.cdiv(K, BK * SPLIT_K)
    # A reduction tile inside one scale group needs a single scale per column,
    # not one per weight element.
    HOIST: tl.constexpr = GROUP == -1 or BK <= GROUP
    # Tiles that divide the problem exactly need no per element bound test.
    # Dropping it is a large gain while the row tile is small and a loss at
    # BM=64, where the tile already amortizes the test over more rows.
    EXACT: tl.constexpr = BM <= 32 and N % BN == 0 and K % (BK * SPLIT_K) == 0
    columns: tl.constexpr = 2 * N if FIRST else N
    for step in range(tiles):
        base = (split * tiles + step) * BK
        k = base + tl.arange(0, BK)
        arow = route // TOPK if FIRST else route
        a = tl.load(
            A + arow[:, None] * K + k[None, :],
            valid[:, None] if EXACT else valid[:, None] & (k[None, :] < K),
            other=0,
        )
        if HOIST:
            g = 0 if GROUP == -1 else base // GROUP
            sv = _group_scale(
                S,
                expert,
                n[None, :],
                g,
                (n[None, :] < columns) & (base < K),
                SE,
                SN,
                SG,
                Q,
                EXACT,
            )
            w = _decode_scaled(
                W,
                sv,
                expert,
                n[None, :],
                k[:, None],
                columns,
                K,
                WE,
                WN,
                WK,
                Q,
                dtype,
                EXACT,
            )
        else:
            w = _decode(
                W,
                S,
                expert,
                n[None, :],
                k[:, None],
                columns,
                K,
                WE,
                WN,
                WK,
                SE,
                SN,
                SG,
                Q,
                GROUP,
                dtype,
                EXACT,
            )
        # A partial final output tile must not read the up half as gate data.
        if not EXACT:
            w = tl.where(n[None, :] < N, w, 0)
        if dtype == tl.float16:
            acc = tl.dot(
                a.to(tl.float32), w.to(tl.float32), acc, input_precision="ieee"
            )
        elif Q == 2:
            acc = tl.trans(tl.dot(tl.trans(w), tl.trans(a), tl.trans(acc)))
        else:
            acc = tl.dot(a, w, acc)
        if FIRST:
            if HOIST:
                svu = _group_scale(
                    S,
                    expert,
                    n[None, :] + N,
                    g,
                    (n[None, :] + N < 2 * N) & (base < K),
                    SE,
                    SN,
                    SG,
                    Q,
                    EXACT,
                )
                wu = _decode_scaled(
                    W,
                    svu,
                    expert,
                    n[None, :] + N,
                    k[:, None],
                    2 * N,
                    K,
                    WE,
                    WN,
                    WK,
                    Q,
                    dtype,
                    EXACT,
                )
            else:
                wu = _decode(
                    W,
                    S,
                    expert,
                    n[None, :] + N,
                    k[:, None],
                    2 * N,
                    K,
                    WE,
                    WN,
                    WK,
                    SE,
                    SN,
                    SG,
                    Q,
                    GROUP,
                    dtype,
                    EXACT,
                )
            if dtype == tl.float16:
                up = tl.dot(
                    a.to(tl.float32), wu.to(tl.float32), up, input_precision="ieee"
                )
            elif Q == 2:
                up = tl.trans(tl.dot(tl.trans(wu), tl.trans(a), tl.trans(up)))
            else:
                up = tl.dot(a, wu, up)
    if SPLIT_K > 1:
        stride = 2 * N if FIRST else N
        offsets = (split * R + route[:, None]) * stride + n[None, :]
        keep = valid[:, None] if EXACT else valid[:, None] & (n[None, :] < N)
        tl.store(C + offsets, acc, keep)
        if FIRST:
            tl.store(C + offsets + N, up, keep)
    else:
        if (FIRST and ROUTER_INPUT) or (not FIRST and not ROUTER_INPUT):
            rw = tl.load(Tw + route, valid, other=0).to(tl.float32)
            acc *= rw[:, None]
            if FIRST:
                up *= rw[:, None]
        if FIRST:
            gate = acc.to(dtype).to(tl.float32)
            up = up.to(dtype).to(tl.float32)
            acc = gate / (1.0 + tl.exp(-gate)) * up
        tl.store(
            C + route[:, None] * N + n[None, :],
            acc,
            valid[:, None] if EXACT else valid[:, None] & (n[None, :] < N),
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hygon_marlin_gemv"),
    key=["N", "K", "R", "Q", "FIRST", "GROUP"],
)
@triton.jit
def _gemv(
    A,
    W,
    S,
    Tw,
    Ids,
    C,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    TOPK: tl.constexpr,
    WE: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    Q: tl.constexpr,
    GROUP: tl.constexpr,
    FIRST: tl.constexpr,
    ROUTER_INPUT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    route = tl.program_id(0).to(tl.int64)
    expert = tl.load(Ids + route)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    arow = route // TOPK if FIRST else route
    dtype: tl.constexpr = A.dtype.element_ty
    accum = tl.zeros((BN, BK), tl.float32)
    if FIRST:
        up_accum = tl.zeros((BN, BK), tl.float32)
    for kb in range(tl.cdiv(K, BK)):
        k = kb * BK + kk
        a = tl.load(A + arow * K + k, k < K, other=0).to(tl.float32)
        w = _decode(
            W,
            S,
            expert,
            n[:, None],
            k[None, :],
            2 * N if FIRST else N,
            K,
            WE,
            WN,
            WK,
            SE,
            SN,
            SG,
            Q,
            GROUP,
            dtype,
        ).to(tl.float32)
        w = tl.where(n[:, None] < N, w, 0)
        accum += w * a[None, :]
        if FIRST:
            wu = _decode(
                W,
                S,
                expert,
                n[:, None] + N,
                k[None, :],
                2 * N,
                K,
                WE,
                WN,
                WK,
                SE,
                SN,
                SG,
                Q,
                GROUP,
                dtype,
            ).to(tl.float32)
            up_accum += wu * a[None, :]
    acc = tl.sum(accum, 1)
    if FIRST:
        up = tl.sum(up_accum, 1)
    if (FIRST and ROUTER_INPUT) or (not FIRST and not ROUTER_INPUT):
        weight = tl.load(Tw + route).to(tl.float32)
        acc *= weight
        if FIRST:
            up *= weight
    if FIRST:
        gate = acc.to(dtype).to(tl.float32)
        up = up.to(dtype).to(tl.float32)
        acc = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(C + route * N + n, acc, n < N)


@triton.jit
def _finish_split(
    P,
    C,
    Tw,
    N: tl.constexpr,
    R: tl.constexpr,
    SPLIT_K: tl.constexpr,
    FIRST: tl.constexpr,
    ROUTER_INPUT: tl.constexpr,
    B: tl.constexpr,
):
    route = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * B + tl.arange(0, B)
    split = tl.arange(0, SPLIT_K)
    stride = 2 * N if FIRST else N
    offsets = (split[:, None] * R + route) * stride + n[None, :]
    acc = tl.sum(tl.load(P + offsets, n[None, :] < N, other=0), 0)
    if FIRST:
        up = tl.sum(tl.load(P + offsets + N, n[None, :] < N, other=0), 0)
    if (FIRST and ROUTER_INPUT) or (not FIRST and not ROUTER_INPUT):
        weight = tl.load(Tw + route).to(tl.float32)
        acc *= weight
        if FIRST:
            up *= weight
    if FIRST:
        dtype: tl.constexpr = C.dtype.element_ty
        gate = acc.to(dtype).to(tl.float32)
        up = up.to(dtype).to(tl.float32)
        acc = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(C + route * N + n, acc, n < N)


@triton.jit
def _sum(C, Out, K: tl.constexpr, TOPK: tl.constexpr, B: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    k = (tl.program_id(1) * B + tl.arange(0, B)).to(tl.int64)
    acc = tl.zeros((B,), tl.float32)
    for t in range(TOPK):
        acc += tl.load(C + (m * TOPK + t) * K + k, k < K, other=0).to(tl.float32)
    tl.store(Out + m * K + k, acc, k < K)


def _fused_marlin_moe_impl(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_ids,
    *,
    apply_router_weight_on_input,
    inplace,
    output,
    group_size,
    quant_type_id,
):
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
    out = hidden_states if inplace else output
    if out is None:
        out = torch.empty_like(hidden_states)
    if m == 0:
        return out
    r = m * topk
    # Route workspaces are O(E * R). Bound memory until a streaming aligner
    # is provided; no hidden host synchronization or CPU routing is used.
    if r > 16384 or e > 1024:
        raise NotImplementedError(
            "Hygon Marlin MoE supports at most 16384 routes and 1024 experts"
        )
    # Reuse expert weights when routes repeat; large banks make repeated
    # direct loads more expensive than the small route-alignment kernels.
    direct = r <= (2 if max(k, n) >= 4096 else 16)
    bm = 64 if r >= 32 * e else 32 if r >= 16 * e else 16
    cap = triton.cdiv(r, bm) * bm
    # Keep the original accumulation order when routing weights precede
    # activation: rounding differences can be amplified by the gated product.
    split_k = (
        8 if max(k, n) >= 4096 and r <= 32 and not apply_router_weight_on_input else 1
    )
    with torch.cuda.device(hidden_states.device):
        routes = (
            torch.empty((e, cap), device=out.device, dtype=torch.int32)
            if not direct
            else topk_ids
        )
        starts = (
            torch.empty((e + 1,), device=out.device, dtype=torch.int32)
            if not direct
            else topk_ids
        )
        if not direct:
            counts = torch.empty((e,), device=out.device, dtype=torch.int32)
            _routes[(e,)](
                topk_ids,
                routes,
                counts,
                r,
                cap,
                bm,
                triton.next_power_of_2(r),
                num_warps=4,
            )
            _prefix[(1,)](counts, starts, e, triton.next_power_of_2(e), num_warps=4)
        intermediate = torch.empty((r, n), dtype=out.dtype, device=out.device)
        result = torch.empty((r, k), dtype=out.dtype, device=out.device)
        blocks = r if direct else min(r, triton.cdiv(r, bm) + e - 1)
        for a, w, s, c, first, nk, kk in (
            (hidden_states, w1, w1_scale, intermediate, True, n, k),
            (intermediate, w2, w2_scale, result, False, k, n),
        ):
            kernel_q = quant_type_id
            if quant_type_id == QUANT_TYPE_FP4_E2M1:
                w = _cached_transpose(w, 1)
                s = _cached_transpose(s, 2)
                kernel_q = 7
            if quant_type_id == QUANT_TYPE_UINT4B8:
                w = _cached_transpose(w)
                s = _cached_transpose(s)
            if quant_type_id == QUANT_TYPE_FP8_E4M3:
                w = w.view(torch.uint8)
            if direct and quant_type_id == QUANT_TYPE_UINT8B128:
                _gemv[lambda meta: (r, triton.cdiv(nk, meta["BN"]))](
                    a,
                    w,
                    s,
                    topk_weights,
                    topk_ids,
                    c,
                    nk,
                    kk,
                    r,
                    topk,
                    *w.stride(),
                    *s.stride(),
                    kernel_q,
                    group_size,
                    first,
                    apply_router_weight_on_input,
                )
                continue
            partial = (
                torch.empty(
                    (split_k, r, nk * (2 if first else 1)),
                    dtype=torch.float32,
                    device=out.device,
                )
                if split_k > 1
                else c
            )
            _gemm[lambda meta: (triton.cdiv(nk, meta["BN"]), blocks, split_k)](
                a,
                w,
                s,
                topk_weights,
                topk_ids,
                routes,
                starts,
                partial,
                nk,
                kk,
                r,
                e,
                topk,
                cap,
                *w.stride(),
                *s.stride(),
                kernel_q,
                group_size,
                first,
                apply_router_weight_on_input,
                direct,
                split_k,
                bm,
            )
            if split_k > 1:
                _finish_split[(r, triton.cdiv(nk, 256))](
                    partial,
                    c,
                    topk_weights,
                    nk,
                    r,
                    split_k,
                    first,
                    apply_router_weight_on_input,
                    256,
                    num_warps=4,
                )
        _sum[(m, triton.cdiv(k, 256))](result, out, k, topk, 256, num_warps=4)
    return out


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
        raise NotImplementedError("Hygon fused Marlin MoE requires FP16 or BF16")
    if ids.dtype not in (torch.int32, torch.int64) or tw.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise NotImplementedError("Unsupported routing dtype")
    tensors = (a, w1, w2, s1, s2, tw, ids)
    if a.device.type != "cuda" or any(t.device != a.device for t in tensors):
        raise ValueError("All tensors must reside on the same Hygon device")
    if any(t.requires_grad for t in tensors):
        raise NotImplementedError("Hygon fused Marlin MoE is inference-only")
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
    """Hygon override for INT4, MXFP4, INT8 and FP8 fused Marlin MoE."""
    if quant_type_id not in _SUPPORTED_QUANT_TYPES:
        raise NotImplementedError("Unsupported quant_type_id")
    activation_str = getattr(
        activation, "value", getattr(activation, "name", activation)
    )
    if activation_str is not None and str(activation_str).lower() != "silu":
        raise NotImplementedError("Hygon fused Marlin MoE supports only SiLU")
    if any(
        value is not None for value in (g_idx1, g_idx2, sort_indices1, sort_indices2)
    ):
        raise NotImplementedError("Hygon fused Marlin MoE does not support act_order")
    if input_dtype is not None:
        raise NotImplementedError("Hygon fused Marlin MoE does not support FP8 input")
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
            "Unsupported Hygon fused Marlin MoE option "
            "(bias/map/scaling/workspace/activation)"
        )
    if w1.ndim != 3 or global_num_experts not in (-1, w1.shape[0]):
        raise NotImplementedError("Hygon fused Marlin MoE requires local weights")
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
