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

GEMM tiles decode weights directly from the caller's quantized storage.
Expert IDs must be in [0, E). Nonfinite values follow floating point
arithmetic. Backward, expert parallelism, and Marlin int32 layouts are
unsupported.
"""

from typing import Any, Callable, Optional

import torch
import triton
import triton.language as tl
from flag_gems import zeros

from flaggems_vllm import runtime
from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_no_tle
from flaggems_vllm.utils import libentry, libtuner

QUANT_TYPE_UINT4B8 = 0
QUANT_TYPE_UINT8B128 = 1
QUANT_TYPE_FP8_E4M3 = 2
QUANT_TYPE_FP4_E2M1 = 6
_SUPPORTED_QUANT_TYPES = {0, 1, 2, 6}
_SHARED_ROUTING_MIN_ROUTES = 1024
_SHARED_ROUTING_MIN_ROUTES_PER_EXPERT = 4
# Resident inputs and the caller-visible output are outside the scratch budget.
_MAX_WORKSPACE_BYTES = 2 << 30


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
    """Load scales for the requested output columns and reduction groups."""
    expert = tl.cast(expert, tl.int64)
    n = tl.cast(n, tl.int64)
    address = S + expert * SE + n * SN + tl.cast(g, tl.int64) * SG
    if EXACT:
        s = tl.load(address)
    else:
        s = tl.load(address, valid, other=0)
    if Q == 6:
        # E8M0 255 denotes NaN, including when the weight is zero.
        sb = s.to(tl.uint32)
        scale = (sb << 23).to(tl.float32, bitcast=True)
        scale = tl.where(sb == 0, 5.877471754111438e-39, scale)
        s = tl.where(sb == 255, float("nan"), scale)
    if Q == 2:
        # Canonicalize NaN once per scale group before broadcasting to weights.
        s = tl.where(s == s, s, float("nan"))
    return s.to(tl.float32)


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
    SCALE_CANONICAL: tl.constexpr = False,
    FOLD_SCALE: tl.constexpr = False,
    FP8X4: tl.constexpr = False,
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
    value = _scale_weight_codes(q, s, Q, DTYPE, SCALE_CANONICAL, FOLD_SCALE, FP8X4)
    if not EXACT:
        # A padded zero code can decode to NaN against an infinite scale.
        value = tl.where((n < N) & (k < K), value, 0.0)
    return value


@triton.jit
def _round_bfloat16(value):
    # Callers preserve canonical NaNs or payloads originating from BF16.
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _scale_weight_codes(
    q,
    s,
    Q: tl.constexpr,
    DTYPE: tl.constexpr,
    SCALE_CANONICAL: tl.constexpr = False,
    FOLD_SCALE: tl.constexpr = False,
    FP8X4: tl.constexpr = False,
):
    # Leave products visible so gfx936 retains paired FP32 multiplies.
    if Q == 2 and DTYPE == tl.bfloat16 and SCALE_CANONICAL and FP8X4:
        # The second result reserves scratch registers for the byte decode.
        decoded, _ = tl.inline_asm_elementwise(
            """
            v_bfe_u32 $0, $6, 0, 7;
            v_cmp_eq_u32_e32 vcc, 127, $0;
            v_lshlrev_b32_e32 $0, 20, $0;
            v_lshlrev_b32_e32 $4, 24, $6;
            v_and_or_b32 $0, $4, $11, $0;
            v_cndmask_b32_e32 $0, $0, $7, vcc;
            v_bfe_u32 $1, $6, 8, 7;
            v_cmp_eq_u32_e32 vcc, 127, $1;
            v_lshlrev_b32_e32 $1, 20, $1;
            v_lshlrev_b32_e32 $4, 16, $6;
            v_and_or_b32 $1, $4, $11, $1;
            v_cndmask_b32_e32 $1, $1, $8, vcc;
            v_bfe_u32 $2, $6, 16, 7;
            v_cmp_eq_u32_e32 vcc, 127, $2;
            v_lshlrev_b32_e32 $2, 20, $2;
            v_lshlrev_b32_e32 $4, 8, $6;
            v_and_or_b32 $2, $4, $11, $2;
            v_cndmask_b32_e32 $2, $2, $9, vcc;
            v_bfe_u32 $3, $6, 24, 7;
            v_cmp_eq_u32_e32 vcc, 127, $3;
            v_lshlrev_b32_e32 $3, 20, $3;
            v_and_or_b32 $3, $6, $11, $3;
            v_cndmask_b32_e32 $3, $3, $10, vcc;
            """,
            constraints="=&v,=&v,=&v,=&v,=&v,=&v,v,v,v,v,v,s,s,s,s,~{vcc}",
            args=[
                q.to(tl.uint8),
                tl.full((), 0x7FC00000, tl.uint32).to(tl.float32, bitcast=True),
                tl.full((), 0x80000000, tl.uint32),
            ],
            dtype=(tl.float32, tl.bfloat16),
            is_pure=True,
            pack=4,
        )
        if FOLD_SCALE:
            s = s * 1.329227995784916e36
        else:
            decoded = decoded * 1.329227995784916e36
        return _round_bfloat16(decoded * s)
    if Q == 0:
        v = (q - 8).to(tl.float32)
    elif Q == 1:
        # Every uint8 code is exactly representable before subtracting the bias.
        v = q.to(tl.uint8).to(tl.float32) - 128.0
    elif Q == 6:
        # The FP16 exponent bias is 14 larger than E2M1's.
        bits = ((q & 8) << 12) | ((q & 7) << 9)
        v = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32) * 16384.0
    else:
        bits = ((q & 127) << 20) | ((q & 128) << 24)
        v = bits.to(tl.float32, bitcast=True)
        if not FOLD_SCALE:
            v = v * 1.329227995784916e36
        v = tl.where((q & 127) == 127, float("nan"), v)
    if Q == 2 and FOLD_SCALE:
        s = s * 1.329227995784916e36
    if DTYPE == tl.bfloat16 and Q == 6:
        # E2M1 times an E8M0 scale is exact in BF16, including subnormals.
        bits = (v * s).to(tl.uint32, bitcast=True)
        return (bits >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    if Q == 2 and DTYPE == tl.bfloat16 and SCALE_CANONICAL:
        # Canonical NaNs stay NaN under integer round-to-nearest-even.
        return _round_bfloat16(v * s)
    value = (v * s).to(DTYPE)
    if Q == 6 and DTYPE == tl.float16:
        # Preserve E2M1 negative zero through the FP16 conversion.
        value = (value.to(tl.uint16, bitcast=True) | ((q == 8).to(tl.uint16) << 15)).to(
            DTYPE, bitcast=True
        )
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
def _load_gemm_weight(
    W,
    S,
    expert,
    n,
    base,
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
    BN: tl.constexpr,
    BK: tl.constexpr,
    EXACT: tl.constexpr,
    WEIGHT_KN: tl.constexpr,
    FOLD_SCALE: tl.constexpr = False,
):
    if WEIGHT_KN:
        k = (base + tl.arange(0, BK))[:, None]
        n = n[None, :]
        if GROUP == -1 or BK <= GROUP:
            group = 0 if GROUP == -1 else base // GROUP
            scale = _group_scale(
                S, expert, n, group, (n < N) & (base < K), SE, SN, SG, Q, EXACT
            )
            return _decode_scaled(
                W,
                scale,
                expert,
                n,
                k,
                N,
                K,
                WE,
                WN,
                WK,
                Q,
                DTYPE,
                EXACT,
                True,
                FOLD_SCALE,
                Q == 2 and DTYPE == tl.bfloat16 and N >= 2048 and K == 256 and BK == 64,
            )
        return _decode(
            W,
            S,
            expert,
            n,
            k,
            N,
            K,
            WE,
            WN,
            WK,
            SE,
            SN,
            SG,
            Q,
            GROUP,
            DTYPE,
            EXACT,
        )

    else:
        # Both operands use rows that are contiguous along K in the native layout.
        expert = expert.to(tl.int64)
        n = n.to(tl.int64)[:, None]
        k = base + tl.arange(0, BK)
        row = W + expert * WE + n * WN
        if Q == 0 or Q == 6:
            packed_k = (base // 2 + tl.arange(0, BK // 2)).to(tl.int64)
            if EXACT:
                packed = tl.load(row + packed_k[None, :] * WK)
            else:
                packed = tl.load(
                    row + packed_k[None, :] * WK,
                    (n < N) & (packed_k[None, :] < K // 2),
                    other=0x88 if Q == 0 else 0,
                )
            packed = packed.to(tl.int32)
            # Join on the innermost axis so every byte contributes even then odd K.
            q = tl.reshape(tl.join(packed & 15, packed >> 4), (BN, BK))
        else:
            address = row + k[None, :].to(tl.int64) * WK
            if EXACT:
                q = tl.load(address).to(tl.int32)
            else:
                q = tl.load(
                    address, (n < N) & (k[None, :] < K), other=128 if Q == 1 else 0
                ).to(tl.int32)

        # A tile inside one scale group uses one scale per output column.
        if GROUP == -1 or BK <= GROUP:
            g = 0 if GROUP == -1 else base // GROUP
            scale = _group_scale(
                S, expert, n, g, (n < N) & (base < K), SE, SN, SG, Q, EXACT
            )
            value = _scale_weight_codes(q, scale, Q, DTYPE, True, FOLD_SCALE)
            if not EXACT:
                value = tl.where((n < N) & (k[None, :] < K), value, 0.0)
            return value
        else:
            num_groups: tl.constexpr = BK // GROUP
            groups = base // GROUP + tl.arange(0, num_groups)
            scale = _group_scale(
                S,
                expert,
                n,
                groups[None, :],
                (n < N) & (groups[None, :] < K // GROUP),
                SE,
                SN,
                SG,
                Q,
                EXACT,
            )
            grouped_codes = tl.reshape(q, (BN, num_groups, GROUP))
            value = _scale_weight_codes(
                grouped_codes, scale[:, :, None], Q, DTYPE, True, FOLD_SCALE
            )
            value = tl.reshape(value, (BN, BK))
            if not EXACT:
                value = tl.where((n < N) & (k[None, :] < K), value, 0.0)
            return value


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


def _prune_gemm_configs(configs, named_args, **kwargs):
    bm, q, first = named_args["BM"], named_args["Q"], named_args["FIRST"]
    n, k, e, r = (named_args[key] for key in ("N", "K", "E", "R"))
    w8 = q in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3)
    packed_w4 = q in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1)
    bf16 = named_args["A"].dtype == torch.bfloat16
    single_k = bf16 and named_args["SPLIT_K"] == 1
    mxfp4 = q == QUANT_TYPE_FP4_E2M1
    narrow = e >= 128 and (n <= 512 and k >= 2048 if first else k == 256 and n >= 2048)
    early_paired = narrow and r >= max(256, e)
    wide = single_k and max(n, k) >= 8192
    wide_w8 = wide and w8 and bm >= 32
    wide_fp8 = wide_w8 and q == QUANT_TYPE_FP8_E4M3
    wide_mxfp4 = wide and mxfp4 and bm >= 64
    short_down = (
        single_k
        and not first
        and narrow
        and (r >= 512 or (early_paired and q == QUANT_TYPE_UINT8B128))
        and bm == 16
    )
    dense_narrow = single_k and narrow and bm >= 128
    dense_fp8 = dense_narrow and q == QUANT_TYPE_FP8_E4M3
    middle_fp8 = (
        single_k
        and narrow
        and q == QUANT_TYPE_FP8_E4M3
        and bm in (32, 64)
        and not named_args["ROUTER_INPUT"]
    )

    dot_layouts = {0}
    if (
        single_k
        and first
        and bm == 16
        and (r >= max(512, 2 * e) or early_paired)
        and k >= 2048
        and n % 32 == 0
        and (q == QUANT_TYPE_UINT8B128 or (narrow and (w8 or mxfp4)))
    ):
        dot_layouts.add(1)
    if (dense_fp8 or middle_fp8) and first:
        dot_layouts.add(1)
    if (
        wide_fp8
        or wide_mxfp4
        or (single_k and q == QUANT_TYPE_FP8_E4M3 and first and bm == 128)
        or (dense_narrow and w8)
        or (mxfp4 and not first and (short_down or dense_narrow))
    ):
        dot_layouts.add(2)
    if wide_w8 or (short_down and w8) or ((dense_fp8 or middle_fp8) and not first):
        dot_layouts.add(3)
    if (
        q == QUANT_TYPE_FP8_E4M3
        and bf16
        and first
        and bm == 16
        and narrow
        and k >= 4096
        and (r <= 6 or 192 < r <= 256)
        and not named_args["ROUTER_INPUT"]
    ):
        dot_layouts.add(1)
    multi_stage = (
        wide_fp8
        or wide_mxfp4
        or (bf16 and r >= 8192 and (bm >= 128 or (bm >= 64 and packed_w4)))
    )
    return [
        config
        for config in configs
        if config.kwargs.get("DOT_LAYOUT", 0) in dot_layouts
        and (
            kwargs.get("KPACK", 1) == 1
            or (
                config.kwargs.get("DOT_LAYOUT", 0) == (1 if first else 3)
                and config.kwargs["BK"] == 64
                and config.num_warps == 4
                and config.num_stages == 1
            )
        )
        and (multi_stage or config.num_stages == 1)
        and (
            config.kwargs["BN"] <= 128
            or (
                not first
                and (
                    dense_fp8 or middle_fp8 or (short_down and q == QUANT_TYPE_FP8_E4M3)
                )
            )
        )
        and (
            config.kwargs.get("DOT_LAYOUT", 0) != 1
            or config.kwargs["BN"] <= 32
            or dense_fp8
            or middle_fp8
        )
        and (
            (
                config.kwargs.get("DOT_LAYOUT", 0),
                config.kwargs["BN"],
                config.kwargs["BK"],
                config.num_warps,
                config.num_stages,
            )
            != (2, 128, 64, 8, 2)
            or (q == QUANT_TYPE_FP8_E4M3 and dense_narrow)
        )
    ]


@triton.jit
def _accumulate_gemm(
    a,
    w,
    acc,
    STANDARD: tl.constexpr,
    WEIGHT_KN: tl.constexpr,
    FP16: tl.constexpr,
):
    if WEIGHT_KN:
        lhs = a
        rhs = w
    elif STANDARD:
        lhs = a
        rhs = tl.trans(w)
    else:
        lhs = w
        rhs = a
    if FP16:
        return tl.dot(
            lhs.to(tl.float32), rhs.to(tl.float32), acc, input_precision="ieee"
        )
    else:
        return tl.dot(lhs, rhs, acc)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hygon_marlin_gemm"),
    key=[
        "N",
        "K",
        "R",
        "E",
        "Q",
        "GROUP",
        "KPACK",
        "FIRST",
        "DIRECT",
        "SHARED_ROUTING",
        "SPLIT_K",
        "BM",
    ],
    prune_configs_by={"early_config_prune": _prune_gemm_configs},
)
@triton.jit
def _gemm(
    A,
    W,
    S,
    Tw,
    ExpertIds,
    Routes,
    RouteMetadata,
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
    SHARED_ROUTING: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DOT_LAYOUT: tl.constexpr = 0,
    KPACK: tl.constexpr = 1,
):
    # The dot K fragment must hold every packed BF16 value.
    tl.static_assert(BK >= 16 * KPACK)
    block = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    if DIRECT:
        expert = tl.load(ExpertIds + block)
        route = tl.where(tl.arange(0, BM) == 0, block, R)
    elif SHARED_ROUTING:
        first_route = block * BM
        if first_route >= tl.load(RouteMetadata):
            return
        expert = tl.load(ExpertIds + block)
        route = tl.load(Routes + first_route + tl.arange(0, BM))
    else:
        total = tl.load(RouteMetadata + E)
        if block >= total:
            return
        # Upper bound search also handles empty experts (repeated starts).
        lo = 0
        hi = E
        while lo < hi:
            mid = (lo + hi) // 2
            start = tl.load(RouteMetadata + mid)
            take = start <= block
            lo = tl.where(take, mid + 1, lo)
            hi = tl.where(take, hi, mid)
        expert = lo - 1
        start = tl.load(RouteMetadata + expert)
        row = (block - start) * BM + tl.arange(0, BM)
        route = tl.load(Routes + expert * CAP + row)
    route = route.to(tl.int64)
    valid = (route < R) & (expert >= 0) & (expert < E)
    dtype: tl.constexpr = A.dtype.element_ty
    PAIRED: tl.constexpr = DOT_LAYOUT == 1
    STANDARD: tl.constexpr = DOT_LAYOUT == 2
    WEIGHT_KN: tl.constexpr = DOT_LAYOUT == 3
    PAIRED_ROW_MAJOR: tl.constexpr = PAIRED and (Q == 2 or Q == 6)
    ROW_MAJOR_ACC: tl.constexpr = STANDARD or WEIGHT_KN or PAIRED_ROW_MAJOR
    if PAIRED:
        weight_rows = tl.arange(0, 2 * BN)
        weight_n = tl.program_id(0) * BN + weight_rows % BN + (weight_rows // BN) * N
        WEIGHT_N: tl.constexpr = 2 * N
        WEIGHT_BN: tl.constexpr = 2 * BN
    else:
        weight_n = n
        WEIGHT_N: tl.constexpr = N
        WEIGHT_BN: tl.constexpr = BN
    ACC_M: tl.constexpr = BM if ROW_MAJOR_ACC else WEIGHT_BN
    ACC_N: tl.constexpr = WEIGHT_BN if ROW_MAJOR_ACC else BM
    acc_t = tl.zeros((ACC_M, ACC_N), tl.float32)
    if FIRST and not PAIRED:
        up_t = tl.zeros((ACC_M, ACC_N), tl.float32)
    split = tl.program_id(2)
    tiles = tl.cdiv(K, BK * SPLIT_K)
    # Weight bounds depend on N and K, even when routing uses a wide M tile.
    WEIGHT_EXACT: tl.constexpr = N % BN == 0 and K % (BK * SPLIT_K) == 0
    EXACT: tl.constexpr = BM <= 32 and WEIGHT_EXACT
    LOAD_EXACT: tl.constexpr = WEIGHT_EXACT and (not WEIGHT_KN or Q == 2 or BM <= 32)
    arow = route // TOPK if FIRST else route
    unroll: tl.constexpr = (
        (8 if FIRST else 4)
        if Q == 2
        and dtype == tl.bfloat16
        and BM == 16
        and (DOT_LAYOUT == 1 or DOT_LAYOUT == 3)
        else 1
    )
    # Folding 2**120 into finite scales with abs(scale) <= 128 is exact.
    # Check each tile once so the K loop keeps a single weight multiply.
    CHECK_ROWS: tl.constexpr = 2 * BN if FIRST else BN
    CHECK_GROUPS: tl.constexpr = triton.next_power_of_2(triton.cdiv(K, 128))
    CAN_FOLD: tl.constexpr = (
        FIRST
        and Q == 2
        and dtype == tl.bfloat16
        and GROUP == 128
        and BK <= 128
        and K % 128 == 0
        and SPLIT_K == 1
        and R >= 256
        and CHECK_ROWS * CHECK_GROUPS <= 8192
    )
    if CAN_FOLD:
        rr = tl.arange(0, CHECK_ROWS)
        cn = tl.program_id(0) * BN + rr % BN + (rr // BN) * N
        gg = tl.arange(0, CHECK_GROUPS)
        sv = tl.load(
            S
            + expert.to(tl.int64) * SE
            + cn[:, None].to(tl.int64) * SN
            + gg[None, :].to(tl.int64) * SG,
            (tl.program_id(0) * BN + (rr % BN)[:, None] < N) & (gg[None, :] < K // 128),
            other=0,
        )
        if S.dtype.element_ty == tl.float32:
            magnitude = sv.to(tl.uint32, bitcast=True) & 0x7FFFFFFF
            limit: tl.constexpr = 0x43000000
        else:
            magnitude = sv.to(tl.uint16, bitcast=True) & 0x7FFF
            limit: tl.constexpr = (
                0x4300 if S.dtype.element_ty == tl.bfloat16 else 0x5800
            )
        can_fold = tl.max(magnitude) <= limit
    else:
        can_fold = False
    for fold in tl.static_range(2):
        if can_fold == (fold == 0):
            for step in tl.range(tiles, loop_unroll_factor=unroll):
                base = (split * tiles + step) * BK
                k = base + tl.arange(0, BK)
                a = tl.load(
                    A + arow[:, None] * K + k[None, :],
                    valid[:, None] if EXACT else valid[:, None] & (k[None, :] < K),
                    other=0,
                )
                a_t = a if ROW_MAJOR_ACC else tl.trans(a)
                if dtype == tl.float16:
                    a_t = a_t.to(tl.float32)
                w_t = _load_gemm_weight(
                    W,
                    S,
                    expert,
                    weight_n,
                    base,
                    WEIGHT_N,
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
                    WEIGHT_BN,
                    BK,
                    LOAD_EXACT,
                    WEIGHT_KN,
                    fold == 0,
                )
                acc_t = _accumulate_gemm(
                    a_t, w_t, acc_t, ROW_MAJOR_ACC, WEIGHT_KN, dtype == tl.float16
                )
                if FIRST and not PAIRED:
                    wu_t = _load_gemm_weight(
                        W,
                        S,
                        expert,
                        n + N,
                        base,
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
                        BN,
                        BK,
                        LOAD_EXACT,
                        WEIGHT_KN,
                        fold == 0,
                    )
                    up_t = _accumulate_gemm(
                        a_t, wu_t, up_t, STANDARD, WEIGHT_KN, dtype == tl.float16
                    )
    if PAIRED:
        # Keep gate/up in the caller's two separate weight banks and share the dot.
        if PAIRED_ROW_MAJOR:
            acc, up = tl.split(tl.permute(tl.reshape(acc_t, (BM, 2, BN)), (0, 2, 1)))
        else:
            acc, up = tl.split(tl.permute(tl.reshape(acc_t, (2, BN, BM)), (2, 1, 0)))
    else:
        acc = acc_t if ROW_MAJOR_ACC else tl.trans(acc_t)
        if FIRST:
            up = up_t if ROW_MAJOR_ACC else tl.trans(up_t)
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
            if Q == 2 and dtype == tl.bfloat16 and BM >= 32:
                rw = tl.where(rw == rw, rw, float("nan"))
            acc *= rw[:, None]
            if FIRST:
                up *= rw[:, None]
        if FIRST:
            if Q == 2 and dtype == tl.bfloat16 and BM >= 32:
                gate = _round_bfloat16(acc).to(tl.float32)
                up = _round_bfloat16(up).to(tl.float32)
            else:
                gate = acc.to(dtype).to(tl.float32)
                up = up.to(dtype).to(tl.float32)
            acc = gate / (1.0 + tl.exp(-gate)) * up
        if Q == 2 and dtype == tl.bfloat16 and BM >= 32:
            acc = _round_bfloat16(acc)
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
    if e > 1024:
        raise NotImplementedError("Hygon Marlin MoE supports at most 1024 experts")

    element_size = hidden_states.element_size()
    projection_bytes_per_token = topk * (n + k) * element_size

    def auxiliary_workspace_bytes(tokens):
        if tokens == 0:
            return 0
        r = tokens * topk
        direct, bm, shared, gate_split, down_split = _batch_config(
            r, e, k, n, quant_type_id, hidden_states.dtype, apply_router_weight_on_input
        )
        if direct:
            routing = 0
        elif shared:
            blocks = triton.cdiv(r + e * (bm - 1), bm)
            routing = blocks * (bm + 1) + (e + 1) ** 2 + 1
        else:
            routing = e * triton.cdiv(r, bm) * bm + 2 * e + 1
        # The two projection partials are allocated and released in order.
        split = r * max(
            gate_split * 2 * n if gate_split > 1 else 0,
            down_split * k if down_split > 1 else 0,
        )
        return 4 * (routing + split)

    batch_tokens = m
    while (
        batch_tokens * projection_bytes_per_token
        + max(
            auxiliary_workspace_bytes(batch_tokens),
            auxiliary_workspace_bytes(m % batch_tokens),
        )
        > _MAX_WORKSPACE_BYTES
    ):
        if batch_tokens == 1:
            raise NotImplementedError(
                "One token exceeds the Hygon MoE workspace budget"
            )
        batch_tokens = triton.cdiv(batch_tokens, 2)

    starts = range(0, m, batch_tokens)
    # Preserve the output alias contract when a shifted output overlaps input.
    if batch_tokens < m and (
        0 < out.data_ptr() - hidden_states.data_ptr() < m * k * element_size
    ):
        starts = reversed(starts)
    intermediate = torch.empty(
        (batch_tokens * topk, n), dtype=out.dtype, device=out.device
    )
    result = torch.empty((batch_tokens * topk, k), dtype=out.dtype, device=out.device)
    for start in starts:
        batch = slice(start, start + batch_tokens)
        _run_moe_batch(
            hidden_states[batch],
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights[batch],
            topk_ids[batch],
            out[batch],
            intermediate,
            result,
            apply_router_weight_on_input,
            group_size,
            quant_type_id,
        )
    return out


def _batch_config(r, e, k, n, quant_type_id, dtype, apply_router_weight_on_input):
    # Reuse expert weights when routes repeat; large banks make repeated
    # direct loads more expensive than the small route-alignment kernels.
    direct = r <= (2 if max(k, n) >= 4096 else 16)
    small_quantized = (
        dtype == torch.bfloat16
        and quant_type_id in (QUANT_TYPE_FP8_E4M3, QUANT_TYPE_FP4_E2M1)
        and e >= 128
        and k >= 4096
        and n == 256
        and not apply_router_weight_on_input
    )
    direct = direct or (small_quantized and r <= 24)
    bm = 64 if r >= 32 * e else 32 if r >= 16 * e else 16
    # Dense BF16 batches can amortize each decoded weight tile over more routes.
    if dtype == torch.bfloat16 and r >= max(8192, 80 * e):
        bm = 128
        if (
            quant_type_id == QUANT_TYPE_FP4_E2M1
            and n == 256
            and k >= 2048
            and e >= 128
            and r >= 384 * e
        ):
            bm = 256
    # The shared aligner has four stages and a quadratic expert-count buffer.
    # Amortize that overhead on grouped batches with enough routes per expert.
    shared_routing = r >= max(
        _SHARED_ROUTING_MIN_ROUTES, _SHARED_ROUTING_MIN_ROUTES_PER_EXPERT * e
    )
    # Keep the original accumulation order when routing weights precede
    # activation: rounding differences can be amplified by the gated product.
    split_k = (
        8
        if r <= 32
        and not apply_router_weight_on_input
        and not (direct and quant_type_id == QUANT_TYPE_UINT8B128)
        else 1
    )
    gate_split = split_k if k >= 4096 else 1
    down_split = split_k if n >= 4096 else 1
    # More gate CTAs help hide the long K reduction in sparse quantized batches.
    if small_quantized and r <= 192:
        gate_split = 16 if quant_type_id == QUANT_TYPE_FP4_E2M1 and r <= 6 else 8
    # Sparse routes through a narrow gate tile leave too few CTAs to hide
    # the long INT8 weight load. Partition the gate reduction only.
    if (
        dtype == torch.bfloat16
        and quant_type_id == QUANT_TYPE_UINT8B128
        and e >= 128
        and k >= 4096
        and n <= 512
        and 32 < r <= 96
        and not apply_router_weight_on_input
    ):
        gate_split = 8 if r <= 64 else 4
    return direct, bm, shared_routing, gate_split, down_split


def _run_moe_batch(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_ids,
    out,
    intermediate,
    result,
    apply_router_weight_on_input,
    group_size,
    quant_type_id,
):
    m, k = hidden_states.shape
    e, n2, _ = w1.shape
    n = n2 // 2
    topk = topk_ids.shape[1]
    r = m * topk
    direct, bm, shared_routing, gate_split, down_split = _batch_config(
        r, e, k, n, quant_type_id, hidden_states.dtype, apply_router_weight_on_input
    )
    cap = triton.cdiv(r, bm) * bm
    with torch.cuda.device(hidden_states.device):
        expert_ids = topk_ids
        if direct:
            routes = route_metadata = topk_ids
        elif shared_routing:
            workspace = zeros(((e + 1) ** 2,), device=out.device, dtype=torch.int32)
            routes, expert_ids, route_metadata = moe_align_block_size_no_tle(
                topk_ids,
                bm,
                e,
                pad_sorted_ids=True,
                workspace=(workspace[: e + 1], workspace[e + 1 :]),
            )
        else:
            routes = torch.empty((e, cap), device=out.device, dtype=torch.int32)
            route_metadata = torch.empty((e + 1,), device=out.device, dtype=torch.int32)
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
            _prefix[(1,)](
                counts, route_metadata, e, triton.next_power_of_2(e), num_warps=4
            )
        blocks = r if direct else min(r, triton.cdiv(r, bm) + e - 1)
        for a, w, s, c, first, nk, kk, split_k in (
            (hidden_states, w1, w1_scale, intermediate, True, n, k, gate_split),
            (intermediate, w2, w2_scale, result, False, k, n, down_split),
        ):
            if quant_type_id == QUANT_TYPE_FP8_E4M3:
                w = w.view(torch.uint8)
            elif quant_type_id == QUANT_TYPE_FP4_E2M1:
                s = s.view(torch.uint8)
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
                    quant_type_id,
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
            packed_k = (
                4
                if quant_type_id == QUANT_TYPE_FP8_E4M3
                and hidden_states.dtype == torch.bfloat16
                and not apply_router_weight_on_input
                and e >= 128
                and k >= 4096
                and n == 256
                and split_k == 1
                and (bm >= 128 or (not first and bm >= 16 and r >= 512))
                else 1
            )
            _gemm[lambda meta: (triton.cdiv(nk, meta["BN"]), blocks, split_k)](
                a,
                w,
                s,
                topk_weights,
                expert_ids,
                routes,
                route_metadata,
                partial,
                nk,
                kk,
                r,
                e,
                topk,
                cap,
                *w.stride(),
                *s.stride(),
                quant_type_id,
                group_size,
                first,
                apply_router_weight_on_input,
                direct,
                shared_routing,
                split_k,
                bm,
                KPACK=packed_k,
                kpack=packed_k,
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
                del partial
        sum_block, sum_warps = (1024, 8) if m >= 2048 and k >= 1024 else (256, 4)
        _sum[(m, triton.cdiv(k, sum_block))](
            result, out, k, topk, sum_block, num_warps=sum_warps
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
        output_begin = target.data_ptr()
        output_end = output_begin + target.numel() * target.element_size()
        for tensor in (w1, w2, s1, s2, tw, ids):
            if not tensor.numel():
                continue
            input_begin = tensor.data_ptr()
            # Weight and scale views may have padding between rows/groups.
            input_end = input_begin + tensor.element_size() * (
                1
                + sum(
                    (size - 1) * stride
                    for size, stride in zip(tensor.shape, tensor.stride())
                )
            )
            if output_begin < input_end and input_begin < output_end:
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
