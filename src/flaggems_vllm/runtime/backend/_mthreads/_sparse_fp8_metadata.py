# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Current-input precision bounds and selected KV metadata for sparse FP8 MLA."""

import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_native import query_abs4

KV_TOKEN_BYTES = tl.constexpr(512)
KV_OFFSET_TOKEN_LIMIT = (1 << 32) // KV_TOKEN_BYTES.value
BOUNDED_EXP_PV_BUDGET = tl.constexpr(8.1e-5)
FAST_NATIVE_EXP = tl.constexpr(1)
FAST_BOUNDED_EXP = tl.constexpr(2)

PACKED_METADATA_TOKEN_LIMIT = 1 << 17

# Packet bits 0..7 contain a positive normal float32 scale exponent;
# bits 9..25 contain the token's 512-byte aligned offset. Invalid indices
# encode the cache-end offset and a zero scale. All cache scales are checked
# on every call. Other formats retain exact float32 scale bits and raw indices.
# Fast flags retain the original exponent mode in bits 0..1; bit 2 selects
# this packet format. The original error bound still selects FP32 repair.


@triton.jit
def encode_kv_byte_offset(token, valid, TOKENS: tl.constexpr):
    # Clamp invalid token IDs before the 32-bit multiplication can wrap.
    return (tl.where(valid, token, TOKENS).to(tl.uint32) * KV_TOKEN_BYTES).to(tl.int32)


@triton.jit
def gather_scales(
    Indices,
    KS,
    Selected,
    Offsets,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
    PACKED_META: tl.constexpr = False,
    Global=None,
    CacheBits=None,
):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(Indices + x, x < ROWS * K, -1)
    valid = (x < ROWS * K) & (ids >= 0) & (ids < TOKENS)
    if PACKED_META:
        minimum = tl.load(Global + 1).to(tl.int32)
        if minimum > 0:
            safe_ids = tl.where(valid, ids, 0).to(tl.uint32)
            token_bits = tl.load(CacheBits + (safe_ids >> 5), valid, 0).to(tl.uint32)
            parity = (token_bits >> (safe_ids & 31)) & 1
            exponent = tl.where(
                valid, minimum.to(tl.uint32) + (parity ^ (minimum & 1).to(tl.uint32)), 0
            )
            location = encode_kv_byte_offset(ids, valid, TOKENS).to(tl.uint32)
            word = exponent | location
        else:
            scale = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
            word = scale.to(tl.uint32, bitcast=True)
        tl.store(Selected + x, word, x < ROWS * K)
    else:
        scale = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        tl.store(Selected + x, scale, x < ROWS * K)
        if Offsets is not None:
            tl.store(
                Offsets + x, encode_kv_byte_offset(ids, valid, TOKENS), x < ROWS * K
            )


@triton.jit
def cache_scale_bounds(
    KS,
    Indices,
    Selected,
    Offsets,
    ScaleParts,
    BadCount,
    TOKENS: tl.constexpr,
    SELECTED: tl.constexpr,
    BT: tl.constexpr,
    PACKED_META: tl.constexpr = False,
    CacheBits=None,
):
    if tl.program_id(0) == 0:
        tl.store(BadCount, 0)
    offset = tl.program_id(0) * BT + tl.arange(0, BT)
    if SELECTED:
        token = tl.load(Indices + offset, offset < SELECTED, -1)
        valid = (offset < SELECTED) & (token >= 0) & (token < TOKENS)
    else:
        token = offset
        valid = token < TOKENS
    scale = tl.load(KS + tl.where(valid, token, 0), valid, 0.0)
    if SELECTED:
        tl.store(Selected + offset, scale, offset < SELECTED)
        if Offsets is not None:
            tl.store(
                Offsets + offset,
                encode_kv_byte_offset(token, valid, TOKENS),
                offset < SELECTED,
            )
    tl.store(ScaleParts + tl.program_id(0), tl.max(scale, 0))
    if PACKED_META:
        tl.static_assert(SELECTED == 0)
        PARTS: tl.constexpr = triton.cdiv(TOKENS, BT)
        bits = scale.to(tl.uint32, bitcast=True)
        exponent = (bits >> 23) & 255
        exact = ((bits & 0x807FFFFF) == 0) & (exponent > 0) & (exponent < 255)
        bad = tl.max((valid & ~exact).to(tl.int32), 0)
        minimum = tl.min(tl.where(valid, exponent, 255), 0)
        parity = tl.reshape(tl.where(valid, exponent & 1, 0), (BT // 32, 32))
        word = tl.sum(parity << tl.arange(0, 32)[None, :], 1)
        pos = tl.program_id(0) * (BT // 32) + tl.arange(0, BT // 32)
        tl.store(CacheBits + pos, word, pos < triton.cdiv(TOKENS, 32))
        tl.store(ScaleParts + PARTS + tl.program_id(0), minimum.to(tl.float32))
        tl.store(ScaleParts + 2 * PARTS + tl.program_id(0), bad.to(tl.float32))


@triton.jit
def reduce_scales(
    ScaleParts,
    Global,
    PARTS: tl.constexpr,
    BP: tl.constexpr,
    PACKED_META: tl.constexpr = False,
):
    part = tl.arange(0, BP)
    maximum = tl.max(tl.load(ScaleParts + part, part < PARTS, 0.0), 0)
    tl.store(Global, maximum)
    if PACKED_META:
        minimum = tl.min(tl.load(ScaleParts + PARTS + part, part < PARTS, 255.0), 0).to(
            tl.int32
        )
        bad = tl.max(tl.load(ScaleParts + 2 * PARTS + part, part < PARTS, 0.0), 0)
        maximum_exponent = ((maximum.to(tl.uint32, bitcast=True) >> 23) & 255).to(
            tl.int32
        )
        codec_ok = (
            (bad == 0)
            & (maximum_exponent >= minimum)
            & (maximum_exponent - minimum <= 1)
        )
        tl.store(Global + 1, tl.where(codec_ok, minimum, 0).to(tl.float32))


@triton.jit(noinline=True)
def repair_k128(
    Q,
    QS,
    KV,
    KS,
    Indices,
    Output,
    LSE,
    row,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
):
    selected = tl.arange(0, 128)
    indices = tl.load(Indices + row * 128 + selected)
    valid = (indices >= 0) & (indices < TOKENS)
    if tl.max(valid.to(tl.int32), 0) == 0:
        lanes = tl.arange(0, 1024)
        for start in range(0, 64 * 512, 1024):
            tl.store(Output + row * (64 * 512) + start + lanes, 0.0)
        tl.store(LSE + row * 64 + tl.arange(0, 64), float("inf"))
    else:
        HEADS: tl.constexpr = 8
        FEATURES: tl.constexpr = 8
        tokens = tl.where(valid, indices, 0).to(tl.int64)
        gains = tl.load(KS + tokens, valid, 0.0)
        columns = tl.arange(0, FEATURES)
        # Reuse each key tile across a head group while bounding register use.
        for group in range(64 // HEADS):
            h = row * 64 + group * HEADS + tl.arange(0, HEADS)
            q_scale = tl.load(QS + h)
            scores = tl.full((HEADS, 128), 0.0, tl.float32)
            for start in range(0, 512, FEATURES):
                dims = start + columns
                query = tl.load(Q + h[:, None] * 512 + dims[None, :]).to(tl.float32)
                query *= q_scale[:, None]
                key = tl.load(
                    KV + tokens[:, None] * 512 + dims[None, :], valid[:, None], 0.0
                ).to(tl.float32)
                key *= gains[:, None]
                scores += tl.sum(query[:, None, :] * key[None, :, :], 2)
            logits = tl.where(valid[None, :], scores * SCALE, float("-inf"))
            maximum = tl.max(logits, 1)
            center = tl.where(maximum == float("-inf"), 0.0, maximum)
            weights = tl.where(valid[None, :], tl.exp(logits - center[:, None]), 0.0)
            total = tl.sum(weights, 1)
            normal = tl.where(total > 0.0, total, 1.0)
            tl.store(
                LSE + h, tl.where(total > 0.0, center + tl.log(total), float("inf"))
            )
            weights /= normal[:, None]
            for start in range(0, 512, FEATURES):
                dims = start + columns
                value = tl.load(
                    KV + tokens[:, None] * 512 + dims[None, :], valid[:, None], 0.0
                ).to(tl.float32)
                value *= gains[:, None]
                acc = tl.sum(value[None, :, :] * weights[:, :, None], 1)
                tl.store(Output + h[:, None] * 512 + dims[None, :], acc)


@triton.jit
def query_bounds_global(
    Q,
    QS,
    Global,
    PScale,
    Fast,
    BadRows,
    BadCount,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    PACKED: tl.constexpr = False,
    Indices=None,
    KS=None,
    Selected=None,
    Offsets=None,
    TOKENS: tl.constexpr = 0,
    PACKED_META: tl.constexpr = False,
    KV=None,
    Output=None,
    LSE=None,
):
    row = tl.program_id(0).to(tl.int64)
    if Indices is None:
        kscale = tl.load(Global)
    else:
        col = tl.arange(0, triton.next_power_of_2(K))
        ids = tl.load(Indices + row * K + col, col < K, -1)
        valid = (col < K) & (ids >= 0) & (ids < TOKENS)
        scales = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        tl.store(Selected + row * K + col, scales, col < K)
        if Offsets is not None:
            byte_offset = encode_kv_byte_offset(ids, valid, TOKENS)
            tl.store(Offsets + row * K + col, byte_offset, col < K)
        # Each query only uses its own selected KV rows.
        kscale = tl.max(scales, 0)
    h = tl.arange(0, 64)
    qs = tl.load(QS + row * 64 + h)
    if PACKED:
        d = tl.arange(0, 128)
        words = tl.load(
            Q.to(tl.pointer_type(tl.uint32))
            + (row * 64 + h[:, None]) * 128
            + d[None, :]
        )
        q_parts = query_abs4(words)
        # Cover both FP16 addition levels while retaining the original L1 margin.
        bound = tl.sum(q_parts, 1) * qs * (448.0 * kscale) * SCALE * 1.00154 + 1e-5
    else:
        d = tl.arange(0, 512)
        q = tl.load(Q + (row * 64 + h[:, None]) * 512 + d[None, :]).to(tl.float32)
        bound = tl.sum(tl.abs(q), 1) * qs * (448.0 * kscale) * SCALE * 1.0005 + 1e-5
    safe_bound = tl.minimum(bound, 20.0)
    exp_bound = tl.exp(safe_bound)
    scale = kscale * exp_bound * (1.0001 / 448.0)
    if K == 640 and PACKED and Indices is not None:
        # For this common range, the original score and PV budgets are
        # bounded by 7.2e-5 and 0.0024 respectively (including rounding).
        head_certified = (
            (qs >= 0)
            & (bound >= 0)
            & (bound <= 2.0)
            & (exp_bound <= 8.0)
            & (scale >= 2.0**-126)
        )
        certified = (
            (kscale <= 2.0**-9)
            & (kscale > 0)
            & (tl.min(head_certified.to(tl.int32), 0) != 0)
        )
    else:
        certified = False
    if certified:
        use_poly = True
        eligible = True
    else:
        U: tl.constexpr = 2.0**-24
        GAMMA_PV: tl.constexpr = 2 * K * U / (1 - 2 * K * U)
        DENOM_STEPS: tl.constexpr = triton.cdiv(K, 128) + 10
        GAMMA_DENOM: tl.constexpr = (
            DENOM_STEPS * 2.0**-23 / (1 - DENOM_STEPS * 2.0**-23)
        )
        # Two FP8 terms have a relative error plus a subnormal rounding floor.
        coefficient = (
            2.0**-9 + 1.25 * GAMMA_PV + GAMMA_DENOM + 4 * 2.0**-19 + 8 * U
        ) / (1 - GAMMA_DENOM)
        score_error = bound * (512 * U / (1 - 512 * U))
        floor_error = (
            448 * scale * exp_bound * 2.0**-10 * (1 + 4 * GAMMA_PV) / (1 - GAMMA_DENOM)
        )
        pv_error = (
            448 * kscale * (coefficient + tl.exp(4.1 * score_error) - 1) + floor_error
        )
        if K > 128:
            use_poly = tl.min(((qs >= 0) & (bound <= 2.0)).to(tl.int32), 0) != 0
            pv_error += tl.where(use_poly, 448 * kscale * BOUNDED_EXP_PV_BUDGET, 0.0)
        else:
            use_poly = False
        eligible = (
            tl.min(
                (
                    (score_error <= 1e-4)
                    & (pv_error < 0.0038)
                    & (bound < 20.0)
                    & (scale >= 2.0**-126)
                ).to(tl.int32),
                0,
            )
            != 0
        )
    tl.store(PScale + row * 64 + h, scale)
    fast_mode = tl.where(use_poly, FAST_BOUNDED_EXP, FAST_NATIVE_EXP)
    if PACKED_META:
        exponent_base = tl.load(Global + 1).to(tl.int32)
        format_tag = tl.where(exponent_base > 0, 4, 0)
        fast_mode = fast_mode | format_tag
    tl.store(Fast + row, tl.where(eligible, fast_mode, 0))
    if BadCount is not None and not eligible:
        index = tl.atomic_add(BadCount, 1)
        tl.store(BadRows + index, row)
    if KV is not None:
        tl.static_assert(K == 128 and Indices is not None)
        if not eligible:
            repair_k128(Q, QS, KV, KS, Indices, Output, LSE, row, TOKENS, SCALE)
