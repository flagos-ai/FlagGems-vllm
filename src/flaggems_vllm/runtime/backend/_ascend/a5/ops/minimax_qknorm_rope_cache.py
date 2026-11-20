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

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import extension as _cann_ext

from flaggems_vllm.runtime import torch_device_fn

_extract_slice = _cann_ext.extract_slice
_insert_slice = _cann_ext.insert_slice

_T = 4
_NCHUNK = 2
_NSTAGE = 2


@triton.jit
def _rope_inplace(y, cos, sin, ROWS: tl.constexpr, HALF: tl.constexpr):
    """NeoX rotate dims [0, 2*HALF) of a (ROWS, HEAD_DIM) tile in registers.

    cos/sin are (ROWS, HALF) tiles so each row may carry its own position.
    """
    a = _extract_slice(y, (0, 0), (ROWS, HALF), (1, 1))
    b = _extract_slice(y, (0, HALF), (ROWS, HALF), (1, 1))
    y = _insert_slice(y, a * cos - b * sin, (0, 0), (ROWS, HALF), (1, 1))
    y = _insert_slice(y, b * cos + a * sin, (0, HALF), (ROWS, HALF), (1, 1))
    return y


@triton.jit
def _rope_inplace3(y, cos, sin, T: tl.constexpr, H: tl.constexpr, HALF: tl.constexpr):
    """NeoX rotate dims [0, 2*HALF) of a (T, H, HEAD_DIM) tile in registers.

    cos/sin are (T, HALF) tiles broadcast over the head axis, so one gather per
    chunk serves every head row of every token. `y` is a bf16 tile: only the
    2*HALF rotated lanes are widened to fp32, and the rotated result is written
    back as bf16. That is exactly the reference value, because the reference
    rotates bf16-rounded inputs in fp32 and then stores to bf16.
    """
    c3 = cos[:, None, :]
    s3 = sin[:, None, :]
    a = _extract_slice(y, (0, 0, 0), (T, H, HALF), (1, 1, 1)).to(tl.float32)
    b = _extract_slice(y, (0, 0, HALF), (T, H, HALF), (1, 1, 1)).to(tl.float32)
    y = _insert_slice(
        y, (a * c3 - b * s3).to(tl.bfloat16), (0, 0, 0), (T, H, HALF), (1, 1, 1)
    )
    y = _insert_slice(
        y, (b * c3 + a * s3).to(tl.bfloat16), (0, 0, HALF), (T, H, HALF), (1, 1, 1)
    )
    return y


@triton.jit
def _rms_norm3(x, w, HEAD_DIM: tl.constexpr, EPS: tl.constexpr):
    """(T, H, HEAD_DIM) fp32 tile -> the reference's bf16-rounded, (1+w)-scaled tile.

    bf16 is deliberately kept as the tile type: the store rounds to bf16 anyway,
    so the half-size tile keeps the rotation working set small.
    """
    inv = tl.rsqrt(tl.sum(x * x, 2) / HEAD_DIM + EPS)
    y = x * inv[:, :, None] * (1.0 + w)[None, None, :]
    return y.to(tl.bfloat16)


@triton.jit
def _rms_norm(x, w, HEAD_DIM: tl.constexpr, EPS: tl.constexpr):
    """(ROWS, HEAD_DIM) fp32 tile -> bf16 round-tripped, (1+w)-scaled fp32 tile."""
    inv = tl.rsqrt(tl.sum(x * x, 1) / HEAD_DIM + EPS)
    y = x * inv[:, None] * (1.0 + w)[None, :]
    return y.to(tl.bfloat16).to(tl.float32)


# --------------------------------------------------------------------------- #
# fallback: one token per program (masked lane-partner reload for RoPE)
# --------------------------------------------------------------------------- #
@triton.jit
def _norm_rope_row(
    base, roff, w_ptr, cs, HEAD_DIM: tl.constexpr, HALF: tl.constexpr, EPS: tl.constexpr
):
    cols = tl.arange(0, HEAD_DIM)
    x = tl.load(base + roff + cols).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x) / HEAD_DIM + EPS)
    y = x * inv * (1.0 + w)
    y = y.to(tl.bfloat16).to(tl.float32)

    lo = cols < HALF
    hi = (cols >= HALF) & (cols < 2 * HALF)
    m = lo | hi
    poff = tl.where(lo, cols + HALF, cols - HALF)
    p = tl.load(base + roff + poff, mask=m, other=0.0).to(tl.float32)
    pw = tl.load(w_ptr + poff, mask=m, other=0.0).to(tl.float32)
    p = p * inv * (1.0 + pw)
    p = p.to(tl.bfloat16).to(tl.float32)

    cidx = tl.where(lo, cols, cols - HALF)
    cos = tl.load(cs + cidx, mask=m, other=0.0).to(tl.float32)
    sin = tl.load(cs + HALF + cidx, mask=m, other=0.0).to(tl.float32)
    rot = tl.where(lo, y * cos - p * sin, y * cos + p * sin)
    return tl.where(m, rot, y)


@triton.jit
def _norm_rope_tile(
    base,
    roff,
    w_ptr,
    cs,
    ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
    EPS: tl.constexpr,
):
    cols = tl.arange(0, HEAD_DIM)
    rows = tl.arange(0, ROWS)
    x = tl.load(base + roff[:, None] + cols[None, :]).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / HEAD_DIM + EPS)
    y = x * inv[:, None] * (1.0 + w)[None, :]
    y = y.to(tl.bfloat16).to(tl.float32)

    lo = cols < HALF
    hi = (cols >= HALF) & (cols < 2 * HALF)
    m = lo | hi
    poff = tl.where(lo, cols + HALF, cols - HALF)
    p = tl.load(base + roff[:, None] + poff[None, :], mask=m[None, :], other=0.0).to(
        tl.float32
    )
    pw = tl.load(w_ptr + poff, mask=m, other=0.0).to(tl.float32)
    p = p * inv[:, None] * (1.0 + pw)[None, :]
    p = p.to(tl.bfloat16).to(tl.float32)

    cidx = tl.where(lo, cols, cols - HALF)
    cos = tl.load(cs + cidx, mask=m, other=0.0).to(tl.float32)
    sin = tl.load(cs + HALF + cidx, mask=m, other=0.0).to(tl.float32)
    rot = tl.where(
        lo[None, :],
        y * cos[None, :] - p * sin[None, :],
        y * cos[None, :] + p * sin[None, :],
    )
    return tl.where(m[None, :], rot, y)


# Short-token path adapted from vllm-ascend/ops/triton/msa/minimax_qknorm_rope_cache.py.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
@triton.jit
def _short_norm_rope(
    input_ptr,
    weight_ptr,
    cos_sin_ptr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    EPS: tl.constexpr,
    other_weight_ptr,
    SPLIT_HEADS: tl.constexpr,
):
    cols = tl.arange(0, HEAD_DIM)
    heads = tl.arange(0, HEADS)
    x = tl.load(input_ptr + heads[:, None] * HEAD_DIM + cols[None, :]).to(tl.float32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    other_weight = tl.load(other_weight_ptr + cols).to(tl.float32)
    weight = tl.where(
        heads[:, None] < SPLIT_HEADS, weight[None, :], other_weight[None, :]
    )
    x = x * tl.rsqrt(tl.sum(x * x, 1) / HEAD_DIM + EPS)[:, None] * (1.0 + weight)
    x = x.to(input_ptr.dtype.element_ty).to(tl.float32)
    half: tl.constexpr = ROTARY_DIM // 2
    cos = tl.load(cos_sin_ptr + tl.arange(0, half)).to(tl.float32)
    sin = tl.load(cos_sin_ptr + half + tl.arange(0, half)).to(tl.float32)
    a = _extract_slice(x, (0, 0), (HEADS, half), (1, 1))
    b = _extract_slice(x, (0, half), (HEADS, half), (1, 1))
    x = _insert_slice(
        x, a * cos[None, :] - b * sin[None, :], (0, 0), (HEADS, half), (1, 1)
    )
    x = _insert_slice(
        x, b * cos[None, :] + a * sin[None, :], (0, half), (HEADS, half), (1, 1)
    )
    return x


# Request-dependent bounds must not specialize on their value or alignment.
# Head dimensions and layout strides below are fixed for a warmed deployment.
@triton.jit(do_not_specialize=["NUM_SLOTS"])
def _kernel_single(
    packed,
    index_packed,
    cos_sin,
    positions,
    q_weight,
    k_weight,
    iq_weight,
    ik_weight,
    query,
    index_query,
    key_cache,
    value_cache,
    index_cache,
    slots,
    index_slots,
    NUM_SLOTS,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    IQ_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    INPUT_STRIDE: tl.constexpr,
    INDEX_INPUT_STRIDE: tl.constexpr,
    INDEX_OFFSET: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    COS_SIN_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    K_BLOCK: tl.constexpr,
    K_TOKEN: tl.constexpr,
    K_HEAD: tl.constexpr,
    V_BLOCK: tl.constexpr,
    V_TOKEN: tl.constexpr,
    V_HEAD: tl.constexpr,
    I_BLOCK: tl.constexpr,
    I_TOKEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INDEX_BLOCK_SIZE: tl.constexpr,
    CACHE_FP8: tl.constexpr,
    INDEX_FP8: tl.constexpr,
    FP8_E4M3_MAX: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    position = tl.load(positions + token * POSITION_STRIDE).to(tl.int64)
    cs = cos_sin + position * COS_SIN_STRIDE
    dim = tl.arange(0, HEAD_DIM)
    row = packed + token * INPUT_STRIDE
    qk = _short_norm_rope(
        row,
        q_weight,
        cs,
        Q_HEADS + KV_HEADS,
        HEAD_DIM,
        ROTARY_DIM,
        EPS,
        k_weight,
        Q_HEADS,
    )
    q = _extract_slice(qk, (0, 0), (Q_HEADS, HEAD_DIM), (1, 1))
    offsets = (
        token * Q_HEADS * HEAD_DIM
        + tl.arange(0, Q_HEADS)[:, None] * HEAD_DIM
        + dim[None, :]
    )
    tl.store(query + offsets, q)
    slot = tl.load(slots + token, token < NUM_SLOTS, other=-1).to(tl.int64)
    if slot >= 0:
        k = _extract_slice(qk, (Q_HEADS, 0), (KV_HEADS, HEAD_DIM), (1, 1))
        k = k.to(packed.dtype.element_ty).to(tl.float32)
        v = tl.load(
            row
            + (Q_HEADS + KV_HEADS) * HEAD_DIM
            + tl.arange(0, KV_HEADS)[:, None] * HEAD_DIM
            + dim[None, :]
        )
        if CACHE_FP8:
            k = tl.minimum(tl.maximum(k, -FP8_E4M3_MAX), FP8_E4M3_MAX)
            v = tl.minimum(tl.maximum(v.to(tl.float32), -FP8_E4M3_MAX), FP8_E4M3_MAX)
        k_offset = slot // BLOCK_SIZE * K_BLOCK + slot % BLOCK_SIZE * K_TOKEN
        v_offset = slot // BLOCK_SIZE * V_BLOCK + slot % BLOCK_SIZE * V_TOKEN
        tl.store(
            key_cache
            + k_offset
            + tl.arange(0, KV_HEADS)[:, None] * K_HEAD
            + dim[None, :],
            k,
        )
        tl.store(
            value_cache
            + v_offset
            + tl.arange(0, KV_HEADS)[:, None] * V_HEAD
            + dim[None, :],
            v,
        )
    row = index_packed + token * INDEX_INPUT_STRIDE + INDEX_OFFSET
    iqk = _short_norm_rope(
        row, iq_weight, cs, IQ_HEADS + 1, HEAD_DIM, ROTARY_DIM, EPS, ik_weight, IQ_HEADS
    )
    if INDEX_FP8:
        iqk = tl.minimum(tl.maximum(iqk, -FP8_E4M3_MAX), FP8_E4M3_MAX)
    iq = _extract_slice(iqk, (0, 0), (IQ_HEADS, HEAD_DIM), (1, 1))
    offsets = (
        token * IQ_HEADS * HEAD_DIM
        + tl.arange(0, IQ_HEADS)[:, None] * HEAD_DIM
        + dim[None, :]
    )
    tl.store(index_query + offsets, iq)
    slot = tl.load(index_slots + token, token < NUM_SLOTS, other=-1).to(tl.int64)
    if slot >= 0:
        ik = _extract_slice(iqk, (IQ_HEADS, 0), (1, HEAD_DIM), (1, 1))
        offset = slot // INDEX_BLOCK_SIZE * I_BLOCK + slot % INDEX_BLOCK_SIZE * I_TOKEN
        tl.store(index_cache + offset + dim, ik.reshape(HEAD_DIM))


# --------------------------------------------------------------------------- #
# tiled path: NCHUNK pipelined chunks of T tokens per program
# --------------------------------------------------------------------------- #
@triton.jit
def _chunk(
    tok0,
    packed,
    cos_sin,
    positions,
    wq,
    wk,
    wiq,
    wik,
    query,
    index_query,
    key_cache,
    value_cache,
    index_cache,
    slots,
    index_slots,
    NUM_SLOTS,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    IQ_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    INPUT_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    COS_SIN_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    K_BLOCK: tl.constexpr,
    K_TOKEN: tl.constexpr,
    V_BLOCK: tl.constexpr,
    V_TOKEN: tl.constexpr,
    I_BLOCK: tl.constexpr,
    I_TOKEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INDEX_BLOCK_SIZE: tl.constexpr,
    CACHE_FP8: tl.constexpr,
    INDEX_FP8: tl.constexpr,
    FP8_E4M3_MAX: tl.constexpr,
    T: tl.constexpr,
):
    """One chunk of T tokens. Body is identical to the un-pipelined kernel."""
    HALF: tl.constexpr = ROTARY_DIM // 2
    cols = tl.arange(0, HEAD_DIM)
    tk = tok0 + tl.arange(0, T)

    # Issue every independent transfer before the first compute, largest first,
    # and leave the dependent positions -> cos/sin gather until last so its
    # serial latency is covered by the already-queued bursts.
    off3 = (
        (tk * INPUT_STRIDE)[:, None, None]
        + (tl.arange(0, Q_HEADS) * HEAD_DIM)[None, :, None]
        + cols[None, None, :]
    )
    xq = tl.load(packed + off3).to(tl.float32)

    # k / v / index q / index k are the four rows right after the q rows, so
    # one affine (T, 4, HEAD_DIM) burst of 1 KB per token replaces four 256 B
    # bursts; the rows are sliced apart in registers.
    toff = (
        (tk * INPUT_STRIDE + Q_HEADS * HEAD_DIM)[:, None, None]
        + (tl.arange(0, 4) * HEAD_DIM)[None, :, None]
        + cols[None, None, :]
    )
    tail = tl.load(packed + toff)
    st = tl.load(slots + tk, tk < NUM_SLOTS, other=-1)
    ist = tl.load(index_slots + tk, tk < NUM_SLOTS, other=-1)

    pos = tl.load(positions + tk * POSITION_STRIDE).to(tl.int64)
    cs = tl.load(
        cos_sin + pos[:, None] * COS_SIN_STRIDE + tl.arange(0, ROTARY_DIM)[None, :]
    ).to(tl.float32)
    cos_t = _extract_slice(cs, (0, 0), (T, HALF), (1, 1))
    sin_t = _extract_slice(cs, (0, HALF), (T, HALF), (1, 1))

    kidx: tl.constexpr = 0
    vidx: tl.constexpr = 1
    iqidx: tl.constexpr = 2
    ikidx: tl.constexpr = 3
    xk = _extract_slice(tail, (0, kidx, 0), (T, 1, HEAD_DIM), (1, 1, 1)).to(tl.float32)
    yk = _rope_inplace3(_rms_norm3(xk, wk, HEAD_DIM, EPS), cos_t, sin_t, T, 1, HALF)

    xiq = _extract_slice(tail, (0, iqidx, 0), (T, 1, HEAD_DIM), (1, 1, 1)).to(
        tl.float32
    )
    yiq = _rope_inplace3(_rms_norm3(xiq, wiq, HEAD_DIM, EPS), cos_t, sin_t, T, 1, HALF)
    if INDEX_FP8:
        yiq = tl.minimum(tl.maximum(yiq.to(tl.float32), -FP8_E4M3_MAX), FP8_E4M3_MAX)
    tl.store(
        index_query + (tk * (IQ_HEADS * HEAD_DIM))[:, None, None] + cols[None, None, :],
        yiq,
    )

    xik = _extract_slice(tail, (0, ikidx, 0), (T, 1, HEAD_DIM), (1, 1, 1)).to(
        tl.float32
    )
    yik = _rope_inplace3(_rms_norm3(xik, wik, HEAD_DIM, EPS), cos_t, sin_t, T, 1, HALF)

    # ---- paged cache scatter: one scalar-branched row per token
    for i in tl.static_range(T):
        s = tl.sum(_extract_slice(st, (i,), (1,), (1,))).to(tl.int64)
        if s >= 0:
            krow = _extract_slice(yk, (i, 0, 0), (1, 1, HEAD_DIM), (1, 1, 1))
            vr = _extract_slice(tail, (i, vidx, 0), (1, 1, HEAD_DIM), (1, 1, 1))
            if CACHE_FP8:
                krow = tl.minimum(
                    tl.maximum(krow.to(tl.float32), -FP8_E4M3_MAX), FP8_E4M3_MAX
                )
                vr = tl.minimum(
                    tl.maximum(vr.to(tl.float32), -FP8_E4M3_MAX), FP8_E4M3_MAX
                )
            k_offset = s // BLOCK_SIZE * K_BLOCK + s % BLOCK_SIZE * K_TOKEN
            v_offset = s // BLOCK_SIZE * V_BLOCK + s % BLOCK_SIZE * V_TOKEN
            tl.store(key_cache + k_offset + cols[None, None, :], krow)
            tl.store(value_cache + v_offset + cols[None, None, :], vr)
        si = tl.sum(_extract_slice(ist, (i,), (1,), (1,))).to(tl.int64)
        if si >= 0:
            ikrow = _extract_slice(yik, (i, 0, 0), (1, 1, HEAD_DIM), (1, 1, 1))
            if INDEX_FP8:
                ikrow = tl.minimum(tl.maximum(ikrow, -FP8_E4M3_MAX), FP8_E4M3_MAX)
            i_offset = (
                si // INDEX_BLOCK_SIZE * I_BLOCK + si % INDEX_BLOCK_SIZE * I_TOKEN
            )
            tl.store(index_cache + i_offset + cols[None, None, :], ikrow)

    # ---- q last: the chunk ends on the single largest contiguous burst.
    yq = _rms_norm3(xq, wq, HEAD_DIM, EPS)
    yq = _rope_inplace3(yq, cos_t, sin_t, T, Q_HEADS, HALF)
    tl.store(
        query
        + (
            (tk * (Q_HEADS * HEAD_DIM))[:, None, None]
            + (tl.arange(0, Q_HEADS) * HEAD_DIM)[None, :, None]
            + cols[None, None, :]
        ),
        yq,
    )


@triton.jit(do_not_specialize=["NUM_SLOTS"])
def _kernel_tiled(
    packed,
    index_packed,
    cos_sin,
    positions,
    q_weight,
    k_weight,
    iq_weight,
    ik_weight,
    query,
    index_query,
    key_cache,
    value_cache,
    index_cache,
    slots,
    index_slots,
    NUM_SLOTS,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    IQ_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    INPUT_STRIDE: tl.constexpr,
    INDEX_INPUT_STRIDE: tl.constexpr,
    INDEX_OFFSET: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    COS_SIN_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    K_BLOCK: tl.constexpr,
    K_TOKEN: tl.constexpr,
    K_HEAD: tl.constexpr,
    V_BLOCK: tl.constexpr,
    V_TOKEN: tl.constexpr,
    V_HEAD: tl.constexpr,
    I_BLOCK: tl.constexpr,
    I_TOKEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INDEX_BLOCK_SIZE: tl.constexpr,
    CACHE_FP8: tl.constexpr,
    INDEX_FP8: tl.constexpr,
    FP8_E4M3_MAX: tl.constexpr,
    T: tl.constexpr,
    NCHUNK: tl.constexpr,
    NSTAGE: tl.constexpr,
):
    cols = tl.arange(0, HEAD_DIM)
    # Loop-invariant per-program state: the four weight vectors, loaded once.
    wq = tl.load(q_weight + cols).to(tl.float32)
    wk = tl.load(k_weight + cols).to(tl.float32)
    wiq = tl.load(iq_weight + cols).to(tl.float32)
    wik = tl.load(ik_weight + cols).to(tl.float32)
    tok_base = tl.program_id(0) * (T * NCHUNK)

    for c in tl.range(0, NCHUNK, num_stages=NSTAGE):
        _chunk(
            tok_base + c * T,
            packed,
            cos_sin,
            positions,
            wq,
            wk,
            wiq,
            wik,
            query,
            index_query,
            key_cache,
            value_cache,
            index_cache,
            slots,
            index_slots,
            NUM_SLOTS,
            Q_HEADS,
            KV_HEADS,
            IQ_HEADS,
            HEAD_DIM,
            ROTARY_DIM,
            INPUT_STRIDE,
            POSITION_STRIDE,
            COS_SIN_STRIDE,
            EPS,
            K_BLOCK,
            K_TOKEN,
            V_BLOCK,
            V_TOKEN,
            I_BLOCK,
            I_TOKEN,
            BLOCK_SIZE,
            INDEX_BLOCK_SIZE,
            CACHE_FP8,
            INDEX_FP8,
            FP8_E4M3_MAX,
            T,
        )


def _launch(
    packed,
    index_packed,
    cos_sin,
    positions,
    q_weight,
    k_weight,
    iq_weight,
    ik_weight,
    query,
    index_query,
    key_cache,
    value_cache,
    index_cache,
    slots,
    index_slots,
    NUM_SLOTS,
    Q_HEADS,
    KV_HEADS,
    IQ_HEADS,
    HEAD_DIM,
    ROTARY_DIM,
    INPUT_STRIDE,
    INDEX_INPUT_STRIDE,
    INDEX_OFFSET,
    POSITION_STRIDE,
    COS_SIN_STRIDE,
    EPS,
    K_BLOCK,
    K_TOKEN,
    K_HEAD,
    V_BLOCK,
    V_TOKEN,
    V_HEAD,
    I_BLOCK,
    I_TOKEN,
    BLOCK_SIZE,
    INDEX_BLOCK_SIZE,
    CACHE_FP8,
    INDEX_FP8,
    FP8_E4M3_MAX,
    *,
    enable_fp_fusion,
    grid,
    warmup,
):
    num_tokens = query.shape[0]
    if num_tokens == 0:
        return

    args = (
        packed,
        index_packed,
        cos_sin,
        positions,
        q_weight,
        k_weight,
        iq_weight,
        ik_weight,
        query,
        index_query,
        key_cache,
        value_cache,
        index_cache,
        slots,
        index_slots,
        NUM_SLOTS,
        Q_HEADS,
        KV_HEADS,
        IQ_HEADS,
        HEAD_DIM,
        ROTARY_DIM,
        INPUT_STRIDE,
        INDEX_INPUT_STRIDE,
        INDEX_OFFSET,
        POSITION_STRIDE,
        COS_SIN_STRIDE,
        EPS,
        K_BLOCK,
        K_TOKEN,
        K_HEAD,
        V_BLOCK,
        V_TOKEN,
        V_HEAD,
        I_BLOCK,
        I_TOKEN,
        BLOCK_SIZE,
        INDEX_BLOCK_SIZE,
        CACHE_FP8,
        INDEX_FP8,
        FP8_E4M3_MAX,
    )

    # The tiled path reads the k/v/index-q/index-k rows as one affine burst, so it
    # requires those four rows to be physically contiguous after the q rows.
    tiled = (
        KV_HEADS == 1
        and IQ_HEADS == 1
        and packed.data_ptr() == index_packed.data_ptr()
        and INDEX_INPUT_STRIDE == INPUT_STRIDE
        and INDEX_OFFSET == (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
        and INDEX_OFFSET + (IQ_HEADS + 1) * HEAD_DIM <= INPUT_STRIDE
        and (ROTARY_DIM & (ROTARY_DIM - 1)) == 0
        and ROTARY_DIM == 2 * (ROTARY_DIM // 2)
        and 2 * (ROTARY_DIM // 2) <= HEAD_DIM
        and num_tokens % (_T * _NCHUNK) == 0
        and num_tokens * INPUT_STRIDE < 2**31
    )

    if tiled:
        _kernel_tiled[(num_tokens // (_T * _NCHUNK),)](
            *args, T=_T, NCHUNK=_NCHUNK, NSTAGE=_NSTAGE
        )
    else:
        _kernel_single[(num_tokens,)](*args)


def minimax_qknorm_rope_cache(
    packed,
    cos_sin_cache,
    positions,
    q_weight,
    k_weight,
    index_q_weight,
    index_k_weight,
    key_cache,
    value_cache,
    index_cache,
    slot_mapping,
    index_slot_mapping,
    num_actual_tokens,
    num_q_heads,
    num_kv_heads,
    num_index_heads,
    eps,
    index_packed=None,
):
    """Return Q/index-Q and update caller-owned BF16 main and index caches.

    Position IDs must index the supplied trig table. Each nonnegative active
    slot names a unique cache row; negative slots skip the corresponding write.
    """
    if (
        packed.device.type != "npu"
        or packed.ndim != 2
        or packed.dtype != torch.bfloat16
        or not packed.is_contiguous()
    ):
        raise ValueError("expected a contiguous 2D BF16 packed tensor on NPU")
    if (num_q_heads, num_kv_heads, num_index_heads) != (16, 1, 1):
        raise ValueError("supported head counts are (16, 1, 1)")
    tokens = packed.shape[0]
    width = 128
    main_width = (num_q_heads + 2 * num_kv_heads) * width
    index_width = (num_index_heads + 1) * width
    if index_packed is None:
        if packed.shape[1] != main_width + index_width:
            raise ValueError("invalid combined packed width")
        index_packed = packed
        offset = main_width
    else:
        if (
            packed.shape[1] != main_width
            or index_packed.shape != (tokens, index_width)
            or index_packed.dtype != packed.dtype
        ):
            raise ValueError("invalid separate index input")
        offset = 0
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[-1] != 64
        or cos_sin_cache.shape[0] == 0
        or positions.shape != (tokens,)
        or positions.dtype not in (torch.int64, torch.int32)
    ):
        raise ValueError("expected a 64-wide trig table and integer position IDs")
    weights = (q_weight, k_weight, index_q_weight, index_k_weight)
    if any(t.shape != (width,) or t.dtype != torch.bfloat16 for t in weights):
        raise ValueError("normalization weights must be BF16 vectors of width 128")
    if (
        key_cache.ndim != 4
        or key_cache.shape[1:] != (128, 1, 128)
        or value_cache.shape != key_cache.shape
        or index_cache.ndim != 3
        or index_cache.shape[1:] != (128, 128)
    ):
        raise ValueError("expected 128-token BF16 cache pages")
    if any(
        t.dtype != packed.dtype
        for t in (key_cache, value_cache, index_cache, cos_sin_cache)
    ):
        raise ValueError("all caches and trig table must be BF16")
    if not 0 <= num_actual_tokens <= tokens or any(
        t.ndim != 1
        or t.numel() < num_actual_tokens
        or t.dtype not in (torch.int32, torch.int64)
        for t in (slot_mapping, index_slot_mapping)
    ):
        raise ValueError("slot mappings must cover every active token")
    if any(
        t.device != packed.device or not t.is_contiguous()
        for t in (
            *weights,
            index_packed,
            cos_sin_cache,
            positions,
            slot_mapping,
            index_slot_mapping,
            key_cache,
            value_cache,
            index_cache,
        )
    ):
        raise ValueError("expected contiguous inputs on one NPU")
    q = torch.empty(
        (tokens, num_q_heads * width), dtype=packed.dtype, device=packed.device
    )
    iq = torch.empty((tokens, width), dtype=packed.dtype, device=packed.device)
    with torch_device_fn.device(packed.device):
        _launch(
            packed,
            index_packed,
            cos_sin_cache,
            positions,
            *weights,
            q,
            iq,
            key_cache,
            value_cache,
            index_cache,
            slot_mapping,
            index_slot_mapping,
            num_actual_tokens,
            num_q_heads,
            num_kv_heads,
            num_index_heads,
            width,
            64,
            packed.stride(0),
            index_packed.stride(0),
            offset,
            positions.stride(0),
            cos_sin_cache.stride(0),
            eps,
            *key_cache.stride()[:3],
            *value_cache.stride()[:3],
            *index_cache.stride()[:2],
            128,
            128,
            False,
            False,
            448.0,
            grid=(tokens,),
            warmup=False,
            enable_fp_fusion=False,
        )
    return q, iq
