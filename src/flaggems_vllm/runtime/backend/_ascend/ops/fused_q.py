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

"""Ascend specialization for the full BF16 fused_q workload.

The fixed four-row and 256-column NoPE tiles were validated on Ascend 910B4-1.
Use one launch for the nine small workloads and split launches for other
supported token counts. These fixed Ascend schedules do not autotune.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

_ROW_TILE = 4
_NOPE_TILE = 256
_MAX_PROGRAMS = 40
_SMALL_TOKEN_COUNTS = (1, 6, 24)
_SUPPORTED_HEADS = (16, 32, 64)


@triton.jit
def _encode_bias_bytes(values):
    bits = values.to(tl.float32).to(tl.int32, bitcast=True)
    one = tl.full((), 1, tl.int32)
    magnitude = bits & tl.full((), 0x7FFFFFFF, tl.int32)
    sign = (bits >> tl.full((), 24, tl.int32)) & tl.full((), 128, tl.int32)
    shift = tl.full((), 20, tl.int32)
    normal = (
        (magnitude + tl.full((), 0x7FFFF, tl.int32) + ((magnitude >> shift) & one))
        >> shift
    ) - tl.full((), 960, tl.int32)
    # At 2**14, FP32 ULP is 2**-9, the E4M3FN subnormal spacing.
    # FP32 addition rounds to nearest-even. The integer bit difference is
    # the rounded subnormal code, including the carry into the first normal.
    absolute = magnitude.to(tl.float32, bitcast=True)
    rounded = absolute + tl.full((), 16384.0, tl.float32)
    subnormal = rounded.to(tl.int32, bitcast=True) - tl.full((), 0x46800000, tl.int32)
    code = tl.where(magnitude >= tl.full((), 0x3C800000, tl.int32), normal, subnormal)
    code = tl.where(
        magnitude > tl.full((), 0x43E80000, tl.int32), tl.full((), 127, tl.int32), code
    )
    return (code | sign).to(tl.uint8)


@triton.jit
def _rope_contiguous(
    x,
    cos,
    sin,
    BR: tl.constexpr,
    DIM: tl.constexpr,
    ROT: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    """Rotate the first ROT values after a single contiguous GM input load."""
    tl.static_assert(DIM % ROT == 0, "rotation must divide the contiguous tile")
    if INTERLEAVED:
        # MQA: keep the loaded tensor two-dimensional. The previous 4D
        # reshape/split/join route compiled but failed NPU validation; this
        # isolates that lowering while preserving the native arithmetic order.
        col = tl.arange(0, DIM)
        rotating = col < ROT
        partner_col = tl.where(rotating, col ^ 1, col)
        partner = tl.gather(x, tl.broadcast_to(partner_col[None, :], (BR, DIM)), axis=1)
        cache_col = tl.minimum(col // 2, ROT // 2 - 1)
        cache_indices = tl.broadcast_to(cache_col[None, :], (BR, DIM))
        expanded_cos = tl.gather(cos, cache_indices, axis=1)
        expanded_sin = tl.gather(sin, cache_indices, axis=1)
        sign = tl.where(col % 2 == 0, -1.0, 1.0)
        rotated = x * expanded_cos + sign[None, :] * partner * expanded_sin
        return tl.where(rotating[None, :], rotated, x)
    else:
        # Indexer NeoX retains the previously validated path for this ablation.
        pairs = tl.trans(tl.reshape(x, (BR, DIM // ROT, 2, ROT // 2)), (0, 1, 3, 2))
        # Keep every use of pairs in this constexpr branch. Ascend's frontend
        # still resolves statements after an early return in the other branch.
        a, b = tl.split(pairs)
        first = a * cos[:, None, :] - b * sin[:, None, :]
        second = b * cos[:, None, :] + a * sin[:, None, :]
        rotating = tl.arange(0, DIM // ROT) == 0
        first = tl.where(rotating[None, :, None], first, a)
        second = tl.where(rotating[None, :, None], second, b)
        merged = tl.join(first, second)
        merged = tl.trans(merged, (0, 1, 3, 2))
        return tl.reshape(merged, (BR, DIM))


@triton.jit
def _encode_e4m3fn_bytes(values):
    """FP32 -> OCP E4M3FN bytes, nearest-even, including gradual underflow.

    Integer rounding avoids FP8 types/instructions in the compiled software path.
    The caller passes uint8 views of FP8 output storage, preserving the public ABI.
    """
    # Keep every operand explicitly signed i32. Mixing uint32 tensors with
    # Python integer literals can introduce a uint32 -> uint64 promotion that
    # Ascend910B's BiShengHIR vcast lowering rejects. Arithmetic right shift is
    # safe here: magnitude is positive, and the sign extraction is masked.
    one = tl.full((), 1, tl.int32)
    bits = values.to(tl.float32).to(tl.int32, bitcast=True)
    magnitude = bits & tl.full((), 0x7FFFFFFF, tl.int32)
    sign = (bits >> tl.full((), 24, tl.int32)) & tl.full((), 0x80, tl.int32)
    mantissa_shift = tl.full((), 20, tl.int32)
    normal = (
        (
            magnitude
            + tl.full((), 0x7FFFF, tl.int32)
            + ((magnitude >> mantissa_shift) & one)
        )
        >> mantissa_shift
    ) - tl.full((), 960, tl.int32)
    exponent = (magnitude >> tl.full((), 23, tl.int32)) & tl.full((), 0xFF, tl.int32)
    # E4M3FN subnormal unit is 2**-9. For FP32 mantissa M, x*512 =
    # M * 2**(exponent-141). Clamp shifts on unused normal/very tiny lanes.
    shift = tl.minimum(
        tl.maximum(tl.full((), 141, tl.int32) - exponent, one),
        tl.full((), 25, tl.int32),
    )
    mantissa = (magnitude & tl.full((), 0x7FFFFF, tl.int32)) | tl.full(
        (), 0x800000, tl.int32
    )
    halfway = one << (shift - one)
    subnormal = (mantissa + halfway - one + ((mantissa >> shift) & one)) >> shift
    code = tl.where(magnitude >= tl.full((), 0x3C800000, tl.int32), normal, subnormal)
    # 464 is the last tie, rounded down to 448 (even). Larger values, Inf
    # and NaN become the format's NaN code; preserve the input sign bit.
    code = tl.where(
        magnitude > tl.full((), 0x43E80000, tl.int32), tl.full((), 127, tl.int32), code
    )
    return (code | sign).to(tl.uint8)


encode_bias = _encode_bias_bytes
encode_shift = _encode_e4m3fn_bytes


rope_contiguous = _rope_contiguous


@triton.jit
def cast_fp8(values, USE_BIAS: tl.constexpr):
    if USE_BIAS:
        return encode_bias(values)
    else:
        return encode_shift(values)


@triton.jit
def fused_q_kernel(
    POS,
    QPE,
    QCACHE,
    INDEX,
    ICACHE,
    NOPE,
    SCALE,
    WEIGHTS,
    IQOUT,
    IWOUT,
    OUT,
    softmax_scale,
    head_scale,
    N: tl.constexpr,
    H: tl.constexpr,
    IH: tl.constexpr,
    R: tl.constexpr,
    ID: tl.constexpr,
    D: tl.constexpr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    cs0: tl.constexpr,
    is0: tl.constexpr,
    is1: tl.constexpr,
    ics0: tl.constexpr,
    ws0: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
    PART: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    rows: tl.constexpr = N * (IH if PART == 1 else H)
    columns: tl.constexpr = tl.cdiv(D, BN) if PART == 2 else 1
    row_offsets = tl.arange(0, BR).to(tl.int64)
    for task in range(
        tl.program_id(0), tl.cdiv(rows, BR) * columns, tl.num_programs(0)
    ):
        row = (task // columns).to(tl.int64) * BR + row_offsets
        if PART == 0:
            q_token, q_head = row // H, row % H
            q_valid = q_token < N
            q_pos = tl.load(POS + q_token, q_valid, other=0)
            q_col = tl.arange(0, R)
            q_half = tl.arange(0, R // 2)
            q_base = QPE + q_token[:, None] * qs0 + q_head[:, None] * qs1
            q_x = tl.load(q_base + q_col[None, :], q_valid[:, None], other=0).to(
                tl.float32
            )
            q_cache = QCACHE + q_pos[:, None] * cs0 + q_half[None, :]
            q_cos = tl.load(q_cache, q_valid[:, None], other=1).to(tl.float32)
            q_sin = tl.load(q_cache + R // 2, q_valid[:, None], other=0).to(tl.float32)
            q_rotated = rope_contiguous(q_x, q_cos, q_sin, BR, R, R, True)
            q_scale = tl.load(SCALE)
            q_byte = cast_fp8(q_rotated / q_scale, USE_BIAS)
            tl.store(
                OUT + row[:, None] * (D + R) + D + q_col[None, :],
                q_byte,
                q_valid[:, None],
            )
        elif PART == 1:
            i_token, i_head = row // IH, row % IH
            i_valid = i_token < N
            i_pos = tl.load(POS + i_token, i_valid, other=0)
            i_col = tl.arange(0, ID)
            i_half = tl.arange(0, R // 2)
            i_base = INDEX + i_token[:, None] * is0 + i_head[:, None] * is1
            i_x = tl.load(i_base + i_col[None, :], i_valid[:, None], other=0).to(
                tl.float32
            )
            i_cache = ICACHE + i_pos[:, None] * ics0 + i_half[None, :]
            i_cos = tl.load(i_cache, i_valid[:, None], other=1).to(tl.float32)
            i_sin = tl.load(i_cache + R // 2, i_valid[:, None], other=0).to(tl.float32)
            i_rotated = rope_contiguous(i_x, i_cos, i_sin, BR, ID, R, False)
            i_scale = tl.div_rn(
                tl.maximum(tl.max(tl.abs(i_rotated), axis=1), 1e-4), 448.0
            )
            i_scale = tl.exp2(tl.ceil(tl.log2(i_scale)))
            i_byte = cast_fp8(tl.div_rn(i_rotated, i_scale[:, None]), USE_BIAS)
            tl.store(
                IQOUT + row[:, None] * ID + i_col[None, :], i_byte, i_valid[:, None]
            )
            i_weight = tl.load(WEIGHTS + i_token * ws0 + i_head, i_valid, other=0).to(
                tl.float32
            )
            i_weight = i_weight * i_scale
            i_weight = i_weight * softmax_scale
            i_weight = i_weight * head_scale
            tl.store(IWOUT + row, i_weight, i_valid)
        else:
            no_col = (task % columns) * BN + tl.arange(0, BN)
            no_valid = (row[:, None] < rows) & (no_col[None, :] < D)
            no_x = tl.load(
                NOPE + row[:, None] * D + no_col[None, :], no_valid, other=0
            ).to(tl.float32)
            no_scale = tl.load(SCALE)
            no_byte = cast_fp8(no_x / no_scale, USE_BIAS)
            tl.store(OUT + row[:, None] * (D + R) + no_col[None, :], no_byte, no_valid)


@triton.jit
def load_tile(
    POS,
    INDEX,
    ICACHE,
    WEIGHTS,
    row_begin,
    N: tl.constexpr,
    IH: tl.constexpr,
    R: tl.constexpr,
    ID: tl.constexpr,
    is0: tl.constexpr,
    is1: tl.constexpr,
    ics0: tl.constexpr,
    ws0: tl.constexpr,
    BR: tl.constexpr,
):
    row = row_begin + tl.arange(0, BR).to(tl.int64)
    token, head = row // IH, row % IH
    valid = token < N
    col = tl.arange(0, ID)
    x = tl.load(
        INDEX + token[:, None] * is0 + head[:, None] * is1 + col[None, :],
        valid[:, None],
        other=0,
    ).to(tl.float32)
    cache_token = row_begin // IH
    cache_valid = cache_token < N
    position = tl.load(POS + cache_token, cache_valid, other=0)
    half = tl.arange(0, R // 2)
    base = ICACHE + position * ics0 + half
    cos = tl.load(base, cache_valid, other=1).to(tl.float32)
    sin = tl.load(base + R // 2, cache_valid, other=0).to(tl.float32)
    weight = tl.load(WEIGHTS + token * ws0 + head, valid, other=0).to(tl.float32)
    return (
        x,
        tl.broadcast_to(cos[None, :], (BR, R // 2)),
        tl.broadcast_to(sin[None, :], (BR, R // 2)),
        weight,
    )


@triton.jit
def direct_rope(x, cos, sin, BR: tl.constexpr, ID: tl.constexpr, R: tl.constexpr):
    tl.static_assert(
        ID == 2 * R, "direct layout requires one rotating and one unchanged half"
    )
    half = tl.arange(0, R // 2)
    left = tl.gather(x, tl.broadcast_to(half[None, :], (BR, R // 2)), axis=1)
    right = tl.gather(
        x, tl.broadcast_to((half + R // 2)[None, :], (BR, R // 2)), axis=1
    )
    first = left * cos - right * sin
    second = right * cos + left * sin
    col = tl.arange(0, ID)
    indices = tl.broadcast_to((col % (R // 2))[None, :], (BR, ID))
    first_full = tl.gather(first, indices, axis=1)
    second_full = tl.gather(second, indices, axis=1)
    return tl.where(
        col[None, :] < R // 2, first_full, tl.where(col[None, :] < R, second_full, x)
    )


@triton.jit
def store_tile(
    x,
    cos,
    sin,
    weight,
    row_begin,
    IQOUT,
    IWOUT,
    softmax_scale,
    head_scale,
    BR: tl.constexpr,
    ID: tl.constexpr,
    R: tl.constexpr,
    DIRECT_ROPE: tl.constexpr,
):
    if DIRECT_ROPE:
        rotated = direct_rope(x, cos, sin, BR, ID, R)
    else:
        rotated = rope_contiguous(x, cos, sin, BR, ID, R, False)
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(rotated), axis=1), 1e-4), 448.0)
    scale = tl.exp2(tl.ceil(tl.log2(scale)))
    quantized = encode_bias(tl.div_rn(rotated, scale[:, None]))
    row = row_begin + tl.arange(0, BR).to(tl.int64)
    col = tl.arange(0, ID)
    tl.store(IQOUT + row[:, None] * ID + col[None, :], quantized)
    weight = weight * scale
    weight = weight * softmax_scale
    weight = weight * head_scale
    tl.store(IWOUT + row, weight)


@triton.jit
def indexer_kernel(
    POS,
    QPE,
    QCACHE,
    INDEX,
    ICACHE,
    NOPE,
    SCALE,
    WEIGHTS,
    IQOUT,
    IWOUT,
    OUT,
    softmax_scale,
    head_scale,
    N: tl.constexpr,
    H: tl.constexpr,
    IH: tl.constexpr,
    R: tl.constexpr,
    ID: tl.constexpr,
    D: tl.constexpr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    cs0: tl.constexpr,
    is0: tl.constexpr,
    is1: tl.constexpr,
    ics0: tl.constexpr,
    ws0: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
    PART: tl.constexpr,
    UNROLL: tl.constexpr,
    LOOKAHEAD: tl.constexpr,
    DIRECT_ROPE: tl.constexpr,
):
    tl.static_assert(PART == 1, "Indexer kernel requires PART=1")
    tl.static_assert(IH % BR == 0, "shared cache tile must stay within one token")
    tl.static_assert(N * IH % (BR * UNROLL) == 0, "complete tile groups required")
    tl.static_assert(UNROLL == 1 or UNROLL == 2, "only one or two blocks supported")
    tl.static_assert(not LOOKAHEAD or UNROLL == 2, "lookahead requires two blocks")
    for job in range(
        tl.program_id(0), tl.cdiv(N * IH, BR * UNROLL), tl.num_programs(0)
    ):
        begin = job.to(tl.int64) * BR * UNROLL
        if LOOKAHEAD:
            x0, c0, s0, w0 = load_tile(
                POS,
                INDEX,
                ICACHE,
                WEIGHTS,
                begin,
                N,
                IH,
                R,
                ID,
                is0,
                is1,
                ics0,
                ws0,
                BR,
            )
            x1, c1, s1, w1 = load_tile(
                POS,
                INDEX,
                ICACHE,
                WEIGHTS,
                begin + BR,
                N,
                IH,
                R,
                ID,
                is0,
                is1,
                ics0,
                ws0,
                BR,
            )
            store_tile(
                x0,
                c0,
                s0,
                w0,
                begin,
                IQOUT,
                IWOUT,
                softmax_scale,
                head_scale,
                BR,
                ID,
                R,
                DIRECT_ROPE,
            )
            store_tile(
                x1,
                c1,
                s1,
                w1,
                begin + BR,
                IQOUT,
                IWOUT,
                softmax_scale,
                head_scale,
                BR,
                ID,
                R,
                DIRECT_ROPE,
            )
        else:
            for lane in tl.static_range(0, UNROLL):
                row_begin = begin + lane * BR
                x, cos, sin, weight = load_tile(
                    POS,
                    INDEX,
                    ICACHE,
                    WEIGHTS,
                    row_begin,
                    N,
                    IH,
                    R,
                    ID,
                    is0,
                    is1,
                    ics0,
                    ws0,
                    BR,
                )
                store_tile(
                    x,
                    cos,
                    sin,
                    weight,
                    row_begin,
                    IQOUT,
                    IWOUT,
                    softmax_scale,
                    head_scale,
                    BR,
                    ID,
                    R,
                    DIRECT_ROPE,
                )


@triton.jit
def mqa_kernel(
    POS,
    QPE,
    QCACHE,
    INDEX,
    ICACHE,
    NOPE,
    SCALE,
    WEIGHTS,
    IQOUT,
    IWOUT,
    OUT,
    softmax_scale,
    head_scale,
    N: tl.constexpr,
    H: tl.constexpr,
    IH: tl.constexpr,
    R: tl.constexpr,
    ID: tl.constexpr,
    D: tl.constexpr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    cs0: tl.constexpr,
    is0: tl.constexpr,
    is1: tl.constexpr,
    ics0: tl.constexpr,
    ws0: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
    PART: tl.constexpr,
    SHARED_CACHE: tl.constexpr,
    DIRECT_PAIRS: tl.constexpr,
    FULL_STORE: tl.constexpr,
):
    tl.static_assert(PART == 0, "MQA kernel requires PART=0")
    tl.static_assert(R % 2 == 0, "interleaved RoPE requires even width")
    if SHARED_CACHE:
        tl.static_assert(
            H % BR == 0, "shared MQA cache tile must stay within one token"
        )
    if FULL_STORE:
        tl.static_assert(
            N * H % BR == 0, "unmasked MQA store requires complete row tiles"
        )
    offsets = tl.arange(0, BR).to(tl.int64)
    col = tl.arange(0, R)
    half = tl.arange(0, R // 2)
    for task in range(tl.program_id(0), tl.cdiv(N * H, BR), tl.num_programs(0)):
        begin = task.to(tl.int64) * BR
        row = begin + offsets
        token, head = row // H, row % H
        valid = token < N
        if SHARED_CACHE:
            cache_token = begin // H
            cache_valid = cache_token < N
            position = tl.load(POS + cache_token, cache_valid, other=0)
        else:
            position = tl.load(POS + token, valid, other=0)
        x = tl.load(
            QPE + token[:, None] * qs0 + head[:, None] * qs1 + col[None, :],
            valid[:, None],
            other=0,
        ).to(tl.float32)
        if SHARED_CACHE:
            base = QCACHE + position * cs0 + half
            shared_cos = tl.load(base, cache_valid, other=1).to(tl.float32)
            shared_sin = tl.load(base + R // 2, cache_valid, other=0).to(tl.float32)
            cos = tl.broadcast_to(shared_cos[None, :], (BR, R // 2))
            sin = tl.broadcast_to(shared_sin[None, :], (BR, R // 2))
        else:
            base = QCACHE + position[:, None] * cs0 + half[None, :]
            cos = tl.load(base, valid[:, None], other=1).to(tl.float32)
            sin = tl.load(base + R // 2, valid[:, None], other=0).to(tl.float32)
        if DIRECT_PAIRS:
            even = tl.gather(
                x, tl.broadcast_to((half * 2)[None, :], (BR, R // 2)), axis=1
            )
            odd = tl.gather(
                x, tl.broadcast_to((half * 2 + 1)[None, :], (BR, R // 2)), axis=1
            )
            first = even * cos + (-1.0 * odd) * sin
            second = odd * cos + (1.0 * even) * sin
            # Keep reconstruction 2D. join/reshape was lowered into
            # 4x32x2 UB buffers with alignment failures on Ascend910B4.
            indices = tl.broadcast_to((col // 2)[None, :], (BR, R))
            first_full = tl.gather(first, indices, axis=1)
            second_full = tl.gather(second, indices, axis=1)
            rotated = tl.where(col[None, :] % 2 == 0, first_full, second_full)
        else:
            rotated = rope_contiguous(x, cos, sin, BR, R, R, True)
        scale = tl.load(SCALE)
        quantized = encode_bias(rotated / scale)
        target = OUT + row[:, None] * (D + R) + D + col[None, :]
        if FULL_STORE:
            tl.store(target, quantized)
        else:
            tl.store(target, quantized, valid[:, None])


index_load = load_tile
index_store = store_tile


@triton.jit
def small_fused_q_kernel(
    POS,
    QPE,
    QCACHE,
    INDEX,
    ICACHE,
    NOPE,
    SCALE,
    WEIGHTS,
    IQOUT,
    IWOUT,
    OUT,
    softmax_scale,
    head_scale,
    N: tl.constexpr,
    H: tl.constexpr,
    IH: tl.constexpr,
    R: tl.constexpr,
    ID: tl.constexpr,
    D: tl.constexpr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    cs0: tl.constexpr,
    is0: tl.constexpr,
    is1: tl.constexpr,
    ics0: tl.constexpr,
    ws0: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
):
    tl.static_assert(
        N == 1 or N == 6 or N == 24, "only the targeted token counts are supported"
    )
    tl.static_assert(H % BR == 0 and IH % BR == 0, "tiles must stay within tokens")
    tl.static_assert(D % BN == 0, "NoPE requires complete column tiles")
    mqa_tiles: tl.constexpr = N * H // BR
    index_tiles: tl.constexpr = N * IH // BR
    nope_columns: tl.constexpr = D // BN
    total: tl.constexpr = mqa_tiles + index_tiles + mqa_tiles * nope_columns
    for task in range(tl.program_id(0), total, tl.num_programs(0)):
        if task < mqa_tiles:
            begin = task.to(tl.int64) * BR
            row = begin + tl.arange(0, BR).to(tl.int64)
            token, head = row // H, row % H
            valid = token < N
            cache_token = begin // H
            cache_valid = cache_token < N
            position = tl.load(POS + cache_token, cache_valid, other=0)
            col = tl.arange(0, R)
            x = tl.load(
                QPE + token[:, None] * qs0 + head[:, None] * qs1 + col[None, :],
                valid[:, None],
                other=0,
            ).to(tl.float32)
            half = tl.arange(0, R // 2)
            cache = QCACHE + position * cs0 + half
            cos = tl.load(cache, cache_valid, other=1).to(tl.float32)
            sin = tl.load(cache + R // 2, cache_valid, other=0).to(tl.float32)
            cos = tl.broadcast_to(cos[None, :], (BR, R // 2))
            sin = tl.broadcast_to(sin[None, :], (BR, R // 2))
            rotated = rope_contiguous(x, cos, sin, BR, R, R, True)
            quantized = encode_bias(rotated / tl.load(SCALE))
            tl.store(OUT + row[:, None] * (D + R) + D + col[None, :], quantized)
        elif task < mqa_tiles + index_tiles:
            index_begin = (task - mqa_tiles).to(tl.int64) * BR
            ix, ic, isin, iw = index_load(
                POS,
                INDEX,
                ICACHE,
                WEIGHTS,
                index_begin,
                N,
                IH,
                R,
                ID,
                is0,
                is1,
                ics0,
                ws0,
                BR,
            )
            index_store(
                ix,
                ic,
                isin,
                iw,
                index_begin,
                IQOUT,
                IWOUT,
                softmax_scale,
                head_scale,
                BR,
                ID,
                R,
                True,
            )
        else:
            local_task = task - mqa_tiles - index_tiles
            no_row = (local_task // nope_columns).to(tl.int64) * BR + tl.arange(
                0, BR
            ).to(tl.int64)
            no_col = (local_task % nope_columns) * BN + tl.arange(0, BN)
            no_valid = (no_row[:, None] < N * H) & (no_col[None, :] < D)
            no_x = tl.load(
                NOPE + no_row[:, None] * D + no_col[None, :], no_valid, other=0
            ).to(tl.float32)
            no_byte = encode_bias(no_x / tl.load(SCALE))
            tl.store(
                OUT + no_row[:, None] * (D + R) + no_col[None, :], no_byte, no_valid
            )


def _signature(inputs):
    return tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in inputs)


@lru_cache(maxsize=16)
def _program_limit(device):
    properties = triton.runtime.driver.active.utils.get_device_properties(device.index)
    vector_cores = properties.get("num_vectorcore")
    if isinstance(vector_cores, int) and vector_cores > 0:
        return min(_MAX_PROGRAMS, vector_cores)
    return _MAX_PROGRAMS


@lru_cache(maxsize=128)
def _make_plan(signature, program_limit):
    positions, q_pe, q_cache, index_q, index_cache, nope, scale, weights = signature
    q_shape, q_stride, _, device = q_pe
    n, h, r = q_shape
    index_heads, index_dim = 32, 128
    nope_dim, max_pos = 512, 16384
    expected = (
        ((n,), torch.int64),
        ((n, h, 64), torch.bfloat16),
        ((max_pos, 64), torch.bfloat16),
        ((n, index_heads, index_dim), torch.bfloat16),
        ((max_pos, 64), torch.bfloat16),
        ((n, h, nope_dim), torch.bfloat16),
        (scale[0], torch.float32),
        ((n, index_heads), torch.float32),
    )
    for (shape, stride, dtype, tensor_device), (expected_shape, expected_dtype) in zip(
        signature, expected
    ):
        if shape != expected_shape or dtype != expected_dtype:
            raise ValueError(
                "Input shapes and dtypes must match the BF16 fused_q workload"
            )
        if tensor_device != device:
            raise ValueError("All fused_q inputs must be on the same NPU")
        if stride and stride[-1] != 1:
            raise ValueError("fused_q requires unit innermost strides")

    if (
        index_q[1] != (index_heads * index_dim, index_dim, 1)
        or nope[1] != (h * nope_dim, nope_dim, 1)
        or weights[1] != (index_heads, 1)
    ):
        raise ValueError("Indexer query, index weights and NoPE must be contiguous")

    metadata = dict(
        N=n,
        H=h,
        IH=index_heads,
        R=r,
        ID=index_dim,
        D=nope_dim,
        qs0=q_stride[0],
        qs1=q_stride[1],
        cs0=q_cache[1][0],
        is0=index_q[1][0],
        is1=index_q[1][1],
        ics0=index_cache[1][0],
        ws0=weights[1][0],
        BR=_ROW_TILE,
        BN=_NOPE_TILE,
        num_warps=1,
        num_stages=1,
        multibuffer=False,
    )
    q_tiles = n * h // _ROW_TILE
    index_tiles = n * index_heads // _ROW_TILE
    nope_tiles = q_tiles * (nope_dim // _NOPE_TILE)
    if n in _SMALL_TOKEN_COUNTS:
        grids = ((min(program_limit, q_tiles + index_tiles + nope_tiles),),)
    else:
        grids = tuple(
            (min(program_limit, tiles),) for tiles in (q_tiles, index_tiles, nope_tiles)
        )
    return grids, metadata


def fused_q(
    positions,
    q_pe,
    q_pe_cos_sin_cache,
    index_q,
    index_q_cos_sin_cache,
    ql_nope,
    q_scale,
    index_weights,
    index_weights_softmax_scale,
    index_weights_head_scale,
    has_indexer=True,
    index_rope_interleave=False,
    quantize_mqa=True,
):
    """Return indexer FP8, FP32 weights and packed MQA FP8 on Ascend.

    The Ascend specialization supports the default quantized, NeoX indexer
    configuration. Positions must be within both RoPE caches and q_scale
    must be finite and positive.
    """
    if not has_indexer or index_rope_interleave or not quantize_mqa:
        raise NotImplementedError(
            "Ascend fused_q requires the default quantized indexer mode"
        )
    device = q_pe.device
    if device.type != "npu":
        raise ValueError("Ascend fused_q requires an NPU query")
    if q_pe.ndim != 3:
        raise ValueError("q_pe must have shape [N, H, 64]")
    n, h, r = q_pe.shape
    if n <= 0 or h not in _SUPPORTED_HEADS or r != 64:
        raise NotImplementedError(
            "Ascend fused_q requires N > 0 and H=16, 32 or 64 with RoPE=64"
        )
    inputs = (
        positions,
        q_pe,
        q_pe_cos_sin_cache,
        index_q,
        index_q_cos_sin_cache,
        ql_nope,
        q_scale,
        index_weights,
    )
    if any(t is None for t in inputs):
        raise ValueError("All eight fused_q tensors are required in the default mode")
    if q_scale.numel() != 1:
        raise ValueError("q_scale must contain one FP32 value")

    with torch_device_fn.device(device):
        grids, metadata = _make_plan(_signature(inputs), _program_limit(device))
        index_out = torch.empty(index_q.shape, dtype=torch.float8_e4m3fn, device=device)
        weights_out = torch.empty(
            index_weights.shape, dtype=torch.float32, device=device
        )
        mqa_out = torch.empty((n, h, 576), dtype=torch.float8_e4m3fn, device=device)
        args = (
            *inputs,
            index_out.view(torch.uint8),
            weights_out,
            mqa_out.view(torch.uint8),
            index_weights_softmax_scale,
            index_weights_head_scale,
        )
        if len(grids) == 1:
            small_fused_q_kernel[grids[0]](*args, **metadata)
        else:
            mqa_kernel[grids[0]](
                *args,
                **metadata,
                PART=0,
                SHARED_CACHE=True,
                DIRECT_PAIRS=False,
                FULL_STORE=True,
            )
            indexer_kernel[grids[1]](
                *args, **metadata, PART=1, UNROLL=1, LOOKAHEAD=False, DIRECT_ROPE=True
            )
            fused_q_kernel[grids[2]](*args, **metadata, PART=2, USE_BIAS=True)
    return index_out, weights_out, mqa_out


__all__ = ["fused_q"]
