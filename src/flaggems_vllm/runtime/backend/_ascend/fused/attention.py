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


from dataclasses import dataclass, replace
from functools import wraps
from typing import Optional, Tuple, Union

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.ops.attention import (
    flash_attn_varlen_func as upstream_flash_attn_varlen_func,
)
from flaggems_vllm.ops.flash_kernel import (
    apply_alibi,
    apply_mask,
    apply_softcap,
    virtual_to_cache_offset,
)
from flaggems_vllm.runtime import torch_device_fn

PIPE = al.PIPE
RESIDUAL_SCALE_BITS = tl.constexpr(1048576000)
LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)
DESCALE_BLOCK = tl.constexpr(128)
BLOCK_M = tl.constexpr(64)
BLOCK_N = tl.constexpr(128)
LAUNCH_CHUNK = 2048


# Native attention kernels


@triton.jit
def ub_zero():
    # Prevent loop hoisting of the initialization scalar.
    bits = tl.inline_asm_elementwise("MOV $0, 0", "=l", [], tl.int64, False, 1)
    return bits.to(tl.uint32).to(tl.float32, bitcast=True)


@dataclass(frozen=True)
class AttentionConfig:
    __triton_compile_time_value__ = True
    batch: int
    kv_heads: int
    group: int
    dim: int
    query_capacity: int
    key_capacity: int
    query_tiles: int
    total_query: int
    output_stride: int
    q_scale_strides: Tuple[int, int, int]
    k_scale_strides: Tuple[int, int, int]
    v_scale_strides: Tuple[int, int, int]
    causal: bool
    write_lse: bool
    scale: float
    block_m: int
    block_n: int
    cores: int
    left: int
    right: int
    softcap: float
    has_alibi: bool
    alibi_batch_stride: int
    alibi_head_stride: int
    has_worklist: bool = False
    direct_output: bool = False
    filter_flags: bool = False
    kv_head_repeat: int = 1
    uniform_q_length: int = 0
    vector_elements: int = 8192
    aggregate_flags: bool = False
    query_stride: int = 0
    task_offset: int = 0
    tile_shift: int = 0
    key_head_major: bool = False


@triton.jit
def stage_sync(slot, kind: tl.constexpr, WAIT: tl.constexpr):
    if WAIT:
        if kind == 1:
            if slot == 0:
                al.sync_block_wait("vector", "cube", 1, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
            else:
                al.sync_block_wait("vector", "cube", 4, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
        elif slot == 0:
            al.sync_block_wait("cube", "vector", kind, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
        else:
            al.sync_block_wait(
                "cube", "vector", kind + 3, PIPE.PIPE_FIX, PIPE.PIPE_MTE2
            )
    elif kind == 1:
        if slot == 0:
            al.sync_block_set("vector", "cube", 1, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
        else:
            al.sync_block_set("vector", "cube", 4, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
    elif slot == 0:
        al.sync_block_set("cube", "vector", kind, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
    else:
        al.sync_block_set("cube", "vector", kind + 3, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)


@triton.jit
def native_cube_score(Query, Key, Score, column_tile, CONFIG: tl.constexpr):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    CM: tl.constexpr = 128 if M >= 128 else M
    CN: tl.constexpr = N if N <= 32768 // CM else 32768 // CM
    for column in range(0, N, CN):
        kp = tl.make_block_ptr(
            Key,
            (CONFIG.key_capacity, D),
            (
                (
                    D
                    if CONFIG.key_head_major
                    else CONFIG.kv_heads // CONFIG.kv_head_repeat * D
                ),
                1,
            ),
            (column_tile * N + column, 0),
            (CN, D),
            (1, 0),
        )
        key = tl.trans(tl.load(kp))
        for row in range(0, M, CM):
            qp = tl.make_block_ptr(
                Query,
                (M, D),
                (CONFIG.query_stride if CONFIG.query_stride > 0 else D, 1),
                (row, 0),
                (CM, D),
                (1, 0),
            )
            score = tl.dot(tl.load(qp), key, out_dtype=tl.int32)
            sp = tl.make_block_ptr(
                Score, (M, N), (N, 1), (row, column), (CM, CN), (1, 0)
            )
            tl.store(sp, score)


@triton.jit
def native_cube_product(
    Probability,
    Value,
    Product,
    column_tile,
    CONFIG: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    CM: tl.constexpr = 128 if M >= 128 else M
    vp = tl.make_block_ptr(
        Value, (CONFIG.key_capacity, D), (D, 1), (column_tile * N, 0), (N, D), (1, 0)
    )
    value = tl.load(vp)
    for row in range(0, M, CM):
        pp = tl.make_block_ptr(Probability, (M, N), (N, 1), (row, 0), (CM, N), (1, 0))
        product = tl.dot(tl.load(pp), value, out_dtype=tl.float32)
        if RESIDUAL:
            rp = tl.make_block_ptr(
                Probability + M * N, (M, N), (N, 1), (row, 0), (CM, N), (1, 0)
            )
            product = tl.dot(tl.load(rp), value, product, out_dtype=tl.float32)
        else:
            pass
        op = tl.make_block_ptr(Product, (M, D), (D, 1), (row, 0), (CM, D), (1, 0))
        tl.store(op, product)


@triton.jit
def native_cube_main(
    Query,
    Key,
    Value,
    Cuq,
    Used,
    QueryBatch,
    QueryTile,
    Counts,
    ValueScale,
    Score,
    Probability,
    Product,
    CONFIG: tl.constexpr,
    Flags,
    UnsafeCount,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    G: tl.constexpr = CONFIG.group
    HK: tl.constexpr = CONFIG.kv_heads
    B: tl.constexpr = CONFIG.batch
    Q_ROWS: tl.constexpr = M // G
    core = tl.program_id(0)
    if CONFIG.has_worklist:
        work_count = tl.load(Counts)
    else:
        work_count = CONFIG.query_tiles * B
    if CONFIG.aggregate_flags:
        if tl.load(UnsafeCount) == 0:
            work_count = 0
        else:
            pass
    else:
        pass
    for task in tl.range(core, work_count * HK, CONFIG.cores, disable_licm=True):
        if CONFIG.has_worklist:
            tile = tl.load(QueryTile + task // HK)
            batch = tl.load(QueryBatch + task // HK)
        else:
            tile = task // (B * HK)
            batch = task // HK % B
        head = task % HK
        if CONFIG.uniform_q_length > 0:
            nq = CONFIG.uniform_q_length
        else:
            nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        nk = tl.load(Used + batch)
        if CONFIG.filter_flags:
            required = tl.load(Flags + task * 256) | tl.load(Flags + task * 256 + 128)
        else:
            required = 1
        if (tile * Q_ROWS < nq) & (required != 0):
            if CONFIG.query_stride > 0:
                query = (
                    Query
                    + (batch * CONFIG.uniform_q_length + tile * M) * CONFIG.query_stride
                    + head * D
                )
            else:
                query = (
                    Query
                    + ((batch * HK + head) * CONFIG.query_capacity * G + tile * M) * D
                )
            if CONFIG.key_head_major:
                key_base = (
                    Key
                    + (
                        batch * (HK // CONFIG.kv_head_repeat)
                        + head // CONFIG.kv_head_repeat
                    )
                    * CONFIG.key_capacity
                    * D
                )
            else:
                key_base = (
                    Key
                    + (
                        batch * CONFIG.key_capacity * (HK // CONFIG.kv_head_repeat)
                        + head // CONFIG.kv_head_repeat
                    )
                    * D
                )
            value_base = (
                Value
                + (
                    batch * (HK // CONFIG.kv_head_repeat)
                    + head // CONFIG.kv_head_repeat
                )
                * CONFIG.key_capacity
                * D
            )
            end = nk
            if CONFIG.causal:
                end = tl.minimum(nk, (tile + 1) * Q_ROWS + nk - nq)
            else:
                pass
            count = tl.cdiv(tl.maximum(end, 0), N)
            if count > 0:
                native_cube_score(query, key_base, Score + core * 2 * M * N, 0, CONFIG)
                stage_sync(0, 0, False)
                for column_tile in tl.range(0, count, disable_licm=True):
                    slot = column_tile % 2
                    if column_tile + 1 < count:
                        next_slot = (column_tile + 1) % 2
                        native_cube_score(
                            query,
                            key_base,
                            Score + (core * 2 + next_slot) * M * N,
                            column_tile + 1,
                            CONFIG,
                        )
                        stage_sync(next_slot, 0, False)
                    else:
                        pass
                    stage_sync(slot, 1, True)
                    descale_block = column_tile * (N // 128)
                    value_scale = tl.load(
                        ValueScale
                        + batch * CONFIG.v_scale_strides[0]
                        + head // CONFIG.kv_head_repeat * CONFIG.v_scale_strides[1]
                        + descale_block * CONFIG.v_scale_strides[2]
                    )
                    scale_bits = value_scale.to(tl.int32, bitcast=True) & 2147483647
                    probability_base = Probability + (core * 2 + slot) * 2 * M * N
                    product_base = Product + (core * 2 + slot) * M * D
                    if scale_bits > RESIDUAL_SCALE_BITS:
                        native_cube_product(
                            probability_base,
                            value_base,
                            product_base,
                            column_tile,
                            CONFIG,
                            True,
                        )
                    else:
                        native_cube_product(
                            probability_base,
                            value_base,
                            product_base,
                            column_tile,
                            CONFIG,
                            False,
                        )
                    stage_sync(slot, 2, False)
            else:
                pass
        else:
            pass
    al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)


@triton.jit
def apply_score_modifiers(
    scores,
    Alibi,
    batch,
    query_heads,
    query_rows,
    columns,
    nq,
    nk,
    CONFIG: tl.constexpr,
):
    if CONFIG.softcap > 0:
        scores = CONFIG.softcap * apply_softcap(
            scores, 1.0 / CONFIG.softcap, is_softcap=True
        )
    else:
        pass
    if CONFIG.has_alibi:
        slopes = tl.load(
            Alibi
            + batch * CONFIG.alibi_batch_stride
            + query_heads * CONFIG.alibi_head_stride
        )
        scores = apply_alibi(
            scores,
            columns,
            query_rows,
            nq,
            nk,
            is_causal=False,
            is_alibi=True,
            alibi_slope=slopes[:, None],
        )
    else:
        pass
    # apply_mask expects finite window bounds; normalize each unbounded side.
    scores = apply_mask(
        scores,
        columns,
        query_rows,
        nq,
        nk,
        nk if CONFIG.left < 0 else CONFIG.left,
        0 if CONFIG.causal else nq if CONFIG.right < 0 else CONFIG.right,
        is_even_mn=False,
        is_causal=CONFIG.causal,
        is_local=(CONFIG.left >= 0 or CONFIG.right >= 0),
    )
    return tl.where(query_rows[:, None] < nq, scores * LOG2E, float("-inf"))


@triton.jit
def native_softmax_block(
    Score,
    Probability,
    QueryScale,
    KeyScale,
    ValueScale,
    Alibi,
    MaxState,
    SumState,
    AlphaState,
    batch,
    head,
    tile,
    nq,
    nk,
    column_tile,
    sub,
    CONFIG: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    VR: tl.constexpr = M // 2
    CHUNK: tl.constexpr = (
        16 if VR >= 16 and M <= 256 and (N <= 256) else 8 if VR >= 8 else VR
    )
    G: tl.constexpr = CONFIG.group
    columns = column_tile * N + tl.arange(0, N)
    descale_block = column_tile * (N // 128)
    key_scale = tl.load(
        KeyScale
        + batch * CONFIG.k_scale_strides[0]
        + head // CONFIG.kv_head_repeat * CONFIG.k_scale_strides[1]
        + descale_block * CONFIG.k_scale_strides[2]
    )
    slot = column_tile % 2
    for piece in tl.range(0, VR, CHUNK, loop_unroll_factor=1, disable_licm=True):
        rows = sub * VR + piece + tl.arange(0, CHUNK)
        query_rows = tile * (M // G) + rows // G
        query_heads = head * G + rows % G
        valid_rows = query_rows < nq
        query_scale = tl.load(
            QueryScale
            + batch * CONFIG.q_scale_strides[0]
            + query_heads * CONFIG.q_scale_strides[1]
            + query_rows // 128 * CONFIG.q_scale_strides[2],
            valid_rows,
            0,
        )
        sp = tl.make_block_ptr(
            Score, (M, N), (N, 1), (sub * VR + piece, 0), (CHUNK, N), (1, 0)
        )
        scores = tl.load(sp).to(tl.float32)
        scores *= (query_scale * key_scale * CONFIG.scale)[:, None]
        scores = apply_score_modifiers(
            scores, Alibi, batch, query_heads, query_rows, columns, nq, nk, CONFIG
        )
        max_slot = tle.dsa.subview(MaxState, [piece], [CHUNK], [1])
        sum_slot = tle.dsa.subview(SumState, [piece], [CHUNK], [1])
        alpha_slot = tle.dsa.subview(AlphaState, [slot * VR + piece], [CHUNK], [1])
        maximum = tle.dsa.to_tensor(max_slot, writable=False)
        denominator = tle.dsa.to_tensor(sum_slot, writable=False)
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(new_maximum == float("-inf"), 0, new_maximum)
        alpha = tl.exp2(maximum - safe_maximum)
        probability = tl.exp2(scores - safe_maximum[:, None])
        new_denominator = denominator * alpha + tl.sum(probability, 1)
        tle.dsa.to_buffer(alpha, bind_buffer=alpha_slot)
        tle.dsa.to_buffer(new_maximum, bind_buffer=max_slot)
        tle.dsa.to_buffer(new_denominator, bind_buffer=sum_slot)
        probability_half = probability.to(tl.float16)
        pp = tl.make_block_ptr(
            Probability, (M, N), (N, 1), (sub * VR + piece, 0), (CHUNK, N), (1, 0)
        )
        tl.store(pp, probability_half)
        if RESIDUAL:
            residual = (probability - probability_half.to(tl.float32)).to(tl.float16)
            rp = tl.make_block_ptr(
                Probability + M * N,
                (M, N),
                (N, 1),
                (sub * VR + piece, 0),
                (CHUNK, N),
                (1, 0),
            )
            tl.store(rp, residual)
        else:
            pass


@triton.jit
def native_prepare_probability(
    Score,
    Probability,
    QueryScale,
    KeyScale,
    ValueScale,
    Alibi,
    MaxState,
    SumState,
    AlphaState,
    batch,
    head,
    tile,
    nq,
    nk,
    column_tile,
    sub,
    CONFIG: tl.constexpr,
):
    N: tl.constexpr = CONFIG.block_n
    value_scale = tl.load(
        ValueScale
        + batch * CONFIG.v_scale_strides[0]
        + head // CONFIG.kv_head_repeat * CONFIG.v_scale_strides[1]
        + column_tile * (N // 128) * CONFIG.v_scale_strides[2]
    )
    scale_bits = value_scale.to(tl.int32, bitcast=True) & 2147483647
    if scale_bits > RESIDUAL_SCALE_BITS:
        native_softmax_block(
            Score,
            Probability,
            QueryScale,
            KeyScale,
            ValueScale,
            Alibi,
            MaxState,
            SumState,
            AlphaState,
            batch,
            head,
            tile,
            nq,
            nk,
            column_tile,
            sub,
            CONFIG,
            True,
        )
    else:
        native_softmax_block(
            Score,
            Probability,
            QueryScale,
            KeyScale,
            ValueScale,
            Alibi,
            MaxState,
            SumState,
            AlphaState,
            batch,
            head,
            tile,
            nq,
            nk,
            column_tile,
            sub,
            CONFIG,
            False,
        )


@triton.jit
def native_accumulate_product(
    Product,
    ValueScale,
    AccState,
    AlphaState,
    batch,
    head,
    column_tile,
    sub,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    VR: tl.constexpr = M // 2
    CHUNK: tl.constexpr = (
        16 if VR >= 16 and M <= 256 and (N <= 256) else 8 if VR >= 8 else VR
    )
    value_scale = tl.load(
        ValueScale
        + batch * CONFIG.v_scale_strides[0]
        + head // CONFIG.kv_head_repeat * CONFIG.v_scale_strides[1]
        + column_tile * (N // 128) * CONFIG.v_scale_strides[2]
    )
    slot = column_tile % 2
    for piece in tl.range(0, VR, CHUNK, loop_unroll_factor=1, disable_licm=True):
        acc_slot = tle.dsa.subview(AccState, [piece, 0], [CHUNK, D], [1, 1])
        alpha_slot = tle.dsa.subview(AlphaState, [slot * VR + piece], [CHUNK], [1])
        accumulator = tle.dsa.to_tensor(acc_slot, writable=False)
        alpha = tle.dsa.to_tensor(alpha_slot, writable=False)
        product_ptr = tl.make_block_ptr(
            Product, (M, D), (D, 1), (sub * VR + piece, 0), (CHUNK, D), (1, 0)
        )
        product = tl.load(product_ptr)
        updated = accumulator * alpha[:, None] + product * value_scale
        tle.dsa.to_buffer(updated, bind_buffer=acc_slot)


@triton.jit
def write_output(
    Output,
    LSE,
    Cuq,
    AccState,
    MaxState,
    SumState,
    batch,
    head,
    tile,
    nq,
    sub,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    G: tl.constexpr = CONFIG.group
    VR: tl.constexpr = M // 2
    CHUNK: tl.constexpr = (
        16 if VR >= 16 and M <= 256 and (N <= 256) else 8 if VR >= 8 else VR
    )
    output_base = (batch * CONFIG.kv_heads + head) * CONFIG.query_capacity * G
    for piece in tl.range(0, VR, CHUNK, loop_unroll_factor=1, disable_licm=True):
        acc_slot = tle.dsa.subview(AccState, [piece, 0], [CHUNK, D], [1, 1])
        sum_slot = tle.dsa.subview(SumState, [piece], [CHUNK], [1])
        accumulator = tle.dsa.to_tensor(acc_slot, writable=False)
        denominator = tle.dsa.to_tensor(sum_slot, writable=False)
        output = accumulator / tl.where(denominator > 0, denominator, 1)[:, None]
        row = tile * M + sub * VR + piece
        if CONFIG.direct_output:
            if CONFIG.uniform_q_length > 0:
                begin = batch * CONFIG.uniform_q_length
            else:
                begin = tl.load(Cuq + batch)
            row = tile * M + sub * VR + piece
            target = tl.make_block_ptr(
                Output + begin * CONFIG.output_stride + head * G * D,
                (nq, G * D),
                (CONFIG.output_stride, 1),
                (row // G, 0),
                (CHUNK // G, G * D),
                (1, 0),
            )
            if (
                CONFIG.uniform_q_length > 0
                and CONFIG.uniform_q_length == CONFIG.query_capacity
            ):
                tl.store(
                    target,
                    output.to(Output.dtype.element_ty).reshape((CHUNK // G, G * D)),
                )
            elif row // G + CHUNK // G <= nq:
                tl.store(
                    target,
                    output.to(Output.dtype.element_ty).reshape((CHUNK // G, G * D)),
                )
            else:
                tl.store(
                    target,
                    output.to(Output.dtype.element_ty).reshape((CHUNK // G, G * D)),
                    boundary_check=(0,),
                )
        else:
            target = tl.make_block_ptr(
                Output + output_base * D,
                (CONFIG.query_capacity * G, D),
                (D, 1),
                (row, 0),
                (CHUNK, D),
                (1, 0),
            )
            tl.store(target, output.to(Output.dtype.element_ty))
        if CONFIG.write_lse:
            max_slot = tle.dsa.subview(MaxState, [piece], [CHUNK], [1])
            maximum = tle.dsa.to_tensor(max_slot, writable=False)
            lse = tl.where(
                denominator > 0,
                maximum * 0.6931471805599453 + tl.log(denominator),
                float("inf"),
            )
            if CONFIG.direct_output:
                lane = tl.arange(0, CHUNK)
                query_row = (row + lane) // G
                query_head = head * G + (row + lane) % G
                if (
                    CONFIG.uniform_q_length > 0
                    and CONFIG.uniform_q_length == CONFIG.query_capacity
                ):
                    tl.store(
                        LSE + query_head * CONFIG.total_query + begin + query_row, lse
                    )
                else:
                    tl.store(
                        LSE + query_head * CONFIG.total_query + begin + query_row,
                        lse,
                        query_row < nq,
                    )
            else:
                lse_ptr = tl.make_block_ptr(
                    LSE + output_base,
                    (CONFIG.query_capacity * G,),
                    (1,),
                    (row,),
                    (CHUNK,),
                    (0,),
                )
                tl.store(lse_ptr, lse)
        else:
            pass


@triton.jit
def native_vector_main(
    Output,
    LSE,
    Cuq,
    Used,
    QueryBatch,
    QueryTile,
    Counts,
    QueryScale,
    KeyScale,
    ValueScale,
    Alibi,
    Score,
    Probability,
    Product,
    CONFIG: tl.constexpr,
    Flags,
    UnsafeCount,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    VR: tl.constexpr = M // 2
    G: tl.constexpr = CONFIG.group
    B: tl.constexpr = CONFIG.batch
    HK: tl.constexpr = CONFIG.kv_heads
    core = tl.program_id(0)
    sub = al.sub_vec_id().to(tl.int32)
    max_state = tle.dsa.alloc([VR], tl.float32, tle.dsa.ascend.UB)
    sum_state = tle.dsa.alloc([VR], tl.float32, tle.dsa.ascend.UB)
    alpha_state = tle.dsa.alloc([2 * VR], tl.float32, tle.dsa.ascend.UB)
    acc_state = tle.dsa.alloc([VR, D], tl.float32, tle.dsa.ascend.UB)
    if CONFIG.has_worklist:
        work_count = tl.load(Counts)
    else:
        work_count = CONFIG.query_tiles * B
    if CONFIG.aggregate_flags:
        if tl.load(UnsafeCount) == 0:
            work_count = 0
        else:
            pass
    else:
        pass
    for task in tl.range(core, work_count * HK, CONFIG.cores, disable_licm=True):
        if CONFIG.has_worklist:
            tile = tl.load(QueryTile + task // HK)
            batch = tl.load(QueryBatch + task // HK)
        else:
            tile = task // (B * HK)
            batch = task // HK % B
        head = task % HK
        if CONFIG.uniform_q_length > 0:
            nq = CONFIG.uniform_q_length
        else:
            nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        nk = tl.load(Used + batch)
        if CONFIG.filter_flags:
            required = tl.load(Flags + task * 256) | tl.load(Flags + task * 256 + 128)
        else:
            required = 1
        if (tile * (M // G) < nq) & (required != 0):
            zero = ub_zero()
            negative_infinity = (zero.to(tl.uint32, bitcast=True) | 0xFF800000).to(
                tl.float32, bitcast=True
            )
            tle.dsa.to_buffer(
                tl.full((VR,), negative_infinity, tl.float32), bind_buffer=max_state
            )
            tle.dsa.to_buffer(tl.full((VR,), zero, tl.float32), bind_buffer=sum_state)
            tle.dsa.to_buffer(tl.full((VR, D), zero, tl.float32), bind_buffer=acc_state)
            end = nk
            if CONFIG.causal:
                end = tl.minimum(nk, (tile + 1) * (M // G) + nk - nq)
            else:
                pass
            count = tl.cdiv(tl.maximum(end, 0), N)
            if count > 0:
                stage_sync(0, 0, True)
                native_prepare_probability(
                    Score + core * 2 * M * N,
                    Probability + core * 4 * M * N,
                    QueryScale,
                    KeyScale,
                    ValueScale,
                    Alibi,
                    max_state,
                    sum_state,
                    alpha_state,
                    batch,
                    head,
                    tile,
                    nq,
                    nk,
                    0,
                    sub,
                    CONFIG,
                )
                stage_sync(0, 1, False)
                for column_tile in tl.range(0, count, disable_licm=True):
                    slot = column_tile % 2
                    if column_tile + 1 < count:
                        next_slot = (column_tile + 1) % 2
                        stage_sync(next_slot, 0, True)
                        native_prepare_probability(
                            Score + (core * 2 + next_slot) * M * N,
                            Probability + (core * 2 + next_slot) * 2 * M * N,
                            QueryScale,
                            KeyScale,
                            ValueScale,
                            Alibi,
                            max_state,
                            sum_state,
                            alpha_state,
                            batch,
                            head,
                            tile,
                            nq,
                            nk,
                            column_tile + 1,
                            sub,
                            CONFIG,
                        )
                        stage_sync(next_slot, 1, False)
                    else:
                        pass
                    stage_sync(slot, 2, True)
                    native_accumulate_product(
                        Product + (core * 2 + slot) * M * D,
                        ValueScale,
                        acc_state,
                        alpha_state,
                        batch,
                        head,
                        column_tile,
                        sub,
                        CONFIG,
                    )
            else:
                pass
            write_output(
                Output,
                LSE,
                Cuq,
                acc_state,
                max_state,
                sum_state,
                batch,
                head,
                tile,
                nq,
                sub,
                CONFIG,
            )
        else:
            pass
    al.sync_block_set("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)


@triton.jit
def native_tle_attention(
    Query,
    Key,
    Value,
    Output,
    LSE,
    Cuq,
    Used,
    QueryBatch,
    QueryTile,
    Counts,
    QueryScale,
    KeyScale,
    ValueScale,
    Alibi,
    Score,
    Probability,
    Product,
    CONFIG: tl.constexpr,
    Flags=None,
    UnsafeCount=None,
):
    with al.scope(core_mode="cube"):
        native_cube_main(
            Query,
            Key,
            Value,
            Cuq,
            Used,
            QueryBatch,
            QueryTile,
            Counts,
            ValueScale,
            Score,
            Probability,
            Product,
            CONFIG,
            Flags,
            UnsafeCount,
        )
    with al.scope(core_mode="vector"):
        native_vector_main(
            Output,
            LSE,
            Cuq,
            Used,
            QueryBatch,
            QueryTile,
            Counts,
            QueryScale,
            KeyScale,
            ValueScale,
            Alibi,
            Score,
            Probability,
            Product,
            CONFIG,
            Flags,
            UnsafeCount,
        )


# Prefill kernels


@triton.jit
def prefill_cube_score(query, Key, Score, column, CONFIG: tl.constexpr):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    CN: tl.constexpr = 128 if N >= 128 else N
    for piece in range(0, N, CN):
        kp = tl.make_block_ptr(
            Key,
            (CONFIG.key_capacity, D),
            (
                (
                    D
                    if CONFIG.key_head_major
                    else CONFIG.kv_heads // CONFIG.kv_head_repeat * D
                ),
                1,
            ),
            (column * N + piece, 0),
            (CN, D),
            (1, 0),
        )
        score = tl.dot(query, tl.trans(tl.load(kp)), out_dtype=tl.int32)
        sp = tl.make_block_ptr(Score, (M, N), (N, 1), (0, piece), (M, CN), (1, 0))
        tl.store(sp, score)


@triton.jit
def cube_loop(
    Query,
    Key,
    Value,
    Score,
    Probability,
    Product,
    Flags,
    core,
    count,
    CONFIG: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    qp = tl.make_block_ptr(
        Query,
        (M, D),
        (CONFIG.query_stride if CONFIG.query_stride > 0 else D, 1),
        (0, 0),
        (M, D),
        (1, 0),
    )
    query = tl.load(qp)
    accumulator = tl.zeros((M, D), tl.float32)
    prefill_cube_score(query, Key, Score + core * (2 * M * N) + 0 * M * N, 0, CONFIG)
    stage_sync(0, 0, False)
    for column in tl.range(0, count, disable_licm=True):
        slot = column & 1
        if column + 1 < count:
            prefill_cube_score(
                query,
                Key,
                Score + core * (2 * M * N) + (column + 1 & 1) * M * N,
                column + 1,
                CONFIG,
            )
            stage_sync(column + 1 & 1, 0, False)
        else:
            pass
        if N <= 256:
            value_ptr = tl.make_block_ptr(
                Value, (CONFIG.key_capacity, D), (D, 1), (column * N, 0), (N, D), (1, 0)
            )
            prefetched_value = tl.load(value_ptr)
        else:
            pass
        stage_sync(slot, 1, True)
        KN: tl.constexpr = 256 if N >= 256 else N
        for piece in range(0, N, KN):
            pp = tl.make_block_ptr(
                Probability + core * (4 * M * N) + slot * 2 * M * N,
                (M, N),
                (N, 1),
                (0, piece),
                (M, KN),
                (1, 0),
            )
            vp = tl.make_block_ptr(
                Value,
                (CONFIG.key_capacity, D),
                (D, 1),
                (column * N + piece, 0),
                (KN, D),
                (1, 0),
            )
            values = prefetched_value if N <= 256 else tl.load(vp)
            accumulator = tl.dot(tl.load(pp), values, accumulator, out_dtype=tl.float32)
            if RESIDUAL:
                rp = tl.make_block_ptr(
                    Probability + core * (4 * M * N) + slot * 2 * M * N + M * N,
                    (M, N),
                    (N, 1),
                    (0, piece),
                    (M, KN),
                    (1, 0),
                )
                accumulator = tl.dot(
                    tl.load(rp), values, accumulator, out_dtype=tl.float32
                )
            else:
                pass
    target = tl.make_block_ptr(
        Product + core * (M * D), (M, D), (D, 1), (0, 0), (M, D), (1, 0)
    )
    tl.store(target, accumulator)
    al.sync_block_set("cube", "vector", 7, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)


@triton.jit
def prefill_cube_main(
    Query,
    Key,
    Value,
    Cuq,
    Used,
    QBatch,
    QTile,
    Counts,
    VS,
    Score,
    Probability,
    Product,
    Flags,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    HK: tl.constexpr = CONFIG.kv_heads
    B: tl.constexpr = CONFIG.batch
    G: tl.constexpr = CONFIG.group
    core = tl.program_id(0)
    if CONFIG.has_worklist:
        work_count = tl.load(Counts)
    else:
        work_count = CONFIG.query_tiles * B
    for linear_task in tl.range(core, work_count * HK, CONFIG.cores, disable_licm=True):
        if CONFIG.task_offset > 0:
            shifted = (linear_task + CONFIG.task_offset) % (CONFIG.query_tiles * B * HK)
            shifted_head = shifted % HK
            shifted_tile = (
                shifted // (B * HK)
                + shifted_head // CONFIG.kv_head_repeat * CONFIG.tile_shift
            ) % CONFIG.query_tiles
            task = (shifted_tile * B + shifted // HK % B) * HK + shifted_head
        else:
            task = linear_task
        if CONFIG.has_worklist:
            batch = tl.load(QBatch + task // HK)
            tile = tl.load(QTile + task // HK)
        else:
            batch = task // HK % B
            tile = task // (B * HK)
        head = task % HK
        if CONFIG.uniform_q_length > 0:
            nq = CONFIG.uniform_q_length
        else:
            nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        nk = tl.load(Used + batch)
        if tile * (M // G) < nq:
            end = nk
            if CONFIG.causal:
                end = tl.minimum(nk, (tile + 1) * (M // G) + nk - nq)
            else:
                pass
            count = tl.cdiv(tl.maximum(end, 0), N)
            if count > 0:
                if CONFIG.query_stride > 0:
                    query = (
                        Query
                        + (batch * CONFIG.uniform_q_length + tile * M)
                        * CONFIG.query_stride
                        + head * D
                    )
                else:
                    query = (
                        Query
                        + ((batch * HK + head) * CONFIG.query_capacity * G + tile * M)
                        * D
                    )
                if CONFIG.key_head_major:
                    key = (
                        Key
                        + (
                            batch * (HK // CONFIG.kv_head_repeat)
                            + head // CONFIG.kv_head_repeat
                        )
                        * CONFIG.key_capacity
                        * D
                    )
                else:
                    key = (
                        Key
                        + (
                            batch * CONFIG.key_capacity * (HK // CONFIG.kv_head_repeat)
                            + head // CONFIG.kv_head_repeat
                        )
                        * D
                    )
                value = (
                    Value
                    + (
                        batch * (HK // CONFIG.kv_head_repeat)
                        + head // CONFIG.kv_head_repeat
                    )
                    * CONFIG.key_capacity
                    * D
                )
                flag = Flags + task * 256
                vs = tl.load(
                    VS
                    + batch * CONFIG.v_scale_strides[0]
                    + head // CONFIG.kv_head_repeat * CONFIG.v_scale_strides[1]
                )
                if not vs.to(tl.int32, bitcast=True) & 2147483647 > RESIDUAL_SCALE_BITS:
                    cube_loop(
                        query,
                        key,
                        value,
                        Score,
                        Probability,
                        Product,
                        flag,
                        core,
                        count,
                        CONFIG,
                        False,
                    )
                else:
                    pass
            else:
                pass
        else:
            pass
    al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)


@triton.jit
def prepare_block(
    Score,
    Probability,
    Flags,
    Coefficients,
    Denominator,
    batch,
    head,
    tile,
    nq,
    nk,
    column,
    sub,
    CONFIG: tl.constexpr,
    RESIDUAL: tl.constexpr,
    MASKED: tl.constexpr,
    FIRST: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    VR: tl.constexpr = M // 2
    C: tl.constexpr = (
        CONFIG.vector_elements // N if VR >= CONFIG.vector_elements // N else VR
    )
    any_update = tl.full((), 0, tl.int32)
    for piece in tl.range(0, VR, C, disable_licm=True, loop_unroll_factor=1):
        rows = sub * VR + piece + tl.arange(0, C)
        qp = Coefficients
        sp = tl.make_block_ptr(
            Score, (M, N), (N, 1), (sub * VR + piece, 0), (C, N), (1, 0)
        )
        if MASKED:
            qr = tile * (M // CONFIG.group) + rows // CONFIG.group
            kc = column * N + tl.arange(0, N)
            valid = (qr[:, None] < nq) & (kc[None, :] < nk)
            if CONFIG.causal:
                valid &= kc[None, :] <= qr[:, None] + nk - nq
            else:
                pass
        else:
            pass
        if CONFIG.group == 1:
            scores = tl.load(sp).to(tl.float32) * qp
        else:
            scores = tl.load(sp).to(tl.float32) * qp[:, None]
        if MASKED:
            scores = tl.where(valid, scores, float("-inf"))
        else:
            pass
        ds = tle.dsa.subview(Denominator, [piece], [C], [1])
        denominator = tle.dsa.to_tensor(ds, writable=False)
        # A fixed reference avoids per-block rescaling; unsafe rows are recomputed.
        reference = tl.full((C,), -2.772588722239781, tl.float32)
        safe = tl.where(reference == float("-inf"), 0, reference)
        p = tl.exp(scores - safe[:, None])
        block_sum = tl.sum(p, 1)
        tle.dsa.to_buffer(denominator + block_sum, bind_buffer=ds)
        half = p.to(tl.float16)
        pp = tl.make_block_ptr(
            Probability, (M, N), (N, 1), (sub * VR + piece, 0), (C, N), (1, 0)
        )
        tl.store(pp, half)
        if RESIDUAL:
            rp = tl.make_block_ptr(
                Probability + M * N,
                (M, N),
                (N, 1),
                (sub * VR + piece, 0),
                (C, N),
                (1, 0),
            )
            tl.store(rp, (p - half.to(tl.float32)).to(tl.float16))
        else:
            pass
    return any_update


@triton.jit
def prepare_selected(
    Score,
    Probability,
    Flags,
    Coefficients,
    Denominator,
    batch,
    head,
    tile,
    nq,
    nk,
    column,
    sub,
    bits,
    CONFIG: tl.constexpr,
    FIRST: tl.constexpr,
):
    query_rows: tl.constexpr = CONFIG.block_m // CONFIG.group
    half_rows: tl.constexpr = query_rows // 2
    rows_full = tile * query_rows + (sub + 1) * half_rows <= nq
    columns_full = (column + 1) * CONFIG.block_n <= nk
    if CONFIG.causal:
        columns_full &= (
            column + 1
        ) * CONFIG.block_n <= tile * query_rows + sub * half_rows + nk - nq + 1
    else:
        pass
    is_full = rows_full & columns_full
    if is_full:
        required = prepare_block(
            Score,
            Probability,
            Flags,
            Coefficients,
            Denominator,
            batch,
            head,
            tile,
            nq,
            nk,
            column,
            sub,
            CONFIG,
            False,
            False,
            FIRST,
        )
    else:
        required = prepare_block(
            Score,
            Probability,
            Flags,
            Coefficients,
            Denominator,
            batch,
            head,
            tile,
            nq,
            nk,
            column,
            sub,
            CONFIG,
            False,
            True,
            FIRST,
        )
    return required


@triton.jit
def prefill_vector_main(
    Output,
    LSE,
    Cuq,
    Used,
    QBatch,
    QTile,
    Counts,
    QS,
    KS,
    VS,
    Score,
    Probability,
    Product,
    Flags,
    CONFIG: tl.constexpr,
    UnsafeCount,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    G: tl.constexpr = CONFIG.group
    HK: tl.constexpr = CONFIG.kv_heads
    B: tl.constexpr = CONFIG.batch
    VR: tl.constexpr = M // 2
    C: tl.constexpr = (
        CONFIG.vector_elements // N if VR >= CONFIG.vector_elements // N else VR
    )
    core = tl.program_id(0)
    sub = al.sub_vec_id().to(tl.int32)
    denominator = tle.dsa.alloc([VR], tl.float32, tle.dsa.ascend.UB)
    if CONFIG.has_worklist:
        work_count = tl.load(Counts)
    else:
        work_count = CONFIG.query_tiles * B
    for linear_task in tl.range(core, work_count * HK, CONFIG.cores, disable_licm=True):
        if CONFIG.task_offset > 0:
            shifted = (linear_task + CONFIG.task_offset) % (CONFIG.query_tiles * B * HK)
            shifted_head = shifted % HK
            shifted_tile = (
                shifted // (B * HK)
                + shifted_head // CONFIG.kv_head_repeat * CONFIG.tile_shift
            ) % CONFIG.query_tiles
            task = (shifted_tile * B + shifted // HK % B) * HK + shifted_head
        else:
            task = linear_task
        if CONFIG.has_worklist:
            batch = tl.load(QBatch + task // HK)
            tile = tl.load(QTile + task // HK)
        else:
            batch = task // HK % B
            tile = task // (B * HK)
        head = task % HK
        if CONFIG.uniform_q_length > 0:
            nq = CONFIG.uniform_q_length
        else:
            nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        nk = tl.load(Used + batch)
        if tile * (M // G) < nq:
            vs = tl.load(
                VS
                + batch * CONFIG.v_scale_strides[0]
                + head // CONFIG.kv_head_repeat * CONFIG.v_scale_strides[1]
            )
            bits = vs.to(tl.int32, bitcast=True) & 2147483647
            if bits <= RESIDUAL_SCALE_BITS:
                zero = ub_zero()
                tle.dsa.to_buffer(
                    tl.full((VR,), zero, tl.float32), bind_buffer=denominator
                )
                if G == 1:
                    qs = tl.load(
                        QS
                        + batch * CONFIG.q_scale_strides[0]
                        + head * CONFIG.q_scale_strides[1]
                    )
                else:
                    qheads = head * G + (sub * VR + tl.arange(0, C) & G - 1)
                    qs = tl.load(
                        QS
                        + batch * CONFIG.q_scale_strides[0]
                        + qheads * CONFIG.q_scale_strides[1]
                    )
                ks = tl.load(
                    KS
                    + batch * CONFIG.k_scale_strides[0]
                    + head // CONFIG.kv_head_repeat * CONFIG.k_scale_strides[1]
                )
                coefficients = qs * ks * CONFIG.scale
                end = nk
                if CONFIG.causal:
                    end = tl.minimum(nk, (tile + 1) * (M // G) + nk - nq)
                else:
                    pass
                count = tl.cdiv(tl.maximum(end, 0), N)
                flag = Flags + task * 256
                if count > 0:
                    stage_sync(0, 0, True)
                    score = Score + core * (2 * M * N) + 0 * M * N
                    probability = Probability + core * (4 * M * N) + 0 * 2 * M * N
                    prepare_selected(
                        score,
                        probability,
                        flag,
                        coefficients,
                        denominator,
                        batch,
                        head,
                        tile,
                        nq,
                        nk,
                        0,
                        sub,
                        bits,
                        CONFIG,
                        True,
                    )
                    stage_sync(0, 1, False)
                    for column in tl.range(1, count, disable_licm=True):
                        slot = column & 1
                        stage_sync(slot, 0, True)
                        score = Score + core * (2 * M * N) + slot * M * N
                        probability = (
                            Probability + core * (4 * M * N) + slot * 2 * M * N
                        )
                        prepare_selected(
                            score,
                            probability,
                            flag,
                            coefficients,
                            denominator,
                            batch,
                            head,
                            tile,
                            nq,
                            nk,
                            column,
                            sub,
                            bits,
                            CONFIG,
                            False,
                        )
                        stage_sync(slot, 1, False)
                    al.sync_block_wait(
                        "cube", "vector", 7, PIPE.PIPE_FIX, PIPE.PIPE_MTE2
                    )
                else:
                    pass
                output_requires = tl.zeros((16,), tl.int32)
                for piece in range(0, VR, 16):
                    ds = tle.dsa.subview(denominator, [piece], [16], [1])
                    sums = tle.dsa.to_tensor(ds, writable=False)
                    if count > 0:
                        source = tl.make_block_ptr(
                            Product + core * (M * D),
                            (M, D),
                            (D, 1),
                            (sub * VR + piece, 0),
                            (16, D),
                            (1, 0),
                        )
                        numerator = tl.load(source)
                    else:
                        numerator = tl.zeros((16, D), tl.float32)
                    # Any nonfinite FP16 P makes every PV column nonfinite for finite V.
                    sample = tl.load(
                        Product
                        + core * (M * D)
                        + (sub * VR + piece + tl.arange(0, 16)) * D
                    )
                    query_row = (
                        tile * (M // G) + (sub * VR + piece + tl.arange(0, 16)) // G
                    )
                    has_keys = (query_row < nq) & (nk > 0)
                    if CONFIG.causal:
                        has_keys &= query_row + nk - nq >= 0
                    else:
                        pass
                    output_requires = output_requires | (
                        has_keys & ((sums < 1.0) | ~(tl.abs(sample) < float("inf")))
                    ).to(tl.int32)
                    output = numerator / tl.where(sums > 0, sums, 1)[:, None] * vs
                    base = (batch * HK + head) * CONFIG.query_capacity * G
                    if CONFIG.direct_output:
                        if CONFIG.uniform_q_length > 0:
                            begin = batch * CONFIG.uniform_q_length
                        else:
                            begin = tl.load(Cuq + batch)
                        row = tile * M + sub * VR + piece
                        target = tl.make_block_ptr(
                            Output + begin * CONFIG.output_stride + head * G * D,
                            (nq, G * D),
                            (CONFIG.output_stride, 1),
                            (row // G, 0),
                            (16 // G, G * D),
                            (1, 0),
                        )
                        if (
                            CONFIG.uniform_q_length > 0
                            and CONFIG.uniform_q_length == CONFIG.query_capacity
                        ):
                            tl.store(
                                target,
                                output.to(Output.dtype.element_ty).reshape(
                                    (16 // G, G * D)
                                ),
                            )
                        elif row // G + 16 // G <= nq:
                            tl.store(
                                target,
                                output.to(Output.dtype.element_ty).reshape(
                                    (16 // G, G * D)
                                ),
                            )
                        else:
                            tl.store(
                                target,
                                output.to(Output.dtype.element_ty).reshape(
                                    (16 // G, G * D)
                                ),
                                boundary_check=(0,),
                            )
                    else:
                        row = tile * M + sub * VR + piece
                        base = (batch * HK + head) * CONFIG.query_capacity * G
                        target = tl.make_block_ptr(
                            Output + base * D,
                            (CONFIG.query_capacity * G, D),
                            (D, 1),
                            (row, 0),
                            (16, D),
                            (1, 0),
                        )
                        tl.store(target, output.to(Output.dtype.element_ty))
                    if CONFIG.write_lse:
                        maxima = tl.full((16,), -2.772588722239781, tl.float32)
                        lse = tl.where(sums > 0, maxima + tl.log(sums), float("inf"))
                        if CONFIG.direct_output:
                            lane = tl.arange(0, 16)
                            query_row = (row + lane) // G
                            query_head = head * G + (row + lane) % G
                            if (
                                CONFIG.uniform_q_length > 0
                                and CONFIG.uniform_q_length == CONFIG.query_capacity
                            ):
                                tl.store(
                                    LSE
                                    + query_head * CONFIG.total_query
                                    + begin
                                    + query_row,
                                    lse,
                                )
                            else:
                                tl.store(
                                    LSE
                                    + query_head * CONFIG.total_query
                                    + begin
                                    + query_row,
                                    lse,
                                    query_row < nq,
                                )
                        else:
                            lp = tl.make_block_ptr(
                                LSE + base,
                                (CONFIG.query_capacity * G,),
                                (1,),
                                (row,),
                                (16,),
                                (0,),
                            )
                            tl.store(lp, lse)
                    else:
                        pass
                task_unsafe = tl.max(output_requires, 0)
                tl.store(flag + sub * 128, task_unsafe)
                # Only zero/nonzero is used by the subsequent native recomputation kernel.
                if task_unsafe != 0:
                    tl.atomic_add(UnsafeCount, 1)
                else:
                    pass
            else:
                task_unsafe = 1
                tl.store(Flags + task * 256 + sub * 128, task_unsafe)
                # Only zero/nonzero is used by the subsequent native recomputation kernel.
                if task_unsafe != 0:
                    tl.atomic_add(UnsafeCount, 1)
                else:
                    pass
        else:
            task_unsafe = 0
            tl.store(Flags + task * 256 + sub * 128, task_unsafe)
            if task_unsafe != 0:
                tl.atomic_add(UnsafeCount, 1)
            else:
                pass
    al.sync_block_set("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)


@triton.jit
def prefill_attention(
    Query,
    Key,
    Value,
    Output,
    LSE,
    Cuq,
    Used,
    QBatch,
    QTile,
    Counts,
    QS,
    KS,
    VS,
    Alibi,
    Score,
    Probability,
    Product,
    Flags,
    CONFIG: tl.constexpr,
    UnsafeCount,
):
    with al.scope(core_mode="cube"):
        prefill_cube_main(
            Query,
            Key,
            Value,
            Cuq,
            Used,
            QBatch,
            QTile,
            Counts,
            VS,
            Score,
            Probability,
            Product,
            Flags,
            CONFIG,
        )
    with al.scope(core_mode="vector"):
        prefill_vector_main(
            Output,
            LSE,
            Cuq,
            Used,
            QBatch,
            QTile,
            Counts,
            QS,
            KS,
            VS,
            Score,
            Probability,
            Product,
            Flags,
            CONFIG,
            UnsafeCount,
        )


# Decode kernels


@dataclass(frozen=True)
class DecodeConfig(AttentionConfig):
    head_pack: int = 4
    query_cap: int = 1
    probability_digits: int = 3
    query_stride: int = 0
    key_stride: int = 0
    key_page_stride: int = 0
    value_stride: int = 0
    value_page_stride: int = 0
    table_stride: int = 0
    scale_blocks: int = 0
    table_blocks: int = 0


@triton.jit
def pack_queries_diagonal(
    Query,
    Packed,
    Cuq,
    B: tl.constexpr,
    HKG: tl.constexpr,
    D: tl.constexpr,
    SQ: tl.constexpr,
    QC: tl.constexpr,
    PACK: tl.constexpr,
):
    M: tl.constexpr = QC * 4 * PACK
    DP: tl.constexpr = D * PACK
    for task in range(tl.program_id(0), B * HKG, 40):
        batch = task // HKG
        head_group = task % HKG
        begin = tl.load(Cuq + batch).to(tl.int64)
        length = tl.load(Cuq + batch + 1) - begin
        if length > 0:
            if PACK == 1:
                source = tl.make_block_ptr(
                    Query + begin * SQ + head_group * 4 * D,
                    (length, 4 * D),
                    (SQ, 1),
                    (0, 0),
                    (QC, 4 * D),
                    (1, 0),
                )
                target = tl.make_block_ptr(
                    Packed + task * M * DP,
                    (QC, 4 * D),
                    (4 * D, 1),
                    (0, 0),
                    (QC, 4 * D),
                    (1, 0),
                )
                tl.store(
                    target, tl.load(source, boundary_check=(0,), padding_option="zero")
                )
            else:
                target = tl.make_block_ptr(
                    Packed + task * M * DP, (M, DP), (DP, 1), (0, 0), (M, DP), (1, 0)
                )
                tl.store(target, tl.zeros((M, DP), tl.int8))
                for local_head in range(PACK):
                    for query in range(QC):
                        if query < length:
                            source = tl.make_block_ptr(
                                Query
                                + (begin + query) * SQ
                                + (head_group * PACK + local_head) * 4 * D,
                                (4, D),
                                (D, 1),
                                (0, 0),
                                (4, D),
                                (1, 0),
                            )
                            diagonal = tl.make_block_ptr(
                                Packed + task * M * DP,
                                (M, DP),
                                (DP, 1),
                                ((local_head * QC + query) * 4, local_head * D),
                                (4, D),
                                (1, 0),
                            )
                            tl.store(diagonal, tl.load(source))
                        else:
                            pass
        else:
            pass


@triton.jit
def decode_pack_key_value(
    K,
    V,
    PackedK,
    PackedV,
    Table,
    Used,
    batch: tl.constexpr,
    heads: tl.constexpr,
    capacity: tl.constexpr,
    page: tl.constexpr,
    table_stride: tl.constexpr,
    sk: tl.constexpr,
    pk: tl.constexpr,
    sv: tl.constexpr,
    pv: tl.constexpr,
    D: tl.constexpr,
    Meta,
    PAGES: tl.constexpr,
    ATTENTION_N: tl.constexpr,
):
    W: tl.constexpr = heads * D
    pages = tl.arange(0, PAGES)
    for sequence in range(tl.program_id(0), batch, 40):
        length = tl.load(Used + sequence)
        first = tl.load(Table + sequence * table_stride, length > 0, 0).to(tl.int64)
        physical = tl.load(
            Table + sequence * table_stride + pages, pages * page < length, 0
        ).to(tl.int64)
        contiguous = (
            tl.sum(
                tl.where(
                    pages * page < length, (physical == first + pages).to(tl.int32), 1
                ),
                0,
            )
            == PAGES
        )
        contiguous &= length % ATTENTION_N == 0
        tl.store(Meta + sequence, tl.where(contiguous, first, -1).to(tl.int32))
        if not contiguous:
            for logical in range(tl.cdiv(length, ATTENTION_N) * (ATTENTION_N // page)):
                source_page = tl.load(
                    Table + sequence * table_stride + logical,
                    logical * page < length,
                    0,
                ).to(tl.int64)
                source_k = tl.make_block_ptr(
                    K + source_page * pk, (page, W), (sk, 1), (0, 0), (page, W), (1, 0)
                )
                target_k = tl.make_block_ptr(
                    PackedK + sequence * capacity * W,
                    (capacity, W),
                    (W, 1),
                    (logical * page, 0),
                    (page, W),
                    (1, 0),
                )
                tl.store(target_k, tl.load(source_k))
                source_v = tl.make_block_ptr(
                    V + source_page * pv, (page, W), (sv, 1), (0, 0), (page, W), (1, 0)
                )
                target_v = tl.make_block_ptr(
                    PackedV + sequence * capacity * W,
                    (capacity, W),
                    (W, 1),
                    (logical * page, 0),
                    (page, W),
                    (1, 0),
                )
                tl.store(target_v, tl.load(source_v))
        else:
            pass


@triton.jit
def prepare_decode_inputs(
    Q, Packed, K, V, Key, Value, Cuq, Used, Table, VS, Flags, Meta, CONFIG: tl.constexpr
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    B: tl.constexpr = CONFIG.batch
    HKG: tl.constexpr = CONFIG.kv_heads // CONFIG.head_pack
    D: tl.constexpr = CONFIG.dim
    QC: tl.constexpr = CONFIG.query_cap
    PACK: tl.constexpr = CONFIG.head_pack
    SQ: tl.constexpr = CONFIG.query_stride
    SVB: tl.constexpr = CONFIG.v_scale_strides[0]
    SVH: tl.constexpr = CONFIG.v_scale_strides[1]
    SVK: tl.constexpr = CONFIG.v_scale_strides[2]
    SCALE_PAD: tl.constexpr = CONFIG.scale_blocks
    N: tl.constexpr = CONFIG.block_n
    pack_queries_diagonal(Q, Packed, Cuq, B, HKG, D, SQ, QC, PACK)
    scale_blocks = tl.arange(0, SCALE_PAD)
    for task in range(tl.program_id(0), B * HKG, 40):
        batch = task // HKG
        head_group = task % HKG
        nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        if nq > 0:
            nk = tl.load(Used + batch)
            heads = head_group * PACK + tl.arange(0, PACK)
            if SVK == 0:
                values = tl.load(VS + batch * SVB + heads * SVH)
                maximum = tl.max(tl.abs(values), 0)
            else:
                values = tl.load(
                    VS
                    + batch * SVB
                    + heads[:, None] * SVH
                    + scale_blocks[None, :] * SVK,
                    scale_blocks[None, :] < tl.cdiv(nk, 128),
                    0,
                )
                maximum = tl.max(tl.max(tl.abs(values), 1), 0)
            rounding = (
                8.0 * (tl.cdiv(nk, N) + 4) * 5.960464477539063e-08 * (1.0 + 0.008 * nk)
            )
            if DIGITS == 2:
                bound = 128.0 * maximum * rounding
                index = tl.arange(0, 256)
                payload = tl.where(
                    index & 127 == 1,
                    maximum.to(tl.int32, bitcast=True),
                    bound.to(tl.int32, bitcast=True),
                )
                target = tl.make_block_ptr(
                    Flags + task * 256, (256,), (1,), (0,), (256,), (0,)
                )
                tl.store(target, payload)
            else:
                quantization = nk * (1.0 / (2.0 * 2048383.0) + 1e-09)
                bound = 128.0 * maximum * (quantization + rounding)
                tl.store(Flags + task, (~(bound <= 0.01)).to(tl.int32))
        elif DIGITS == 2:
            target = tl.make_block_ptr(
                Flags + task * 256, (256,), (1,), (0,), (256,), (0,)
            )
            tl.store(target, tl.zeros((256,), tl.int32))
        else:
            tl.store(Flags + task, 0)
    decode_pack_key_value(
        K,
        V,
        Key,
        Value,
        Table,
        Used,
        B,
        CONFIG.kv_heads,
        CONFIG.key_capacity,
        16,
        CONFIG.table_stride,
        CONFIG.key_stride,
        CONFIG.key_page_stride,
        CONFIG.value_stride,
        CONFIG.value_page_stride,
        D,
        Meta,
        CONFIG.table_blocks,
        N,
    )


@triton.jit
def decode_cube_score(Query, Key, Score, column_tile, nk, CONFIG: tl.constexpr):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    DP: tl.constexpr = CONFIG.dim * CONFIG.head_pack
    CN0: tl.constexpr = N if N <= 32768 // M else 32768 // M
    CN: tl.constexpr = CN0 if CN0 <= 131072 // DP else 131072 // DP
    query_ptr = tl.make_block_ptr(Query, (M, DP), (DP, 1), (0, 0), (M, DP), (1, 0))
    query = tl.load(query_ptr)
    for column in range(0, N, CN):
        key_ptr = tl.make_block_ptr(
            Key,
            (CONFIG.key_capacity, DP),
            (CONFIG.kv_heads * CONFIG.dim, 1),
            (column_tile * N + column, 0),
            (CN, DP),
            (1, 0),
        )
        score = tl.dot(query, tl.trans(tl.load(key_ptr)), out_dtype=tl.int32)
        target = tl.make_block_ptr(Score, (M, N), (N, 1), (0, column), (M, CN), (1, 0))
        tl.store(target, score)


@triton.jit
def decode_cube_product(
    Probability,
    Value,
    Product,
    column_tile,
    CONFIG: tl.constexpr,
    RESIDUAL: tl.constexpr,
    nk,
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    PLANES: tl.constexpr = 2 if DIGITS == 2 else 4
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    DP: tl.constexpr = CONFIG.dim * CONFIG.head_pack
    KN: tl.constexpr = (
        256
        if DIGITS == 2 and CONFIG.query_cap == 1 and (N >= 256)
        else 512 if N >= 512 else N
    )
    accumulator = tl.zeros((PLANES * M, DP), tl.int32)
    for column in range(0, N, KN):
        vp = tl.make_block_ptr(
            Value,
            (CONFIG.key_capacity, DP),
            (CONFIG.kv_heads * CONFIG.dim, 1),
            (column_tile * N + column, 0),
            (KN, DP),
            (1, 0),
        )
        pp = tl.make_block_ptr(
            Probability, (DIGITS * M, N), (N, 1), (0, column), (PLANES * M, KN), (1, 0)
        )
        if DIGITS == 2:
            p = tl.load(pp)
        else:
            p = tl.load(pp, boundary_check=(0,), padding_option="zero")
        accumulator = tl.dot(p, tl.load(vp), accumulator, out_dtype=tl.int32)
    target = tl.make_block_ptr(
        Product, (PLANES * M, DP), (DP, 1), (0, 0), (PLANES * M, DP), (1, 0)
    )
    tl.store(target, accumulator)


@triton.jit
def cube_task(
    query,
    key_base,
    value_base,
    Score,
    Probability,
    Product,
    core,
    nq,
    nk,
    tile,
    CONFIG: tl.constexpr,
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    PLANES: tl.constexpr = 2 if CONFIG.probability_digits == 2 else 4
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim * CONFIG.head_pack
    Q_ROWS: tl.constexpr = M // CONFIG.group
    end = nk
    if CONFIG.causal:
        end = tl.minimum(nk, (tile + 1) * Q_ROWS + nk - nq)
    else:
        pass
    count = tl.cdiv(tl.maximum(end, 0), N)
    if count > 0:
        decode_cube_score(query, key_base, Score + core * 2 * M * N, 0, nk, CONFIG)
        stage_sync(0, 0, False)
        for column_tile in tl.range(0, count, disable_licm=True):
            slot = column_tile % 2
            if column_tile + 1 < count:
                next_slot = (column_tile + 1) % 2
                decode_cube_score(
                    query,
                    key_base,
                    Score + (core * 2 + next_slot) * M * N,
                    column_tile + 1,
                    nk,
                    CONFIG,
                )
                stage_sync(next_slot, 0, False)
            else:
                pass
            stage_sync(slot, 1, True)
            probability_base = Probability + (core * 2 + slot) * DIGITS * M * N
            product_base = Product + (core * 2 + slot) * PLANES * M * D
            decode_cube_product(
                probability_base,
                value_base,
                product_base,
                column_tile,
                CONFIG,
                False,
                nk,
            )
            stage_sync(slot, 2, False)
    else:
        pass


@triton.jit
def decode_cube_main(
    Query,
    Key,
    Value,
    Cuq,
    Used,
    RawK,
    RawV,
    Meta,
    ValueScale,
    Score,
    Probability,
    Product,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    D: tl.constexpr = CONFIG.dim
    PACK: tl.constexpr = CONFIG.head_pack
    HK: tl.constexpr = CONFIG.kv_heads
    HKG: tl.constexpr = HK // PACK
    core = tl.program_id(0)
    for task in tl.range(core, CONFIG.batch * HKG, CONFIG.cores, disable_licm=True):
        batch = task // HKG
        head = task % HKG
        nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        nk = tl.load(Used + batch)
        if nq > 0:
            query = Query + (batch * HKG + head) * M * D * PACK
            first = tl.load(Meta + batch).to(tl.int64)
            if first >= 0:
                cube_task(
                    query,
                    RawK + (first * 16 * HK + head * PACK) * D,
                    RawV + (first * 16 * HK + head * PACK) * D,
                    Score,
                    Probability,
                    Product,
                    core,
                    nq,
                    nk,
                    0,
                    CONFIG,
                )
            else:
                cube_task(
                    query,
                    Key + (batch * CONFIG.key_capacity * HK + head * PACK) * D,
                    Value + (batch * CONFIG.key_capacity * HK + head * PACK) * D,
                    Score,
                    Probability,
                    Product,
                    core,
                    nq,
                    nk,
                    0,
                    CONFIG,
                )
        else:
            pass
    al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_S)


@triton.jit
def decode_softmax_block(
    Score,
    Probability,
    QueryScale,
    KeyScale,
    ValueScale,
    Alibi,
    MaxState,
    SumState,
    AlphaState,
    batch,
    head,
    tile,
    nq,
    nk,
    column_tile,
    sub,
    CONFIG: tl.constexpr,
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    VR: tl.constexpr = M // 2
    CHUNK: tl.constexpr = CONFIG.block_m // 2 if CONFIG.query_cap > 1 else CONFIG.group
    PAD: tl.constexpr = 1 if CONFIG.query_cap > 1 else 2
    G: tl.constexpr = CONFIG.group
    QSHIFT: tl.constexpr = (
        0
        if CONFIG.query_cap == 1
        else 1 if CONFIG.query_cap == 2 else 2 if CONFIG.query_cap == 4 else 3
    )
    columns = column_tile * N + tl.arange(0, N)
    descale_block = column_tile * (N // 128)
    slot = column_tile % 2
    for piece in tl.range(0, VR, CHUNK, loop_unroll_factor=1, disable_licm=True):
        query_rows = sub * VR + piece + tl.arange(0, CHUNK) >> 2 & CONFIG.query_cap - 1
        kv_head = head * CONFIG.head_pack + (sub * VR + piece >> QSHIFT + 2)
        query_heads = kv_head * G + (sub * VR + piece + tl.arange(0, CHUNK) & 3)
        valid_rows = query_rows < nq
        query_scale = tl.load(
            QueryScale
            + batch * CONFIG.q_scale_strides[0]
            + query_heads * CONFIG.q_scale_strides[1],
            valid_rows,
            0,
        )
        sp = tl.make_block_ptr(
            Score, (M, N), (N, 1), (sub * VR + piece, 0), (CHUNK, N), (1, 0)
        )
        scores = tl.load(sp).to(tl.float32)
        key_scale = tl.load(
            KeyScale
            + batch * CONFIG.k_scale_strides[0]
            + kv_head * CONFIG.k_scale_strides[1]
            + descale_block * CONFIG.k_scale_strides[2]
        )
        scores *= (query_scale * key_scale * CONFIG.scale)[:, None]
        scores = apply_score_modifiers(
            scores, Alibi, batch, query_heads, query_rows, columns, nq, nk, CONFIG
        )
        max_slot = tle.dsa.subview(MaxState, [piece * PAD], [CHUNK], [1])
        sum_slot = tle.dsa.subview(SumState, [piece * PAD], [CHUNK], [1])
        alpha_slot = tle.dsa.subview(
            AlphaState, [slot * VR * PAD + piece * PAD], [CHUNK], [1]
        )
        maximum = tle.dsa.to_tensor(max_slot, writable=False)
        denominator = tle.dsa.to_tensor(sum_slot, writable=False)
        block_maximum = tl.max(scores, 1)
        new_maximum = tl.maximum(maximum, block_maximum)
        safe_maximum = tl.where(new_maximum == float("-inf"), 0, new_maximum)
        alpha = tl.exp2(maximum - safe_maximum)
        if DIGITS == 2:
            safe_block_maximum = tl.where(
                block_maximum == float("-inf"), 0.0, block_maximum
            )
            beta = tl.exp2(block_maximum - safe_maximum)
            probability = tl.exp2(scores - safe_block_maximum[:, None])
            new_denominator = denominator * alpha + beta * tl.sum(probability, 1)
        else:
            probability = tl.exp2(scores - safe_maximum[:, None])
            new_denominator = denominator * alpha + tl.sum(probability, 1)
        tle.dsa.to_buffer(alpha, bind_buffer=alpha_slot)
        if DIGITS == 2:
            beta_slot = tle.dsa.subview(
                AlphaState, [3 * VR * PAD + slot * VR * PAD + piece * PAD], [CHUNK], [1]
            )
            tle.dsa.to_buffer(beta, bind_buffer=beta_slot)
        else:
            pass
        tle.dsa.to_buffer(new_maximum, bind_buffer=max_slot)
        tle.dsa.to_buffer(new_denominator, bind_buffer=sum_slot)
        remainder = probability
        for digit in tl.static_range(DIGITS):
            integer = tl.floor(remainder * 127.0 ** (digit + 1) + 0.5)
            quantized = al.cast(integer, tl.int8, overflow_mode="saturate")
            pp = tl.make_block_ptr(
                Probability + digit * M * N,
                (M, N),
                (N, 1),
                (sub * VR + piece, 0),
                (CHUNK, N),
                (1, 0),
            )
            tl.store(pp, quantized)
            remainder -= integer * (1.0 / 127.0 ** (digit + 1))
        if DIGITS == 2:
            error_slot = tle.dsa.subview(
                AlphaState, [2 * VR * PAD + piece * PAD], [CHUNK], [1]
            )
            previous_error = tle.dsa.to_tensor(error_slot, writable=False)
            value_scale = tl.load(
                ValueScale
                + batch * CONFIG.v_scale_strides[0]
                + kv_head * CONFIG.v_scale_strides[1]
                + descale_block * CONFIG.v_scale_strides[2]
            )
            block_error = (
                128.0 * tl.abs(value_scale) * tl.sum(tl.abs(remainder) + 2e-08, 1)
            )
            tle.dsa.to_buffer(
                previous_error * alpha + block_error * beta, bind_buffer=error_slot
            )
        else:
            pass


@triton.jit
def decode_accumulate_product(
    Product,
    ValueScale,
    AccState,
    AlphaState,
    batch,
    head,
    column_tile,
    sub,
    CONFIG: tl.constexpr,
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    DP: tl.constexpr = D * CONFIG.head_pack
    VR: tl.constexpr = M // 2
    G: tl.constexpr = CONFIG.group
    QSHIFT: tl.constexpr = (
        0
        if CONFIG.query_cap == 1
        else 1 if CONFIG.query_cap == 2 else 2 if CONFIG.query_cap == 4 else 3
    )
    PAD: tl.constexpr = 1 if CONFIG.query_cap > 1 else 2
    C: tl.constexpr = VR if CONFIG.query_cap > 1 else G
    slot = column_tile % 2
    for piece in range(0, VR, C):
        local_head = sub * VR + piece >> QSHIFT + 2
        kv_head = head * CONFIG.head_pack + local_head
        value_scale = tl.load(
            ValueScale
            + batch * CONFIG.v_scale_strides[0]
            + kv_head * CONFIG.v_scale_strides[1]
            + column_tile * (N // 128) * CONFIG.v_scale_strides[2]
        )
        acc_slot = tle.dsa.subview(AccState, [piece, 0], [C, D], [1, 1])
        alpha_slot = tle.dsa.subview(
            AlphaState, [slot * VR * PAD + piece * PAD], [C], [1]
        )
        accumulator = tle.dsa.to_tensor(acc_slot, writable=False)
        alpha = tle.dsa.to_tensor(alpha_slot, writable=False)
        if DIGITS == 2:
            beta_slot = tle.dsa.subview(
                AlphaState, [3 * VR * PAD + slot * VR * PAD + piece * PAD], [C], [1]
            )
            beta = tle.dsa.to_tensor(beta_slot, writable=False)
        else:
            pass
        product = tl.zeros((C, D), tl.float32)
        for digit in tl.static_range(DIGITS):
            source = tl.make_block_ptr(
                Product + digit * M * DP,
                (M, DP),
                (DP, 1),
                (sub * VR + piece, local_head * D),
                (C, D),
                (1, 0),
            )
            product += tl.load(source).to(tl.float32) * (1.0 / 127.0 ** (digit + 1))
        if DIGITS == 2:
            tle.dsa.to_buffer(
                accumulator * alpha[:, None] + product * (value_scale * beta)[:, None],
                bind_buffer=acc_slot,
            )
        else:
            tle.dsa.to_buffer(
                accumulator * alpha[:, None] + product * value_scale,
                bind_buffer=acc_slot,
            )


@triton.jit
def write_single_output(
    Output,
    LSE,
    Cuq,
    AccState,
    MaxState,
    SumState,
    batch,
    head,
    sub,
    CONFIG: tl.constexpr,
    Flags,
    nk,
    task,
    ErrorState,
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    G: tl.constexpr = CONFIG.group
    D: tl.constexpr = CONFIG.dim
    VR: tl.constexpr = CONFIG.block_m // 2
    QSHIFT: tl.constexpr = (
        0
        if CONFIG.query_cap == 1
        else 1 if CONFIG.query_cap == 2 else 2 if CONFIG.query_cap == 4 else 3
    )
    PAD: tl.constexpr = 1 if CONFIG.query_cap > 1 else 2
    begin = tl.load(Cuq + batch).to(tl.int64)
    nq = tl.load(Cuq + batch + 1) - begin
    if DIGITS == 2:
        metadata = Flags + task * 256 + sub * 128
        rounding_bound = tl.load(metadata).to(tl.float32, bitcast=True)
        unsafe = tl.full((), 0, tl.int32)
    else:
        pass
    for piece in range(0, VR, G):
        query_index = sub * VR + piece >> 2 & CONFIG.query_cap - 1
        local_head = sub * VR + piece >> QSHIFT + 2
        query_head = (head * CONFIG.head_pack + local_head) * G
        if query_index < nq:
            acc_slot = tle.dsa.subview(AccState, [piece, 0], [G, D], [1, 1])
            sum_slot = tle.dsa.subview(SumState, [piece * PAD], [G], [1])
            accumulator = tle.dsa.to_tensor(acc_slot, writable=False)
            denominator = tle.dsa.to_tensor(sum_slot, writable=False)
            if DIGITS == 2:
                error_slot = tle.dsa.subview(
                    ErrorState, [2 * VR * PAD + piece * PAD], [G], [1]
                )
                quantization_bound = tle.dsa.to_tensor(
                    error_slot, writable=False
                ) / tl.where(denominator > 0, denominator, 1)
                bound = tl.where(
                    denominator > 0, quantization_bound + rounding_bound, 0.0
                )
                unsafe = tl.maximum(unsafe, tl.max((~(bound <= 0.01)).to(tl.int32), 0))
            else:
                pass
            output = accumulator / tl.where(denominator > 0, denominator, 1)[:, None]
            target = tl.make_block_ptr(
                Output + (begin + query_index) * CONFIG.output_stride + query_head * D,
                (G, D),
                (D, 1),
                (0, 0),
                (G, D),
                (1, 0),
            )
            tl.store(target, output.to(Output.dtype.element_ty))
            if CONFIG.write_lse:
                max_slot = tle.dsa.subview(MaxState, [piece * PAD], [G], [1])
                maximum = tle.dsa.to_tensor(max_slot, writable=False)
                lse = tl.where(
                    denominator > 0,
                    maximum * 0.6931471805599453 + tl.log(denominator),
                    float("inf"),
                )
                tl.store(
                    LSE
                    + (query_head + tl.arange(0, G)) * CONFIG.total_query
                    + begin
                    + query_index,
                    lse,
                )
            else:
                pass
        else:
            pass
    if DIGITS == 2:
        flag_target = tl.make_block_ptr(metadata, (128,), (1,), (0,), (128,), (0,))
        tl.store(flag_target, tl.full((128,), unsafe, tl.int32))
    else:
        pass


@triton.jit
def decode_vector_main(
    Output,
    LSE,
    Cuq,
    Used,
    QueryScale,
    KeyScale,
    ValueScale,
    Alibi,
    Score,
    Probability,
    Product,
    Flags,
    CONFIG: tl.constexpr,
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    PLANES: tl.constexpr = 2 if CONFIG.probability_digits == 2 else 4
    SLOTS: tl.constexpr = 5 if DIGITS == 2 else 2
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    VR: tl.constexpr = M // 2
    PAD: tl.constexpr = 1 if CONFIG.query_cap > 1 else 2
    G: tl.constexpr = CONFIG.group
    B: tl.constexpr = CONFIG.batch
    HK: tl.constexpr = CONFIG.kv_heads // CONFIG.head_pack
    core = tl.program_id(0)
    sub = al.sub_vec_id().to(tl.int32)
    max_state = tle.dsa.alloc([VR * PAD], tl.float32, tle.dsa.ascend.UB)
    sum_state = tle.dsa.alloc([VR * PAD], tl.float32, tle.dsa.ascend.UB)
    alpha_state = tle.dsa.alloc([SLOTS * VR * PAD], tl.float32, tle.dsa.ascend.UB)
    acc_state = tle.dsa.alloc([VR, D], tl.float32, tle.dsa.ascend.UB)
    for task in tl.range(
        core, CONFIG.query_tiles * B * HK, CONFIG.cores, disable_licm=True
    ):
        tile = task // (B * HK)
        batch = task // HK % B
        head = task % HK
        nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        nk = tl.load(Used + batch)
        if tile * (M // G) < nq:
            zero = ub_zero()
            negative_infinity = (zero.to(tl.uint32, bitcast=True) | 0xFF800000).to(
                tl.float32, bitcast=True
            )
            tle.dsa.to_buffer(
                tl.full((VR * PAD,), negative_infinity, tl.float32),
                bind_buffer=max_state,
            )
            tle.dsa.to_buffer(
                tl.full((VR * PAD,), zero, tl.float32), bind_buffer=sum_state
            )
            tle.dsa.to_buffer(tl.full((VR, D), zero, tl.float32), bind_buffer=acc_state)
            if DIGITS == 2:
                error_state = tle.dsa.subview(
                    alpha_state, [2 * VR * PAD], [VR * PAD], [1]
                )
                tle.dsa.to_buffer(
                    tl.full((VR * PAD,), zero, tl.float32), bind_buffer=error_state
                )
            else:
                pass
            end = nk
            if CONFIG.causal:
                end = tl.minimum(nk, (tile + 1) * (M // G) + nk - nq)
            else:
                pass
            count = tl.cdiv(tl.maximum(end, 0), N)
            if count > 0:
                stage_sync(0, 0, True)
                decode_softmax_block(
                    Score + core * 2 * M * N,
                    Probability + core * 2 * DIGITS * M * N,
                    QueryScale,
                    KeyScale,
                    ValueScale,
                    Alibi,
                    max_state,
                    sum_state,
                    alpha_state,
                    batch,
                    head,
                    tile,
                    nq,
                    nk,
                    0,
                    sub,
                    CONFIG,
                )
                stage_sync(0, 1, False)
                for column_tile in tl.range(0, count, disable_licm=True):
                    slot = column_tile % 2
                    if column_tile + 1 < count:
                        next_slot = (column_tile + 1) % 2
                        stage_sync(next_slot, 0, True)
                        decode_softmax_block(
                            Score + (core * 2 + next_slot) * M * N,
                            Probability + (core * 2 + next_slot) * DIGITS * M * N,
                            QueryScale,
                            KeyScale,
                            ValueScale,
                            Alibi,
                            max_state,
                            sum_state,
                            alpha_state,
                            batch,
                            head,
                            tile,
                            nq,
                            nk,
                            column_tile + 1,
                            sub,
                            CONFIG,
                        )
                        stage_sync(next_slot, 1, False)
                    else:
                        pass
                    stage_sync(slot, 2, True)
                    decode_accumulate_product(
                        Product + (core * 2 + slot) * PLANES * M * D * CONFIG.head_pack,
                        ValueScale,
                        acc_state,
                        alpha_state,
                        batch,
                        head,
                        column_tile,
                        sub,
                        CONFIG,
                    )
            else:
                pass
            write_single_output(
                Output,
                LSE,
                Cuq,
                acc_state,
                max_state,
                sum_state,
                batch,
                head,
                sub,
                CONFIG,
                Flags,
                nk,
                task,
                alpha_state,
            )
        else:
            pass
    al.sync_block_set("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_S)


@triton.jit
def replay_ready(WAIT: tl.constexpr):
    if WAIT:
        al.sync_block_wait("cube", "vector", 0, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
    else:
        al.sync_block_set("cube", "vector", 0, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)


@triton.jit
def replay_consumed(WAIT: tl.constexpr):
    if WAIT:
        al.sync_block_wait("vector", "cube", 1, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)
    else:
        al.sync_block_set("vector", "cube", 1, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)


@triton.jit
def replay_cube_task(Query, Key, Score, nk, CONFIG: tl.constexpr):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    for column in range(tl.cdiv(nk, N)):
        decode_cube_score(Query, Key, Score + column * M * N, column, nk, CONFIG)
    replay_ready(False)
    replay_consumed(True)


@triton.jit
def replay_cube_main(
    Query, Key, RawK, Meta, Cuq, Used, Flags, Score, CONFIG: tl.constexpr
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    PACK: tl.constexpr = CONFIG.head_pack
    HK: tl.constexpr = CONFIG.kv_heads
    HKG: tl.constexpr = HK // PACK
    BLOCKS: tl.constexpr = tl.cdiv(CONFIG.key_capacity, N)
    core = tl.program_id(0)
    for task in tl.range(core, CONFIG.batch * HKG, CONFIG.cores, disable_licm=True):
        batch = task // HKG
        head = task % HKG
        if CONFIG.probability_digits == 2:
            flag = tl.load(Flags + task * 256, volatile=True) | tl.load(
                Flags + task * 256 + 128, volatile=True
            )
        else:
            flag = tl.load(Flags + task, volatile=True)
        if flag != 0:
            nk = tl.load(Used + batch)
            first = tl.load(Meta + batch).to(tl.int64)
            query = Query + task * M * D * PACK
            target = Score + core * BLOCKS * M * N
            if first >= 0:
                replay_cube_task(
                    query,
                    RawK + (first * 16 * HK + head * PACK) * D,
                    target,
                    nk,
                    CONFIG,
                )
            else:
                replay_cube_task(
                    query,
                    Key + (batch * CONFIG.key_capacity * HK + head * PACK) * D,
                    target,
                    nk,
                    CONFIG,
                )
        else:
            pass
    al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)


@triton.jit
def replay_score_values(
    Score,
    QS,
    KS,
    Alibi,
    batch,
    query_head,
    kv_head,
    query,
    nq,
    nk,
    column,
    offset,
    logical_row,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    columns = column * N + offset + tl.arange(0, 64)
    values = tl.load(
        Score + column * M * N + logical_row * N + offset + tl.arange(0, 64)
    ).to(tl.float32)
    qs = tl.load(
        QS + batch * CONFIG.q_scale_strides[0] + query_head * CONFIG.q_scale_strides[1]
    )
    ks = tl.load(
        KS
        + batch * CONFIG.k_scale_strides[0]
        + kv_head * CONFIG.k_scale_strides[1]
        + column * (N // 128) * CONFIG.k_scale_strides[2]
    )
    values *= qs * ks * CONFIG.scale
    scores = apply_score_modifiers(
        values[None, :],
        Alibi,
        batch,
        tl.full((1,), query_head, tl.int32),
        tl.full((1,), query, tl.int32),
        columns,
        nq,
        nk,
        CONFIG,
    )
    return tl.reshape(scores, (64,))


@triton.jit
def replay_rows(
    Score,
    Value,
    Output,
    QS,
    KS,
    VS,
    Alibi,
    begin,
    batch,
    head,
    sub,
    nq,
    nk,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    G: tl.constexpr = CONFIG.group
    VR: tl.constexpr = M // 2
    QSHIFT: tl.constexpr = (
        0
        if CONFIG.query_cap == 1
        else 1 if CONFIG.query_cap == 2 else 2 if CONFIG.query_cap == 4 else 3
    )
    for local_row in range(VR):
        logical_row = sub * VR + local_row
        query = logical_row >> 2 & CONFIG.query_cap - 1
        local_head = logical_row >> QSHIFT + 2
        kv_head = head * CONFIG.head_pack + local_head
        query_head = kv_head * G + (logical_row & 3)
        if query < nq:
            maximum = tl.full((), float("-inf"), tl.float32)
            for column in range(tl.cdiv(nk, N)):
                for offset in range(0, N, 64):
                    score = replay_score_values(
                        Score,
                        QS,
                        KS,
                        Alibi,
                        batch,
                        query_head,
                        kv_head,
                        query,
                        nq,
                        nk,
                        column,
                        offset,
                        logical_row,
                        CONFIG,
                    )
                    maximum = tl.maximum(maximum, tl.max(score, 0))
            safe_maximum = tl.where(maximum == float("-inf"), 0, maximum)
            numerator = tl.zeros((D,), tl.float32)
            denominator = tl.full((), 0.0, tl.float32)
            for column in range(tl.cdiv(nk, N)):
                scale = tl.load(
                    VS
                    + batch * CONFIG.v_scale_strides[0]
                    + kv_head * CONFIG.v_scale_strides[1]
                    + column * (N // 128) * CONFIG.v_scale_strides[2]
                )
                for offset in range(0, N, 64):
                    score = replay_score_values(
                        Score,
                        QS,
                        KS,
                        Alibi,
                        batch,
                        query_head,
                        kv_head,
                        query,
                        nq,
                        nk,
                        column,
                        offset,
                        logical_row,
                        CONFIG,
                    )
                    probability = tl.exp2(score - safe_maximum)
                    denominator += tl.sum(probability, 0)
                    vp = tl.make_block_ptr(
                        Value,
                        (CONFIG.key_capacity, CONFIG.head_pack * D),
                        (CONFIG.kv_heads * D, 1),
                        (column * N + offset, local_head * D),
                        (64, D),
                        (1, 0),
                    )
                    values = tl.load(vp).to(tl.float32)
                    numerator += tl.sum(probability[:, None] * values, 0) * scale
            output = numerator / tl.where(denominator > 0, denominator, 1)
            tl.store(
                Output
                + (begin + query) * CONFIG.output_stride
                + query_head * D
                + tl.arange(0, D),
                output.to(Output.dtype.element_ty),
            )
        else:
            pass


@triton.jit
def replay_vector_main(
    Value,
    RawV,
    Meta,
    Output,
    Cuq,
    Used,
    QS,
    KS,
    VS,
    Alibi,
    Flags,
    Score,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    PACK: tl.constexpr = CONFIG.head_pack
    HK: tl.constexpr = CONFIG.kv_heads
    HKG: tl.constexpr = HK // PACK
    BLOCKS: tl.constexpr = tl.cdiv(CONFIG.key_capacity, N)
    core = tl.program_id(0)
    sub = al.sub_vec_id().to(tl.int32)
    for task in tl.range(core, CONFIG.batch * HKG, CONFIG.cores, disable_licm=True):
        if CONFIG.probability_digits == 2:
            flag_pointer = tl.make_block_ptr(
                Flags + task * 256, (256,), (1,), (0,), (256,), (0,)
            )
            flag = tl.max(tl.load(flag_pointer), 0)
        else:
            flag = tl.load(Flags + task)
        if flag != 0:
            batch = task // HKG
            head = task % HKG
            begin = tl.load(Cuq + batch).to(tl.int64)
            nq = tl.load(Cuq + batch + 1) - begin
            nk = tl.load(Used + batch)
            first = tl.load(Meta + batch).to(tl.int64)
            replay_ready(True)
            if first >= 0:
                replay_rows(
                    Score + core * BLOCKS * M * N,
                    RawV + (first * 16 * HK + head * PACK) * D,
                    Output,
                    QS,
                    KS,
                    VS,
                    Alibi,
                    begin,
                    batch,
                    head,
                    sub,
                    nq,
                    nk,
                    CONFIG,
                )
            else:
                replay_rows(
                    Score + core * BLOCKS * M * N,
                    Value + (batch * CONFIG.key_capacity * HK + head * PACK) * D,
                    Output,
                    QS,
                    KS,
                    VS,
                    Alibi,
                    begin,
                    batch,
                    head,
                    sub,
                    nq,
                    nk,
                    CONFIG,
                )
            replay_consumed(False)
        else:
            pass
    al.sync_block_set("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)


@triton.jit
def decode_attention_kernel(
    Query,
    Key,
    Value,
    Output,
    LSE,
    Cuq,
    Used,
    RawK,
    RawV,
    Meta,
    QS,
    KS,
    VS,
    Alibi,
    Score,
    Probability,
    Product,
    Flags,
    ReplayScore,
    CONFIG: tl.constexpr,
):
    with al.scope(core_mode="cube"):
        decode_cube_main(
            Query,
            Key,
            Value,
            Cuq,
            Used,
            RawK,
            RawV,
            Meta,
            VS,
            Score,
            Probability,
            Product,
            CONFIG,
        )
        if CONFIG.probability_digits == 2:
            al.sync_block_set("cube", "vector", 7, PIPE.PIPE_S, PIPE.PIPE_MTE2)
        else:
            pass
        replay_cube_main(Query, Key, RawK, Meta, Cuq, Used, Flags, ReplayScore, CONFIG)
    with al.scope(core_mode="vector"):
        decode_vector_main(
            Output,
            LSE,
            Cuq,
            Used,
            QS,
            KS,
            VS,
            Alibi,
            Score,
            Probability,
            Product,
            Flags,
            CONFIG,
        )
        if CONFIG.probability_digits == 2:
            al.sync_block_wait("cube", "vector", 7, PIPE.PIPE_S, PIPE.PIPE_MTE2)
        else:
            pass
        replay_vector_main(
            Value,
            RawV,
            Meta,
            Output,
            Cuq,
            Used,
            QS,
            KS,
            VS,
            Alibi,
            Flags,
            ReplayScore,
            CONFIG,
        )


def decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    cuq: torch.Tensor,
    used: torch.Tensor,
    table: torch.Tensor,
    qs: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    max_query: int,
    max_key: int,
    scale: float,
    causal: bool,
    left: int,
    right: int,
    softcap: float,
    alibi: Optional[torch.Tensor],
    write_lse: bool,
) -> None:
    batch = cuq.shape[0] - 1
    heads = k.shape[-2]
    dim = q.shape[-1]
    query_cap = triton.next_power_of_2(max_query)
    pack = 4 if query_cap == 1 else 2 if query_cap <= 4 else 1
    digits = 2 if max_query == 1 and max_key > 1024 else 3
    planes = 2 if digits == 2 else 4
    rows = query_cap * 4 * pack
    columns = (
        (1024 if query_cap == 1 else 512)
        if ks.stride(2) == 0 and vs.stride(2) == 0
        else 128
    )
    capacity = triton.cdiv(max_key, columns) * columns
    groups = heads // pack
    cores = min(20, batch * groups)
    query = torch.empty(
        (batch, groups, rows, dim * pack), dtype=torch.int8, device=q.device
    )
    key = torch.empty((batch, capacity, heads, dim), dtype=torch.int8, device=q.device)
    value = torch.empty(
        (batch, capacity, heads, dim), dtype=torch.int8, device=q.device
    )
    metadata = torch.empty(batch, dtype=torch.int32, device=q.device)
    flag_words = 256 if digits == 2 else 1
    flags = torch.empty(batch * groups * flag_words, dtype=torch.int32, device=q.device)
    score = torch.empty((cores, 2, rows, columns), dtype=torch.int32, device=q.device)
    probability = torch.empty(
        (cores, 2, digits, rows, columns), dtype=torch.int8, device=q.device
    )
    product = torch.empty(
        (cores, 2, planes, rows, dim * pack), dtype=torch.int32, device=q.device
    )
    replay_score = torch.empty(
        (cores, triton.cdiv(capacity, columns), rows, columns),
        dtype=torch.int32,
        device=q.device,
    )
    alibi_batch_stride = 0
    alibi_head_stride = 0
    if alibi is not None:
        alibi_head_stride = alibi.stride(-1)
        if alibi.ndim == 2:
            alibi_batch_stride = alibi.stride(0)
        else:
            pass
    else:
        pass
    configuration = DecodeConfig(
        batch=batch,
        kv_heads=heads,
        group=4,
        dim=dim,
        query_capacity=16,
        key_capacity=capacity,
        query_tiles=1,
        total_query=q.shape[0],
        output_stride=output.stride(0),
        q_scale_strides=qs.stride(),
        k_scale_strides=ks.stride(),
        v_scale_strides=vs.stride(),
        causal=causal,
        write_lse=write_lse,
        scale=scale,
        block_m=rows,
        block_n=columns,
        cores=cores,
        left=left,
        right=right,
        softcap=softcap,
        has_alibi=alibi is not None,
        alibi_batch_stride=alibi_batch_stride,
        alibi_head_stride=alibi_head_stride,
        head_pack=pack,
        query_cap=query_cap,
        probability_digits=digits,
        query_stride=q.stride(0),
        key_stride=k.stride(-3),
        key_page_stride=k.stride(0),
        value_stride=v.stride(-3),
        value_page_stride=v.stride(0),
        table_stride=table.stride(0),
        scale_blocks=triton.next_power_of_2(vs.shape[2]),
        table_blocks=triton.next_power_of_2(table.shape[1]),
    )
    prepare_decode_inputs[40,](
        q, query, k, v, key, value, cuq, used, table, vs, flags, metadata, configuration
    )
    decode_attention_kernel[cores,](
        query,
        key,
        value,
        output,
        lse,
        cuq,
        used,
        k,
        v,
        metadata,
        qs,
        ks,
        vs,
        alibi if alibi is not None else qs,
        score,
        probability,
        product,
        flags,
        replay_score,
        configuration,
        multibuffer=digits == 2 and query_cap == 1,
        enable_auto_bind_sub_block=False,
        disable_auto_inject_block_sync=True,
    )


# Input packing and public dispatch


@triton.jit
def build_worklist(
    Cuq,
    Used,
    QueryBatch,
    QueryTile,
    KeyBatch,
    KeyPage,
    Counts,
    B: tl.constexpr,
    Q_ROWS: tl.constexpr,
    N: tl.constexpr,
    PAGE: tl.constexpr,
    BB: tl.constexpr,
):
    sequences = tl.arange(0, BB)
    begin = tl.load(Cuq + sequences, sequences < B, 0)
    end = tl.load(Cuq + sequences + 1, sequences < B, 0)
    query_counts = tl.cdiv(end - begin, Q_ROWS)
    key_lengths = tl.load(Used + sequences, sequences < B, 0)
    key_counts = tl.cdiv(key_lengths, N) * (N // PAGE)
    if tl.program_id(0) == 0:
        tl.store(Counts, tl.sum(query_counts, 0))
        tl.store(Counts + 1, tl.sum(key_counts, 0))
    else:
        pass
    for sequence in range(tl.program_id(0), B, 40):
        query_offset = tl.sum(tl.where(sequences < sequence, query_counts, 0), 0)
        key_offset = tl.sum(tl.where(sequences < sequence, key_counts, 0), 0)
        query_count = tl.cdiv(
            tl.load(Cuq + sequence + 1) - tl.load(Cuq + sequence), Q_ROWS
        )
        key_count = tl.cdiv(tl.load(Used + sequence), N) * (N // PAGE)
        for start in range(0, query_count, 256):
            index = start + tl.arange(0, 256)
            tl.store(QueryBatch + query_offset + index, sequence, index < query_count)
            tl.store(QueryTile + query_offset + index, index, index < query_count)
        for start in range(0, key_count, 256):
            index = start + tl.arange(0, 256)
            tl.store(KeyBatch + key_offset + index, sequence, index < key_count)
            tl.store(KeyPage + key_offset + index, index, index < key_count)


@triton.jit
def pack_queries(
    Q,
    Packed,
    Cuq,
    QueryBatch,
    QueryTile,
    Counts,
    Q_ROWS: tl.constexpr,
    batch: tl.constexpr,
    heads: tl.constexpr,
    capacity: tl.constexpr,
    group: tl.constexpr,
    sq: tl.constexpr,
    D: tl.constexpr,
    HAS_WORKLIST: tl.constexpr,
    UNIFORM_Q_LENGTH: tl.constexpr = 0,
):
    BLOCK: tl.constexpr = 16 if Q_ROWS >= 16 else Q_ROWS
    tiles: tl.constexpr = capacity // 16
    if HAS_WORKLIST:
        count = tl.load(Counts) * heads
    else:
        count = batch * heads * tiles
    for task in range(tl.program_id(0), count, 40):
        if HAS_WORKLIST:
            head = task % heads
            sequence = tl.load(QueryBatch + task // heads)
            tile = tl.load(QueryTile + task // heads)
        else:
            head = task // tiles % heads
            sequence = task // (tiles * heads)
            tile = task % tiles
        for piece in tl.static_range(0, Q_ROWS, BLOCK):
            row = tile * Q_ROWS + piece
            if UNIFORM_Q_LENGTH > 0:
                begin = (sequence * UNIFORM_Q_LENGTH).to(tl.int64)
            else:
                begin = tl.load(Cuq + sequence).to(tl.int64)
            if UNIFORM_Q_LENGTH > 0:
                length = UNIFORM_Q_LENGTH
            else:
                length = tl.load(Cuq + sequence + 1) - begin
            source = tl.make_block_ptr(
                Q + begin * sq + head * group * D,
                (length, group * D),
                (sq, 1),
                (row, 0),
                (BLOCK, group * D),
                (1, 0),
            )
            target = tl.make_block_ptr(
                Packed + (sequence * heads + head) * capacity * group * D,
                (capacity, group * D),
                (group * D, 1),
                (row, 0),
                (BLOCK, group * D),
                (1, 0),
            )
            tl.store(
                target,
                (
                    tl.load(source)
                    if UNIFORM_Q_LENGTH > 0 and UNIFORM_Q_LENGTH == capacity
                    else tl.load(source, boundary_check=(0,), padding_option="zero")
                ),
                boundary_check=(0,),
            )


@triton.jit
def unpack_output(
    Packed,
    PackedLSE,
    Output,
    LSE,
    Cuq,
    QueryBatch,
    QueryTile,
    Counts,
    Q_ROWS: tl.constexpr,
    B: tl.constexpr,
    HK: tl.constexpr,
    CAP: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    STRIDE: tl.constexpr,
    TOTAL: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    HAS_WORKLIST: tl.constexpr,
    UNIFORM_Q_LENGTH: tl.constexpr = 0,
):
    BLOCK: tl.constexpr = 16 if Q_ROWS >= 16 else Q_ROWS
    tiles: tl.constexpr = CAP // 16
    if HAS_WORKLIST:
        count = tl.load(Counts) * HK
    else:
        count = B * HK * tiles
    for task in range(tl.program_id(0), count, 40):
        if HAS_WORKLIST:
            head = task % HK
            batch = tl.load(QueryBatch + task // HK)
            tile = tl.load(QueryTile + task // HK)
        else:
            head = task // tiles % HK
            batch = task // (tiles * HK)
            tile = task % tiles
        for piece in tl.static_range(0, Q_ROWS, BLOCK):
            row = tile * Q_ROWS + piece
            if UNIFORM_Q_LENGTH > 0:
                begin = batch * UNIFORM_Q_LENGTH
            else:
                begin = tl.load(Cuq + batch)
            if UNIFORM_Q_LENGTH > 0:
                length = UNIFORM_Q_LENGTH
            else:
                length = tl.load(Cuq + batch + 1) - begin
            source = tl.make_block_ptr(
                Packed + (batch * HK + head) * CAP * G * D,
                (CAP, G * D),
                (G * D, 1),
                (row, 0),
                (BLOCK, G * D),
                (1, 0),
            )
            target = tl.make_block_ptr(
                Output + begin.to(tl.int64) * STRIDE + head * G * D,
                (length, G * D),
                (STRIDE, 1),
                (row, 0),
                (BLOCK, G * D),
                (1, 0),
            )
            if (
                UNIFORM_Q_LENGTH > 0
                and UNIFORM_Q_LENGTH == CAP
                or row + BLOCK <= length
            ):
                tl.store(target, tl.load(source))
            else:
                tl.store(target, tl.load(source), boundary_check=(0,))
            if WRITE_LSE:
                source_lse = tl.make_block_ptr(
                    PackedLSE + (batch * HK + head) * CAP * G,
                    (CAP, G),
                    (G, 1),
                    (row, 0),
                    (BLOCK, G),
                    (1, 0),
                )
                target_lse = tl.make_block_ptr(
                    LSE + head * G * TOTAL + begin.to(tl.int64),
                    (G, length),
                    (TOTAL, 1),
                    (0, row),
                    (G, BLOCK),
                    (1, 0),
                )
                tl.store(target_lse, tl.trans(tl.load(source_lse)), boundary_check=(1,))
            else:
                pass


@triton.jit
def pack_key_value(
    K,
    V,
    PackedK,
    PackedV,
    Table,
    Used,
    KeyBatch,
    KeyPage,
    Counts,
    batch: tl.constexpr,
    heads: tl.constexpr,
    capacity: tl.constexpr,
    page: tl.constexpr,
    table_stride: tl.constexpr,
    sk: tl.constexpr,
    pk: tl.constexpr,
    sv: tl.constexpr,
    pv: tl.constexpr,
    D: tl.constexpr,
    HAS_WORKLIST: tl.constexpr,
    KEY_MAJOR: tl.constexpr = False,
):
    pages: tl.constexpr = capacity // page
    width: tl.constexpr = heads * D
    if HAS_WORKLIST:
        count = tl.load(Counts + 1)
    else:
        count = batch * pages
    for task in range(tl.program_id(0), count, 40):
        if HAS_WORKLIST:
            logical_page = tl.load(KeyPage + task)
            sequence = tl.load(KeyBatch + task)
        else:
            logical_page = task % pages
            sequence = task // pages
        length = tl.load(Used + sequence)
        physical_page = tl.load(
            Table + sequence * table_stride + logical_page,
            logical_page * page < length,
            0,
        ).to(tl.int64)
        key = tl.make_block_ptr(
            K + physical_page * pk,
            (page, width),
            (sk, 1),
            (0, 0),
            (page, width),
            (1, 0),
        )
        value = tl.make_block_ptr(
            V + physical_page * pv,
            (page, heads, D),
            (sv, D, 1),
            (0, 0, 0),
            (page, heads, D),
            (2, 1, 0),
        )
        key_target = tl.make_block_ptr(
            PackedK + sequence * capacity * width,
            (capacity, width),
            (width, 1),
            (logical_page * page, 0),
            (page, width),
            (1, 0),
        )
        value_target = tl.make_block_ptr(
            PackedV + sequence * heads * capacity * D,
            (heads, capacity, D),
            (capacity * D, D, 1),
            (0, logical_page * page, 0),
            (heads, page, D),
            (2, 1, 0),
        )
        if KEY_MAJOR:
            key3 = tl.make_block_ptr(
                K + physical_page * pk,
                (page, heads, D),
                (sk, D, 1),
                (0, 0, 0),
                (page, heads, D),
                (2, 1, 0),
            )
            values3 = tl.load(key3)
            target3 = tl.make_block_ptr(
                PackedK + sequence * heads * capacity * D,
                (heads, capacity, D),
                (capacity * D, D, 1),
                (0, logical_page * page, 0),
                (heads, page, D),
                (2, 1, 0),
            )
            tl.store(target3, tl.permute(values3, (1, 0, 2)))
        else:
            tl.store(key_target, tl.load(key))
        tl.store(value_target, tl.permute(tl.load(value).to(tl.float16), (1, 0, 2)))


@triton.jit
def pack_inputs(
    Q,
    K,
    V,
    Query,
    Key,
    Value,
    Cuq,
    Used,
    Table,
    QBatch,
    QTile,
    Counts,
    KBatch,
    KPage,
    Q_ROWS: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    QCAP: tl.constexpr,
    G: tl.constexpr,
    KCAP: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    PK: tl.constexpr,
    SV: tl.constexpr,
    PV: tl.constexpr,
    D: tl.constexpr,
    HAS_WORKLIST: tl.constexpr,
    CACHE_REPEAT: tl.constexpr = 1,
    UNIFORM_Q_LENGTH: tl.constexpr = 0,
    UnsafeCount=None,
    INIT_UNSAFE: tl.constexpr = False,
    DIRECT_QUERY: tl.constexpr = False,
    KEY_MAJOR: tl.constexpr = False,
):
    if INIT_UNSAFE:
        if tl.program_id(0) == 0:
            tl.store(UnsafeCount, 0)
        else:
            pass
    else:
        pass
    if not DIRECT_QUERY:
        pack_queries(
            Q,
            Query,
            Cuq,
            QBatch,
            QTile,
            Counts,
            Q_ROWS,
            B,
            H,
            QCAP,
            G,
            SQ,
            D,
            HAS_WORKLIST,
            UNIFORM_Q_LENGTH,
        )
    else:
        pass
    pack_key_value(
        K,
        V,
        Key,
        Value,
        Table,
        Used,
        KBatch,
        KPage,
        Counts,
        B,
        H // CACHE_REPEAT,
        KCAP,
        PAGE,
        TABLE_STRIDE,
        SK,
        PK,
        SV,
        PV,
        D,
        HAS_WORKLIST,
        KEY_MAJOR,
    )


@triton.jit
def flash_int8_fwd(
    Q,
    K,
    V,
    Output,
    LSE,
    PART_OUT,
    PART_MAX,
    PART_DENOM,
    CUQ,
    CUK,
    USED,
    TABLE,
    QS,
    KS,
    VS,
    ALIBI,
    task_begin,
    batch_size,
    heads,
    group: tl.constexpr,
    max_blocks,
    sq,
    hq,
    sk,
    hk,
    pk,
    sv,
    hv,
    pv,
    so,
    ho,
    qs0,
    qs1,
    qs2,
    ks0,
    ks1,
    ks2,
    vs0,
    vs1,
    vs2,
    table_stride,
    alibi_batch_stride,
    alibi_head_stride,
    total_q,
    scale,
    causal: tl.constexpr,
    left: tl.constexpr,
    right: tl.constexpr,
    softcap: tl.constexpr,
    has_alibi: tl.constexpr,
    write_lse: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    SPLITS: tl.constexpr,
    PACKED_NATIVE: tl.constexpr,
    CAPACITY: tl.constexpr,
    CAPACITY_Q: tl.constexpr,
):
    task = task_begin + tl.program_id(0)
    tile_part = task // (batch_size * heads)
    part = tile_part % SPLITS
    tile = tile_part // SPLITS
    rem = task % (batch_size * heads)
    batch = rem // heads
    kv_head = rem % heads
    lane = tl.arange(0, BLOCK_M)
    rows_per_tile = BLOCK_M // group
    q_row = tile * rows_per_tile + lane // group
    q_head = kv_head * group + lane % group
    q_start_i32 = tl.load(CUQ + batch)
    q_end_i32 = tl.load(CUQ + batch + 1)
    q_start = q_start_i32.to(tl.int64)
    nq = q_end_i32 - q_start_i32
    if PAGED or PACKED_NATIVE:
        k_start = tl.full((), 0, tl.int64)
        nk = tl.load(USED + batch)
    else:
        k_start_i32 = tl.load(CUK + batch)
        k_end_i32 = tl.load(CUK + batch + 1)
        k_start = k_start_i32.to(tl.int64)
        nk = k_end_i32 - k_start_i32
    row_ok = (lane < rows_per_tile * group) & (q_row < nq)
    rows_safe = tl.where(row_ok, q_row, 0).to(tl.int64)
    d = tl.arange(0, D)
    if tile * rows_per_tile < nq:
        if PACKED_NATIVE:
            query_ptr = tl.make_block_ptr(
                Q + (batch * heads + kv_head) * CAPACITY_Q * group * D,
                (CAPACITY_Q * group, D),
                (D, 1),
                (tile * BLOCK_M, 0),
                (BLOCK_M, D),
                (1, 0),
            )
            q = tl.load(query_ptr)
        else:
            q = tl.load(
                Q
                + (q_start + rows_safe[:, None]) * sq
                + q_head[:, None] * hq
                + d[None, :],
                row_ok[:, None],
                0,
            )
        q_scale = tl.load(
            QS + batch * qs0 + q_head * qs1 + q_row // DESCALE_BLOCK * qs2, row_ok, 0
        )
        maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        denom = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        slope = tl.zeros((BLOCK_M,), dtype=tl.float32)
        if has_alibi:
            slope = tl.load(
                ALIBI + batch * alibi_batch_stride + q_head * alibi_head_stride
            )
        else:
            pass
        first = 0
        end = nk
        if left >= 0:
            first = tl.maximum(0, tile * rows_per_tile + nk - nq - left) // BLOCK_N
        else:
            pass
        if causal:
            end = tl.minimum(end, (tile + 1) * rows_per_tile + nk - nq)
        else:
            pass
        if right >= 0:
            end = tl.minimum(end, (tile + 1) * rows_per_tile + nk - nq + right)
        else:
            pass
        part_blocks = (max_blocks + SPLITS - 1) // SPLITS
        for start in tl.range(part * part_blocks, (part + 1) * part_blocks):
            if (start >= first) & (start < max_blocks) & (start * BLOCK_N < end):
                n = start * BLOCK_N + tl.arange(0, BLOCK_N)
                n_ok = n < nk
                n_safe = tl.where(n_ok, n, 0).to(tl.int64)
                if PAGED:
                    k_row = virtual_to_cache_offset(
                        n_safe,
                        nk,
                        TABLE + batch * table_stride,
                        PAGE,
                        sk,
                        pk,
                        boundary_check=True,
                    )
                    v_row = virtual_to_cache_offset(
                        n_safe,
                        nk,
                        TABLE + batch * table_stride,
                        PAGE,
                        sv,
                        pv,
                        boundary_check=True,
                    )
                else:
                    k_row = (k_start + n_safe) * sk
                    v_row = (k_start + n_safe) * sv
                if PACKED_NATIVE:
                    packed_base = (batch * CAPACITY * heads + kv_head) * D
                    key_ptr = tl.make_block_ptr(
                        K + packed_base,
                        (CAPACITY, D),
                        (heads * D, 1),
                        (start * BLOCK_N, 0),
                        (BLOCK_N, D),
                        (1, 0),
                    )
                    k = tl.trans(tl.load(key_ptr))
                elif PAGED and PAGE == BLOCK_N:
                    page_id = tl.load(
                        TABLE + batch * table_stride + start, start * PAGE < nk, 0
                    ).to(tl.int64)
                    k_page = tl.make_block_ptr(
                        base=K + page_id * pk + kv_head * hk,
                        shape=(PAGE, D),
                        strides=(sk, 1),
                        offsets=(0, 0),
                        block_shape=(BLOCK_N, D),
                        order=(1, 0),
                    )
                    k = tl.trans(tl.load(k_page))
                elif PAGED and PAGE == 16:
                    k_rows = tl.full((BLOCK_N, D), 0, tl.int8)
                    first_page = start * (BLOCK_N // PAGE)
                    for page_in_tile in tl.static_range(0, BLOCK_N // PAGE):
                        page_number = first_page + page_in_tile
                        physical_page = tl.load(
                            TABLE + batch * table_stride + page_number,
                            page_number < (nk + PAGE - 1) // PAGE,
                            0,
                        ).to(tl.int64)
                        k_page = tl.make_block_ptr(
                            base=K + physical_page * pk + kv_head * hk,
                            shape=(PAGE, D),
                            strides=(sk, 1),
                            offsets=(0, 0),
                            block_shape=(PAGE, D),
                            order=(1, 0),
                        )
                        k_rows = tle.dsa.insert_slice(
                            k_rows,
                            tl.load(k_page),
                            offsets=[page_in_tile * PAGE, 0],
                            sizes=[PAGE, D],
                            strides=[1, 1],
                        )
                    k = tl.trans(k_rows)
                elif not PAGED and (start + 1) * BLOCK_N <= nk:
                    k_block_ptr = tl.make_block_ptr(
                        base=K + k_start * sk + kv_head * hk,
                        shape=(nk, D),
                        strides=(sk, 1),
                        offsets=(start * BLOCK_N, 0),
                        block_shape=(BLOCK_N, D),
                        order=(1, 0),
                    )
                    k = tl.trans(tl.load(k_block_ptr))
                else:
                    k = tl.load(
                        K + k_row[None, :] + kv_head * hk + d[:, None], n_ok[None, :], 0
                    )
                descale_block = start * BLOCK_N // DESCALE_BLOCK
                ks = tl.load(KS + batch * ks0 + kv_head * ks1 + descale_block * ks2)
                vs = tl.load(VS + batch * vs0 + kv_head * vs1 + descale_block * vs2)
                scores = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
                scores = scores * (q_scale * ks * scale)[:, None]
                if softcap > 0:
                    scores = softcap * apply_softcap(
                        scores, 1.0 / softcap, is_softcap=True
                    )
                else:
                    scores = apply_softcap(scores, 1.0, is_softcap=False)
                scores = apply_alibi(
                    scores,
                    n,
                    q_row,
                    nq,
                    nk,
                    is_causal=False,
                    is_alibi=has_alibi,
                    alibi_slope=slope[:, None],
                )
                scores = apply_mask(
                    scores,
                    n,
                    q_row,
                    nq,
                    nk,
                    nk if left < 0 else left,
                    (
                        (0 if causal else nq)
                        if right < 0
                        else tl.minimum(right, 0) if causal else right
                    ),
                    is_even_mn=False,
                    is_causal=causal,
                    is_local=left >= 0 or right >= 0,
                )
                scores = tl.where(row_ok[:, None], scores * LOG2E, float("-inf"))
                tile_maximum = tl.max(scores, 1)
                difference = maximum - tile_maximum
                alpha = tl.where(
                    maximum == float("-inf"), 0, tl.exp2(tl.minimum(difference, 0))
                )
                beta = tl.where(
                    tile_maximum == float("-inf"),
                    0,
                    tl.exp2(tl.minimum(-difference, 0)),
                )
                safe_tile_maximum = tl.where(
                    tile_maximum == float("-inf"), 0, tile_maximum
                )
                p = tl.exp2(scores - safe_tile_maximum[:, None])
                denom = denom * alpha + tl.sum(p, 1) * beta
                if PACKED_NATIVE:
                    value_ptr = tl.make_block_ptr(
                        V + (batch * heads + kv_head) * CAPACITY * D,
                        (CAPACITY, D),
                        (D, 1),
                        (start * BLOCK_N, 0),
                        (BLOCK_N, D),
                        (1, 0),
                    )
                    v = tl.load(value_ptr)
                else:
                    v = tl.load(
                        V + v_row[:, None] + kv_head * hv + d[None, :], n_ok[:, None], 0
                    ).to(tl.float16)
                probability = p.to(tl.float16)
                partial = tl.dot(probability, v, out_dtype=tl.float32)
                if vs > 0.25:
                    residual = (p - probability.to(tl.float32)).to(tl.float16)
                    partial += tl.dot(residual, v, out_dtype=tl.float32)
                else:
                    pass
                acc = acc * alpha[:, None] + partial * (vs * beta)[:, None]
                maximum = tl.maximum(maximum, tile_maximum)
            else:
                pass
        if SPLITS > 1:
            part_row = task * BLOCK_M + lane
            tl.store(PART_OUT + part_row[:, None] * D + d[None, :], acc)
            tl.store(PART_MAX + part_row, maximum)
            tl.store(PART_DENOM + part_row, denom)
        else:
            result = acc / tl.where(denom > 0, denom, 1)[:, None]
            if PACKED_NATIVE:
                output_ptr = tl.make_block_ptr(
                    Output + q_start * so + kv_head * group * D,
                    (nq, group * D),
                    (so, 1),
                    (tile * rows_per_tile, 0),
                    (BLOCK_M // group, group * D),
                    (1, 0),
                )
                tl.store(
                    output_ptr,
                    result.reshape((BLOCK_M // group, group * D)).to(OUT_DTYPE),
                    boundary_check=(0,),
                )
            else:
                tl.store(
                    Output
                    + (q_start + rows_safe[:, None]) * so
                    + q_head[:, None] * ho
                    + d[None, :],
                    result.to(OUT_DTYPE),
                    row_ok[:, None],
                )
            if write_lse:
                lse = tl.where(denom > 0, maximum * LN2 + tl.log(denom), float("inf"))
                tl.store(LSE + q_head * total_q + q_start + rows_safe, lse, row_ok)
            else:
                pass
    else:
        pass


@triton.jit
def flash_int8_merge(
    Output,
    LSE,
    PART_OUT,
    PART_MAX,
    PART_DENOM,
    CUQ,
    batch_size,
    heads,
    group: tl.constexpr,
    total_q,
    so,
    ho,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLITS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    task = tl.program_id(0)
    tile = task // (batch_size * heads)
    rem = task % (batch_size * heads)
    batch = rem // heads
    kv_head = rem % heads
    lane = tl.arange(0, BLOCK_M)
    d = tl.arange(0, D)
    q_start = tl.load(CUQ + batch).to(tl.int64)
    nq = tl.load(CUQ + batch + 1) - q_start
    q_row = tile * (BLOCK_M // group) + lane // group
    q_head = kv_head * group + lane % group
    row_ok = (lane < BLOCK_M // group * group) & (q_row < nq)
    rows_safe = tl.where(row_ok, q_row, 0).to(tl.int64)
    if tile * (BLOCK_M // group) < nq:
        maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        denom = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)
        for part in tl.static_range(0, SPLITS):
            part_row = (
                (tile * SPLITS + part) * batch_size * heads + rem
            ) * BLOCK_M + lane
            part_m = tl.load(PART_MAX + part_row)
            part_d = tl.load(PART_DENOM + part_row)
            part_a = tl.load(PART_OUT + part_row[:, None] * D + d[None, :])
            new_max = tl.maximum(maximum, part_m)
            safe_max = tl.where(new_max == float("-inf"), 0, new_max)
            alpha = tl.exp2(maximum - safe_max)
            beta = tl.exp2(part_m - safe_max)
            denom = denom * alpha + part_d * beta
            acc = acc * alpha[:, None] + part_a * beta[:, None]
            maximum = new_max
        result = acc / tl.where(denom > 0, denom, 1)[:, None]
        tl.store(
            Output
            + (q_start + rows_safe[:, None]) * so
            + q_head[:, None] * ho
            + d[None, :],
            result.to(OUT_DTYPE),
            row_ok[:, None],
        )
        if WRITE_LSE:
            lse = tl.where(denom > 0, maximum * LN2 + tl.log(denom), float("inf"))
            tl.store(LSE + q_head * total_q + q_start + rows_safe, lse, row_ok)
        else:
            pass
    else:
        pass


def check_descale(
    name: str,
    scale: Optional[torch.Tensor],
    batch: int,
    nheads: int,
    length: int,
    device: torch.device,
) -> None:
    if scale is None or scale.dtype != torch.float32 or scale.ndim != 3:
        raise TypeError(f"{name} must be a rank-3 float32 tensor")
    if scale.device != device:
        raise ValueError(f"{name} must be on the same device as Q")
    if scale.shape[0] != batch or scale.shape[1] != nheads:
        raise ValueError(f"{name} must have shape [batch, heads, blocks]")
    if length > 0:
        needed = (length + DESCALE_BLOCK.value - 1) // DESCALE_BLOCK.value
        if scale.stride(-1) != 0 and scale.shape[-1] < needed:
            raise ValueError(f"{name} covers {scale.shape[-1]} blocks, need {needed}")
    else:
        pass


@wraps(upstream_flash_attn_varlen_func, assigned=(), updated=())
def flash_attn_varlen_func_w8a8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    q_v: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[Tuple[int, int]] = None,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    scheduler_metadata: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
    fa_version: int = 2,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Ascend INT8 FA2 with native Triton QK, FP32 softmax and FP16 PV."""
    if dropout_p != 0 or return_attn_probs or q_v is not None:
        raise NotImplementedError(
            "Only attention inference without dropout is supported"
        )
    if scheduler_metadata is not None or s_aux is not None or num_splits != 0:
        raise NotImplementedError(
            "Scheduler, auxiliary and split-KV modes are unsupported"
        )
    if cp_world_size != 1 or cp_rank != 0 or cp_tot_seqused_k is not None:
        raise NotImplementedError("Context parallel attention is unsupported")
    if fa_version != 2:
        raise NotImplementedError("Only FA2 is implemented")
    from flaggems_vllm.ops.flash_api import CHECK_DEVICE

    CHECK_DEVICE(q)
    CHECK_DEVICE(k)
    CHECK_DEVICE(v)
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
        raise NotImplementedError("Q/K/V must use INT8")
    if q.ndim != 3:
        raise ValueError("Q must have shape [total_q, heads, D]")
    if q.shape[-1] not in (64, 128):
        raise NotImplementedError("Only head dimensions 64 and 128 are supported")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise NotImplementedError("The last dimension of Q/K/V must be contiguous")
    paged = block_table is not None
    if paged and (k.ndim != 4 or v.ndim != 4):
        raise ValueError("Paged K/V must have shape [pages, page_size, kv_heads, D]")
    if not paged and (k.ndim != 3 or v.ndim != 3):
        raise ValueError("Packed K/V must have shape [total_k, kv_heads, D]")
    if seqused_k is not None and (not paged):
        raise NotImplementedError("seqused_k requires paged KV")
    if not paged and cu_seqlens_k is None:
        raise ValueError("Packed KV requires cu_seqlens_k")
    if paged and (seqused_k is None or block_table.ndim != 2):
        raise ValueError("Paged KV requires seqused_k and a rank-2 block_table")
    total, heads, dim = q.shape
    kv_heads = k.shape[-2]
    if kv_heads == 0 or heads % kv_heads != 0:
        raise ValueError("Query heads must be a positive multiple of KV heads")
    batch = cu_seqlens_q.shape[0] - 1
    if batch < 1:
        raise ValueError("cu_seqlens_q must contain at least one sequence")
    group = heads // kv_heads
    page = k.shape[1] if paged else 1
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    elif out.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("out must be FP16 or BF16")
    elif out.shape != q.shape or out.device != q.device or out.stride(-1) != 1:
        raise ValueError(
            "out must match Q shape and device, with a contiguous last dim"
        )
    lse = (
        torch.empty((heads, total), dtype=torch.float32, device=q.device)
        if return_softmax_lse
        else None
    )
    if total == 0:
        return (out, lse) if return_softmax_lse else out
    check_descale("q_descale", q_descale, batch, heads, max_seqlen_q, q.device)
    check_descale("k_descale", k_descale, batch, kv_heads, max_seqlen_k, q.device)
    check_descale("v_descale", v_descale, batch, kv_heads, max_seqlen_k, q.device)
    left, right = (-1, -1) if window_size is None else window_size
    packed_native = (
        paged
        and max_seqlen_k > 0
        and (page == 16)
        and (group <= 16)
        and (group & group - 1 == 0)
        and (q.stride(-2) == dim)
        and (k.stride(-2) == dim)
        and (v.stride(-2) == dim)
        and (out.stride(-2) == dim)
    )
    is_prefill_fast = (
        packed_native
        and max_seqlen_q > 8
        and (q_descale.stride(2) == 0 or max_seqlen_q <= 128)
        and (k_descale.stride(2) == 0 or max_seqlen_k <= 128)
        and (v_descale.stride(2) == 0 or max_seqlen_k <= 128)
        and (softcap == 0)
        and (alibi_slopes is None)
        and ((left, right) == (-1, -1))
    )
    uniform_q_length = (
        max_seqlen_q if is_prefill_fast and total == batch * max_seqlen_q else 0
    )
    is_query_head_layout = (
        is_prefill_fast
        and uniform_q_length >= 512
        and (max_seqlen_k >= 512)
        and (uniform_q_length & uniform_q_length - 1 == 0)
        and (max_seqlen_k & max_seqlen_k - 1 == 0)
        and causal
        and (dim == 128)
        and (not return_softmax_lse)
    )
    use_direct_output = (
        is_query_head_layout
        and (not return_softmax_lse)
        and (dim == 128)
        and (uniform_q_length >= 512)
        and (uniform_q_length % 128 == 0)
    )
    kv_head_repeat = group if is_query_head_layout else 1
    if is_query_head_layout:
        kv_heads = heads
        group = 1
    else:
        pass
    tile_n = (
        256
        if packed_native and k_descale.stride(2) == 0 and (v_descale.stride(2) == 0)
        else 128
    )
    capacity = triton.cdiv(max_seqlen_k, tile_n) * tile_n
    use_tle = packed_native and max_seqlen_q > 0
    tle_rows = (
        128
        if is_prefill_fast
        else (
            256
            if max_seqlen_q > 8
            else max(16, 2 * group, triton.next_power_of_2(group * max_seqlen_q))
        )
    )
    query_alignment = max(16, tle_rows // group) if use_tle else 16
    query_capacity = triton.cdiv(max_seqlen_q, query_alignment) * query_alignment
    if group > BLOCK_M.value:
        raise NotImplementedError("KV group larger than the M tile is unsupported")
    block_m = min(BLOCK_M.value, triton.next_power_of_2(max(16, group * max_seqlen_q)))
    block_m = min(block_m, 32) if group <= 32 else block_m
    query_tile = block_m // group
    num_q_tiles = triton.cdiv(max_seqlen_q, query_tile)
    num_tasks = num_q_tiles * batch * kv_heads
    splits = 4 if not use_tle and max_seqlen_q == 1 and (max_seqlen_k >= 512) else 1
    block_n = 128 if packed_native else BLOCK_N.value if block_m <= 32 else 32
    max_blocks = triton.cdiv(max_seqlen_k, block_n)
    alibi_batch_stride = 0
    alibi_head_stride = 0
    alibi_ptr = q_descale
    if alibi_slopes is not None:
        alibi_ptr = alibi_slopes
        if alibi_slopes.ndim == 1:
            alibi_head_stride = alibi_slopes.stride(0)
        else:
            alibi_batch_stride = alibi_slopes.stride(0)
            alibi_head_stride = alibi_slopes.stride(1)
    else:
        pass
    lse_ptr = (
        lse if lse is not None else torch.empty(1, dtype=torch.float32, device=q.device)
    )
    out_dtype = tl.bfloat16 if out.dtype == torch.bfloat16 else tl.float16
    scale = dim ** (-0.5) if softmax_scale is None else softmax_scale
    k_token_stride = k.stride(-3)
    v_token_stride = v.stride(-3)
    k_page_stride = k.stride(0) if paged else 0
    v_page_stride = v.stride(0) if paged else 0
    part_out = (
        torch.empty(
            (num_tasks * splits * block_m, dim), dtype=torch.float32, device=q.device
        )
        if splits > 1
        else out
    )
    part_max = (
        torch.empty(num_tasks * splits * block_m, dtype=torch.float32, device=q.device)
        if splits > 1
        else out
    )
    part_denom = (
        torch.empty(num_tasks * splits * block_m, dtype=torch.float32, device=q.device)
        if splits > 1
        else out
    )
    decode_cap = triton.next_power_of_2(max(1, max_seqlen_q))
    decode_pack = 4 if decode_cap == 1 else 2 if decode_cap <= 4 else 1
    is_decode_fast = (
        packed_native
        and 0 < max_seqlen_q <= 8
        and (group == 4)
        and (kv_heads % decode_pack == 0)
        and (k_token_stride == kv_heads * dim)
        and (v_token_stride == kv_heads * dim)
        and (k_page_stride == page * kv_heads * dim)
        and (v_page_stride == page * kv_heads * dim)
    )
    with torch_device_fn.device(q.device):
        if is_decode_fast:
            decode_attention(
                q,
                k,
                v,
                out,
                lse_ptr,
                cu_seqlens_q,
                seqused_k,
                block_table,
                q_descale,
                k_descale,
                v_descale,
                max_seqlen_q,
                max_seqlen_k,
                scale,
                causal,
                left,
                right,
                softcap,
                alibi_slopes,
                return_softmax_lse,
            )
            return (out, lse) if return_softmax_lse else out
        else:
            pass
        if packed_native:
            query = (
                torch.empty(
                    (batch, kv_heads, query_capacity, group, dim),
                    dtype=torch.int8,
                    device=q.device,
                )
                if not is_query_head_layout
                else q
            )
            key_major_enabled = (
                use_direct_output
                and batch == 1
                and (group == 1)
                and (kv_heads == 32)
                and (kv_head_repeat == 4)
                and (query_capacity == 2048)
            )
            key = torch.empty(
                (
                    (batch, kv_heads // kv_head_repeat, capacity, dim)
                    if key_major_enabled
                    else (batch, capacity, kv_heads // kv_head_repeat, dim)
                ),
                dtype=torch.int8,
                device=q.device,
            )
            value = torch.empty(
                (batch, kv_heads // kv_head_repeat, capacity, dim),
                dtype=torch.float16,
                device=q.device,
            )
            has_worklist = (
                batch > 1 and batch <= 128 and (total * 2 < batch * max_seqlen_q)
            )
            query_batch = query_tile = work_counts = key_batch = key_page = cu_seqlens_q
            if has_worklist:
                work_capacity = triton.cdiv(total, tle_rows // group) + batch
                query_batch = torch.empty(
                    work_capacity, dtype=torch.int32, device=q.device
                )
                query_tile = torch.empty(
                    work_capacity, dtype=torch.int32, device=q.device
                )
                key_batch = torch.empty(
                    batch * (capacity // page), dtype=torch.int32, device=q.device
                )
                key_page = torch.empty(
                    batch * (capacity // page), dtype=torch.int32, device=q.device
                )
                work_counts = torch.empty(2, dtype=torch.int32, device=q.device)
                build_worklist[40,](
                    cu_seqlens_q,
                    seqused_k,
                    query_batch,
                    query_tile,
                    key_batch,
                    key_page,
                    work_counts,
                    batch,
                    tle_rows // group,
                    tile_n,
                    page,
                    triton.next_power_of_2(batch),
                )
            else:
                pass
            unsafe_count = (
                torch.empty(1, dtype=torch.int32, device=q.device)
                if is_prefill_fast
                else cu_seqlens_q
            )
            pack_inputs[40,](
                q,
                k,
                v,
                query,
                key,
                value,
                cu_seqlens_q,
                seqused_k,
                block_table,
                query_batch,
                query_tile,
                work_counts,
                key_batch,
                key_page,
                tle_rows // group if has_worklist else 16,
                batch,
                kv_heads,
                query_capacity,
                group,
                capacity,
                page,
                block_table.stride(0),
                q.stride(0),
                k_token_stride,
                k_page_stride,
                v_token_stride,
                v_page_stride,
                dim,
                has_worklist,
                kv_head_repeat,
                uniform_q_length,
                UnsafeCount=unsafe_count,
                INIT_UNSAFE=is_prefill_fast,
                DIRECT_QUERY=is_query_head_layout,
                KEY_MAJOR=key_major_enabled,
            )
        else:
            query, key, value = (q, k, v)
        if use_tle:
            query_tiles = triton.cdiv(max_seqlen_q, tle_rows // group)
            core_count = min(20, query_tiles * batch * kv_heads)
            score = torch.empty(
                (core_count * (2 * tle_rows * tile_n),),
                dtype=torch.int32,
                device=q.device,
            )
            probability = torch.empty(
                (core_count * (4 * tle_rows * tile_n),),
                dtype=torch.float16,
                device=q.device,
            )
            product = torch.empty(
                (
                    (core_count * tle_rows * dim,)
                    if is_prefill_fast
                    else (core_count, 2, tle_rows, dim)
                ),
                dtype=torch.float32,
                device=q.device,
            )
            task_offset, tile_shift = (
                (
                    (72, 14)
                    if query_tiles == 16
                    else (80, 8) if query_tiles == 32 else (0, 0)
                )
                if is_query_head_layout and batch == 1 and (kv_heads == 32)
                else (0, 0)
            )
            configuration = AttentionConfig(
                batch=batch,
                kv_heads=kv_heads,
                group=group,
                dim=dim,
                query_capacity=query_capacity,
                key_capacity=capacity,
                query_tiles=query_tiles,
                total_query=total,
                output_stride=out.stride(0),
                q_scale_strides=q_descale.stride(),
                k_scale_strides=k_descale.stride(),
                v_scale_strides=v_descale.stride(),
                causal=causal,
                write_lse=return_softmax_lse,
                scale=scale,
                block_m=tle_rows,
                block_n=tile_n,
                cores=core_count,
                left=left,
                right=right,
                softcap=softcap,
                has_alibi=alibi_slopes is not None,
                alibi_batch_stride=alibi_batch_stride,
                alibi_head_stride=alibi_head_stride,
                has_worklist=has_worklist,
                direct_output=use_direct_output,
                kv_head_repeat=kv_head_repeat,
                uniform_q_length=uniform_q_length,
                vector_elements=16384 if use_direct_output else 8192,
                query_stride=q.stride(0) if is_query_head_layout else 0,
                task_offset=task_offset,
                tile_shift=tile_shift,
                key_head_major=key_major_enabled,
            )
            native_output = (
                out
                if use_direct_output
                else torch.empty(
                    (batch, kv_heads, query_capacity * group, dim),
                    dtype=out.dtype,
                    device=q.device,
                )
            )
            native_lse = (
                lse_ptr
                if use_direct_output
                else (
                    torch.empty(
                        (batch, kv_heads, query_capacity * group),
                        dtype=torch.float32,
                        device=q.device,
                    )
                    if return_softmax_lse
                    else lse_ptr
                )
            )
            native_arguments = (
                query,
                key,
                value,
                native_output,
                native_lse,
                cu_seqlens_q,
                seqused_k,
                query_batch,
                query_tile,
                work_counts,
                q_descale,
                k_descale,
                v_descale,
                alibi_ptr,
                score,
                probability,
                product,
            )
            launch_options = dict(
                enable_reduce_sum_strength=True,
                multibuffer=False,
                enable_auto_bind_sub_block=False,
                disable_auto_inject_block_sync=True,
            )
            if key_major_enabled:
                launch_options["disable_auto_cv_work_space_manage"] = True
            if is_prefill_fast:
                flags = torch.empty(
                    (query_tiles * batch * kv_heads, 256),
                    dtype=torch.int32,
                    device=q.device,
                )
                prefill_attention[core_count,](
                    *native_arguments,
                    flags,
                    configuration,
                    **launch_options,
                    UnsafeCount=unsafe_count,
                )
                fallback_configuration = replace(
                    configuration, block_n=128, filter_flags=True, aggregate_flags=True
                )
                fallback_product = torch.empty(
                    (core_count, 2, tle_rows, dim), dtype=torch.float32, device=q.device
                )
                fallback_arguments = (*native_arguments[:-1], fallback_product)
                native_tle_attention[core_count,](
                    *fallback_arguments,
                    fallback_configuration,
                    Flags=flags,
                    **launch_options,
                    UnsafeCount=unsafe_count,
                )
            else:
                native_tle_attention[core_count,](
                    *native_arguments, configuration, **launch_options
                )
            if not use_direct_output:
                unpack_output[40,](
                    native_output,
                    native_lse,
                    out,
                    lse_ptr,
                    cu_seqlens_q,
                    query_batch,
                    query_tile,
                    work_counts,
                    tle_rows // group if has_worklist else 16,
                    batch,
                    kv_heads,
                    query_capacity,
                    group,
                    dim,
                    out.stride(0),
                    total,
                    return_softmax_lse,
                    HAS_WORKLIST=has_worklist,
                    UNIFORM_Q_LENGTH=uniform_q_length,
                )
            else:
                pass
            return (out, lse) if return_softmax_lse else out
        task_begin = 0
        while task_begin < num_tasks * splits:
            count = min(LAUNCH_CHUNK, num_tasks * splits - task_begin)
            flash_int8_fwd[count,](
                query,
                key,
                value,
                out,
                lse_ptr,
                part_out,
                part_max,
                part_denom,
                cu_seqlens_q,
                cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q,
                seqused_k if seqused_k is not None else cu_seqlens_q,
                block_table if block_table is not None else cu_seqlens_q,
                q_descale,
                k_descale,
                v_descale,
                alibi_ptr,
                task_begin,
                batch,
                kv_heads,
                group,
                max_blocks,
                q.stride(0),
                q.stride(1),
                k_token_stride,
                k.stride(-2),
                k_page_stride,
                v_token_stride,
                v.stride(-2),
                v_page_stride,
                out.stride(0),
                out.stride(1),
                q_descale.stride(0),
                q_descale.stride(1),
                q_descale.stride(2),
                k_descale.stride(0),
                k_descale.stride(1),
                k_descale.stride(2),
                v_descale.stride(0),
                v_descale.stride(1),
                v_descale.stride(2),
                block_table.stride(0) if paged else 0,
                alibi_batch_stride,
                alibi_head_stride,
                total,
                scale,
                causal,
                left,
                right,
                softcap,
                alibi_slopes is not None,
                return_softmax_lse,
                D=dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                PAGE=page,
                PAGED=paged,
                OUT_DTYPE=out_dtype,
                SPLITS=splits,
                PACKED_NATIVE=packed_native,
                CAPACITY=capacity,
                CAPACITY_Q=query_capacity,
            )
            task_begin += count
        if splits > 1:
            flash_int8_merge[num_tasks,](
                out,
                lse_ptr,
                part_out,
                part_max,
                part_denom,
                cu_seqlens_q,
                batch,
                kv_heads,
                group,
                total,
                out.stride(0),
                out.stride(1),
                D=dim,
                BLOCK_M=block_m,
                SPLITS=splits,
                OUT_DTYPE=out_dtype,
                WRITE_LSE=return_softmax_lse,
            )
        else:
            pass
    return (out, lse) if return_softmax_lse else out


@wraps(upstream_flash_attn_varlen_func)
def flash_attn_varlen_func(q, *args, **kwargs):
    if q.dtype == torch.int8:
        return flash_attn_varlen_func_w8a8(q, *args, **kwargs)
    return upstream_flash_attn_varlen_func(q, *args, **kwargs)
