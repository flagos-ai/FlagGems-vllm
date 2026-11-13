# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend group128 SwiGLU MoE with symmetric INT8 or INT4 weights.

INT8 accepts FP16 or BF16 activations/scales; INT4 accepts BF16.
FP16 weight decoding uses native INT8 conversion and FP16 multiplication.
GEMM accumulation and router-weighted reduction use FP32.
"""

import weakref
from functools import lru_cache
from typing import Callable, Optional

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al

GROUP_SIZE = 128
MAX_EXPERTS = 512
MAX_HIDDEN_SIZE = 7168
MAX_INTERMEDIATE_SIZE = 14336
INT32_INDEX_LIMIT = 2**31
EXACT_ROUTE_INDEX_LIMIT = 2**24


@triton.jit
def prepare_packed_kernel(
    W,
    S,
    Q,
    T,
    Safe,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    TASKS: tl.constexpr,
    BITS: tl.constexpr,
):
    PACK: tl.constexpr = 8 // BITS
    BYTES: tl.constexpr = 128 // PACK
    for pid in range(tl.program_id(0), TASKS, tl.num_programs(0)):
        pn = pid % tl.cdiv(N, BN)
        g = (pid // tl.cdiv(N, BN)) % (K // 128)
        # Packed expert weights can exceed 2 GiB; widen before address products.
        e = (pid // (tl.cdiv(N, BN) * (K // 128))).to(tl.int64)
        ns = pn * BN + tl.arange(0, BN)
        kh = tl.arange(0, BYTES)
        v = tl.load(
            W + (e * N + ns[:, None]) * (K // PACK) + g * BYTES + kh[None, :],
            ns[:, None] < N,
            other=0,
        )
        tl.store(
            Q + ((e * (K // 128) + g) * N + ns[:, None]) * BYTES + kh[None, :],
            v ^ (0x88 if BITS == 4 else 0x80),
            ns[:, None] < N,
        )
        s = tl.load(S + e * N * (K // 128) + ns * (K // 128) + g, ns < N, other=0)
        tl.store(T + (e * (K // 128) + g) * N + ns, s, ns < N)
        magnitude = tl.abs(s.to(tl.float32))
        safe = (magnitude == 0) | (
            (magnitude >= 0.00006103515625) & (magnitude <= 4096.0)
        )
        flag = tl.min(tl.where(ns < N, safe, True).to(tl.int32), 0)
        tl.store(Safe + (e * (K // 128) + g) * tl.cdiv(N, BN) + pn, flag)


weight_cache = {}


def prepare_weights(weight, scale, bits=4):
    key = (id(weight), id(scale), bits)
    version = (weight._version, scale._version, weight.data_ptr(), scale.data_ptr())
    item = weight_cache.get(key)
    if (
        item is not None
        and item[0]() is weight
        and item[1]() is scale
        and item[2] == version
    ):
        return item[3:]
    expert_count, output_features, packed_k = weight.shape
    hidden_size = packed_k * (8 // bits)
    group_count = hidden_size // GROUP_SIZE
    packed_group = GROUP_SIZE // (8 // bits)
    tile_n = 32
    packed = torch.empty(
        (expert_count, group_count, output_features, packed_group),
        device=weight.device,
        dtype=torch.uint8,
    )
    grouped_scale = torch.empty(
        (expert_count, group_count, output_features),
        device=scale.device,
        dtype=scale.dtype,
    )
    scale_is_fast = torch.empty(
        (expert_count, group_count, triton.cdiv(output_features, tile_n)),
        device=weight.device,
        dtype=torch.int32,
    )
    tasks = expert_count * group_count * triton.cdiv(output_features, tile_n)
    prepare_packed_kernel[(min(tasks, 1024),)](
        weight,
        scale,
        packed,
        grouped_scale,
        scale_is_fast,
        output_features,
        hidden_size,
        tile_n,
        tasks,
        bits,
    )

    def _drop_cached_weight(_weight_ref):
        weight_cache.pop(key, None)

    weight_cache[key] = (
        weakref.ref(weight, _drop_cached_weight),
        weakref.ref(scale, _drop_cached_weight),
        version,
        packed,
        grouped_scale,
        scale_is_fast,
    )
    return packed, grouped_scale, scale_is_fast


@triton.jit
def load_cube_weight_stage(
    Work,
    pid,
    iteration,
    BN: tl.constexpr,
    PANEL_N: tl.constexpr = 0,
    STREAM_FIRST: tl.constexpr = False,
):
    if PANEL_N:
        ni = tl.arange(0, BN)
        ki = tl.arange(0, 128)
        if STREAM_FIRST:
            al.sync_block_wait(
                "vector",
                "cube",
                2,
                sender_pipe=al.PIPE.PIPE_MTE3,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
        b = tl.load(Work + iteration * PANEL_N * 128 + ni[None, :] * 128 + ki[:, None])
        if STREAM_FIRST:
            al.sync_block_set(
                "cube",
                "vector",
                4,
                sender_pipe=al.PIPE.PIPE_MTE2,
                receiver_pipe=al.PIPE.PIPE_MTE3,
            )
    else:
        STAGES: tl.constexpr = (
            4 if Work.dtype.element_ty == tl.float16 and BN == 512 else 2
        )
        tl.static_assert((STAGES & (STAGES - 1)) == 0)
        ni = tl.arange(0, BN)
        ki = tl.arange(0, 128)
        tle.dsa.ascend.sync_block_wait(
            "vector",
            "cube",
            2,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
        )
        b = tl.load(
            Work
            + (pid * STAGES + (iteration & (STAGES - 1))) * BN * 128
            + ni[None, :] * 128
            + ki[:, None]
        )
        tle.dsa.ascend.sync_block_set(
            "cube",
            "vector",
            3,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
        )
    return b


@triton.jit
def cube_tile(
    A,
    Work,
    Output,
    row,
    col,
    pid,
    iteration,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    RowEnd=None,
):
    mi = tl.arange(0, BM)
    ni = tl.arange(0, BN)
    ki = tl.arange(0, 128)
    acc = tl.full((BM, BN), 0, tl.float32)
    for kb in range(K // 128):
        if RowEnd is None:
            a = tl.load(A + (row + mi[:, None]) * K + kb * 128 + ki[None, :])
        else:
            a = tl.load(
                A + (row + mi[:, None]) * K + kb * 128 + ki[None, :],
                (row + mi[:, None]) < RowEnd,
                other=0,
            )
        b = load_cube_weight_stage(Work, pid, iteration, BN)
        acc = tl.dot(a, b, acc)
        iteration += 1
    tl.store(
        Output + (row + mi[:, None]) * N + col + ni[None, :],
        acc.to(Output.dtype.element_ty),
    )
    return iteration


@triton.jit
def cube_tile_pair(
    A,
    Work,
    Output,
    row,
    col,
    pid,
    iteration,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    RowEnd=None,
    PANEL_N: tl.constexpr = 0,
    STREAM_FIRST: tl.constexpr = False,
):
    mi = tl.arange(0, BM)
    ni = tl.arange(0, BN)
    ki = tl.arange(0, 128)
    acc = tl.full((BM, BN), 0, tl.float32)
    RESIDENT: tl.constexpr = (
        K == 4096
        and (
            (BN == 512 and (BM == 32 or (BM == 16 and RowEnd is not None)))
            or (PANEL_N > 0 and BN == 256 and (BM == 16 or BM == 32))
        )
        and (A.dtype.element_ty == tl.float16)
    )
    if RESIDENT:
        all_k = tl.arange(0, K)
        if RowEnd is None:
            a_all = tl.load(A + (row + mi[:, None]) * K + all_k[None, :])
        else:
            a_all = tl.load(
                A + (row + mi[:, None]) * K + all_k[None, :],
                row + mi[:, None] < RowEnd,
                other=0,
            )
        zero = pid * 0
    for pair in range(K // 256):
        if RESIDENT:
            a0 = tle.dsa.extract_slice(a_all, [zero, pair * 256], [BM, 128], [1, 1])
        elif RowEnd is None:
            a0 = tl.load(A + (row + mi[:, None]) * K + pair * 256 + ki[None, :])
        else:
            a0 = tl.load(
                A + (row + mi[:, None]) * K + pair * 256 + ki[None, :],
                row + mi[:, None] < RowEnd,
                other=0,
            )
        b0 = load_cube_weight_stage(Work, pid, iteration, BN, PANEL_N, STREAM_FIRST)
        if RESIDENT:
            a1 = tle.dsa.extract_slice(
                a_all, [zero, pair * 256 + 128], [BM, 128], [1, 1]
            )
        elif RowEnd is None:
            a1 = tl.load(A + (row + mi[:, None]) * K + pair * 256 + 128 + ki[None, :])
        else:
            a1 = tl.load(
                A + (row + mi[:, None]) * K + pair * 256 + 128 + ki[None, :],
                row + mi[:, None] < RowEnd,
                other=0,
            )
        b1 = load_cube_weight_stage(Work, pid, iteration + 1, BN, PANEL_N, STREAM_FIRST)
        acc = tl.dot(a0, b0, acc)
        acc = tl.dot(a1, b1, acc)
        iteration += 2
    if K % 256:
        if RowEnd is None:
            a0 = tl.load(A + (row + mi[:, None]) * K + (K - 128) + ki[None, :])
        else:
            a0 = tl.load(
                A + (row + mi[:, None]) * K + (K - 128) + ki[None, :],
                row + mi[:, None] < RowEnd,
                other=0,
            )
        b0 = load_cube_weight_stage(Work, pid, iteration, BN, PANEL_N, STREAM_FIRST)
        acc = tl.dot(a0, b0, acc)
        iteration += 1
    tl.store(
        Output + (row + mi[:, None]) * N + col + ni[None, :],
        acc.to(Output.dtype.element_ty),
    )
    return iteration


@triton.jit
def direct_cube_tile_pair(
    A,
    Routes,
    Counts,
    Offsets,
    Work,
    Output,
    expert,
    row,
    col,
    pid,
    iteration,
    K: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    TopK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    mi = tl.arange(0, BM)
    ni = tl.arange(0, BN)
    begin = tl.load(Offsets + expert)
    count = tl.load(Counts + expert)
    a_buffer = tle.dsa.alloc((BM, K), A.dtype.element_ty, tle.dsa.ascend.L1)
    for local_row in tl.static_range(BM):
        route_row = row + local_row - begin
        if route_row < count:
            token = tl.load(Routes + expert * R + route_row) // TopK
            source = tl.make_block_ptr(
                A + token * K,
                (1, K),
                (K, 1),
                (0, 0),
                (1, K),
                (1, 0),
            )
            destination = tle.dsa.subview(a_buffer, [local_row, 0], [1, K], [1, 1])
            tle.dsa.copy(source, destination, [1, K])
    a_all = tle.dsa.to_tensor(a_buffer, writable=False)
    zero = pid * 0
    acc = tl.full((BM, BN), 0, tl.float32)
    for pair in range(K // 256):
        a0 = tle.dsa.extract_slice(a_all, [zero, pair * 256], [BM, 128], [1, 1])
        b0 = load_cube_weight_stage(Work, pid, iteration, BN)
        a1 = tle.dsa.extract_slice(a_all, [zero, pair * 256 + 128], [BM, 128], [1, 1])
        b1 = load_cube_weight_stage(Work, pid, iteration + 1, BN)
        acc = tl.dot(a0, b0, acc)
        acc = tl.dot(a1, b1, acc)
        iteration += 2
    tl.store(
        Output + (row + mi[:, None]) * N + col + ni[None, :],
        acc.to(Output.dtype.element_ty),
    )
    return iteration


@triton.jit
def direct_cube_gemm(
    A,
    Routes,
    Counts,
    Offsets,
    Work,
    Experts,
    Output,
    PID,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TASKS: tl.constexpr,
    GRID: tl.constexpr,
    R: tl.constexpr,
    TopK: tl.constexpr,
):
    iteration = 0
    for task in range(PID, TASKS, GRID):
        tile_id = task // (N // BN)
        col = task % (N // BN) * BN
        expert = tl.load(Experts + tile_id)
        if expert >= 0:
            iteration = direct_cube_tile_pair(
                A,
                Routes,
                Counts,
                Offsets,
                Work,
                Output,
                expert,
                tile_id * BM,
                col,
                PID,
                iteration,
                K,
                N,
                R,
                TopK,
                BM,
                BN,
            )


@triton.jit
def tail_task_info(
    Experts, Counts, Offsets, task, N: tl.constexpr, TASKS: tl.constexpr
):
    tile = task // (N // 128)
    panel = task % (N // 128)
    expert = tl.load(Experts + tile)
    active = expert >= 0
    paired = False
    remaining = 0
    if active:
        begin = tl.load(Offsets + expert)
        local_tile = tile - begin // 128
        active = local_tile % 2 == 0
        if active:
            remaining = tl.load(Counts + expert) - local_tile * 128
            paired = remaining > 128
    return active, paired, expert, tile, panel, remaining


@triton.jit
def load_adaptive_weight(Work, pid, iteration, BN: tl.constexpr):
    ni = tl.arange(0, BN)
    ki = tl.arange(0, 128)
    al.sync_block_wait(
        "vector",
        "cube",
        2,
        sender_pipe=al.PIPE.PIPE_MTE3,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    value = tl.load(
        Work + (pid * 2 + iteration % 2) * 512 * 128 + ni[None, :] * 128 + ki[:, None]
    )
    al.sync_block_set(
        "cube",
        "vector",
        3,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_MTE3,
    )
    return value


@triton.jit
def cube_adaptive_tile(
    A,
    Work,
    Output,
    row,
    col,
    pid,
    iteration,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    acc = tl.full((BM, BN), 0, tl.float32)
    for pair in range(K // 256):
        ap0 = tl.make_block_ptr(
            A + row * K, (BM, K), (K, 1), (0, pair * 256), (BM, 128), (1, 0)
        )
        a0 = tl.load(ap0)
        b0 = load_adaptive_weight(Work, pid, iteration, BN)
        ap1 = tl.make_block_ptr(
            A + row * K, (BM, K), (K, 1), (0, pair * 256 + 128), (BM, 128), (1, 0)
        )
        a1 = tl.load(ap1)
        b1 = load_adaptive_weight(Work, pid, iteration + 1, BN)
        acc = tl.dot(a0, b0, acc)
        acc = tl.dot(a1, b1, acc)
        iteration += 2
    cp = tl.make_block_ptr(
        Output + row * N + col, (BM, BN), (N, 1), (0, 0), (BM, BN), (1, 0)
    )
    tl.store(cp, acc.to(Output.dtype.element_ty))
    return iteration


@triton.jit
def cube_adaptive_gemm(
    A,
    Work,
    Experts,
    Output,
    Counts,
    Offsets,
    PID,
    N: tl.constexpr,
    K: tl.constexpr,
    TASKS: tl.constexpr,
    GRID: tl.constexpr,
):
    iteration = 0
    for task in range(PID, TASKS, GRID):
        active, paired, expert, tile, panel, remaining = tail_task_info(
            Experts, Counts, Offsets, task, N, TASKS
        )
        if active:
            if paired:
                iteration = cube_adaptive_tile(
                    A,
                    Work,
                    Output,
                    tile * 128,
                    panel * 128,
                    PID,
                    iteration,
                    K,
                    N,
                    256,
                    128,
                )
            else:
                if remaining <= 64:
                    if panel < N // 512:
                        iteration = cube_adaptive_tile(
                            A,
                            Work,
                            Output,
                            tile * 128,
                            panel * 512,
                            PID,
                            iteration,
                            K,
                            N,
                            64,
                            512,
                        )
                else:
                    if panel < N // 256:
                        iteration = cube_adaptive_tile(
                            A,
                            Work,
                            Output,
                            tile * 128,
                            panel * 256,
                            PID,
                            iteration,
                            K,
                            N,
                            128,
                            256,
                        )


@triton.jit
def dequantize_adaptive_task(
    Q,
    S,
    Work,
    expert,
    col,
    PID,
    SUB,
    iteration,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
):
    VBN: tl.constexpr = BN // 2
    CB: tl.constexpr = 128 if VBN >= 128 else VBN
    ns = tl.arange(0, CB)
    kk = tl.arange(0, 128)
    for kb in range(K // 128):
        base = (expert.to(tl.int64) * (K // 128) + kb) * N + col + SUB * VBN
        for chunk in range(VBN // CB):
            q = tl.load(Q + (base + chunk * CB) * 128 + tl.arange(0, CB * 128))
            scale = tl.load(S + base + chunk * CB + ns)
            value = dequantize_tile(q, scale, CB, 8)
            if iteration >= 2 and chunk == 0:
                al.sync_block_wait(
                    "cube",
                    "vector",
                    3,
                    sender_pipe=al.PIPE.PIPE_MTE2,
                    receiver_pipe=al.PIPE.PIPE_MTE3,
                )
            offset = (
                (PID * 2 + iteration % 2) * 512 * 128
                + SUB * VBN * 128
                + chunk * CB * 128
            )
            tl.store(Work + offset + ns[:, None] * 128 + kk[None, :], value)
        al.sync_block_set(
            "vector",
            "cube",
            2,
            sender_pipe=al.PIPE.PIPE_MTE3,
            receiver_pipe=al.PIPE.PIPE_MTE2,
        )
        iteration += 1
    return iteration


@triton.jit
def dequantize_adaptive_weights(
    Q,
    S,
    Experts,
    Counts,
    Offsets,
    Work,
    PID,
    SUB,
    N: tl.constexpr,
    K: tl.constexpr,
    TASKS: tl.constexpr,
    GRID: tl.constexpr,
):
    iteration = 0
    for task in range(PID, TASKS, GRID):
        active, paired, expert, tile, panel, remaining = tail_task_info(
            Experts, Counts, Offsets, task, N, TASKS
        )
        if active:
            if paired:
                iteration = dequantize_adaptive_task(
                    Q, S, Work, expert, panel * 128, PID, SUB, iteration, N, K, 128
                )
            else:
                if remaining <= 64:
                    if panel < N // 512:
                        iteration = dequantize_adaptive_task(
                            Q,
                            S,
                            Work,
                            expert,
                            panel * 512,
                            PID,
                            SUB,
                            iteration,
                            N,
                            K,
                            512,
                        )
                else:
                    if panel < N // 256:
                        iteration = dequantize_adaptive_task(
                            Q,
                            S,
                            Work,
                            expert,
                            panel * 256,
                            PID,
                            SUB,
                            iteration,
                            N,
                            K,
                            256,
                        )
    for _ in range(tl.minimum(iteration, 2)):
        al.sync_block_wait(
            "cube",
            "vector",
            3,
            sender_pipe=al.PIPE.PIPE_MTE2,
            receiver_pipe=al.PIPE.PIPE_MTE3,
        )


@triton.jit
def cube_gemm(
    A,
    Work,
    Experts,
    Output,
    PID,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TASKS: tl.constexpr,
    GRID: tl.constexpr,
    MERGE: tl.constexpr,
    TILE: tl.constexpr = cube_tile,
    Counts=None,
    Offsets=None,
):
    iteration = 0
    for task in range(PID, TASKS, GRID):
        tile_id = task // (N // BN)
        col = task % (N // BN) * BN
        expert = tl.load(Experts + tile_id)
        active = expert >= 0
        if MERGE:
            previous = tile_id - 1
            local_tile = 0
            same = active
            while (previous >= 0) & same:
                before = tl.load(Experts + previous)
                same = before == expert
                local_tile += same.to(tl.int32)
                previous -= 1
            active = active & (local_tile % 2 == 0)
        if active:
            if Counts is None:
                row_end = None
            else:
                row_end = tl.load(Offsets + expert) + tl.load(Counts + expert)
            if MERGE:
                next_expert = -2
                if tile_id + 1 < TASKS // (N // BN):
                    next_expert = tl.load(Experts + tile_id + 1)
                if next_expert == expert:
                    iteration = TILE(
                        A,
                        Work,
                        Output,
                        tile_id * 128,
                        col,
                        PID,
                        iteration,
                        K,
                        N,
                        256,
                        BN,
                        row_end,
                    )
                else:
                    iteration = TILE(
                        A,
                        Work,
                        Output,
                        tile_id * 128,
                        col,
                        PID,
                        iteration,
                        K,
                        N,
                        128,
                        BN,
                        row_end,
                    )
            else:
                iteration = TILE(
                    A,
                    Work,
                    Output,
                    tile_id * BM,
                    col,
                    PID,
                    iteration,
                    K,
                    N,
                    BM,
                    BN,
                    row_end,
                )


@triton.jit
def dequantize_tile(q, scales, RB: tl.constexpr, BITS: tl.constexpr):
    if BITS == 4:
        h = tl.full((RB * 128,), 0, tl.float16)
        h = al.custom("cast_int4_to_fp16", q, 0, RB * 128, out=h)
    else:
        # Preparation converts offset-binary UINT8 codes to signed INT8 bits.
        h = q.to(tl.int8, bitcast=True).to(tl.float16)
    values_h = tl.reshape(h, (RB, 128))
    if scales.dtype == tl.float16:
        result = values_h * tl.reshape(scales, (RB, 1))
    else:
        result = (values_h.to(tl.float32) * scales.to(tl.float32)[:, None]).to(
            tl.bfloat16
        )
    return result


@triton.jit
def copy_quant_matrix(
    Source, destination, offset, ROWS: tl.constexpr, COLS: tl.constexpr = 128
):
    source = tl.make_block_ptr(
        base=Source + offset,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(0, 0),
        block_shape=(ROWS, COLS),
        order=(1, 0),
    )
    tle.dsa.copy(source, destination, [ROWS, COLS])


@triton.jit
def store_pipeline_chunk(
    Work,
    value,
    PID,
    SUB,
    iteration,
    chunk,
    BN: tl.constexpr,
    CB: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    STAGES: tl.constexpr = 4 if Work.dtype.element_ty == tl.float16 and BN == 512 else 2
    tl.static_assert((STAGES & (STAGES - 1)) == 0)
    ns = tl.arange(0, CB)
    kk = tl.arange(0, 128)
    if iteration >= STAGES and chunk == 0:
        al.sync_block_wait(
            "cube",
            "vector",
            3,
            sender_pipe=al.PIPE.PIPE_MTE2,
            receiver_pipe=al.PIPE.PIPE_MTE3,
        )
    offset = (
        (PID * STAGES + (iteration & (STAGES - 1))) * BN * 128
        + SUB * (BN // 2) * 128
        + chunk * CB * 128
    )
    tl.store(Work + offset + ns[:, None] * 128 + kk[None, :], value)
    if chunk == CHUNKS - 1:
        al.sync_block_set(
            "vector",
            "cube",
            2,
            sender_pipe=al.PIPE.PIPE_MTE3,
            receiver_pipe=al.PIPE.PIPE_MTE2,
        )
        iteration += 1
    return iteration


@triton.jit
def dequantize_int8_pipeline(
    Q,
    S,
    Experts,
    Work,
    PID,
    SUB,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    TASKS: tl.constexpr,
    GRID: tl.constexpr,
    MERGE: tl.constexpr,
):
    STAGES: tl.constexpr = 4 if Work.dtype.element_ty == tl.float16 and BN == 512 else 2
    tl.static_assert((STAGES & (STAGES - 1)) == 0)
    VBN: tl.constexpr = BN // 2
    CB: tl.constexpr = VBN if VBN < 128 else 128
    CHUNKS: tl.constexpr = VBN // CB
    TOTAL: tl.constexpr = K // 128 * CHUNKS
    q0 = tle.dsa.alloc((CB, 128), tl.uint8, tle.dsa.ascend.UB)
    q1 = tle.dsa.alloc((CB, 128), tl.uint8, tle.dsa.ascend.UB)
    s0 = tle.dsa.alloc((CB, 1), Work.dtype.element_ty, tle.dsa.ascend.UB)
    s1 = tle.dsa.alloc((CB, 1), Work.dtype.element_ty, tle.dsa.ascend.UB)
    iteration = 0
    for task in range(PID, TASKS, GRID):
        tile = task // (N // BN)
        pn = task % (N // BN) * 2 + SUB
        expert = tl.load(Experts + tile)
        active = expert >= 0
        if MERGE:
            previous = tile - 1
            local_tile = 0
            same = active
            while (previous >= 0) & same:
                before = tl.load(Experts + previous)
                same = before == expert
                local_tile += same.to(tl.int32)
                previous -= 1
            active = active & (local_tile % 2 == 0)
        if active:
            start = expert.to(tl.int64) * (K // 128) * N + pn * VBN
            copy_quant_matrix(Q, q0, start * 128, CB)
            copy_quant_matrix(S, s0, start, CB, 1)
            for pair in range(tl.cdiv(TOTAL, 2)):
                j = pair * 2
                if j + 1 < TOTAL:
                    next_row = (
                        (expert.to(tl.int64) * (K // 128) + (j + 1) // CHUNKS) * N
                        + pn * VBN
                        + (j + 1) % CHUNKS * CB
                    )
                    copy_quant_matrix(Q, q1, next_row * 128, CB)
                    copy_quant_matrix(S, s1, next_row, CB, 1)
                h0 = dequantize_tile(
                    tle.dsa.to_tensor(q0, writable=False),
                    tle.dsa.to_tensor(s0, writable=False),
                    CB,
                    8,
                )
                if Work.dtype.element_ty == tl.float16 and CHUNKS == 2:
                    split_ns = tl.arange(0, CB)
                    split_ks = tl.arange(0, 128)
                    if iteration >= STAGES:
                        al.sync_block_wait(
                            "cube",
                            "vector",
                            3,
                            sender_pipe=al.PIPE.PIPE_MTE2,
                            receiver_pipe=al.PIPE.PIPE_MTE3,
                        )
                    split_offset = (
                        PID * STAGES + (iteration & (STAGES - 1))
                    ) * BN * 128 + SUB * VBN * 128
                    tl.store(
                        Work
                        + split_offset
                        + split_ns[:, None] * 128
                        + split_ks[None, :],
                        h0,
                        split_ns[:, None] < 112,
                    )
                else:
                    iteration = store_pipeline_chunk(
                        Work, h0, PID, SUB, iteration, j % CHUNKS, BN, CB, CHUNKS
                    )
                if j + 2 < TOTAL:
                    next_row = (
                        (expert.to(tl.int64) * (K // 128) + (j + 2) // CHUNKS) * N
                        + pn * VBN
                        + (j + 2) % CHUNKS * CB
                    )
                    copy_quant_matrix(Q, q0, next_row * 128, CB)
                    copy_quant_matrix(S, s0, next_row, CB, 1)
                if j + 1 < TOTAL:
                    h1 = dequantize_tile(
                        tle.dsa.to_tensor(q1, writable=False),
                        tle.dsa.to_tensor(s1, writable=False),
                        CB,
                        8,
                    )
                    if Work.dtype.element_ty == tl.float16 and CHUNKS == 2:
                        tl.store(
                            Work
                            + split_offset
                            + split_ns[:, None] * 128
                            + split_ks[None, :],
                            h0,
                            split_ns[:, None] >= 112,
                        )
                    iteration = store_pipeline_chunk(
                        Work, h1, PID, SUB, iteration, (j + 1) % CHUNKS, BN, CB, CHUNKS
                    )
    for _ in range(tl.minimum(iteration, STAGES)):
        al.sync_block_wait(
            "cube",
            "vector",
            3,
            sender_pipe=al.PIPE.PIPE_MTE2,
            receiver_pipe=al.PIPE.PIPE_MTE3,
        )


@triton.jit
def dequantize_weights(
    Q,
    S,
    Safe,
    Experts,
    Work,
    PID,
    SUB,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    TASKS: tl.constexpr,
    GRID: tl.constexpr,
    MERGE: tl.constexpr,
    BITS: tl.constexpr,
):
    STAGES: tl.constexpr = 4 if Work.dtype.element_ty == tl.float16 and BN == 512 else 2
    tl.static_assert((STAGES & (STAGES - 1)) == 0)
    BYTES: tl.constexpr = 128 // (8 // BITS)
    VBN: tl.constexpr = BN // 2
    # Keep INT4's existing UB bound; INT8's native cast permits the full Vector tile.
    CB: tl.constexpr = (
        VBN
        if BITS == 8 and Work.dtype.element_ty == tl.float16
        else 64 if VBN > 64 else VBN
    )
    ns = tl.arange(0, CB)
    ks = tl.arange(0, 128)
    iteration = 0
    for task in range(PID, TASKS, GRID):
        tile = task // (N // BN)
        pn = task % (N // BN) * 2 + SUB
        expert = tl.load(Experts + tile)
        active = expert >= 0
        if MERGE:
            previous = tile - 1
            local_tile = 0
            same = active
            while (previous >= 0) & same:
                before = tl.load(Experts + previous)
                same = before == expert
                local_tile += same.to(tl.int32)
                previous -= 1
            active = active & (local_tile % 2 == 0)
        if active:
            for kb in range(K // 128):
                base = (expert.to(tl.int64) * (K // 128) + kb) * N + pn * VBN
                for chunk in range(VBN // CB):
                    packed = tl.load(
                        Q + (base + chunk * CB) * BYTES + tl.arange(0, CB * BYTES)
                    )
                    scale = tl.load(S + base + chunk * CB + ns)
                    result = dequantize_tile(packed, scale, CB, BITS)
                    # Wait once before overwriting either half of the GM stage.
                    if iteration >= STAGES and chunk == 0:
                        tle.dsa.ascend.sync_block_wait(
                            "cube",
                            "vector",
                            3,
                            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
                            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
                        )
                    offset = (
                        (PID * STAGES + (iteration & (STAGES - 1))) * BN * 128
                        + SUB * VBN * 128
                        + chunk * CB * 128
                    )
                    tl.store(Work + offset + ns[:, None] * 128 + ks[None, :], result)
                # Notify Cube only after all subtiles have been stored.
                tle.dsa.ascend.sync_block_set(
                    "vector",
                    "cube",
                    2,
                    sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
                    receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
                )
                iteration += 1
    for _ in range(tl.minimum(iteration, STAGES)):
        tle.dsa.ascend.sync_block_wait(
            "cube",
            "vector",
            3,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
        )


@triton.jit
def route_kernel(
    IDs,
    Routes,
    Counts,
    R: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
):
    lane = tl.arange(0, B)
    source = lane.to(tl.float32)
    for expert in range(tl.program_id(0), E, tl.num_programs(0)):
        count = 0
        for start in range(0, R, B):
            ids = tl.load(IDs + start + lane, start + lane < R, other=-1)
            mask = tl.full((B // 32,), 0, tl.uint32)
            mask = al.custom(
                "compare_scalar",
                ids.to(tl.float32),
                expert.to(tl.float32),
                2,
                B,
                out=mask,
            )
            output = tl.full((B,), 0, tl.float32)
            number = tl.full((1,), 0, tl.int64)
            output, number = al.custom(
                "gather_mask_custom_pattern",
                source,
                mask,
                True,
                B,
                1,
                1,
                8,
                1,
                out=[output, number],
            )
            found = tl.max(number, 0).to(tl.int32)
            if found > 0:
                indices = output.to(tl.int32) + start
                tl.store(Routes + expert * R + count + lane, indices, lane < found)
                count += found
        tl.store(Counts + expert, count)


def route_experts(ids, output, counts):
    cores = triton.runtime.driver.active.utils.get_device_properties(ids.device.index)[
        "num_vectorcore"
    ]
    grid = min(counts.numel(), cores)
    block = 4096
    route_kernel[(grid,)](
        ids,
        output,
        counts,
        ids.numel(),
        counts.numel(),
        block,
        disable_auto_inject_block_sync=True,
        num_warps=1,
    )


@triton.jit
def gemm_kernel(
    A,
    Q,
    S,
    Safe,
    Experts,
    Work,
    Output,
    BM: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    TASKS: tl.constexpr,
    GRID: tl.constexpr,
    MERGE: tl.constexpr,
    BITS: tl.constexpr,
    Counts=None,
    Offsets=None,
    TAIL_TILES: tl.constexpr = False,
    Routes=None,
    R: tl.constexpr = 1,
    TopK: tl.constexpr = 1,
):
    with tle.scope(core_mode="cube"):
        if TAIL_TILES:
            cube_adaptive_gemm(
                A,
                Work,
                Experts,
                Output,
                Counts,
                Offsets,
                tl.program_id(0),
                N,
                K,
                TASKS,
                GRID,
            )
        elif Routes is not None:
            direct_cube_gemm(
                A,
                Routes,
                Counts,
                Offsets,
                Work,
                Experts,
                Output,
                tl.program_id(0),
                N,
                K,
                BM,
                BN,
                TASKS,
                GRID,
                R,
                TopK,
            )
        elif BITS == 8:
            cube_gemm(
                A,
                Work,
                Experts,
                Output,
                tl.program_id(0),
                N,
                K,
                BM,
                BN,
                TASKS,
                GRID,
                MERGE,
                cube_tile_pair,
                Counts,
                Offsets,
            )
        else:
            cube_gemm(
                A,
                Work,
                Experts,
                Output,
                tl.program_id(0),
                N,
                K,
                BM,
                BN,
                TASKS,
                GRID,
                MERGE,
            )
    with tle.scope(core_mode="vector"):
        if TAIL_TILES:
            dequantize_adaptive_weights(
                Q,
                S,
                Experts,
                Counts,
                Offsets,
                Work,
                tl.program_id(0),
                tle.dsa.ascend.sub_vec_id(),
                N,
                K,
                TASKS,
                GRID,
            )
        elif (
            BITS == 8 and BM <= 64 and Work.dtype.element_ty == tl.float16 and BN == 512
        ):
            dequantize_int8_pipeline(
                Q,
                S,
                Experts,
                Work,
                tl.program_id(0),
                tle.dsa.ascend.sub_vec_id(),
                N,
                K,
                BN,
                TASKS,
                GRID,
                MERGE,
            )
        else:
            dequantize_weights(
                Q,
                S,
                Safe,
                Experts,
                Work,
                tl.program_id(0),
                tle.dsa.ascend.sub_vec_id(),
                N,
                K,
                BN,
                TASKS,
                GRID,
                MERGE,
                BITS,
            )


@triton.jit
def expert_panel_kernel(
    A,
    Q,
    S,
    Counts,
    Offsets,
    Work,
    Output,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GRID: tl.constexpr,
):
    STAGES: tl.constexpr = 1
    pid = tl.program_id(0)
    with tle.scope(core_mode="cube"):
        cube_iteration = 0
        for cube_task in range(pid, E * (N // BN), GRID):
            cube_expert = cube_task // (N // BN)
            cube_col = cube_task % (N // BN) * BN
            cube_count = tl.load(Counts + cube_expert)
            if cube_count > 0:
                cube_begin = tl.load(Offsets + cube_expert)
                cube_stage = (pid * STAGES + cube_iteration % STAGES) * BN * K
                cube_tile_pair(
                    A,
                    Work + cube_stage,
                    Output,
                    cube_begin,
                    cube_col,
                    pid,
                    pid * 0,
                    K,
                    N,
                    BM,
                    BN,
                    None,
                    BN,
                    True,
                )
                for cube_start in range(BM, cube_count, BM):
                    remaining = cube_count - cube_start
                    if remaining <= 16:
                        cube_tile_pair(
                            A,
                            Work + cube_stage,
                            Output,
                            cube_begin + cube_start,
                            cube_col,
                            pid,
                            pid * 0,
                            K,
                            N,
                            16,
                            BN,
                            None,
                            BN,
                            False,
                        )
                    elif remaining <= 32:
                        cube_tile_pair(
                            A,
                            Work + cube_stage,
                            Output,
                            cube_begin + cube_start,
                            cube_col,
                            pid,
                            pid * 0,
                            K,
                            N,
                            32,
                            BN,
                            None,
                            BN,
                            False,
                        )
                    elif remaining <= 64:
                        cube_tile_pair(
                            A,
                            Work + cube_stage,
                            Output,
                            cube_begin + cube_start,
                            cube_col,
                            pid,
                            pid * 0,
                            K,
                            N,
                            64,
                            BN,
                            None,
                            BN,
                            False,
                        )
                    else:
                        cube_tile_pair(
                            A,
                            Work + cube_stage,
                            Output,
                            cube_begin + cube_start,
                            cube_col,
                            pid,
                            pid * 0,
                            K,
                            N,
                            BM,
                            BN,
                            None,
                            BN,
                            False,
                        )
                al.sync_block_set(
                    "cube",
                    "vector",
                    3,
                    sender_pipe=al.PIPE.PIPE_MTE2,
                    receiver_pipe=al.PIPE.PIPE_MTE3,
                )
                cube_iteration += 1
    with tle.scope(core_mode="vector"):
        sub = al.sub_vec_id()
        VBN: tl.constexpr = BN // 2
        ns = tl.arange(0, VBN)
        ks = tl.arange(0, 128)
        vector_iteration = 0
        for vector_task in range(pid, E * (N // BN), GRID):
            vector_expert = vector_task // (N // BN)
            vector_col = vector_task % (N // BN) * BN + sub * VBN
            vector_count = tl.load(Counts + vector_expert)
            if vector_count > 0:
                if vector_iteration >= STAGES:
                    al.sync_block_wait(
                        "cube",
                        "vector",
                        3,
                        sender_pipe=al.PIPE.PIPE_MTE2,
                        receiver_pipe=al.PIPE.PIPE_MTE3,
                    )
                vector_stage = (pid * STAGES + vector_iteration % STAGES) * BN * K
                for kb in range(K // 128):
                    base = (
                        vector_expert.to(tl.int64) * (K // 128) + kb
                    ) * N + vector_col
                    codes = tl.load(Q + base * 128 + tl.arange(0, VBN * 128))
                    scales = tl.load(S + base + ns)
                    decoded = dequantize_tile(codes, scales, VBN, 8)
                    if kb >= 2:
                        al.sync_block_wait(
                            "cube",
                            "vector",
                            4,
                            sender_pipe=al.PIPE.PIPE_MTE2,
                            receiver_pipe=al.PIPE.PIPE_MTE3,
                        )
                    tl.store(
                        Work
                        + vector_stage
                        + kb * BN * 128
                        + (sub * VBN + ns[:, None]) * 128
                        + ks[None, :],
                        decoded,
                    )
                    al.sync_block_set(
                        "vector",
                        "cube",
                        2,
                        sender_pipe=al.PIPE.PIPE_MTE3,
                        receiver_pipe=al.PIPE.PIPE_MTE2,
                    )
                for group_drain in range(tl.minimum(K // 128, 2)):
                    al.sync_block_wait(
                        "cube",
                        "vector",
                        4,
                        sender_pipe=al.PIPE.PIPE_MTE2,
                        receiver_pipe=al.PIPE.PIPE_MTE3,
                    )
                vector_iteration += 1
        for panel_drain in range(tl.minimum(vector_iteration, STAGES)):
            al.sync_block_wait(
                "cube",
                "vector",
                3,
                sender_pipe=al.PIPE.PIPE_MTE2,
                receiver_pipe=al.PIPE.PIPE_MTE3,
            )


def grouped_quantized_gemm(
    a,
    w,
    s,
    experts,
    out,
    bm,
    bits=4,
    counts=None,
    offsets=None,
    tail_tiles=False,
    expert_panel=False,
    routes=None,
    route_count=0,
    topk=1,
):
    n, k = out.shape[1], a.shape[1]
    cores = triton.runtime.driver.active.utils.get_device_properties(a.device.index)[
        "num_aicore"
    ]
    if (
        expert_panel
        and bits == 8
        and a.dtype == torch.float16
        and bm == 128
        and k == 4096
        and n == 512
        and w.shape[0] >= cores
    ):
        q, scale, safe = prepare_weights(w, s, bits)
        panel_bn = 256
        panel_grid = min(w.shape[0] * (n // panel_bn), cores)
        panel_work = torch.empty(
            panel_grid * 1 * panel_bn * k, device=a.device, dtype=a.dtype
        )
        expert_panel_kernel[(panel_grid,)](
            a,
            q,
            scale,
            counts,
            offsets,
            panel_work,
            out,
            k,
            n,
            w.shape[0],
            bm,
            panel_bn,
            panel_grid,
            disable_auto_inject_block_sync=True,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
        return
    # Limit BN to fit two decoded BF16 weight tiles and both A tiles in L1.
    bn = min(n & -n, 512 if bits == 8 and bm <= 64 else 256, 32768 // bm)
    large_fp16 = bits == 8 and a.dtype == torch.float16 and bm == 128
    merge = (
        bm == 128
        and ((k == 256 and n == 4096) or (k == 4096 and n == 512))
        and not large_fp16
    )
    tail_tiles = tail_tiles and not large_fp16
    counts = None if large_fp16 else counts
    offsets = None if large_fp16 else offsets
    assert not tail_tiles or (
        bits == 8 and merge and counts is not None and offsets is not None
    )
    input_bm = bm
    if merge:
        bm, bn = 256, 128
    tasks = out.shape[0] // input_bm * (n // bn)
    grid = min(tasks, cores)
    if merge and k == 4096:
        grid = min(grid, 19)
    q, scale, safe = prepare_weights(w, s, bits)
    stages = 4 if a.dtype == torch.float16 and bn == 512 else 2
    work = torch.empty(
        (grid * stages * (512 if tail_tiles else bn) * 128,),
        device=a.device,
        dtype=a.dtype,
    )
    gemm_kernel[(grid,)](
        a,
        q,
        scale,
        safe,
        experts,
        work,
        out,
        bm,
        n,
        k,
        bn,
        tasks,
        grid,
        merge,
        bits,
        counts,
        offsets,
        tail_tiles,
        routes,
        route_count if routes is not None else 1,
        topk,
        disable_auto_inject_block_sync=True,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )


@triton.jit
def small_fused_kernel(
    X,
    Q1,
    S1,
    F1,
    Q2,
    S2,
    F2,
    IDs,
    P,
    EP,
    AX,
    H,
    Act,
    Z,
    W,
    O,
    K: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    PK: tl.constexpr,
    BN2: tl.constexpr,
    BITS: tl.constexpr,
):
    SM: tl.constexpr = 1 if BITS == 8 else 16
    # INT8 uses a full Vector column tile; INT4 keeps its existing UB bound.
    BN1: tl.constexpr = (
        512
        if BITS == 8
        and X.dtype.element_ty == tl.float16
        and K == 4096
        and N == 256
        and M * T > G // 2
        else 256 if BITS == 8 or K > 4096 else 128
    )
    TILE: tl.constexpr = (
        cube_tile_pair
        if BITS == 8 and X.dtype.element_ty == tl.float16 and K == 4096 and N == 256
        else cube_tile
    )
    ep = EP
    ax = AX
    h = H
    a = Act
    z = Z
    work = W
    pid = tl.program_id(0)
    # Every physical pair enters each barrier, including pairs with no routes.
    # Global barrier 10 is separate from the GEMM ring flags 2 and 3.
    with al.scope(core_mode="cube"):
        al.sync_block_all("all", 10)
        cube_gemm(
            ax,
            work,
            ep,
            h,
            pid,
            2 * N,
            K,
            SM,
            BN1,
            M * T * (2 * N // BN1),
            G,
            False,
            TILE,
        )
        al.sync_block_all("all", 10)
        al.sync_block_all("all", 10)
        cube_gemm(
            a, work, ep, z, pid, K, N, SM, BN2, M * T * (K // BN2), G, False, TILE
        )
        al.sync_block_all("all", 10)
        al.sync_block_all("all", 10)
    with al.scope(core_mode="vector"):
        sub = al.sub_vec_id()
        vp = pid * 2 + sub
        kk = tl.arange(0, PK)
        for route in range(vp, M * T, G * 2):
            expert = tl.load(IDs + route)
            tl.store(ep + route, expert)
            x = tl.load(X + (route // T) * K + kk, kk < K, other=0)
            tl.store(ax + route * SM * K + kk, x, kk < K)
            for row in range(1, SM):
                tl.store(ax + (route * SM + row) * K + kk, 0, kk < K)
        al.sync_block_all("all", 10)
        dequantize_weights(
            Q1,
            S1,
            F1,
            ep,
            work,
            pid,
            sub,
            2 * N,
            K,
            BN1,
            M * T * (2 * N // BN1),
            G,
            False,
            BITS,
        )
        al.sync_block_all("all", 10)
        for act_block in range(vp, M * T * SM * tl.cdiv(N, 256), G * 2):
            act_row = act_block // tl.cdiv(N, 256)
            act_col = act_block % tl.cdiv(N, 256) * 256 + tl.arange(0, 256)
            av = tl.load(h + act_row * N * 2 + act_col, act_col < N, other=0).to(
                tl.float32
            )
            bv = tl.load(h + act_row * N * 2 + act_col + N, act_col < N, other=0).to(
                tl.float32
            )
            value = av / (1 + tl.exp(-av)) * bv
            tl.store(a + act_row * N + act_col, value, act_col < N)
        al.sync_block_all("all", 10)
        dequantize_weights(
            Q2,
            S2,
            F2,
            ep,
            work,
            pid,
            sub,
            K,
            N,
            BN2,
            M * T * (K // BN2),
            G,
            False,
            BITS,
        )
        al.sync_block_all("all", 10)
        for combine_block in range(vp, M * tl.cdiv(K, 256), G * 2):
            combine_row = combine_block // tl.cdiv(K, 256)
            combine_col = combine_block % tl.cdiv(K, 256) * 256 + tl.arange(0, 256)
            total = tl.full((256,), 0, tl.float32)
            for j in range(T):
                restore_route = combine_row * T + j
                zv = tl.load(
                    z + restore_route * SM * K + combine_col, combine_col < K, other=0
                ).to(tl.float32)
                prob = tl.load(P + restore_route)
                total = total + zv * prob
            tl.store(O + combine_row * K + combine_col, total, combine_col < K)
        al.sync_block_all("all", 10)


@lru_cache(maxsize=128)
def small_route_workspace_sizes(
    m, k, n, t, g, bits=4, wide_first=False, wide_second=False
):
    r = m * t
    sm = 1 if bits == 8 else 16
    bn1 = 512 if wide_first else 256 if bits == 8 or k > 4096 else GROUP_SIZE
    bn2 = min(k & -k, 512 if wide_second else 256)
    sizes = (
        r * sm * k,
        r * sm * 2 * n,
        r * sm * n,
        r * sm * k,
        g * (4 if wide_first or wide_second else 2) * max(bn1, bn2) * GROUP_SIZE,
    )
    return sizes


def small_route_moe(x, w1, w2, s1, s2, p, ids, bits=4):
    m, k = x.shape
    n = w1.shape[1] // 2
    t = ids.shape[1]
    g = triton.runtime.driver.active.utils.get_device_properties(x.device.index)[
        "num_aicore"
    ]
    wide_first = (
        bits == 8
        and x.dtype == torch.float16
        and k == 4096
        and n == 256
        and m * t > g // 2
    )
    wide_second = bits == 8 and x.dtype == torch.float16 and k == 4096 and n == 256
    sizes = small_route_workspace_sizes(m, k, n, t, g, bits, wide_first, wide_second)
    q1, s1, f1 = prepare_weights(w1, s1, bits)
    q2, s2, f2 = prepare_weights(w2, s2, bits)
    meta = torch.empty(m * t, device=x.device, dtype=torch.int32)
    # Direct typed allocations avoid unsupported pointer casts and view-dispatch overhead.
    buffers = [torch.empty(size, device=x.device, dtype=x.dtype) for size in sizes]
    out = torch.empty_like(x)
    small_fused_kernel[(g,)](
        x,
        q1,
        s1,
        f1,
        q2,
        s2,
        f2,
        ids,
        p,
        meta,
        *buffers,
        out,
        k,
        n,
        m,
        t,
        g,
        triton.next_power_of_2(k),
        min(k & -k, 512 if wide_second else 256),
        bits,
        disable_auto_inject_block_sync=True,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    return out


@triton.jit
def silu_kernel(
    H,
    Offsets,
    O,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    E: tl.constexpr,
    RB: tl.constexpr,
    NC: tl.constexpr,
    TileExperts=None,
    Counts=None,
    BM: tl.constexpr = 128,
):
    active = TOTAL // N
    if E >= 0:
        active = tl.load(Offsets + E)
    for block in range(
        tl.program_id(0), tl.cdiv(active, RB) * tl.cdiv(N, NC), tl.num_programs(0)
    ):
        row = (block // tl.cdiv(N, NC)) * RB
        col = (block % tl.cdiv(N, NC)) * NC
        input_rows = active
        if TileExperts is not None:
            expert = tl.load(TileExperts + row // BM)
            input_rows = tl.load(Offsets + expert) + tl.load(Counts + expert)
        pa = tl.make_block_ptr(
            H,
            shape=(input_rows, 2 * N),
            strides=(2 * N, 1),
            offsets=(row, col),
            block_shape=(RB, NC),
            order=(1, 0),
        )
        pb = tl.make_block_ptr(
            H,
            shape=(input_rows, 2 * N),
            strides=(2 * N, 1),
            offsets=(row, col + N),
            block_shape=(RB, NC),
            order=(1, 0),
        )
        po = tl.make_block_ptr(
            O,
            shape=(active, N),
            strides=(N, 1),
            offsets=(row, col),
            block_shape=(RB, NC),
            order=(1, 0),
        )
        a = tl.load(pa, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b = tl.load(pb, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        out = a / (1 + tl.exp(-a)) * b
        tl.store(po, out.to(O.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def combine_kernel(
    A,
    P,
    Inv,
    O,
    K: tl.constexpr,
    T: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    HAS_INV: tl.constexpr,
):
    UNROLL: tl.constexpr = T if A.dtype.element_ty == tl.float16 and T <= 8 else 1
    for block in range(tl.program_id(0), M * tl.cdiv(K, B), tl.num_programs(0)):
        row = block // tl.cdiv(K, B)
        cols = block % tl.cdiv(K, B) * B + tl.arange(0, B)
        acc = tl.full((B,), 0, tl.float32)
        for t in tl.range(T, loop_unroll_factor=UNROLL):
            route = row * T + t
            pos = route
            if HAS_INV:
                pos = tl.load(Inv + route)
            prob = tl.load(P + route)
            a = tl.load(A + pos * K + cols, cols < K, other=0).to(tl.float32)
            acc += a * prob
        tl.store(O + row * K + cols, acc, cols < K)


def apply_swiglu(h, out, off=None, b=32, tile_experts=None, counts=None, bm=128):
    silu_kernel[(vector_core_count(h),)](
        h,
        off if off is not None else out,
        out,
        out.shape[1],
        out.numel(),
        off.numel() - 1 if off is not None else -1,
        b,
        min(out.shape[1], 256),
        tile_experts,
        counts,
        bm,
        num_warps=1,
        multibuffer=False,
        enable_fp_fusion=False,
    )


def combine_expert_outputs(a, p, out, inv=None, b=4096):
    combine_kernel[(vector_core_count(a),)](
        a,
        p,
        inv if inv is not None else p,
        out,
        out.shape[1],
        p.shape[1],
        out.shape[0],
        min(b, triton.next_power_of_2(out.shape[1])),
        inv is not None,
        num_warps=1,
        multibuffer=True,
        enable_fp_fusion=False,
    )


@triton.jit
def pack_kernel(
    X,
    R,
    C,
    Offsets,
    Experts,
    O,
    K: tl.constexpr,
    T: tl.constexpr,
    RR: tl.constexpr,
    BM: tl.constexpr,
    TILES: tl.constexpr,
    BR: tl.constexpr,
    PK: tl.constexpr,
):
    for task in range(tl.program_id(0), TILES * (BM // BR), tl.num_programs(0)):
        tile = task // (BM // BR)
        ex = tl.load(Experts + tile)
        if ex >= 0:
            begin = tl.load(Offsets + ex)
            count = tl.load(C + ex)
            row = task * BR + tl.arange(0, BR)
            local = row - begin
            route = tl.load(R + ex * RR + local, local < count, other=0)
            kk = tl.arange(0, PK)
            x = tl.load(
                X + (route // T)[:, None] * K + kk[None, :],
                (local < count)[:, None] & (kk < K)[None, :],
                other=0,
            )
            if PK == K:
                # The loaded tensor already zeroes invalid routes.
                # All destinations fit a complete padded M tile.
                buf = tle.dsa.to_buffer(x, space=tle.dsa.ascend.UB)
                with tle.dsa.hint(inter_no_alias=True):
                    tle.dsa.copy(buf, O + row[:, None] * K + kk[None, :], [BR, K])
            else:
                tl.store(O + row[:, None] * K + kk[None, :], x, (kk < K)[None, :])


def vector_core_count(a):
    return triton.runtime.driver.active.utils.get_device_properties(a.device.index)[
        "num_vectorcore"
    ]


def pack_hidden_states(x, r, c, off, e, out, bm, t, br=4):
    if r.shape[1] <= c.numel() * bm // 2:
        # Sparse routing produces mostly padding: skip input loads for zero rows.
        pack_rows_kernel[(min(e.numel(), vector_core_count(x)),)](
            x,
            r,
            c,
            off,
            e,
            out,
            x.shape[1],
            t,
            r.shape[1],
            bm,
            e.numel(),
            triton.next_power_of_2(x.shape[1]),
            num_warps=1,
            multibuffer=False,
        )
        return
    # Padded 8192-column dense loads need room for masks and temporary buffers.
    if x.shape[1] > 4096:
        br = min(br, 2)
    pack_kernel[(vector_core_count(x),)](
        x,
        r,
        c,
        off,
        e,
        out,
        x.shape[1],
        t,
        r.shape[1],
        bm,
        e.numel(),
        br,
        triton.next_power_of_2(x.shape[1]),
        num_warps=1,
        multibuffer=False,
    )


@triton.jit
def pack_rows_kernel(
    X,
    R,
    C,
    Offsets,
    Experts,
    O,
    K: tl.constexpr,
    T: tl.constexpr,
    RR: tl.constexpr,
    BM: tl.constexpr,
    TILES: tl.constexpr,
    PK: tl.constexpr,
):
    kk = tl.arange(0, PK)
    for tile in range(tl.program_id(0), TILES, tl.num_programs(0)):
        ex = tl.load(Experts + tile)
        if ex >= 0:
            begin = tl.load(Offsets + ex)
            count = tl.load(C + ex)
            for row in range(BM):
                local = tile * BM + row - begin
                value = tl.full((PK,), 0, X.dtype.element_ty)
                if local < count:
                    route = tl.load(R + ex * RR + local)
                    value = tl.load(X + (route // T) * K + kk, kk < K, other=0)
                tl.store(O + (tile * BM + row) * K + kk, value, kk < K)


@triton.jit
def write_expert_offsets(
    Counts, Offsets, E: tl.constexpr, EP: tl.constexpr, BM: tl.constexpr
):
    ei = tl.arange(0, EP)
    count = tl.load(Counts + ei, ei < E, other=0)
    padded = tl.cdiv(count, BM) * BM
    end = tl.cumsum(padded)
    tl.store(Offsets + ei, end - padded, ei < E)
    tl.store(Offsets + E, tl.sum(padded, 0))


@triton.jit
def pack_expert_maps(
    Routes,
    Counts,
    Offsets,
    Experts,
    Inv,
    E: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    TILES: tl.constexpr,
):
    lane = tl.arange(0, BM)
    for expert in range(tl.program_id(0), E, tl.num_programs(0)):
        count = tl.load(Counts + expert)
        begin = tl.load(Offsets + expert)
        for local_tile in range(tl.cdiv(count, BM)):
            row = local_tile * BM + lane
            tile = begin // BM + local_tile
            tl.store(Experts + tile, expert)
            route = tl.load(Routes + expert * R + row, row < count, other=0)
            tl.store(Inv + route, begin + row, row < count)
    if tl.program_id(0) == 0:
        first_unused = tl.load(Offsets + E) // BM
        tail_lane = tl.arange(0, 256)
        for start in range(first_unused, TILES, 256):
            i = start + tail_lane
            tl.store(Experts + i, -1, i < TILES)


@triton.jit
def pack_tiles(
    X,
    Routes,
    Counts,
    Offsets,
    Experts,
    Inv,
    A,
    E: tl.constexpr,
    EP: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
    BM: tl.constexpr,
):
    tile = tl.program_id(0)
    ei = tl.arange(0, EP)
    ends = tl.load(Offsets + ei + 1, ei < E, other=2147483647)
    ex = tl.sum((tile * BM >= ends).to(tl.int32), 0)
    tl.store(Experts + tile, tl.where(ex < E, ex, -1))
    if ex < E:
        begin = tl.load(Offsets + ex)
        count = tl.load(Counts + ex)
        rs = tile * BM - begin + tl.arange(0, BM)
        route = tl.load(Routes + ex * R + rs, rs < count, other=0)
        tl.store(Inv + route, tile * BM + tl.arange(0, BM), rs < count)


@triton.jit
def cast_expert_ids(Input, Output, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(Input + i, i < TOTAL, other=0).to(tl.int32)
    tl.store(Output + i, v, i < TOTAL)


@triton.jit
def scatter_token_rows(
    X, Inv, PackedX, M: tl.constexpr, K: tl.constexpr, T: tl.constexpr
):
    kk = tl.arange(0, K)
    for token in range(tl.program_id(0), M, tl.num_programs(0)):
        x = tl.load(X + token * K + kk)
        buf = tle.dsa.to_buffer(x, space=tle.dsa.ascend.UB)
        for rank in range(T):
            row = tl.load(Inv + token * T + rank)
            with tle.dsa.hint(inter_no_alias=True):
                tle.dsa.copy(buf, PackedX + row * K + kk, [K])


@triton.jit
def pack_maps_and_padding(
    Routes,
    Counts,
    Offsets,
    Experts,
    Inv,
    PackedX,
    E: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    TILES: tl.constexpr,
    K: tl.constexpr,
):
    lane = tl.arange(0, BM)
    kk = tl.arange(0, K)
    zeros = tle.dsa.to_buffer(
        tl.full((K,), 0, PackedX.dtype.element_ty), space=tle.dsa.ascend.UB
    )
    PAD_ROWS: tl.constexpr = 16
    if BM >= 32:
        zero_rows = tl.full((PAD_ROWS, K), 0, PackedX.dtype.element_ty)
        pad_lane = tl.arange(0, PAD_ROWS)
    else:
        zero_rows = None
        pad_lane = None
    for expert in range(tl.program_id(0), E, tl.num_programs(0)):
        count = tl.load(Counts + expert)
        begin = tl.load(Offsets + expert)
        end = tl.load(Offsets + expert + 1)
        for local_tile in range(tl.cdiv(count, BM)):
            row = local_tile * BM + lane
            tile = begin // BM + local_tile
            tl.store(Experts + tile, expert)
            route = tl.load(Routes + expert * R + row, row < count, other=0)
            tl.store(Inv + route, begin + row, row < count)
        first_pad = begin + count
        if BM >= 32:
            tail_begin = first_pad + (end - first_pad) // PAD_ROWS * PAD_ROWS
            for row in range(first_pad, tail_begin, PAD_ROWS):
                with tle.dsa.hint(inter_no_alias=True):
                    tl.store(
                        PackedX + (row + pad_lane[:, None]) * K + kk[None, :],
                        zero_rows,
                    )
        else:
            tail_begin = first_pad
        for row in range(tail_begin, end):
            with tle.dsa.hint(inter_no_alias=True):
                tle.dsa.copy(zeros, PackedX + row * K + kk, [K])
    if tl.program_id(0) == 0:
        first_unused = tl.load(Offsets + E) // BM
        tail_lane = tl.arange(0, 256)
        for start in range(first_unused, TILES, 256):
            i = start + tail_lane
            tl.store(Experts + i, -1, i < TILES)


def run_quantized_moe(x, w1, w2, s1, s2, topk_weights, topk_ids, bits):
    """Return routed SwiGLU MoE with symmetric group128 weights."""
    if x.dtype not in (torch.bfloat16, torch.float16) or x.device.type != "npu":
        raise NotImplementedError(
            "The Ascend implementation supports BF16 and FP16 activations"
        )
    if bits == 4 and x.dtype != torch.bfloat16:
        raise NotImplementedError("Ascend INT4 requires BF16 activations")
    tensors = (x, w1, w2, s1, s2, topk_weights, topk_ids)
    if any((not a.is_contiguous() or a.device != x.device for a in tensors)):
        raise NotImplementedError("All inputs must be contiguous on the same NPU")
    if x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("Expected rank-2 activations and rank-3 weights")
    m, k = x.shape
    e, n2, kp = w1.shape
    n = n2 // 2
    pack = 8 // bits
    if (
        min(e, k, n) <= 0
        or e > MAX_EXPERTS
        or k > MAX_HIDDEN_SIZE
        or n > MAX_INTERMEDIATE_SIZE
    ):
        raise NotImplementedError(
            "Geometry exceeds the tested Ascend indexing and UB limits"
        )
    if (
        k % GROUP_SIZE
        or n % GROUP_SIZE
        or n2 % 2
        or (kp != k // pack)
        or (w2.shape != (e, k, n // pack))
    ):
        raise ValueError(f"Invalid INT{bits} weight geometry")
    if w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise NotImplementedError("Weights must contain uint8 quantization codes")
    if (
        s1.shape != (e, 2 * n, k // GROUP_SIZE)
        or s2.shape != (e, k, n // GROUP_SIZE)
        or s1.dtype != x.dtype
        or (s2.dtype != x.dtype)
    ):
        raise ValueError("Invalid group128 scale shape or dtype")
    if (
        topk_ids.ndim != 2
        or topk_ids.shape[0] != m
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError("Invalid routing shape")
    if (
        topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_weights.dtype != torch.float32
    ):
        raise ValueError("Routing requires integer IDs and FP32 weights")
    t = topk_ids.shape[1]
    if t < 1:
        raise ValueError("top_k must be positive")
    if x.requires_grad:
        raise NotImplementedError("Forward inference only")
    if m == 0:
        return torch.empty((m, k), device=x.device, dtype=x.dtype)
    # Wide weights favor grouped reuse over one padded GEMM tile per route.
    small_routes = 16 if n >= 2048 else 64
    if n > 4096:
        small_routes = min(small_routes, max(e - 1, 1))
    if m * t <= small_routes:
        return small_route_moe(x, w1, w2, s1, s2, topk_weights, topk_ids, bits)
    out = torch.empty((m, k), device=x.device, dtype=x.dtype)

    r = m * t
    routes = torch.empty((e, r), device=x.device, dtype=torch.int32)
    counts = torch.empty((e,), device=x.device, dtype=torch.int32)
    route_experts(topk_ids, routes, counts)
    # Dense expert batches amortize dequantization with a larger M tile.
    bm = (
        128
        if m >= 8192 or (n > 4096 and r >= e * 64)
        else 64 if m >= 1024 or r >= e * 64 else 32 if m > 32 else 16
    )
    use_small_m = (
        bits == 8
        and x.dtype == torch.float16
        and k == 4096
        and n == 256
        and bm in (16, 32)
        and r <= e * 16
    )
    if use_small_m:
        bm = 16
    padded = triton.cdiv(r + e * (bm - 1), bm) * bm
    if (
        max(e * r, padded * k, padded * 2 * n) >= INT32_INDEX_LIMIT
        or r >= EXACT_ROUTE_INDEX_LIMIT
    ):
        raise NotImplementedError(
            "Geometry exceeds 32-bit addressing or exact routing-index limits"
        )
    offsets = torch.empty((e + 1,), device=x.device, dtype=torch.int32)
    experts = torch.empty((padded // bm,), device=x.device, dtype=torch.int32)
    inv = torch.empty((r,), device=x.device, dtype=torch.int32)
    packed_x = (
        None
        if use_small_m
        else torch.empty((padded, k), device=x.device, dtype=x.dtype)
    )
    h = torch.empty((padded, 2 * n), device=x.device, dtype=x.dtype)
    a = torch.empty((padded, n), device=x.device, dtype=x.dtype)
    z = torch.empty((padded, k), device=x.device, dtype=x.dtype)
    write_expert_offsets[1,](counts, offsets, e, triton.next_power_of_2(e), bm)
    if bits == 8 and k == 4096:
        if (
            bm in (32, 64)
            or use_small_m
            or (x.dtype == torch.float16 and n == 256 and bm == 128 and r >= e * 256)
        ):
            pack_expert_maps[(min(e, vector_core_count(x)),)](
                routes,
                counts,
                offsets,
                experts,
                inv,
                e,
                r,
                bm,
                padded // bm,
                num_warps=1,
                multibuffer=False,
            )
        else:
            pack_maps_and_padding[(min(e, vector_core_count(x)),)](
                routes,
                counts,
                offsets,
                experts,
                inv,
                packed_x,
                e,
                r,
                bm,
                padded // bm,
                k,
                num_warps=1,
                multibuffer=False,
            )
        if not use_small_m:
            scatter_token_rows[(vector_core_count(x),)](
                x, inv, packed_x, m, k, t, num_warps=1, multibuffer=True
            )
    else:
        if bits == 8 and bm == 32:
            pack_expert_maps[(min(e, vector_core_count(x)),)](
                routes,
                counts,
                offsets,
                experts,
                inv,
                e,
                r,
                bm,
                padded // bm,
                num_warps=1,
                multibuffer=False,
            )
        else:
            pack_tiles[padded // bm,](
                x,
                routes,
                counts,
                offsets,
                experts,
                inv,
                packed_x,
                e,
                triton.next_power_of_2(e),
                r,
                k,
                t,
                bm,
            )
        pack_hidden_states(x, routes, counts, offsets, experts, packed_x, bm, t)
    if use_small_m:
        grouped_quantized_gemm(
            x,
            w1,
            s1,
            experts,
            h,
            bm,
            bits,
            counts,
            offsets,
            routes=routes,
            route_count=r,
            topk=t,
        )
    elif bits == 8 and k == 4096 and bm in (32, 64):
        grouped_quantized_gemm(packed_x, w1, s1, experts, h, bm, bits, counts, offsets)
    elif bits == 8 and k == 4096 and n == 256 and bm == 128:
        grouped_quantized_gemm(
            packed_x, w1, s1, experts, h, bm, bits, counts, offsets, True, r >= e * 256
        )
    else:
        grouped_quantized_gemm(packed_x, w1, s1, experts, h, bm, bits)
    mask_tail = (
        bits == 8
        and x.dtype == torch.float16
        and k == 4096
        and n == 256
        and bm == 128
        and r >= e * 256
    )
    apply_swiglu(
        h,
        a,
        offsets,
        tile_experts=experts if mask_tail else None,
        counts=counts if mask_tail else None,
        bm=bm,
    )
    grouped_quantized_gemm(a, w2, s2, experts, z, bm, bits)
    combine_expert_outputs(z, topk_weights, out, inv)
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
    activation: Optional[str] = None,
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
    """Ascend group128 INT8 with BF16/FP16 and INT4 with BF16 activations."""
    from flaggems_vllm.ops.fused_marlin_moe import (
        QUANT_TYPE_UINT4B8,
        QUANT_TYPE_UINT8B128,
    )

    if (
        quant_type_id not in (QUANT_TYPE_UINT4B8, QUANT_TYPE_UINT8B128)
        or group_size != GROUP_SIZE
    ):
        raise NotImplementedError(
            "Only symmetric uint4b8 and uint8b128 with group_size=128 are supported"
        )
    if (
        activation not in (None, "silu")
        or apply_router_weight_on_input
        or inplace
        or not is_k_full
    ):
        raise NotImplementedError(
            "Only out-of-place SiLU with output router weights is supported"
        )
    options = (
        bias1,
        bias2,
        activation_func,
        moe_sum,
        expert_map,
        input_global_scale1,
        input_global_scale2,
        global_scale1,
        global_scale2,
        g_idx1,
        g_idx2,
        sort_indices1,
        sort_indices2,
        w1_zeros,
        w2_zeros,
        workspace,
        intermediate_cache13,
        intermediate_cache2,
        output,
        input_dtype,
        clamp_limit,
    )
    if any(v is not None for v in options):
        raise NotImplementedError(
            "Bias, extra quantization metadata, callbacks and caller-owned workspaces are unsupported"
        )
    if global_num_experts not in (-1, w1.shape[0]):
        raise NotImplementedError("Expert parallel mappings are unsupported")
    if hidden_states.device.type != "npu" or hidden_states.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        raise NotImplementedError("Ascend BF16 or FP16 activations are required")
    if topk_ids.device != hidden_states.device or not topk_ids.is_contiguous():
        raise NotImplementedError("Routing IDs must be contiguous on the input NPU")
    if topk_ids.dtype == torch.int64 and topk_ids.numel() > 0:
        ids32 = torch.empty(topk_ids.shape, device=topk_ids.device, dtype=torch.int32)
        cast_expert_ids[(triton.cdiv(topk_ids.numel(), 1024),)](
            topk_ids, ids32, topk_ids.numel(), 1024
        )
        topk_ids = ids32
    bits = 4 if quant_type_id == QUANT_TYPE_UINT4B8 else 8
    return run_quantized_moe(
        hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, bits
    )
