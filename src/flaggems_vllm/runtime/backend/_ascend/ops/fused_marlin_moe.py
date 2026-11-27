# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend group128 SwiGLU MoE with symmetric INT8 or INT4 weights.

INT8 accepts FP16 or BF16 activations/scales; INT4 accepts BF16.
FP16 weight decoding uses native INT8 conversion and FP16 multiplication.
GEMM accumulation and router-weighted reduction use FP32.
"""

import inspect
import os
import weakref
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable, Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from torch.utils.weak import WeakTensorKeyDictionary

try:
    import triton.experimental.tle as tle_api
    import triton.experimental.tle.language.raw as tle_raw
except (AttributeError, ImportError):
    tle_api = None
    tle_raw = None

try:
    from triton.experimental.tle import is_tle_supported as tle_is_supported
except (AttributeError, ImportError):
    tle_is_supported = None

tle = tle_api
USE_TLE_RAW_CALL = tl.constexpr(
    tle_raw is not None and "out" in inspect.signature(tle_raw.call).parameters
)

TLE_COMMON_FEATURES = (
    "dsa.alloc",
    "dsa.copy",
    "dsa.extract_slice",
    "dsa.hint",
    "dsa.subview",
    "dsa.to_buffer",
    "dsa.to_tensor",
)
TLE_INT4_FEATURES = TLE_COMMON_FEATURES + ("raw.call",)

GROUP_SIZE = 128
MAX_EXPERTS = 512
MAX_HIDDEN_SIZE = 7168
MAX_INTERMEDIATE_SIZE = 14336
# Temporary tensor payloads per grouped batch, excluding inputs, output and allocator rounding.
GROUPED_WORKSPACE_BYTES = 2 << 30


@contextmanager
def disable_all_blocks_parallel():
    # FlagGems exports TRITON_ALL_BLOCKS_PARALLEL=1 on import, which forces
    # AutoBlockify and degrades Ascend Cube/Vector multi-stage pipelines.
    saved = os.environ.pop("TRITON_ALL_BLOCKS_PARALLEL", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = saved


def check_tle_support(bits: int) -> None:
    if tle_api is None:
        raise NotImplementedError("Ascend fused_marlin_moe requires Triton TLE support")
    features = TLE_INT4_FEATURES if bits == 4 else TLE_COMMON_FEATURES
    if tle_is_supported is not None and not tle_is_supported(features):
        raise NotImplementedError(
            "Current Triton/TLE backend does not support required Ascend Marlin MoE primitives"
        )


def use_expert_panel(bits, dtype, k, n, bm, r, e, cores) -> bool:
    return (
        bits == 8
        and dtype == torch.float16
        and bm == 128
        and (
            (k == 4096 and n == 256 and r >= e * 256 and e >= cores)
            or (
                n > 256
                and k % 256 == 0
                and n % 256 == 0
                and r >= e * (128 if e < cores else 64)
            )
        )
    )


@triton.jit
def sync_ring_wait(SRC: tl.constexpr, DST: tl.constexpr, FLAG: tl.constexpr):
    if SRC == "vector":
        tle.dsa.ascend.sync_block_wait(
            SRC,
            DST,
            FLAG,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
        )
    else:
        tle.dsa.ascend.sync_block_wait(
            SRC,
            DST,
            FLAG,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
        )


@triton.jit
def sync_ring_set(SRC: tl.constexpr, DST: tl.constexpr, FLAG: tl.constexpr):
    if SRC == "vector":
        tle.dsa.ascend.sync_block_set(
            SRC,
            DST,
            FLAG,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
        )
    else:
        tle.dsa.ascend.sync_block_set(
            SRC,
            DST,
            FLAG,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
        )


@triton.jit
def load_a_tile(
    A,
    row,
    k_offset,
    K: tl.constexpr,
    BM: tl.constexpr,
    RowEnd=None,
    USE_BLOCK_PTR: tl.constexpr = False,
):
    if USE_BLOCK_PTR:
        a = tl.load(
            tl.make_block_ptr(
                A + row * K, (BM, K), (K, 1), (0, k_offset), (BM, 128), (1, 0)
            )
        )
    else:
        mi = tl.arange(0, BM)
        ki = tl.arange(0, 128)
        ptr = A + (row + mi[:, None]) * K + k_offset + ki[None, :]
        if RowEnd is None:
            a = tl.load(ptr)
        else:
            a = tl.load(ptr, (row + mi[:, None]) < RowEnd, other=0)
    return a


@triton.jit
def load_cube_weight_stage(
    Work,
    pid,
    iteration,
    BN: tl.constexpr,
    PANEL_N: tl.constexpr = 0,
    STREAM_FIRST: tl.constexpr = False,
    STAGE_N: tl.constexpr = 0,
):
    ni = tl.arange(0, BN)
    ki = tl.arange(0, 128)
    if PANEL_N:
        if STREAM_FIRST:
            sync_ring_wait("vector", "cube", 2)
        b = tl.load(Work + iteration * PANEL_N * 128 + ni[None, :] * 128 + ki[:, None])
        if STREAM_FIRST:
            sync_ring_set("cube", "vector", 4)
    else:
        EFF_STAGE_N: tl.constexpr = STAGE_N if STAGE_N > 0 else BN
        STAGES: tl.constexpr = (
            2
            if STAGE_N > 0
            else (4 if Work.dtype.element_ty == tl.float16 and BN == 512 else 2)
        )
        tl.static_assert((STAGES & (STAGES - 1)) == 0)
        sync_ring_wait("vector", "cube", 2)
        b = tl.load(
            Work
            + (pid * STAGES + (iteration & (STAGES - 1))) * EFF_STAGE_N * 128
            + ni[None, :] * 128
            + ki[:, None]
        )
        sync_ring_set("cube", "vector", 3)
    return b


@triton.jit
def dot_resident_pairs(
    a_all,
    Work,
    pid,
    iteration,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PANEL_N: tl.constexpr = 0,
    STREAM_FIRST: tl.constexpr = False,
    acc=None,
):
    zero = pid * 0
    if acc is None:
        acc = tl.full((BM, BN), 0, tl.float32)
    for pair in range(K // 256):
        a0 = tle.dsa.extract_slice(a_all, [zero, pair * 256], [BM, 128], [1, 1])
        b0 = load_cube_weight_stage(Work, pid, iteration, BN, PANEL_N, STREAM_FIRST)
        a1 = tle.dsa.extract_slice(a_all, [zero, pair * 256 + 128], [BM, 128], [1, 1])
        b1 = load_cube_weight_stage(Work, pid, iteration + 1, BN, PANEL_N, STREAM_FIRST)
        acc = tl.dot(a0, b0, acc)
        acc = tl.dot(a1, b1, acc)
        iteration += 2
    if K % 256:
        a0 = tle.dsa.extract_slice(a_all, [zero, K - 128], [BM, 128], [1, 1])
        b0 = load_cube_weight_stage(Work, pid, iteration, BN, PANEL_N, STREAM_FIRST)
        acc = tl.dot(a0, b0, acc)
        iteration += 1
    return acc, iteration


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
    acc = tl.full((BM, BN), 0, tl.float32)
    for kb in range(K // 128):
        a = load_a_tile(A, row, kb * 128, K, BM, RowEnd)
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
    STAGE_N: tl.constexpr = 0,
    USE_BLOCK_PTR: tl.constexpr = False,
):
    mi = tl.arange(0, BM)
    ni = tl.arange(0, BN)
    RESIDENT: tl.constexpr = (
        not USE_BLOCK_PTR
        and K == 4096
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
        acc, iteration = dot_resident_pairs(
            a_all, Work, pid, iteration, K, BM, BN, PANEL_N, STREAM_FIRST
        )
    elif (
        A.dtype.element_ty == tl.float16
        and BM == 64
        and BN == 512
        and not PANEL_N
        and not STAGE_N
        and K % 1024 == 0
        and N % 1024 == 0
    ):
        acc = tl.full((BM, BN), 0, tl.float32)
        for slab in range(K // 1024):
            kk = slab * 1024 + tl.arange(0, 1024)
            ptr = A + (row + mi[:, None]) * K + kk[None, :]
            if RowEnd is None:
                a_slab = tl.load(ptr)
            else:
                a_slab = tl.load(ptr, row + mi[:, None] < RowEnd, other=0)
            acc, iteration = dot_resident_pairs(
                a_slab, Work, pid, iteration, 1024, BM, BN, acc=acc
            )
    else:
        acc = tl.full((BM, BN), 0, tl.float32)
        for pair in range(K // 256):
            a0 = load_a_tile(A, row, pair * 256, K, BM, RowEnd, USE_BLOCK_PTR)
            b0 = load_cube_weight_stage(
                Work, pid, iteration, BN, PANEL_N, STREAM_FIRST, STAGE_N
            )
            a1 = load_a_tile(A, row, pair * 256 + 128, K, BM, RowEnd, USE_BLOCK_PTR)
            b1 = load_cube_weight_stage(
                Work, pid, iteration + 1, BN, PANEL_N, STREAM_FIRST, STAGE_N
            )
            acc = tl.dot(a0, b0, acc)
            acc = tl.dot(a1, b1, acc)
            iteration += 2
        if K % 256:
            a0 = load_a_tile(A, row, K - 128, K, BM, RowEnd, USE_BLOCK_PTR)
            b0 = load_cube_weight_stage(
                Work, pid, iteration, BN, PANEL_N, STREAM_FIRST, STAGE_N
            )
            acc = tl.dot(a0, b0, acc)
            iteration += 1
    if USE_BLOCK_PTR:
        cp = tl.make_block_ptr(
            Output + row * N + col, (BM, BN), (N, 1), (0, 0), (BM, BN), (1, 0)
        )
        tl.store(cp, acc.to(Output.dtype.element_ty))
    else:
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
            token = tl.load(Routes + row + local_row) // TopK
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
    acc, iteration = dot_resident_pairs(a_all, Work, pid, iteration, K, BM, BN)
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
                TopK,
                BM,
                BN,
            )


@triton.jit
def tail_task_info(Experts, Counts, Offsets, task, N: tl.constexpr):
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
            Experts, Counts, Offsets, task, N
        )
        if active:
            if paired:
                iteration = cube_tile_pair(
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
                    STAGE_N=512,
                    USE_BLOCK_PTR=True,
                )
            elif remaining <= 64:
                if panel < N // 512:
                    iteration = cube_tile_pair(
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
                        STAGE_N=512,
                        USE_BLOCK_PTR=True,
                    )
            else:
                if panel < N // 256:
                    iteration = cube_tile_pair(
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
                        STAGE_N=512,
                        USE_BLOCK_PTR=True,
                    )


@triton.jit
def dequantize_k_stages(
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
    BITS: tl.constexpr,
    STAGE_N: tl.constexpr = 0,
):
    EFF_STAGE_N: tl.constexpr = STAGE_N if STAGE_N > 0 else BN
    STAGES: tl.constexpr = (
        2
        if STAGE_N > 0
        else (4 if Work.dtype.element_ty == tl.float16 and BN == 512 else 2)
    )
    tl.static_assert((STAGES & (STAGES - 1)) == 0)
    VBN: tl.constexpr = BN // 2
    MAX_ROWS: tl.constexpr = (
        128 if BITS == 8 and Work.dtype.element_ty == tl.float16 else 64
    )
    CB: tl.constexpr = VBN if VBN <= MAX_ROWS else MAX_ROWS
    ns = tl.arange(0, CB)
    ks = tl.arange(0, 128)
    for kb in range(K // 128):
        for chunk in range(VBN // CB):
            result = load_weight_tile(
                Q,
                S,
                expert,
                col + SUB * VBN + chunk * CB,
                kb,
                N,
                K,
                CB,
                BITS,
            )
            # Wait once before overwriting either half of the GM stage.
            if iteration >= STAGES and chunk == 0:
                sync_ring_wait("cube", "vector", 3)
            offset = (
                (PID * STAGES + (iteration & (STAGES - 1))) * EFF_STAGE_N * 128
                + SUB * VBN * 128
                + chunk * CB * 128
            )
            tl.store(Work + offset + ns[:, None] * 128 + ks[None, :], result)
        # Notify Cube only after all subtiles have been stored.
        sync_ring_set("vector", "cube", 2)
        iteration += 1
    return iteration


@triton.jit
def drain_weight_ring(iteration, STAGES: tl.constexpr):
    for _ in range(tl.minimum(iteration, STAGES)):
        sync_ring_wait("cube", "vector", 3)


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
            Experts, Counts, Offsets, task, N
        )
        if active:
            if paired:
                iteration = dequantize_k_stages(
                    Q,
                    S,
                    Work,
                    expert,
                    panel * 128,
                    PID,
                    SUB,
                    iteration,
                    N,
                    K,
                    128,
                    8,
                    512,
                )
            elif remaining <= 64:
                if panel < N // 512:
                    iteration = dequantize_k_stages(
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
                        8,
                        512,
                    )
            else:
                if panel < N // 256:
                    iteration = dequantize_k_stages(
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
                        8,
                        512,
                    )
    drain_weight_ring(iteration, 2)


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
            begin = tl.load(Offsets + tl.maximum(expert, 0))
            active = active & ((tile_id - begin // 128) % 2 == 0)
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
def prepare_packed_kernel(
    W,
    S,
    Q,
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    TASKS: tl.constexpr,
    BITS: tl.constexpr,
):
    PACK: tl.constexpr = 8 // BITS
    BYTES: tl.constexpr = 128 // PACK
    GROUPS: tl.constexpr = K // 128
    TILES_N: tl.constexpr = tl.cdiv(N, BN)
    for pid in range(tl.program_id(0), TASKS, tl.num_programs(0)):
        pn = pid % TILES_N
        g = (pid // TILES_N) % GROUPS
        # Packed expert weights can exceed 2 GiB; widen before address products.
        e = (pid // (TILES_N * GROUPS)).to(tl.int64)
        ns = pn * BN + tl.arange(0, BN)
        kh = tl.arange(0, BYTES)
        v = tl.load(
            W + (e * N + ns[:, None]) * (K // PACK) + g * BYTES + kh[None, :],
            ns[:, None] < N,
            other=0,
        )
        tl.store(
            Q + ((e * GROUPS + g) * N + ns[:, None]) * BYTES + kh[None, :],
            v ^ (0x88 if BITS == 4 else 0x80),
            ns[:, None] < N,
        )
        s = tl.load(S + (e * N + ns) * GROUPS + g, ns < N, other=0)
        tl.store(T + (e * GROUPS + g) * N + ns, s, ns < N)


_WEIGHT_CACHE: WeakTensorKeyDictionary = WeakTensorKeyDictionary()


def prepare_weights(weight: torch.Tensor, scale: torch.Tensor, bits: int = 4):
    """Repack weights to (E, K//128, N, 128//pack) with WeakTensorKeyDictionary caching."""
    try:
        w_version = weight._version
        s_version = scale._version
    except RuntimeError:
        # Inference tensors have no version counter: never cache them.
        w_version = None
        s_version = None
    stream_id = torch.npu.current_stream(weight.device).npu_stream
    meta = (
        bits,
        w_version,
        s_version,
        weight.data_ptr(),
        scale.data_ptr(),
        tuple(weight.shape),
        tuple(scale.shape),
        weight.dtype,
        scale.dtype,
        stream_id,
    )
    if w_version is not None:
        entry = _WEIGHT_CACHE.get(weight)
        if entry is not None and entry[0]() is scale and entry[1] == meta:
            return entry[2], entry[3]

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
    tasks = expert_count * group_count * triton.cdiv(output_features, tile_n)
    prepare_packed_kernel[(min(tasks, 1024),)](
        weight,
        scale,
        packed,
        grouped_scale,
        output_features,
        hidden_size,
        tile_n,
        tasks,
        bits,
    )
    if w_version is not None:
        _WEIGHT_CACHE[weight] = (weakref.ref(scale), meta, packed, grouped_scale)
    return packed, grouped_scale


@triton.jit
def load_weight_tile(
    Q,
    S,
    expert,
    col,
    kb,
    N: tl.constexpr,
    K: tl.constexpr,
    RB: tl.constexpr,
    BITS: tl.constexpr,
):
    group_bytes: tl.constexpr = 128 // (8 // BITS)
    groups: tl.constexpr = K // 128
    base = (expert.to(tl.int64) * groups + kb) * N + col
    q = tl.load(Q + base * group_bytes + tl.arange(0, RB * group_bytes))
    scales = tl.load(S + base + tl.arange(0, RB))
    if BITS == 4:
        h = tl.full((RB * 128,), 0, tl.float16)
        if USE_TLE_RAW_CALL:
            h = tle_raw.call("ascend_cast_int4_to_fp16", q, 0, RB * 128, out=h)
        else:
            h = tle.dsa.ascend.raw("cast_int4_to_fp16", q, 0, RB * 128, out=h)
    else:
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
def copy_decode_input(Source, Dest, row, COLS: tl.constexpr):
    ptr = tl.make_block_ptr(
        Source + row * COLS, (128, COLS), (COLS, 1), (0, 0), (128, COLS), (1, 0)
    )
    tle.dsa.copy(ptr, Dest, [128, COLS])


@triton.jit
def decode_ub_tile(q, s):
    signed = tle.dsa.to_tensor(q, writable=False).to(tl.int8, bitcast=True)
    scale = tle.dsa.to_tensor(s, writable=False)
    return signed.to(tl.float16) * scale


@triton.jit
def dequantize_fp16_wide(
    Q,
    S,
    Work,
    expert,
    col,
    pid,
    sub,
    iteration,
    N: tl.constexpr,
    K: tl.constexpr,
    groups=None,
):
    if groups is None:
        groups = K // 128
    q0 = tle.dsa.alloc((128, 128), tl.uint8, tle.dsa.ascend.UB)
    q1 = tle.dsa.alloc((128, 128), tl.uint8, tle.dsa.ascend.UB)
    s0 = tle.dsa.alloc((128, 1), tl.float16, tle.dsa.ascend.UB)
    s1 = tle.dsa.alloc((128, 1), tl.float16, tle.dsa.ascend.UB)
    # Prefetch the next half tile while converting the current signed INT8 rows.
    first = expert.to(tl.int64) * (K // 128) * N + col + sub * 256
    ix = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    copy_decode_input(Q, q0, first, 128)
    copy_decode_input(S, s0, first, 1)
    for kb in range(groups):
        row = first + kb * N
        copy_decode_input(Q, q1, row + 128, 128)
        copy_decode_input(S, s1, row + 128, 1)
        h0 = decode_ub_tile(q0, s0)
        if iteration >= 4:
            sync_ring_wait("cube", "vector", 3)
        dest = Work + ((pid * 4 + (iteration & 3)) * 512 + sub * 256) * 128
        tl.store(dest + ix, h0)
        if kb + 1 < groups:
            copy_decode_input(Q, q0, row + N, 128)
            copy_decode_input(S, s0, row + N, 1)
        h1 = decode_ub_tile(q1, s1)
        tl.store(dest + 128 * 128 + ix, h1)
        sync_ring_set("vector", "cube", 2)
        iteration += 1
    return iteration


@triton.jit
def dequantize_weights(
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
    BITS: tl.constexpr,
    Offsets=None,
):
    STAGES: tl.constexpr = 4 if Work.dtype.element_ty == tl.float16 and BN == 512 else 2
    iteration = 0
    for task in range(PID, TASKS, GRID):
        tile = task // (N // BN)
        col = task % (N // BN) * BN
        expert = tl.load(Experts + tile)
        active = expert >= 0
        if MERGE:
            begin = tl.load(Offsets + tl.maximum(expert, 0))
            active = active & ((tile - begin // 128) % 2 == 0)
        if active:
            if BITS == 8 and Work.dtype.element_ty == tl.float16 and BN == 512:
                iteration = dequantize_fp16_wide(
                    Q, S, Work, expert, col, PID, SUB, iteration, N, K
                )
            else:
                iteration = dequantize_k_stages(
                    Q,
                    S,
                    Work,
                    expert,
                    col,
                    PID,
                    SUB,
                    iteration,
                    N,
                    K,
                    BN,
                    BITS,
                )
    drain_weight_ring(iteration, STAGES)


@triton.jit
def count_routes(
    IDs,
    ChunkCounts,
    Inv,
    R: tl.constexpr,
    E: tl.constexpr,
    EP: tl.constexpr,
    B: tl.constexpr,
):
    chunk = tl.program_id(0)
    lane = tl.arange(0, B)
    route = chunk * B + lane
    expert = tl.load(IDs + route, route < R, other=E)
    bins = tl.arange(0, EP)
    # Keep the histogram input in range and subtract the tail from its bin.
    values = tl.where(route < R, expert, 0)
    count = tl.histogram(values, EP) - tl.where(
        bins == 0, tl.maximum((chunk + 1) * B - R, 0), 0
    )
    tl.store(ChunkCounts + chunk * E + bins, count, bins < E)
    # The low bits keep routes stable within an expert. All keys fit FP32 exactly.
    key = al.sort((expert * B + lane).to(tl.float32), descending=False).to(tl.int32)
    group = key // B
    begin = tl.cumsum(count, 0) - count
    rank = lane - tl.gather(begin.to(tl.float32), tl.minimum(group, EP - 1), 0).to(
        tl.int32
    )
    # A second on-chip sort restores input order for a contiguous GM write.
    ordered = al.sort(((key % B) * B + rank).to(tl.float32), descending=False).to(
        tl.int32
    )
    tl.store(Inv + route, ordered % B, route < R)


@triton.jit
def prefix_route_chunks(
    ChunkCounts, Counts, E: tl.constexpr, CHUNKS: tl.constexpr, C: tl.constexpr
):
    group = tl.program_id(0) * 16 + tl.arange(0, 16)
    lane = tl.arange(0, C)
    total = tl.full((16,), 0, tl.int32)
    for start in range(0, CHUNKS, C):
        chunk = start + lane
        ptr = ChunkCounts + chunk[:, None] * E + group[None, :]
        mask = (chunk < CHUNKS)[:, None] & (group < E)[None, :]
        count = tl.load(ptr, mask, other=0)
        tl.store(ptr, total[None, :] + tl.cumsum(count, 0) - count, mask)
        total += tl.sum(count, 0)
    tl.store(Counts + group, total, group < E)


@triton.jit
def scatter_routes(
    IDs,
    ChunkStarts,
    Offsets,
    Inv,
    Routes,
    Experts,
    R: tl.constexpr,
    E: tl.constexpr,
    EP: tl.constexpr,
    BM: tl.constexpr,
    B: tl.constexpr,
    CHUNKS: tl.constexpr,
    TILES: tl.constexpr,
):
    chunk = tl.program_id(0)
    if chunk < CHUNKS:
        route = chunk * B + tl.arange(0, B)
        expert = tl.load(IDs + route, route < R, other=0)
        bins = tl.arange(0, EP)
        starts = tl.load(Offsets + bins, bins < E, other=0)
        if CHUNKS > 1:
            starts += tl.load(ChunkStarts + chunk * E + bins, bins < E, other=0)
        begin = tl.gather(starts.to(tl.float32, bitcast=True), expert, 0).to(
            tl.int32, bitcast=True
        )
        row = begin + tl.load(Inv + route, route < R, other=0)
        tl.store(Inv + route, row, route < R)
        if Routes is not None:
            tl.store(Routes + row, route, route < R)
    if chunk <= E:
        first = tl.load(Offsets + chunk) // BM
        if chunk < E:
            end = tl.load(Offsets + chunk + 1) // BM
            expert_id = chunk
        else:
            end = TILES
            expert_id = -1
        tile_lane = tl.arange(0, 64)
        for start in range(first, end, 64):
            tile = start + tile_lane
            tl.store(Experts + tile, expert_id, tile < end)


@triton.jit
def sort_route_indices(
    IDs,
    Keys,
    Starts,
    Counts,
    Offsets,
    E: tl.constexpr,
    EP: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    BM: tl.constexpr,
    STEPS: tl.constexpr,
):
    # The composite key orders routes by expert and then by input position.
    lane = tl.arange(0, B)
    expert = tl.load(IDs + lane, lane < R, other=E)
    keys = al.sort((expert * B + lane).to(tl.float32), descending=False)
    tl.store(Keys + lane, keys.to(tl.int32), lane < R)
    bins = tl.arange(0, EP)
    lo = tl.full((EP,), 0, tl.int32)
    hi = tl.full((EP,), R, tl.int32)
    for _ in tl.static_range(STEPS):
        mid = (lo + hi) // 2
        val = tl.gather(keys, tl.minimum(mid, B - 1), 0)
        before = val < (bins * B).to(tl.float32)
        lo = tl.where((lo < hi) & before, mid + 1, lo)
        hi = tl.where(~before & (mid < hi), mid, hi)
    following = tl.gather(lo.to(tl.float32), tl.minimum(bins + 1, EP - 1), 0).to(
        tl.int32
    )
    count = tl.where(bins < E, tl.where(bins + 1 < E, following, R) - lo, 0)
    padded = tl.cdiv(count, BM) * BM
    end = tl.cumsum(padded.to(tl.float32), 0).to(tl.int32)
    tl.store(Starts + bins, lo, bins < E)
    tl.store(Counts + bins, count, bins < E)
    tl.store(Offsets + bins, end - padded, bins < E)
    tl.store(Offsets + E, tl.sum(padded, 0))


@triton.jit
def unpack_route_indices(
    Keys,
    Starts,
    Counts,
    Offsets,
    Routes,
    Experts,
    Inv,
    E: tl.constexpr,
    B: tl.constexpr,
    BM: tl.constexpr,
    TILES: tl.constexpr,
):
    lane = tl.arange(0, BM)
    for e in range(tl.program_id(0), E, tl.num_programs(0)):
        count = tl.load(Counts + e)
        start = tl.load(Starts + e)
        begin = tl.load(Offsets + e)
        for i in range(tl.cdiv(count, BM)):
            off = i * BM + lane
            route = tl.load(Keys + start + off, off < count, other=0) % B
            tl.store(Routes + begin + off, route, off < count)
            tl.store(Inv + route, begin + off, off < count)
            tl.store(Experts + begin // BM + i, e)
    if tl.program_id(0) == 0:
        used = tl.load(Offsets + E) // BM
        for i in range(used, TILES, 128):
            cols = i + tl.arange(0, 128)
            tl.store(Experts + cols, -1, cols < TILES)


def route_experts(ids, e, bm, padded, block, need_routes, sorted_routes=False):
    r = ids.numel()
    chunks = triton.cdiv(r, block)
    ep = max(32, triton.next_power_of_2(e))
    options = {"device": ids.device, "dtype": torch.int32}
    counts = torch.empty(e, **options)
    offsets = torch.empty(e + 1, **options)
    routes = torch.empty(padded, **options) if need_routes else None
    experts = torch.empty(padded // bm, **options)
    inv = torch.empty(r, **options)
    if sorted_routes:
        size = triton.next_power_of_2(r)
        keys = torch.empty(r, **options)
        starts = torch.empty(e, **options)
        sort_route_indices[(1,)](
            ids,
            keys,
            starts,
            counts,
            offsets,
            e,
            ep,
            r,
            size,
            bm,
            size.bit_length(),
            num_warps=1,
            multibuffer=False,
        )
        unpack_route_indices[(min(e, vector_core_count(ids)),)](
            keys,
            starts,
            counts,
            offsets,
            routes,
            experts,
            inv,
            e,
            size,
            bm,
            padded // bm,
            num_warps=1,
            multibuffer=False,
        )
        return routes, counts, offsets, experts, inv
    chunk_counts = torch.empty((chunks, e), **options) if chunks > 1 else counts
    count_routes[(chunks,)](
        ids, chunk_counts, inv, r, e, ep, block, num_warps=1, multibuffer=False
    )
    if chunks > 1:
        prefix_route_chunks[(triton.cdiv(e, 16),)](
            chunk_counts, counts, e, chunks, min(32, triton.next_power_of_2(chunks))
        )
    write_expert_offsets[(1,)](counts, offsets, e, ep, bm)
    scatter_routes[(max(chunks, e + 1),)](
        ids,
        chunk_counts,
        offsets,
        inv,
        routes,
        experts,
        r,
        e,
        ep,
        bm,
        block,
        chunks,
        padded // bm,
        num_warps=1,
        multibuffer=False,
    )
    return routes, counts, offsets, experts, inv


@triton.jit
def gemm_kernel(
    A,
    Q,
    S,
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
                Offsets=Offsets,
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
        else:
            dequantize_weights(
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
                BITS,
                Offsets,
            )


@triton.jit
def compact_panel_tile(
    A,
    panel,
    output,
    row,
    col,
    pid,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    FIRST: tl.constexpr = False,
):
    cube_tile_pair(
        A, panel, output, row, col, pid, pid * 0, K, N, BM, BN, None, BN, FIRST
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
    WEIGHT_N: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GRID: tl.constexpr,
    STAGES: tl.constexpr = 1,
    EXPERT_OUTER: tl.constexpr = False,
    GateCache=None,
):
    pid = tl.program_id(0)
    COLS: tl.constexpr = N // BN
    with tle.scope(core_mode="cube"):
        cube_iteration = 0
        if GateCache is not None:
            # A panel contains matching gate/up columns. Vector consumes one
            # projection tile while Cube writes the next into the other slot.
            projection_tile = 0
            for task in range(pid, E * COLS, GRID):
                expert = task // COLS
                count = tl.load(Counts + expert)
                if count > 0:
                    begin = tl.load(Offsets + expert)
                    panel = Work + pid * BN * K
                    for start in range(0, count, 128):
                        if projection_tile >= 2:
                            tle.dsa.ascend.sync_block_wait(
                                "vector",
                                "cube",
                                6,
                                sender_pipe=al.PIPE.PIPE_MTE2,
                                receiver_pipe=al.PIPE.PIPE_FIX,
                            )
                        cache_tile = (
                            GateCache + (pid * 2 + (projection_tile & 1)) * 128 * BN
                        )
                        inputs = A + (begin + start) * K
                        if start == 0:
                            cube_tile_pair(
                                inputs,
                                panel,
                                cache_tile,
                                pid * 0,
                                0,
                                pid,
                                pid * 0,
                                K,
                                BN,
                                128,
                                BN,
                                count - start,
                                BN,
                                True,
                            )
                        else:
                            cube_tile_pair(
                                inputs,
                                panel,
                                cache_tile,
                                pid * 0,
                                0,
                                pid,
                                pid * 0,
                                K,
                                BN,
                                128,
                                BN,
                                count - start,
                                BN,
                                False,
                            )
                        tle.dsa.ascend.sync_block_set(
                            "cube",
                            "vector",
                            5,
                            sender_pipe=al.PIPE.PIPE_FIX,
                            receiver_pipe=al.PIPE.PIPE_MTE2,
                        )
                        projection_tile += 1
                    sync_ring_set("cube", "vector", 3)
            for _ in range(tl.minimum(projection_tile, 2)):
                tle.dsa.ascend.sync_block_wait(
                    "vector",
                    "cube",
                    6,
                    sender_pipe=al.PIPE.PIPE_MTE2,
                    receiver_pipe=al.PIPE.PIPE_FIX,
                )
        elif BM == 32:
            # Consume only this expert's 32-row-aligned range.
            for task in range(pid, E * COLS, GRID):
                expert = task // COLS
                col = task % COLS * BN
                count = tl.load(Counts + expert)
                if count > 0:
                    row = tl.load(Offsets + expert)
                    remaining = tl.cdiv(count, 32) * 32
                    panel = Work + (pid * STAGES + cube_iteration % STAGES) * BN * K
                    if remaining >= 128:
                        first_rows = 128
                        compact_panel_tile(
                            A, panel, Output, row, col, pid, K, N, 128, BN, True
                        )
                    elif remaining >= 64:
                        first_rows = 64
                        compact_panel_tile(
                            A, panel, Output, row, col, pid, K, N, 64, BN, True
                        )
                    else:
                        first_rows = 32
                        compact_panel_tile(
                            A, panel, Output, row, col, pid, K, N, 32, BN, True
                        )
                    row += first_rows
                    remaining -= first_rows
                    while remaining >= 128:
                        compact_panel_tile(
                            A, panel, Output, row, col, pid, K, N, 128, BN
                        )
                        row += 128
                        remaining -= 128
                    if remaining >= 64:
                        compact_panel_tile(
                            A, panel, Output, row, col, pid, K, N, 64, BN
                        )
                        row += 64
                        remaining -= 64
                    if remaining:
                        compact_panel_tile(
                            A, panel, Output, row, col, pid, K, N, 32, BN
                        )
                    sync_ring_set("cube", "vector", 3)
                    cube_iteration += 1
        elif EXPERT_OUTER:
            for cube_expert in range(pid, E, GRID):
                cube_count = tl.load(Counts + cube_expert)
                if cube_count > 0:
                    cube_begin = tl.load(Offsets + cube_expert)
                    for col_idx in range(COLS):
                        cube_col = col_idx * BN
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
                            if remaining <= 32:
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
                        sync_ring_set("cube", "vector", 3)
                        cube_iteration += 1
        else:
            for cube_task in range(pid, E * COLS, GRID):
                cube_expert = cube_task // COLS
                cube_col = cube_task % COLS * BN
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
                    sync_ring_set("cube", "vector", 3)
                    cube_iteration += 1
    with tle.scope(core_mode="vector"):
        sub = tle.dsa.ascend.sub_vec_id()
        VBN: tl.constexpr = BN // 2
        ns = tl.arange(0, VBN)
        ks = tl.arange(0, 128)
        vector_iteration = 0
        if EXPERT_OUTER:
            for vector_expert in range(pid, E, GRID):
                vector_count = tl.load(Counts + vector_expert)
                if vector_count > 0:
                    for col_idx in range(COLS):
                        vector_col = col_idx * BN + sub * VBN
                        if vector_iteration >= STAGES:
                            sync_ring_wait("cube", "vector", 3)
                        vector_stage = (
                            (pid * STAGES + vector_iteration % STAGES) * BN * K
                        )
                        for kb in range(K // 128):
                            decoded = load_weight_tile(
                                Q,
                                S,
                                vector_expert,
                                vector_col,
                                kb,
                                WEIGHT_N,
                                K,
                                VBN,
                                8,
                            )
                            if kb >= 2:
                                sync_ring_wait("cube", "vector", 4)
                            tl.store(
                                Work
                                + vector_stage
                                + kb * BN * 128
                                + (sub * VBN + ns[:, None]) * 128
                                + ks[None, :],
                                decoded,
                            )
                            sync_ring_set("vector", "cube", 2)
                        for _ in range(tl.minimum(K // 128, 2)):
                            sync_ring_wait("cube", "vector", 4)
                        vector_iteration += 1
        else:
            projection_tile = 0
            for vector_task in range(pid, E * COLS, GRID):
                vector_expert = vector_task // COLS
                if GateCache is not None:
                    vector_col = vector_task % COLS * VBN + sub * (N // 2)
                else:
                    vector_col = vector_task % COLS * BN + sub * VBN
                vector_count = tl.load(Counts + vector_expert)
                if vector_count > 0:
                    if vector_iteration >= STAGES:
                        sync_ring_wait("cube", "vector", 3)
                    vector_stage = (pid * STAGES + vector_iteration % STAGES) * BN * K
                    for kb in range(K // 128):
                        decoded = load_weight_tile(
                            Q,
                            S,
                            vector_expert,
                            vector_col,
                            kb,
                            WEIGHT_N,
                            K,
                            VBN,
                            8,
                        )
                        if kb >= 2:
                            sync_ring_wait("cube", "vector", 4)
                        tl.store(
                            Work
                            + vector_stage
                            + kb * BN * 128
                            + (sub * VBN + ns[:, None]) * 128
                            + ks[None, :],
                            decoded,
                        )
                        sync_ring_set("vector", "cube", 2)
                    for _ in range(tl.minimum(K // 128, 2)):
                        sync_ring_wait("cube", "vector", 4)
                    vector_iteration += 1
                    if GateCache is not None:
                        begin = tl.load(Offsets + vector_expert)
                        col = vector_task % COLS * VBN
                        rows = sub * 64 + tl.arange(0, 64)
                        tile_offsets = rows[:, None] * BN + ns[None, :]
                        for start in range(0, vector_count, 128):
                            tle.dsa.ascend.sync_block_wait(
                                "cube",
                                "vector",
                                5,
                                sender_pipe=al.PIPE.PIPE_FIX,
                                receiver_pipe=al.PIPE.PIPE_MTE2,
                            )
                            cache_tile = (
                                GateCache + (pid * 2 + (projection_tile & 1)) * 128 * BN
                            )
                            gate = tl.load(cache_tile + tile_offsets).to(tl.float32)
                            up = tl.load(cache_tile + tile_offsets + VBN).to(tl.float32)
                            activated = gate / (1 + tl.exp(-gate)) * up
                            tle.dsa.ascend.sync_block_set(
                                "vector",
                                "cube",
                                6,
                                sender_pipe=al.PIPE.PIPE_MTE2,
                                receiver_pipe=al.PIPE.PIPE_FIX,
                            )
                            tl.store(
                                Output
                                + (begin + start + rows[:, None]) * (N // 2)
                                + col
                                + ns[None, :],
                                activated.to(tl.float16),
                                start + rows[:, None] < vector_count,
                            )
                            projection_tile += 1
        drain_weight_ring(vector_iteration, STAGES)


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
    topk=1,
    fuse_swiglu=False,
):
    n, k = out.shape[1], a.shape[1]
    cores = triton.runtime.driver.active.utils.get_device_properties(a.device.index)[
        "num_aicore"
    ]
    if expert_panel:
        if fuse_swiglu:
            n *= 2
        panel_bn = 256
        expert_outer = k <= 512 and bm != 32
        panel_stages = 2 if expert_outer else 1
        panel_grid = min(
            w.shape[0] if expert_outer else w.shape[0] * (n // panel_bn), cores
        )
        panel_work = torch.empty(
            panel_grid * panel_stages * panel_bn * k, device=a.device, dtype=a.dtype
        )
        gate_cache = (
            torch.empty((panel_grid, 2, 128, panel_bn), device=a.device, dtype=a.dtype)
            if fuse_swiglu
            else None
        )
        expert_panel_kernel[(panel_grid,)](
            a,
            w,
            s,
            counts,
            offsets,
            panel_work,
            out,
            k,
            n,
            s.stride(1),
            w.shape[0],
            bm,
            panel_bn,
            panel_grid,
            panel_stages,
            expert_outer,
            GateCache=gate_cache,
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
    stages = 4 if a.dtype == torch.float16 and bn == 512 else 2
    work = torch.empty(
        (grid * stages * (512 if tail_tiles else bn) * 128,),
        device=a.device,
        dtype=a.dtype,
    )
    gemm_kernel[(grid,)](
        a,
        w,
        s,
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
        topk,
        disable_auto_inject_block_sync=True,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
        enable_preload=(
            True if bits == 8 and a.dtype == torch.float16 and bn == 512 else None
        ),
    )


@triton.jit
def split_route_projection(
    X,
    Partial,
    Work,
    pid,
    R: tl.constexpr,
    T: tl.constexpr,
    GRID: tl.constexpr,
    SPLITS: tl.constexpr,
):
    # Distribute the 32 quantization groups across Cube cores for a small batch.
    KP: tl.constexpr = 4096 // SPLITS
    iteration = 0
    for task in range(pid, R * SPLITS, GRID):
        route = task // SPLITS
        activation = X + (route // T) * 4096 + (task % SPLITS) * KP
        iteration = cube_tile_pair(
            activation,
            Work,
            Partial + task * 512,
            pid * 0,
            pid * 0,
            pid,
            iteration,
            KP,
            512,
            1,
            512,
        )


@triton.jit
def decode_route_projection(
    Q, S, IDs, Work, pid, sub, R: tl.constexpr, GRID: tl.constexpr, SPLITS: tl.constexpr
):
    KP: tl.constexpr = 4096 // SPLITS
    iteration = 0
    for task in range(pid, R * SPLITS, GRID):
        expert = tl.load(IDs + task // SPLITS)
        group = expert.to(tl.int64) * 32 + (task % SPLITS) * (KP // 128)
        iteration = dequantize_fp16_wide(
            Q + group * 512 * 128,
            S + group * 512,
            Work,
            pid * 0,
            0,
            pid,
            sub,
            iteration,
            512,
            KP,
        )
    drain_weight_ring(iteration, 4)


@triton.jit
def small_fused_kernel(
    X,
    Q1,
    S1,
    Q2,
    S2,
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
    BN1: tl.constexpr,
    SPLITS: tl.constexpr = 1,
):
    SM: tl.constexpr = 1 if BITS == 8 else 16
    TILE: tl.constexpr = cube_tile_pair if BN2 == 512 else cube_tile
    ep = EP
    ax = AX
    h = H
    a = Act
    z = Z
    work = W
    pid = tl.program_id(0)
    # Every physical pair enters each barrier, including pairs with no routes.
    # Global barrier 10 is separate from the GEMM ring flags 2 and 3.
    with tle.scope(core_mode="cube"):
        if SPLITS > 1:
            split_route_projection(X, h, work, pid, M * T, T, G, SPLITS)
        else:
            tle.dsa.ascend.sync_block_all("all", 10)
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
        tle.dsa.ascend.sync_block_all("all", 10)
        tle.dsa.ascend.sync_block_all("all", 10)
        cube_gemm(
            a, work, ep, z, pid, K, N, SM, BN2, M * T * (K // BN2), G, False, TILE
        )
        tle.dsa.ascend.sync_block_all("all", 10)
        if SPLITS == 1:
            tle.dsa.ascend.sync_block_all("all", 10)
    with tle.scope(core_mode="vector"):
        sub = tle.dsa.ascend.sub_vec_id()
        vp = pid * 2 + sub
        kk = tl.arange(0, PK)
        if SPLITS > 1:
            decode_route_projection(Q1, S1, IDs, work, pid, sub, M * T, G, SPLITS)
        else:
            for route in range(vp, M * T, G * 2):
                expert = tl.load(IDs + route)
                tl.store(ep + route, expert)
                x = tl.load(X + (route // T) * K + kk, kk < K, other=0)
                tl.store(ax + route * SM * K + kk, x, kk < K)
                for row in range(1, SM):
                    tl.store(ax + (route * SM + row) * K + kk, 0, kk < K)
            tle.dsa.ascend.sync_block_all("all", 10)
            dequantize_weights(
                Q1,
                S1,
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
        tle.dsa.ascend.sync_block_all("all", 10)
        for act_block in range(vp, M * T * SM * tl.cdiv(N, 256), G * 2):
            act_row = act_block // tl.cdiv(N, 256)
            act_col = act_block % tl.cdiv(N, 256) * 256 + tl.arange(0, 256)
            if SPLITS > 1:
                # Accumulate partial products in FP32, then apply the original
                # FP16 projection boundary before evaluating SwiGLU in FP32.
                av = tl.full((256,), 0, tl.float32)
                bv = tl.full((256,), 0, tl.float32)
                for part in tl.static_range(SPLITS):
                    base = (act_row * SPLITS + part) * 512 + act_col
                    av += tl.load(h + base)
                    bv += tl.load(h + base + 256)
                av = av.to(tl.float16).to(tl.float32)
                bv = bv.to(tl.float16).to(tl.float32)
            else:
                av = tl.load(h + act_row * N * 2 + act_col, act_col < N, other=0).to(
                    tl.float32
                )
                bv = tl.load(
                    h + act_row * N * 2 + act_col + N, act_col < N, other=0
                ).to(tl.float32)
            value = av / (1 + tl.exp(-av)) * bv
            tl.store(a + act_row * N + act_col, value, act_col < N)
        tle.dsa.ascend.sync_block_all("all", 10)
        dequantize_weights(
            Q2,
            S2,
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
        tle.dsa.ascend.sync_block_all("all", 10)
        COLUMNS: tl.constexpr = 512 if SPLITS > 1 else 256
        for combine_block in range(vp, M * tl.cdiv(K, COLUMNS), G * 2):
            combine_row = combine_block // tl.cdiv(K, COLUMNS)
            combine_col = combine_block % tl.cdiv(K, COLUMNS) * COLUMNS + tl.arange(
                0, COLUMNS
            )
            total = tl.full((COLUMNS,), 0, tl.float32)
            for j in tl.range(T, loop_unroll_factor=T if SPLITS > 1 and T <= 8 else 1):
                restore_route = combine_row * T + j
                zv = tl.load(
                    z + restore_route * SM * K + combine_col, combine_col < K, other=0
                ).to(tl.float32)
                prob = tl.load(P + restore_route).to(tl.float32)
                total = total + zv * prob
            tl.store(O + combine_row * K + combine_col, total, combine_col < K)
        if SPLITS == 1:
            tle.dsa.ascend.sync_block_all("all", 10)


@triton.jit
def project_route_intervals(
    A,
    Q,
    S,
    IDs,
    Work,
    Partial,
    K: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    TOP: tl.constexpr,
    GRID: tl.constexpr,
    VECTOR: tl.constexpr,
):
    # Partition the flattened (route, output tile, K pair) space across cores.
    # A core writes one FP32 fragment for each output tile its interval touches.
    pid = tl.program_id(0)
    pairs: tl.constexpr = K // 256
    tiles: tl.constexpr = R * (N // 512)
    total: tl.constexpr = tiles * pairs
    start = pid * total // GRID
    stop = (pid + 1) * total // GRID
    iteration = 0
    for tile in range(start // pairs, tl.cdiv(stop, pairs)):
        route = tile // (N // 512)
        first_pair = tl.maximum(0, start - tile * pairs)
        stop_pair = tl.minimum(pairs, stop - tile * pairs)
        if VECTOR:
            col = (tile % (N // 512)) * 512
            expert = tl.load(IDs + route)
            offset = first_pair * 2 * N
            iteration = dequantize_fp16_wide(
                Q + offset * 128,
                S + offset,
                Work,
                expert,
                col,
                pid,
                tle.dsa.ascend.sub_vec_id(),
                iteration,
                N,
                K,
                2 * (stop_pair - first_pair),
            )
        else:
            row = A + (route // TOP) * K
            k_offsets = tl.arange(0, 128)[None, :]
            acc = tl.full((1, 512), 0, tl.float32)
            for pair in range(first_pair, stop_pair):
                a0 = tl.load(row + pair * 256 + k_offsets)
                b0 = load_cube_weight_stage(Work, pid, iteration, 512)
                a1 = tl.load(row + pair * 256 + 128 + k_offsets)
                b1 = load_cube_weight_stage(Work, pid, iteration + 1, 512)
                acc = tl.dot(a0, b0, acc)
                acc = tl.dot(a1, b1, acc)
                iteration += 2
            tl.store(Partial + (tile + pid) * 512 + tl.arange(0, 512)[None, :], acc)
    if VECTOR:
        drain_weight_ring(iteration, 4)


@triton.jit
def read_route_projection(
    Partial, tile, lanes, K: tl.constexpr, TILES: tl.constexpr, GRID: tl.constexpr
):
    # Invert the integer interval partition. Fragments for a tile are adjacent
    # at tile + core_id and are added in their original K order.
    pairs: tl.constexpr = K // 256
    total: tl.constexpr = TILES * pairs
    first = tl.cdiv((tile * pairs + 1) * GRID, total) - 1
    last = tl.cdiv((tile + 1) * pairs * GRID, total) - 1
    offset = Partial + (tile + first) * 512 + lanes
    value = tl.load(offset)
    for fragment in range(1, last - first + 1):
        value += tl.load(offset + fragment * 512)
    return value.to(tl.float16).to(tl.float32)


@triton.jit
def small_route_interval_kernel(
    X,
    Q1,
    S1,
    Q2,
    S2,
    IDs,
    Probabilities,
    Work,
    GateUp,
    Activated,
    Projected,
    Output,
    K: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    T: tl.constexpr,
    GRID: tl.constexpr,
):
    routes: tl.constexpr = M * T
    with tle.scope(core_mode="cube"):
        project_route_intervals(
            X, Q1, S1, IDs, Work, GateUp, K, 2 * N, routes, T, GRID, False
        )
        tle.dsa.ascend.sync_block_all("all", 10)
        tle.dsa.ascend.sync_block_all("all", 10)
        project_route_intervals(
            Activated, Q2, S2, IDs, Work, Projected, N, K, routes, 1, GRID, False
        )
        tle.dsa.ascend.sync_block_all("all", 10)
    with tle.scope(core_mode="vector"):
        vector_id = 2 * tl.program_id(0) + tle.dsa.ascend.sub_vec_id()
        project_route_intervals(
            X, Q1, S1, IDs, Work, GateUp, K, 2 * N, routes, T, GRID, True
        )
        tle.dsa.ascend.sync_block_all("all", 10)
        lanes = tl.arange(0, 256)
        for block in range(vector_id, routes * (N // 256), 2 * GRID):
            route = block // (N // 256)
            col = block % (N // 256) * 256
            tile = route * (2 * N // 512) + col // 512
            gate = read_route_projection(
                GateUp, tile, col % 512 + lanes, K, routes * (2 * N // 512), GRID
            )
            up = read_route_projection(
                GateUp,
                tile + N // 512,
                col % 512 + lanes,
                K,
                routes * (2 * N // 512),
                GRID,
            )
            tl.store(
                Activated + route * N + col + lanes, gate / (1 + tl.exp(-gate)) * up
            )
        tle.dsa.ascend.sync_block_all("all", 10)
        project_route_intervals(
            Activated, Q2, S2, IDs, Work, Projected, N, K, routes, 1, GRID, True
        )
        tle.dsa.ascend.sync_block_all("all", 10)
        for block in range(vector_id, M * (K // 256), 2 * GRID):
            row = block // (K // 256)
            col = block % (K // 256) * 256
            acc = tl.full((256,), 0, tl.float32)
            for slot in range(T):
                route = row * T + slot
                tile = route * (K // 512) + col // 512
                value = read_route_projection(
                    Projected, tile, col % 512 + lanes, N, routes * (K // 512), GRID
                )
                acc += value * tl.load(Probabilities + route).to(tl.float32)
            tl.store(Output + row * K + col + lanes, acc)


@lru_cache(maxsize=128)
def small_route_workspace_sizes(m, k, n, t, g, bits, first_columns, second_columns):
    r = m * t
    sm = 1 if bits == 8 else 16
    sizes = (
        r * sm * k,
        r * sm * 2 * n,
        r * sm * n,
        r * sm * k,
        g
        * (4 if max(first_columns, second_columns) == 512 else 2)
        * max(first_columns, second_columns)
        * GROUP_SIZE,
    )
    return sizes


def small_route_moe(x, w1, w2, s1, s2, p, ids, out, bits):
    m, k = x.shape
    n = w1.shape[2] // 2
    t = ids.shape[1]
    g = triton.runtime.driver.active.utils.get_device_properties(x.device.index)[
        "num_aicore"
    ]
    fp16_int8 = bits == 8 and x.dtype == torch.float16
    split_projection = fp16_int8 and k == 4096 and n == 256
    wide_projection = fp16_int8 and k % 1024 == 0 and n % 1024 == 0
    if wide_projection:
        routes = m * t
        g = min(
            g, routes * (2 * n // 512) * (k // 256), routes * (k // 512) * (n // 256)
        )
        # Each of the G-1 interval boundaries adds at most one extra fragment.
        gate_up = torch.empty(
            routes * 2 * n + (g - 1) * 512, device=x.device, dtype=torch.float32
        )
        projected = torch.empty(
            routes * k + (g - 1) * 512, device=x.device, dtype=torch.float32
        )
        activated = torch.empty((routes, n), device=x.device, dtype=x.dtype)
        decode_buffer = torch.empty(g * 4 * 512 * 128, device=x.device, dtype=x.dtype)
        small_route_interval_kernel[(g,)](
            x,
            w1,
            s1,
            w2,
            s2,
            ids,
            p,
            decode_buffer,
            gate_up,
            activated,
            projected,
            out,
            k,
            n,
            m,
            t,
            g,
            disable_auto_inject_block_sync=True,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
        return
    first_columns = (
        512
        if split_projection and m * t > g // 2
        else (256 if bits == 8 or k > 4096 else GROUP_SIZE)
    )
    second_columns = min(k & -k, 512 if split_projection else 256)
    sizes = small_route_workspace_sizes(
        m, k, n, t, g, bits, first_columns, second_columns
    )
    splits = (16 if m * t <= g else 4) if split_projection else 1
    if splits > 1:
        meta = ids
        buffers = [
            x,
            torch.empty((m * t, splits, 2 * n), device=x.device, dtype=torch.float32),
        ]
        buffers.extend(
            torch.empty(size, device=x.device, dtype=x.dtype) for size in sizes[2:]
        )
    else:
        meta = torch.empty(m * t, device=x.device, dtype=torch.int32)
        buffers = [torch.empty(size, device=x.device, dtype=x.dtype) for size in sizes]
    small_fused_kernel[(g,)](
        x,
        w1,
        s1,
        w2,
        s2,
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
        second_columns,
        bits,
        first_columns,
        SPLITS=splits,
        disable_auto_inject_block_sync=True,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )


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
    OUTPUT_STRIDE: tl.constexpr,
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
            prob = tl.load(P + route).to(tl.float32)
            a = tl.load(A + pos * K + cols, cols < K, other=0).to(tl.float32)
            acc += a * prob
        tl.store(O + row * OUTPUT_STRIDE + cols, acc, cols < K)


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
        out.stride(0),
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
            route = tl.load(R + row, local < count, other=0)
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
    if x.shape[0] * t <= c.numel() * bm // 2:
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
                    route = tl.load(R + tile * BM + row)
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
    total = tl.sum(padded, 0)
    tl.store(Offsets + E, total)


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
def zero_expert_padding(
    Counts, Offsets, PackedX, E: tl.constexpr, BM: tl.constexpr, K: tl.constexpr
):
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


@triton.jit
def clip_expert_ranges(
    Counts,
    Offsets,
    LocalCounts,
    LocalOffsets,
    start,
    rows,
    E: tl.constexpr,
    EP: tl.constexpr,
):
    expert = tl.arange(0, EP)
    begin = tl.load(Offsets + expert, expert < E, other=0)
    count = tl.load(Counts + expert, expert < E, other=0)
    local_begin = tl.minimum(rows, tl.maximum(0, begin - start))
    local_end = tl.minimum(rows, tl.maximum(0, begin + count - start))
    tl.store(LocalCounts + expert, local_end - local_begin, expert < E)
    tl.store(LocalOffsets + expert, local_begin, expert < E)
    tl.store(
        LocalOffsets + E, tl.minimum(rows, tl.maximum(0, tl.load(Offsets + E) - start))
    )


def segmented_expert_moe(x, w1, w2, s1, s2, probabilities, ids, out, cores):
    """Process contiguous expert rows so a token batch split does not reread all experts."""
    m, k = x.shape
    # Token-row offsets in pack and combine are signed 32-bit element indices.
    if x.numel() > 2**31:
        return False
    e, _, gate_up, _ = w1.shape
    n = gate_up // 2
    routes_total = ids.numel()
    topk = ids.shape[1]
    align = 32
    padded = (
        triton.cdiv(routes_total + min(e, routes_total) * (align - 1), align) * align
    )
    block = 1024
    chunks = triton.cdiv(routes_total, block)
    # Both projections reuse the same decode workspace.
    metadata_bytes = 4 * (
        padded
        + padded // align
        + routes_total * (1 + int(ids.dtype == torch.int64))
        + chunks * e
        + 4 * e
        + 2
    )
    panel_bytes = cores * 256 * max(k, n) * 2
    available = GROUPED_WORKSPACE_BYTES - metadata_bytes - panel_bytes
    bytes_per_row = 2 * max(k + 2 * n, 3 * n)
    step = min(padded, (available - 2 * padded * k) // bytes_per_row // align * align)
    # Keeping SwiGLU outputs instead of projected rows can leave room for much
    # larger up-projection segments. The down projection then runs in column
    # strips, visiting all experts once per strip.
    strip_budget = available - 2 * padded * n
    up_step = min(padded, strip_budget // (2 * (k + 2 * n)) // align * align)
    # A 1024-column strip keeps each top-k gather contiguous and bounds its working set.
    strip_width = min(k, 1024, strip_budget // (2 * padded) // 256 * 256)
    use_strips = (
        n < k
        and up_step >= align
        and strip_width >= 256
        and (
            step < align
            or triton.cdiv(padded, step)
            >= 2 * (triton.cdiv(padded, up_step) + triton.cdiv(k, strip_width))
        )
    )
    if use_strips:
        step = up_step
    if step < align:
        return False
    if ids.dtype == torch.int64:
        ids32 = torch.empty((m, topk), device=x.device, dtype=torch.int32)
        cast_expert_ids[(triton.cdiv(routes_total, 1024),)](
            ids, ids32, routes_total, 1024
        )
        ids = ids32
    routes, counts, offsets, experts, inverse = route_experts(
        ids, e, align, padded, block, True
    )
    saved = torch.empty(
        (padded, n if use_strips else k), device=x.device, dtype=x.dtype
    )
    local_counts = torch.empty_like(counts)
    local_offsets = torch.empty_like(offsets)
    for start in range(0, padded, step):
        rows = min(step, padded - start)
        tiles = experts.narrow(0, start // align, rows // align)
        clip_expert_ranges[(1,)](
            counts,
            offsets,
            local_counts,
            local_offsets,
            start,
            rows,
            e,
            max(32, triton.next_power_of_2(e)),
        )
        packed = torch.empty((rows, k), device=x.device, dtype=x.dtype)
        pack_hidden_states(
            x,
            routes.narrow(0, start, rows),
            local_counts,
            local_offsets,
            tiles,
            packed,
            align,
            topk,
        )
        hidden = torch.empty((rows, 2 * n), device=x.device, dtype=x.dtype)
        grouped_quantized_gemm(
            packed,
            w1,
            s1,
            tiles,
            hidden,
            align,
            8,
            local_counts,
            local_offsets,
            expert_panel=True,
        )
        del packed
        activated = (
            saved.narrow(0, start, rows)
            if use_strips
            else torch.empty((rows, n), device=x.device, dtype=x.dtype)
        )
        apply_swiglu(
            hidden,
            activated,
            local_offsets,
            tile_experts=tiles,
            counts=local_counts,
            bm=align,
        )
        del hidden
        if not use_strips:
            grouped_quantized_gemm(
                activated,
                w2,
                s2,
                tiles,
                saved.narrow(0, start, rows),
                align,
                8,
                local_counts,
                local_offsets,
                expert_panel=True,
            )
        del activated
    if use_strips:
        for col in range(0, k, strip_width):
            width = min(k - col, strip_width)
            projected = torch.empty((padded, width), device=x.device, dtype=x.dtype)
            grouped_quantized_gemm(
                saved,
                w2.narrow(2, col, width),
                s2.narrow(2, col, width),
                experts,
                projected,
                align,
                8,
                counts,
                offsets,
                expert_panel=True,
            )
            combine_expert_outputs(
                projected, probabilities, out.narrow(1, col, width), inverse
            )
            del projected
    else:
        combine_expert_outputs(saved, probabilities, out, inverse)
    return True


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
    if topk_ids.dtype not in (torch.int32, torch.int64) or topk_weights.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError("Routing requires integer IDs and floating-point weights")
    t = topk_ids.shape[1]
    if t < 1:
        raise ValueError("top_k must be positive")
    if x.requires_grad:
        raise NotImplementedError("Forward inference only")
    out = torch.empty_like(x)
    if m == 0:
        return out
    w1, s1 = prepare_weights(w1, s1, bits)
    w2, s2 = prepare_weights(w2, s2, bits)

    small_routes = 16 if n >= 2048 else 64
    if n > 4096:
        small_routes = min(small_routes, max(e - 1, 1))
    properties = triton.runtime.driver.active.utils.get_device_properties(
        x.device.index
    )
    cores = properties["num_aicore"]
    start = 0
    batch = m
    while start < m:
        batch = min(batch, m - start)
        r = batch * t
        if r > small_routes:
            bm, use_small_m, padded = grouped_route_geometry(
                batch, t, e, k, n, bits, x.dtype
            )
            # Use a single chunk for small batches to avoid the prefix-sum kernel;
            # spread larger batches across vector cores.
            route_block = (
                min(1024, max(32, triton.next_power_of_2(r)))
                if r <= 1024
                else min(
                    1024,
                    max(
                        32,
                        triton.next_power_of_2(
                            triton.cdiv(r, properties["num_vectorcore"])
                        ),
                    ),
                )
            )
            route_chunks = triton.cdiv(r, route_block)
            need_routes = use_small_m or bits != 8 or k != 4096
            # packed_x and the down projection share storage. Both GEMMs run on the
            # same stream, so only the larger of their decode workspaces is live.
            panel = use_expert_panel(bits, x.dtype, k, n, bm, r, e, cores)
            if panel and n > 256 and r >= e * 128:
                # Keep more tokens in a bounded batch by reducing per-expert padding.
                bm = 32
                padded = triton.cdiv(r + min(e, r) * (bm - 1), bm) * bm
            decode_bytes = (
                cores * (256 * max(k, n, 1024) if panel else 4 * 512 * GROUP_SIZE) * 2
            )
            # Retain only activated rows when many routes reuse each decoded panel.
            fuse_up = (
                panel and bm == 32 and k % 1024 == 0 and n % 1024 == 0 and r > e * 256
            )
            if fuse_up:
                # Two 128x256 FP16 projection tiles per Cube core.
                decode_bytes += cores * 2 * 128 * 256 * 2
            workspace = (
                padded * (k + (n if fuse_up else 3 * n)) * 2
                + decode_bytes
                + 4
                * (
                    (padded if need_routes else 0)
                    + (e * route_chunks if route_chunks > 1 else 0)
                    + r
                    + 2 * e
                    + 1
                    + padded // bm
                    + (r if topk_ids.dtype == torch.int64 else 0)
                )
            )
            if workspace > GROUPED_WORKSPACE_BYTES:
                if (
                    start == 0
                    and batch == m
                    and panel
                    and e >= cores
                    and k % 1024 == 0
                    and n % 1024 == 0
                ):
                    if segmented_expert_moe(
                        x, w1, w2, s1, s2, topk_weights, topk_ids, out, cores
                    ):
                        return out
                if batch == 1:
                    raise NotImplementedError(
                        "One token exceeds the Ascend MoE workspace or indexing limit"
                    )
                batch = (batch + 1) // 2
                continue
        end = start + batch
        ids = topk_ids[start:end]
        if ids.dtype == torch.int64:
            ids32 = torch.empty((batch, t), device=x.device, dtype=torch.int32)
            cast_expert_ids[(triton.cdiv(r, 1024),)](ids, ids32, r, 1024)
            ids = ids32
            del ids32
        if r <= small_routes:
            small_route_moe(
                x[start:end],
                w1,
                w2,
                s1,
                s2,
                topk_weights[start:end],
                ids,
                out[start:end],
                bits,
            )
        else:
            grouped_route_moe(
                x[start:end],
                w1,
                w2,
                s1,
                s2,
                topk_weights[start:end],
                ids,
                out[start:end],
                bits,
                bm,
                use_small_m,
                padded,
                route_block,
                need_routes,
                panel,
                fuse_up,
            )
        del ids
        start = end
    return out


def grouped_route_geometry(m, t, e, k, n, bits, dtype):
    r = m * t
    bm = (
        128
        if m >= 8192
        or (
            bits == 8
            and dtype == torch.float16
            and k == 4096
            and n == 256
            and r >= e * 72
        )
        or (
            bits == 8
            and dtype == torch.float16
            and n > 256
            and k % 256 == 0
            and n % 256 == 0
            and r >= e * 64
        )
        or (n > 4096 and r >= e * 64)
        else 64 if m >= 1024 or r >= e * 64 else 32 if m > 32 else 16
    )
    if bits == 8 and dtype == torch.float16 and k % 1024 == 0 and n % 1024 == 0:
        # Skewed experts can occupy several tiles. Shorter row tiles reduce
        # padding for wide inputs until the average exceeds eight routes.
        if r <= e:
            bm = 4
        elif k > 4096 and r <= 8 * e:
            bm = 16
        elif n > 4096 and 32 * e <= r <= 64 * e:
            # Keep 512-column GEMM tiles when the intermediate is wide.
            bm = 64
    use_small_m = (
        bits == 8
        and dtype == torch.float16
        and k == 4096
        and n == 256
        and bm in (16, 32)
        and r <= e * 16
    )
    if use_small_m:
        bm = 16
    padded = triton.cdiv(r + min(e, r) * (bm - 1), bm) * bm
    return bm, use_small_m, padded


def grouped_route_moe(
    x,
    w1,
    w2,
    s1,
    s2,
    topk_weights,
    topk_ids,
    out,
    bits,
    bm,
    use_small_m,
    padded,
    route_block,
    need_routes,
    panel,
    fuse_up,
):
    m, k = x.shape
    e, _, n2, _ = w1.shape
    n = n2 // 2
    t = topk_ids.shape[1]
    routes, counts, offsets, experts, inv = route_experts(
        topk_ids,
        e,
        bm,
        padded,
        route_block,
        need_routes,
        use_small_m and e >= 128 and m * t <= 4096,
    )
    packed_x = (
        None
        if use_small_m
        else torch.empty((padded, k), device=x.device, dtype=x.dtype)
    )
    if not use_small_m:
        if bits == 8 and k == 4096:
            if bm not in (32, 64) and not panel:
                zero_expert_padding[(min(e, vector_core_count(x)),)](
                    counts, offsets, packed_x, e, bm, k, num_warps=1, multibuffer=False
                )
            scatter_token_rows[(vector_core_count(x),)](
                x, inv, packed_x, m, k, t, num_warps=1, multibuffer=True
            )
        else:
            pack_hidden_states(x, routes, counts, offsets, experts, packed_x, bm, t)
    h = torch.empty((padded, n if fuse_up else 2 * n), device=x.device, dtype=x.dtype)
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
            topk=t,
        )
    elif not panel and bits == 8 and k == 4096 and bm in (32, 64):
        grouped_quantized_gemm(packed_x, w1, s1, experts, h, bm, bits, counts, offsets)
    elif panel or (bits == 8 and k == 4096 and n == 256 and bm == 128):
        grouped_quantized_gemm(
            packed_x,
            w1,
            s1,
            experts,
            h,
            bm,
            bits,
            counts,
            offsets,
            True,
            panel,
            fuse_swiglu=fuse_up,
        )
    else:
        grouped_quantized_gemm(packed_x, w1, s1, experts, h, bm, bits, offsets=offsets)
    del routes
    if fuse_up:
        a = h
    else:
        a = torch.empty((padded, n), device=x.device, dtype=x.dtype)
        apply_swiglu(
            h,
            a,
            offsets,
            tile_experts=experts if panel else None,
            counts=counts if panel else None,
            bm=bm,
        )
    del h
    z = (
        packed_x
        if packed_x is not None
        else torch.empty((padded, k), device=x.device, dtype=x.dtype)
    )
    grouped_quantized_gemm(
        a,
        w2,
        s2,
        experts,
        z,
        bm,
        bits,
        counts=counts if panel else None,
        offsets=offsets,
        expert_panel=panel,
    )
    del a
    combine_expert_outputs(z, topk_weights, out, inv)


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
            f"Unsupported quant_type_id={quant_type_id} or group_size={group_size}. "
            "Ascend requires symmetric uint4b8 or uint8b128 weights with group_size=128."
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
        if any(v is not None for v in (g_idx1, g_idx2, sort_indices1, sort_indices2)):
            raise NotImplementedError("Ascend Marlin MoE does not support act_order")
        if input_dtype is not None:
            raise NotImplementedError(
                f"Ascend Marlin MoE does not support FP8 or other input_dtype overrides ({input_dtype})"
            )
        raise NotImplementedError(
            "Bias, extra quantization metadata, callbacks and caller-owned workspaces are unsupported"
        )
    bits = 4 if quant_type_id == QUANT_TYPE_UINT4B8 else 8
    check_tle_support(bits)
    if w1.ndim == 3 and global_num_experts not in (-1, w1.shape[0]):
        raise NotImplementedError("Expert parallel mappings are unsupported")
    with disable_all_blocks_parallel():
        return run_quantized_moe(
            hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, bits
        )
