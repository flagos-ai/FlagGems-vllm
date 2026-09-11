# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Forward W4A16 INT4 MoE for Ascend 910B.

Requires FlagTree compare_scalar, gather_mask, cast_int4_to_fp16 and
cube_begin/cube_end. Routing, scaling, tl.dot, activation and output reduction
are implemented here. The Cube boundary primitives are local barriers;
TLE sync_block_set/wait synchronizes the two-stage Vector/Cube pipeline.

Weight scaling uses FP32 unconditionally to avoid the unused Vector-to-Cube
transfer generated for the former scale-flag reduction. Prepared scale flags
remain in the existing weight interface. CANN compatibility and custom-op
registration belong to FlagTree's Ascend/fused_marlin_moe_custom branch.
"""

import weakref
from functools import lru_cache
from typing import Any, Callable, Optional

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def _prepare_packed_kernel(
    W,
    S,
    Q,
    T,
    Safe,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    TASKS: tl.constexpr,
):
    for pid in range(tl.program_id(0), TASKS, tl.num_programs(0)):
        pn = pid % tl.cdiv(N, BN)
        g = (pid // tl.cdiv(N, BN)) % (K // 128)
        # Packed expert weights can exceed 2 GiB; widen before address products.
        e = (pid // (tl.cdiv(N, BN) * (K // 128))).to(tl.int64)
        ns = pn * BN + tl.arange(0, BN)
        kh = tl.arange(0, 64)
        v = tl.load(
            W + e * N * (K // 2) + ns[:, None] * (K // 2) + g * 64 + kh[None, :],
            ns[:, None] < N,
            other=0,
        )
        tl.store(
            Q + ((e * (K // 128) + g) * N + ns[:, None]) * 64 + kh[None, :],
            v ^ 0x88,
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


_weight_cache = {}


def _prepare_weights(w, s):
    key = (id(w), id(s))
    try:
        version = (w._version, s._version, w.data_ptr(), s.data_ptr())
    except RuntimeError:
        version = None
    item = _weight_cache.get(key)
    if (
        version is not None
        and item is not None
        and item[0]() is w
        and item[1]() is s
        and item[2] == version
    ):
        return item[3:]
    e, n, k2 = w.shape
    k = k2 * 2
    q = torch.empty((e, k // 128, n, 64), device=w.device, dtype=torch.uint8)
    scale = torch.empty((e, k // 128, n), device=s.device, dtype=s.dtype)
    safe = torch.empty(
        (e, k // 128, triton.cdiv(n, 32)), device=w.device, dtype=torch.int32
    )
    tasks = e * (k // 128) * triton.cdiv(n, 32)
    _prepare_packed_kernel[(min(tasks, 1024),)](w, s, q, scale, safe, n, k, 32, tasks)

    def remove(_):
        _weight_cache.pop(key, None)

    if version is not None:
        _weight_cache[key] = (
            weakref.ref(w, remove),
            weakref.ref(s, remove),
            version,
            q,
            scale,
            safe,
        )
    return q, scale, safe


@triton.jit
def _cube_tile(
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
    mi = tl.arange(0, BM)
    ni = tl.arange(0, BN)
    ki = tl.arange(0, 128)
    acc = tl.full((BM, BN), 0, tl.float32)
    for kb in range(K // 128):
        tle.dsa.ascend.sync_block_wait(
            "vector",
            "cube",
            2,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
        )
        a = tl.load(A + (row + mi[:, None]) * K + kb * 128 + ki[None, :])
        b = tl.load(
            Work
            + (pid * 2 + iteration % 2) * BN * 128
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
        acc = tl.dot(a, b, acc)
        iteration += 1
    tl.store(Output + (row + mi[:, None]) * N + col + ni[None, :], acc.to(tl.bfloat16))
    return iteration


@triton.jit
def _cube_gemm(
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
):
    tle.dsa.ascend.raw("cube_begin", PID)
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
            if MERGE:
                next_expert = -2
                if tile_id + 1 < TASKS // (N // BN):
                    next_expert = tl.load(Experts + tile_id + 1)
                if next_expert == expert:
                    iteration = _cube_tile(
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
                    )
                else:
                    iteration = _cube_tile(
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
                    )
            else:
                iteration = _cube_tile(
                    A, Work, Output, tile_id * BM, col, PID, iteration, K, N, BM, BN
                )

    tle.dsa.ascend.raw("cube_end", PID)


@triton.jit
def _dequantize_tile(q, scales, fast, RB: tl.constexpr, OP: tl.constexpr):
    h = tl.full((RB * 128,), 0, tl.float16)
    h = tle.dsa.ascend.raw("cast_int4_to_fp16", q, out=h)
    values_h = tl.reshape(h, (RB, 128))
    result = (values_h.to(tl.float32) * scales.to(tl.float32)[:, None]).to(tl.bfloat16)
    return result


@triton.jit
def _dequantize(
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
    OP: tl.constexpr,
):
    VBN: tl.constexpr = BN // 2
    # Limit temporary UB usage for FP32 scaling on CANN 9.0.
    CB: tl.constexpr = 64 if VBN > 64 else VBN
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
                fast = False
                for chunk in range(VBN // CB):
                    packed = tl.load(
                        Q + (base + chunk * CB) * 64 + tl.arange(0, CB * 64)
                    )
                    scale = tl.load(S + base + chunk * CB + ns)
                    result = _dequantize_tile(packed, scale, fast, CB, OP)
                    # Wait once before overwriting either half of the GM stage.
                    if iteration >= 2 and chunk == 0:
                        tle.dsa.ascend.sync_block_wait(
                            "cube",
                            "vector",
                            3,
                            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
                            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
                        )
                    offset = (
                        (PID * 2 + iteration % 2) * BN * 128
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
    for _ in range(tl.minimum(iteration, 2)):
        tle.dsa.ascend.sync_block_wait(
            "cube",
            "vector",
            3,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
        )


@triton.jit
def _route_kernel(
    IDs,
    Routes,
    Counts,
    R: tl.constexpr,
    E: tl.constexpr,
    OP: tl.constexpr,
    CMP: tl.constexpr,
    B: tl.constexpr,
):
    lane = tl.arange(0, B)
    source = lane.to(tl.float32)
    for expert in range(tl.program_id(0), E, tl.num_programs(0)):
        count = 0
        for start in range(0, R, B):
            ids = tl.load(IDs + start + lane, start + lane < R, other=-1)
            mask = tl.full((B // 16,), 0, tl.uint16)
            mask = tle.dsa.ascend.raw(
                "compare_scalar", ids.to(tl.float32), expert.to(tl.float32), out=mask
            )
            output = tl.full((B,), 0, tl.float32)
            number = tl.full((8,), 0, tl.int32)
            output, number = tle.dsa.ascend.raw(
                "gather_mask", source, mask, out=[output, number]
            )
            found = tl.sum(tl.where(tl.arange(0, 8) == 0, number, 0), 0)
            if found > 0:
                indices = output.to(tl.int32) + start
                tl.store(Routes + expert * R + count + lane, indices, lane < found)
                count += found
        tl.store(Counts + expert, count)


def _route_experts(ids, output, counts):
    cores = triton.runtime.driver.active.utils.get_device_properties(ids.device.index)[
        "num_vectorcore"
    ]
    grid = min(counts.numel(), cores)
    block = 4096
    _route_kernel[(grid,)](
        ids,
        output,
        counts,
        ids.numel(),
        counts.numel(),
        "gather_mask",
        "compare_scalar",
        block,
        disable_auto_inject_block_sync=True,
        num_warps=1,
    )


@triton.jit
def _gemm_kernel(
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
    OP: tl.constexpr,
):
    with tle.scope(core_mode="cube"):
        _cube_gemm(
            A, Work, Experts, Output, tl.program_id(0), N, K, BM, BN, TASKS, GRID, MERGE
        )
    with tle.scope(core_mode="vector"):
        _dequantize(
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
            OP,
        )


def _gemm(a, w, s, experts, out, bm, bn=128):
    n, k = out.shape[1], a.shape[1]
    bn = min(n & -n, 256, 32768 // bm)
    merge = bm == 128 and ((k == 256 and n == 4096) or (k == 4096 and n == 512))
    input_bm = bm
    if merge:
        bm, bn = 256, 128
    tasks = a.shape[0] // input_bm * (n // bn)
    cores = triton.runtime.driver.active.utils.get_device_properties(a.device.index)[
        "num_aicore"
    ]
    grid = min(tasks, cores)
    if merge and k == 4096:
        grid = min(grid, 19)
    q, scale, safe = _prepare_weights(w, s)
    work = torch.empty((grid * 2 * bn * 128,), device=a.device, dtype=a.dtype)
    _gemm_kernel[(grid,)](
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
        "cast_int4_to_fp16",
        disable_auto_inject_block_sync=True,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )


@triton.jit
def _small_fused_kernel(
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
    C1: tl.constexpr,
    C2: tl.constexpr,
):
    # Larger K amortizes the wider first-GEMM column tile.
    BN1: tl.constexpr = 256 if K > 4096 else 128
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
        _cube_gemm(
            ax, work, ep, h, pid, 2 * N, K, 16, BN1, M * T * (2 * N // BN1), G, False
        )
        al.sync_block_all("all", 10)
        al.sync_block_all("all", 10)
        _cube_gemm(a, work, ep, z, pid, K, N, 16, BN2, M * T * (K // BN2), G, False)
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
            tl.store(ax + route * 16 * K + kk, x, kk < K)
            for row in range(1, 16):
                tl.store(ax + (route * 16 + row) * K + kk, 0, kk < K)
        al.sync_block_all("all", 10)
        _dequantize(
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
            "cast_int4_to_fp16",
        )
        al.sync_block_all("all", 10)
        for act_block in range(vp, M * T * 16 * tl.cdiv(N, 256), G * 2):
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
        _dequantize(
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
            "cast_int4_to_fp16",
        )
        al.sync_block_all("all", 10)
        for combine_block in range(vp, M * tl.cdiv(K, 256), G * 2):
            combine_row = combine_block // tl.cdiv(K, 256)
            combine_col = combine_block % tl.cdiv(K, 256) * 256 + tl.arange(0, 256)
            total = tl.full((256,), 0, tl.float32)
            for j in range(T):
                restore_route = combine_row * T + j
                zv = tl.load(
                    z + restore_route * 16 * K + combine_col, combine_col < K, other=0
                ).to(tl.float32)
                prob = tl.load(P + restore_route)
                total = total + zv * prob
            tl.store(O + combine_row * K + combine_col, total, combine_col < K)
        al.sync_block_all("all", 10)


@lru_cache(maxsize=128)
def _small_config(m, k, n, t, g):
    r = m * t
    sizes = (
        r * 16 * k,
        r * 16 * 2 * n,
        r * 16 * n,
        r * 16 * k,
        g * 2 * max(256 if k > 4096 else 128, min(k & -k, 256)) * 128,
    )
    c1 = 0
    c2 = 0
    return sizes, (c1, c2)


def _small_moe(x, w1, w2, s1, s2, p, ids):
    m, k = x.shape
    n = w1.shape[1] // 2
    t = ids.shape[1]
    g = triton.runtime.driver.active.utils.get_device_properties(x.device.index)[
        "num_aicore"
    ]
    sizes, ops = _small_config(m, k, n, t, g)
    q1, s1, f1 = _prepare_weights(w1, s1)
    q2, s2, f2 = _prepare_weights(w2, s2)
    meta = torch.empty(m * t, device=x.device, dtype=torch.int32)
    # Direct typed allocations avoid unsupported pointer casts and view-dispatch overhead.
    buffers = [torch.empty(size, device=x.device, dtype=x.dtype) for size in sizes]
    out = torch.empty_like(x)
    _small_fused_kernel[(g,)](
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
        min(k & -k, 256),
        *ops,
        disable_auto_inject_block_sync=True,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    return out


@triton.jit
def _silu_kernel(
    H,
    Offsets,
    O,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    E: tl.constexpr,
    RB: tl.constexpr,
    NC: tl.constexpr,
):
    active = TOTAL // N
    if E >= 0:
        active = tl.load(Offsets + E)
    for block in range(
        tl.program_id(0), tl.cdiv(active, RB) * tl.cdiv(N, NC), tl.num_programs(0)
    ):
        row = (block // tl.cdiv(N, NC)) * RB
        col = (block % tl.cdiv(N, NC)) * NC
        pa = tl.make_block_ptr(
            H,
            shape=(active, 2 * N),
            strides=(2 * N, 1),
            offsets=(row, col),
            block_shape=(RB, NC),
            order=(1, 0),
        )
        pb = tl.make_block_ptr(
            H,
            shape=(active, 2 * N),
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
def _combine_kernel(
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
    for block in range(tl.program_id(0), M * tl.cdiv(K, B), tl.num_programs(0)):
        row = block // tl.cdiv(K, B)
        cols = block % tl.cdiv(K, B) * B + tl.arange(0, B)
        acc = tl.full((B,), 0, tl.float32)
        for t in range(T):
            route = row * T + t
            pos = route
            if HAS_INV:
                pos = tl.load(Inv + route)
            prob = tl.load(P + route)
            a = tl.load(A + pos * K + cols, cols < K, other=0).to(tl.float32)
            acc += a * prob
        tl.store(O + row * K + cols, acc, cols < K)


def _silu(h, out, off=None, b=32):
    _silu_kernel[(_cores(h),)](
        h,
        off if off is not None else out,
        out,
        out.shape[1],
        out.numel(),
        off.numel() - 1 if off is not None else -1,
        b,
        min(out.shape[1], 256),
        num_warps=1,
        multibuffer=False,
        enable_fp_fusion=False,
    )


def _combine(a, p, out, inv=None, b=4096):
    _combine_kernel[(_cores(a),)](
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
def _pack_kernel(
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


def _cores(a):
    return triton.runtime.driver.active.utils.get_device_properties(a.device.index)[
        "num_vectorcore"
    ]


def _pack(x, r, c, off, e, out, bm, t, br=4):
    if r.shape[1] <= c.numel() * bm // 2:
        # Sparse routing produces mostly padding: skip input loads for zero rows.
        _pack_rows_kernel[(min(e.numel(), _cores(x)),)](
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
    _pack_kernel[(_cores(x),)](
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
def _pack_rows_kernel(
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
                value = tl.full((PK,), 0, tl.bfloat16)
                if local < count:
                    route = tl.load(R + ex * RR + local)
                    value = tl.load(X + (route // T) * K + kk, kk < K, other=0)
                tl.store(O + (tile * BM + row) * K + kk, value, kk < K)


@triton.jit
def _offsets(Counts, Offsets, E: tl.constexpr, EP: tl.constexpr, BM: tl.constexpr):
    ei = tl.arange(0, EP)
    count = tl.load(Counts + ei, ei < E, other=0)
    padded = tl.cdiv(count, BM) * BM
    end = tl.cumsum(padded)
    tl.store(Offsets + ei, end - padded, ei < E)
    tl.store(Offsets + E, tl.sum(padded, 0))


@triton.jit
def _pack_tiles(
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
def _cast_ids(Input, Output, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(Input + i, i < TOTAL, other=0).to(tl.int32)
    tl.store(Output + i, v, i < TOTAL)


def _run(x, w1, w2, s1, s2, topk_weights, topk_ids):
    """Return routed SwiGLU MoE for symmetric uint4b8, group size 128."""
    if x.dtype != torch.bfloat16 or x.device.type != "npu":
        raise NotImplementedError("The Ascend implementation supports BF16 activations")
    tensors = (x, w1, w2, s1, s2, topk_weights, topk_ids)
    if any((not a.is_contiguous() or a.device != x.device for a in tensors)):
        raise NotImplementedError("All inputs must be contiguous on the same NPU")
    if x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("Expected rank-2 activations and rank-3 weights")
    m, k = x.shape
    e, n2, kp = w1.shape
    n = n2 // 2
    if min(e, k, n) <= 0 or e > 512 or k > 7168 or n > 14336:
        raise NotImplementedError(
            "Geometry exceeds the tested Ascend indexing and UB limits"
        )
    if k % 128 or n % 128 or n2 % 2 or (kp != k // 2) or (w2.shape != (e, k, n // 2)):
        raise ValueError("Invalid packed INT4 weight geometry")
    if w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise NotImplementedError("Weights must contain uint8 nibble pairs")
    if (
        s1.shape != (e, 2 * n, k // 128)
        or s2.shape != (e, k, n // 128)
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
        return _small_moe(x, w1, w2, s1, s2, topk_weights, topk_ids)
    out = torch.empty((m, k), device=x.device, dtype=x.dtype)

    r = m * t
    routes = torch.empty((e, r), device=x.device, dtype=torch.int32)
    counts = torch.empty((e,), device=x.device, dtype=torch.int32)
    _route_experts(topk_ids, routes, counts)
    # Dense expert batches amortize dequantization with a larger M tile.
    bm = (
        128
        if m >= 8192 or (n > 4096 and r >= e * 64)
        else 64 if m >= 1024 or r >= e * 64 else 32 if m > 32 else 16
    )
    padded = triton.cdiv(r + e * (bm - 1), bm) * bm
    if max(e * r, padded * k, padded * 2 * n) >= 2**31 or r >= 2**24:
        raise NotImplementedError(
            "Geometry exceeds 32-bit addressing or exact routing-index limits"
        )
    offsets = torch.empty((e + 1,), device=x.device, dtype=torch.int32)
    experts = torch.empty((padded // bm,), device=x.device, dtype=torch.int32)
    inv = torch.empty((r,), device=x.device, dtype=torch.int32)
    packed_x = torch.empty((padded, k), device=x.device, dtype=x.dtype)
    h = torch.empty((padded, 2 * n), device=x.device, dtype=x.dtype)
    a = torch.empty((padded, n), device=x.device, dtype=x.dtype)
    z = torch.empty((padded, k), device=x.device, dtype=x.dtype)
    _offsets[1,](counts, offsets, e, triton.next_power_of_2(e), bm)
    _pack_tiles[padded // bm,](
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
    _pack(x, routes, counts, offsets, experts, packed_x, bm, t)
    _gemm(packed_x, w1, s1, experts, h, bm, 256 if m <= 32 else 128)
    _silu(h, a, offsets)
    _gemm(a, w2, s2, experts, z, bm, 256 if m <= 32 else 128)
    _combine(z, topk_weights, out, inv)
    return out


def fused_marlin_moe_w4a16_int4(
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
    """Ascend BF16/uint4b8 specialization; unsupported options raise explicitly."""
    from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT4B8

    if quant_type_id != QUANT_TYPE_UINT4B8 or group_size != 128:
        raise NotImplementedError(
            "Only symmetric uint4b8 with group_size=128 is supported"
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
    if hidden_states.device.type != "npu" or hidden_states.dtype != torch.bfloat16:
        raise NotImplementedError("Ascend BF16 activations are required")
    if topk_ids.device != hidden_states.device or not topk_ids.is_contiguous():
        raise NotImplementedError("Routing IDs must be contiguous on the input NPU")
    if topk_ids.dtype == torch.int64 and topk_ids.numel() > 0:
        ids32 = torch.empty(topk_ids.shape, device=topk_ids.device, dtype=torch.int32)
        _cast_ids[(triton.cdiv(topk_ids.numel(), 1024),)](
            topk_ids, ids32, topk_ids.numel(), 1024
        )
        topk_ids = ids32
    return _run(hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids)
