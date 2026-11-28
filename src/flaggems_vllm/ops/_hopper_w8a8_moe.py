# SPDX-License-Identifier: Apache-2.0
"""Hopper W8A8 schedules with indexed INT8 activations and TMA expert weights."""

import torch
import triton as tr
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._layouts import (
    BlockedLayout,
    NVMMADistributedLayout,
    NVMMASharedLayout,
    SliceLayout,
)
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonDescriptor
from triton.tools.tensor_descriptor import TensorDescriptor

from flaggems_vllm.runtime import torch_device_fn


@gluon.jit
def _issue(
    A, W, SA, SB, Bar, ptrs, mask, b_row, k, slot, BK: gl.constexpr, BN: gl.constexpr
):
    mbarrier.expect(Bar.index(slot), BN * BK)
    tma.async_copy_global_to_shared(W, [b_row, k * BK], Bar.index(slot), SB.index(slot))
    async_copy.async_copy_global_to_shared(
        SA.index(slot), A + ptrs + k * BK, mask=mask, cache_modifier=".cg"
    )
    async_copy.commit_group()


@gluon.jit
def _tile(
    A,
    W,
    SA,
    SB,
    Bar,
    AS,
    BS,
    RW,
    C,
    Sorted,
    Experts,
    n_routes,
    mi,
    ni,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    S: gl.constexpr,
    HOLD: gl.constexpr,
    N: gl.constexpr,
    DIV: gl.constexpr,
    ROUTED: gl.constexpr,
    WARPS: gl.constexpr,
):
    LOAD: gl.constexpr = BlockedLayout([1, 16], [4, 8], [WARPS, 1], [1, 0])
    STORE: gl.constexpr = BlockedLayout([1, 8], [4, 8], [WARPS, 1], [1, 0])
    MMA: gl.constexpr = NVMMADistributedLayout([3, 0], [WARPS, 1], [16, BN, 32])
    LEAD: gl.constexpr = S - 1 - HOLD
    KK: gl.constexpr = 1536 // BK
    expert = gl.load(Experts + mi)
    indices = mi * BM + gl.arange(0, BM, layout=gl.constexpr(SliceLayout(1, LOAD)))
    r = gl.load(Sorted + indices)
    valid = (r >= 0) & (r < n_routes)
    kcols = gl.arange(0, BK, layout=gl.constexpr(SliceLayout(0, LOAD)))
    ptrs = (r // DIV)[:, None] * 1536 + kcols[None, :]
    mask = valid[:, None] & (kcols[None, :] >= 0)
    out_r = gl.convert_layout(r, gl.constexpr(SliceLayout(1, STORE)))
    out_valid = (out_r >= 0) & (out_r < n_routes)
    out_cols = ni * BN + gl.arange(0, BN, layout=gl.constexpr(SliceLayout(0, STORE)))
    if expert >= 0:
        for j in gl.static_range(S):
            mbarrier.init(Bar.index(j), count=1)
        for j in gl.static_range(LEAD):
            _issue(A, W, SA, SB, Bar, ptrs, mask, expert * N + ni * BN, j, j, BK, BN)
        acc = gl.full((BM, BN), 0, gl.int32, layout=gl.constexpr(MMA))
        for j in gl.static_range(KK):
            if j + LEAD < KK:
                _issue(
                    A,
                    W,
                    SA,
                    SB,
                    Bar,
                    ptrs,
                    mask,
                    expert * N + ni * BN,
                    j + LEAD,
                    (j + LEAD) % S,
                    BK,
                    BN,
                )
            async_copy.wait_group(min(LEAD, KK - j - 1))
            mbarrier.wait(Bar.index(j % S), (j // S) & 1, deps=(SB.index(j % S),))
            gl.thread_barrier()
            pending = warpgroup_mma(
                SA.index(j % S), SB.index(j % S).permute((1, 0)), acc, is_async=True
            )
            acc = warpgroup_mma_wait(HOLD if j + 1 < KK else 0, deps=(pending,))
            # Both warpgroups can read the same B slot. Every consumer must
            # retire that slot before any warp starts refilling it.
            gl.thread_barrier()
        for j in gl.static_range(S):
            mbarrier.invalidate(Bar.index(j))
        gl.thread_barrier()
        v = gl.convert_layout(acc, gl.constexpr(STORE)).to(gl.float32)
        a_scale = gl.load(AS + out_r // DIV, out_valid, other=0)
        b_scale = gl.load(BS + expert * N + out_cols)
        out = v * a_scale[:, None] * b_scale[None, :]
        if ROUTED:
            out *= gl.load(RW + out_r, out_valid, other=0)[:, None]
        gl.store(
            C + out_r[:, None] * N + out_cols[None, :],
            out.to(gl.bfloat16),
            out_valid[:, None],
        )
    else:
        gl.store(C + out_r[:, None] * N + out_cols[None, :], 0, out_valid[:, None])


@gluon.jit
def grouped(
    A,
    W,
    AS,
    BS,
    RW,
    C,
    Sorted,
    Experts,
    Padded,
    n_routes,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    GROUP: gl.constexpr,
    N: gl.constexpr,
    DIV: gl.constexpr,
    ROUTED: gl.constexpr,
    S: gl.constexpr,
    HOLD: gl.constexpr,
    WARPS: gl.constexpr,
):
    SA = gl.allocate_shared_memory(gl.int8, [S, BM, BK], W.layout)
    SB = gl.allocate_shared_memory(gl.int8, [S, BN, BK], W.layout)
    Bar = gl.allocate_shared_memory(
        gl.int64, [S, 1], layout=gl.constexpr(mbarrier.MBarrierLayout())
    )
    MT = gl.load(Padded) // BM
    NT: gl.constexpr = N // BN
    for pid in range(gl.program_id(0), MT * NT, gl.num_programs(0)):
        first = (pid // (GROUP * NT)) * GROUP
        count = gl.minimum(MT - first, GROUP)
        inside = pid % (GROUP * NT)
        _tile(
            A,
            W,
            SA,
            SB,
            Bar,
            AS,
            BS,
            RW,
            C,
            Sorted,
            Experts,
            n_routes,
            first + inside % count,
            inside // count,
            BM,
            BN,
            BK,
            S,
            HOLD,
            N,
            DIV,
            ROUTED,
            WARPS,
        )


@tr.jit
def _scale_store(
    acc,
    AS,
    BS,
    RW,
    C,
    r,
    expert,
    col,
    n_routes,
    N: tl.constexpr,
    DIV: tl.constexpr,
    ROUTED: tl.constexpr,
):
    valid = (r >= 0) & (r < n_routes)
    sa = tl.load(AS + r // DIV, valid, other=0)
    sb = tl.load(BS + expert * N + col)
    result = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    if ROUTED:
        result *= tl.load(RW + r, valid, other=0)[:, None]
    tl.store(C + r[:, None] * N + col[None, :], result.to(tl.bfloat16), valid[:, None])


@tr.jit
def _overlap_gather(
    A,
    W,
    row,
    valid,
    b_row,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    Hidden: tl.constexpr,
    DOUBLE: tl.constexpr,
):
    acc = tl.full((BM, BN), 0, tl.int32)
    sa = tle.gpu.alloc([BM, BK], tl.int8, scope=tle.gpu.smem)
    off = tl.arange(0, BK)
    mask = tl.broadcast_to(valid[:, None], (BM, BK))
    if DOUBLE:
        sb = tle.gpu.alloc([2, BN, BK], tl.int8, scope=tle.gpu.smem)
        ready = tle.gpu.alloc_barriers(2, expect_bytes=BN * BK)
        tle.gpu.copy(W, sb.slot(0), [BN, BK], [b_row, 0], barrier=ready[0])
        for k in range(Hidden // BK):
            buf = k % 2
            ahead = (k + 1) % 2
            if k + 1 < Hidden // BK:
                tle.gpu.copy(
                    W,
                    sb.slot(ahead),
                    [BN, BK],
                    [b_row, (k + 1) * BK],
                    barrier=ready[ahead],
                )
            tle.gpu.copy(
                A + row[:, None] * Hidden + (off + k * BK)[None, :],
                sa,
                [BM, BK],
                mask=mask,
            )
            tle.gpu.barrier_wait(ready[buf], phaseIdx=k // 2)
            acc = tle.gpu.wgmma(sa, sb.slot(buf), acc, out_dtype=tl.int32, trans_b=True)
            acc = tle.gpu.wgmma_wait(0, acc)
    else:
        sb = tle.gpu.alloc([BN, BK], tl.int8, scope=tle.gpu.smem)
        ready = tle.gpu.alloc_barrier(expect_bytes=BN * BK)
        for k in range(Hidden // BK):
            tle.gpu.copy(W, sb, [BN, BK], [b_row, k * BK], barrier=ready)
            tle.gpu.copy(
                A + row[:, None] * Hidden + (off + k * BK)[None, :],
                sa,
                [BM, BK],
                mask=mask,
            )
            tle.gpu.barrier_wait(ready, phaseIdx=k)
            acc = tle.gpu.wgmma(sa, sb, acc, out_dtype=tl.int32, trans_b=True)
            acc = tle.gpu.wgmma_wait(0, acc)
    return acc


@tr.jit
def _auto_dot(
    A,
    W,
    row,
    valid,
    w_row,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    Hidden: tl.constexpr,
    BLOCKED: tl.constexpr,
    PIPE: tl.constexpr,
    FLIP: tl.constexpr = False,
):
    if FLIP:
        acc = tl.full((BN, BM), 0, tl.int32)
    else:
        acc = tl.full((BM, BN), 0, tl.int32)
    offsets = row[:, None] * Hidden + tl.arange(0, BK)[None, :]
    for k in tl.range(0, Hidden // BK, num_stages=PIPE):
        aa = tl.load(A + offsets + k * BK, valid[:, None], 0)
        if BLOCKED:
            bb = W.load([w_row * (Hidden // BK) + k * BN, 0])
        else:
            bb = W.load([w_row, k * BK])
        if FLIP:
            acc = tl.dot(bb, tl.trans(aa), acc, out_dtype=tl.int32)
        else:
            acc = tl.dot(aa, tl.trans(bb), acc, out_dtype=tl.int32)
    if FLIP:
        return tl.trans(acc)
    else:
        return acc


# Offline-selected shapes, avoiding runtime autotuning during graph capture.
_CONFIGS = {
    (1024, "down"): (64, 128, 128, 16, 16, 3, 4),
    (1024, "gate_up"): (64, 128, 256, 4, -3, 3, 4),
    (2048, "down"): (64, 128, 128, 32, 33, 2, 4),
    (2048, "gate_up"): (128, 128, 128, 16, 33, 4, 8),
    (4096, "down"): (64, 128, 128, 16, 16, 3, 4),
    (4096, "gate_up"): (128, 128, 128, 32, 33, 2, 8),
    (8192, "down"): (128, 128, 128, 16, 16, 2, 4),
    (8192, "gate_up"): (128, 128, 128, 64, 33, 2, 8),
}


@tr.jit
def _w8a8_persistent(
    A,
    W,
    AS,
    BS,
    RW,
    C,
    Sorted,
    Experts,
    Padded,
    n_routes,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    N: tl.constexpr,
    DIV: tl.constexpr,
    PIPE: tl.constexpr,
):
    m_tiles = tl.load(Padded) // BM
    num_cols: tl.constexpr = N // BN
    for pid in range(tl.program_id(0), m_tiles * num_cols, tl.num_programs(0)):
        group = pid // (GROUP * num_cols)
        first = group * GROUP
        live_group = tl.minimum(m_tiles - first, GROUP)
        within = pid % (GROUP * num_cols)
        tile_m = first + within % live_group
        col = within // live_group * BN
        expert = tl.load(Experts + tile_m)
        row = tl.load(Sorted + tile_m * BM + tl.arange(0, BM))
        valid = (row >= 0) & (row < n_routes)
        if expert >= 0:
            if PIPE == -3:
                acc = _overlap_gather(
                    A, W, row // DIV, valid, expert * N + col, BM, BN, BK, 1536, False
                )
            else:
                acc = _auto_dot(
                    A,
                    W,
                    row // DIV,
                    valid,
                    expert * N + col,
                    BM,
                    BN,
                    BK,
                    1536,
                    False,
                    3,
                )
            _scale_store(
                acc,
                AS,
                BS,
                RW,
                C,
                row,
                expert,
                col + tl.arange(0, BN),
                n_routes,
                N,
                DIV,
                DIV == 1,
            )
        else:
            columns = col + tl.arange(0, BN)
            tl.store(C + row[:, None] * N + columns[None, :], 0, valid[:, None])


def try_invoke(
    a, weight, out, sa, sb, probs, sorted_ids, owners, padded, weighted, topk, config
):
    if (
        out.ndim != 3
        or out.shape[1] != 16
        or weight.ndim != 3
        or weight.shape[0] != 64
        or sa is None
        or sb is None
        or sorted_ids is None
    ):
        return False
    m, _, n = out.shape
    gate = n == 3072
    cfg = _CONFIGS.get((m, "gate_up" if gate else "down"))
    if (
        cfg is None
        or n not in (3072, 1536)
        or weight.shape[1:] != (n, 1536)
        or topk != (16 if gate else 1)
        or weighted == gate
        or a.shape != (m if gate else m * 16, 1536)
        or a.dtype != torch.int8
        or weight.dtype != torch.int8
        or out.dtype != torch.bfloat16
        or sa.dtype != torch.float32
        or sb.dtype != torch.float32
        or sa.numel() != a.shape[0]
        or sb.numel() != 64 * n
        or config["BLOCK_SIZE_M"] != cfg[0]
        or not all(
            x.is_contiguous() and x.device == a.device
            for x in (a, weight, out, sa, sb, sorted_ids, owners, padded)
        )
        or any(x.dtype != torch.int32 for x in (sorted_ids, owners, padded))
        or (
            not gate
            and (
                probs is None
                or probs.dtype != torch.float32
                or probs.device != a.device
                or not probs.is_contiguous()
                or probs.numel() != m * 16
            )
        )
    ):
        return False
    # TMA needs the base and row stride to be 16-byte aligned.
    if weight.data_ptr() % 16:
        return False
    bm, bn, bk, group, pipe, waves, warps = cfg
    matrix = weight.view(64 * n, 1536)
    grid = (waves * torch.cuda.get_device_properties(a.device).multi_processor_count,)
    common = (
        sa,
        sb,
        sa if gate else probs,
        out,
        sorted_ids,
        owners,
        padded,
        m * 16,
        bm,
        bn,
        bk,
        group,
        n,
    )
    with torch_device_fn.device(a.device):
        if pipe == 33:
            desc = GluonDescriptor.from_tensor(
                matrix, [bn, bk], NVMMASharedLayout(min(128, bk), 8)
            )
            grouped[grid](
                a,
                desc,
                *common,
                topk,
                weighted,
                3,
                0,
                warps,
                num_warps=warps,
                num_stages=1,
            )
        else:
            desc = TensorDescriptor.from_tensor(matrix, [bn, bk])
            _w8a8_persistent[grid](
                a, desc, *common, topk, pipe, num_warps=warps, num_stages=1
            )
    return True
