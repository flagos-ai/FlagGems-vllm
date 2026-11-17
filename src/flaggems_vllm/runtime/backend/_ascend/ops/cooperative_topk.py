# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Exact cooperative top-k using Ascend vector sort and four-way merging.

Each segment contributes a bounded candidate list. If an omitted value could
exceed the final threshold, the row is recomputed using canonical FP32 integer
keys. NaN, infinity and subnormal rows take the same exact refinement, avoiding
hardware-sort differences at those boundaries. Ties at the threshold may use
any index, as permitted by the public interface.

Segment and merge geometries are constrained by the 192 KiB UB and the packed
(value, index) proposal ABI. Fixed vendor configurations preserve these bounds.
Latency-sensitive 2048-index rows use the existing sorted-topk merge tree, with
finite temporary inputs and mutually exclusive exact refinement for exceptions.
"""

import importlib

import torch
import triton
import triton.experimental.tle as tle
import triton.experimental.tle.language.dsa.ascend.custom_ops
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.cooperative_topk import (
    _cooperative_topk_impl,
    _cooperative_topk_small,
    _key_greater,
    _masked_histogram,
    _ordered_key,
    _radix_digit,
)
from flaggems_vllm.utils import libentry, libtuner

_ascend_topk = importlib.import_module("flaggems_vllm.runtime.backend._ascend.ops.topk")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_ascend_sort"),
    key=["WIDTH", "STRIDE", "ROWS", "K", "LOCAL"],
    strategy=["default"] * 5,
)
@triton.jit
def _sort(
    X,
    Lengths,
    Props,
    Counts,
    Flags,
    Tails,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    LOCAL: tl.constexpr,
    SEG: tl.constexpr,
    SEGS: tl.constexpr,
    GRID: tl.constexpr,
):
    src = tle.dsa.alloc([SEG], tl.float32, tle.dsa.ascend.UB)
    out = tle.dsa.alloc([2 * LOCAL], tl.float32, tle.dsa.ascend.UB)
    SORT_IMPL: tl.constexpr = 1 if K == 2048 and LOCAL == 256 and SEG == 4096 else 0
    TMP_SIZE: tl.constexpr = 16 * LOCAL + 4096 if SORT_IMPL == 1 else 4 * SEG
    tmp = tl.zeros([TMP_SIZE], tl.float32)
    for task in tl.range(tl.program_id(0), ROWS * SEGS, GRID):
        row = (task // SEGS).to(tl.int64)
        seg = task % SEGS
        n = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
        if n > K:
            offs = seg * SEG + tl.arange(0, SEG)
            valid = offs < n
            x = tl.load(X + row * STRIDE + offs, valid, other=float("-inf"))
            bits = x.to(tl.int32, bitcast=True) & 0x7FFFFFFF
            special = valid & ((bits >= 0x7F800000) | ((bits > 0) & (bits < 0x800000)))
            tl.store(Flags + task, tl.max(special.to(tl.int32), 0))
            tle.dsa.to_buffer(x, space=tle.dsa.ascend.UB, bind_buffer=src)
            props = tle.dsa.ascend.raw(
                "sort_1d_pack",
                tle.dsa.to_tensor(src),
                tmp,
                True,
                LOCAL,
                seg * SEG,
                SORT_IMPL,
                out=tle.dsa.to_tensor(out),
            )
            tl.store(Props + task * 2 * K + tl.arange(0, 2 * LOCAL), props)
            tail = tl.sum(tl.gather(props, tl.full([1], 2 * LOCAL - 2, tl.int32), 0), 0)
            tl.store(Tails + task, tl.where(n - seg * SEG > LOCAL, tail, float("-inf")))
            valid_count = tl.minimum(LOCAL, tl.maximum(n - seg * SEG, 0))
            # Keep 32-proposal run alignment. Padded values are -inf;
            # insufficient real candidates force exact refinement.
            tl.store(Counts + task, ((valid_count + 31) // 32) * 32)
        else:
            tl.store(Counts + task, 0)
            tl.store(Flags + task, 0)
            tl.store(Tails + task, float("-inf"))


@triton.jit
def _repair(
    X,
    Lengths,
    Output,
    Flags,
    Tails,
    boundary,
    selected_count,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    SEGS: tl.constexpr,
    FLAG_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
    if n <= K:
        i = tl.arange(0, K)
        tl.store(Output + row * K + i, tl.where(i < n, i, -1))
    else:
        fs = tl.arange(0, FLAG_BLOCK)
        special = tl.max(tl.load(Flags + row * SEGS + fs, fs < SEGS, other=0), 0)
        tail = tl.max(
            tl.load(Tails + row * SEGS + fs, fs < SEGS, other=float("-inf")), 0
        )
        if special or tail > boundary or selected_count < K:
            prefix = tl.full((), 0, tl.uint32)
            remaining = tl.full((), K, tl.int32)
            for shift in tl.static_range(24, -1, -8):
                h = tl.zeros([256], tl.int32)
                for start in range(0, n, BLOCK):
                    i = start + tl.arange(0, BLOCK)
                    valid = i < n
                    keys = _ordered_key(tl.load(X + row * STRIDE + i, valid, other=0.0))
                    mask: tl.constexpr = (0xFFFFFFFF << (shift + 8)) & 0xFFFFFFFF
                    candidates = valid & ((keys & mask) == prefix)
                    digits = ((keys >> shift) & 255).to(tl.int32)
                    h += _masked_histogram(digits, candidates, 256)
                digit, remaining = _radix_digit(h, remaining)
                prefix = prefix | (digit << shift)
            out_base = 0
            tie_base = 0
            for start in range(0, n, BLOCK):
                i = start + tl.arange(0, BLOCK)
                valid = i < n
                keys = _ordered_key(tl.load(X + row * STRIDE + i, valid, other=0.0))
                eq = valid & (keys == prefix)
                er = tl.cumsum(eq.to(tl.int32))
                selected = valid & (
                    _key_greater(keys, prefix) | (eq & ((tie_base + er) <= remaining))
                )
                sr = tl.cumsum(selected.to(tl.int32))
                tl.store(Output + row * K + out_base + sr - 1, i, selected)
                tie_base += tl.sum(eq.to(tl.int32), 0)
                out_base += tl.sum(selected.to(tl.int32), 0)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_ascend_merge"),
    key=["TOPK", "LOAD_CAP", "nsrc", "ngrp", "total_tasks"],
    strategy=["default"] * 5,
)
@triton.jit
def _merge_level(
    src_ptr,
    nv_src_ptr,
    dst_ptr,
    nv_dst_ptr,
    nsrc,
    ngrp,
    total_tasks,
    TOPK: tl.constexpr,
    LOAD_CAP: tl.constexpr,
    GRID: tl.constexpr,
):
    """One strided CTA per (row, group): 4-way exhaustion merge into TOPK slots.

    Way ``w`` of the group lives at proposal offset ``w * TOPK`` of ``src`` and
    holds ``nv_src[row * nsrc + w]`` valid proposals; only that prefix is ever
    described to the op.
    """
    pid = tl.program_id(0)
    in_ub = tle.dsa.alloc(
        [4 * 2 * TOPK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    # Exhaustion merge may emit all four ways before the prefix is truncated.
    out_ub = tle.dsa.alloc([8 * TOPK], tl.float32, tle.dsa.ascend.UB)
    for t in tl.range(pid, total_tasks, GRID):
        row = t // ngrp
        grp = t % ngrp
        first = grp * 4
        ways = tl.minimum(4, nsrc - first)
        base = src_ptr + row * nsrc * 2 * TOPK
        nv_base = nv_src_ptr + row * nsrc
        for w in tl.static_range(0, 4):
            if w < ways:
                view = tle.dsa.subview(in_ub, [w * 2 * TOPK], [2 * LOAD_CAP], [1])
                tle.dsa.copy(
                    base + (first + w) * 2 * TOPK + tl.arange(0, 2 * LOAD_CAP),
                    view,
                    [2 * LOAD_CAP],
                )
        tl.debug_barrier()
        wi = tl.minimum(first, nsrc - 1)
        l0 = tl.load(nv_base + wi)
        wi = tl.minimum(first + 1, nsrc - 1)
        l1 = tl.where(first + 1 < nsrc, tl.load(nv_base + wi), 0)
        wi = tl.minimum(first + 2, nsrc - 1)
        l2 = tl.where(first + 2 < nsrc, tl.load(nv_base + wi), 0)
        wi = tl.minimum(first + 3, nsrc - 1)
        l3 = tl.where(first + 3 < nsrc, tl.load(nv_base + wi), 0)
        c0 = tl.zeros((), dtype=tl.int32)
        c1 = tl.zeros((), dtype=tl.int32)
        c2 = tl.zeros((), dtype=tl.int32)
        c3 = tl.zeros((), dtype=tl.int32)
        got = tl.zeros((), dtype=tl.int32)
        cons = tl.zeros([4], dtype=tl.int32)
        idx4 = tl.arange(0, 4)
        dst_base = dst_ptr + (row * ngrp + grp) * 2 * TOPK
        for _ in tl.static_range(0, 4):
            if got < TOPK and l0 + l1 + l2 + l3 > 0:
                _merged, cons = tle.dsa.ascend.raw(
                    "merge_exhaust_sort4",
                    tle.dsa.to_tensor(in_ub),
                    ways,
                    c0,
                    TOPK + c1,
                    2 * TOPK + c2,
                    3 * TOPK + c3,
                    l0,
                    l1,
                    l2,
                    l3,
                    out=[tle.dsa.to_tensor(out_ub), cons],
                )
                prefix_buf = tle.dsa.subview(
                    out_ub,
                    [0],
                    [2 * TOPK],
                    [1],
                )
                prefix_out = tle.dsa.to_tensor(prefix_buf)
                cons0 = tl.sum(tl.where(idx4 == 0, cons, 0))
                cons1 = tl.sum(tl.where(idx4 == 1, cons, 0))
                cons2 = tl.sum(tl.where(idx4 == 2, cons, 0))
                cons3 = tl.sum(tl.where(idx4 == 3, cons, 0))
                take = tl.minimum(cons0 + cons1 + cons2 + cons3, TOPK - got)
                c0 += cons0
                c1 += cons1
                c2 += cons2
                c3 += cons3
                l0 -= cons0
                l1 -= cons1
                l2 -= cons2
                l3 -= cons3
                oo = tl.arange(0, 2 * TOPK)
                tl.store(dst_base + got * 2 + oo, prefix_out, mask=oo < take * 2)
                tl.debug_barrier()
                got += take
        tl.store(nv_dst_ptr + t, got)


@triton.jit
def _merge_final(
    src_ptr,
    nv_src_ptr,
    scratch_ptr,
    out_ptr,
    nsrc,
    num_rows,
    TOPK: tl.constexpr,
    GRID: tl.constexpr,
):
    """Last level: strided over rows, merge the remaining ways and unpack indices.

    Rows whose ways are all empty are skipped here: ``got`` stays 0 and
    the subsequent repair stage writes their short-row indices and padding.
    """
    boundary = tl.full((), float("-inf"), tl.float32)
    total_found = tl.full((), 0, tl.int32)
    pid = tl.program_id(0)
    # Only FANOUT ways are resident at a time: with TOPK == 2048 the previous
    # "all nsrc ways at once" form needed 4*2*TOPK + 8*TOPK floats and did not
    # fit the UB budget (measured ub overflow).  Ways are merged in groups of
    # four into independent group scratch before the final merge. The input
    # proposal slots remain intact while each group's DMA writes complete.
    in_ub = tle.dsa.alloc(
        [4 * 2 * TOPK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    # Exhaustion merge may emit all four ways before the prefix is truncated.
    out_ub = tle.dsa.alloc([8 * TOPK], tl.float32, tle.dsa.ascend.UB)
    oo = tl.arange(0, 2 * TOPK)
    for row in tl.range(pid, num_rows, GRID):
        row_base = src_ptr + row * nsrc * 2 * TOPK
        row_scratch = scratch_ptr + row * 10 * TOPK
        group_base = row_scratch + 2 * TOPK
        nv_base = nv_src_ptr + row * nsrc
        ng = (nsrc + 4 - 1) // 4
        group_counts = tl.zeros([4], tl.int32)
        for g in tl.range(0, ng):
            first = g * 4
            for w in tl.static_range(0, 4):
                if w < nsrc - first:
                    view = tle.dsa.subview(in_ub, [w * 2 * TOPK], [2 * TOPK], [1])
                    tle.dsa.copy(
                        row_base + (first + w) * 2 * TOPK + tl.arange(0, 2 * TOPK),
                        view,
                        [2 * TOPK],
                    )
            tl.debug_barrier()
            l0 = tl.load(nv_base + first + 0, first + 0 < nsrc, other=0)
            l1 = tl.load(nv_base + first + 1, first + 1 < nsrc, other=0)
            l2 = tl.load(nv_base + first + 2, first + 2 < nsrc, other=0)
            l3 = tl.load(nv_base + first + 3, first + 3 < nsrc, other=0)
            c0 = tl.zeros((), dtype=tl.int32)
            c1 = tl.zeros((), dtype=tl.int32)
            c2 = tl.zeros((), dtype=tl.int32)
            c3 = tl.zeros((), dtype=tl.int32)
            got = tl.zeros((), dtype=tl.int32)
            cons = tl.zeros([4], dtype=tl.int32)
            idx4 = tl.arange(0, 4)
            for _ in tl.static_range(0, 4):
                if got < TOPK and l0 + l1 + l2 + l3 > 0:
                    _merged, cons = tle.dsa.ascend.raw(
                        "merge_exhaust_sort4",
                        tle.dsa.to_tensor(in_ub),
                        tl.minimum(4, nsrc - first),
                        c0,
                        TOPK + c1,
                        2 * TOPK + c2,
                        3 * TOPK + c3,
                        l0,
                        l1,
                        l2,
                        l3,
                        out=[tle.dsa.to_tensor(out_ub), cons],
                    )
                    prefix_buf = tle.dsa.subview(
                        out_ub,
                        [0],
                        [2 * TOPK],
                        [1],
                    )
                    prefix_out = tle.dsa.to_tensor(prefix_buf)
                    cons0 = tl.sum(tl.where(idx4 == 0, cons, 0))
                    cons1 = tl.sum(tl.where(idx4 == 1, cons, 0))
                    cons2 = tl.sum(tl.where(idx4 == 2, cons, 0))
                    cons3 = tl.sum(tl.where(idx4 == 3, cons, 0))
                    take = tl.minimum(cons0 + cons1 + cons2 + cons3, TOPK - got)
                    c0 += cons0
                    c1 += cons1
                    c2 += cons2
                    c3 += cons3
                    l0 -= cons0
                    l1 -= cons1
                    l2 -= cons2
                    l3 -= cons3
                    if ng == 1:
                        tl.store(
                            row_scratch + got * 2 + oo, prefix_out, mask=oo < take * 2
                        )
                    else:
                        tl.store(
                            group_base + (g * 2 * TOPK + got * 2) + oo,
                            prefix_out,
                            mask=oo < take * 2,
                        )
                    tl.debug_barrier()
                    got += take
            group_counts = tl.where(tl.arange(0, 4) == g, got, group_counts)
        tl.debug_barrier()
        if ng > 1:
            # final pass over the (at most FANOUT) merged groups, then unpack
            first = 0
            for w in tl.static_range(0, 4):
                if w < ng:
                    view = tle.dsa.subview(in_ub, [w * 2 * TOPK], [2 * TOPK], [1])
                    tle.dsa.copy(
                        group_base + (first + w) * 2 * TOPK + tl.arange(0, 2 * TOPK),
                        view,
                        [2 * TOPK],
                    )
            tl.debug_barrier()
            l0 = tl.sum(tl.where(tl.arange(0, 4) == 0, group_counts, 0))
            l1 = tl.sum(tl.where(tl.arange(0, 4) == 1, group_counts, 0))
            l2 = tl.sum(tl.where(tl.arange(0, 4) == 2, group_counts, 0))
            l3 = tl.sum(tl.where(tl.arange(0, 4) == 3, group_counts, 0))
            c0 = tl.zeros((), dtype=tl.int32)
            c1 = tl.zeros((), dtype=tl.int32)
            c2 = tl.zeros((), dtype=tl.int32)
            c3 = tl.zeros((), dtype=tl.int32)
            got = tl.zeros((), dtype=tl.int32)
            cons = tl.zeros([4], dtype=tl.int32)
            idx4 = tl.arange(0, 4)
            for _ in tl.static_range(0, 4):
                if got < TOPK and l0 + l1 + l2 + l3 > 0:
                    _merged, cons = tle.dsa.ascend.raw(
                        "merge_exhaust_sort4",
                        tle.dsa.to_tensor(in_ub),
                        ng,
                        c0,
                        TOPK + c1,
                        2 * TOPK + c2,
                        3 * TOPK + c3,
                        l0,
                        l1,
                        l2,
                        l3,
                        out=[tle.dsa.to_tensor(out_ub), cons],
                    )
                    prefix_buf = tle.dsa.subview(
                        out_ub,
                        [0],
                        [2 * TOPK],
                        [1],
                    )
                    prefix_out = tle.dsa.to_tensor(prefix_buf)
                    cons0 = tl.sum(tl.where(idx4 == 0, cons, 0))
                    cons1 = tl.sum(tl.where(idx4 == 1, cons, 0))
                    cons2 = tl.sum(tl.where(idx4 == 2, cons, 0))
                    cons3 = tl.sum(tl.where(idx4 == 3, cons, 0))
                    take = tl.minimum(cons0 + cons1 + cons2 + cons3, TOPK - got)
                    c0 += cons0
                    c1 += cons1
                    c2 += cons2
                    c3 += cons3
                    l0 -= cons0
                    l1 -= cons1
                    l2 -= cons2
                    l3 -= cons3
                    tl.store(row_scratch + got * 2 + oo, prefix_out, mask=oo < take * 2)
                    tl.debug_barrier()
                    got += take
        else:
            got = tl.sum(group_counts, 0)
        total_found = got
        if got > 0:
            acc = tle.dsa.subview(in_ub, [0], [2 * TOPK], [1])
            tl.debug_barrier()
            tle.dsa.copy(row_scratch + tl.arange(0, 2 * TOPK), acc, [2 * TOPK])
            tl.debug_barrier()
            dval = tl.zeros([TOPK], dtype=tl.float32)
            didx = tl.zeros([TOPK], dtype=tl.int32)
            dval, didx = tle.dsa.ascend.raw(
                "unpack_sort", tle.dsa.to_tensor(acc), TOPK, out=[dval, didx]
            )
            boundary = tl.sum(tl.gather(dval, tl.full([1], TOPK - 1, tl.int32), 0), 0)
            pos = tl.arange(0, TOPK)
            tl.store(out_ptr + row * TOPK + pos, tl.where(pos < got, didx, -1))
    return boundary, total_found


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_ascend_finish"),
    key=["WIDTH", "ROWS", "K", "SOURCES"],
    strategy=["default"] * 4,
)
@triton.jit
def _finish(
    Props,
    Counts,
    Scratch,
    Output,
    X,
    Lengths,
    Flags,
    Tails,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    SEGS: tl.constexpr,
    SOURCES: tl.constexpr,
    FLAG_BLOCK: tl.constexpr,
):
    boundary, selected_count = _merge_final(
        Props, Counts, Scratch, Output, SOURCES, ROWS, TOPK=K, GRID=ROWS
    )
    tl.debug_barrier()
    _repair(
        X,
        Lengths,
        Output,
        Flags,
        Tails,
        boundary,
        selected_count,
        WIDTH,
        STRIDE,
        K,
        SEGS,
        FLAG_BLOCK,
        512,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_ascend_prepare_sorted"),
    key=["WIDTH", "STRIDE", "K", "PARTS"],
    strategy=["default"] * 4,
)
@triton.jit
def _cooperative_topk_ascend_prepare_sorted(
    X,
    Lengths,
    Masked,
    Flags,
    Tails,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    part = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    n = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
    i = part * BLOCK + tl.arange(0, BLOCK)
    if n <= K:
        # The final stage produces ascending short-row indices. Keeping this
        # temporary row finite avoids passing an all-inf row to the sorter.
        tl.store(Masked + row * WIDTH + i, 0.0, i < WIDTH)
        tl.store(Flags + row * PARTS + part, 0)
    else:
        value = tl.load(X + row * STRIDE + i, i < WIDTH, other=0.0)
        bits = value.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        valid = i < n
        special = valid & ((bits >= 0x7F800000) | ((bits > 0) & (bits < 0x800000)))
        # At least K+1 positions stay finite, including exceptional values
        # temporarily replaced by zero. The exact stage reads original X.
        sanitized = tl.where(special, 0.0, value)
        tl.store(
            Masked + row * WIDTH + i,
            tl.where(valid, sanitized, float("-inf")),
            i < WIDTH,
        )
        tl.store(Flags + row * PARTS + part, tl.max(special.to(tl.int32), 0))
    tl.store(Tails + row * PARTS + part, float("-inf"))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_ascend_finish_sorted"),
    key=["WIDTH", "STRIDE", "K", "PARTS", "RUNS"],
    strategy=["default"] * 5,
)
@triton.jit
def _cooperative_topk_ascend_finish_sorted(
    Work,
    Values,
    Output,
    X,
    Lengths,
    Flags,
    Tails,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    PARTS: tl.constexpr,
    FLAG_BLOCK: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    RUNS: tl.constexpr,
    STREAM_CAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
    p = tl.arange(0, FLAG_BLOCK)
    special = tl.max(tl.load(Flags + row * PARTS + p, p < PARTS, other=0), 0)
    if n <= K or special:
        _repair(
            X,
            Lengths,
            Output,
            Flags,
            Tails,
            tl.full((), float("inf"), tl.float32),
            0,
            WIDTH,
            STRIDE,
            K,
            PARTS,
            FLAG_BLOCK,
            512,
        )
    else:
        if RUNS <= 4:
            _ascend_topk.smallk_stream_merge4_unpack_kernel(
                Work, Values, Output, ROW_WORDS, K, RUNS, K, STREAM_CAP, K
            )
        else:
            _ascend_topk.smallk_stream_merge_unpack_kernel(
                Work, Values, Output, ROW_WORDS, K, RUNS, K, STREAM_CAP, K
            )


def _cooperative_topk_ascend_sorted_impl(
    logits, lengths, output, workspace, k, max_seq_len
):
    rows, width = logits.shape
    parts = triton.cdiv(width, 4096)
    masked = torch.empty((rows, width), device=logits.device, dtype=torch.float32)
    flags = torch.empty((rows, parts), device=logits.device, dtype=torch.int32)
    tails = torch.empty((rows, parts), device=logits.device, dtype=torch.float32)
    _cooperative_topk_ascend_prepare_sorted[(parts, rows)](
        logits,
        lengths,
        masked,
        flags,
        tails,
        width,
        logits.stride(0),
        k,
        parts,
        4096,
        multibuffer=False,
    )
    sort_gm, row_words, run_len, num_runs = _ascend_topk._build_sorted_runs_round_robin(
        masked, k, seg_len=4096
    )
    values = torch.empty((rows, k), device=logits.device, dtype=torch.float32)
    out_segs = _ascend_topk._out_segs(rows, num_runs, k, run_len)
    local_words = out_segs * k * 2
    local_gm = torch.empty(
        (rows * 2 * local_words,), device=logits.device, dtype=torch.float32
    )
    _ascend_topk.smallk_core_local_stream_merge_kernel[(rows, out_segs)](
        sort_gm,
        local_gm,
        NUM_SEG_PER_ROW=num_runs,
        SORT_ROW_WORDS=row_words,
        LOCAL_ROW_WORDS=local_words,
        RUN_LEN=run_len,
        K=k,
        NUM_CORES_C=out_segs,
        STREAM_CAP=triton.next_power_of_2(4 * run_len * 2),
        multibuffer=False,
        enable_select_analysis=False,
    )
    _cooperative_topk_ascend_finish_sorted[(rows,)](
        local_gm,
        values,
        output,
        logits,
        lengths,
        flags,
        tails,
        width,
        logits.stride(0),
        k,
        parts,
        triton.next_power_of_2(parts),
        local_words,
        out_segs,
        triton.next_power_of_2(4 * k * 2),
        multibuffer=False,
        enable_select_analysis=False,
    )


def _run(logits, lengths, output, workspace, k, max_seq_len):
    rows, width = logits.shape
    if width > 1048577:
        raise NotImplementedError("Ascend cooperative_topk supports width <= 1048577")
    if k == 2048 and rows <= 8 and 65536 < width <= 131072:
        return _cooperative_topk_ascend_sorted_impl(
            logits, lengths, output, workspace, k, max_seq_len
        )
    seg = 2048 if k == 2048 and width < 16384 else 4096
    segs = triton.cdiv(width, seg)
    grid = min(40, rows * segs)
    props = torch.empty((rows, segs, 2 * k), dtype=torch.float32, device=logits.device)
    counts = torch.empty((rows, segs), dtype=torch.int32, device=logits.device)
    flags = torch.empty_like(counts)
    tails = torch.empty((rows, segs), dtype=torch.float32, device=logits.device)
    local = 256 if k == 2048 and width >= 65536 else (512 if segs >= 4 else k)
    _sort[(grid,)](
        logits,
        lengths,
        props,
        counts,
        flags,
        tails,
        width,
        logits.stride(0),
        rows,
        k,
        local,
        seg,
        segs,
        grid,
        multibuffer=False,
    )
    groups = triton.cdiv(segs, 4)
    buf_a = torch.empty(
        (rows, groups, 2 * k), dtype=torch.float32, device=logits.device
    )
    buf_b = torch.empty_like(buf_a)
    count_a = torch.empty((rows, groups), dtype=torch.int32, device=logits.device)
    count_b = torch.empty_like(count_a)
    scratch = torch.empty((rows, 10 * k), dtype=torch.float32, device=logits.device)
    current, nv, nsrc, use_a = props, counts, segs, True
    load_cap = local if k == 2048 else k
    while nsrc > 16:
        ngrp = triton.cdiv(nsrc, 4)
        dst, ndst = (buf_a, count_a) if use_a else (buf_b, count_b)
        _merge_level[(40,)](
            current,
            nv,
            dst,
            ndst,
            nsrc,
            ngrp,
            rows * ngrp,
            TOPK=k,
            LOAD_CAP=load_cap,
            GRID=40,
            multibuffer=False,
        )
        current, nv, nsrc, use_a = dst, ndst, ngrp, not use_a
        load_cap = k
    _finish[(rows,)](
        current,
        nv,
        scratch,
        output,
        logits,
        lengths,
        flags,
        tails,
        width,
        logits.stride(0),
        rows,
        k,
        segs,
        nsrc,
        triton.next_power_of_2(segs),
        multibuffer=False,
    )


class _SmallLauncher:
    def __getitem__(self, grid):
        def launch(logits, lengths, output, width, stride, k, rows, *, BLOCK):
            if width <= k:
                return _cooperative_topk_small[grid](
                    logits, lengths, output, width, stride, k, rows, BLOCK=BLOCK
                )
            return _run(logits, lengths, output, None, k, width)

        return launch


_small_launcher = _SmallLauncher()


def cooperative_topk(logits, lengths, output, workspace, k, max_seq_len):
    return _cooperative_topk_impl(
        logits,
        lengths,
        output,
        workspace,
        k,
        max_seq_len,
        small_kernel=_small_launcher,
        wide_kernel=_run,
    )
