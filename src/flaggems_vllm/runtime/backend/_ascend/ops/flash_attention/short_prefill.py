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

import math
from functools import lru_cache
from threading import Lock

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.experimental.tle.language.dsa.ascend import (
    PIPE,
    sub_vec_id,
    sync_block_set,
    sync_block_wait,
)
from triton.runtime import driver

from flaggems_vllm.runtime.backend._ascend.ops.flash_attention.submission import (
    _raw_stream_nowait,
    bind_runner,
    cube_core_limit,
)


@triton.jit(do_not_specialize_on_alignment=["K", "V", "TABLE", "USED", "KP", "VP"])
def _pack_short_prefill_kv(
    K, V, TABLE, USED, KP, VP, HK: tl.constexpr, PAGE: tl.constexpr, W: tl.constexpr
):
    _pack_page(
        K,
        V,
        TABLE,
        tl.load(USED),
        KP,
        VP,
        tl.program_id(0),
        tl.program_id(1),
        HK,
        PAGE,
        W,
    )


@triton.jit
def _pack_page(
    K,
    V,
    TABLE,
    used,
    KP,
    VP,
    logical,
    head,
    HK: tl.constexpr,
    PAGE: tl.constexpr,
    W: tl.constexpr,
):
    row = logical * PAGE + tl.arange(0, PAGE)
    d = tl.arange(0, 256)
    physical = tl.load(TABLE + logical, logical * PAGE < used, 0).to(tl.int64)
    src = ((physical * PAGE + tl.arange(0, PAGE)[:, None]) * HK + head) * 256 + d[
        None, :
    ]
    valid = row < used
    k = tl.load(K + src, valid[:, None], 0)
    v = tl.load(V + src, valid[:, None], 0)
    dst = (head * W + row[:, None]) * 256 + d[None, :]
    tl.store(KP + dst, k)
    tl.store(VP + dst, v)


@triton.jit
def _short_softmax(
    ta,
    pa,
    row_norm,
    limit,
    load_qr,
    qlen,
    klen,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    W: tl.constexpr,
    BM: tl.constexpr,
    VM: tl.constexpr,
    WIDTH: tl.constexpr,
    SPLIT_PADDED_WIDTH: tl.constexpr = False,
    FIRST_VALID: tl.constexpr = 0,
):
    with tle.scope(core_mode="vector"):
        vec = sub_vec_id().to(tl.int32)
        SPLIT_END: tl.constexpr = SPLIT_PADDED_WIDTH and WIDTH > 1024 and BN == 256
        col = tl.arange(0, 1024 if SPLIT_END else WIDTH)
        begin = 0
        if FIRST_VALID:
            tl.static_assert(SPLIT_END and W == 1280)
            # The tail's first rows repeat already written prefix rows.
            # PV still reads those rows, so explicitly zero their probabilities.
            begin = tl.maximum(0, FIRST_VALID - vec * (BM // 2)) // VM * VM
            for i in range(0, begin, VM):
                rows = vec * (BM // 2) + i + tl.arange(0, VM)
                offs = rows[:, None] * W
                tl.store(pa + offs + col[None, :], tl.full((VM, 1024), 0, tl.float32))
                tl.store(
                    pa + offs + 1024 + tl.arange(0, 256)[None, :],
                    tl.full((VM, 256), 0, tl.float32),
                )
                tl.store(row_norm + rows, 1.0)
        for i in range(begin, BM // 2, VM):
            rows = vec * (BM // 2) + i + tl.arange(0, VM)
            offsets = rows[:, None] * W + col[None, :]
            val = tl.load(ta + offsets, col[None, :] < limit, 0, volatile=True)
            row_end = tl.minimum(limit, load_qr + rows[:, None] + klen - qlen + 1)
            keep = col[None, :] < row_end
            val = tl.where(keep, val * SCALE, -float("inf"))
            maximum = tl.max(val, 1)
            if SPLIT_END:
                # This short path caps K at 1152. The final Cube tile
                # consumes only columns 1024:1280 of the padded workspace.
                end_col = 1024 + tl.arange(0, 256)
                end_offsets = rows[:, None] * W + end_col[None, :]
                end_values = tl.load(
                    ta + end_offsets, end_col[None, :] < limit, 0, volatile=True
                )
                end_values = tl.where(
                    end_col[None, :] < row_end, end_values * SCALE, -float("inf")
                )
                maximum = tl.maximum(maximum, tl.max(end_values, 1))
            maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
            exp = tl.exp(val - maximum[:, None])
            norm = tl.sum(exp, 1)
            if SPLIT_END:
                end_exp = tl.exp(end_values - maximum[:, None])
                norm += tl.sum(end_exp, 1)
                tl.store(pa + end_offsets, end_exp)
            tl.store(row_norm + rows, tl.where(norm > 0, norm, 1.0))
            tl.store(pa + offsets, exp, col[None, :] < tl.cdiv(limit, BN) * BN)


@triton.jit(
    do_not_specialize_on_alignment=[
        "Q",
        "K",
        "V",
        "O",
        "USED",
        "SCORES",
        "PROBS",
        "PARTIAL",
    ]
)
def _short_prefill(
    Q,
    K,
    V,
    O,
    USED,
    SCORES,
    PROBS,
    PARTIAL,
    SCALE: tl.constexpr,
    QLEN: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    BN: tl.constexpr,
    W: tl.constexpr,
    BM: tl.constexpr,
    VM: tl.constexpr,
    TASKS: tl.constexpr,
):
    worker = tl.program_id(0).to(tl.int64)
    slots = tl.num_programs(0)
    qr_count: tl.constexpr = TASKS // H
    qlen = QLEN
    klen = tl.load(USED)
    ta = SCORES + worker * W * BM
    pa = PROBS + worker * W * BM
    wa = PARTIAL + worker * BM * 257
    row_norm = wa + BM * 256
    dd = tl.arange(0, 256)
    qb = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    kb = tle.dsa.alloc([BN, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    vb = tle.dsa.alloc([BN, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    pb = tle.dsa.alloc([BM, BN], Q.dtype.element_ty, tle.dsa.ascend.L1)
    for ticket in range(worker, TASKS, slots):
        # Pair long and short causal tiles so workers do similar amounts of work.
        head = (ticket % H).to(tl.int32)
        kh = head // (H // HK)
        idx = (ticket // H).to(tl.int32)
        qr = tl.where(idx % 2 == 0, qr_count - 1 - idx // 2, idx // 2) * BM
        limit = tl.maximum(0, tl.minimum(klen, qr + BM + klen - qlen))
        if qr < qlen:
            # Match the scalable pipeline's affine Q reads. A non-full last block
            # reads valid physical rows by sliding its source tile left. Its
            # duplicate rows are ignored and the logical output is still disjoint.
            tail_begin = tl.maximum(0, qlen - BM)
            load_qr = tl.minimum(qr, tail_begin)
            if limit > 0:
                with tle.scope(core_mode="cube"):
                    qp = tl.make_block_ptr(
                        Q + (load_qr.to(tl.int64) * H + head) * 256,
                        (BM, 256),
                        (H * 256, 1),
                        (0, 0),
                        (BM, 256),
                        (1, 0),
                    )
                    tle.dsa.copy(qp, qb, [BM, 256])
                    query = tle.dsa.to_tensor(qb, writable=False)
                    for start in range(0, limit, BN):
                        kp = tl.make_block_ptr(
                            K + (kh * W + start) * 256,
                            (BN, 256),
                            (256, 1),
                            (0, 0),
                            (BN, 256),
                            (1, 0),
                        )
                        tle.dsa.copy(kp, kb, [BN, 256])
                        score = tl.dot(
                            query, tl.trans(tle.dsa.to_tensor(kb, writable=False))
                        )
                        tl.store(
                            ta
                            + tl.arange(0, BM)[:, None] * W
                            + start
                            + tl.arange(0, BN)[None, :],
                            score,
                        )
                sync_block_set("cube", "vector", 12, PIPE.PIPE_FIX, PIPE.PIPE_S)
                sync_block_wait("cube", "vector", 12, PIPE.PIPE_FIX, PIPE.PIPE_S)
                # A causal tile only needs the rounded-up visible KV prefix.
                if limit <= 256:
                    _short_softmax(
                        ta,
                        pa,
                        row_norm,
                        limit,
                        load_qr,
                        qlen,
                        klen,
                        SCALE,
                        BN,
                        W,
                        BM,
                        VM,
                        256,
                    )
                elif limit <= 512:
                    _short_softmax(
                        ta,
                        pa,
                        row_norm,
                        limit,
                        load_qr,
                        qlen,
                        klen,
                        SCALE,
                        BN,
                        W,
                        BM,
                        VM,
                        512,
                    )
                elif limit <= 1024 or W == 1024:
                    _short_softmax(
                        ta,
                        pa,
                        row_norm,
                        limit,
                        load_qr,
                        qlen,
                        klen,
                        SCALE,
                        BN,
                        W,
                        BM,
                        VM,
                        1024,
                    )
                else:
                    _short_softmax(
                        ta,
                        pa,
                        row_norm,
                        limit,
                        load_qr,
                        qlen,
                        klen,
                        SCALE,
                        BN,
                        W,
                        BM,
                        VM,
                        W,
                        H <= 8 and Q.dtype.element_ty == tl.bfloat16,
                    )
                sync_block_set("vector", "cube", 13, PIPE.PIPE_MTE3, PIPE.PIPE_S)
                sync_block_wait("vector", "cube", 13, PIPE.PIPE_MTE3, PIPE.PIPE_S)
                with tle.scope(core_mode="cube"):
                    acc = tl.full((BM, 256), 0.0, tl.float32)
                    for start in range(0, limit, BN):
                        vp = tl.make_block_ptr(
                            V + (kh * W + start) * 256,
                            (BN, 256),
                            (256, 1),
                            (0, 0),
                            (BN, 256),
                            (1, 0),
                        )
                        pp = tl.make_block_ptr(
                            pa + start,
                            (BM, BN),
                            (W, 1),
                            (0, 0),
                            (BM, BN),
                            (1, 0),
                        )
                        tle.dsa.copy(vp, vb, [BN, 256])
                        tle.dsa.copy(pp, pb, [BM, BN])
                        acc = tl.dot(
                            tle.dsa.to_tensor(pb, writable=False),
                            tle.dsa.to_tensor(vb, writable=False),
                            acc,
                        )
                    tl.store(wa + tl.arange(0, BM)[:, None] * 256 + dd[None, :], acc)
                sync_block_set("cube", "vector", 14, PIPE.PIPE_FIX, PIPE.PIPE_S)
                sync_block_wait("cube", "vector", 14, PIPE.PIPE_FIX, PIPE.PIPE_S)
            with tle.scope(core_mode="vector"):
                vec = sub_vec_id().to(tl.int32)
                for i in range(0, BM // 2, VM):
                    rows = vec * (BM // 2) + i + tl.arange(0, VM)
                    result = tl.zeros((VM, 256), tl.float32)
                    if limit > 0:
                        result = tl.load(
                            wa + rows[:, None] * 256 + dd[None, :], volatile=True
                        )
                        result = (
                            result / tl.load(row_norm + rows, volatile=True)[:, None]
                        )
                    qr_abs = load_qr + rows
                    tl.store(
                        O
                        + (qr_abs.to(tl.int64)[:, None] * H + head) * 256
                        + dd[None, :],
                        result,
                        ((qr_abs >= qr) & (qr_abs < qlen))[:, None],
                    )
        sync_block_set("vector", "cube", 15, PIPE.PIPE_MTE3, PIPE.PIPE_S)
        sync_block_wait("vector", "cube", 15, PIPE.PIPE_MTE3, PIPE.PIPE_S)


@triton.jit
def _overlap_tile(
    ticket,
    klen,
    H: tl.constexpr,
    HK: tl.constexpr,
    QLEN: tl.constexpr,
    BM: tl.constexpr,
    BALANCE: tl.constexpr,
):
    nrows: tl.constexpr = triton.cdiv(QLEN, BM)
    # One worker owns a fixed KV head for the whole task loop.
    group: tl.constexpr = H // HK
    owner = tl.program_id(0).to(tl.int32) % HK
    h = (owner * group + ticket % group).to(tl.int32)
    i = (ticket // group).to(tl.int32)
    if BALANCE:
        # With eight heads and ten lanes per KV head, this keeps each lane
        # at 20 panels for the common equal-length short prefill.
        row = ((tl.full((), 0x357684210, tl.int64) >> (4 * i)) & 15).to(tl.int32) * BM
    else:
        row = tl.where(i % 2 == 0, nrows - 1 - i // 2, i // 2) * BM
    return (
        h,
        h // (H // HK),
        row,
        tl.minimum(row, QLEN - BM),
        tl.maximum(0, tl.minimum(klen, row + BM + klen - QLEN)),
    )


@triton.jit
def _overlap_qk(
    Q,
    K,
    ta,
    qb,
    kb,
    first_k,
    head,
    kh,
    qrow,
    limit,
    H: tl.constexpr,
    W: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    qp = tl.make_block_ptr(
        Q + (qrow.to(tl.int64) * H + head) * 256,
        (BM, 256),
        (H * 256, 1),
        (0, 0),
        (BM, 256),
        (1, 0),
    )
    tle.dsa.copy(qp, qb, [BM, 256])
    query = tle.dsa.to_tensor(qb, writable=False)
    first_score = tl.dot(query, tl.trans(tle.dsa.to_tensor(first_k, writable=False)))
    tl.store(
        ta + tl.arange(0, BM)[:, None] * W + tl.arange(0, BN)[None, :],
        first_score,
    )
    for col in range(BN, limit, BN):
        kp = tl.make_block_ptr(
            K + (kh * W + col) * 256,
            (BN, 256),
            (256, 1),
            (0, 0),
            (BN, 256),
            (1, 0),
        )
        tle.dsa.copy(kp, kb, [BN, 256])
        score = tl.dot(query, tl.trans(tle.dsa.to_tensor(kb, writable=False)))
        tl.store(
            ta + tl.arange(0, BM)[:, None] * W + col + tl.arange(0, BN)[None, :], score
        )


@triton.jit
def _overlap_pv(
    V,
    pa,
    wa,
    vb,
    pb,
    first_v,
    kh,
    limit,
    W: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pp0 = tl.make_block_ptr(pa, (BM, BN), (W, 1), (0, 0), (BM, BN), (1, 0))
    tle.dsa.copy(pp0, pb, [BM, BN])
    acc = tl.dot(
        tle.dsa.to_tensor(pb, writable=False),
        tle.dsa.to_tensor(first_v, writable=False),
    )
    for col in range(BN, limit, BN):
        vp = tl.make_block_ptr(
            V + (kh * W + col) * 256,
            (BN, 256),
            (256, 1),
            (0, 0),
            (BN, 256),
            (1, 0),
        )
        pp = tl.make_block_ptr(
            pa + col,
            (BM, BN),
            (W, 1),
            (0, 0),
            (BM, BN),
            (1, 0),
        )
        tle.dsa.copy(vp, vb, [BN, 256])
        tle.dsa.copy(pp, pb, [BM, BN])
        acc = tl.dot(
            tle.dsa.to_tensor(pb, writable=False),
            tle.dsa.to_tensor(vb, writable=False),
            acc,
        )
    tl.store(wa + tl.arange(0, BM)[:, None] * 256 + tl.arange(0, 256)[None, :], acc)


@triton.jit(
    do_not_specialize_on_alignment=[
        "Q",
        "K",
        "V",
        "O",
        "USED",
        "SCORES",
        "PROBS",
        "PARTIAL",
    ]
)
def _short_prefill_overlap(
    Q,
    K,
    V,
    O,
    USED,
    SCORES,
    PROBS,
    PARTIAL,
    SCALE: tl.constexpr,
    QLEN: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    BN: tl.constexpr,
    W: tl.constexpr,
    BM: tl.constexpr,
    VM: tl.constexpr,
    TASKS: tl.constexpr,
    WORKERS: tl.constexpr = 0,
):
    DEFER_RELEASE: tl.constexpr = 1024 < QLEN and QLEN <= 1040
    BALANCE: tl.constexpr = (
        DEFER_RELEASE and BM == 128 and H == 16 and HK == 2 and WORKERS == 20
    )
    # Keep two work slots per worker. Cube can produce the next QK scores
    # while Vector consumes the previous scores, before Cube consumes those
    # probabilities. Vector releases a slot only after writing its output.
    qb = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    kb = tle.dsa.alloc([BN, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    first_k = tle.dsa.alloc([BN, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    pb = tle.dsa.alloc([BM, BN], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    first_v = tle.dsa.alloc([BN, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    with tle.scope(core_mode="cube"):
        worker = tl.program_id(0).to(tl.int64)
        workers = tl.num_programs(0)
        klen = tl.load(USED)
        owner = worker % HK
        group_worker = worker // HK
        group_workers = workers // HK
        n = tl.cdiv(TASKS // HK - group_worker, group_workers)
        if klen > 0:
            vp0 = tl.make_block_ptr(
                V + owner * W * 256, (BN, 256), (256, 1), (0, 0), (BN, 256), (1, 0)
            )
            kp0 = tl.make_block_ptr(
                K + owner * W * 256, (BN, 256), (256, 1), (0, 0), (BN, 256), (1, 0)
            )
            tle.dsa.copy(vp0, first_v, [BN, 256])
            tle.dsa.copy(kp0, first_k, [BN, 256])
        for tick in range(n + 1):
            slot = tick % 2
            prev = 1 - slot
            if tick < n:
                if not DEFER_RELEASE and tick >= 2:
                    sync_block_wait(
                        "vector", "cube", 9 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                    )
                head, kh, _, qrow, limit = _overlap_tile(
                    group_worker + group_workers * tick, klen, H, HK, QLEN, BM, BALANCE
                )
                if limit > 0:
                    _overlap_qk(
                        Q,
                        K,
                        SCORES + (worker * 2 + slot) * W * BM,
                        qb,
                        kb,
                        first_k,
                        head,
                        kh,
                        qrow,
                        limit,
                        H,
                        W,
                        BM,
                        BN,
                    )
                sync_block_set("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            if tick > 0:
                _, kh0, _, _, limit0 = _overlap_tile(
                    group_worker + group_workers * (tick - 1),
                    klen,
                    H,
                    HK,
                    QLEN,
                    BM,
                    BALANCE,
                )
                sync_block_wait("vector", "cube", 3 + prev, PIPE.PIPE_MTE3, PIPE.PIPE_S)
                if DEFER_RELEASE and tick >= 3:
                    # Score and P were retired by the previous PV. Keep the O
                    # slot protected until the next PV is ready to overwrite it.
                    sync_block_wait(
                        "vector", "cube", 9 + prev, PIPE.PIPE_MTE3, PIPE.PIPE_S
                    )
                if limit0 > 0:
                    _overlap_pv(
                        V,
                        PROBS + (worker * 2 + prev) * W * BM,
                        PARTIAL + (worker * 2 + prev) * BM * 257,
                        kb,
                        pb,
                        first_v,
                        kh0,
                        limit0,
                        W,
                        BM,
                        BN,
                    )
                sync_block_set("cube", "vector", 6 + prev, PIPE.PIPE_FIX, PIPE.PIPE_S)
    with tle.scope(core_mode="vector"):
        worker = tl.program_id(0).to(tl.int64)
        workers = tl.num_programs(0)
        vec = sub_vec_id().to(tl.int32)
        klen = tl.load(USED)
        group_worker = worker // HK
        group_workers = workers // HK
        n = tl.cdiv(TASKS // HK - group_worker, group_workers)
        for tick in range(n + 1):
            slot = tick % 2
            prev = 1 - slot
            if tick < n:
                _, _, logical_row, qrow, limit = _overlap_tile(
                    group_worker + group_workers * tick, klen, H, HK, QLEN, BM, BALANCE
                )
                ta = SCORES + (worker * 2 + slot) * W * BM
                pa = PROBS + (worker * 2 + slot) * W * BM
                norm = PARTIAL + (worker * 2 + slot) * BM * 257 + BM * 256
                sync_block_wait("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
                if limit > 0:
                    if limit <= 256:
                        _short_softmax(
                            ta,
                            pa,
                            norm,
                            limit,
                            qrow,
                            QLEN,
                            klen,
                            SCALE,
                            BN,
                            W,
                            BM,
                            VM,
                            256,
                        )
                    elif limit <= 512:
                        _short_softmax(
                            ta,
                            pa,
                            norm,
                            limit,
                            qrow,
                            QLEN,
                            klen,
                            SCALE,
                            BN,
                            W,
                            BM,
                            VM,
                            512,
                        )
                    elif limit <= 1024 or W == 1024:
                        _short_softmax(
                            ta,
                            pa,
                            norm,
                            limit,
                            qrow,
                            QLEN,
                            klen,
                            SCALE,
                            BN,
                            W,
                            BM,
                            VM,
                            1024,
                        )
                    elif (
                        H == 16
                        and HK == 2
                        and (QLEN > 1024 and QLEN <= 1040)
                        and QLEN % BM != 0
                        and BN == 256
                        and W == 1280
                        and logical_row > qrow
                    ):
                        _short_softmax(
                            ta,
                            pa,
                            norm,
                            limit,
                            qrow,
                            QLEN,
                            klen,
                            SCALE,
                            BN,
                            W,
                            BM,
                            VM,
                            W,
                            True,
                            BM - QLEN % BM,
                        )
                    else:
                        _short_softmax(
                            ta,
                            pa,
                            norm,
                            limit,
                            qrow,
                            QLEN,
                            klen,
                            SCALE,
                            BN,
                            W,
                            BM,
                            VM,
                            W,
                            W == 1280,
                        )
                sync_block_set("vector", "cube", 3 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
            if tick > 0:
                h0, _, row0, start0, limit0 = _overlap_tile(
                    group_worker + group_workers * (tick - 1),
                    klen,
                    H,
                    HK,
                    QLEN,
                    BM,
                    BALANCE,
                )
                wa = PARTIAL + (worker * 2 + prev) * BM * 257
                sync_block_wait("cube", "vector", 6 + prev, PIPE.PIPE_FIX, PIPE.PIPE_S)
                begin = 0
                if (
                    H == 16
                    and HK == 2
                    and (QLEN > 1024 and QLEN <= 1040)
                    and QLEN % BM != 0
                ):
                    begin = tl.maximum(0, row0 - start0 - vec * (BM // 2)) // VM * VM
                for r in range(begin, BM // 2, VM):
                    rows = vec * (BM // 2) + r + tl.arange(0, VM)
                    d = tl.arange(0, 256)
                    output = tl.zeros((VM, 256), tl.float32)
                    if limit0 > 0:
                        output = tl.load(
                            wa + rows[:, None] * 256 + d[None, :], volatile=True
                        )
                        output = (
                            output
                            / tl.load(wa + BM * 256 + rows, volatile=True)[:, None]
                        )
                    row = start0 + rows
                    tl.store(
                        O + (row[:, None].to(tl.int64) * H + h0) * 256 + d[None, :],
                        output,
                        ((row >= row0) & (row < QLEN))[:, None],
                    )
                if tick + 1 < n:
                    sync_block_set(
                        "vector", "cube", 9 + prev, PIPE.PIPE_MTE3, PIPE.PIPE_S
                    )


@triton.jit(
    do_not_specialize_on_alignment=[
        "Q",
        "K",
        "V",
        "O",
        "USED",
        "TABLE",
        "WORK",
        "KV",
        "PROBS",
    ]
)
def _short_prefill_fused(
    Q,
    K,
    V,
    O,
    USED,
    TABLE,
    WORK,
    KV,
    PROBS,
    SCALE: tl.constexpr,
    QLEN: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    BN: tl.constexpr,
    W: tl.constexpr,
    BM: tl.constexpr,
    VM: tl.constexpr,
    TASKS: tl.constexpr,
    PAGE: tl.constexpr,
    WORKERS: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    packed_v = KV + HK * W * 256
    with tle.scope(core_mode="vector"):
        lane = tl.program_id(0) * 2 + sub_vec_id()
        used = tl.load(USED)
        pages = tl.cdiv(used, BN) * (BN // PAGE)
        for page_task in range(lane, pages * HK, WORKERS * 2):
            _pack_page(
                K,
                V,
                TABLE,
                used,
                KV,
                packed_v,
                page_task // HK,
                page_task % HK,
                HK,
                PAGE,
                W,
            )
    tle.dsa.ascend.sync_block_all("all", 15 if OVERLAP else 0)
    if OVERLAP:
        _short_prefill_overlap(
            Q,
            KV,
            packed_v,
            O,
            USED,
            WORK,
            PROBS,
            WORK + 2 * WORKERS * BM * W,
            SCALE,
            QLEN,
            H,
            HK,
            BN,
            W,
            BM,
            VM,
            TASKS,
            WORKERS,
        )
    else:
        _short_prefill(
            Q,
            KV,
            packed_v,
            O,
            USED,
            WORK,
            PROBS,
            WORK + WORKERS * BM * W,
            SCALE,
            QLEN,
            H,
            HK,
            BN,
            W,
            BM,
            VM,
            TASKS,
        )


_short_prefill_launchers = {}
_fused_prefill_launchers = {}
_workspace = {}
_workspace_lock = Lock()
_MAX_WORKSPACE_BYTES = 64 << 20


def _make_workspace(device, dtype, required, stream):
    key = None
    if stream is not None and not torch.npu.is_current_stream_capturing():
        key = (device.index, stream, dtype)
        saved = _workspace.get(key)
        if (
            saved is not None
            and saved[0][0] >= required[0]
            and saved[0][1] >= required[1]
            and saved[0][2] >= required[2]
        ):
            return saved[2]
    scratch = (
        torch.empty(required[0], dtype=torch.float32, device=device),
        torch.empty(required[1], dtype=dtype, device=device),
        torch.empty(required[2], dtype=dtype, device=device),
    )
    if key is not None:
        size = 4 * required[0] + 2 * (required[1] + required[2])
        with _workspace_lock:
            total = sum(
                saved_bytes
                for other_key, (_, saved_bytes, _) in _workspace.items()
                if other_key != key
            )
            if size + total <= _MAX_WORKSPACE_BYTES:
                _workspace[key] = (required, size, scratch)
    return scratch


@lru_cache(maxsize=128)
def _launch_plan(dtype, max_q, max_k, h, hk, page, scale, core_limit):
    # Wide-head workloads have enough Q tiles to use a larger row block.
    bm, bn, vm = (128 if h >= 16 else 64), 256, 4
    compact_width = (
        dtype == torch.bfloat16 and (h, hk) in ((8, 1), (16, 2)) and max_k > 1024
    )
    if compact_width:
        vm = 8
        w = ((max_k + bn - 1) // bn) * bn
    else:
        w = 1 << (max_k - 1).bit_length()
    tasks = ((max_q + bm - 1) // bm) * h
    workers = min(tasks, core_limit)
    overlap = dtype == torch.bfloat16 and h == 16 and hk == 2 and tasks >= 2 * workers
    if overlap:
        workers -= workers % hk
        overlap = workers > 0
        workers = workers if overlap else 1
    slots = 2 if overlap else 1
    kernel = _short_prefill_overlap if overlap else _short_prefill
    score_size = slots * workers * bm * w
    partial_size = slots * workers * bm * 257
    kv_size = hk * w * 256
    compute_grid = (workers, 1, 1)
    fused = dtype == torch.bfloat16 and (overlap or h <= 8) and workers <= 32
    requirements = (score_size + partial_size, 2 * kv_size, score_size)
    scalars = (scale, max_q, h, hk, bn, w, bm, vm, tasks, page, workers, overlap)
    return fused, compute_grid, kernel, requirements, partial_size, scalars


def short_prefill_output(q, k, v, used, table, max_q, max_k, scale, device):
    """Compute small causal prefill, with an optional fused KV pack."""
    if (
        q.ndim != 3
        or k.ndim != 4
        or k.shape != v.shape
        or q.shape[0] != max_q
        or q.shape[-1] != 256
        or k.shape[-1] != 256
        or k.shape[1] not in (16, 32)
        or not k.shape[2]
        or q.shape[1] % k.shape[2]
        or q.shape[1] // k.shape[2] not in (4, 8)
        or not 512 <= max_q <= 1152
        or not max_q <= max_k <= 1152
        or table.ndim != 2
        or table.shape[0] != 1
        or table.shape[1] < (max_k + k.shape[1] - 1) // k.shape[1]
        or used.shape != (1,)
        or q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous())
        or q.data_ptr() % 32
    ):
        return None
    if scale is None:
        scale = 0.0625
    elif not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
        return None

    _, h, _ = q.shape
    _, page, hk, _ = k.shape
    index = device.index
    fused, compute_grid, kernel, required, partial_size, scalars = _launch_plan(
        q.dtype, max_q, max_k, h, hk, page, scale, cube_core_limit(index)
    )
    w, workers = scalars[5], scalars[10]
    key = (index, q.dtype, max_q, w, page, h, hk, scale, workers)
    out = torch.empty_like(q)
    if fused:
        active = torch.npu.current_device() == index
        get_stream = _raw_stream_nowait or driver.active.get_current_stream
        stream = get_stream(index) if active else None
        scratch, packed, p = _make_workspace(device, q.dtype, required, stream)
        args = (q, k, v, out, used, table, scratch, packed, p) + scalars
        runner = _fused_prefill_launchers.get(key)
        if runner is not None and active:
            runner(args)
        else:
            with torch.npu.device(device):
                entry = _short_prefill_fused[compute_grid](
                    *args,
                    enable_fp_fusion=False,
                    disable_auto_inject_block_sync=True,
                    multibuffer=False,
                )
            if len(_fused_prefill_launchers) < 128 and hasattr(entry, "__getitem__"):
                _fused_prefill_launchers[key] = bind_runner(entry, compute_grid, index)
        return out
    score_size = required[2]
    kv_size = required[1] // 2
    pack_grid = (w // page, hk, 1)
    compiled = _short_prefill_launchers.get(key)
    scratch = torch.empty(
        (score_size + partial_size + score_size // 2,),
        dtype=torch.float32,
        device=device,
    )
    packed = torch.empty((2 * kv_size,), dtype=q.dtype, device=device)
    scores = scratch[:score_size]
    partial = scratch[score_size : score_size + partial_size]
    probs = scratch[score_size + partial_size :].view(q.dtype)
    kp, vp = packed[:kv_size], packed[kv_size:]
    pack_args = (k, v, table, used, kp, vp, hk, page, w)
    compute_args = (
        q,
        kp,
        vp,
        out,
        used,
        scores,
        probs,
        partial,
    ) + scalars[:9]
    if compiled is not None and torch.npu.current_device() == index:
        stream = driver.active.get_current_stream(index)
        pack, compute = compiled
        pack(*pack_args, stream=stream)
        compute(*compute_args, stream=stream)
    else:
        with torch.npu.device(device):
            pack = _pack_short_prefill_kv[pack_grid](*pack_args)
            compute = kernel[compute_grid](
                *compute_args,
                enable_fp_fusion=False,
                disable_auto_inject_block_sync=True,
                multibuffer=False,
            )
        if (
            len(_short_prefill_launchers) < 128
            and hasattr(pack, "__getitem__")
            and hasattr(compute, "__getitem__")
        ):
            _short_prefill_launchers[key] = (
                pack[pack_grid],
                compute[compute_grid],
            )
    return out
