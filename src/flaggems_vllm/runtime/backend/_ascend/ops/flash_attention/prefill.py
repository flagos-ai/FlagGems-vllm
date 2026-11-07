# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.experimental.tle.language.dsa.ascend import (
    PIPE,
    sub_vec_id,
    sync_block_set,
    sync_block_wait,
)

from .direct import (
    _attention_request_stream,
    _softmax_rows_dispatch,
    _softmax_stack,
    _whole_task_finish,
    _whole_task_pv,
    _whole_task_qk,
    _whole_task_softmax,
    ascend_attention_persistent_pipelined,
)

# ---- decoupled_prefill ----


@triton.jit
def _core_init_state(
    state_buf,
    row_end_buf,
    qt,
    qs,
    headstart,
    qlen,
    klen,
    group: tl.constexpr,
    BM: tl.constexpr,
):
    row_rr = sub_vec_id().to(tl.int32) * (BM // 2) + tl.arange(0, BM // 2)
    row_q = qt * 128 + row_rr
    row_valid = (row_rr < qs) & (headstart < group) & (row_q < qlen)
    row_end = tl.where(row_valid, row_q + klen - qlen + 1, 0)
    tle.dsa.to_buffer(row_end, bind_buffer=row_end_buf)
    init = tl.where(tl.arange(0, 5)[:, None] == 0, -float("inf"), 0.0)
    tle.dsa.to_buffer(tl.broadcast_to(init, (5, BM // 2)), bind_buffer=state_buf)


# ---- decoupled_prefill ----


@triton.jit
def _core_init_o(
    acc_buf,
    ACCWORK,
    worker,
    BM: tl.constexpr,
    ACC_UB_ROWS: tl.constexpr,
    COMPACT_O: tl.constexpr = False,
):
    for init_vs in range(0, ACC_UB_ROWS, 8):
        init_acc = tle.dsa.subview(acc_buf, [init_vs, 0], [8, 256], [1, 1])
        if COMPACT_O:
            tle.dsa.to_buffer(tl.zeros((8, 256), tl.bfloat16), bind_buffer=init_acc)
        else:
            tle.dsa.to_buffer(tl.zeros((8, 256), tl.float32), bind_buffer=init_acc)
    for init_vs in range(ACC_UB_ROWS, BM // 2, 8):
        init_rows = sub_vec_id().to(tl.int32) * (BM // 2) + init_vs + tl.arange(0, 8)
        tl.store(
            ACCWORK
            + worker * 128 * 256
            + init_rows[:, None] * 256
            + tl.arange(0, 256)[None, :],
            tl.zeros((8, 256), tl.float32),
        )


# ---- decoupled_prefill ----


@triton.jit
def _core_load_q(
    Q,
    Q_RAW,
    query_buf,
    batch,
    qbase,
    qlen,
    task,
    qt,
    kh,
    headstart,
    group: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    QCAP: tl.constexpr,
    Q_DIRECT: tl.constexpr,
    BM: tl.constexpr,
):
    if Q_DIRECT:
        direct = qt * 128 + 128 <= qlen
        if direct:
            raw_head = kh * group + headstart
            raw_base = (
                Q_RAW + qbase + qt.to(tl.int64) * 128 * QR + raw_head.to(tl.int64) * QH
            )
            query_ptr = tl.make_block_ptr(
                raw_base, (128, 256), (QR, 1), (0, 0), (128, 256), (1, 0)
            )
            tle.dsa.copy(query_ptr, query_buf, [128, 256])
        else:
            query_ptr = tl.make_block_ptr(
                Q + (batch.to(tl.int64) * QCAP + task) * 128 * 256,
                (BM, 256),
                (256, 1),
                (0, 0),
                (BM, 256),
                (1, 0),
            )
            tle.dsa.copy(query_ptr, query_buf, [BM, 256])
    else:
        query_ptr = tl.make_block_ptr(
            Q + (batch.to(tl.int64) * QCAP + task) * 128 * 256,
            (BM, 256),
            (256, 1),
            (0, 0),
            (BM, 256),
            (1, 0),
        )
        tle.dsa.copy(query_ptr, query_buf, [BM, 256])


# ---- decoupled_prefill ----


@triton.jit
def _core_qk_only(
    K,
    scratch,
    prob,
    query_buf,
    key_buf_0,
    key_buf_1,
    state_buf,
    row_end_buf,
    batch,
    kh,
    begin,
    stack,
    slot,
    qs,
    qt,
    headstart,
    qlen,
    klen,
    HK: tl.constexpr,
    group: tl.constexpr,
    CONTIG_K: tl.constexpr,
    BM: tl.constexpr,
    BS: tl.constexpr,
    SCALE: tl.constexpr,
    SINGLE_HEAD_INDEX: tl.constexpr,
):
    q1 = tle.dsa.to_tensor(query_buf, writable=False)
    pair_end = (stack + 127) // 256 * 256
    for ck in range(0, pair_end, 256):
        kbase = K + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + ck) * 256
        kptr_0 = tl.make_block_ptr(
            kbase, (128, 256), (256, 1), (0, 0), (128, 256), (1, 0)
        )
        kptr_1 = tl.make_block_ptr(
            kbase + 128 * 256, (128, 256), (256, 1), (0, 0), (128, 256), (1, 0)
        )
        tle.dsa.copy(kptr_0, key_buf_0, [128, 256])
        tle.dsa.copy(kptr_1, key_buf_1, [128, 256])
        key_0 = tle.dsa.to_tensor(key_buf_0, writable=False)
        key_1 = tle.dsa.to_tensor(key_buf_1, writable=False)
        qk_0 = tl.dot(q1, tl.trans(key_0))
        qk_1 = tl.dot(q1, tl.trans(key_1))
        soff_0 = tl.arange(0, BM)[:, None] * BS + ck + tl.arange(0, 128)[None, :]
        tl.store(scratch + soff_0, qk_0)
        tl.store(scratch + soff_0 + 128, qk_1)
    if pair_end < stack:
        kbase = K + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + pair_end) * 256
        kptr_0 = tl.make_block_ptr(
            kbase, (128, 256), (256, 1), (0, 0), (128, 256), (1, 0)
        )
        tle.dsa.copy(kptr_0, key_buf_0, [128, 256])
        key_0 = tle.dsa.to_tensor(key_buf_0, writable=False)
        qk_0 = tl.dot(q1, tl.trans(key_0))
        soff_0 = tl.arange(0, BM)[:, None] * BS + pair_end + tl.arange(0, 128)[None, :]
        tl.store(scratch + soff_0, qk_0)
    sync_block_set("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)


# ---- decoupled_prefill ----


@triton.jit
def _core_softmax_only(
    K,
    scratch,
    prob,
    query_buf,
    key_buf_0,
    key_buf_1,
    state_buf,
    row_end_buf,
    batch,
    kh,
    begin,
    stack,
    slot,
    qs,
    qt,
    headstart,
    qlen,
    klen,
    HK: tl.constexpr,
    group: tl.constexpr,
    CONTIG_K: tl.constexpr,
    BM: tl.constexpr,
    BS: tl.constexpr,
    SCALE: tl.constexpr,
    SINGLE_HEAD_INDEX: tl.constexpr,
):
    sub = 0
    head_rows = 1
    CAUSAL: tl.constexpr = True
    VM: tl.constexpr = 16
    SOFTMAX_VM_256: tl.constexpr = 32
    sync_block_wait("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
    x2_vf = sub_vec_id().to(tl.int32) * (BM // 2)
    if stack <= 256:
        if CONTIG_K >= 2048 and BM == 128:
            _softmax_stack(
                scratch,
                prob,
                state_buf,
                begin,
                stack,
                slot,
                sub,
                qs,
                qt,
                head_rows,
                headstart,
                group,
                qlen,
                klen,
                x2_vf,
                BM,
                BS,
                256,
                CAUSAL,
                SCALE,
                SOFTMAX_VM_256,
                SINGLE_HEAD_INDEX,
                row_end_buf,
            )
        else:
            _softmax_rows_dispatch(
                scratch,
                prob,
                state_buf,
                begin,
                stack,
                slot,
                sub,
                qs,
                qt,
                head_rows,
                headstart,
                group,
                qlen,
                klen,
                x2_vf,
                BM,
                BS,
                256,
                CAUSAL,
                SCALE,
                SOFTMAX_VM_256,
                SINGLE_HEAD_INDEX,
                row_end_buf,
            )
    elif CONTIG_K >= 2048 and BM == 128:
        _softmax_stack(
            scratch,
            prob,
            state_buf,
            begin,
            stack,
            slot,
            sub,
            qs,
            qt,
            head_rows,
            headstart,
            group,
            qlen,
            klen,
            x2_vf,
            BM,
            BS,
            BS,
            CAUSAL,
            SCALE,
            VM,
            SINGLE_HEAD_INDEX,
            row_end_buf,
        )
    else:
        _softmax_rows_dispatch(
            scratch,
            prob,
            state_buf,
            begin,
            stack,
            slot,
            sub,
            qs,
            qt,
            head_rows,
            headstart,
            group,
            qlen,
            klen,
            x2_vf,
            BM,
            BS,
            BS,
            CAUSAL,
            SCALE,
            VM,
            SINGLE_HEAD_INDEX,
            row_end_buf,
        )
    sync_block_set("vector", "cube", 3 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)


# ---- decoupled_prefill ----


@triton.jit
def _core_finish(
    O,
    LSE,
    ACCWORK,
    state_buf,
    acc_buf,
    worker,
    qt,
    qs,
    headstart,
    kh,
    qlen,
    qstart,
    obase,
    H: tl.constexpr,
    group: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    BM: tl.constexpr,
    ACC_UB_ROWS: tl.constexpr,
):
    VM: tl.constexpr = 16
    x5_vf = sub_vec_id().to(tl.int32) * (BM // 2)
    for x5_region in tl.static_range(0, 2):
        if x5_region == 0:
            X5_BEGIN = 0
            X5_END = ACC_UB_ROWS
        else:
            X5_BEGIN = ACC_UB_ROWS
            X5_END = BM // 2
        for x5_vs in range(X5_BEGIN, X5_END, VM):
            x5_mv = tle.dsa.subview(state_buf, [0, x5_vs], [1, VM], [1, 1])
            x5_lv = tle.dsa.subview(state_buf, [1, x5_vs], [1, VM], [1, 1])
            x5_ab = (
                ACCWORK
                + worker * 128 * 256
                + (x5_vf + x5_vs + tl.arange(0, VM))[:, None] * 256
                + tl.arange(0, 256)[None, :]
            )
            x5_m = tl.reshape(tle.dsa.to_tensor(x5_mv, writable=False), (VM,))
            x5_l = tl.reshape(tle.dsa.to_tensor(x5_lv, writable=False), (VM,))
            if x5_region == 0 and ACC_UB_ROWS > 0:
                x5_accview = tle.dsa.subview(acc_buf, [x5_vs, 0], [VM, 256], [1, 1])
                x5_acc = tle.dsa.to_tensor(x5_accview, writable=False).to(tl.float32)
            else:
                x5_acc = tl.load(x5_ab, volatile=True)
            x5_safe = tl.where(x5_l > 0, x5_l, 1.0)
            x5_result = x5_acc / x5_safe[:, None]
            x5_lse = tl.where(x5_l > 0, x5_m + tl.log(x5_safe), -float("inf"))
            store_rows = qt * 128 + x5_vf + x5_vs + tl.arange(0, VM)
            store_head = kh * group + headstart
            store_mask = (x5_vf + x5_vs + tl.arange(0, VM) < qs) & (store_rows < qlen)
            tl.store(
                O
                + obase
                + store_rows.to(tl.int64)[:, None] * OR
                + store_head * OH
                + tl.arange(0, 256)[None, :],
                x5_result,
                store_mask[:, None],
            )
            tl.store(
                LSE + store_head.to(tl.int64) * TOTAL_Q + qstart + store_rows,
                x5_lse,
                store_mask,
            )


# ---- decoupled_prefill ----


@triton.jit
def _core_pv(
    V,
    PROBS,
    PVWORK,
    ACCWORK,
    value_buf,
    prob_buf,
    state_buf,
    acc_buf,
    batch,
    kh,
    worker,
    previous,
    prevslot,
    begin,
    stack,
    HK: tl.constexpr,
    CONTIG_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    ACC_UB_ROWS: tl.constexpr,
    COMPACT_O: tl.constexpr = False,
):
    prob = PROBS + (worker * 3 + prevslot) * 128 * BS
    pvwork = PVWORK + (worker * 3 + prevslot) * 128 * 256
    _: tl.constexpr = 16 if ACC_UB_ROWS == 16 else 32
    first_vptr = tl.make_block_ptr(
        V + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin) * 256,
        (BN, 256),
        (256, 1),
        (0, 0),
        (BN, 256),
        (1, 0),
    )
    tle.dsa.copy(first_vptr, value_buf, [BN, 256])
    sync_block_wait("vector", "cube", 3 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    c2 = tl.zeros((BM, 256), tl.float32)
    for cv in range(0, stack, BN):
        if cv > 0:
            vptr = tl.make_block_ptr(
                V + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + cv) * 256,
                (BN, 256),
                (256, 1),
                (0, 0),
                (BN, 256),
                (1, 0),
            )
            tle.dsa.copy(vptr, value_buf, [BN, 256])
        pptr = tl.make_block_ptr(prob + cv, (BM, BS), (BS, 1), (0, 0), (BM, BN), (1, 0))
        tle.dsa.copy(pptr, prob_buf, [BM, BN])
        value2 = tle.dsa.to_tensor(value_buf, writable=False)
        weights2 = tle.dsa.to_tensor(prob_buf, writable=False)
        c2 = tl.dot(weights2, value2, c2)
    tl.store(pvwork + tl.arange(0, BM)[:, None] * 256 + tl.arange(0, 256)[None, :], c2)
    sync_block_set("cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S)


# ---- decoupled_prefill ----


@triton.jit
def _core_o(
    V,
    PROBS,
    PVWORK,
    ACCWORK,
    value_buf,
    prob_buf,
    state_buf,
    acc_buf,
    batch,
    kh,
    worker,
    previous,
    prevslot,
    begin,
    stack,
    HK: tl.constexpr,
    CONTIG_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    ACC_UB_ROWS: tl.constexpr,
    COMPACT_O: tl.constexpr = False,
):
    _ = PROBS + (worker * 3 + prevslot) * 128 * BS
    pvwork = PVWORK + (worker * 3 + prevslot) * 128 * 256
    O_VM: tl.constexpr = 16 if ACC_UB_ROWS == 16 else 32
    sync_block_wait("cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S)
    x4_vf = sub_vec_id().to(tl.int32) * (BM // 2)
    if ACC_UB_ROWS > 0:
        for x4_vs in range(0, ACC_UB_ROWS, O_VM):
            x4_av = tle.dsa.subview(state_buf, [2 + prevslot, x4_vs], [1, O_VM], [1, 1])
            x4_acc = tle.dsa.subview(acc_buf, [x4_vs, 0], [O_VM, 256], [1, 1])
            x4_alpha = tl.reshape(tle.dsa.to_tensor(x4_av, writable=False), (O_VM,))
            x4_old = tle.dsa.to_tensor(x4_acc, writable=False).to(tl.float32)
            x4_pv = tl.load(
                pvwork
                + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
                + tl.arange(0, 256)[None, :],
                volatile=True,
            )
            if COMPACT_O:
                tle.dsa.to_buffer(
                    (x4_old * x4_alpha[:, None] + x4_pv).to(tl.bfloat16),
                    bind_buffer=x4_acc,
                )
            else:
                tle.dsa.to_buffer(
                    x4_old * x4_alpha[:, None] + x4_pv, bind_buffer=x4_acc
                )
    for x4_vs in range(ACC_UB_ROWS, BM // 2, O_VM):
        x4_av = tle.dsa.subview(state_buf, [2 + prevslot, x4_vs], [1, O_VM], [1, 1])
        x4_ab = (
            ACCWORK
            + worker * 128 * 256
            + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
            + tl.arange(0, 256)[None, :]
        )
        x4_alpha = tl.reshape(tle.dsa.to_tensor(x4_av, writable=False), (O_VM,))
        x4_old = tl.load(x4_ab, volatile=True)
        x4_pv = tl.load(
            pvwork
            + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
            + tl.arange(0, 256)[None, :],
            volatile=True,
        )
        tl.store(x4_ab, x4_old * x4_alpha[:, None] + x4_pv)
    sync_block_set("vector", "cube", 9 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S)


# ---- decoupled_prefill ----


@triton.jit
def ascend_attention_prefill_coreloops(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    CONTIG_K: tl.constexpr,
    QCAP: tl.constexpr,
    Q_DIRECT: tl.constexpr,
):
    BM: tl.constexpr = 128
    BN: tl.constexpr = 256
    BS: tl.constexpr = 512
    group: tl.constexpr = H // HK
    query_buf = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([BN, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([BM, BN], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    state_buf = tle.dsa.alloc([5, BM // 2], tl.float32, tle.dsa.ascend.UB)
    row_end_buf = tle.dsa.alloc([BM // 2], tl.int32, tle.dsa.ascend.UB)
    acc_buf = tle.dsa.alloc([64, 256], tl.bfloat16, tle.dsa.ascend.UB)
    # Every scalar loop index and length is produced independently on each core.
    with tle.scope(core_mode="cube"):
        worker = tl.program_id(0)
        workers = tl.num_programs(0)
        qstart = tl.load(CUQ)
        qlen = tl.load(CUQ + 1) - qstart
        klen = tl.load(USED)
        ntiles = tl.cdiv(qlen, 128) * H
        batch = tl.full((), 0, tl.int32)
        generation = 0
        for task in range(worker, ntiles, workers):
            qt = task // H
            kh = task % H // group
            headstart = task % group
            qs = tl.maximum(1, tl.minimum(qlen, 128))
            limit = tl.minimum(
                klen, tl.maximum(0, tl.minimum(qlen, qt * 128 + 128) + klen - qlen)
            )
            stacks = tl.cdiv(limit, BS)
            _core_load_q(
                Q,
                Q_RAW,
                query_buf,
                batch,
                qstart.to(tl.int64) * QR,
                qlen,
                task,
                qt,
                kh,
                headstart,
                group,
                QR,
                QH,
                QCAP,
                Q_DIRECT,
                BM,
            )
            for step in range(0, stacks + 2):
                if step < stacks:
                    slot = (generation + step) % 3
                    begin = step * BS
                    stack = tl.minimum(BS, limit - begin)
                    scratch = SCORES + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                    prob = PROBS + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                    _core_qk_only(
                        K,
                        scratch,
                        prob,
                        query_buf,
                        key_buf_0,
                        key_buf_1,
                        state_buf,
                        row_end_buf,
                        batch,
                        kh,
                        begin,
                        stack,
                        slot,
                        qs,
                        qt,
                        headstart,
                        qlen,
                        klen,
                        HK,
                        group,
                        CONTIG_K,
                        BM,
                        BS,
                        SCALE,
                        True,
                    )
                if step >= 2:
                    previous = step - 2
                    prevslot = (generation + previous) % 3
                    # Lifetime spans adjacent Q tasks: wait only before slot reuse.
                    if generation + previous >= 3:
                        sync_block_wait(
                            "vector", "cube", 9 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                        )
                    begin = previous * BS
                    stack = tl.minimum(BS, limit - begin)
                    _core_pv(
                        V,
                        PROBS,
                        PVWORK,
                        ACCWORK,
                        value_buf,
                        prob_buf,
                        state_buf,
                        acc_buf,
                        batch,
                        kh,
                        worker.to(tl.int64),
                        previous,
                        prevslot,
                        begin,
                        stack,
                        HK,
                        CONTIG_K,
                        BM,
                        BN,
                        BS,
                        64,
                        True,
                    )
            generation += stacks
        for drain in range(tl.minimum(generation, 3)):
            slot = (generation - tl.minimum(generation, 3) + drain) % 3
            sync_block_wait("vector", "cube", 9 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    with tle.scope(core_mode="vector"):
        worker = tl.program_id(0)
        workers = tl.num_programs(0)
        qstart = tl.load(CUQ)
        qlen = tl.load(CUQ + 1) - qstart
        klen = tl.load(USED)
        ntiles = tl.cdiv(qlen, 128) * H
        batch = tl.full((), 0, tl.int32)
        generation = 0
        for task in range(worker, ntiles, workers):
            qt = task // H
            kh = task % H // group
            headstart = task % group
            qs = tl.maximum(1, tl.minimum(qlen, 128))
            limit = tl.minimum(
                klen, tl.maximum(0, tl.minimum(qlen, qt * 128 + 128) + klen - qlen)
            )
            stacks = tl.cdiv(limit, BS)
            _core_init_o(acc_buf, ACCWORK, worker.to(tl.int64), BM, 64, True)
            _core_init_state(
                state_buf, row_end_buf, qt, qs, headstart, qlen, klen, group, BM
            )
            for step in range(0, stacks + 2):
                if step < stacks:
                    slot = (generation + step) % 3
                    begin = step * BS
                    stack = tl.minimum(BS, limit - begin)
                    scratch = SCORES + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                    prob = PROBS + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                    _core_softmax_only(
                        K,
                        scratch,
                        prob,
                        query_buf,
                        key_buf_0,
                        key_buf_1,
                        state_buf,
                        row_end_buf,
                        batch,
                        kh,
                        begin,
                        stack,
                        slot,
                        qs,
                        qt,
                        headstart,
                        qlen,
                        klen,
                        HK,
                        group,
                        CONTIG_K,
                        BM,
                        BS,
                        SCALE,
                        True,
                    )
                if step >= 2:
                    previous = step - 2
                    prevslot = (generation + previous) % 3
                    begin = previous * BS
                    stack = tl.minimum(BS, limit - begin)
                    _core_o(
                        V,
                        PROBS,
                        PVWORK,
                        ACCWORK,
                        value_buf,
                        prob_buf,
                        state_buf,
                        acc_buf,
                        batch,
                        kh,
                        worker.to(tl.int64),
                        previous,
                        prevslot,
                        begin,
                        stack,
                        HK,
                        CONTIG_K,
                        BM,
                        BN,
                        BS,
                        64,
                        True,
                    )
            _core_finish(
                O,
                LSE,
                ACCWORK,
                state_buf,
                acc_buf,
                worker.to(tl.int64),
                qt,
                qs,
                headstart,
                kh,
                qlen,
                qstart,
                qstart.to(tl.int64) * OR,
                H,
                group,
                TOTAL_Q,
                OR,
                OH,
                BM,
                64,
            )
            generation += stacks


# ---- vfirst_prefill ----


@triton.jit
def _pv_after_prefetch(
    V,
    PROBS,
    PVWORK,
    ACCWORK,
    value_buf,
    prob_buf,
    state_buf,
    acc_buf,
    batch,
    kh,
    worker,
    previous,
    prevslot,
    begin,
    stack,
    HK: tl.constexpr,
    CONTIG_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    ACC_UB_ROWS: tl.constexpr,
    COMPACT_O: tl.constexpr = False,
):
    prob = PROBS + (worker * 3 + prevslot) * 128 * BS
    pvwork = PVWORK + (worker * 3 + prevslot) * 128 * 256
    _: tl.constexpr = 16 if ACC_UB_ROWS == 16 else 32
    sync_block_wait("vector", "cube", 3 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    c2 = tl.zeros((BM, 256), tl.float32)
    for cv in range(0, stack, BN):
        if cv > 0:
            vptr = tl.make_block_ptr(
                V + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + cv) * 256,
                (BN, 256),
                (256, 1),
                (0, 0),
                (BN, 256),
                (1, 0),
            )
            tle.dsa.copy(vptr, value_buf, [BN, 256])
        pptr = tl.make_block_ptr(prob + cv, (BM, BS), (BS, 1), (0, 0), (BM, BN), (1, 0))
        tle.dsa.copy(pptr, prob_buf, [BM, BN])
        value2 = tle.dsa.to_tensor(value_buf, writable=False)
        weights2 = tle.dsa.to_tensor(prob_buf, writable=False)
        c2 = tl.dot(weights2, value2, c2)
    tl.store(pvwork + tl.arange(0, BM)[:, None] * 256 + tl.arange(0, 256)[None, :], c2)
    sync_block_set("cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S)


# ---- vfirst_prefill ----


@triton.jit
def _prefill_tail_softmax_rows(
    scratch,
    prob,
    state_buf,
    begin,
    stack,
    slot,
    sub,
    qs,
    qt,
    head_rows,
    headstart,
    group,
    qlen,
    klen,
    x2_vf,
    BM: tl.constexpr,
    BS: tl.constexpr,
    WIDTH: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    VM: tl.constexpr,
    SINGLE_HEAD: tl.constexpr = False,
    ROW_END=None,
):
    valid_rows = tl.minimum(BM // 2, tl.maximum(0, qlen - qt * 128 - x2_vf))
    for x2_vs in range(0, valid_rows, VM):
        if ROW_END is None:
            x2_rr = sub * BM + x2_vf + x2_vs + tl.arange(0, VM)
            if SINGLE_HEAD:
                x2_row = qt * 128 + x2_rr
                x2_valid = (x2_rr < qs) & (headstart < group) & (x2_row < qlen)
            else:
                x2_row = qt * 128 + x2_rr % qs
                x2_hl = headstart + x2_rr // qs
                x2_valid = (x2_rr < qs * head_rows) & (x2_hl < group) & (x2_row < qlen)
        x2_cols = begin + tl.arange(0, WIDTH)
        x2_offsets = (x2_vf + x2_vs + tl.arange(0, VM))[:, None] * BS + tl.arange(
            0, WIDTH
        )[None, :]
        x2_first = sub * BM + x2_vf + x2_vs
        x2_last = x2_first + VM - 1
        if SINGLE_HEAD:
            x2_minrow = qt * 128 + x2_first
            x2_maxrow = qt * 128 + x2_last
            x2_full = (
                (stack == WIDTH)
                & (x2_last < qs)
                & (headstart < group)
                & (x2_maxrow < qlen)
            )
        else:
            x2_same = x2_first // qs == x2_last // qs
            x2_minrow = qt * 128 + tl.where(x2_same, x2_first % qs, 0)
            x2_maxrow = qt * 128 + tl.where(x2_same, x2_last % qs, qs - 1)
            x2_full = (
                (stack == WIDTH)
                & (x2_last < qs * head_rows)
                & (headstart + x2_last // qs < group)
                & (x2_maxrow < qlen)
            )
        if CAUSAL:
            x2_full &= begin + WIDTH - 1 <= x2_minrow + klen - qlen
        if x2_full:
            x2_score = tl.load(scratch + x2_offsets, volatile=True) * SCALE
        else:
            x2_score = (
                tl.load(
                    scratch + x2_offsets,
                    x2_cols[None, :] < begin + stack,
                    0,
                    volatile=True,
                )
                * SCALE
            )
            if ROW_END is not None:
                x2_ev = tle.dsa.subview(ROW_END, [x2_vs], [VM], [1])
                x2_end = tl.minimum(
                    begin + stack, tle.dsa.to_tensor(x2_ev, writable=False)
                )
            else:
                x2_end = tl.full((VM,), begin + stack, tl.int32)
                if CAUSAL:
                    x2_end = tl.minimum(x2_end, x2_row + klen - qlen + 1)
                x2_end = tl.where(x2_valid, x2_end, begin)
            x2_score = tl.where(
                x2_cols[None, :] < x2_end[:, None], x2_score, -float("inf")
            )
        x2_mv = tle.dsa.subview(state_buf, [0, x2_vs], [1, VM], [1, 1])
        x2_lv = tle.dsa.subview(state_buf, [1, x2_vs], [1, VM], [1, 1])
        x2_av = tle.dsa.subview(state_buf, [2 + slot, x2_vs], [1, VM], [1, 1])
        x2_m = tl.reshape(tle.dsa.to_tensor(x2_mv, writable=False), (VM,))
        x2_l = tl.reshape(tle.dsa.to_tensor(x2_lv, writable=False), (VM,))
        x2_newm = tl.maximum(x2_m, tl.max(x2_score, 1))
        x2_safe = tl.where(x2_newm == -float("inf"), 0.0, x2_newm)
        x2_p = tl.exp(x2_score - x2_safe[:, None])
        x2_alpha = tl.exp(x2_m - x2_safe)
        x2_l = x2_l * x2_alpha + tl.sum(x2_p, 1)
        tl.store(prob + x2_offsets, x2_p.to(prob.dtype.element_ty))
        tle.dsa.to_buffer(x2_newm[None, :], bind_buffer=x2_mv)
        tle.dsa.to_buffer(x2_l[None, :], bind_buffer=x2_lv)
        tle.dsa.to_buffer(x2_alpha[None, :], bind_buffer=x2_av)


# ---- vfirst_prefill ----


@triton.jit
def _prefill_tail_softmax(
    scratch,
    prob,
    state_buf,
    row_end_buf,
    begin,
    stack,
    slot,
    qt,
    headstart,
    qlen,
    klen,
    group: tl.constexpr,
    SCALE: tl.constexpr,
):
    sync_block_wait("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
    vf = sub_vec_id().to(tl.int32) * 64
    if stack <= 256:
        _prefill_tail_softmax_rows(
            scratch,
            prob,
            state_buf,
            begin,
            stack,
            slot,
            0,
            128,
            qt,
            1,
            headstart,
            group,
            qlen,
            klen,
            vf,
            128,
            512,
            256,
            True,
            SCALE,
            32,
            True,
            row_end_buf,
        )
    else:
        _prefill_tail_softmax_rows(
            scratch,
            prob,
            state_buf,
            begin,
            stack,
            slot,
            0,
            128,
            qt,
            1,
            headstart,
            group,
            qlen,
            klen,
            vf,
            128,
            512,
            512,
            True,
            SCALE,
            16,
            True,
            row_end_buf,
        )
    sync_block_set("vector", "cube", 3 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)


# ---- vfirst_prefill ----


@triton.jit
def ascend_attention_prefill_vfirst(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    CONTIG_K: tl.constexpr,
    QCAP: tl.constexpr,
    Q_DIRECT: tl.constexpr,
):
    BM: tl.constexpr = 128
    BN: tl.constexpr = 256
    BS: tl.constexpr = 512
    group: tl.constexpr = H // HK
    BALANCED_TASKS: tl.constexpr = (TOTAL_Q >= 4096) and (TOTAL_Q <= 4352)
    query_buf = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([BN, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([BM, BN], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    state_buf = tle.dsa.alloc([5, BM // 2], tl.float32, tle.dsa.ascend.UB)
    row_end_buf = tle.dsa.alloc([BM // 2], tl.int32, tle.dsa.ascend.UB)
    acc_buf = tle.dsa.alloc([64, 256], tl.bfloat16, tle.dsa.ascend.UB)
    with tle.scope(core_mode="cube"):
        worker = tl.program_id(0)
        workers = tl.num_programs(0)
        qstart = tl.load(CUQ)
        qlen = tl.load(CUQ + 1) - qstart
        klen = tl.load(USED)
        ntiles = tl.cdiv(qlen, 128) * H
        batch = tl.full((), 0, tl.int32)
        generation = 0
        loop_limit = tl.cdiv(ntiles, workers) * workers if BALANCED_TASKS else ntiles
        for item in range(worker, loop_limit, workers):
            if BALANCED_TASKS:
                wave = item // workers
                lane = tl.where(wave % 2 == 0, worker, workers - 1 - worker)
                task = ntiles - 1 - wave * workers - lane
            else:
                task = item
            if not BALANCED_TASKS or task >= 0:
                qt = task // H
                kh = task % H // group
                headstart = task % group
                qs = tl.maximum(1, tl.minimum(qlen, 128))
                limit = tl.minimum(
                    klen, tl.maximum(0, tl.minimum(qlen, qt * 128 + 128) + klen - qlen)
                )
                stacks = tl.cdiv(limit, BS)
                _core_load_q(
                    Q,
                    Q_RAW,
                    query_buf,
                    batch,
                    qstart.to(tl.int64) * QR,
                    qlen,
                    task,
                    qt,
                    kh,
                    headstart,
                    group,
                    QR,
                    QH,
                    QCAP,
                    Q_DIRECT,
                    BM,
                )
                for step in range(0, stacks + 2):
                    if step >= 2:
                        # V is independent of this iteration's QK and P readiness.
                        prefetch_begin = (step - 2) * BS
                        prefetch_vptr = tl.make_block_ptr(
                            V
                            + (
                                (batch.to(tl.int64) * HK + kh) * CONTIG_K
                                + prefetch_begin
                            )
                            * 256,
                            (BN, 256),
                            (256, 1),
                            (0, 0),
                            (BN, 256),
                            (1, 0),
                        )
                        tle.dsa.copy(prefetch_vptr, value_buf, [BN, 256])
                    if step < stacks:
                        slot = (generation + step) % 3
                        begin = step * BS
                        stack = tl.minimum(BS, limit - begin)
                        scratch = SCORES + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                        prob = PROBS + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                        _core_qk_only(
                            K,
                            scratch,
                            prob,
                            query_buf,
                            key_buf_0,
                            key_buf_1,
                            state_buf,
                            row_end_buf,
                            batch,
                            kh,
                            begin,
                            stack,
                            slot,
                            qs,
                            qt,
                            headstart,
                            qlen,
                            klen,
                            HK,
                            group,
                            CONTIG_K,
                            BM,
                            BS,
                            SCALE,
                            True,
                        )
                    if step >= 2:
                        previous = step - 2
                        prevslot = (generation + previous) % 3
                        if generation + previous >= 3:
                            sync_block_wait(
                                "vector",
                                "cube",
                                9 + prevslot,
                                PIPE.PIPE_MTE3,
                                PIPE.PIPE_S,
                            )
                        begin = previous * BS
                        stack = tl.minimum(BS, limit - begin)
                        _pv_after_prefetch(
                            V,
                            PROBS,
                            PVWORK,
                            ACCWORK,
                            value_buf,
                            prob_buf,
                            state_buf,
                            acc_buf,
                            batch,
                            kh,
                            worker.to(tl.int64),
                            previous,
                            prevslot,
                            begin,
                            stack,
                            HK,
                            CONTIG_K,
                            BM,
                            BN,
                            BS,
                            64,
                            True,
                        )
                generation += stacks
        for drain in range(tl.minimum(generation, 3)):
            slot = (generation - tl.minimum(generation, 3) + drain) % 3
            sync_block_wait("vector", "cube", 9 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    with tle.scope(core_mode="vector"):
        worker = tl.program_id(0)
        workers = tl.num_programs(0)
        qstart = tl.load(CUQ)
        qlen = tl.load(CUQ + 1) - qstart
        klen = tl.load(USED)
        ntiles = tl.cdiv(qlen, 128) * H
        batch = tl.full((), 0, tl.int32)
        generation = 0
        loop_limit = tl.cdiv(ntiles, workers) * workers if BALANCED_TASKS else ntiles
        for item in range(worker, loop_limit, workers):
            if BALANCED_TASKS:
                wave = item // workers
                lane = tl.where(wave % 2 == 0, worker, workers - 1 - worker)
                task = ntiles - 1 - wave * workers - lane
            else:
                task = item
            if not BALANCED_TASKS or task >= 0:
                qt = task // H
                kh = task % H // group
                headstart = task % group
                qs = tl.maximum(1, tl.minimum(qlen, 128))
                limit = tl.minimum(
                    klen, tl.maximum(0, tl.minimum(qlen, qt * 128 + 128) + klen - qlen)
                )
                stacks = tl.cdiv(limit, BS)
                _core_init_o(acc_buf, ACCWORK, worker.to(tl.int64), BM, 64, True)
                _core_init_state(
                    state_buf, row_end_buf, qt, qs, headstart, qlen, klen, group, BM
                )
                for step in range(0, stacks + 2):
                    if step < stacks:
                        slot = (generation + step) % 3
                        begin = step * BS
                        stack = tl.minimum(BS, limit - begin)
                        scratch = SCORES + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                        prob = PROBS + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                        if TOTAL_Q % 128 != 0:
                            if qt * 128 + 128 > qlen:
                                _prefill_tail_softmax(
                                    scratch,
                                    prob,
                                    state_buf,
                                    row_end_buf,
                                    begin,
                                    stack,
                                    slot,
                                    qt,
                                    headstart,
                                    qlen,
                                    klen,
                                    group,
                                    SCALE,
                                )
                            else:
                                _core_softmax_only(
                                    K,
                                    scratch,
                                    prob,
                                    query_buf,
                                    key_buf_0,
                                    key_buf_1,
                                    state_buf,
                                    row_end_buf,
                                    batch,
                                    kh,
                                    begin,
                                    stack,
                                    slot,
                                    qs,
                                    qt,
                                    headstart,
                                    qlen,
                                    klen,
                                    HK,
                                    group,
                                    CONTIG_K,
                                    BM,
                                    BS,
                                    SCALE,
                                    True,
                                )
                        else:
                            _core_softmax_only(
                                K,
                                scratch,
                                prob,
                                query_buf,
                                key_buf_0,
                                key_buf_1,
                                state_buf,
                                row_end_buf,
                                batch,
                                kh,
                                begin,
                                stack,
                                slot,
                                qs,
                                qt,
                                headstart,
                                qlen,
                                klen,
                                HK,
                                group,
                                CONTIG_K,
                                BM,
                                BS,
                                SCALE,
                                True,
                            )
                    if step >= 2:
                        previous = step - 2
                        prevslot = (generation + previous) % 3
                        begin = previous * BS
                        stack = tl.minimum(BS, limit - begin)
                        _core_o(
                            V,
                            PROBS,
                            PVWORK,
                            ACCWORK,
                            value_buf,
                            prob_buf,
                            state_buf,
                            acc_buf,
                            batch,
                            kh,
                            worker.to(tl.int64),
                            previous,
                            prevslot,
                            begin,
                            stack,
                            HK,
                            CONTIG_K,
                            BM,
                            BN,
                            BS,
                            64,
                            True,
                        )
                _core_finish(
                    O,
                    LSE,
                    ACCWORK,
                    state_buf,
                    acc_buf,
                    worker.to(tl.int64),
                    qt,
                    qs,
                    headstart,
                    kh,
                    qlen,
                    qstart,
                    qstart.to(tl.int64) * OR,
                    H,
                    group,
                    TOTAL_Q,
                    OR,
                    OH,
                    BM,
                    64,
                )
                generation += stacks


# ---- vfirst_prefill ----


@triton.jit
def _prefill_recycle_o(
    V,
    PROBS,
    PVWORK,
    ACCWORK,
    value_buf,
    prob_buf,
    state_buf,
    acc_buf,
    batch,
    kh,
    worker,
    previous,
    prevslot,
    begin,
    stack,
    HK: tl.constexpr,
    CONTIG_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    ACC_UB_ROWS: tl.constexpr,
    COMPACT_O: tl.constexpr = False,
    TASK_STACKS=0,
    FULL_ROWS=False,
):
    _ = PROBS + (worker * 3 + prevslot) * 128 * BS
    pvwork = PVWORK + (worker * 3 + prevslot) * 128 * 256
    O_VM: tl.constexpr = 16 if ACC_UB_ROWS == 16 else 32
    sync_block_wait("cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S)
    x4_vf = sub_vec_id().to(tl.int32) * (BM // 2)
    if ACC_UB_ROWS > 0:
        for x4_vs in range(0, ACC_UB_ROWS, O_VM):
            x4_av = tle.dsa.subview(state_buf, [2 + prevslot, x4_vs], [1, O_VM], [1, 1])
            x4_acc = tle.dsa.subview(acc_buf, [x4_vs, 0], [O_VM, 256], [1, 1])
            x4_alpha = tl.reshape(tle.dsa.to_tensor(x4_av, writable=False), (O_VM,))
            x4_old = tle.dsa.to_tensor(x4_acc, writable=False).to(tl.float32)
            x4_pv = tl.load(
                pvwork
                + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
                + tl.arange(0, 256)[None, :],
                volatile=True,
            )
            if COMPACT_O:
                tle.dsa.to_buffer(
                    (x4_old * x4_alpha[:, None] + x4_pv).to(tl.bfloat16),
                    bind_buffer=x4_acc,
                )
            else:
                tle.dsa.to_buffer(
                    x4_old * x4_alpha[:, None] + x4_pv, bind_buffer=x4_acc
                )
    for x4_vs in range(ACC_UB_ROWS, BM // 2, O_VM):
        x4_av = tle.dsa.subview(state_buf, [2 + prevslot, x4_vs], [1, O_VM], [1, 1])
        x4_ab = (
            ACCWORK
            + worker * 128 * 256
            + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
            + tl.arange(0, 256)[None, :]
        )
        x4_alpha = tl.reshape(tle.dsa.to_tensor(x4_av, writable=False), (O_VM,))
        x4_old = tl.load(x4_ab, volatile=True)
        x4_pv = tl.load(
            pvwork
            + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
            + tl.arange(0, 256)[None, :],
            volatile=True,
        )
        tl.store(x4_ab, x4_old * x4_alpha[:, None] + x4_pv)
    if (previous + 3 >= TASK_STACKS) | (not FULL_ROWS):
        sync_block_set("vector", "cube", 9 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S)


# ---- vfirst_prefill ----


@triton.jit
def ascend_attention_prefill_bounded_release(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    CONTIG_K: tl.constexpr,
    QCAP: tl.constexpr,
    Q_DIRECT: tl.constexpr,
):
    BM: tl.constexpr = 128
    BN: tl.constexpr = 256
    BS: tl.constexpr = 512
    group: tl.constexpr = H // HK
    BALANCED_TASKS: tl.constexpr = TOTAL_Q >= 4096 and TOTAL_Q <= 4352
    query_buf = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([BN, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([BM, BN], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    state_buf = tle.dsa.alloc([5, BM // 2], tl.float32, tle.dsa.ascend.UB)
    row_end_buf = tle.dsa.alloc([BM // 2], tl.int32, tle.dsa.ascend.UB)
    acc_buf = tle.dsa.alloc([64, 256], tl.bfloat16, tle.dsa.ascend.UB)
    with tle.scope(core_mode="cube"):
        worker = tl.program_id(0)
        workers = tl.num_programs(0)
        qstart = tl.load(CUQ)
        qlen = tl.load(CUQ + 1) - qstart
        klen = tl.load(USED)
        ntiles = tl.cdiv(qlen, 128) * H
        batch = tl.full((), 0, tl.int32)
        generation = 0
        loop_limit = tl.cdiv(ntiles, workers) * workers if BALANCED_TASKS else ntiles
        for item in range(worker, loop_limit, workers):
            if BALANCED_TASKS:
                wave = item // workers
                lane = tl.where(wave % 2 == 0, worker, workers - 1 - worker)
                task = ntiles - 1 - wave * workers - lane
            else:
                task = item
            if not BALANCED_TASKS or task >= 0:
                qt = task // H
                kh = task % H // group
                headstart = task % group
                qs = tl.maximum(1, tl.minimum(qlen, 128))
                limit = tl.minimum(
                    klen, tl.maximum(0, tl.minimum(qlen, qt * 128 + 128) + klen - qlen)
                )
                stacks = tl.cdiv(limit, BS)
                _core_load_q(
                    Q,
                    Q_RAW,
                    query_buf,
                    batch,
                    qstart.to(tl.int64) * QR,
                    qlen,
                    task,
                    qt,
                    kh,
                    headstart,
                    group,
                    QR,
                    QH,
                    QCAP,
                    Q_DIRECT,
                    BM,
                )
                for step in range(0, stacks + 2):
                    if step >= 2:
                        prefetch_begin = (step - 2) * BS
                        prefetch_vptr = tl.make_block_ptr(
                            V
                            + (
                                (batch.to(tl.int64) * HK + kh) * CONTIG_K
                                + prefetch_begin
                            )
                            * 256,
                            (BN, 256),
                            (256, 1),
                            (0, 0),
                            (BN, 256),
                            (1, 0),
                        )
                        tle.dsa.copy(prefetch_vptr, value_buf, [BN, 256])
                    if step < stacks:
                        slot = (generation + step) % 3
                        begin = step * BS
                        stack = tl.minimum(BS, limit - begin)
                        scratch = SCORES + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                        prob = PROBS + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                        _core_qk_only(
                            K,
                            scratch,
                            prob,
                            query_buf,
                            key_buf_0,
                            key_buf_1,
                            state_buf,
                            row_end_buf,
                            batch,
                            kh,
                            begin,
                            stack,
                            slot,
                            qs,
                            qt,
                            headstart,
                            qlen,
                            klen,
                            HK,
                            group,
                            CONTIG_K,
                            BM,
                            BS,
                            SCALE,
                            True,
                        )
                    if step >= 2:
                        previous = step - 2
                        prevslot = (generation + previous) % 3
                        if (generation + previous >= 3) & (
                            (previous < 3) | (qt * 128 + 128 > qlen)
                        ):
                            sync_block_wait(
                                "vector",
                                "cube",
                                9 + prevslot,
                                PIPE.PIPE_MTE3,
                                PIPE.PIPE_S,
                            )
                        begin = previous * BS
                        stack = tl.minimum(BS, limit - begin)
                        _pv_after_prefetch(
                            V,
                            PROBS,
                            PVWORK,
                            ACCWORK,
                            value_buf,
                            prob_buf,
                            state_buf,
                            acc_buf,
                            batch,
                            kh,
                            worker.to(tl.int64),
                            previous,
                            prevslot,
                            begin,
                            stack,
                            HK,
                            CONTIG_K,
                            BM,
                            BN,
                            BS,
                            64,
                            True,
                        )
                generation += stacks
        for drain in range(tl.minimum(generation, 3)):
            slot = (generation - tl.minimum(generation, 3) + drain) % 3
            sync_block_wait("vector", "cube", 9 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    with tle.scope(core_mode="vector"):
        worker = tl.program_id(0)
        workers = tl.num_programs(0)
        qstart = tl.load(CUQ)
        qlen = tl.load(CUQ + 1) - qstart
        klen = tl.load(USED)
        ntiles = tl.cdiv(qlen, 128) * H
        batch = tl.full((), 0, tl.int32)
        generation = 0
        loop_limit = tl.cdiv(ntiles, workers) * workers if BALANCED_TASKS else ntiles
        for item in range(worker, loop_limit, workers):
            if BALANCED_TASKS:
                wave = item // workers
                lane = tl.where(wave % 2 == 0, worker, workers - 1 - worker)
                task = ntiles - 1 - wave * workers - lane
            else:
                task = item
            if not BALANCED_TASKS or task >= 0:
                qt = task // H
                kh = task % H // group
                headstart = task % group
                qs = tl.maximum(1, tl.minimum(qlen, 128))
                limit = tl.minimum(
                    klen, tl.maximum(0, tl.minimum(qlen, qt * 128 + 128) + klen - qlen)
                )
                stacks = tl.cdiv(limit, BS)
                _core_init_o(acc_buf, ACCWORK, worker.to(tl.int64), BM, 64, True)
                _core_init_state(
                    state_buf, row_end_buf, qt, qs, headstart, qlen, klen, group, BM
                )
                for step in range(0, stacks + 2):
                    if step < stacks:
                        slot = (generation + step) % 3
                        begin = step * BS
                        stack = tl.minimum(BS, limit - begin)
                        scratch = SCORES + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                        prob = PROBS + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                        if TOTAL_Q % 128 != 0:
                            if qt * 128 + 128 > qlen:
                                _prefill_tail_softmax(
                                    scratch,
                                    prob,
                                    state_buf,
                                    row_end_buf,
                                    begin,
                                    stack,
                                    slot,
                                    qt,
                                    headstart,
                                    qlen,
                                    klen,
                                    group,
                                    SCALE,
                                )
                            else:
                                _core_softmax_only(
                                    K,
                                    scratch,
                                    prob,
                                    query_buf,
                                    key_buf_0,
                                    key_buf_1,
                                    state_buf,
                                    row_end_buf,
                                    batch,
                                    kh,
                                    begin,
                                    stack,
                                    slot,
                                    qs,
                                    qt,
                                    headstart,
                                    qlen,
                                    klen,
                                    HK,
                                    group,
                                    CONTIG_K,
                                    BM,
                                    BS,
                                    SCALE,
                                    True,
                                )
                        else:
                            _core_softmax_only(
                                K,
                                scratch,
                                prob,
                                query_buf,
                                key_buf_0,
                                key_buf_1,
                                state_buf,
                                row_end_buf,
                                batch,
                                kh,
                                begin,
                                stack,
                                slot,
                                qs,
                                qt,
                                headstart,
                                qlen,
                                klen,
                                HK,
                                group,
                                CONTIG_K,
                                BM,
                                BS,
                                SCALE,
                                True,
                            )
                    if step >= 2:
                        previous = step - 2
                        prevslot = (generation + previous) % 3
                        begin = previous * BS
                        stack = tl.minimum(BS, limit - begin)
                        _prefill_recycle_o(
                            V,
                            PROBS,
                            PVWORK,
                            ACCWORK,
                            value_buf,
                            prob_buf,
                            state_buf,
                            acc_buf,
                            batch,
                            kh,
                            worker.to(tl.int64),
                            previous,
                            prevslot,
                            begin,
                            stack,
                            HK,
                            CONTIG_K,
                            BM,
                            BN,
                            BS,
                            64,
                            True,
                            stacks,
                            qt * 128 + 128 <= qlen,
                        )
                _core_finish(
                    O,
                    LSE,
                    ACCWORK,
                    state_buf,
                    acc_buf,
                    worker.to(tl.int64),
                    qt,
                    qs,
                    headstart,
                    kh,
                    qlen,
                    qstart,
                    qstart.to(tl.int64) * OR,
                    H,
                    group,
                    TOTAL_Q,
                    OR,
                    OH,
                    BM,
                    64,
                )
                generation += stacks


# ---- vfirst_prefill ----


@triton.jit
def _pv_whole_probability(
    V,
    PROBS,
    PVWORK,
    ACCWORK,
    value_buf,
    prob_buf,
    state_buf,
    acc_buf,
    batch,
    kh,
    worker,
    previous,
    prevslot,
    begin,
    stack,
    HK: tl.constexpr,
    CONTIG_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    ACC_UB_ROWS: tl.constexpr,
    COMPACT_O: tl.constexpr = False,
):
    if stack > BN:
        full_prob_buf = tle.dsa.alloc(
            [BM, BS], PROBS.dtype.element_ty, tle.dsa.ascend.L1
        )
        prob = PROBS + (worker * 3 + prevslot) * 128 * BS
        pvwork = PVWORK + (worker * 3 + prevslot) * 128 * 256
        sync_block_wait("vector", "cube", 3 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
        pptr = tl.make_block_ptr(prob, (BM, BS), (BS, 1), (0, 0), (BM, BS), (1, 0))
        tle.dsa.copy(pptr, full_prob_buf, [BM, BS])
        all_weights = tle.dsa.to_tensor(full_prob_buf, writable=False)
        weights0 = tle.dsa.extract_slice(all_weights, [0, 0], [BM, BN], [1, 1])
        value0 = tle.dsa.to_tensor(value_buf, writable=False)
        c2 = tl.dot(weights0, value0, tl.zeros((BM, 256), tl.float32))
        vptr = tl.make_block_ptr(
            V + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + BN) * 256,
            (BN, 256),
            (256, 1),
            (0, 0),
            (BN, 256),
            (1, 0),
        )
        tle.dsa.copy(vptr, value_buf, [BN, 256])
        weights1 = tle.dsa.extract_slice(all_weights, [0, BN], [BM, BN], [1, 1])
        value1 = tle.dsa.to_tensor(value_buf, writable=False)
        c2 = tl.dot(weights1, value1, c2)
        tl.store(
            pvwork + tl.arange(0, BM)[:, None] * 256 + tl.arange(0, 256)[None, :], c2
        )
        sync_block_set("cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S)
    else:
        _pv_after_prefetch(
            V,
            PROBS,
            PVWORK,
            ACCWORK,
            value_buf,
            prob_buf,
            state_buf,
            acc_buf,
            batch,
            kh,
            worker,
            previous,
            prevslot,
            begin,
            stack,
            HK,
            CONTIG_K,
            BM,
            BN,
            BS,
            ACC_UB_ROWS,
            COMPACT_O,
        )


# ---- vfirst_prefill ----


@triton.jit
def ascend_attention_prefill_prob_stack(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    CONTIG_K: tl.constexpr,
    QCAP: tl.constexpr,
    Q_DIRECT: tl.constexpr,
):
    BM: tl.constexpr = 128
    BN: tl.constexpr = 256
    BS: tl.constexpr = 512
    group: tl.constexpr = H // HK
    BALANCED_TASKS: tl.constexpr = TOTAL_Q >= 4096 and TOTAL_Q <= 4352
    query_buf = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([BN, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([BM, BN], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    state_buf = tle.dsa.alloc([5, BM // 2], tl.float32, tle.dsa.ascend.UB)
    row_end_buf = tle.dsa.alloc([BM // 2], tl.int32, tle.dsa.ascend.UB)
    acc_buf = tle.dsa.alloc([64, 256], tl.bfloat16, tle.dsa.ascend.UB)
    with tle.scope(core_mode="cube"):
        worker = tl.program_id(0)
        workers = tl.num_programs(0)
        qstart = tl.load(CUQ)
        qlen = tl.load(CUQ + 1) - qstart
        klen = tl.load(USED)
        ntiles = tl.cdiv(qlen, 128) * H
        batch = tl.full((), 0, tl.int32)
        generation = 0
        loop_limit = tl.cdiv(ntiles, workers) * workers if BALANCED_TASKS else ntiles
        for item in range(worker, loop_limit, workers):
            if BALANCED_TASKS:
                wave = item // workers
                lane = tl.where(wave % 2 == 0, worker, workers - 1 - worker)
                task = ntiles - 1 - wave * workers - lane
            else:
                task = item
            if not BALANCED_TASKS or task >= 0:
                qt = task // H
                kh = task % H // group
                headstart = task % group
                qs = tl.maximum(1, tl.minimum(qlen, 128))
                limit = tl.minimum(
                    klen, tl.maximum(0, tl.minimum(qlen, qt * 128 + 128) + klen - qlen)
                )
                stacks = tl.cdiv(limit, BS)
                _core_load_q(
                    Q,
                    Q_RAW,
                    query_buf,
                    batch,
                    qstart.to(tl.int64) * QR,
                    qlen,
                    task,
                    qt,
                    kh,
                    headstart,
                    group,
                    QR,
                    QH,
                    QCAP,
                    Q_DIRECT,
                    BM,
                )
                for step in range(0, stacks + 2):
                    if step >= 2:
                        prefetch_begin = (step - 2) * BS
                        prefetch_vptr = tl.make_block_ptr(
                            V
                            + (
                                (batch.to(tl.int64) * HK + kh) * CONTIG_K
                                + prefetch_begin
                            )
                            * 256,
                            (BN, 256),
                            (256, 1),
                            (0, 0),
                            (BN, 256),
                            (1, 0),
                        )
                        tle.dsa.copy(prefetch_vptr, value_buf, [BN, 256])
                    if step < stacks:
                        slot = (generation + step) % 3
                        begin = step * BS
                        stack = tl.minimum(BS, limit - begin)
                        scratch = SCORES + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                        prob = PROBS + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                        _core_qk_only(
                            K,
                            scratch,
                            prob,
                            query_buf,
                            key_buf_0,
                            key_buf_1,
                            state_buf,
                            row_end_buf,
                            batch,
                            kh,
                            begin,
                            stack,
                            slot,
                            qs,
                            qt,
                            headstart,
                            qlen,
                            klen,
                            HK,
                            group,
                            CONTIG_K,
                            BM,
                            BS,
                            SCALE,
                            True,
                        )
                    if step >= 2:
                        previous = step - 2
                        prevslot = (generation + previous) % 3
                        if (generation + previous >= 3) & (
                            (previous < 3) | (qt * 128 + 128 > qlen)
                        ):
                            sync_block_wait(
                                "vector",
                                "cube",
                                9 + prevslot,
                                PIPE.PIPE_MTE3,
                                PIPE.PIPE_S,
                            )
                        begin = previous * BS
                        stack = tl.minimum(BS, limit - begin)
                        _pv_whole_probability(
                            V,
                            PROBS,
                            PVWORK,
                            ACCWORK,
                            value_buf,
                            prob_buf,
                            state_buf,
                            acc_buf,
                            batch,
                            kh,
                            worker.to(tl.int64),
                            previous,
                            prevslot,
                            begin,
                            stack,
                            HK,
                            CONTIG_K,
                            BM,
                            BN,
                            BS,
                            64,
                            True,
                        )
                generation += stacks
        for drain in range(tl.minimum(generation, 3)):
            slot = (generation - tl.minimum(generation, 3) + drain) % 3
            sync_block_wait("vector", "cube", 9 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    with tle.scope(core_mode="vector"):
        worker = tl.program_id(0)
        workers = tl.num_programs(0)
        qstart = tl.load(CUQ)
        qlen = tl.load(CUQ + 1) - qstart
        klen = tl.load(USED)
        ntiles = tl.cdiv(qlen, 128) * H
        batch = tl.full((), 0, tl.int32)
        generation = 0
        loop_limit = tl.cdiv(ntiles, workers) * workers if BALANCED_TASKS else ntiles
        for item in range(worker, loop_limit, workers):
            if BALANCED_TASKS:
                wave = item // workers
                lane = tl.where(wave % 2 == 0, worker, workers - 1 - worker)
                task = ntiles - 1 - wave * workers - lane
            else:
                task = item
            if not BALANCED_TASKS or task >= 0:
                qt = task // H
                kh = task % H // group
                headstart = task % group
                qs = tl.maximum(1, tl.minimum(qlen, 128))
                limit = tl.minimum(
                    klen, tl.maximum(0, tl.minimum(qlen, qt * 128 + 128) + klen - qlen)
                )
                stacks = tl.cdiv(limit, BS)
                _core_init_o(acc_buf, ACCWORK, worker.to(tl.int64), BM, 64, True)
                _core_init_state(
                    state_buf, row_end_buf, qt, qs, headstart, qlen, klen, group, BM
                )
                for step in range(0, stacks + 2):
                    if step < stacks:
                        slot = (generation + step) % 3
                        begin = step * BS
                        stack = tl.minimum(BS, limit - begin)
                        scratch = SCORES + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                        prob = PROBS + (worker.to(tl.int64) * 3 + slot) * 128 * BS
                        if TOTAL_Q % 128 != 0:
                            if qt * 128 + 128 > qlen:
                                _prefill_tail_softmax(
                                    scratch,
                                    prob,
                                    state_buf,
                                    row_end_buf,
                                    begin,
                                    stack,
                                    slot,
                                    qt,
                                    headstart,
                                    qlen,
                                    klen,
                                    group,
                                    SCALE,
                                )
                            else:
                                _core_softmax_only(
                                    K,
                                    scratch,
                                    prob,
                                    query_buf,
                                    key_buf_0,
                                    key_buf_1,
                                    state_buf,
                                    row_end_buf,
                                    batch,
                                    kh,
                                    begin,
                                    stack,
                                    slot,
                                    qs,
                                    qt,
                                    headstart,
                                    qlen,
                                    klen,
                                    HK,
                                    group,
                                    CONTIG_K,
                                    BM,
                                    BS,
                                    SCALE,
                                    True,
                                )
                        else:
                            _core_softmax_only(
                                K,
                                scratch,
                                prob,
                                query_buf,
                                key_buf_0,
                                key_buf_1,
                                state_buf,
                                row_end_buf,
                                batch,
                                kh,
                                begin,
                                stack,
                                slot,
                                qs,
                                qt,
                                headstart,
                                qlen,
                                klen,
                                HK,
                                group,
                                CONTIG_K,
                                BM,
                                BS,
                                SCALE,
                                True,
                            )
                    if step >= 2:
                        previous = step - 2
                        prevslot = (generation + previous) % 3
                        begin = previous * BS
                        stack = tl.minimum(BS, limit - begin)
                        _prefill_recycle_o(
                            V,
                            PROBS,
                            PVWORK,
                            ACCWORK,
                            value_buf,
                            prob_buf,
                            state_buf,
                            acc_buf,
                            batch,
                            kh,
                            worker.to(tl.int64),
                            previous,
                            prevslot,
                            begin,
                            stack,
                            HK,
                            CONTIG_K,
                            BM,
                            BN,
                            BS,
                            64,
                            True,
                            stacks,
                            qt * 128 + 128 <= qlen,
                        )
                _core_finish(
                    O,
                    LSE,
                    ACCWORK,
                    state_buf,
                    acc_buf,
                    worker.to(tl.int64),
                    qt,
                    qs,
                    headstart,
                    kh,
                    qlen,
                    qstart,
                    qstart.to(tl.int64) * OR,
                    H,
                    group,
                    TOTAL_Q,
                    OR,
                    OH,
                    BM,
                    64,
                )
                generation += stacks


# ---- long_prefill ----


@triton.jit
def ascend_attention_long_prefill(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    CONTIG_K: tl.constexpr,
    QCAP: tl.constexpr,
    Q_DIRECT: tl.constexpr,
):
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    batch = tl.full((), 0, tl.int32)
    qlen = tl.load(CUQ + 1) - tl.load(CUQ)
    klen = tl.load(USED)
    ntiles = tl.cdiv(qlen, 128) * H
    _attention_request_stream(
        Q,
        Q_RAW,
        K,
        V,
        O,
        LSE,
        CUQ,
        SCORES,
        PROBS,
        PVWORK,
        ACCWORK,
        batch,
        worker,
        ntiles,
        workers,
        worker,
        qlen,
        klen,
        H,
        HK,
        TOTAL_Q,
        SCALE,
        QR,
        QH,
        OR,
        OH,
        CONTIG_K,
        QCAP,
        Q_DIRECT,
        True,
        128,
        256,
        512,
        64,  # All O rows reside in UB, stored as BF16 between FP32 updates.
        True,
    )


# ---- long_prefill ----


def launch_long_prefill(
    params, workers, packed, scores, probs, pvwork, accwork, scale=None
):
    q, k, v, qcap, capacity, q_direct = packed
    kernel = ascend_attention_prefill_vfirst
    if 2304 < params.total_q <= 4608:
        kernel = ascend_attention_prefill_prob_stack

    kernel[(workers,)](
        q,
        params.q_ptr,
        k,
        v,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        scores,
        probs,
        pvwork,
        accwork,
        params.h,
        params.hk,
        params.total_q,
        params.scale_softmax if scale is None else scale,
        params.q_row_stride,
        params.q_head_stride,
        params.o_row_stride,
        params.o_head_stride,
        capacity,
        qcap,
        q_direct,
        disable_auto_inject_block_sync=True,
        enable_fp_fusion=False,
        multibuffer=False,
        enable_reduce_sum_strength=True,
    )


# ---- prefill_clipped ----


@triton.jit
def _attention_prefill_clipped_queue(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    REQUEST_IDS,
    TASK_SCORES,
    TASK_PROBS,
    PVWORK,
    ACCWORK,
    total_tasks,
    workers,
    worker,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    CONTIG_K: tl.constexpr,
    QCAP: tl.constexpr,
    Q_DIRECT: tl.constexpr,
    REQUEST_LIST: tl.constexpr,
    SKIP_SHORT_KV: tl.constexpr,
):
    group: tl.constexpr = H // HK
    query_buf = tle.dsa.alloc([128, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([256, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([128, 256], TASK_PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    with tle.scope(core_mode="cube"):
        c_worker = tl.program_id(0).to(tl.int64)
        c_workers = tl.num_programs(0)
        c_queue_qstart = tl.load(CUQ)
        c_queue_qlen = tl.load(CUQ + 1) - c_queue_qstart
        c_queue_klen = tl.load(USED)
        c_worker_id = c_worker.to(tl.int32)
        c_total_tasks = tl.cdiv(c_queue_qlen, 128) * H
        c_count = tl.cdiv(tl.maximum(c_total_tasks - c_worker_id, 0), c_workers)
        c_old_begin = 0
        c_old_batch = 0
        c_old_qlen = 0
        c_old_klen = 0
        _ = 0
        _ = 0
        _ = 0
        _ = 0
        c_current_begin = 0
        c_current_batch = 0
        c_current_qlen = 0
        c_current_klen = 0
        c_current_qstart = 0
        for c_tick in range(c_count + 2):
            if c_tick >= 2:
                c_old_index = c_tick - 2
                c_old_slot = c_old_index % 3
                (
                    _,
                    c_old_begin,
                    _,
                    c_old_batch,
                    c_old_qlen,
                    c_old_klen,
                    _,
                ) = (0, 0, c_total_tasks, 0, c_queue_qlen, c_queue_klen, c_queue_qstart)
                c_old_task = c_worker_id + c_old_index * c_workers - c_old_begin
                c_old_qt = c_old_task // H
                if H == 4:
                    c_q_tiles = tl.cdiv(c_queue_qlen, 128)
                    c_first_wave = tl.minimum(c_q_tiles, tl.maximum(1, c_workers // H))
                    c_old_qt = tl.where(
                        c_old_qt < c_first_wave,
                        c_q_tiles - 1 - c_old_qt,
                        c_old_qt - c_first_wave,
                    )
                c_old_head = c_old_task % H
                c_old_kh = c_old_head // group
                c_old_limit = tl.minimum(
                    c_old_klen,
                    tl.maximum(
                        0,
                        tl.minimum(c_old_qlen, c_old_qt * 128 + 128)
                        + c_old_klen
                        - c_old_qlen,
                    ),
                )
                c_old_prob = TASK_PROBS + (c_worker * 3 + c_old_slot) * 128 * 1536
                c_old_pv = PVWORK + (c_worker * 3 + c_old_slot) * 128 * 256
                sync_block_wait(
                    "vector", "cube", 3 + c_old_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                )
                _whole_task_pv(
                    V,
                    c_old_prob,
                    c_old_pv,
                    value_buf,
                    prob_buf,
                    c_old_batch,
                    c_old_kh,
                    c_old_limit,
                    HK,
                    CONTIG_K,
                )
                sync_block_set(
                    "cube", "vector", 6 + c_old_slot, PIPE.PIPE_FIX, PIPE.PIPE_S
                )
            if c_tick < c_count:
                c_slot = c_tick % 3
                (
                    _,
                    c_current_begin,
                    _,
                    c_current_batch,
                    c_current_qlen,
                    c_current_klen,
                    c_current_qstart,
                ) = (0, 0, c_total_tasks, 0, c_queue_qlen, c_queue_klen, c_queue_qstart)
                c_task = c_worker_id + c_tick * c_workers - c_current_begin
                c_qt = c_task // H
                if H == 4:
                    c_q_tiles = tl.cdiv(c_queue_qlen, 128)
                    c_first_wave = tl.minimum(c_q_tiles, tl.maximum(1, c_workers // H))
                    c_qt = tl.where(
                        c_qt < c_first_wave, c_q_tiles - 1 - c_qt, c_qt - c_first_wave
                    )
                c_head = c_task % H
                if H == 4:
                    c_task = c_qt * H + c_head
                c_kh = c_head // group
                c_limit = tl.minimum(
                    c_current_klen,
                    tl.maximum(
                        0,
                        tl.minimum(c_current_qlen, c_qt * 128 + 128)
                        + c_current_klen
                        - c_current_qlen,
                    ),
                )
                c_score = TASK_SCORES + (c_worker * 3 + c_slot) * 128 * 1536
                if c_tick >= 3:
                    sync_block_wait(
                        "vector", "cube", 9 + c_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                    )
                _whole_task_qk(
                    Q,
                    Q_RAW,
                    K,
                    c_score,
                    query_buf,
                    key_buf_0,
                    key_buf_1,
                    c_current_batch,
                    c_task,
                    c_qt,
                    c_head,
                    c_kh,
                    c_current_qstart,
                    c_current_qlen,
                    c_limit,
                    HK,
                    QR,
                    QH,
                    CONTIG_K,
                    QCAP,
                    Q_DIRECT,
                )
                sync_block_set("cube", "vector", c_slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
        for c_drain in range(tl.maximum(0, c_count - 3), c_count):
            sync_block_wait(
                "vector", "cube", 9 + c_drain % 3, PIPE.PIPE_MTE3, PIPE.PIPE_S
            )
    with tle.scope(core_mode="vector"):
        v_worker = tl.program_id(0).to(tl.int64)
        v_workers = tl.num_programs(0)
        v_queue_qstart = tl.load(CUQ)
        v_queue_qlen = tl.load(CUQ + 1) - v_queue_qstart
        v_queue_klen = tl.load(USED)
        v_worker_id = v_worker.to(tl.int32)
        v_total_tasks = tl.cdiv(v_queue_qlen, 128) * H
        v_count = tl.cdiv(tl.maximum(v_total_tasks - v_worker_id, 0), v_workers)
        v_old_begin = 0
        _ = 0
        v_old_qlen = v_queue_qlen
        _ = v_queue_klen
        v_old_qstart = v_queue_qstart
        v_middle_begin = 0
        v_middle_qlen = 0
        v_middle_klen = 0
        _ = 0
        _ = 0
        _ = 0
        _ = 0
        _ = 0
        for v_tick in range(v_count + 2):
            if (v_tick >= 1) & (v_tick < v_count + 1):
                v_middle_index = v_tick - 1
                v_middle_slot = v_middle_index % 3
                _, v_middle_begin, _, _, v_middle_qlen, v_middle_klen, _ = (
                    0,
                    0,
                    v_total_tasks,
                    0,
                    v_queue_qlen,
                    v_queue_klen,
                    v_queue_qstart,
                )
                v_middle_task = (
                    v_worker_id + v_middle_index * v_workers - v_middle_begin
                )
                v_middle_qt = v_middle_task // H
                if H == 4:
                    v_q_tiles = tl.cdiv(v_queue_qlen, 128)
                    v_first_wave = tl.minimum(v_q_tiles, tl.maximum(1, v_workers // H))
                    v_middle_qt = tl.where(
                        v_middle_qt < v_first_wave,
                        v_q_tiles - 1 - v_middle_qt,
                        v_middle_qt - v_first_wave,
                    )
                v_middle_limit = tl.minimum(
                    v_middle_klen,
                    tl.maximum(
                        0,
                        tl.minimum(v_middle_qlen, v_middle_qt * 128 + 128)
                        + v_middle_klen
                        - v_middle_qlen,
                    ),
                )
                v_middle_score = (
                    TASK_SCORES + (v_worker * 3 + v_middle_slot) * 128 * 1536
                )
                v_middle_prob = TASK_PROBS + (v_worker * 3 + v_middle_slot) * 128 * 1536
                v_middle_state = ACCWORK + v_worker * 128 * 256 + v_middle_slot * 256
                sync_block_wait(
                    "cube", "vector", v_middle_slot, PIPE.PIPE_FIX, PIPE.PIPE_S
                )
                with tle.scope(core_mode="vector"):
                    if v_middle_limit <= 256:
                        _prefill_softmax_valid_rows(
                            v_middle_score,
                            v_middle_prob,
                            v_middle_state,
                            v_middle_limit,
                            v_middle_qt,
                            v_middle_qlen,
                            v_middle_klen,
                            SCALE,
                            256,
                            32,
                        )
                    elif v_middle_limit <= 512:
                        _prefill_softmax_valid_rows(
                            v_middle_score,
                            v_middle_prob,
                            v_middle_state,
                            v_middle_limit,
                            v_middle_qt,
                            v_middle_qlen,
                            v_middle_klen,
                            SCALE,
                            512,
                            16,
                        )
                    elif v_middle_limit <= 1024:
                        _prefill_softmax_valid_rows(
                            v_middle_score,
                            v_middle_prob,
                            v_middle_state,
                            v_middle_limit,
                            v_middle_qt,
                            v_middle_qlen,
                            v_middle_klen,
                            SCALE,
                            1024,
                            8,
                        )
                    else:
                        _prefill_softmax_valid_rows(
                            v_middle_score,
                            v_middle_prob,
                            v_middle_state,
                            v_middle_limit,
                            v_middle_qt,
                            v_middle_qlen,
                            v_middle_klen,
                            SCALE,
                            2048,
                            4,
                        )
                sync_block_set(
                    "vector", "cube", 3 + v_middle_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                )
            if v_tick >= 2:
                v_out_index = v_tick - 2
                v_out_slot = v_out_index % 3
                v_out_task = v_worker_id + v_out_index * v_workers - v_old_begin
                v_out_qt = v_out_task // H
                if H == 4:
                    v_q_tiles = tl.cdiv(v_queue_qlen, 128)
                    v_first_wave = tl.minimum(v_q_tiles, tl.maximum(1, v_workers // H))
                    v_out_qt = tl.where(
                        v_out_qt < v_first_wave,
                        v_q_tiles - 1 - v_out_qt,
                        v_out_qt - v_first_wave,
                    )
                v_out_head = v_out_task % H
                v_out_pv = PVWORK + (v_worker * 3 + v_out_slot) * 128 * 256
                v_out_state = ACCWORK + v_worker * 128 * 256 + v_out_slot * 256
                sync_block_wait(
                    "cube", "vector", 6 + v_out_slot, PIPE.PIPE_FIX, PIPE.PIPE_S
                )
                _whole_task_finish(
                    O,
                    LSE,
                    v_out_pv,
                    v_out_state,
                    v_out_qt,
                    v_out_head,
                    v_old_qlen,
                    v_old_qstart,
                    TOTAL_Q,
                    OR,
                    OH,
                )
                sync_block_set(
                    "vector", "cube", 9 + v_out_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                )


# ---- prefill_clipped ----


@triton.jit
def ascend_attention_prefill_clipped(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    TABLE,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SEQLEN_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    QB: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    OB: tl.constexpr,
    KP: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    VARLEN: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    SPLITS: tl.constexpr = 1,
    CONTIG_K: tl.constexpr = 0,
    QCAP: tl.constexpr = 0,
    ACC_UB_ROWS: tl.constexpr = 0,
    Q_DIRECT: tl.constexpr = False,
    SINGLE_HEAD_INDEX: tl.constexpr = False,
    TASK_M: tl.constexpr = 0,
    SKIP_SHORT_KV: tl.constexpr = 0,
    REQUEST_IDS=None,
    REQUEST_COUNT=None,
    REQUEST_LIST: tl.constexpr = False,
    COMPACT_Q1: tl.constexpr = False,
    FINAL_O=None,
    FINAL_LSE=None,
    FINAL_OR: tl.constexpr = 0,
    FINAL_OH: tl.constexpr = 0,
    FINAL_TOTAL_Q: tl.constexpr = 0,
    BYPASS_SHORT: tl.constexpr = False,
    MERGE_O=None,
    MERGE_LSE=None,
    MERGE_IDS=None,
    MERGE_COUNT=None,
    MERGE_SPLITS: tl.constexpr = 0,
    CROSS_Q_TASK: tl.constexpr = False,
    BODY_TASK_PIPELINE: tl.constexpr = False,
    TASK_SCORES=None,
    TASK_PROBS=None,
):
    tl.static_assert(
        B == 1
        and VARLEN
        and CAUSAL
        and (BM == 128)
        and (BN == 256)
        and (BS == 512)
        and (SPLITS == 1)
    )
    tl.static_assert(
        TASK_M == 0
        and (not REQUEST_LIST)
        and (not COMPACT_Q1)
        and (not MERGE_SPLITS)
        and (not CROSS_Q_TASK)
        and (not BODY_TASK_PIPELINE)
    )
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    qstart = tl.load(CUQ)
    qlen = tl.load(CUQ + 1) - qstart
    klen = tl.load(USED)
    supported = (
        (qlen > 64)
        & (klen >= qlen)
        & (klen <= 1536)
        & (qstart >= 0)
        & (qstart + qlen <= TOTAL_Q)
    )
    if supported:
        total_tasks = tl.cdiv(qlen, 128) * H
        _attention_prefill_clipped_queue(
            Q,
            Q_RAW,
            K,
            V,
            O,
            LSE,
            CUQ,
            USED,
            REQUEST_IDS,
            TASK_SCORES,
            TASK_PROBS,
            PVWORK,
            ACCWORK,
            total_tasks,
            workers,
            worker,
            H,
            HK,
            TOTAL_Q,
            SCALE,
            QR,
            QH,
            OR,
            OH,
            CONTIG_K,
            QCAP,
            Q_DIRECT,
            False,
            0,
        )
    else:
        ascend_attention_persistent_pipelined(
            Q,
            Q_RAW,
            K,
            V,
            O,
            LSE,
            CUQ,
            USED,
            TABLE,
            SCORES,
            PROBS,
            PVWORK,
            ACCWORK,
            B,
            H,
            HK,
            TOTAL_Q,
            SEQLEN_Q,
            SCALE,
            QR,
            QH,
            QB,
            OR,
            OH,
            OB,
            KP,
            KR,
            KH,
            VP,
            VR,
            VH,
            TS,
            PAGE,
            VARLEN,
            CAUSAL,
            BM,
            BN,
            BS,
            SPLITS,
            CONTIG_K,
            QCAP,
            ACC_UB_ROWS,
            Q_DIRECT,
            SINGLE_HEAD_INDEX,
            TASK_M,
            SKIP_SHORT_KV,
            REQUEST_IDS,
            REQUEST_COUNT,
            REQUEST_LIST,
            COMPACT_Q1,
            FINAL_O,
            FINAL_LSE,
            FINAL_OR,
            FINAL_OH,
            FINAL_TOTAL_Q,
            BYPASS_SHORT,
            MERGE_O,
            MERGE_LSE,
            MERGE_IDS,
            MERGE_COUNT,
            MERGE_SPLITS,
            CROSS_Q_TASK,
            BODY_TASK_PIPELINE,
            TASK_SCORES,
            TASK_PROBS,
        )


# ---- prefill_clipped ----


@triton.jit
def _prefill_softmax_valid_rows(
    SCORES,
    PROBS,
    STATE,
    limit,
    qt,
    qlen,
    klen,
    SCALE: tl.constexpr,
    WIDTH: tl.constexpr,
    VM: tl.constexpr,
):
    vf = sub_vec_id().to(tl.int32) * 64
    cols = tl.arange(0, WIDTH)
    write_limit = tl.cdiv(limit, 256) * 256
    valid_rows = tl.minimum(64, tl.maximum(0, qlen - qt * 128 - vf))
    for vs in range(0, valid_rows, VM):
        rows = vf + vs + tl.arange(0, VM)
        offsets = rows[:, None] * 1536 + cols[None, :]
        scores = (
            tl.load(SCORES + offsets, cols[None, :] < limit, 0, volatile=True) * SCALE
        )
        qrows = qt * 128 + rows
        end = tl.minimum(limit, qrows + klen - qlen + 1)
        end = tl.where(qrows < qlen, end, 0)
        scores = tl.where(
            cols[None, :].to(tl.float32) < end[:, None].to(tl.float32),
            scores,
            -float("inf"),
        )
        m = tl.max(scores, 1)
        safe = tl.where(m == -float("inf"), 0.0, m)
        p = tl.exp(scores - safe[:, None])
        denom = tl.sum(p, 1)
        tl.store(
            PROBS + offsets, p.to(PROBS.dtype.element_ty), cols[None, :] < write_limit
        )
        tl.store(STATE + rows, m)
        tl.store(STATE + 128 + rows, denom)


# ---- balanced_prefill ----


@triton.jit
def _attention_prefill_balanced_queue(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    REQUEST_IDS,
    TASK_SCORES,
    TASK_PROBS,
    PVWORK,
    ACCWORK,
    total_tasks,
    workers,
    worker,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    CONTIG_K: tl.constexpr,
    QCAP: tl.constexpr,
    Q_DIRECT: tl.constexpr,
    REQUEST_LIST: tl.constexpr,
    SKIP_SHORT_KV: tl.constexpr,
):
    queue_qstart = tl.load(CUQ)
    queue_qlen = tl.load(CUQ + 1) - queue_qstart
    queue_klen = tl.load(USED)
    worker_id = worker.to(tl.int32)
    worker = worker.to(tl.int64)
    group: tl.constexpr = H // HK
    query_buf = tle.dsa.alloc([128, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([256, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([128, 256], TASK_PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    count = tl.cdiv(tl.maximum(total_tasks - worker_id, 0), workers)
    old_begin = 0
    old_batch = 0
    old_qlen = 0
    old_klen = 0
    old_qstart = 0
    middle_begin = 0
    middle_qlen = 0
    middle_klen = 0
    current_begin = 0
    current_batch = 0
    current_qlen = 0
    current_klen = 0
    current_qstart = 0
    for tick in range(count + 2):
        if tick >= 2:
            old_index = tick - 2
            old_slot = old_index % 3
            _, old_begin, _, old_batch, old_qlen, old_klen, old_qstart = (
                0,
                0,
                total_tasks,
                0,
                queue_qlen,
                queue_klen,
                queue_qstart,
            )
            old_task = worker_id + old_index * workers - old_begin
            old_qt = old_task // H
            if H == 4:
                q_tiles = tl.cdiv(queue_qlen, 128)
                first_wave = tl.minimum(q_tiles, tl.maximum(1, workers // H))
                old_qt = tl.where(
                    old_qt < first_wave, q_tiles - 1 - old_qt, old_qt - first_wave
                )
            # Bijective 9-tile schedule lowers the longest per-core QK/PV load.
            if H == 8:
                if tl.cdiv(queue_qlen, 128) == 9:
                    old_qt = (old_qt + 4) % 9
            elif H == 16:
                if tl.cdiv(queue_qlen, 128) == 9:
                    old_qt = (old_qt * 4 + 8) % 9
            old_head = old_task % H
            old_kh = old_head // group
            old_limit = tl.minimum(
                old_klen,
                tl.maximum(
                    0, tl.minimum(old_qlen, old_qt * 128 + 128) + old_klen - old_qlen
                ),
            )
            old_prob = TASK_PROBS + (worker * 3 + old_slot) * 128 * 1536
            old_pv = PVWORK + (worker * 3 + old_slot) * 128 * 256
            sync_block_wait("vector", "cube", 3 + old_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
            _whole_task_pv(
                V,
                old_prob,
                old_pv,
                value_buf,
                prob_buf,
                old_batch,
                old_kh,
                old_limit,
                HK,
                CONTIG_K,
            )
            sync_block_set("cube", "vector", 6 + old_slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
        if (tick >= 1) & (tick < count + 1):
            middle_index = tick - 1
            middle_slot = middle_index % 3
            _, middle_begin, _, _, middle_qlen, middle_klen, _ = (
                0,
                0,
                total_tasks,
                0,
                queue_qlen,
                queue_klen,
                queue_qstart,
            )
            middle_task = worker_id + middle_index * workers - middle_begin
            middle_qt = middle_task // H
            if H == 4:
                q_tiles = tl.cdiv(queue_qlen, 128)
                first_wave = tl.minimum(q_tiles, tl.maximum(1, workers // H))
                middle_qt = tl.where(
                    middle_qt < first_wave,
                    q_tiles - 1 - middle_qt,
                    middle_qt - first_wave,
                )
            # Bijective 9-tile schedule lowers the longest per-core QK/PV load.
            if H == 8:
                if tl.cdiv(queue_qlen, 128) == 9:
                    middle_qt = (middle_qt + 4) % 9
            elif H == 16:
                if tl.cdiv(queue_qlen, 128) == 9:
                    middle_qt = (middle_qt * 4 + 8) % 9
            middle_limit = tl.minimum(
                middle_klen,
                tl.maximum(
                    0,
                    tl.minimum(middle_qlen, middle_qt * 128 + 128)
                    + middle_klen
                    - middle_qlen,
                ),
            )
            middle_score = TASK_SCORES + (worker * 3 + middle_slot) * 128 * 1536
            middle_prob = TASK_PROBS + (worker * 3 + middle_slot) * 128 * 1536
            middle_state = ACCWORK + worker * 128 * 256 + middle_slot * 256
            sync_block_wait("cube", "vector", middle_slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            with tle.scope(core_mode="vector"):
                if middle_limit <= 256:
                    _prefill_softmax_valid_rows(
                        middle_score,
                        middle_prob,
                        middle_state,
                        middle_limit,
                        middle_qt,
                        middle_qlen,
                        middle_klen,
                        SCALE,
                        256,
                        32,
                    )
                elif middle_limit <= 512:
                    _prefill_softmax_valid_rows(
                        middle_score,
                        middle_prob,
                        middle_state,
                        middle_limit,
                        middle_qt,
                        middle_qlen,
                        middle_klen,
                        SCALE,
                        512,
                        16,
                    )
                elif middle_limit <= 1024:
                    _prefill_softmax_valid_rows(
                        middle_score,
                        middle_prob,
                        middle_state,
                        middle_limit,
                        middle_qt,
                        middle_qlen,
                        middle_klen,
                        SCALE,
                        1024,
                        8,
                    )
                else:
                    _prefill_softmax_valid_rows(
                        middle_score,
                        middle_prob,
                        middle_state,
                        middle_limit,
                        middle_qt,
                        middle_qlen,
                        middle_klen,
                        SCALE,
                        2048,
                        4,
                    )
            sync_block_set(
                "vector", "cube", 3 + middle_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
            )
        if tick < count:
            slot = tick % 3
            (
                _,
                current_begin,
                _,
                current_batch,
                current_qlen,
                current_klen,
                current_qstart,
            ) = (0, 0, total_tasks, 0, queue_qlen, queue_klen, queue_qstart)
            task = worker_id + tick * workers - current_begin
            qt = task // H
            if H == 4:
                q_tiles = tl.cdiv(queue_qlen, 128)
                first_wave = tl.minimum(q_tiles, tl.maximum(1, workers // H))
                qt = tl.where(qt < first_wave, q_tiles - 1 - qt, qt - first_wave)
            # Bijective 9-tile schedule lowers the longest per-core QK/PV load.
            if H == 8:
                if tl.cdiv(queue_qlen, 128) == 9:
                    qt = (qt + 4) % 9
            elif H == 16:
                if tl.cdiv(queue_qlen, 128) == 9:
                    qt = (qt * 4 + 8) % 9
            head = task % H
            task = qt * H + head
            kh = head // group
            limit = tl.minimum(
                current_klen,
                tl.maximum(
                    0,
                    tl.minimum(current_qlen, qt * 128 + 128)
                    + current_klen
                    - current_qlen,
                ),
            )
            score = TASK_SCORES + (worker * 3 + slot) * 128 * 1536
            if tick >= 3:
                sync_block_wait("vector", "cube", 9 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
            _whole_task_qk(
                Q,
                Q_RAW,
                K,
                score,
                query_buf,
                key_buf_0,
                key_buf_1,
                current_batch,
                task,
                qt,
                head,
                kh,
                current_qstart,
                current_qlen,
                limit,
                HK,
                QR,
                QH,
                CONTIG_K,
                QCAP,
                Q_DIRECT,
            )
            sync_block_set("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
        if tick >= 2:
            out_index = tick - 2
            out_slot = out_index % 3
            out_task = worker_id + out_index * workers - old_begin
            out_qt = out_task // H
            if H == 4:
                q_tiles = tl.cdiv(queue_qlen, 128)
                first_wave = tl.minimum(q_tiles, tl.maximum(1, workers // H))
                out_qt = tl.where(
                    out_qt < first_wave, q_tiles - 1 - out_qt, out_qt - first_wave
                )
            # Bijective 9-tile schedule lowers the longest per-core QK/PV load.
            if H == 8:
                if tl.cdiv(queue_qlen, 128) == 9:
                    out_qt = (out_qt + 4) % 9
            elif H == 16:
                if tl.cdiv(queue_qlen, 128) == 9:
                    out_qt = (out_qt * 4 + 8) % 9
            out_head = out_task % H
            out_pv = PVWORK + (worker * 3 + out_slot) * 128 * 256
            out_state = ACCWORK + worker * 128 * 256 + out_slot * 256
            sync_block_wait("cube", "vector", 6 + out_slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            _whole_task_finish(
                O,
                LSE,
                out_pv,
                out_state,
                out_qt,
                out_head,
                old_qlen,
                old_qstart,
                TOTAL_Q,
                OR,
                OH,
            )
            sync_block_set("vector", "cube", 9 + out_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    for drain in range(tl.maximum(0, count - 3), count):
        sync_block_wait("vector", "cube", 9 + drain % 3, PIPE.PIPE_MTE3, PIPE.PIPE_S)


# ---- balanced_prefill ----


@triton.jit
def ascend_attention_prefill_balanced(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    TABLE,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SEQLEN_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    QB: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    OB: tl.constexpr,
    KP: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    VARLEN: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    SPLITS: tl.constexpr = 1,
    CONTIG_K: tl.constexpr = 0,
    QCAP: tl.constexpr = 0,
    ACC_UB_ROWS: tl.constexpr = 0,
    Q_DIRECT: tl.constexpr = False,
    SINGLE_HEAD_INDEX: tl.constexpr = False,
    TASK_M: tl.constexpr = 0,
    SKIP_SHORT_KV: tl.constexpr = 0,
    REQUEST_IDS=None,
    REQUEST_COUNT=None,
    REQUEST_LIST: tl.constexpr = False,
    COMPACT_Q1: tl.constexpr = False,
    FINAL_O=None,
    FINAL_LSE=None,
    FINAL_OR: tl.constexpr = 0,
    FINAL_OH: tl.constexpr = 0,
    FINAL_TOTAL_Q: tl.constexpr = 0,
    BYPASS_SHORT: tl.constexpr = False,
    MERGE_O=None,
    MERGE_LSE=None,
    MERGE_IDS=None,
    MERGE_COUNT=None,
    MERGE_SPLITS: tl.constexpr = 0,
    CROSS_Q_TASK: tl.constexpr = False,
    BODY_TASK_PIPELINE: tl.constexpr = False,
    TASK_SCORES=None,
    TASK_PROBS=None,
):
    tl.static_assert(
        B == 1
        and VARLEN
        and CAUSAL
        and (BM == 128)
        and (BN == 256)
        and (BS == 512)
        and (SPLITS == 1)
    )
    tl.static_assert(
        TASK_M == 0
        and (not REQUEST_LIST)
        and (not COMPACT_Q1)
        and (not MERGE_SPLITS)
        and (not CROSS_Q_TASK)
        and (not BODY_TASK_PIPELINE)
    )
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    qstart = tl.load(CUQ)
    qlen = tl.load(CUQ + 1) - qstart
    klen = tl.load(USED)
    supported = (
        (qlen > 64)
        & (klen >= qlen)
        & (klen <= 1536)
        & (qstart >= 0)
        & (qstart + qlen <= TOTAL_Q)
    )
    if supported:
        total_tasks = tl.cdiv(qlen, 128) * H
        _attention_prefill_balanced_queue(
            Q,
            Q_RAW,
            K,
            V,
            O,
            LSE,
            CUQ,
            USED,
            REQUEST_IDS,
            TASK_SCORES,
            TASK_PROBS,
            PVWORK,
            ACCWORK,
            total_tasks,
            workers,
            worker,
            H,
            HK,
            TOTAL_Q,
            SCALE,
            QR,
            QH,
            OR,
            OH,
            CONTIG_K,
            QCAP,
            Q_DIRECT,
            False,
            0,
        )
    else:
        ascend_attention_persistent_pipelined(
            Q,
            Q_RAW,
            K,
            V,
            O,
            LSE,
            CUQ,
            USED,
            TABLE,
            SCORES,
            PROBS,
            PVWORK,
            ACCWORK,
            B,
            H,
            HK,
            TOTAL_Q,
            SEQLEN_Q,
            SCALE,
            QR,
            QH,
            QB,
            OR,
            OH,
            OB,
            KP,
            KR,
            KH,
            VP,
            VR,
            VH,
            TS,
            PAGE,
            VARLEN,
            CAUSAL,
            BM,
            BN,
            BS,
            SPLITS,
            CONTIG_K,
            QCAP,
            ACC_UB_ROWS,
            Q_DIRECT,
            SINGLE_HEAD_INDEX,
            TASK_M,
            SKIP_SHORT_KV,
            REQUEST_IDS,
            REQUEST_COUNT,
            REQUEST_LIST,
            COMPACT_Q1,
            FINAL_O,
            FINAL_LSE,
            FINAL_OR,
            FINAL_OH,
            FINAL_TOTAL_Q,
            BYPASS_SHORT,
            MERGE_O,
            MERGE_LSE,
            MERGE_IDS,
            MERGE_COUNT,
            MERGE_SPLITS,
            CROSS_Q_TASK,
            BODY_TASK_PIPELINE,
            TASK_SCORES,
            TASK_PROBS,
        )


# ---- prefill_corequeue ----


@triton.jit
def _attention_prefill_corequeue(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    REQUEST_IDS,
    TASK_SCORES,
    TASK_PROBS,
    PVWORK,
    ACCWORK,
    total_tasks,
    workers,
    worker,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    CONTIG_K: tl.constexpr,
    QCAP: tl.constexpr,
    Q_DIRECT: tl.constexpr,
    REQUEST_LIST: tl.constexpr,
    SKIP_SHORT_KV: tl.constexpr,
):
    group: tl.constexpr = H // HK
    query_buf = tle.dsa.alloc([128, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([256, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([128, 256], TASK_PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    with tle.scope(core_mode="cube"):
        c_worker = tl.program_id(0).to(tl.int64)
        c_workers = tl.num_programs(0)
        c_queue_qstart = tl.load(CUQ)
        c_queue_qlen = tl.load(CUQ + 1) - c_queue_qstart
        c_queue_klen = tl.load(USED)
        c_worker_id = c_worker.to(tl.int32)
        c_total_tasks = tl.cdiv(c_queue_qlen, 128) * H
        c_count = tl.cdiv(tl.maximum(c_total_tasks - c_worker_id, 0), c_workers)
        c_old_begin = 0
        c_old_batch = 0
        c_old_qlen = 0
        c_old_klen = 0
        _ = 0
        _ = 0
        _ = 0
        _ = 0
        c_current_begin = 0
        c_current_batch = 0
        c_current_qlen = 0
        c_current_klen = 0
        c_current_qstart = 0
        for c_tick in range(c_count + 2):
            if c_tick >= 2:
                c_old_index = c_tick - 2
                c_old_slot = c_old_index % 3
                (
                    _,
                    c_old_begin,
                    _,
                    c_old_batch,
                    c_old_qlen,
                    c_old_klen,
                    _,
                ) = (0, 0, c_total_tasks, 0, c_queue_qlen, c_queue_klen, c_queue_qstart)
                c_old_task = c_worker_id + c_old_index * c_workers - c_old_begin
                c_old_qt = c_old_task // H
                if H == 4:
                    c_q_tiles = tl.cdiv(c_queue_qlen, 128)
                    c_first_wave = tl.minimum(c_q_tiles, tl.maximum(1, c_workers // H))
                    c_old_qt = tl.where(
                        c_old_qt < c_first_wave,
                        c_q_tiles - 1 - c_old_qt,
                        c_old_qt - c_first_wave,
                    )
                c_old_head = c_old_task % H
                c_old_kh = c_old_head // group
                c_old_limit = tl.minimum(
                    c_old_klen,
                    tl.maximum(
                        0,
                        tl.minimum(c_old_qlen, c_old_qt * 128 + 128)
                        + c_old_klen
                        - c_old_qlen,
                    ),
                )
                c_old_prob = TASK_PROBS + (c_worker * 3 + c_old_slot) * 128 * 1536
                c_old_pv = PVWORK + (c_worker * 3 + c_old_slot) * 128 * 256
                sync_block_wait(
                    "vector", "cube", 3 + c_old_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                )
                _whole_task_pv(
                    V,
                    c_old_prob,
                    c_old_pv,
                    value_buf,
                    prob_buf,
                    c_old_batch,
                    c_old_kh,
                    c_old_limit,
                    HK,
                    CONTIG_K,
                )
                sync_block_set(
                    "cube", "vector", 6 + c_old_slot, PIPE.PIPE_FIX, PIPE.PIPE_S
                )
            if c_tick < c_count:
                c_slot = c_tick % 3
                (
                    _,
                    c_current_begin,
                    _,
                    c_current_batch,
                    c_current_qlen,
                    c_current_klen,
                    c_current_qstart,
                ) = (0, 0, c_total_tasks, 0, c_queue_qlen, c_queue_klen, c_queue_qstart)
                c_task = c_worker_id + c_tick * c_workers - c_current_begin
                c_qt = c_task // H
                if H == 4:
                    c_q_tiles = tl.cdiv(c_queue_qlen, 128)
                    c_first_wave = tl.minimum(c_q_tiles, tl.maximum(1, c_workers // H))
                    c_qt = tl.where(
                        c_qt < c_first_wave, c_q_tiles - 1 - c_qt, c_qt - c_first_wave
                    )
                c_head = c_task % H
                if H == 4:
                    c_task = c_qt * H + c_head
                c_kh = c_head // group
                c_limit = tl.minimum(
                    c_current_klen,
                    tl.maximum(
                        0,
                        tl.minimum(c_current_qlen, c_qt * 128 + 128)
                        + c_current_klen
                        - c_current_qlen,
                    ),
                )
                c_score = TASK_SCORES + (c_worker * 3 + c_slot) * 128 * 1536
                if c_tick >= 3:
                    sync_block_wait(
                        "vector", "cube", 9 + c_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                    )
                _whole_task_qk(
                    Q,
                    Q_RAW,
                    K,
                    c_score,
                    query_buf,
                    key_buf_0,
                    key_buf_1,
                    c_current_batch,
                    c_task,
                    c_qt,
                    c_head,
                    c_kh,
                    c_current_qstart,
                    c_current_qlen,
                    c_limit,
                    HK,
                    QR,
                    QH,
                    CONTIG_K,
                    QCAP,
                    Q_DIRECT,
                )
                sync_block_set("cube", "vector", c_slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
        for c_drain in range(tl.maximum(0, c_count - 3), c_count):
            sync_block_wait(
                "vector", "cube", 9 + c_drain % 3, PIPE.PIPE_MTE3, PIPE.PIPE_S
            )
    with tle.scope(core_mode="vector"):
        v_worker = tl.program_id(0).to(tl.int64)
        v_workers = tl.num_programs(0)
        v_queue_qstart = tl.load(CUQ)
        v_queue_qlen = tl.load(CUQ + 1) - v_queue_qstart
        v_queue_klen = tl.load(USED)
        v_worker_id = v_worker.to(tl.int32)
        v_total_tasks = tl.cdiv(v_queue_qlen, 128) * H
        v_count = tl.cdiv(tl.maximum(v_total_tasks - v_worker_id, 0), v_workers)
        v_old_begin = 0
        _ = 0
        v_old_qlen = v_queue_qlen
        _ = v_queue_klen
        v_old_qstart = v_queue_qstart
        v_middle_begin = 0
        v_middle_qlen = 0
        v_middle_klen = 0
        _ = 0
        _ = 0
        _ = 0
        _ = 0
        _ = 0
        for v_tick in range(v_count + 2):
            if (v_tick >= 1) & (v_tick < v_count + 1):
                v_middle_index = v_tick - 1
                v_middle_slot = v_middle_index % 3
                _, v_middle_begin, _, _, v_middle_qlen, v_middle_klen, _ = (
                    0,
                    0,
                    v_total_tasks,
                    0,
                    v_queue_qlen,
                    v_queue_klen,
                    v_queue_qstart,
                )
                v_middle_task = (
                    v_worker_id + v_middle_index * v_workers - v_middle_begin
                )
                v_middle_qt = v_middle_task // H
                if H == 4:
                    v_q_tiles = tl.cdiv(v_queue_qlen, 128)
                    v_first_wave = tl.minimum(v_q_tiles, tl.maximum(1, v_workers // H))
                    v_middle_qt = tl.where(
                        v_middle_qt < v_first_wave,
                        v_q_tiles - 1 - v_middle_qt,
                        v_middle_qt - v_first_wave,
                    )
                v_middle_limit = tl.minimum(
                    v_middle_klen,
                    tl.maximum(
                        0,
                        tl.minimum(v_middle_qlen, v_middle_qt * 128 + 128)
                        + v_middle_klen
                        - v_middle_qlen,
                    ),
                )
                v_middle_score = (
                    TASK_SCORES + (v_worker * 3 + v_middle_slot) * 128 * 1536
                )
                v_middle_prob = TASK_PROBS + (v_worker * 3 + v_middle_slot) * 128 * 1536
                v_middle_state = ACCWORK + v_worker * 128 * 256 + v_middle_slot * 256
                sync_block_wait(
                    "cube", "vector", v_middle_slot, PIPE.PIPE_FIX, PIPE.PIPE_S
                )
                with tle.scope(core_mode="vector"):
                    if v_middle_limit <= 256:
                        _whole_task_softmax(
                            v_middle_score,
                            v_middle_prob,
                            v_middle_state,
                            v_middle_limit,
                            v_middle_qt,
                            v_middle_qlen,
                            v_middle_klen,
                            SCALE,
                            256,
                            32,
                        )
                    elif v_middle_limit <= 512:
                        _whole_task_softmax(
                            v_middle_score,
                            v_middle_prob,
                            v_middle_state,
                            v_middle_limit,
                            v_middle_qt,
                            v_middle_qlen,
                            v_middle_klen,
                            SCALE,
                            512,
                            16,
                        )
                    elif v_middle_limit <= 1024:
                        _whole_task_softmax(
                            v_middle_score,
                            v_middle_prob,
                            v_middle_state,
                            v_middle_limit,
                            v_middle_qt,
                            v_middle_qlen,
                            v_middle_klen,
                            SCALE,
                            1024,
                            8,
                        )
                    else:
                        _whole_task_softmax(
                            v_middle_score,
                            v_middle_prob,
                            v_middle_state,
                            v_middle_limit,
                            v_middle_qt,
                            v_middle_qlen,
                            v_middle_klen,
                            SCALE,
                            2048,
                            4,
                        )
                sync_block_set(
                    "vector", "cube", 3 + v_middle_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                )
            if v_tick >= 2:
                v_out_index = v_tick - 2
                v_out_slot = v_out_index % 3
                v_out_task = v_worker_id + v_out_index * v_workers - v_old_begin
                v_out_qt = v_out_task // H
                if H == 4:
                    v_q_tiles = tl.cdiv(v_queue_qlen, 128)
                    v_first_wave = tl.minimum(v_q_tiles, tl.maximum(1, v_workers // H))
                    v_out_qt = tl.where(
                        v_out_qt < v_first_wave,
                        v_q_tiles - 1 - v_out_qt,
                        v_out_qt - v_first_wave,
                    )
                v_out_head = v_out_task % H
                v_out_pv = PVWORK + (v_worker * 3 + v_out_slot) * 128 * 256
                v_out_state = ACCWORK + v_worker * 128 * 256 + v_out_slot * 256
                sync_block_wait(
                    "cube", "vector", 6 + v_out_slot, PIPE.PIPE_FIX, PIPE.PIPE_S
                )
                _whole_task_finish(
                    O,
                    LSE,
                    v_out_pv,
                    v_out_state,
                    v_out_qt,
                    v_out_head,
                    v_old_qlen,
                    v_old_qstart,
                    TOTAL_Q,
                    OR,
                    OH,
                )
                sync_block_set(
                    "vector", "cube", 9 + v_out_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                )


# ---- prefill_corequeue ----


@triton.jit
def ascend_attention_prefill_corequeue(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    TABLE,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SEQLEN_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    QB: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    OB: tl.constexpr,
    KP: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    VARLEN: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    SPLITS: tl.constexpr = 1,
    CONTIG_K: tl.constexpr = 0,
    QCAP: tl.constexpr = 0,
    ACC_UB_ROWS: tl.constexpr = 0,
    Q_DIRECT: tl.constexpr = False,
    SINGLE_HEAD_INDEX: tl.constexpr = False,
    TASK_M: tl.constexpr = 0,
    SKIP_SHORT_KV: tl.constexpr = 0,
    REQUEST_IDS=None,
    REQUEST_COUNT=None,
    REQUEST_LIST: tl.constexpr = False,
    COMPACT_Q1: tl.constexpr = False,
    FINAL_O=None,
    FINAL_LSE=None,
    FINAL_OR: tl.constexpr = 0,
    FINAL_OH: tl.constexpr = 0,
    FINAL_TOTAL_Q: tl.constexpr = 0,
    BYPASS_SHORT: tl.constexpr = False,
    MERGE_O=None,
    MERGE_LSE=None,
    MERGE_IDS=None,
    MERGE_COUNT=None,
    MERGE_SPLITS: tl.constexpr = 0,
    CROSS_Q_TASK: tl.constexpr = False,
    BODY_TASK_PIPELINE: tl.constexpr = False,
    TASK_SCORES=None,
    TASK_PROBS=None,
):
    tl.static_assert(
        B == 1
        and VARLEN
        and CAUSAL
        and (BM == 128)
        and (BN == 256)
        and (BS == 512)
        and (SPLITS == 1)
    )
    tl.static_assert(
        TASK_M == 0
        and (not REQUEST_LIST)
        and (not COMPACT_Q1)
        and (not MERGE_SPLITS)
        and (not CROSS_Q_TASK)
        and (not BODY_TASK_PIPELINE)
    )
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    qstart = tl.load(CUQ)
    qlen = tl.load(CUQ + 1) - qstart
    klen = tl.load(USED)
    supported = (
        (qlen > 64)
        & (klen >= qlen)
        & (klen <= 1536)
        & (qstart >= 0)
        & (qstart + qlen <= TOTAL_Q)
    )
    if supported:
        total_tasks = tl.cdiv(qlen, 128) * H
        _attention_prefill_corequeue(
            Q,
            Q_RAW,
            K,
            V,
            O,
            LSE,
            CUQ,
            USED,
            REQUEST_IDS,
            TASK_SCORES,
            TASK_PROBS,
            PVWORK,
            ACCWORK,
            total_tasks,
            workers,
            worker,
            H,
            HK,
            TOTAL_Q,
            SCALE,
            QR,
            QH,
            OR,
            OH,
            CONTIG_K,
            QCAP,
            Q_DIRECT,
            False,
            0,
        )
    else:
        ascend_attention_persistent_pipelined(
            Q,
            Q_RAW,
            K,
            V,
            O,
            LSE,
            CUQ,
            USED,
            TABLE,
            SCORES,
            PROBS,
            PVWORK,
            ACCWORK,
            B,
            H,
            HK,
            TOTAL_Q,
            SEQLEN_Q,
            SCALE,
            QR,
            QH,
            QB,
            OR,
            OH,
            OB,
            KP,
            KR,
            KH,
            VP,
            VR,
            VH,
            TS,
            PAGE,
            VARLEN,
            CAUSAL,
            BM,
            BN,
            BS,
            SPLITS,
            CONTIG_K,
            QCAP,
            ACC_UB_ROWS,
            Q_DIRECT,
            SINGLE_HEAD_INDEX,
            TASK_M,
            SKIP_SHORT_KV,
            REQUEST_IDS,
            REQUEST_COUNT,
            REQUEST_LIST,
            COMPACT_Q1,
            FINAL_O,
            FINAL_LSE,
            FINAL_OR,
            FINAL_OH,
            FINAL_TOTAL_Q,
            BYPASS_SHORT,
            MERGE_O,
            MERGE_LSE,
            MERGE_IDS,
            MERGE_COUNT,
            MERGE_SPLITS,
            CROSS_Q_TASK,
            BODY_TASK_PIPELINE,
            TASK_SCORES,
            TASK_PROBS,
        )


# ---- prefill_task_queue ----


@triton.jit
def _attention_prefill_task_queue(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    REQUEST_IDS,
    TASK_SCORES,
    TASK_PROBS,
    PVWORK,
    ACCWORK,
    total_tasks,
    workers,
    worker,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    CONTIG_K: tl.constexpr,
    QCAP: tl.constexpr,
    Q_DIRECT: tl.constexpr,
    REQUEST_LIST: tl.constexpr,
    SKIP_SHORT_KV: tl.constexpr,
):
    queue_qstart = tl.load(CUQ)
    queue_qlen = tl.load(CUQ + 1) - queue_qstart
    queue_klen = tl.load(USED)
    worker_id = worker.to(tl.int32)
    worker = worker.to(tl.int64)
    group: tl.constexpr = H // HK
    query_buf = tle.dsa.alloc([128, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([256, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([128, 256], TASK_PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    count = tl.cdiv(tl.maximum(total_tasks - worker_id, 0), workers)
    old_begin = 0
    old_batch = 0
    old_qlen = 0
    old_klen = 0
    old_qstart = 0
    middle_begin = 0
    middle_qlen = 0
    middle_klen = 0
    current_begin = 0
    current_batch = 0
    current_qlen = 0
    current_klen = 0
    current_qstart = 0
    for tick in range(count + 2):
        if tick >= 2:
            old_index = tick - 2
            old_slot = old_index % 3
            (
                _,
                old_begin,
                _,
                old_batch,
                old_qlen,
                old_klen,
                old_qstart,
            ) = (0, 0, total_tasks, 0, queue_qlen, queue_klen, queue_qstart)
            old_task = worker_id + old_index * workers - old_begin
            old_qt = old_task // H
            if H == 4:
                q_tiles = tl.cdiv(queue_qlen, 128)
                first_wave = tl.minimum(q_tiles, tl.maximum(1, workers // H))
                old_qt = tl.where(
                    old_qt < first_wave,
                    q_tiles - 1 - old_qt,
                    old_qt - first_wave,
                )
            old_head = old_task % H
            old_kh = old_head // group
            old_limit = tl.minimum(
                old_klen,
                tl.maximum(
                    0, tl.minimum(old_qlen, old_qt * 128 + 128) + old_klen - old_qlen
                ),
            )
            old_prob = TASK_PROBS + (worker * 3 + old_slot) * 128 * 1536
            old_pv = PVWORK + (worker * 3 + old_slot) * 128 * 256
            sync_block_wait("vector", "cube", 3 + old_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
            _whole_task_pv(
                V,
                old_prob,
                old_pv,
                value_buf,
                prob_buf,
                old_batch,
                old_kh,
                old_limit,
                HK,
                CONTIG_K,
            )
            sync_block_set("cube", "vector", 6 + old_slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
        if (tick >= 1) & (tick < count + 1):
            middle_index = tick - 1
            middle_slot = middle_index % 3
            (
                _,
                middle_begin,
                _,
                _,
                middle_qlen,
                middle_klen,
                _,
            ) = (0, 0, total_tasks, 0, queue_qlen, queue_klen, queue_qstart)
            middle_task = worker_id + middle_index * workers - middle_begin
            middle_qt = middle_task // H
            if H == 4:
                q_tiles = tl.cdiv(queue_qlen, 128)
                first_wave = tl.minimum(q_tiles, tl.maximum(1, workers // H))
                middle_qt = tl.where(
                    middle_qt < first_wave,
                    q_tiles - 1 - middle_qt,
                    middle_qt - first_wave,
                )
            middle_limit = tl.minimum(
                middle_klen,
                tl.maximum(
                    0,
                    tl.minimum(middle_qlen, middle_qt * 128 + 128)
                    + middle_klen
                    - middle_qlen,
                ),
            )
            middle_score = TASK_SCORES + (worker * 3 + middle_slot) * 128 * 1536
            middle_prob = TASK_PROBS + (worker * 3 + middle_slot) * 128 * 1536
            middle_state = ACCWORK + worker * 128 * 256 + middle_slot * 256
            sync_block_wait("cube", "vector", middle_slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            with tle.scope(core_mode="vector"):
                if middle_limit <= 256:
                    _prefill_softmax_valid_rows(
                        middle_score,
                        middle_prob,
                        middle_state,
                        middle_limit,
                        middle_qt,
                        middle_qlen,
                        middle_klen,
                        SCALE,
                        256,
                        32,
                    )
                elif middle_limit <= 512:
                    _prefill_softmax_valid_rows(
                        middle_score,
                        middle_prob,
                        middle_state,
                        middle_limit,
                        middle_qt,
                        middle_qlen,
                        middle_klen,
                        SCALE,
                        512,
                        16,
                    )
                elif middle_limit <= 1024:
                    _prefill_softmax_valid_rows(
                        middle_score,
                        middle_prob,
                        middle_state,
                        middle_limit,
                        middle_qt,
                        middle_qlen,
                        middle_klen,
                        SCALE,
                        1024,
                        8,
                    )
                else:
                    _prefill_softmax_valid_rows(
                        middle_score,
                        middle_prob,
                        middle_state,
                        middle_limit,
                        middle_qt,
                        middle_qlen,
                        middle_klen,
                        SCALE,
                        2048,
                        4,
                    )
            sync_block_set(
                "vector", "cube", 3 + middle_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
            )
        if tick < count:
            slot = tick % 3
            (
                _,
                current_begin,
                _,
                current_batch,
                current_qlen,
                current_klen,
                current_qstart,
            ) = (0, 0, total_tasks, 0, queue_qlen, queue_klen, queue_qstart)
            task = worker_id + tick * workers - current_begin
            qt = task // H
            if H == 4:
                q_tiles = tl.cdiv(queue_qlen, 128)
                first_wave = tl.minimum(q_tiles, tl.maximum(1, workers // H))
                qt = tl.where(
                    qt < first_wave,
                    q_tiles - 1 - qt,
                    qt - first_wave,
                )
            head = task % H
            if H == 4:
                task = qt * H + head
            kh = head // group
            limit = tl.minimum(
                current_klen,
                tl.maximum(
                    0,
                    tl.minimum(current_qlen, qt * 128 + 128)
                    + current_klen
                    - current_qlen,
                ),
            )
            score = TASK_SCORES + (worker * 3 + slot) * 128 * 1536
            if tick >= 3:
                sync_block_wait("vector", "cube", 9 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
            _whole_task_qk(
                Q,
                Q_RAW,
                K,
                score,
                query_buf,
                key_buf_0,
                key_buf_1,
                current_batch,
                task,
                qt,
                head,
                kh,
                current_qstart,
                current_qlen,
                limit,
                HK,
                QR,
                QH,
                CONTIG_K,
                QCAP,
                Q_DIRECT,
            )
            sync_block_set("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
        if tick >= 2:
            out_index = tick - 2
            out_slot = out_index % 3
            out_task = worker_id + out_index * workers - old_begin
            out_qt = out_task // H
            if H == 4:
                q_tiles = tl.cdiv(queue_qlen, 128)
                first_wave = tl.minimum(q_tiles, tl.maximum(1, workers // H))
                out_qt = tl.where(
                    out_qt < first_wave,
                    q_tiles - 1 - out_qt,
                    out_qt - first_wave,
                )
            out_head = out_task % H
            out_pv = PVWORK + (worker * 3 + out_slot) * 128 * 256
            out_state = ACCWORK + worker * 128 * 256 + out_slot * 256
            sync_block_wait("cube", "vector", 6 + out_slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            _whole_task_finish(
                O,
                LSE,
                out_pv,
                out_state,
                out_qt,
                out_head,
                old_qlen,
                old_qstart,
                TOTAL_Q,
                OR,
                OH,
            )
            sync_block_set("vector", "cube", 9 + out_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    for drain in range(tl.maximum(0, count - 3), count):
        sync_block_wait("vector", "cube", 9 + drain % 3, PIPE.PIPE_MTE3, PIPE.PIPE_S)


# ---- prefill_task_queue ----


@triton.jit
def ascend_attention_prefill_task_queue(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    TABLE,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SEQLEN_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    QB: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    OB: tl.constexpr,
    KP: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    VARLEN: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    SPLITS: tl.constexpr = 1,
    CONTIG_K: tl.constexpr = 0,
    QCAP: tl.constexpr = 0,
    ACC_UB_ROWS: tl.constexpr = 0,
    Q_DIRECT: tl.constexpr = False,
    SINGLE_HEAD_INDEX: tl.constexpr = False,
    TASK_M: tl.constexpr = 0,
    SKIP_SHORT_KV: tl.constexpr = 0,
    REQUEST_IDS=None,
    REQUEST_COUNT=None,
    REQUEST_LIST: tl.constexpr = False,
    COMPACT_Q1: tl.constexpr = False,
    FINAL_O=None,
    FINAL_LSE=None,
    FINAL_OR: tl.constexpr = 0,
    FINAL_OH: tl.constexpr = 0,
    FINAL_TOTAL_Q: tl.constexpr = 0,
    BYPASS_SHORT: tl.constexpr = False,
    MERGE_O=None,
    MERGE_LSE=None,
    MERGE_IDS=None,
    MERGE_COUNT=None,
    MERGE_SPLITS: tl.constexpr = 0,
    CROSS_Q_TASK: tl.constexpr = False,
    BODY_TASK_PIPELINE: tl.constexpr = False,
    TASK_SCORES=None,
    TASK_PROBS=None,
):
    tl.static_assert(
        B == 1
        and VARLEN
        and CAUSAL
        and (BM == 128)
        and (BN == 256)
        and (BS == 512)
        and (SPLITS == 1)
    )
    tl.static_assert(
        TASK_M == 0
        and (not REQUEST_LIST)
        and (not COMPACT_Q1)
        and (not MERGE_SPLITS)
        and (not CROSS_Q_TASK)
        and (not BODY_TASK_PIPELINE)
    )
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    qstart = tl.load(CUQ)
    qlen = tl.load(CUQ + 1) - qstart
    klen = tl.load(USED)
    supported = (
        (qlen > 64)
        & (klen >= qlen)
        & (klen <= 1536)
        & (qstart >= 0)
        & (qstart + qlen <= TOTAL_Q)
    )
    if supported:
        total_tasks = tl.cdiv(qlen, 128) * H
        _attention_prefill_task_queue(
            Q,
            Q_RAW,
            K,
            V,
            O,
            LSE,
            CUQ,
            USED,
            REQUEST_IDS,
            TASK_SCORES,
            TASK_PROBS,
            PVWORK,
            ACCWORK,
            total_tasks,
            workers,
            worker,
            H,
            HK,
            TOTAL_Q,
            SCALE,
            QR,
            QH,
            OR,
            OH,
            CONTIG_K,
            QCAP,
            Q_DIRECT,
            False,
            0,
        )
    else:
        ascend_attention_persistent_pipelined(
            Q,
            Q_RAW,
            K,
            V,
            O,
            LSE,
            CUQ,
            USED,
            TABLE,
            SCORES,
            PROBS,
            PVWORK,
            ACCWORK,
            B,
            H,
            HK,
            TOTAL_Q,
            SEQLEN_Q,
            SCALE,
            QR,
            QH,
            QB,
            OR,
            OH,
            OB,
            KP,
            KR,
            KH,
            VP,
            VR,
            VH,
            TS,
            PAGE,
            VARLEN,
            CAUSAL,
            BM,
            BN,
            BS,
            SPLITS,
            CONTIG_K,
            QCAP,
            ACC_UB_ROWS,
            Q_DIRECT,
            SINGLE_HEAD_INDEX,
            TASK_M,
            SKIP_SHORT_KV,
            REQUEST_IDS,
            REQUEST_COUNT,
            REQUEST_LIST,
            COMPACT_Q1,
            FINAL_O,
            FINAL_LSE,
            FINAL_OR,
            FINAL_OH,
            FINAL_TOTAL_Q,
            BYPASS_SHORT,
            MERGE_O,
            MERGE_LSE,
            MERGE_IDS,
            MERGE_COUNT,
            MERGE_SPLITS,
            CROSS_Q_TASK,
            BODY_TASK_PIPELINE,
            TASK_SCORES,
            TASK_PROBS,
        )
