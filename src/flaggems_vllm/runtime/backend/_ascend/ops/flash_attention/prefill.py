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

from flaggems_vllm.runtime.backend._ascend.ops.flash_attention.common import (
    _packed_q_base,
    _raw_stream_nowait,
    bind_runner,
    cube_core_limit,
    pack_kv_prescaled,
    packed_input_layout,
)
from flaggems_vllm.runtime.backend._ascend.ops.flash_attention.direct import (
    _attention_request_stream,
    _init_attention_accumulator,
    _init_attention_state,
    _load_attention_query,
    _online_softmax_weights,
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
    K_FIRST: tl.constexpr = False,
    V0_BASE=None,
    V0_BUF=None,
    MOVE_V0=False,
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
        if K_FIRST:
            if MOVE_V0 & (ck == 0):
                vptr = tl.make_block_ptr(
                    V0_BASE, (256, 256), (256, 1), (0, 0), (256, 256), (1, 0)
                )
                tle.dsa.copy(vptr, V0_BUF, [256, 256])
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
            _load_attention_query(
                Q,
                Q_RAW,
                query_buf,
                batch,
                qstart,
                H,
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
            _init_attention_accumulator(
                acc_buf, ACCWORK, worker.to(tl.int64), BM, 64, True
            )
            _init_attention_state(
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
        x2_newm, x2_p, x2_alpha = _online_softmax_weights(x2_score, x2_m)
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
                _load_attention_query(
                    Q,
                    Q_RAW,
                    query_buf,
                    batch,
                    qstart,
                    H,
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
                _init_attention_accumulator(
                    acc_buf, ACCWORK, worker.to(tl.int64), BM, 64, True
                )
                _init_attention_state(
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
                _load_attention_query(
                    Q,
                    Q_RAW,
                    query_buf,
                    batch,
                    qstart,
                    H,
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
                _init_attention_accumulator(
                    acc_buf, ACCWORK, worker.to(tl.int64), BM, 64, True
                )
                _init_attention_state(
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
    EARLY_SECOND_V: tl.constexpr = False,
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
        if EARLY_SECOND_V:
            v_next = tle.dsa.alloc([BN, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
            vptr = tl.make_block_ptr(
                V + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + BN) * 256,
                (BN, 256),
                (256, 1),
                (0, 0),
                (BN, 256),
                (1, 0),
            )
            tle.dsa.copy(vptr, v_next, [BN, 256])
        all_weights = tle.dsa.to_tensor(full_prob_buf, writable=False)
        weights0 = tle.dsa.extract_slice(all_weights, [0, 0], [BM, BN], [1, 1])
        value0 = tle.dsa.to_tensor(value_buf, writable=False)
        c2 = tl.dot(weights0, value0, tl.zeros((BM, 256), tl.float32))
        if not EARLY_SECOND_V:
            vptr = tl.make_block_ptr(
                V + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + BN) * 256,
                (BN, 256),
                (256, 1),
                (0, 0),
                (BN, 256),
                (1, 0),
            )
            tle.dsa.copy(vptr, value_buf, [BN, 256])
            v_next = value_buf
        weights1 = tle.dsa.extract_slice(all_weights, [0, BN], [BM, BN], [1, 1])
        value1 = tle.dsa.to_tensor(v_next, writable=False)
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
    EARLY_SECOND_V: tl.constexpr = False,
):
    BM: tl.constexpr = 128
    BN: tl.constexpr = 256
    BS: tl.constexpr = 512
    group: tl.constexpr = H // HK
    BALANCED_TASKS: tl.constexpr = TOTAL_Q >= 4096 and TOTAL_Q <= 4352
    query_buf = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    if EARLY_SECOND_V:
        key_buf_0 = None
        key_buf_1 = None
    else:
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
                _load_attention_query(
                    Q,
                    Q_RAW,
                    query_buf,
                    batch,
                    qstart,
                    H,
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
                        # K staging can reuse the L1 space of the second V tile.
                        if EARLY_SECOND_V:
                            qk_key_0 = tle.dsa.alloc(
                                [128, 256], K.dtype.element_ty, tle.dsa.ascend.L1
                            )
                            qk_key_1 = tle.dsa.alloc(
                                [128, 256], K.dtype.element_ty, tle.dsa.ascend.L1
                            )
                        else:
                            qk_key_0, qk_key_1 = key_buf_0, key_buf_1
                        _core_qk_only(
                            K,
                            scratch,
                            prob,
                            query_buf,
                            qk_key_0,
                            qk_key_1,
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
                            EARLY_SECOND_V,
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
                _init_attention_accumulator(
                    acc_buf, ACCWORK, worker.to(tl.int64), BM, 64, True
                )
                _init_attention_state(
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


_prefill_launchers = {}
_fused_4k_launchers = {}
_fused_4k_workspaces = {}
_fused_4k_workspace_lock = Lock()
_MAX_FUSED_4K_WORKSPACE_BYTES = 64 << 20


def _allocate_4k_workspace(q, q_shape, kv_shape, workers, reuse):
    device = q.device
    key = None
    if reuse and not torch.npu.is_current_stream_capturing():
        # This route consumes all seven buffers in a single launch. Reuse on
        # the same stream is ordered; a captured graph owns distinct buffers.
        current_stream = _raw_stream_nowait or driver.active.get_current_stream
        key = (device.index, current_stream(device.index))
        layout = (q_shape, kv_shape, workers, q.dtype)
        cached = _fused_4k_workspaces.get(key)
        if cached is not None and cached[0] == layout:
            return cached[2]

    qp = torch.empty(q_shape, dtype=q.dtype, device=device)
    kp = torch.empty(kv_shape, dtype=q.dtype, device=device)
    vp = torch.empty_like(kp)
    scores = torch.empty((workers, 3, 128, 512), dtype=torch.float32, device=device)
    probs = torch.empty_like(scores, dtype=q.dtype)
    pvwork = torch.empty((workers, 3, 128, 256), dtype=torch.float32, device=device)
    accwork = torch.empty((workers, 128, 256), dtype=torch.float32, device=device)
    workspace = qp, kp, vp, scores, probs, pvwork, accwork
    if key is not None:
        size = sum(x.numel() * x.element_size() for x in workspace)
        with _fused_4k_workspace_lock:
            retained = sum(
                value[1]
                for cached_key, value in _fused_4k_workspaces.items()
                if cached_key != key
            )
            if retained + size <= _MAX_FUSED_4K_WORKSPACE_BYTES:
                _fused_4k_workspaces[key] = (layout, size, workspace)
    return workspace


@triton.jit
def _prefill_4k_fused(
    Q,
    QQ,
    K,
    V,
    KP,
    VP,
    O,
    LSE,
    CUQ,
    USED,
    TABLE,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    CAPACITY: tl.constexpr,
    PAGE: tl.constexpr,
    QCAP: tl.constexpr,
    GROUP_PAGES: tl.constexpr,
):
    with tle.scope(core_mode="vector"):
        lane = tl.program_id(0) * 2 + sub_vec_id()
        lanes = tl.num_programs(0) * 2
        length = tl.load(USED)
        qstart = tl.load(CUQ)
        qlen = tl.load(CUQ + 1) - qstart
        tiled_rows: tl.constexpr = PAGE * GROUP_PAGES
        tiles = tl.cdiv(length, 512) * (512 // tiled_rows)
        d = tl.arange(0, 256)
        for tile in range(lane, tiles * HK, lanes):
            kh = tile % HK
            logical = (tile // HK) * tiled_rows + tl.arange(0, tiled_rows)
            valid = logical < length
            physical = tl.load(TABLE + logical // PAGE, valid, 0).to(tl.int64)
            raw = ((physical * PAGE + logical % PAGE) * HK + kh) * 256
            key = tl.load(K + raw[:, None] + d[None, :], valid[:, None], 0)
            value = tl.load(V + raw[:, None] + d[None, :], valid[:, None], 0)
            offset = (kh * CAPACITY + logical[:, None]) * 256 + d[None, :]
            tl.store(KP + offset, key.to(tl.float32) * 0.0625)
            tl.store(VP + offset, value)
        tail = qlen % 128
        if tail:
            for part in range(lane, H * 4, lanes):
                head = part // 4
                row = tl.arange(0, 32) + (part % 4) * 32
                batch_row = qlen // 128 * 128 + row
                q_value = tl.load(
                    Q + ((qstart + batch_row[:, None]) * H + head) * 256 + d[None, :],
                    row[:, None] < tail,
                    0,
                )
                dest = _packed_q_base(tl.full((), 0, tl.int32), qstart, QCAP, H)
                dest += (qlen // 128) * H + head
                tl.store(QQ + (dest * 128 + row[:, None]) * 256 + d[None, :], q_value)
    tle.dsa.ascend.sync_block_all("all", 15)
    ascend_attention_prefill_prob_stack(
        QQ,
        Q,
        KP,
        VP,
        O,
        LSE,
        CUQ,
        USED,
        SCORES,
        PROBS,
        PVWORK,
        ACCWORK,
        H,
        HK,
        TOTAL_Q,
        1.0,
        H * 256,
        256,
        H * 256,
        256,
        CAPACITY,
        QCAP,
        True,
    )


def try_launch_cached_prefill(params, workers):
    """Submit the existing 4K BF16 pack and compute kernels on the current stream."""
    q, k, v, o = params.q_ptr, params.k_ptr, params.v_ptr, params.o_ptr
    cu, used, table = (
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.page_table_ptr,
    )
    lse = params.softmax_lse_ptr
    if any(
        x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 32
        for x in (q, k, v, o)
    ) or any(x.dtype != torch.int32 or x.data_ptr() % 32 for x in (cu, used, table)):
        return False
    if lse is not None and (lse.dtype != torch.float32 or lse.data_ptr() % 32):
        return False
    q_shape, kv_shape, _, qcap, capacity = packed_input_layout(params)
    page, h, hk = params.block_size, params.h, params.hk
    n_pages = 4 if h == 4 and hk == 1 else 1
    wave = workers * 2 if n_pages == 4 else workers * 8 // hk
    programs = min(capacity // (n_pages * page), wave)
    if programs * hk < 4 * h:
        return False
    device = q.device
    layout = (
        *k.stride()[:3],
        *v.stride()[:3],
        params.page_table_batch_stride,
        capacity,
        page,
        hk,
        programs,
    )
    grid = (programs, 1, hk)
    compute_grid = (workers, 1, 1)
    compute_layout = (
        h,
        hk,
        params.total_q,
        1.0,
        params.q_row_stride,
        params.q_head_stride,
        params.o_row_stride,
        params.o_head_stride,
        capacity,
        qcap,
        True,
    )
    key = (device.index, grid, compute_grid, layout, compute_layout, lse is None)
    fused = h == 4 and hk == 1 and cu.is_contiguous() and workers <= 32
    qp, kp, vp, scores, probs, pvwork, accwork = _allocate_4k_workspace(
        q, q_shape, kv_shape, workers, reuse=fused
    )
    if fused:
        args = (
            q,
            qp,
            k,
            v,
            kp,
            vp,
            o,
            lse,
            cu,
            used,
            table,
            scores,
            probs,
            pvwork,
            accwork,
            h,
            hk,
            params.total_q,
            capacity,
            page,
            qcap,
            2,
        )
        runner = _fused_4k_launchers.get(key)
        if runner is not None:
            runner(args)
        else:
            compiled = _prefill_4k_fused[compute_grid](
                *args,
                disable_auto_inject_block_sync=True,
                enable_fp_fusion=False,
                multibuffer=False,
                enable_reduce_sum_strength=True,
            )
            if len(_fused_4k_launchers) < 64 and hasattr(compiled, "__getitem__"):
                _fused_4k_launchers[key] = bind_runner(
                    compiled, compute_grid, device.index
                )
        return True
    pack_args = (
        k,
        v,
        table,
        used,
        kp,
        vp,
        *layout,
        None,
        None,
        False,
        True,
        q,
        cu,
        qp,
        h,
        qcap,
        params.q_row_stride,
        params.q_head_stride,
        False,
        True,
        None,
        None,
        None,
        0,
        n_pages,
    )
    early_second_v = page == 32 and h // hk == 8 and 4096 <= params.total_q <= 4352
    compute_args = (
        qp,
        q,
        kp,
        vp,
        o,
        lse,
        cu,
        used,
        scores,
        probs,
        pvwork,
        accwork,
        *compute_layout,
        early_second_v,
    )
    runners = _prefill_launchers.get(key)
    if runners is not None:
        pack, compute = runners
        pack(pack_args)
        compute(compute_args)
    else:
        pack_options = {"multibuffer": False} if n_pages > 1 else {}
        pack = pack_kv_prescaled[grid](*pack_args, **pack_options)
        stack_options = {"enable_ubuf_saving": True} if early_second_v else {}
        compute = ascend_attention_prefill_prob_stack[compute_grid](
            *compute_args,
            disable_auto_inject_block_sync=True,
            enable_fp_fusion=False,
            multibuffer=False,
            enable_reduce_sum_strength=True,
            **stack_options,
        )
        if (
            len(_prefill_launchers) < 64
            and hasattr(pack, "__getitem__")
            and hasattr(compute, "__getitem__")
        ):
            _prefill_launchers[key] = (
                bind_runner(pack, grid, device.index),
                bind_runner(compute, compute_grid, device.index),
            )
    return True


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
                    H,
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
                H,
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
                    H,
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
                H,
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


# ---- Short Prefill schedules ----


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
