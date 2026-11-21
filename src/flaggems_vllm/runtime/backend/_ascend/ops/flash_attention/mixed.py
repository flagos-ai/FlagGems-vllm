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

from math import gcd

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.experimental.tle.language.dsa import extract_slice
from triton.experimental.tle.language.dsa.ascend import (
    PIPE,
    sub_vec_id,
    sync_block_set,
    sync_block_wait,
)

from .common import use_packed_k_prescale
from .direct import (
    _attention_body_task_queue,
    _attention_task,
    _body_queue_partition_preflight,
    _body_queue_preflight,
    _softmax_rows_dispatch,
    _softmax_stack,
    stage_q1_window,
)
from .prefill import (
    _core_finish,
    _core_init_o,
    _core_init_state,
    _core_load_q,
    _core_o,
    _core_pv,
    _core_qk_only,
    _core_softmax_only,
    _pv_after_prefetch,
    _pv_whole_probability,
)
from .scheduling import effective_kv_splits, merge_q_rows, merge_request_list

# ---- grouped_tail ----


@triton.jit
def _attention_task_tail_gqa(
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
    batch,
    task,
    split,
    worker,
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
    COMPACT_Q1: tl.constexpr = False,
    FINAL_O=None,
    FINAL_LSE=None,
    FINAL_OR: tl.constexpr = 0,
    FINAL_OH: tl.constexpr = 0,
    FINAL_TOTAL_Q: tl.constexpr = 0,
    BYPASS_SHORT: tl.constexpr = False,
    Q1_STREAM: tl.constexpr = False,
    RAW_K=None,
    RAW_V=None,
    TAIL_BATCH: tl.constexpr = 0,
):
    VM: tl.constexpr = 8 if BM == 16 else 16
    SOFTMAX_VM_256: tl.constexpr = 32 if BM == 128 else VM
    O_VM: tl.constexpr = 32 if BM == 128 and ACC_UB_ROWS == 32 else VM
    worker = worker.to(tl.int64)
    if VARLEN:
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        qbase = qstart.to(tl.int64) * QR  # noqa: F841
        obase = qstart.to(tl.int64) * OR
    else:
        # Shared mha_varlan_fwd folds GQA heads into the Q sequence for decode.
        qlen = SEQLEN_Q
        qstart = batch * SEQLEN_Q
        qbase = batch.to(tl.int64) * QB  # noqa: F841
        obase = batch.to(tl.int64) * OB
    output_qstart = qstart  # noqa: F841
    if COMPACT_Q1:
        output_qstart = batch  # noqa: F841
        obase = batch.to(tl.int64) * OR
    if SPLITS > 1:
        obase += split.to(tl.int64) * TOTAL_Q * H * 256  # noqa: F841
    klen = tl.load(USED + batch)
    write_final = False
    if BYPASS_SHORT:
        active_output_splits, _ = effective_kv_splits(klen, SPLITS, BS)
        write_final = active_output_splits == 1  # noqa: F841
    group: tl.constexpr = H // HK
    head_rows = group
    qs = tl.maximum(1, qlen % 128)
    sub = 0
    qt = qlen // 128
    kh = task
    headstart = 0
    query_buf = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([BN, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([BM, BN], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    state_buf = tle.dsa.alloc([5, BM // 2], tl.float32, tle.dsa.ascend.UB)
    row_end_buf = None
    if BM == 128:
        row_end_buf = tle.dsa.alloc([BM // 2], tl.int32, tle.dsa.ascend.UB)
    if ACC_UB_ROWS > 0:
        acc_buf = tle.dsa.alloc([ACC_UB_ROWS, 256], tl.float32, tle.dsa.ascend.UB)
    scratch = SCORES + worker * 128 * BS
    prob = PROBS + worker * 128 * BS
    pvwork = PVWORK + worker * 128 * 256
    limit = klen
    if CAUSAL:
        # Bound by this BM subtile, including a possible wrap to the next
        # grouped head, rather than by the entire 128-query tile.
        first_row = sub * BM
        last_row = first_row + BM - 1
        end_row = tl.where(first_row // qs == last_row // qs, last_row % qs + 1, qs)
        end_row = tl.minimum(qlen, qt * 128 + end_row)
        limit = tl.minimum(klen, tl.maximum(0, end_row + klen - qlen))
    start = 0
    if SPLITS > 1:
        # Preserve the native 512-key probability-rounding boundaries.
        _, span = effective_kv_splits(klen, SPLITS, BS)
        start = split * span
        limit = tl.minimum(limit, start + span)
    with tle.scope(core_mode="vector"):
        if BM == 128:
            # Per-task/AIV row visibility is invariant across all KV stacks.
            row_rr = (
                sub * BM + sub_vec_id().to(tl.int32) * (BM // 2) + tl.arange(0, BM // 2)
            )
            if head_rows == 1:
                row_q = qt * 128 + row_rr
                row_valid = (row_rr < qs) & (headstart < group) & (row_q < qlen)
            else:
                row_q = qt * 128 + row_rr % qs
                row_h = headstart + row_rr // qs
                row_valid = (row_rr < qs * head_rows) & (row_h < group) & (row_q < qlen)
            row_end = tl.full((BM // 2,), klen, tl.int32)
            if CAUSAL:
                row_end = row_q + klen - qlen + 1
            # begin and every key column are >=0; zero hides invalid rows.
            row_end = tl.where(row_valid, row_end, 0)
            tle.dsa.to_buffer(row_end, bind_buffer=row_end_buf)
        if ACC_UB_ROWS > 0:
            for init_vs in range(0, ACC_UB_ROWS, 8):
                init_acc = tle.dsa.subview(acc_buf, [init_vs, 0], [8, 256], [1, 1])
                tle.dsa.to_buffer(tl.zeros((8, 256), tl.float32), bind_buffer=init_acc)
        for init_vs in range(ACC_UB_ROWS, BM // 2, 8):
            init_rows = (
                sub_vec_id().to(tl.int32) * (BM // 2) + init_vs + tl.arange(0, 8)
            )
            tl.store(
                ACCWORK
                + worker * 128 * 256
                + init_rows[:, None] * 256
                + tl.arange(0, 256)[None, :],
                tl.zeros((8, 256), tl.float32),
            )
        x0_state_init = tl.where(tl.arange(0, 5)[:, None] == 0, -float("inf"), 0.0)
        x0_state_init = tl.broadcast_to(x0_state_init, (5, BM // 2))
        tle.dsa.to_buffer(x0_state_init, bind_buffer=state_buf)
    with tle.scope(core_mode="cube"):
        query_ptr = tl.make_block_ptr(
            Q + worker * BM * 256, (BM, 256), (256, 1), (0, 0), (BM, 256), (1, 0)
        )
        tle.dsa.copy(query_ptr, query_buf, [BM, 256])
    stacks = tl.cdiv(tl.maximum(limit - start, 0), BS)
    for step in range(stacks + 2):
        if step < stacks:
            if Q1_STREAM:
                tl.static_assert(BM == 16 and SPLITS == 1)
                if qlen == 1:
                    if step > 0:
                        sync_block_wait(
                            "vector", "cube", 13 + step % 2, PIPE.PIPE_MTE3, PIPE.PIPE_S
                        )
                    if step + 1 < stacks:
                        with tle.scope(core_mode="vector"):
                            stage_q1_window(
                                RAW_K,
                                RAW_V,
                                TABLE,
                                K,
                                V,
                                batch,
                                kh,
                                start + (step + 1) * BS,
                                klen,
                                KP,
                                KR,
                                KH,
                                VP,
                                VR,
                                VH,
                                TS,
                                PAGE,
                                CONTIG_K,
                                HK,
                                BS,
                            )
                        sync_block_set(
                            "vector",
                            "cube",
                            13 + (step + 1) % 2,
                            PIPE.PIPE_MTE3,
                            PIPE.PIPE_S,
                        )
            slot = step % 3
            begin = start + step * BS
            stack = tl.minimum(BS, limit - begin)
            scratch = SCORES + (worker * 3 + slot) * 128 * BS
            prob = PROBS + (worker * 3 + slot) * 128 * BS
            if BM != 128:
                if step >= 3:
                    sync_block_wait(
                        "vector", "cube", 9 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                    )
            with tle.scope(core_mode="cube"):
                q1 = tle.dsa.to_tensor(query_buf, writable=False)
                pair_end = ((stack + 127) // 256) * 256
                for ck in range(0, pair_end, 256):
                    kbase = (
                        K
                        + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + ck) * 256
                    )
                    kptr_0 = tl.make_block_ptr(
                        kbase, (128, 256), (256, 1), (0, 0), (128, 256), (1, 0)
                    )
                    kptr_1 = tl.make_block_ptr(
                        kbase + 128 * 256,
                        (128, 256),
                        (256, 1),
                        (0, 0),
                        (128, 256),
                        (1, 0),
                    )
                    tle.dsa.copy(kptr_0, key_buf_0, [128, 256])
                    tle.dsa.copy(kptr_1, key_buf_1, [128, 256])
                    key_0 = tle.dsa.to_tensor(key_buf_0, writable=False)
                    key_1 = tle.dsa.to_tensor(key_buf_1, writable=False)
                    qk_0 = tl.dot(q1, tl.trans(key_0))
                    qk_1 = tl.dot(q1, tl.trans(key_1))
                    soff_0 = (
                        tl.arange(0, BM)[:, None] * BS + ck + tl.arange(0, 128)[None, :]
                    )
                    tl.store(scratch + soff_0, qk_0)
                    tl.store(scratch + soff_0 + 128, qk_1)
                if pair_end < stack:
                    kbase = (
                        K
                        + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + pair_end)
                        * 256
                    )
                    kptr_0 = tl.make_block_ptr(
                        kbase, (128, 256), (256, 1), (0, 0), (128, 256), (1, 0)
                    )
                    tle.dsa.copy(kptr_0, key_buf_0, [128, 256])
                    key_0 = tle.dsa.to_tensor(key_buf_0, writable=False)
                    qk_0 = tl.dot(q1, tl.trans(key_0))
                    soff_0 = (
                        tl.arange(0, BM)[:, None] * BS
                        + pair_end
                        + tl.arange(0, 128)[None, :]
                    )
                    tl.store(scratch + soff_0, qk_0)
            sync_block_set("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            sync_block_wait("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            with tle.scope(core_mode="vector"):
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
                else:
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
        if step >= 2:
            previous = step - 2
            prevslot = previous % 3
            begin = start + previous * BS
            stack = tl.minimum(BS, limit - begin)
            prob = PROBS + (worker * 3 + prevslot) * 128 * BS
            pvwork = PVWORK + (worker * 3 + prevslot) * 128 * 256
            if BM == 128:
                if previous >= 3:
                    sync_block_wait(
                        "vector", "cube", 9 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                    )
            # V does not depend on Softmax. Issue the first tile transfer
            # before waiting for P, reusing the existing L1 value buffer.
            with tle.scope(core_mode="cube"):
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
            with tle.scope(core_mode="cube"):
                c2 = tl.zeros((BM, 256), tl.float32)
                for cv in range(0, stack, BN):
                    if cv > 0:
                        vptr = tl.make_block_ptr(
                            V
                            + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + cv)
                            * 256,
                            (BN, 256),
                            (256, 1),
                            (0, 0),
                            (BN, 256),
                            (1, 0),
                        )
                        tle.dsa.copy(vptr, value_buf, [BN, 256])
                    pptr = tl.make_block_ptr(
                        prob + cv, (BM, BS), (BS, 1), (0, 0), (BM, BN), (1, 0)
                    )
                    tle.dsa.copy(pptr, prob_buf, [BM, BN])
                    value2 = tle.dsa.to_tensor(value_buf, writable=False)
                    weights2 = tle.dsa.to_tensor(prob_buf, writable=False)
                    c2 = tl.dot(weights2, value2, c2)
                tl.store(
                    pvwork
                    + tl.arange(0, BM)[:, None] * 256
                    + tl.arange(0, 256)[None, :],
                    c2,
                )
            sync_block_set("cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            sync_block_wait("cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            with tle.scope(core_mode="vector"):
                x4_vf = sub_vec_id().to(tl.int32) * (BM // 2)
                if ACC_UB_ROWS > 0:
                    for x4_vs in range(0, ACC_UB_ROWS, O_VM):
                        x4_av = tle.dsa.subview(
                            state_buf, [2 + prevslot, x4_vs], [1, O_VM], [1, 1]
                        )
                        x4_acc = tle.dsa.subview(
                            acc_buf, [x4_vs, 0], [O_VM, 256], [1, 1]
                        )
                        x4_alpha = tl.reshape(
                            tle.dsa.to_tensor(x4_av, writable=False), (O_VM,)
                        )
                        x4_old = tle.dsa.to_tensor(x4_acc, writable=False)
                        x4_pv = tl.load(
                            pvwork
                            + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
                            + tl.arange(0, 256)[None, :],
                            volatile=True,
                        )
                        tle.dsa.to_buffer(
                            x4_old * x4_alpha[:, None] + x4_pv, bind_buffer=x4_acc
                        )
                for x4_vs in range(ACC_UB_ROWS, BM // 2, O_VM):
                    x4_av = tle.dsa.subview(
                        state_buf, [2 + prevslot, x4_vs], [1, O_VM], [1, 1]
                    )
                    x4_ab = (
                        ACCWORK
                        + worker * 128 * 256
                        + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
                        + tl.arange(0, 256)[None, :]
                    )
                    x4_alpha = tl.reshape(
                        tle.dsa.to_tensor(x4_av, writable=False), (O_VM,)
                    )
                    x4_old = tl.load(x4_ab, volatile=True)
                    x4_pv = tl.load(
                        pvwork
                        + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
                        + tl.arange(0, 256)[None, :],
                        volatile=True,
                    )
                    tl.store(x4_ab, x4_old * x4_alpha[:, None] + x4_pv)
            sync_block_set("vector", "cube", 9 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    for drain in range(tl.minimum(stacks, 3)):
        sync_block_wait("vector", "cube", 9 + drain, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    with tle.scope(core_mode="vector"):
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
                    x5_acc = tle.dsa.to_tensor(x5_accview, writable=False)
                else:
                    x5_acc = tl.load(x5_ab, volatile=True)
                x5_safe = tl.where(x5_l > 0, x5_l, 1.0)
                x5_result = x5_acc / x5_safe[:, None]
                x5_lse = tl.where(x5_l > 0, x5_m + tl.log(x5_safe), -float("inf"))
                x5_rr = x5_vf + x5_vs + tl.arange(0, VM)
                packed_base = ((split.to(tl.int64) * TAIL_BATCH + batch) * HK + kh) * BM
                tl.store(
                    O
                    + (packed_base + x5_rr)[:, None] * 256
                    + tl.arange(0, 256)[None, :],
                    x5_result,
                )
                tl.store(LSE + packed_base + x5_rr, x5_lse)


# ---- grouped_tail ----


@triton.jit
def ascend_attention_grouped_tail(
    Q_RAW,
    Q_STAGE,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    IDS,
    COUNT,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SCALE: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    CONTIG_K: tl.constexpr,
    SPLITS: tl.constexpr,
):
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    group: tl.constexpr = H // HK
    cursor = 0
    count = tl.load(COUNT)
    for request in range(count):
        batch = tl.load(IDS + request)
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        tail = qlen % 128
        eligible = (qlen > 128) & (tail > 0) & (tail <= 4)
        klen = tl.load(USED + batch)
        active_splits, _ = effective_kv_splits(klen, SPLITS, 512)
        tasks = tl.where(eligible, HK * active_splits, 0)
        first = (worker + workers - cursor) % workers
        for task in range(first, tasks, workers):
            kh = task // active_splits
            split = task % active_splits
            with tle.scope(core_mode="vector"):
                rr = sub_vec_id().to(tl.int32) * 16 + tl.arange(0, 16)
                safe_tail = tl.maximum(tail, 1)
                token = qlen - tail + rr % safe_tail
                head = kh * group + tl.minimum(rr // safe_tail, group - 1)
                valid = rr < tail * group
                d = tl.arange(0, 256)
                q = tl.load(
                    Q_RAW
                    + (qstart + token).to(tl.int64)[:, None] * QR
                    + head.to(tl.int64)[:, None] * QH
                    + d[None, :],
                    valid[:, None],
                    0,
                )
                tl.store(
                    Q_STAGE
                    + worker.to(tl.int64) * 32 * 256
                    + rr[:, None] * 256
                    + d[None, :],
                    q,
                )
            sync_block_set("vector", "cube", 12, PIPE.PIPE_MTE3, PIPE.PIPE_S)
            sync_block_wait("vector", "cube", 12, PIPE.PIPE_MTE3, PIPE.PIPE_S)
            _attention_task_tail_gqa(
                Q_STAGE,
                Q_RAW,
                K,
                V,
                O,
                LSE,
                CUQ,
                USED,
                IDS,
                SCORES,
                PROBS,
                PVWORK,
                ACCWORK,
                batch,
                kh,
                split,
                worker,
                H,
                HK,
                TOTAL_Q,
                132,
                SCALE,
                QR,
                QH,
                0,
                H * 256,
                256,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                32,
                True,
                True,
                32,
                256,
                512,
                SPLITS,
                CONTIG_K,
                0,
                16,
                FINAL_O=O,
                FINAL_LSE=LSE,
                FINAL_OR=H * 256,
                FINAL_OH=256,
                FINAL_TOTAL_Q=TOTAL_Q,
                TAIL_BATCH=B,
            )
        cursor = (cursor + tasks) % workers


# ---- grouped_tail ----


@triton.jit
def merge_grouped_tail(
    PARTIAL,
    PARTIAL_LSE,
    O,
    LSE,
    CUQ,
    USED,
    IDS,
    COUNT,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    SPLITS: tl.constexpr,
    NS: tl.constexpr,
):
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    count = tl.load(COUNT)
    for task in range(worker, count * H, workers):
        request = task // H
        head = task % H
        batch = tl.load(IDS + request)
        start = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - start
        tail = qlen % 128
        if (qlen > 128) & (tail > 0) & (tail <= 4):
            klen = tl.load(USED + batch)
            active_splits, _ = effective_kv_splits(klen, SPLITS, 512)
            splits = tl.arange(0, NS)
            d = tl.arange(0, 256)
            for r in range(tail):
                row = start + qlen - tail + r
                packed_row = (head % (H // HK)) * tail + r
                offset = (
                    (splits * B + batch) * HK + head // (H // HK)
                ) * 32 + packed_row
                ls = tl.load(
                    PARTIAL_LSE + offset, splits < active_splits, -float("inf")
                )
                m = tl.max(ls, 0)
                safe_m = tl.where(m > -float("inf"), m, 0.0)
                weights = tl.exp(ls - safe_m)
                total = tl.sum(weights, 0)
                safe_total = tl.where(total > 0.0, total, 1.0)
                values = tl.load(
                    PARTIAL + offset[:, None] * 256 + d[None, :],
                    (splits < active_splits)[:, None],
                    0.0,
                )
                result = tl.sum(values * weights[:, None], 0) / safe_total
                tl.store(O + row.to(tl.int64) * OR + head * OH + d, result)
                out_lse = tl.where(
                    total > 0.0, safe_m + tl.log(safe_total), -float("inf")
                )
                tl.store(LSE + head * TOTAL_Q + row, out_lse)


# ---- grouped_tail ----


def launch_grouped_tail(params, plan, workers, packed, request_ids, request_count):
    splits = min(16, workers // gcd(workers, params.hk))
    device = params.q_ptr.device
    stage = torch.empty((workers, 32, 256), dtype=params.q_ptr.dtype, device=device)
    scores = torch.empty((workers, 3, 128, 512), dtype=torch.float32, device=device)
    probs = torch.empty((workers, 3, 128, 512), dtype=params.q_ptr.dtype, device=device)
    pvwork = torch.empty((workers, 3, 128, 256), dtype=torch.float32, device=device)
    accwork = torch.empty((workers, 128, 256), dtype=torch.float32, device=device)
    partial = torch.empty(
        (splits, params.b, params.hk, 32, 256), dtype=torch.float32, device=device
    )
    partial_lse = torch.empty(
        (splits, params.b, params.hk, 32), dtype=torch.float32, device=device
    )
    _, k, v, _, contig_k, _ = packed
    ascend_attention_grouped_tail[(workers,)](
        params.q_ptr,
        stage,
        k,
        v,
        partial,
        partial_lse,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        request_ids,
        request_count,
        scores,
        probs,
        pvwork,
        accwork,
        params.b,
        params.h,
        params.hk,
        params.total_q,
        params.scale_softmax,
        params.q_row_stride,
        params.q_head_stride,
        contig_k,
        splits,
        disable_auto_inject_block_sync=True,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    merge_grouped_tail[(min(workers * 2, params.b * params.h),)](
        partial,
        partial_lse,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        request_ids,
        request_count,
        params.b,
        params.h,
        params.hk,
        params.total_q,
        params.o_row_stride,
        params.o_head_stride,
        splits,
        triton.next_power_of_2(splits),
        enable_fp_fusion=False,
    )


# ---- short_kv ----


@triton.jit
def ascend_attention_short_kv(
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
    TAIL_ONLY: tl.constexpr = False,
):
    COMPUTE_M: tl.constexpr = TASK_M if TASK_M else BM
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    if MERGE_SPLITS:
        with tle.scope(core_mode="vector"):
            merge_request_list(
                MERGE_O,
                MERGE_LSE,
                O,
                LSE,
                CUQ,
                USED,
                MERGE_IDS,
                MERGE_COUNT,
                OR,
                OH,
                TOTAL_Q,
                H,
                256,
                MERGE_SPLITS,
                triton.next_power_of_2(MERGE_SPLITS),
                BS,
                0,
                B,
                True,
                worker * 2 + sub_vec_id(),
                workers * 2,
            )
    cursor = 0
    group: tl.constexpr = H // HK
    count = B
    if REQUEST_LIST:
        count = tl.load(REQUEST_COUNT)
    for request in range(count):
        batch = request
        if REQUEST_LIST:
            batch = tl.load(REQUEST_IDS + request)
        if VARLEN:
            qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
        else:
            qlen = SEQLEN_Q
        klen = tl.load(USED + batch)
        handled = False
        if SKIP_SHORT_KV:
            handled = (qlen == 1) & (klen <= SKIP_SHORT_KV)
        if not handled:
            head_rows = tl.minimum(
                group, tl.maximum(1, (128 // tl.maximum(qlen, 1)) // 2 * 2)
            )
            ntiles = tl.cdiv(qlen, 128) * tl.cdiv(group, head_rows) * HK
            ntiles *= tl.cdiv(tl.minimum(qlen, 128) * head_rows, BM)
            active_splits, _ = effective_kv_splits(klen, SPLITS, BS)
            ntiles *= active_splits
            # Eligibility implies head_rows==1 and one original task per head.
            # Keep the tail numbering compact without scanning full Q tiles.
            tail_rows = qlen % 128
            eligible_tail = (qlen > 128) & (tail_rows > 0) & (tail_rows <= 16)
            original_offset = 0
            if TAIL_ONLY:
                # Q1 keeps its original grouped-head task; eligible long-Q
                # contributes only its compact final head tasks, both M16.
                ntiles = tl.where(qlen == 1, ntiles, tl.where(eligible_tail, H, 0))
                original_offset = tl.where(qlen == 1, 0, (tl.cdiv(qlen, 128) - 1) * H)
            else:
                ntiles -= tl.where(eligible_tail, H, 0)
            first = (worker + workers - cursor) % workers
            for task in range(first, ntiles, workers):
                _attention_task(
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
                    batch,
                    (task + original_offset) // active_splits,
                    task % active_splits,
                    worker,
                    H,
                    HK,
                    B if COMPACT_Q1 else TOTAL_Q,
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
                    COMPUTE_M,
                    BN,
                    BS,
                    SPLITS,
                    CONTIG_K,
                    QCAP,
                    ACC_UB_ROWS,
                    Q_DIRECT,
                    SINGLE_HEAD_INDEX,
                    COMPACT_Q1,
                    FINAL_O,
                    FINAL_LSE,
                    FINAL_OR,
                    FINAL_OH,
                    FINAL_TOTAL_Q,
                    BYPASS_SHORT,
                )
            cursor = (cursor + ntiles) % workers


# ---- split_body ----


@triton.jit
def ascend_attention_split_body(
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
):
    COMPUTE_M: tl.constexpr = TASK_M if TASK_M else BM
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    if MERGE_SPLITS:
        with tle.scope(core_mode="vector"):
            merge_request_list(
                MERGE_O,
                MERGE_LSE,
                FINAL_O if SPLITS > 1 else O,
                FINAL_LSE if SPLITS > 1 else LSE,
                CUQ,
                USED,
                MERGE_IDS,
                MERGE_COUNT,
                FINAL_OR if SPLITS > 1 else OR,
                FINAL_OH if SPLITS > 1 else OH,
                FINAL_TOTAL_Q if SPLITS > 1 else TOTAL_Q,
                H,
                256,
                MERGE_SPLITS,
                triton.next_power_of_2(MERGE_SPLITS),
                BS,
                0,
                B,
                True,
                worker * 2 + sub_vec_id(),
                workers * 2,
            )
    cursor = 0
    group: tl.constexpr = H // HK
    count = B
    if REQUEST_LIST:
        count = tl.load(REQUEST_COUNT)
    for request in range(count):
        batch = request
        if REQUEST_LIST:
            batch = tl.load(REQUEST_IDS + request)
        if VARLEN:
            qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
        else:
            qlen = SEQLEN_Q
        klen = tl.load(USED + batch)
        handled = False
        if SKIP_SHORT_KV:
            handled = (qlen == 1) & (klen <= SKIP_SHORT_KV)
        if not handled:
            head_rows = tl.minimum(
                group, tl.maximum(1, 128 // tl.maximum(qlen, 1) // 2 * 2)
            )
            ntiles = tl.cdiv(qlen, 128) * tl.cdiv(group, head_rows) * HK
            ntiles *= tl.cdiv(tl.minimum(qlen, 128) * head_rows, BM)
            tail_rows = qlen % 128
            eligible_tail = (qlen > 128) & (tail_rows > 0) & (tail_rows <= 4)
            ntiles -= tl.where(eligible_tail, H, 0)
            active_splits, _ = effective_kv_splits(klen, SPLITS, BS)
            ntiles *= active_splits
            first = (worker + workers - cursor) % workers
            for task in range(first, ntiles, workers):
                _attention_task(
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
                    batch,
                    task // active_splits,
                    task % active_splits,
                    worker,
                    H,
                    HK,
                    B if COMPACT_Q1 else TOTAL_Q,
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
                    COMPUTE_M,
                    BN,
                    BS,
                    SPLITS,
                    CONTIG_K,
                    QCAP,
                    ACC_UB_ROWS,
                    Q_DIRECT,
                    SINGLE_HEAD_INDEX,
                    COMPACT_Q1,
                    FINAL_O,
                    FINAL_LSE,
                    FINAL_OR,
                    FINAL_OH,
                    FINAL_TOTAL_Q,
                    BYPASS_SHORT,
                )
            cursor = (cursor + ntiles) % workers


# ---- split_body ----


@triton.jit
def _merge_body_request(
    PARTIAL,
    PARTIAL_LSE,
    O,
    LSE,
    CUQ,
    USED,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    STACK: tl.constexpr,
    batch,
    head,
    SKIP_SHORT_KV: tl.constexpr = 0,
    PARTIAL_Q: tl.constexpr = 0,
    BYPASS_SHORT: tl.constexpr = False,
):
    PARTIAL_ROWS: tl.constexpr = PARTIAL_Q if PARTIAL_Q else TOTAL_Q
    first = tl.load(CUQ + batch)
    end = tl.load(CUQ + batch + 1)
    qlen = end - first
    tail = qlen % 128
    end -= tl.where((qlen > 128) & (tail > 0) & (tail <= 4), tail, 0)
    length = tl.load(USED + batch)
    handled = (SKIP_SHORT_KV > 0) & (end - first == 1) & (length <= SKIP_SHORT_KV)
    active, _ = effective_kv_splits(length, SPLITS, STACK)
    if BYPASS_SHORT:
        handled = handled | (active == 1)
    if not handled:
        parts = tl.arange(0, BLOCK_SPLITS)
        dims = tl.arange(0, D)
        for index in range(first, end):
            row = index.to(tl.int64)
            partial_row = batch.to(tl.int64) if PARTIAL_Q else row
            if active == 1:
                result = tl.load(PARTIAL + (partial_row * HEADS + head) * D + dims)
                merged_lse = tl.load(PARTIAL_LSE + head * PARTIAL_ROWS + partial_row)
            else:
                ls = tl.load(
                    PARTIAL_LSE
                    + parts * HEADS * PARTIAL_ROWS
                    + head * PARTIAL_ROWS
                    + partial_row,
                    parts < active,
                    -float("inf"),
                )
                maximum = tl.max(ls, 0)
                valid = maximum > -float("inf")
                safe_max = tl.where(valid, maximum, 0.0)
                weights = tl.exp(ls - safe_max)
                denom = tl.sum(weights, 0)
                safe_denom = tl.where(valid, denom, 1.0)
                values = tl.load(
                    PARTIAL
                    + parts[:, None].to(tl.int64) * PARTIAL_ROWS * HEADS * D
                    + (partial_row * HEADS + head) * D
                    + dims[None, :],
                    parts[:, None] < active,
                    0.0,
                )
                result = tl.sum(values * (weights / safe_denom)[:, None], 0)
                merged_lse = tl.where(
                    valid, safe_max + tl.log(safe_denom), -float("inf")
                )
            tl.store(O + row * OR + head * OH + dims, result)
            tl.store(LSE + head * TOTAL_Q + row, merged_lse)


# ---- split_body ----


@triton.jit
def merge_split_body(
    PARTIAL,
    PARTIAL_LSE,
    O,
    LSE,
    CUQ,
    USED,
    REQUEST_IDS,
    REQUEST_COUNT,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    H: tl.constexpr,
    BODY_SPLITS: tl.constexpr = 2,
):
    merge_q_rows(
        PARTIAL,
        PARTIAL_LSE,
        O,
        LSE,
        CUQ,
        USED,
        REQUEST_IDS,
        REQUEST_COUNT,
        OR,
        OH,
        TOTAL_Q,
        H,
        256,
        BODY_SPLITS,
        triton.next_power_of_2(BODY_SPLITS),
        512,
        0,
        True,
        0,
        False,
        32768,
        CLIP_TAIL=True,
    )


# ---- split_body ----


def launch_split_full_body(
    params, plan, workers, packed, request_ids, request_count, merge_q1, body_splits=2
):
    q, k, v, qcap, contig_k, q_direct = packed
    device = params.q_ptr.device
    scores = torch.empty((workers, 3, 128, 512), dtype=torch.float32, device=device)
    probs = torch.empty((workers, 3, 128, 512), dtype=params.q_ptr.dtype, device=device)
    pvwork = torch.empty((workers, 3, 128, 256), dtype=torch.float32, device=device)
    accwork = torch.empty((workers, 128, 256), dtype=torch.float32, device=device)
    partial = torch.empty(
        (body_splits, params.total_q, params.h, 256), dtype=torch.float32, device=device
    )
    partial_lse = torch.empty(
        (body_splits, params.h, params.total_q), dtype=torch.float32, device=device
    )
    merge_o = merge_lse = merge_ids = merge_count = None
    merge_splits = 0
    if merge_q1 is not None:
        merge_o, merge_lse, merge_ids, merge_count, merge_splits = merge_q1
    ascend_attention_split_body[(workers,)](
        q,
        params.q_ptr,
        k,
        v,
        partial,
        partial_lse,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.page_table_ptr,
        scores,
        probs,
        pvwork,
        accwork,
        params.b,
        params.h,
        params.hk,
        params.total_q,
        params.seqlen_q,
        params.scale_softmax,
        params.q_row_stride,
        params.q_head_stride,
        params.q_batch_stride,
        params.h * 256,
        256,
        params.o_batch_stride,
        params.k_page_stride,
        params.k_row_stride,
        params.k_head_stride,
        params.v_ptr.stride(0),
        params.v_row_stride,
        params.v_head_stride,
        params.page_table_batch_stride,
        params.block_size,
        params.is_cu_seqlens_q,
        params.is_causal,
        128,
        256,
        512,
        body_splits,
        contig_k,
        QCAP=qcap,
        Q_DIRECT=q_direct,
        SINGLE_HEAD_INDEX=False,
        ACC_UB_ROWS=32,
        REQUEST_IDS=request_ids,
        REQUEST_COUNT=request_count,
        REQUEST_LIST=True,
        FINAL_O=params.o_ptr,
        FINAL_LSE=params.softmax_lse_ptr,
        FINAL_OR=params.o_row_stride,
        FINAL_OH=params.o_head_stride,
        FINAL_TOTAL_Q=params.total_q,
        MERGE_O=merge_o,
        MERGE_LSE=merge_lse,
        MERGE_IDS=merge_ids,
        MERGE_COUNT=merge_count,
        MERGE_SPLITS=merge_splits,
        disable_auto_inject_block_sync=True,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    merge_split_body[(workers * 2,)](
        partial,
        partial_lse,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        request_ids,
        request_count,
        params.o_row_stride,
        params.o_head_stride,
        params.total_q,
        params.h,
        BODY_SPLITS=body_splits,
        enable_fp_fusion=False,
    )


# ---- mixed_coreloops ----


@triton.jit
def _attention_request_coreloops(
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
    batch,
    first,
    ntiles,
    workers,
    worker,
    qlen,
    klen,
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
    SINGLE_HEAD_INDEX: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    ACC_UB_ROWS: tl.constexpr,
    COMPACT_O: tl.constexpr = False,
):
    tl.static_assert(BM == 128 and BN == 256 and (BS == 512))
    tl.static_assert(
        (COMPACT_O and ACC_UB_ROWS == 64) or (not COMPACT_O and ACC_UB_ROWS == 32)
    )
    group: tl.constexpr = H // HK
    query_buf = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([BN, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([BM, BN], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    state_buf = tle.dsa.alloc([5, BM // 2], tl.float32, tle.dsa.ascend.UB)
    row_end_buf = tle.dsa.alloc([BM // 2], tl.int32, tle.dsa.ascend.UB)
    if COMPACT_O:
        acc_buf = tle.dsa.alloc([ACC_UB_ROWS, 256], tl.bfloat16, tle.dsa.ascend.UB)
    else:
        acc_buf = tle.dsa.alloc([ACC_UB_ROWS, 256], tl.float32, tle.dsa.ascend.UB)
    with tle.scope(core_mode="cube"):
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        klen = tl.load(USED + batch)
        generation = 0
        for task in range(first, ntiles, workers):
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
                if CONTIG_K >= 4096 and (H // HK == 4 or H // HK == 8):
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
                    if generation + previous >= 3:
                        sync_block_wait(
                            "vector", "cube", 9 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                        )
                    begin = previous * BS
                    stack = tl.minimum(BS, limit - begin)
                    if CONTIG_K >= 4096 and (H // HK == 4 or H // HK == 8):
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
                            ACC_UB_ROWS,
                            COMPACT_O,
                        )
                    else:
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
                            ACC_UB_ROWS,
                            COMPACT_O,
                        )
            generation += stacks
        for drain in range(tl.minimum(generation, 3)):
            slot = (generation - tl.minimum(generation, 3) + drain) % 3
            sync_block_wait("vector", "cube", 9 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    with tle.scope(core_mode="vector"):
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        klen = tl.load(USED + batch)
        generation = 0
        for task in range(first, ntiles, workers):
            qt = task // H
            kh = task % H // group
            headstart = task % group
            qs = tl.maximum(1, tl.minimum(qlen, 128))
            limit = tl.minimum(
                klen, tl.maximum(0, tl.minimum(qlen, qt * 128 + 128) + klen - qlen)
            )
            stacks = tl.cdiv(limit, BS)
            _core_init_o(
                acc_buf, ACCWORK, worker.to(tl.int64), BM, ACC_UB_ROWS, COMPACT_O
            )
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
                        ACC_UB_ROWS,
                        COMPACT_O,
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
                ACC_UB_ROWS,
            )
            generation += stacks


# ---- mixed_coreloops ----


@triton.jit
def ascend_attention_mixed_coreloops(
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
    MERGE_MIN_SPLIT_K: tl.constexpr = 32768,
    CROSS_Q_TASK: tl.constexpr = False,
    LONG_MIXED_STREAM: tl.constexpr = False,
    INCLUDE_COMPACT_TAIL: tl.constexpr = False,
    COMPACT_BODY_O: tl.constexpr = False,
    BODY_TASK_PIPELINE: tl.constexpr = False,
    PARTITION_SHORT_BODY: tl.constexpr = False,
    TASK_SCORES=None,
    TASK_PROBS=None,
):
    COMPUTE_M: tl.constexpr = TASK_M if TASK_M else BM
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    if MERGE_SPLITS:
        with tle.scope(core_mode="vector"):
            merge_request_list(
                MERGE_O,
                MERGE_LSE,
                FINAL_O if SPLITS > 1 else O,
                FINAL_LSE if SPLITS > 1 else LSE,
                CUQ,
                USED,
                MERGE_IDS,
                MERGE_COUNT,
                FINAL_OR if SPLITS > 1 else OR,
                FINAL_OH if SPLITS > 1 else OH,
                FINAL_TOTAL_Q if SPLITS > 1 else TOTAL_Q,
                H,
                256,
                MERGE_SPLITS,
                triton.next_power_of_2(MERGE_SPLITS),
                BS,
                0,
                B,
                True,
                worker * 2 + sub_vec_id(),
                workers * 2,
                MIN_SPLIT_K=MERGE_MIN_SPLIT_K,
            )
    queue_supported = False
    queue_total = 0
    queue_has_short = False
    if BODY_TASK_PIPELINE:
        if PARTITION_SHORT_BODY:
            queue_supported, queue_total, queue_has_short = (
                _body_queue_partition_preflight(
                    CUQ,
                    USED,
                    REQUEST_IDS,
                    REQUEST_COUNT,
                    B,
                    H,
                    REQUEST_LIST,
                    SKIP_SHORT_KV,
                )
            )
        else:
            queue_supported, queue_total = _body_queue_preflight(
                CUQ,
                USED,
                REQUEST_IDS,
                REQUEST_COUNT,
                B,
                H,
                HK,
                REQUEST_LIST,
                SKIP_SHORT_KV,
            )
        if queue_supported:
            _attention_body_task_queue(
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
                queue_total,
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
                REQUEST_LIST,
                SKIP_SHORT_KV,
                PARTITION_SHORT_BODY,
            )
    if (not queue_supported) | queue_has_short:
        cursor = 0
        group: tl.constexpr = H // HK
        count = B
        if REQUEST_LIST:
            count = tl.load(REQUEST_COUNT)
        for request in range(count):
            batch = request
            if REQUEST_LIST:
                batch = tl.load(REQUEST_IDS + request)
            if VARLEN:
                qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
            else:
                qlen = SEQLEN_Q
            klen = tl.load(USED + batch)
            handled = False
            if SKIP_SHORT_KV:
                handled = (qlen == 1) & (klen <= SKIP_SHORT_KV)
            if PARTITION_SHORT_BODY:
                handled = handled | queue_supported & (qlen > 64)
            if not handled:
                head_rows = tl.minimum(
                    group, tl.maximum(1, 128 // tl.maximum(qlen, 1) // 2 * 2)
                )
                ntiles = tl.cdiv(qlen, 128) * tl.cdiv(group, head_rows) * HK
                ntiles *= tl.cdiv(tl.minimum(qlen, 128) * head_rows, BM)
                active_splits, _ = effective_kv_splits(klen, SPLITS, BS)
                ntiles *= active_splits
                if CROSS_Q_TASK and (not INCLUDE_COMPACT_TAIL):
                    tail_rows = qlen % 128
                    eligible_tail = (qlen > 128) & (tail_rows > 0) & (tail_rows <= 16)
                    ntiles -= tl.where(eligible_tail, H, 0)
                first = (worker + workers - cursor) % workers
                use_stream = False
                if CROSS_Q_TASK:
                    use_stream = (head_rows == 1) & (klen <= 8192)
                if LONG_MIXED_STREAM:
                    use_stream = (head_rows == 1) & (qlen > 64) & (klen <= 73728)
                if use_stream:
                    if CROSS_Q_TASK or LONG_MIXED_STREAM:
                        _attention_request_coreloops(
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
                            batch,
                            first,
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
                            SINGLE_HEAD_INDEX,
                            COMPUTE_M,
                            BN,
                            BS,
                            64 if COMPACT_BODY_O else ACC_UB_ROWS,
                            COMPACT_BODY_O,
                        )
                else:
                    for task in range(first, ntiles, workers):
                        _attention_task(
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
                            batch,
                            task // active_splits,
                            task % active_splits,
                            worker,
                            H,
                            HK,
                            B if COMPACT_Q1 else TOTAL_Q,
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
                            COMPUTE_M,
                            BN,
                            BS,
                            SPLITS,
                            CONTIG_K,
                            QCAP,
                            ACC_UB_ROWS,
                            Q_DIRECT,
                            SINGLE_HEAD_INDEX,
                            COMPACT_Q1,
                            FINAL_O,
                            FINAL_LSE,
                            FINAL_OR,
                            FINAL_OH,
                            FINAL_TOTAL_Q,
                            BYPASS_SHORT,
                        )
                cursor = (cursor + ntiles) % workers


# ---- mixed_coreloops ----


@triton.jit
def _attention_request_prob_stack(
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
    batch,
    first,
    ntiles,
    workers,
    worker,
    qlen,
    klen,
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
    SINGLE_HEAD_INDEX: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    ACC_UB_ROWS: tl.constexpr,
    COMPACT_O: tl.constexpr = False,
):
    tl.static_assert(BM == 128 and BN == 256 and (BS == 512))
    tl.static_assert(
        (COMPACT_O and ACC_UB_ROWS == 64) or (not COMPACT_O and ACC_UB_ROWS == 32)
    )
    group: tl.constexpr = H // HK
    query_buf = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([BN, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([BM, BN], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    state_buf = tle.dsa.alloc([5, BM // 2], tl.float32, tle.dsa.ascend.UB)
    row_end_buf = tle.dsa.alloc([BM // 2], tl.int32, tle.dsa.ascend.UB)
    if COMPACT_O:
        acc_buf = tle.dsa.alloc([ACC_UB_ROWS, 256], tl.bfloat16, tle.dsa.ascend.UB)
    else:
        acc_buf = tle.dsa.alloc([ACC_UB_ROWS, 256], tl.float32, tle.dsa.ascend.UB)
    with tle.scope(core_mode="cube"):
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        klen = tl.load(USED + batch)
        generation = 0
        for task in range(first, ntiles, workers):
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
                if CONTIG_K >= 4096 and (H // HK == 4 or H // HK == 8):
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
                    if generation + previous >= 3:
                        sync_block_wait(
                            "vector", "cube", 9 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                        )
                    begin = previous * BS
                    stack = tl.minimum(BS, limit - begin)
                    if CONTIG_K >= 4096 and (H // HK == 4 or H // HK == 8):
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
                            ACC_UB_ROWS,
                            COMPACT_O,
                        )
                    else:
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
                            ACC_UB_ROWS,
                            COMPACT_O,
                        )
            generation += stacks
        for drain in range(tl.minimum(generation, 3)):
            slot = (generation - tl.minimum(generation, 3) + drain) % 3
            sync_block_wait("vector", "cube", 9 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    with tle.scope(core_mode="vector"):
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        klen = tl.load(USED + batch)
        generation = 0
        for task in range(first, ntiles, workers):
            qt = task // H
            kh = task % H // group
            headstart = task % group
            qs = tl.maximum(1, tl.minimum(qlen, 128))
            limit = tl.minimum(
                klen, tl.maximum(0, tl.minimum(qlen, qt * 128 + 128) + klen - qlen)
            )
            stacks = tl.cdiv(limit, BS)
            _core_init_o(
                acc_buf, ACCWORK, worker.to(tl.int64), BM, ACC_UB_ROWS, COMPACT_O
            )
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
                        ACC_UB_ROWS,
                        COMPACT_O,
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
                ACC_UB_ROWS,
            )
            generation += stacks


# ---- mixed_coreloops ----


@triton.jit
def ascend_attention_mixed_prob_stack(
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
    MERGE_MIN_SPLIT_K: tl.constexpr = 32768,
    CROSS_Q_TASK: tl.constexpr = False,
    LONG_MIXED_STREAM: tl.constexpr = False,
    INCLUDE_COMPACT_TAIL: tl.constexpr = False,
    COMPACT_BODY_O: tl.constexpr = False,
    BODY_TASK_PIPELINE: tl.constexpr = False,
    PARTITION_SHORT_BODY: tl.constexpr = False,
    TASK_SCORES=None,
    TASK_PROBS=None,
):
    COMPUTE_M: tl.constexpr = TASK_M if TASK_M else BM
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    if MERGE_SPLITS:
        with tle.scope(core_mode="vector"):
            merge_request_list(
                MERGE_O,
                MERGE_LSE,
                FINAL_O if SPLITS > 1 else O,
                FINAL_LSE if SPLITS > 1 else LSE,
                CUQ,
                USED,
                MERGE_IDS,
                MERGE_COUNT,
                FINAL_OR if SPLITS > 1 else OR,
                FINAL_OH if SPLITS > 1 else OH,
                FINAL_TOTAL_Q if SPLITS > 1 else TOTAL_Q,
                H,
                256,
                MERGE_SPLITS,
                triton.next_power_of_2(MERGE_SPLITS),
                BS,
                0,
                B,
                True,
                worker * 2 + sub_vec_id(),
                workers * 2,
                MIN_SPLIT_K=MERGE_MIN_SPLIT_K,
            )
    queue_supported = False
    queue_total = 0
    queue_has_short = False
    if BODY_TASK_PIPELINE:
        if PARTITION_SHORT_BODY:
            queue_supported, queue_total, queue_has_short = (
                _body_queue_partition_preflight(
                    CUQ,
                    USED,
                    REQUEST_IDS,
                    REQUEST_COUNT,
                    B,
                    H,
                    REQUEST_LIST,
                    SKIP_SHORT_KV,
                )
            )
        else:
            queue_supported, queue_total = _body_queue_preflight(
                CUQ,
                USED,
                REQUEST_IDS,
                REQUEST_COUNT,
                B,
                H,
                HK,
                REQUEST_LIST,
                SKIP_SHORT_KV,
            )
        if queue_supported:
            _attention_body_task_queue(
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
                queue_total,
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
                REQUEST_LIST,
                SKIP_SHORT_KV,
                PARTITION_SHORT_BODY,
            )
    if (not queue_supported) | queue_has_short:
        cursor = 0
        group: tl.constexpr = H // HK
        count = B
        if REQUEST_LIST:
            count = tl.load(REQUEST_COUNT)
        for request in range(count):
            batch = request
            if REQUEST_LIST:
                batch = tl.load(REQUEST_IDS + request)
            if VARLEN:
                qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
            else:
                qlen = SEQLEN_Q
            klen = tl.load(USED + batch)
            handled = False
            if SKIP_SHORT_KV:
                handled = (qlen == 1) & (klen <= SKIP_SHORT_KV)
            if PARTITION_SHORT_BODY:
                handled = handled | queue_supported & (qlen > 64)
            if not handled:
                head_rows = tl.minimum(
                    group, tl.maximum(1, 128 // tl.maximum(qlen, 1) // 2 * 2)
                )
                ntiles = tl.cdiv(qlen, 128) * tl.cdiv(group, head_rows) * HK
                ntiles *= tl.cdiv(tl.minimum(qlen, 128) * head_rows, BM)
                active_splits, _ = effective_kv_splits(klen, SPLITS, BS)
                ntiles *= active_splits
                if CROSS_Q_TASK and (not INCLUDE_COMPACT_TAIL):
                    tail_rows = qlen % 128
                    eligible_tail = (qlen > 128) & (tail_rows > 0) & (tail_rows <= 16)
                    ntiles -= tl.where(eligible_tail, H, 0)
                first = (worker + workers - cursor) % workers
                use_stream = False
                if CROSS_Q_TASK:
                    use_stream = (head_rows == 1) & (klen <= 8192)
                if LONG_MIXED_STREAM:
                    use_stream = (head_rows == 1) & (qlen > 64) & (klen <= 73728)
                if use_stream:
                    if CROSS_Q_TASK or LONG_MIXED_STREAM:
                        _attention_request_prob_stack(
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
                            batch,
                            first,
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
                            SINGLE_HEAD_INDEX,
                            COMPUTE_M,
                            BN,
                            BS,
                            64 if COMPACT_BODY_O else ACC_UB_ROWS,
                            COMPACT_BODY_O,
                        )
                else:
                    for task in range(first, ntiles, workers):
                        _attention_task(
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
                            batch,
                            task // active_splits,
                            task % active_splits,
                            worker,
                            H,
                            HK,
                            B if COMPACT_Q1 else TOTAL_Q,
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
                            COMPUTE_M,
                            BN,
                            BS,
                            SPLITS,
                            CONTIG_K,
                            QCAP,
                            ACC_UB_ROWS,
                            Q_DIRECT,
                            SINGLE_HEAD_INDEX,
                            COMPACT_Q1,
                            FINAL_O,
                            FINAL_LSE,
                            FINAL_OR,
                            FINAL_OH,
                            FINAL_TOTAL_Q,
                            BYPASS_SHORT,
                        )
                cursor = (cursor + ntiles) % workers


# ---- long_short_split ----


@triton.jit
def ascend_attention_long_short_main(
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
    MERGE_MIN_SPLIT_K: tl.constexpr = 32768,
    CROSS_Q_TASK: tl.constexpr = False,
    LONG_MIXED_STREAM: tl.constexpr = False,
    INCLUDE_COMPACT_TAIL: tl.constexpr = False,
    COMPACT_BODY_O: tl.constexpr = False,
    BODY_TASK_PIPELINE: tl.constexpr = False,
    PARTITION_SHORT_BODY: tl.constexpr = False,
    TASK_SCORES=None,
    TASK_PROBS=None,
):
    COMPUTE_M: tl.constexpr = TASK_M if TASK_M else BM
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    if MERGE_SPLITS:
        with tle.scope(core_mode="vector"):
            merge_request_list(
                MERGE_O,
                MERGE_LSE,
                FINAL_O if SPLITS > 1 else O,
                FINAL_LSE if SPLITS > 1 else LSE,
                CUQ,
                USED,
                MERGE_IDS,
                MERGE_COUNT,
                FINAL_OR if SPLITS > 1 else OR,
                FINAL_OH if SPLITS > 1 else OH,
                FINAL_TOTAL_Q if SPLITS > 1 else TOTAL_Q,
                H,
                256,
                MERGE_SPLITS,
                triton.next_power_of_2(MERGE_SPLITS),
                BS,
                0,
                B,
                True,
                worker * 2 + sub_vec_id(),
                workers * 2,
                MIN_SPLIT_K=MERGE_MIN_SPLIT_K,
            )
    queue_supported = False
    queue_total = 0
    queue_has_short = False
    if BODY_TASK_PIPELINE:
        if PARTITION_SHORT_BODY:
            queue_supported, queue_total, queue_has_short = (
                _body_queue_partition_preflight(
                    CUQ,
                    USED,
                    REQUEST_IDS,
                    REQUEST_COUNT,
                    B,
                    H,
                    REQUEST_LIST,
                    SKIP_SHORT_KV,
                )
            )
        else:
            queue_supported, queue_total = _body_queue_preflight(
                CUQ,
                USED,
                REQUEST_IDS,
                REQUEST_COUNT,
                B,
                H,
                HK,
                REQUEST_LIST,
                SKIP_SHORT_KV,
            )
        if queue_supported:
            _attention_body_task_queue(
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
                queue_total,
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
                REQUEST_LIST,
                SKIP_SHORT_KV,
                PARTITION_SHORT_BODY,
            )
    if (not queue_supported) | queue_has_short:
        cursor = 0
        group: tl.constexpr = H // HK
        count = B
        if REQUEST_LIST:
            count = tl.load(REQUEST_COUNT)
        for request in range(count):
            batch = request
            if REQUEST_LIST:
                batch = tl.load(REQUEST_IDS + request)
            if VARLEN:
                qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
            else:
                qlen = SEQLEN_Q
            klen = tl.load(USED + batch)
            handled = False
            if SKIP_SHORT_KV:
                handled = (qlen == 1) & (klen <= SKIP_SHORT_KV)
            if PARTITION_SHORT_BODY:
                handled = handled | queue_supported & (qlen > 64)
            if not handled:
                head_rows = tl.minimum(
                    group, tl.maximum(1, 128 // tl.maximum(qlen, 1) // 2 * 2)
                )
                ntiles = tl.cdiv(qlen, 128) * tl.cdiv(group, head_rows) * HK
                ntiles *= tl.cdiv(tl.minimum(qlen, 128) * head_rows, BM)
                active_splits, _ = effective_kv_splits(klen, SPLITS, BS)
                ntiles *= active_splits
                if CROSS_Q_TASK and (not INCLUDE_COMPACT_TAIL):
                    tail_rows = qlen % 128
                    eligible_tail = (qlen > 128) & (tail_rows > 0) & (tail_rows <= 16)
                    ntiles -= tl.where(eligible_tail, H, 0)
                first = (worker + workers - cursor) % workers
                use_stream = False
                if CROSS_Q_TASK:
                    use_stream = (head_rows == 1) & (klen <= 8192)
                if LONG_MIXED_STREAM:
                    use_stream = (head_rows == 1) & (qlen > 64) & (klen <= 73728)
                if not (qlen > 1) & (qlen <= 128) & (klen >= 32768):
                    if use_stream:
                        if CROSS_Q_TASK or LONG_MIXED_STREAM:
                            _attention_request_coreloops(
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
                                batch,
                                first,
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
                                SINGLE_HEAD_INDEX,
                                COMPUTE_M,
                                BN,
                                BS,
                                64 if COMPACT_BODY_O else ACC_UB_ROWS,
                                COMPACT_BODY_O,
                            )
                    else:
                        for task in range(first, ntiles, workers):
                            _attention_task(
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
                                batch,
                                task // active_splits,
                                task % active_splits,
                                worker,
                                H,
                                HK,
                                B if COMPACT_Q1 else TOTAL_Q,
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
                                COMPUTE_M,
                                BN,
                                BS,
                                SPLITS,
                                CONTIG_K,
                                QCAP,
                                ACC_UB_ROWS,
                                Q_DIRECT,
                                SINGLE_HEAD_INDEX,
                                COMPACT_Q1,
                                FINAL_O,
                                FINAL_LSE,
                                FINAL_OR,
                                FINAL_OH,
                                FINAL_TOTAL_Q,
                                BYPASS_SHORT,
                            )
                cursor = (cursor + ntiles) % workers


# ---- long_short_split ----


@triton.jit
def prepare_long_short_descriptors(
    CUQ, COMPACT_CU, B: tl.constexpr, BLOCK: tl.constexpr
):
    batch = tl.arange(0, BLOCK)
    begin = tl.load(CUQ + batch, batch < B, 0)
    end = tl.load(CUQ + batch + 1, batch < B, 0)
    tl.store(COMPACT_CU + batch * 2, batch * 128, batch < B)
    tl.store(COMPACT_CU + batch * 2 + 1, batch * 128 + end - begin, batch < B)


# ---- long_short_split ----


@triton.jit
def ascend_attention_long_short_split(
    Q,
    Q_RAW,
    K,
    V,
    PARTIAL,
    PARTIAL_LSE,
    CUQ,
    USED,
    TABLE,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    IDS,
    COUNT,
    COMPACT_CU,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    QCAP: tl.constexpr,
    CONTIG_K: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    KP: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
):
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    cursor = 0
    count = tl.load(COUNT)
    group: tl.constexpr = H // HK
    for request in range(count):
        batch = tl.load(IDS + request)
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        klen = tl.load(USED + batch)
        if (qlen > 1) & (qlen <= 128) & (klen >= 32768):
            compact_start = batch * 128
            descriptor = COMPACT_CU + batch * 2
            head_rows = tl.minimum(group, tl.maximum(1, 128 // qlen // 2 * 2))
            base_tasks = tl.cdiv(group, head_rows) * HK
            base_tasks *= tl.cdiv(qlen * head_rows, 128)
            active, _ = effective_kv_splits(klen, SPLITS, 512)
            tasks = base_tasks * active
            first = (worker + workers - cursor) % workers
            raw_delta = (qstart.to(tl.int64) - compact_start) * QR
            for task in range(first, tasks, workers):
                _attention_task(
                    Q,
                    Q_RAW + raw_delta,
                    K,
                    V,
                    PARTIAL,
                    PARTIAL_LSE,
                    descriptor - batch,
                    USED,
                    TABLE,
                    SCORES,
                    PROBS,
                    PVWORK,
                    ACCWORK,
                    batch,
                    task // active,
                    task % active,
                    worker,
                    H,
                    HK,
                    B * 128,
                    128,
                    SCALE,
                    QR,
                    QH,
                    0,
                    H * 256,
                    256,
                    0,
                    KP,
                    KR,
                    KH,
                    VP,
                    VR,
                    VH,
                    TS,
                    PAGE,
                    True,
                    True,
                    128,
                    256,
                    512,
                    SPLITS,
                    CONTIG_K,
                    QCAP=QCAP,
                    ACC_UB_ROWS=32,
                    Q_DIRECT=True,
                    SINGLE_HEAD_INDEX=False,
                    FINAL_O=PARTIAL,
                    FINAL_LSE=PARTIAL_LSE,
                    FINAL_OR=H * 256,
                    FINAL_OH=256,
                    FINAL_TOTAL_Q=B * 128,
                )
            cursor = (cursor + tasks) % workers


# ---- long_short_split ----


@triton.jit
def merge_long_short_split(
    PARTIAL,
    PARTIAL_LSE,
    O,
    LSE,
    CUQ,
    USED,
    IDS,
    COUNT,
    B: tl.constexpr,
    H: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    ss = tl.arange(0, BLOCK_SPLITS)
    dd = tl.arange(0, 256)
    count = tl.load(COUNT)
    for request in range(count):
        batch = tl.load(IDS + request)
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        klen = tl.load(USED + batch)
        if (qlen > 1) & (qlen <= 128) & (klen >= 32768):
            active, _ = effective_kv_splits(klen, SPLITS, 512)
            for task in range(worker, qlen * H, workers):
                row, head = task // H, task % H
                compact_row = batch * 128 + row
                sl = tl.load(
                    PARTIAL_LSE + ss * H * (B * 128) + head * (B * 128) + compact_row,
                    ss < active,
                    -float("inf"),
                )
                maximum = tl.max(sl, 0)
                safe = tl.where(maximum == -float("inf"), 0.0, maximum)
                weights = tl.exp(sl - safe)
                denominator = tl.sum(weights, 0)
                part = tl.load(
                    PARTIAL
                    + ss[:, None] * (B * 128) * H * 256
                    + compact_row.to(tl.int64) * H * 256
                    + head * 256
                    + dd[None, :],
                    ss[:, None] < active,
                    0.0,
                )
                result = tl.sum(part * weights[:, None], 0)
                result /= tl.where(denominator > 0, denominator, 1.0)
                out_lse = tl.where(
                    denominator > 0, maximum + tl.log(denominator), -float("inf")
                )
                tl.store(O + (qstart + row).to(tl.int64) * OR + head * OH + dd, result)
                tl.store(LSE + head.to(tl.int64) * TOTAL_Q + qstart + row, out_lse)


# ---- long_short_split ----


def launch_long_short_split(
    params, workers, packed, scores, probs, pvwork, accwork, request_ids, request_count
):
    q, k, v, qcap, contig_k, q_direct = packed
    splits = min(16, workers // gcd(workers, params.h))
    partial = torch.empty(
        (splits, params.b * 128, params.h, 256),
        dtype=torch.float32,
        device=params.q_ptr.device,
    )
    partial_lse = torch.empty(
        (splits, params.h, params.b * 128),
        dtype=torch.float32,
        device=params.q_ptr.device,
    )
    compact_cu = torch.empty(
        (params.b, 2), dtype=torch.int32, device=params.q_ptr.device
    )
    prepare_long_short_descriptors[(1,)](
        params.cu_seqlens_q_ptr,
        compact_cu,
        params.b,
        triton.next_power_of_2(params.b),
    )
    ascend_attention_long_short_split[(workers,)](
        q,
        params.q_ptr,
        k,
        v,
        partial,
        partial_lse,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.page_table_ptr,
        scores,
        probs,
        pvwork,
        accwork,
        request_ids,
        request_count,
        compact_cu,
        params.b,
        params.h,
        params.hk,
        qcap,
        contig_k,
        params.q_row_stride,
        params.q_head_stride,
        params.k_page_stride,
        params.k_row_stride,
        params.k_head_stride,
        params.v_ptr.stride(0),
        params.v_row_stride,
        params.v_head_stride,
        params.page_table_batch_stride,
        params.block_size,
        1.0 if use_packed_k_prescale(params) else params.scale_softmax,
        splits,
        disable_auto_inject_block_sync=True,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    merge_long_short_split[(workers * 2,)](
        partial,
        partial_lse,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        request_ids,
        request_count,
        params.b,
        params.h,
        params.total_q,
        params.o_row_stride,
        params.o_head_stride,
        splits,
        triton.next_power_of_2(splits),
        enable_fp_fusion=False,
    )


# ---- Q1 native paged tails ----


@triton.jit
def _stage_q1_native_tail(
    Q_RAW,
    RAW_K,
    RAW_V,
    TABLE,
    QSTAGE,
    KSTAGE,
    VSTAGE,
    batch,
    kh,
    worker,
    qstart,
    length,
    H: tl.constexpr,
    HK: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    KP: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    CAP: tl.constexpr,
    B: tl.constexpr,
    QCAP: tl.constexpr,
    STREAM_Q1: tl.constexpr = False,
    Q_SCALE: tl.constexpr = 1.0,
):
    with tle.scope(core_mode="vector"):
        native_vh = sub_vec_id().to(tl.int32)
        native_w = worker.to(tl.int64)
        if native_vh == 0:
            native_rr = tl.arange(0, 16)
            native_dd = tl.arange(0, 256)
            native_group: tl.constexpr = H // HK
            native_head = kh * native_group + tl.minimum(native_rr, native_group - 1)
            native_query = tl.load(
                Q_RAW
                + qstart.to(tl.int64) * QR
                + native_head[:, None] * QH
                + native_dd[None, :],
                native_rr[:, None] < native_group,
                0,
            )
            if Q_SCALE != 1.0:
                native_query = (native_query.to(tl.float32) * Q_SCALE).to(
                    QSTAGE.dtype.element_ty
                )
            tl.store(
                QSTAGE
                + (B * QCAP + native_w) * 128 * 256
                + native_rr[:, None] * 256
                + native_dd[None, :],
                native_query,
            )
        native_tail = length % (2 * PAGE)
        if native_tail > 0:
            native_page = length // (2 * PAGE) * 2
            native_rows = tl.arange(0, 2 * PAGE)
            native_logical = native_page + native_rows // PAGE
            native_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + native_logical,
                native_rows < native_tail,
                0,
            ).to(tl.int64)
            native_dims = tl.arange(0, 256)
            native_keep = native_rows < native_tail
            native_dest = (
                (B * HK + native_w) * CAP + native_page * PAGE + native_rows[:, None]
            ) * 256 + native_dims[None, :]
            if native_vh == 0:
                native_key = tl.load(
                    RAW_K
                    + native_physical[:, None] * KP
                    + native_rows[:, None] % PAGE * KR
                    + kh.to(tl.int64) * KH
                    + native_dims[None, :],
                    native_keep[:, None],
                    0,
                )
                tl.store(KSTAGE + native_dest, native_key)
            else:
                native_value = tl.load(
                    RAW_V
                    + native_physical[:, None] * VP
                    + native_rows[:, None] % PAGE * VR
                    + kh.to(tl.int64) * VH
                    + native_dims[None, :],
                    native_keep[:, None],
                    0,
                )
                tl.store(VSTAGE + native_dest, native_value)
    sync_block_set("vector", "cube", 12, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    sync_block_wait("vector", "cube", 12, PIPE.PIPE_MTE3, PIPE.PIPE_S)


@triton.jit
def _attention_task_native_pair(
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
    batch,
    task,
    split,
    worker,
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
    COMPACT_Q1: tl.constexpr = False,
    FINAL_O=None,
    FINAL_LSE=None,
    FINAL_OR: tl.constexpr = 0,
    FINAL_OH: tl.constexpr = 0,
    FINAL_TOTAL_Q: tl.constexpr = 0,
    BYPASS_SHORT: tl.constexpr = False,
    Q1_STREAM: tl.constexpr = False,
    RAW_K=None,
    RAW_V=None,
    NATIVE_SPLIT_Q1: tl.constexpr = False,
    MIN_SPLIT_K: tl.constexpr = 32768,
):
    tl.static_assert(
        BM == 16
        and BN == 256
        and BS == 512
        and (SPLITS == 1 or (NATIVE_SPLIT_Q1 and (SPLITS == 3 or SPLITS == 4)))
        and (H // HK == 4 or H // HK == 8)
    )
    tl.static_assert(
        not NATIVE_SPLIT_Q1
        or (
            COMPACT_Q1
            and BYPASS_SHORT
            and (PAGE == 16 or PAGE == 32)
            and ACC_UB_ROWS == 8
        )
    )
    VM: tl.constexpr = 8 if BM == 16 else 16
    SOFTMAX_VM_256: tl.constexpr = 32 if BM == 128 else VM
    O_VM: tl.constexpr = 32 if BM == 128 and ACC_UB_ROWS == 32 else VM
    worker = worker.to(tl.int64)
    if VARLEN:
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        qbase = qstart.to(tl.int64) * QR
        obase = qstart.to(tl.int64) * OR
    else:
        qlen = SEQLEN_Q
        qstart = batch * SEQLEN_Q
        qbase = batch.to(tl.int64) * QB
        obase = batch.to(tl.int64) * OB
    output_qstart = qstart
    if COMPACT_Q1:
        output_qstart = batch
        obase = batch.to(tl.int64) * OR
    if SPLITS > 1:
        obase += split.to(tl.int64) * TOTAL_Q * H * 256
    klen = tl.load(USED + batch)
    write_final = False
    if BYPASS_SHORT:
        active_output_splits, _ = effective_kv_splits(klen, SPLITS, BS, MIN_SPLIT_K)
        write_final = active_output_splits == 1
    group: tl.constexpr = H // HK
    head_rows = tl.minimum(group, tl.maximum(1, 128 // tl.maximum(qlen, 1) // 2 * 2))
    head_tiles = tl.cdiv(group, head_rows)
    qs = tl.maximum(1, tl.minimum(qlen, 128))
    subtiles = tl.cdiv(qs * head_rows, 128)
    sub = task % subtiles
    outer = task // subtiles
    qt = outer // (head_tiles * HK)
    headtile = outer % (head_tiles * HK)
    kh = headtile // head_tiles
    headstart = headtile % head_tiles * head_rows
    query_buf = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([BN, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([BM, BN], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    native_key_0 = tle.dsa.alloc([PAGE, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    native_key_1 = tle.dsa.alloc([PAGE, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    native_key_2 = tle.dsa.alloc([PAGE, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    native_key_3 = tle.dsa.alloc([PAGE, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    native_value_0 = tle.dsa.alloc([PAGE, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    native_value_1 = tle.dsa.alloc([PAGE, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    native_value_2 = tle.dsa.alloc([PAGE, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    native_value_3 = tle.dsa.alloc([PAGE, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    native_prob_0 = tle.dsa.alloc([BM, PAGE], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    native_prob_1 = tle.dsa.alloc([BM, PAGE], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    native_prob_full = tle.dsa.alloc(
        [BM, BS], PROBS.dtype.element_ty, tle.dsa.ascend.L1
    )
    state_buf = tle.dsa.alloc([5, BM // 2], tl.float32, tle.dsa.ascend.UB)
    row_end_buf = None
    if BM == 128:
        row_end_buf = tle.dsa.alloc([BM // 2], tl.int32, tle.dsa.ascend.UB)
    if ACC_UB_ROWS > 0:
        acc_buf = tle.dsa.alloc([ACC_UB_ROWS, 256], tl.float32, tle.dsa.ascend.UB)
    scratch = SCORES + worker * 128 * BS
    prob = PROBS + worker * 128 * BS
    pvwork = PVWORK + worker * 128 * 256
    limit = klen
    if CAUSAL:
        first_row = sub * BM
        last_row = first_row + BM - 1
        end_row = tl.where(first_row // qs == last_row // qs, last_row % qs + 1, qs)
        end_row = tl.minimum(qlen, qt * 128 + end_row)
        limit = tl.minimum(klen, tl.maximum(0, end_row + klen - qlen))
    start = 0
    if SPLITS > 1:
        _, span = effective_kv_splits(klen, SPLITS, BS, MIN_SPLIT_K)
        start = split * span
        limit = tl.minimum(limit, start + span)
    with tle.scope(core_mode="vector"):
        if BM == 128:
            row_rr = (
                sub * BM + sub_vec_id().to(tl.int32) * (BM // 2) + tl.arange(0, BM // 2)
            )
            if head_rows == 1:
                row_q = qt * 128 + row_rr
                row_valid = (row_rr < qs) & (headstart < group) & (row_q < qlen)
            else:
                row_q = qt * 128 + row_rr % qs
                row_h = headstart + row_rr // qs
                row_valid = (row_rr < qs * head_rows) & (row_h < group) & (row_q < qlen)
            row_end = tl.full((BM // 2,), klen, tl.int32)
            if CAUSAL:
                row_end = row_q + klen - qlen + 1
            row_end = tl.where(row_valid, row_end, 0)
            tle.dsa.to_buffer(row_end, bind_buffer=row_end_buf)
        if ACC_UB_ROWS > 0:
            for init_vs in range(0, ACC_UB_ROWS, 8):
                init_acc = tle.dsa.subview(acc_buf, [init_vs, 0], [8, 256], [1, 1])
                tle.dsa.to_buffer(tl.zeros((8, 256), tl.float32), bind_buffer=init_acc)
        for init_vs in range(ACC_UB_ROWS, BM // 2, 8):
            init_rows = (
                sub_vec_id().to(tl.int32) * (BM // 2) + init_vs + tl.arange(0, 8)
            )
            tl.store(
                ACCWORK
                + worker * 128 * 256
                + init_rows[:, None] * 256
                + tl.arange(0, 256)[None, :],
                tl.zeros((8, 256), tl.float32),
            )
        x0_state_init = tl.where(tl.arange(0, 5)[:, None] == 0, -float("inf"), 0.0)
        x0_state_init = tl.broadcast_to(x0_state_init, (5, BM // 2))
        tle.dsa.to_buffer(x0_state_init, bind_buffer=state_buf)
    with tle.scope(core_mode="cube"):
        if Q_DIRECT and VARLEN and (BM == 128):
            direct = (head_rows == 1) & (qt * 128 + 128 <= qlen)
            if direct:
                raw_head = kh * group + headstart
                raw_base = (
                    Q_RAW
                    + qbase
                    + qt.to(tl.int64) * 128 * QR
                    + raw_head.to(tl.int64) * QH
                )
                query_ptr = tl.make_block_ptr(
                    raw_base, (128, 256), (QR, 1), (0, 0), (128, 256), (1, 0)
                )
                tle.dsa.copy(query_ptr, query_buf, [128, 256])
            else:
                query_ptr = tl.make_block_ptr(
                    Q + ((batch.to(tl.int64) * QCAP + outer) * 128 + sub * BM) * 256,
                    (BM, 256),
                    (256, 1),
                    (0, 0),
                    (BM, 256),
                    (1, 0),
                )
                tle.dsa.copy(query_ptr, query_buf, [BM, 256])
        else:
            query_ptr = tl.make_block_ptr(
                Q + ((batch.to(tl.int64) * QCAP + outer) * 128 + sub * BM) * 256,
                (BM, 256),
                (256, 1),
                (0, 0),
                (BM, 256),
                (1, 0),
            )
            tle.dsa.copy(query_ptr, query_buf, [BM, 256])
    stacks = tl.cdiv(tl.maximum(limit - start, 0), BS)
    for step in range(stacks + 2):
        if step >= 2:
            previous = step - 2
            prevslot = previous % 3
            begin = start + previous * BS
            stack = tl.minimum(BS, limit - begin)
            prob = PROBS + (worker * 3 + prevslot) * 128 * BS
            pvwork = PVWORK + (worker * 3 + prevslot) * 128 * 256
            if BM == 128:
                if previous >= 3:
                    sync_block_wait(
                        "vector", "cube", 9 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                    )
            with tle.scope(core_mode="cube"):
                if qlen == 1:
                    if begin + 0 + PAGE <= klen:
                        native_vpre0_physical = tl.load(
                            TABLE + batch.to(tl.int64) * TS + (begin + 0) // PAGE
                        ).to(tl.int64)
                        native_vpre0_full_base = (
                            RAW_V + native_vpre0_physical * VP + kh.to(tl.int64) * VH
                        )
                        native_vpre0_full_ptr = tl.make_block_ptr(
                            native_vpre0_full_base,
                            (PAGE, 256),
                            (VR, 1),
                            (0, 0),
                            (PAGE, 256),
                            (1, 0),
                        )
                        tle.dsa.copy(native_vpre0_full_ptr, native_value_0, [PAGE, 256])
                    else:
                        native_vpre0_tail_base = (
                            V
                            + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + 0)
                            * 256
                        )
                        native_vpre0_tail_ptr = tl.make_block_ptr(
                            native_vpre0_tail_base,
                            (PAGE, 256),
                            (256, 1),
                            (0, 0),
                            (PAGE, 256),
                            (1, 0),
                        )
                        tle.dsa.copy(native_vpre0_tail_ptr, native_value_0, [PAGE, 256])
                    if begin + PAGE + PAGE <= klen:
                        native_vpre1_physical = tl.load(
                            TABLE + batch.to(tl.int64) * TS + (begin + PAGE) // PAGE
                        ).to(tl.int64)
                        native_vpre1_full_base = (
                            RAW_V + native_vpre1_physical * VP + kh.to(tl.int64) * VH
                        )
                        native_vpre1_full_ptr = tl.make_block_ptr(
                            native_vpre1_full_base,
                            (PAGE, 256),
                            (VR, 1),
                            (0, 0),
                            (PAGE, 256),
                            (1, 0),
                        )
                        tle.dsa.copy(native_vpre1_full_ptr, native_value_1, [PAGE, 256])
                    else:
                        native_vpre1_tail_base = (
                            V
                            + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + PAGE)
                            * 256
                        )
                        native_vpre1_tail_ptr = tl.make_block_ptr(
                            native_vpre1_tail_base,
                            (PAGE, 256),
                            (256, 1),
                            (0, 0),
                            (PAGE, 256),
                            (1, 0),
                        )
                        tle.dsa.copy(native_vpre1_tail_ptr, native_value_1, [PAGE, 256])
                else:
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
            with tle.scope(core_mode="cube"):
                if qlen == 1:
                    if stack == BS:
                        native_prob_full_ptr = tl.make_block_ptr(
                            prob, (BM, BS), (BS, 1), (0, 0), (BM, BS), (1, 0)
                        )
                        tle.dsa.copy(native_prob_full_ptr, native_prob_full, [BM, BS])
                        native_prob_tensor_full = tle.dsa.to_tensor(
                            native_prob_full, writable=False
                        )
                        c2 = tl.zeros((BM, 256), tl.float32)
                        native_vpair_end = (
                            (stack + 2 * PAGE - 1) // (2 * PAGE) * (2 * PAGE)
                        )
                        if PAGE == 32:
                            pv_pre_32_0_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 64) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_0_0_base = (
                                RAW_V
                                + pv_pre_32_0_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_0_0_ptr = tl.make_block_ptr(
                                pv_pre_32_0_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_0_0_ptr, native_value_2, [PAGE, 256])
                            pv_pre_32_0_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 96) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_0_1_base = (
                                RAW_V
                                + pv_pre_32_0_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_0_1_ptr = tl.make_block_ptr(
                                pv_pre_32_0_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_0_1_ptr, native_value_3, [PAGE, 256])
                            pv_use_32_0_0_p = extract_slice(
                                native_prob_tensor_full, (0, 0), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_0_0_v = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            c2 = tl.dot(pv_use_32_0_0_p, pv_use_32_0_0_v, c2)
                            pv_use_32_0_1_p = extract_slice(
                                native_prob_tensor_full, (0, 32), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_0_1_v = tle.dsa.to_tensor(
                                native_value_1, writable=False
                            )
                            c2 = tl.dot(pv_use_32_0_1_p, pv_use_32_0_1_v, c2)
                            pv_pre_32_1_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 128) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_1_0_base = (
                                RAW_V
                                + pv_pre_32_1_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_1_0_ptr = tl.make_block_ptr(
                                pv_pre_32_1_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_1_0_ptr, native_value_0, [PAGE, 256])
                            pv_pre_32_1_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 160) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_1_1_base = (
                                RAW_V
                                + pv_pre_32_1_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_1_1_ptr = tl.make_block_ptr(
                                pv_pre_32_1_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_1_1_ptr, native_value_1, [PAGE, 256])
                            pv_use_32_1_0_p = extract_slice(
                                native_prob_tensor_full, (0, 64), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_1_0_v = tle.dsa.to_tensor(
                                native_value_2, writable=False
                            )
                            c2 = tl.dot(pv_use_32_1_0_p, pv_use_32_1_0_v, c2)
                            pv_use_32_1_1_p = extract_slice(
                                native_prob_tensor_full, (0, 96), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_1_1_v = tle.dsa.to_tensor(
                                native_value_3, writable=False
                            )
                            c2 = tl.dot(pv_use_32_1_1_p, pv_use_32_1_1_v, c2)
                            pv_pre_32_2_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 192) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_2_0_base = (
                                RAW_V
                                + pv_pre_32_2_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_2_0_ptr = tl.make_block_ptr(
                                pv_pre_32_2_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_2_0_ptr, native_value_2, [PAGE, 256])
                            pv_pre_32_2_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 224) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_2_1_base = (
                                RAW_V
                                + pv_pre_32_2_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_2_1_ptr = tl.make_block_ptr(
                                pv_pre_32_2_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_2_1_ptr, native_value_3, [PAGE, 256])
                            pv_use_32_2_0_p = extract_slice(
                                native_prob_tensor_full, (0, 128), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_2_0_v = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            c2 = tl.dot(pv_use_32_2_0_p, pv_use_32_2_0_v, c2)
                            pv_use_32_2_1_p = extract_slice(
                                native_prob_tensor_full, (0, 160), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_2_1_v = tle.dsa.to_tensor(
                                native_value_1, writable=False
                            )
                            c2 = tl.dot(pv_use_32_2_1_p, pv_use_32_2_1_v, c2)
                            pv_pre_32_3_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 256) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_3_0_base = (
                                RAW_V
                                + pv_pre_32_3_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_3_0_ptr = tl.make_block_ptr(
                                pv_pre_32_3_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_3_0_ptr, native_value_0, [PAGE, 256])
                            pv_pre_32_3_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 288) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_3_1_base = (
                                RAW_V
                                + pv_pre_32_3_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_3_1_ptr = tl.make_block_ptr(
                                pv_pre_32_3_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_3_1_ptr, native_value_1, [PAGE, 256])
                            pv_use_32_3_0_p = extract_slice(
                                native_prob_tensor_full, (0, 192), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_3_0_v = tle.dsa.to_tensor(
                                native_value_2, writable=False
                            )
                            c2 = tl.dot(pv_use_32_3_0_p, pv_use_32_3_0_v, c2)
                            pv_use_32_3_1_p = extract_slice(
                                native_prob_tensor_full, (0, 224), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_3_1_v = tle.dsa.to_tensor(
                                native_value_3, writable=False
                            )
                            c2 = tl.dot(pv_use_32_3_1_p, pv_use_32_3_1_v, c2)
                            pv_pre_32_4_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 320) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_4_0_base = (
                                RAW_V
                                + pv_pre_32_4_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_4_0_ptr = tl.make_block_ptr(
                                pv_pre_32_4_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_4_0_ptr, native_value_2, [PAGE, 256])
                            pv_pre_32_4_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 352) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_4_1_base = (
                                RAW_V
                                + pv_pre_32_4_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_4_1_ptr = tl.make_block_ptr(
                                pv_pre_32_4_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_4_1_ptr, native_value_3, [PAGE, 256])
                            pv_use_32_4_0_p = extract_slice(
                                native_prob_tensor_full, (0, 256), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_4_0_v = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            c2 = tl.dot(pv_use_32_4_0_p, pv_use_32_4_0_v, c2)
                            pv_use_32_4_1_p = extract_slice(
                                native_prob_tensor_full, (0, 288), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_4_1_v = tle.dsa.to_tensor(
                                native_value_1, writable=False
                            )
                            c2 = tl.dot(pv_use_32_4_1_p, pv_use_32_4_1_v, c2)
                            pv_pre_32_5_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 384) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_5_0_base = (
                                RAW_V
                                + pv_pre_32_5_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_5_0_ptr = tl.make_block_ptr(
                                pv_pre_32_5_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_5_0_ptr, native_value_0, [PAGE, 256])
                            pv_pre_32_5_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 416) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_5_1_base = (
                                RAW_V
                                + pv_pre_32_5_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_5_1_ptr = tl.make_block_ptr(
                                pv_pre_32_5_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_5_1_ptr, native_value_1, [PAGE, 256])
                            pv_use_32_5_0_p = extract_slice(
                                native_prob_tensor_full, (0, 320), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_5_0_v = tle.dsa.to_tensor(
                                native_value_2, writable=False
                            )
                            c2 = tl.dot(pv_use_32_5_0_p, pv_use_32_5_0_v, c2)
                            pv_use_32_5_1_p = extract_slice(
                                native_prob_tensor_full, (0, 352), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_5_1_v = tle.dsa.to_tensor(
                                native_value_3, writable=False
                            )
                            c2 = tl.dot(pv_use_32_5_1_p, pv_use_32_5_1_v, c2)
                            pv_pre_32_6_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 448) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_6_0_base = (
                                RAW_V
                                + pv_pre_32_6_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_6_0_ptr = tl.make_block_ptr(
                                pv_pre_32_6_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_6_0_ptr, native_value_2, [PAGE, 256])
                            pv_pre_32_6_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 480) // PAGE
                            ).to(tl.int64)
                            pv_pre_32_6_1_base = (
                                RAW_V
                                + pv_pre_32_6_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_32_6_1_ptr = tl.make_block_ptr(
                                pv_pre_32_6_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_32_6_1_ptr, native_value_3, [PAGE, 256])
                            pv_use_32_6_0_p = extract_slice(
                                native_prob_tensor_full, (0, 384), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_6_0_v = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            c2 = tl.dot(pv_use_32_6_0_p, pv_use_32_6_0_v, c2)
                            pv_use_32_6_1_p = extract_slice(
                                native_prob_tensor_full, (0, 416), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_6_1_v = tle.dsa.to_tensor(
                                native_value_1, writable=False
                            )
                            c2 = tl.dot(pv_use_32_6_1_p, pv_use_32_6_1_v, c2)
                            pv_use_32_7_0_p = extract_slice(
                                native_prob_tensor_full, (0, 448), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_7_0_v = tle.dsa.to_tensor(
                                native_value_2, writable=False
                            )
                            c2 = tl.dot(pv_use_32_7_0_p, pv_use_32_7_0_v, c2)
                            pv_use_32_7_1_p = extract_slice(
                                native_prob_tensor_full, (0, 480), (BM, PAGE), (1, 1)
                            )
                            pv_use_32_7_1_v = tle.dsa.to_tensor(
                                native_value_3, writable=False
                            )
                            c2 = tl.dot(pv_use_32_7_1_p, pv_use_32_7_1_v, c2)
                        else:
                            pv_pre_16_0_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 32) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_0_0_base = (
                                RAW_V
                                + pv_pre_16_0_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_0_0_ptr = tl.make_block_ptr(
                                pv_pre_16_0_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_0_0_ptr, native_value_2, [PAGE, 256])
                            pv_pre_16_0_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 48) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_0_1_base = (
                                RAW_V
                                + pv_pre_16_0_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_0_1_ptr = tl.make_block_ptr(
                                pv_pre_16_0_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_0_1_ptr, native_value_3, [PAGE, 256])
                            pv_use_16_0_0_p = extract_slice(
                                native_prob_tensor_full, (0, 0), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_0_0_v = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            c2 = tl.dot(pv_use_16_0_0_p, pv_use_16_0_0_v, c2)
                            pv_use_16_0_1_p = extract_slice(
                                native_prob_tensor_full, (0, 16), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_0_1_v = tle.dsa.to_tensor(
                                native_value_1, writable=False
                            )
                            c2 = tl.dot(pv_use_16_0_1_p, pv_use_16_0_1_v, c2)
                            pv_pre_16_1_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 64) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_1_0_base = (
                                RAW_V
                                + pv_pre_16_1_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_1_0_ptr = tl.make_block_ptr(
                                pv_pre_16_1_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_1_0_ptr, native_value_0, [PAGE, 256])
                            pv_pre_16_1_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 80) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_1_1_base = (
                                RAW_V
                                + pv_pre_16_1_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_1_1_ptr = tl.make_block_ptr(
                                pv_pre_16_1_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_1_1_ptr, native_value_1, [PAGE, 256])
                            pv_use_16_1_0_p = extract_slice(
                                native_prob_tensor_full, (0, 32), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_1_0_v = tle.dsa.to_tensor(
                                native_value_2, writable=False
                            )
                            c2 = tl.dot(pv_use_16_1_0_p, pv_use_16_1_0_v, c2)
                            pv_use_16_1_1_p = extract_slice(
                                native_prob_tensor_full, (0, 48), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_1_1_v = tle.dsa.to_tensor(
                                native_value_3, writable=False
                            )
                            c2 = tl.dot(pv_use_16_1_1_p, pv_use_16_1_1_v, c2)
                            pv_pre_16_2_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 96) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_2_0_base = (
                                RAW_V
                                + pv_pre_16_2_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_2_0_ptr = tl.make_block_ptr(
                                pv_pre_16_2_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_2_0_ptr, native_value_2, [PAGE, 256])
                            pv_pre_16_2_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 112) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_2_1_base = (
                                RAW_V
                                + pv_pre_16_2_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_2_1_ptr = tl.make_block_ptr(
                                pv_pre_16_2_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_2_1_ptr, native_value_3, [PAGE, 256])
                            pv_use_16_2_0_p = extract_slice(
                                native_prob_tensor_full, (0, 64), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_2_0_v = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            c2 = tl.dot(pv_use_16_2_0_p, pv_use_16_2_0_v, c2)
                            pv_use_16_2_1_p = extract_slice(
                                native_prob_tensor_full, (0, 80), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_2_1_v = tle.dsa.to_tensor(
                                native_value_1, writable=False
                            )
                            c2 = tl.dot(pv_use_16_2_1_p, pv_use_16_2_1_v, c2)
                            pv_pre_16_3_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 128) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_3_0_base = (
                                RAW_V
                                + pv_pre_16_3_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_3_0_ptr = tl.make_block_ptr(
                                pv_pre_16_3_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_3_0_ptr, native_value_0, [PAGE, 256])
                            pv_pre_16_3_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 144) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_3_1_base = (
                                RAW_V
                                + pv_pre_16_3_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_3_1_ptr = tl.make_block_ptr(
                                pv_pre_16_3_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_3_1_ptr, native_value_1, [PAGE, 256])
                            pv_use_16_3_0_p = extract_slice(
                                native_prob_tensor_full, (0, 96), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_3_0_v = tle.dsa.to_tensor(
                                native_value_2, writable=False
                            )
                            c2 = tl.dot(pv_use_16_3_0_p, pv_use_16_3_0_v, c2)
                            pv_use_16_3_1_p = extract_slice(
                                native_prob_tensor_full, (0, 112), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_3_1_v = tle.dsa.to_tensor(
                                native_value_3, writable=False
                            )
                            c2 = tl.dot(pv_use_16_3_1_p, pv_use_16_3_1_v, c2)
                            pv_pre_16_4_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 160) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_4_0_base = (
                                RAW_V
                                + pv_pre_16_4_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_4_0_ptr = tl.make_block_ptr(
                                pv_pre_16_4_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_4_0_ptr, native_value_2, [PAGE, 256])
                            pv_pre_16_4_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 176) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_4_1_base = (
                                RAW_V
                                + pv_pre_16_4_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_4_1_ptr = tl.make_block_ptr(
                                pv_pre_16_4_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_4_1_ptr, native_value_3, [PAGE, 256])
                            pv_use_16_4_0_p = extract_slice(
                                native_prob_tensor_full, (0, 128), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_4_0_v = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            c2 = tl.dot(pv_use_16_4_0_p, pv_use_16_4_0_v, c2)
                            pv_use_16_4_1_p = extract_slice(
                                native_prob_tensor_full, (0, 144), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_4_1_v = tle.dsa.to_tensor(
                                native_value_1, writable=False
                            )
                            c2 = tl.dot(pv_use_16_4_1_p, pv_use_16_4_1_v, c2)
                            pv_pre_16_5_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 192) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_5_0_base = (
                                RAW_V
                                + pv_pre_16_5_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_5_0_ptr = tl.make_block_ptr(
                                pv_pre_16_5_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_5_0_ptr, native_value_0, [PAGE, 256])
                            pv_pre_16_5_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 208) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_5_1_base = (
                                RAW_V
                                + pv_pre_16_5_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_5_1_ptr = tl.make_block_ptr(
                                pv_pre_16_5_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_5_1_ptr, native_value_1, [PAGE, 256])
                            pv_use_16_5_0_p = extract_slice(
                                native_prob_tensor_full, (0, 160), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_5_0_v = tle.dsa.to_tensor(
                                native_value_2, writable=False
                            )
                            c2 = tl.dot(pv_use_16_5_0_p, pv_use_16_5_0_v, c2)
                            pv_use_16_5_1_p = extract_slice(
                                native_prob_tensor_full, (0, 176), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_5_1_v = tle.dsa.to_tensor(
                                native_value_3, writable=False
                            )
                            c2 = tl.dot(pv_use_16_5_1_p, pv_use_16_5_1_v, c2)
                            pv_pre_16_6_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 224) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_6_0_base = (
                                RAW_V
                                + pv_pre_16_6_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_6_0_ptr = tl.make_block_ptr(
                                pv_pre_16_6_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_6_0_ptr, native_value_2, [PAGE, 256])
                            pv_pre_16_6_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 240) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_6_1_base = (
                                RAW_V
                                + pv_pre_16_6_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_6_1_ptr = tl.make_block_ptr(
                                pv_pre_16_6_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_6_1_ptr, native_value_3, [PAGE, 256])
                            pv_use_16_6_0_p = extract_slice(
                                native_prob_tensor_full, (0, 192), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_6_0_v = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            c2 = tl.dot(pv_use_16_6_0_p, pv_use_16_6_0_v, c2)
                            pv_use_16_6_1_p = extract_slice(
                                native_prob_tensor_full, (0, 208), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_6_1_v = tle.dsa.to_tensor(
                                native_value_1, writable=False
                            )
                            c2 = tl.dot(pv_use_16_6_1_p, pv_use_16_6_1_v, c2)
                            pv_pre_16_7_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 256) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_7_0_base = (
                                RAW_V
                                + pv_pre_16_7_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_7_0_ptr = tl.make_block_ptr(
                                pv_pre_16_7_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_7_0_ptr, native_value_0, [PAGE, 256])
                            pv_pre_16_7_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 272) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_7_1_base = (
                                RAW_V
                                + pv_pre_16_7_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_7_1_ptr = tl.make_block_ptr(
                                pv_pre_16_7_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_7_1_ptr, native_value_1, [PAGE, 256])
                            pv_use_16_7_0_p = extract_slice(
                                native_prob_tensor_full, (0, 224), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_7_0_v = tle.dsa.to_tensor(
                                native_value_2, writable=False
                            )
                            c2 = tl.dot(pv_use_16_7_0_p, pv_use_16_7_0_v, c2)
                            pv_use_16_7_1_p = extract_slice(
                                native_prob_tensor_full, (0, 240), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_7_1_v = tle.dsa.to_tensor(
                                native_value_3, writable=False
                            )
                            c2 = tl.dot(pv_use_16_7_1_p, pv_use_16_7_1_v, c2)
                            pv_pre_16_8_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 288) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_8_0_base = (
                                RAW_V
                                + pv_pre_16_8_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_8_0_ptr = tl.make_block_ptr(
                                pv_pre_16_8_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_8_0_ptr, native_value_2, [PAGE, 256])
                            pv_pre_16_8_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 304) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_8_1_base = (
                                RAW_V
                                + pv_pre_16_8_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_8_1_ptr = tl.make_block_ptr(
                                pv_pre_16_8_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_8_1_ptr, native_value_3, [PAGE, 256])
                            pv_use_16_8_0_p = extract_slice(
                                native_prob_tensor_full, (0, 256), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_8_0_v = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            c2 = tl.dot(pv_use_16_8_0_p, pv_use_16_8_0_v, c2)
                            pv_use_16_8_1_p = extract_slice(
                                native_prob_tensor_full, (0, 272), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_8_1_v = tle.dsa.to_tensor(
                                native_value_1, writable=False
                            )
                            c2 = tl.dot(pv_use_16_8_1_p, pv_use_16_8_1_v, c2)
                            pv_pre_16_9_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 320) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_9_0_base = (
                                RAW_V
                                + pv_pre_16_9_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_9_0_ptr = tl.make_block_ptr(
                                pv_pre_16_9_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_9_0_ptr, native_value_0, [PAGE, 256])
                            pv_pre_16_9_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 336) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_9_1_base = (
                                RAW_V
                                + pv_pre_16_9_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_9_1_ptr = tl.make_block_ptr(
                                pv_pre_16_9_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(pv_pre_16_9_1_ptr, native_value_1, [PAGE, 256])
                            pv_use_16_9_0_p = extract_slice(
                                native_prob_tensor_full, (0, 288), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_9_0_v = tle.dsa.to_tensor(
                                native_value_2, writable=False
                            )
                            c2 = tl.dot(pv_use_16_9_0_p, pv_use_16_9_0_v, c2)
                            pv_use_16_9_1_p = extract_slice(
                                native_prob_tensor_full, (0, 304), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_9_1_v = tle.dsa.to_tensor(
                                native_value_3, writable=False
                            )
                            c2 = tl.dot(pv_use_16_9_1_p, pv_use_16_9_1_v, c2)
                            pv_pre_16_10_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 352) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_10_0_base = (
                                RAW_V
                                + pv_pre_16_10_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_10_0_ptr = tl.make_block_ptr(
                                pv_pre_16_10_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(
                                pv_pre_16_10_0_ptr, native_value_2, [PAGE, 256]
                            )
                            pv_pre_16_10_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 368) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_10_1_base = (
                                RAW_V
                                + pv_pre_16_10_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_10_1_ptr = tl.make_block_ptr(
                                pv_pre_16_10_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(
                                pv_pre_16_10_1_ptr, native_value_3, [PAGE, 256]
                            )
                            pv_use_16_10_0_p = extract_slice(
                                native_prob_tensor_full, (0, 320), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_10_0_v = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            c2 = tl.dot(pv_use_16_10_0_p, pv_use_16_10_0_v, c2)
                            pv_use_16_10_1_p = extract_slice(
                                native_prob_tensor_full, (0, 336), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_10_1_v = tle.dsa.to_tensor(
                                native_value_1, writable=False
                            )
                            c2 = tl.dot(pv_use_16_10_1_p, pv_use_16_10_1_v, c2)
                            pv_pre_16_11_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 384) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_11_0_base = (
                                RAW_V
                                + pv_pre_16_11_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_11_0_ptr = tl.make_block_ptr(
                                pv_pre_16_11_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(
                                pv_pre_16_11_0_ptr, native_value_0, [PAGE, 256]
                            )
                            pv_pre_16_11_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 400) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_11_1_base = (
                                RAW_V
                                + pv_pre_16_11_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_11_1_ptr = tl.make_block_ptr(
                                pv_pre_16_11_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(
                                pv_pre_16_11_1_ptr, native_value_1, [PAGE, 256]
                            )
                            pv_use_16_11_0_p = extract_slice(
                                native_prob_tensor_full, (0, 352), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_11_0_v = tle.dsa.to_tensor(
                                native_value_2, writable=False
                            )
                            c2 = tl.dot(pv_use_16_11_0_p, pv_use_16_11_0_v, c2)
                            pv_use_16_11_1_p = extract_slice(
                                native_prob_tensor_full, (0, 368), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_11_1_v = tle.dsa.to_tensor(
                                native_value_3, writable=False
                            )
                            c2 = tl.dot(pv_use_16_11_1_p, pv_use_16_11_1_v, c2)
                            pv_pre_16_12_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 416) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_12_0_base = (
                                RAW_V
                                + pv_pre_16_12_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_12_0_ptr = tl.make_block_ptr(
                                pv_pre_16_12_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(
                                pv_pre_16_12_0_ptr, native_value_2, [PAGE, 256]
                            )
                            pv_pre_16_12_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 432) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_12_1_base = (
                                RAW_V
                                + pv_pre_16_12_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_12_1_ptr = tl.make_block_ptr(
                                pv_pre_16_12_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(
                                pv_pre_16_12_1_ptr, native_value_3, [PAGE, 256]
                            )
                            pv_use_16_12_0_p = extract_slice(
                                native_prob_tensor_full, (0, 384), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_12_0_v = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            c2 = tl.dot(pv_use_16_12_0_p, pv_use_16_12_0_v, c2)
                            pv_use_16_12_1_p = extract_slice(
                                native_prob_tensor_full, (0, 400), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_12_1_v = tle.dsa.to_tensor(
                                native_value_1, writable=False
                            )
                            c2 = tl.dot(pv_use_16_12_1_p, pv_use_16_12_1_v, c2)
                            pv_pre_16_13_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 448) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_13_0_base = (
                                RAW_V
                                + pv_pre_16_13_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_13_0_ptr = tl.make_block_ptr(
                                pv_pre_16_13_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(
                                pv_pre_16_13_0_ptr, native_value_0, [PAGE, 256]
                            )
                            pv_pre_16_13_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 464) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_13_1_base = (
                                RAW_V
                                + pv_pre_16_13_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_13_1_ptr = tl.make_block_ptr(
                                pv_pre_16_13_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(
                                pv_pre_16_13_1_ptr, native_value_1, [PAGE, 256]
                            )
                            pv_use_16_13_0_p = extract_slice(
                                native_prob_tensor_full, (0, 416), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_13_0_v = tle.dsa.to_tensor(
                                native_value_2, writable=False
                            )
                            c2 = tl.dot(pv_use_16_13_0_p, pv_use_16_13_0_v, c2)
                            pv_use_16_13_1_p = extract_slice(
                                native_prob_tensor_full, (0, 432), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_13_1_v = tle.dsa.to_tensor(
                                native_value_3, writable=False
                            )
                            c2 = tl.dot(pv_use_16_13_1_p, pv_use_16_13_1_v, c2)
                            pv_pre_16_14_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 480) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_14_0_base = (
                                RAW_V
                                + pv_pre_16_14_0_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_14_0_ptr = tl.make_block_ptr(
                                pv_pre_16_14_0_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(
                                pv_pre_16_14_0_ptr, native_value_2, [PAGE, 256]
                            )
                            pv_pre_16_14_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 496) // PAGE
                            ).to(tl.int64)
                            pv_pre_16_14_1_base = (
                                RAW_V
                                + pv_pre_16_14_1_physical * VP
                                + kh.to(tl.int64) * VH
                            )
                            pv_pre_16_14_1_ptr = tl.make_block_ptr(
                                pv_pre_16_14_1_base,
                                (PAGE, 256),
                                (VR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(
                                pv_pre_16_14_1_ptr, native_value_3, [PAGE, 256]
                            )
                            pv_use_16_14_0_p = extract_slice(
                                native_prob_tensor_full, (0, 448), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_14_0_v = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            c2 = tl.dot(pv_use_16_14_0_p, pv_use_16_14_0_v, c2)
                            pv_use_16_14_1_p = extract_slice(
                                native_prob_tensor_full, (0, 464), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_14_1_v = tle.dsa.to_tensor(
                                native_value_1, writable=False
                            )
                            c2 = tl.dot(pv_use_16_14_1_p, pv_use_16_14_1_v, c2)
                            pv_use_16_15_0_p = extract_slice(
                                native_prob_tensor_full, (0, 480), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_15_0_v = tle.dsa.to_tensor(
                                native_value_2, writable=False
                            )
                            c2 = tl.dot(pv_use_16_15_0_p, pv_use_16_15_0_v, c2)
                            pv_use_16_15_1_p = extract_slice(
                                native_prob_tensor_full, (0, 496), (BM, PAGE), (1, 1)
                            )
                            pv_use_16_15_1_v = tle.dsa.to_tensor(
                                native_value_3, writable=False
                            )
                            c2 = tl.dot(pv_use_16_15_1_p, pv_use_16_15_1_v, c2)
                        tl.store(
                            pvwork
                            + tl.arange(0, BM)[:, None] * 256
                            + tl.arange(0, 256)[None, :],
                            c2,
                        )
                    else:
                        if (
                            H // HK == 8
                            and CONTIG_K == 2560
                            and SPLITS == 1
                            and Q1_STREAM
                        ) and stack <= PAGE:
                            tail_prob_ptr = tl.make_block_ptr(
                                prob, (BM, BS), (BS, 1), (0, 0), (BM, PAGE), (1, 0)
                            )
                            tle.dsa.copy(tail_prob_ptr, native_prob_0, [BM, PAGE])
                            tail_weights = tle.dsa.to_tensor(
                                native_prob_0, writable=False
                            )
                            tail_values = tle.dsa.to_tensor(
                                native_value_0, writable=False
                            )
                            tail_pv = tl.dot(tail_weights, tail_values)
                            tl.store(
                                pvwork
                                + tl.arange(0, BM)[:, None] * 256
                                + tl.arange(0, 256)[None, :],
                                tail_pv,
                            )
                        else:
                            c2 = tl.zeros((BM, 256), tl.float32)
                            native_vpair_end = (
                                (stack + 2 * PAGE - 1) // (2 * PAGE) * (2 * PAGE)
                            )
                            for native_cv in range(0, native_vpair_end, 2 * PAGE):
                                if native_cv > 0:
                                    if begin + native_cv + PAGE <= klen:
                                        native_v0_physical = tl.load(
                                            TABLE
                                            + batch.to(tl.int64) * TS
                                            + (begin + native_cv) // PAGE
                                        ).to(tl.int64)
                                        native_v0_full_base = (
                                            RAW_V
                                            + native_v0_physical * VP
                                            + kh.to(tl.int64) * VH
                                        )
                                        native_v0_full_ptr = tl.make_block_ptr(
                                            native_v0_full_base,
                                            (PAGE, 256),
                                            (VR, 1),
                                            (0, 0),
                                            (PAGE, 256),
                                            (1, 0),
                                        )
                                        tle.dsa.copy(
                                            native_v0_full_ptr,
                                            native_value_0,
                                            [PAGE, 256],
                                        )
                                    else:
                                        native_v0_tail_base = (
                                            V
                                            + (
                                                (batch.to(tl.int64) * HK + kh)
                                                * CONTIG_K
                                                + begin
                                                + native_cv
                                            )
                                            * 256
                                        )
                                        native_v0_tail_ptr = tl.make_block_ptr(
                                            native_v0_tail_base,
                                            (PAGE, 256),
                                            (256, 1),
                                            (0, 0),
                                            (PAGE, 256),
                                            (1, 0),
                                        )
                                        tle.dsa.copy(
                                            native_v0_tail_ptr,
                                            native_value_0,
                                            [PAGE, 256],
                                        )
                                    if begin + (native_cv + PAGE) + PAGE <= klen:
                                        native_v1_physical = tl.load(
                                            TABLE
                                            + batch.to(tl.int64) * TS
                                            + (begin + (native_cv + PAGE)) // PAGE
                                        ).to(tl.int64)
                                        native_v1_full_base = (
                                            RAW_V
                                            + native_v1_physical * VP
                                            + kh.to(tl.int64) * VH
                                        )
                                        native_v1_full_ptr = tl.make_block_ptr(
                                            native_v1_full_base,
                                            (PAGE, 256),
                                            (VR, 1),
                                            (0, 0),
                                            (PAGE, 256),
                                            (1, 0),
                                        )
                                        tle.dsa.copy(
                                            native_v1_full_ptr,
                                            native_value_1,
                                            [PAGE, 256],
                                        )
                                    else:
                                        native_v1_tail_base = (
                                            V
                                            + (
                                                (batch.to(tl.int64) * HK + kh)
                                                * CONTIG_K
                                                + begin
                                                + (native_cv + PAGE)
                                            )
                                            * 256
                                        )
                                        native_v1_tail_ptr = tl.make_block_ptr(
                                            native_v1_tail_base,
                                            (PAGE, 256),
                                            (256, 1),
                                            (0, 0),
                                            (PAGE, 256),
                                            (1, 0),
                                        )
                                        tle.dsa.copy(
                                            native_v1_tail_ptr,
                                            native_value_1,
                                            [PAGE, 256],
                                        )
                                native_p0_ptr = tl.make_block_ptr(
                                    prob + native_cv,
                                    (BM, BS),
                                    (BS, 1),
                                    (0, 0),
                                    (BM, PAGE),
                                    (1, 0),
                                )
                                tle.dsa.copy(native_p0_ptr, native_prob_0, [BM, PAGE])
                                native_p1_ptr = tl.make_block_ptr(
                                    prob + (native_cv + PAGE),
                                    (BM, BS),
                                    (BS, 1),
                                    (0, 0),
                                    (BM, PAGE),
                                    (1, 0),
                                )
                                tle.dsa.copy(native_p1_ptr, native_prob_1, [BM, PAGE])
                                native_v_tensor_0 = tle.dsa.to_tensor(
                                    native_value_0, writable=False
                                )
                                native_v_tensor_1 = tle.dsa.to_tensor(
                                    native_value_1, writable=False
                                )
                                native_p_tensor_0 = tle.dsa.to_tensor(
                                    native_prob_0, writable=False
                                )
                                native_p_tensor_1 = tle.dsa.to_tensor(
                                    native_prob_1, writable=False
                                )
                                c2 = tl.dot(native_p_tensor_0, native_v_tensor_0, c2)
                                c2 = tl.dot(native_p_tensor_1, native_v_tensor_1, c2)
                            tl.store(
                                pvwork
                                + tl.arange(0, BM)[:, None] * 256
                                + tl.arange(0, 256)[None, :],
                                c2,
                            )
                else:
                    c2 = tl.zeros((BM, 256), tl.float32)
                    for cv in range(0, stack, BN):
                        if cv > 0:
                            vptr = tl.make_block_ptr(
                                V
                                + (
                                    (batch.to(tl.int64) * HK + kh) * CONTIG_K
                                    + begin
                                    + cv
                                )
                                * 256,
                                (BN, 256),
                                (256, 1),
                                (0, 0),
                                (BN, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(vptr, value_buf, [BN, 256])
                        pptr = tl.make_block_ptr(
                            prob + cv, (BM, BS), (BS, 1), (0, 0), (BM, BN), (1, 0)
                        )
                        tle.dsa.copy(pptr, prob_buf, [BM, BN])
                        value2 = tle.dsa.to_tensor(value_buf, writable=False)
                        weights2 = tle.dsa.to_tensor(prob_buf, writable=False)
                        c2 = tl.dot(weights2, value2, c2)
                    tl.store(
                        pvwork
                        + tl.arange(0, BM)[:, None] * 256
                        + tl.arange(0, 256)[None, :],
                        c2,
                    )
            sync_block_set("cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            sync_block_wait("cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            with tle.scope(core_mode="vector"):
                x4_vf = sub_vec_id().to(tl.int32) * (BM // 2)
                if ACC_UB_ROWS > 0:
                    for x4_vs in range(0, ACC_UB_ROWS, O_VM):
                        x4_av = tle.dsa.subview(
                            state_buf, [2 + prevslot, x4_vs], [1, O_VM], [1, 1]
                        )
                        x4_acc = tle.dsa.subview(
                            acc_buf, [x4_vs, 0], [O_VM, 256], [1, 1]
                        )
                        x4_alpha = tl.reshape(
                            tle.dsa.to_tensor(x4_av, writable=False), (O_VM,)
                        )
                        x4_old = tle.dsa.to_tensor(x4_acc, writable=False)
                        x4_pv = tl.load(
                            pvwork
                            + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
                            + tl.arange(0, 256)[None, :],
                            volatile=True,
                        )
                        tle.dsa.to_buffer(
                            x4_old * x4_alpha[:, None] + x4_pv, bind_buffer=x4_acc
                        )
                for x4_vs in range(ACC_UB_ROWS, BM // 2, O_VM):
                    x4_av = tle.dsa.subview(
                        state_buf, [2 + prevslot, x4_vs], [1, O_VM], [1, 1]
                    )
                    x4_ab = (
                        ACCWORK
                        + worker * 128 * 256
                        + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
                        + tl.arange(0, 256)[None, :]
                    )
                    x4_alpha = tl.reshape(
                        tle.dsa.to_tensor(x4_av, writable=False), (O_VM,)
                    )
                    x4_old = tl.load(x4_ab, volatile=True)
                    x4_pv = tl.load(
                        pvwork
                        + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
                        + tl.arange(0, 256)[None, :],
                        volatile=True,
                    )
                    tl.store(x4_ab, x4_old * x4_alpha[:, None] + x4_pv)
            sync_block_set("vector", "cube", 9 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
        if step < stacks:
            slot = step % 3
            begin = start + step * BS
            stack = tl.minimum(BS, limit - begin)
            scratch = SCORES + (worker * 3 + slot) * 128 * BS
            prob = PROBS + (worker * 3 + slot) * 128 * BS
            if BM != 128:
                if step >= 3:
                    sync_block_wait(
                        "vector", "cube", 9 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                    )
            with tle.scope(core_mode="cube"):
                q1 = tle.dsa.to_tensor(query_buf, writable=False)
                if qlen == 1:
                    if stack == BS:
                        if PAGE == 32:
                            qk_pre_32_0_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 0) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_0_0_base = (
                                RAW_K
                                + qk_pre_32_0_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_0_0_ptr = tl.make_block_ptr(
                                qk_pre_32_0_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_0_0_ptr, native_key_0, [PAGE, 256])
                            qk_pre_32_0_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 32) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_0_1_base = (
                                RAW_K
                                + qk_pre_32_0_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_0_1_ptr = tl.make_block_ptr(
                                qk_pre_32_0_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_0_1_ptr, native_key_1, [PAGE, 256])
                            qk_pre_32_1_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 64) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_1_0_base = (
                                RAW_K
                                + qk_pre_32_1_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_1_0_ptr = tl.make_block_ptr(
                                qk_pre_32_1_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_1_0_ptr, native_key_2, [PAGE, 256])
                            qk_pre_32_1_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 96) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_1_1_base = (
                                RAW_K
                                + qk_pre_32_1_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_1_1_ptr = tl.make_block_ptr(
                                qk_pre_32_1_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_1_1_ptr, native_key_3, [PAGE, 256])
                            qk_use_32_0_0_key = tle.dsa.to_tensor(
                                native_key_0, writable=False
                            )
                            qk_use_32_0_0_score = tl.dot(
                                q1, tl.trans(qk_use_32_0_0_key)
                            )
                            qk_use_32_0_1_key = tle.dsa.to_tensor(
                                native_key_1, writable=False
                            )
                            qk_use_32_0_1_score = tl.dot(
                                q1, tl.trans(qk_use_32_0_1_key)
                            )
                            qk_use_32_0_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 0
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_0_0_offset, qk_use_32_0_0_score
                            )
                            qk_use_32_0_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 32
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_0_1_offset, qk_use_32_0_1_score
                            )
                            qk_pre_32_2_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 128) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_2_0_base = (
                                RAW_K
                                + qk_pre_32_2_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_2_0_ptr = tl.make_block_ptr(
                                qk_pre_32_2_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_2_0_ptr, native_key_0, [PAGE, 256])
                            qk_pre_32_2_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 160) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_2_1_base = (
                                RAW_K
                                + qk_pre_32_2_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_2_1_ptr = tl.make_block_ptr(
                                qk_pre_32_2_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_2_1_ptr, native_key_1, [PAGE, 256])
                            qk_use_32_1_0_key = tle.dsa.to_tensor(
                                native_key_2, writable=False
                            )
                            qk_use_32_1_0_score = tl.dot(
                                q1, tl.trans(qk_use_32_1_0_key)
                            )
                            qk_use_32_1_1_key = tle.dsa.to_tensor(
                                native_key_3, writable=False
                            )
                            qk_use_32_1_1_score = tl.dot(
                                q1, tl.trans(qk_use_32_1_1_key)
                            )
                            qk_use_32_1_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 64
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_1_0_offset, qk_use_32_1_0_score
                            )
                            qk_use_32_1_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 96
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_1_1_offset, qk_use_32_1_1_score
                            )
                            qk_pre_32_3_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 192) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_3_0_base = (
                                RAW_K
                                + qk_pre_32_3_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_3_0_ptr = tl.make_block_ptr(
                                qk_pre_32_3_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_3_0_ptr, native_key_2, [PAGE, 256])
                            qk_pre_32_3_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 224) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_3_1_base = (
                                RAW_K
                                + qk_pre_32_3_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_3_1_ptr = tl.make_block_ptr(
                                qk_pre_32_3_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_3_1_ptr, native_key_3, [PAGE, 256])
                            qk_use_32_2_0_key = tle.dsa.to_tensor(
                                native_key_0, writable=False
                            )
                            qk_use_32_2_0_score = tl.dot(
                                q1, tl.trans(qk_use_32_2_0_key)
                            )
                            qk_use_32_2_1_key = tle.dsa.to_tensor(
                                native_key_1, writable=False
                            )
                            qk_use_32_2_1_score = tl.dot(
                                q1, tl.trans(qk_use_32_2_1_key)
                            )
                            qk_use_32_2_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 128
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_2_0_offset, qk_use_32_2_0_score
                            )
                            qk_use_32_2_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 160
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_2_1_offset, qk_use_32_2_1_score
                            )
                            qk_pre_32_4_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 256) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_4_0_base = (
                                RAW_K
                                + qk_pre_32_4_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_4_0_ptr = tl.make_block_ptr(
                                qk_pre_32_4_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_4_0_ptr, native_key_0, [PAGE, 256])
                            qk_pre_32_4_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 288) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_4_1_base = (
                                RAW_K
                                + qk_pre_32_4_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_4_1_ptr = tl.make_block_ptr(
                                qk_pre_32_4_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_4_1_ptr, native_key_1, [PAGE, 256])
                            qk_use_32_3_0_key = tle.dsa.to_tensor(
                                native_key_2, writable=False
                            )
                            qk_use_32_3_0_score = tl.dot(
                                q1, tl.trans(qk_use_32_3_0_key)
                            )
                            qk_use_32_3_1_key = tle.dsa.to_tensor(
                                native_key_3, writable=False
                            )
                            qk_use_32_3_1_score = tl.dot(
                                q1, tl.trans(qk_use_32_3_1_key)
                            )
                            qk_use_32_3_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 192
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_3_0_offset, qk_use_32_3_0_score
                            )
                            qk_use_32_3_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 224
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_3_1_offset, qk_use_32_3_1_score
                            )
                            qk_pre_32_5_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 320) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_5_0_base = (
                                RAW_K
                                + qk_pre_32_5_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_5_0_ptr = tl.make_block_ptr(
                                qk_pre_32_5_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_5_0_ptr, native_key_2, [PAGE, 256])
                            qk_pre_32_5_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 352) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_5_1_base = (
                                RAW_K
                                + qk_pre_32_5_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_5_1_ptr = tl.make_block_ptr(
                                qk_pre_32_5_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_5_1_ptr, native_key_3, [PAGE, 256])
                            qk_use_32_4_0_key = tle.dsa.to_tensor(
                                native_key_0, writable=False
                            )
                            qk_use_32_4_0_score = tl.dot(
                                q1, tl.trans(qk_use_32_4_0_key)
                            )
                            qk_use_32_4_1_key = tle.dsa.to_tensor(
                                native_key_1, writable=False
                            )
                            qk_use_32_4_1_score = tl.dot(
                                q1, tl.trans(qk_use_32_4_1_key)
                            )
                            qk_use_32_4_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 256
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_4_0_offset, qk_use_32_4_0_score
                            )
                            qk_use_32_4_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 288
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_4_1_offset, qk_use_32_4_1_score
                            )
                            qk_pre_32_6_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 384) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_6_0_base = (
                                RAW_K
                                + qk_pre_32_6_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_6_0_ptr = tl.make_block_ptr(
                                qk_pre_32_6_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_6_0_ptr, native_key_0, [PAGE, 256])
                            qk_pre_32_6_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 416) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_6_1_base = (
                                RAW_K
                                + qk_pre_32_6_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_6_1_ptr = tl.make_block_ptr(
                                qk_pre_32_6_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_6_1_ptr, native_key_1, [PAGE, 256])
                            qk_use_32_5_0_key = tle.dsa.to_tensor(
                                native_key_2, writable=False
                            )
                            qk_use_32_5_0_score = tl.dot(
                                q1, tl.trans(qk_use_32_5_0_key)
                            )
                            qk_use_32_5_1_key = tle.dsa.to_tensor(
                                native_key_3, writable=False
                            )
                            qk_use_32_5_1_score = tl.dot(
                                q1, tl.trans(qk_use_32_5_1_key)
                            )
                            qk_use_32_5_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 320
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_5_0_offset, qk_use_32_5_0_score
                            )
                            qk_use_32_5_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 352
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_5_1_offset, qk_use_32_5_1_score
                            )
                            qk_pre_32_7_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 448) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_7_0_base = (
                                RAW_K
                                + qk_pre_32_7_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_7_0_ptr = tl.make_block_ptr(
                                qk_pre_32_7_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_7_0_ptr, native_key_2, [PAGE, 256])
                            qk_pre_32_7_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 480) // PAGE
                            ).to(tl.int64)
                            qk_pre_32_7_1_base = (
                                RAW_K
                                + qk_pre_32_7_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_32_7_1_ptr = tl.make_block_ptr(
                                qk_pre_32_7_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_32_7_1_ptr, native_key_3, [PAGE, 256])
                            qk_use_32_6_0_key = tle.dsa.to_tensor(
                                native_key_0, writable=False
                            )
                            qk_use_32_6_0_score = tl.dot(
                                q1, tl.trans(qk_use_32_6_0_key)
                            )
                            qk_use_32_6_1_key = tle.dsa.to_tensor(
                                native_key_1, writable=False
                            )
                            qk_use_32_6_1_score = tl.dot(
                                q1, tl.trans(qk_use_32_6_1_key)
                            )
                            qk_use_32_6_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 384
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_6_0_offset, qk_use_32_6_0_score
                            )
                            qk_use_32_6_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 416
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_6_1_offset, qk_use_32_6_1_score
                            )
                            qk_use_32_7_0_key = tle.dsa.to_tensor(
                                native_key_2, writable=False
                            )
                            qk_use_32_7_0_score = tl.dot(
                                q1, tl.trans(qk_use_32_7_0_key)
                            )
                            qk_use_32_7_1_key = tle.dsa.to_tensor(
                                native_key_3, writable=False
                            )
                            qk_use_32_7_1_score = tl.dot(
                                q1, tl.trans(qk_use_32_7_1_key)
                            )
                            qk_use_32_7_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 448
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_7_0_offset, qk_use_32_7_0_score
                            )
                            qk_use_32_7_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 480
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_32_7_1_offset, qk_use_32_7_1_score
                            )
                        else:
                            qk_pre_16_0_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 0) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_0_0_base = (
                                RAW_K
                                + qk_pre_16_0_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_0_0_ptr = tl.make_block_ptr(
                                qk_pre_16_0_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_0_0_ptr, native_key_0, [PAGE, 256])
                            qk_pre_16_0_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 16) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_0_1_base = (
                                RAW_K
                                + qk_pre_16_0_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_0_1_ptr = tl.make_block_ptr(
                                qk_pre_16_0_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_0_1_ptr, native_key_1, [PAGE, 256])
                            qk_pre_16_1_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 32) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_1_0_base = (
                                RAW_K
                                + qk_pre_16_1_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_1_0_ptr = tl.make_block_ptr(
                                qk_pre_16_1_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_1_0_ptr, native_key_2, [PAGE, 256])
                            qk_pre_16_1_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 48) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_1_1_base = (
                                RAW_K
                                + qk_pre_16_1_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_1_1_ptr = tl.make_block_ptr(
                                qk_pre_16_1_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_1_1_ptr, native_key_3, [PAGE, 256])
                            qk_use_16_0_0_key = tle.dsa.to_tensor(
                                native_key_0, writable=False
                            )
                            qk_use_16_0_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_0_0_key)
                            )
                            qk_use_16_0_1_key = tle.dsa.to_tensor(
                                native_key_1, writable=False
                            )
                            qk_use_16_0_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_0_1_key)
                            )
                            qk_use_16_0_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 0
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_0_0_offset, qk_use_16_0_0_score
                            )
                            qk_use_16_0_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 16
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_0_1_offset, qk_use_16_0_1_score
                            )
                            qk_pre_16_2_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 64) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_2_0_base = (
                                RAW_K
                                + qk_pre_16_2_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_2_0_ptr = tl.make_block_ptr(
                                qk_pre_16_2_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_2_0_ptr, native_key_0, [PAGE, 256])
                            qk_pre_16_2_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 80) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_2_1_base = (
                                RAW_K
                                + qk_pre_16_2_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_2_1_ptr = tl.make_block_ptr(
                                qk_pre_16_2_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_2_1_ptr, native_key_1, [PAGE, 256])
                            qk_use_16_1_0_key = tle.dsa.to_tensor(
                                native_key_2, writable=False
                            )
                            qk_use_16_1_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_1_0_key)
                            )
                            qk_use_16_1_1_key = tle.dsa.to_tensor(
                                native_key_3, writable=False
                            )
                            qk_use_16_1_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_1_1_key)
                            )
                            qk_use_16_1_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 32
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_1_0_offset, qk_use_16_1_0_score
                            )
                            qk_use_16_1_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 48
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_1_1_offset, qk_use_16_1_1_score
                            )
                            qk_pre_16_3_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 96) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_3_0_base = (
                                RAW_K
                                + qk_pre_16_3_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_3_0_ptr = tl.make_block_ptr(
                                qk_pre_16_3_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_3_0_ptr, native_key_2, [PAGE, 256])
                            qk_pre_16_3_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 112) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_3_1_base = (
                                RAW_K
                                + qk_pre_16_3_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_3_1_ptr = tl.make_block_ptr(
                                qk_pre_16_3_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_3_1_ptr, native_key_3, [PAGE, 256])
                            qk_use_16_2_0_key = tle.dsa.to_tensor(
                                native_key_0, writable=False
                            )
                            qk_use_16_2_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_2_0_key)
                            )
                            qk_use_16_2_1_key = tle.dsa.to_tensor(
                                native_key_1, writable=False
                            )
                            qk_use_16_2_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_2_1_key)
                            )
                            qk_use_16_2_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 64
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_2_0_offset, qk_use_16_2_0_score
                            )
                            qk_use_16_2_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 80
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_2_1_offset, qk_use_16_2_1_score
                            )
                            qk_pre_16_4_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 128) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_4_0_base = (
                                RAW_K
                                + qk_pre_16_4_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_4_0_ptr = tl.make_block_ptr(
                                qk_pre_16_4_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_4_0_ptr, native_key_0, [PAGE, 256])
                            qk_pre_16_4_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 144) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_4_1_base = (
                                RAW_K
                                + qk_pre_16_4_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_4_1_ptr = tl.make_block_ptr(
                                qk_pre_16_4_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_4_1_ptr, native_key_1, [PAGE, 256])
                            qk_use_16_3_0_key = tle.dsa.to_tensor(
                                native_key_2, writable=False
                            )
                            qk_use_16_3_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_3_0_key)
                            )
                            qk_use_16_3_1_key = tle.dsa.to_tensor(
                                native_key_3, writable=False
                            )
                            qk_use_16_3_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_3_1_key)
                            )
                            qk_use_16_3_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 96
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_3_0_offset, qk_use_16_3_0_score
                            )
                            qk_use_16_3_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 112
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_3_1_offset, qk_use_16_3_1_score
                            )
                            qk_pre_16_5_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 160) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_5_0_base = (
                                RAW_K
                                + qk_pre_16_5_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_5_0_ptr = tl.make_block_ptr(
                                qk_pre_16_5_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_5_0_ptr, native_key_2, [PAGE, 256])
                            qk_pre_16_5_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 176) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_5_1_base = (
                                RAW_K
                                + qk_pre_16_5_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_5_1_ptr = tl.make_block_ptr(
                                qk_pre_16_5_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_5_1_ptr, native_key_3, [PAGE, 256])
                            qk_use_16_4_0_key = tle.dsa.to_tensor(
                                native_key_0, writable=False
                            )
                            qk_use_16_4_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_4_0_key)
                            )
                            qk_use_16_4_1_key = tle.dsa.to_tensor(
                                native_key_1, writable=False
                            )
                            qk_use_16_4_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_4_1_key)
                            )
                            qk_use_16_4_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 128
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_4_0_offset, qk_use_16_4_0_score
                            )
                            qk_use_16_4_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 144
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_4_1_offset, qk_use_16_4_1_score
                            )
                            qk_pre_16_6_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 192) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_6_0_base = (
                                RAW_K
                                + qk_pre_16_6_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_6_0_ptr = tl.make_block_ptr(
                                qk_pre_16_6_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_6_0_ptr, native_key_0, [PAGE, 256])
                            qk_pre_16_6_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 208) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_6_1_base = (
                                RAW_K
                                + qk_pre_16_6_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_6_1_ptr = tl.make_block_ptr(
                                qk_pre_16_6_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_6_1_ptr, native_key_1, [PAGE, 256])
                            qk_use_16_5_0_key = tle.dsa.to_tensor(
                                native_key_2, writable=False
                            )
                            qk_use_16_5_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_5_0_key)
                            )
                            qk_use_16_5_1_key = tle.dsa.to_tensor(
                                native_key_3, writable=False
                            )
                            qk_use_16_5_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_5_1_key)
                            )
                            qk_use_16_5_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 160
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_5_0_offset, qk_use_16_5_0_score
                            )
                            qk_use_16_5_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 176
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_5_1_offset, qk_use_16_5_1_score
                            )
                            qk_pre_16_7_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 224) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_7_0_base = (
                                RAW_K
                                + qk_pre_16_7_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_7_0_ptr = tl.make_block_ptr(
                                qk_pre_16_7_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_7_0_ptr, native_key_2, [PAGE, 256])
                            qk_pre_16_7_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 240) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_7_1_base = (
                                RAW_K
                                + qk_pre_16_7_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_7_1_ptr = tl.make_block_ptr(
                                qk_pre_16_7_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_7_1_ptr, native_key_3, [PAGE, 256])
                            qk_use_16_6_0_key = tle.dsa.to_tensor(
                                native_key_0, writable=False
                            )
                            qk_use_16_6_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_6_0_key)
                            )
                            qk_use_16_6_1_key = tle.dsa.to_tensor(
                                native_key_1, writable=False
                            )
                            qk_use_16_6_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_6_1_key)
                            )
                            qk_use_16_6_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 192
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_6_0_offset, qk_use_16_6_0_score
                            )
                            qk_use_16_6_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 208
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_6_1_offset, qk_use_16_6_1_score
                            )
                            qk_pre_16_8_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 256) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_8_0_base = (
                                RAW_K
                                + qk_pre_16_8_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_8_0_ptr = tl.make_block_ptr(
                                qk_pre_16_8_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_8_0_ptr, native_key_0, [PAGE, 256])
                            qk_pre_16_8_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 272) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_8_1_base = (
                                RAW_K
                                + qk_pre_16_8_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_8_1_ptr = tl.make_block_ptr(
                                qk_pre_16_8_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_8_1_ptr, native_key_1, [PAGE, 256])
                            qk_use_16_7_0_key = tle.dsa.to_tensor(
                                native_key_2, writable=False
                            )
                            qk_use_16_7_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_7_0_key)
                            )
                            qk_use_16_7_1_key = tle.dsa.to_tensor(
                                native_key_3, writable=False
                            )
                            qk_use_16_7_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_7_1_key)
                            )
                            qk_use_16_7_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 224
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_7_0_offset, qk_use_16_7_0_score
                            )
                            qk_use_16_7_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 240
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_7_1_offset, qk_use_16_7_1_score
                            )
                            qk_pre_16_9_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 288) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_9_0_base = (
                                RAW_K
                                + qk_pre_16_9_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_9_0_ptr = tl.make_block_ptr(
                                qk_pre_16_9_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_9_0_ptr, native_key_2, [PAGE, 256])
                            qk_pre_16_9_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 304) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_9_1_base = (
                                RAW_K
                                + qk_pre_16_9_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_9_1_ptr = tl.make_block_ptr(
                                qk_pre_16_9_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_9_1_ptr, native_key_3, [PAGE, 256])
                            qk_use_16_8_0_key = tle.dsa.to_tensor(
                                native_key_0, writable=False
                            )
                            qk_use_16_8_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_8_0_key)
                            )
                            qk_use_16_8_1_key = tle.dsa.to_tensor(
                                native_key_1, writable=False
                            )
                            qk_use_16_8_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_8_1_key)
                            )
                            qk_use_16_8_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 256
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_8_0_offset, qk_use_16_8_0_score
                            )
                            qk_use_16_8_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 272
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_8_1_offset, qk_use_16_8_1_score
                            )
                            qk_pre_16_10_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 320) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_10_0_base = (
                                RAW_K
                                + qk_pre_16_10_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_10_0_ptr = tl.make_block_ptr(
                                qk_pre_16_10_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_10_0_ptr, native_key_0, [PAGE, 256])
                            qk_pre_16_10_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 336) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_10_1_base = (
                                RAW_K
                                + qk_pre_16_10_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_10_1_ptr = tl.make_block_ptr(
                                qk_pre_16_10_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_10_1_ptr, native_key_1, [PAGE, 256])
                            qk_use_16_9_0_key = tle.dsa.to_tensor(
                                native_key_2, writable=False
                            )
                            qk_use_16_9_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_9_0_key)
                            )
                            qk_use_16_9_1_key = tle.dsa.to_tensor(
                                native_key_3, writable=False
                            )
                            qk_use_16_9_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_9_1_key)
                            )
                            qk_use_16_9_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 288
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_9_0_offset, qk_use_16_9_0_score
                            )
                            qk_use_16_9_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 304
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_9_1_offset, qk_use_16_9_1_score
                            )
                            qk_pre_16_11_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 352) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_11_0_base = (
                                RAW_K
                                + qk_pre_16_11_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_11_0_ptr = tl.make_block_ptr(
                                qk_pre_16_11_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_11_0_ptr, native_key_2, [PAGE, 256])
                            qk_pre_16_11_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 368) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_11_1_base = (
                                RAW_K
                                + qk_pre_16_11_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_11_1_ptr = tl.make_block_ptr(
                                qk_pre_16_11_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_11_1_ptr, native_key_3, [PAGE, 256])
                            qk_use_16_10_0_key = tle.dsa.to_tensor(
                                native_key_0, writable=False
                            )
                            qk_use_16_10_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_10_0_key)
                            )
                            qk_use_16_10_1_key = tle.dsa.to_tensor(
                                native_key_1, writable=False
                            )
                            qk_use_16_10_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_10_1_key)
                            )
                            qk_use_16_10_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 320
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_10_0_offset, qk_use_16_10_0_score
                            )
                            qk_use_16_10_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 336
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_10_1_offset, qk_use_16_10_1_score
                            )
                            qk_pre_16_12_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 384) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_12_0_base = (
                                RAW_K
                                + qk_pre_16_12_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_12_0_ptr = tl.make_block_ptr(
                                qk_pre_16_12_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_12_0_ptr, native_key_0, [PAGE, 256])
                            qk_pre_16_12_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 400) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_12_1_base = (
                                RAW_K
                                + qk_pre_16_12_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_12_1_ptr = tl.make_block_ptr(
                                qk_pre_16_12_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_12_1_ptr, native_key_1, [PAGE, 256])
                            qk_use_16_11_0_key = tle.dsa.to_tensor(
                                native_key_2, writable=False
                            )
                            qk_use_16_11_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_11_0_key)
                            )
                            qk_use_16_11_1_key = tle.dsa.to_tensor(
                                native_key_3, writable=False
                            )
                            qk_use_16_11_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_11_1_key)
                            )
                            qk_use_16_11_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 352
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_11_0_offset, qk_use_16_11_0_score
                            )
                            qk_use_16_11_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 368
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_11_1_offset, qk_use_16_11_1_score
                            )
                            qk_pre_16_13_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 416) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_13_0_base = (
                                RAW_K
                                + qk_pre_16_13_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_13_0_ptr = tl.make_block_ptr(
                                qk_pre_16_13_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_13_0_ptr, native_key_2, [PAGE, 256])
                            qk_pre_16_13_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 432) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_13_1_base = (
                                RAW_K
                                + qk_pre_16_13_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_13_1_ptr = tl.make_block_ptr(
                                qk_pre_16_13_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_13_1_ptr, native_key_3, [PAGE, 256])
                            qk_use_16_12_0_key = tle.dsa.to_tensor(
                                native_key_0, writable=False
                            )
                            qk_use_16_12_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_12_0_key)
                            )
                            qk_use_16_12_1_key = tle.dsa.to_tensor(
                                native_key_1, writable=False
                            )
                            qk_use_16_12_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_12_1_key)
                            )
                            qk_use_16_12_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 384
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_12_0_offset, qk_use_16_12_0_score
                            )
                            qk_use_16_12_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 400
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_12_1_offset, qk_use_16_12_1_score
                            )
                            qk_pre_16_14_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 448) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_14_0_base = (
                                RAW_K
                                + qk_pre_16_14_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_14_0_ptr = tl.make_block_ptr(
                                qk_pre_16_14_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_14_0_ptr, native_key_0, [PAGE, 256])
                            qk_pre_16_14_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 464) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_14_1_base = (
                                RAW_K
                                + qk_pre_16_14_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_14_1_ptr = tl.make_block_ptr(
                                qk_pre_16_14_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_14_1_ptr, native_key_1, [PAGE, 256])
                            qk_use_16_13_0_key = tle.dsa.to_tensor(
                                native_key_2, writable=False
                            )
                            qk_use_16_13_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_13_0_key)
                            )
                            qk_use_16_13_1_key = tle.dsa.to_tensor(
                                native_key_3, writable=False
                            )
                            qk_use_16_13_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_13_1_key)
                            )
                            qk_use_16_13_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 416
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_13_0_offset, qk_use_16_13_0_score
                            )
                            qk_use_16_13_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 432
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_13_1_offset, qk_use_16_13_1_score
                            )
                            qk_pre_16_15_0_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 480) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_15_0_base = (
                                RAW_K
                                + qk_pre_16_15_0_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_15_0_ptr = tl.make_block_ptr(
                                qk_pre_16_15_0_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_15_0_ptr, native_key_2, [PAGE, 256])
                            qk_pre_16_15_1_physical = tl.load(
                                TABLE + batch.to(tl.int64) * TS + (begin + 496) // PAGE
                            ).to(tl.int64)
                            qk_pre_16_15_1_base = (
                                RAW_K
                                + qk_pre_16_15_1_physical * KP
                                + kh.to(tl.int64) * KH
                            )
                            qk_pre_16_15_1_ptr = tl.make_block_ptr(
                                qk_pre_16_15_1_base,
                                (PAGE, 256),
                                (KR, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(qk_pre_16_15_1_ptr, native_key_3, [PAGE, 256])
                            qk_use_16_14_0_key = tle.dsa.to_tensor(
                                native_key_0, writable=False
                            )
                            qk_use_16_14_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_14_0_key)
                            )
                            qk_use_16_14_1_key = tle.dsa.to_tensor(
                                native_key_1, writable=False
                            )
                            qk_use_16_14_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_14_1_key)
                            )
                            qk_use_16_14_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 448
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_14_0_offset, qk_use_16_14_0_score
                            )
                            qk_use_16_14_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 464
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_14_1_offset, qk_use_16_14_1_score
                            )
                            qk_use_16_15_0_key = tle.dsa.to_tensor(
                                native_key_2, writable=False
                            )
                            qk_use_16_15_0_score = tl.dot(
                                q1, tl.trans(qk_use_16_15_0_key)
                            )
                            qk_use_16_15_1_key = tle.dsa.to_tensor(
                                native_key_3, writable=False
                            )
                            qk_use_16_15_1_score = tl.dot(
                                q1, tl.trans(qk_use_16_15_1_key)
                            )
                            qk_use_16_15_0_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 480
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_15_0_offset, qk_use_16_15_0_score
                            )
                            qk_use_16_15_1_offset = (
                                tl.arange(0, BM)[:, None] * BS
                                + 496
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(
                                scratch + qk_use_16_15_1_offset, qk_use_16_15_1_score
                            )
                    else:
                        if (
                            H // HK == 8
                            and CONTIG_K == 2560
                            and SPLITS == 1
                            and Q1_STREAM
                        ) and stack <= PAGE:
                            tail_key_base = (
                                K
                                + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin)
                                * 256
                            )
                            tail_key_ptr = tl.make_block_ptr(
                                tail_key_base,
                                (PAGE, 256),
                                (256, 1),
                                (0, 0),
                                (PAGE, 256),
                                (1, 0),
                            )
                            tle.dsa.copy(tail_key_ptr, native_key_0, [PAGE, 256])
                            tail_key = tle.dsa.to_tensor(native_key_0, writable=False)
                            tail_score = tl.dot(q1, tl.trans(tail_key))
                            tail_offsets = (
                                tl.arange(0, BM)[:, None] * BS
                                + tl.arange(0, PAGE)[None, :]
                            )
                            tl.store(scratch + tail_offsets, tail_score)
                        else:
                            native_pair_end = (
                                (stack + 2 * PAGE - 1) // (2 * PAGE) * (2 * PAGE)
                            )
                            for native_ck in range(0, native_pair_end, 2 * PAGE):
                                if begin + native_ck + PAGE <= klen:
                                    native_k0_physical = tl.load(
                                        TABLE
                                        + batch.to(tl.int64) * TS
                                        + (begin + native_ck) // PAGE
                                    ).to(tl.int64)
                                    native_k0_full_base = (
                                        RAW_K
                                        + native_k0_physical * KP
                                        + kh.to(tl.int64) * KH
                                    )
                                    native_k0_full_ptr = tl.make_block_ptr(
                                        native_k0_full_base,
                                        (PAGE, 256),
                                        (KR, 1),
                                        (0, 0),
                                        (PAGE, 256),
                                        (1, 0),
                                    )
                                    tle.dsa.copy(
                                        native_k0_full_ptr, native_key_0, [PAGE, 256]
                                    )
                                else:
                                    native_k0_tail_base = (
                                        K
                                        + (
                                            (batch.to(tl.int64) * HK + kh) * CONTIG_K
                                            + begin
                                            + native_ck
                                        )
                                        * 256
                                    )
                                    native_k0_tail_ptr = tl.make_block_ptr(
                                        native_k0_tail_base,
                                        (PAGE, 256),
                                        (256, 1),
                                        (0, 0),
                                        (PAGE, 256),
                                        (1, 0),
                                    )
                                    tle.dsa.copy(
                                        native_k0_tail_ptr, native_key_0, [PAGE, 256]
                                    )
                                if begin + (native_ck + PAGE) + PAGE <= klen:
                                    native_k1_physical = tl.load(
                                        TABLE
                                        + batch.to(tl.int64) * TS
                                        + (begin + (native_ck + PAGE)) // PAGE
                                    ).to(tl.int64)
                                    native_k1_full_base = (
                                        RAW_K
                                        + native_k1_physical * KP
                                        + kh.to(tl.int64) * KH
                                    )
                                    native_k1_full_ptr = tl.make_block_ptr(
                                        native_k1_full_base,
                                        (PAGE, 256),
                                        (KR, 1),
                                        (0, 0),
                                        (PAGE, 256),
                                        (1, 0),
                                    )
                                    tle.dsa.copy(
                                        native_k1_full_ptr, native_key_1, [PAGE, 256]
                                    )
                                else:
                                    native_k1_tail_base = (
                                        K
                                        + (
                                            (batch.to(tl.int64) * HK + kh) * CONTIG_K
                                            + begin
                                            + (native_ck + PAGE)
                                        )
                                        * 256
                                    )
                                    native_k1_tail_ptr = tl.make_block_ptr(
                                        native_k1_tail_base,
                                        (PAGE, 256),
                                        (256, 1),
                                        (0, 0),
                                        (PAGE, 256),
                                        (1, 0),
                                    )
                                    tle.dsa.copy(
                                        native_k1_tail_ptr, native_key_1, [PAGE, 256]
                                    )
                                native_k_tensor_0 = tle.dsa.to_tensor(
                                    native_key_0, writable=False
                                )
                                native_k_tensor_1 = tle.dsa.to_tensor(
                                    native_key_1, writable=False
                                )
                                native_qk_0 = tl.dot(q1, tl.trans(native_k_tensor_0))
                                native_qk_1 = tl.dot(q1, tl.trans(native_k_tensor_1))
                                native_offsets = (
                                    tl.arange(0, BM)[:, None] * BS
                                    + native_ck
                                    + tl.arange(0, PAGE)[None, :]
                                )
                                tl.store(scratch + native_offsets, native_qk_0)
                                tl.store(scratch + native_offsets + PAGE, native_qk_1)
                else:
                    pair_end = (stack + 127) // 256 * 256
                    for ck in range(0, pair_end, 256):
                        kbase = (
                            K
                            + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + ck)
                            * 256
                        )
                        kptr_0 = tl.make_block_ptr(
                            kbase, (128, 256), (256, 1), (0, 0), (128, 256), (1, 0)
                        )
                        kptr_1 = tl.make_block_ptr(
                            kbase + 128 * 256,
                            (128, 256),
                            (256, 1),
                            (0, 0),
                            (128, 256),
                            (1, 0),
                        )
                        tle.dsa.copy(kptr_0, key_buf_0, [128, 256])
                        tle.dsa.copy(kptr_1, key_buf_1, [128, 256])
                        key_0 = tle.dsa.to_tensor(key_buf_0, writable=False)
                        key_1 = tle.dsa.to_tensor(key_buf_1, writable=False)
                        qk_0 = tl.dot(q1, tl.trans(key_0))
                        qk_1 = tl.dot(q1, tl.trans(key_1))
                        soff_0 = (
                            tl.arange(0, BM)[:, None] * BS
                            + ck
                            + tl.arange(0, 128)[None, :]
                        )
                        tl.store(scratch + soff_0, qk_0)
                        tl.store(scratch + soff_0 + 128, qk_1)
                    if pair_end < stack:
                        kbase = (
                            K
                            + (
                                (batch.to(tl.int64) * HK + kh) * CONTIG_K
                                + begin
                                + pair_end
                            )
                            * 256
                        )
                        kptr_0 = tl.make_block_ptr(
                            kbase, (128, 256), (256, 1), (0, 0), (128, 256), (1, 0)
                        )
                        tle.dsa.copy(kptr_0, key_buf_0, [128, 256])
                        key_0 = tle.dsa.to_tensor(key_buf_0, writable=False)
                        qk_0 = tl.dot(q1, tl.trans(key_0))
                        soff_0 = (
                            tl.arange(0, BM)[:, None] * BS
                            + pair_end
                            + tl.arange(0, 128)[None, :]
                        )
                        tl.store(scratch + soff_0, qk_0)
            sync_block_set("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            sync_block_wait("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            with tle.scope(core_mode="vector"):
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
    for drain in range(tl.minimum(stacks, 3)):
        sync_block_wait("vector", "cube", 9 + drain, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    with tle.scope(core_mode="vector"):
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
                    x5_acc = tle.dsa.to_tensor(x5_accview, writable=False)
                else:
                    x5_acc = tl.load(x5_ab, volatile=True)
                x5_safe = tl.where(x5_l > 0, x5_l, 1.0)
                x5_result = x5_acc / x5_safe[:, None]
                x5_lse = tl.where(x5_l > 0, x5_m + tl.log(x5_safe), -float("inf"))
                if head_rows == 1:
                    store_rows = qt * 128 + sub * BM + x5_vf + x5_vs + tl.arange(0, VM)
                    store_head = kh * group + headstart
                    store_mask = (sub * BM + x5_vf + x5_vs + tl.arange(0, VM) < qs) & (
                        store_rows < qlen
                    )
                    if write_final:
                        final_rows = qstart + store_rows
                        tl.store(
                            FINAL_O
                            + final_rows.to(tl.int64)[:, None] * FINAL_OR
                            + store_head * FINAL_OH
                            + tl.arange(0, 256)[None, :],
                            x5_result,
                            store_mask[:, None],
                        )
                        tl.store(
                            FINAL_LSE
                            + store_head.to(tl.int64) * FINAL_TOTAL_Q
                            + final_rows,
                            x5_lse,
                            store_mask,
                        )
                    else:
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
                            LSE
                            + split.to(tl.int64) * H * TOTAL_Q
                            + store_head.to(tl.int64) * TOTAL_Q
                            + output_qstart
                            + store_rows,
                            x5_lse,
                            store_mask,
                        )
                else:
                    for x5_r in range(VM):
                        x5_combined = sub * BM + x5_vf + x5_vs + x5_r
                        x5_token = qt * 128 + x5_combined % qs
                        x5_local_head = headstart + x5_combined // qs
                        if (
                            x5_combined < qs * head_rows
                            and x5_local_head < group
                            and (x5_token < qlen)
                        ):
                            x5_oh = kh * group + x5_local_head
                            x5_values = extract_slice(
                                x5_result, (x5_r, 0), (1, 256), (1, 1)
                            )
                            x5_out_lse = extract_slice(x5_lse, (x5_r,), (1,), (1,))
                            if write_final:
                                final_row = qstart + x5_token
                                tl.store(
                                    FINAL_O
                                    + final_row.to(tl.int64) * FINAL_OR
                                    + x5_oh * FINAL_OH
                                    + tl.arange(0, 256),
                                    tl.reshape(x5_values, (256,)),
                                )
                                tl.store(
                                    FINAL_LSE
                                    + x5_oh.to(tl.int64) * FINAL_TOTAL_Q
                                    + final_row,
                                    tl.reshape(x5_out_lse, ()),
                                )
                            else:
                                tl.store(
                                    O
                                    + obase
                                    + x5_token.to(tl.int64) * OR
                                    + x5_oh * OH
                                    + tl.arange(0, 256),
                                    tl.reshape(x5_values, (256,)),
                                )
                                tl.store(
                                    LSE
                                    + split.to(tl.int64) * H * TOTAL_Q
                                    + x5_oh.to(tl.int64) * TOTAL_Q
                                    + output_qstart
                                    + x5_token,
                                    tl.reshape(x5_out_lse, ()),
                                )
