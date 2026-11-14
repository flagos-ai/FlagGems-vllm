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

from .common import load_kv_tile
from .scheduling import _merge_request, effective_kv_splits, merge_request_list

# ---- body_task_queue ----


@triton.jit
def _whole_task_qk(
    Q,
    Q_RAW,
    K,
    score,
    query_buf,
    key_buf_0,
    key_buf_1,
    batch,
    task,
    qt,
    head,
    kh,
    qstart,
    qlen,
    limit,
    HK: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    CONTIG_K: tl.constexpr,
    QCAP: tl.constexpr,
    Q_DIRECT: tl.constexpr,
):
    with tle.scope(core_mode="cube"):
        if Q_DIRECT and qt * 128 + 128 <= qlen:
            qbase = (
                Q_RAW + (qstart.to(tl.int64) + qt * 128) * QR + head.to(tl.int64) * QH
            )
            qp = tl.make_block_ptr(
                qbase, (128, 256), (QR, 1), (0, 0), (128, 256), (1, 0)
            )
            tle.dsa.copy(qp, query_buf, [128, 256])
        else:
            qp = tl.make_block_ptr(
                Q + (batch.to(tl.int64) * QCAP + task) * 128 * 256,
                (128, 256),
                (256, 1),
                (0, 0),
                (128, 256),
                (1, 0),
            )
            tle.dsa.copy(qp, query_buf, [128, 256])
        query = tle.dsa.to_tensor(query_buf, writable=False)
        pair_end = (limit + 127) // 256 * 256
        for ck in range(0, pair_end, 256):
            kb = K + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + ck) * 256
            kp0 = tl.make_block_ptr(
                kb, (128, 256), (256, 1), (0, 0), (128, 256), (1, 0)
            )
            kp1 = tl.make_block_ptr(
                kb + 128 * 256, (128, 256), (256, 1), (0, 0), (128, 256), (1, 0)
            )
            tle.dsa.copy(kp0, key_buf_0, [128, 256])
            tle.dsa.copy(kp1, key_buf_1, [128, 256])
            key0 = tle.dsa.to_tensor(key_buf_0, writable=False)
            key1 = tle.dsa.to_tensor(key_buf_1, writable=False)
            s0 = tl.dot(query, tl.trans(key0))
            s1 = tl.dot(query, tl.trans(key1))
            so = tl.arange(0, 128)[:, None] * 1536 + ck + tl.arange(0, 128)[None, :]
            tl.store(score + so, s0)
            tl.store(score + so + 128, s1)
        if pair_end < limit:
            kb = K + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + pair_end) * 256
            kp0 = tl.make_block_ptr(
                kb, (128, 256), (256, 1), (0, 0), (128, 256), (1, 0)
            )
            tle.dsa.copy(kp0, key_buf_0, [128, 256])
            key0 = tle.dsa.to_tensor(key_buf_0, writable=False)
            s0 = tl.dot(query, tl.trans(key0))
            so = (
                tl.arange(0, 128)[:, None] * 1536
                + pair_end
                + tl.arange(0, 128)[None, :]
            )
            tl.store(score + so, s0)


# ---- body_task_queue ----


@triton.jit
def _whole_task_softmax(
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
    STATE_UB: tl.constexpr = False,
    STATE_SLOT=0,
):
    vf = sub_vec_id().to(tl.int32) * 64
    cols = tl.arange(0, WIDTH)
    write_limit = tl.cdiv(limit, 256) * 256
    for vs in range(0, 64, VM):
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
        if STATE_UB:
            max_view = tle.dsa.subview(STATE, [STATE_SLOT * 128 + vs], [VM], [1])
            sum_view = tle.dsa.subview(STATE, [STATE_SLOT * 128 + 64 + vs], [VM], [1])
            tle.dsa.to_buffer(m, bind_buffer=max_view)
            tle.dsa.to_buffer(denom, bind_buffer=sum_view)
        else:
            tl.store(STATE + rows, m)
            tl.store(STATE + 128 + rows, denom)


# ---- body_task_queue ----


@triton.jit
def _whole_task_pv(
    V,
    prob,
    pvwork,
    value_buf,
    prob_buf,
    batch,
    kh,
    limit,
    HK: tl.constexpr,
    CONTIG_K: tl.constexpr,
):
    with tle.scope(core_mode="cube"):
        acc = tl.zeros((128, 256), tl.float32)
        for cv in range(0, limit, 256):
            vp = tl.make_block_ptr(
                V + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + cv) * 256,
                (256, 256),
                (256, 1),
                (0, 0),
                (256, 256),
                (1, 0),
            )
            pp = tl.make_block_ptr(
                prob + cv, (128, 1536), (1536, 1), (0, 0), (128, 256), (1, 0)
            )
            tle.dsa.copy(vp, value_buf, [256, 256])
            tle.dsa.copy(pp, prob_buf, [128, 256])
            value = tle.dsa.to_tensor(value_buf, writable=False)
            weights = tle.dsa.to_tensor(prob_buf, writable=False)
            acc = tl.dot(weights, value, acc)
        tl.store(
            pvwork + tl.arange(0, 128)[:, None] * 256 + tl.arange(0, 256)[None, :], acc
        )


# ---- body_task_queue ----


@triton.jit
def _whole_task_finish(
    O,
    LSE,
    pvwork,
    STATE,
    qt,
    head,
    qlen,
    qstart,
    TOTAL_Q: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    STATE_UB: tl.constexpr = False,
    STATE_SLOT=0,
):
    with tle.scope(core_mode="vector"):
        vf = sub_vec_id().to(tl.int32) * 64
        for vs in range(0, 64, 16):
            if STATE_UB:
                max_view = tle.dsa.subview(STATE, [STATE_SLOT * 128 + vs], [16], [1])
                sum_view = tle.dsa.subview(
                    STATE, [STATE_SLOT * 128 + 64 + vs], [16], [1]
                )
                m = tle.dsa.to_tensor(max_view, writable=False)
                denom = tle.dsa.to_tensor(sum_view, writable=False)
            else:
                m = tl.load(STATE + vf + vs + tl.arange(0, 16), volatile=True)
                denom = tl.load(STATE + 128 + vf + vs + tl.arange(0, 16), volatile=True)
            rows = vf + vs + tl.arange(0, 16)
            val = tl.load(
                pvwork + rows[:, None] * 256 + tl.arange(0, 256)[None, :], volatile=True
            )
            safe = tl.where(denom > 0, denom, 1.0)
            result = val / safe[:, None]
            lse = tl.where(denom > 0, m + tl.log(safe), -float("inf"))
            qrows = qt * 128 + rows
            keep = qrows < qlen
            tl.store(
                O
                + (qstart.to(tl.int64) + qrows.to(tl.int64))[:, None] * OR
                + head * OH
                + tl.arange(0, 256)[None, :],
                result,
                keep[:, None],
            )
            tl.store(LSE + head.to(tl.int64) * TOTAL_Q + qstart + qrows, lse, keep)


# ---- body_task_queue ----


@triton.jit
def _body_queue_preflight(
    CUQ,
    USED,
    REQUEST_IDS,
    REQUEST_COUNT,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    REQUEST_LIST: tl.constexpr,
    SKIP_SHORT_KV: tl.constexpr,
):
    request_count = B
    if REQUEST_LIST:
        request_count = tl.load(REQUEST_COUNT)
    supported = True
    total = 0
    for request in range(request_count):
        batch = request
        if REQUEST_LIST:
            batch = tl.load(REQUEST_IDS + request)
        qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
        klen = tl.load(USED + batch)
        handled = False
        if SKIP_SHORT_KV:
            handled = (qlen == 1) & (klen <= SKIP_SHORT_KV)
        if not handled:
            head_rows = tl.minimum(
                H // HK, tl.maximum(1, (128 // tl.maximum(qlen, 1)) // 2 * 2)
            )
            supported = supported & (head_rows == 1) & (klen >= qlen) & (klen <= 1536)
            ntiles = tl.cdiv(qlen, 128) * tl.cdiv(H // HK, head_rows) * HK
            ntiles *= tl.cdiv(tl.minimum(qlen, 128) * head_rows, 128)
            tail = qlen % 128
            ntiles -= tl.where((qlen > 128) & (tail > 0) & (tail <= 16), H, 0)
            total += ntiles
    return supported, total


# ---- body_task_queue ----


@triton.jit
def _body_queue_partition_preflight(
    CUQ,
    USED,
    REQUEST_IDS,
    REQUEST_COUNT,
    B: tl.constexpr,
    H: tl.constexpr,
    REQUEST_LIST: tl.constexpr,
    SKIP_SHORT_KV: tl.constexpr,
):
    """Queue long requests without letting packed short requests reject them."""
    request_count = B
    if REQUEST_LIST:
        request_count = tl.load(REQUEST_COUNT)
    supported = True
    total = 0
    has_short = False
    for request in range(request_count):
        batch = request
        if REQUEST_LIST:
            batch = tl.load(REQUEST_IDS + request)
        qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
        klen = tl.load(USED + batch)
        handled = False
        if SKIP_SHORT_KV:
            handled = (qlen == 1) & (klen <= SKIP_SHORT_KV)
        if not handled:
            if qlen > 64:
                supported = supported & (klen >= qlen) & (klen <= 1536)
                ntiles = tl.cdiv(qlen, 128) * H
                tail = qlen % 128
                ntiles -= tl.where((qlen > 128) & (tail > 0) & (tail <= 16), H, 0)
                total += ntiles
            else:
                has_short = has_short | (qlen > 0)
    return supported, total, has_short


# ---- body_task_queue ----


@triton.jit
def _seek_body_task(
    global_task,
    request,
    begin,
    end,
    batch,
    qlen,
    klen,
    qstart,
    CUQ,
    USED,
    REQUEST_IDS,
    H: tl.constexpr,
    REQUEST_LIST: tl.constexpr,
    SKIP_SHORT_KV: tl.constexpr,
    PARTITION_SHORT_BODY: tl.constexpr = False,
):
    while global_task >= end:
        request += 1
        begin = end
        batch = request
        if REQUEST_LIST:
            batch = tl.load(REQUEST_IDS + request)
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        klen = tl.load(USED + batch)
        ntiles = tl.cdiv(qlen, 128) * H
        tail = qlen % 128
        ntiles -= tl.where((qlen > 128) & (tail > 0) & (tail <= 16), H, 0)
        if SKIP_SHORT_KV:
            ntiles = tl.where((qlen == 1) & (klen <= SKIP_SHORT_KV), 0, ntiles)
        if PARTITION_SHORT_BODY:
            ntiles = tl.where(qlen > 64, ntiles, 0)
        end += ntiles
    return request, begin, end, batch, qlen, klen, qstart


# ---- body_task_queue ----


@triton.jit
def _attention_body_task_queue(
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
    PARTITION_SHORT_BODY: tl.constexpr = False,
):
    worker_id = worker.to(tl.int32)
    worker = worker.to(tl.int64)
    group: tl.constexpr = H // HK
    STATE_UB: tl.constexpr = H // HK == 8
    if STATE_UB:
        state_ring = tle.dsa.alloc([384], tl.float32, tle.dsa.ascend.UB)
    query_buf = tle.dsa.alloc([128, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([256, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([128, 256], TASK_PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    count = tl.cdiv(tl.maximum(total_tasks - worker_id, 0), workers)
    old_request = -1
    old_begin = 0
    old_end = 0
    old_batch = 0
    old_qlen = 0
    old_klen = 0
    old_qstart = 0
    middle_request = -1
    middle_begin = 0
    middle_end = 0
    middle_batch = 0
    middle_qlen = 0
    middle_klen = 0
    middle_qstart = 0
    current_request = -1
    current_begin = 0
    current_end = 0
    current_batch = 0
    current_qlen = 0
    current_klen = 0
    current_qstart = 0
    for tick in range(count + 2):
        if tick >= 2:
            old_index = tick - 2
            old_slot = old_index % 3
            (
                old_request,
                old_begin,
                old_end,
                old_batch,
                old_qlen,
                old_klen,
                old_qstart,
            ) = _seek_body_task(
                worker_id + old_index * workers,
                old_request,
                old_begin,
                old_end,
                old_batch,
                old_qlen,
                old_klen,
                old_qstart,
                CUQ,
                USED,
                REQUEST_IDS,
                H,
                REQUEST_LIST,
                SKIP_SHORT_KV,
                PARTITION_SHORT_BODY,
            )
            old_task = worker_id + old_index * workers - old_begin
            old_qt = old_task // H
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
                middle_request,
                middle_begin,
                middle_end,
                middle_batch,
                middle_qlen,
                middle_klen,
                middle_qstart,
            ) = _seek_body_task(
                worker_id + middle_index * workers,
                middle_request,
                middle_begin,
                middle_end,
                middle_batch,
                middle_qlen,
                middle_klen,
                middle_qstart,
                CUQ,
                USED,
                REQUEST_IDS,
                H,
                REQUEST_LIST,
                SKIP_SHORT_KV,
                PARTITION_SHORT_BODY,
            )
            middle_task = worker_id + middle_index * workers - middle_begin
            middle_qt = middle_task // H
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
            if STATE_UB:
                middle_state = state_ring
            else:
                middle_state = ACCWORK + worker * 128 * 256 + middle_slot * 256
            sync_block_wait("cube", "vector", middle_slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            with tle.scope(core_mode="vector"):
                if middle_limit <= 256:
                    _whole_task_softmax(
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
                        STATE_UB,
                        middle_slot,
                    )
                elif middle_limit <= 512:
                    _whole_task_softmax(
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
                        STATE_UB,
                        middle_slot,
                    )
                elif middle_limit <= 1024:
                    _whole_task_softmax(
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
                        STATE_UB,
                        middle_slot,
                    )
                else:
                    _whole_task_softmax(
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
                        STATE_UB,
                        middle_slot,
                    )
            sync_block_set(
                "vector", "cube", 3 + middle_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
            )
        if tick < count:
            slot = tick % 3
            (
                current_request,
                current_begin,
                current_end,
                current_batch,
                current_qlen,
                current_klen,
                current_qstart,
            ) = _seek_body_task(
                worker_id + tick * workers,
                current_request,
                current_begin,
                current_end,
                current_batch,
                current_qlen,
                current_klen,
                current_qstart,
                CUQ,
                USED,
                REQUEST_IDS,
                H,
                REQUEST_LIST,
                SKIP_SHORT_KV,
                PARTITION_SHORT_BODY,
            )
            task = worker_id + tick * workers - current_begin
            qt = task // H
            head = task % H
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
            out_head = out_task % H
            out_pv = PVWORK + (worker * 3 + out_slot) * 128 * 256
            if STATE_UB:
                out_state = state_ring
            else:
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
                STATE_UB,
                out_slot,
            )
            sync_block_set("vector", "cube", 9 + out_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    for drain in range(tl.maximum(0, count - 3), count):
        sync_block_wait("vector", "cube", 9 + drain % 3, PIPE.PIPE_MTE3, PIPE.PIPE_S)


# ---- q1_streaming ----


@triton.jit
def stage_q1_window(
    RAW_K,
    RAW_V,
    TABLE,
    K,
    V,
    batch,
    kh,
    begin,
    length,
    KP: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    CAP: tl.constexpr,
    HK: tl.constexpr,
    BS: tl.constexpr,
):
    # K/V already have the native task's worker-local pointer rebase.
    vh = sub_vec_id().to(tl.int32)
    end = tl.minimum(length, begin + BS)
    end_page = tl.cdiv(end, 256) * (256 // PAGE)
    for page in range(begin // PAGE + vh, end_page, 2):
        safe_page = tl.minimum(page, tl.maximum(tl.cdiv(length, PAGE) - 1, 0))
        physical = tl.load(TABLE + batch.to(tl.int64) * TS + safe_page).to(tl.int64)
        rows = tl.arange(0, PAGE)[:, None]
        dims = tl.arange(0, 256)[None, :]
        valid = page * PAGE + rows < length
        key = tl.load(
            RAW_K + physical * KP + rows * KR + kh.to(tl.int64) * KH + dims, valid, 0
        )
        value = tl.load(
            RAW_V + physical * VP + rows * VR + kh.to(tl.int64) * VH + dims, valid, 0
        )
        offset = (
            (batch.to(tl.int64) * HK + kh) * CAP + page * PAGE + rows
        ) * 256 + dims
        tl.store(K + offset, key)
        tl.store(V + offset, value)


# ---- direct ----


@triton.jit
def _attention_task_direct(
    Q,
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
    SM_N: tl.constexpr = 0,
    D: tl.constexpr = 256,
    BD: tl.constexpr = 256,
):
    SM_WIDTH: tl.constexpr = SM_N if SM_N else BS
    worker = worker.to(tl.int64)
    if VARLEN:
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        qbase = qstart.to(tl.int64) * QR
        obase = qstart.to(tl.int64) * OR
    else:
        # Shared mha_varlan_fwd folds GQA heads into the Q sequence for decode.
        qlen = SEQLEN_Q
        qstart = batch * SEQLEN_Q
        qbase = batch.to(tl.int64) * QB
        obase = batch.to(tl.int64) * OB
    if SPLITS > 1:
        obase += split.to(tl.int64) * TOTAL_Q * H * D
    klen = tl.load(USED + batch)
    group: tl.constexpr = H // HK
    head_rows = tl.minimum(group, tl.maximum(1, (128 // tl.maximum(qlen, 1)) // 2 * 2))
    head_tiles = tl.cdiv(group, head_rows)
    qs = tl.maximum(1, tl.minimum(qlen, 128))
    subtiles = tl.cdiv(qs * head_rows, BM)
    sub = task % subtiles
    outer = task // subtiles
    qt = outer // (head_tiles * HK)
    headtile = outer % (head_tiles * HK)
    kh = headtile // head_tiles
    headstart = (headtile % head_tiles) * head_rows
    rr = sub * BM + tl.arange(0, BM)
    row = qt * 128 + rr % qs
    headlocal = headstart + rr // qs
    head = kh * group + headlocal
    validq = (rr < qs * head_rows) & (headlocal < group) & (row < qlen)
    dd = tl.arange(0, BD)
    if D == BD:
        q = tl.load(
            Q
            + qbase
            + row[:, None].to(tl.int64) * QR
            + head[:, None] * QH
            + dd[None, :],
            validq[:, None],
            0,
        )
    else:
        q = tl.load(
            Q
            + qbase
            + row[:, None].to(tl.int64) * QR
            + head[:, None] * QH
            + dd[None, :],
            validq[:, None] & (dd[None, :] < D),
            0,
        )
    sm_offsets = tl.arange(0, BM)[:, None] * BS + tl.arange(0, SM_WIDTH)[None, :]
    pv_offsets = tl.arange(0, BM)[:, None] * BD + dd[None, :]
    scratch = SCORES + worker * BM * BS
    prob = PROBS + worker * BM * BS
    pvwork = PVWORK + worker * BM * BD
    m = tl.full((BM,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
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
    for begin in range(start, limit, BS):
        stack = tl.minimum(BS, limit - begin)
        for chunk in range(0, stack, BN):
            key = load_kv_tile(
                K,
                TABLE,
                batch,
                kh,
                begin + chunk,
                limit,
                KP,
                KR,
                KH,
                TS,
                PAGE,
                BN,
                CONTIG_K,
                HK,
                D,
                BD,
            )
            qk = tl.dot(q, tl.trans(key))
            part = tl.arange(0, BM)[:, None] * BS + chunk + tl.arange(0, BN)[None, :]
            tl.store(scratch + part, qk)
        sync_block_set("cube", "vector", 12, PIPE.PIPE_FIX, PIPE.PIPE_S)
        sync_block_wait("cube", "vector", 12, PIPE.PIPE_FIX, PIPE.PIPE_S)
        cols = begin + tl.arange(0, SM_WIDTH)
        score = (
            tl.load(
                scratch + sm_offsets, cols[None, :] < begin + stack, 0, volatile=True
            )
            * SCALE
        )
        keep = validq[:, None] & (cols[None, :] < begin + stack)
        if CAUSAL:
            keep &= cols[None, :] <= row[:, None] + klen - qlen
        score = tl.where(keep, score, -float("inf"))
        newm = tl.maximum(m, tl.max(score, 1))
        safe_m = tl.where(newm == -float("inf"), 0.0, newm)
        p = tl.exp(score - safe_m[:, None])
        alpha = tl.exp(m - safe_m)
        normalizer = normalizer * alpha + tl.sum(p, 1)
        tl.store(prob + sm_offsets, p.to(PROBS.dtype.element_ty))
        sync_block_set("vector", "cube", 13, PIPE.PIPE_MTE3, PIPE.PIPE_S)
        sync_block_wait("vector", "cube", 13, PIPE.PIPE_MTE3, PIPE.PIPE_S)
        contribution = tl.zeros((BM, BD), tl.float32)
        for chunk in range(0, stack, BN):
            value = load_kv_tile(
                V,
                TABLE,
                batch,
                kh,
                begin + chunk,
                limit,
                VP,
                VR,
                VH,
                TS,
                PAGE,
                BN,
                CONTIG_K,
                HK,
                D,
                BD,
            )
            part = tl.arange(0, BM)[:, None] * BS + chunk + tl.arange(0, BN)[None, :]
            weights = tl.load(prob + part, volatile=True)
            contribution = tl.dot(weights, value, contribution)
        tl.store(pvwork + pv_offsets, contribution)
        sync_block_set("cube", "vector", 14, PIPE.PIPE_FIX, PIPE.PIPE_S)
        sync_block_wait("cube", "vector", 14, PIPE.PIPE_FIX, PIPE.PIPE_S)
        contribution = tl.load(pvwork + pv_offsets, volatile=True)
        acc = acc * alpha[:, None] + contribution
        m = newm
        # Vector has finished reading the previous Cube result before reuse.
        sync_block_set("vector", "cube", 15, PIPE.PIPE_MTE2, PIPE.PIPE_S)
        sync_block_wait("vector", "cube", 15, PIPE.PIPE_MTE2, PIPE.PIPE_S)
    safe_l = tl.where(normalizer > 0, normalizer, 1.0)
    result = acc / safe_l[:, None]
    lse = tl.where(normalizer > 0, m + tl.log(safe_l), -float("inf"))
    # Scalar row bases retain affine contiguous stores in the Ascend lowering.
    for r in range(BM):
        combined = sub * BM + r
        token = qt * 128 + combined % qs
        local_head = headstart + combined // qs
        if combined < qs * head_rows and local_head < group and token < qlen:
            output_head = kh * group + local_head
            values = extract_slice(result, (r, 0), (1, BD), (1, 1))
            if D == BD:
                tl.store(
                    O + obase + token.to(tl.int64) * OR + output_head * OH + dd,
                    tl.reshape(values, (BD,)),
                )
            else:
                tl.store(
                    O + obase + token.to(tl.int64) * OR + output_head * OH + dd,
                    tl.reshape(values, (BD,)),
                    dd < D,
                )
            row_lse = extract_slice(lse, (r,), (1,), (1,))
            tl.store(
                LSE
                + split.to(tl.int64) * H * TOTAL_Q
                + output_head.to(tl.int64) * TOTAL_Q
                + qstart
                + token,
                tl.reshape(row_lse, ()),
            )


# ---- direct ----


@triton.jit
def ascend_attention_persistent(
    Q,
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
    SKIP_SHORT_KV: tl.constexpr = 0,
    REQUEST_IDS=None,
    REQUEST_COUNT=None,
    REQUEST_LIST: tl.constexpr = False,
    D: tl.constexpr = 256,
    BD: tl.constexpr = 256,
):
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
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
            first = (worker + workers - cursor) % workers
            for task in range(first, ntiles, workers):
                _attention_task_direct(
                    Q,
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
                    batch,
                    task // active_splits,
                    task % active_splits,
                    worker,
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
                    D=D,
                    BD=BD,
                )
            cursor = (cursor + ntiles) % workers


# ---- direct_pipelined ----


@triton.jit
def _softmax_rows(
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
    for x2_vs in range(0, BM // 2, VM):
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


# ---- direct_pipelined ----


@triton.jit
def _softmax_rows_dispatch(
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
    SINGLE_HEAD_INDEX: tl.constexpr,
    ROW_END=None,
):
    if not SINGLE_HEAD_INDEX:
        _softmax_rows(
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
            WIDTH,
            CAUSAL,
            SCALE,
            VM,
            False,
            ROW_END,
        )
    elif head_rows == 1:
        _softmax_rows(
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
            WIDTH,
            CAUSAL,
            SCALE,
            VM,
            True,
            ROW_END,
        )
    else:
        _softmax_rows(
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
            WIDTH,
            CAUSAL,
            SCALE,
            VM,
            False,
            ROW_END,
        )


# ---- direct_pipelined ----


@triton.jit
def _softmax_rows_nomask(
    scratch,
    prob,
    state_buf,
    slot,
    x2_vf,
    BM: tl.constexpr,
    BS: tl.constexpr,
    WIDTH: tl.constexpr,
    SCALE: tl.constexpr,
    VM: tl.constexpr,
):
    for x2_vs in range(0, BM // 2, VM):
        x2_offsets = (x2_vf + x2_vs + tl.arange(0, VM))[:, None] * BS + tl.arange(
            0, WIDTH
        )[None, :]
        x2_score = tl.load(scratch + x2_offsets, volatile=True) * SCALE
        x2_mv = tle.dsa.subview(state_buf, [0, x2_vs], [1, VM], [1, 1])
        x2_lv = tle.dsa.subview(state_buf, [1, x2_vs], [1, VM], [1, 1])
        x2_av = tle.dsa.subview(state_buf, [2 + slot, x2_vs], [1, VM], [1, 1])
        x2_m = tl.reshape(tle.dsa.to_tensor(x2_mv, writable=False), (VM,))
        x2_l = tl.reshape(tle.dsa.to_tensor(x2_lv, writable=False), (VM,))
        x2_newm = tl.maximum(x2_m, tl.max(x2_score, 1))
        x2_safe = tl.where(x2_newm == -float("inf"), 0.0, x2_newm)
        x2_p = tl.exp(x2_score - x2_safe[:, None])
        x2_alpha = tl.exp(x2_m - x2_safe)
        x2_p_cast = x2_p.to(prob.dtype.element_ty)
        x2_local_sum = tl.sum(x2_p, 1)
        tl.store(prob + x2_offsets, x2_p_cast)
        x2_l = x2_l * x2_alpha + x2_local_sum
        tle.dsa.to_buffer(x2_newm[None, :], bind_buffer=x2_mv)
        tle.dsa.to_buffer(x2_l[None, :], bind_buffer=x2_lv)
        tle.dsa.to_buffer(x2_alpha[None, :], bind_buffer=x2_av)


# ---- direct_pipelined ----


@triton.jit
def _softmax_stack(
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
    SINGLE_HEAD_INDEX: tl.constexpr,
    ROW_END=None,
):
    # Prove visibility for all BM rows once, outside the per-VM loop.
    first = sub * BM
    last = first + BM - 1
    same_head = first // qs == last // qs
    min_row = qt * 128 + tl.where(same_head, first % qs, 0)
    max_row = qt * 128 + tl.where(same_head, last % qs, qs - 1)
    full = (
        (stack == WIDTH)
        & (last < qs * head_rows)
        & (headstart + last // qs < group)
        & (max_row < qlen)
    )
    if CAUSAL:
        full &= begin + WIDTH - 1 <= min_row + klen - qlen
    if full:
        _softmax_rows_nomask(
            scratch, prob, state_buf, slot, x2_vf, BM, BS, WIDTH, SCALE, VM
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
            WIDTH,
            CAUSAL,
            SCALE,
            VM,
            SINGLE_HEAD_INDEX,
            ROW_END,
        )


# ---- direct_pipelined ----


@triton.jit
def _attention_task(
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
):
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
        # Shared mha_varlan_fwd folds GQA heads into the Q sequence for decode.
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
        active_output_splits, _ = effective_kv_splits(klen, SPLITS, BS)
        write_final = active_output_splits == 1
    group: tl.constexpr = H // HK
    head_rows = tl.minimum(group, tl.maximum(1, (128 // tl.maximum(qlen, 1)) // 2 * 2))
    head_tiles = tl.cdiv(group, head_rows)
    qs = tl.maximum(1, tl.minimum(qlen, 128))
    subtiles = tl.cdiv(qs * head_rows, 128)
    sub = task % subtiles
    outer = task // subtiles
    qt = outer // (head_tiles * HK)
    headtile = outer % (head_tiles * HK)
    kh = headtile // head_tiles
    headstart = (headtile % head_tiles) * head_rows
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
        if Q_DIRECT and VARLEN and BM == 128:
            direct = (head_rows == 1) & (qt * 128 + 128 <= qlen)
            if direct:
                raw_head = kh * group + headstart
                raw_base = (
                    Q_RAW
                    + qbase
                    + (qt.to(tl.int64) * 128) * QR
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
                    x5_acc = tle.dsa.to_tensor(x5_accview, writable=False).to(
                        tl.float32
                    )
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
                            and x5_token < qlen
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


# ---- direct_pipelined ----


@triton.jit
def _stream_init_state(
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
    with tle.scope(core_mode="vector"):
        row_rr = sub_vec_id().to(tl.int32) * (BM // 2) + tl.arange(0, BM // 2)
        row_q = qt * 128 + row_rr
        row_valid = (row_rr < qs) & (headstart < group) & (row_q < qlen)
        row_end = tl.where(row_valid, row_q + klen - qlen + 1, 0)
        tle.dsa.to_buffer(row_end, bind_buffer=row_end_buf)
        init = tl.where(tl.arange(0, 5)[:, None] == 0, -float("inf"), 0.0)
        tle.dsa.to_buffer(tl.broadcast_to(init, (5, BM // 2)), bind_buffer=state_buf)


# ---- direct_pipelined ----


@triton.jit
def _stream_init_o(
    acc_buf,
    ACCWORK,
    worker,
    BM: tl.constexpr,
    ACC_UB_ROWS: tl.constexpr,
    COMPACT_O: tl.constexpr = False,
):
    with tle.scope(core_mode="vector"):
        for init_vs in range(0, ACC_UB_ROWS, 8):
            init_acc = tle.dsa.subview(acc_buf, [init_vs, 0], [8, 256], [1, 1])
            if COMPACT_O:
                tle.dsa.to_buffer(tl.zeros((8, 256), tl.bfloat16), bind_buffer=init_acc)
            else:
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


# ---- direct_pipelined ----


@triton.jit
def _stream_load_q(
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
    with tle.scope(core_mode="cube"):
        if Q_DIRECT:
            direct = qt * 128 + 128 <= qlen
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
                    Q + ((batch.to(tl.int64) * QCAP + task) * 128) * 256,
                    (BM, 256),
                    (256, 1),
                    (0, 0),
                    (BM, 256),
                    (1, 0),
                )
                tle.dsa.copy(query_ptr, query_buf, [BM, 256])
        else:
            query_ptr = tl.make_block_ptr(
                Q + ((batch.to(tl.int64) * QCAP + task) * 128) * 256,
                (BM, 256),
                (256, 1),
                (0, 0),
                (BM, 256),
                (1, 0),
            )
            tle.dsa.copy(query_ptr, query_buf, [BM, 256])


# ---- direct_pipelined ----


@triton.jit
def _stream_qk_only(
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
    with tle.scope(core_mode="cube"):
        q1 = tle.dsa.to_tensor(query_buf, writable=False)
        pair_end = ((stack + 127) // 256) * 256
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
            kbase = (
                K + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + pair_end) * 256
            )
            kptr_0 = tl.make_block_ptr(
                kbase, (128, 256), (256, 1), (0, 0), (128, 256), (1, 0)
            )
            tle.dsa.copy(kptr_0, key_buf_0, [128, 256])
            key_0 = tle.dsa.to_tensor(key_buf_0, writable=False)
            qk_0 = tl.dot(q1, tl.trans(key_0))
            soff_0 = (
                tl.arange(0, BM)[:, None] * BS + pair_end + tl.arange(0, 128)[None, :]
            )
            tl.store(scratch + soff_0, qk_0)
    sync_block_set("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)


# ---- direct_pipelined ----


@triton.jit
def _stream_softmax_only(
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


# ---- direct_pipelined ----


@triton.jit
def _stream_pv_o(
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
    O_VM: tl.constexpr = 16 if ACC_UB_ROWS == 16 else 32
    if BM == 128:
        if previous >= 3:
            sync_block_wait("vector", "cube", 9 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
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
                    V + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + cv) * 256,
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
            pvwork + tl.arange(0, BM)[:, None] * 256 + tl.arange(0, 256)[None, :], c2
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


# ---- direct_pipelined ----


@triton.jit
def _stream_finish(
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
                    x5_acc = tle.dsa.to_tensor(x5_accview, writable=False).to(
                        tl.float32
                    )
                else:
                    x5_acc = tl.load(x5_ab, volatile=True)
                x5_safe = tl.where(x5_l > 0, x5_l, 1.0)
                x5_result = x5_acc / x5_safe[:, None]
                x5_lse = tl.where(x5_l > 0, x5_m + tl.log(x5_safe), -float("inf"))
                store_rows = qt * 128 + x5_vf + x5_vs + tl.arange(0, VM)
                store_head = kh * group + headstart
                store_mask = (x5_vf + x5_vs + tl.arange(0, VM) < qs) & (
                    store_rows < qlen
                )
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


# ---- direct_pipelined ----


@triton.jit
def _attention_stream_task(
    Q,
    Q_RAW,
    K,
    V,
    O,
    LSE,
    SCORES,
    PROBS,
    PVWORK,
    ACCWORK,
    query_buf,
    key_buf_0,
    key_buf_1,
    value_buf,
    prob_buf,
    acc_buf,
    state_buf,
    row_end_buf,
    batch,
    task,
    ntiles,
    workers,
    worker,
    qlen,
    klen,
    qstart,
    slot_base,
    seeded,
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
    group: tl.constexpr = H // HK
    qs = tl.maximum(1, tl.minimum(qlen, 128))
    qt = task // H
    headtile = task % H
    kh = headtile // group
    headstart = headtile % group
    qbase = qstart.to(tl.int64) * QR
    obase = qstart.to(tl.int64) * OR
    limit = tl.minimum(
        klen, tl.maximum(0, tl.minimum(qlen, qt * 128 + 128) + klen - qlen)
    )
    stacks = tl.cdiv(limit, BS)
    next_task = task + workers
    next_qt = next_task // H
    next_headtile = next_task % H
    next_kh = next_headtile // group
    next_headstart = next_headtile % group
    next_limit = tl.minimum(
        klen, tl.maximum(0, tl.minimum(qlen, next_qt * 128 + 128) + klen - qlen)
    )
    next_stacks = tl.cdiv(next_limit, BS)
    next_base = (slot_base + stacks) % 3
    prepare_next = (next_task < ntiles) & (next_stacks > 0)

    # Only the Softmax state was initialized while the preceding task owned O.
    # Its final output has now consumed O; one shared O accumulator is safe here.
    _stream_init_o(acc_buf, ACCWORK, worker, BM, ACC_UB_ROWS, COMPACT_O)
    _stream_init_state(state_buf, row_end_buf, qt, qs, headstart, qlen, klen, group, BM)
    if seeded == 0:
        _stream_load_q(
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
            group,
            QR,
            QH,
            QCAP,
            Q_DIRECT,
            BM,
        )
    for step in range(0, stacks + 2):
        if step < stacks:
            slot = (slot_base + step) % 3
            begin = step * BS
            stack = tl.minimum(BS, limit - begin)
            scratch = SCORES + (worker * 3 + slot) * 128 * BS
            prob = PROBS + (worker * 3 + slot) * 128 * BS
            if (step != 0) | (seeded == 0):
                _stream_qk_only(
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
                    SINGLE_HEAD_INDEX,
                )
            _stream_softmax_only(
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
                SINGLE_HEAD_INDEX,
            )
        if (step == stacks) & prepare_next:
            # Current no longer reads Q. Seed exactly one next stack, without
            # touching its O/PVWORK or consuming any old O_done generation.
            _stream_load_q(
                Q,
                Q_RAW,
                query_buf,
                batch,
                qbase,
                qlen,
                next_task,
                next_qt,
                next_kh,
                next_headstart,
                group,
                QR,
                QH,
                QCAP,
                Q_DIRECT,
                BM,
            )
            next_scratch = SCORES + (worker * 3 + next_base) * 128 * BS
            next_prob = PROBS + (worker * 3 + next_base) * 128 * BS
            _stream_qk_only(
                K,
                next_scratch,
                next_prob,
                query_buf,
                key_buf_0,
                key_buf_1,
                state_buf,
                row_end_buf,
                batch,
                next_kh,
                0,
                tl.minimum(BS, next_limit),
                next_base,
                qs,
                next_qt,
                next_headstart,
                qlen,
                klen,
                HK,
                group,
                CONTIG_K,
                BM,
                BS,
                SCALE,
                SINGLE_HEAD_INDEX,
            )
        if step >= 2:
            previous = step - 2
            prevslot = (slot_base + previous) % 3
            begin = previous * BS
            stack = tl.minimum(BS, limit - begin)
            _stream_pv_o(
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
    for drain in range(tl.minimum(stacks, 3)):
        drain_slot = (slot_base + drain) % 3
        sync_block_wait("vector", "cube", 9 + drain_slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    _stream_finish(
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
        H,
        group,
        TOTAL_Q,
        OR,
        OH,
        BM,
        ACC_UB_ROWS,
    )
    return next_base, prepare_next


# ---- direct_pipelined ----


@triton.jit
def _attention_request_stream(
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
    worker = worker.to(tl.int64)
    qstart = tl.load(CUQ + batch)
    query_buf = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    key_buf_1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
    value_buf = tle.dsa.alloc([BN, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
    prob_buf = tle.dsa.alloc([BM, BN], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
    state_0 = tle.dsa.alloc([5, BM // 2], tl.float32, tle.dsa.ascend.UB)
    row_end_0 = tle.dsa.alloc([BM // 2], tl.int32, tle.dsa.ascend.UB)
    if COMPACT_O:
        acc_buf = tle.dsa.alloc([ACC_UB_ROWS, 256], tl.bfloat16, tle.dsa.ascend.UB)
    else:
        acc_buf = tle.dsa.alloc([ACC_UB_ROWS, 256], tl.float32, tle.dsa.ascend.UB)
    slot_base = 0
    seeded = False
    for task in range(first, ntiles, workers):
        slot_base, seeded = _attention_stream_task(
            Q,
            Q_RAW,
            K,
            V,
            O,
            LSE,
            SCORES,
            PROBS,
            PVWORK,
            ACCWORK,
            query_buf,
            key_buf_0,
            key_buf_1,
            value_buf,
            prob_buf,
            acc_buf,
            state_0,
            row_end_0,
            batch,
            task,
            ntiles,
            workers,
            worker,
            qlen,
            klen,
            qstart,
            slot_base,
            seeded,
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
            BM,
            BN,
            BS,
            ACC_UB_ROWS,
            COMPACT_O,
        )


# ---- direct_pipelined ----


@triton.jit
def ascend_attention_persistent_pipelined(
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
    RESIDUAL_MERGE: tl.constexpr = False,
    WAVE_PARTIAL=None,
    WAVE_LSE=None,
    WAVE_IDS=None,
    WAVE_COUNT=None,
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
    if RESIDUAL_MERGE:
        tl.static_assert(TASK_M == 0 and SPLITS == 1 and H // HK == 8)
        with tle.scope(core_mode="vector"):
            wave_count = tl.load(WAVE_COUNT)
            wave_prefix = (wave_count * HK // workers * workers) // HK
            wave_residual = wave_count - wave_prefix
            wave_enabled = (
                (workers % HK == 0)
                & (wave_residual > 0)
                & (wave_residual * HK <= workers // 4)
            )
            if wave_enabled:
                for wave_task in range(
                    worker * 2 + sub_vec_id(), wave_residual * H, workers * 2
                ):
                    wave_batch = tl.load(WAVE_IDS + wave_prefix + wave_task // H)
                    wave_head = wave_task % H
                    _merge_request(
                        WAVE_PARTIAL,
                        WAVE_LSE,
                        O,
                        LSE,
                        CUQ,
                        USED,
                        OR,
                        OH,
                        TOTAL_Q,
                        H,
                        256,
                        4,
                        4,
                        BS,
                        wave_batch,
                        wave_head,
                        PARTIAL_Q=0,
                        MIN_SPLIT_K=0,
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
                # The queue drained all long requests; keep short packed tasks.
                # On failed preflight retain the entire original general path.
                handled = handled | (queue_supported & (qlen > 64))
            if not handled:
                head_rows = tl.minimum(
                    group, tl.maximum(1, (128 // tl.maximum(qlen, 1)) // 2 * 2)
                )
                ntiles = tl.cdiv(qlen, 128) * tl.cdiv(group, head_rows) * HK
                ntiles *= tl.cdiv(tl.minimum(qlen, 128) * head_rows, BM)
                active_splits, _ = effective_kv_splits(klen, SPLITS, BS)
                ntiles *= active_splits
                if CROSS_Q_TASK and not INCLUDE_COMPACT_TAIL:
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
