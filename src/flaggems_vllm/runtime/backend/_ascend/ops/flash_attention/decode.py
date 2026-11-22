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
from threading import Lock

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
from triton.runtime import driver

from flaggems_vllm.runtime.backend._ascend.ops.flash_attention.common import (
    _decode_actual_task,
    _packed_q_base,
    _packed_q_stage_base,
    _raw_stream_nowait,
    bind_runner,
    cube_core_limit,
    load_paged_tile,
)
from flaggems_vllm.runtime.backend._ascend.ops.flash_attention.direct import (
    _attention_task,
    _attention_task_direct,
    _native_compute_pv,
    _native_compute_qk,
    _native_update_accumulator,
    _native_update_softmax,
)
from flaggems_vllm.runtime.backend._ascend.ops.flash_attention.mixed import (
    _attention_task_native_pair,
    _stage_q1_native_tail,
)
from flaggems_vllm.runtime.backend._ascend.ops.flash_attention.scheduling import (
    _classify_long_requests,
    effective_kv_splits,
    merge_request_list,
    merge_splits,
)

# ---- decode ----


@triton.jit
def ascend_flash_attention_decode(
    Q,
    K,
    V,
    O,
    LSE,
    CUQ,
    CUK,
    USED,
    TABLE,
    scale,
    sq0: tl.constexpr,
    sq1: tl.constexpr,
    sk0: tl.constexpr,
    sk1: tl.constexpr,
    sk2: tl.constexpr,
    sv0: tl.constexpr,
    sv1: tl.constexpr,
    sv2: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    st0: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    row = tl.load(CUQ + batch)
    end = tl.load(CUQ + batch + 1)
    if row < end:
        if PAGED:
            length = tl.load(USED + batch)
            first = 0
        else:
            first = tl.load(CUK + batch)
            length = tl.load(CUK + batch + 1) - first
        cols, dims = tl.arange(0, BN), tl.arange(0, BD)
        kh = head // GROUP
        q = tl.load(Q + row.to(tl.int64) * sq0 + head * sq1 + dims, dims < D, 0).to(
            tl.float32
        )
        if PAGED:
            k = load_paged_tile(
                K, TABLE, batch, kh, 0, length, sk0, sk1, sk2, st0, PAGE, BN
            ).to(tl.float32)
        else:
            ko = (first + cols).to(tl.int64) * sk0 + kh * sk1
            k = tl.load(
                K + ko[:, None] + dims[None, :],
                (cols[:, None] < length) & (dims[None, :] < D),
                0,
            ).to(tl.float32)
        scores = tl.sum(k * q[None, :], 1) * scale
        scores = tl.where(cols < length, scores, -float("inf"))
        maximum = tl.max(scores, 0)
        safe_max = tl.where(length > 0, maximum, 0.0)
        weights = tl.exp(scores - safe_max)
        normalizer = tl.sum(weights, 0)
        denom = tl.where(length > 0, normalizer, 1.0)
        if PAGED:
            v = load_paged_tile(
                V, TABLE, batch, kh, 0, length, sv0, sv1, sv2, st0, PAGE, BN
            ).to(tl.float32)
        else:
            vo = (first + cols).to(tl.int64) * sv0 + kh * sv1
            v = tl.load(
                V + vo[:, None] + dims[None, :],
                (cols[:, None] < length) & (dims[None, :] < D),
                0,
            ).to(tl.float32)
        result = tl.sum(weights[:, None] * v, 0) / denom
        tl.store(O + row.to(tl.int64) * so0 + head * so1 + dims, result, dims < D)
        lse = tl.where(length > 0, maximum + tl.log(denom), -float("inf"))
        tl.store(LSE + head * TOTAL_Q + row, lse)


# ---- decode ----


@triton.jit
def ascend_flash_attention_decode_group(
    Q,
    K,
    V,
    O,
    LSE,
    CUQ,
    CUK,
    USED,
    TABLE,
    scale,
    sq0: tl.constexpr,
    sq1: tl.constexpr,
    sk0: tl.constexpr,
    sk1: tl.constexpr,
    sk2: tl.constexpr,
    sv0: tl.constexpr,
    sv1: tl.constexpr,
    sv2: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    st0: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    batch, kh = tl.program_id(0), tl.program_id(1)
    row = tl.load(CUQ + batch)
    end = tl.load(CUQ + batch + 1)
    if row < end:
        if PAGED:
            length = tl.load(USED + batch)
            first = 0
        else:
            first = tl.load(CUK + batch)
            length = tl.load(CUK + batch + 1) - first
        cols, dims = tl.arange(0, BN), tl.arange(0, BD)
        if length > 0:
            if PAGED:
                k = load_paged_tile(
                    K, TABLE, batch, kh, 0, length, sk0, sk1, sk2, st0, PAGE, BN
                ).to(tl.float32)
            else:
                ko = (first + cols).to(tl.int64) * sk0 + kh * sk1
                k = tl.load(
                    K + ko[:, None] + dims[None, :],
                    (cols[:, None] < length) & (dims[None, :] < D),
                    0,
                ).to(tl.float32)
            if PAGED:
                v = load_paged_tile(
                    V, TABLE, batch, kh, 0, length, sv0, sv1, sv2, st0, PAGE, BN
                ).to(tl.float32)
            else:
                vo = (first + cols).to(tl.int64) * sv0 + kh * sv1
                v = tl.load(
                    V + vo[:, None] + dims[None, :],
                    (cols[:, None] < length) & (dims[None, :] < D),
                    0,
                ).to(tl.float32)
            for gh in range(GROUP):
                head = kh * GROUP + gh
                q = tl.load(
                    Q + row.to(tl.int64) * sq0 + head * sq1 + dims, dims < D, 0
                ).to(tl.float32)
                scores = tl.sum(k * q[None, :], 1) * scale
                scores = tl.where(cols < length, scores, -float("inf"))
                maximum = tl.max(scores, 0)
                safe_max = tl.where(length > 0, maximum, 0.0)
                weights = tl.exp(scores - safe_max)
                normalizer = tl.sum(weights, 0)
                denom = tl.where(length > 0, normalizer, 1.0)
                result = tl.sum(weights[:, None] * v, 0) / denom
                tl.store(
                    O + row.to(tl.int64) * so0 + head * so1 + dims, result, dims < D
                )
                lse = tl.where(length > 0, maximum + tl.log(denom), -float("inf"))
                tl.store(LSE + head * TOTAL_Q + row, lse)
        else:
            for empty_gh in range(GROUP):
                empty_head = kh * GROUP + empty_gh
                tl.store(
                    O + row.to(tl.int64) * so0 + empty_head * so1 + dims, 0.0, dims < D
                )
                tl.store(LSE + empty_head * TOTAL_Q + row, -float("inf"))


# ---- decode ----


@triton.jit
def ascend_flash_attention_single_kv(
    Q,
    K,
    V,
    O,
    LSE,
    CUQ,
    CUK,
    USED,
    TABLE,
    scale,
    sq0: tl.constexpr,
    sq1: tl.constexpr,
    sk0: tl.constexpr,
    sk1: tl.constexpr,
    sk2: tl.constexpr,
    sv0: tl.constexpr,
    sv1: tl.constexpr,
    sv2: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    st0: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    PAGED: tl.constexpr,
    BG: tl.constexpr,
    BD: tl.constexpr,
):
    kh, batch = tl.program_id(0), tl.program_id(1)
    row = tl.load(CUQ + batch)
    end = tl.load(CUQ + batch + 1)
    if row < end:
        if PAGED:
            length = tl.load(USED + batch)
            physical = tl.load(TABLE + batch * st0, length > 0, 0).to(tl.int64)
            ko, vo = physical * sk0 + kh * sk2, physical * sv0 + kh * sv2
        else:
            first = tl.load(CUK + batch)
            length = tl.load(CUK + batch + 1) - first
            ko = first.to(tl.int64) * sk0 + kh * sk1
            vo = first.to(tl.int64) * sv0 + kh * sv1
        g, dims = tl.arange(0, BG), tl.arange(0, BD)
        heads = kh * GROUP + g
        valid = (g[:, None] < GROUP) & (dims[None, :] < D)
        q = tl.load(
            Q + row.to(tl.int64) * sq0 + heads[:, None] * sq1 + dims[None, :], valid, 0
        ).to(tl.float32)
        k = tl.load(K + ko + dims, (dims < D) & (length > 0), 0).to(tl.float32)
        v = tl.load(V + vo + dims, (dims < D) & (length > 0), 0)
        scores = tl.sum(q * k[None, :], 1) * scale
        lse = tl.where(length > 0, scores, -float("inf"))
        tl.store(
            O + row.to(tl.int64) * so0 + heads[:, None] * so1 + dims[None, :],
            v[None, :],
            valid,
        )
        tl.store(LSE + heads * TOTAL_Q + row, lse, g < GROUP)


# ---- decode ----


def launch_decode(q, k, v, out, lse, cuq, cuk, used, table, scale, max_k):
    total, heads, dim = q.shape
    batch = cuq.numel() - 1
    group = heads // k.shape[-2]
    paged = table is not None
    args = (
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        used,
        table,
        float(scale),
        *q.stride()[:2],
        *k.stride()[:3],
        *v.stride()[:3],
        *out.stride()[:2],
        table.stride(0) if paged else 0,
        total,
        group,
        dim,
    )
    if max_k <= 1 and group <= 32:
        # Issue adjacent KV heads before moving to the next request. This keeps
        # their grouped Q/output rows adjacent in the singleton decode wave.
        ascend_flash_attention_single_kv[(k.shape[-2], batch)](
            *args,
            paged,
            triton.next_power_of_2(group),
            triton.next_power_of_2(dim),
            enable_fp_fusion=False,
        )
    elif max_k <= 32 and group <= 8:
        ascend_flash_attention_decode_group[(batch, k.shape[-2])](
            *args,
            k.shape[1] if paged else 1,
            paged,
            triton.next_power_of_2(max(1, max_k)),
            triton.next_power_of_2(dim),
            enable_fp_fusion=False,
        )
    else:
        ascend_flash_attention_decode[(batch, heads)](
            *args,
            k.shape[1] if paged else 1,
            paged,
            triton.next_power_of_2(max(1, max_k)),
            triton.next_power_of_2(dim),
            enable_fp_fusion=False,
        )


# ---- decode ----


@triton.jit
def ascend_flash_attention_decode_actual(
    Q,
    K,
    V,
    O,
    LSE,
    CUQ,
    CUK,
    USED,
    TABLE,
    scale,
    sq0: tl.constexpr,
    sq1: tl.constexpr,
    sk0: tl.constexpr,
    sk1: tl.constexpr,
    sk2: tl.constexpr,
    sv0: tl.constexpr,
    sv1: tl.constexpr,
    sv2: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    st0: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    BG: tl.constexpr,
    REQUEST_IDS=None,
    REQUEST_COUNT=None,
    B: tl.constexpr = 0,
    CLASSIFY_BLOCK: tl.constexpr = 0,
):
    batch = tl.program_id(0)
    kh = tl.program_id(1)
    if CLASSIFY_BLOCK:
        if batch == B:
            if kh == 0:
                _classify_long_requests(
                    CUQ, USED, REQUEST_IDS, REQUEST_COUNT, B, CLASSIFY_BLOCK
                )
        else:
            _decode_actual_task(
                Q,
                K,
                V,
                O,
                LSE,
                CUQ,
                CUK,
                USED,
                TABLE,
                scale,
                sq0,
                sq1,
                sk0,
                sk1,
                sk2,
                sv0,
                sv1,
                sv2,
                so0,
                so1,
                st0,
                TOTAL_Q,
                GROUP,
                D,
                PAGE,
                PAGED,
                BN,
                BD,
                BG,
                tl.program_id(0),
                tl.program_id(1),
                0,
            )
    else:
        _decode_actual_task(
            Q,
            K,
            V,
            O,
            LSE,
            CUQ,
            CUK,
            USED,
            TABLE,
            scale,
            sq0,
            sq1,
            sk0,
            sk1,
            sk2,
            sv0,
            sv1,
            sv2,
            so0,
            so1,
            st0,
            TOTAL_Q,
            GROUP,
            D,
            PAGE,
            PAGED,
            BN,
            BD,
            BG,
            tl.program_id(0),
            tl.program_id(1),
            0,
        )


# ---- decode ----


def launch_decode_actual(params, request_ids=None, request_count=None):
    """Finish requests with one Q token and actual KV length at most 32."""
    classify = request_ids is not None
    ascend_flash_attention_decode_actual[(params.b + int(classify), params.hk)](
        params.q_ptr,
        params.k_ptr,
        params.v_ptr,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.cu_seqlens_q_ptr,
        None,
        params.seqused_k_ptr,
        params.page_table_ptr,
        params.scale_softmax,
        params.q_row_stride,
        params.q_head_stride,
        *params.k_ptr.stride()[:3],
        *params.v_ptr.stride()[:3],
        params.o_row_stride,
        params.o_head_stride,
        params.page_table_batch_stride,
        params.total_q,
        params.h // params.hk,
        params.d,
        params.block_size,
        True,
        32,
        256,
        triton.next_power_of_2(params.h // params.hk),
        REQUEST_IDS=request_ids,
        REQUEST_COUNT=request_count,
        B=params.b,
        CLASSIFY_BLOCK=triton.next_power_of_2(params.b) if classify else 0,
        enable_fp_fusion=False,
    )


# ---- decode_cube ----


@triton.jit
def ascend_flash_attention_decode_bounded(
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
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    KP: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
):
    batch, kh = tl.program_id(0), tl.program_id(1)
    row = tl.load(CUQ + batch)
    end = tl.load(CUQ + batch + 1)
    length = tl.load(USED + batch)
    if end - row == 1:
        if length <= 32:
            with tle.scope(core_mode="vector"):
                _decode_actual_task(
                    Q,
                    K,
                    V,
                    O,
                    LSE,
                    CUQ,
                    None,
                    USED,
                    TABLE,
                    SCALE,
                    QR,
                    QH,
                    KP,
                    KR,
                    KH,
                    VP,
                    VR,
                    VH,
                    OR,
                    OH,
                    TS,
                    TOTAL_Q,
                    H // HK,
                    256,
                    PAGE,
                    True,
                    32,
                    256,
                    H // HK // 2,
                    batch,
                    kh,
                    sub_vec_id(),
                    2,
                )
        elif length <= 256:
            worker = batch * HK + kh
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
                kh,
                worker - worker,
                worker,
                H,
                HK,
                TOTAL_Q,
                1,
                SCALE,
                QR,
                QH,
                0,
                OR,
                OH,
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
                False,
                16,
                32,
                512,
                SM_N=256,
            )


# ---- decode_cube ----


def launch_paged_decode(params):
    """Write final O/LSE only for actual KV lengths at most 256."""
    programs = params.b * params.hk
    scores = torch.empty(
        (programs, 16, 512), dtype=torch.float32, device=params.q_ptr.device
    )
    probs = torch.empty_like(scores, dtype=params.q_ptr.dtype)
    pv = torch.empty(
        (programs, 16, 256), dtype=torch.float32, device=params.q_ptr.device
    )
    ascend_flash_attention_decode_bounded[(params.b, params.hk)](
        params.q_ptr,
        params.k_ptr,
        params.v_ptr,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.page_table_ptr,
        scores,
        probs,
        pv,
        params.h,
        params.hk,
        params.total_q,
        params.q_row_stride,
        params.q_head_stride,
        params.o_row_stride,
        params.o_head_stride,
        *params.k_ptr.stride()[:3],
        *params.v_ptr.stride()[:3],
        params.page_table_batch_stride,
        params.block_size,
        params.scale_softmax,
        enable_fp_fusion=False,
        multibuffer=False,
    )


# ---- decode_cube ----


@triton.jit
def _decode256_pack_request(
    Q,
    K,
    V,
    PQ,
    PK,
    PV,
    TABLE,
    task,
    row,
    length,
    kh,
    batch,
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
):
    vh = sub_vec_id().to(tl.int32)
    if vh == 0:
        qr = tl.arange(0, 16)
        qd = tl.arange(0, 256)
        safe = kh * (H // HK) + tl.minimum(qr, H // HK - 1)
        query0 = tl.load(
            Q + row.to(tl.int64) * QR + safe[:, None] * QH + qd[None, :],
            qr[:, None] < H // HK,
            0,
        )
        tl.store(
            PQ + task.to(tl.int64) * 128 * 256 + qr[:, None] * 256 + qd[None, :],
            query0,
        )
    for page in range(vh, 256 // PAGE, 2):
        safe_page = tl.minimum(page, tl.cdiv(length, PAGE) - 1)
        physical = tl.load(TABLE + batch.to(tl.int64) * TS + safe_page).to(tl.int64)
        pr = tl.arange(0, PAGE)
        pd = tl.arange(0, 256)
        valid = page * PAGE + pr < length
        key0 = tl.load(
            K + physical * KP + pr[:, None] * KR + kh * KH + pd[None, :],
            valid[:, None],
            0,
        )
        val0 = tl.load(
            V + physical * VP + pr[:, None] * VR + kh * VH + pd[None, :],
            valid[:, None],
            0,
        )
        off = (task.to(tl.int64) * 512 + page * PAGE + pr[:, None]) * 256 + pd[None, :]
        tl.store(PK + off, key0)
        tl.store(PV + off, val0)


# ---- decode_cube ----


@triton.jit
def _decode256_fused_stage(
    Q,
    K,
    V,
    PQ,
    PK,
    PV,
    O,
    LSE,
    CUQ,
    USED,
    TABLE,
    SCORES,
    PROBS,
    PVWORK,
    COUNT,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    KP: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    worker = tl.program_id(0)
    worker_count = tl.num_programs(0)
    with tle.scope(core_mode="vector"):
        if (worker == 0) & (sub_vec_id() == 0):
            ids = tl.arange(0, BLOCK_B)
            lens = tl.load(USED + ids, ids < B, 0)
            starts = tl.load(CUQ + ids, ids < B, 0)
            ends = tl.load(CUQ + ids + 1, ids < B, 0)
            count = tl.sum(
                ((ids < B) & (ends - starts == 1) & (lens > 256)).to(tl.int32), 0
            )
            tl.store(COUNT, count)
    first_batch = worker // HK
    first_len = tl.load(USED + first_batch)
    all_singleton = first_len == 1
    if all_singleton:
        first_row = tl.load(CUQ + first_batch)
        first_end = tl.load(CUQ + first_batch + 1)
        all_singleton = first_end - first_row == 1
    if all_singleton:
        for inspect_task in range(worker + worker_count, B * HK, worker_count):
            inspect_batch = inspect_task // HK
            inspect_row = tl.load(CUQ + inspect_batch)
            inspect_end = tl.load(CUQ + inspect_batch + 1)
            inspect_len = tl.load(USED + inspect_batch)
            all_singleton = (
                all_singleton & (inspect_end - inspect_row == 1) & (inspect_len == 1)
            )
    if all_singleton:
        with tle.scope(core_mode="vector"):
            singleton_vec = sub_vec_id().to(tl.int32)
            for singleton_task in range(
                worker + singleton_vec * worker_count, B * HK, worker_count * 2
            ):
                singleton_batch = singleton_task // HK
                singleton_kh = singleton_task % HK
                _decode_actual_task(
                    Q,
                    K,
                    V,
                    O,
                    LSE,
                    CUQ,
                    None,
                    USED,
                    TABLE,
                    SCALE,
                    QR,
                    QH,
                    KP,
                    KR,
                    KH,
                    VP,
                    VR,
                    VH,
                    OR,
                    OH,
                    TS,
                    TOTAL_Q,
                    H // HK,
                    256,
                    PAGE,
                    True,
                    1,
                    256,
                    H // HK,
                    singleton_batch,
                    singleton_kh,
                    0,
                    1,
                )
    else:
        with tle.scope(core_mode="vector"):
            batch0 = worker // HK
            kh0 = worker % HK
            row0 = tl.load(CUQ + batch0)
            end0 = tl.load(CUQ + batch0 + 1)
            len0 = tl.load(USED + batch0)
            if (end0 - row0 == 1) & (len0 > 32) & (len0 <= 256):
                _decode256_pack_request(
                    Q,
                    K,
                    V,
                    PQ,
                    PK,
                    PV,
                    TABLE,
                    worker,
                    row0,
                    len0,
                    kh0,
                    batch0,
                    H,
                    HK,
                    QR,
                    QH,
                    KP,
                    KR,
                    KH,
                    VP,
                    VR,
                    VH,
                    TS,
                    PAGE,
                )
                sync_block_set("vector", "cube", 11, PIPE.PIPE_MTE3, PIPE.PIPE_S)
        qb = tle.dsa.alloc([16, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
        kb0 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
        kb1 = tle.dsa.alloc([128, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
        vb = tle.dsa.alloc([256, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
        pb = tle.dsa.alloc([16, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
        state = tle.dsa.alloc([2, 8], tl.float32, tle.dsa.ascend.UB)
        for task in range(worker, B * HK, worker_count):
            batch = task // HK
            kh = task % HK
            row = tl.load(CUQ + batch)
            end = tl.load(CUQ + batch + 1)
            length = tl.load(USED + batch)
            score = SCORES + task.to(tl.int64) * 3 * 128 * 512
            prob = PROBS + task.to(tl.int64) * 3 * 128 * 512
            work = PVWORK + task.to(tl.int64) * 3 * 128 * 256
            if (end - row == 1) & (length > 32) & (length <= 256):
                sync_block_wait("vector", "cube", 11, PIPE.PIPE_MTE3, PIPE.PIPE_S)
                with tle.scope(core_mode="cube"):
                    qp = tl.make_block_ptr(
                        PQ + task.to(tl.int64) * 128 * 256,
                        (16, 256),
                        (256, 1),
                        (0, 0),
                        (16, 256),
                        (1, 0),
                    )
                    kp0 = tl.make_block_ptr(
                        PK + task.to(tl.int64) * 512 * 256,
                        (128, 256),
                        (256, 1),
                        (0, 0),
                        (128, 256),
                        (1, 0),
                    )
                    kp1 = tl.make_block_ptr(
                        PK + (task.to(tl.int64) * 512 + 128) * 256,
                        (128, 256),
                        (256, 1),
                        (0, 0),
                        (128, 256),
                        (1, 0),
                    )
                    tle.dsa.copy(qp, qb, [16, 256])
                    tle.dsa.copy(kp0, kb0, [128, 256])
                    tle.dsa.copy(kp1, kb1, [128, 256])
                    query = tle.dsa.to_tensor(qb, writable=False)
                    a = tl.dot(query, tl.trans(tle.dsa.to_tensor(kb0, writable=False)))
                    b = tl.dot(query, tl.trans(tle.dsa.to_tensor(kb1, writable=False)))
                    rr = tl.arange(0, 16)[:, None] * 256
                    cc = tl.arange(0, 128)[None, :]
                    tl.store(score + rr + cc, a)
                    tl.store(score + rr + 128 + cc, b)
                sync_block_set("cube", "vector", 12, PIPE.PIPE_FIX, PIPE.PIPE_S)
                sync_block_wait("cube", "vector", 12, PIPE.PIPE_FIX, PIPE.PIPE_S)
                with tle.scope(core_mode="vector"):
                    rows = sub_vec_id().to(tl.int32) * 8 + tl.arange(0, 8)
                    cols = tl.arange(0, 256)
                    scores = (
                        tl.load(
                            score + rows[:, None] * 256 + cols[None, :], volatile=True
                        )
                        * SCALE
                    )
                    scores = tl.where(cols[None, :] < length, scores, -float("inf"))
                    maximum = tl.max(scores, 1)
                    weights = tl.exp(scores - maximum[:, None])
                    denom = tl.sum(weights, 1)
                    tl.store(
                        prob + rows[:, None] * 256 + cols[None, :],
                        weights.to(Q.dtype.element_ty),
                    )
                    mv = tle.dsa.subview(state, [0, 0], [1, 8], [1, 1])
                    lv = tle.dsa.subview(state, [1, 0], [1, 8], [1, 1])
                    tle.dsa.to_buffer(maximum[None, :], bind_buffer=mv)
                    tle.dsa.to_buffer(denom[None, :], bind_buffer=lv)
                sync_block_set("vector", "cube", 13, PIPE.PIPE_MTE3, PIPE.PIPE_S)
                sync_block_wait("vector", "cube", 13, PIPE.PIPE_MTE3, PIPE.PIPE_S)
                with tle.scope(core_mode="cube"):
                    pp = tl.make_block_ptr(
                        prob, (16, 256), (256, 1), (0, 0), (16, 256), (1, 0)
                    )
                    vp = tl.make_block_ptr(
                        PV + task.to(tl.int64) * 512 * 256,
                        (256, 256),
                        (256, 1),
                        (0, 0),
                        (256, 256),
                        (1, 0),
                    )
                    tle.dsa.copy(pp, pb, [16, 256])
                    tle.dsa.copy(vp, vb, [256, 256])
                    cube_pv = tl.dot(
                        tle.dsa.to_tensor(pb, writable=False),
                        tle.dsa.to_tensor(vb, writable=False),
                    )
                    tl.store(
                        work
                        + tl.arange(0, 16)[:, None] * 256
                        + tl.arange(0, 256)[None, :],
                        cube_pv,
                    )
                sync_block_set("cube", "vector", 14, PIPE.PIPE_FIX, PIPE.PIPE_S)
            # Distinct per-request GM slots allow next staging during current PV.
            with tle.scope(core_mode="vector"):
                next_task = task + worker_count
                if next_task < B * HK:
                    next_batch = next_task // HK
                    next_kh = next_task % HK
                    next_row = tl.load(CUQ + next_batch)
                    next_end = tl.load(CUQ + next_batch + 1)
                    next_len = tl.load(USED + next_batch)
                    if (next_end - next_row == 1) & (next_len > 32) & (next_len <= 256):
                        _decode256_pack_request(
                            Q,
                            K,
                            V,
                            PQ,
                            PK,
                            PV,
                            TABLE,
                            next_task,
                            next_row,
                            next_len,
                            next_kh,
                            next_batch,
                            H,
                            HK,
                            QR,
                            QH,
                            KP,
                            KR,
                            KH,
                            VP,
                            VR,
                            VH,
                            TS,
                            PAGE,
                        )
                        sync_block_set(
                            "vector", "cube", 11, PIPE.PIPE_MTE3, PIPE.PIPE_S
                        )
            if (end - row == 1) & (length > 32) & (length <= 256):
                sync_block_wait("cube", "vector", 14, PIPE.PIPE_FIX, PIPE.PIPE_S)
                with tle.scope(core_mode="vector"):
                    first = sub_vec_id().to(tl.int32) * 8
                    rows = first + tl.arange(0, 8)
                    dims = tl.arange(0, 256)
                    values = tl.load(
                        work + rows[:, None] * 256 + dims[None, :], volatile=True
                    )
                    mv = tle.dsa.subview(state, [0, 0], [1, 8], [1, 1])
                    lv = tle.dsa.subview(state, [1, 0], [1, 8], [1, 1])
                    maximum = tl.reshape(tle.dsa.to_tensor(mv, writable=False), (8,))
                    denom = tl.reshape(tle.dsa.to_tensor(lv, writable=False), (8,))
                    result = values / denom[:, None]
                    lse = maximum + tl.log(denom)
                    for idx in range(8):
                        head = kh * (H // HK) + first + idx
                        if first + idx < H // HK:
                            one = extract_slice(result, (idx, 0), (1, 256), (1, 1))
                            val = extract_slice(lse, (idx,), (1,), (1,))
                            tl.store(
                                O + row.to(tl.int64) * OR + head * OH + dims,
                                tl.reshape(one, (256,)),
                            )
                            tl.store(LSE + head * TOTAL_Q + row, tl.reshape(val, ()))
            with tle.scope(core_mode="vector"):
                if (end - row == 1) & (length <= 32):
                    _decode_actual_task(
                        Q,
                        K,
                        V,
                        O,
                        LSE,
                        CUQ,
                        None,
                        USED,
                        TABLE,
                        SCALE,
                        QR,
                        QH,
                        KP,
                        KR,
                        KH,
                        VP,
                        VR,
                        VH,
                        OR,
                        OH,
                        TS,
                        TOTAL_Q,
                        H // HK,
                        256,
                        PAGE,
                        True,
                        32,
                        256,
                        (H // HK) // 2,
                        batch,
                        kh,
                        sub_vec_id(),
                        2,
                    )


# ---- decode_cube ----


@triton.jit
def _decode256_long_counted(
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
    COUNT,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    KP: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
):
    if tl.load(COUNT) > 0:
        for task in range(tl.program_id(0), B * HK, tl.num_programs(0)):
            batch = task // HK
            kh = task % HK
            row = tl.load(CUQ + batch)
            end = tl.load(CUQ + batch + 1)
            length = tl.load(USED + batch)
            if (end - row == 1) & (length > 256):
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
                    kh,
                    task - task,
                    task,
                    H,
                    HK,
                    TOTAL_Q,
                    1,
                    SCALE,
                    QR,
                    QH,
                    0,
                    OR,
                    OH,
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
                    False,
                    16,
                    32,
                    512,
                    SM_N=512,
                )


# ---- decode_cube ----


def launch_decode256_staged(params):
    n = params.b * params.hk
    dev = params.q_ptr.device
    pq = torch.empty((n, 128, 256), dtype=params.q_ptr.dtype, device=dev)
    pk = torch.empty((n, 512, 256), dtype=params.q_ptr.dtype, device=dev)
    pv = torch.empty_like(pk)
    scores = torch.empty((n, 3, 128, 512), dtype=torch.float32, device=dev)
    probs = torch.empty_like(scores, dtype=params.q_ptr.dtype)
    work = torch.empty((n, 3, 128, 256), dtype=torch.float32, device=dev)
    count = torch.empty((), dtype=torch.int32, device=dev)
    constants = (
        params.b,
        params.h,
        params.hk,
        params.total_q,
        params.q_row_stride,
        params.q_head_stride,
        params.o_row_stride,
        params.o_head_stride,
        *params.k_ptr.stride()[:3],
        *params.v_ptr.stride()[:3],
        params.page_table_batch_stride,
        params.block_size,
        params.scale_softmax,
    )
    stage_workers = min(
        n, torch.npu.get_device_limit(params.q_ptr.device.index)["cube_core_num"]
    )
    _decode256_fused_stage[(stage_workers,)](
        params.q_ptr,
        params.k_ptr,
        params.v_ptr,
        pq,
        pk,
        pv,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.page_table_ptr,
        scores,
        probs,
        work,
        count,
        *constants,
        triton.next_power_of_2(params.b),
        enable_fp_fusion=False,
        disable_auto_inject_block_sync=True,
        multibuffer=False,
    )
    if params.seqlen_k > 256:
        workers = min(
            n, torch.npu.get_device_limit(params.q_ptr.device.index)["cube_core_num"]
        )
        _decode256_long_counted[(workers,)](
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            params.o_ptr,
            params.softmax_lse_ptr,
            params.cu_seqlens_q_ptr,
            params.seqused_k_ptr,
            params.page_table_ptr,
            scores,
            probs,
            work,
            count,
            *constants,
            enable_fp_fusion=False,
            multibuffer=False,
        )


# ---- decode_localstage ----


@triton.jit
def _stage_decode_split(
    Q,
    K,
    V,
    TABLE,
    QP,
    KP,
    VP,
    batch,
    kh,
    worker,
    qstart,
    first,
    length,
    H: tl.constexpr,
    HK: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    CAP: tl.constexpr,
):
    with tle.scope(core_mode="vector"):
        vh = sub_vec_id().to(tl.int32)
        w = worker.to(tl.int64)
        if vh == 0:
            rr = tl.arange(0, 16)
            dd = tl.arange(0, 256)
            group: tl.constexpr = H // HK
            head = kh * group + tl.minimum(rr, group - 1)
            query = tl.load(
                Q + qstart.to(tl.int64) * QR + head[:, None] * QH + dd[None, :],
                rr[:, None] < group,
                0,
            )
            tl.store(QP + w * 128 * 256 + rr[:, None] * 256 + dd[None, :], query)
        pages = tl.cdiv(length, 512) * (512 // PAGE)
        for page in range(vh, pages, 2):
            rows = tl.arange(0, PAGE)
            dd = tl.arange(0, 256)
            absolute_page = first // PAGE + page
            # Padded split pages never dereference a page beyond this request.
            safe_page = tl.minimum(absolute_page, tl.cdiv(first + length, PAGE) - 1)
            physical = tl.load(TABLE + batch.to(tl.int64) * TS + safe_page).to(tl.int64)
            keep = page * PAGE + rows < length
            key = tl.load(
                K
                + physical * KS0
                + rows[:, None] * KS1
                + kh.to(tl.int64) * KS2
                + dd[None, :],
                keep[:, None],
                0,
            )
            value = tl.load(
                V
                + physical * VS0
                + rows[:, None] * VS1
                + kh.to(tl.int64) * VS2
                + dd[None, :],
                keep[:, None],
                0,
            )
            dst = (w * CAP + page * PAGE + rows[:, None]) * 256 + dd[None, :]
            tl.store(KP + dst, key)
            tl.store(VP + dst, value)
    sync_block_set("vector", "cube", 12, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    sync_block_wait("vector", "cube", 12, PIPE.PIPE_MTE3, PIPE.PIPE_S)


# ---- decode_localstage ----


@triton.jit
def ascend_flash_attention_decode_split_staged(
    Q,
    K,
    V,
    O,
    LSE,
    PARTIAL,
    PARTIAL_LSE,
    CUQ,
    USED,
    TABLE,
    QP,
    KP,
    VP,
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
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    CAP: tl.constexpr,
    SPLITS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    cursor = 0
    for batch in range(B):
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        length = tl.load(USED + batch)
        active, span = effective_kv_splits(length, SPLITS, 512)
        active = tl.where(length <= 32, 1, active)
        tasks = tl.where(qlen == 1, HK * active, 0)
        first_task = (worker + workers - cursor) % workers
        for task in range(first_task, tasks, workers):
            kh = task // active
            split = task % active
            if length <= 32:
                with tle.scope(core_mode="vector"):
                    if sub_vec_id() == 0:
                        _decode_actual_task(
                            Q,
                            K,
                            V,
                            O,
                            LSE,
                            CUQ,
                            None,
                            USED,
                            TABLE,
                            SCALE,
                            QR,
                            QH,
                            KS0,
                            KS1,
                            KS2,
                            VS0,
                            VS1,
                            VS2,
                            OR,
                            OH,
                            TS,
                            TOTAL_Q,
                            H // HK,
                            256,
                            PAGE,
                            True,
                            32,
                            256,
                            H // HK,
                            batch,
                            kh,
                            0,
                        )
            else:
                first = split * span
                split_length = tl.minimum(span, length - first)
                _stage_decode_split(
                    Q,
                    K,
                    V,
                    TABLE,
                    QP,
                    KP,
                    VP,
                    batch,
                    kh,
                    worker,
                    qstart,
                    first,
                    split_length,
                    H,
                    HK,
                    QR,
                    QH,
                    KS0,
                    KS1,
                    KS2,
                    VS0,
                    VS1,
                    VS2,
                    TS,
                    PAGE,
                    CAP,
                )
                # Native helper retains original batch/head and absolute KV
                # coordinates; rebase only its private operand stage pointers.
                qoff = (worker.to(tl.int64) - batch.to(tl.int64) * HK - kh) * 128 * 256
                koff = (
                    (worker.to(tl.int64) - batch.to(tl.int64) * HK - kh) * CAP - first
                ) * 256
                _attention_task(
                    QP + qoff,
                    Q,
                    KP + koff,
                    VP + koff,
                    PARTIAL,
                    PARTIAL_LSE,
                    CUQ,
                    USED,
                    TABLE,
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
                    1,
                    SCALE,
                    QR,
                    QH,
                    0,
                    H * 256,
                    256,
                    0,
                    KS0,
                    KS1,
                    KS2,
                    VS0,
                    VS1,
                    VS2,
                    TS,
                    PAGE,
                    True,
                    CAUSAL,
                    16,
                    256,
                    512,
                    SPLITS,
                    CAP,
                    QCAP=HK,
                    ACC_UB_ROWS=8,
                    FINAL_O=O,
                    FINAL_LSE=LSE,
                    FINAL_OR=OR,
                    FINAL_OH=OH,
                    FINAL_TOTAL_Q=TOTAL_Q,
                )
        cursor = (cursor + tasks) % workers


# ---- decode_localstage ----


def decode_split_stage_layout(params, plan, workers):
    splits = plan.kv_splits
    # Below32K the existing splitter deliberately uses one span. Above32K its
    # aligned span is monotonic in length. Both bounds are needed for graph
    # replay with different device lengths but unchanged host metadata.
    capacity = max(
        triton.cdiv(min(params.seqlen_k, 32767), 512) * 512,
        triton.cdiv(params.seqlen_k, splits * 512) * 512,
    )
    workers = min(workers, params.b * params.hk * splits)
    return workers, capacity


def launch_decode_split_staged(params, plan, workers):
    """Use one stage/partial launch followed by the existing stable merge."""
    splits = plan.kv_splits
    workers, capacity = decode_split_stage_layout(params, plan, workers)
    alloc = lambda shape, dtype: torch.empty(
        shape, dtype=dtype, device=params.q_ptr.device
    )
    qstage = alloc((workers, 128, 256), params.q_ptr.dtype)
    kstage = alloc((workers, capacity, 256), params.q_ptr.dtype)
    vstage = alloc((workers, capacity, 256), params.q_ptr.dtype)
    scores = alloc((workers, 3, 128, 512), torch.float32)
    probs = alloc((workers, 3, 128, 512), params.q_ptr.dtype)
    pvwork = alloc((workers, 3, 128, 256), torch.float32)
    accwork = alloc((workers, 128, 256), torch.float32)
    partial = alloc((splits, params.total_q, params.h, 256), torch.float32)
    partial_lse = alloc((splits, params.h, params.total_q), torch.float32)
    ascend_flash_attention_decode_split_staged[(workers,)](
        params.q_ptr,
        params.k_ptr,
        params.v_ptr,
        params.o_ptr,
        params.softmax_lse_ptr,
        partial,
        partial_lse,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.page_table_ptr,
        qstage,
        kstage,
        vstage,
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
        *params.k_ptr.stride()[:3],
        *params.v_ptr.stride()[:3],
        params.o_row_stride,
        params.o_head_stride,
        params.page_table_batch_stride,
        params.block_size,
        capacity,
        splits,
        params.is_causal,
        disable_auto_inject_block_sync=True,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    merge_splits[(params.b, params.h)](
        partial,
        partial_lse,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.o_row_stride,
        params.o_head_stride,
        params.total_q,
        params.h,
        256,
        splits,
        triton.next_power_of_2(splits),
        512,
        SKIP_SHORT_KV=32,
        enable_fp_fusion=False,
    )


# ---- decode_qdirect_cube ----


@triton.jit
def _qdirect_short_fallback(
    Q,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    TABLE,
    SCALE,
    QR: tl.constexpr,
    QH: tl.constexpr,
    KP: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TS: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    FP32_P: tl.constexpr,
    batch,
    kh,
    part,
):
    if not FP32_P:
        _decode_actual_task(
            Q,
            K,
            V,
            O,
            LSE,
            CUQ,
            None,
            USED,
            TABLE,
            SCALE,
            QR,
            QH,
            KP,
            KR,
            KH,
            VP,
            VR,
            VH,
            OR,
            OH,
            TS,
            TOTAL_Q,
            8,
            256,
            32,
            True,
            32,
            256,
            4,
            batch,
            kh,
            part,
            2,
        )
    else:
        row = tl.load(CUQ + batch)
        end = tl.load(CUQ + batch + 1)
        length = tl.load(USED + batch)
        if end - row == 1:
            dims, cols = (tl.arange(0, 256), tl.arange(0, 32))
            if length > 0:
                keys = load_paged_tile(
                    K, TABLE, batch, kh, 0, length, KP, KR, KH, TS, 32, 32
                ).to(tl.float32)
                values = load_paged_tile(
                    V, TABLE, batch, kh, 0, length, VP, VR, VH, TS, 32, 32
                ).to(tl.float32)
                for gh in range(4):
                    head = kh * 8 + part * 4 + gh
                    query = tl.load(Q + row.to(tl.int64) * QR + head * QH + dims).to(
                        tl.float32
                    )
                    scores = tl.sum(keys * query[None, :], 1) * SCALE
                    scores = tl.where(cols < length, scores, -float("inf"))
                    maximum = tl.max(scores, 0)
                    weights = tl.exp(scores - maximum)
                    denom = tl.sum(weights, 0)
                    result = tl.sum(weights[:, None] * values, 0) / denom
                    tl.store(O + row.to(tl.int64) * OR + head * OH + dims, result)
                    tl.store(LSE + head * TOTAL_Q + row, maximum + tl.log(denom))
            else:
                for gh in range(4):
                    head = kh * 8 + part * 4 + gh
                    tl.store(O + row.to(tl.int64) * OR + head * OH + dims, 0.0)
                    tl.store(LSE + head * TOTAL_Q + row, -float("inf"))


# ---- decode_qdirect_cube ----


@triton.jit
def ascend_attention_decode_qdirect_persistent(
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
    REQUEST_IDS,
    REQUEST_COUNT,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    KP: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    CLASSIFY: tl.constexpr,
    BLOCK_B: tl.constexpr,
    FP32_P: tl.constexpr,
    WORKERS: tl.constexpr,
):
    worker = tl.program_id(0)
    for job in range(worker, B * HK, WORKERS):
        batch, kh = job // HK, job % HK
        if batch < B:
            row = tl.load(CUQ + batch)
            end = tl.load(CUQ + batch + 1)
            length = tl.load(USED + batch)
            if (end - row == 1) & (length == 32):
                task = worker
                qb = tle.dsa.alloc([16, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
                kb = tle.dsa.alloc([32, 256], K.dtype.element_ty, tle.dsa.ascend.L1)
                vb = tle.dsa.alloc([32, 256], V.dtype.element_ty, tle.dsa.ascend.L1)
                pb = tle.dsa.alloc([16, 32], PROBS.dtype.element_ty, tle.dsa.ascend.L1)
                state = tle.dsa.alloc([2, 8], tl.float32, tle.dsa.ascend.UB)
                score = SCORES + task.to(tl.int64) * 16 * 32
                prob = PROBS + task.to(tl.int64) * 16 * 32
                work = PVWORK + task.to(tl.int64) * 16 * 256
                physical = tl.load(TABLE + batch.to(tl.int64) * TS).to(tl.int64)
                with tle.scope(core_mode="cube"):
                    qp = tl.make_block_ptr(
                        Q + row.to(tl.int64) * QR,
                        (16, 256),
                        (QH, 1),
                        (0, 0),
                        (16, 256),
                        (1, 0),
                    )
                    kp = tl.make_block_ptr(
                        K + physical * KP + kh * KH,
                        (32, 256),
                        (KR, 1),
                        (0, 0),
                        (32, 256),
                        (1, 0),
                    )
                    tle.dsa.copy(qp, qb, [16, 256])
                    tle.dsa.copy(kp, kb, [32, 256])
                    cube_qk = tl.dot(
                        tle.dsa.to_tensor(qb, writable=False),
                        tl.trans(tle.dsa.to_tensor(kb, writable=False)),
                    )
                    tl.store(
                        score
                        + tl.arange(0, 16)[:, None] * 32
                        + tl.arange(0, 32)[None, :],
                        cube_qk,
                    )
                sync_block_set("cube", "vector", 12, PIPE.PIPE_FIX, PIPE.PIPE_S)
                sync_block_wait("cube", "vector", 12, PIPE.PIPE_FIX, PIPE.PIPE_S)
                with tle.scope(core_mode="vector"):
                    rows = kh * 8 + sub_vec_id().to(tl.int32) * 4 + tl.arange(0, 4)
                    cols = tl.arange(0, 32)
                    scores = (
                        tl.load(
                            score + rows[:, None] * 32 + cols[None, :], volatile=True
                        )
                        * SCALE
                    )
                    maximum = tl.max(scores, 1)
                    weights = tl.exp(scores - maximum[:, None])
                    denom = tl.sum(weights, 1)
                    tl.store(
                        prob + rows[:, None] * 32 + cols[None, :],
                        weights.to(Q.dtype.element_ty),
                    )
                    unused_rows = (
                        (1 - kh) * 8 + sub_vec_id().to(tl.int32) * 4 + tl.arange(0, 4)
                    )
                    tl.store(
                        prob + unused_rows[:, None] * 32 + cols[None, :],
                        tl.full((4, 32), 0, Q.dtype.element_ty),
                    )
                    mv = tle.dsa.subview(state, [0, 0], [1, 4], [1, 1])
                    lv = tle.dsa.subview(state, [1, 0], [1, 4], [1, 1])
                    tle.dsa.to_buffer(maximum[None, :], bind_buffer=mv)
                    tle.dsa.to_buffer(denom[None, :], bind_buffer=lv)
                sync_block_set("vector", "cube", 13, PIPE.PIPE_MTE3, PIPE.PIPE_S)
                sync_block_wait("vector", "cube", 13, PIPE.PIPE_MTE3, PIPE.PIPE_S)
                with tle.scope(core_mode="cube"):
                    pp = tl.make_block_ptr(
                        prob, (16, 32), (32, 1), (0, 0), (16, 32), (1, 0)
                    )
                    vp = tl.make_block_ptr(
                        V + physical * VP + kh * VH,
                        (32, 256),
                        (VR, 1),
                        (0, 0),
                        (32, 256),
                        (1, 0),
                    )
                    tle.dsa.copy(pp, pb, [16, 32])
                    tle.dsa.copy(vp, vb, [32, 256])
                    cube_pv = tl.dot(
                        tle.dsa.to_tensor(pb, writable=False),
                        tle.dsa.to_tensor(vb, writable=False),
                    )
                    tl.store(
                        work
                        + tl.arange(0, 16)[:, None] * 256
                        + tl.arange(0, 256)[None, :],
                        cube_pv,
                    )
                sync_block_set("cube", "vector", 14, PIPE.PIPE_FIX, PIPE.PIPE_S)
                sync_block_wait("cube", "vector", 14, PIPE.PIPE_FIX, PIPE.PIPE_S)
                with tle.scope(core_mode="vector"):
                    rows = kh * 8 + sub_vec_id().to(tl.int32) * 4 + tl.arange(0, 4)
                    dims = tl.arange(0, 256)
                    values = tl.load(
                        work + rows[:, None] * 256 + dims[None, :], volatile=True
                    )
                    mv = tle.dsa.subview(state, [0, 0], [1, 4], [1, 1])
                    lv = tle.dsa.subview(state, [1, 0], [1, 4], [1, 1])
                    maximum = tl.reshape(tle.dsa.to_tensor(mv, writable=False), (4,))
                    denom = tl.reshape(tle.dsa.to_tensor(lv, writable=False), (4,))
                    result = values / denom[:, None]
                    lse = maximum + tl.log(denom)
                    tl.store(
                        O + row.to(tl.int64) * OR + rows[:, None] * OH + dims[None, :],
                        result,
                    )
                    tl.store(LSE + rows * TOTAL_Q + row, lse)
            with tle.scope(core_mode="vector"):
                if (end - row == 1) & (length != 32):
                    _qdirect_short_fallback(
                        Q,
                        K,
                        V,
                        O,
                        LSE,
                        CUQ,
                        USED,
                        TABLE,
                        SCALE,
                        QR,
                        QH,
                        KP,
                        KR,
                        KH,
                        VP,
                        VR,
                        VH,
                        OR,
                        OH,
                        TS,
                        TOTAL_Q,
                        FP32_P,
                        batch,
                        kh,
                        sub_vec_id(),
                    )

        # Both Vector subcores finish their output stores before this worker
        # reuses score/prob/PV buffers for another request/head task.
        sync_block_set("vector", "cube", 15, PIPE.PIPE_MTE3, PIPE.PIPE_S)
        sync_block_wait("vector", "cube", 15, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    if CLASSIFY:
        with tle.scope(core_mode="vector"):
            if (worker == 0) & (sub_vec_id() == 0):
                rows = tl.arange(0, BLOCK_B)
                first = tl.load(CUQ + rows, rows < B, 0)
                end = tl.load(CUQ + rows + 1, rows < B, 0)
                length = tl.load(USED + rows, rows < B, 0)
                qlen = end - first
                keep = (rows < B) & ((qlen > 1) | ((qlen == 1) & (length > 32)))
                count = tl.sum(keep.to(tl.int32), 0)
                # The next kernel consumes IDS only when COUNT is nonzero.
                if count > 0:
                    positions = tl.cumsum(keep.to(tl.int32), 0) - 1
                    tl.store(REQUEST_IDS + positions, rows, keep)
                tl.store(REQUEST_COUNT, count)


# ---- decode_qdirect_cube ----


def launch_qdirect_decode(params, request_ids=None, request_count=None):
    workers = torch.npu.get_device_limit(params.q_ptr.device.index)["cube_core_num"]
    n, dev = workers, params.q_ptr.device
    scores = torch.empty((n, 16, 32), dtype=torch.float32, device=dev)
    probs = torch.empty_like(scores, dtype=params.q_ptr.dtype)
    work = torch.empty((n, 16, 256), dtype=torch.float32, device=dev)
    classify = request_ids is not None
    ascend_attention_decode_qdirect_persistent[(workers,)](
        params.q_ptr,
        params.k_ptr,
        params.v_ptr,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.page_table_ptr,
        scores,
        probs,
        work,
        request_ids,
        request_count,
        params.b,
        params.h,
        params.hk,
        params.total_q,
        params.q_row_stride,
        params.q_head_stride,
        params.o_row_stride,
        params.o_head_stride,
        *params.k_ptr.stride()[:3],
        *params.v_ptr.stride()[:3],
        params.page_table_batch_stride,
        params.block_size,
        params.scale_softmax,
        classify,
        triton.next_power_of_2(params.b),
        params.seqlen_k <= 32,
        workers,
        enable_fp_fusion=False,
        disable_auto_inject_block_sync=True,
        multibuffer=False,
    )


# ---- q1_native_pair ----


# ---- q1_native_pair ----


# ---- q1_localstage ----


@triton.jit
def _stage_q1_request(
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
        vh = sub_vec_id().to(tl.int32)
        w64 = worker.to(tl.int64)
        if vh == 0:
            rr = tl.arange(0, 16)
            dd = tl.arange(0, 256)
            group: tl.constexpr = H // HK
            safe_head = kh * group + tl.minimum(rr, group - 1)
            query = tl.load(
                Q_RAW
                + qstart.to(tl.int64) * QR
                + safe_head[:, None] * QH
                + dd[None, :],
                rr[:, None] < group,
                0,
            )
            if Q_SCALE != 1.0:
                query = (query.to(tl.float32) * Q_SCALE).to(QSTAGE.dtype.element_ty)
            tl.store(
                QSTAGE
                + (_packed_q_stage_base(B, QCAP) + w64) * 128 * 256
                + rr[:, None] * 256
                + dd[None, :],
                query,
            )
        stage_length = length
        if STREAM_Q1:
            stage_length = tl.minimum(length, 512)
        pages = tl.cdiv(stage_length, 256) * (256 // PAGE)
        for page in range(vh, pages, 2):
            safe_page = tl.minimum(page, tl.maximum(tl.cdiv(length, PAGE) - 1, 0))
            physical = tl.load(TABLE + batch.to(tl.int64) * TS + safe_page).to(tl.int64)
            rows = tl.arange(0, PAGE)
            dims = tl.arange(0, 256)
            keep = page * PAGE + rows < length
            key = tl.load(
                RAW_K
                + physical * KP
                + rows[:, None] * KR
                + kh.to(tl.int64) * KH
                + dims[None, :],
                keep[:, None],
                0,
            )
            value = tl.load(
                RAW_V
                + physical * VP
                + rows[:, None] * VR
                + kh.to(tl.int64) * VH
                + dims[None, :],
                keep[:, None],
                0,
            )
            offset = ((B * HK + w64) * CAP + page * PAGE + rows[:, None]) * 256 + dims[
                None, :
            ]
            tl.store(KSTAGE + offset, key)
            tl.store(VSTAGE + offset, value)
    sync_block_set("vector", "cube", 12, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    sync_block_wait("vector", "cube", 12, PIPE.PIPE_MTE3, PIPE.PIPE_S)


# ---- q1_localstage ----


@triton.jit
def ascend_attention_short_kv_staged(
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
    STAGED_Q1: tl.constexpr = False,
    STREAM_Q1: tl.constexpr = False,
    RAW_K=None,
    RAW_V=None,
    RAW_PAGE_PAIR: tl.constexpr = False,
    NATIVE_SPLIT_Q1: tl.constexpr = False,
    MIN_SPLIT_K: tl.constexpr = 32768,
    QUERY_SCALE: tl.constexpr = 1.0,
    COMPACT_NATIVE_TAIL: tl.constexpr = False,
):
    tl.static_assert(not COMPACT_NATIVE_TAIL or (STAGED_Q1 and RAW_PAGE_PAIR))
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
                group, tl.maximum(1, 128 // tl.maximum(qlen, 1) // 2 * 2)
            )
            ntiles = tl.cdiv(qlen, 128) * tl.cdiv(group, head_rows) * HK
            ntiles *= tl.cdiv(tl.minimum(qlen, 128) * head_rows, BM)
            active_splits, _ = effective_kv_splits(klen, SPLITS, BS, MIN_SPLIT_K)
            ntiles *= active_splits
            tail_rows = qlen % 128
            eligible_tail = (qlen > 128) & (tail_rows > 0) & (tail_rows <= 16)
            original_offset = 0
            if TAIL_ONLY:
                ntiles = tl.where(qlen == 1, ntiles, tl.where(eligible_tail, H, 0))
                original_offset = tl.where(qlen == 1, 0, (tl.cdiv(qlen, 128) - 1) * H)
            else:
                ntiles -= tl.where(eligible_tail, H, 0)
            first = (worker + workers - cursor) % workers
            for task in range(first, ntiles, workers):
                q_offset = tl.full((), 0, tl.int64)
                kv_offset = tl.full((), 0, tl.int64)
                if STAGED_Q1:
                    if qlen == 1:
                        qstart = tl.load(CUQ + batch)
                        stage_head = task
                        if NATIVE_SPLIT_Q1:
                            stage_head = task // active_splits
                        if RAW_PAGE_PAIR:
                            _stage_q1_native_tail(
                                Q_RAW,
                                RAW_K,
                                RAW_V,
                                TABLE,
                                Q,
                                K,
                                V,
                                batch,
                                stage_head,
                                worker,
                                qstart,
                                klen,
                                H,
                                HK,
                                QR,
                                QH,
                                KP,
                                KR,
                                KH,
                                VP,
                                VR,
                                VH,
                                TS,
                                PAGE,
                                CONTIG_K,
                                B,
                                QCAP,
                                STREAM_Q1,
                                Q_SCALE=QUERY_SCALE,
                                COMPACT_NATIVE_TAIL=COMPACT_NATIVE_TAIL,
                            )
                        else:
                            _stage_q1_request(
                                Q_RAW,
                                RAW_K,
                                RAW_V,
                                TABLE,
                                Q,
                                K,
                                V,
                                batch,
                                stage_head,
                                worker,
                                qstart,
                                klen,
                                H,
                                HK,
                                QR,
                                QH,
                                KP,
                                KR,
                                KH,
                                VP,
                                VR,
                                VH,
                                TS,
                                PAGE,
                                CONTIG_K,
                                B,
                                QCAP,
                                STREAM_Q1,
                                Q_SCALE=QUERY_SCALE,
                            )
                        q_offset = (
                            (
                                _packed_q_stage_base(B, QCAP)
                                + worker.to(tl.int64)
                                - _packed_q_base(batch, qstart, QCAP, H)
                                - stage_head
                            )
                            * 128
                            * 256
                        )
                        packed_k_base = (
                            batch.to(tl.int64) * HK + stage_head
                        ) * CONTIG_K
                        if COMPACT_NATIVE_TAIL:
                            # Bias the packed base so the native helper's final
                            # logical page pair lands in this worker's tail slot.
                            tail_start = klen.to(tl.int64) // (2 * PAGE) * (2 * PAGE)
                            kv_offset = (
                                B * HK * CONTIG_K
                                + worker.to(tl.int64) * (2 * PAGE)
                                - packed_k_base
                                - tail_start
                            ) * 256
                        else:
                            kv_offset = (
                                (B * HK + worker.to(tl.int64)) * CONTIG_K
                                - packed_k_base
                            ) * 256
                if RAW_PAGE_PAIR:
                    _attention_task_native_pair(
                        Q + q_offset,
                        Q_RAW,
                        K + kv_offset,
                        V + kv_offset,
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
                        Q1_STREAM=STREAM_Q1,
                        RAW_K=RAW_K,
                        RAW_V=RAW_V,
                        NATIVE_SPLIT_Q1=NATIVE_SPLIT_Q1,
                        MIN_SPLIT_K=MIN_SPLIT_K,
                    )
                else:
                    _attention_task(
                        Q + q_offset,
                        Q_RAW,
                        K + kv_offset,
                        V + kv_offset,
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
                        Q1_STREAM=STREAM_Q1,
                        RAW_K=RAW_K,
                        RAW_V=RAW_V,
                    )
            cursor = (cursor + ntiles) % workers


# ---- q1_only ----


@triton.jit
def _attention_task_q1_only(
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
        and (BS == 512)
        and (SPLITS == 1 or (NATIVE_SPLIT_Q1 and SPLITS == 4))
        and (H // HK == 4 or H // HK == 8)
    )
    tl.static_assert(
        not NATIVE_SPLIT_Q1
        or (
            COMPACT_Q1
            and BYPASS_SHORT
            and (PAGE == 16 or PAGE == 32)
            and (ACC_UB_ROWS == 8)
        )
    )
    VM: tl.constexpr = 8 if BM == 16 else 16
    SOFTMAX_VM_256: tl.constexpr = 32 if BM == 128 else VM
    O_VM: tl.constexpr = 32 if BM == 128 and ACC_UB_ROWS == 32 else VM
    worker = worker.to(tl.int64)
    qstart = tl.load(CUQ + batch)
    qlen = 1
    qbase = qstart.to(tl.int64) * QR
    obase = qstart.to(tl.int64) * OR
    output_qstart = qstart
    if SPLITS > 1:
        obase += split.to(tl.int64) * TOTAL_Q * H * 256
    klen = tl.load(USED + batch)
    write_final = False
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
                    Q
                    + (
                        (_packed_q_base(batch, qstart, QCAP, H) + outer) * 128
                        + sub * BM
                    )
                    * 256,
                    (BM, 256),
                    (256, 1),
                    (0, 0),
                    (BM, 256),
                    (1, 0),
                )
                tle.dsa.copy(query_ptr, query_buf, [BM, 256])
        else:
            query_ptr = tl.make_block_ptr(
                Q
                + ((_packed_q_base(batch, qstart, QCAP, H) + outer) * 128 + sub * BM)
                * 256,
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
                        + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + 0) * 256
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
            sync_block_wait("vector", "cube", 3 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
            with tle.scope(core_mode="cube"):
                _native_compute_pv(
                    batch,
                    begin,
                    kh,
                    klen,
                    native_prob_0,
                    native_prob_1,
                    native_prob_full,
                    native_value_0,
                    native_value_1,
                    native_value_2,
                    native_value_3,
                    prob,
                    pvwork,
                    stack,
                    BM,
                    BS,
                    CONTIG_K,
                    H,
                    HK,
                    PAGE,
                    Q1_STREAM,
                    RAW_V,
                    SPLITS,
                    TABLE,
                    TS,
                    V,
                    VH,
                    VP,
                    VR,
                )
            sync_block_set("cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            sync_block_wait("cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            with tle.scope(core_mode="vector"):
                x4_vf = sub_vec_id().to(tl.int32) * (BM // 2)
                _native_update_accumulator(
                    acc_buf,
                    prevslot,
                    pvwork,
                    state_buf,
                    worker,
                    x4_vf,
                    ACCWORK,
                    ACC_UB_ROWS,
                    BM,
                    O_VM,
                )
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
                _native_compute_qk(
                    batch,
                    begin,
                    kh,
                    klen,
                    native_key_0,
                    native_key_1,
                    native_key_2,
                    native_key_3,
                    q1,
                    scratch,
                    stack,
                    BM,
                    BS,
                    CONTIG_K,
                    H,
                    HK,
                    K,
                    KH,
                    KP,
                    KR,
                    PAGE,
                    Q1_STREAM,
                    RAW_K,
                    SPLITS,
                    TABLE,
                    TS,
                )
            sync_block_set("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            sync_block_wait("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
            with tle.scope(core_mode="vector"):
                x2_vf = sub_vec_id().to(tl.int32) * (BM // 2)
                _native_update_softmax(
                    begin,
                    group,
                    head_rows,
                    headstart,
                    klen,
                    prob,
                    qlen,
                    qs,
                    qt,
                    row_end_buf,
                    scratch,
                    slot,
                    stack,
                    state_buf,
                    sub,
                    x2_vf,
                    BM,
                    BS,
                    CAUSAL,
                    CONTIG_K,
                    SCALE,
                    SINGLE_HEAD_INDEX,
                    SOFTMAX_VM_256,
                    VM,
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


# ---- q1_only ----


@triton.jit
def ascend_attention_q1_specialized_staged(
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
    STAGED_Q1: tl.constexpr = False,
    STREAM_Q1: tl.constexpr = False,
    RAW_K=None,
    RAW_V=None,
    RAW_PAGE_PAIR: tl.constexpr = False,
    NATIVE_SPLIT_Q1: tl.constexpr = False,
    MIN_SPLIT_K: tl.constexpr = 32768,
    QUERY_SCALE: tl.constexpr = 1.0,
):
    tl.static_assert(
        VARLEN
        and STAGED_Q1
        and STREAM_Q1
        and RAW_PAGE_PAIR
        and TAIL_ONLY
        and (SPLITS == 1)
        and (not COMPACT_Q1)
        and (not BYPASS_SHORT)
        and (not NATIVE_SPLIT_Q1)
        and (not MERGE_SPLITS)
        and (not SKIP_SHORT_KV)
        and (TASK_M == 16)
        and (H // HK == 8)
    )
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
                group, tl.maximum(1, 128 // tl.maximum(qlen, 1) // 2 * 2)
            )
            ntiles = tl.cdiv(qlen, 128) * tl.cdiv(group, head_rows) * HK
            ntiles *= tl.cdiv(tl.minimum(qlen, 128) * head_rows, BM)
            active_splits, _ = effective_kv_splits(klen, SPLITS, BS, MIN_SPLIT_K)
            ntiles *= active_splits
            tail_rows = qlen % 128
            eligible_tail = (qlen > 128) & (tail_rows > 0) & (tail_rows <= 16)
            original_offset = 0
            if TAIL_ONLY:
                ntiles = tl.where(qlen == 1, ntiles, tl.where(eligible_tail, H, 0))
                original_offset = tl.where(qlen == 1, 0, (tl.cdiv(qlen, 128) - 1) * H)
            else:
                ntiles -= tl.where(eligible_tail, H, 0)
            first = (worker + workers - cursor) % workers
            for task in range(first, ntiles, workers):
                q_offset = tl.full((), 0, tl.int64)
                kv_offset = tl.full((), 0, tl.int64)
                if STAGED_Q1:
                    if qlen == 1:
                        qstart = tl.load(CUQ + batch)
                        stage_head = task
                        if NATIVE_SPLIT_Q1:
                            stage_head = task // active_splits
                        if RAW_PAGE_PAIR:
                            _stage_q1_native_tail(
                                Q_RAW,
                                RAW_K,
                                RAW_V,
                                TABLE,
                                Q,
                                K,
                                V,
                                batch,
                                stage_head,
                                worker,
                                qstart,
                                klen,
                                H,
                                HK,
                                QR,
                                QH,
                                KP,
                                KR,
                                KH,
                                VP,
                                VR,
                                VH,
                                TS,
                                PAGE,
                                CONTIG_K,
                                B,
                                QCAP,
                                STREAM_Q1,
                                Q_SCALE=QUERY_SCALE,
                            )
                        else:
                            _stage_q1_request(
                                Q_RAW,
                                RAW_K,
                                RAW_V,
                                TABLE,
                                Q,
                                K,
                                V,
                                batch,
                                stage_head,
                                worker,
                                qstart,
                                klen,
                                H,
                                HK,
                                QR,
                                QH,
                                KP,
                                KR,
                                KH,
                                VP,
                                VR,
                                VH,
                                TS,
                                PAGE,
                                CONTIG_K,
                                B,
                                QCAP,
                                STREAM_Q1,
                            )
                        q_offset = (
                            (
                                _packed_q_stage_base(B, QCAP)
                                + worker.to(tl.int64)
                                - _packed_q_base(batch, qstart, QCAP, H)
                                - stage_head
                            )
                            * 128
                            * 256
                        )
                        kv_offset = (
                            (
                                B * HK
                                + worker.to(tl.int64)
                                - batch.to(tl.int64) * HK
                                - stage_head
                            )
                            * CONTIG_K
                            * 256
                        )
                if RAW_PAGE_PAIR:
                    if qlen == 1:
                        _attention_task_q1_only(
                            Q + q_offset,
                            Q_RAW,
                            K + kv_offset,
                            V + kv_offset,
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
                            Q1_STREAM=STREAM_Q1,
                            RAW_K=RAW_K,
                            RAW_V=RAW_V,
                            NATIVE_SPLIT_Q1=NATIVE_SPLIT_Q1,
                            MIN_SPLIT_K=MIN_SPLIT_K,
                        )
                    else:
                        _attention_task_native_pair(
                            Q + q_offset,
                            Q_RAW,
                            K + kv_offset,
                            V + kv_offset,
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
                            Q1_STREAM=STREAM_Q1,
                            RAW_K=RAW_K,
                            RAW_V=RAW_V,
                            NATIVE_SPLIT_Q1=NATIVE_SPLIT_Q1,
                            MIN_SPLIT_K=MIN_SPLIT_K,
                        )
                else:
                    _attention_task(
                        Q + q_offset,
                        Q_RAW,
                        K + kv_offset,
                        V + kv_offset,
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
                        Q1_STREAM=STREAM_Q1,
                        RAW_K=RAW_K,
                        RAW_V=RAW_V,
                    )
            cursor = (cursor + ntiles) % workers


# ---- q1_coreloops ----


@triton.jit
def _attention_task_q1_coreloops(
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
    worker64 = worker.to(tl.int64)
    tl.static_assert(
        BM == 16
        and BN == 256
        and (BS == 512)
        and (SPLITS == 1 or (NATIVE_SPLIT_Q1 and SPLITS == 4))
        and (H // HK == 4 or H // HK == 8)
    )
    tl.static_assert(
        not NATIVE_SPLIT_Q1
        or (
            COMPACT_Q1
            and BYPASS_SHORT
            and (PAGE == 16 or PAGE == 32)
            and (ACC_UB_ROWS == 8)
        )
    )
    VM: tl.constexpr = 8 if BM == 16 else 16
    SOFTMAX_VM_256: tl.constexpr = 32 if BM == 128 else VM
    O_VM: tl.constexpr = 32 if BM == 128 and ACC_UB_ROWS == 32 else VM
    group: tl.constexpr = H // HK
    query_buf = tle.dsa.alloc([BM, 256], Q.dtype.element_ty, tle.dsa.ascend.L1)
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
    with tle.scope(core_mode="cube"):
        qstart = tl.load(CUQ + batch)
        qlen = 1
        qbase = qstart.to(tl.int64) * QR
        obase = qstart.to(tl.int64) * OR
        output_qstart = qstart
        if SPLITS > 1:
            obase += split.to(tl.int64) * TOTAL_Q * H * 256
        klen = tl.load(USED + batch)
        write_final = False
        head_rows = tl.minimum(
            group, tl.maximum(1, 128 // tl.maximum(qlen, 1) // 2 * 2)
        )
        head_tiles = tl.cdiv(group, head_rows)
        qs = tl.maximum(1, tl.minimum(qlen, 128))
        subtiles = tl.cdiv(qs * head_rows, 128)
        sub = task % subtiles
        outer = task // subtiles
        qt = outer // (head_tiles * HK)
        headtile = outer % (head_tiles * HK)
        kh = headtile // head_tiles
        headstart = headtile % head_tiles * head_rows
        scratch = SCORES + worker64 * 128 * BS
        prob = PROBS + worker64 * 128 * BS
        pvwork = PVWORK + worker64 * 128 * 256
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
                    Q
                    + (
                        (_packed_q_base(batch, qstart, QCAP, H) + outer) * 128
                        + sub * BM
                    )
                    * 256,
                    (BM, 256),
                    (256, 1),
                    (0, 0),
                    (BM, 256),
                    (1, 0),
                )
                tle.dsa.copy(query_ptr, query_buf, [BM, 256])
        else:
            query_ptr = tl.make_block_ptr(
                Q
                + ((_packed_q_base(batch, qstart, QCAP, H) + outer) * 128 + sub * BM)
                * 256,
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
                prob = PROBS + (worker64 * 3 + prevslot) * 128 * BS
                pvwork = PVWORK + (worker64 * 3 + prevslot) * 128 * 256
                if BM == 128:
                    if previous >= 3:
                        sync_block_wait(
                            "vector", "cube", 9 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                        )
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
                        + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + 0) * 256
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
                sync_block_wait(
                    "vector", "cube", 3 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                )
                _native_compute_pv(
                    batch,
                    begin,
                    kh,
                    klen,
                    native_prob_0,
                    native_prob_1,
                    native_prob_full,
                    native_value_0,
                    native_value_1,
                    native_value_2,
                    native_value_3,
                    prob,
                    pvwork,
                    stack,
                    BM,
                    BS,
                    CONTIG_K,
                    H,
                    HK,
                    PAGE,
                    Q1_STREAM,
                    RAW_V,
                    SPLITS,
                    TABLE,
                    TS,
                    V,
                    VH,
                    VP,
                    VR,
                )
                sync_block_set(
                    "cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S
                )
            if step < stacks:
                slot = step % 3
                begin = start + step * BS
                stack = tl.minimum(BS, limit - begin)
                scratch = SCORES + (worker64 * 3 + slot) * 128 * BS
                prob = PROBS + (worker64 * 3 + slot) * 128 * BS
                if BM != 128:
                    if step >= 3:
                        sync_block_wait(
                            "vector", "cube", 9 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                        )
                q1 = tle.dsa.to_tensor(query_buf, writable=False)
                _native_compute_qk(
                    batch,
                    begin,
                    kh,
                    klen,
                    native_key_0,
                    native_key_1,
                    native_key_2,
                    native_key_3,
                    q1,
                    scratch,
                    stack,
                    BM,
                    BS,
                    CONTIG_K,
                    H,
                    HK,
                    K,
                    KH,
                    KP,
                    KR,
                    PAGE,
                    Q1_STREAM,
                    RAW_K,
                    SPLITS,
                    TABLE,
                    TS,
                )
                sync_block_set("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
        for drain in range(tl.minimum(stacks, 3)):
            sync_block_wait("vector", "cube", 9 + drain, PIPE.PIPE_MTE3, PIPE.PIPE_S)
    with tle.scope(core_mode="vector"):
        qstart = tl.load(CUQ + batch)
        qlen = 1
        qbase = qstart.to(tl.int64) * QR
        obase = qstart.to(tl.int64) * OR
        output_qstart = qstart
        if SPLITS > 1:
            obase += split.to(tl.int64) * TOTAL_Q * H * 256
        klen = tl.load(USED + batch)
        write_final = False
        head_rows = tl.minimum(
            group, tl.maximum(1, 128 // tl.maximum(qlen, 1) // 2 * 2)
        )
        head_tiles = tl.cdiv(group, head_rows)
        qs = tl.maximum(1, tl.minimum(qlen, 128))
        subtiles = tl.cdiv(qs * head_rows, 128)
        sub = task % subtiles
        outer = task // subtiles
        qt = outer // (head_tiles * HK)
        headtile = outer % (head_tiles * HK)
        kh = headtile // head_tiles
        headstart = headtile % head_tiles * head_rows
        scratch = SCORES + worker64 * 128 * BS
        prob = PROBS + worker64 * 128 * BS
        pvwork = PVWORK + worker64 * 128 * 256
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
                + worker64 * 128 * 256
                + init_rows[:, None] * 256
                + tl.arange(0, 256)[None, :],
                tl.zeros((8, 256), tl.float32),
            )
        x0_state_init = tl.where(tl.arange(0, 5)[:, None] == 0, -float("inf"), 0.0)
        x0_state_init = tl.broadcast_to(x0_state_init, (5, BM // 2))
        tle.dsa.to_buffer(x0_state_init, bind_buffer=state_buf)
        stacks = tl.cdiv(tl.maximum(limit - start, 0), BS)
        for step in range(stacks + 2):
            if step >= 2:
                previous = step - 2
                prevslot = previous % 3
                begin = start + previous * BS
                stack = tl.minimum(BS, limit - begin)
                prob = PROBS + (worker64 * 3 + prevslot) * 128 * BS
                pvwork = PVWORK + (worker64 * 3 + prevslot) * 128 * 256
                sync_block_wait(
                    "cube", "vector", 6 + prevslot, PIPE.PIPE_FIX, PIPE.PIPE_S
                )
                x4_vf = sub_vec_id().to(tl.int32) * (BM // 2)
                _native_update_accumulator(
                    acc_buf,
                    prevslot,
                    pvwork,
                    state_buf,
                    worker64,
                    x4_vf,
                    ACCWORK,
                    ACC_UB_ROWS,
                    BM,
                    O_VM,
                )
                sync_block_set(
                    "vector", "cube", 9 + prevslot, PIPE.PIPE_MTE3, PIPE.PIPE_S
                )
            if step < stacks:
                slot = step % 3
                begin = start + step * BS
                stack = tl.minimum(BS, limit - begin)
                scratch = SCORES + (worker64 * 3 + slot) * 128 * BS
                prob = PROBS + (worker64 * 3 + slot) * 128 * BS
                sync_block_wait("cube", "vector", slot, PIPE.PIPE_FIX, PIPE.PIPE_S)
                x2_vf = sub_vec_id().to(tl.int32) * (BM // 2)
                _native_update_softmax(
                    begin,
                    group,
                    head_rows,
                    headstart,
                    klen,
                    prob,
                    qlen,
                    qs,
                    qt,
                    row_end_buf,
                    scratch,
                    slot,
                    stack,
                    state_buf,
                    sub,
                    x2_vf,
                    BM,
                    BS,
                    CAUSAL,
                    CONTIG_K,
                    SCALE,
                    SINGLE_HEAD_INDEX,
                    SOFTMAX_VM_256,
                    VM,
                )
                sync_block_set("vector", "cube", 3 + slot, PIPE.PIPE_MTE3, PIPE.PIPE_S)
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
                    + worker64 * 128 * 256
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
                    if OH == 256:
                        # Q1 rows are consecutive heads of one KV group.
                        if x5_vf == 0:
                            x5_group_values = extract_slice(
                                x5_result, (0, 0), (group, 256), (1, 1)
                            )
                            tl.store(
                                O
                                + obase
                                + (kh * group + tl.arange(0, group))[:, None] * OH
                                + tl.arange(0, 256)[None, :],
                                x5_group_values,
                            )
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
                                if OH != 256:
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


# ---- q1_coreloops ----


@triton.jit
def ascend_attention_q1_coreloops_staged(
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
    STAGED_Q1: tl.constexpr = False,
    STREAM_Q1: tl.constexpr = False,
    RAW_K=None,
    RAW_V=None,
    RAW_PAGE_PAIR: tl.constexpr = False,
    NATIVE_SPLIT_Q1: tl.constexpr = False,
    MIN_SPLIT_K: tl.constexpr = 32768,
    QUERY_SCALE: tl.constexpr = 1.0,
    RESIDUAL_WAVE: tl.constexpr = False,
    WAVE_PARTIAL=None,
    WAVE_LSE=None,
):
    tl.static_assert(
        VARLEN
        and STAGED_Q1
        and STREAM_Q1
        and RAW_PAGE_PAIR
        and TAIL_ONLY
        and (SPLITS == 1)
        and (not COMPACT_Q1)
        and (not BYPASS_SHORT)
        and (not NATIVE_SPLIT_Q1)
        and (not MERGE_SPLITS)
        and (not SKIP_SHORT_KV)
        and (TASK_M == 16)
        and (H // HK == 4 or H // HK == 8)
    )
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
    residual_enabled = False
    prefix_count = 0
    decode_count = 0
    last_full_batch = -1
    if RESIDUAL_WAVE:
        tl.static_assert(H // HK == 8)
        decode_count = tl.load(REQUEST_COUNT)
        prefix_count = (decode_count * HK // workers * workers) // HK
        residual_count = decode_count - prefix_count
        residual_enabled = (
            (workers % HK == 0)
            & (residual_count > 0)
            & (residual_count * HK <= workers // 4)
        )
        if prefix_count > 0:
            last_full_batch = tl.load(REQUEST_IDS + prefix_count - 1)
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
            active_splits, _ = effective_kv_splits(klen, SPLITS, BS, MIN_SPLIT_K)
            ntiles *= active_splits
            tail_rows = qlen % 128
            eligible_tail = (qlen > 128) & (tail_rows > 0) & (tail_rows <= 16)
            original_offset = 0
            if TAIL_ONLY:
                ntiles = tl.where(qlen == 1, ntiles, tl.where(eligible_tail, H, 0))
                original_offset = tl.where(qlen == 1, 0, (tl.cdiv(qlen, 128) - 1) * H)
            else:
                ntiles -= tl.where(eligible_tail, H, 0)
            if RESIDUAL_WAVE:
                is_residual = residual_enabled & (qlen == 1) & (batch > last_full_batch)
                ntiles = tl.where(is_residual, 0, ntiles)
            first = (worker + workers - cursor) % workers
            for task in range(first, ntiles, workers):
                q_offset = tl.full((), 0, tl.int64)
                kv_offset = tl.full((), 0, tl.int64)
                if STAGED_Q1:
                    if qlen == 1:
                        qstart = tl.load(CUQ + batch)
                        stage_head = task
                        if NATIVE_SPLIT_Q1:
                            stage_head = task // active_splits
                        if RAW_PAGE_PAIR:
                            _stage_q1_native_tail(
                                Q_RAW,
                                RAW_K,
                                RAW_V,
                                TABLE,
                                Q,
                                K,
                                V,
                                batch,
                                stage_head,
                                worker,
                                qstart,
                                klen,
                                H,
                                HK,
                                QR,
                                QH,
                                KP,
                                KR,
                                KH,
                                VP,
                                VR,
                                VH,
                                TS,
                                PAGE,
                                CONTIG_K,
                                B,
                                QCAP,
                                STREAM_Q1,
                                Q_SCALE=QUERY_SCALE,
                            )
                        else:
                            _stage_q1_request(
                                Q_RAW,
                                RAW_K,
                                RAW_V,
                                TABLE,
                                Q,
                                K,
                                V,
                                batch,
                                stage_head,
                                worker,
                                qstart,
                                klen,
                                H,
                                HK,
                                QR,
                                QH,
                                KP,
                                KR,
                                KH,
                                VP,
                                VR,
                                VH,
                                TS,
                                PAGE,
                                CONTIG_K,
                                B,
                                QCAP,
                                STREAM_Q1,
                            )
                        q_offset = (
                            (
                                _packed_q_stage_base(B, QCAP)
                                + worker.to(tl.int64)
                                - _packed_q_base(batch, qstart, QCAP, H)
                                - stage_head
                            )
                            * 128
                            * 256
                        )
                        kv_offset = (
                            (
                                B * HK
                                + worker.to(tl.int64)
                                - batch.to(tl.int64) * HK
                                - stage_head
                            )
                            * CONTIG_K
                            * 256
                        )
                if RAW_PAGE_PAIR:
                    if qlen == 1:
                        _attention_task_q1_coreloops(
                            Q + q_offset,
                            Q_RAW,
                            K + kv_offset,
                            V + kv_offset,
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
                            Q1_STREAM=STREAM_Q1,
                            RAW_K=RAW_K,
                            RAW_V=RAW_V,
                            NATIVE_SPLIT_Q1=NATIVE_SPLIT_Q1,
                            MIN_SPLIT_K=MIN_SPLIT_K,
                        )
                    else:
                        _attention_task_native_pair(
                            Q + q_offset,
                            Q_RAW,
                            K + kv_offset,
                            V + kv_offset,
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
                            Q1_STREAM=STREAM_Q1,
                            RAW_K=RAW_K,
                            RAW_V=RAW_V,
                            NATIVE_SPLIT_Q1=NATIVE_SPLIT_Q1,
                            MIN_SPLIT_K=MIN_SPLIT_K,
                        )
                else:
                    _attention_task(
                        Q + q_offset,
                        Q_RAW,
                        K + kv_offset,
                        V + kv_offset,
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
                        Q1_STREAM=STREAM_Q1,
                        RAW_K=RAW_K,
                        RAW_V=RAW_V,
                    )
            cursor = (cursor + ntiles) % workers
    if RESIDUAL_WAVE:
        if residual_enabled:
            for request in range(decode_count):
                if request >= prefix_count:
                    batch = tl.load(REQUEST_IDS + request)
                    qstart = tl.load(CUQ + batch)
                    klen = tl.load(USED + batch)
                    active_splits, _ = effective_kv_splits(klen, 4, BS, 0)
                    ntiles = HK * active_splits
                    first = (worker + workers - cursor) % workers
                    for task in range(first, ntiles, workers):
                        stage_head = task // active_splits
                        _stage_q1_native_tail(
                            Q_RAW,
                            RAW_K,
                            RAW_V,
                            TABLE,
                            Q,
                            K,
                            V,
                            batch,
                            stage_head,
                            worker,
                            qstart,
                            klen,
                            H,
                            HK,
                            QR,
                            QH,
                            KP,
                            KR,
                            KH,
                            VP,
                            VR,
                            VH,
                            TS,
                            PAGE,
                            CONTIG_K,
                            B,
                            QCAP,
                            STREAM_Q1,
                            Q_SCALE=QUERY_SCALE,
                        )
                        q_offset = (
                            (
                                _packed_q_stage_base(B, QCAP)
                                + worker.to(tl.int64)
                                - _packed_q_base(batch, qstart, QCAP, H)
                                - stage_head
                            )
                            * 128
                            * 256
                        )
                        kv_offset = (
                            (
                                B * HK
                                + worker.to(tl.int64)
                                - batch.to(tl.int64) * HK
                                - stage_head
                            )
                            * CONTIG_K
                            * 256
                        )
                        partial_o = WAVE_PARTIAL
                        partial_lse = WAVE_LSE
                        _attention_task_q1_coreloops(
                            Q + q_offset,
                            Q_RAW,
                            K + kv_offset,
                            V + kv_offset,
                            partial_o,
                            partial_lse,
                            CUQ,
                            USED,
                            TABLE,
                            SCORES,
                            PROBS,
                            PVWORK,
                            ACCWORK,
                            batch,
                            stage_head,
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
                            H * 256,
                            256,
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
                            4,
                            CONTIG_K,
                            QCAP,
                            ACC_UB_ROWS,
                            Q_DIRECT,
                            SINGLE_HEAD_INDEX,
                            True,
                            FINAL_O,
                            FINAL_LSE,
                            FINAL_OR,
                            FINAL_OH,
                            FINAL_TOTAL_Q,
                            True,
                            Q1_STREAM=STREAM_Q1,
                            RAW_K=RAW_K,
                            RAW_V=RAW_V,
                            NATIVE_SPLIT_Q1=True,
                            MIN_SPLIT_K=0,
                        )
                    cursor = (cursor + ntiles) % workers


# ---- q1_coreloops ----


@triton.jit
def ascend_attention_q1_native_coreloops_staged(
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
    STAGED_Q1: tl.constexpr = False,
    STREAM_Q1: tl.constexpr = False,
    RAW_K=None,
    RAW_V=None,
    RAW_PAGE_PAIR: tl.constexpr = False,
    NATIVE_SPLIT_Q1: tl.constexpr = False,
    MIN_SPLIT_K: tl.constexpr = 32768,
    QUERY_SCALE: tl.constexpr = 1.0,
    RESIDUAL_WAVE: tl.constexpr = False,
    WAVE_PARTIAL=None,
    WAVE_LSE=None,
):
    tl.static_assert(
        VARLEN
        and STAGED_Q1
        and STREAM_Q1
        and RAW_PAGE_PAIR
        and TAIL_ONLY
        and SPLITS == 4
        and COMPACT_Q1
        and BYPASS_SHORT
        and NATIVE_SPLIT_Q1
        and not MERGE_SPLITS
        and not SKIP_SHORT_KV
        and TASK_M == 16
        and H // HK == 4
        and MIN_SPLIT_K == 2048
        and not RESIDUAL_WAVE
    )
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
    residual_enabled = False
    prefix_count = 0
    decode_count = 0
    last_full_batch = -1
    if RESIDUAL_WAVE:
        tl.static_assert(H // HK == 8)
        decode_count = tl.load(REQUEST_COUNT)
        prefix_count = (decode_count * HK // workers * workers) // HK
        residual_count = decode_count - prefix_count
        residual_enabled = (
            (workers % HK == 0)
            & (residual_count > 0)
            & (residual_count * HK <= workers // 4)
        )
        if prefix_count > 0:
            last_full_batch = tl.load(REQUEST_IDS + prefix_count - 1)
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
            active_splits, _ = effective_kv_splits(klen, SPLITS, BS, MIN_SPLIT_K)
            ntiles *= active_splits
            tail_rows = qlen % 128
            eligible_tail = (qlen > 128) & (tail_rows > 0) & (tail_rows <= 16)
            original_offset = 0
            if TAIL_ONLY:
                ntiles = tl.where(qlen == 1, ntiles, tl.where(eligible_tail, H, 0))
                original_offset = tl.where(qlen == 1, 0, (tl.cdiv(qlen, 128) - 1) * H)
            else:
                ntiles -= tl.where(eligible_tail, H, 0)
            if RESIDUAL_WAVE:
                is_residual = residual_enabled & (qlen == 1) & (batch > last_full_batch)
                ntiles = tl.where(is_residual, 0, ntiles)
            first = (worker + workers - cursor) % workers
            for task in range(first, ntiles, workers):
                q_offset = tl.full((), 0, tl.int64)
                kv_offset = tl.full((), 0, tl.int64)
                if STAGED_Q1:
                    if qlen == 1:
                        qstart = tl.load(CUQ + batch)
                        stage_head = task
                        if NATIVE_SPLIT_Q1:
                            stage_head = task // active_splits
                        if RAW_PAGE_PAIR:
                            _stage_q1_native_tail(
                                Q_RAW,
                                RAW_K,
                                RAW_V,
                                TABLE,
                                Q,
                                K,
                                V,
                                batch,
                                stage_head,
                                worker,
                                qstart,
                                klen,
                                H,
                                HK,
                                QR,
                                QH,
                                KP,
                                KR,
                                KH,
                                VP,
                                VR,
                                VH,
                                TS,
                                PAGE,
                                CONTIG_K,
                                B,
                                QCAP,
                                STREAM_Q1,
                                Q_SCALE=QUERY_SCALE,
                            )
                        else:
                            _stage_q1_request(
                                Q_RAW,
                                RAW_K,
                                RAW_V,
                                TABLE,
                                Q,
                                K,
                                V,
                                batch,
                                stage_head,
                                worker,
                                qstart,
                                klen,
                                H,
                                HK,
                                QR,
                                QH,
                                KP,
                                KR,
                                KH,
                                VP,
                                VR,
                                VH,
                                TS,
                                PAGE,
                                CONTIG_K,
                                B,
                                QCAP,
                                STREAM_Q1,
                            )
                        q_offset = (
                            (
                                _packed_q_stage_base(B, QCAP)
                                + worker.to(tl.int64)
                                - _packed_q_base(batch, qstart, QCAP, H)
                                - stage_head
                            )
                            * 128
                            * 256
                        )
                        kv_offset = (
                            (
                                B * HK
                                + worker.to(tl.int64)
                                - batch.to(tl.int64) * HK
                                - stage_head
                            )
                            * CONTIG_K
                            * 256
                        )
                if RAW_PAGE_PAIR:
                    if qlen == 1 and klen >= MIN_SPLIT_K:
                        _attention_task_q1_coreloops(
                            Q + q_offset,
                            Q_RAW,
                            K + kv_offset,
                            V + kv_offset,
                            O
                            + (batch.to(tl.int64) - tl.load(CUQ + batch).to(tl.int64))
                            * OR,
                            LSE + batch - tl.load(CUQ + batch),
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
                            Q1_STREAM=STREAM_Q1,
                            RAW_K=RAW_K,
                            RAW_V=RAW_V,
                            NATIVE_SPLIT_Q1=NATIVE_SPLIT_Q1,
                            MIN_SPLIT_K=MIN_SPLIT_K,
                        )
                    else:
                        _attention_task_native_pair(
                            Q + q_offset,
                            Q_RAW,
                            K + kv_offset,
                            V + kv_offset,
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
                            Q1_STREAM=STREAM_Q1,
                            RAW_K=RAW_K,
                            RAW_V=RAW_V,
                            NATIVE_SPLIT_Q1=NATIVE_SPLIT_Q1,
                            MIN_SPLIT_K=MIN_SPLIT_K,
                        )
                else:
                    _attention_task(
                        Q + q_offset,
                        Q_RAW,
                        K + kv_offset,
                        V + kv_offset,
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
                        Q1_STREAM=STREAM_Q1,
                        RAW_K=RAW_K,
                        RAW_V=RAW_V,
                    )
            cursor = (cursor + ntiles) % workers
    if RESIDUAL_WAVE:
        if residual_enabled:
            for request in range(decode_count):
                if request >= prefix_count:
                    batch = tl.load(REQUEST_IDS + request)
                    qstart = tl.load(CUQ + batch)
                    klen = tl.load(USED + batch)
                    active_splits, _ = effective_kv_splits(klen, 4, BS, 0)
                    ntiles = HK * active_splits
                    first = (worker + workers - cursor) % workers
                    for task in range(first, ntiles, workers):
                        stage_head = task // active_splits
                        _stage_q1_native_tail(
                            Q_RAW,
                            RAW_K,
                            RAW_V,
                            TABLE,
                            Q,
                            K,
                            V,
                            batch,
                            stage_head,
                            worker,
                            qstart,
                            klen,
                            H,
                            HK,
                            QR,
                            QH,
                            KP,
                            KR,
                            KH,
                            VP,
                            VR,
                            VH,
                            TS,
                            PAGE,
                            CONTIG_K,
                            B,
                            QCAP,
                            STREAM_Q1,
                            Q_SCALE=QUERY_SCALE,
                        )
                        q_offset = (
                            (
                                _packed_q_stage_base(B, QCAP)
                                + worker.to(tl.int64)
                                - _packed_q_base(batch, qstart, QCAP, H)
                                - stage_head
                            )
                            * 128
                            * 256
                        )
                        kv_offset = (
                            (
                                B * HK
                                + worker.to(tl.int64)
                                - batch.to(tl.int64) * HK
                                - stage_head
                            )
                            * CONTIG_K
                            * 256
                        )
                        partial_o = WAVE_PARTIAL
                        partial_lse = WAVE_LSE
                        _attention_task_q1_coreloops(
                            Q + q_offset,
                            Q_RAW,
                            K + kv_offset,
                            V + kv_offset,
                            partial_o,
                            partial_lse,
                            CUQ,
                            USED,
                            TABLE,
                            SCORES,
                            PROBS,
                            PVWORK,
                            ACCWORK,
                            batch,
                            stage_head,
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
                            H * 256,
                            256,
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
                            4,
                            CONTIG_K,
                            QCAP,
                            ACC_UB_ROWS,
                            Q_DIRECT,
                            SINGLE_HEAD_INDEX,
                            True,
                            FINAL_O,
                            FINAL_LSE,
                            FINAL_OR,
                            FINAL_OH,
                            FINAL_TOTAL_Q,
                            True,
                            Q1_STREAM=STREAM_Q1,
                            RAW_K=RAW_K,
                            RAW_V=RAW_V,
                            NATIVE_SPLIT_Q1=True,
                            MIN_SPLIT_K=0,
                        )
                    cursor = (cursor + ntiles) % workers


# ---- Direct paged Q1 Decode ----


@triton.jit
def _merge_one_q1(
    O,
    PARTIAL,
    LSE,
    b,
    head,
    n,
    H: tl.constexpr,
    B: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    row = (b * H + head).to(tl.int64)
    d = tl.arange(0, 256)
    active, _ = effective_kv_splits(n, SPLITS, 512)
    if active == 1:
        out = tl.load(PARTIAL + row * 256 + d, volatile=True)
    else:
        split = tl.arange(0, BLOCK_SPLITS)
        ls = tl.load(
            LSE + (split * H + head) * B + b,
            split < active,
            -float("inf"),
            volatile=True,
        )
        max_lse = tl.max(ls, 0)
        valid = max_lse > -float("inf")
        weight = tl.exp(ls - tl.where(valid, max_lse, 0.0))
        partial = tl.load(
            PARTIAL + (split[:, None] * B * H + row) * 256 + d[None, :],
            split[:, None] < active,
            0.0,
            volatile=True,
        )
        denom = tl.where(valid, tl.sum(weight, 0), 1.0)
        out = tl.sum(partial * (weight / denom)[:, None], 0)
    tl.store(O + row * 256 + d, out)


@triton.jit(
    do_not_specialize_on_alignment=[
        "Q",
        "K",
        "V",
        "O",
        "USED",
        "TABLE",
        "WORK",
        "PROBS",
    ]
)
def _raw_q1_one(
    Q,
    K,
    V,
    O,
    USED,
    TABLE,
    WORK,
    PROBS,
    H: tl.constexpr,
    HK: tl.constexpr,
    B: tl.constexpr,
    PAGE: tl.constexpr,
    TS: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
):
    W = tl.num_programs(0)
    worker = tl.program_id(0)
    G: tl.constexpr = H // HK
    qs: tl.constexpr = H * 256
    kp: tl.constexpr = PAGE * HK * 256
    kr: tl.constexpr = HK * 256
    scores = WORK
    pv = WORK + W * 16 * 512
    partial = pv + W * 16 * 256
    lse = partial + SPLITS * B * H * 256
    for item in range(worker, B * HK * SPLITS, W):
        task, split = item // SPLITS, item % SPLITS
        b, kh = task // HK, task % HK
        n = tl.load(USED + b)
        if n <= 1:
            with tle.scope(core_mode="vector"):
                if split == 0 and sub_vec_id() == 0:
                    d = tl.arange(0, 256)
                    g = tl.arange(0, G)
                    physical = tl.load(TABLE + b.to(tl.int64) * TS, n == 1, 0).to(
                        tl.int64
                    )
                    val = tl.load(V + physical * kp + kh * 256 + d, n == 1, 0)
                    dst = O + b.to(tl.int64) * qs + (kh * G + g[:, None]) * 256
                    tl.store(dst + d[None, :], val[None, :])
        else:
            active, _ = effective_kv_splits(n, SPLITS, 512)
            if split < active:
                _attention_task_direct(
                    Q,
                    K,
                    V,
                    partial,
                    lse,
                    None,
                    USED,
                    TABLE,
                    scores,
                    PROBS,
                    pv,
                    b,
                    kh,
                    split,
                    worker,
                    H,
                    HK,
                    B,
                    1,
                    SCALE,
                    qs,
                    256,
                    qs,
                    qs,
                    256,
                    qs,
                    kp,
                    kr,
                    256,
                    kp,
                    kr,
                    256,
                    TS,
                    PAGE,
                    False,
                    False,
                    16,
                    32,
                    512,
                    SPLITS,
                )
    # A persistent grid occupies at most one mixed program per Cube.
    # All workers reach the same global barrier, including when every K is 0/1.
    tle.dsa.ascend.sync_block_all("all", 0)
    with tle.scope(core_mode="vector"):
        lane = worker * 2 + sub_vec_id()
        for output_row in range(lane, B * H, W * 2):
            batch, head = output_row // H, output_row % H
            n = tl.load(USED + batch)
            if n > 1:
                _merge_one_q1(
                    O,
                    partial,
                    lse,
                    batch,
                    head,
                    n,
                    H,
                    B,
                    SPLITS,
                    triton.next_power_of_2(SPLITS),
                )


_launchers = {}
_stream_scratch = {}
_scratch_lock = Lock()
_MAX_SCRATCH_BYTES = 48 * 1024 * 1024


def _q1_scratch(q, work_size, prob_size, stream):
    dtype, device = q.dtype, q.device
    key = None
    if stream is not None and not torch.npu.is_current_stream_capturing():
        # One buffer per stream is sufficient: the single persistent launch
        # consumes both regions before the next launch on this stream.
        key = (device.index, stream)
        workspace = _stream_scratch.get(key)
        if workspace is not None and (
            workspace[0].numel() >= work_size and workspace[1].numel() >= prob_size
        ):
            return workspace
    workspace = (
        torch.empty((work_size,), dtype=torch.float32, device=device),
        torch.empty((prob_size,), dtype=dtype, device=device),
    )
    if key is not None:
        with _scratch_lock:
            retained = sum(
                4 * w.numel() + 2 * p.numel()
                for k, (w, p) in _stream_scratch.items()
                if k != key
            )
            if retained + 4 * work_size + 2 * prob_size <= _MAX_SCRATCH_BYTES:
                _stream_scratch[key] = workspace
    return workspace


def raw_q1_output(q, k, v, cu, used, table, max_k, scale, device):
    """Fuse singleton output with sparse work on the raw-paged Decode path."""
    if (
        q.ndim != 3
        or k.ndim != 4
        or k.shape != v.shape
        or q.dtype != torch.bfloat16
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or q.shape[2] != 256
        or k.shape[3] != 256
        or not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous())
        or cu.dtype != torch.int32
        or cu.ndim != 1
        or not cu.is_contiguous()
        or used.ndim != 1
        or not used.is_contiguous()
        or table.ndim != 2
        or not 1 <= q.size(0) <= 1024
        or not 32768 <= max_k <= table.size(1) * k.size(1)
    ):
        return None
    b, h, _ = q.shape
    pages, page, hk, _ = k.shape
    if (
        not hk
        or h % hk
        or h // hk not in (4, 8)
        or page not in (16, 32)
        or b != table.size(0)
        or b != used.numel()
        or cu.numel() != b + 1
        # Keep the pre-existing small-cache staged and Q-direct paths.
        or (h == 8 and hk == 1 and 32 <= b <= 128 and pages * page <= b * 4096)
        or (
            page == 32
            and h == 16
            and hk == 2
            and b >= 512
            and b % 4 == 0
            and pages * page <= b * 2048
        )
    ):
        return None
    if scale is None:
        scale = 0.0625
    elif type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        return None
    index = device.index
    w = cube_core_limit(index)
    # Bound the partial workspace while allowing large batches to remain on
    # the direct path. Keep at least two splits for sparse long requests.
    if b * h > 8192:
        return None
    tasks = (b * h + 15) // 16
    splits = max(2, min(16, w // tasks)) if tasks <= w else 4
    out = torch.empty_like(q)
    partials = splits * h * b * 256
    active = torch.npu.current_device() == index
    stream = None
    if active:
        current_stream = _raw_stream_nowait or driver.active.get_current_stream
        stream = current_stream(index)
    work, probs = _q1_scratch(
        q, w * 16 * (512 + 256) + splits * h * b + partials, w * 16 * 512, stream
    )
    constexprs = (h, hk, b, page, table.stride(0), scale, splits)
    args = (q, k, v, out, used, table, work, probs, *constexprs)
    key = (index, q.dtype, w, constexprs)
    compiled = _launchers.get(key)
    if compiled is not None and active:
        compiled(args)
    else:
        grid = (w, 1, 1)
        with torch.npu.device(device):
            entry = _raw_q1_one[grid](*args, enable_fp_fusion=False, multibuffer=False)
        if len(_launchers) < 256 and hasattr(entry, "__getitem__"):
            _launchers[key] = bind_runner(entry, grid, index)
    return out


# ---- Singleton paged Decode ----


@triton.jit(do_not_specialize_on_alignment=["V", "O", "CUQ", "USED", "TABLE"])
def _singleton_paged_output(
    V,
    O,
    CUQ,
    USED,
    TABLE,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    PAGE: tl.constexpr,
    TS: tl.constexpr,
    GROUP: tl.constexpr,
    BG: tl.constexpr,
    BATCH_TILE: tl.constexpr,
    DENSE_Q: tl.constexpr,
):
    kh, first = tl.program_id(0), tl.program_id(1) * BATCH_TILE
    d = tl.arange(0, 256)
    g = tl.arange(0, BG)
    for offset in range(BATCH_TILE):
        batch = first + offset
        if B % BATCH_TILE == 0 or batch < B:
            if DENSE_Q:
                # With Q <= 1 and as many query rows as requests, every
                # request owns exactly one row. No device metadata read.
                row, has_query = batch, True
            else:
                row = tl.load(CUQ + batch)
                has_query = tl.load(CUQ + batch + 1) > row
            if has_query:
                has_kv = tl.load(USED + batch) > 0
                page = tl.load(TABLE + batch.to(tl.int64) * TS, has_kv, 0).to(tl.int64)
                val = tl.load(V + (page * PAGE * HK + kh) * 256 + d, has_kv, 0)
                dst = O + (row.to(tl.int64) * H + kh * GROUP + g[:, None]) * 256
                tl.store(dst + d[None, :], val[None, :], g[:, None] < GROUP)


_singleton_launchers = {}


def singleton_paged_output(q, k, v, cuq, used, table, device):
    """Fast output-only path; caller guarantees the maximum KV length is one."""
    q_shape, kv_shape, table_shape = q.shape, k.shape, table.shape
    dtype = q.dtype
    if (
        len(q_shape) != 3
        or len(kv_shape) != 4
        or q_shape[-1] != 256
        or kv_shape[-1] != 256
        or kv_shape != v.shape
        or kv_shape[1] not in (16, 32)
        or dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != dtype
        or v.dtype != dtype
        or not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous())
        or cuq.ndim != 1
        or cuq.dtype != torch.int32
        or not cuq.is_contiguous()
        or not used.is_contiguous()
        or len(table_shape) != 2
        or table_shape[0] != cuq.numel() - 1
        or table_shape[1] < 1
        or used.shape != (table_shape[0],)
    ):
        return None
    total, h, _ = q_shape
    hk, b = kv_shape[2], table_shape[0]
    if not hk or not h or h % hk or h // hk > 32 or not b or total > b:
        return None
    out = torch.empty_like(q)
    if not total:
        return out
    # Bound the number of program waves for large batches; keep each
    # physical-page read contiguous within the Vector program.
    tasks = b * hk
    batch_tile = 8 if tasks > 512 else 4 if tasks > 192 else 1
    group = h // hk
    configs = (
        b,
        h,
        hk,
        kv_shape[1],
        table.stride(0),
        group,
        1 << (group - 1).bit_length(),
        batch_tile,
        total == b,
    )
    index = device.index
    key = (index, dtype, configs)
    compiled = _singleton_launchers.get(key)
    inputs = (v, out, None if total == b else cuq, used, table, *configs)
    # Only the compiled code/grid is cached. Metadata and output pointers are
    # supplied each call, and the runner resolves the current stream.
    if compiled is not None and torch.npu.current_device() == index:
        compiled(inputs)
    else:
        grid = (hk, triton.cdiv(b, batch_tile), 1)
        with torch.npu.device(device):
            entry = _singleton_paged_output[grid](*inputs)
        if len(_singleton_launchers) < 256 and hasattr(entry, "__getitem__"):
            _singleton_launchers[key] = bind_runner(entry, grid, index)
    return out
