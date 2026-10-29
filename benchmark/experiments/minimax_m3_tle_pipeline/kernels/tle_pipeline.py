# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Isolated BF16 union attention with a real TMA producer/consumer ring.

No serving or compiler source is changed. See CONTRACT.md for numerical and
slot-lifetime requirements. This module remains experimental until receipts pass.
"""

import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from tle_nows import _dispatch_nows


@triton.jit
def _bitmap_or(a, b):
    return a | b


@triton.jit
def _producer(
    desc,
    bt_row,
    stride_bt_k,
    raw_k,
    raw_v,
    metadata,
    empty,
    k_full,
    v_full,
    meta_full,
    lo_s,
    span,
    active0,
    active1,
    active2,
    active3,
    STAGES: tl.constexpr,
    SPARSE: tl.constexpr,
):
    iteration = 0
    g = 0
    while g < span:
        bits = tl.where(g == 0, active0, active1)
        bits = tl.where(g == 64, active2, bits)
        bits = tl.where(g == 96, active3, bits)
        if SPARSE:
            visits = tl.extra.cuda.libdevice.popc(bits)
        else:
            visits = tl.minimum(32, span - g)
        for visit in range(visits):
            if SPARSE:
                offset = tl.extra.cuda.libdevice.ffs(bits) - 1
                bits = bits & (bits - 1)
            else:
                offset = visit
            blk = lo_s + g + offset
            slot = iteration % STAGES
            phase = iteration // STAGES
            tle.gpu.barrier_wait(empty[slot], phaseIdx=phase)
            page = tl.load(bt_row + blk * stride_bt_k)
            # Metadata is covered by the same empty/release lifetime as K/V.
            # Its full barrier is a release/acquire edge independent of TMA.
            tl.store(
                tle.gpu.local_ptr(metadata.slot(slot)), tl.full((1, 1), blk, tl.int32)
            )
            tle.gpu.barrier_arrive(meta_full[slot], phaseIdx=phase)
            tle.gpu.copy(
                desc,
                raw_k.slot(slot),
                [128, 128],
                [page * 128, 0],
                barrier=k_full[slot],
            )
            tle.gpu.copy(
                desc,
                raw_v.slot(slot),
                [128, 128],
                [page * 128, 128],
                barrier=v_full[slot],
            )
            iteration += 1
        g += 32
    # Drain every populated slot, including workloads shorter than the ring.
    # This protects shared-memory lifetime at the WS join.
    for i in range(tl.maximum(0, iteration - STAGES), iteration):
        tle.gpu.barrier_wait(empty[i % STAGES], phaseIdx=i // STAGES + 1)


@triton.jit
def _producer_convert(
    desc,
    bt_row,
    stride_bt_k,
    raw_k,
    raw_v,
    metadata,
    empty,
    k_full,
    v_full,
    meta_full,
    bf16_k,
    bf16_v,
    lo_s,
    span,
    active0,
    active1,
    active2,
    active3,
    STAGES: tl.constexpr,
    SPARSE: tl.constexpr,
    seq_len,
):
    iteration = 0
    g = 0
    while g < span:
        bits = tl.where(g == 0, active0, active1)
        bits = tl.where(g == 64, active2, bits)
        bits = tl.where(g == 96, active3, bits)
        if SPARSE:
            visits = tl.extra.cuda.libdevice.popc(bits)
        else:
            visits = tl.minimum(32, span - g)
        for visit in range(visits):
            if SPARSE:
                offset = tl.extra.cuda.libdevice.ffs(bits) - 1
                bits = bits & (bits - 1)
            else:
                offset = visit
            blk = lo_s + g + offset
            slot = iteration % STAGES
            phase = iteration // STAGES
            tle.gpu.barrier_wait(empty[slot], phaseIdx=phase)
            page = tl.load(bt_row + blk * stride_bt_k)
            # Metadata is covered by the same empty/release lifetime as K/V.
            # Its full barrier is a release/acquire edge independent of TMA.
            tl.store(
                tle.gpu.local_ptr(metadata.slot(slot)), tl.full((1, 1), blk, tl.int32)
            )
            tle.gpu.copy(
                desc, raw_k.slot(0), [128, 128], [page * 128, 0], barrier=k_full[0]
            )
            tle.gpu.copy(
                desc, raw_v.slot(0), [128, 128], [page * 128, 128], barrier=v_full[0]
            )
            tle.gpu.barrier_wait(k_full[0], phaseIdx=iteration)
            tle.gpu.barrier_wait(v_full[0], phaseIdx=iteration)
            valid = blk * 128 + tl.arange(0, 128) < seq_len
            k = tl.where(
                valid[:, None], tl.load(tle.gpu.local_ptr(raw_k.slot(0))), 0.0
            ).to(tl.bfloat16)
            v = tl.where(
                valid[:, None], tl.load(tle.gpu.local_ptr(raw_v.slot(0))), 0.0
            ).to(tl.bfloat16)
            tl.store(tle.gpu.local_ptr(bf16_k.slot(slot)), k)
            tl.store(tle.gpu.local_ptr(bf16_v.slot(slot)), v)
            # All conversion warps finish their stores before publishing the slot.
            tl.debug_barrier()
            tle.gpu.barrier_arrive(meta_full[slot], phaseIdx=phase)
            iteration += 1
        g += 32
    # Drain every populated slot, including workloads shorter than the ring.
    # This protects shared-memory lifetime at the WS join.
    for i in range(tl.maximum(0, iteration - STAGES), iteration):
        tle.gpu.barrier_wait(empty[i % STAGES], phaseIdx=i // STAGES + 1)


@triton.jit
def _consumer(
    q,
    raw_k,
    raw_v,
    metadata,
    scratch,
    empty,
    k_full,
    v_full,
    meta_full,
    bm0,
    bm1,
    bm2,
    bm3,
    lo_s,
    visits,
    q_abs,
    seq_len,
    qk_scale,
    o_ptr,
    row_off,
    row_ok,
    stride_qn,
    v_scale_ptr,
    M: tl.constexpr,
    BQ: tl.constexpr,
    STAGES: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    PRODUCER_CONVERT: tl.constexpr,
):
    off_n = tl.arange(0, 128)
    off_d = tl.arange(0, 128)
    m_i = tl.full((M,), float("-inf"), tl.float32)
    l_i = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, 128), tl.float32)
    for iteration in range(visits):
        slot = iteration % STAGES
        phase = iteration // STAGES
        tle.gpu.barrier_wait(meta_full[slot], phaseIdx=phase)
        blk = tl.sum(tl.sum(tl.load(tle.gpu.local_ptr(metadata.slot(slot))), 0), 0)
        offset = blk - lo_s
        bm = tl.where(offset < 32, bm0, bm1)
        bm = tl.where((offset >= 64) & (offset < 96), bm2, bm)
        bm = tl.where(offset >= 96, bm3, bm)
        sel = ((bm >> (offset & 31)) & 1) != 0
        sel = tl.reshape(tl.broadcast_to(sel[:, None], (BQ, 8)), (M,))
        pos = blk * 128 + off_n
        pos_ok = pos < seq_len
        if PRODUCER_CONVERT:
            qk_async = tle.gpu.wgmma(
                q, raw_k.slot(slot), trans_b=True, out_dtype=tl.float32
            )
        else:
            tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
            k = tl.load(tle.gpu.local_ptr(raw_k.slot(slot)))
            # Match baseline zero-fill before BF16 QK for sequence tails.
            k = tl.where(pos_ok[:, None], k, 0.0).to(tl.bfloat16)
            tl.store(tle.gpu.local_ptr(scratch.slot(0)), k)
            qk_async = tle.gpu.wgmma(
                q, scratch.slot(0), trans_b=True, out_dtype=tl.float32
            )
        qk = tle.gpu.wgmma_wait(0, qk_async) * qk_scale
        keep = sel[:, None] & (q_abs[:, None] >= pos[None, :]) & pos_ok[None, :]
        qk = tl.where(keep, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if PRODUCER_CONVERT:
            pv_async = tle.gpu.wgmma(
                p.to(tl.bfloat16),
                raw_v.slot(slot),
                acc=acc * alpha[:, None],
                out_dtype=tl.float32,
            )
        else:
            tle.gpu.barrier_wait(v_full[slot], phaseIdx=phase)
            v = tl.load(tle.gpu.local_ptr(raw_v.slot(slot)))
            v = tl.where(pos_ok[:, None], v, 0.0).to(tl.bfloat16)
            # QK has completed before its scratch bytes are reused for V.
            tl.store(tle.gpu.local_ptr(scratch.slot(0)), v)
            # Frozen Triton combines its scaled-acc + dot into WGMMA's C operand.
            pv_async = tle.gpu.wgmma(
                p.to(tl.bfloat16),
                scratch.slot(0),
                acc=acc * alpha[:, None],
                out_dtype=tl.float32,
            )
        acc = tle.gpu.wgmma_wait(0, pv_async)
        m_i = m_new
        # Both async reads have completed; only now may the producer reuse slot.
        tle.gpu.barrier_arrive(empty[slot], phaseIdx=phase)
    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    if HAS_SCALE:
        inv_l = inv_l * tl.load(v_scale_ptr)
    acc = acc * inv_l[:, None]
    tl.store(
        o_ptr + row_off[:, None] * stride_qn + off_d[None, :],
        acc.to(tl.bfloat16),
        mask=row_ok[:, None],
    )


@triton.jit(do_not_specialize_on_alignment=["t_ptr", "cu_ptr", "seq_ptr", "pre_ptr"])
def _union_tle_kernel(
    q_ptr,
    kv_ptr,
    kv_desc,
    t_ptr,
    o_ptr,
    bt_ptr,
    cu_ptr,
    seq_ptr,
    pre_ptr,
    k_scale_ptr,
    v_scale_ptr,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    heads_total,
    stride_qn,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_bt_b,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOPK: tl.constexpr,
    QH: tl.constexpr,
    SPAN: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    D_POW2: tl.constexpr,
    INT8_QK: tl.constexpr,
    auto_flags_ptr=None,
    AUTO_SPAN: tl.constexpr = 0,
    stride_bt_k=1,
    SPARSE_WALK: tl.constexpr = False,
    parent_flags_ptr=None,
    PARENT_BLOCK_Q: tl.constexpr = 0,
    PARENT_TILES: tl.constexpr = 0,
    PIPE_STAGES: tl.constexpr = 2,
    WARP_SPECIALIZE: tl.constexpr = True,
    PRODUCER_CONVERT: tl.constexpr = False,
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_ptr + pid_b)
    q_len = tl.load(cu_ptr + pid_b + 1) - q_start
    q0 = pid_q * BLOCK_Q
    if q0 >= q_len:
        return
    # Parent compact output already owns these rows; still overwrite every active wide flag.
    if PARENT_BLOCK_Q:
        parent_off = (
            pid_b * tl.num_programs(1) + pid_h
        ) * PARENT_TILES + q0 // PARENT_BLOCK_Q
        if tl.load(parent_flags_ptr + parent_off) != 0:
            wide_off = (pid_b * tl.num_programs(1) + pid_h) * tl.num_programs(0) + pid_q
            tl.store(auto_flags_ptr + wide_off, 1)
            return
    seq_len = tl.load(seq_ptr + pid_b)
    prefix_len = tl.load(pre_ptr + pid_b)

    M: tl.constexpr = BLOCK_Q * QH
    off_m = tl.arange(0, M)
    off_d = tl.arange(0, BLOCK_D)
    tok = off_m // QH
    n_tok = tl.minimum(BLOCK_Q, q_len - q0)
    row_ok = tok < n_tok
    d_ok = off_d < head_dim
    q_abs = prefix_len + q0 + tok

    qk_scale = sm_scale * 1.4426950409
    if HAS_SCALE:
        qk_scale = qk_scale * tl.load(k_scale_ptr)

    # `q` arrives as a flat [total_q * heads_total, head_dim] view, so one
    # token spans heads_total heads, of which this program owns the QH heads of
    # KV head ``pid_h``. Both the tile base and the in-tile row offset must
    # account for that: a row is (token, head-within-KV-head), so
    #   row_offset = tok * heads_total + off_m % QH
    # Using ``off_m * stride_qn`` (the old code) is only right while
    # heads_total == QH, i.e. num_kv_heads == 1; with more KV heads it walked
    # into the neighbouring KV heads and left ~47% of the output unwritten.
    row_off = tok * heads_total + (off_m % QH)
    qbase = q_ptr + ((q_start + q0) * heads_total + pid_h * QH) * stride_qn
    if D_POW2:
        q = tl.load(
            qbase + row_off[:, None] * stride_qn + off_d[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
    else:
        q = tl.load(
            qbase + row_off[:, None] * stride_qn + off_d[None, :],
            mask=row_ok[:, None] & d_ok[None, :],
            other=0.0,
        )

    if INT8_QK:
        qi, qs = _quantize_int8_rows(q)  # noqa: F821 - archived, untested INT8 branch
    # Selection metadata has one row per query token, shared by the eight GQA heads.
    # Keep it at BLOCK_Q, then expand the chosen bitmap for the QK tile.
    topk_tok = tl.arange(0, BLOCK_Q)
    topk_row_ok = topk_tok < n_tok
    tbase = t_ptr + pid_h * stride_th + (q_start + q0 + topk_tok) * stride_tn
    big = 2147483647
    lo = tl.full((BLOCK_Q,), big, tl.int32)
    hi = tl.full((BLOCK_Q,), -1, tl.int32)
    if AUTO_SPAN:
        n_selected = tl.zeros((BLOCK_Q,), tl.int32)
    for j in tl.static_range(TOPK):
        t = tl.load(tbase + j * stride_tk, mask=topk_row_ok, other=-1)
        lo = tl.minimum(lo, tl.where(t >= 0, t, big))
        hi = tl.maximum(hi, t)
        if AUTO_SPAN:
            n_selected += ((t >= 0) & topk_row_ok).to(tl.int32)
    lo_s = tl.min(lo, axis=0)
    hi_s = tl.max(hi, axis=0)
    span = hi_s - lo_s + 1
    auto_bm_hi = tl.zeros((BLOCK_Q,), tl.int32)
    auto_bm_hi2 = tl.zeros((BLOCK_Q,), tl.int32)
    auto_bm_hi3 = tl.zeros((BLOCK_Q,), tl.int32)
    if AUTO_SPAN:
        # Recomputed from device indices on every invocation / graph replay.
        # No host readback and no assumption that indexer outputs stay local.
        # Sparse lists with few valid entries need a proportionally tighter
        # span cap. A full TOPK=16 list permits 32 / 64 blocks respectively.
        density = max(1, AUTO_SPAN // TOPK)
        selected = tl.sum(n_selected, 0)
        compact = (selected == 0) | (
            (span <= AUTO_SPAN) & (span * n_tok <= density * selected)
        )
        flag_off = (pid_b * tl.num_programs(1) + pid_h) * tl.num_programs(0) + pid_q
        tl.store(auto_flags_ptr + flag_off, compact.to(tl.int32))
        if not compact:
            return
        auto_bm = tl.zeros((BLOCK_Q,), tl.int32)
        if AUTO_SPAN > 32:
            auto_bm_hi = tl.zeros((BLOCK_Q,), tl.int32)
        if AUTO_SPAN > 64:
            auto_bm_hi2 = tl.zeros((BLOCK_Q,), tl.int32)
            auto_bm_hi3 = tl.zeros((BLOCK_Q,), tl.int32)
        for j in tl.static_range(TOPK):
            t = tl.load(tbase + j * stride_tk, mask=topk_row_ok, other=-1)
            sh = (t - lo_s) & 31
            valid = topk_row_ok & (t >= 0)
            auto_bm |= tl.where(valid & (t < lo_s + 32), 1 << sh, 0)
            if AUTO_SPAN > 32:
                auto_bm_hi |= tl.where(
                    valid & (t >= lo_s + 32) & (t < lo_s + 64), 1 << sh, 0
                )
            if AUTO_SPAN > 64:
                auto_bm_hi2 |= tl.where(
                    valid & (t >= lo_s + 64) & (t < lo_s + 96), 1 << sh, 0
                )
                auto_bm_hi3 |= tl.where(valid & (t >= lo_s + 96), 1 << sh, 0)
        # Bitmap membership collapses repeated ids; preserve selected-list
        # semantics by sending duplicate-bearing tiles to fallback.
        pop = tl.extra.cuda.libdevice.popc(auto_bm)
        if AUTO_SPAN > 32:
            pop += tl.extra.cuda.libdevice.popc(auto_bm_hi)
        if AUTO_SPAN > 64:
            pop += tl.extra.cuda.libdevice.popc(auto_bm_hi2)
            pop += tl.extra.cuda.libdevice.popc(auto_bm_hi3)
        duplicates = pop != n_selected
        if tl.sum(duplicates.to(tl.int32), axis=0) != 0:
            tl.store(auto_flags_ptr + flag_off, 0)
            return

    # Metadata decisions match the frozen wrapper; only payload scheduling changes.
    active0 = tl.reduce(auto_bm, 0, _bitmap_or)
    active1 = tl.reduce(auto_bm_hi, 0, _bitmap_or)
    active2 = tl.reduce(auto_bm_hi2, 0, _bitmap_or)
    active3 = tl.reduce(auto_bm_hi3, 0, _bitmap_or)
    if SPARSE_WALK:
        visits = (
            tl.extra.cuda.libdevice.popc(active0)
            + tl.extra.cuda.libdevice.popc(active1)
            + tl.extra.cuda.libdevice.popc(active2)
            + tl.extra.cuda.libdevice.popc(active3)
        )
    else:
        visits = tl.maximum(0, span)
    if not WARP_SPECIALIZE:
        obase = o_ptr + ((q_start + q0) * heads_total + pid_h * QH) * stride_qn
        _dispatch_nows(
            q,
            kv_desc,
            bt_ptr + pid_b * stride_bt_b,
            stride_bt_k,
            auto_bm,
            auto_bm_hi,
            auto_bm_hi2,
            auto_bm_hi3,
            active0,
            active1,
            active2,
            active3,
            lo_s,
            span,
            visits,
            q_abs,
            seq_len,
            qk_scale,
            obase,
            row_off,
            row_ok,
            stride_qn,
            v_scale_ptr,
            M,
            BLOCK_Q,
            PIPE_STAGES,
            HAS_SCALE,
            SPARSE_WALK,
            kv_ptr.dtype.element_ty,
        )
    else:
        RAW_STAGES: tl.constexpr = 1 if PRODUCER_CONVERT else PIPE_STAGES
        raw_k = tle.gpu.alloc(
            [RAW_STAGES, 128, 128], dtype=kv_ptr.dtype.element_ty, scope=tle.gpu.smem
        )
        raw_v = tle.gpu.alloc(
            [RAW_STAGES, 128, 128], dtype=kv_ptr.dtype.element_ty, scope=tle.gpu.smem
        )
        metadata = tle.gpu.alloc(
            [PIPE_STAGES, 1, 1],
            dtype=tl.int32,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        scratch = tle.gpu.alloc([1, 128, 128], dtype=tl.bfloat16, scope=tle.gpu.smem)
        empty = tle.gpu.alloc_barriers(
            triton.next_power_of_2(PIPE_STAGES), init=tle.gpu.READY
        )
        meta_full = tle.gpu.alloc_barriers(triton.next_power_of_2(PIPE_STAGES))
        BYTES: tl.constexpr = (
            128 * 128 * (kv_ptr.dtype.element_ty.primitive_bitwidth // 8)
        )
        k_full = tle.gpu.alloc_barriers(
            triton.next_power_of_2(PIPE_STAGES), expect_bytes=BYTES
        )
        v_full = tle.gpu.alloc_barriers(
            triton.next_power_of_2(PIPE_STAGES), expect_bytes=BYTES
        )
        obase = o_ptr + ((q_start + q0) * heads_total + pid_h * QH) * stride_qn
        if PRODUCER_CONVERT:
            bf16_k = tle.gpu.alloc(
                [PIPE_STAGES, 128, 128], dtype=tl.bfloat16, scope=tle.gpu.smem
            )
            bf16_v = tle.gpu.alloc(
                [PIPE_STAGES, 128, 128], dtype=tl.bfloat16, scope=tle.gpu.smem
            )
            tle.gpu.warp_specialize(
                [
                    (
                        _consumer,
                        (
                            q,
                            bf16_k,
                            bf16_v,
                            metadata,
                            scratch,
                            empty,
                            k_full,
                            v_full,
                            meta_full,
                            auto_bm,
                            auto_bm_hi,
                            auto_bm_hi2,
                            auto_bm_hi3,
                            lo_s,
                            visits,
                            q_abs,
                            seq_len,
                            qk_scale,
                            obase,
                            row_off,
                            row_ok,
                            stride_qn,
                            v_scale_ptr,
                            M,
                            BLOCK_Q,
                            PIPE_STAGES,
                            HAS_SCALE,
                            True,
                        ),
                    ),
                    (
                        _producer_convert,
                        (
                            kv_desc,
                            bt_ptr + pid_b * stride_bt_b,
                            stride_bt_k,
                            raw_k,
                            raw_v,
                            metadata,
                            empty,
                            k_full,
                            v_full,
                            meta_full,
                            bf16_k,
                            bf16_v,
                            lo_s,
                            span,
                            active0,
                            active1,
                            active2,
                            active3,
                            PIPE_STAGES,
                            SPARSE_WALK,
                            seq_len,
                        ),
                    ),
                ],
                [4],
                [168],
            )
        else:
            tle.gpu.warp_specialize(
                [
                    (
                        _consumer,
                        (
                            q,
                            raw_k,
                            raw_v,
                            metadata,
                            scratch,
                            empty,
                            k_full,
                            v_full,
                            meta_full,
                            auto_bm,
                            auto_bm_hi,
                            auto_bm_hi2,
                            auto_bm_hi3,
                            lo_s,
                            visits,
                            q_abs,
                            seq_len,
                            qk_scale,
                            obase,
                            row_off,
                            row_ok,
                            stride_qn,
                            v_scale_ptr,
                            M,
                            BLOCK_Q,
                            PIPE_STAGES,
                            HAS_SCALE,
                            False,
                        ),
                    ),
                    (
                        _producer,
                        (
                            kv_desc,
                            bt_ptr + pid_b * stride_bt_b,
                            stride_bt_k,
                            raw_k,
                            raw_v,
                            metadata,
                            empty,
                            k_full,
                            v_full,
                            meta_full,
                            lo_s,
                            span,
                            active0,
                            active1,
                            active2,
                            active3,
                            PIPE_STAGES,
                            SPARSE_WALK,
                        ),
                    ),
                ],
                [1],
                [48],
            )
