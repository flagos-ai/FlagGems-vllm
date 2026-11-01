# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""MiniMax M3 selected-block attention on native vLLM 0.28 packed KV pages.

The mathematical kernels are extracted from the measured MiniMax integration;
server hooks, environment overrides and native compute fallbacks are omitted.
Precision is an explicit request independent of the shape-dependent schedule.
"""

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_int8_rows(x):
    """Symmetric per-row D-axis quantization, [-127,127], ties to even.

    Zero rows use scale=1. Nonfinite rows propagate NaN through their scale.
    D<=128 gives |INT32 dot| <= 128*127**2 = 2,064,512 < 2**31.
    """
    xf = x.to(tl.float32)
    finite = tl.min((tl.abs(xf) < float("inf")).to(tl.int32), axis=1) != 0
    amax = tl.max(tl.abs(xf), axis=1)
    safe_scale = tl.where((amax > 0) & finite, tl.div_rn(amax, 127.0), 1.0)
    normalized = tl.where(finite[:, None], tl.div_rn(xf, safe_scale[:, None]), 0.0)
    rounded = tl.extra.cuda.libdevice.nearbyint(normalized)
    xi = tl.minimum(127.0, tl.maximum(-127.0, rounded)).to(tl.int8)
    scale = tl.where(finite, safe_scale, float("nan"))
    return xi, scale


@triton.jit
def _int8_qk_dot(qi, q_scale, k_rows, q_rows):
    # Q rows are [query,head], K rows are [key_token,kv_head], both reduced
    # over head_dim. Accumulation is integer; dequantization is FP32.
    ki, k_scale = _quantize_int8_rows(k_rows)
    accum = tl.dot(qi, tl.trans(ki), out_dtype=tl.int32)
    result = accum.to(tl.float32) * q_scale[:, None] * k_scale[None, :]
    # INT8 has no representation for NaN/Inf. Never cast those values to
    # integers. A device-only rare branch restores the original BF16 dot for
    # affected row pairs, including signed +/-Inf (which must not all become
    # NaN). All finite pairs keep INT8/INT32 math. On ordinary finite tiles
    # the BF16 dot is not executed; no host readback or dispatch is involved.
    bad_q = q_scale != q_scale
    bad_k = k_scale != k_scale
    if tl.sum(bad_q.to(tl.int32), 0) + tl.sum(bad_k.to(tl.int32), 0) > 0:
        exceptional = tl.dot(q_rows.to(tl.bfloat16), tl.trans(k_rows.to(tl.bfloat16)))
        result = tl.where(bad_q[:, None] | bad_k[None, :], exceptional, result)
    return result


@triton.jit(do_not_specialize_on_alignment=["t_ptr", "cu_ptr", "seq_ptr", "pre_ptr"])
def _union_span_kernel(
    q_ptr,
    kv_ptr,
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
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_ptr + pid_b)
    q_len = tl.load(cu_ptr + pid_b + 1) - q_start
    q0 = pid_q * BLOCK_Q
    if q0 >= q_len:
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
        qi, qs = _quantize_int8_rows(q)
    tbase = t_ptr + pid_h * stride_th + (q_start + q0 + tok) * stride_tn
    big = 2147483647
    lo = tl.full((M,), big, tl.int32)
    hi = tl.full((M,), -1, tl.int32)
    if AUTO_SPAN:
        n_selected = tl.zeros((M,), tl.int32)
    for j in tl.static_range(TOPK):
        t = tl.load(tbase + j * stride_tk, mask=row_ok, other=-1)
        lo = tl.minimum(lo, tl.where(t >= 0, t, big))
        hi = tl.maximum(hi, t)
        if AUTO_SPAN:
            n_selected += ((t >= 0) & row_ok).to(tl.int32)
    lo_s = tl.min(lo, axis=0)
    hi_s = tl.max(hi, axis=0)
    span = hi_s - lo_s + 1
    if AUTO_SPAN:
        # Recomputed from device indices on every invocation / graph replay.
        # No host readback and no assumption that indexer outputs stay local.
        # Sparse lists with few valid entries need a proportionally tighter
        # span cap. A full TOPK=16 list permits 32 / 64 blocks respectively.
        density = max(1, AUTO_SPAN // TOPK)
        selected = tl.sum(n_selected, 0)
        compact = (selected == 0) | (
            (span <= AUTO_SPAN) & (span * n_tok * QH <= density * selected)
        )
        flag_off = (pid_b * tl.num_programs(1) + pid_h) * tl.num_programs(0) + pid_q
        tl.store(auto_flags_ptr + flag_off, compact.to(tl.int32))
        if not compact:
            return
        auto_bm = tl.zeros((M,), tl.int32)
        if AUTO_SPAN > 32:
            auto_bm_hi = tl.zeros((M,), tl.int32)
        for j in tl.static_range(TOPK):
            t = tl.load(tbase + j * stride_tk, mask=row_ok, other=-1)
            sh = (t - lo_s) & 31
            valid = row_ok & (t >= 0)
            auto_bm |= tl.where(valid & (t < lo_s + 32), 1 << sh, 0)
            if AUTO_SPAN > 32:
                auto_bm_hi |= tl.where(valid & (t >= lo_s + 32), 1 << sh, 0)
        # Bitmap membership collapses repeated ids; preserve selected-list
        # semantics by sending duplicate-bearing tiles to fallback.
        pop = tl.extra.cuda.libdevice.popc(auto_bm)
        if AUTO_SPAN > 32:
            pop += tl.extra.cuda.libdevice.popc(auto_bm_hi)
        duplicates = pop != n_selected
        if tl.sum(duplicates.to(tl.int32), axis=0) != 0:
            tl.store(auto_flags_ptr + flag_off, 0)
            return

    bt_row = bt_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_K)
    kv_k = (
        off_n[:, None] * stride_kv_pos
        + off_d[None, :] * stride_kv_d
        + pid_h * stride_kv_h
    )
    kv_v = (
        off_n[:, None] * stride_kv_pos
        + (head_dim + off_d[None, :]) * stride_kv_d
        + pid_h * stride_kv_h
    )

    m_i = tl.full((M,), float("-inf"), tl.float32)
    l_i = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, BLOCK_D), tl.float32)

    g = 0
    while g < span:
        lo_g = lo_s + g
        if AUTO_SPAN:
            bm = auto_bm
            if AUTO_SPAN > 32:
                bm = tl.where(g == 0, auto_bm, auto_bm_hi)
        else:
            bm = tl.zeros((M,), tl.int32)
            for j in tl.static_range(TOPK):
                t = tl.load(tbase + j * stride_tk, mask=row_ok, other=-1)
                sh = (t - lo_g) & 31
                ok = row_ok & (t >= lo_g) & (t < lo_g + SPAN)
                bm = bm | tl.where(ok, 1 << sh, 0)
        inner = tl.minimum(SPAN, span - g)
        for i in range(inner):
            blk = lo_g + i
            page = tl.load(bt_row + blk * stride_bt_k).to(tl.int64)
            c = blk * BLOCK_K
            pos = c + off_n
            pos_ok = pos < seq_len
            sel = ((bm >> i) & 1) != 0
            kbase = kv_ptr + page * stride_kv_blk
            if D_POW2:
                k = tl.load(kbase + kv_k, mask=pos_ok[:, None], other=0.0)
            else:
                k = tl.load(
                    kbase + kv_k, mask=pos_ok[:, None] & d_ok[None, :], other=0.0
                )
            if INT8_QK:
                qk = _int8_qk_dot(qi, qs, k, q) * qk_scale
            else:
                qk = tl.dot(q, tl.trans(k.to(q.dtype))) * qk_scale
            keep = sel[:, None] & (q_abs[:, None] >= pos[None, :]) & pos_ok[None, :]
            qk = tl.where(keep, qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp2(qk - m_safe[:, None])
            alpha = tl.exp2(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            if D_POW2:
                v = tl.load(kbase + kv_v, mask=pos_ok[:, None], other=0.0)
            else:
                v = tl.load(
                    kbase + kv_v, mask=pos_ok[:, None] & d_ok[None, :], other=0.0
                )
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype))
            m_i = m_new
        g += SPAN

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    if HAS_SCALE:
        inv_l = inv_l * tl.load(v_scale_ptr)
    acc = acc * inv_l[:, None]
    obase = o_ptr + ((q_start + q0) * heads_total + pid_h * QH) * stride_qn
    if D_POW2:
        tl.store(
            obase + row_off[:, None] * stride_qn + off_d[None, :],
            acc.to(o_ptr.dtype.element_ty),
            mask=row_ok[:, None],
        )
    else:
        tl.store(
            obase + row_off[:, None] * stride_qn + off_d[None, :],
            acc.to(o_ptr.dtype.element_ty),
            mask=row_ok[:, None] & d_ok[None, :],
        )


@triton.jit(do_not_specialize_on_alignment=["t_ptr", "cu_ptr", "seq_ptr"])
def _split_kernel(
    q_ptr,
    kv_ptr,
    t_ptr,
    o_ptr,
    l_ptr,
    bt_ptr,
    cu_ptr,
    seq_ptr,
    k_scale_ptr,
    v_scale_ptr,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SINGLE: tl.constexpr,
    FP8_QK: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    D_POW2: tl.constexpr,
    INT8_QK: tl.constexpr,
    USE_PREFIX: tl.constexpr,
    auto_flags_ptr=None,
    auto_prefix_ptr=None,
    AUTO_BLOCK_Q: tl.constexpr = 0,
    AUTO_TILES: tl.constexpr = 0,
    stride_bt_k=1,
):
    """One CTA per (query token, top-k chunk, kv head).

    Every entry of the chunk is visited; a negative (``-1`` padding) id is
    masked out instead of being trusted to sit after the last valid id, so a
    row that pads in the middle is still scored exactly once per valid block.
    ``SINGLE`` (NUM_CHUNKS == 1) writes the final output and skips the merge.
    """
    pid_qc = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_q = pid_qc // NUM_CHUNKS
    pid_c = pid_qc % NUM_CHUNKS

    q_start = tl.load(cu_ptr + pid_b)
    q_len = tl.load(cu_ptr + pid_b + 1) - q_start
    if pid_q >= q_len:
        return
    if AUTO_BLOCK_Q:
        flag_off = (
            pid_b * tl.num_programs(1) + pid_kh
        ) * AUTO_TILES + pid_q // AUTO_BLOCK_Q
        if tl.load(auto_flags_ptr + flag_off) != 0:
            return
        query_pos = tl.load(auto_prefix_ptr + pid_b) + pid_q
    elif USE_PREFIX:
        query_pos = tl.load(auto_prefix_ptr + pid_b) + pid_q
    else:
        query_pos = tl.load(seq_ptr + pid_b) - q_len + pid_q
    kv_len = tl.maximum(query_pos + 1, 0)
    if AUTO_BLOCK_Q or USE_PREFIX:
        kv_len = tl.minimum(kv_len, tl.load(seq_ptr + pid_b))

    chunk_size = (max_topk + NUM_CHUNKS - 1) // NUM_CHUNKS
    c0 = pid_c * chunk_size

    off_t = tl.arange(0, BLOCK_T)  # noqa: F841 - preserve the measured kernel AST
    idx_base = t_ptr + pid_kh * stride_th + (q_start + pid_q) * stride_tn
    off_n = tl.arange(0, BLOCK_K)
    off_d = tl.arange(0, BLOCK_D)
    d_ok = off_d < head_dim
    pid_h = pid_kh * gqa_group_size

    m_i = tl.full((BLOCK_H,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    safe_cur = tl.zeros((BLOCK_H,), tl.float32)
    off_h = tl.arange(0, BLOCK_H)
    hmask = off_h < gqa_group_size
    qptr = (
        q_ptr
        + (q_start + pid_q) * stride_qn
        + (pid_h + off_h)[:, None] * stride_qh
        + off_d[None, :] * stride_qd
    )
    if D_POW2:
        qb = tl.load(qptr, mask=hmask[:, None], other=0.0)
    else:
        qb = tl.load(qptr, mask=hmask[:, None] & d_ok[None, :], other=0.0)

    qk_scale = sm_scale * 1.4426950409
    if HAS_SCALE:
        qk_scale = qk_scale * tl.load(k_scale_ptr)
    if INT8_QK:
        qi, qscale = _quantize_int8_rows(qb)
    elif FP8_QK:
        # per-row e4m3 quantization of q; k stays fp8, so the QK dot runs at
        # 2x the bf16 tensor rate and the K dequant (the decode bottleneck)
        # disappears.
        qf32 = qb.to(tl.float32)
        amax = tl.max(tl.abs(qf32), axis=1)
        qscale = tl.where(amax > 0, amax / 448.0, 1.0)
        qf8 = (qf32 / qscale[:, None]).to(tl.float8e4nv)
    else:
        qscale = tl.full((BLOCK_H,), 1.0, tl.float32)

    cur = idx_base + c0 * stride_tk
    for i in tl.range(0, chunk_size):
        # chunk_size is cdiv(max_topk, NUM_CHUNKS), so the LAST chunk overruns
        # the list whenever NUM_CHUNKS does not divide TOPK (TOPK=16, chunks=3
        # reads 2 elements past the row and corrupts the result). Skip them.
        in_range = (c0 + i) < max_topk
        blk_raw = tl.load(cur, mask=in_range, other=-1).to(tl.int32)
        cur = cur + stride_tk
        ok_blk = (blk_raw >= 0) & in_range
        blk = tl.maximum(blk_raw, 0)
        page = tl.load(bt_ptr + pid_b * stride_bt_b + blk * stride_bt_k).to(tl.int64)
        pos = blk * BLOCK_K + off_n
        pos_ok = (pos < kv_len) & ok_blk
        kv_base = kv_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
        if D_POW2:
            k = tl.load(
                kv_base + off_n[None, :] * stride_kv_pos + off_d[:, None] * stride_kv_d,
                mask=pos_ok[None, :],
                other=0.0,
            )
        else:
            k = tl.load(
                kv_base + off_n[None, :] * stride_kv_pos + off_d[:, None] * stride_kv_d,
                mask=pos_ok[None, :] & d_ok[:, None],
                other=0.0,
            )
        qk = tl.zeros((BLOCK_H, BLOCK_K), tl.float32)
        qk += tl.where(pos_ok[None, :], 0.0, float("-inf"))
        if INT8_QK:
            qk += _int8_qk_dot(qi, qscale, tl.trans(k), qb) * qk_scale
        elif FP8_QK:
            qk += tl.dot(qf8, k) * (qk_scale * qscale)[:, None]
        else:
            qk += tl.dot(qb, k.to(qb.dtype)) * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        alpha = tl.exp2(m_i - safe)
        p = tl.exp2(qk - safe[:, None])
        l_ij = tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        if D_POW2:
            v = tl.load(
                kv_base
                + off_n[:, None] * stride_kv_pos
                + (head_dim + off_d[None, :]) * stride_kv_d,
                mask=pos_ok[:, None],
                other=0.0,
            )
        else:
            v = tl.load(
                kv_base
                + off_n[:, None] * stride_kv_pos
                + (head_dim + off_d[None, :]) * stride_kv_d,
                mask=pos_ok[:, None] & d_ok[None, :],
                other=0.0,
            )
        pv = v.to(qb.dtype)
        acc += tl.dot(p.to(qb.dtype), pv)
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        safe_cur = safe

    scale = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    if HAS_SCALE:
        scale = scale * tl.load(v_scale_ptr)
    acc = acc * scale[:, None]
    hh = off_h
    # BLOCK_D is next_power_of_2(head_dim), so for a non-power-of-two head_dim
    # (e.g. 96 -> 128) the store must mask the tail or it scribbles over the
    # next row of the output tensor.
    store_mask = (
        (hh < gqa_group_size)[:, None] & d_ok[None, :]
        if not D_POW2
        else (hh < gqa_group_size)[:, None]
    )
    if SINGLE:
        tl.store(
            o_ptr
            + (q_start + pid_q) * stride_o_b
            + (pid_h + hh)[:, None] * stride_o_h
            + off_d[None, :] * stride_o_d,
            acc.to(o_ptr.dtype.element_ty),
            mask=store_mask,
        )
    else:
        tl.store(
            o_ptr
            + pid_c * stride_o_c
            + (q_start + pid_q) * stride_o_b
            + (pid_h + hh)[:, None] * stride_o_h
            + off_d[None, :] * stride_o_d,
            acc.to(o_ptr.dtype.element_ty),
            mask=store_mask,
        )
        lse = tl.where(
            l_i > 0, safe_cur + tl.log2(tl.maximum(l_i, 1e-30)), float("-inf")
        )
        tl.store(
            l_ptr
            + pid_c * stride_l_c
            + (q_start + pid_q) * stride_l_b
            + (pid_h + hh) * stride_l_h,
            lse,
            mask=hmask,
        )


@triton.jit
def _merge_split_kernel(
    o_ptr,
    l_ptr,
    out_ptr,
    cu_ptr,
    gqa_group_size,
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_POW2: tl.constexpr,
    auto_flags_ptr=None,
    AUTO_BLOCK_Q: tl.constexpr = 0,
    AUTO_TILES: tl.constexpr = 0,
):
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_ptr + pid_b)
    q_len = tl.load(cu_ptr + pid_b + 1) - q_start
    if pid_q >= q_len:
        return
    if AUTO_BLOCK_Q:
        flag_off = (
            pid_b * tl.num_programs(1) + pid_kh
        ) * AUTO_TILES + pid_q // AUTO_BLOCK_Q
        if tl.load(auto_flags_ptr + flag_off) != 0:
            return
    off_h = tl.arange(0, BLOCK_H)
    off_d = tl.arange(0, BLOCK_D)
    hmask = off_h < gqa_group_size
    pid_h = pid_kh * gqa_group_size
    tok = q_start + pid_q
    lb = l_ptr + tok * stride_l_b + (pid_h + off_h) * stride_l_h
    lse = tl.load(lb + 0 * stride_l_c, mask=hmask, other=float("-inf"))
    m = lse
    for c in tl.static_range(1, NUM_CHUNKS):
        m = tl.maximum(m, tl.load(lb + c * stride_l_c, mask=hmask, other=float("-inf")))
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    denom = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    for c in tl.static_range(NUM_CHUNKS):
        w = tl.exp2(
            tl.load(lb + c * stride_l_c, mask=hmask, other=float("-inf")) - m_safe
        )
        denom += w
        if D_POW2:
            o = tl.load(
                o_ptr
                + c * stride_o_c
                + tok * stride_o_b
                + (pid_h + off_h)[:, None] * stride_o_h
                + off_d[None, :] * stride_o_d,
                mask=hmask[:, None],
                other=0.0,
            )
        else:
            o = tl.load(
                o_ptr
                + c * stride_o_c
                + tok * stride_o_b
                + (pid_h + off_h)[:, None] * stride_o_h
                + off_d[None, :] * stride_o_d,
                mask=hmask[:, None] & (off_d < head_dim)[None, :],
                other=0.0,
            )
        acc += w[:, None] * o.to(tl.float32)
    acc = acc / tl.where(denom > 0, denom, 1.0)[:, None]
    if D_POW2:
        tl.store(
            out_ptr
            + tok * stride_out_n
            + (pid_h + off_h)[:, None] * stride_out_h
            + off_d[None, :] * stride_out_d,
            acc.to(out_ptr.dtype.element_ty),
            mask=hmask[:, None],
        )
    else:
        tl.store(
            out_ptr
            + tok * stride_out_n
            + (pid_h + off_h)[:, None] * stride_out_h
            + off_d[None, :] * stride_out_d,
            acc.to(out_ptr.dtype.element_ty),
            mask=hmask[:, None] & (off_d < head_dim)[None, :],
        )


# Fixed launch schedules preserve the measured source. See the explicit
# no-autotune exemption and pending package-integration gates in the PR docs.
_WORKSPACE = {}


@lru_cache(maxsize=None)
def _device_capability(device_index):
    # Probe only after supported CUDA inputs exist; never at module import.
    if torch.version.hip is not None:
        raise NotImplementedError("MiniMax M3 attention currently supports NVIDIA only")
    return torch.cuda.get_device_capability(device_index)


@lru_cache(maxsize=None)
def _is_h100(device_index):
    props = torch.cuda.get_device_properties(device_index)
    return (props.major, props.minor) == (9, 0) and props.name.upper().startswith(
        "NVIDIA H100"
    )


def _resolve_qk_dtype(requested, kv_dtype, capability):
    """Never change a BF16 request merely because hardware lacks FP8.

    An explicit FP8 request can use INT8 on SM80/86 only with BF16 storage
    already selected by the caller before allocation. FP8 bytes are never
    reinterpreted as INT8 or BF16.
    """
    if requested not in ("bf16", "fp8", "int8"):
        raise ValueError("qk_dtype must be 'bf16', 'fp8', or 'int8'")
    sm = capability[0] * 10 + capability[1]
    if sm < 80:
        raise NotImplementedError("MiniMax M3 attention requires NVIDIA SM80+")
    if kv_dtype == torch.float8_e4m3fn and sm < 89:
        raise ValueError(
            "FP8 KV storage requires FP8 hardware; the upstream precision planner "
            "must allocate BF16 KV before requesting the INT8 compute fallback"
        )
    if requested == "fp8":
        if sm < 89:
            return "int8"
        if kv_dtype != torch.float8_e4m3fn:
            raise ValueError("FP8 QK requires typed float8_e4m3fn KV storage")
    if requested == "int8" and kv_dtype != torch.bfloat16:
        raise ValueError("INT8 QK requires BF16 KV storage, not reinterpreted FP8")
    return requested


def _validate_inputs(
    q,
    kv_cache,
    topk_idx,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    num_kv_heads,
    sm_scale,
    output,
    k_scale,
    v_scale,
):
    if q.ndim != 3 or output.shape != q.shape:
        raise ValueError("Q/output must have matching [tokens, heads, D] shapes")
    if q.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise NotImplementedError("Q/output must be BF16")
    if q.device.type != "cuda" or torch.version.hip is not None:
        raise NotImplementedError("MiniMax M3 attention currently supports NVIDIA CUDA")
    if not q.is_contiguous() or not output.is_contiguous():
        raise NotImplementedError("Q/output must be contiguous")
    if q.requires_grad or kv_cache.requires_grad or output.requires_grad:
        raise NotImplementedError("MiniMax M3 sparse attention is inference only")
    if (
        isinstance(num_kv_heads, bool)
        or not isinstance(num_kv_heads, int)
        or num_kv_heads < 1
        or q.shape[1] != 8 * num_kv_heads
    ):
        raise NotImplementedError("Only GQA8 is validated")
    if q.shape[-1] not in (96, 128):
        raise NotImplementedError("Only head dimensions 96 and 128 are validated")
    if kv_cache.ndim != 4 or kv_cache.shape[1:] != (num_kv_heads, 128, 2 * q.shape[-1]):
        raise ValueError("KV must use packed [pages, KV_heads, 128, 2*D] storage")
    if kv_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise NotImplementedError("KV must be BF16 or typed float8_e4m3fn")
    if topk_idx.ndim != 3 or topk_idx.shape[:2] != (num_kv_heads, q.shape[0]):
        raise ValueError("topk_idx must be [KV_heads, tokens, TOPK]")
    if not 1 <= topk_idx.shape[-1] <= 32:
        raise NotImplementedError("Only 1 <= TOPK <= 32 is validated")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q must have shape [batch+1]")
    batch = cu_seqlens_q.numel() - 1
    if seq_lens.shape != (batch,) or prefix_lens.shape != (batch,):
        raise ValueError("seq_lens/prefix_lens must have shape [batch]")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] < 1
    ):
        raise ValueError("block_table must be [batch, positive capacity]")
    metadata = (topk_idx, block_table, cu_seqlens_q, seq_lens, prefix_lens)
    if any(t.dtype != torch.int32 for t in metadata):
        raise ValueError("Attention index metadata must use int32")
    if any(t.stride(0) != 1 for t in (cu_seqlens_q, seq_lens, prefix_lens)):
        raise NotImplementedError("Length metadata must be contiguous")
    if any(t.device != q.device for t in (kv_cache, output) + metadata):
        raise ValueError("All attention tensors must be on the same device")
    if any(
        any(stride <= 0 for stride in t.stride())
        for t in (q, kv_cache, output) + metadata
    ):
        raise NotImplementedError("Only positive strides are supported")
    if (
        isinstance(max_query_len, bool)
        or not isinstance(max_query_len, int)
        or max_query_len < 0
    ):
        raise ValueError("max_query_len must be a nonnegative host integer")
    if q.shape[0] and max_query_len == 0:
        raise ValueError("Nonempty Q requires positive max_query_len")
    # The retained kernels use int32 logical index expressions before page
    # pointers are promoted to int64. Reject larger logical views explicitly.
    if any(
        sum((n - 1) * st for n, st in zip(t.shape, t.stride())) >= 2**31
        for t in (q, output) + metadata
    ):
        raise NotImplementedError("Logical attention offsets must fit int32")
    if (
        isinstance(sm_scale, bool)
        or not isinstance(sm_scale, (float, int))
        or abs(sm_scale) > 3.4028234663852886e38
        or not math.isfinite(sm_scale)
    ):
        raise ValueError("sm_scale must be finite in FP32")
    if (k_scale is None) != (v_scale is None):
        raise ValueError("K/V scales must be provided together")
    if k_scale is not None:
        if any(
            t.numel() != 1 or t.dtype != torch.float32 or t.device != q.device
            for t in (k_scale, v_scale)
        ):
            raise ValueError("K/V scales must be scalar FP32 tensors on the Q device")


def _workspace(chunks, q, num_kv_heads, max_query_len, block_q):
    # Stream isolation prevents concurrent calls from overwriting partials.
    # Strong references preserve captured pointers. Warm every shape/stream
    # before capture; replay performs no allocation and no host tensor reads.
    stream = torch.cuda.current_stream(q.device).cuda_stream
    key = (
        q.device.index,
        stream,
        q.shape,
        chunks,
        num_kv_heads,
        max_query_len,
        block_q,
    )
    found = _WORKSPACE.get(key)
    if found is None:
        partial = torch.empty((chunks,) + q.shape, dtype=q.dtype, device=q.device)
        lse = torch.empty((chunks,) + q.shape[:2], dtype=torch.float32, device=q.device)
        found = (partial, lse)
        _WORKSPACE[key] = found
    return found


def _launch_selected(
    q,
    kv,
    topk,
    bt,
    cu,
    seq,
    prefix,
    max_q,
    num_kv_heads,
    sm_scale,
    output,
    k_scale,
    v_scale,
    chunks,
    qk_dtype,
    flags=None,
    block_q=0,
):
    total_q, heads, d = q.shape
    gqa = heads // num_kv_heads
    bh = max(16, gqa) if qk_dtype == "int8" else gqa
    if chunks == 1:
        partial, lse = output, output
        so_c = sl_c = sl_b = sl_h = 0
        so_b, so_h, so_d = output.stride()
    else:
        partial, lse = _workspace(chunks, q, num_kv_heads, max_q, block_q)
        so_c, so_b, so_h, so_d = partial.stride()
        sl_c, sl_b, sl_h = lse.stride()
    batch = cu.numel() - 1
    has_scale = k_scale is not None
    k_arg, v_arg = (k_scale, v_scale) if has_scale else (output, output)
    _split_kernel[(max_q * chunks, num_kv_heads, batch)](
        q.view(-1, d),
        kv,
        topk,
        partial,
        lse,
        bt,
        cu,
        seq,
        k_arg,
        v_arg,
        gqa,
        d,
        topk.shape[-1],
        sm_scale,
        *q.stride(),
        *kv.stride(),
        *topk.stride(),
        so_c,
        so_b,
        so_h,
        so_d,
        sl_c,
        sl_b,
        sl_h,
        bt.stride(0),
        NUM_CHUNKS=chunks,
        BLOCK_K=128,
        BLOCK_H=bh,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_T=16,
        SINGLE=chunks == 1,
        FP8_QK=qk_dtype == "fp8",
        HAS_SCALE=has_scale,
        D_POW2=d == triton.next_power_of_2(d),
        INT8_QK=qk_dtype == "int8",
        USE_PREFIX=True,
        auto_flags_ptr=flags,
        auto_prefix_ptr=prefix,
        AUTO_BLOCK_Q=block_q,
        AUTO_TILES=triton.cdiv(max_q, block_q) if block_q else 0,
        stride_bt_k=bt.stride(1),
        num_warps=4,
        num_stages=1,
    )
    if chunks > 1:
        _merge_split_kernel[(max_q, num_kv_heads, batch)](
            partial,
            lse,
            output,
            cu,
            gqa,
            d,
            so_c,
            so_b,
            so_h,
            so_d,
            sl_c,
            sl_b,
            sl_h,
            *output.stride(),
            NUM_CHUNKS=chunks,
            BLOCK_H=bh,
            BLOCK_D=triton.next_power_of_2(d),
            D_POW2=d == triton.next_power_of_2(d),
            auto_flags_ptr=flags,
            AUTO_BLOCK_Q=block_q,
            AUTO_TILES=triton.cdiv(max_q, block_q) if block_q else 0,
            num_warps=4,
            num_stages=1,
        )


def minimax_m3_sparse_attn(
    q,
    kv_cache,
    topk_idx,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    num_kv_heads,
    sm_scale,
    output,
    k_scale=None,
    v_scale=None,
    *,
    qk_dtype="bf16",
    schedule="auto",
    chunks=None,
):
    """Write causal sparse attention into ``output`` and return it.

    ``qk_dtype`` is the requested computation, not the KV storage dtype.
    BF16 is preserved on all supported platforms. Explicit FP8 computation
    requires typed FP8 KV on SM89+; on SM80/86 that request uses INT8 only if
    the caller has already allocated BF16 KV. INT8 can also be requested
    explicitly. PV remains BF16. No FP8 capability is inferred from dtype.

    ``schedule='auto'`` uses a GPU union predicate on the measured H100 domain
    (q>=384, TOPK16, GQA8) and selected-block work elsewhere. The predicate is
    recomputed on every graph replay and preserves duplicates through selected
    fallback. ``schedule='selected'`` forces exact selected-block traversal.
    FP8 QK always uses selected traversal, so a call never mixes QK precision.

    Caller contracts: CU starts at zero and ends at Q token count; per-request
    Q length <= max_query_len; nonnegative sequence/prefix lengths fit the block
    table; every nonnegative selected ID fits that table and physical pages.
    Metadata contents stay on device and are not read/synchronized here.
    Output must not overlap inputs. Inference only; no autograd registration.
    """
    _validate_inputs(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        sm_scale,
        output,
        k_scale,
        v_scale,
    )
    if schedule not in ("auto", "selected"):
        raise ValueError("schedule must be 'auto' or 'selected'")
    if chunks is not None and (
        isinstance(chunks, bool)
        or not isinstance(chunks, int)
        or chunks not in (1, 2, 3, 4, 8, 16)
    ):
        raise ValueError("chunks must be one of 1, 2, 3, 4, 8, 16")
    if qk_dtype not in ("bf16", "fp8", "int8"):
        raise ValueError("qk_dtype must be 'bf16', 'fp8', or 'int8'")
    if not q.shape[0]:
        return output
    precision = _resolve_qk_dtype(
        qk_dtype, kv_cache.dtype, _device_capability(q.device.index)
    )
    batch = cu_seqlens_q.numel() - 1
    use_union = (
        schedule == "auto"
        and max_query_len >= 384
        and topk_idx.shape[-1] == 16
        and precision != "fp8"
        and _is_h100(q.device.index)
    )
    block_q = (16 if max_query_len >= 2048 else 8) if use_union else 0
    if chunks is None:
        if use_union:
            chunks = 1 if max_query_len >= 2048 else 2
        elif max_query_len <= 4:
            chunks = 16 if batch < 32 else (8 if batch <= 48 else 4)
        else:
            chunks = 4
    if chunks * q.numel() >= 2**31:
        raise NotImplementedError("Split workspace offsets must fit int32")
    flags = None
    if use_union:
        # The union kernel writes every flag that the selected/merge kernels
        # can read. Invalid tail tiles return in all three before flag reads.
        # Allocation is uninitialized by design: no torch.zeros compute path.
        stream = torch.cuda.current_stream(q.device).cuda_stream
        tiles = triton.cdiv(max_query_len, block_q)
        flag_key = ("flags", q.device.index, stream, batch, num_kv_heads, tiles)
        flags = _WORKSPACE.get(flag_key)
        if flags is None:
            flags = torch.empty(
                (batch, num_kv_heads, tiles), dtype=torch.int32, device=q.device
            )
            _WORKSPACE[flag_key] = flags
        has_scale = k_scale is not None
        k_arg, v_arg = (k_scale, v_scale) if has_scale else (output, output)
        d = q.shape[-1]
        flat_q = q.view(-1, d)
        _union_span_kernel[(tiles, num_kv_heads, batch)](
            flat_q,
            kv_cache,
            topk_idx,
            output.view(-1, d),
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            k_arg,
            v_arg,
            8,
            d,
            topk_idx.shape[-1],
            sm_scale,
            q.shape[1],
            flat_q.stride(0),
            *kv_cache.stride(),
            *topk_idx.stride(),
            block_table.stride(0),
            BLOCK_Q=block_q,
            BLOCK_K=128,
            BLOCK_D=triton.next_power_of_2(d),
            TOPK=topk_idx.shape[-1],
            QH=8,
            SPAN=32,
            HAS_SCALE=has_scale,
            D_POW2=d == triton.next_power_of_2(d),
            INT8_QK=precision == "int8",
            auto_flags_ptr=flags,
            AUTO_SPAN=64 if max_query_len >= 2048 else 32,
            stride_bt_k=block_table.stride(1),
            num_warps=8,
            num_stages=3,
        )
    _launch_selected(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        sm_scale,
        output,
        k_scale,
        v_scale,
        chunks,
        precision,
        flags,
        block_q,
    )
    return output
