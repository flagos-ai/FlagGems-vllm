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

"""Triton (v0) and TLE (v1/v2) implementations of TND LightningIndexer on Hygon."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.lightning_indexer import (
    _validate_lightning_indexer_inputs,
    compute_row_bounds,
    fill_empty_outputs,
    select_topk_sorted,
)

_hcu_tle = None


def _init_hcu_tle():
    global _hcu_tle
    if _hcu_tle is None:
        import triton.experimental.tle as extension

        try:
            supported = extension.get_supported_primitives("hcu")
        except TypeError:
            supported = extension.get_supported_primitives()
        if "extract_tile" not in supported:
            raise RuntimeError("Hygon LightningIndexer requires TLE extract_tile")
        from triton.experimental.tle import language

        _hcu_tle = language


@triton.jit
def _row_meta(
    QEnd, KMeta, row, B: tl.constexpr, SPARSE_MODE: tl.constexpr, IS_PAGED: tl.constexpr
):
    """Resolve request index, ragged key start, and visible key count on device."""
    req = 0
    end = tl.load(QEnd)
    while req < B - 1 and row >= end:
        req += 1
        end = tl.load(QEnd + req)
    if IS_PAGED:
        k_start = 0
        k_len = tl.load(KMeta + req)
    else:
        k_end = tl.load(KMeta + req)
        k_start = tl.load(KMeta + req - 1) if req > 0 else 0
        k_len = k_end - k_start
    raw_valid = (k_len - (end - row) + 1) if SPARSE_MODE == 3 else k_len
    valid = tl.maximum(0, raw_valid)
    return req, k_start, valid


@triton.jit
def _hygon_direct_indices(
    QEnd,
    KMeta,
    Out,
    B: tl.constexpr,
    SPARSE_COUNT: tl.constexpr,
    SPARSE_MODE: tl.constexpr,
    IS_PAGED: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Write sequential visible indices when `active_max <= sparse_count`."""
    row = tl.program_id(0)
    _, _, valid = _row_meta(QEnd, KMeta, row, B, SPARSE_MODE, IS_PAGED)
    row_out = Out + row.to(tl.int64) * SPARSE_COUNT
    for chunk_idx in tl.static_range((SPARSE_COUNT + CHUNK - 1) // CHUNK):
        ix = chunk_idx * CHUNK + tl.arange(0, CHUNK)
        tl.store(
            row_out + ix,
            tl.where(ix < valid, ix, -1),
            mask=ix < SPARSE_COUNT,
        )


@triton.jit
def _paged_scores(
    Q,
    K,
    W,
    QEnd,
    KMeta,
    BT,
    Scores,
    H: tl.constexpr,
    D: tl.constexpr,
    PS: tl.constexpr,
    BTW: tl.constexpr,
    MAXK: tl.constexpr,
    B: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BK: tl.constexpr,
    K_STEPS: tl.constexpr,
    SPARSE_COUNT: tl.constexpr,
    SPARSE_MODE: tl.constexpr,
    IS_PAGED: tl.constexpr,
    SKIP_SHORT_ROWS: tl.constexpr,
    USE_TLE: tl.constexpr,
):
    """Q-stationary K@Q^T scorer with coalesced row-major K, split-D MFMA dot, and axis-1 reduction."""
    row = tl.program_id(0)
    k_group = tl.program_id(1)
    req, k_start, raw_valid = _row_meta(QEnd, KMeta, row, B, SPARSE_MODE, IS_PAGED)
    valid = tl.minimum(MAXK, raw_valid)
    need_scores = valid > SPARSE_COUNT if SKIP_SHORT_ROWS else valid > 0
    base_k = k_group * (K_STEPS * BK)
    if need_scores and base_k < valid:
        hs = tl.arange(0, BH)
        w = tl.load(W + row.to(tl.int64) * H + hs, mask=hs < H, other=0).to(tl.float32)
        q_base = Q + row.to(tl.int64) * H * D
        ds0 = tl.arange(0, 64)
        ds1 = 64 + ds0
        q0 = tl.load(
            q_base + hs[:, None] * D + ds0[None, :], mask=hs[:, None] < H, other=0
        )
        q1 = tl.load(
            q_base + hs[:, None] * D + ds1[None, :], mask=hs[:, None] < H, other=0
        )
        qt0 = tl.trans(q0)
        qt1 = tl.trans(q1)
        row_scores = Scores + row.to(tl.int64) * MAXK

        for step in range(0, K_STEPS):
            k0 = base_k + step * BK
            if k0 < valid:
                ks = k0 + tl.arange(0, BK)
                if IS_PAGED:
                    pages = tl.load(BT + req * BTW + ks // PS, mask=ks < valid, other=0)
                    k_rows = pages.to(tl.int64) * PS + (ks % PS)
                else:
                    k_rows = k_start.to(tl.int64) + ks
                k0v = tl.load(
                    K + k_rows[:, None] * D + ds0[None, :],
                    mask=ks[:, None] < valid,
                    other=0,
                )
                k1v = tl.load(
                    K + k_rows[:, None] * D + ds1[None, :],
                    mask=ks[:, None] < valid,
                    other=0,
                )
                qk = tl.dot(k0v, qt0)
                qk = tl.dot(k1v, qt1, qk)
                score = tl.sum(tl.maximum(qk, 0.0) * w[None, :], axis=1)
                if USE_TLE:
                    for subtile in tl.static_range(BK // 64):
                        part = _hcu_tle.extract_tile(score, subtile, (64,))
                        key_ix = k0 + subtile * 64 + tl.arange(0, 64)
                        tl.store(
                            row_scores + key_ix,
                            tl.where(key_ix < valid, part, -float("inf")),
                            mask=key_ix < MAXK,
                        )
                else:
                    tl.store(
                        row_scores + ks,
                        tl.where(ks < valid, score, -float("inf")),
                        mask=ks < MAXK,
                    )


@triton.jit
def _fp32_to_inv_i32(x):
    """Map FP32 monotonically into descending signed int32 (NaN first, -inf last)."""
    bits = x.to(tl.int32, bitcast=True)
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    oi32 = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return tl.where(is_nan, -2147483648, ~oi32)


@triton.jit
def _find_threshold_bin(hist_ptr, need, NBINS: tl.constexpr, BLOCK: tl.constexpr):
    """Scan histogram in register chunks to locate the threshold bin, prefix count, and bin size."""
    lane = tl.arange(0, BLOCK)
    running = tl.zeros((), dtype=tl.int32)
    thr_bin = tl.full((), NBINS - 1, dtype=tl.int32)
    lt_total = tl.zeros((), dtype=tl.int32)
    eq_total = tl.zeros((), dtype=tl.int32)
    found_flag = tl.zeros((), dtype=tl.int32)
    for chunk_idx in tl.static_range(NBINS // BLOCK):
        if found_flag == 0:
            bins = chunk_idx * BLOCK + lane
            cnts = tl.load(hist_ptr + bins)
            inc = tl.cumsum(cnts, axis=0)
            exc = inc - cnts
            pref = running + exc
            nxt = running + inc
            hit = (pref < need) & (nxt >= need)
            any_hit = tl.sum(hit.to(tl.int32), axis=0)
            if any_hit > 0:
                thr_bin = tl.sum(tl.where(hit, bins, 0), axis=0)
                lt_total = tl.sum(tl.where(hit, pref, 0), axis=0)
                eq_total = tl.sum(tl.where(hit, cnts, 0), axis=0)
                found_flag = 1
            running += tl.sum(cnts, axis=0)
    return thr_bin, lt_total, eq_total


@triton.jit
def _finish_compact_bucket(
    inv_ptr,
    col_ptr,
    out_row,
    found,
    need,
    eq_cnt,
    SB: tl.constexpr,
):
    """Sort compacted threshold-bucket pairs in registers/LDS and emit remaining top-k indices."""
    s_ix = tl.arange(0, SB)
    in_b = s_ix < eq_cnt
    inv_v = tl.load(inv_ptr + s_ix, mask=in_b, other=2147483647)
    col_v = tl.load(col_ptr + s_ix, mask=in_b, other=0)
    oi32 = tl.where(in_b, ~inv_v, -2147483648)
    packed = (oi32.to(tl.int64) << 32) | (0x7FFFFFFF - col_v).to(tl.int64)
    sp = tl.sort(packed, dim=0, descending=True)
    sel_id = (0x7FFFFFFF - (sp & 0x7FFFFFFF)).to(tl.int32)
    tl.store(out_row + found + s_ix, sel_id, mask=s_ix < need)


@triton.jit
def _hygon_radix_topk(
    Scores,
    QEnd,
    KMeta,
    Hist,
    Out,
    MAXK: tl.int64,
    B: tl.constexpr,
    TOPK: tl.constexpr,
    SPARSE_MODE: tl.constexpr,
    IS_PAGED: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Single-kernel radix TopK with compact boundary-bucket sort (2 scans on typical rows)."""
    row = tl.program_id(0)
    _, _, raw_valid = _row_meta(QEnd, KMeta, row, B, SPARSE_MODE, IS_PAGED)
    valid = tl.minimum(MAXK.to(tl.int32), raw_valid)
    out_row = Out + row.to(tl.int64) * TOPK

    if valid <= TOPK:
        for base in tl.static_range(0, TOPK, 512):
            ix = base + tl.arange(0, 512)
            tl.store(
                out_row + ix,
                tl.where(ix < valid, ix, -1),
                mask=ix < TOPK,
            )
        return

    score_row = Scores + row.to(tl.int64) * MAXK
    h0 = Hist + row.to(tl.int64) * 4096
    h1 = h0 + 2048
    lane512 = tl.arange(0, 512)
    for c in tl.static_range(4):
        tl.store(h0 + c * 512 + lane512, 0)
    tl.debug_barrier()

    offs = tl.arange(0, CHUNK)
    ones = tl.full([CHUNK], 1, tl.int32)
    n_tiles = tl.cdiv(valid, CHUNK)

    # Pass 0: top 11 bits -> 2048 bins.
    for t in range(0, n_tiles):
        cols = t * CHUNK + offs
        in_r = cols < valid
        x = tl.load(score_row + cols, mask=in_r, other=-float("inf"))
        inv = _fp32_to_inv_i32(x)
        b0 = ((inv >> 21) & 0x7FF) ^ 0x400
        tl.atomic_add(h0 + b0, ones, mask=in_r, sem="relaxed", scope="cta")
    tl.debug_barrier()

    thr0, _, eq0_total = _find_threshold_bin(h0, TOPK, NBINS=2048, BLOCK=512)
    tl.debug_barrier()

    # Fast 2-scan path: compact threshold bucket (<= 2048 items) into h0/h1 and sort directly.
    if eq0_total <= 2048:
        found = tl.zeros((), dtype=tl.int32)
        eq_cnt = tl.zeros((), dtype=tl.int32)
        for t in range(0, n_tiles):
            cols = t * CHUNK + offs
            in_r = cols < valid
            x = tl.load(score_row + cols, mask=in_r, other=-float("inf"))
            inv = _fp32_to_inv_i32(x)
            b0 = ((inv >> 21) & 0x7FF) ^ 0x400
            take = in_r & (b0 < thr0)
            ti = take.to(tl.int32)
            n_take = tl.sum(ti, axis=0)
            if n_take > 0:
                pos = found + tl.cumsum(ti, axis=0) - 1
                tl.store(out_row + pos, cols, mask=take)
                found += n_take
            eq0 = in_r & (b0 == thr0)
            ei = eq0.to(tl.int32)
            n_eq = tl.sum(ei, axis=0)
            if n_eq > 0:
                epos = eq_cnt + tl.cumsum(ei, axis=0) - 1
                tl.store(h0 + epos, inv, mask=eq0)
                tl.store(h1 + epos, cols, mask=eq0)
                eq_cnt += n_eq
        tl.debug_barrier()
        need = TOPK - found
        if need <= 0:
            return
        if eq_cnt <= need:
            for base in tl.static_range(0, 2048, 512):
                s_ix = base + lane512
                m = s_ix < eq_cnt
                tl.store(
                    out_row + found + s_ix, tl.load(h1 + s_ix, mask=m, other=0), mask=m
                )
            return
        if eq_cnt <= 512:
            _finish_compact_bucket(h0, h1, out_row, found, need, eq_cnt, SB=512)
        elif eq_cnt <= 1024:
            _finish_compact_bucket(h0, h1, out_row, found, need, eq_cnt, SB=1024)
        else:
            _finish_compact_bucket(h0, h1, out_row, found, need, eq_cnt, SB=2048)
        return

    # Full-precision fallback when threshold bucket exceeds 2048 (e.g. heavy ties).
    for c in tl.static_range(4):
        tl.store(h1 + c * 512 + lane512, 0)
    tl.debug_barrier()

    found = tl.zeros((), dtype=tl.int32)
    for t in range(0, n_tiles):
        cols = t * CHUNK + offs
        in_r = cols < valid
        x = tl.load(score_row + cols, mask=in_r, other=-float("inf"))
        inv = _fp32_to_inv_i32(x)
        b0 = ((inv >> 21) & 0x7FF) ^ 0x400
        take = in_r & (b0 < thr0)
        ti = take.to(tl.int32)
        n_take = tl.sum(ti, axis=0)
        if n_take > 0:
            pos = found + tl.cumsum(ti, axis=0) - 1
            tl.store(out_row + pos, cols, mask=take)
            found += n_take
        eq0 = in_r & (b0 == thr0)
        b1 = (inv >> 10) & 0x7FF
        tl.atomic_add(h1 + b1, ones, mask=eq0, sem="relaxed", scope="cta")
    tl.debug_barrier()

    need = TOPK - found
    if need <= 0:
        return

    for c in tl.static_range(2):
        tl.store(h0 + c * 512 + lane512, 0)
    tl.debug_barrier()

    thr1, _, _ = _find_threshold_bin(h1, need, NBINS=2048, BLOCK=512)

    for t in range(0, n_tiles):
        cols = t * CHUNK + offs
        in_r = cols < valid
        x = tl.load(score_row + cols, mask=in_r, other=-float("inf"))
        inv = _fp32_to_inv_i32(x)
        b0 = ((inv >> 21) & 0x7FF) ^ 0x400
        m0 = in_r & (b0 == thr0)
        b1 = (inv >> 10) & 0x7FF
        take = m0 & (b1 < thr1)
        ti = take.to(tl.int32)
        n_take = tl.sum(ti, axis=0)
        if n_take > 0:
            pos = found + tl.cumsum(ti, axis=0) - 1
            tl.store(out_row + pos, cols, mask=take)
            found += n_take
        eq1 = m0 & (b1 == thr1)
        b2 = inv & 0x3FF
        tl.atomic_add(h0 + b2, ones, mask=eq1, sem="relaxed", scope="cta")
    tl.debug_barrier()

    need = TOPK - found
    if need <= 0:
        return

    thr2, lt2, _ = _find_threshold_bin(h0, need, NBINS=1024, BLOCK=512)
    eq_found = found + lt2

    for t in range(0, n_tiles):
        cols = t * CHUNK + offs
        in_r = cols < valid
        x = tl.load(score_row + cols, mask=in_r, other=-float("inf"))
        inv = _fp32_to_inv_i32(x)
        b0 = ((inv >> 21) & 0x7FF) ^ 0x400
        b1 = (inv >> 10) & 0x7FF
        b2 = inv & 0x3FF
        m1 = in_r & (b0 == thr0) & (b1 == thr1)
        take_lt = m1 & (b2 < thr2)
        t_lt = take_lt.to(tl.int32)
        n_lt = tl.sum(t_lt, axis=0)
        if n_lt > 0:
            pos_lt = found + tl.cumsum(t_lt, axis=0) - 1
            tl.store(out_row + pos_lt, cols, mask=take_lt)
            found += n_lt

        take_eq = m1 & (b2 == thr2)
        t_eq = take_eq.to(tl.int32)
        n_eq = tl.sum(t_eq, axis=0)
        if n_eq > 0:
            pos_eq = eq_found + tl.cumsum(t_eq, axis=0) - 1
            tl.store(out_row + pos_eq, cols, mask=take_eq & (pos_eq < TOPK))
            eq_found += n_eq


@triton.jit
def _hygon_bitonic_topk(
    Scores,
    QEnd,
    KMeta,
    OutIndices,
    OutValues,
    MAXK: tl.int64,
    B: tl.constexpr,
    BLOCK: tl.constexpr,
    SPARSE_COUNT: tl.constexpr,
    SPARSE_MODE: tl.constexpr,
    IS_PAGED: tl.constexpr,
    RETURN_VALUE: tl.constexpr,
):
    """Single-kernel descending packed-int64 bitonic sort for `max_keys <= 8192`."""
    row = tl.program_id(0)
    _, _, raw_valid = _row_meta(QEnd, KMeta, row, B, SPARSE_MODE, IS_PAGED)
    valid = tl.minimum(MAXK.to(tl.int32), raw_valid)
    cols = tl.arange(0, BLOCK)
    active = cols < valid
    sc = tl.load(
        Scores + row.to(tl.int64) * MAXK + cols,
        mask=active,
        other=-float("inf"),
    )
    bits = sc.to(tl.int32, bitcast=True)
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    oi32 = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    oi32 = tl.where(is_nan, 0x7FFFFFFF, oi32)
    oi32 = tl.where(active, oi32, -2147483648)
    inv_id = (0x7FFFFFFF - cols).to(tl.int64)
    packed = (oi32.to(tl.int64) << 32) | inv_id
    sp = tl.sort(packed, dim=0, descending=True)

    out_mask = cols < SPARSE_COUNT
    valid_out = cols < valid
    row_out = row.to(tl.int64) * SPARSE_COUNT
    out_id = (0x7FFFFFFF - (sp & 0x7FFFFFFF)).to(tl.int32)
    tl.store(
        OutIndices + row_out + cols,
        tl.where(valid_out, out_id, -1),
        mask=out_mask,
    )
    if RETURN_VALUE:
        hi = (sp >> 32).to(tl.int32)
        orig_bits = hi ^ ((hi >> 31) & 0x7FFFFFFF)
        out_val = orig_bits.to(tl.float32, bitcast=True)
        out_val = tl.where(valid_out, out_val, -float("inf")).to(
            OutValues.dtype.element_ty
        )
        tl.store(OutValues + row_out + cols, out_val, mask=out_mask)
    if SPARSE_COUNT > BLOCK:
        for base in tl.static_range(BLOCK, SPARSE_COUNT, 512):
            pos = base + tl.arange(0, 512)
            pmask = pos < SPARSE_COUNT
            tl.store(OutIndices + row_out + pos, -1, mask=pmask)
            if RETURN_VALUE:
                neg_inf = tl.full([512], -float("inf"), dtype=tl.float32).to(
                    OutValues.dtype.element_ty
                )
                tl.store(OutValues + row_out + pos, neg_inf, mask=pmask)


def _hygon_lightning_indexer_impl(
    query,
    key,
    weights,
    ends,
    key_meta,
    block_table,
    *,
    layout_key="PA_BSND",
    sparse_count=512,
    sparse_mode=3,
    return_value=False,
    use_tle,
    active_max,
    host_lengths=None,
):
    t, h, d = query.shape
    out = torch.empty((t, 1, sparse_count), dtype=torch.int32, device=query.device)
    values = (
        torch.empty((t, 1, sparse_count), dtype=query.dtype, device=query.device)
        if return_value
        else torch.empty((0,), dtype=query.dtype, device=query.device)
    )
    if not t:
        return out, values
    b = key_meta.numel()
    max_keys = active_max
    if not max_keys:
        fill_empty_outputs(out, values, return_value)
        return out, values

    is_paged = layout_key == "PA_BSND"
    if not return_value and max_keys <= sparse_count:
        chunk = min(512, max(16, triton.next_power_of_2(sparse_count)))
        _hygon_direct_indices[(t,)](
            ends,
            key_meta,
            out,
            b,
            SPARSE_COUNT=sparse_count,
            SPARSE_MODE=sparse_mode,
            IS_PAGED=is_paged,
            CHUNK=chunk,
        )
        return out, values

    if use_tle:
        _init_hcu_tle()
    tile_k = 128
    k_steps = 8 if (t >= 32 and max_keys >= 1024) else (4 if t >= 32 else 1)
    k_groups = triton.cdiv(max_keys, tile_k * k_steps)
    if k_groups > 65535 or t * max_keys >= 2**31:
        raise NotImplementedError("score workspace exceeds supported launch limits")

    scores = torch.empty((t, max_keys), dtype=torch.float32, device=query.device)
    page_size = key.shape[1] if is_paged else 1
    table_width = block_table.shape[1] if is_paged else 1
    bt_ptr = block_table if is_paged else ends
    bh = max(16, triton.next_power_of_2(h))
    bd = max(32, triton.next_power_of_2(d))
    _paged_scores[(t, k_groups)](
        query,
        key,
        weights,
        ends,
        key_meta,
        bt_ptr,
        scores,
        h,
        d,
        page_size,
        table_width,
        max_keys,
        b,
        bh,
        bd,
        tile_k,
        k_steps,
        SPARSE_COUNT=sparse_count,
        SPARSE_MODE=sparse_mode,
        IS_PAGED=is_paged,
        SKIP_SHORT_ROWS=(not return_value),
        USE_TLE=use_tle,
        num_warps=4,
    )

    if not return_value:
        hist = torch.empty((t, 4096), dtype=torch.int32, device=query.device)
        radix_chunk = 4096 if max_keys >= 16384 else 2048
        _hygon_radix_topk[(t,)](
            scores,
            ends,
            key_meta,
            hist,
            out,
            max_keys,
            b,
            TOPK=sparse_count,
            SPARSE_MODE=sparse_mode,
            IS_PAGED=is_paged,
            CHUNK=radix_chunk,
            num_warps=8,
        )
        return out, values

    if max_keys <= 8192:
        block = max(64, triton.next_power_of_2(max_keys))
        nw = 4 if block <= 2048 else 8
        _hygon_bitonic_topk[(t,)](
            scores,
            ends,
            key_meta,
            out,
            values,
            max_keys,
            b,
            BLOCK=block,
            SPARSE_COUNT=sparse_count,
            SPARSE_MODE=sparse_mode,
            IS_PAGED=is_paged,
            RETURN_VALUE=True,
            num_warps=nw,
        )
        return out, values

    lengths = (
        key_meta
        if is_paged
        else torch.tensor(host_lengths, dtype=torch.int32, device=query.device)
    )
    starts = torch.empty((t,), dtype=torch.int32, device=query.device)
    row_ends = torch.empty((t,), dtype=torch.int32, device=query.device)
    compute_row_bounds(ends, lengths, starts, row_ends, b, max_keys, sparse_mode)
    select_topk_sorted(
        scores,
        row_ends,
        out,
        values,
        sparse_count=sparse_count,
        return_value=True,
    )
    return out, values


def lightning_indexer(
    query,
    key,
    weights,
    *,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=512,
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    return_value=False,
):
    """Hygon v0/v1/v2 TND LightningIndexer entry registered in backend._hygon.ops."""
    (
        version,
        query,
        key,
        weights,
        ends,
        key_meta,
        block_table,
        active_max,
        _host_ends,
        host_lengths,
    ) = _validate_lightning_indexer_inputs(
        query,
        key,
        weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        return_value=return_value,
    )
    with torch.cuda.device(query.device):
        return _hygon_lightning_indexer_impl(
            query,
            key,
            weights,
            ends,
            key_meta,
            block_table,
            layout_key=layout_key,
            sparse_count=sparse_count,
            sparse_mode=sparse_mode,
            return_value=return_value,
            use_tle=(version != "v0"),
            active_max=active_max,
            host_lengths=host_lengths,
        )
