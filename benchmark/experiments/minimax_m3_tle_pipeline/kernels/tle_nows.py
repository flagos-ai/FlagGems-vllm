# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Single-partition full-row TMA ring; separate real depth from launch stages."""

import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def _prefetch(
    desc,
    bt_row,
    stride_bt_k,
    visitlist,
    metadata,
    raw,
    full,
    iteration,
    STAGES: tl.constexpr,
):
    blk = tl.load(tle.gpu.local_ptr(visitlist, (0, iteration)))
    slot = iteration % STAGES
    page = tl.load(bt_row + blk * stride_bt_k)
    tl.store(tle.gpu.local_ptr(metadata.slot(slot)), tl.full((1, 1), blk, tl.int32))
    tle.gpu.copy(desc, raw.slot(slot), [128, 256], [page * 128, 0], barrier=full[slot])


@triton.jit
def _dispatch_nows(
    q,
    desc,
    bt_row,
    stride_bt_k,
    bm0,
    bm1,
    bm2,
    bm3,
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
    o_ptr,
    row_off,
    row_ok,
    stride_qn,
    v_scale_ptr,
    M: tl.constexpr,
    BQ: tl.constexpr,
    STAGES: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    SPARSE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    off = tl.arange(0, 128)
    groups = tl.where(off < 32, active0, active1)
    groups = tl.where((off >= 64) & (off < 96), active2, groups)
    groups = tl.where(off >= 96, active3, groups)
    valid = off < span
    if SPARSE:
        valid = valid & (((groups >> (off & 31)) & 1) != 0)
    ranks = tl.cumsum(valid.to(tl.int32), 0) - 1
    visitlist = tle.gpu.alloc(
        [1, 128], dtype=tl.int32, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    tl.store(
        tle.gpu.local_ptr(
            visitlist, (tl.zeros((128,), tl.int32), tl.maximum(0, ranks))
        ),
        lo_s + off,
        mask=valid,
    )
    tl.debug_barrier()
    raw = tle.gpu.alloc([STAGES, 128, 256], dtype=DTYPE, scope=tle.gpu.smem)
    metadata = tle.gpu.alloc(
        [STAGES, 1, 1], dtype=tl.int32, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    scratch = tle.gpu.alloc([1, 128, 128], dtype=tl.bfloat16, scope=tle.gpu.smem)
    BYTES: tl.constexpr = 128 * 256 * (DTYPE.primitive_bitwidth // 8)
    full = tle.gpu.alloc_barriers(triton.next_power_of_2(STAGES), expect_bytes=BYTES)
    off_n = tl.arange(0, 128)
    off_d = tl.arange(0, 128)
    m_i = tl.full((M,), float("-inf"), tl.float32)
    l_i = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, 128), tl.float32)
    # One slot remains free for prefetch while PV is outstanding. A slot is
    # reused only after the preceding iteration's WGMMA has finished.
    for i in range(tl.minimum(STAGES - 1, visits)):
        _prefetch(desc, bt_row, stride_bt_k, visitlist, metadata, raw, full, i, STAGES)
    for iteration in range(visits):
        slot = iteration % STAGES
        tle.gpu.barrier_wait(full[slot], phaseIdx=iteration // STAGES)
        blk = tl.sum(tl.sum(tl.load(tle.gpu.local_ptr(metadata.slot(slot))), 0), 0)
        offset = blk - lo_s
        bm = tl.where(offset < 32, bm0, bm1)
        bm = tl.where((offset >= 64) & (offset < 96), bm2, bm)
        bm = tl.where(offset >= 96, bm3, bm)
        sel = ((bm >> (offset & 31)) & 1) != 0
        sel = tl.reshape(tl.broadcast_to(sel[:, None], (BQ, 8)), (M,))
        pos = blk * 128 + off_n
        pos_ok = pos < seq_len
        k = tl.load(tle.gpu.local_ptr(raw.slot(slot).subslice(0, 128, dim=1)))
        k = tl.where(pos_ok[:, None], k, 0.0).to(tl.bfloat16)
        tl.store(tle.gpu.local_ptr(scratch.slot(0)), k)
        qk_async = tle.gpu.wgmma(q, scratch.slot(0), trans_b=True, out_dtype=tl.float32)
        qk = tle.gpu.wgmma_wait(0, qk_async) * qk_scale
        keep = sel[:, None] & (q_abs[:, None] >= pos[None, :]) & pos_ok[None, :]
        qk = tl.where(keep, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(tle.gpu.local_ptr(raw.slot(slot).subslice(128, 128, dim=1)))
        v = tl.where(pos_ok[:, None], v, 0.0).to(tl.bfloat16)
        tl.store(tle.gpu.local_ptr(scratch.slot(0)), v)
        pv_async = tle.gpu.wgmma(
            p.to(tl.bfloat16),
            scratch.slot(0),
            acc=acc * alpha[:, None],
            out_dtype=tl.float32,
        )
        future = iteration + STAGES - 1
        if future < visits:
            _prefetch(
                desc,
                bt_row,
                stride_bt_k,
                visitlist,
                metadata,
                raw,
                full,
                future,
                STAGES,
            )
        acc = tle.gpu.wgmma_wait(0, pv_async)
        m_i = m_new
    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    if HAS_SCALE:
        inv_l = inv_l * tl.load(v_scale_ptr)
    tl.store(
        o_ptr + row_off[:, None] * stride_qn + off_d[None, :],
        (acc * inv_l[:, None]).to(tl.bfloat16),
        mask=row_ok[:, None],
    )
