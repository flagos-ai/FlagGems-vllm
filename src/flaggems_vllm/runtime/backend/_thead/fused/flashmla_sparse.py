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

"""Sparse MLA prefill for PPU, with the generic operator's ABI and semantics.

Persistent CTAs distribute query heads and TOPK splits over matrix-capable SMs.
An eight-warp QK group gathers KV blocks two positions ahead, computes scores,
and hands each block to an eight-warp PV group for online softmax and P @ V.
A combine kernel merges TOPK splits when query parallelism is low.

The default KV ring uses cp.async inline assembly. The opt-in tle.pipe variant
uses compiler-managed stage ownership for the ring. Each ready/free barrier
generation is consumed exactly once. Without PPU TLE warp specialization, dispatch
uses the original PPU Triton implementation.
"""

import functools
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle():
    try:
        import triton.experimental.tle as _tle_api
        import triton.experimental.tle.language as tle
    except ImportError:
        _tle_api = tle = None
else:
    _tle_api = tle = None

LOG2E = 1.4426950408889634
# Initial running max (log2 units): finite, so fully masked blocks give no NaN in
# exp2(m_old - m_new) or exp2(s - m_new) without extra selects.
M_FLOOR = -1.0e30
BLOCK_K = 64  # keys per block
# Lazy rescaling threshold (log2 units): P stays below 2^THR between rescales.
RESCALE_THR = 2.0

# SMs that run matrix-instruction kernels, by device model (others: all SMs).
_MMA_SM_COUNT = {"PPU-ZW810E": 20}

# Latency model fitted on PPU-ZW810E (us), by (DQK, heads per item): cost of an item
# in a CTA's stream, of a key block, and of filling / draining a CTA's stream. The
# combine of a TOPK split costs a fixed part plus a part per partial row.
_ITEM_US = {
    (512, 64): 3.1,
    (512, 32): 1.9,
    (512, 16): 1.2,
    (576, 64): 3.5,
    (576, 32): 2.4,
    (576, 16): 1.3,
}
_BLOCK_US = {
    (512, 64): 2.09,
    (512, 32): 1.51,
    (512, 16): 1.33,
    (576, 64): 2.25,
    (576, 32): 1.61,
    (576, 16): 1.45,
}
_FILL_US = {
    (512, 64): 4.3,
    (512, 32): 4.0,
    (512, 16): 3.9,
    (576, 64): 4.7,
    (576, 32): 4.3,
    (576, 16): 4.2,
}
_COMBINE_US = 3.5
_COMBINE_ROW_US = 0.046 / 64  # 0.046 us per 64 partial rows
# The combine kernel unrolls over the splits.
_MAX_SPLIT = 16
# Heads per program of the combine kernel.
_COMBINE_HEADS = 8
# Warps of each of the two warp groups.
_GROUP_WARPS = 8

_TRITON_BLOCK_K = 128
_TRITON_ITEM_US = {64: 10.6, 32: 8.0, 16: 6.5}
_TRITON_BLOCK_US = {64: 7.9, 32: 5.55, 16: 4.1}
_TRITON_COMBINE_US = 3.0

_TLE_PRIMITIVES = (
    "gpu.alloc",
    "gpu.local_ptr",
    "gpu.warp_specialize",
    "gpu.alloc_barriers",
    "gpu.barrier_wait",
    "gpu.barrier_arrive",
    "gpu.buffered_tensor.slot",
    "gpu.buffered_tensor.subslice",
)
_TLE_PIPE_PRIMITIVES = _TLE_PRIMITIVES + (
    "pipe",
    "pipe.writer",
    "pipe.writer.acquire",
    "pipe.writer.commit",
    "pipe.reader",
    "pipe.reader.wait",
    "pipe.reader.release",
    "gpu.copy",
)


@triton.jit
def _ppu_sparse_mla_fwd_kernel(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale_log2,
    output,
    max_logits,
    lse,
    o_part,
    ml_part,
    stride_qm,
    stride_qh,
    stride_kvn,
    stride_tm,
    stride_om,
    stride_oh,
    stride_mm,
    stride_lm,
    SKV,
    TOPK,
    nsplit,
    chunk,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    PARTIAL: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    M_FLOOR: tl.constexpr,
):
    """One work item: BH heads of one query token over `chunk` key blocks."""
    NHB: tl.constexpr = HQ // BH
    BD: tl.constexpr = 256
    LN2: tl.constexpr = 0.6931471805599453
    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BD)
    offs_t = tl.arange(0, BK)

    item = tl.program_id(0)
    hb = item % NHB
    ts = item // NHB
    split = ts % nsplit
    tok = (ts // nsplit).to(tl.int64)
    h0 = hb * BH

    q_ptr = q + tok * stride_qm + (h0 + offs_h)[:, None] * stride_qh
    q0 = tl.load(q_ptr + offs_d[None, :])
    q1 = tl.load(q_ptr + BD + offs_d[None, :])
    if DQK == 576:
        offs_td = tl.arange(0, 64)
        qt = tl.load(q_ptr + 2 * BD + offs_td[None, :])

    if HAVE_TOPK_LENGTH:
        topk_len = tl.minimum(tl.maximum(tl.load(topk_length + tok), 0), TOPK)
    else:
        topk_len = TOPK
    lo = split * chunk
    hi = tl.minimum(lo + chunk, tl.cdiv(topk_len, BK))

    # online softmax in log2 units
    m_i = tl.full([BH], M_FLOOR, dtype=tl.float32)
    l_i = tl.zeros([BH], dtype=tl.float32)
    acc0 = tl.zeros([BH, BD], dtype=tl.float32)
    acc1 = tl.zeros([BH, BD], dtype=tl.float32)
    t_base = indices + tok * stride_tm
    for ck in range(lo, hi):
        t_offs = ck * BK + offs_t
        ids = tl.load(t_base + t_offs, mask=t_offs < topk_len, other=-1)
        valid = (ids >= 0) & (ids < SKV)
        ids = tl.where(valid, ids, 0)
        kv_ptr = kv + ids[:, None].to(tl.int64) * stride_kvn
        k0 = tl.load(kv_ptr + offs_d[None, :])
        k1 = tl.load(kv_ptr + BD + offs_d[None, :])
        s = tl.dot(q0, tl.trans(k0))
        s = tl.dot(q1, tl.trans(k1), s)
        if DQK == 576:
            kt = tl.load(kv_ptr + 2 * BD + offs_td[None, :])
            s = tl.dot(qt, tl.trans(kt), s)
        s = tl.where(valid[None, :], s * sm_scale_log2, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pb = p.to(tl.bfloat16)
        acc0 = tl.dot(pb, k0, acc0 * alpha[:, None])
        acc1 = tl.dot(pb, k1, acc1 * alpha[:, None])
        m_i = m_new

    row_valid = l_i > 0
    inv_l = tl.where(row_valid, 1.0 / l_i, 0.0)
    if PARTIAL:
        # partial of (token, split): normalized O, (max, sum) in log2 units
        prow = (tok * nsplit + split) * HQ + h0 + offs_h
        op_ptr = o_part + prow[:, None] * (2 * BD)
        tl.store(op_ptr + offs_d[None, :], acc0 * inv_l[:, None])
        tl.store(op_ptr + BD + offs_d[None, :], acc1 * inv_l[:, None])
        tl.store(ml_part + prow * 2, m_i)
        tl.store(ml_part + prow * 2 + 1, l_i)
    else:
        lse_i = (m_i + tl.math.log2(l_i)) * LN2
        factor = inv_l
        if HAVE_ATTN_SINK:
            sink = tl.load(attn_sink + h0 + offs_h)
            # no valid key: lse_i = -inf, sink - lse_i is NaN for sink = -inf
            factor = tl.where(
                row_valid, factor / (1.0 + tl.math.exp(sink - lse_i)), 0.0
            )
        tl.store(
            max_logits + tok * stride_mm + h0 + offs_h,
            tl.where(row_valid, m_i * LN2, float("-inf")),
        )
        tl.store(
            lse + tok * stride_lm + h0 + offs_h,
            tl.where(row_valid, lse_i, float("inf")),
        )
        o_ptr = output + tok * stride_om + (h0 + offs_h)[:, None] * stride_oh
        tl.store(o_ptr + offs_d[None, :], (acc0 * factor[:, None]).to(tl.bfloat16))
        tl.store(o_ptr + BD + offs_d[None, :], (acc1 * factor[:, None]).to(tl.bfloat16))


_FWD_KERNELS = {w: libentry()(_ppu_sparse_mla_fwd_kernel) for w in (8, 16)}


@libentry()
@triton.jit
def _ppu_sparse_mla_combine_kernel(
    o_part,
    ml_part,
    attn_sink,
    output,
    max_logits,
    lse,
    stride_om,
    stride_oh,
    stride_mm,
    stride_lm,
    HQ: tl.constexpr,
    DV: tl.constexpr,
    NSPLIT: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    BHC: tl.constexpr,
):
    """Merge the NSPLIT partials of BHC heads of one query token."""
    NHG: tl.constexpr = HQ // BHC
    LN2: tl.constexpr = 0.6931471805599453
    pid = tl.program_id(0)
    tok = (pid // NHG).to(tl.int64)
    offs_h = (pid % NHG) * BHC + tl.arange(0, BHC)
    offs_d = tl.arange(0, DV)

    m_max = tl.full([BHC], float("-inf"), dtype=tl.float32)
    for s in tl.static_range(NSPLIT):
        prow = (tok * NSPLIT + s) * HQ + offs_h
        l_s = tl.load(ml_part + prow * 2 + 1)
        m_s = tl.load(ml_part + prow * 2)
        m_max = tl.maximum(m_max, tl.where(l_s > 0, m_s, float("-inf")))
    m_ref = tl.where(m_max == float("-inf"), 0.0, m_max)

    total = tl.zeros([BHC], dtype=tl.float32)
    acc = tl.zeros([BHC, DV], dtype=tl.float32)
    for s in tl.static_range(NSPLIT):
        prow = (tok * NSPLIT + s) * HQ + offs_h
        l_s = tl.load(ml_part + prow * 2 + 1)
        m_s = tl.load(ml_part + prow * 2)
        w = tl.where(l_s > 0, l_s * tl.math.exp2(m_s - m_ref), 0.0)
        total += w
        acc += w[:, None] * tl.load(o_part + prow[:, None] * DV + offs_d[None, :])

    row_valid = total > 0
    lse_i = (m_ref + tl.math.log2(total)) * LN2
    factor = tl.where(row_valid, 1.0 / total, 0.0)
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink + offs_h)
        factor = tl.where(row_valid, factor / (1.0 + tl.math.exp(sink - lse_i)), 0.0)
    tl.store(
        max_logits + tok * stride_mm + offs_h,
        tl.where(row_valid, m_max * LN2, float("-inf")),
    )
    tl.store(lse + tok * stride_lm + offs_h, tl.where(row_valid, lse_i, float("inf")))
    tl.store(
        output + tok * stride_om + offs_h[:, None] * stride_oh + offs_d[None, :],
        (acc * factor[:, None]).to(tl.bfloat16),
    )


if tle is not None:

    @triton.constexpr_function
    def _ring_layout(vec, max_phase):
        # rank-3 ring [slot, rows, cols], swizzled row-major slots
        return tle.gpu.swizzled_shared_layout(
            vec, 1, max_phase, [2, 1, 0], [1] * 3, [1] * 3, [1] * 3, [2, 1, 0]
        )

    @triton.constexpr_function
    def _tile_layout(vec, max_phase):
        return tle.gpu.swizzled_shared_layout(
            vec, 1, max_phase, [1, 0], [1] * 2, [1] * 2, [1] * 2, [1, 0]
        )

    # Barrier slots are compile-time indices: select one of two by a runtime index.
    @triton.jit
    def _wait_one_of_two(bars, i):
        if i == 0:
            tle.gpu.barrier_wait(bars[0])
        else:
            tle.gpu.barrier_wait(bars[1])

    @triton.jit
    def _arrive_one_of_two(bars, i):
        if i == 0:
            tle.gpu.barrier_arrive(bars[0])
        else:
            tle.gpu.barrier_arrive(bars[1])

    @triton.jit
    def _wait_barrier(bar):
        tle.gpu.barrier_wait(bar[0])

    # Asynchronous copies. The KV ring is filled two positions ahead of its use, so
    # the copies are issued as cp.async (masked-off rows are zero-filled) and waited
    # for by commit group, a few steps after they are issued. The compiler does not
    # see these shared-memory writes: each wait is followed by a barrier of the QK
    # group, and a slot is refilled only after a barrier that follows its last read.
    @triton.constexpr_function
    def _cp_async_asm_text(nbytes, zfill):
        # with zfill the source size $3 is nbytes or 0, which zero-fills
        cache = "cg" if nbytes == 16 else "ca"
        size = ", $3" if zfill else ""
        return (
            f"cp.async.{cache}.shared.global [ $1 + 0 ], [ $2 + 0 ], {nbytes}{size}; "
            "mov.u32 $0, 0;"
        )

    @triton.jit
    def _cp_async(dst, src, src_bytes, NBYTES: tl.constexpr, ZFILL: tl.constexpr):
        """Copy NBYTES (4 or 16) from global `src` to shared `dst` per element; with
        ZFILL, only the first `src_bytes` (NBYTES or 0) are read, the rest is
        zero-filled. `src` stays a pointer, so the code generator can keep a
        warp-uniform row base in scalar registers."""
        # shared addresses are 32-bit
        dst_addr = dst.to(tl.int64).to(tl.int32)
        if ZFILL:
            tl.inline_asm_elementwise(
                _cp_async_asm_text(NBYTES, True),
                "=r,r,l,r",
                [dst_addr, src, src_bytes],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        else:
            tl.inline_asm_elementwise(
                _cp_async_asm_text(NBYTES, False),
                "=r,r,l",
                [dst_addr, src],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )

    @triton.jit
    def _cp_async_ctrl(asm: tl.constexpr):
        """A cp.async control instruction (commit or wait) of each thread."""
        tl.inline_asm_elementwise(
            asm + " mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )

    @triton.jit
    def _commit_copies():
        """Close the copies issued since the last commit into one group."""
        _cp_async_ctrl("cp.async.commit_group;")

    @triton.jit
    def _drain_copies():
        """Wait for all copy groups of this thread."""
        _cp_async_ctrl("cp.async.wait_group 0;")

    @triton.jit
    def _wait_copies(n, qkbar):
        """Wait for all copy groups of this thread but the newest n (0 .. 2), then
        make the copies of the whole QK group visible to it. Each _gather_* call and
        each index copy is one group."""
        if n == 2:
            _cp_async_ctrl("cp.async.wait_group 2;")
        elif n == 1:
            _cp_async_ctrl("cp.async.wait_group 1;")
        else:
            _cp_async_ctrl("cp.async.wait_group 0;")
        _wait_barrier(qkbar)

    @triton.jit
    def _copy_tile(
        dst,
        src_rows,
        col0,
        valid,
        R: tl.constexpr,
        C: tl.constexpr,
        ZFILL: tl.constexpr,
    ):
        """Issue the copy of the R x C bf16 tile at columns col0 .. col0 + C of the
        rows `src_rows` (pointers) to the shared tile `dst`, in 16-byte chunks; with
        ZFILL, rows that are not `valid` are zero-filled."""
        rows = tl.broadcast_to(tl.arange(0, R)[:, None], (R, C // 8))
        cols = tl.broadcast_to((tl.arange(0, C // 8) * 8)[None, :], (R, C // 8))
        dst_ptrs = tle.gpu.local_ptr(dst, (rows, cols))
        src_ptrs = src_rows[:, None] + col0 + cols
        if ZFILL:
            src_bytes = tl.where(valid, 16, 0)[:, None]
            _cp_async(
                dst_ptrs,
                src_ptrs,
                tl.broadcast_to(src_bytes, (R, C // 8)),
                16,
                True,
            )
        else:
            _cp_async(dst_ptrs, src_ptrs, None, 16, False)

    @triton.jit
    def _copy_ids(dst, src, valid, BK: tl.constexpr):
        """Issue the copy of BK key indices to the shared index slot `dst` (0 where
        not `valid`) as one commit group."""
        _cp_async(
            tle.gpu.local_ptr(dst, (tl.arange(0, BK),)),
            src,
            tl.where(valid, 4, 0),
            4,
            True,
        )
        _commit_copies()

    @triton.jit
    def _q_rows(ring, slot, BH: tl.constexpr, BK: tl.constexpr):
        """The first BH rows of a ring slot of BK rows (an item's Q rows)."""
        if BH == BK:
            return ring.slot(slot)
        else:
            return ring.slot(slot).subslice(0, BH, 0)

    @triton.jit
    def _valid(ids, SKV):
        return (ids >= 0) & (ids < SKV)

    @triton.jit
    def _item(
        k,
        pid,
        grid,
        n_my,
        topk_length,
        TOPK,
        nsplit,
        chunk,
        NHB: tl.constexpr,
        BH: tl.constexpr,
        BK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
    ):
        """(token, first head, split, first key, key count, block count >= 1) of the
        CTA's k-th work item. An item without keys still gets one (all invalid)
        block, so every item has a Q position and at least one block."""
        it = pid + tl.minimum(k, n_my - 1) * grid
        h0 = (it % NHB) * BH
        ts = it // NHB
        split = ts % nsplit
        tok = (ts // nsplit).to(tl.int64)
        lo = split * chunk * BK
        if HAVE_TOPK_LENGTH:
            tlen = tl.minimum(tl.maximum(tl.load(topk_length + tok), 0), TOPK)
        else:
            tlen = TOPK
        ln = tl.minimum(tl.maximum(tlen - lo, 0), chunk * BK)
        return tok, h0, split, lo, ln, tl.maximum(tl.cdiv(ln, BK), 1)

    @triton.jit
    def _load_ids(t_row, d, ln, BK: tl.constexpr):
        """Indices of block d of an item (-1 past its ln keys)."""
        t_offs = d * BK + tl.arange(0, BK)
        return tl.load(t_row + t_offs, mask=t_offs < ln, other=-1)

    @triton.jit
    def _ring_ids(sIdx, d, ln, BK: tl.constexpr, NIDX: tl.constexpr):
        """Indices of block d of the item from its index slot (-1 past its ln keys),
        read where used, in the layout of the use."""
        t_offs = d * BK + tl.arange(0, BK)
        ids = tl.load(tle.gpu.local_ptr(sIdx.slot(d % NIDX)))
        return tl.where(t_offs < ln, ids, -1)

    @triton.jit
    def _gather_q(
        q,
        tok,
        h0,
        sK0,
        sK1,
        sKT,
        p,
        stride_qm,
        stride_qh,
        BH: tl.constexpr,
        BK: tl.constexpr,
        BD: tl.constexpr,
        NKV: tl.constexpr,
        TD: tl.constexpr,
    ):
        """Issue (do not wait for) the copy of an item's Q rows to position p, as one
        commit group."""
        q_rows = q + tok * stride_qm + (h0 + tl.arange(0, BH)) * stride_qh
        _copy_tile(_q_rows(sK0, p % NKV, BH, BK), q_rows, 0, None, BH, BD, False)
        _copy_tile(_q_rows(sK1, p % NKV, BH, BK), q_rows, BD, None, BH, BD, False)
        if TD > 0:
            _copy_tile(_q_rows(sKT, p % 2, BH, BK), q_rows, 2 * BD, None, BH, TD, False)
        _commit_copies()

    @triton.jit
    def _gather_kv(
        kv,
        ids,
        SKV,
        sK0,
        sK1,
        sKT,
        p,
        stride_kvn,
        BK: tl.constexpr,
        BD: tl.constexpr,
        NKV: tl.constexpr,
        TD: tl.constexpr,
        KV_I64: tl.constexpr,
    ):
        """Issue (do not wait for) the copy of the key block with indices `ids` to
        position p, as one commit group; invalid keys are zero-filled."""
        valid = _valid(ids, SKV)
        row_ids = tl.where(valid, ids, 0)
        if KV_I64:
            rows = kv + row_ids.to(tl.int64) * stride_kvn
        else:
            rows = kv + row_ids * stride_kvn
        _copy_tile(sK0.slot(p % NKV), rows, 0, valid, BK, BD, True)
        _copy_tile(sK1.slot(p % NKV), rows, BD, valid, BK, BD, True)
        if TD > 0:
            _copy_tile(sKT.slot(p % 2), rows, 2 * BD, valid, BK, TD, True)
        _commit_copies()

    @triton.jit
    def _scores(
        q0,
        q1,
        sQT,
        ids,
        sK0,
        sK1,
        sKT,
        p,
        sm_scale_log2,
        SKV,
        NKV: tl.constexpr,
        TD: tl.constexpr,
    ):
        """Scores (log2 units, invalid keys -inf) of the key block at position p."""
        slot = p % NKV
        k0 = tl.load(tle.gpu.local_ptr(sK0.slot(slot)))
        k1 = tl.load(tle.gpu.local_ptr(sK1.slot(slot)))
        if TD > 0:
            # the short tail dot first: its operand loads overlap the long dots
            kt = tl.load(tle.gpu.local_ptr(sKT.slot(p % 2)))
            s = tl.dot(tl.load(tle.gpu.local_ptr(sQT)), tl.trans(kt))
            s = tl.dot(q0, tl.trans(k0), s)
        else:
            s = tl.dot(q0, tl.trans(k0))
        s = tl.dot(q1, tl.trans(k1), s)
        return tl.where(_valid(ids, SKV)[None, :], s * sm_scale_log2, float("-inf"))

    @triton.jit
    def _qk_group(
        q,
        kv,
        indices,
        topk_length,
        sK0,
        sK1,
        sKT,
        sQT,
        sS,
        sIdx,
        ready,
        free,
        qkbar,
        sfree,
        pid,
        grid,
        n_my,
        TOPK,
        nsplit,
        chunk,
        SKV,
        sm_scale_log2,
        stride_qm,
        stride_qh,
        stride_kvn,
        stride_tm,
        NHB: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        BH: tl.constexpr,
        BK: tl.constexpr,
        BD: tl.constexpr,
        NKV: tl.constexpr,
        TD: tl.constexpr,
        NIDX: tl.constexpr,
        NS: tl.constexpr,
        KV_I64: tl.constexpr,
    ):
        """Step p computes position p and then gathers position p + 2, once position
        p - 1 is consumed (its KV slot; the K-tail slot is that of position p, read by
        every QK thread by then): block g + 2 of the item, past its end the next
        item's Q and block 0. The indices of block d of an item live in index slot
        d % NIDX: blocks 0 .. 3 are loaded into registers while the previous item runs
        and stored at the item's Q step, block d >= 4 is copied at step d - 5. A step
        waits for every copy but those of the previous step; _wait_copies then
        synchronizes the QK group, which makes them visible to it."""
        # the indices of blocks 0 .. 3 of the current item (a*) and of the next one
        # (n_a*) are loaded into registers ahead; later blocks go through sIdx
        tok, h0, _, lo, ln, nblk = _item(
            0,
            pid,
            grid,
            n_my,
            topk_length,
            TOPK,
            nsplit,
            chunk,
            NHB,
            BH,
            BK,
            HAVE_TOPK_LENGTH,
        )
        t_row = indices + tok * stride_tm + lo
        a0 = _load_ids(t_row, 0, ln, BK)
        a1 = _load_ids(t_row, 1, ln, BK)
        a2 = _load_ids(t_row, 2, ln, BK)
        a3 = _load_ids(t_row, 3, ln, BK)
        # positions 0 (Q of item 0) and 1 (its block 0)
        _gather_q(
            q, tok, h0, sK0, sK1, sKT, 0, stride_qm, stride_qh, BH, BK, BD, NKV, TD
        )
        _gather_kv(kv, a0, SKV, sK0, sK1, sKT, 1, stride_kvn, BK, BD, NKV, TD, KV_I64)
        issued = 1
        # loop-carried Q registers: placeholders until the first Q position
        q0 = tl.load(tle.gpu.local_ptr(_q_rows(sK0, 2, BH, BK)))
        q1 = tl.load(tle.gpu.local_ptr(_q_rows(sK1, 2, BH, BK)))
        p = 0
        r = 0
        prev_blk = False
        for k in range(n_my):
            # the next item: its metadata and the indices of its blocks 0 .. 3
            n_tok, n_h0, _, n_lo, n_ln, n_nblk = _item(
                k + 1,
                pid,
                grid,
                n_my,
                topk_length,
                TOPK,
                nsplit,
                chunk,
                NHB,
                BH,
                BK,
                HAVE_TOPK_LENGTH,
            )
            has_next = k + 1 < n_my
            n_row = indices + n_tok * stride_tm + n_lo
            n_len = tl.where(has_next, n_ln, 0)
            n_a0 = _load_ids(n_row, 0, n_len, BK)
            n_a1 = _load_ids(n_row, 1, n_len, BK)
            n_a2 = _load_ids(n_row, 2, n_len, BK)
            n_a3 = _load_ids(n_row, 3, n_len, BK)
            # the Q position
            _wait_copies(issued, qkbar)
            q0 = tl.load(tle.gpu.local_ptr(_q_rows(sK0, p % NKV, BH, BK)))
            q1 = tl.load(tle.gpu.local_ptr(_q_rows(sK1, p % NKV, BH, BK)))
            if TD > 0:
                # the Q tail stays in shared memory (held in registers it costs the
                # dots their loads in flight)
                tl.store(
                    tle.gpu.local_ptr(sQT),
                    tl.load(tle.gpu.local_ptr(_q_rows(sKT, p % 2, BH, BK))),
                )
            tl.store(tle.gpu.local_ptr(sIdx.slot(0)), a0)
            tl.store(tle.gpu.local_ptr(sIdx.slot(1)), a1)
            tl.store(tle.gpu.local_ptr(sIdx.slot(2)), a2)
            tl.store(tle.gpu.local_ptr(sIdx.slot(3)), a3)
            if prev_blk:
                _wait_one_of_two(free, (r - 1) % 2)
            else:
                _wait_barrier(qkbar)
            prev_blk = False
            issued = 0
            if nblk > 1:
                _gather_kv(
                    kv,
                    a1,
                    SKV,
                    sK0,
                    sK1,
                    sKT,
                    p + 2,
                    stride_kvn,
                    BK,
                    BD,
                    NKV,
                    TD,
                    KV_I64,
                )
                issued += 1
            elif has_next:
                _gather_q(
                    q,
                    n_tok,
                    n_h0,
                    sK0,
                    sK1,
                    sKT,
                    p + 2,
                    stride_qm,
                    stride_qh,
                    BH,
                    BK,
                    BD,
                    NKV,
                    TD,
                )
                issued += 1
            if 4 < nblk:
                t_offs = 4 * BK + tl.arange(0, BK)
                _copy_ids(sIdx.slot(4 % NIDX), t_row + t_offs, t_offs < ln, BK)
                issued += 1
            p += 1
            for g in range(nblk):
                _wait_copies(issued, qkbar)
                # the indices this step gathers, read before the dots so that the read
                # overlaps them
                ids_g = _ring_ids(sIdx, g + 2, ln, BK, NIDX)
                s = _scores(
                    q0,
                    q1,
                    sQT,
                    _ring_ids(sIdx, g, ln, BK, NIDX),
                    sK0,
                    sK1,
                    sKT,
                    p,
                    sm_scale_log2,
                    SKV,
                    NKV,
                    TD,
                )
                if NS == 1:
                    if r >= 1:
                        _wait_barrier(sfree)  # the PV group read S(r - 1)
                tl.store(tle.gpu.local_ptr(sS.slot(r % NS)), s)
                _arrive_one_of_two(ready, r % 2)
                # position p - 1 consumed: a key block by the PV group (every FREE
                # arrival is waited for once, in order), the Q position by every QK
                # thread
                if prev_blk:
                    _wait_one_of_two(free, (r - 1) % 2)
                else:
                    _wait_barrier(qkbar)
                prev_blk = True
                r += 1
                d = g + 2
                issued = 0
                if d < nblk:
                    _gather_kv(
                        kv,
                        ids_g,
                        SKV,
                        sK0,
                        sK1,
                        sKT,
                        p + 2,
                        stride_kvn,
                        BK,
                        BD,
                        NKV,
                        TD,
                        KV_I64,
                    )
                    issued += 1
                elif has_next:
                    if d == nblk:
                        _gather_q(
                            q,
                            n_tok,
                            n_h0,
                            sK0,
                            sK1,
                            sKT,
                            p + 2,
                            stride_qm,
                            stride_qh,
                            BH,
                            BK,
                            BD,
                            NKV,
                            TD,
                        )
                    else:
                        _gather_kv(
                            kv,
                            n_a0,
                            SKV,
                            sK0,
                            sK1,
                            sKT,
                            p + 2,
                            stride_kvn,
                            BK,
                            BD,
                            NKV,
                            TD,
                            KV_I64,
                        )
                    issued += 1
                if g + 5 < nblk:
                    t_offs = (g + 5) * BK + tl.arange(0, BK)
                    _copy_ids(
                        sIdx.slot((g + 5) % NIDX), t_row + t_offs, t_offs < ln, BK
                    )
                    issued += 1
                p += 1
            tok, h0, lo, ln, nblk = n_tok, n_h0, n_lo, n_ln, n_nblk
            t_row = n_row
            a0 = n_a0
            a1 = n_a1
            a2 = n_a2
            a3 = n_a3
        if prev_blk:
            _wait_one_of_two(free, (r - 1) % 2)
        _drain_copies()

    @triton.jit
    def _pv_dot(pb, sKx, slot, col0: tl.constexpr, acc, BD: tl.constexpr):
        """acc + P @ V[:, col0:col0 + BD // 2] of KV slot `slot` of sKx."""
        v = tl.load(tle.gpu.local_ptr(sKx.slot(slot).subslice(col0, BD // 2, 1)))
        return tl.dot(pb, v, acc)

    @triton.jit
    def _split_columns(x, n: tl.constexpr):
        """The two column halves (interleaved n-column chunks) of a tile: a register
        selection of the MMA layout, so each layout conversion of a half needs half
        the scratch."""
        R: tl.constexpr = x.shape[0]
        C: tl.constexpr = x.shape[1]
        return tl.split(
            tl.permute(tl.reshape(x, [R, C // (2 * n), 2, n]), [0, 1, 3, 2])
        )

    @triton.jit
    def _store_o(o_ptr, col0, acc, factor, BD: tl.constexpr, SPLIT: tl.constexpr):
        """A BH x BD/2 accumulator as bf16 output; with SPLIT in two 64-column halves
        (half the conversion scratch, for a tight shared-memory budget)."""
        o = (acc * factor[:, None]).to(tl.bfloat16)
        if SPLIT:
            lo, hi = _split_columns(o, BD // 4)
            offs = col0 + tl.arange(0, BD // 4)
            tl.store(o_ptr + offs[None, :], tl.reshape(lo, [lo.shape[0], BD // 4]))
            tl.store(
                o_ptr + BD // 4 + offs[None, :], tl.reshape(hi, [hi.shape[0], BD // 4])
            )
        else:
            offs = col0 + tl.arange(0, BD // 2)
            tl.store(o_ptr + offs[None, :], o)

    @triton.jit
    def _store_partial(op_ptr, col0, acc, inv_l, BD: tl.constexpr):
        """A BH x BD/2 accumulator as an fp32 partial, in four 32-column quarters."""
        QB: tl.constexpr = BD // 8
        a, b = _split_columns(acc * inv_l[:, None], 2 * QB)
        a = tl.reshape(a, [a.shape[0], 2 * QB])
        b = tl.reshape(b, [b.shape[0], 2 * QB])
        a0, a1 = _split_columns(a, QB)
        b0, b1 = _split_columns(b, QB)
        offs = col0 + tl.arange(0, QB)
        tl.store(op_ptr + offs[None, :], tl.reshape(a0, [a0.shape[0], QB]))
        tl.store(op_ptr + QB + offs[None, :], tl.reshape(a1, [a1.shape[0], QB]))
        tl.store(op_ptr + 2 * QB + offs[None, :], tl.reshape(b0, [b0.shape[0], QB]))
        tl.store(op_ptr + 3 * QB + offs[None, :], tl.reshape(b1, [b1.shape[0], QB]))

    @triton.jit
    def _pv_group(
        sK0,
        sK1,
        sS,
        ready,
        free,
        sfree,
        topk_length,
        output,
        max_logits,
        lse,
        attn_sink,
        o_part,
        ml_part,
        pid,
        grid,
        n_my,
        TOPK,
        nsplit,
        chunk,
        stride_om,
        stride_oh,
        stride_mm,
        stride_lm,
        NHB: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        PARTIAL: tl.constexpr,
        HQ: tl.constexpr,
        BH: tl.constexpr,
        BK: tl.constexpr,
        BD: tl.constexpr,
        NKV: tl.constexpr,
        NS: tl.constexpr,
        M_FLOOR: tl.constexpr,
        RESCALE_THR: tl.constexpr,
    ):
        """Per key block: wait READY, read S, online softmax with lazy rescaling,
        O += P V from the block's KV slot, arrive FREE. Writes O, max_logits and lse
        (a partial with PARTIAL) per item."""
        LN2: tl.constexpr = 0.6931471805599453
        HB: tl.constexpr = BD // 2
        offs_h = tl.arange(0, BH)
        r = 0
        pos = 0
        for k in range(n_my):
            tok, h0, split, lo, ln, nblk = _item(
                k,
                pid,
                grid,
                n_my,
                topk_length,
                TOPK,
                nsplit,
                chunk,
                NHB,
                BH,
                BK,
                HAVE_TOPK_LENGTH,
            )
            pos += 1  # the item's Q position
            m_i = tl.full([BH], M_FLOOR, dtype=tl.float32)  # reference max of P
            # true max (max_logits) per row and key residue mod 4, reduced per item
            m_max = tl.full([BH, 4], M_FLOOR, dtype=tl.float32)
            l_i = tl.zeros([BH], dtype=tl.float32)
            # O (BH x 512) as four BH x 128 accumulators: smaller dot tiles
            acc0 = tl.zeros([BH, HB], dtype=tl.float32)
            acc1 = tl.zeros([BH, HB], dtype=tl.float32)
            acc2 = tl.zeros([BH, HB], dtype=tl.float32)
            acc3 = tl.zeros([BH, HB], dtype=tl.float32)
            for j in range(nblk):
                rr = r + j
                slot = (pos + j) % NKV
                _wait_one_of_two(ready, rr % 2)
                s = tl.load(tle.gpu.local_ptr(sS.slot(rr % NS)))
                if NS == 1:
                    if (k + 1 < n_my) | (j + 1 < nblk):
                        tle.gpu.barrier_arrive(sfree[0])
                # lazy rescaling: the reference max (and O, l with it) only moves when
                # some score of a row exceeds it by more than RESCALE_THR. With two S
                # slots the scores are tested directly, which saves the cross-warp row
                # max of every block; with one S slot (the K tail) the row max form is
                # faster on PPU-ZW810E.
                if NS == 1:
                    m_blk = tl.max(s, axis=1)
                    m_max = tl.maximum(m_max, m_blk[:, None])
                    rescale = tl.max(m_blk - m_i) > RESCALE_THR
                else:
                    m_max = tl.maximum(
                        m_max, tl.max(tl.reshape(s, [BH, BK // 4, 4]), axis=1)
                    )
                    rescale = tl.max(s - m_i[:, None]) > RESCALE_THR
                if rescale:
                    if NS != 1:
                        m_blk = tl.max(s, axis=1)
                    m_new = tl.maximum(m_i, m_blk)
                    alpha = tl.math.exp2(m_i - m_new)
                    l_i = l_i * alpha
                    acc0 = acc0 * alpha[:, None]
                    acc1 = acc1 * alpha[:, None]
                    acc2 = acc2 * alpha[:, None]
                    acc3 = acc3 * alpha[:, None]
                    m_i = m_new
                p = tl.math.exp2(s - m_i[:, None])
                l_i = l_i + tl.sum(p, axis=1)
                pb = p.to(tl.bfloat16)
                acc0 = _pv_dot(pb, sK0, slot, 0, acc0, BD)
                acc1 = _pv_dot(pb, sK0, slot, HB, acc1, BD)
                acc2 = _pv_dot(pb, sK1, slot, 0, acc2, BD)
                acc3 = _pv_dot(pb, sK1, slot, HB, acc3, BD)
                _arrive_one_of_two(free, rr % 2)
            pos += nblk
            r += nblk
            m_max = tl.max(m_max, axis=1)
            row_valid = l_i > 0
            inv_l = tl.where(row_valid, 1.0 / l_i, 0.0)
            if PARTIAL:
                # partial of (token, split): normalized O, (max, sum) in log2 units; the
                # sum is taken to the true max (the combine reports it as max_logits)
                prow = (tok * nsplit + split) * HQ + h0 + offs_h
                tl.store(ml_part + prow * 2, m_max)
                tl.store(ml_part + prow * 2 + 1, l_i * tl.math.exp2(m_i - m_max))
                op_ptr = o_part + prow[:, None] * (2 * BD)
                _store_partial(op_ptr, 0, acc0, inv_l, BD)
                _store_partial(op_ptr, HB, acc1, inv_l, BD)
                _store_partial(op_ptr, BD, acc2, inv_l, BD)
                _store_partial(op_ptr, BD + HB, acc3, inv_l, BD)
            else:
                lse_i = (m_i + tl.math.log2(l_i)) * LN2
                factor = inv_l
                if HAVE_ATTN_SINK:
                    sink = tl.load(attn_sink + h0 + offs_h)
                    factor = tl.where(
                        row_valid, factor / (1.0 + tl.math.exp(sink - lse_i)), 0.0
                    )
                tl.store(
                    max_logits + tok * stride_mm + h0 + offs_h,
                    tl.where(row_valid, m_max * LN2, float("-inf")),
                )
                tl.store(
                    lse + tok * stride_lm + h0 + offs_h,
                    tl.where(row_valid, lse_i, float("inf")),
                )
                o_ptr = output + tok * stride_om + (h0 + offs_h)[:, None] * stride_oh
                # with the K tail (NS == 1) the shared memory budget is tight
                _store_o(o_ptr, 0, acc0, factor, BD, NS == 1)
                _store_o(o_ptr, HB, acc1, factor, BD, NS == 1)
                _store_o(o_ptr, BD, acc2, factor, BD, NS == 1)
                _store_o(o_ptr, BD + HB, acc3, factor, BD, NS == 1)

    # tle.pipe variant (FLAGGEMS_FLASHMLA_SPARSE_RING=pipe). The KV ring is one pipe: the QK
    # group writes a position with tle.gpu.copy and is its own reader ("qk"), the PV group
    # is the second reader ("pv"). The index ring is replaced by register-held indices.
    @triton.jit
    def _rows(buf, BH: tl.constexpr, BK: tl.constexpr):
        """The first BH rows of a slot of BK rows."""
        if BH == BK:
            return buf
        else:
            return buf.subslice(0, BH, 0)

    @triton.jit
    def _write_q(
        writer,
        q,
        tok,
        h0,
        p,
        stride_qm,
        stride_qh,
        BH: tl.constexpr,
        BK: tl.constexpr,
        BD: tl.constexpr,
        TD: tl.constexpr,
    ):
        """Write an item's Q rows to position p of the KV pipe."""
        q_rows = q + tok * stride_qm + (h0 + tl.arange(0, BH)) * stride_qh
        cols = tl.arange(0, BD)[None, :]
        slot = writer.acquire(p)
        tle.gpu.copy(q_rows[:, None] + cols, _rows(slot.k0, BH, BK), [BH, BD])
        tle.gpu.copy(q_rows[:, None] + BD + cols, _rows(slot.k1, BH, BK), [BH, BD])
        if TD > 0:
            tcols = tl.arange(0, TD)[None, :]
            tle.gpu.copy(
                q_rows[:, None] + 2 * BD + tcols, _rows(slot.kt, BH, BK), [BH, TD]
            )
        writer.commit(p)

    @triton.jit
    def _write_kv(
        writer,
        kv,
        ids,
        SKV,
        p,
        stride_kvn,
        BK: tl.constexpr,
        BD: tl.constexpr,
        TD: tl.constexpr,
        KV_I64: tl.constexpr,
    ):
        """Write the key block with indices `ids` to position p of the KV pipe; invalid
        keys are zero-filled."""
        valid = _valid(ids, SKV)
        row_ids = tl.where(valid, ids, 0)
        if KV_I64:
            rows = kv + row_ids.to(tl.int64) * stride_kvn
        else:
            rows = kv + row_ids * stride_kvn
        cols = tl.arange(0, BD)[None, :]
        mask = tl.broadcast_to(valid[:, None], (BK, BD))
        slot = writer.acquire(p)
        tle.gpu.copy(rows[:, None] + cols, slot.k0, [BK, BD], mask=mask)
        tle.gpu.copy(rows[:, None] + BD + cols, slot.k1, [BK, BD], mask=mask)
        if TD > 0:
            tcols = tl.arange(0, TD)[None, :]
            tle.gpu.copy(
                rows[:, None] + 2 * BD + tcols,
                slot.kt,
                [BK, TD],
                mask=tl.broadcast_to(valid[:, None], (BK, TD)),
            )
        writer.commit(p)

    @triton.jit
    def _scores_slot(q0, q1, sQT, ids, slot, sm_scale_log2, SKV, TD: tl.constexpr):
        """Scores (log2 units, invalid keys -inf) of the key block in a KV pipe slot."""
        k0 = tl.load(tle.gpu.local_ptr(slot.k0))
        k1 = tl.load(tle.gpu.local_ptr(slot.k1))
        if TD > 0:
            kt = tl.load(tle.gpu.local_ptr(slot.kt))
            s = tl.dot(tl.load(tle.gpu.local_ptr(sQT)), tl.trans(kt))
            s = tl.dot(q0, tl.trans(k0), s)
        else:
            s = tl.dot(q0, tl.trans(k0))
        s = tl.dot(q1, tl.trans(k1), s)
        return tl.where(_valid(ids, SKV)[None, :], s * sm_scale_log2, float("-inf"))

    @triton.jit
    def _qk_group_pipe(
        q,
        kv,
        indices,
        topk_length,
        writer,
        qk_reader,
        sQT,
        sS,
        ready,
        sfree,
        pid,
        grid,
        n_my,
        TOPK,
        nsplit,
        chunk,
        SKV,
        sm_scale_log2,
        stride_qm,
        stride_qh,
        stride_kvn,
        stride_tm,
        NHB: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        BH: tl.constexpr,
        BK: tl.constexpr,
        BD: tl.constexpr,
        TD: tl.constexpr,
        NS: tl.constexpr,
        KV_I64: tl.constexpr,
    ):
        """The schedule of _qk_group with the KV ring as a pipe: position p holds the Q
        rows of an item or one of its key blocks; step p reads position p, computes the
        scores and writes position p + 2 (acquiring it waits until the PV group has
        released position p - 1). The indices of the block written at a step are loaded
        two steps ahead."""
        tok, h0, _, lo, ln, nblk = _item(
            0,
            pid,
            grid,
            n_my,
            topk_length,
            TOPK,
            nsplit,
            chunk,
            NHB,
            BH,
            BK,
            HAVE_TOPK_LENGTH,
        )
        t_row = indices + tok * stride_tm + lo
        a0 = _load_ids(t_row, 0, ln, BK)
        _write_q(writer, q, tok, h0, 0, stride_qm, stride_qh, BH, BK, BD, TD)
        _write_kv(writer, kv, a0, SKV, 1, stride_kvn, BK, BD, TD, KV_I64)
        # loop-carried Q registers: placeholders until the first Q position
        q0 = tl.zeros([BH, BD], dtype=tl.bfloat16)
        q1 = tl.zeros([BH, BD], dtype=tl.bfloat16)
        p = 0
        r = 0
        for k in range(n_my):
            n_tok, n_h0, _, n_lo, n_ln, n_nblk = _item(
                k + 1,
                pid,
                grid,
                n_my,
                topk_length,
                TOPK,
                nsplit,
                chunk,
                NHB,
                BH,
                BK,
                HAVE_TOPK_LENGTH,
            )
            has_next = k + 1 < n_my
            n_row = indices + n_tok * stride_tm + n_lo
            n_a0 = _load_ids(n_row, 0, tl.where(has_next, n_ln, 0), BK)
            a1 = _load_ids(t_row, 1, ln, BK)
            # the Q position
            qs = qk_reader.wait(p).slot
            q0 = tl.load(tle.gpu.local_ptr(_rows(qs.k0, BH, BK)))
            q1 = tl.load(tle.gpu.local_ptr(_rows(qs.k1, BH, BK)))
            if TD > 0:
                # the Q tail stays in shared memory (held in registers it costs the
                # dots their loads in flight)
                tl.store(
                    tle.gpu.local_ptr(sQT),
                    tl.load(tle.gpu.local_ptr(_rows(qs.kt, BH, BK))),
                )
            if nblk > 1:
                _write_kv(writer, kv, a1, SKV, p + 2, stride_kvn, BK, BD, TD, KV_I64)
            elif has_next:
                _write_q(
                    writer,
                    q,
                    n_tok,
                    n_h0,
                    p + 2,
                    stride_qm,
                    stride_qh,
                    BH,
                    BK,
                    BD,
                    TD,
                )
            p += 1
            cur = a0
            nx1 = a1
            nx2 = _load_ids(t_row, 2, ln, BK)
            for g in range(nblk):
                ks = qk_reader.wait(p).slot
                s = _scores_slot(q0, q1, sQT, cur, ks, sm_scale_log2, SKV, TD)
                if NS == 1:
                    if r >= 1:
                        _wait_barrier(sfree)  # the PV group read S(r - 1)
                tl.store(tle.gpu.local_ptr(sS.slot(r % NS)), s)
                _arrive_one_of_two(ready, r % 2)
                r += 1
                d = g + 2
                if d < nblk:
                    _write_kv(
                        writer, kv, nx2, SKV, p + 2, stride_kvn, BK, BD, TD, KV_I64
                    )
                elif has_next:
                    if d == nblk:
                        _write_q(
                            writer,
                            q,
                            n_tok,
                            n_h0,
                            p + 2,
                            stride_qm,
                            stride_qh,
                            BH,
                            BK,
                            BD,
                            TD,
                        )
                    else:
                        _write_kv(
                            writer,
                            kv,
                            n_a0,
                            SKV,
                            p + 2,
                            stride_kvn,
                            BK,
                            BD,
                            TD,
                            KV_I64,
                        )
                p += 1
                cur = nx1
                nx1 = nx2
                nx2 = _load_ids(t_row, g + 3, ln, BK)
            tok, h0, lo, ln, nblk = n_tok, n_h0, n_lo, n_ln, n_nblk
            t_row = n_row
            a0 = n_a0

    @triton.jit
    def _pv_dot_slot(pb, kslot, col0: tl.constexpr, acc, BD: tl.constexpr):
        """acc + P @ V[:, col0:col0 + BD // 2] of a KV pipe slot field."""
        v = tl.load(tle.gpu.local_ptr(kslot.subslice(col0, BD // 2, 1)))
        return tl.dot(pb, v, acc)

    @triton.jit
    def _pv_group_pipe(
        pv_reader,
        sS,
        ready,
        sfree,
        topk_length,
        output,
        max_logits,
        lse,
        attn_sink,
        o_part,
        ml_part,
        pid,
        grid,
        n_my,
        TOPK,
        nsplit,
        chunk,
        stride_om,
        stride_oh,
        stride_mm,
        stride_lm,
        NHB: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        PARTIAL: tl.constexpr,
        HQ: tl.constexpr,
        BH: tl.constexpr,
        BK: tl.constexpr,
        BD: tl.constexpr,
        NS: tl.constexpr,
        M_FLOOR: tl.constexpr,
        RESCALE_THR: tl.constexpr,
    ):
        """_pv_group with the KV ring as a pipe: the V of position pos + j is read from the
        slot that wait(pos + j) returns and released after the last dot."""
        LN2: tl.constexpr = 0.6931471805599453
        HB: tl.constexpr = BD // 2
        offs_h = tl.arange(0, BH)
        r = 0
        pos = 0
        for k in range(n_my):
            tok, h0, split, lo, ln, nblk = _item(
                k,
                pid,
                grid,
                n_my,
                topk_length,
                TOPK,
                nsplit,
                chunk,
                NHB,
                BH,
                BK,
                HAVE_TOPK_LENGTH,
            )
            # the Q position is read by the QK group only: skip it
            pv_reader.wait(pos)
            pv_reader.release(pos)
            pos += 1
            m_i = tl.full([BH], M_FLOOR, dtype=tl.float32)  # reference max of P
            # true max (max_logits) per row and key residue mod 4, reduced per item
            m_max = tl.full([BH, 4], M_FLOOR, dtype=tl.float32)
            l_i = tl.zeros([BH], dtype=tl.float32)
            # O (BH x 512) as four BH x 128 accumulators: smaller dot tiles
            acc0 = tl.zeros([BH, HB], dtype=tl.float32)
            acc1 = tl.zeros([BH, HB], dtype=tl.float32)
            acc2 = tl.zeros([BH, HB], dtype=tl.float32)
            acc3 = tl.zeros([BH, HB], dtype=tl.float32)
            for j in range(nblk):
                rr = r + j
                _wait_one_of_two(ready, rr % 2)
                s = tl.load(tle.gpu.local_ptr(sS.slot(rr % NS)))
                if NS == 1:
                    if (k + 1 < n_my) | (j + 1 < nblk):
                        tle.gpu.barrier_arrive(sfree[0])
                # lazy rescaling: see _pv_group
                if NS == 1:
                    m_blk = tl.max(s, axis=1)
                    m_max = tl.maximum(m_max, m_blk[:, None])
                    rescale = tl.max(m_blk - m_i) > RESCALE_THR
                else:
                    m_max = tl.maximum(
                        m_max, tl.max(tl.reshape(s, [BH, BK // 4, 4]), axis=1)
                    )
                    rescale = tl.max(s - m_i[:, None]) > RESCALE_THR
                if rescale:
                    if NS != 1:
                        m_blk = tl.max(s, axis=1)
                    m_new = tl.maximum(m_i, m_blk)
                    alpha = tl.math.exp2(m_i - m_new)
                    l_i = l_i * alpha
                    acc0 = acc0 * alpha[:, None]
                    acc1 = acc1 * alpha[:, None]
                    acc2 = acc2 * alpha[:, None]
                    acc3 = acc3 * alpha[:, None]
                    m_i = m_new
                p = tl.math.exp2(s - m_i[:, None])
                l_i = l_i + tl.sum(p, axis=1)
                pb = p.to(tl.bfloat16)
                vs = pv_reader.wait(pos + j).slot
                acc0 = _pv_dot_slot(pb, vs.k0, 0, acc0, BD)
                acc1 = _pv_dot_slot(pb, vs.k0, HB, acc1, BD)
                acc2 = _pv_dot_slot(pb, vs.k1, 0, acc2, BD)
                acc3 = _pv_dot_slot(pb, vs.k1, HB, acc3, BD)
                pv_reader.release(pos + j)
            pos += nblk
            r += nblk
            m_max = tl.max(m_max, axis=1)
            row_valid = l_i > 0
            inv_l = tl.where(row_valid, 1.0 / l_i, 0.0)
            if PARTIAL:
                prow = (tok * nsplit + split) * HQ + h0 + offs_h
                tl.store(ml_part + prow * 2, m_max)
                tl.store(ml_part + prow * 2 + 1, l_i * tl.math.exp2(m_i - m_max))
                op_ptr = o_part + prow[:, None] * (2 * BD)
                _store_partial(op_ptr, 0, acc0, inv_l, BD)
                _store_partial(op_ptr, HB, acc1, inv_l, BD)
                _store_partial(op_ptr, BD, acc2, inv_l, BD)
                _store_partial(op_ptr, BD + HB, acc3, inv_l, BD)
            else:
                lse_i = (m_i + tl.math.log2(l_i)) * LN2
                factor = inv_l
                if HAVE_ATTN_SINK:
                    sink = tl.load(attn_sink + h0 + offs_h)
                    factor = tl.where(
                        row_valid, factor / (1.0 + tl.math.exp(sink - lse_i)), 0.0
                    )
                tl.store(
                    max_logits + tok * stride_mm + h0 + offs_h,
                    tl.where(row_valid, m_max * LN2, float("-inf")),
                )
                tl.store(
                    lse + tok * stride_lm + h0 + offs_h,
                    tl.where(row_valid, lse_i, float("inf")),
                )
                o_ptr = output + tok * stride_om + (h0 + offs_h)[:, None] * stride_oh
                _store_o(o_ptr, 0, acc0, factor, BD, NS == 1)
                _store_o(o_ptr, HB, acc1, factor, BD, NS == 1)
                _store_o(o_ptr, BD, acc2, factor, BD, NS == 1)
                _store_o(o_ptr, BD + HB, acc3, factor, BD, NS == 1)

    @triton.jit
    def _ppu_sparse_mla_kernel_pipe(
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        sm_scale_log2,
        output,
        max_logits,
        lse,
        o_part,
        ml_part,
        stride_qm,
        stride_qh,
        stride_kvn,
        stride_tm,
        stride_om,
        stride_oh,
        stride_mm,
        stride_lm,
        SKV,
        TOPK,
        n_items,
        nsplit,
        chunk,
        HQ: tl.constexpr,
        DQK: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        PARTIAL: tl.constexpr,
        BK: tl.constexpr,
        M_FLOOR: tl.constexpr,
        RESCALE_THR: tl.constexpr,
        QK_WARPS: tl.constexpr,
        PV_WARPS: tl.constexpr,
        KV_I64: tl.constexpr,
        BH: tl.constexpr,
    ):
        """_ppu_sparse_mla_kernel with the KV ring as a tle.pipe (see _qk_group_pipe)."""
        BD: tl.constexpr = 256
        NKV: tl.constexpr = 3  # KV ring
        TD: tl.constexpr = DQK - 2 * BD
        NS: tl.constexpr = 1 if TD > 0 else 2
        TDA: tl.constexpr = TD if TD > 0 else 16  # placeholders without a tail
        NHB: tl.constexpr = HQ // BH
        pid = tl.program_id(0)
        grid = tl.num_programs(0)
        n_my = tl.cdiv(n_items - pid, grid)
        sK0 = tle.gpu.alloc(
            [NKV, BK, BD],
            dtype=tl.bfloat16,
            layout=_ring_layout(8, 16),
            scope=tle.gpu.smem,
        )
        sK1 = tle.gpu.alloc(
            [NKV, BK, BD],
            dtype=tl.bfloat16,
            layout=_ring_layout(8, 16),
            scope=tle.gpu.smem,
        )
        sKT = tle.gpu.alloc(
            [NKV, BK, TDA],
            dtype=tl.bfloat16,
            layout=_ring_layout(8, 8),
            scope=tle.gpu.smem,
        )
        sS = tle.gpu.alloc(
            [NS, BH, BK],
            dtype=tl.float32,
            layout=_ring_layout(4, 8),
            scope=tle.gpu.smem,
        )
        sQT = tle.gpu.alloc(
            [BH, TDA],
            dtype=tl.bfloat16,
            layout=_tile_layout(8, 8),
            scope=tle.gpu.smem,
        )
        if TD > 0:
            kv_pipe = tle.pipe(
                capacity=NKV,
                name="kv",
                readers=("qk", "pv"),
                k0=sK0,
                k1=sK1,
                kt=sKT,
            )
        else:
            kv_pipe = tle.pipe(
                capacity=NKV, name="kv", readers=("qk", "pv"), k0=sK0, k1=sK1
            )
        NTHR: tl.constexpr = (QK_WARPS + PV_WARPS) * 32
        ready = tle.gpu.alloc_barriers(2, arrive_count=NTHR)
        sfree = tle.gpu.alloc_barriers(1, arrive_count=NTHR)
        tle.gpu.warp_specialize(
            [
                (
                    _qk_group_pipe,
                    (
                        q,
                        kv,
                        indices,
                        topk_length,
                        kv_pipe.writer(),
                        kv_pipe.reader("qk"),
                        sQT,
                        sS,
                        ready,
                        sfree,
                        pid,
                        grid,
                        n_my,
                        TOPK,
                        nsplit,
                        chunk,
                        SKV,
                        sm_scale_log2,
                        stride_qm,
                        stride_qh,
                        stride_kvn,
                        stride_tm,
                        NHB,
                        HAVE_TOPK_LENGTH,
                        BH,
                        BK,
                        BD,
                        TD,
                        NS,
                        KV_I64,
                    ),
                ),
                (
                    _pv_group_pipe,
                    (
                        kv_pipe.reader("pv"),
                        sS,
                        ready,
                        sfree,
                        topk_length,
                        output,
                        max_logits,
                        lse,
                        attn_sink,
                        o_part,
                        ml_part,
                        pid,
                        grid,
                        n_my,
                        TOPK,
                        nsplit,
                        chunk,
                        stride_om,
                        stride_oh,
                        stride_mm,
                        stride_lm,
                        NHB,
                        HAVE_TOPK_LENGTH,
                        HAVE_ATTN_SINK,
                        PARTIAL,
                        HQ,
                        BH,
                        BK,
                        BD,
                        NS,
                        M_FLOOR,
                        RESCALE_THR,
                    ),
                ),
            ],
            [PV_WARPS],
            [256],  # requested registers of the PV group; PPU ignores them
        )

    @triton.jit
    def _ppu_sparse_mla_kernel(
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        sm_scale_log2,
        output,
        max_logits,
        lse,
        o_part,
        ml_part,
        stride_qm,
        stride_qh,
        stride_kvn,
        stride_tm,
        stride_om,
        stride_oh,
        stride_mm,
        stride_lm,
        SKV,
        TOPK,
        n_items,
        nsplit,
        chunk,
        HQ: tl.constexpr,
        DQK: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        PARTIAL: tl.constexpr,
        BK: tl.constexpr,
        M_FLOOR: tl.constexpr,
        RESCALE_THR: tl.constexpr,
        QK_WARPS: tl.constexpr,
        PV_WARPS: tl.constexpr,
        KV_I64: tl.constexpr,
        BH: tl.constexpr,
    ):
        """Persistent CTA: work items pid, pid + grid, ... (see the module docstring).
        Shared memory: KV ring sK0 / sK1 (dims 0 .. 255 / 256 .. 511), K-tail ring
        sKT and Q tail sQT (DQK = 576), scores sS, key index ring sIdx."""
        BD: tl.constexpr = 256
        NKV: tl.constexpr = 3  # KV ring
        NIDX: tl.constexpr = 8  # index ring
        TD: tl.constexpr = DQK - 2 * BD
        # S slots: 2, or 1 with the K tail (its room holds the Q tail; the QK group
        # then waits until the PV group has read S)
        NS: tl.constexpr = 1 if TD > 0 else 2
        TDA: tl.constexpr = TD if TD > 0 else 16  # placeholders without a tail
        NHB: tl.constexpr = HQ // BH
        pid = tl.program_id(0)
        grid = tl.num_programs(0)
        n_my = tl.cdiv(n_items - pid, grid)
        sK0 = tle.gpu.alloc(
            [NKV, BK, BD],
            dtype=tl.bfloat16,
            layout=_ring_layout(8, 16),
            scope=tle.gpu.smem,
        )
        sK1 = tle.gpu.alloc(
            [NKV, BK, BD],
            dtype=tl.bfloat16,
            layout=_ring_layout(8, 16),
            scope=tle.gpu.smem,
        )
        sS = tle.gpu.alloc(
            [NS, BH, BK],
            dtype=tl.float32,
            layout=_ring_layout(4, 8),
            scope=tle.gpu.smem,
        )
        sKT = tle.gpu.alloc(
            [2, BK, TDA],
            dtype=tl.bfloat16,
            layout=_ring_layout(8, 8),
            scope=tle.gpu.smem,
        )
        sQT = tle.gpu.alloc(
            [BH, TDA],
            dtype=tl.bfloat16,
            layout=_tile_layout(8, 8),
            scope=tle.gpu.smem,
        )
        sIdx = tle.gpu.alloc(
            [NIDX, BK], dtype=tl.int32, layout=_tile_layout(1, 1), scope=tle.gpu.smem
        )
        NTHR: tl.constexpr = (QK_WARPS + PV_WARPS) * 32
        ready = tle.gpu.alloc_barriers(2, arrive_count=NTHR)
        free = tle.gpu.alloc_barriers(2, arrive_count=NTHR)
        qkbar = tle.gpu.alloc_barriers(1, arrive_count=QK_WARPS * 32)
        sfree = tle.gpu.alloc_barriers(1, arrive_count=NTHR)
        tle.gpu.warp_specialize(
            [
                (
                    _qk_group,
                    (
                        q,
                        kv,
                        indices,
                        topk_length,
                        sK0,
                        sK1,
                        sKT,
                        sQT,
                        sS,
                        sIdx,
                        ready,
                        free,
                        qkbar,
                        sfree,
                        pid,
                        grid,
                        n_my,
                        TOPK,
                        nsplit,
                        chunk,
                        SKV,
                        sm_scale_log2,
                        stride_qm,
                        stride_qh,
                        stride_kvn,
                        stride_tm,
                        NHB,
                        HAVE_TOPK_LENGTH,
                        BH,
                        BK,
                        BD,
                        NKV,
                        TD,
                        NIDX,
                        NS,
                        KV_I64,
                    ),
                ),
                (
                    _pv_group,
                    (
                        sK0,
                        sK1,
                        sS,
                        ready,
                        free,
                        sfree,
                        topk_length,
                        output,
                        max_logits,
                        lse,
                        attn_sink,
                        o_part,
                        ml_part,
                        pid,
                        grid,
                        n_my,
                        TOPK,
                        nsplit,
                        chunk,
                        stride_om,
                        stride_oh,
                        stride_mm,
                        stride_lm,
                        NHB,
                        HAVE_TOPK_LENGTH,
                        HAVE_ATTN_SINK,
                        PARTIAL,
                        HQ,
                        BH,
                        BK,
                        BD,
                        NKV,
                        NS,
                        M_FLOOR,
                        RESCALE_THR,
                    ),
                ),
            ],
            [PV_WARPS],
            [256],  # requested registers of the PV group; PPU ignores them
        )


@functools.lru_cache(maxsize=None)
def _has_tle() -> bool:
    return _tle_api is not None and _tle_api.is_tle_supported(_TLE_PRIMITIVES)


@functools.lru_cache(maxsize=None)
def _has_tle_pipe() -> bool:
    return _tle_api is not None and _tle_api.is_tle_supported(_TLE_PIPE_PRIMITIVES)


def _use_pipe() -> bool:
    """Select the KV-ring implementation independently of TLE dispatch."""
    value = os.environ.get("FLAGGEMS_FLASHMLA_SPARSE_RING", "auto").strip().lower()
    if value not in {"auto", "inline_asm", "pipe"}:
        raise ValueError(
            f"invalid FLAGGEMS_FLASHMLA_SPARSE_RING={value!r}; "
            "expected auto, inline_asm or pipe"
        )
    if value == "pipe":
        if not _has_tle_pipe():
            raise NotImplementedError(
                "FLAGGEMS_FLASHMLA_SPARSE_RING=pipe requires PPU TLE pipe support"
            )
        return True
    if value == "inline_asm" or "FLAGGEMS_FLASHMLA_SPARSE_RING" in os.environ:
        return False
    return (
        os.environ.get("FLAGGEMS_FLASHMLA_SPARSE_PIPE", "0").lower()
        in {"1", "true", "on", "yes"}
        and _has_tle_pipe()
    )


@functools.lru_cache(maxsize=None)
def _mma_sm_count(device_index: int) -> int:
    props = torch_device_fn.get_device_properties(device_index)
    return _MMA_SM_COUNT.get(props.name, props.multi_processor_count)


@functools.lru_cache(maxsize=None)
def _plan(sq: int, hq: int, dqk: int, topk: int, n_sm: int) -> Tuple[int, int, int]:
    """(nsplit, chunk, head_block): an item is `head_block` heads of one token over
    `chunk` key blocks. Persistent CTAs run the items round robin, so the cost is that
    of the busiest CTA; fewer heads per item and TOPK splits spread few tokens over the
    SMs."""
    nblk = max(1, triton.cdiv(topk, BLOCK_K))
    best = None
    for bh in (64, 32, 16):
        key = (dqk, bh)
        for nsplit in range(1, min(nblk, _MAX_SPLIT) + 1):
            chunk = triton.cdiv(nblk, nsplit)
            if triton.cdiv(nblk, chunk) != nsplit:
                continue
            per_cta = triton.cdiv(sq * nsplit * (hq // bh), n_sm)
            cost = _FILL_US[key] + per_cta * (_ITEM_US[key] + chunk * _BLOCK_US[key])
            if nsplit > 1:
                cost += _COMBINE_US + _COMBINE_ROW_US * sq * nsplit * hq
            if best is None or cost < best[0]:
                best = (cost, nsplit, chunk, bh)
    return best[1:]


@functools.lru_cache(maxsize=None)
def _plan_triton(
    sq: int, hq: int, dqk: int, topk: int, n_sm: int
) -> Tuple[int, int, int, int]:
    """(nsplit, chunk, head_block, num_warps): a work item is `head_block` heads of
    one token over `chunk` 128-key blocks. TOPK is not split once 64-head items fill
    the MMA-capable SMs; otherwise the latency model picks the split and head block.
    16 warps pay off for 64-head DQK=512 items that are few or long."""
    nblk = max(1, triton.cdiv(topk, _TRITON_BLOCK_K))
    few = sq * (hq // 64) < n_sm
    if not few:
        nsplit, chunk, bh = 1, nblk, 64
    else:
        best = None
        for bh in (64, 32, 16):
            for nsplit in range(1, min(nblk, _MAX_SPLIT) + 1):
                chunk = triton.cdiv(nblk, nsplit)
                if triton.cdiv(nblk, chunk) != nsplit:
                    continue
                waves = triton.cdiv(sq * nsplit * (hq // bh), n_sm)
                cost = waves * (_TRITON_ITEM_US[bh] + chunk * _TRITON_BLOCK_US[bh])
                if nsplit > 1:
                    cost += _TRITON_COMBINE_US
                if best is None or cost < best[0]:
                    best = (cost, nsplit, chunk, bh)
        nsplit, chunk, bh = best[1:]
    wide = dqk == 512 and bh == 64 and (few or nblk >= 32)
    return nsplit, chunk, bh, 16 if wide else 8


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse attention prefill kernel; see flaggems_vllm.ops.flashmla_sparse."""
    enabled = os.environ.get("FLAGGEMS_FLASHMLA_SPARSE_TLE", "1").lower()
    use_tle = enabled not in {"0", "false", "off", "no"} and _has_tle()
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    # TLE copies Q and KV rows as 16-byte vectors; offset tensors use Triton.
    use_tle = use_tle and q.data_ptr() % 16 == 0 and kv.data_ptr() % 16 == 0
    assert (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
    )
    SQ, HQ, DQK = q.shape
    SKV, HKV, _ = kv.shape

    assert d_v == 512, "Unsupported d_v"
    DV = d_v

    assert kv.shape[-1] == DQK
    _, _, TOPK = indices.shape
    assert indices.shape == (SQ, HKV, TOPK)
    if attn_sink is not None:
        assert attn_sink.is_contiguous()
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (HQ,), "attn_sink error shape"
    if topk_length is not None:
        assert topk_length.is_contiguous()
        assert topk_length.dtype == torch.int32
        assert topk_length.shape == (SQ,), "topk_length error shape"

    # check from FlashMLA
    assert HKV == 1, "h_kv is expected to be 1"
    assert HQ == 64 or HQ == 128, "Unsupported h_q"
    assert DQK == 576 or DQK == 512, "Unsupported d_qk"

    output = torch.empty((SQ, HQ, DV), device=q.device, dtype=q.dtype)
    max_logits = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    lse = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    if SQ == 0:
        return output, max_logits, lse

    device_index = q.device.index
    if device_index is None:
        device_index = torch_device_fn.current_device()
    n_sm = _mma_sm_count(device_index)
    if use_tle:
        nsplit, chunk, bh = _plan(SQ, HQ, DQK, TOPK, n_sm)
    else:
        nsplit, chunk, bh, num_warps = _plan_triton(SQ, HQ, DQK, TOPK, n_sm)
    if nsplit > 1:
        o_part = torch.empty(
            (SQ * nsplit * HQ, DV), device=q.device, dtype=torch.float32
        )
        ml_part = torch.empty(
            (SQ * nsplit * HQ, 2), device=q.device, dtype=torch.float32
        )
    else:
        o_part = ml_part = max_logits  # unused

    with torch_device_fn.device(q.device):
        n_items = SQ * nsplit * (HQ // bh)
        if use_tle:
            # warp_specialize cannot capture None: unused pointers get a placeholder
            kernel = (
                _ppu_sparse_mla_kernel_pipe if _use_pipe() else _ppu_sparse_mla_kernel
            )
            kernel[(min(n_items, n_sm),)](
                q,
                kv,
                indices,
                attn_sink if attn_sink is not None else max_logits,
                topk_length if topk_length is not None else indices,
                sm_scale * LOG2E,
                output,
                max_logits,
                lse,
                o_part,
                ml_part,
                q.stride(0),
                q.stride(1),
                kv.stride(0),
                indices.stride(0),
                output.stride(0),
                output.stride(1),
                max_logits.stride(0),
                lse.stride(0),
                SKV,
                TOPK,
                n_items,
                nsplit,
                chunk,
                HQ=HQ,
                DQK=DQK,
                HAVE_ATTN_SINK=attn_sink is not None,
                HAVE_TOPK_LENGTH=topk_length is not None,
                PARTIAL=nsplit > 1,
                BK=BLOCK_K,
                M_FLOOR=M_FLOOR,
                RESCALE_THR=RESCALE_THR,
                QK_WARPS=_GROUP_WARPS,
                PV_WARPS=_GROUP_WARPS,
                # 32-bit KV row offsets when they cannot overflow: cheaper addresses
                KV_I64=kv.numel() >= 2**31,
                BH=bh,
                num_warps=_GROUP_WARPS,
                num_stages=1,
            )
        else:
            _FWD_KERNELS[num_warps][(n_items,)](
                q,
                kv,
                indices,
                attn_sink,
                topk_length,
                sm_scale * LOG2E,
                output,
                max_logits,
                lse,
                o_part,
                ml_part,
                q.stride(0),
                q.stride(1),
                kv.stride(0),
                indices.stride(0),
                output.stride(0),
                output.stride(1),
                max_logits.stride(0),
                lse.stride(0),
                SKV,
                TOPK,
                nsplit,
                chunk,
                HQ=HQ,
                DQK=DQK,
                HAVE_ATTN_SINK=attn_sink is not None,
                HAVE_TOPK_LENGTH=topk_length is not None,
                PARTIAL=nsplit > 1,
                BH=bh,
                BK=_TRITON_BLOCK_K,
                M_FLOOR=M_FLOOR,
                num_warps=num_warps,
                num_stages=3 if num_warps == 16 else 2,
            )
        if nsplit > 1:
            _ppu_sparse_mla_combine_kernel[(SQ * HQ // _COMBINE_HEADS,)](
                o_part,
                ml_part,
                attn_sink,
                output,
                max_logits,
                lse,
                output.stride(0),
                output.stride(1),
                max_logits.stride(0),
                lse.stride(0),
                HQ=HQ,
                DV=DV,
                NSPLIT=nsplit,
                HAVE_ATTN_SINK=attn_sink is not None,
                BHC=_COMBINE_HEADS,
                num_warps=4,
                num_stages=1,
            )
    return output, max_logits, lse
