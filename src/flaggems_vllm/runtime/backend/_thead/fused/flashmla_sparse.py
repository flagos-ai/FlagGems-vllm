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

"""Sparse MLA prefill (flash_mla_sparse_fwd) for PPU.

Same ABI and semantics as ``flaggems_vllm.ops.flashmla_sparse``. PPU-specific:

* KV is gathered as [BK, D] tiles, read transposed by QK and as is by PV, so the
  gather is software-pipelined (the generic [D, BK] tile spills on PPU).
* Matrix-instruction kernels only run on some SMs (20 of 64 on PPU-ZW810E); few
  query tokens are spread over them by splitting TOPK and by head blocks < 64.
"""

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

LOG2E = 1.4426950408889634
# Initial running max (log2 units): finite, so fully masked blocks give no NaN in
# exp2(m_old - m_new) or exp2(s - m_new) without extra selects.
M_FLOOR = -1.0e30
BLOCK_K = 128

# SMs that run matrix-instruction kernels, by device model (others: all SMs).
_MMA_SM_COUNT = {"PPU-ZW810E": 20}

# Work-item latency (us) fitted on PPU-ZW810E: fixed and per-128-key-block cost by
# head block size, plus the combine launch of a TOPK split.
_ITEM_US = {64: 10.6, 32: 8.0, 16: 6.5}
_BLOCK_US = {64: 7.9, 32: 5.55, 16: 4.1}
_COMBINE_US = 3.0
# The combine kernel unrolls over the splits.
_MAX_SPLIT = 16


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


# LibEntry caches compiled kernels by argument values, not by num_warps: one per warp count.
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


@functools.lru_cache(maxsize=None)
def _mma_sm_count(device_index: int) -> int:
    props = torch_device_fn.get_device_properties(device_index)
    return _MMA_SM_COUNT.get(props.name, props.multi_processor_count)


@functools.lru_cache(maxsize=None)
def _plan(
    sq: int, hq: int, dqk: int, topk: int, n_sm: int
) -> Tuple[int, int, int, int]:
    """(nsplit, chunk, head_block, num_warps): a work item is `head_block` heads of
    one token over `chunk` 128-key blocks. TOPK is not split once 64-head items fill
    the MMA-capable SMs; otherwise the latency model picks the split and head block.
    16 warps pay off for 64-head DQK=512 items that are few or long."""
    nblk = max(1, triton.cdiv(topk, BLOCK_K))
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
                cost = waves * (_ITEM_US[bh] + chunk * _BLOCK_US[bh])
                if nsplit > 1:
                    cost += _COMBINE_US
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
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
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
    nsplit, chunk, bh, num_warps = _plan(SQ, HQ, DQK, TOPK, _mma_sm_count(device_index))
    if nsplit > 1:
        o_part = torch.empty(
            (SQ * nsplit * HQ, DV), device=q.device, dtype=torch.float32
        )
        ml_part = torch.empty(
            (SQ * nsplit * HQ, 2), device=q.device, dtype=torch.float32
        )
    else:
        o_part = ml_part = output  # unused

    with torch_device_fn.device(q.device):
        _FWD_KERNELS[num_warps][(SQ * nsplit * (HQ // bh),)](
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
            BK=BLOCK_K,
            M_FLOOR=M_FLOOR,
            num_warps=num_warps,
            num_stages=3 if num_warps == 16 else 2,  # fastest on PPU-ZW810E
        )
        if nsplit > 1:
            _ppu_sparse_mla_combine_kernel[(SQ * HQ // 8,)](
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
                BHC=8,
                num_warps=4,
                num_stages=1,
            )
    return output, max_logits, lse
