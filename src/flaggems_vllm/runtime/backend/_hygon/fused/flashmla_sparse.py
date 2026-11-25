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

"""Sparse MLA prefill (flash_mla_sparse_fwd) for HCU.

Same ABI and semantics as ``flaggems_vllm.ops.flashmla_sparse``. gfx938 has no
warp specialization (the backend's lowering is gated to gfx946), no TMA and no
global-to-LDS direct loads, so this is a plain single-stage kernel whose shape
is dictated by what BW1101 (gfx938: 64 CU, 64 KiB LDS, 256 VGPR/thread,
wave64, ``mmac_f32_16x16x16bf16``) actually allows:

* One work item owns ``BH`` heads of one query token, so every gathered KV row
  feeds ``BH`` heads of QK and PV. ``BH=64`` is the largest value the MFMA tile
  splits evenly over 4 waves. Splitting the heads or the value dimension over
  more work items duplicates the KV gather, which costs far more than the
  register pressure it relieves.
* ``Q`` costs BH*DQK*2 and the f32 accumulator BH*DV*4 bytes of register file:
  together 192 of the 256 VGPRs per thread, which is all gfx938 can address.
  Only the remaining quarter is left for the KV pipeline, which is why the KV
  tile is gathered in ``DV // BD`` chunks of whole rows - wide loads for
  bandwidth, small live tiles for LDS and registers. The tile cannot be made to
  fit: QK consumes it with a free layout while PV needs it transposed, so it
  stays live across both and the compiler spills (see ``BLOCK_K_BY_DQK``).
* The accumulator is rescaled only when the running max actually grows, which
  removes a [BH, DV] multiply from almost every iteration. The result is still
  the exact online softmax, not an approximation.
* Few query tokens are spread over the CUs by splitting TOPK; the partial
  results are recombined by a second kernel.
"""

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

LOG2E = 1.4426950408889634
LN2 = tl.constexpr(0.6931471805599453)
# Initial running max (log2 units): finite, so a fully masked block gives no NaN
# in exp2(m_old - m_new) or exp2(s - m_new) without extra selects.
M_FLOOR = tl.constexpr(-1.0e30)

BLOCK_H = 64
# The key block trades register spilling against per-iteration overhead, and the
# two DQK values land on opposite sides of it: at 512 the narrower block cuts
# spilled values from 119 to 45 and is worth 3%, while at 576 the extra pe
# columns make the per-iteration cost dominate, so halving the block costs 15%
# even though it halves the spills too.
BLOCK_K_BY_DQK = {512: 32, 576: 64}
# When TOPK fits in one key block the loop is peeled into straight-line code and
# the register allocator stops spilling (41 vs 134 spilled values), which is
# worth ~1.4x. With more than one iteration the wider block spills far more, so
# this only applies to the single-block case.
BLOCK_K_SINGLE = 128
BLOCK_D = 128
_NUM_WARPS = 4
# Occupancy target for the split decision below. The kernel is register-limited
# to 3 work groups per CU (256 of the 768 VGPRs per SIMD), but splitting pays off
# only well short of that: every extra split writes and re-reads a [HQ, DV] f32
# partial, and measurements turn over at 2.
_CTAS_PER_CU = 2
# The combine kernel unrolls over the splits.
_MAX_SPLIT = 16
# Below this many key blocks per token, the combine launch costs more host time
# than the split saves on the device.
_MIN_SPLIT_BLOCKS = 4


@triton.jit
def _hcu_sparse_mla_fwd_kernel(
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
    BD: tl.constexpr,
    DV: tl.constexpr,
):
    """One work item: BH heads of one query token over `chunk` key blocks."""
    NHB: tl.constexpr = HQ // BH
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
    q2 = tl.load(q_ptr + 2 * BD + offs_d[None, :])
    q3 = tl.load(q_ptr + 3 * BD + offs_d[None, :])
    if DQK == 576:
        offs_td = tl.arange(0, 64)
        qt = tl.load(q_ptr + DV + offs_td[None, :])

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
    acc2 = tl.zeros([BH, BD], dtype=tl.float32)
    acc3 = tl.zeros([BH, BD], dtype=tl.float32)
    t_base = indices + tok * stride_tm

    for ck in range(lo, hi):
        t_offs = ck * BK + offs_t
        ids = tl.load(t_base + t_offs, mask=t_offs < topk_len, other=-1)
        valid = (ids >= 0) & (ids < SKV)
        ids = tl.where(valid, ids, 0)
        kv_ptr = kv + ids[:, None].to(tl.int64) * stride_kvn

        # whole KV rows, one chunk per load: all NBD loads are issued before the
        # first dot, which is what gives the gather its memory-level parallelism
        k0 = tl.load(kv_ptr + offs_d[None, :])
        k1 = tl.load(kv_ptr + BD + offs_d[None, :])
        k2 = tl.load(kv_ptr + 2 * BD + offs_d[None, :])
        k3 = tl.load(kv_ptr + 3 * BD + offs_d[None, :])

        s = tl.dot(q0, tl.trans(k0))
        s = tl.dot(q1, tl.trans(k1), s)
        s = tl.dot(q2, tl.trans(k2), s)
        s = tl.dot(q3, tl.trans(k3), s)
        if DQK == 576:
            kt = tl.load(kv_ptr + DV + offs_td[None, :])
            s = tl.dot(qt, tl.trans(kt), s)
        s = tl.where(valid[None, :], s * sm_scale_log2, float("-inf"))

        # Rescale only when the max grows: p = exp2(s - m_i) stays <= 1 whenever
        # every head's block max is already covered by m_i, so this is the exact
        # online softmax with the [BH, DV] multiply skipped.
        blk_max = tl.max(s, axis=1)
        if tl.max(blk_max - m_i) > 0.0:
            m_new = tl.maximum(m_i, blk_max)
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc2 = acc2 * alpha[:, None]
            acc3 = acc3 * alpha[:, None]
            m_i = m_new

        p = tl.math.exp2(s - m_i[:, None])
        l_i += tl.sum(p, axis=1)
        pb = p.to(tl.bfloat16)
        acc0 = tl.dot(pb, k0, acc0)
        acc1 = tl.dot(pb, k1, acc1)
        acc2 = tl.dot(pb, k2, acc2)
        acc3 = tl.dot(pb, k3, acc3)

    row_valid = l_i > 0
    inv_l = tl.where(row_valid, 1.0 / l_i, 0.0)
    if PARTIAL:
        # partial of (token, split): normalized O, (max, sum) in log2 units
        prow = (tok * nsplit + split) * HQ + h0 + offs_h
        op_ptr = o_part + prow[:, None] * DV
        tl.store(op_ptr + offs_d[None, :], acc0 * inv_l[:, None])
        tl.store(op_ptr + BD + offs_d[None, :], acc1 * inv_l[:, None])
        tl.store(op_ptr + 2 * BD + offs_d[None, :], acc2 * inv_l[:, None])
        tl.store(op_ptr + 3 * BD + offs_d[None, :], acc3 * inv_l[:, None])
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
        tl.store(
            o_ptr + 2 * BD + offs_d[None, :], (acc2 * factor[:, None]).to(tl.bfloat16)
        )
        tl.store(
            o_ptr + 3 * BD + offs_d[None, :], (acc3 * factor[:, None]).to(tl.bfloat16)
        )


@libentry()
@triton.jit
def _hcu_sparse_mla_combine_kernel(
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


# LibEntry caches compiled kernels by argument values, not by launch options.
_FWD_KERNEL = libentry()(_hcu_sparse_mla_fwd_kernel)


@functools.lru_cache(maxsize=None)
def _backend_launch_options(dqk: int, split: bool) -> dict:
    """Backend-specific launch options, skipped on backends that lack them.

    ``kpack=2`` widens the per-lane K footprint of the MFMA operands, which pays
    off for the 512-wide rows (5-12%) but regresses the 576-wide ones, so it is
    selected per DQK.

    ``sched_latency="mmac5-ds32"`` makes the scheduler assume a much longer LDS
    latency, so it hoists the ds_reads of the PV operand away from the mmacs
    that consume them. The inner loop otherwise spends 71 of its 762
    instructions on a fully blocking ``s_waitcnt lgkmcnt(0)``, and with only 3
    work groups per CU there is no other wave to cover that latency.

    ``waves_per_eu_exact`` makes ``waves_per_eu`` actually pin the register
    budget, which this kernel needs: without it the allocator keeps sizing for
    occupancy and spills 321 values instead of 45 (2.9x on the TOPK=128 rows).

    ``in_thread_transpose`` transposes the PV operand in registers before the
    local stores instead of reading it out of LDS two bytes at a time, worth
    1.17-1.35x here. Both are opt-in on gfx938 because they cost other kernels.
    """
    try:
        import importlib

        from triton.backends import backends
    except ImportError:
        return {}
    for backend in backends.values():
        module = importlib.import_module(backend.compiler.__module__)
        fields = getattr(
            getattr(module, "HIPOptions", None), "__dataclass_fields__", {}
        )
        if "schedule_hint" not in fields:
            continue
        # iglp_opt interleaves the MFMA stream with the KV gather and the LDS
        # reads of the transposed PV operand (4-11% on BW1101).
        options = {"schedule_hint": "attention"}
        if "waves_per_eu_exact" in fields:
            options["waves_per_eu_exact"] = True
        if "in_thread_transpose" in fields:
            options["in_thread_transpose"] = True
        if dqk == 512 and "kpack" in fields:
            options["kpack"] = 2
        if not split and "sched_latency" in fields:
            options["sched_latency"] = "mmac5-ds32"
        return options
    return {}


@functools.lru_cache(maxsize=None)
def _cu_count(device_index: int) -> int:
    return torch_device_fn.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=None)
def _plan(sq: int, hq: int, topk: int, dqk: int, n_cu: int) -> Tuple[int, int, int]:
    """(block_k, nsplit, chunk): a work item covers `chunk` key blocks of one token.

    TOPK is only split when BH-head items cannot fill the CUs, which is the
    regime the latency is dominated by idle CUs rather than by the gather.
    Splitting costs a second kernel launch (~80us of host time on this stack),
    so it has to buy back more than a couple of key blocks of device time.
    """
    block_k = BLOCK_K_SINGLE if topk <= BLOCK_K_SINGLE else BLOCK_K_BY_DQK[dqk]
    nblk = max(1, triton.cdiv(topk, block_k))
    items = sq * (hq // BLOCK_H)
    target = _CTAS_PER_CU * n_cu
    if items >= target or nblk < _MIN_SPLIT_BLOCKS:
        return block_k, 1, nblk
    nsplit = min(nblk, _MAX_SPLIT, triton.cdiv(target, items))
    chunk = triton.cdiv(nblk, nsplit)
    return block_k, triton.cdiv(nblk, chunk), chunk


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
    block_k, nsplit, chunk = _plan(SQ, HQ, TOPK, DQK, _cu_count(device_index))
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
        _FWD_KERNEL[(SQ * nsplit * (HQ // BLOCK_H),)](
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
            BH=BLOCK_H,
            BK=block_k,
            BD=BLOCK_D,
            DV=DV,
            num_warps=_NUM_WARPS,
            num_stages=1,  # staging the gather doubles LDS without hiding it
            waves_per_eu=1,  # one register budget to aim at; anything higher
            # caps the kernel below the 256 VGPRs it already needs and spills
            # hundreds of values (416 at 192 VGPRs)
            **_backend_launch_options(DQK, nsplit > 1),
        )
        if nsplit > 1:
            _hcu_sparse_mla_combine_kernel[(SQ * HQ // 8,)](
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
