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
# The KV tile is [BK, DV] of shared memory, so the single-block path above would
# need 32 KiB of the 64 KiB and only two work groups would fit per CU. Gathering
# that path in twice as many chunks halves the live tile, which buys a third and
# fourth work group and 1.08-1.11x on the bandwidth-bound TOPK=128 rows. It only
# helps where occupancy binds: the multi-block path pays for the extra chunks on
# every iteration (up to 20%), and too few tokens to fill the CUs leaves the
# extra work groups with nothing to do.
BLOCK_D_SINGLE = 64
_NUM_WARPS = 4
# Stacked waves per SIMD. Above 4 the budget stops fitting the 256 VGPRs the
# accumulator needs and the spills return (94 at 5, 506 at 8), so the useful
# range is narrow. Three is worth 3.5% on the DQK=512 multi-block path and is
# applied there only; see the launch site.
_WAVES_PER_EU = 3
_NUM_STAGES = 1
# Key blocks between rescales. A rescale is only needed when the running max
# grows, but detecting that costs a cross-wave max over [BH] - a reduction plus
# an s_barrier every iteration. A fixed cadence turns it into a uniform branch
# on the loop counter.
#
# `max_logits` is unaffected: m_i still tracks the exact running max and only
# the max the accumulator is scaled against (m_ref) lags, which the epilogue
# divides back out.
RESCALE_PERIOD = 32
# Read the KV tile twice - once already transposed for QK, once natural for PV -
# rather than transposing it in registers.
#
# QK wants k^T [BD, BK] and PV wants k [BK, BD], so one of them has to be
# turned. Doing it in registers costs an LDS round trip on the critical path
# (120 ds_write + 120 ds_read per iteration against 128 mmac) and is what
# presses the live tile into spilling. Reading the same bytes with the indices
# swapped gives QK its operand straight from memory, and PV re-reads the tile
# the first pass left in L1/L2.
#
# Only affordable where the gather is not the constraint, which is the case on
# the DeepSeek-V4.1-Flash replay: the KV is 344 MiB there yet costs 107us of
# wall time, so the kernel is compute-bound with bandwidth to spare.
QK_TRANSPOSED = True
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
# Splitting only fills the gap between the live work items and the occupancy
# target, while its cost - a second launch plus a [HQ, DV] f32 partial per split
# - does not shrink with that gap. Half-empty is where the two meet: at 64 of
# the 128 work items splitting is still worth 1.13-1.59x, at 96 and 120 it
# already costs 10-21%.
_MIN_SPLIT_GAIN = 2
# Same idea for the short rows, where splitting also has to give up the peeled
# single-block body: there the turnover is an eighth of the target rather than
# half of it. See _plan.
_SHORT_SPLIT_GAIN = 16


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
    RESCALE_PERIOD: tl.constexpr,
    QK_TRANSPOSED: tl.constexpr,
):
    """One work item: BH heads of one query token over `chunk` key blocks."""
    NHB: tl.constexpr = HQ // BH
    NBD: tl.constexpr = DV // BD
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
    qs = ()
    for c in tl.static_range(NBD):
        qs = qs + (tl.load(q_ptr + c * BD + offs_d[None, :]),)
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
    # The max the accumulator and l_i are carried against; see the rescale
    # comment in the loop body. Identical to m_i when the period is 0.
    m_ref = tl.full([BH], M_FLOOR, dtype=tl.float32)
    l_i = tl.zeros([BH], dtype=tl.float32)
    accs = ()
    for _c in tl.static_range(NBD):
        accs = accs + (tl.zeros([BH, BD], dtype=tl.float32),)
    t_base = indices + tok * stride_tm

    for ck in range(lo, hi):
        t_offs = ck * BK + offs_t
        ids = tl.load(t_base + t_offs, mask=t_offs < topk_len, other=-1)
        valid = (ids >= 0) & (ids < SKV)
        ids = tl.where(valid, ids, 0)
        kv_ptr = kv + ids[:, None].to(tl.int64) * stride_kvn

        # whole KV rows, one chunk per load: all NBD loads are issued before the
        # first dot, which is what gives the gather its memory-level parallelism
        if QK_TRANSPOSED:
            # Each `t` is one KV row and each `d` a contiguous run along it, so
            # swapping the indices yields [BD, BK] directly - the same bytes,
            # just a different order - and the QK operand needs no transpose.
            # PV still wants the natural [BK, BD] and reads it back below, from
            # the L1/L2 lines this pass leaves behind.
            i64 = ids.to(tl.int64) * stride_kvn
            base_t = kv + i64[None, :]
            ks = ()
            for c in tl.static_range(NBD):
                ks = ks + (tl.load(base_t + c * BD + offs_d[:, None]),)
            s = tl.dot(qs[0], ks[0])
            for c in tl.static_range(1, NBD):
                s = tl.dot(qs[c], ks[c], s)
            if DQK == 576:
                kt = tl.load(kv_ptr + DV + offs_td[None, :])
                s = tl.dot(qt, tl.trans(kt), s)
        else:
            ks = ()
            for c in tl.static_range(NBD):
                ks = ks + (tl.load(kv_ptr + c * BD + offs_d[None, :]),)

            s = tl.dot(qs[0], tl.trans(ks[0]))
            for c in tl.static_range(1, NBD):
                s = tl.dot(qs[c], tl.trans(ks[c]), s)
            if DQK == 576:
                kt = tl.load(kv_ptr + DV + offs_td[None, :])
                s = tl.dot(qt, tl.trans(kt), s)
        s = tl.where(valid[None, :], s * sm_scale_log2, float("-inf"))

        # Rescale on a fixed cadence rather than on a data-dependent gate.
        #
        # m_i stays the exact running max (it is only ever raised), so it is
        # what max_logits reports. The accumulator and l_i are carried against
        # m_ref, which may trail it by up to RESCALE_PERIOD blocks: p is then a
        # factor 2^(m_i - m_ref) larger than exact, which f32 absorbs and the
        # epilogue divides back out.
        blk_max = tl.max(s, axis=1)
        if RESCALE_PERIOD == 0:
            # Peeled single-block path: the loop runs once, so accs and l_i are
            # still zero on entry and there is nothing to renormalise.
            m_i = tl.maximum(m_i, blk_max)
            m_ref = m_i
        else:
            m_i = tl.maximum(m_i, blk_max)
            # Counted from the work item's first block: a split starting off a
            # period boundary would otherwise run with m_ref at M_FLOOR and
            # overflow. The last block always rescales, so that p is cast to
            # bf16 against the exact max.
            if ((ck - lo) % RESCALE_PERIOD == 0) | (ck == hi - 1):
                alpha = tl.math.exp2(m_ref - m_i)
                l_i = l_i * alpha
                rescaled = ()
                for c in tl.static_range(NBD):
                    rescaled = rescaled + (accs[c] * alpha[:, None],)
                accs = rescaled
                m_ref = m_i

        if RESCALE_PERIOD == 0:
            # m_ref is m_i here; naming it would keep a second copy live.
            p = tl.math.exp2(s - m_i[:, None])
        else:
            p = tl.math.exp2(s - m_ref[:, None])
        l_i += tl.sum(p, axis=1)
        pb = p.to(tl.bfloat16)
        if QK_TRANSPOSED:
            # PV wants the natural [BK, BD]; re-read it. The first pass left it
            # in L1/L2, so this is a hit rather than a second trip to DRAM.
            pvp = kv + i64[:, None] + offs_d[None, :]
            updated = ()
            for c in tl.static_range(NBD):
                updated = updated + (tl.dot(pb, tl.load(pvp + c * BD), accs[c]),)
        else:
            updated = ()
            for c in tl.static_range(NBD):
                updated = updated + (tl.dot(pb, ks[c], accs[c]),)
        accs = updated

    row_valid = l_i > 0
    inv_l = tl.where(row_valid, 1.0 / l_i, 0.0)
    if PARTIAL:
        # partial of (token, split): normalized O, (max, sum) in log2 units
        prow = (tok * nsplit + split) * HQ + h0 + offs_h
        op_ptr = o_part + prow[:, None] * DV
        for c in tl.static_range(NBD):
            tl.store(op_ptr + c * BD + offs_d[None, :], accs[c] * inv_l[:, None])
        # The combiner assumes acc and the sum share one max, so hand it m_ref.
        # With the period at 0 the two are the same tensor, and naming m_ref
        # here would keep a second copy alive across the whole kernel.
        tl.store(ml_part + prow * 2, m_ref if RESCALE_PERIOD > 0 else m_i)
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
        for c in tl.static_range(NBD):
            tl.store(
                o_ptr + c * BD + offs_d[None, :],
                (accs[c] * factor[:, None]).to(tl.bfloat16),
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
def _backend_launch_options(dqk: int, split: bool, single: bool) -> dict:
    """Backend-specific launch options, skipped on backends that lack them.

    ``kpack=2`` widens the per-lane K footprint of the MFMA operands, which pays
    off for the 512-wide rows (5-12%) but regresses the 576-wide ones, so it is
    selected per DQK.

    ``sched_latency`` makes the scheduler assume a longer LDS latency, so it
    hoists the ds_reads of the PV operand away from the mmacs that consume
    them; 23 of the inner loop's 44 ``s_waitcnt`` are otherwise a fully
    blocking ``lgkmcnt(0)``, and with only 3 work groups per CU there is no
    other wave to cover that latency. How far to go depends on how much loop
    there is to hoist across: the multi-block loop wants ``ds80`` (1.05-1.09x
    over ``ds32``), while the peeled single-block body has nothing to hide
    behind and the same value costs it 2-4%.

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
            # The latency tier interacts with the waves-per-SIMD hint - both are
            # handed to the same instruction scheduler - so they are tuned as a
            # pair. DQK=512 multi-block runs at three waves and wants ds40
            # (1.146x, against 1.095x at ds64); the single-block path keeps its
            # peeled tuning, and DQK=576 stays at one wave, where ds80 remains
            # the best tier (1.031x vs 0.983x at ds40).
            if single:
                tier = 64
            elif dqk == 512:
                tier = 40
            else:
                tier = 80
            options["sched_latency"] = f"mmac5-ds{tier}"
        return options
    return {}


@functools.lru_cache(maxsize=None)
def _cu_count(device_index: int) -> int:
    return torch_device_fn.get_device_properties(device_index).multi_processor_count


# Compiled launchers and their argument templates, keyed the same way `_plan`
# is. `LibEntry` re-derives a cache key from all 34 kernel parameters on every
# call - specializing each tensor argument individually, which walks the
# storage of every pointer operand - and that costs ~58us of host time against
# a ~7us launch. The device work on the short rows is well under that, so the
# Python entry overhead is the whole latency there: 72.2us of host time per
# call becomes 17.5us, which is also where the vendor's own entry sits.
#
# A captured template is the launcher's own argument tuple. Only the operand
# slots and the stream are rewritten on replay; everything else (grid, kernel
# handle, metadata, hooks, strides, extents) is fixed for a given key, which is
# why the key has to name every one of them.
_LAUNCH_CACHE = {}

# Argument layout of the generated C entry point, which `HIPLauncher.__call__`
# only prefixes two fixed values onto:
#
#   launch(coop, gridX, gridY, gridZ, stream, function, profile_scratch,
#          packed_metadata, launch_metadata, enter_hook, exit_hook, *kernel_args)
#
# so kernel argument i sits at _ARG0 + i and the stream at _STREAM_SLOT.
_STREAM_SLOT = 4
_ARG0 = 11

_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)

# Triton specializes every pointer argument on 16-byte alignment, and this
# backend additionally on whether the storage fits the 32-bit addressing that
# buffer ops use (`HIPBackend.get_tensor_specialization`). Both are properties
# of the tensor rather than of its shape - a view into a larger cache has the
# same shape and a different specialization - so a captured launcher is only
# valid for operands that agree on them, and they have to be part of the key.
# The outputs are allocated here and always agree; only the caller's do not.
_PTR_RANGE_MAX = 2**31 - 1


def _spec_bits(tensors):
    bits = 0
    for t in tensors:
        bits <<= 2
        if t is not None:
            if t.data_ptr() % 16:
                bits |= 1
            if t.untyped_storage().size() > _PTR_RANGE_MAX:
                bits |= 2
    return bits


# Partial-accumulator scratch for the split path. It never escapes the op, so it
# is kept across calls: the two torch.empty calls cost ~5us of host time against
# a ~7us kernel launch, and on the rows this path serves the host is most of the
# latency. One base buffer per (device, stream) grows to the largest shape seen
# and the per-shape views are cached, so a steady-state call is a dict lookup.
# Keying on the stream keeps concurrent callers from sharing an accumulator;
# within one stream the launches are ordered, so reuse is safe.
_WORKSPACE = {}


def _split_workspace(device, stream, rows, dv):
    state = _WORKSPACE.get((device, stream))
    if state is None:
        state = _WORKSPACE[(device, stream)] = [None, None, {}]
    else:
        pair = state[2].get(rows)
        if pair is not None:
            return pair
    views = state[2]
    if state[0] is None or state[0].numel() < rows * dv:
        state[0] = torch.empty(rows * dv, device=device, dtype=torch.float32)
        views.clear()
    if state[1] is None or state[1].numel() < rows * 2:
        state[1] = torch.empty(rows * 2, device=device, dtype=torch.float32)
        views.clear()
    pair = (
        torch.as_strided(state[0], (rows, dv), (dv, 1)),
        torch.as_strided(state[1], (rows, 2), (2, 1)),
    )
    views[rows] = pair
    return pair


def _capture_launch(run_through_entry, n_operands):
    """Run one ordinary launch, returning (C entry, argument template) for it.

    The `n_operands` leading kernel arguments are the ones a replay rewrites,
    so they are cleared out of the template: a template that kept them would
    pin the first call's tensors, and a captured KV cache is never freed.

    Returns None when the launch cannot be replayed - a non-HCU build, a
    kernel that wants per-launch profile scratch, or anything that did not
    resolve to exactly one launcher call.
    """
    try:
        from triton.backends.hcu.driver import HIPLauncher
    except ImportError:  # not an HCU build, keep the portable path
        run_through_entry()
        return None

    captured = []
    original = HIPLauncher.__call__

    def spy(self, *call_args):
        captured.append((self, call_args))
        return original(self, *call_args)

    HIPLauncher.__call__ = spy
    try:
        run_through_entry()
    finally:
        HIPLauncher.__call__ = original

    if len(captured) != 1:
        return None
    launcher, call_args = captured[0]
    if launcher.profile_scratch_size > 0:
        return None  # the scratch allocation needs the Python wrapper
    template = [launcher.launch_cooperative_grid]
    template += call_args[:5]  # grid x3, stream, function
    template.append(None)  # profile scratch
    template += call_args[5:]  # metadata, hooks, kernel arguments
    for slot in range(_ARG0, _ARG0 + n_operands):
        template[slot] = None
    return launcher.launch, template


@functools.lru_cache(maxsize=None)
def _plan(
    sq: int, hq: int, topk: int, dqk: int, n_cu: int
) -> Tuple[int, int, int, int]:
    """(block_k, block_d, nsplit, chunk): a work item covers `chunk` key blocks.

    TOPK is only split when BH-head items cannot fill the CUs, which is the
    regime the latency is dominated by idle CUs rather than by the gather.
    Splitting costs a second kernel launch, which the replayed launcher brings
    down to ~7us of host time, so it has to buy back about that much of device
    time.
    """
    items = sq * (hq // BLOCK_H)
    target = _CTAS_PER_CU * n_cu
    saturated = items * _MIN_SPLIT_GAIN > target
    # The wide block that lets a short TOPK be peeled is straight-line code that
    # does not spill, and that is worth more per unit of work than the CUs it
    # leaves idle - until there are almost no work items at all. Measured on
    # TOPK=128 against the vendor, trading the peeled block for a narrow one
    # split four ways: 0.83x -> 0.98x at 4 items and 0.83x -> 0.88x at 8, then
    # 0.83x -> 0.78x at 12 and 0.82x -> 0.54x at 32. Only the first eighth of
    # the occupancy target is desperate enough.
    if topk <= BLOCK_K_SINGLE and items * _SHORT_SPLIT_GAIN > target:
        block_k = BLOCK_K_SINGLE
    else:
        block_k = BLOCK_K_BY_DQK[dqk]
    nblk = max(1, triton.cdiv(topk, block_k))
    if saturated or nblk < _MIN_SPLIT_BLOCKS:
        # The narrow value block only pays off where the loop runs once and the
        # CUs are saturated, which is what makes occupancy the binding
        # constraint; see BLOCK_D_SINGLE.
        narrow = nblk == 1 and items >= target
        return block_k, BLOCK_D_SINGLE if narrow else BLOCK_D, 1, nblk
    nsplit = min(nblk, _MAX_SPLIT, triton.cdiv(target, items))
    chunk = triton.cdiv(nblk, nsplit)
    return block_k, BLOCK_D, triton.cdiv(nblk, chunk), chunk


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
    stream = _raw_stream(device_index) if _raw_stream is not None else None
    block_k, block_d, nsplit, chunk = _plan(SQ, HQ, TOPK, DQK, _cu_count(device_index))
    if nsplit > 1:
        if stream is None:  # cannot tell callers apart, do not share scratch
            o_part = torch.empty(
                (SQ * nsplit * HQ, DV), device=q.device, dtype=torch.float32
            )
            ml_part = torch.empty(
                (SQ * nsplit * HQ, 2), device=q.device, dtype=torch.float32
            )
        else:
            o_part, ml_part = _split_workspace(q.device, stream, SQ * nsplit * HQ, DV)
    else:
        o_part = ml_part = output  # unused

    # Everything the template holds fixed has to be part of the key: the plan,
    # the shapes it was specialized against, the two optional operands that
    # select a kernel variant, and the per-tensor specialization bits.
    key = (
        SQ,
        HQ,
        TOPK,
        DQK,
        SKV,
        nsplit,
        chunk,
        block_k,
        block_d,
        attn_sink is not None,
        topk_length is not None,
        _spec_bits((q, kv, indices, attn_sink, topk_length)),
    )
    entry = _LAUNCH_CACHE.get(key) if _raw_stream is not None else False

    if entry is None or entry is False:
        # Only the cold and portable paths need the strides; a replay keeps them.
        q_s0, q_s1, _ = q.stride()
        kv_s0 = kv.stride(0)
        idx_s0 = indices.stride(0)
        out_s0, out_s1, _ = output.stride()
        ml_s0 = max_logits.stride(0)
        lse_s0 = lse.stride(0)

    def _fwd_args():
        return (
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
            q_s0,
            q_s1,
            kv_s0,
            idx_s0,
            out_s0,
            out_s1,
            ml_s0,
            lse_s0,
            SKV,
            TOPK,
            nsplit,
            chunk,
        )

    def _fwd_launch():
        return _FWD_KERNEL[(SQ * nsplit * (HQ // BLOCK_H),)](
            *_fwd_args(),
            HQ=HQ,
            DQK=DQK,
            HAVE_ATTN_SINK=attn_sink is not None,
            HAVE_TOPK_LENGTH=topk_length is not None,
            PARTIAL=nsplit > 1,
            BH=BLOCK_H,
            BK=block_k,
            BD=block_d,
            DV=DV,
            RESCALE_PERIOD=RESCALE_PERIOD if chunk > 1 else 0,
            # Only where the loop actually iterates. With one key block the
            # tile is read once and never re-read from cache, so the second
            # read is a second miss: measured 0.831x -> 0.735x on the TOPK=128
            # band. Across the replay's TOPK=640 rows the same switch is worth
            # 1.022x -> 1.067x.
            #
            # Not at DQK=576. The tail there is a [64, BK] read and turning it
            # transposed is not the problem - leaving just the tail natural and
            # transposing everything else still measures 0.65-0.68x across the
            # whole TOPK=2048 band, so the combination of the deeper K extent
            # with a second read of the tile is what loses. DQK=512 only.
            QK_TRANSPOSED=QK_TRANSPOSED and DQK == DV and chunk > 1,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
            # Three waves only pays on the DQK=512 multi-block path, which is the
            # one the transposed read also serves: both are the same register
            # budget landing in a better place. Measured there 1.057x -> 1.093x
            # (1478 vs 1530us). It is flat on the peeled single-block path
            # (0.832x either way) and costs 7% at DQK=576 (1.001x -> 0.934x), so
            # it is scoped the same way QK_TRANSPOSED is.
            waves_per_eu=(
                _WAVES_PER_EU if QK_TRANSPOSED and DQK == DV and chunk > 1 else 1
            ),
            **_backend_launch_options(DQK, nsplit > 1, chunk == 1),
        )

    def _combine_launch():
        return _hcu_sparse_mla_combine_kernel[(SQ * HQ // 8,)](
            o_part,
            ml_part,
            attn_sink,
            output,
            max_logits,
            lse,
            out_s0,
            out_s1,
            ml_s0,
            lse_s0,
            HQ=HQ,
            DV=DV,
            NSPLIT=nsplit,
            HAVE_ATTN_SINK=attn_sink is not None,
            BHC=8,
            num_warps=4,
            num_stages=1,
        )

    if entry is None:
        with torch_device_fn.device(q.device):
            fwd = _capture_launch(_fwd_launch, 11)
            cmb = _capture_launch(_combine_launch, 6) if nsplit > 1 else None
        # Either capture coming back empty means the launch is not replayable;
        # remember that so the portable path is taken without retrying.
        _LAUNCH_CACHE[key] = (
            (fwd[0], fwd[1], cmb) if fwd is not None and (nsplit == 1 or cmb) else False
        )
        return output, max_logits, lse

    if entry is False:  # portable path
        with torch_device_fn.device(q.device):
            _fwd_launch()
            if nsplit > 1:
                _combine_launch()
        return output, max_logits, lse

    fwd_launch, fwd_template, cmb_entry = entry
    if torch_device_fn.current_device() != device_index:
        torch_device_fn.set_device(device_index)

    args = fwd_template[:]
    args[_STREAM_SLOT] = stream
    args[_ARG0 : _ARG0 + 11] = (
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
    )
    fwd_launch(*args)

    if cmb_entry is not None:
        cmb_launch, cmb_template = cmb_entry
        args = cmb_template[:]
        args[_STREAM_SLOT] = stream
        args[_ARG0 : _ARG0 + 6] = (
            o_part,
            ml_part,
            attn_sink,
            output,
            max_logits,
            lse,
        )
        cmb_launch(*args)

    return output, max_logits, lse
