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

"""Ascend KDA prefill kernels (BT=64, chunk-major workspaces).

Two forward paths with identical semantics share one public entry, ``chunk_kda``:

* TLE path (``ascend_chunk_kda_fwd_infer``): three kernels mirroring the
  fla-npu AscendC ``aclnnChunkKdaFwdV2`` chain -- Prepare (gate cumsum +
  intra-chunk Gram matrices + triangular inverse + workspace pack), FwdH (the
  serial 3-dot chunk recurrence) and Finalize (the fully chunk-parallel
  ``o = scale*qg@h + Aqk@v_new``).  All intermediates are chunk-major, so each
  chunk-head is one contiguous 8-48 KiB burst instead of BT strided rows.
* Triton fallback (``chunk_kda_fwd_infer_triton``): portable plain-Triton
  kernels shared verbatim with ``flaggems_vllm.ops.FLA.chunk_kda``.  Used when
  TLE is unavailable or the chunk-major h buffer would exceed its budget.

``chunk_kda`` validates inputs first, then dispatches: TLE when
available, and Triton otherwise. Set ``FLAGGEMS_CHUNK_KDA_BACKEND`` to
``tle`` or ``triton`` to force a backend; the default is ``auto``.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.FLA.chunk_kda import (
    _validate_chunk_kda_inputs,
    chunk_kda_fwd_infer_triton,
)
from flaggems_vllm.ops.FLA.index import prepare_chunk_indices
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

# The kernels below need exactly two slice primitives. FlagTree provides them
# as ``tle.dsa.extract_slice`` / ``insert_slice``; stock triton-ascend ships the
# same two ops (same offsets/sizes/strides signature, verified equivalent on a
# sequential cumsum) under ``triton.language.extra.cann.extension``. Prefer TLE
# when present so FlagTree behaviour is unchanged, and fall back to the
# extension API so a stock triton-ascend build runs these kernels too instead of
# dropping to the portable Triton path.
_extract_slice = None
_insert_slice = None
_SLICE_SOURCE = "none"

if has_triton_tle(3, 5, 0):
    try:
        import triton.experimental.tle as tle

        _extract_slice = tle.dsa.extract_slice
        _insert_slice = tle.dsa.insert_slice
        _SLICE_SOURCE = "tle"
    except ImportError:
        tle = None
else:
    tle = None

if _extract_slice is None:
    try:
        from triton.language.extra.cann.extension import extract_slice as _extract_slice
        from triton.language.extra.cann.extension import insert_slice as _insert_slice

        _SLICE_SOURCE = "cann_extension"
    except ImportError:
        _extract_slice = None
        _insert_slice = None

HAS_TLE_KDA = _extract_slice is not None

__all__ = ["chunk_kda"]

# =============================================================================
# Shared helpers
# =============================================================================

RCP_LN2 = 1.4426950216
MAX_GRID_BLOCKS = 32768

# Internal chunk width for the Ascend kernels. The caller-visible chunk_size
# stays 16 (the cross-backend contract); only the tile the Ascend kernels walk
# is widened, which is safe because both the intra and state kernels recompute
# everything they need per tile from q/k/g rather than from a BT=16 layout.
#
# BT trades the serial chunk recurrence against the factored-decay range:
# the gate span grows ~linearly in BT, and Aqk/Akk operands are cast to bf16
# (8 mantissa bits) for tl.dot. A single chunk-wide midpoint anchor keeps
# that span bit-exact through BT=32 but goes non-finite at BT=64 (3.5M-17.8M
# non-finite outputs, see tools/_kda_bt64_subblock_math_check.py for a CPU
# repro). _ascend_kda_fwd_intra_kernel splits BT=64 into two 32-row
# sub-blocks, each rebased on its own local midpoint (mirrors
# AscendC arch35's 32-row score reference span -- see
# flash-linear-attention-npu/.../chunk_kda_fwd_prepare.h ScoreRefToken), which
# keeps every span within the BT=32 bit-exact envelope while still scanning
# in 64-row steps. Verified on Ascend950DT (tools/_kda_bt_accuracy.py: 0
# non-finite at BT=64 with the split, vs. NaN across all 4 harness shapes
# without it) and faster throughout (tools/_kda_shape_bench.py chunk, us):
#   T=8192  H=24 seq=2   BT=32 2632.7   BT=64 2239.2   (1.18x)
#   T=16384 H=12 seq=4   BT=32 2655.5   BT=64 2241.1   (1.18x)
#   T=24576 H=6  seq=8   BT=32 2010.5   BT=64 1700.7   (1.18x)
#   T=32768 H=24 seq=2   BT=32 10667.6  BT=64 8915.1   (1.20x)
_ASCEND_BT = 64
_ASCEND_BT_ENV = "FLAGGEMS_ASCEND_KDA_BT"


def _ascend_chunk_width() -> int:
    """Chunk width the Ascend KDA kernels tile with (override for tuning)."""
    raw = os.environ.get(_ASCEND_BT_ENV)
    if not raw:
        return _ASCEND_BT
    try:
        val = int(raw)
    except ValueError:
        return _ASCEND_BT
    # The intra kernel evaluates the Gram matrices in two BT//2 sub-blocks;
    # sub-blocks below 16 rows cannot feed tl.dot, so 16 is no longer a valid
    # internal width (it maps back to the default).
    return val if val in (32, 64) else _ASCEND_BT


# Neumann truncation order for (I+L)^-1 in _ascend_kda_fwd_intra_kernel.
# L (our b_Akk) is strictly lower triangular, so L^BT == 0 and the series
# terminates exactly; truncation error is the only error source, bounded by
# ||L||^(stage_order+1) / (1-||L||). The doubling scheme (I-L, +Ai@L2,
# +Ai@L4, +Ai@L8) reaches effective order 1/3/7/15 at stage 0/1/2/3.
#
# tools/_kda_neumann_order_check.py measured max||L||_2 under our actual
# safe-gate distribution (q/k/v/g/beta matching tools/_kda_shape_bench.py):
# ~0.02-0.025 across BT in {16,32,64} and a beta-stress sweep. At that norm
# the *stage-0* remainder bound (||L||^2/(1-||L||), the r=1 case) is already
# ~4-6e-4, an order of magnitude below bf16 eps (~3.9e-3), so stage 0 (I-L,
# 0 extra dots) was tried against the 3-stage default on Ascend950DT:
#   tools/_kda_bt_accuracy.py: identical max_abs/state max_abs at every BT
#   and shape (0 non-finite either way) -- no measurable accuracy loss.
#   tools/test_kda_vs_vllm_ascend.py: 15/15 correct; chunk_kda speedup vs
#   AscendC improved at every shape (T=8192/16384/24576: 1.37-1.41x ->
#   1.50-1.55x; T=32768: 0.41x -> 0.45x).
# So 0 extra dots is the default; set >0 only if a different gate/beta/k
# distribution needs the extra margin (re-run _kda_neumann_order_check.py
# first to confirm ||L|| actually warrants it before paying the dot cost).
_ASCEND_NEUMANN_STAGES = 0
_ASCEND_NEUMANN_STAGES_ENV = "FLAGGEMS_ASCEND_KDA_NEUMANN_STAGES"


def _ascend_neumann_stages() -> int:
    """Doubling stages for the (I+L)^-1 Neumann series (override for tuning)."""
    raw = os.environ.get(_ASCEND_NEUMANN_STAGES_ENV)
    if not raw:
        return _ASCEND_NEUMANN_STAGES
    try:
        val = int(raw)
    except ValueError:
        return _ASCEND_NEUMANN_STAGES
    return val if val in (0, 1, 2, 3) else _ASCEND_NEUMANN_STAGES


# The state scan used to carry an experimental ``tl.range(num_stages=...)``
# switch.  Measured on Ascend950DT with T=32768 H=24 (tools/_kda_shape_bench.py)
# num_stages 0/1/2/3 all gave the same 3.14 ms state-kernel time, i.e. the
# triton-ascend backend does not software-pipeline this loop (matching the
# category note "program 间不重叠、num_stages 无效").  It was removed rather
# than kept as a dead knob.
#
# The state kernel is split into two kernels, mirroring fla-npu's AscendC
# ``aclnnChunkKdaFwdV2`` chain (Prepare -> ChunkFwdH -> ChunkKdaFwdFinalize):
# the serial chunk scan (_ascend_kda_fwd_state_kernel) only carries the true
# recurrence (3 dots/chunk), while the o = qg@h + Aqk@v_new half runs in a
# fully chunk-parallel kernel (_ascend_kda_fwd_output_kernel, 2 dots/chunk).
# FLAGGEMS_ASCEND_KDA_FUSED_STATE=1 forces the portable chunk-at-a-time Triton
# path instead (A/B measurement; also the memory fallback below).
_FUSED_STATE_ENV = "FLAGGEMS_ASCEND_KDA_FUSED_STATE"


def _ascend_fused_state() -> bool:
    """Use the fused state+output scan instead of FwdH + Finalize."""
    return os.environ.get(_FUSED_STATE_ENV, "0").strip().lower() not in (
        "",
        "0",
        "false",
        "off",
        "no",
    )


# Budget for the chunk-major h buffer Finalize consumes.  h is [chunks, HV, K,
# V] bf16: 402 MiB for the T=32768 H=24 case, the same intermediate the AscendC
# V2 chain materializes.  Above this budget the portable chunk-at-a-time Triton
# path runs instead so an unexpectedly large (shape, head count) combination
# cannot OOM the device.
_H_BUDGET_ENV = "FLAGGEMS_ASCEND_KDA_H_BUDGET_MB"
_H_BUDGET_MB = 4096


def _ascend_h_budget_bytes() -> int:
    raw = os.environ.get(_H_BUDGET_ENV)
    if not raw:
        return _H_BUDGET_MB << 20
    try:
        val = int(raw)
    except ValueError:
        return _H_BUDGET_MB << 20
    return max(val, 0) << 20


_BACKEND_ENV = "FLAGGEMS_CHUNK_KDA_BACKEND"
_BACKEND_ALIASES = {
    "auto": "auto",
    "tle": "tle",
    "generic_tle": "tle",
    "triton": "triton",
    "triton_fuse": "triton",
}


def _chunk_kda_backend() -> str:
    value = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    try:
        return _BACKEND_ALIASES[value]
    except KeyError as exc:
        choices = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(
            f"invalid {_BACKEND_ENV}={value!r}; expected one of: {choices}"
        ) from exc


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit
def _block_mul_lower(a00, a10, a11, b00, b10, b11):
    """Product of two block lower-triangular [2H, 2H] matrices.

    A = [[a00, 0], [a10, a11]], B likewise; returns (c00, c10, c11) with
    c10 = a10@b00 + a11@b10.  fp32 in/out, bf16 operands for the dots.
    """
    c00 = tl.dot(a00.to(tl.bfloat16), b00.to(tl.bfloat16), out_dtype=tl.float32)
    c11 = tl.dot(a11.to(tl.bfloat16), b11.to(tl.bfloat16), out_dtype=tl.float32)
    c10 = tl.dot(a10.to(tl.bfloat16), b00.to(tl.bfloat16), out_dtype=tl.float32)
    c10 += tl.dot(a11.to(tl.bfloat16), b10.to(tl.bfloat16), out_dtype=tl.float32)
    return c00, c10, c11


# =============================================================================
# TLE path -- default when available
# =============================================================================

if HAS_TLE_KDA:
    # Bind the resolved primitives to module-level names so the @triton.jit
    # kernels below capture them as globals regardless of which source
    # (FlagTree TLE or the cann extension) provided them.
    extract_slice = _extract_slice
    insert_slice = _insert_slice

    # =================================================================
    # Workspace layouts -- all chunk-major, mirroring the AscendC V2
    # chain's (`aclnnChunkKdaFwdV2`: Prepare -> ChunkFwdH ->
    # ChunkKdaFwdFinalize) internal tensors.
    #
    # ``gid`` is the canonical chunk id: sequence-major for varlen (the row
    # index into ``chunk_indices``), ``i_n * NT + i_t`` for dense.
    #
    #   ws       [NC, HV, BT, 3K]  bf16   : w | kg | qg
    #   Aqk      [NC, HV, BT, BT]  bf16
    #   Akk      [NC, HV, BT, BT]  bf16   : (I+L)^-1, already applied
    #   g_out    [B, NC*BT, HV, K] fp32   : chunk-local gate cumsum (FwdH reads
    #                                      only each chunk's last row)
    #   v_new    [NC, HV, BT, V]   bf16
    #   h_buf    [NC, HV, K, V]    bf16   : state at the *start* of the chunk
    #
    # Chunk-major matters.  The old token-major form stored one chunk-head as
    # BT x (BT*2 B) rows strided by HV*BT (Aqk/Akk) and BT x (256 B) rows
    # strided by HV*3K (ws); the serial scan's profiler ratios were
    # MTE2 49% / scalar 36% and almost no MAC, i.e. it was paying for exactly
    # those short strided bursts.  Here each chunk-head is one contiguous
    # 8 KiB (Aqk/Akk) / 48 KiB (ws) / 16 KiB (v_new) / 32 KiB (h) object,
    # matching AscendC's chunk-major h/w/A workspaces.
    # =================================================================

    def _pick_bv(K: int, V: int) -> int:
        # The chunk recurrence is serial, so BV decides how many waves of
        # programs each pay the full per-chunk dot chain: the grid is
        # (V/BV, N*HV). Taking BV = V runs that chain once per (sequence, head)
        # instead of 2x at BV=64. Measured on the T=24576 H=12 D=128 Ascend950DT
        # prefill: 5.66 ms (BV=32), 4.38 ms (BV=64), 3.90 ms (BV=128).
        if K <= 128:
            return min(V, 128)
        return 32

    # ------------------------------------------------------------------
    # Kernel 1: gate cumsum + intra-chunk grans + triangular inverse +
    # workspace packing.  Parallel over (chunk, head).
    #
    # Mirrors AscendC arch35 Prepare: both Gram matrices share their right
    # operand, and the gate decay is applied *after* the dot as the
    # [BT, BT] elementwise factor exp2(gc_i - gc_j) instead of being folded
    # into the bf16 operands.  On the causal half gc_i - gc_j <= 0, so the
    # factor is in [0, 1]: no bf16/fp32 overflow, no per-32-row re-basing,
    # and only two dots (q@k^T, k@k^T) per chunk instead of the six the
    # sub-block factored form needed at BT=64.
    # ------------------------------------------------------------------
    @triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
    @triton.jit(do_not_specialize=["T"])
    def _ascend_kda_fwd_intra_kernel(
        q,
        k,
        g,
        beta,
        ws,
        Aqk,
        Akk,
        g_out,
        chunk_offsets,
        A_log,
        dt_bias,
        lower_bound,
        scale,
        g_scale,
        l2norm_eps,
        cu_seqlens,
        chunk_indices,
        T,
        NC,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        CPT: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        NEUMANN_STAGES: tl.constexpr,
    ):
        i_tc, i_bh = tl.program_id(0), tl.program_id(1)
        i_hv = i_bh % HV
        i_h = i_hv // (HV // H)

        for c in tl.static_range(CPT):
            i_tg = i_tc * CPT + c
            if i_tg < NC:
                if IS_VARLEN:
                    i_n = tl.load(chunk_indices + i_tg * 2).to(tl.int32)
                    i_t = tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int32)
                    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
                    T = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
                    if (i_bh == 0) & (i_t == 0):
                        # canonical chunk -> [N] launch prefix, consumed by FwdH
                        tl.store(chunk_offsets + i_n, i_tg)
                else:
                    i_t = i_tg
                    bos = i_bh // HV * T

                if i_t * BT < T:
                    p_q = q + (bos * H + i_h).to(tl.int64) * K
                    p_k = k + (bos * H + i_h).to(tl.int64) * K
                    p_g = g + (bos * HV + i_hv).to(tl.int64) * K
                    p_g_out = g_out + (bos * HV + i_hv).to(tl.int64) * K
                    p_beta = beta + bos * HV + i_hv
                    p_ws = ws + (i_tg.to(tl.int64) * HV + i_hv) * BT * 3 * K
                    p_Aqk = Aqk + (i_tg.to(tl.int64) * HV + i_hv) * BT * BT
                    p_Akk = Akk + (i_tg.to(tl.int64) * HV + i_hv) * BT * BT

                    o_i = tl.arange(0, BT)
                    o_c = i_t * BT + o_i
                    m_c = o_c < T
                    o_k = tl.arange(0, K)

                    # [BT, K] tiles; q/k/g are the real (unpadded) [B, T, H/HV, K] tensors.
                    ptrs_qk = o_c[:, None] * (H * K) + o_k[None, :]
                    ptrs_g = o_c[:, None] * (HV * K) + o_k[None, :]
                    b_q = tl.load(p_q + ptrs_qk, mask=m_c[:, None], other=0.0)
                    b_k = tl.load(p_k + ptrs_qk, mask=m_c[:, None], other=0.0)
                    b_g = tl.load(p_g + ptrs_g, mask=m_c[:, None], other=0.0).to(
                        tl.float32
                    )

                    b_qf = b_q.to(tl.float32)
                    b_kf = b_k.to(tl.float32)

                    b_q_rstd = 1.0 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + l2norm_eps)
                    b_k_rstd = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + l2norm_eps)

                    b_A = exp2(tl.load(A_log + i_hv).to(tl.float32) * g_scale)
                    b_bias = tl.load(dt_bias + i_hv * K + o_k).to(tl.float32)
                    b_g = b_g + b_bias[None, :]
                    # FlashKDA-compatible safe gate; lower_bound is required by the verifier.
                    b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)

                    # Chunk-local gate prefix sum along the token axis.
                    # tl.cumsum lowers to a vectorized scan on triton-ascend
                    # (verified bit-exact against torch.cumsum at BT=16/32/64),
                    # replacing a BT-iteration extract_slice/insert_slice chain
                    # that serialized BT read-modify-writes of the [BT, K] tile.
                    b_gc = tl.cumsum(b_g, axis=0)

                    # The whole [BT, K] cumsum is stored even though FwdH reads
                    # only the chunk's last row: a compact [K] store at a
                    # scalar-computed offset was measured *2x slower* than this
                    # wide contiguous 2D burst (see the BT sweep in the history:
                    # 1395 -> 2948 us at T=8192 H=24), so the extra 402 MB of
                    # writes is cheaper than the small-store latency.
                    if IS_VARLEN:
                        tl.store(
                            p_g_out + o_c[:, None] * (HV * K) + o_k[None, :],
                            b_gc,
                            mask=m_c[:, None],
                        )
                    else:
                        tl.store(
                            p_g_out + o_c[:, None] * (HV * K) + o_k[None, :],
                            b_gc,
                        )

                    # Key-wise gate decay cannot be pulled out of the dot: the
                    # factor exp2(gc[i,k] - gc[j,k]) lives inside the K sum.
                    # Fold it into the bf16 operands, anchored per 32-row
                    # sub-block (mirrors AscendC arch35 Prepare's
                    # ScoreRefToken / 32-row reference span): acting as a
                    # single BT=64 anchor makes each side cover <=32 tokens of
                    # decay, which underflows the near-diagonal terms at the
                    # harness gate range.
                    HALF: tl.constexpr = BT // 2
                    in_sb0 = o_i[:, None] < HALF
                    in_sb1 = ~in_sb0
                    b_anchor0 = tl.sum(
                        tl.where(o_i[:, None] == HALF // 2, b_gc, 0.0), 0
                    )[None, :]
                    b_anchor1 = tl.sum(
                        tl.where(o_i[:, None] == HALF + HALF // 2, b_gc, 0.0), 0
                    )[None, :]
                    b_reb0 = b_gc - b_anchor0
                    b_reb1 = b_gc - b_anchor1
                    b_cross_corr = exp2(b_anchor1 - b_anchor0)
                    b_gq0 = tl.where(m_c[:, None] & in_sb0, exp2(b_reb0), 0.0)
                    b_gk0 = tl.where(m_c[:, None] & in_sb0, exp2(-b_reb0), 0.0)
                    b_gq1 = tl.where(m_c[:, None] & in_sb1, exp2(b_reb1), 0.0)
                    b_gk1 = tl.where(m_c[:, None] & in_sb1, exp2(-b_reb1), 0.0)
                    b_gk0_corr = tl.where(
                        m_c[:, None] & in_sb0, b_gk0 * b_cross_corr, 0.0
                    )

                    b_beta = tl.sigmoid(
                        tl.load(p_beta + o_c * HV, mask=m_c, other=0.0).to(tl.float32)
                    )

                    b_qg0_t = (b_qf * b_gq0).to(tl.bfloat16)
                    b_qg1_t = (b_qf * b_gq1).to(tl.bfloat16)
                    b_kg0_t = tl.trans((b_kf * b_gk0).to(tl.bfloat16))
                    b_kg1_t = tl.trans((b_kf * b_gk1).to(tl.bfloat16))
                    b_kg0_corr_t = tl.trans((b_kf * b_gk0_corr).to(tl.bfloat16))

                    m_Aqk = o_i[:, None] >= o_i[None, :]
                    b_Aqk = tl.dot(b_qg0_t, b_kg0_t, out_dtype=tl.float32)
                    b_Aqk += tl.dot(b_qg1_t, b_kg1_t, out_dtype=tl.float32)
                    b_Aqk += tl.dot(b_qg1_t, b_kg0_corr_t, out_dtype=tl.float32)
                    b_Aqk = b_Aqk * b_q_rstd[:, None] * b_k_rstd[None, :]
                    b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
                    tl.store(
                        p_Aqk + o_i[:, None] * BT + o_i[None, :],
                        b_Aqk.to(Aqk.dtype.element_ty),
                    )

                    m_Akk = o_i[:, None] > o_i[None, :]
                    m_I = o_i[:, None] == o_i[None, :]
                    b_Akk = tl.dot(
                        (b_kf * b_gq0).to(tl.bfloat16),
                        b_kg0_t,
                        out_dtype=tl.float32,
                    )
                    b_Akk += tl.dot(
                        (b_kf * b_gq1).to(tl.bfloat16),
                        b_kg1_t,
                        out_dtype=tl.float32,
                    )
                    b_Akk += tl.dot(
                        (b_kf * b_gq1).to(tl.bfloat16),
                        b_kg0_corr_t,
                        out_dtype=tl.float32,
                    )
                    b_Akk = b_Akk * b_k_rstd[:, None] * b_k_rstd[None, :]
                    b_Akk = tl.where(m_Akk, b_Akk * b_beta[:, None], 0.0)

                    # (I+L)^-1 via Neumann series: L is strictly lower
                    # triangular so L^BT == 0 and the series terminates
                    # exactly; truncation is the only error source, bounded
                    # by ||L||^(stage_order+1)/(1-||L||). The doubling scheme
                    # (I-L, +Ai@L2, +Ai@L4, +Ai@L8) reaches effective order
                    # 1/3/7/15 at stage 0/1/2/3.
                    #
                    # tools/_kda_neumann_order_check.py measured
                    # max||L||_2 ~ 0.02-0.025 under our actual safe-gate
                    # distribution across BT in {16,32,64} (q/k L2-normalized,
                    # so (k_i.k_j) in [-1,1] bounds ||L|| independent of the
                    # beta distribution -- also checked with beta stressed to
                    # raw U[0,1), same order of magnitude). At that norm the
                    # stage-0 (r=1, 0 extra dots) remainder bound is already
                    # ~4-6e-4, below bf16 eps (~3.9e-3); NEUMANN_STAGES lets
                    # this be tuned instead of always paying for 3 stages.
                    b_Ai = m_I.to(tl.float32) - b_Akk
                    if NEUMANN_STAGES >= 1:
                        b_L = b_Akk.to(tl.bfloat16)
                        b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float32).to(tl.bfloat16)
                        b_Ai = b_Ai + tl.dot(
                            b_Ai.to(tl.bfloat16), b_L2, out_dtype=tl.float32
                        )
                        if NEUMANN_STAGES >= 2:
                            b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float32).to(
                                tl.bfloat16
                            )
                            b_Ai = b_Ai + tl.dot(
                                b_Ai.to(tl.bfloat16), b_L4, out_dtype=tl.float32
                            )
                            if NEUMANN_STAGES >= 3:
                                b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float32).to(
                                    tl.bfloat16
                                )
                                b_Ai = b_Ai + tl.dot(
                                    b_Ai.to(tl.bfloat16), b_L8, out_dtype=tl.float32
                                )

                    tl.store(
                        p_Akk + o_i[:, None] * BT + o_i[None, :],
                        b_Ai.to(Akk.dtype.element_ty),
                    )

                    # Pack w | kg | qg into one contiguous [BT, 3K] chunk tile.
                    ptrs_ws = o_i[:, None] * (3 * K) + o_k[None, :]
                    b_k3 = b_kf * b_k_rstd[:, None]
                    b_q3 = b_qf * b_q_rstd[:, None]
                    last_local = tl.minimum(BT, T - i_t * BT) - 1
                    b_gn = tl.sum(tl.where(o_i[:, None] == last_local, b_gc, 0.0), 0)
                    b_w_val = b_k3 * b_beta[:, None] * exp2(b_gc)
                    b_kg_val = b_k3 * tl.where(
                        m_c[:, None], exp2(b_gn[None, :] - b_gc), 0.0
                    )
                    b_qg_val = b_q3 * exp2(b_gc)
                    tl.store(p_ws + ptrs_ws, b_w_val.to(ws.dtype.element_ty))
                    tl.store(p_ws + ptrs_ws + K, b_kg_val.to(ws.dtype.element_ty))
                    tl.store(p_ws + ptrs_ws + 2 * K, b_qg_val.to(ws.dtype.element_ty))

    def _ascend_kda_fwd_intra(
        q,
        k,
        g,
        beta,
        scale,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=16,
        lower_bound=None,
        A_log=None,
        dt_bias=None,
    ):
        B, T_len, H, K = q.shape
        HV = g.shape[2]
        BT = chunk_size

        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        if cu_seqlens is None:
            N = B
            NT = triton.cdiv(T_len, BT)
            NC = NT
            NC_global = B * NT
        else:
            N = len(cu_seqlens) - 1
            NC = len(chunk_indices)
            NC_global = NC
        BHV = B * HV

        # Cap the linear grid at MAX_GRID_BLOCKS. Each program then handles CPT
        # consecutive chunks; CPT == 1 reproduces the original grid exactly.
        CPT = triton.cdiv(NC * BHV, MAX_GRID_BLOCKS)
        grid0 = triton.cdiv(NC, CPT)
        while grid0 * BHV > MAX_GRID_BLOCKS:
            CPT += 1
            grid0 = triton.cdiv(NC, CPT)
        grid = (grid0, BHV)

        ws = torch.empty(NC_global, HV, BT, 3 * K, device=q.device, dtype=q.dtype)
        Aqk = torch.empty(NC_global, HV, BT, BT, device=q.device, dtype=q.dtype)
        Akk = torch.empty(NC_global, HV, BT, BT, device=q.device, dtype=q.dtype)
        # [token, HV, K] fp32 cumsum, shared layout with the FwdH gk read.
        g_out = torch.empty(
            1, NC_global * BT, HV, K, device=q.device, dtype=torch.float32
        )
        chunk_offsets = torch.empty(max(N, 1), device=q.device, dtype=torch.int32)

        _ascend_kda_fwd_intra_kernel[grid](
            q=q,
            k=k,
            g=g,
            beta=beta,
            ws=ws,
            Aqk=Aqk,
            Akk=Akk,
            g_out=g_out,
            chunk_offsets=chunk_offsets,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            scale=scale,
            g_scale=RCP_LN2,
            l2norm_eps=1e-6,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T_len,
            NC=NC,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            CPT=CPT,
            NEUMANN_STAGES=_ascend_neumann_stages(),
        )

        return ws, Aqk, Akk, g_out, chunk_offsets, NC, NC_global, N

    # ------------------------------------------------------------------
    # Kernel 2: FwdH -- the only serial chunk scan, 3 dots per chunk
    # (w@h, (I+L)^-1@diff, kg^T@v_new), exactly AscendC's ChunkFwdH.
    # qg/Aqk/o are no longer touched here; they moved to Finalize.
    # ------------------------------------------------------------------
    @triton.heuristics(
        {
            "USE_INITIAL_STATE": lambda args: args["h0"].numel() > 1,
            "STORE_FINAL_STATE": lambda args: args["ht"].numel() > 1,
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @triton.jit(do_not_specialize=["T"])
    def _ascend_kda_fwd_state_kernel(
        v,
        beta,
        gk,
        Akk,
        ws,
        v_new,
        h_buf,
        h0,
        ht,
        chunk_offsets,
        cu_seqlens,
        T,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        STATE_V_FIRST: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,
        STORE_FINAL_STATE: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        i_v, i_nh = tl.program_id(0), tl.program_id(1)

        if IS_VARLEN:
            i_n = i_nh // HV
            i_h = i_nh % HV
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
            NT = tl.cdiv(T, BT)
            hbase = tl.load(chunk_offsets + i_n).to(tl.int32)
        else:
            i_n = i_nh // HV
            i_h = i_nh % HV
            bos = i_n * T
            NT = tl.cdiv(T, BT)
            hbase = i_n * NT

        o_t = tl.arange(0, BT)
        o_k = tl.arange(0, K)
        o_v = tl.arange(0, BV)
        m_v = i_v * BV + o_v < V

        # Initial state h [K, BV] fp32.
        if USE_INITIAL_STATE:
            h0 += i_nh.to(tl.int64) * K * V
            if STATE_V_FIRST:
                b_h = tl.trans(
                    tl.load(
                        h0 + (i_v * BV + o_v)[:, None] * K + o_k[None, :],
                        mask=m_v[:, None],
                        other=0.0,
                    )
                ).to(tl.float32)
            else:
                b_h = tl.load(
                    h0 + o_k[:, None] * V + (i_v * BV + o_v)[None, :],
                    mask=m_v[None, :],
                    other=0.0,
                ).to(tl.float32)
        else:
            b_h = tl.zeros((K, BV), dtype=tl.float32)

        for i_t in range(NT):
            o_c = i_t * BT + o_t
            m_c = o_c < T
            gid = hbase + i_t

            # w | kg | qg, one contiguous chunk tile.
            p_ws = ws + (gid.to(tl.int64) * HV + i_h) * BT * 3 * K
            ptrs_w = o_t[:, None] * (3 * K) + o_k[None, :]
            b_w = tl.load(p_ws + ptrs_w)
            b_kg = tl.load(p_ws + ptrs_w + K)
            b_Akk = tl.load(
                Akk
                + (gid.to(tl.int64) * HV + i_h) * BT * BT
                + o_t[:, None] * BT
                + o_t[None, :]
            )
            gk_row = tl.minimum(i_t * BT + BT, T) - 1
            b_gk_last = tl.load(gk + ((bos + gk_row) * HV + i_h).to(tl.int64) * K + o_k)

            p_v = v + ((bos + i_t * BT) * HV + i_h).to(tl.int64) * V
            b_v = tl.load(
                p_v + o_t[:, None] * (HV * V) + (i_v * BV + o_v)[None, :],
                mask=m_c[:, None] & m_v[None, :],
                other=0.0,
            )
            b_beta = tl.sigmoid(
                tl.load(beta + (bos + o_c) * HV + i_h, mask=m_c, other=0.0).to(
                    tl.float32
                )
            )
            b_vb = b_v.to(tl.float32) * b_beta[:, None]

            b_h_cast = b_h.to(tl.bfloat16)
            # h at the start of this chunk, consumed by Finalize.
            tl.store(
                h_buf
                + ((gid.to(tl.int64) * HV + i_h) * K + o_k[:, None]) * V
                + (i_v * BV + o_v)[None, :],
                b_h_cast,
                mask=m_v[None, :],
            )

            b_kh = tl.dot(b_w.to(tl.bfloat16), b_h_cast, out_dtype=tl.float32)
            b_diff = b_vb - b_kh
            b_v_new = tl.dot(
                b_Akk.to(tl.bfloat16),
                b_diff.to(tl.bfloat16),
                out_dtype=tl.float32,
            )
            b_v_new_bf = b_v_new.to(tl.bfloat16)
            tl.store(
                v_new
                + ((gid.to(tl.int64) * HV + i_h) * BT) * V
                + o_t[:, None] * V
                + (i_v * BV + o_v)[None, :],
                b_v_new_bf,
                mask=m_c[:, None] & m_v[None, :],
            )

            # h = h * exp2(gk_last) + kg^T @ v_new
            b_h = b_h * exp2(b_gk_last)[:, None] + tl.dot(
                tl.trans(b_kg).to(tl.bfloat16),
                b_v_new_bf,
                out_dtype=tl.float32,
            )

        # Final state.
        if STORE_FINAL_STATE:
            ht += i_nh.to(tl.int64) * K * V
            if STATE_V_FIRST:
                tl.store(
                    ht + (i_v * BV + o_v)[:, None] * K + o_k[None, :],
                    tl.trans(b_h).to(ht.dtype.element_ty),
                    mask=m_v[:, None],
                )
            else:
                tl.store(
                    ht + o_k[:, None] * V + (i_v * BV + o_v)[None, :],
                    b_h.to(ht.dtype.element_ty),
                    mask=m_v[None, :],
                )

    def _ascend_kda_fwd_state(
        v,
        beta,
        gk,
        Akk,
        ws,
        v_new,
        h_buf,
        chunk_offsets,
        initial_state=None,
        output_final_state=False,
        state_v_first=True,
        cu_seqlens=None,
    ):
        B = v.shape[0]
        HV = ws.shape[1]
        BT = ws.shape[2]
        K = ws.shape[3] // 3
        V = v.shape[-1]

        if K > 256:
            raise ValueError(f"KDA K must be <= 256, got {K}")

        N = B if cu_seqlens is None else len(cu_seqlens) - 1

        final_state = None
        if output_final_state:
            if state_v_first:
                final_state = v.new_zeros(N, HV, V, K, dtype=torch.float32)
            else:
                final_state = v.new_zeros(N, HV, K, V, dtype=torch.float32)

        h0_arg = (
            initial_state
            if initial_state is not None
            else v.new_empty(1, dtype=torch.float32)
        )
        ht_arg = (
            final_state
            if final_state is not None
            else v.new_empty(1, dtype=torch.float32)
        )

        bv = _pick_bv(K, V)
        grid = (triton.cdiv(V, bv), N * HV)
        _ascend_kda_fwd_state_kernel[grid](
            v=v,
            beta=beta,
            gk=gk,
            Akk=Akk,
            ws=ws,
            v_new=v_new,
            h_buf=h_buf,
            h0=h0_arg,
            ht=ht_arg,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlens,
            T=v.shape[1],
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BV=bv,
            STATE_V_FIRST=state_v_first,
        )
        return final_state

    # ------------------------------------------------------------------
    # Kernel 3: Finalize -- o = scale * qg@h + Aqk@v_new.  Fully parallel
    # over (chunk, head), exactly AscendC's ChunkKdaFwdFinalize.
    # ------------------------------------------------------------------
    @triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
    @triton.jit(do_not_specialize=["T"])
    def _ascend_kda_fwd_output_kernel(
        qg_ws,
        v_new,
        Aqk,
        h_buf,
        o,
        chunk_indices,
        cu_seqlens,
        scale,
        T,
        NC,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_tc = tl.program_id(1)
        i_nh = tl.program_id(2)

        if IS_VARLEN:
            i_n = tl.load(chunk_indices + i_tc * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + i_tc * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
            gid = i_tc
        else:
            i_n = i_nh // HV
            bos = i_n * T
            i_t = i_tc
            gid = i_n * NC + i_t
        i_h = i_nh % HV

        o_t = tl.arange(0, BT)
        o_k = tl.arange(0, K)
        o_v = tl.arange(0, BV)
        m_c = i_t * BT + o_t < T
        m_v = i_v * BV + o_v < V

        p_ws = qg_ws + (gid.to(tl.int64) * HV + i_h) * BT * 3 * K
        b_qg = tl.load(p_ws + o_t[:, None] * (3 * K) + o_k[None, :] + 2 * K)
        b_v_new = tl.load(
            v_new
            + ((gid.to(tl.int64) * HV + i_h) * BT) * V
            + o_t[:, None] * V
            + (i_v * BV + o_v)[None, :],
            mask=m_c[:, None] & m_v[None, :],
            other=0.0,
        )
        b_Aqk = tl.load(
            Aqk
            + ((gid.to(tl.int64) * HV + i_h) * BT * BT)
            + o_t[:, None] * BT
            + o_t[None, :]
        )
        b_h = tl.load(
            h_buf
            + ((gid.to(tl.int64) * HV + i_h) * K + o_k[:, None]) * V
            + (i_v * BV + o_v)[None, :],
            mask=m_v[None, :],
            other=0.0,
        )

        b_qh = tl.dot(b_qg.to(tl.bfloat16), b_h.to(tl.bfloat16), out_dtype=tl.float32)
        b_o = scale * b_qh + tl.dot(
            b_Aqk.to(tl.bfloat16),
            b_v_new.to(tl.bfloat16),
            out_dtype=tl.float32,
        )

        o += ((bos + i_t * BT) * HV + i_h).to(tl.int64) * V
        tl.store(
            o + o_t[:, None] * (HV * V) + (i_v * BV + o_v)[None, :],
            b_o.to(o.dtype.element_ty),
            mask=m_c[:, None] & m_v[None, :],
        )

    def _ascend_kda_fwd_output(
        ws,
        Aqk,
        v_new,
        h_buf,
        NC,
        NC_global,
        v,
        scale,
        cu_seqlens=None,
        chunk_indices=None,
    ):
        V = v.shape[-1]
        HV = ws.shape[1]
        BT = ws.shape[2]
        K = ws.shape[3] // 3
        bv = _pick_bv(K, V)
        o = torch.empty(v.shape[0], v.shape[1], HV, V, device=v.device, dtype=v.dtype)
        if cu_seqlens is None:
            grid = (triton.cdiv(V, bv), NC, v.shape[0] * HV)
        else:
            grid = (triton.cdiv(V, bv), NC_global, HV)
        _ascend_kda_fwd_output_kernel[grid](
            qg_ws=ws,
            v_new=v_new,
            Aqk=Aqk,
            h_buf=h_buf,
            o=o,
            chunk_indices=chunk_indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            T=v.shape[1],
            NC=NC,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BV=bv,
        )
        return o

    def ascend_tle_chunk_kda_fwd_infer(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        chunk_size: int = 16,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if scale is None:
            scale = q.shape[-1] ** -0.5

        # Tile at the Ascend-internal width regardless of the caller's
        # chunk_size: the chunked recurrence is mathematically independent of
        # BT (only fp accumulation order changes), so a wider tile shortens the
        # serial scan without altering the contract. Any chunk_indices built
        # for the caller's width no longer describes our tiling, so rebuild it.
        bt = _ascend_chunk_width()
        if bt != chunk_size:
            chunk_indices = (
                prepare_chunk_indices(cu_seqlens, bt)
                if cu_seqlens is not None
                else None
            )
        elif chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, bt)

        # Memory fallback: the chunk-major h buffer is [chunks, HV, K, V]
        # bf16 (402 MiB at T=32768 H=24). Above the budget -- or when the
        # split is disabled for A/B -- use the portable chunk-at-a-time
        # Triton path instead of materializing h for every chunk.
        if cu_seqlens is None:
            nc_global = q.shape[0] * triton.cdiv(q.shape[1], bt)
        else:
            nc_global = len(chunk_indices)
        h_bytes = nc_global * g.shape[2] * q.shape[-1] * v.shape[-1] * 2
        if _ascend_fused_state() or h_bytes > _ascend_h_budget_bytes():
            return chunk_kda_fwd_infer_triton(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                state_v_first=state_v_first,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=bt,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                A_log=A_log,
                dt_bias=dt_bias,
            )

        ws, Aqk, Akk, g_out, chunk_offsets, NC, NC_global, N = _ascend_kda_fwd_intra(
            q=q,
            k=k,
            g=g,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=bt,
            lower_bound=lower_bound,
            A_log=A_log,
            dt_bias=dt_bias,
        )

        K = q.shape[-1]
        V = v.shape[-1]
        HV = g.shape[2]

        h_buf = torch.empty(NC_global, HV, K, V, device=q.device, dtype=torch.bfloat16)
        v_new = torch.empty(NC_global, HV, bt, V, device=q.device, dtype=torch.bfloat16)

        final_state = _ascend_kda_fwd_state(
            v=v,
            beta=beta,
            gk=g_out,
            Akk=Akk,
            ws=ws,
            v_new=v_new,
            h_buf=h_buf,
            chunk_offsets=chunk_offsets,
            initial_state=initial_state,
            output_final_state=output_final_state,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
        )
        o = _ascend_kda_fwd_output(
            ws=ws,
            Aqk=Aqk,
            v_new=v_new,
            h_buf=h_buf,
            NC=NC,
            NC_global=NC_global,
            v=v,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        return o, final_state


# =============================================================================
# Public entry: validate, then dispatch (TLE if available, else Triton)
# =============================================================================


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 16,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Inference-only implementation of chunk Kimi Delta Attention (Ascend).

    Inputs use seq-first layout: q/k ``[B, T, H, K]``, v/g ``[B, T, HV, *]``,
    and beta ``[B, T, HV]``. q/k L2 norm, gate activation, and beta sigmoid are
    computed inside the kernels. Backend selection defaults to automatic and
    can be forced with ``FLAGGEMS_CHUNK_KDA_BACKEND``.
    """
    A_log = kwargs.get("A_log")
    dt_bias = kwargs.get("dt_bias")

    _validate_chunk_kda_inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    backend = _chunk_kda_backend()
    if backend == "tle" and not HAS_TLE_KDA:
        raise RuntimeError(
            f"{_BACKEND_ENV}=tle requires Triton TLE >= 3.5.0 "
            "(FlagTree triton on Ascend)"
        )

    if backend in {"auto", "tle"} and HAS_TLE_KDA:
        return ascend_tle_chunk_kda_fwd_infer(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    return chunk_kda_fwd_infer_triton(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
    )
