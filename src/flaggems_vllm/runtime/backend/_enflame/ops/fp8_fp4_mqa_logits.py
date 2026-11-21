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

"""fp8_fp4_mqa_logits — Enflame (GCU) host dispatch.

GCU adaptation relative to the device-agnostic version:

**Canonical-module kernel resolution.** The SpecOpRegistrar namespace swap
points ``flaggems_vllm.fp8_fp4_mqa_logits`` at this module's host function,
whose module globals are not the ones consumers patch: contract tests
(``tests/test_indexer_scale_contract.py``) intercept the mxfp4/fp8/clean
kernel launches through ``flaggems_vllm.ops.fp8_fp4_mqa_logits`` (the
canonical module) via ``monkeypatch.setattr``. This copy therefore resolves
every kernel launch through the canonical module namespace at call time,
mirroring the mrope/fp8_storage install precedent (the canonical namespace is
the rendezvous point, and explicit user or test patches always win).

**GCU-tuned kernels (pristine swap).** The local kernel definitions below are
installed into the canonical module at import time (identity-guarded, so test
monkeypatches still win):

- ``_fp8_fp4_mqa_logits_mxfp4_kernel`` — join-free restructure. On GCU the
  generic kernel's nibble interleave (``tl.join(lo, hi)`` + reshape) is a
  pathological layout conversion (~80% of runtime), so the dot is split into
  even/odd K-column tiles matched against the dequantized lo/hi nibble halves,
  the E2M1 mapping uses an fp32 bit-craft instead of a nested ``tl.where``
  chain, and the ue8m0 scale is applied on the wide half tensors (a 4-D
  regroup broadcast multiply was measured ~28 ms slower and is avoided).
- ``_fp8_fp4_mqa_logits_kernel`` — body byte-identical to the generic source;
  only the tuned geometry differs (BLOCK_N=128 halves the per-N-tile Q
  re-read count; measured ~25% faster across the production shapes).
- ``_clean_logits_kernel`` / ``_fp8_fp4_mqa_logits_kernel_tle`` — served from
  the canonical module unchanged (the TLE path never fires on GCU: device
  capability (4, 1) < (9, 0)).

Host-side validation, stride handling and early returns are unchanged.
"""

import logging
import os
from importlib import import_module as _import_module

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.device_info import get_device_capability
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

# The canonical generic module. ``import flaggems_vllm.ops.fp8_fp4_mqa_logits``
# cannot be used: the ops package rebinds that name to the host *function*
# (``from ... import fp8_fp4_mqa_logits``), so only import_module reliably
# yields the module object whose namespace contract tests patch.
_canonical = _import_module("flaggems_vllm.ops.fp8_fp4_mqa_logits")

# Pristine generic kernels, captured before the _install() swap at the bottom
# of this module so the identity guard can tell "untouched generic" apart
# from any explicit user or test patch.
_GENERIC_FP8_KERNEL = _canonical._fp8_fp4_mqa_logits_kernel
_GENERIC_MXFP4_KERNEL = _canonical._fp8_fp4_mqa_logits_mxfp4_kernel

logger = logging.getLogger(__name__)


@triton.jit
def _e2m1_to_f32_bits(nibble):
    """Dequantize an E2M1 nibble (uint8/int8 in [0, 16)) to fp32 by bit-craft.

    E2M1 bit layout: bit3 = sign, bits[2:1] = 2-bit exponent (bias 1),
    bit0 = 1-bit mantissa. Representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    Matches the quantization used by `fused_indexer_q_rope_quant(use_fp4=True)`.

    fp32 exponent field for magnitude mag = n & 7: mag < 2 maps the subnormal
    pair {0 -> +0.0, 0.5 -> 0x3F000000}; mag >= 2 (normal) uses field
    126 + e with the mantissa bit at 22. The sign bit lands at 31 via
    (n & 8) << 28. Replaces the generic nested-tl.where chain (measured ~6%
    of the join-free kernel on GCU400; see case_log j20261111085117).
    """
    n = nibble.to(tl.int32)
    mag = n & 0x7
    bits = tl.where(
        mag < 2,
        mag * 0x3F000000,
        ((126 + (mag >> 1)) << 23) | ((mag & 0x1) << 22),
    )
    bits = bits | ((n & 0x8) << 28)
    return bits.to(tl.float32, bitcast=True)


# =============================================================================
# TLE (WGMMA) fast path — purely additive. The tuned baseline below is left
# byte-for-byte untouched; the TLE path only fires inside its measured win band
# and falls back to the baseline everywhere else (no regression guarantee).
# See TLE_OPTIMIZATION_CASE_STUDY.md for the rationale.
# =============================================================================

HAS_TLE = has_triton_tle(3, 6, 0)
if HAS_TLE:
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = hasattr(tle.gpu, "wgmma")
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None


def _tle_enabled() -> bool:
    """Master switch for the WGMMA fast path (mirrors FLAGGEMS_FLASHMLA_DECODE_TLE)."""
    value = os.environ.get("FLAGGEMS_FP8_FP4_MQA_LOGITS_TLE", "1").lower()
    return value not in {"0", "false", "off", "no"}


# WGMMA fast-path geometry. Validated on H800 (2026-08-10): BLOCK_N=64 keeps the
# K tile at 32 KB for D=512 (the WGMMA sweet spot); HEAD_BLOCK=2 bounds the Q tile
# (BLOCK_M*HEAD_BLOCK*D bytes) so pipelining doesn't overflow SMEM.
_TLE_CONFIG = (64, 64, 2, 4, 2)  # (BLOCK_M, BLOCK_N, HEAD_BLOCK, num_warps, num_stages)

# GCU400-tuned geometry for the fp8 and mxfp4 MQA logits kernels (measured on
# the benchmark core shapes; see case_log j20261111085117). BLOCK_N=128 halves
# the per-N-tile Q reload count and fits the 917,504-byte DSM budget —
# BLOCK_N=256 exceeds it (compile-time OutOfResources). num_warps/num_stages
# are insensitive in [1, 3]/[2, 4]; a single deterministic config avoids the
# multi-config autotune lottery.
_GCU_MQALOGITS_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "HEAD_BLOCK": 2},
        num_warps=4,
        num_stages=2,
    ),
]


@libentry()
@libtuner(configs=_GCU_MQALOGITS_CONFIGS, key=["M", "N", "H", "D"])
@triton.jit
def _fp8_fp4_mqa_logits_kernel(
    Q_ptr,
    K_ptr,
    K_scale_ptr,
    W_ptr,
    O_ptr,
    M,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_om,
    stride_on,
    stride_wm,
    stride_wh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    """Fused FP8 MQA logits with head-batched tiled dot products.

    Computes logits[m, n] = sum_h(ReLU(sum_d(q[m,h,d]*k[n,d]) * k_scale[n])
                                  * weights[m, h])

    K is loaded once per (BLOCK_M, BLOCK_N) tile and reused across HEAD_BLOCK
    heads to minimize global memory traffic.

    GCU: body byte-identical to the generic source; only the tuned geometry
    differs (see _GCU_MQALOGITS_CONFIGS).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, D)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # Load K tile once — reused across all head batches
    k = tl.load(
        K_ptr + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
        mask=n_mask[:, None] & (d_offs[None, :] < D),
        other=0.0,
    )

    k_scale = tl.load(K_scale_ptr + n_offs, mask=n_mask, other=0.0)

    # Accumulator for output: [BLOCK_M, BLOCK_N] in fp32
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for hb in range(0, H, HEAD_BLOCK):
        hb_offs = hb + tl.arange(0, HEAD_BLOCK)

        # Load Q for this head batch: [BLOCK_M, HEAD_BLOCK, D]
        q = tl.load(
            Q_ptr
            + m_offs[:, None, None] * stride_qm
            + hb_offs[None, :, None] * stride_qh
            + d_offs[None, None, :] * stride_qd,
            mask=m_mask[:, None, None]
            & (hb_offs[None, :, None] < H)
            & (d_offs[None, None, :] < D),
            other=0.0,
        )

        # Flatten to 2D for MMA: [BLOCK_M * HEAD_BLOCK, D]
        q_2d = tl.reshape(q, [BLOCK_M * HEAD_BLOCK, D])

        # Dot product: [BLOCK_M * HEAD_BLOCK, BLOCK_N]
        dot = tl.dot(q_2d, tl.trans(k))

        # Fused k_scale + ReLU activation
        dot = tl.maximum(dot * k_scale[None, :], 0.0)

        # Load weights for this head batch: [BLOCK_M, HEAD_BLOCK]
        w = tl.load(
            W_ptr + m_offs[:, None] * stride_wm + hb_offs[None, :] * stride_wh,
            mask=m_mask[:, None] & (hb_offs[None, :] < H),
            other=0.0,
        )

        # Weight each head's contribution
        w_flat = tl.reshape(w, [BLOCK_M * HEAD_BLOCK])
        dot = dot * w_flat[:, None]

        # Reduce over head dimension: [BLOCK_M, HEAD_BLOCK, BLOCK_N] -> sum
        dot_3d = tl.reshape(dot, [BLOCK_M, HEAD_BLOCK, BLOCK_N])
        dot_reduced = tl.sum(dot_3d, axis=1)

        acc += dot_reduced

    # Store output tile
    write_mask = m_mask[:, None] & n_mask[None, :]
    out_ptrs = O_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=write_mask)


@libentry()
@triton.jit
def _clean_logits_kernel(
    O_ptr,
    KS_ptr,
    KE_ptr,
    M,
    N,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fill invalid positions with -inf based on per-row [ks, ke) ranges."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    ks = tl.load(KS_ptr + m_offs, mask=m_mask, other=0)
    ke = tl.load(KE_ptr + m_offs, mask=m_mask, other=0)

    invalid_mask = (n_offs[None, :] < ks[:, None]) | (n_offs[None, :] >= ke[:, None])
    write_mask = m_mask[:, None] & n_mask[None, :] & invalid_mask

    neg_inf = float("-inf")
    out_ptrs = O_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(
        out_ptrs,
        neg_inf + tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32),
        mask=write_mask,
    )


@libentry()
@libtuner(configs=_GCU_MQALOGITS_CONFIGS, key=["M", "N", "H", "D"])
@triton.jit
def _fp8_fp4_mqa_logits_mxfp4_kernel(
    Q_ptr,  # uint8/int8 [M, H, D//2] packed E2M1 (2 nibbles per byte)
    Q_scale_ptr,  # int32 [M, H] or [M, H, 1], packed ue8m0 bytes (little-endian)
    K_ptr,  # fp8 [N, D]
    K_scale_ptr,  # fp32 [N] per-token scale
    W_ptr,  # fp32 [M, H] per-head weights (NO q_scale folded)
    O_ptr,  # fp32 [M, N]
    M,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_qsm,
    stride_qsh,
    stride_kn,
    stride_kd,
    stride_om,
    stride_on,
    stride_wm,
    stride_wh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    """GCU-tuned MXFP4 MQA logits (join-free two-dot restructure).

    Consumes MXFP4 Q (packed E2M1 values + ue8m0 per-32 block scales) against
    FP8 K with per-token fp32 scales. K stays fp8 because the flaggems K-side
    producer (`indexer_k_quant_and_cache`) does not emit fp4.

    logits[m, n] = sum_h( ReLU( sum_d(q[m,h,d]*k[n,d]) * k_scale[n] ) * w[m,h] )
    with q[m,h,d] = e2m1(packed[m,h,d//2], d%2) * 2^(ue8m0[m,h,d//32] - 127).

    GCU restructure (the generic nibble interleave `tl.join(lo, hi)` +
    reshape to [.., D] is a pathological layout conversion, ~80% of runtime
    on GCU400): the packed byte j carries dim 2*j in its lo nibble and dim
    2*j+1 in its hi nibble, so the single dot over D splits into even/odd
    K-column tiles `[BLOCK_N, D//2]` dotted against the dequantized lo/hi
    halves — no interleave anywhere. The ue8m0 scale stays applied on the
    wide half tensors (variable per-lane shift + exp2, measured cheap); a
    [BM, HB, 4]-block regroup broadcast multiply was measured ~28 ms slower
    and is deliberately avoided.
    """
    D2: tl.constexpr = D // 2

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d2_offs = tl.arange(0, D2)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # Even/odd K column tiles, upcast to fp16 for the MMA (K is fp8 [N, D]).
    k_even = tl.load(
        K_ptr + n_offs[:, None] * stride_kn + (2 * d2_offs)[None, :] * stride_kd,
        mask=n_mask[:, None] & (d2_offs[None, :] < D2),
        other=0.0,
    ).to(tl.float16)
    k_odd = tl.load(
        K_ptr + n_offs[:, None] * stride_kn + (2 * d2_offs + 1)[None, :] * stride_kd,
        mask=n_mask[:, None] & (d2_offs[None, :] < D2),
        other=0.0,
    ).to(tl.float16)
    k_even_t = tl.trans(k_even)
    k_odd_t = tl.trans(k_odd)

    k_scale = tl.load(K_scale_ptr + n_offs, mask=n_mask, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Half-dim scale block index: dim d = 2*j + parity belongs to ue8m0
    # block d // 32 == j // 16.
    block_id2 = (d2_offs // 16)[None, None, :]

    for hb in range(0, H, HEAD_BLOCK):
        hb_offs = hb + tl.arange(0, HEAD_BLOCK)

        # Packed Q bytes: [BLOCK_M, HEAD_BLOCK, D//2]
        q_packed = tl.load(
            Q_ptr
            + m_offs[:, None, None] * stride_qm
            + hb_offs[None, :, None] * stride_qh
            + d2_offs[None, None, :] * stride_qd,
            mask=m_mask[:, None, None]
            & (hb_offs[None, :, None] < H)
            & (d2_offs[None, None, :] < D2),
            other=0,
        )

        q_scale_val = tl.load(
            Q_scale_ptr + m_offs[:, None] * stride_qsm + hb_offs[None, :] * stride_qsh,
            mask=m_mask[:, None] & (hb_offs[None, :] < H),
            other=0,
        )  # [BLOCK_M, HEAD_BLOCK] int32

        # Per-32-dim ue8m0 scale for every half-dim lane: [BM, HB, D//2]
        byte = ((q_scale_val[:, :, None] >> (8 * block_id2)) & 0xFF).to(tl.float32)
        scale = tl.exp2(byte - 127.0)

        # Dequantize nibbles and fold the block scale (fp32, then fp16 MMA).
        lo = _e2m1_to_f32_bits(q_packed & 0xF) * scale
        hi = _e2m1_to_f32_bits((q_packed >> 4) & 0xF) * scale

        lo_s = tl.reshape(lo, [BLOCK_M * HEAD_BLOCK, D2]).to(tl.float16)
        hi_s = tl.reshape(hi, [BLOCK_M * HEAD_BLOCK, D2]).to(tl.float16)

        # sum_d q[m,h,d]*k[n,d] = dot(lo, k_even) + dot(hi, k_odd)
        dot = tl.dot(hi_s, k_odd_t, tl.dot(lo_s, k_even_t))

        # Fused k_scale + ReLU activation
        dot = tl.maximum(dot * k_scale[None, :], 0.0)

        # Load weights for this head batch: [BLOCK_M, HEAD_BLOCK]
        w = tl.load(
            W_ptr + m_offs[:, None] * stride_wm + hb_offs[None, :] * stride_wh,
            mask=m_mask[:, None] & (hb_offs[None, :] < H),
            other=0.0,
        )

        # Weight each head's contribution
        w_flat = tl.reshape(w, [BLOCK_M * HEAD_BLOCK])
        dot = dot * w_flat[:, None]

        # Reduce over head dimension: [BLOCK_M, HEAD_BLOCK, BLOCK_N] -> sum
        dot_3d = tl.reshape(dot, [BLOCK_M, HEAD_BLOCK, BLOCK_N])
        dot_reduced = tl.sum(dot_3d, axis=1)

        acc += dot_reduced

    # Store output tile
    write_mask = m_mask[:, None] & n_mask[None, :]
    out_ptrs = O_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=write_mask)


@libentry()
@triton.jit
def _fp8_fp4_mqa_logits_kernel_tle(
    Q_ptr,
    K_ptr,
    K_scale_ptr,
    W_ptr,
    O_ptr,
    M,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_om,
    stride_on,
    stride_wm,
    stride_wh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """TLE WGMMA fast path: K and Q staged in MMA-swizzled SMEM, WGMMA reads
    both operands directly from shared memory.

    logits[m, n] = sum_h(ReLU(sum_d(q[m,h,d]*k[n,d]) * k_scale[n]) * weights[m,h])

    Wins when the K tile exceeds register capacity (D >= 256); at D == 128 the
    tuned baseline is faster, so the fast path is D >= 256 only. Falls back to
    the tuned baseline everywhere else (see fp8_fp4_mqa_logits).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, D)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # K in SMEM with MMA swizzle — WGMMA reads directly from here (loaded once
    # per (BLOCK_M, BLOCK_N) tile, reused across all head batches).
    k_smem = tle.gpu.alloc(
        [BLOCK_N, D],
        dtype=tl.float8e4nv,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    k_gmem_ptrs = K_ptr + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd
    k_mask = n_mask[:, None] & (d_offs[None, :] < D)
    k_vals = tle.load(k_gmem_ptrs, mask=k_mask, other=0.0, is_async=True)
    rows = tl.broadcast_to(n_offs[:, None], (BLOCK_N, D))
    cols = tl.broadcast_to(d_offs[None, :], (BLOCK_N, D))
    k_local_ptr = tle.gpu.local_ptr(k_smem, (rows - pid_n * BLOCK_N, cols))
    tl.store(k_local_ptr, k_vals)

    k_scale = tl.load(K_scale_ptr + n_offs, mask=n_mask, other=0.0)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    Q_ROWS: tl.constexpr = BLOCK_M * HEAD_BLOCK
    q_smem = tle.gpu.alloc(
        [Q_ROWS, D],
        dtype=tl.float8e4nv,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    q_base_rows = tl.arange(0, Q_ROWS)
    qr = tl.broadcast_to(q_base_rows[:, None], (Q_ROWS, D))
    qc = tl.broadcast_to(d_offs[None, :], (Q_ROWS, D))
    q_local_ptr = tle.gpu.local_ptr(q_smem, (qr, qc))

    for hb in tl.range(0, H, HEAD_BLOCK, num_stages=NUM_STAGES):
        hb_offs = hb + tl.arange(0, HEAD_BLOCK)

        q_vals = tle.load(
            Q_ptr
            + m_offs[:, None, None] * stride_qm
            + hb_offs[None, :, None] * stride_qh
            + d_offs[None, None, :] * stride_qd,
            mask=m_mask[:, None, None]
            & (hb_offs[None, :, None] < H)
            & (d_offs[None, None, :] < D),
            other=0.0,
            is_async=True,
        )
        tl.store(q_local_ptr, tl.reshape(q_vals, [Q_ROWS, D]))

        # WGMMA: Q[Q_ROWS, D] @ K[BLOCK_N, D]^T — both read directly from SMEM
        dot = tle.gpu.wgmma(q_smem, k_smem, out_dtype=tl.float32, trans_b=True)
        dot = tle.gpu.wgmma_wait(0, dot)

        dot = tl.maximum(dot * k_scale[None, :], 0.0)

        w = tle.load(
            W_ptr + m_offs[:, None] * stride_wm + hb_offs[None, :] * stride_wh,
            mask=m_mask[:, None] & (hb_offs[None, :] < H),
            other=0.0,
            is_async=True,
        )
        w_flat = tl.reshape(w, [BLOCK_M * HEAD_BLOCK])
        dot = dot * w_flat[:, None]

        dot_3d = tl.reshape(dot, [BLOCK_M, HEAD_BLOCK, BLOCK_N])
        acc += tl.sum(dot_3d, axis=1)

    write_mask = m_mask[:, None] & n_mask[None, :]
    out_ptrs = O_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=write_mask)


def _can_use_tle(M, N, H, D) -> bool:
    """Deterministic WGMMA fast-path guard (fixed, empirical — NOT autotune).

    WGMMA pays off when the K tile exceeds register capacity (D >= 256, K tile
    = BLOCK_N*D bytes fp8); at D == 128 the K tile fits in registers and the
    tuned baseline is faster, so only D >= 256 takes the fast path.
    D > 512 would overflow SMEM once the pipelined Q tile is counted (measured
    OOM), so the fast path is capped at D == 512.
    """
    if not (HAS_TLE and _tle_enabled()):
        return False
    if get_device_capability() < (9, 0):
        return False
    if D > 512:
        return False
    return D >= 256


def _launch_tle_kernel(q_values, k_values, k_scales, weights, logits, M, N, H, D):
    """Launch the WGMMA fast path with the validated fixed geometry."""
    BLOCK_M, BLOCK_N, HEAD_BLOCK, num_warps, num_stages = _TLE_CONFIG
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _canonical._fp8_fp4_mqa_logits_kernel_tle[grid](
        q_values,
        k_values,
        k_scales,
        weights,
        logits,
        M,
        N,
        H,
        D,
        q_values.stride(0),
        q_values.stride(1),
        q_values.stride(2),
        k_values.stride(0),
        k_values.stride(1),
        logits.stride(0),
        logits.stride(1),
        weights.stride(0),
        weights.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_BLOCK=HEAD_BLOCK,
        num_warps=num_warps,
        NUM_STAGES=num_stages,
    )


def fp8_fp4_mqa_logits(
    q: tuple,
    kv: tuple,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton implementation of fp8_fp4_mqa_logits.

    Computes weighted MQA logits with FP8 or MXFP4 quantized Q and FP8 K.
    Uses head-batched tiled dot products with K reuse for high throughput.

    On Hopper+ (SM90) with TLE available, shapes with D >= 256 (K tile exceeds
    register capacity) are dispatched to a TLE WGMMA fast path; every other
    shape keeps the tuned baseline below. See TLE_OPTIMIZATION_CASE_STUDY.md.

    Args:
        q: Tuple of (q_values, q_scale).
            FP8 path: q_values [M, H, D] float8_e4m3fn, q_scale is None
                (per-token q_scale already folded into `weights`).
            FP4 path: q_values [M, H, D//2] uint8 or int8 (packed E2M1),
                q_scale [M, H] or [M, H, 1] int32 (D//32 ue8m0 bytes per
                token-head, little-endian). Token/head strides are preserved.
        kv: Tuple of (k_values [N, D] fp8, k_scales [N] fp32).
        weights: [M, H] fp32 per-head weights.
        cu_seqlen_ks: [M] int32 start indices for valid K range.
        cu_seqlen_ke: [M] int32 end indices for valid K range.
        clean_logits: Whether to fill invalid positions with -inf.

    Returns:
        logits: [M, N] fp32 output tensor.
    """
    logger.debug("GEMS FP8_FP4_MQA_LOGITS")

    q_values, q_scale = q
    k_values, k_scales = kv

    if q_values.ndim != 3 or k_values.ndim != 2:
        raise ValueError("MQA expects Q[M,H,D] and K[N,D]")
    m, h, packed_dim = q_values.shape
    n, dim = k_values.shape
    if k_scales.shape != (n,) or k_scales.dtype != torch.float32:
        raise ValueError("MQA key scales must be float32 [N]")
    if not k_scales.is_contiguous():
        raise ValueError("MQA key scales must be contiguous")
    if weights.shape != (m, h) or weights.dtype != torch.float32:
        raise ValueError("MQA weights must be float32 [M,H]")
    if q_scale is None:
        if q_values.dtype != torch.float8_e4m3fn or packed_dim != dim:
            raise ValueError("FP8 MQA expects E4M3FN query with the key dimension")
    elif (
        q_values.dtype not in (torch.uint8, torch.int8)
        or packed_dim * 2 != dim
        or q_scale.shape not in ((m, h), (m, h, 1))
        or q_scale.dtype != torch.int32
    ):
        raise ValueError(
            "MXFP4 MQA expects packed uint8/int8 Q and int32 [M,H] or [M,H,1] scales"
        )
    tensors = (k_values, k_scales, weights, q_scale, cu_seqlen_ks, cu_seqlen_ke)
    if any(t is not None and t.device != q_values.device for t in tensors):
        raise ValueError("MQA tensors must share the query device")
    if (
        k_values.dtype != torch.float8_e4m3fn
        or q_values.stride(-1) != 1
        or k_values.stride(-1) != 1
    ):
        raise ValueError("MQA expects E4M3FN keys and contiguous feature dimensions")
    if clean_logits and any(
        t.shape != (m,) or t.dtype != torch.int32 or not t.is_contiguous()
        for t in (cu_seqlen_ks, cu_seqlen_ke)
    ):
        raise ValueError("MQA valid ranges must be contiguous int32 [M]")

    logits = torch.empty(
        (q_values.shape[0], k_values.shape[0]),
        dtype=torch.float32,
        device=q_values.device,
    )
    if m == 0 or n == 0:
        return logits

    grid = lambda META: (
        triton.cdiv(q_values.shape[0], META["BLOCK_M"]),
        triton.cdiv(k_values.shape[0], META["BLOCK_N"]),
    )

    if q_scale is not None:
        # MXFP4 Q path: preserve packed uint8/int8 Q and the original rank-2
        # or rank-3 int32 scale object; the kernel uses token/head strides.
        M, H, D2 = q_values.shape
        D = D2 * 2
        N = k_values.shape[0]
        assert k_values.shape[1] == D

        _canonical._fp8_fp4_mqa_logits_mxfp4_kernel[grid](
            q_values,
            q_scale,
            k_values,
            k_scales,
            weights,
            logits,
            M,
            N,
            H,
            D,
            q_values.stride(0),
            q_values.stride(1),
            q_values.stride(2),
            q_scale.stride(0),
            q_scale.stride(1),
            k_values.stride(0),
            k_values.stride(1),
            logits.stride(0),
            logits.stride(1),
            weights.stride(0),
            weights.stride(1),
        )
    else:
        M, H, D = q_values.shape
        N = k_values.shape[0]
        use_tle = _can_use_tle(M, N, H, D)
        if use_tle:
            _launch_tle_kernel(
                q_values, k_values, k_scales, weights, logits, M, N, H, D
            )
        else:
            _canonical._fp8_fp4_mqa_logits_kernel[grid](
                q_values,
                k_values,
                k_scales,
                weights,
                logits,
                M,
                N,
                H,
                D,
                q_values.stride(0),
                q_values.stride(1),
                q_values.stride(2),
                k_values.stride(0),
                k_values.stride(1),
                logits.stride(0),
                logits.stride(1),
                weights.stride(0),
                weights.stride(1),
            )

    if clean_logits:
        CLEAN_BLOCK_M = 8
        CLEAN_BLOCK_N = 128
        clean_grid = (
            triton.cdiv(M, CLEAN_BLOCK_M),
            triton.cdiv(N, CLEAN_BLOCK_N),
        )
        _canonical._clean_logits_kernel[clean_grid](
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            M,
            N,
            logits.stride(0),
            logits.stride(1),
            BLOCK_M=CLEAN_BLOCK_M,
            BLOCK_N=CLEAN_BLOCK_N,
        )

    return logits


def _install():
    """Wire the GCU-tuned kernels into the canonical module namespace.

    Both the canonical (generic) host — used by ``flaggems_vllm.ops`` consumers
    and the benchmark wrapper — and this host resolve the fp8/mxfp4 kernel
    launches through ``flaggems_vllm.ops.fp8_fp4_mqa_logits`` at call time, so
    replacing that module's attributes in place upgrades both entrypoints.
    Only a pristine generic kernel is swapped (identity-guarded against the
    references captured at import, mirroring the mrope/fp8_storage install
    precedent), so explicit user or test monkeypatches always win.
    """
    for name, generic in (
        ("_fp8_fp4_mqa_logits_kernel", _GENERIC_FP8_KERNEL),
        ("_fp8_fp4_mqa_logits_mxfp4_kernel", _GENERIC_MXFP4_KERNEL),
    ):
        if getattr(_canonical, name, None) is generic:
            setattr(_canonical, name, globals()[name])


try:
    _install()
except Exception:
    # pytest assertion-rewrite contexts can break @triton.jit source lookup
    # when collection-time module chains import this module; a failed install
    # only means the generic kernels keep serving (the SpecOpRegistrar host
    # routing is unaffected).
    pass
