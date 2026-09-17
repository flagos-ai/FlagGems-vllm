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

import logging
import os

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.device_info import get_device_capability
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

logger = logging.getLogger(__name__)


@triton.jit
def _e2m1_to_f32(nibble):
    """Dequantize an E2M1 nibble (uint8 in [0, 16)) to fp32.

    E2M1 bit layout: bit3 = sign, bits[2:1] = 2-bit exponent (bias 1),
    bit0 = 1-bit mantissa. Representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    Matches the quantization used by `fused_indexer_q_rope_quant(use_fp4=True)`.
    """
    n = nibble.to(tl.int32)
    sign = 1.0 - 2.0 * ((n >> 3) & 1).to(tl.float32)
    e = (n >> 1) & 0x3
    m = (n & 0x1).to(tl.float32)
    mag = tl.where(
        e == 0,
        0.5 * m,
        tl.where(e == 1, 1.0 + 0.5 * m, tl.where(e == 2, 2.0 + m, 4.0 + 2.0 * m)),
    )
    return sign * mag


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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fp8_fp4_mqa_logits"),
    key=["M", "N", "H", "D"],
)
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
        acc_h = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for i in tl.static_range(HEAD_BLOCK):
            h = hb + i
            q_i = tl.load(
                Q_ptr
                + m_offs[:, None] * stride_qm
                + h * stride_qh
                + d_offs[None, :] * stride_qd,
                mask=m_mask[:, None] & (h < H) & (d_offs[None, :] < D),
                other=0.0,
            )  # [BLOCK_M, D]

            dot_i = tl.dot(q_i, tl.trans(k))  # [BLOCK_M, BLOCK_N]

            w_i = tl.load(
                W_ptr + m_offs * stride_wm + h * stride_wh,
                mask=m_mask & (h < H),
                other=0.0,
            )

            acc_h += tl.maximum(dot_i * k_scale[None, :], 0.0) * w_i[:, None]

        acc += acc_h

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
@libtuner(
    configs=runtime.get_tuned_config("fp8_fp4_mqa_logits"),
    key=["M", "N", "H", "D"],
)
@triton.jit
def _fp8_fp4_mqa_logits_mxfp4_kernel(
    Q_ptr,  # uint8 [M, H, D//2] packed E2M1 (2 nibbles per byte)
    Q_scale_ptr,  # int32 [M, H] holding D//32 ue8m0 bytes (little-endian)
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
    """Fused MXFP4 MQA logits with software E2M1 dequant + fp16 MMA.

    Consumes MXFP4 Q (packed E2M1 values + ue8m0 per-32 block scales) against
    FP8 K with per-token fp32 scales. K stays fp8 because the flaggems K-side
    producer (`indexer_k_quant_and_cache`) does not emit fp4.

    logits[m, n] = sum_h( ReLU( sum_d(q[m,h,d]*k[n,d]) * k_scale[n] ) * w[m,h] )
    with q[m,h,d] = e2m1(packed[m,h,d//2], d%2) * 2^(ue8m0[m,h,d//32] - 127).
    """
    D2: tl.constexpr = D // 2

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, D)
    d2_offs = tl.arange(0, D2)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # K is fp8 [N, D] with per-token fp32 scale; upcast to fp16 for the MMA.
    k = tl.load(
        K_ptr + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
        mask=n_mask[:, None] & (d_offs[None, :] < D),
        other=0.0,
    ).to(tl.float16)

    k_scale = tl.load(K_scale_ptr + n_offs, mask=n_mask, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Per-dim scale block index: dim d belongs to ue8m0 block d // 32.
    block_id = d_offs // 32  # [D] 1D for MUSA Triton compatibility

    for hb in range(0, H, HEAD_BLOCK):
        # Per-head dot products to avoid 3D reshape SQMMA incompatibility on MUSA
        h0 = hb
        h1 = hb + 1

        # Extract per-head Q packed bytes: [BLOCK_M, D//2]
        q_packed_0 = tl.load(
            Q_ptr
            + m_offs[:, None] * stride_qm
            + h0 * stride_qh
            + d2_offs[None, :] * stride_qd,
            mask=m_mask[:, None] & (d2_offs[None, :] < D2),
            other=0,
        )
        q_packed_1 = tl.load(
            Q_ptr
            + m_offs[:, None] * stride_qm
            + h1 * stride_qh
            + d2_offs[None, :] * stride_qd,
            mask=m_mask[:, None] & (h1 < H) & (d2_offs[None, :] < D2),
            other=0,
        )

        # Unpack nibbles for each head
        lo0 = q_packed_0 & 0xF
        hi0 = (q_packed_0 >> 4) & 0xF
        nibble0 = tl.reshape(tl.join(lo0, hi0), [BLOCK_M, D])
        q0_f32 = _e2m1_to_f32(nibble0)

        lo1 = q_packed_1 & 0xF
        hi1 = (q_packed_1 >> 4) & 0xF
        nibble1 = tl.reshape(tl.join(lo1, hi1), [BLOCK_M, D])
        q1_f32 = _e2m1_to_f32(nibble1)

        # Per-block ue8m0 scale for each head
        q_scale_val_0 = tl.load(
            Q_scale_ptr + m_offs * stride_qsm + h0 * stride_qsh,
            mask=m_mask,
            other=0,
        )
        q_scale_val_1 = tl.load(
            Q_scale_ptr + m_offs * stride_qsm + h1 * stride_qsh,
            mask=m_mask & (h1 < H),
            other=0,
        )

        byte0 = ((q_scale_val_0[:, None] >> (8 * block_id[None, :])) & 0xFF).to(
            tl.float32
        )
        scale0 = tl.exp2(byte0 - 127.0)
        q0_f32 = q0_f32 * scale0

        byte1 = ((q_scale_val_1[:, None] >> (8 * block_id[None, :])) & 0xFF).to(
            tl.float32
        )
        scale1 = tl.exp2(byte1 - 127.0)
        q1_f32 = q1_f32 * scale1

        # Dot products: [BLOCK_M, BLOCK_N]
        dot0 = tl.dot(q0_f32.to(tl.float16), tl.trans(k))
        dot1 = tl.dot(q1_f32.to(tl.float16), tl.trans(k))

        # Load weights for each head
        w0 = tl.load(
            W_ptr + m_offs * stride_wm + h0 * stride_wh, mask=m_mask, other=0.0
        )
        w1 = tl.load(
            W_ptr + m_offs * stride_wm + h1 * stride_wh,
            mask=m_mask & (h1 < H),
            other=0.0,
        )

        # Fused k_scale + ReLU + weight for each head, then sum
        dot_reduced = (
            tl.maximum(dot0 * k_scale[None, :], 0.0) * w0[:, None]
            + tl.maximum(dot1 * k_scale[None, :], 0.0) * w1[:, None]
        )

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
    _fp8_fp4_mqa_logits_kernel_tle[grid](
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
            FP4 path: q_values [M, H, D//2] uint8 (packed E2M1), q_scale
                [M, H] int32 (D//32 ue8m0 bytes per token-head, little-endian).
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

    logits = torch.empty(
        (q_values.shape[0], k_values.shape[0]),
        dtype=torch.float32,
        device=q_values.device,
    )

    grid = lambda META: (
        triton.cdiv(q_values.shape[0], META["BLOCK_M"]),
        triton.cdiv(k_values.shape[0], META["BLOCK_N"]),
    )

    if q_scale is not None:
        # MXFP4 Q path: q_values is packed uint8 [M, H, D//2], q_scale is
        # int32 [M, H] holding D//32 ue8m0 bytes per (token, head).
        M, H, D2 = q_values.shape
        D = D2 * 2
        N = k_values.shape[0]
        assert k_values.shape[1] == D

        _fp8_fp4_mqa_logits_mxfp4_kernel[grid](
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
            _fp8_fp4_mqa_logits_kernel[grid](
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
        _clean_logits_kernel[clean_grid](
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
