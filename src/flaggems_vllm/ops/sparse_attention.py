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

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


# ---------------------------------------------------------------------------
# Triton kernel: sparse attention with attention-sink
# grid = (m, b)  — one program per (seq_pos, batch), handles ALL heads
# Aligned with tilelang version: uses tl.dot (GEMM) instead of vector dot
# ---------------------------------------------------------------------------
@triton.jit
def sparse_attn_triton_kernel(
    Q,  # (b, m, h, d)  bf16
    KV,  # (b, n, d)     bf16
    O,  # (b, m, h, d)  bf16
    attn_sink,  # (h,)          fp32
    topk_idxs,  # (b, m, topk)  int32
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_idxb,
    stride_idxm,
    stride_idxk,
    scale,
    topk,
    H_ACTUAL,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    # ---- load Q matrix: (H, D) — all heads at once ----
    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H_ACTUAL
    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)  # (H, D) bf16

    # ---- base pointers ----
    kv_base = KV + pid_b * stride_kvb
    idx_base = topk_idxs + pid_b * stride_idxb + pid_m * stride_idxm

    # ---- online softmax state ----
    acc_o = tl.zeros([H, D], dtype=tl.float32)
    scores_max = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)

    num_blocks = (topk + BLOCK - 1) // BLOCK
    offs_blk = tl.arange(0, BLOCK)

    for t in range(num_blocks):
        # -- gather indices --
        raw_offs = t * BLOCK + offs_blk  # (BLOCK,)
        idx_mask = raw_offs < topk
        idxs = tl.load(
            idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1
        )  # (BLOCK,)
        valid_mask = idxs != -1  # (BLOCK,)

        # -- gather KV block: (BLOCK, D) --
        kv_ptrs = kv_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        kv_block = tl.load(
            kv_ptrs, mask=valid_mask[:, None], other=0.0
        )  # (BLOCK, D) bf16

        # -- scores: Q @ KV^T -> (H, BLOCK) via GEMM --
        acc_s = tl.dot(q_block, tl.trans(kv_block))  # (H, D) @ (D, BLOCK) = (H, BLOCK)
        acc_s = acc_s * scale
        # mask invalid positions to -inf (apply on 2D tensor to avoid layout mismatch)
        acc_s = tl.where(valid_mask[None, :], acc_s, float("-inf"))

        # -- online softmax update --
        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)  # (H,)
        scores_max = tl.maximum(scores_max, block_max)

        correction = tl.exp(scores_max_prev - scores_max)  # (H,)
        p = tl.exp(acc_s - scores_max[:, None])  # (H, BLOCK)

        # -- accumulate output: acc_o = acc_o * correction + P @ KV --
        acc_o = acc_o * correction[:, None]
        acc_o += tl.dot(p.to(tl.bfloat16), kv_block)  # (H, BLOCK) @ (BLOCK, D) = (H, D)

        scores_sum = tl.sum(p, axis=1)  # (H,)
        sum_exp = sum_exp * correction + scores_sum

    # ---- incorporate attn_sink ----
    sink_vals = tl.load(attn_sink + offs_h, mask=h_mask, other=0.0)  # (H,)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    # ---- normalize ----
    acc_o = acc_o / sum_exp[:, None]

    # ---- store output: (H, D) ----
    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    o = torch.empty_like(q)
    BLOCK = 64
    h_padded = max(16, triton.next_power_of_2(h))

    grid = (m, b)  # each program handles ALL h heads
    sparse_attn_triton_kernel[grid](
        q,
        kv,
        o,
        attn_sink,
        topk_idxs,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk_idxs.stride(2),
        softmax_scale,
        topk,
        h,
        BLOCK=BLOCK,
        D=d,
        H=h_padded,
        num_warps=8,  # 256 threads, matching tilelang
    )
    return o


# ---------------------------------------------------------------------------
# Quantized variant: int8 or fp8 symmetric quantization
#
# Quantization points (all symmetric, per-row / per-block scale):
#   1. KV is pre-quantized per 64-position block by _kv_quant_kernel:
#      scale_block = max|kv| / QMAX, kv_q = round(kv / scale_block).
#      The block scales are stored in KV_DESCALE (b, nblocks), and the
#      dequantization factor for each gathered position is
#      kv_desc[pos] = idxs[pos] // KV_BLOCK_SIZE, loaded together with the KV
#      block under the same valid_mask.
#   2. Q is quantized inside the kernel per head:
#      inv_q[h] = QMAX / max|q[h]|, q_q[h] = round(q[h] * inv_q[h]).
#   3. P (softmax output) is quantized per element with the KV descale folded
#      in, so the dequantization of the PV dot happens with a single /QMAX:
#      p_q = round(p * kv_desc * QMAX), out += dot(p_q, kv_q) / QMAX.
#
# QK dequantization happens in fp32 after the dot: acc_s = dot(q_q, kv_q)
# is multiplied by q_scale_h[h] * kv_desc[t] * scale, where q_scale_h =
# max|q[h]| / QMAX. The -inf padding mask is applied after dequantization,
# in the fp32 domain. The attention sink stays fp32 throughout.
# ---------------------------------------------------------------------------

QMAX_INT8 = 127.0
QMAX_FP8 = 448.0
KV_BLOCK_SIZE = 64
SPARSE_ATTN_BLOCK = tl.constexpr(64)


@triton.jit
def _quant_i8(x):
    return tl.clamp(tldevice.nearbyint(x), -128.0, 127.0).to(tl.int8)


@triton.jit
def _quant_fp8(x):
    return tl.clamp(x, -448.0, 448.0).to(tl.float8e4nv)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_attn_kv_quant"),
    key=["kv_len"],
    strategy=["default"],
    flagtune_op_name="sparse_attn_kv_quant",
)
@triton.jit
def _kv_quant_kernel(
    KV,
    KV_Q,
    KV_DESCALE,
    kv_len,
    num_blocks_per_row,
    IS_FP8: tl.constexpr,
    QMAX: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    PLACEHOLDER_UNUSED: tl.constexpr = 1,
):
    """Per-block symmetric quantization of the KV tensor.

    grid = (batch, nblocks). Each program covers one block of KV_BLOCK_SIZE
    positions of one batch row. The tail block (when kv_len is not a multiple
    of KV_BLOCK_SIZE) loads with a mask and padding zero, which does not move
    the block max-abs; the padded lanes store zero.
    """
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)

    offs_n = pid_blk * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
    offs_d = tl.arange(0, D)
    n_mask = offs_n < kv_len
    row_stride = kv_len * D

    kv_ptrs = KV + pid_b * row_stride + offs_n[:, None] * D + offs_d[None, :]
    src = tl.load(kv_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

    blk_max = tl.max(tl.abs(src))
    scale = blk_max / QMAX
    inv = tl.where(blk_max == 0.0, 0.0, 1.0 / scale)
    tl.store(KV_DESCALE + pid_b * num_blocks_per_row + pid_blk, scale)

    if IS_FP8:
        q = _quant_fp8(src * inv)
    else:
        q = _quant_i8(src * inv)
    tl.store(
        KV_Q + pid_b * row_stride + offs_n[:, None] * D + offs_d[None, :],
        q,
        mask=n_mask[:, None],
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_attn_quant"),
    key=["topk"],
    strategy=["default"],
    flagtune_op_name="sparse_attn_quant",
)
@triton.jit
def sparse_attn_triton_quant_kernel(
    Q,
    KV_Q,
    KV_DESCALE,
    O,
    attn_sink,
    topk_idxs,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_idxb,
    stride_idxm,
    stride_idxk,
    scale,
    topk,
    H_ACTUAL,
    nblocks,
    KV_BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    IS_FP8: tl.constexpr,
    QMAX: tl.constexpr,
    PLACEHOLDER_UNUSED: tl.constexpr = 1,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    # ---- load Q matrix: (H, D) — all heads at once ----
    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H_ACTUAL
    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0).to(tl.float32)

    # ---- per-head Q quantization (point 2) ----
    q_max = tl.max(tl.abs(q_block), axis=1)  # (H,)
    inv_q = tl.where(q_max == 0.0, 0.0, QMAX / q_max)
    if IS_FP8:
        q_q = _quant_fp8(q_block * inv_q[:, None])
    else:
        q_q = _quant_i8(q_block * inv_q[:, None])
    q_scale_h = q_max / QMAX  # (H,) row dequantization factor

    kv_q_base = KV_Q + pid_b * stride_kvb
    desc_base = KV_DESCALE + pid_b * nblocks
    idx_base = topk_idxs + pid_b * stride_idxb + pid_m * stride_idxm

    # ---- online softmax state ----
    acc_o = tl.zeros([H, D], dtype=tl.float32)
    scores_max = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)

    num_blocks = (topk + SPARSE_ATTN_BLOCK - 1) // SPARSE_ATTN_BLOCK
    offs_blk = tl.arange(0, SPARSE_ATTN_BLOCK)

    for t in range(num_blocks):
        # -- gather indices --
        raw_offs = t * SPARSE_ATTN_BLOCK + offs_blk  # (BLOCK,)
        idx_mask = raw_offs < topk
        idxs = tl.load(
            idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1
        )  # (BLOCK,)
        valid_mask = idxs != -1  # (BLOCK,)

        # -- gather quantized KV block and its per-position descale --
        kv_q_ptrs = (
            kv_q_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        )
        kv_q = tl.load(kv_q_ptrs, mask=valid_mask[:, None], other=0.0)
        kv_desc_offs = idxs // KV_BLOCK_SIZE  # (BLOCK,)
        kv_desc = tl.load(
            desc_base + kv_desc_offs, mask=valid_mask, other=0.0
        )  # (BLOCK,) fp32

        # -- scores: Q @ KV^T via GEMM, then fp32 dequantization --
        if IS_FP8:
            acc_s = tl.dot(q_q, tl.trans(kv_q), out_dtype=tl.float32)
        else:
            acc_s = tl.dot(q_q, tl.trans(kv_q), out_dtype=tl.int32).to(tl.float32)
        acc_s = acc_s * q_scale_h[:, None] * kv_desc[None, :] * scale
        # mask invalid positions to -inf (fp32 domain, after dequantization)
        acc_s = tl.where(valid_mask[None, :], acc_s, float("-inf"))

        # -- online softmax update --
        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)  # (H,)
        scores_max = tl.maximum(scores_max, block_max)

        correction = tl.exp(scores_max_prev - scores_max)  # (H,)
        p = tl.exp(acc_s - scores_max[:, None])  # (H, BLOCK)

        # -- quantized PV: descale folded into P, single /QMAX after dot --
        if IS_FP8:
            p_q = _quant_fp8(p * kv_desc[None, :] * QMAX)
            pv = tl.dot(p_q, kv_q, out_dtype=tl.float32)
        else:
            p_q = _quant_i8(p * kv_desc[None, :] * QMAX)
            pv = tl.dot(p_q, kv_q, out_dtype=tl.int32).to(tl.float32)

        acc_o = acc_o * correction[:, None]
        acc_o += pv * (1.0 / QMAX)

        scores_sum = tl.sum(p, axis=1)  # (H,)
        sum_exp = sum_exp * correction + scores_sum

    # ---- incorporate attn_sink (stays fp32) ----
    sink_vals = tl.load(attn_sink + offs_h, mask=h_mask, other=0.0)  # (H,)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    # ---- normalize ----
    acc_o = acc_o / sum_exp[:, None]

    # ---- store output: (H, D) ----
    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def sparse_attn_triton_quant(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    quant_dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    b, m, h, d = q.shape
    kv_len = kv.shape[1]
    topk = topk_idxs.shape[-1]
    o = torch.empty_like(q)
    h_padded = max(16, triton.next_power_of_2(h))
    nblocks = triton.cdiv(kv_len, KV_BLOCK_SIZE)

    if quant_dtype == torch.float8_e4m3fn:
        is_fp8 = True
        qmax = QMAX_FP8
    elif quant_dtype == torch.int8:
        is_fp8 = False
        qmax = QMAX_INT8
    else:
        raise NotImplementedError(
            f"quant_dtype {quant_dtype} is not supported; "
            "use torch.int8 or torch.float8_e4m3fn"
        )

    kv_q = torch.empty_like(kv, dtype=quant_dtype)
    kv_descale = torch.empty((b, nblocks), device=kv.device, dtype=torch.float32)

    _kv_quant_kernel[(b, nblocks)](
        kv,
        kv_q,
        kv_descale,
        kv_len,
        nblocks,
        IS_FP8=is_fp8,
        QMAX=qmax,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        D=d,
    )

    grid = (m, b)  # each program handles ALL h heads
    sparse_attn_triton_quant_kernel[grid](
        q,
        kv_q,
        kv_descale,
        o,
        attn_sink,
        topk_idxs,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_q.stride(0),
        kv_q.stride(1),
        kv_q.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk_idxs.stride(2),
        softmax_scale,
        topk,
        h,
        nblocks,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        D=d,
        H=h_padded,
        IS_FP8=is_fp8,
        QMAX=qmax,
    )
    return o
