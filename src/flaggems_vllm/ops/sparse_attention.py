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

QMAX_INT8 = 127.0
QMAX_FP8 = 448.0
KV_BLOCK_SIZE = 64


@triton.jit
def _quant_i8(x):
    return tl.clamp(tldevice.nearbyint(x), -128.0, 127.0).to(tl.int8)


@triton.jit
def _quant_fp8(x):
    return tl.clamp(x, -448.0, 448.0).to(tl.float8e4nv)


# ---------------------------------------------------------------------------
# Triton kernel: sparse attention with attention-sink
# grid = (m, b) — one program per (seq_pos, batch), handles ALL heads
# Aligned with tilelang version: uses tl.dot (GEMM) instead of vector dot
# ---------------------------------------------------------------------------
@triton.jit
def sparse_attn_triton_kernel(
    Q,  # (b, m, h, d) bf16
    KV,  # (b, n, d)    bf16
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
    """Sparse attention over a full-precision (bf16) KV tensor.

    Args:
        q: (B, M, H, D) bf16.
        kv: (B, kv_len, D) bf16.
        attn_sink: (H,) fp32.
        topk_idxs: (B, M, topk) int32.
        softmax_scale: fp32 softmax temperature.

    Returns:
        (B, M, H, D) bf16 attention output.
    """
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_attn_quant"),
    key=["topk"],
    strategy=["default"],
    flagtune_op_name="sparse_attn_quant",
)
@triton.jit
def sparse_attn_triton_quant_kernel(
    Q,  # (b, m, h, d) bf16
    KV_Q,  # (b, n, d) int8 quantized KV
    KV_DESCALE,  # (b, nblocks) fp32 per-64-block descale
    O,  # (b, m, h, d) bf16
    attn_sink,  # (h,) fp32
    topk_idxs,  # (b, m, topk) int32
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
    D: tl.constexpr,
    HP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    h_base = pid_h * HP

    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, HP)
    offs_d = tl.arange(0, D)
    h_mask = (h_base + offs_h) < H_ACTUAL
    q_ptrs = (
        q_base + (h_base + offs_h)[:, None] * stride_qh + offs_d[None, :] * stride_qd
    )
    q_f = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0).to(tl.float32)

    q_max = tl.max(tl.abs(q_f), axis=1)
    inv_q = tl.where(q_max == 0.0, 0.0, 127.0 / q_max)
    q_q = _quant_i8(q_f * inv_q[:, None])
    q_scale_h = q_max / 127.0

    kv_q_base = KV_Q + pid_b * stride_kvb
    desc_base = KV_DESCALE + pid_b * nblocks
    idx_base = topk_idxs + pid_b * stride_idxb + pid_m * stride_idxm

    acc_o = tl.zeros([HP, D], dtype=tl.float32)
    scores_max = tl.full([HP], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([HP], dtype=tl.float32)

    num_blocks = (topk + BLOCK - 1) // BLOCK
    offs_blk = tl.arange(0, BLOCK)

    for t in range(num_blocks):
        raw_offs = t * BLOCK + offs_blk
        idx_mask = raw_offs < topk
        idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1)
        valid_mask = idxs != -1

        kv_desc = tl.load(desc_base + (idxs >> 6), mask=valid_mask, other=0.0)
        kv_q = tl.load(
            kv_q_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd,
            mask=valid_mask[:, None],
            other=0.0,
        )

        qk = tl.dot(q_q, tl.trans(kv_q), out_dtype=tl.int32).to(tl.float32)
        acc_s = qk * (q_scale_h * scale)[:, None] * kv_desc[None, :]
        acc_s = tl.where(valid_mask[None, :], acc_s, float("-inf"))

        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)
        scores_max = tl.maximum(scores_max, block_max)
        correction = tl.exp(scores_max_prev - scores_max)
        p = tl.exp(acc_s - scores_max[:, None])

        p_desc = p * kv_desc[None, :]
        pd_max = tl.max(p_desc, axis=1)
        inv_pd = tl.where(pd_max == 0.0, 0.0, 127.0 / pd_max)
        p_q = _quant_i8(p_desc * inv_pd[:, None])
        pv = tl.dot(p_q, kv_q, out_dtype=tl.int32).to(tl.float32)

        acc_o = acc_o * correction[:, None]
        acc_o += pv * (pd_max / 127.0)[:, None]
        sum_exp = sum_exp * correction + tl.sum(p, axis=1)

    sink_vals = tl.load(attn_sink + (h_base + offs_h), mask=h_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    acc_o = acc_o / sum_exp[:, None]

    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = (
        o_base + (h_base + offs_h)[:, None] * stride_oh + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_attn_quant_fp8"),
    key=["topk"],
    strategy=["default"],
    flagtune_op_name="sparse_attn_quant_fp8",
)
@triton.jit
def sparse_attn_triton_quant_fp8_kernel(
    Q,  # (b, m, h, d) bf16
    KV_Q,  # (b, n, d) fp8_e4m3fn quantized KV
    KV_DESCALE,  # (b, nblocks) fp32 per-64-block descale
    O,  # (b, m, h, d) bf16
    attn_sink,  # (h,) fp32
    topk_idxs,  # (b, m, topk) int32
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
    D: tl.constexpr,
    HP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    h_base = pid_h * HP

    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, HP)
    offs_d = tl.arange(0, D)
    h_mask = (h_base + offs_h) < H_ACTUAL
    q_ptrs = (
        q_base + (h_base + offs_h)[:, None] * stride_qh + offs_d[None, :] * stride_qd
    )
    q_f = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0).to(tl.float32)

    q_max = tl.max(tl.abs(q_f), axis=1)
    inv_q = tl.where(q_max == 0.0, 0.0, 448.0 / q_max)
    q_q = _quant_fp8(q_f * inv_q[:, None])
    q_scale_h = q_max / 448.0

    kv_q_base = KV_Q + pid_b * stride_kvb
    desc_base = KV_DESCALE + pid_b * nblocks
    idx_base = topk_idxs + pid_b * stride_idxb + pid_m * stride_idxm

    acc_o = tl.zeros([HP, D], dtype=tl.float32)
    scores_max = tl.full([HP], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([HP], dtype=tl.float32)

    num_blocks = (topk + BLOCK - 1) // BLOCK
    offs_blk = tl.arange(0, BLOCK)

    for t in range(num_blocks):
        raw_offs = t * BLOCK + offs_blk
        idx_mask = raw_offs < topk
        idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1)
        valid_mask = idxs != -1

        kv_desc = tl.load(desc_base + (idxs >> 6), mask=valid_mask, other=0.0)
        kv_q = tl.load(
            kv_q_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd,
            mask=valid_mask[:, None],
            other=0.0,
        )

        qk = tl.dot(q_q, tl.trans(kv_q), out_dtype=tl.float32)
        acc_s = qk * (q_scale_h * scale)[:, None] * kv_desc[None, :]
        acc_s = tl.where(valid_mask[None, :], acc_s, float("-inf"))

        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)
        scores_max = tl.maximum(scores_max, block_max)
        correction = tl.exp(scores_max_prev - scores_max)
        p = tl.exp(acc_s - scores_max[:, None])

        kv_bf = (kv_q.to(tl.float32) * kv_desc[:, None]).to(tl.bfloat16)
        acc_o = acc_o * correction[:, None]
        acc_o += tl.dot(p.to(tl.bfloat16), kv_bf)
        sum_exp = sum_exp * correction + tl.sum(p, axis=1)

    sink_vals = tl.load(attn_sink + (h_base + offs_h), mask=h_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    acc_o = acc_o / sum_exp[:, None]

    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = (
        o_base + (h_base + offs_h)[:, None] * stride_oh + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])


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


def sparse_attn_triton_quant_int8(
    q: torch.Tensor,
    kv_q: torch.Tensor,
    kv_descale: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse attention over an INT8-quantized KV cache.

    Args:
        q: (B, M, H, D) bf16.
        kv_q: (B, kv_len, D) int8 — quantized K+V cache.
        kv_descale: (B, ceil(kv_len / 64)) fp32 — one descale per 64 block.
        attn_sink: (H,) fp32.
        topk_idxs: (B, M, topk) int32.
        softmax_scale: fp32 softmax temperature.

    Returns:
        (B, M, H, D) bf16 attention output.
    """
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    nblocks = kv_descale.shape[-1]
    o = torch.empty_like(q)
    HP = min(32, triton.next_power_of_2(h))

    grid = (m, b, triton.cdiv(h, HP))
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
        D=d,
        HP=HP,
    )
    return o


def sparse_attn_triton_quant_fp8(
    q: torch.Tensor,
    kv_q: torch.Tensor,
    kv_descale: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse attention over an FP8(e4m3)-quantized KV cache.

    Args:
        q: (B, M, H, D) bf16.
        kv_q: (B, kv_len, D) fp8_e4m3fn — quantized K+V cache.
        kv_descale: (B, ceil(kv_len / 64)) fp32 — one descale per 64 block.
        attn_sink: (H,) fp32.
        topk_idxs: (B, M, topk) int32.
        softmax_scale: fp32 softmax temperature.

    Returns:
        (B, M, H, D) bf16 attention output.
    """
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    nblocks = kv_descale.shape[-1]
    o = torch.empty_like(q)
    HP = min(32, triton.next_power_of_2(h))

    grid = (m, b, triton.cdiv(h, HP))
    sparse_attn_triton_quant_fp8_kernel[grid](
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
        D=d,
        HP=HP,
    )
    return o
