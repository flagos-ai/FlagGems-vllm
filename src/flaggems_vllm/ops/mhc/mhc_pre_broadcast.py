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

"""
Triton implementation of mHC Pre Broadcast operator (first-layer variant).

At model entry all hc_mult residual streams are identical, so residual is
(T, H) instead of (T, hc_mult, H). This lets us:
  - project with a pre-summed fn_broadcast: (hc_mult3, H) instead of
    (hc_mult3, hc_mult*H)  [derive_fn_broadcast does this once after weight
    load, mirroring vLLM's "sum hc_attn_fn over the replicated input-stream
    dimension"]
  - compute the pre-norm sqrsum from a single H-wide pass, scaled by hc_mult
  - collapse the weighted sum into a scalar multiply (all streams equal)
  - broadcast-write residual_out: (T, hc_mult, H)

The output RMSNorm (norm_weight is mandatory here, mirroring vLLM's
mhc_pre_big_fuse_broadcast_with_norm_tilelang) reduces over the BF16-rounded
collapsed value, not the FP32 value, to match the non-fused reference path's
rounding boundary. We recompute the raw value in a second pass rather than
reserve a cross-pass buffer (Triton has no direct TileLang-style persistent
shared-memory stash across two Python-level loops in one kernel instance);
this costs one extra elementwise multiply pass, not an extra memory round-trip.
"""

from __future__ import annotations

import flag_gems
import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.data_movement import contiguous_copy


def derive_fn_broadcast(fn: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """Fold fn: (hc_mult3, hc_mult*H) into fn_broadcast: (hc_mult3, H) by
    summing over the hc_mult input-stream blocks.

    Call this once after checkpoint load / weight update, not per forward
    (mirrors vLLM PR #53972's weight-derivation step).
    """
    hc_mult3, hc_hidden_size = fn.shape
    hidden_size = hc_hidden_size // hc_mult
    return fn.view(hc_mult3, hc_mult, hidden_size).sum(dim=1).contiguous()


@triton.jit
def _mhc_pre_broadcast_fused_kernel_hc_mult_4_impl(
    gemm_out_ptr,       # (num_tokens, hc_mult3), float32 -- projection only
    hc_scale_ptr,       # (3,), float32
    hc_base_ptr,        # (hc_mult3,), float32
    residual_ptr,       # (num_tokens, hidden_size), bfloat16  -- (T, H)
    norm_weight_ptr,    # (hidden_size,), bfloat16
    post_mix_ptr,       # (num_tokens, hc_mult), float32
    comb_mix_ptr,       # (num_tokens, hc_mult*hc_mult), float32
    layer_input_ptr,    # (num_tokens, hidden_size), bfloat16
    residual_out_ptr,   # (num_tokens, hc_mult, hidden_size), bfloat16
    num_tokens,
    num_tokens_bucket,
    res_stride_n, res_stride_h,
    ro_stride_n, ro_stride_i, ro_stride_h,
    li_stride_n, li_stride_h,
    hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    norm_eps: tl.constexpr,
    HC_MULT3: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= num_tokens:
        return

    # ══ Pass 1: sqrsum over the single (T,H) copy, scaled by hc_mult(=4) ══
    # (matches: rms[0] computed from gemm_out_sqrsum in vLLM, which in turn
    #  is the plain sum-of-squares of the (T,H) input before projection;
    #  equivalent to summing sq over 4 identical copies.)
    sq_single = 0.0
    res_base = pid_n * res_stride_n
    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size
        v = tl.load(
            residual_ptr + res_base + h_offsets * res_stride_h,
            mask=h_mask, other=0.0,
        ).to(tl.float32)
        sq_single += tl.sum(v * v)
    sq = 4.0 * sq_single
    rms_inv = tl.rsqrt(sq / (4.0 * hidden_size) + rms_eps)

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)
    go_base = pid_n * HC_MULT3

    # ══ pre_mix: indices 0..3 ══
    pre_mix_0 = tl.sigmoid(tl.load(gemm_out_ptr + go_base + 0) * rms_inv * scale_0 + tl.load(hc_base_ptr + 0)) + hc_pre_eps
    pre_mix_1 = tl.sigmoid(tl.load(gemm_out_ptr + go_base + 1) * rms_inv * scale_0 + tl.load(hc_base_ptr + 1)) + hc_pre_eps
    pre_mix_2 = tl.sigmoid(tl.load(gemm_out_ptr + go_base + 2) * rms_inv * scale_0 + tl.load(hc_base_ptr + 2)) + hc_pre_eps
    pre_mix_3 = tl.sigmoid(tl.load(gemm_out_ptr + go_base + 3) * rms_inv * scale_0 + tl.load(hc_base_ptr + 3)) + hc_pre_eps
    pre_sum = pre_mix_0 + pre_mix_1 + pre_mix_2 + pre_mix_3

    # ══ post_mix: indices 4..7 ══
    post_0 = tl.sigmoid(tl.load(gemm_out_ptr + go_base + 4) * rms_inv * scale_1 + tl.load(hc_base_ptr + 4)) * hc_post_mult_value
    tl.store(post_mix_ptr + pid_n * 4 + 0, post_0)
    post_1 = tl.sigmoid(tl.load(gemm_out_ptr + go_base + 5) * rms_inv * scale_1 + tl.load(hc_base_ptr + 5)) * hc_post_mult_value
    tl.store(post_mix_ptr + pid_n * 4 + 1, post_1)
    post_2 = tl.sigmoid(tl.load(gemm_out_ptr + go_base + 6) * rms_inv * scale_1 + tl.load(hc_base_ptr + 6)) * hc_post_mult_value
    tl.store(post_mix_ptr + pid_n * 4 + 2, post_2)
    post_3 = tl.sigmoid(tl.load(gemm_out_ptr + go_base + 7) * rms_inv * scale_1 + tl.load(hc_base_ptr + 7)) * hc_post_mult_value
    tl.store(post_mix_ptr + pid_n * 4 + 3, post_3)

    # ══ comb_mix: indices 8..23 → 4x4 Sinkhorn (identical to mhc_pre) ══
    cb = 8
    cm_00 = tl.load(gemm_out_ptr + go_base + cb + 0) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 0)
    cm_01 = tl.load(gemm_out_ptr + go_base + cb + 1) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 1)
    cm_02 = tl.load(gemm_out_ptr + go_base + cb + 2) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 2)
    cm_03 = tl.load(gemm_out_ptr + go_base + cb + 3) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 3)
    cm_10 = tl.load(gemm_out_ptr + go_base + cb + 4) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 4)
    cm_11 = tl.load(gemm_out_ptr + go_base + cb + 5) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 5)
    cm_12 = tl.load(gemm_out_ptr + go_base + cb + 6) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 6)
    cm_13 = tl.load(gemm_out_ptr + go_base + cb + 7) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 7)
    cm_20 = tl.load(gemm_out_ptr + go_base + cb + 8) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 8)
    cm_21 = tl.load(gemm_out_ptr + go_base + cb + 9) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 9)
    cm_22 = tl.load(gemm_out_ptr + go_base + cb + 10) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 10)
    cm_23 = tl.load(gemm_out_ptr + go_base + cb + 11) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 11)
    cm_30 = tl.load(gemm_out_ptr + go_base + cb + 12) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 12)
    cm_31 = tl.load(gemm_out_ptr + go_base + cb + 13) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 13)
    cm_32 = tl.load(gemm_out_ptr + go_base + cb + 14) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 14)
    cm_33 = tl.load(gemm_out_ptr + go_base + cb + 15) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 15)

    rm = tl.maximum(tl.maximum(cm_00, cm_01), tl.maximum(cm_02, cm_03))
    cm_00 = tl.exp(cm_00 - rm); cm_01 = tl.exp(cm_01 - rm); cm_02 = tl.exp(cm_02 - rm); cm_03 = tl.exp(cm_03 - rm)
    rs = cm_00 + cm_01 + cm_02 + cm_03; inv_rs = 1.0 / rs
    cm_00 = cm_00 * inv_rs + hc_sinkhorn_eps; cm_01 = cm_01 * inv_rs + hc_sinkhorn_eps
    cm_02 = cm_02 * inv_rs + hc_sinkhorn_eps; cm_03 = cm_03 * inv_rs + hc_sinkhorn_eps

    rm = tl.maximum(tl.maximum(cm_10, cm_11), tl.maximum(cm_12, cm_13))
    cm_10 = tl.exp(cm_10 - rm); cm_11 = tl.exp(cm_11 - rm); cm_12 = tl.exp(cm_12 - rm); cm_13 = tl.exp(cm_13 - rm)
    rs = cm_10 + cm_11 + cm_12 + cm_13; inv_rs = 1.0 / rs
    cm_10 = cm_10 * inv_rs + hc_sinkhorn_eps; cm_11 = cm_11 * inv_rs + hc_sinkhorn_eps
    cm_12 = cm_12 * inv_rs + hc_sinkhorn_eps; cm_13 = cm_13 * inv_rs + hc_sinkhorn_eps

    rm = tl.maximum(tl.maximum(cm_20, cm_21), tl.maximum(cm_22, cm_23))
    cm_20 = tl.exp(cm_20 - rm); cm_21 = tl.exp(cm_21 - rm); cm_22 = tl.exp(cm_22 - rm); cm_23 = tl.exp(cm_23 - rm)
    rs = cm_20 + cm_21 + cm_22 + cm_23; inv_rs = 1.0 / rs
    cm_20 = cm_20 * inv_rs + hc_sinkhorn_eps; cm_21 = cm_21 * inv_rs + hc_sinkhorn_eps
    cm_22 = cm_22 * inv_rs + hc_sinkhorn_eps; cm_23 = cm_23 * inv_rs + hc_sinkhorn_eps

    rm = tl.maximum(tl.maximum(cm_30, cm_31), tl.maximum(cm_32, cm_33))
    cm_30 = tl.exp(cm_30 - rm); cm_31 = tl.exp(cm_31 - rm); cm_32 = tl.exp(cm_32 - rm); cm_33 = tl.exp(cm_33 - rm)
    rs = cm_30 + cm_31 + cm_32 + cm_33; inv_rs = 1.0 / rs
    cm_30 = cm_30 * inv_rs + hc_sinkhorn_eps; cm_31 = cm_31 * inv_rs + hc_sinkhorn_eps
    cm_32 = cm_32 * inv_rs + hc_sinkhorn_eps; cm_33 = cm_33 * inv_rs + hc_sinkhorn_eps

    cs0 = cm_00 + cm_10 + cm_20 + cm_30; cs1 = cm_01 + cm_11 + cm_21 + cm_31
    cs2 = cm_02 + cm_12 + cm_22 + cm_32; cs3 = cm_03 + cm_13 + cm_23 + cm_33
    inv_cs0 = 1.0 / (cs0 + hc_sinkhorn_eps); inv_cs1 = 1.0 / (cs1 + hc_sinkhorn_eps)
    inv_cs2 = 1.0 / (cs2 + hc_sinkhorn_eps); inv_cs3 = 1.0 / (cs3 + hc_sinkhorn_eps)
    cm_00 *= inv_cs0; cm_10 *= inv_cs0; cm_20 *= inv_cs0; cm_30 *= inv_cs0
    cm_01 *= inv_cs1; cm_11 *= inv_cs1; cm_21 *= inv_cs1; cm_31 *= inv_cs1
    cm_02 *= inv_cs2; cm_12 *= inv_cs2; cm_22 *= inv_cs2; cm_32 *= inv_cs2
    cm_03 *= inv_cs3; cm_13 *= inv_cs3; cm_23 *= inv_cs3; cm_33 *= inv_cs3

    for _ in tl.static_range(sinkhorn_repeat - 1):
        rs0 = cm_00 + cm_01 + cm_02 + cm_03; rs1 = cm_10 + cm_11 + cm_12 + cm_13
        rs2 = cm_20 + cm_21 + cm_22 + cm_23; rs3 = cm_30 + cm_31 + cm_32 + cm_33
        inv_rs0 = 1.0 / (rs0 + hc_sinkhorn_eps); inv_rs1 = 1.0 / (rs1 + hc_sinkhorn_eps)
        inv_rs2 = 1.0 / (rs2 + hc_sinkhorn_eps); inv_rs3 = 1.0 / (rs3 + hc_sinkhorn_eps)
        cm_00 *= inv_rs0; cm_01 *= inv_rs0; cm_02 *= inv_rs0; cm_03 *= inv_rs0
        cm_10 *= inv_rs1; cm_11 *= inv_rs1; cm_12 *= inv_rs1; cm_13 *= inv_rs1
        cm_20 *= inv_rs2; cm_21 *= inv_rs2; cm_22 *= inv_rs2; cm_23 *= inv_rs2
        cm_30 *= inv_rs3; cm_31 *= inv_rs3; cm_32 *= inv_rs3; cm_33 *= inv_rs3
        cs0 = cm_00 + cm_10 + cm_20 + cm_30; cs1 = cm_01 + cm_11 + cm_21 + cm_31
        cs2 = cm_02 + cm_12 + cm_22 + cm_32; cs3 = cm_03 + cm_13 + cm_23 + cm_33
        inv_cs0 = 1.0 / (cs0 + hc_sinkhorn_eps); inv_cs1 = 1.0 / (cs1 + hc_sinkhorn_eps)
        inv_cs2 = 1.0 / (cs2 + hc_sinkhorn_eps); inv_cs3 = 1.0 / (cs3 + hc_sinkhorn_eps)
        cm_00 *= inv_cs0; cm_01 *= inv_cs1; cm_02 *= inv_cs2; cm_03 *= inv_cs3
        cm_10 *= inv_cs0; cm_11 *= inv_cs1; cm_12 *= inv_cs2; cm_13 *= inv_cs3
        cm_20 *= inv_cs0; cm_21 *= inv_cs1; cm_22 *= inv_cs2; cm_23 *= inv_cs3
        cm_30 *= inv_cs0; cm_31 *= inv_cs1; cm_32 *= inv_cs2; cm_33 *= inv_cs3

    co = pid_n * 16
    tl.store(comb_mix_ptr + co + 0, cm_00); tl.store(comb_mix_ptr + co + 1, cm_01)
    tl.store(comb_mix_ptr + co + 2, cm_02); tl.store(comb_mix_ptr + co + 3, cm_03)
    tl.store(comb_mix_ptr + co + 4, cm_10); tl.store(comb_mix_ptr + co + 5, cm_11)
    tl.store(comb_mix_ptr + co + 6, cm_12); tl.store(comb_mix_ptr + co + 7, cm_13)
    tl.store(comb_mix_ptr + co + 8, cm_20); tl.store(comb_mix_ptr + co + 9, cm_21)
    tl.store(comb_mix_ptr + co + 10, cm_22); tl.store(comb_mix_ptr + co + 11, cm_23)
    tl.store(comb_mix_ptr + co + 12, cm_30); tl.store(comb_mix_ptr + co + 13, cm_31)
    tl.store(comb_mix_ptr + co + 14, cm_32); tl.store(comb_mix_ptr + co + 15, cm_33)

    # ══ Pass 2a: broadcast residual_out (unmodified residual, hc_mult copies) ══
    # ══           + accumulate sumsq of the BF16-ROUNDED raw collapse ══
    # Mirrors TileLang: output_shared stores bf16(ol); sumsq accumulates
    # float32(bf16(ol))^2, i.e. post-rounding, not the raw FP32 value.
    ro_base = pid_n * ro_stride_n
    sumsq_acc = 0.0
    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size
        r = tl.load(
            residual_ptr + res_base + h_offsets * res_stride_h,
            mask=h_mask, other=0.0,
        ).to(tl.float32)
        r_bf16 = r.to(tl.bfloat16)

        for k in tl.static_range(4):
            tl.store(
                residual_out_ptr + ro_base + k * ro_stride_i + h_offsets * ro_stride_h,
                r_bf16, mask=h_mask,
            )

        raw_bf16 = (pre_sum * r).to(tl.bfloat16)
        raw_rounded = raw_bf16.to(tl.float32)
        sumsq_acc += tl.sum(tl.where(h_mask, raw_rounded * raw_rounded, 0.0))

    norm_inv = tl.rsqrt(sumsq_acc / hidden_size + norm_eps)

    # ══ Pass 2b: recompute the BF16-rounded raw value, scale, write layer_input ══
    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size
        r = tl.load(
            residual_ptr + res_base + h_offsets * res_stride_h,
            mask=h_mask, other=0.0,
        ).to(tl.float32)
        raw_bf16 = (pre_sum * r).to(tl.bfloat16)
        raw_rounded = raw_bf16.to(tl.float32)
        w = tl.load(norm_weight_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
        out = raw_rounded * norm_inv * w
        tl.store(
            layer_input_ptr + pid_n * li_stride_n + h_offsets * li_stride_h,
            out.to(tl.bfloat16), mask=h_mask,
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=16, num_stages=1),
    ],
    key=["hidden_size", "num_tokens_bucket"],
)
@triton.jit
def mhc_pre_broadcast_fused_kernel_hc_mult_4(
    gemm_out_ptr, hc_scale_ptr, hc_base_ptr, residual_ptr, norm_weight_ptr,
    post_mix_ptr, comb_mix_ptr, layer_input_ptr, residual_out_ptr,
    num_tokens, num_tokens_bucket,
    res_stride_n, res_stride_h,
    ro_stride_n, ro_stride_i, ro_stride_h,
    li_stride_n, li_stride_h,
    hidden_size,
    rms_eps: tl.constexpr, hc_pre_eps: tl.constexpr, hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr, sinkhorn_repeat: tl.constexpr, norm_eps: tl.constexpr,
    HC_MULT3: tl.constexpr, BLOCK_H: tl.constexpr,
):
    _mhc_pre_broadcast_fused_kernel_hc_mult_4_impl(
        gemm_out_ptr, hc_scale_ptr, hc_base_ptr, residual_ptr, norm_weight_ptr,
        post_mix_ptr, comb_mix_ptr, layer_input_ptr, residual_out_ptr,
        num_tokens, num_tokens_bucket,
        res_stride_n, res_stride_h,
        ro_stride_n, ro_stride_i, ro_stride_h,
        li_stride_n, li_stride_h,
        hidden_size,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_eps,
        HC_MULT3, BLOCK_H,
    )


def mhc_pre_broadcast(
    residual: torch.Tensor,        # (T, H), bfloat16
    fn_broadcast: torch.Tensor,    # (hc_mult3, H), float32 — see derive_fn_broadcast
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor,     # (H,), required (mirrors vLLM's assert)
    norm_eps: float = 1e-6,
    hc_mult: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-layer mHC pre for a residual broadcast from (T, H).

    Only the hc_mult == 4 fast path is implemented, mirroring mhc_pre's own
    specialization. Port mhc_pre_generic_kernel's style for other hc_mult.
    """
    assert residual.dtype == torch.bfloat16 and residual.dim() == 2
    assert fn_broadcast.dtype == torch.float32
    assert norm_weight is not None, "broadcast mHC pre requires fused RMSNorm"
    assert hc_mult == 4, "only hc_mult == 4 fast path implemented"

    num_tokens, hidden_size = residual.shape
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    assert fn_broadcast.shape == (hc_mult3, hidden_size)
    device = residual.device

    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    norm_weight = contiguous_copy(norm_weight)

    if num_tokens == 0:
        return (
            torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device),
            torch.empty(num_tokens, hc_mult, 1, dtype=torch.float32, device=device),
            torch.empty(num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=device),
            torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device),
        )

    if num_tokens <= 512:
        num_tokens_bucket = 1
    elif num_tokens <= 1024:
        num_tokens_bucket = 2
    elif num_tokens <= 2048:
        num_tokens_bucket = 3
    elif num_tokens <= 4096:
        num_tokens_bucket = 4
    else:
        num_tokens_bucket = 5

    residual = contiguous_copy(residual)
    bmm_out = getattr(flag_gems, "bmm_out", None)
    if not callable(bmm_out):
        raise NotImplementedError("MHC projection requires public FlagGems bmm_out")

    fn_broadcast_bf16 = contiguous_copy(fn_broadcast, torch.bfloat16)
    projection = torch.empty((num_tokens, hc_mult3), dtype=torch.bfloat16, device=device)
    bmm_out(
        residual.unsqueeze(0), fn_broadcast_bf16.t().unsqueeze(0), out=projection.unsqueeze(0)
    )
    gemm_out = contiguous_copy(projection, torch.float32)

    residual_out = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    mhc_pre_broadcast_fused_kernel_hc_mult_4[(num_tokens,)](
        gemm_out, hc_scale, hc_base, residual, norm_weight,
        post_mix, comb_mix, layer_input, residual_out,
        num_tokens, num_tokens_bucket,
        residual.stride(0), residual.stride(1),
        residual_out.stride(0), residual_out.stride(1), residual_out.stride(2),
        layer_input.stride(0), layer_input.stride(1),
        hidden_size,
        rms_eps=rms_eps, hc_pre_eps=hc_pre_eps, hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value, sinkhorn_repeat=sinkhorn_repeat, norm_eps=norm_eps,
        HC_MULT3=hc_mult3,
    )

    return (
        residual_out,
        post_mix.unsqueeze(-1),
        comb_mix.view(num_tokens, hc_mult, hc_mult),
        layer_input,
    )


# ───────────────────────── Reference implementation ─────────────────────────



def _sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """PyTorch reference Sinkhorn normalization (local copy, avoids a
    cross-module dependency on mhc_pre.py's accuracy-test-only helper)."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x

def mhc_pre_broadcast_ref(
    residual: torch.Tensor,        # (T, H), bfloat16
    fn_broadcast: torch.Tensor,    # (hc_mult3, H), float32
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor,
    norm_eps: float = 1e-6,
    hc_mult: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference, mirroring the fused Triton kernel's exact rounding
    order (sqrsum of the BF16-rounded collapsed value, not the FP32 value).
    """

    num_tokens, hidden_size = residual.shape
    residual_f32 = residual.float()
    sqrsum = hc_mult * residual_f32.square().sum(-1)  # 4 identical copies
    mixes = residual_f32 @ fn_broadcast.T * (
        sqrsum.unsqueeze(-1) / hidden_size + rms_eps
    ).rsqrt()

    hc_scale_expanded = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * hc_scale_expanded + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid() + hc_pre_eps  # (T, hc_mult)
    pre_sum = pre_mix.sum(-1, keepdim=True)              # (T, 1)

    post_mix = (mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult)
    res_mix = _sinkhorn_normalize_ref(res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps)

    # Collapse, round to BF16 (matches kernel's output_shared boundary),
    # THEN compute sqrsum from the rounded value.
    raw = pre_sum * residual_f32
    raw_bf16 = raw.bfloat16()
    raw_rounded = raw_bf16.float()
    sumsq = raw_rounded.square().sum(-1, keepdim=True)
    norm_inv = (sumsq / hidden_size + norm_eps).rsqrt()
    layer_input = (raw_rounded * norm_inv * norm_weight.float()).bfloat16()

    residual_out = residual.unsqueeze(1).expand(num_tokens, hc_mult, hidden_size).contiguous()

    return residual_out, post_mix, res_mix, layer_input

