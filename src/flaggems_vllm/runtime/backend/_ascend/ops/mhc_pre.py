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

"""Standalone mHC pre operator for Ascend NPU: ``npu_mhc_pre``.

Port of the generic Triton ``mhc_pre`` variants (copied from
``flaggems_vllm.ops.mhc.mhc_pre`` so that the generic fallback stays
untouched), with two deliberate deltas:

* **Model contract ``clamp(-30, 30)`` before Sinkhorn** - ``clamp_logits``
  is a compile-time flag that is always ``True`` on this path. Provenance:
  XingChen4-29B-A4B checkpoint contract (formal_issues/OP-ACC-003,
  REVIEW.md: "Model contract requires clamp(-30, 30) before Sinkhorn").
* **Dynamic Sinkhorn loop** - ``sinkhorn_repeat`` is a runtime argument
  (``range`` instead of ``tl.static_range``), so ONE compiled artifact
  serves every repeat count; first-call compilation drops from ~215-350s
  (fresh constexpr set x ~10 autotune configs, measured baseline) to a
  single small compile. P0-verified on npu: dynamic bound n=3 -> n=20
  reuses the same artifact (0.000s), numerics bitwise-identical.

Shape/dtype contract (ValueError) and the NPU-only device gate are kept
from the original standalone implementation; outputs follow the generic
contract: post ``(..., hc, 1)``, comb ``(..., hc, hc)``, hin ``(..., D)``.
TLE twins are NOT ported (triton 3.2.0 < 3.6 gate is false on this box).
"""

from __future__ import annotations

import weakref

import torch
import triton
import triton.language as tl

__all__ = ["npu_mhc_pre"]

# XingChen4 checkpoint contract: comb logits are clamped into this range
# before the Sinkhorn rounds (formal_issues/OP-ACC-003).
CLAMP_MIN = -30.0
CLAMP_MAX = 30.0


_FN_BF16_CACHE: weakref.WeakKeyDictionary[torch.Tensor, tuple[int, torch.Tensor]] = (
    weakref.WeakKeyDictionary()
)

def _get_fn_bf16_cached(fn: torch.Tensor) -> torch.Tensor:
    if fn.requires_grad or torch.is_grad_enabled():
        return fn.to(dtype=torch.bfloat16)
    version = fn._version
    cached = _FN_BF16_CACHE.get(fn)
    if cached is not None:
        cached_version, cached_bf16 = cached
        if cached_version == version:
            return cached_bf16
    fn_bf16 = fn.to(dtype=torch.bfloat16)
    _FN_BF16_CACHE[fn] = (version, fn_bf16)
    return fn_bf16

@triton.jit
def _mhc_pre_fused_kernel_hc_mult_4_impl(
    gemm_out_ptr,  # (num_tokens, hc_mult3), float32
    hc_scale_ptr,  # (3,), float32
    hc_base_ptr,  # (hc_mult3,), float32
    residual_ptr,  # (num_tokens, hc_mult, hidden_size), bfloat16
    post_mix_ptr,  # (num_tokens, hc_mult), float32
    comb_mix_ptr,  # (num_tokens, hc_mult*hc_mult), float32
    layer_input_ptr,  # (num_tokens, hidden_size), bfloat16
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat,
    clamp_logits: tl.constexpr,
    HC_MULT3: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fully fused: sqrsum + RMS norm + sigmoid + Sinkhorn + weighted sum. One token per program."""
    pid_n = tl.program_id(0)
    if pid_n >= num_tokens:
        return

    # ══ Pass 1: compute sqrsum over all 4 heads ══
    sq = 0.0
    res_base = pid_n * res_stride_n
    for k in tl.static_range(4):
        head_base = res_base + k * res_stride_i
        for h_start in range(0, hidden_size, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < hidden_size
            v = tl.load(
                residual_ptr + head_base + h_offsets * res_stride_h,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            sq += tl.sum(v * v)

    rms_inv = tl.rsqrt(sq / hc_hidden_size + rms_eps)

    # ══ Load scales ══
    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    go_base = pid_n * HC_MULT3

    # ══ pre_mix: indices 0..3 ══
    pre_mix_0 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 0) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 0)
        )
        + hc_pre_eps
    )
    pre_mix_1 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 1) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 1)
        )
        + hc_pre_eps
    )
    pre_mix_2 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 2) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 2)
        )
        + hc_pre_eps
    )
    pre_mix_3 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 3) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 3)
        )
        + hc_pre_eps
    )

    # ══ post_mix: indices 4..7 ══
    post_0 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 4) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 4)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid_n * 4 + 0, post_0)
    post_1 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 5) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 5)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid_n * 4 + 1, post_1)
    post_2 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 6) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 6)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid_n * 4 + 2, post_2)
    post_3 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 7) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 7)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid_n * 4 + 3, post_3)

    # ══ comb_mix: indices 8..23 → 4x4 Sinkhorn ══
    cb = 8
    cm_00 = tl.load(gemm_out_ptr + go_base + cb + 0) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 0
    )
    cm_01 = tl.load(gemm_out_ptr + go_base + cb + 1) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 1
    )
    cm_02 = tl.load(gemm_out_ptr + go_base + cb + 2) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 2
    )
    cm_03 = tl.load(gemm_out_ptr + go_base + cb + 3) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 3
    )
    cm_10 = tl.load(gemm_out_ptr + go_base + cb + 4) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 4
    )
    cm_11 = tl.load(gemm_out_ptr + go_base + cb + 5) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 5
    )
    cm_12 = tl.load(gemm_out_ptr + go_base + cb + 6) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 6
    )
    cm_13 = tl.load(gemm_out_ptr + go_base + cb + 7) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 7
    )
    cm_20 = tl.load(gemm_out_ptr + go_base + cb + 8) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 8
    )
    cm_21 = tl.load(gemm_out_ptr + go_base + cb + 9) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 9
    )
    cm_22 = tl.load(gemm_out_ptr + go_base + cb + 10) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 10
    )
    cm_23 = tl.load(gemm_out_ptr + go_base + cb + 11) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 11
    )
    cm_30 = tl.load(gemm_out_ptr + go_base + cb + 12) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 12
    )
    cm_31 = tl.load(gemm_out_ptr + go_base + cb + 13) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 13
    )
    cm_32 = tl.load(gemm_out_ptr + go_base + cb + 14) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 14
    )
    cm_33 = tl.load(gemm_out_ptr + go_base + cb + 15) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 15
    )

    if clamp_logits:
        cm_00 = tl.clamp(cm_00, -30.0, 30.0)
        cm_01 = tl.clamp(cm_01, -30.0, 30.0)
        cm_02 = tl.clamp(cm_02, -30.0, 30.0)
        cm_03 = tl.clamp(cm_03, -30.0, 30.0)
        cm_10 = tl.clamp(cm_10, -30.0, 30.0)
        cm_11 = tl.clamp(cm_11, -30.0, 30.0)
        cm_12 = tl.clamp(cm_12, -30.0, 30.0)
        cm_13 = tl.clamp(cm_13, -30.0, 30.0)
        cm_20 = tl.clamp(cm_20, -30.0, 30.0)
        cm_21 = tl.clamp(cm_21, -30.0, 30.0)
        cm_22 = tl.clamp(cm_22, -30.0, 30.0)
        cm_23 = tl.clamp(cm_23, -30.0, 30.0)
        cm_30 = tl.clamp(cm_30, -30.0, 30.0)
        cm_31 = tl.clamp(cm_31, -30.0, 30.0)
        cm_32 = tl.clamp(cm_32, -30.0, 30.0)
        cm_33 = tl.clamp(cm_33, -30.0, 30.0)

    # ── Sinkhorn iteration ──
    rm = tl.maximum(tl.maximum(cm_00, cm_01), tl.maximum(cm_02, cm_03))
    cm_00 = tl.exp(cm_00 - rm)
    cm_01 = tl.exp(cm_01 - rm)
    cm_02 = tl.exp(cm_02 - rm)
    cm_03 = tl.exp(cm_03 - rm)
    rs = cm_00 + cm_01 + cm_02 + cm_03
    inv_rs = 1.0 / rs
    cm_00 = cm_00 * inv_rs + hc_sinkhorn_eps
    cm_01 = cm_01 * inv_rs + hc_sinkhorn_eps
    cm_02 = cm_02 * inv_rs + hc_sinkhorn_eps
    cm_03 = cm_03 * inv_rs + hc_sinkhorn_eps

    rm = tl.maximum(tl.maximum(cm_10, cm_11), tl.maximum(cm_12, cm_13))
    cm_10 = tl.exp(cm_10 - rm)
    cm_11 = tl.exp(cm_11 - rm)
    cm_12 = tl.exp(cm_12 - rm)
    cm_13 = tl.exp(cm_13 - rm)
    rs = cm_10 + cm_11 + cm_12 + cm_13
    inv_rs = 1.0 / rs
    cm_10 = cm_10 * inv_rs + hc_sinkhorn_eps
    cm_11 = cm_11 * inv_rs + hc_sinkhorn_eps
    cm_12 = cm_12 * inv_rs + hc_sinkhorn_eps
    cm_13 = cm_13 * inv_rs + hc_sinkhorn_eps

    rm = tl.maximum(tl.maximum(cm_20, cm_21), tl.maximum(cm_22, cm_23))
    cm_20 = tl.exp(cm_20 - rm)
    cm_21 = tl.exp(cm_21 - rm)
    cm_22 = tl.exp(cm_22 - rm)
    cm_23 = tl.exp(cm_23 - rm)
    rs = cm_20 + cm_21 + cm_22 + cm_23
    inv_rs = 1.0 / rs
    cm_20 = cm_20 * inv_rs + hc_sinkhorn_eps
    cm_21 = cm_21 * inv_rs + hc_sinkhorn_eps
    cm_22 = cm_22 * inv_rs + hc_sinkhorn_eps
    cm_23 = cm_23 * inv_rs + hc_sinkhorn_eps

    rm = tl.maximum(tl.maximum(cm_30, cm_31), tl.maximum(cm_32, cm_33))
    cm_30 = tl.exp(cm_30 - rm)
    cm_31 = tl.exp(cm_31 - rm)
    cm_32 = tl.exp(cm_32 - rm)
    cm_33 = tl.exp(cm_33 - rm)
    rs = cm_30 + cm_31 + cm_32 + cm_33
    inv_rs = 1.0 / rs
    cm_30 = cm_30 * inv_rs + hc_sinkhorn_eps
    cm_31 = cm_31 * inv_rs + hc_sinkhorn_eps
    cm_32 = cm_32 * inv_rs + hc_sinkhorn_eps
    cm_33 = cm_33 * inv_rs + hc_sinkhorn_eps

    cs0 = cm_00 + cm_10 + cm_20 + cm_30
    cs1 = cm_01 + cm_11 + cm_21 + cm_31
    cs2 = cm_02 + cm_12 + cm_22 + cm_32
    cs3 = cm_03 + cm_13 + cm_23 + cm_33
    inv_cs0 = 1.0 / (cs0 + hc_sinkhorn_eps)
    inv_cs1 = 1.0 / (cs1 + hc_sinkhorn_eps)
    inv_cs2 = 1.0 / (cs2 + hc_sinkhorn_eps)
    inv_cs3 = 1.0 / (cs3 + hc_sinkhorn_eps)
    cm_00 *= inv_cs0
    cm_10 *= inv_cs0
    cm_20 *= inv_cs0
    cm_30 *= inv_cs0
    cm_01 *= inv_cs1
    cm_11 *= inv_cs1
    cm_21 *= inv_cs1
    cm_31 *= inv_cs1
    cm_02 *= inv_cs2
    cm_12 *= inv_cs2
    cm_22 *= inv_cs2
    cm_32 *= inv_cs2
    cm_03 *= inv_cs3
    cm_13 *= inv_cs3
    cm_23 *= inv_cs3
    cm_33 *= inv_cs3

    for _ in range(sinkhorn_repeat - 1):
        rs0 = cm_00 + cm_01 + cm_02 + cm_03
        rs1 = cm_10 + cm_11 + cm_12 + cm_13
        rs2 = cm_20 + cm_21 + cm_22 + cm_23
        rs3 = cm_30 + cm_31 + cm_32 + cm_33
        inv_rs0 = 1.0 / (rs0 + hc_sinkhorn_eps)
        inv_rs1 = 1.0 / (rs1 + hc_sinkhorn_eps)
        inv_rs2 = 1.0 / (rs2 + hc_sinkhorn_eps)
        inv_rs3 = 1.0 / (rs3 + hc_sinkhorn_eps)
        cm_00 *= inv_rs0
        cm_01 *= inv_rs0
        cm_02 *= inv_rs0
        cm_03 *= inv_rs0
        cm_10 *= inv_rs1
        cm_11 *= inv_rs1
        cm_12 *= inv_rs1
        cm_13 *= inv_rs1
        cm_20 *= inv_rs2
        cm_21 *= inv_rs2
        cm_22 *= inv_rs2
        cm_23 *= inv_rs2
        cm_30 *= inv_rs3
        cm_31 *= inv_rs3
        cm_32 *= inv_rs3
        cm_33 *= inv_rs3
        cs0 = cm_00 + cm_10 + cm_20 + cm_30
        cs1 = cm_01 + cm_11 + cm_21 + cm_31
        cs2 = cm_02 + cm_12 + cm_22 + cm_32
        cs3 = cm_03 + cm_13 + cm_23 + cm_33
        inv_cs0 = 1.0 / (cs0 + hc_sinkhorn_eps)
        inv_cs1 = 1.0 / (cs1 + hc_sinkhorn_eps)
        inv_cs2 = 1.0 / (cs2 + hc_sinkhorn_eps)
        inv_cs3 = 1.0 / (cs3 + hc_sinkhorn_eps)
        cm_00 *= inv_cs0
        cm_01 *= inv_cs1
        cm_02 *= inv_cs2
        cm_03 *= inv_cs3
        cm_10 *= inv_cs0
        cm_11 *= inv_cs1
        cm_12 *= inv_cs2
        cm_13 *= inv_cs3
        cm_20 *= inv_cs0
        cm_21 *= inv_cs1
        cm_22 *= inv_cs2
        cm_23 *= inv_cs3
        cm_30 *= inv_cs0
        cm_31 *= inv_cs1
        cm_32 *= inv_cs2
        cm_33 *= inv_cs3

    co = pid_n * 16
    tl.store(comb_mix_ptr + co + 0, cm_00)
    tl.store(comb_mix_ptr + co + 1, cm_01)
    tl.store(comb_mix_ptr + co + 2, cm_02)
    tl.store(comb_mix_ptr + co + 3, cm_03)
    tl.store(comb_mix_ptr + co + 4, cm_10)
    tl.store(comb_mix_ptr + co + 5, cm_11)
    tl.store(comb_mix_ptr + co + 6, cm_12)
    tl.store(comb_mix_ptr + co + 7, cm_13)
    tl.store(comb_mix_ptr + co + 8, cm_20)
    tl.store(comb_mix_ptr + co + 9, cm_21)
    tl.store(comb_mix_ptr + co + 10, cm_22)
    tl.store(comb_mix_ptr + co + 11, cm_23)
    tl.store(comb_mix_ptr + co + 12, cm_30)
    tl.store(comb_mix_ptr + co + 13, cm_31)
    tl.store(comb_mix_ptr + co + 14, cm_32)
    tl.store(comb_mix_ptr + co + 15, cm_33)

    # ══ Pass 2: weighted sum  layer_input = sum_k(pre_mix_k * residual[n, k, :]) ══
    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size
        r0 = tl.load(
            residual_ptr + res_base + 0 * res_stride_i + h_offsets * res_stride_h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        r1 = tl.load(
            residual_ptr + res_base + 1 * res_stride_i + h_offsets * res_stride_h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        acc = pre_mix_0 * r0 + pre_mix_1 * r1
        r2 = tl.load(
            residual_ptr + res_base + 2 * res_stride_i + h_offsets * res_stride_h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        r3 = tl.load(
            residual_ptr + res_base + 3 * res_stride_i + h_offsets * res_stride_h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre_mix_2 * r2 + pre_mix_3 * r3
        tl.store(
            layer_input_ptr + pid_n * li_stride_n + h_offsets * li_stride_h,
            acc.to(tl.bfloat16),
            mask=h_mask,
        )



@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
    ],
    key=["hidden_size", "num_tokens_bucket"],
)
@triton.jit
def mhc_pre_fused_kernel_hc_mult_4(
    gemm_out_ptr,  # (num_tokens, hc_mult3), float32
    hc_scale_ptr,  # (3,), float32
    hc_base_ptr,  # (hc_mult3,), float32
    residual_ptr,  # (num_tokens, hc_mult, hidden_size), bfloat16
    post_mix_ptr,  # (num_tokens, hc_mult), float32
    comb_mix_ptr,  # (num_tokens, hc_mult*hc_mult), float32
    layer_input_ptr,  # (num_tokens, hidden_size), bfloat16
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat,
    clamp_logits: tl.constexpr,
    HC_MULT3: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    _mhc_pre_fused_kernel_hc_mult_4_impl(
        gemm_out_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        residual_ptr,
        post_mix_ptr,
        comb_mix_ptr,
        layer_input_ptr,
        num_tokens,
        num_tokens_bucket,
        res_stride_n,
        res_stride_i,
        res_stride_h,
        li_stride_n,
        li_stride_h,
        hidden_size,
        hc_hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        clamp_logits,
        HC_MULT3,
        BLOCK_H,
    )



@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
    ],
    key=["hidden_size", "num_tokens_bucket", "HC"],
)
@triton.jit
def mhc_pre_generic_kernel(
    gemm_out_ptr,  # (num_tokens, hc_mult3), float32
    hc_scale_ptr,  # (3,), float32
    hc_base_ptr,  # (hc_mult3,), float32
    residual_ptr,  # (num_tokens, HC, hidden_size), bfloat16
    post_mix_ptr,  # (num_tokens, HC), float32
    comb_mix_ptr,  # (num_tokens, HC*HC), float32
    layer_input_ptr,  # (num_tokens, hidden_size), bfloat16
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat,
    clamp_logits: tl.constexpr,
    HC,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= num_tokens:
        return

    res_base = pid_n * res_stride_n
    go_base = pid_n * (HC * 2 + HC * HC)
    comb_base = pid_n * (HC * HC)

    sq = 0.0
    for k in range(HC):
        head_base = res_base + k * res_stride_i
        for h_start in range(0, hidden_size, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < hidden_size
            v = tl.load(
                residual_ptr + head_base + h_offsets * res_stride_h,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            sq += tl.sum(v * v)

    rms_inv = tl.rsqrt(sq / hc_hidden_size + rms_eps)

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    for i in range(HC):
        post_i = (
            tl.sigmoid(
                tl.load(gemm_out_ptr + go_base + HC + i) * rms_inv * scale_1
                + tl.load(hc_base_ptr + HC + i)
            )
            * hc_post_mult_value
        )
        tl.store(post_mix_ptr + pid_n * HC + i, post_i)

    cb = 2 * HC
    for i in range(HC):
        for j in range(HC):
            idx = i * HC + j
            v = tl.load(
                gemm_out_ptr + go_base + cb + idx
            ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + idx)
            if clamp_logits:
                v = tl.clamp(v, -30.0, 30.0)
            tl.store(comb_mix_ptr + comb_base + idx, v)

    for i in range(HC):
        row_max = tl.load(comb_mix_ptr + comb_base + i * HC + 0)
        for j in range(1, HC):
            row_max = tl.maximum(
                row_max, tl.load(comb_mix_ptr + comb_base + i * HC + j)
            )

        row_sum = 0.0
        for j in range(HC):
            e = tl.exp(tl.load(comb_mix_ptr + comb_base + i * HC + j) - row_max)
            tl.store(comb_mix_ptr + comb_base + i * HC + j, e)
            row_sum += e

        inv_row_sum = 1.0 / row_sum
        for j in range(HC):
            v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
            tl.store(
                comb_mix_ptr + comb_base + i * HC + j, v * inv_row_sum + hc_sinkhorn_eps
            )

    for j in range(HC):
        col_sum = 0.0
        for i in range(HC):
            col_sum += tl.load(comb_mix_ptr + comb_base + i * HC + j)
        inv_col_sum = 1.0 / (col_sum + hc_sinkhorn_eps)
        for i in range(HC):
            v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
            tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_col_sum)

    for _ in range(sinkhorn_repeat - 1):
        for i in range(HC):
            row_sum = 0.0
            for j in range(HC):
                row_sum += tl.load(comb_mix_ptr + comb_base + i * HC + j)
            inv_row_sum = 1.0 / (row_sum + hc_sinkhorn_eps)
            for j in range(HC):
                v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
                tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_row_sum)

        for j in range(HC):
            col_sum = 0.0
            for i in range(HC):
                col_sum += tl.load(comb_mix_ptr + comb_base + i * HC + j)
            inv_col_sum = 1.0 / (col_sum + hc_sinkhorn_eps)
            for i in range(HC):
                v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
                tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_col_sum)

    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for k in range(HC):
            pre_k = (
                tl.sigmoid(
                    tl.load(gemm_out_ptr + go_base + k) * rms_inv * scale_0
                    + tl.load(hc_base_ptr + k)
                )
                + hc_pre_eps
            )
            rk = tl.load(
                residual_ptr + res_base + k * res_stride_i + h_offsets * res_stride_h,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            acc += pre_k * rk

        tl.store(
            layer_input_ptr + pid_n * li_stride_n + h_offsets * li_stride_h,
            acc.to(tl.bfloat16),
            mask=h_mask,
        )



def _validate_inputs(residual, fn, hc_scale, hc_base):
    """Shape/dtype validation. Raises ValueError on any mismatch."""
    if not isinstance(residual, torch.Tensor) or residual.dim() < 3:
        raise ValueError(
            "residual must be a tensor of shape (..., hc_mult, hidden), got "
            "{!r}".format(getattr(residual, "shape", residual))
        )
    if residual.dtype != torch.bfloat16:
        raise ValueError(
            "residual must be bfloat16, got {}".format(residual.dtype)
        )
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    if not isinstance(fn, torch.Tensor) or fn.dim() != 2 or tuple(fn.shape) != (
        hc_mult3,
        hc_mult * hidden_size,
    ):
        raise ValueError(
            "fn must have shape ({}, {}), got {!r}".format(
                hc_mult3, hc_mult * hidden_size, tuple(getattr(fn, "shape", ()))
            )
        )
    if fn.dtype != torch.float32:
        raise ValueError("fn must be float32, got {}".format(fn.dtype))
    if tuple(hc_scale.shape) != (3,):
        raise ValueError(
            "hc_scale must have shape (3,), got {}".format(tuple(hc_scale.shape))
        )
    if hc_scale.dtype != torch.float32:
        raise ValueError(
            "hc_scale must be float32, got {}".format(hc_scale.dtype)
        )
    if tuple(hc_base.shape) != (hc_mult3,):
        raise ValueError(
            "hc_base must have shape ({},), got {}".format(
                hc_mult3, tuple(hc_base.shape)
            )
        )
    if hc_base.dtype != torch.float32:
        raise ValueError("hc_base must be float32, got {}".format(hc_base.dtype))
    return hc_mult, hidden_size, hc_mult3


def npu_mhc_pre(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
):
    """Standalone mHC pre block for Ascend NPU (ported Triton variants).

    Args:
        residual: (..., hc_mult, hidden), bfloat16.
        fn: (hc_mult3, hc_mult * hidden), float32 projection weights.
        hc_scale: (3,), float32 per-block scales.
        hc_base: (hc_mult3,), float32 per-block biases.
        rms_eps: RMS normalisation epsilon.
        hc_pre_eps: epsilon added to sigmoid(pre_logits).
        hc_sinkhorn_eps: Sinkhorn renormalisation epsilon.
        hc_post_mult_value: multiplier applied to sigmoid(post_logits).
        sinkhorn_repeat: Sinkhorn rounds (model uses 20); runtime argument,
            changing it does not trigger recompilation.
        n_splits: accepted for signature parity with the generic op, unused.

    Returns:
        (post, comb, hin) matching the generic ``mhc_pre`` contract, computed
        with the model-required ``clamp(-30, 30)`` before Sinkhorn.
    """
    if residual.device.type != "npu":
        raise NotImplementedError("npu_mhc_pre requires NPU tensors")
    hc_mult, hidden_size, hc_mult3 = _validate_inputs(
        residual, fn, hc_scale, hc_base
    )

    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
    num_tokens = residual_flat.shape[0]
    hc_hidden_size = hc_mult * hidden_size
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

    # projection: bf16 GEMM (ported generic path)
    x_flat = residual_flat.reshape(num_tokens, hc_hidden_size)
    fn_bf16 = _get_fn_bf16_cached(fn)
    gemm_out = torch.mm(x_flat, fn_bf16.t()).float()

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    common = (
        gemm_out,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        num_tokens,
        num_tokens_bucket,
        residual_flat.stride(0),
        residual_flat.stride(1),
        residual_flat.stride(2),
        layer_input.stride(0),
        layer_input.stride(1),
        hidden_size,
        hc_hidden_size,
    )
    kwargs = dict(
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        clamp_logits=True,  # model contract, always on for this operator
    )
    if hc_mult == 4:
        mhc_pre_fused_kernel_hc_mult_4[(num_tokens,)](
            *common, HC_MULT3=hc_mult3, **kwargs
        )
    else:
        mhc_pre_generic_kernel[(num_tokens,)](*common, HC=hc_mult, **kwargs)

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input
