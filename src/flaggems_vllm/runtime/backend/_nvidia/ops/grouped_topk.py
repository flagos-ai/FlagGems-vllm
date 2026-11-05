# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA score/grouped TopK fusions retaining current generic alternatives."""

import logging
import os

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.grouped_topk import group_idx_and_topk_triton
from flaggems_vllm.ops.grouped_topk import grouped_topk as _generic_grouped_topk
from flaggems_vllm.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)
_GROUPED_TOPK_SCORE_FUSION = (
    os.environ.get("FLAG_GEMS_GROUPED_TOPK_SCORE_FUSION", "0") == "1"
)
_GROUPED_TOPK_SINGLE_GROUP_FUSION = (
    os.environ.get("FLAG_GEMS_GROUPED_TOPK_SINGLE_GROUP_FUSION", "0") == "1"
)
# Fixed launches preserve separate FP32 pointwise rounds and the selected tie tree.


@libentry()
@triton.jit
def _single_group_score_topk_triton(
    scores_ptr,
    bias_ptr,
    topk_values_ptr,
    topk_indices_ptr,
    num_tokens,
    num_experts,
    topk,
    routed_scaling_factor,
    stride_scores_token,
    stride_out_token,
    TOPK: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    renormalize: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    neg_inf = -float("inf")
    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    valid_expert = expert_offsets < num_experts
    x = tl.load(
        scores_ptr + pid * stride_scores_token + expert_offsets,
        mask=valid_expert,
        other=neg_inf,
    )

    # Same four FP32 materialization rounds and tanh provider as the separate score stage.
    x = (0.5 * x).to(tl.float32)
    x = tl_extra_shim.tanh(x.to(tl.float32)).to(tl.float32)
    x = (0.5 * x).to(tl.float32)
    x = (x + 0.5).to(tl.float32)
    processed_scores = x
    x = tl.where(valid_expert, x, neg_inf)
    expert_bias = tl.load(
        bias_ptr + expert_offsets,
        mask=valid_expert,
        other=0.0,
    ).to(INPUT_DTYPE)
    x = x + expert_bias
    x_f32 = x.to(tl.float32)

    # Keep the original duplicate-max group top-two and its invalid-group contract.
    max1 = tl.max(x_f32, axis=0)
    is_max1 = (x_f32 == max1) & valid_expert
    count_max1 = tl.sum(is_max1.to(tl.int32), axis=0)
    x2 = tl.where(
        is_max1 & (count_max1 == 1),
        neg_inf,
        x_f32,
    )
    max2 = tl.max(x2, axis=0)
    group_scores_f32 = (max1 + max2).to(INPUT_DTYPE).to(tl.float32)
    is_finite = (group_scores_f32 == group_scores_f32) & (
        group_scores_f32 != float("inf")
    )
    group_scores_f32 = tl.where(is_finite, group_scores_f32, neg_inf)
    if_proceed = group_scores_f32 != neg_inf

    # n_group=topk_group=1 selects the sole physical group even when its
    # cleaned score is -inf. Only final if_proceed controls default outputs.
    scored = tl.where(valid_expert, processed_scores, neg_inf)
    selection_scores_native = scored + expert_bias
    selection_scores = tl.where(
        valid_expert,
        selection_scores_native.to(tl.float32),
        neg_inf,
    )
    topk_vals = tl.full([TOPK], 0.0, tl.float32)
    topk_idx = tl.full([TOPK], 0, tl.int32)
    pos_range = tl.arange(0, TOPK)

    for i in range(TOPK):
        max_val = tl.max(selection_scores, axis=0)
        is_max = selection_scores == max_val
        candidate_idx = tl.where(is_max, expert_offsets, num_experts + 1)
        selected_idx = tl.min(candidate_idx, axis=0)

        # A bounded register gather preserves the old unbiased selected-score
        # load bits. Do not retrieve a value using an extra floating reduction.
        safe_idx = tl.minimum(selected_idx, num_experts - 1)
        gather_idx = tl.full((1,), 0, tl.int32) + safe_idx
        selected_score = tl.gather(processed_scores, gather_idx, axis=0).reshape(())
        selected_score = tl.where(
            selected_idx < num_experts, selected_score, neg_inf
        ).to(tl.float32)

        topk_vals = tl.where(pos_range == i, selected_score, topk_vals)
        topk_idx = tl.where(pos_range == i, selected_idx.to(tl.int32), topk_idx)
        selection_scores = tl.where(
            expert_offsets == selected_idx, neg_inf, selection_scores
        )

    if renormalize == 1:
        topk_sum = tl.sum(topk_vals, axis=0) + 1e-20
        scale = routed_scaling_factor / topk_sum
    else:
        scale = routed_scaling_factor
    topk_vals = topk_vals * scale

    default_idx = pos_range.to(tl.int32)
    default_vals = tl.full([TOPK], 1.0 / topk, tl.float32)
    final_vals = tl.where(if_proceed, topk_vals, default_vals)
    final_idx = tl.where(if_proceed, topk_idx, default_idx)
    tl.store(
        topk_values_ptr + pid * stride_out_token + pos_range,
        final_vals,
        mask=pos_range < topk,
    )
    tl.store(
        topk_indices_ptr + pid * stride_out_token + pos_range,
        final_idx,
        mask=pos_range < topk,
    )


@libentry()
@triton.jit
def _score_topk_with_k2_triton(
    scores_ptr,
    processed_scores_ptr,
    bias_ptr,
    group_scores_ptr,
    num_experts_per_group,
    n_group,
    stride_scores_token,
    stride_processed_token,
    stride_group_scores_token,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)

    token_id = pid // n_group
    group_id = pid % n_group

    lane = tl.arange(0, BLOCK_SIZE)
    mask = lane < num_experts_per_group

    scores_offset = token_id * stride_scores_token + group_id * num_experts_per_group
    bias_offset = group_id * num_experts_per_group

    x = tl.load(
        scores_ptr + scores_offset + lane,
        mask=mask,
        other=-float("inf"),
    )

    # The original four FP32 pointwise stages, in their original order.
    # The launch disables FP fusion: the second multiply and final add must
    # round separately, as they do across the original kernel boundaries.
    x = (0.5 * x).to(tl.float32)
    x = tl_extra_shim.tanh(x.to(tl.float32)).to(tl.float32)
    x = (0.5 * x).to(tl.float32)
    x = (x + 0.5).to(tl.float32)
    processed_offset = (
        token_id * stride_processed_token + group_id * num_experts_per_group
    )
    tl.store(processed_scores_ptr + processed_offset + lane, x, mask=mask)
    # Invalid lanes must stay -inf for the unchanged group top-two reduction;
    # applying sigmoid to the original masked -inf would otherwise make 0.
    x = tl.where(mask, x, -float("inf"))

    b = tl.load(
        bias_ptr + bias_offset + lane,
        mask=mask,
        other=0.0,
    ).to(INPUT_DTYPE)

    x = x + b

    x_f32 = x.to(tl.float32)

    max1 = tl.max(x_f32, axis=0)
    is_max1 = (x_f32 == max1) & mask
    count_max1 = tl.sum(is_max1.to(tl.int32), axis=0)

    x2 = tl.where(
        is_max1 & (count_max1 == 1),
        -float("inf"),
        x_f32,
    )
    max2 = tl.max(x2, axis=0)

    group_scores_offset = token_id * stride_group_scores_token + group_id
    tl.store(
        group_scores_ptr + group_scores_offset,
        (max1 + max2).to(INPUT_DTYPE),
    )


def grouped_topk(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
):
    logger.debug("GEMS_NVIDIA GROUPED_TOPK")
    supported = (
        _GROUPED_TOPK_SCORE_FUSION
        and isinstance(scores, torch.Tensor)
        and isinstance(bias, torch.Tensor)
        and scores.ndim == 2
        and type(n_group) is int
        and type(topk_group) is int
        and type(topk) is int
        and 0 < n_group <= 32
        and 0 < topk_group <= n_group
        and 0 < topk <= 32
        and scores.shape[1] > 0
        and scores.shape[1] % n_group == 0
        and topk <= scores.shape[1]
        and scores.shape[0] > 0
        and scoring_func == 1
        and scores.is_cuda
        and torch.version.hip is None
        and scores.dtype == torch.float32
        and scores.stride(1) == 1
        and scores.stride(0) >= scores.shape[1]
        and bias.ndim == 1
        and bias.shape[0] == scores.shape[1]
        and bias.device == scores.device
        and bias.stride(0) == 1
        and bias.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    # Keep every current-main small-expert route before the added fusions.
    if not supported or (scores.shape[1] <= 256 and topk <= 8):
        return _generic_grouped_topk(
            scores,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
        )
    num_tokens, num_experts = scores.shape
    num_experts_per_group = num_experts // n_group
    topk_values = torch.empty(
        (num_tokens, topk), device=scores.device, dtype=torch.float32
    )
    topk_indices = torch.empty(
        (num_tokens, topk), device=scores.device, dtype=torch.int32
    )
    with torch.cuda.device(scores.device):
        if (
            _GROUPED_TOPK_SINGLE_GROUP_FUSION
            and n_group == topk_group == 1
            and num_experts <= 1024
            and (topk & (topk - 1)) == 0
        ):
            _single_group_score_topk_triton[(num_tokens,)](
                scores,
                bias,
                topk_values,
                topk_indices,
                num_tokens,
                num_experts,
                topk,
                routed_scaling_factor,
                scores.stride(0),
                topk_values.stride(0),
                TOPK=topk,
                BLOCK_EXPERT=triton.next_power_of_2(num_experts),
                INPUT_DTYPE=tl.float32,
                renormalize=int(renormalize),
                enable_fp_fusion=False,
            )
            return topk_values, topk_indices
        processed = torch.empty(
            (num_tokens, num_experts), device=scores.device, dtype=torch.float32
        )
        group_scores = torch.empty(
            (num_tokens, n_group), device=scores.device, dtype=torch.float32
        )
        _score_topk_with_k2_triton[(num_tokens * n_group,)](
            scores,
            processed,
            bias,
            group_scores,
            num_experts_per_group,
            n_group,
            scores.stride(0),
            processed.stride(0),
            group_scores.stride(0),
            BLOCK_SIZE=triton.next_power_of_2(num_experts_per_group),
            INPUT_DTYPE=tl.float32,
            enable_fp_fusion=False,
        )
        group_idx_and_topk_triton[(num_tokens,)](
            processed,
            group_scores,
            topk_values,
            topk_indices,
            bias,
            num_tokens,
            n_group,
            topk_group,
            topk,
            num_experts,
            num_experts_per_group,
            routed_scaling_factor,
            processed.stride(0),
            group_scores.stride(0),
            topk_values.stride(0),
            N_GROUP=n_group,
            TOPK_GROUP=topk_group,
            TOPK=topk,
            BLOCK_GROUP=triton.next_power_of_2(n_group),
            BLOCK_EXPERT=triton.next_power_of_2(num_experts),
            INPUT_DTYPE=tl.float32,
            renormalize=int(renormalize),
        )
    return topk_values, topk_indices


__all__ = ["grouped_topk"]
