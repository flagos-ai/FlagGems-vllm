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

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flaggems_vllm.utils import tl_extra_shim

logger = logging.getLogger(__name__)


@triton.jit
def topk_with_k2_triton(
    scores_ptr,
    bias_ptr,
    group_scores_ptr,
    num_experts_per_group,
    n_group,
    stride_scores_token,
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


@triton.jit
def group_idx_and_topk_triton(
    scores_ptr,
    group_scores_ptr,
    topk_values_ptr,
    topk_indices_ptr,
    bias_ptr,
    num_tokens,
    n_group,
    topk_group,
    topk,
    num_experts,
    num_experts_per_group,
    routed_scaling_factor,
    stride_scores_token,
    stride_group_scores_token,
    stride_out_token,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    renormalize: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    neg_inf = -float("inf")

    group_offsets = tl.arange(0, BLOCK_GROUP)
    valid_group = group_offsets < n_group

    group_scores = tl.load(
        group_scores_ptr + pid * stride_group_scores_token + group_offsets,
        mask=valid_group,
        other=neg_inf,
    )

    group_scores_f32 = group_scores.to(tl.float32)
    is_finite = (group_scores_f32 == group_scores_f32) & (
        group_scores_f32 != float("inf")
    )
    group_scores_f32 = tl.where(is_finite & valid_group, group_scores_f32, neg_inf)

    max_group_score = tl.max(group_scores_f32, axis=0)
    if_proceed = max_group_score != neg_inf

    value = group_scores_f32
    target_num_min = BLOCK_GROUP - n_group + topk_group
    count_equal_to_top_value = BLOCK_GROUP - n_group
    pre_count_equal_to_top_value = 0
    topk_group_value = neg_inf

    for _ in range(TOPK_GROUP):
        need = count_equal_to_top_value < target_num_min
        max_val = tl.max(value, axis=0)

        is_max = need & (value == max_val)
        value = tl.where(is_max, neg_inf, value)

        newly = tl.sum(is_max.to(tl.int32), axis=0)

        pre_count_equal_to_top_value = tl.where(
            need, count_equal_to_top_value, pre_count_equal_to_top_value
        )
        count_equal_to_top_value = tl.where(
            need, count_equal_to_top_value + newly, count_equal_to_top_value
        )
        topk_group_value = tl.where(need, max_val, topk_group_value)

    num_equalto_topkth_group = target_num_min - pre_count_equal_to_top_value

    group_gt = group_scores_f32 > topk_group_value
    group_eq = group_scores_f32 == topk_group_value

    eq_i = group_eq.to(tl.int32)
    prefix_eq = tl.cumsum(eq_i, axis=0) - eq_i

    group_selected = (
        group_gt | (group_eq & (prefix_eq < num_equalto_topkth_group))
    ) & valid_group

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    valid_expert = expert_offsets < num_experts
    expert_group = expert_offsets // num_experts_per_group

    expert_in_group = expert_group[:, None] == group_offsets[None, :]
    expert_selected = (
        tl.sum((expert_in_group & group_selected[None, :]).to(tl.int32), axis=1) > 0
    ) & valid_expert

    scored = tl.load(
        scores_ptr + pid * stride_scores_token + expert_offsets,
        mask=expert_selected,
        other=neg_inf,
    )

    expert_bias = tl.load(
        bias_ptr + expert_offsets,
        mask=valid_expert,
        other=0.0,
    ).to(INPUT_DTYPE)

    selection_scores_native = scored + expert_bias

    selection_scores = tl.where(
        expert_selected,
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

        selected_score = tl.load(
            scores_ptr + pid * stride_scores_token + selected_idx,
            mask=selected_idx < num_experts,
            other=neg_inf,
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


@triton.jit
def _sigmoid(x):
    log2e: tl.constexpr = 1.4426950408889634
    return 1 / (1 + tl_extra_shim.exp2(-x * log2e))


# Adapted from vLLM:
#   ./vllm/csrc/moe/grouped_topk_kernels.cu
#   ./vllm/csrc/moe/moeTopKFuncs.cuh
@triton.jit
def triton_grouped_topk_fused_small_expert_count_kernel(
    scores_ptr,
    topk_values_ptr,
    topk_indices_ptr,
    routing_bias_ptr,
    num_tokens,
    num_groups,
    topk_group: tl.constexpr,
    topk: tl.constexpr,
    num_experts,
    num_experts_per_group,
    renormalize,
    routed_scaling_factor,
    scores_stride0,
    SCORE_DTYPE: tl.constexpr,
    BIAS_DTYPE: tl.constexpr,
    SCORING_FUNC: tl.constexpr,
    FULL_SHAPE: tl.constexpr,
    NUM_GROUPS_PAD: tl.constexpr,
):
    MAX_TOPK_GROUP: tl.constexpr = 4
    MAX_TOPK: tl.constexpr = 8
    WARP_SIZE: tl.constexpr = 32
    NUM_WARPS: tl.constexpr = NUM_GROUPS_PAD
    neg_inf: tl.constexpr = float("-inf")

    token_id = tl.program_id(0)
    scores_ptr += token_id * scores_stride0
    topk_values_ptr += token_id * topk
    topk_indices_ptr += token_id * topk
    warps = tl.arange(0, NUM_WARPS)
    lane = tl.arange(0, WARP_SIZE)

    # step1: load score and bias, get score_sigmoid and score_bias
    if FULL_SHAPE:
        # num_groups = NUM_GROUPS_PAD, num_experts_per_group = WARP_SIZE
        offs = warps[:, None] * WARP_SIZE + lane[None, :]
        score_ub = tle.dsa.alloc(
            [NUM_WARPS, WARP_SIZE], dtype=SCORE_DTYPE, mem_addr_space=tle.dsa.ascend.UB
        )
        bias_ub = tle.dsa.alloc(
            [NUM_WARPS, WARP_SIZE], dtype=BIAS_DTYPE, mem_addr_space=tle.dsa.ascend.UB
        )
        tle.dsa.copy(scores_ptr + offs, score_ub, [NUM_WARPS, WARP_SIZE])
        tle.dsa.copy(routing_bias_ptr + offs, bias_ub, [NUM_WARPS, WARP_SIZE])
        score = tle.dsa.to_tensor(score_ub).to(tl.float32)
        if SCORING_FUNC == 1:
            score_sigmoid = _sigmoid(score)
        else:
            score_sigmoid = score
        bias_val = tle.dsa.to_tensor(bias_ub)
        score_bias = score_sigmoid + bias_val
    else:
        offs = warps[:, None] * num_experts_per_group + lane[None, :]
        score = tl.load(
            scores_ptr + offs,
            mask=(warps[:, None] < num_groups)
            & (lane[None, :] < num_experts_per_group),
            other=neg_inf,
        ).to(tl.float32)
        if SCORING_FUNC == 1:
            score_sigmoid = _sigmoid(score)
        else:
            score_sigmoid = score
        bias_val = tl.load(
            routing_bias_ptr + offs,
            mask=(warps[:, None] < num_groups)
            & (lane[None, :] < num_experts_per_group),
            other=neg_inf,
        ).to(tl.float32)
        score_bias = score_sigmoid + bias_val

    # step2: get sum(top2) as group_score
    group_max_val0, group_max_index0 = tl.max(
        score_bias, axis=-1, return_indices=True, return_indices_tie_break_left=True
    )
    tmp_score_bias = tl.where(
        group_max_index0[:, None] == lane[None, :], neg_inf, score_bias
    )
    group_max_val1 = tl.max(tmp_score_bias, axis=-1, return_indices_tie_break_left=True)
    group_score = group_max_val0 + group_max_val1  # [NUM_WARPS]

    # step3: get topk_group
    invalid_score = tl.full([1], neg_inf, dtype=tl.float32)
    group_idx = tl.zeros([MAX_TOPK_GROUP], dtype=tl.int32)
    group_score_bias = tl.full([MAX_TOPK_GROUP, WARP_SIZE], neg_inf, dtype=tl.float32)
    for i in tl.static_range(topk_group):
        _1, idx = tl.max(
            group_score,
            axis=-1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        group_score = tle.dsa.insert_slice(
            group_score, invalid_score, offsets=(idx,), sizes=(1,), strides=(1,)
        )
        group_idx = tle.dsa.insert_slice(
            group_idx,
            tl.full([1], idx, dtype=tl.int32),
            offsets=(i,),
            sizes=(1,),
            strides=(1,),
        )
        group_slice = tle.dsa.extract_slice(
            score_bias, offsets=(idx, 0), sizes=(1, WARP_SIZE), strides=(1, 1)
        )
        group_score_bias = tle.dsa.insert_slice(
            group_score_bias,
            group_slice,
            offsets=(i, 0),
            sizes=(1, WARP_SIZE),
            strides=(1, 1),
        )
    group_score_bias = group_score_bias.reshape(MAX_TOPK_GROUP * WARP_SIZE)

    # step4: get topk
    top_experts_idx = tl.zeros([MAX_TOPK], dtype=tl.int32)
    top_experts_pos = tl.zeros([MAX_TOPK], dtype=tl.int32)
    topk_offs = tl.arange(0, MAX_TOPK)
    topk_group_offs = tl.arange(0, MAX_TOPK_GROUP)
    for j in tl.static_range(topk):
        _2, off = tl.max(
            group_score_bias,
            axis=-1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        group_score_bias = tle.dsa.insert_slice(
            group_score_bias,
            tl.full([1], neg_inf, dtype=tl.float32),
            offsets=(off,),
            sizes=(1,),
            strides=(1,),
        )
        local_group_off = off // WARP_SIZE
        inner_off = off % WARP_SIZE
        # TODO: User tle.dsa.extract_element instead of tl.min(tl.where(...)) after the related issue is resolved
        actual_group_off = tl.min(
            tl.where(topk_group_offs == local_group_off, group_idx, NUM_WARPS)
        )
        expert_id = actual_group_off * num_experts_per_group + inner_off
        pos = actual_group_off * WARP_SIZE + inner_off
        top_experts_idx = tle.dsa.insert_slice(
            top_experts_idx,
            tl.full([1], expert_id, dtype=tl.int32),
            offsets=(j,),
            sizes=(1,),
            strides=(1,),
        )
        top_experts_pos = tle.dsa.insert_slice(
            top_experts_pos,
            tl.full([1], pos, dtype=tl.int32),
            offsets=(j,),
            sizes=(1,),
            strides=(1,),
        )

    # step5: output
    topk_unbiased = tl.gather(
        tl.reshape(score_sigmoid, (NUM_WARPS * WARP_SIZE)), top_experts_pos, 0
    )
    topk_unbiased = tl.where(topk_offs < topk, topk_unbiased, 0.0)
    topk_sum = 1e-20
    if renormalize:
        topk_sum += tl.sum(topk_unbiased)
    scale = routed_scaling_factor.to(tl.float32)
    if renormalize:
        scale /= topk_sum
    tl.store(topk_values_ptr + topk_offs, topk_unbiased * scale, mask=topk_offs < topk)
    tl.store(topk_indices_ptr + topk_offs, top_experts_idx, mask=topk_offs < topk)


def grouped_topk(
    scores: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    logger.debug("GEMS_ASCEND GROUPED TOPK")
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D Tensor")
    num_tokens, num_experts = scores.shape
    if num_experts % n_group != 0:
        raise ValueError("num_experts must be divisible by n_group")
    if n_group > 32:
        raise ValueError("n_group should be smaller than or equal to 32")
    if topk > 32:
        raise ValueError("topk should be smaller than or equal to 32 for now")
    if scoring_func not in (0, 1):
        raise ValueError("scoring_func must be 0 (none) or 1 (sigmoid)")

    if bias.ndim != 1:
        bias = bias.flatten()
    if len(bias) != num_experts:
        raise ValueError(
            f"bias length ({len(bias)}) must match num_experts ({num_experts})"
        )

    num_experts_per_group = num_experts // n_group

    if scores.dtype == torch.float32:
        INPUT_DTYPE = tl.float32
    elif scores.dtype == torch.float16:
        INPUT_DTYPE = tl.float16
    elif scores.dtype == torch.bfloat16:
        INPUT_DTYPE = tl.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {scores.dtype}")

    if (
        (n_group > 1)
        & (n_group <= 32)
        & (num_experts <= 256)
        & (num_experts_per_group <= 32)
        & (num_experts_per_group * topk_group <= 128)
        & (topk <= 8)
        & (topk_group <= 4)
    ):
        # DeepSeek-v3.2
        topk_values = torch.empty(
            (num_tokens, topk),
            device=scores.device,
            dtype=torch.float32,
        )
        topk_indices = torch.empty(
            (num_tokens, topk),
            device=scores.device,
            dtype=torch.int32,
        )
        if bias.dtype == torch.float32:
            BIAS_DTYPE = tl.float32
        elif bias.dtype == torch.float16:
            BIAS_DTYPE = tl.float16
        elif bias.dtype == torch.bfloat16:
            BIAS_DTYPE = tl.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {bias.dtype}")
        n_group_pad = triton.next_power_of_2(n_group)

        triton_grouped_topk_fused_small_expert_count_kernel[(num_tokens,)](
            scores,
            topk_values,
            topk_indices,
            bias,
            num_tokens,
            n_group,
            topk_group,
            topk,
            num_experts,
            num_experts_per_group,
            renormalize,
            routed_scaling_factor,
            scores.stride(0),
            SCORE_DTYPE=INPUT_DTYPE,
            BIAS_DTYPE=BIAS_DTYPE,
            SCORING_FUNC=scoring_func,
            FULL_SHAPE=(num_experts_per_group == 32) & (n_group_pad == n_group),
            NUM_GROUPS_PAD=n_group_pad,
            num_warps=1,
        )
        return topk_values, topk_indices

    if scoring_func == 1:
        scores_processed = torch.sigmoid(scores.float()).to(scores.dtype)
    else:
        scores_processed = scores

    group_scores = torch.empty(
        (num_tokens, n_group),
        device=scores.device,
        dtype=scores.dtype,
    )

    topk_values = torch.empty(
        (num_tokens, topk),
        device=scores.device,
        dtype=torch.float32,
    )

    topk_indices = torch.empty(
        (num_tokens, topk),
        device=scores.device,
        dtype=torch.int32,
    )

    BLOCK1 = triton.next_power_of_2(num_experts_per_group)
    grid1 = (num_tokens * n_group,)

    topk_with_k2_triton[grid1](
        scores_processed,
        bias,
        group_scores,
        num_experts_per_group,
        n_group,
        scores_processed.stride(0),
        group_scores.stride(0),
        BLOCK_SIZE=BLOCK1,
        INPUT_DTYPE=INPUT_DTYPE,
    )

    BLOCK_GROUP = triton.next_power_of_2(n_group)
    BLOCK_EXPERT = triton.next_power_of_2(num_experts)
    grid2 = (num_tokens,)

    group_idx_and_topk_triton[grid2](
        scores_processed,
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
        scores_processed.stride(0),
        group_scores.stride(0),
        topk_values.stride(0),
        N_GROUP=n_group,
        TOPK_GROUP=topk_group,
        TOPK=topk,
        BLOCK_GROUP=BLOCK_GROUP,
        BLOCK_EXPERT=BLOCK_EXPERT,
        INPUT_DTYPE=INPUT_DTYPE,
        renormalize=int(renormalize),
    )

    return topk_values, topk_indices
