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
from typing import Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libtuner, tl_extra_shim
from flaggems_vllm.utils.triton_version_utils import has_triton_tle


def backend_tle_enabled():
    from flaggems_vllm.runtime import backend, device

    vendor_info = backend.get_vendor_info(device.vendor_name)
    return vendor_info.tle_enabled


if backend_tle_enabled() & has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = True
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False


logger = logging.getLogger(__name__)


@triton.jit
def topk_with_k2_triton(
    scores_ptr,
    bias_ptr,
    group_scores_ptr,
    num_experts_per_group,
    n_group,
    stride_scores_token,
    stride_scores_expert,
    stride_group_scores_token,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    padding_ptr,
    HAS_PADDING: tl.constexpr,
    MASKED_SIGMOID: tl.constexpr,
):
    pid = tl.program_id(0)

    token_id = pid // n_group
    group_id = pid % n_group

    if HAS_PADDING:
        if tl.load(padding_ptr + token_id):
            return  # Masked output CTA never reads this workspace row.

    lane = tl.arange(0, BLOCK_SIZE)
    mask = lane < num_experts_per_group

    scores_offset = (
        token_id * stride_scores_token
        + group_id * num_experts_per_group * stride_scores_expert
    )
    bias_offset = group_id * num_experts_per_group

    x = tl.load(
        scores_ptr + scores_offset + lane * stride_scores_expert,
        mask=mask,
        other=-float("inf"),
    )

    if MASKED_SIGMOID:
        x = _legacy_sigmoid(x, INPUT_DTYPE)
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
    stride_scores_expert,
    stride_group_scores_token,
    stride_out_token,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    renormalize: tl.constexpr,
    padding_ptr,
    HAS_PADDING: tl.constexpr,
    MASKED_SIGMOID: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    if HAS_PADDING:
        if tl.load(padding_ptr + pid):
            _write_padding(topk_values_ptr, topk_indices_ptr, pid, TOPK, TOPK_BLOCK)
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
        scores_ptr + pid * stride_scores_token + expert_offsets * stride_scores_expert,
        mask=expert_selected,
        other=neg_inf,
    )

    if MASKED_SIGMOID:
        scored = _legacy_sigmoid(scored, INPUT_DTYPE)

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

    topk_vals = tl.full([TOPK_BLOCK], 0.0, tl.float32)
    topk_idx = tl.full([TOPK_BLOCK], 0, tl.int32)
    pos_range = tl.arange(0, TOPK_BLOCK)

    for i in range(TOPK):
        max_val = tl.max(selection_scores, axis=0)
        is_max = selection_scores == max_val

        candidate_idx = tl.where(is_max, expert_offsets, num_experts + 1)
        selected_idx = tl.min(candidate_idx, axis=0)

        selected_score = tl.load(
            scores_ptr
            + pid * stride_scores_token
            + selected_idx * stride_scores_expert,
            mask=selected_idx < num_experts,
            other=neg_inf,
        )
        if MASKED_SIGMOID:
            selected_score = _legacy_sigmoid(selected_score, INPUT_DTYPE)
        selected_score = selected_score.to(tl.float32)

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
    default_vals = tl.full([TOPK_BLOCK], 1.0 / topk, tl.float32)

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


@triton.jit
def _pack_val_idx_fp32(val, idx):
    MAX_IDX: tl.constexpr = 0xFFFF
    bits = val.to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
    high = key.to(tl.uint64) << 32
    low = (0xFFFF & (MAX_IDX - idx)).to(tl.uint64)
    return high | low


@triton.jit
def _unpack_val_idx_fp32(pair):
    MAX_IDX: tl.constexpr = 0xFFFF
    key = (pair >> 32).to(tl.uint32)
    idx = (MAX_IDX - (pair & 0xFFFF)).to(tl.uint32)
    bits = tl.where((key & 0x80000000) != 0, key ^ 0x80000000, ~key)
    val = bits.to(tl.float32, bitcast=True)
    return val, idx


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
    g_score_sigmoid_ptr,
    g_score_bias_ptr,
    SCORING_FUNC: tl.constexpr,
    HAS_TLE: tl.constexpr,
    NUM_GROUPS_PAD: tl.constexpr,
    GROUP_SIZE_PAD: tl.constexpr,
    FULL_SHAPE: tl.constexpr,
    USE_GROUPS: tl.constexpr,
    padding_ptr,
    HAS_PADDING: tl.constexpr,
    USE_UINT32_REDUX: tl.constexpr = False,
):
    WARP_SIZE: tl.constexpr = 32
    neg_inf: tl.constexpr = float("-inf")
    MAX_IDX: tl.constexpr = 65535

    token_id = tl.program_id(0)
    if HAS_PADDING:
        if tl.load(padding_ptr + token_id):
            _write_padding(topk_values_ptr, topk_indices_ptr, token_id, topk, WARP_SIZE)
            return
    scores_ptr += token_id * scores_stride0
    topk_values_ptr += token_id * topk
    topk_indices_ptr += token_id * topk
    groups = tl.arange(0, NUM_GROUPS_PAD)
    group_experts = tl.arange(0, GROUP_SIZE_PAD)
    lane = tl.arange(0, WARP_SIZE)

    if HAS_TLE:
        s_score_sigmoid = tle.gpu.alloc(
            [NUM_GROUPS_PAD, GROUP_SIZE_PAD],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        s_score_bias = tle.gpu.alloc(
            [NUM_GROUPS_PAD, GROUP_SIZE_PAD],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        s_score_sigmoid_ptr = tle.gpu.local_ptr(s_score_sigmoid, (0, 0))
        s_score_bias_ptr = tle.gpu.local_ptr(s_score_bias, (0, 0))
    else:
        # Masked scores may have a padded row stride, but this scratch is dense.
        scratch_stride = num_experts
        s_score_sigmoid_ptr = g_score_sigmoid_ptr + token_id * scratch_stride
        s_score_bias_ptr = g_score_bias_ptr + token_id * scratch_stride

    # step1: load score/bias, get score_sigmoid/score_bias
    offs = groups[:, None] * num_experts_per_group + group_experts[None, :]
    if FULL_SHAPE:
        score = tl.load(scores_ptr + offs).to(tl.float32)
    else:
        score = tl.load(
            scores_ptr + offs,
            mask=(groups[:, None] < num_groups)
            & (group_experts[None, :] < num_experts_per_group),
            other=neg_inf,
        ).to(tl.float32)
    if SCORING_FUNC == 1:
        score_sigmoid = _sigmoid(score)
    else:
        score_sigmoid = score
    if FULL_SHAPE:
        tl.store(s_score_sigmoid_ptr + offs, score_sigmoid)
        bias_val = tl.load(routing_bias_ptr + offs).to(tl.float32)
    else:
        tl.store(
            s_score_sigmoid_ptr + offs,
            score_sigmoid,
            mask=(groups[:, None] < num_groups)
            & (group_experts[None, :] < num_experts_per_group),
        )
        bias_val = tl.load(
            routing_bias_ptr + offs,
            mask=(groups[:, None] < num_groups)
            & (group_experts[None, :] < num_experts_per_group),
            other=neg_inf,
        ).to(tl.float32)
    score_bias = score_sigmoid + bias_val
    if FULL_SHAPE:
        tl.store(s_score_bias_ptr + offs, score_bias)
    else:
        tl.store(
            s_score_bias_ptr + offs,
            score_bias,
            mask=(groups[:, None] < num_groups)
            & (group_experts[None, :] < num_experts_per_group),
        )

    if USE_GROUPS:
        # step2: get top2 as group_score
        min_val0 = tl.full((NUM_GROUPS_PAD, GROUP_SIZE_PAD), neg_inf, dtype=tl.float32)
        comp_val_idx0 = _pack_val_idx_fp32(score_bias, offs)
        if USE_UINT32_REDUX:
            packed_max00 = _packed_max_uint32(comp_val_idx0, 1)
        else:
            packed_max00 = tl.max(comp_val_idx0, axis=-1)
        val_max0, _0 = _unpack_val_idx_fp32(packed_max00)
        comp_val_idx0 = tl.where(
            comp_val_idx0 == packed_max00[:, None],
            _pack_val_idx_fp32(min_val0, offs),
            comp_val_idx0,
        )
        if USE_UINT32_REDUX:
            packed_max01 = _packed_max_uint32(comp_val_idx0, 1)
        else:
            packed_max01 = tl.max(comp_val_idx0, axis=-1)
        val_max1, _0 = _unpack_val_idx_fp32(packed_max01)
        group_score = val_max0 + val_max1

        # step3: get topk_group, topk_group <= MAX_NUM_TOP_GROUPS, where MAX_NUM_TOP_GROUPS = 4
        min_val1 = tl.full((NUM_GROUPS_PAD,), neg_inf, dtype=tl.float32)
        comp_val_idx1 = _pack_val_idx_fp32(group_score, groups)
        packed_max10 = tl.max(comp_val_idx1)
        _2, group_idx0 = _unpack_val_idx_fp32(packed_max10)
        comp_val_idx1 = tl.where(
            comp_val_idx1 == packed_max10,
            _pack_val_idx_fp32(min_val1, groups),
            comp_val_idx1,
        )
        packed_max11 = tl.max(comp_val_idx1)
        _2, group_idx1 = _unpack_val_idx_fp32(packed_max11)
        comp_val_idx1 = tl.where(
            comp_val_idx1 == packed_max11,
            _pack_val_idx_fp32(min_val1, groups),
            comp_val_idx1,
        )
        packed_max12 = tl.max(comp_val_idx1)
        _2, group_idx2 = _unpack_val_idx_fp32(packed_max12)
        comp_val_idx1 = tl.where(
            comp_val_idx1 == packed_max12,
            _pack_val_idx_fp32(min_val1, groups),
            comp_val_idx1,
        )
        packed_max13 = tl.max(comp_val_idx1)
        _2, group_idx3 = _unpack_val_idx_fp32(packed_max13)

        # step4: get topk, topk <= MAX_NUM_TOP_EXPERTS, where MAX_NUM_TOP_EXPERTS = 8
        if FULL_SHAPE and WARP_SIZE == GROUP_SIZE_PAD:
            expert_idx_group0 = group_idx0 * GROUP_SIZE_PAD + lane
            expert_idx_group1 = group_idx1 * GROUP_SIZE_PAD + lane
            expert_idx_group2 = group_idx2 * GROUP_SIZE_PAD + lane
            expert_idx_group3 = group_idx3 * GROUP_SIZE_PAD + lane
            expert_score_group0 = tl.load(
                s_score_bias_ptr + expert_idx_group0,
                mask=0 < topk_group,
                other=neg_inf,
            )
            expert_score_group1 = tl.load(
                s_score_bias_ptr + expert_idx_group1,
                mask=1 < topk_group,
                other=neg_inf,
            )
            expert_score_group2 = tl.load(
                s_score_bias_ptr + expert_idx_group2,
                mask=2 < topk_group,
                other=neg_inf,
            )
            expert_score_group3 = tl.load(
                s_score_bias_ptr + expert_idx_group3,
                mask=3 < topk_group,
                other=neg_inf,
            )
        else:
            expert_idx_group0 = group_idx0 * num_experts_per_group + lane
            expert_idx_group1 = group_idx1 * num_experts_per_group + lane
            expert_idx_group2 = group_idx2 * num_experts_per_group + lane
            expert_idx_group3 = group_idx3 * num_experts_per_group + lane
            lane_mask = lane < num_experts_per_group
            expert_score_group0 = tl.load(
                s_score_bias_ptr + expert_idx_group0,
                mask=(0 < topk_group) & lane_mask,
                other=neg_inf,
            )
            expert_score_group1 = tl.load(
                s_score_bias_ptr + expert_idx_group1,
                mask=(1 < topk_group) & lane_mask,
                other=neg_inf,
            )
            expert_score_group2 = tl.load(
                s_score_bias_ptr + expert_idx_group2,
                mask=(2 < topk_group) & lane_mask,
                other=neg_inf,
            )
            expert_score_group3 = tl.load(
                s_score_bias_ptr + expert_idx_group3,
                mask=(3 < topk_group) & lane_mask,
                other=neg_inf,
            )
        comp_val_idx20 = _pack_val_idx_fp32(expert_score_group0, expert_idx_group0)
        comp_val_idx21 = _pack_val_idx_fp32(expert_score_group1, expert_idx_group1)
        comp_val_idx22 = _pack_val_idx_fp32(expert_score_group2, expert_idx_group2)
        comp_val_idx23 = _pack_val_idx_fp32(expert_score_group3, expert_idx_group3)
        # TOPK_SWAP(0, 2); TOPK_SWAP(1, 3); TOPK_SWAP(0, 1); TOPK_SWAP(2, 3); TOPK_SWAP(1, 2);
        comp_val_idx20, comp_val_idx22 = max(comp_val_idx20, comp_val_idx22), min(
            comp_val_idx20, comp_val_idx22
        )
        comp_val_idx21, comp_val_idx23 = max(comp_val_idx21, comp_val_idx23), min(
            comp_val_idx21, comp_val_idx23
        )
        comp_val_idx20, comp_val_idx21 = max(comp_val_idx20, comp_val_idx21), min(
            comp_val_idx20, comp_val_idx21
        )
        comp_val_idx22, comp_val_idx23 = max(comp_val_idx22, comp_val_idx23), min(
            comp_val_idx22, comp_val_idx23
        )
        comp_val_idx21, comp_val_idx22 = max(comp_val_idx21, comp_val_idx22), min(
            comp_val_idx21, comp_val_idx22
        )

        min_val2 = tl.full((WARP_SIZE,), neg_inf, dtype=tl.float32)
        top_experts = tl.full((WARP_SIZE,), MAX_IDX, dtype=tl.uint32)
        packed_max20 = tl.full((), 0, dtype=tl.uint64)
        for kk in tl.static_range(0, topk):
            update = (kk > 0) & (comp_val_idx20 == packed_max20)
            comp_val_idx20 = tl.where(
                update,
                comp_val_idx21,
                comp_val_idx20,
            )
            comp_val_idx21 = tl.where(
                update,
                comp_val_idx22,
                comp_val_idx21,
            )
            comp_val_idx22 = tl.where(
                update,
                comp_val_idx23,
                comp_val_idx22,
            )
            comp_val_idx23 = tl.where(
                update,
                _pack_val_idx_fp32(min_val2, expert_idx_group3),
                comp_val_idx23,
            )
            if USE_UINT32_REDUX:
                packed_max20 = _packed_max_uint32(comp_val_idx20, 0)
            else:
                packed_max20 = tl.max(comp_val_idx20)
            _3, out_idx = _unpack_val_idx_fp32(packed_max20)
            top_experts = tl.where(lane == kk, out_idx, top_experts)
    else:
        # single group
        expert_idx_group0 = lane
        expert_idx_group1 = WARP_SIZE + lane
        expert_idx_group2 = WARP_SIZE * 2 + lane
        expert_idx_group3 = WARP_SIZE * 3 + lane
        expert_idx_group4 = WARP_SIZE * 4 + lane
        expert_idx_group5 = WARP_SIZE * 5 + lane
        expert_idx_group6 = WARP_SIZE * 6 + lane
        expert_idx_group7 = WARP_SIZE * 7 + lane
        expert_score_group0 = tl.load(
            s_score_bias_ptr + expert_idx_group0,
            mask=expert_idx_group0 < num_experts,
            other=neg_inf,
        )
        expert_score_group1 = tl.load(
            s_score_bias_ptr + expert_idx_group1,
            mask=expert_idx_group1 < num_experts,
            other=neg_inf,
        )
        expert_score_group2 = tl.load(
            s_score_bias_ptr + expert_idx_group2,
            mask=expert_idx_group2 < num_experts,
            other=neg_inf,
        )
        expert_score_group3 = tl.load(
            s_score_bias_ptr + expert_idx_group3,
            mask=expert_idx_group3 < num_experts,
            other=neg_inf,
        )
        expert_score_group4 = tl.load(
            s_score_bias_ptr + expert_idx_group4,
            mask=expert_idx_group4 < num_experts,
            other=neg_inf,
        )
        expert_score_group5 = tl.load(
            s_score_bias_ptr + expert_idx_group5,
            mask=expert_idx_group5 < num_experts,
            other=neg_inf,
        )
        expert_score_group6 = tl.load(
            s_score_bias_ptr + expert_idx_group6,
            mask=expert_idx_group6 < num_experts,
            other=neg_inf,
        )
        expert_score_group7 = tl.load(
            s_score_bias_ptr + expert_idx_group7,
            mask=expert_idx_group7 < num_experts,
            other=neg_inf,
        )
        comp_val_idx20 = _pack_val_idx_fp32(expert_score_group0, expert_idx_group0)
        comp_val_idx21 = _pack_val_idx_fp32(expert_score_group1, expert_idx_group1)
        comp_val_idx22 = _pack_val_idx_fp32(expert_score_group2, expert_idx_group2)
        comp_val_idx23 = _pack_val_idx_fp32(expert_score_group3, expert_idx_group3)
        comp_val_idx24 = _pack_val_idx_fp32(expert_score_group4, expert_idx_group4)
        comp_val_idx25 = _pack_val_idx_fp32(expert_score_group5, expert_idx_group5)
        comp_val_idx26 = _pack_val_idx_fp32(expert_score_group6, expert_idx_group6)
        comp_val_idx27 = _pack_val_idx_fp32(expert_score_group7, expert_idx_group7)
        # descending bitonic sort
        # k = 2, j = 1
        comp_val_idx20, comp_val_idx21 = max(comp_val_idx20, comp_val_idx21), min(
            comp_val_idx20, comp_val_idx21
        )
        comp_val_idx22, comp_val_idx23 = min(comp_val_idx22, comp_val_idx23), max(
            comp_val_idx22, comp_val_idx23
        )
        comp_val_idx24, comp_val_idx25 = max(comp_val_idx24, comp_val_idx25), min(
            comp_val_idx24, comp_val_idx25
        )
        comp_val_idx26, comp_val_idx27 = min(comp_val_idx26, comp_val_idx27), max(
            comp_val_idx26, comp_val_idx27
        )
        # k = 4, j = 2
        comp_val_idx20, comp_val_idx22 = max(comp_val_idx20, comp_val_idx22), min(
            comp_val_idx20, comp_val_idx22
        )
        comp_val_idx21, comp_val_idx23 = max(comp_val_idx21, comp_val_idx23), min(
            comp_val_idx21, comp_val_idx23
        )
        comp_val_idx24, comp_val_idx26 = min(comp_val_idx24, comp_val_idx26), max(
            comp_val_idx24, comp_val_idx26
        )
        comp_val_idx25, comp_val_idx27 = min(comp_val_idx25, comp_val_idx27), max(
            comp_val_idx25, comp_val_idx27
        )
        # k = 4, j = 1
        comp_val_idx20, comp_val_idx21 = max(comp_val_idx20, comp_val_idx21), min(
            comp_val_idx20, comp_val_idx21
        )
        comp_val_idx22, comp_val_idx23 = max(comp_val_idx22, comp_val_idx23), min(
            comp_val_idx22, comp_val_idx23
        )
        comp_val_idx24, comp_val_idx25 = min(comp_val_idx24, comp_val_idx25), max(
            comp_val_idx24, comp_val_idx25
        )
        comp_val_idx26, comp_val_idx27 = min(comp_val_idx26, comp_val_idx27), max(
            comp_val_idx26, comp_val_idx27
        )
        # k = 8, j = 4
        comp_val_idx20, comp_val_idx24 = max(comp_val_idx20, comp_val_idx24), min(
            comp_val_idx20, comp_val_idx24
        )
        comp_val_idx21, comp_val_idx25 = max(comp_val_idx21, comp_val_idx25), min(
            comp_val_idx21, comp_val_idx25
        )
        comp_val_idx22, comp_val_idx26 = max(comp_val_idx22, comp_val_idx26), min(
            comp_val_idx22, comp_val_idx26
        )
        comp_val_idx23, comp_val_idx27 = max(comp_val_idx23, comp_val_idx27), min(
            comp_val_idx23, comp_val_idx27
        )
        # k = 8, j = 2
        comp_val_idx20, comp_val_idx22 = max(comp_val_idx20, comp_val_idx22), min(
            comp_val_idx20, comp_val_idx22
        )
        comp_val_idx21, comp_val_idx23 = max(comp_val_idx21, comp_val_idx23), min(
            comp_val_idx21, comp_val_idx23
        )
        comp_val_idx24, comp_val_idx26 = max(comp_val_idx24, comp_val_idx26), min(
            comp_val_idx24, comp_val_idx26
        )
        comp_val_idx25, comp_val_idx27 = max(comp_val_idx25, comp_val_idx27), min(
            comp_val_idx25, comp_val_idx27
        )
        # k = 8, j = 1
        comp_val_idx20, comp_val_idx21 = max(comp_val_idx20, comp_val_idx21), min(
            comp_val_idx20, comp_val_idx21
        )
        comp_val_idx22, comp_val_idx23 = max(comp_val_idx22, comp_val_idx23), min(
            comp_val_idx22, comp_val_idx23
        )
        comp_val_idx24, comp_val_idx25 = max(comp_val_idx24, comp_val_idx25), min(
            comp_val_idx24, comp_val_idx25
        )
        comp_val_idx26, comp_val_idx27 = max(comp_val_idx26, comp_val_idx27), min(
            comp_val_idx26, comp_val_idx27
        )

        min_val2 = tl.full((WARP_SIZE,), neg_inf, dtype=tl.float32)
        top_experts = tl.full((WARP_SIZE,), MAX_IDX, dtype=tl.uint32)
        packed_max20 = tl.full((), 0, dtype=tl.uint64)
        for kk in tl.static_range(0, topk):
            update = (kk > 0) & (comp_val_idx20 == packed_max20)
            comp_val_idx20 = tl.where(
                update,
                comp_val_idx21,
                comp_val_idx20,
            )
            comp_val_idx21 = tl.where(
                update,
                comp_val_idx22,
                comp_val_idx21,
            )
            comp_val_idx22 = tl.where(
                update,
                comp_val_idx23,
                comp_val_idx22,
            )
            comp_val_idx23 = tl.where(
                update,
                comp_val_idx24,
                comp_val_idx23,
            )
            comp_val_idx24 = tl.where(
                update,
                comp_val_idx25,
                comp_val_idx24,
            )
            comp_val_idx25 = tl.where(
                update,
                comp_val_idx26,
                comp_val_idx25,
            )
            comp_val_idx26 = tl.where(
                update,
                comp_val_idx27,
                comp_val_idx26,
            )
            comp_val_idx27 = tl.where(
                update,
                _pack_val_idx_fp32(min_val2, expert_idx_group7),
                comp_val_idx27,
            )
            if USE_UINT32_REDUX:
                packed_max20 = _packed_max_uint32(comp_val_idx20, 0)
            else:
                packed_max20 = tl.max(comp_val_idx20)
            _3, out_idx = _unpack_val_idx_fp32(packed_max20)
            top_experts = tl.where(lane == kk, out_idx, top_experts)

    # step5: renormalize and output
    lane_unbiased = tl.load(
        s_score_sigmoid_ptr + top_experts, mask=lane < topk, other=0.0
    )
    topk_sum = 1e-20
    if renormalize:
        topk_sum += tl.sum(lane_unbiased)
    scale = routed_scaling_factor.to(tl.float32)
    if renormalize:
        scale /= topk_sum
    tl.store(topk_values_ptr + lane, lane_unbiased * scale, mask=lane < topk)
    tl.store(topk_indices_ptr + lane, top_experts, mask=lane < topk)


@triton.jit
def _legacy_sigmoid(x, dtype: tl.constexpr):
    # Match the legacy grouped path: separate same-dtype half/tanh/half/add.
    half = (x.to(tl.float32) * 0.5).to(dtype)
    t = tl_extra_shim.tanh(half.to(tl.float32)).to(dtype)
    half_t = (t.to(tl.float32) * 0.5).to(dtype)
    return (half_t.to(tl.float32) + 0.5).to(dtype)


@triton.jit
def _write_padding(weights, ids, token, topk: tl.constexpr, block: tl.constexpr):
    slot = tl.arange(0, block)
    tl.store(weights + token * topk + slot, 0.0, slot < topk)
    tl.store(ids + token * topk + slot, -1, slot < topk)


@triton.jit
def _masked_single_group_kernel(
    scores,
    bias,
    padding,
    out_weights,
    out_ids,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    stride_score,
    routed_scaling_factor,
    RENORMALIZE: tl.constexpr,
    SCORING_FUNC: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    USE_FAST_SIGMOID: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    if HAS_PADDING:
        if tl.load(padding + row):
            _write_padding(out_weights, out_ids, row, topk, BLOCK_TOPK)
            return
    expert = tl.arange(0, BLOCK_EXPERTS)
    unb = tl.load(scores + row * stride_score + expert, expert < num_experts, 0.0)
    if SCORING_FUNC == 1:
        if USE_FAST_SIGMOID:
            unb = _sigmoid(unb.to(tl.float32)).to(INPUT_DTYPE)
        else:
            unb = _legacy_sigmoid(unb, INPUT_DTYPE)
    b = tl.load(bias + expert, expert < num_experts, 0.0).to(INPUT_DTYPE)
    s = (unb + b).to(tl.float32)
    s = tl.where(s == 0.0, 0.0, s)  # Signed zeros compare equal on the legacy path.
    # Zero key for invalid lanes sorts behind every valid expert including -inf.
    keys = tl.where(expert < num_experts, _pack_val_idx_fp32(s, expert), 0)
    winners = tl.topk(keys, BLOCK_TOPK)
    chosen, ids = _unpack_val_idx_fp32(winners)
    slot = tl.arange(0, BLOCK_TOPK)
    vals = tl.load(scores + row * stride_score + ids, slot < topk, 0.0)
    if SCORING_FUNC == 1:
        if USE_FAST_SIGMOID:
            vals = _sigmoid(vals.to(tl.float32)).to(INPUT_DTYPE)
        else:
            vals = _legacy_sigmoid(vals, INPUT_DTYPE)
    vals = tl.where(slot < topk, vals.to(tl.float32), 0.0)
    scale = routed_scaling_factor
    if RENORMALIZE:
        scale = scale / (tl.sum(vals) + 1e-20)
    # Old generic routing emits uniform weights when the single group is invalid.
    group_score = tl.sum(tl.where(slot < 2, chosen, 0.0)).to(INPUT_DTYPE).to(tl.float32)
    valid = (group_score > -float("inf")) & (group_score < float("inf"))
    tl.store(
        out_weights + row * topk + slot,
        tl.where(valid, vals * scale, 1.0 / topk),
        slot < topk,
    )
    tl.store(out_ids + row * topk + slot, tl.where(valid, ids, slot), slot < topk)


@triton.jit
def _load_packed_contiguous_expert(
    scores, bias, offsets, experts, SCORING_FUNC: tl.constexpr
):
    valid = offsets < experts
    raw = tl.load(scores + offsets, valid, -float("inf")).to(tl.float32)
    if SCORING_FUNC == 1:
        raw = _sigmoid(raw)
    correction = tl.load(bias + offsets, valid, 0.0).to(tl.float32)
    ranked = tl.where(valid, raw + correction, -float("inf"))
    return _pack_val_idx_fp32(ranked, offsets)


@libtuner(
    configs=runtime.get_tuned_config("grouped_topk_single_register"),
    key=["tokens", "experts", "topk", "SCORING_FUNC", "RENORMALIZE"],
    strategy=["log", "default", "default", "default", "default"],
)
@triton.jit
def _grouped_topk_single_register_kernel(
    scores,
    bias,
    weights,
    ids,
    tokens,
    experts: tl.constexpr,
    topk: tl.constexpr,
    stride,
    scale,
    SCORING_FUNC: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    padding=None,
    HAS_PADDING: tl.constexpr = False,
):
    row = tl.program_id(0)
    if HAS_PADDING:
        if tl.load(padding + row):
            _write_padding(weights, ids, row, topk, 32)
            return
    WARP_SIZE: tl.constexpr = 32
    MAX_IDX: tl.constexpr = 65535
    neg_inf: tl.constexpr = -float("inf")
    lane = tl.arange(0, 32)
    # Each lane owns eight candidates. Sort locally before extracting K winners.
    expert_idx_group7 = lane + 224
    comp_val_idx20 = _load_packed_contiguous_expert(
        scores + row * stride, bias, lane + 0, experts, SCORING_FUNC
    )
    comp_val_idx21 = _load_packed_contiguous_expert(
        scores + row * stride, bias, lane + 32, experts, SCORING_FUNC
    )
    comp_val_idx22 = _load_packed_contiguous_expert(
        scores + row * stride, bias, lane + 64, experts, SCORING_FUNC
    )
    comp_val_idx23 = _load_packed_contiguous_expert(
        scores + row * stride, bias, lane + 96, experts, SCORING_FUNC
    )
    comp_val_idx24 = _load_packed_contiguous_expert(
        scores + row * stride, bias, lane + 128, experts, SCORING_FUNC
    )
    comp_val_idx25 = _load_packed_contiguous_expert(
        scores + row * stride, bias, lane + 160, experts, SCORING_FUNC
    )
    comp_val_idx26 = _load_packed_contiguous_expert(
        scores + row * stride, bias, lane + 192, experts, SCORING_FUNC
    )
    comp_val_idx27 = _load_packed_contiguous_expert(
        scores + row * stride, bias, lane + 224, experts, SCORING_FUNC
    )
    # descending bitonic sort
    # k = 2, j = 1
    comp_val_idx20, comp_val_idx21 = max(comp_val_idx20, comp_val_idx21), min(
        comp_val_idx20, comp_val_idx21
    )
    comp_val_idx22, comp_val_idx23 = min(comp_val_idx22, comp_val_idx23), max(
        comp_val_idx22, comp_val_idx23
    )
    comp_val_idx24, comp_val_idx25 = max(comp_val_idx24, comp_val_idx25), min(
        comp_val_idx24, comp_val_idx25
    )
    comp_val_idx26, comp_val_idx27 = min(comp_val_idx26, comp_val_idx27), max(
        comp_val_idx26, comp_val_idx27
    )
    # k = 4, j = 2
    comp_val_idx20, comp_val_idx22 = max(comp_val_idx20, comp_val_idx22), min(
        comp_val_idx20, comp_val_idx22
    )
    comp_val_idx21, comp_val_idx23 = max(comp_val_idx21, comp_val_idx23), min(
        comp_val_idx21, comp_val_idx23
    )
    comp_val_idx24, comp_val_idx26 = min(comp_val_idx24, comp_val_idx26), max(
        comp_val_idx24, comp_val_idx26
    )
    comp_val_idx25, comp_val_idx27 = min(comp_val_idx25, comp_val_idx27), max(
        comp_val_idx25, comp_val_idx27
    )
    # k = 4, j = 1
    comp_val_idx20, comp_val_idx21 = max(comp_val_idx20, comp_val_idx21), min(
        comp_val_idx20, comp_val_idx21
    )
    comp_val_idx22, comp_val_idx23 = max(comp_val_idx22, comp_val_idx23), min(
        comp_val_idx22, comp_val_idx23
    )
    comp_val_idx24, comp_val_idx25 = min(comp_val_idx24, comp_val_idx25), max(
        comp_val_idx24, comp_val_idx25
    )
    comp_val_idx26, comp_val_idx27 = min(comp_val_idx26, comp_val_idx27), max(
        comp_val_idx26, comp_val_idx27
    )
    # k = 8, j = 4
    comp_val_idx20, comp_val_idx24 = max(comp_val_idx20, comp_val_idx24), min(
        comp_val_idx20, comp_val_idx24
    )
    comp_val_idx21, comp_val_idx25 = max(comp_val_idx21, comp_val_idx25), min(
        comp_val_idx21, comp_val_idx25
    )
    comp_val_idx22, comp_val_idx26 = max(comp_val_idx22, comp_val_idx26), min(
        comp_val_idx22, comp_val_idx26
    )
    comp_val_idx23, comp_val_idx27 = max(comp_val_idx23, comp_val_idx27), min(
        comp_val_idx23, comp_val_idx27
    )
    # k = 8, j = 2
    comp_val_idx20, comp_val_idx22 = max(comp_val_idx20, comp_val_idx22), min(
        comp_val_idx20, comp_val_idx22
    )
    comp_val_idx21, comp_val_idx23 = max(comp_val_idx21, comp_val_idx23), min(
        comp_val_idx21, comp_val_idx23
    )
    comp_val_idx24, comp_val_idx26 = max(comp_val_idx24, comp_val_idx26), min(
        comp_val_idx24, comp_val_idx26
    )
    comp_val_idx25, comp_val_idx27 = max(comp_val_idx25, comp_val_idx27), min(
        comp_val_idx25, comp_val_idx27
    )
    # k = 8, j = 1
    comp_val_idx20, comp_val_idx21 = max(comp_val_idx20, comp_val_idx21), min(
        comp_val_idx20, comp_val_idx21
    )
    comp_val_idx22, comp_val_idx23 = max(comp_val_idx22, comp_val_idx23), min(
        comp_val_idx22, comp_val_idx23
    )
    comp_val_idx24, comp_val_idx25 = max(comp_val_idx24, comp_val_idx25), min(
        comp_val_idx24, comp_val_idx25
    )
    comp_val_idx26, comp_val_idx27 = max(comp_val_idx26, comp_val_idx27), min(
        comp_val_idx26, comp_val_idx27
    )

    min_val2 = tl.full((WARP_SIZE,), neg_inf, dtype=tl.float32)
    top_experts = tl.full((WARP_SIZE,), MAX_IDX, dtype=tl.uint32)
    packed_max20 = tl.full((), 0, dtype=tl.uint64)
    for kk in tl.static_range(0, topk):
        update = (kk > 0) & (comp_val_idx20 == packed_max20)
        comp_val_idx20 = tl.where(
            update,
            comp_val_idx21,
            comp_val_idx20,
        )
        comp_val_idx21 = tl.where(
            update,
            comp_val_idx22,
            comp_val_idx21,
        )
        comp_val_idx22 = tl.where(
            update,
            comp_val_idx23,
            comp_val_idx22,
        )
        comp_val_idx23 = tl.where(
            update,
            comp_val_idx24,
            comp_val_idx23,
        )
        comp_val_idx24 = tl.where(
            update,
            comp_val_idx25,
            comp_val_idx24,
        )
        comp_val_idx25 = tl.where(
            update,
            comp_val_idx26,
            comp_val_idx25,
        )
        comp_val_idx26 = tl.where(
            update,
            comp_val_idx27,
            comp_val_idx26,
        )
        comp_val_idx27 = tl.where(
            update,
            _pack_val_idx_fp32(min_val2, expert_idx_group7),
            comp_val_idx27,
        )
        packed_max20 = _packed_max_uint32(comp_val_idx20, 0)
        _3, out_idx = _unpack_val_idx_fp32(packed_max20)
        top_experts = tl.where(lane == kk, out_idx, top_experts)

    chosen = tl.load(scores + row * stride + top_experts, lane < topk, 0.0).to(
        tl.float32
    )
    if SCORING_FUNC == 1:
        chosen = _sigmoid(chosen)
    chosen = tl.where(lane < topk, chosen, 0.0)
    if RENORMALIZE:
        scale = scale / (tl.sum(chosen) + 1e-20)
    tl.store(weights + row * topk + lane, chosen * scale, lane < topk)
    tl.store(ids + row * topk + lane, top_experts, lane < topk)


@triton.jit
def _ordered_fp32_key(x):
    bits = x.to(tl.uint32, bitcast=True)
    return tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)


@triton.jit
def _load_packed_expert(
    scores,
    bias,
    offsets,
    experts,
    SCORING_FUNC: tl.constexpr,
    stride_expert,
    stride_bias,
):
    valid = offsets < experts
    raw = tl.load(scores + offsets * stride_expert, valid, -float("inf")).to(tl.float32)
    if SCORING_FUNC == 1:
        raw = _sigmoid(raw)
    correction = tl.load(bias + offsets * stride_bias, valid, 0.0).to(tl.float32)
    ranked = tl.where(valid, raw + correction, -float("inf"))
    return _pack_val_idx_fp32(ranked, offsets)


@triton.jit
def _packed_max_uint32(values, axis: tl.constexpr):
    high = (values >> 32).to(tl.uint32)
    low = values.to(tl.uint32)
    high_max = tl.max(high, axis)
    low_max = tl.max(tl.where(high == tl.expand_dims(high_max, axis), low, 0), axis)
    return (high_max.to(tl.uint64) << 32) | low_max.to(tl.uint64)


@triton.jit
def _load_compact_expert(
    scores, bias, offsets, experts, SF: tl.constexpr, stride_expert, stride_bias
):
    v = tl.load(scores + offsets * stride_expert, offsets < experts, 0.0).to(tl.float32)
    if SF:
        v = _sigmoid(v)
    b = tl.load(bias + offsets * stride_bias, offsets < experts, 0.0).to(tl.float32)
    return tl.where(
        offsets < experts,
        (_ordered_fp32_key(v + b) & 0xFFFFFF00) | (255 - offsets).to(tl.uint32),
        0,
    )


@triton.jit
def _select_register(
    scores,
    bias,
    tokens,
    experts: tl.constexpr,
    topk: tl.constexpr,
    stride,
    stride_expert,
    stride_bias,
    SCORING_FUNC: tl.constexpr,
    COMPACT: tl.constexpr,
):
    row = tl.program_id(0)
    WARP_SIZE: tl.constexpr = 32
    MAX_IDX: tl.constexpr = 65535
    neg_inf: tl.constexpr = -float("inf")
    lane = tl.arange(0, 32)
    # Each lane owns eight candidates. Sort locally before extracting K winners.
    expert_idx_group7 = lane + 224
    load_fn: tl.constexpr = _load_compact_expert if COMPACT else _load_packed_expert
    comp_val_idx20 = load_fn(
        scores + row * stride,
        bias,
        lane + 0,
        experts,
        SCORING_FUNC,
        stride_expert,
        stride_bias,
    )
    comp_val_idx21 = load_fn(
        scores + row * stride,
        bias,
        lane + 32,
        experts,
        SCORING_FUNC,
        stride_expert,
        stride_bias,
    )
    comp_val_idx22 = load_fn(
        scores + row * stride,
        bias,
        lane + 64,
        experts,
        SCORING_FUNC,
        stride_expert,
        stride_bias,
    )
    comp_val_idx23 = load_fn(
        scores + row * stride,
        bias,
        lane + 96,
        experts,
        SCORING_FUNC,
        stride_expert,
        stride_bias,
    )
    comp_val_idx24 = load_fn(
        scores + row * stride,
        bias,
        lane + 128,
        experts,
        SCORING_FUNC,
        stride_expert,
        stride_bias,
    )
    comp_val_idx25 = load_fn(
        scores + row * stride,
        bias,
        lane + 160,
        experts,
        SCORING_FUNC,
        stride_expert,
        stride_bias,
    )
    comp_val_idx26 = load_fn(
        scores + row * stride,
        bias,
        lane + 192,
        experts,
        SCORING_FUNC,
        stride_expert,
        stride_bias,
    )
    comp_val_idx27 = load_fn(
        scores + row * stride,
        bias,
        lane + 224,
        experts,
        SCORING_FUNC,
        stride_expert,
        stride_bias,
    )
    orig0 = comp_val_idx20
    orig1 = comp_val_idx21
    orig2 = comp_val_idx22
    orig3 = comp_val_idx23
    orig4 = comp_val_idx24
    orig5 = comp_val_idx25
    orig6 = comp_val_idx26
    orig7 = comp_val_idx27
    # descending bitonic sort
    # k = 2, j = 1
    comp_val_idx20, comp_val_idx21 = max(comp_val_idx20, comp_val_idx21), min(
        comp_val_idx20, comp_val_idx21
    )
    comp_val_idx22, comp_val_idx23 = min(comp_val_idx22, comp_val_idx23), max(
        comp_val_idx22, comp_val_idx23
    )
    comp_val_idx24, comp_val_idx25 = max(comp_val_idx24, comp_val_idx25), min(
        comp_val_idx24, comp_val_idx25
    )
    comp_val_idx26, comp_val_idx27 = min(comp_val_idx26, comp_val_idx27), max(
        comp_val_idx26, comp_val_idx27
    )
    # k = 4, j = 2
    comp_val_idx20, comp_val_idx22 = max(comp_val_idx20, comp_val_idx22), min(
        comp_val_idx20, comp_val_idx22
    )
    comp_val_idx21, comp_val_idx23 = max(comp_val_idx21, comp_val_idx23), min(
        comp_val_idx21, comp_val_idx23
    )
    comp_val_idx24, comp_val_idx26 = min(comp_val_idx24, comp_val_idx26), max(
        comp_val_idx24, comp_val_idx26
    )
    comp_val_idx25, comp_val_idx27 = min(comp_val_idx25, comp_val_idx27), max(
        comp_val_idx25, comp_val_idx27
    )
    # k = 4, j = 1
    comp_val_idx20, comp_val_idx21 = max(comp_val_idx20, comp_val_idx21), min(
        comp_val_idx20, comp_val_idx21
    )
    comp_val_idx22, comp_val_idx23 = max(comp_val_idx22, comp_val_idx23), min(
        comp_val_idx22, comp_val_idx23
    )
    comp_val_idx24, comp_val_idx25 = min(comp_val_idx24, comp_val_idx25), max(
        comp_val_idx24, comp_val_idx25
    )
    comp_val_idx26, comp_val_idx27 = min(comp_val_idx26, comp_val_idx27), max(
        comp_val_idx26, comp_val_idx27
    )
    # k = 8, j = 4
    comp_val_idx20, comp_val_idx24 = max(comp_val_idx20, comp_val_idx24), min(
        comp_val_idx20, comp_val_idx24
    )
    comp_val_idx21, comp_val_idx25 = max(comp_val_idx21, comp_val_idx25), min(
        comp_val_idx21, comp_val_idx25
    )
    comp_val_idx22, comp_val_idx26 = max(comp_val_idx22, comp_val_idx26), min(
        comp_val_idx22, comp_val_idx26
    )
    comp_val_idx23, comp_val_idx27 = max(comp_val_idx23, comp_val_idx27), min(
        comp_val_idx23, comp_val_idx27
    )
    # k = 8, j = 2
    comp_val_idx20, comp_val_idx22 = max(comp_val_idx20, comp_val_idx22), min(
        comp_val_idx20, comp_val_idx22
    )
    comp_val_idx21, comp_val_idx23 = max(comp_val_idx21, comp_val_idx23), min(
        comp_val_idx21, comp_val_idx23
    )
    comp_val_idx24, comp_val_idx26 = max(comp_val_idx24, comp_val_idx26), min(
        comp_val_idx24, comp_val_idx26
    )
    comp_val_idx25, comp_val_idx27 = max(comp_val_idx25, comp_val_idx27), min(
        comp_val_idx25, comp_val_idx27
    )
    # k = 8, j = 1
    comp_val_idx20, comp_val_idx21 = max(comp_val_idx20, comp_val_idx21), min(
        comp_val_idx20, comp_val_idx21
    )
    comp_val_idx22, comp_val_idx23 = max(comp_val_idx22, comp_val_idx23), min(
        comp_val_idx22, comp_val_idx23
    )
    comp_val_idx24, comp_val_idx25 = max(comp_val_idx24, comp_val_idx25), min(
        comp_val_idx24, comp_val_idx25
    )
    comp_val_idx26, comp_val_idx27 = max(comp_val_idx26, comp_val_idx27), min(
        comp_val_idx26, comp_val_idx27
    )

    min_val2 = tl.full((WARP_SIZE,), neg_inf, dtype=tl.float32)
    top_experts = tl.full((WARP_SIZE,), MAX_IDX, dtype=tl.uint32)
    packed_max20 = tl.full((), 0, dtype=tl.uint32 if COMPACT else tl.uint64)
    for kk in tl.static_range(0, topk):
        update = (kk > 0) & (comp_val_idx20 == packed_max20)
        comp_val_idx20 = tl.where(
            update,
            comp_val_idx21,
            comp_val_idx20,
        )
        comp_val_idx21 = tl.where(
            update,
            comp_val_idx22,
            comp_val_idx21,
        )
        comp_val_idx22 = tl.where(
            update,
            comp_val_idx23,
            comp_val_idx22,
        )
        comp_val_idx23 = tl.where(
            update,
            comp_val_idx24,
            comp_val_idx23,
        )
        comp_val_idx24 = tl.where(
            update,
            comp_val_idx25,
            comp_val_idx24,
        )
        comp_val_idx25 = tl.where(
            update,
            comp_val_idx26,
            comp_val_idx25,
        )
        comp_val_idx26 = tl.where(
            update,
            comp_val_idx27,
            comp_val_idx26,
        )
        comp_val_idx27 = tl.where(
            update,
            0 if COMPACT else _pack_val_idx_fp32(min_val2, expert_idx_group7),
            comp_val_idx27,
        )
        if COMPACT:
            packed_max20 = tl.max(comp_val_idx20, 0)
        else:
            packed_max20 = _packed_max_uint32(comp_val_idx20, 0)
        if COMPACT:
            out_idx = 255 - (packed_max20 & 255).to(tl.int32)
        else:
            _3, out_idx = _unpack_val_idx_fp32(packed_max20)
        top_experts = tl.where(lane == kk, out_idx, top_experts)

    if COMPACT:
        threshold = packed_max20 >> 8
        count = (
            ((orig0 >> 8) >= threshold).to(tl.int32)
            + ((orig1 >> 8) >= threshold).to(tl.int32)
            + ((orig2 >> 8) >= threshold).to(tl.int32)
            + ((orig3 >> 8) >= threshold).to(tl.int32)
            + ((orig4 >> 8) >= threshold).to(tl.int32)
            + ((orig5 >> 8) >= threshold).to(tl.int32)
            + ((orig6 >> 8) >= threshold).to(tl.int32)
            + ((orig7 >> 8) >= threshold).to(tl.int32)
        )
        ambig = tl.sum(count) > topk
    else:
        ambig = False
    return top_experts, ambig


@libtuner(
    configs=runtime.get_tuned_config("grouped_topk_register_select"),
    use_cuda_graph=True,
    key=["tokens", "experts", "topk", "SCORING_FUNC", "RENORMALIZE", "HAS_PADDING"],
    strategy=["log", "default", "default", "default", "default", "default"],
)
@triton.jit
def _grouped_topk_register_select_kernel(
    scores,
    bias,
    weights,
    ids,
    padding,
    tokens,
    experts: tl.constexpr,
    topk: tl.constexpr,
    stride,
    stride_expert,
    stride_bias,
    scale,
    SCORING_FUNC: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    COMPACT: tl.constexpr,
):
    if HAS_PADDING:
        if tl.load(padding + tl.program_id(0)):
            _write_padding(weights, ids, tl.program_id(0), topk, 32)
            return
    # The compact key carries 24 order bits plus an 8-bit expert index.
    # It only constructs a candidate set; full FP32 keys determine the result.
    top_experts, ambig = _select_register(
        scores,
        bias,
        tokens,
        experts,
        topk,
        stride,
        stride_expert,
        stride_bias,
        SCORING_FUNC,
        COMPACT,
    )
    row = tl.program_id(0)
    lane = tl.arange(0, 32)
    # More than K experts in the cutoff prefix can hide a true winner.
    # Retry the row with full-width pairs instead of dropping any FP32 bits.
    if COMPACT and ambig:
        top_experts, unused_ambig = _select_register(
            scores,
            bias,
            tokens,
            experts,
            topk,
            stride,
            stride_expert,
            stride_bias,
            SCORING_FUNC,
            False,
        )
    elif COMPACT:
        # Every possible winner is present, but their coarse order may differ.
        kk = tl.arange(0, triton.next_power_of_2(topk))
        small = tl.gather(top_experts, kk, 0)
        v = tl.load(
            scores + row * stride + small * stride_expert,
            kk < topk,
            0.0,
        ).to(tl.float32)
        if SCORING_FUNC:
            v = _sigmoid(v)
        b = tl.load(bias + small * stride_bias, kk < topk, 0.0).to(tl.float32)
        pp = tl.where(kk < topk, _pack_val_idx_fp32(v + b, small), 0)
        ordered = tl.sort(pp, dim=0, descending=True)
        sorted_values, sids = _unpack_val_idx_fp32(ordered)
        top_experts = tl.gather(sids, tl.minimum(lane, topk - 1), 0)
    chosen = tl.load(
        scores + row * stride + top_experts * stride_expert,
        lane < topk,
        0.0,
    ).to(tl.float32)
    if SCORING_FUNC:
        chosen = _sigmoid(chosen)
    chosen = tl.where(lane < topk, chosen, 0.0)
    if RENORMALIZE:
        scale = scale / (tl.sum(chosen) + 1e-20)
    tl.store(weights + row * topk + lane, chosen * scale, lane < topk)
    tl.store(ids + row * topk + lane, top_experts, lane < topk)


def _prune_fused_select_configs(configs, named_args, **kwargs):
    # Large batches are throughput-bound. Cross-warp sorting costs more than
    # the single-warp reduction; retaining it in an event-based tuning trial
    # can also select an unstable result for these short kernels.
    if named_args["TOKENS"] > 512:
        return [c for c in configs if c.num_warps == 1 and not c.kwargs["USE_SORT"]]
    return [c for c in configs if c.num_warps == 4]


@libtuner(
    configs=runtime.get_tuned_config("grouped_topk_fused_select"),
    prune_configs_by={"early_config_prune": _prune_fused_select_configs},
    use_cuda_graph=True,
    key=[
        "TOKENS",
        "EXPERTS",
        "GROUPS",
        "KEEP",
        "TOPK",
        "SCORING_FUNC",
        "RENORMALIZE",
        "HAS_PADDING",
    ],
    strategy=[
        "log",
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
    ],
)
@triton.jit
def _grouped_topk_fused_select_kernel(
    scores,
    bias,
    weights,
    indices,
    padding,
    TOKENS: tl.constexpr,
    EXPERTS: tl.constexpr,
    GROUPS: tl.constexpr,
    KEEP: tl.constexpr,
    TOPK: tl.constexpr,
    stride_row,
    stride_expert,
    stride_bias,
    scale,
    SCORING_FUNC: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    USE_SORT: tl.constexpr,
    FAST_RANK: tl.constexpr = False,
):
    row = tl.program_id(0)
    if HAS_PADDING:
        if tl.load(padding + row):
            _write_padding(weights, indices, row, TOPK, triton.next_power_of_2(TOPK))
            return
    GP: tl.constexpr = triton.next_power_of_2(GROUPS)
    EP: tl.constexpr = triton.next_power_of_2(EXPERTS // GROUPS)
    KP: tl.constexpr = triton.next_power_of_2(TOPK)
    DT: tl.constexpr = scores.dtype.element_ty
    group = tl.arange(0, GP)
    lane = tl.arange(0, EP)
    expert = group[:, None] * (EXPERTS // GROUPS) + lane[None, :]
    valid = (group[:, None] < GROUPS) & (lane[None, :] < EXPERTS // GROUPS)
    unbiased = tl.load(scores + row * stride_row + expert * stride_expert, valid, 0.0)
    if FAST_RANK:
        unbiased = unbiased.to(tl.float32)
        if SCORING_FUNC == 1:
            unbiased = _sigmoid(unbiased)
        correction = tl.load(bias + expert * stride_bias, valid, 0.0).to(tl.float32)
        ranked = unbiased + correction
    else:
        if SCORING_FUNC == 1:
            unbiased = _legacy_sigmoid(unbiased, DT)
        correction = tl.load(bias + expert * stride_bias, valid, 0.0).to(DT)
        ranked = (unbiased + correction).to(DT).to(tl.float32)
    ranked = tl.where(valid, ranked, -float("inf"))
    if not FAST_RANK:
        # tl.max on the original generic route ignores NaNs. Packed integer
        # keys must exclude them explicitly before ranking.
        ranked = tl.where(ranked == ranked, ranked, -float("inf"))

    # Two reductions over values plus the deterministic removal of one maximum.
    # There is no global group-score or expert-score workspace.
    maximum = tl.max(ranked, 1)
    first = tl.min(tl.where(ranked == maximum[:, None], lane[None, :], 0x7FFFFFFF), 1)
    second = tl.max(tl.where(lane[None, :] == first[:, None], -float("inf"), ranked), 1)
    group_scores = maximum + second
    if not FAST_RANK:
        group_scores = group_scores.to(DT).to(tl.float32)
    if not FAST_RANK:
        group_scores = tl.where(
            (group_scores == group_scores) & (group_scores != float("inf")),
            group_scores,
            -float("inf"),
        )
    proceed = tl.max(group_scores) != -float("inf")
    if not FAST_RANK:
        group_scores = tl.where(group_scores == 0.0, 0.0, group_scores)
    group_keys = tl.where(group < GROUPS, _pack_val_idx_fp32(group_scores, group), 0)
    if triton.next_power_of_2(KEEP) == KEEP:
        threshold = tl.min(tl.topk(group_keys, KEEP))
    else:
        remaining = group_keys
        for _ in tl.static_range(KEEP):
            threshold = tl.max(remaining)
            remaining = tl.where(remaining == threshold, 0, remaining)
    enabled = (group_keys >= threshold) & (group < GROUPS)
    ranked = tl.where(enabled[:, None], ranked, -float("inf"))
    # The generic route compares signed zeros as equal, then uses the index.
    if not FAST_RANK:
        ranked = tl.where(ranked == 0.0, 0.0, ranked)

    if DT == tl.float32 or FAST_RANK:
        candidates = _pack_val_idx_fp32(ranked, expert)
    else:
        # DeepSelect's narrow pair format is exact here: ranking was rounded
        # to the input dtype before selection. FP32 bias is never narrowed on
        # the separate fast routing path.
        bits = ranked.to(DT).to(tl.uint16, bitcast=True).to(tl.uint32)
        ordered = bits ^ tl.where((bits & 0x8000) != 0, 0xFFFF, 0x8000)
        candidates = (ordered << 16) | (0xFFFF - expert).to(tl.uint32)
    candidates = tl.reshape(tl.where(valid, candidates, 0), (GP * EP,))
    slot = tl.arange(0, KP)
    if USE_SORT:
        winners = tl.topk(candidates, KP)
    else:
        winners = tl.full((KP,), 0, candidates.dtype)
        for k in tl.static_range(TOPK):
            best = tl.max(candidates)
            winners = tl.where(slot == k, best, winners)
            candidates = tl.where(candidates == best, 0, candidates)
    selected = (0xFFFF - (winners & 0xFFFF)).to(tl.int32)
    values = tl.load(
        scores + row * stride_row + selected * stride_expert, slot < TOPK, 0.0
    )
    if SCORING_FUNC == 1:
        if FAST_RANK:
            values = _sigmoid(values.to(tl.float32))
        else:
            values = _legacy_sigmoid(values, DT)
    values = tl.where(slot < TOPK, values.to(tl.float32), 0.0)
    if RENORMALIZE:
        scale = scale / (tl.sum(values) + 1e-20)
    if FAST_RANK:
        proceed = True
    tl.store(
        weights + row * TOPK + slot,
        tl.where(proceed, values * scale, 1.0 / TOPK),
        slot < TOPK,
    )
    tl.store(
        indices + row * TOPK + slot, tl.where(proceed, selected, slot), slot < TOPK
    )


def _validate_padding(scores, bias, is_padding, n_group, topk_group, topk):
    if runtime.device.vendor_name != "nvidia" or scores.device.type != "cuda":
        raise NotImplementedError("grouped_topk padding requires NVIDIA CUDA")
    rows, experts = scores.shape
    if (
        is_padding.dtype != torch.bool
        or is_padding.shape != (rows,)
        or is_padding.device != scores.device
        or not is_padding.is_contiguous()
    ):
        raise ValueError(
            "is_padding must be contiguous bool [num_tokens] on the scores device"
        )
    if (
        bias.shape != (experts,)
        or bias.device != scores.device
        or not bias.is_contiguous()
        or scores.stride(1) != 1
        or scores.stride(0) < experts
    ):
        raise ValueError(
            "masked grouped_topk needs row-major scores and contiguous same-device bias"
        )
    if not 1 <= topk_group <= n_group <= 32 or not 1 <= topk <= 32:
        raise ValueError("expected valid group counts and topk in [1, 32]")
    if (
        experts // n_group < 2
        or experts > 4096
        or topk > (experts // n_group) * topk_group
    ):
        raise ValueError(
            "masked grouped_topk needs at least 2 experts per group, at most 4096 "
            "experts, and at least topk candidates after group selection"
        )


def grouped_topk(
    scores: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
    is_padding: Optional[torch.Tensor] = None,
):
    """Route scores; optional CUDA padding rows emit zero weights and -1 IDs."""
    logger.debug("GEMS GROUPED TOPK")
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D Tensor")
    num_tokens, num_experts = scores.shape
    has_padding = is_padding is not None
    if has_padding:
        _validate_padding(scores, bias, is_padding, n_group, topk_group, topk)
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

    # GLM-5.2
    is_single_group = (
        (n_group == 1) & (topk_group == 1) & (num_experts <= 256) & (topk <= 8)
    )
    # DeepSeek-v3.2
    is_multi_group = (
        (n_group > 1)
        & (n_group <= 32)
        & (num_experts <= 256)
        & (num_experts_per_group <= 32)
        & (num_experts_per_group * topk_group <= 128)
        & (topk <= 8)
        & (topk_group <= 4)
    )
    if num_tokens == 0:
        shape = (0, topk)
        return (
            torch.empty(shape, device=scores.device, dtype=torch.float32),
            torch.empty(shape, device=scores.device, dtype=torch.int32),
        )
    if (
        runtime.device.vendor_name == "nvidia"
        and (
            not (is_single_group or is_multi_group)
            or (is_multi_group and (scores.stride(1) != 1 or bias.stride(0) != 1))
        )
        and 2 <= num_experts_per_group
        and num_experts <= 1024
        and 1 <= topk_group <= n_group
        and 1 <= topk <= min(32, num_experts_per_group * topk_group)
    ):
        shape = (num_tokens, topk)
        values = torch.empty(shape, device=scores.device, dtype=torch.float32)
        indices = torch.empty(shape, device=scores.device, dtype=torch.int32)
        _grouped_topk_fused_select_kernel[(num_tokens,)](
            scores,
            bias,
            values,
            indices,
            is_padding,
            num_tokens,
            num_experts,
            n_group,
            topk_group,
            topk,
            scores.stride(0),
            scores.stride(1),
            bias.stride(0),
            routed_scaling_factor,
            SCORING_FUNC=scoring_func,
            RENORMALIZE=renormalize,
            HAS_PADDING=has_padding,
            FAST_RANK=is_single_group or is_multi_group,
        )
        return values, indices

    if has_padding and (num_tokens == 0 or (n_group == 1 and not is_single_group)):
        shape = (num_tokens, topk)
        topk_values = torch.empty(shape, device=scores.device, dtype=torch.float32)
        topk_indices = torch.empty(shape, device=scores.device, dtype=torch.int32)
        if num_tokens:
            _masked_single_group_kernel[(num_tokens,)](
                scores,
                bias,
                is_padding,
                topk_values,
                topk_indices,
                num_tokens,
                num_experts,
                topk,
                scores.stride(0),
                routed_scaling_factor,
                RENORMALIZE=renormalize,
                SCORING_FUNC=scoring_func,
                HAS_PADDING=True,
                USE_FAST_SIGMOID=False,
                INPUT_DTYPE=INPUT_DTYPE,
                BLOCK_EXPERTS=triton.next_power_of_2(num_experts),
                BLOCK_TOPK=triton.next_power_of_2(max(2, topk)),
                # Use the native JIT binder, not LibEntry's 3.7 cache-hit launcher.
                num_warps=4,
            )
        return topk_values, topk_indices

    if is_single_group and runtime.device.vendor_name == "nvidia":
        shape = (num_tokens, topk)
        values = torch.empty(shape, device=scores.device, dtype=torch.float32)
        indices = torch.empty(shape, device=scores.device, dtype=torch.int32)
        if num_tokens < 1024 and scores.stride(1) == 1 and bias.stride(0) == 1:
            _grouped_topk_single_register_kernel[(num_tokens,)](
                scores,
                bias,
                values,
                indices,
                num_tokens,
                num_experts,
                topk,
                scores.stride(0),
                routed_scaling_factor,
                SCORING_FUNC=scoring_func,
                RENORMALIZE=renormalize,
                padding=is_padding,
                HAS_PADDING=has_padding,
            )
        else:
            _grouped_topk_register_select_kernel[(num_tokens,)](
                scores,
                bias,
                values,
                indices,
                is_padding,
                num_tokens,
                num_experts,
                topk,
                scores.stride(0),
                scores.stride(1),
                bias.stride(0),
                routed_scaling_factor,
                SCORING_FUNC=scoring_func,
                RENORMALIZE=renormalize,
                HAS_PADDING=has_padding,
            )
        return values, indices

    if is_single_group | is_multi_group:
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
        if not HAS_TLE:
            g_scores_sigmoid = torch.empty(
                (num_tokens, num_experts),
                device=scores.device,
                dtype=torch.float32,
            )
            g_scores_bias = torch.empty(
                (num_tokens, num_experts),
                device=scores.device,
                dtype=torch.float32,
            )
        else:
            g_scores_sigmoid = None
            g_scores_bias = None

        n_group_pad = triton.next_power_of_2(n_group)
        group_size_pad = triton.next_power_of_2(num_experts_per_group)
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
            g_scores_sigmoid,
            g_scores_bias,
            SCORING_FUNC=scoring_func,
            HAS_TLE=HAS_TLE,
            NUM_GROUPS_PAD=n_group_pad,
            GROUP_SIZE_PAD=group_size_pad,
            FULL_SHAPE=num_experts_per_group == group_size_pad
            and n_group == n_group_pad,
            USE_GROUPS=is_multi_group,
            USE_UINT32_REDUX=runtime.device.vendor_name == "nvidia",
            padding_ptr=is_padding,
            HAS_PADDING=has_padding,
            num_warps=1,
        )

        return topk_values, topk_indices

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
        scores_processed.stride(1),
        group_scores.stride(0),
        BLOCK_SIZE=BLOCK1,
        INPUT_DTYPE=INPUT_DTYPE,
        padding_ptr=is_padding,
        HAS_PADDING=has_padding,
        MASKED_SIGMOID=scoring_func == 1,
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
        scores_processed.stride(1),
        group_scores.stride(0),
        topk_values.stride(0),
        N_GROUP=n_group,
        TOPK_GROUP=topk_group,
        TOPK=topk,
        BLOCK_GROUP=BLOCK_GROUP,
        BLOCK_EXPERT=BLOCK_EXPERT,
        INPUT_DTYPE=INPUT_DTYPE,
        renormalize=int(renormalize),
        padding_ptr=is_padding,
        HAS_PADDING=has_padding,
        MASKED_SIGMOID=scoring_func == 1,
        TOPK_BLOCK=triton.next_power_of_2(topk) if has_padding else topk,
    )

    return topk_values, topk_indices
