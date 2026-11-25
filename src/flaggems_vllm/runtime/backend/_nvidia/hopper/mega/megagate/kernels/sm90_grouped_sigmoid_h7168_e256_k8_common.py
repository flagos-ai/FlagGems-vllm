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

"""Exact grouped-sigmoid routing shared by DSV3.2 MegaGate leaves."""

from __future__ import annotations

from typing import Final

import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import routing_primitives as _routing

HIDDEN: Final = 7168
EXPERTS: Final = 256
TOPK: Final = 8
ROUTED_SCALING_FACTOR: Final = 2.5
STATE_WORDS_PER_TILE: Final = 16
TICKET_WORD: Final = 4

JIT_EXPERTS = tl.constexpr(EXPERTS)
JIT_TOPK = tl.constexpr(TOPK)


@triton.jit
def _precise_sigmoid(raw):
    """Overflow-safe FP32 sigmoid without importing an old kernel family."""

    z = libdevice.exp(-tl.abs(raw))
    return tl.where(raw >= 0.0, 1.0 / (1.0 + z), z / (1.0 + z))


@triton.jit
def _compare_swap_local(lhs_score, lhs_offset, rhs_score, rhs_offset):
    take_rhs = (rhs_score > lhs_score) | (
        (rhs_score == lhs_score) & (rhs_offset < lhs_offset)
    )
    return (
        tl.where(take_rhs, rhs_score, lhs_score),
        tl.where(take_rhs, rhs_offset, lhs_offset),
        tl.where(take_rhs, lhs_score, rhs_score),
        tl.where(take_rhs, lhs_offset, rhs_offset),
    )


@triton.jit
def _sort4_pack(s0, s1, s2, s3):
    """Stable descending sort of four lane-local experts."""

    i0 = tl.zeros((128,), dtype=tl.uint32)
    i1 = tl.full((128,), 1, dtype=tl.uint32)
    i2 = tl.full((128,), 2, dtype=tl.uint32)
    i3 = tl.full((128,), 3, dtype=tl.uint32)
    s0, i0, s1, i1 = _compare_swap_local(s0, i0, s1, i1)
    s2, i2, s3, i3 = _compare_swap_local(s2, i2, s3, i3)
    s0, i0, s2, i2 = _compare_swap_local(s0, i0, s2, i2)
    s1, i1, s3, i3 = _compare_swap_local(s1, i1, s3, i3)
    s1, i1, s2, i2 = _compare_swap_local(s1, i1, s2, i2)
    permutation = i0 | (i1 << 2) | (i2 << 4) | (i3 << 6)
    return s0, s1, s2, s3, permutation


@triton.jit
def _sorted_head(cursor, s0, s1, s2, s3):
    return tl.where(
        cursor == 0,
        s0,
        tl.where(
            cursor == 1,
            s1,
            tl.where(
                cursor == 2,
                s2,
                tl.where(cursor == 3, s3, -float("inf")),
            ),
        ),
    )


@triton.jit
def _subwarp8_stable_best_expert(score_key, expert):
    """Stable argmax in four independent width-eight segments."""

    return tl.inline_asm_elementwise(
        """
        {
            .reg .b32 best_key, best_expert, peer_key, peer_expert;
            .reg .pred p_key_gt, p_idx_lt, p_tie, p_take;
            mov.u32 best_key, $1;
            mov.u32 best_expert, $2;

            shfl.sync.down.b32 peer_key, best_key, 4, 0x1807, 0xffffffff;
            shfl.sync.down.b32 peer_expert, best_expert, 4, 0x1807, 0xffffffff;
            setp.gt.u32 p_key_gt, peer_key, best_key;
            setp.lt.u32 p_idx_lt, peer_expert, best_expert;
            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;
            or.pred p_take, p_key_gt, p_tie;
            selp.u32 best_key, peer_key, best_key, p_take;
            selp.u32 best_expert, peer_expert, best_expert, p_take;

            shfl.sync.down.b32 peer_key, best_key, 2, 0x1807, 0xffffffff;
            shfl.sync.down.b32 peer_expert, best_expert, 2, 0x1807, 0xffffffff;
            setp.gt.u32 p_key_gt, peer_key, best_key;
            setp.lt.u32 p_idx_lt, peer_expert, best_expert;
            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;
            or.pred p_take, p_key_gt, p_tie;
            selp.u32 best_key, peer_key, best_key, p_take;
            selp.u32 best_expert, peer_expert, best_expert, p_take;

            shfl.sync.down.b32 peer_key, best_key, 1, 0x1807, 0xffffffff;
            shfl.sync.down.b32 peer_expert, best_expert, 1, 0x1807, 0xffffffff;
            setp.gt.u32 p_key_gt, peer_key, best_key;
            setp.lt.u32 p_idx_lt, peer_expert, best_expert;
            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;
            or.pred p_take, p_key_gt, p_tie;
            selp.u32 best_key, peer_key, best_key, p_take;
            selp.u32 best_expert, peer_expert, best_expert, p_take;

            shfl.sync.idx.b32 $0, best_expert, 0, 0x1807, 0xffffffff;
        }
        """,
        "=r,r,r",
        [score_key.to(tl.uint32), expert.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _subwarp8_broadcast_f32(value, source_lane):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b32 input_bits, output_bits;
            mov.b32 input_bits, $1;
            shfl.sync.idx.b32 output_bits, input_bits, $2, 0x1807, 0xffffffff;
            mov.b32 $0, output_bits;
        }
        """,
        "=f,f,r",
        [value, source_lane.to(tl.uint32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _parallel_subwarp8_group_top2_sum_packed(
    local_best_score,
    local_second_score,
    permutation,
    lane,
    wave_base: tl.constexpr,
):
    lane_u32 = lane.to(tl.uint32)
    lane_expert_base = wave_base + lane_u32 * 4
    local_best_expert = lane_expert_base + (permutation & 3)
    local_second_expert = lane_expert_base + ((permutation >> 2) & 3)

    best_key = _routing.ordered_score_key(local_best_score).to(tl.uint32)
    best_expert = _subwarp8_stable_best_expert(best_key, local_best_expert)
    best_owner_lane = (best_expert - wave_base) // 4
    best_score = _subwarp8_broadcast_f32(local_best_score, best_owner_lane)

    owns_best = local_best_expert == best_expert
    second_candidate_score = tl.where(owns_best, local_second_score, local_best_score)
    second_candidate_expert = tl.where(
        owns_best, local_second_expert, local_best_expert
    )
    second_key = _routing.ordered_score_key(second_candidate_score).to(tl.uint32)
    second_expert = _subwarp8_stable_best_expert(second_key, second_candidate_expert)
    second_owner_lane = (second_expert - wave_base) // 4
    second_score = _subwarp8_broadcast_f32(second_candidate_score, second_owner_lane)
    return best_score + second_score


@triton.jit
def _gather_four_subwarp8_roots(group_score, lane):
    source_lane = (lane & 3).to(tl.uint32) * 8
    return _routing.warp_broadcast_f32(group_score, source_lane)


@triton.jit
def _subwarp8_bfly_u32(value, J: tl.constexpr):
    if J == 1:
        return tl.inline_asm_elementwise(
            "shfl.sync.bfly.b32 $0, $1, 1, 0x1807, 0xffffffff;",
            "=r,r",
            [value.to(tl.uint32)],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
    if J == 2:
        return tl.inline_asm_elementwise(
            "shfl.sync.bfly.b32 $0, $1, 2, 0x1807, 0xffffffff;",
            "=r,r",
            [value.to(tl.uint32)],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
    return tl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0, $1, 4, 0x1807, 0xffffffff;",
        "=r,r",
        [value.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bitonic_group_stage(
    score_key,
    group_id,
    lane_in_segment,
    J: tl.constexpr,
    K: tl.constexpr,
):
    peer_key = _subwarp8_bfly_u32(score_key, J)
    peer_group = _subwarp8_bfly_u32(group_id, J)
    peer_better = (peer_key > score_key) | (
        (peer_key == score_key) & (peer_group < group_id)
    )
    lane_low = (lane_in_segment & J) == 0
    ascending = (lane_in_segment & K) != 0
    keep_better = lane_low != ascending
    take_peer = tl.where(keep_better, peer_better, peer_better == 0)
    return (
        tl.where(take_peer, peer_key, score_key),
        tl.where(take_peer, peer_group, group_id),
    )


@triton.jit
def _stable_sort8_groups(score_key, group_id, lane_in_segment):
    score_key, group_id = _bitonic_group_stage(
        score_key, group_id, lane_in_segment, 1, 2
    )
    score_key, group_id = _bitonic_group_stage(
        score_key, group_id, lane_in_segment, 2, 4
    )
    score_key, group_id = _bitonic_group_stage(
        score_key, group_id, lane_in_segment, 1, 4
    )
    score_key, group_id = _bitonic_group_stage(
        score_key, group_id, lane_in_segment, 4, 8
    )
    score_key, group_id = _bitonic_group_stage(
        score_key, group_id, lane_in_segment, 2, 8
    )
    return _bitonic_group_stage(score_key, group_id, lane_in_segment, 1, 8)


@triton.jit
def _load_warp_local_ranked_wave(
    logits_ptr,
    bias_ptr,
    score_row,
    lane,
    valid_token,
    WAVE_BASE: tl.constexpr,
):
    """Load one E128 wave while preserving the BF16 producer boundary.

    Each lane owns four consecutive experts.  The precise sigmoid values are
    kept separately from ``unbiased + bias`` because correction bias affects
    selection only; normalized output weights must use the unbiased values.
    """

    expert_base = WAVE_BASE + lane * 4
    e0 = expert_base + 0
    e1 = expert_base + 1
    e2 = expert_base + 2
    e3 = expert_base + 3
    raw0 = (
        tl.load(
            logits_ptr + score_row + e0,
            mask=valid_token,
            other=0.0,
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    raw1 = (
        tl.load(
            logits_ptr + score_row + e1,
            mask=valid_token,
            other=0.0,
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    raw2 = (
        tl.load(
            logits_ptr + score_row + e2,
            mask=valid_token,
            other=0.0,
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    raw3 = (
        tl.load(
            logits_ptr + score_row + e3,
            mask=valid_token,
            other=0.0,
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )

    unbiased0 = _precise_sigmoid(raw0)
    unbiased1 = _precise_sigmoid(raw1)
    unbiased2 = _precise_sigmoid(raw2)
    unbiased3 = _precise_sigmoid(raw3)
    score0 = unbiased0 + tl.load(bias_ptr + e0).to(tl.float32)
    score1 = unbiased1 + tl.load(bias_ptr + e1).to(tl.float32)
    score2 = unbiased2 + tl.load(bias_ptr + e2).to(tl.float32)
    score3 = unbiased3 + tl.load(bias_ptr + e3).to(tl.float32)
    score0 = tl.where(valid_token & (score0 == score0), score0, -float("inf"))
    score1 = tl.where(valid_token & (score1 == score1), score1, -float("inf"))
    score2 = tl.where(valid_token & (score2 == score2), score2, -float("inf"))
    score3 = tl.where(valid_token & (score3 == score3), score3, -float("inf"))
    score0, score1, score2, score3, permutation = _sort4_pack(
        score0, score1, score2, score3
    )
    return (
        score0,
        score1,
        score2,
        score3,
        permutation,
        unbiased0,
        unbiased1,
        unbiased2,
        unbiased3,
    )


@triton.jit
def _warp_local_group_is_selected(sorted_group, group, lane):
    """Return whether ``group`` belongs to stable ranks 0..3."""

    segment_base = (lane // 8) * 8
    selected = tl.zeros((128,), dtype=tl.int1)
    for rank in tl.static_range(0, 4):
        selected_group = _subwarp8_broadcast_f32(
            sorted_group.to(tl.float32), segment_base + rank
        ).to(tl.int32)
        selected |= group == selected_group
    return selected


@triton.jit
def route_grouped_sigmoid_warp_rows(
    logits_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    token_base,
    TOKENS,
    BLOCK_TOKENS: tl.constexpr,
    ROLE_ID: tl.constexpr,
    TOTAL_ROLES: tl.constexpr,
    RELOAD_SELECTED: tl.constexpr,
):
    """Route one token row per physical warp across all role-owned rounds.

    ``TOKENS`` is deliberately a runtime scalar so exact shapes sharing one
    geometry can reuse a cubin. ``BLOCK_TOKENS`` and ``TOTAL_ROLES`` remain
    compile-time scheduling constants. ``ROLE_ID`` is a compile-time role
    partition. A 128-thread role owns four physical warps, hence row ownership
    is

    ``ROLE_ID * 4 + routing_warp + round * (TOTAL_ROLES * 4)``.

    The selector implements the DSV3.2 contract exactly: BF16 logits are
    converted back to FP32, bias participates only in ranking, each E32
    group's best two scores are summed, and the stable best four groups
    survive. Existing v371/v372/v373 leaves retain their cached-unbiased
    normalization path.  v435 sets ``RELOAD_SELECTED`` and instead reloads the
    eight selected BF16 logits before normalization, matching CUDA's
    non-cached score path while shortening the selector's register live range.
    """

    route = tl.arange(0, 128)
    routing_warp = route // 32
    lane = route % 32
    tokens_per_round: tl.constexpr = TOTAL_ROLES * 4
    num_route_rounds: tl.constexpr = (
        BLOCK_TOKENS + tokens_per_round - 1
    ) // tokens_per_round

    for owner_round in tl.range(0, num_route_rounds, loop_unroll_factor=1):
        local_token = ROLE_ID * 4 + routing_warp + owner_round * tokens_per_round
        token = token_base + local_token
        valid_token = (local_token < BLOCK_TOKENS) & (token < TOKENS)
        safe_token = tl.where(valid_token, token, 0)
        score_row = safe_token * JIT_EXPERTS

        (
            s00,
            s01,
            s02,
            s03,
            permutation0,
            u00,
            u01,
            u02,
            u03,
        ) = _load_warp_local_ranked_wave(
            logits_ptr, bias_ptr, score_row, lane, valid_token, 0
        )
        (
            s10,
            s11,
            s12,
            s13,
            permutation1,
            u10,
            u11,
            u12,
            u13,
        ) = _load_warp_local_ranked_wave(
            logits_ptr, bias_ptr, score_row, lane, valid_token, 128
        )

        wave0_group_score = _parallel_subwarp8_group_top2_sum_packed(
            s00, s01, permutation0, lane, 0
        )
        wave1_group_score = _parallel_subwarp8_group_top2_sum_packed(
            s10, s11, permutation1, lane, 128
        )
        gathered0 = _gather_four_subwarp8_roots(wave0_group_score, lane)
        gathered1 = _gather_four_subwarp8_roots(wave1_group_score, lane)

        # Replicate the same eight group records in each width-eight segment;
        # the local network sorts all four segments in six bitonic stages.
        lane_in_segment = lane & 7
        group_score = tl.where(lane_in_segment < 4, gathered0, gathered1)
        group_id = lane_in_segment.to(tl.uint32)
        group_key = _routing.ordered_score_key(group_score).to(tl.uint32)
        _, sorted_group = _stable_sort8_groups(group_key, group_id, lane_in_segment)
        group0 = lane // 8
        group1 = group0 + 4
        wave0_selected = _warp_local_group_is_selected(sorted_group, group0, lane)
        wave1_selected = _warp_local_group_is_selected(sorted_group, group1, lane)
        s00 = tl.where(wave0_selected, s00, -float("inf"))
        s01 = tl.where(wave0_selected, s01, -float("inf"))
        s02 = tl.where(wave0_selected, s02, -float("inf"))
        s03 = tl.where(wave0_selected, s03, -float("inf"))
        s10 = tl.where(wave1_selected, s10, -float("inf"))
        s11 = tl.where(wave1_selected, s11, -float("inf"))
        s12 = tl.where(wave1_selected, s12, -float("inf"))
        s13 = tl.where(wave1_selected, s13, -float("inf"))

        one = tl.full((128,), 1, dtype=tl.uint32)
        invalid_expert = tl.full((128,), 0xFFFFFFFF, dtype=tl.uint32)
        packed_cursors = tl.zeros((128,), dtype=tl.uint32)
        selected_for_lane = tl.zeros((128,), dtype=tl.int32)
        selected_unbiased_for_lane = tl.zeros((128,), dtype=tl.float32)
        lane_expert_base = lane.to(tl.uint32) * 4
        for output_idx in tl.static_range(0, JIT_TOPK):
            cursor0 = packed_cursors & 7
            cursor1 = (packed_cursors >> 3) & 7
            head0 = _sorted_head(cursor0, s00, s01, s02, s03)
            head1 = _sorted_head(cursor1, s10, s11, s12, s13)
            offset0 = (permutation0 >> (cursor0 * 2)) & 3
            offset1 = (permutation1 >> (cursor1 * 2)) & 3
            expert0 = lane_expert_base + offset0
            expert1 = lane_expert_base + offset1 + 128
            # Wave 0 owns lower expert IDs, so keeping it on an exact score tie
            # is the required stable expert-index ordering.
            take1 = head1 > head0
            local_score = tl.where(take1, head1, head0)
            local_expert = tl.where(take1, expert1, expert0)
            local_wave = tl.where(take1, 1, 0).to(tl.uint32)
            if not RELOAD_SELECTED:
                local_unbiased0 = tl.where(
                    offset0 == 0,
                    u00,
                    tl.where(offset0 == 1, u01, tl.where(offset0 == 2, u02, u03)),
                )
                local_unbiased1 = tl.where(
                    offset1 == 0,
                    u10,
                    tl.where(offset1 == 1, u11, tl.where(offset1 == 2, u12, u13)),
                )
                local_unbiased = tl.where(take1, local_unbiased1, local_unbiased0)
            local_key = _routing.ordered_score_key(local_score).to(tl.uint32)
            best_key = _routing.warp_max_u32(local_key)
            best_expert = _routing.warp_min_u32(
                tl.where(
                    local_key == best_key,
                    local_expert,
                    invalid_expert,
                )
            )
            owns_best = (local_key == best_key) & (local_expert == best_expert)
            packed_cursors += tl.where(
                owns_best,
                one << (local_wave * 3),
                tl.zeros((128,), dtype=tl.uint32),
            )
            selected_for_lane = tl.where(
                lane == output_idx,
                best_expert.to(tl.int32),
                selected_for_lane,
            )
            if not RELOAD_SELECTED:
                best_owner_lane = (best_expert & 127) // 4
                best_unbiased = _routing.warp_broadcast_f32(
                    local_unbiased, best_owner_lane
                )
                selected_unbiased_for_lane = tl.where(
                    lane == output_idx,
                    best_unbiased,
                    selected_unbiased_for_lane,
                )

        valid_output = valid_token & (lane < JIT_TOPK)
        if RELOAD_SELECTED:
            # CUDA-v021's non-cached path keeps only expert IDs live across
            # top-k, then reloads the selected BF16 producer values.  The
            # producer boundary makes this exactly equivalent to the cached
            # normalization path while reducing selector live ranges.
            selected_expert = tl.where(lane < JIT_TOPK, selected_for_lane, 0)
            selected_raw = (
                tl.load(
                    logits_ptr + score_row + selected_expert,
                    mask=valid_output,
                    other=0.0,
                )
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            selected_unbiased = _precise_sigmoid(selected_raw)
            selected_unbiased = tl.where(valid_output, selected_unbiased, 0.0)
        else:
            selected_unbiased = tl.where(valid_output, selected_unbiased_for_lane, 0.0)
        denominator = _routing.subwarp8_sum_f32(selected_unbiased)
        denominator = (
            _routing.warp_broadcast_f32(
                denominator,
                tl.zeros((128,), dtype=tl.uint32),
            )
            + 1.0e-20
        )
        output_offset = safe_token * JIT_TOPK + lane
        tl.store(
            topk_indices_ptr + output_offset,
            selected_for_lane,
            mask=valid_output,
        )
        tl.store(
            topk_weights_ptr + output_offset,
            selected_unbiased * (routed_scaling_factor / denominator),
            mask=valid_output,
        )


__all__ = [
    "EXPERTS",
    "HIDDEN",
    "ROUTED_SCALING_FACTOR",
    "STATE_WORDS_PER_TILE",
    "TICKET_WORD",
    "TOPK",
    "route_grouped_sigmoid_warp_rows",
]
