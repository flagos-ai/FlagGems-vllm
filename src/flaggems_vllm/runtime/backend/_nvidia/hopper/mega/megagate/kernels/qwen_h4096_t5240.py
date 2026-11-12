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

"""Flattened exact-shape Qwen H4096 MegaGate TLE leaf."""

from __future__ import annotations

import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language.extra import libdevice

from .qwen_h4096_primitives import load_score_float4 as v370__load_score_float4
from .qwen_h4096_primitives import (
    quadlane4_broadcast_f32 as v368__quadlane4_broadcast_f32,
)
from .qwen_h4096_primitives import (
    quadlane4_stable_best_expert as v368__quadlane4_stable_best_expert,
)
from .qwen_h4096_primitives import quadlane4_sum_f32 as v368__quadlane4_sum_f32
from .qwen_h4096_primitives import sort4_pack as v166__sort4_pack
from .qwen_h4096_primitives import sorted_head as v166__sorted_head
from .routing_primitives import ordered_score_key as routing__ordered_score_key
from .routing_primitives import warp_broadcast_f32 as routing__warp_broadcast_f32
from .routing_primitives import warp_max_u32 as routing__warp_max_u32
from .routing_primitives import warp_min_u32 as routing__warp_min_u32

v368__HIDDEN = 4096
v368__EXPERTS = 512
v368__TOPK = 10
v368__LOCAL_TOPK = v368__TOPK
v368__BLOCK_TOKENS = 160
v368__TOKENS_N64 = 64
v368__EXPERTS_PER_CTA = 128
v368__EXPERTS_PER_MATH = 64
v368__LOGICAL_EXPERT_CTAS = 4
v368__BLOCK_K = 128
v368__NUM_K_TILES = v368__HIDDEN // v368__BLOCK_K
v368__PIPE_STAGES = 3
v368__MATH_ROLES = 4
v368__LOGICAL_ROLES = 1 + v368__MATH_ROLES
v368__THREADS_PER_ROLE = 128
v368__TOTAL_THREADS = v368__LOGICAL_ROLES * v368__THREADS_PER_ROLE
v368__MATH_MAXNREG = 104
v368__KERNEL_MAXNREG = 96
v368__STATE_WORDS_PER_TILE = 16
v368__TICKET_WORD = 4


v390__EXPERTS = v368__EXPERTS
v390__TOPK = v368__TOPK
v390__BLOCK_TOKENS = v368__BLOCK_TOKENS
v390__TOKENS_N64 = v368__TOKENS_N64
v390__ACTIVATION_MAIN_TOKENS = 128
v390__ACTIVATION_RESIDUE_TOKENS = 32
v390__EXPERTS_PER_CTA = v368__EXPERTS_PER_CTA
v390__EXPERTS_PER_MATH = v368__EXPERTS_PER_MATH
v390__LOGICAL_EXPERT_CTAS = v368__LOGICAL_EXPERT_CTAS
v390__BLOCK_K = v368__BLOCK_K
v390__NUM_K_TILES = v368__NUM_K_TILES
v390__PIPE_STAGES = v368__PIPE_STAGES
v390__TOTAL_THREADS = v368__TOTAL_THREADS
v390__MATH_MAXNREG = v368__MATH_MAXNREG
v390__KERNEL_MAXNREG = v368__KERNEL_MAXNREG
v390__STATE_WORDS_PER_TILE = v368__STATE_WORDS_PER_TILE
v390__TICKET_WORD = v368__TICKET_WORD
v390__JIT_EXPERTS = tl.constexpr(v390__EXPERTS)
v390__JIT_BLOCK_TOKENS = tl.constexpr(v390__BLOCK_TOKENS)
v390__JIT_TOKENS_N64 = tl.constexpr(v390__TOKENS_N64)
v390__JIT_ACTIVATION_MAIN_TOKENS = tl.constexpr(v390__ACTIVATION_MAIN_TOKENS)
v390__JIT_ACTIVATION_RESIDUE_TOKENS = tl.constexpr(v390__ACTIVATION_RESIDUE_TOKENS)
v390__JIT_EXPERTS_PER_CTA = tl.constexpr(v390__EXPERTS_PER_CTA)
v390__JIT_EXPERTS_PER_MATH = tl.constexpr(v390__EXPERTS_PER_MATH)
v390__JIT_LOGICAL_EXPERT_CTAS = tl.constexpr(v390__LOGICAL_EXPERT_CTAS)
v390__JIT_BLOCK_K = tl.constexpr(v390__BLOCK_K)
v390__JIT_NUM_K_TILES = tl.constexpr(v390__NUM_K_TILES)
v390__JIT_PIPE_STAGES = tl.constexpr(v390__PIPE_STAGES)
v390__JIT_TOTAL_THREADS = tl.constexpr(v390__TOTAL_THREADS)
v390__JIT_MATH_MAXNREG = tl.constexpr(v390__MATH_MAXNREG)
v390__JIT_STATE_WORDS_PER_TILE = tl.constexpr(v390__STATE_WORDS_PER_TILE)
v390__JIT_TICKET_WORD = tl.constexpr(v390__TICKET_WORD)
v390__JIT_NUM_PRODUCERS = tl.constexpr(v390__LOGICAL_EXPERT_CTAS)


@triton.jit
def v390__load_ticket_packed_b160_e128(
    writer, weight_desc, x128_desc, x32_desc, token_base, expert_base, ticket
):
    slot = writer.acquire(ticket)
    k_base = ticket * v390__JIT_BLOCK_K
    tle.gpu.copy(
        weight_desc,
        slot.weight,
        [v390__JIT_EXPERTS_PER_CTA, v390__JIT_BLOCK_K],
        [expert_base, k_base],
    )
    tle.gpu.copy(
        x128_desc,
        slot.x128,
        [v390__JIT_ACTIVATION_MAIN_TOKENS, v390__JIT_BLOCK_K],
        [token_base, k_base],
    )
    tle.gpu.copy(
        x32_desc,
        slot.x32,
        [v390__JIT_ACTIVATION_RESIDUE_TOKENS, v390__JIT_BLOCK_K],
        [token_base + v390__JIT_ACTIVATION_MAIN_TOKENS, k_base],
    )
    writer.commit(ticket)


@triton.jit
def v390__load_packed_b160_e128(
    writer, weight_desc, x128_desc, x32_desc, token_base, expert_base
):
    for ticket in tl.range(0, v390__JIT_NUM_K_TILES):
        v390__load_ticket_packed_b160_e128(
            writer, weight_desc, x128_desc, x32_desc, token_base, expert_base, ticket
        )


@triton.jit
def v390__issue_head_tail32(stage, acc64, acc32, math_id: tl.constexpr):
    weight_atom = stage.slot.weight.subslice(
        math_id * v390__JIT_EXPERTS_PER_MATH, v390__JIT_EXPERTS_PER_MATH, 0
    )
    x64_atom = stage.slot.x128.subslice(0, v390__JIT_TOKENS_N64, 0)
    acc64 = tle.gpu.wgmma(weight_atom, x64_atom, acc64, trans_b=True)
    acc32 = tle.gpu.wgmma(weight_atom, stage.slot.x32, acc32, trans_b=True)
    return (acc64, acc32)


@triton.jit
def v390__issue_middle64(stage, accumulator, math_id: tl.constexpr):
    weight_atom = stage.slot.weight.subslice(
        (math_id - 2) * v390__JIT_EXPERTS_PER_MATH,
        v390__JIT_EXPERTS_PER_MATH,
        0,
    )
    x64_atom = stage.slot.x128.subslice(v390__JIT_TOKENS_N64, v390__JIT_TOKENS_N64, 0)
    return tle.gpu.wgmma(weight_atom, x64_atom, accumulator, trans_b=True)


@triton.jit
def v390__store_head_tail32(
    acc64, acc32, logits_ptr, token_base, expert_base, tokens: tl.constexpr
):
    token64 = token_base + tl.arange(0, v390__JIT_TOKENS_N64)[None, :]
    token32 = (
        token_base
        + v390__JIT_ACTIVATION_MAIN_TOKENS
        + tl.arange(0, v390__JIT_ACTIVATION_RESIDUE_TOKENS)[None, :]
    )
    expert = expert_base + tl.arange(0, v390__JIT_EXPERTS_PER_MATH)[:, None]
    tl.store(
        logits_ptr + token64 * v390__JIT_EXPERTS + expert,
        acc64.to(tl.bfloat16).to(tl.float32),
        mask=token64 < tokens,
    )
    tl.store(
        logits_ptr + token32 * v390__JIT_EXPERTS + expert,
        acc32.to(tl.bfloat16).to(tl.float32),
        mask=token32 < tokens,
    )


@triton.jit
def v390__store_middle64(
    accumulator, logits_ptr, token_base, expert_base, tokens: tl.constexpr
):
    token = (
        token_base + v390__JIT_TOKENS_N64 + tl.arange(0, v390__JIT_TOKENS_N64)[None, :]
    )
    expert = expert_base + tl.arange(0, v390__JIT_EXPERTS_PER_MATH)[:, None]
    tl.store(
        logits_ptr + token * v390__JIT_EXPERTS + expert,
        accumulator.to(tl.bfloat16).to(tl.float32),
        mask=token < tokens,
    )


@triton.jit
def v390__gemm_head_tail32_wait0(
    reader,
    logits_ptr,
    token_base,
    expert_base,
    tokens: tl.constexpr,
    math_id: tl.constexpr,
):
    acc64 = tl.zeros(
        (v390__JIT_EXPERTS_PER_MATH, v390__JIT_TOKENS_N64), dtype=tl.float32
    )
    acc32 = tl.zeros(
        (v390__JIT_EXPERTS_PER_MATH, v390__JIT_ACTIVATION_RESIDUE_TOKENS),
        dtype=tl.float32,
    )
    for ticket in tl.range(0, v390__JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        acc64, acc32 = v390__issue_head_tail32(stage, acc64, acc32, math_id)
        acc64 = tle.gpu.wgmma_wait(0, acc64)
        acc32 = tle.gpu.wgmma_wait(0, acc32)
        reader.release(ticket)
    local_expert = expert_base + math_id * v390__JIT_EXPERTS_PER_MATH
    v390__store_head_tail32(acc64, acc32, logits_ptr, token_base, local_expert, tokens)


@triton.jit
def v390__gemm_middle64_wait0(
    reader,
    logits_ptr,
    token_base,
    expert_base,
    tokens: tl.constexpr,
    math_id: tl.constexpr,
):
    accumulator = tl.zeros(
        (v390__JIT_EXPERTS_PER_MATH, v390__JIT_TOKENS_N64), dtype=tl.float32
    )
    for ticket in tl.range(0, v390__JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        accumulator = v390__issue_middle64(stage, accumulator, math_id)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        reader.release(ticket)
    local_expert = expert_base + (math_id - 2) * v390__JIT_EXPERTS_PER_MATH
    v390__store_middle64(accumulator, logits_ptr, token_base, local_expert, tokens)


v402__TOPK = v390__TOPK
v402__LOCAL_TOPK = v368__LOCAL_TOPK
v402__KERNEL_MAXNREG = v390__KERNEL_MAXNREG
v402__COMPACT_CANDIDATE_WORDS = v402__LOCAL_TOPK
v402__JIT_EXPERTS = v390__JIT_EXPERTS
v402__JIT_TOPK = tl.constexpr(v402__TOPK)
v402__JIT_LOCAL_TOPK = tl.constexpr(v402__LOCAL_TOPK)
v402__JIT_BLOCK_TOKENS = v390__JIT_BLOCK_TOKENS
v402__JIT_ACTIVATION_MAIN_TOKENS = v390__JIT_ACTIVATION_MAIN_TOKENS
v402__JIT_ACTIVATION_RESIDUE_TOKENS = v390__JIT_ACTIVATION_RESIDUE_TOKENS
v402__JIT_EXPERTS_PER_CTA = v390__JIT_EXPERTS_PER_CTA
v402__JIT_LOGICAL_EXPERT_CTAS = v390__JIT_LOGICAL_EXPERT_CTAS
v402__JIT_BLOCK_K = v390__JIT_BLOCK_K
v402__JIT_PIPE_STAGES = v390__JIT_PIPE_STAGES
v402__JIT_TOTAL_THREADS = v390__JIT_TOTAL_THREADS
v402__JIT_MATH_MAXNREG = v390__JIT_MATH_MAXNREG
v402__JIT_STATE_WORDS_PER_TILE = v390__JIT_STATE_WORDS_PER_TILE
v402__JIT_TICKET_WORD = v390__JIT_TICKET_WORD
v402__JIT_NUM_PRODUCERS = v390__JIT_NUM_PRODUCERS
v402__JIT_COMPACT_CANDIDATE_WORDS = tl.constexpr(v402__COMPACT_CANDIDATE_WORDS)


@triton.jit
def v402__store_packed_candidate_word(ptr, score, local_expert, valid):
    """Store the v399 compact word; stores have no cache-policy qualifier."""
    address = ptr.to(tl.uint64)
    score_bits = score.to(tl.uint32, bitcast=True)
    packed = score_bits & 4294901760 | local_expert.to(tl.uint32) & 255
    _ = tl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            setp.ne.u32 p, $3, 0;
            @p st.global.u32 [$1], $2;
            mov.u32 $0, $3;
        }
        """,
        "=r,l,r,r,~{memory}",
        [address, packed, valid.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def v402__load_packed_candidate_word_l1(ptr, valid):
    """Load one compact head while retaining its 40-byte list in L1."""
    address = ptr.to(tl.uint64)
    packed = tl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            mov.u32 $0, 0xff8000ff;
            setp.ne.u32 p, $2, 0;
            @p ld.weak.global.L1::evict_last.b32 $0, [$1];
        }
        """,
        "=r,l,r,~{memory}",
        [address, valid.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    score_bits = packed & 4294901760
    return (score_bits.to(tl.float32, bitcast=True), packed & 255)


@triton.jit
def v402__publish_local_top10_b160_compact_role(
    logits_ptr,
    candidate_scratch_ptr,
    expert_base,
    token_base,
    tokens: tl.constexpr,
    ROLE_ID: tl.constexpr,
    ROUTE_BLOCK_TOKENS: tl.constexpr,
):
    """Publish CUDA-v025 local K10 for 32 B160 rows per static role."""
    route = tl.arange(0, 128)
    routing_warp = route // 32
    lane = route % 32
    lane_u32 = lane.to(tl.uint32)
    invalid_expert = tl.full((128,), 4294967295, dtype=tl.uint32)
    for owner_round in tl.range(0, 8, loop_unroll_factor=1):
        local_token = ROLE_ID * 4 + routing_warp + owner_round * 20
        token = token_base + local_token
        valid_token = (local_token < ROUTE_BLOCK_TOKENS) & (token < tokens)
        score_row = token * v402__JIT_EXPERTS
        local_expert_base = expert_base + lane * 4
        s0, s1, s2, s3 = v370__load_score_float4(
            logits_ptr + score_row + local_expert_base, valid_token
        )
        s0 = s0.to(tl.bfloat16).to(tl.float32)
        s1 = s1.to(tl.bfloat16).to(tl.float32)
        s2 = s2.to(tl.bfloat16).to(tl.float32)
        s3 = s3.to(tl.bfloat16).to(tl.float32)
        s0 = tl.where(s0 == s0, s0, -float("inf"))
        s1 = tl.where(s1 == s1, s1, -float("inf"))
        s2 = tl.where(s2 == s2, s2, -float("inf"))
        s3 = tl.where(s3 == s3, s3, -float("inf"))
        s0, s1, s2, s3, permutation = v166__sort4_pack(s0, s1, s2, s3)
        cursor = tl.zeros((128,), dtype=tl.uint32)
        selected_score = tl.zeros((128,), dtype=tl.float32)
        selected_expert = tl.zeros((128,), dtype=tl.uint32)
        for output_idx in tl.static_range(0, v402__JIT_LOCAL_TOPK):
            head = v166__sorted_head(cursor, s0, s1, s2, s3)
            offset = permutation >> cursor * 2 & 3
            local_expert = (expert_base + lane_u32 * 4 + offset).to(tl.uint32)
            local_key = routing__ordered_score_key(head).to(tl.uint32)
            best_key = routing__warp_max_u32(local_key)
            best_expert = routing__warp_min_u32(
                tl.where(local_key == best_key, local_expert, invalid_expert)
            )
            source_lane = best_expert - expert_base >> 2
            best_score = routing__warp_broadcast_f32(head, source_lane)
            cursor += (local_expert == best_expert).to(tl.uint32)
            selected_score = tl.where(lane == output_idx, best_score, selected_score)
            selected_expert = tl.where(lane == output_idx, best_expert, selected_expert)
        output_lane = lane < v402__JIT_LOCAL_TOPK
        producer = expert_base // v402__JIT_EXPERTS_PER_CTA
        candidate = (
            candidate_scratch_ptr
            + (token * v402__JIT_NUM_PRODUCERS + producer)
            * v402__JIT_COMPACT_CANDIDATE_WORDS
            + lane
        )
        v402__store_packed_candidate_word(
            candidate,
            selected_score,
            selected_expert - expert_base,
            valid_token & output_lane,
        )


@triton.jit
def v402__merge_four_local_top10_b160_compact_role(
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    token_base,
    tokens: tl.constexpr,
    ROLE_ID: tl.constexpr,
    ROUTE_BLOCK_TOKENS: tl.constexpr,
):
    """Merge four lists for eight tokens per warp, matching CUDA-v025."""
    terminal = tl.load(tle.gpu.local_ptr(terminal_smem, (0,)))
    if terminal != 0:
        route = tl.arange(0, 128)
        routing_warp = route // 32
        lane = route % 32
        token_octet = lane // 4
        subwarp_lane = lane % 4
        producer = subwarp_lane.to(tl.uint32)
        producer_lane = subwarp_lane < v402__JIT_NUM_PRODUCERS
        for owner_round in tl.range(0, 1, loop_unroll_factor=1):
            local_token = ROLE_ID * 32 + routing_warp * 8 + token_octet
            token = token_base + local_token
            valid_token = (local_token < ROUTE_BLOCK_TOKENS) & (token < tokens)
            safe_token = tl.where(valid_token, token, token_base)
            cursor = tl.zeros((128,), dtype=tl.uint32)
            selected_score0 = tl.zeros((128,), dtype=tl.float32)
            selected_score1 = tl.zeros((128,), dtype=tl.float32)
            selected_score2 = tl.zeros((128,), dtype=tl.float32)
            selected_expert0 = tl.zeros((128,), dtype=tl.uint32)
            selected_expert1 = tl.zeros((128,), dtype=tl.uint32)
            selected_expert2 = tl.zeros((128,), dtype=tl.uint32)
            candidate = (
                candidate_scratch_ptr
                + (safe_token * v402__JIT_NUM_PRODUCERS + producer)
                * v402__JIT_COMPACT_CANDIDATE_WORDS
            )
            head_score, head_local = v402__load_packed_candidate_word_l1(
                candidate, valid_token & producer_lane
            )
            head_unbiased = head_score
            head_expert = producer * v402__JIT_EXPERTS_PER_CTA + head_local
            for output_idx in tl.static_range(0, v402__JIT_TOPK):
                head_key = routing__ordered_score_key(head_score).to(tl.uint32)
                best_expert = v368__quadlane4_stable_best_expert(head_key, head_expert)
                source_list = best_expert // v402__JIT_EXPERTS_PER_CTA
                best_score = v368__quadlane4_broadcast_f32(head_unbiased, source_list)
                owns_best = head_expert == best_expert
                if output_idx < 4:
                    selected_score0 = tl.where(
                        subwarp_lane == output_idx, best_score, selected_score0
                    )
                    selected_expert0 = tl.where(
                        subwarp_lane == output_idx, best_expert, selected_expert0
                    )
                elif output_idx < 8:
                    selected_score1 = tl.where(
                        subwarp_lane == output_idx - 4, best_score, selected_score1
                    )
                    selected_expert1 = tl.where(
                        subwarp_lane == output_idx - 4, best_expert, selected_expert1
                    )
                else:
                    selected_score2 = tl.where(
                        subwarp_lane == output_idx - 8, best_score, selected_score2
                    )
                    selected_expert2 = tl.where(
                        subwarp_lane == output_idx - 8, best_expert, selected_expert2
                    )
                if output_idx + 1 < v402__JIT_TOPK:
                    cursor += owns_best.to(tl.uint32)
                    next_candidate = (
                        candidate_scratch_ptr
                        + (safe_token * v402__JIT_NUM_PRODUCERS + producer)
                        * v402__JIT_COMPACT_CANDIDATE_WORDS
                        + cursor
                    )
                    next_score, next_local = v402__load_packed_candidate_word_l1(
                        next_candidate, valid_token & owns_best
                    )
                    next_expert = producer * v402__JIT_EXPERTS_PER_CTA + next_local
                    head_score = tl.where(owns_best, next_score, head_score)
                    head_unbiased = tl.where(owns_best, next_score, head_unbiased)
                    head_expert = tl.where(owns_best, next_expert, head_expert)
            top1 = v368__quadlane4_broadcast_f32(
                selected_score0, tl.zeros((128,), dtype=tl.uint32)
            )
            has_finite = top1 != -float("inf")
            exp0 = tl.where(has_finite, libdevice.exp(selected_score0 - top1), 0.0)
            exp1 = tl.where(has_finite, libdevice.exp(selected_score1 - top1), 0.0)
            valid_rank2 = subwarp_lane < 2
            exp2 = tl.where(
                valid_rank2 & has_finite, libdevice.exp(selected_score2 - top1), 0.0
            )
            local_sum = exp0 + exp2
            local_sum += exp1
            denominator = v368__quadlane4_sum_f32(local_sum) + 1e-20
            output_base = token * v402__JIT_TOPK
            tl.store(
                topk_indices_ptr + output_base + subwarp_lane,
                selected_expert0.to(tl.int32),
                mask=valid_token,
            )
            tl.store(
                topk_weights_ptr + output_base + subwarp_lane,
                exp0 / denominator,
                mask=valid_token,
            )
            tl.store(
                topk_indices_ptr + output_base + subwarp_lane + 4,
                selected_expert1.to(tl.int32),
                mask=valid_token,
            )
            tl.store(
                topk_weights_ptr + output_base + subwarp_lane + 4,
                exp1 / denominator,
                mask=valid_token,
            )
            tl.store(
                topk_indices_ptr + output_base + subwarp_lane + 8,
                selected_expert2.to(tl.int32),
                mask=valid_token & valid_rank2,
            )
            tl.store(
                topk_weights_ptr + output_base + subwarp_lane + 8,
                exp2 / denominator,
                mask=valid_token & valid_rank2,
            )


@triton.jit
def v402__loader_publish_merge0(
    writer,
    weight_desc,
    x128_desc,
    x32_desc,
    logits_ptr,
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    logits_ready_barrier,
    candidates_ready_barrier,
    route_handoff_barrier,
    token_base,
    expert_base,
    tokens: tl.constexpr,
):
    v390__load_packed_b160_e128(
        writer, weight_desc, x128_desc, x32_desc, token_base, expert_base
    )
    tle.gpu.barrier_wait(logits_ready_barrier)
    v402__publish_local_top10_b160_compact_role(
        logits_ptr,
        candidate_scratch_ptr,
        expert_base,
        token_base,
        tokens,
        0,
        v402__JIT_BLOCK_TOKENS,
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)
    tle.gpu.barrier_wait(route_handoff_barrier)
    v402__merge_four_local_top10_b160_compact_role(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        tokens,
        0,
        v402__JIT_BLOCK_TOKENS,
    )


v428__KERNEL_MAXNREG = v402__KERNEL_MAXNREG
v428__JIT_EXPERTS = v402__JIT_EXPERTS
v428__JIT_TOPK = v402__JIT_TOPK
v428__JIT_LOCAL_TOPK = v402__JIT_LOCAL_TOPK
v428__JIT_BLOCK_TOKENS = v402__JIT_BLOCK_TOKENS
v428__JIT_ACTIVATION_MAIN_TOKENS = v402__JIT_ACTIVATION_MAIN_TOKENS
v428__JIT_ACTIVATION_RESIDUE_TOKENS = v402__JIT_ACTIVATION_RESIDUE_TOKENS
v428__JIT_EXPERTS_PER_CTA = v402__JIT_EXPERTS_PER_CTA
v428__JIT_LOGICAL_EXPERT_CTAS = v402__JIT_LOGICAL_EXPERT_CTAS
v428__JIT_BLOCK_K = v402__JIT_BLOCK_K
v428__JIT_PIPE_STAGES = v402__JIT_PIPE_STAGES
v428__JIT_TOTAL_THREADS = v402__JIT_TOTAL_THREADS
v428__JIT_MATH_MAXNREG = v402__JIT_MATH_MAXNREG
v428__JIT_STATE_WORDS_PER_TILE = v402__JIT_STATE_WORDS_PER_TILE
v428__JIT_TICKET_WORD = v402__JIT_TICKET_WORD
v428__JIT_NUM_PRODUCERS = v402__JIT_NUM_PRODUCERS
v428__JIT_COMPACT_CANDIDATE_WORDS = v402__JIT_COMPACT_CANDIDATE_WORDS


@triton.jit(noinline=True)
def v428__publish_local_top10_phase_nw4(
    logits_ptr,
    candidate_scratch_ptr,
    token_base,
    expert_base,
    role_id,
    tokens: tl.constexpr,
):
    """Publish one role's local K10; shared only by math roles 1--4."""
    route = tl.arange(0, 128)
    routing_warp = route // 32
    lane = route % 32
    lane_u32 = lane.to(tl.uint32)
    invalid_expert = tl.full((128,), 4294967295, dtype=tl.uint32)
    for owner_round in tl.range(0, 8, loop_unroll_factor=1):
        local_token = role_id * 4 + routing_warp + owner_round * 20
        token = token_base + local_token
        valid_token = (local_token < v428__JIT_BLOCK_TOKENS) & (token < tokens)
        score_row = token * v428__JIT_EXPERTS
        local_expert_base = expert_base + lane * 4
        s0, s1, s2, s3 = v370__load_score_float4(
            logits_ptr + score_row + local_expert_base, valid_token
        )
        s0 = s0.to(tl.bfloat16).to(tl.float32)
        s1 = s1.to(tl.bfloat16).to(tl.float32)
        s2 = s2.to(tl.bfloat16).to(tl.float32)
        s3 = s3.to(tl.bfloat16).to(tl.float32)
        s0 = tl.where(s0 == s0, s0, -float("inf"))
        s1 = tl.where(s1 == s1, s1, -float("inf"))
        s2 = tl.where(s2 == s2, s2, -float("inf"))
        s3 = tl.where(s3 == s3, s3, -float("inf"))
        s0, s1, s2, s3, permutation = v166__sort4_pack(s0, s1, s2, s3)
        cursor = tl.zeros((128,), dtype=tl.uint32)
        selected_score = tl.zeros((128,), dtype=tl.float32)
        selected_expert = tl.zeros((128,), dtype=tl.uint32)
        for output_idx in tl.range(0, v428__JIT_LOCAL_TOPK, loop_unroll_factor=1):
            head = v166__sorted_head(cursor, s0, s1, s2, s3)
            offset = permutation >> cursor * 2 & 3
            local_expert = (expert_base + lane_u32 * 4 + offset).to(tl.uint32)
            local_key = routing__ordered_score_key(head).to(tl.uint32)
            best_key = routing__warp_max_u32(local_key)
            best_expert = routing__warp_min_u32(
                tl.where(local_key == best_key, local_expert, invalid_expert)
            )
            source_lane = best_expert - expert_base >> 2
            best_score = routing__warp_broadcast_f32(head, source_lane)
            cursor += (local_expert == best_expert).to(tl.uint32)
            selected_score = tl.where(lane == output_idx, best_score, selected_score)
            selected_expert = tl.where(lane == output_idx, best_expert, selected_expert)
        output_lane = lane < v428__JIT_LOCAL_TOPK
        producer = expert_base // v428__JIT_EXPERTS_PER_CTA
        candidate = (
            candidate_scratch_ptr
            + (token * v428__JIT_NUM_PRODUCERS + producer)
            * v428__JIT_COMPACT_CANDIDATE_WORDS
            + lane
        )
        v402__store_packed_candidate_word(
            candidate,
            selected_score,
            selected_expert - expert_base,
            valid_token & output_lane,
        )


@triton.jit(noinline=True)
def v428__terminal_merge_phase_shared_nw4(
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    token_base,
    role_id,
    tokens: tl.constexpr,
):
    """Merge four compact K10 lists; shared only by math roles 1--4."""
    terminal = tl.load(tle.gpu.local_ptr(terminal_smem, (0,)))
    if terminal != 0:
        route = tl.arange(0, 128)
        routing_warp = route // 32
        lane = route % 32
        token_octet = lane // 4
        subwarp_lane = lane % 4
        producer = subwarp_lane.to(tl.uint32)
        producer_lane = subwarp_lane < v428__JIT_NUM_PRODUCERS
        local_token = role_id * 32 + routing_warp * 8 + token_octet
        token = token_base + local_token
        valid_token = (local_token < v428__JIT_BLOCK_TOKENS) & (token < tokens)
        safe_token = tl.where(valid_token, token, token_base)
        cursor = tl.zeros((128,), dtype=tl.uint32)
        selected_score0 = tl.zeros((128,), dtype=tl.float32)
        selected_score1 = tl.zeros((128,), dtype=tl.float32)
        selected_score2 = tl.zeros((128,), dtype=tl.float32)
        selected_expert0 = tl.zeros((128,), dtype=tl.uint32)
        selected_expert1 = tl.zeros((128,), dtype=tl.uint32)
        selected_expert2 = tl.zeros((128,), dtype=tl.uint32)
        candidate = (
            candidate_scratch_ptr
            + (safe_token * v428__JIT_NUM_PRODUCERS + producer)
            * v428__JIT_COMPACT_CANDIDATE_WORDS
        )
        head_score, head_local = v402__load_packed_candidate_word_l1(
            candidate, valid_token & producer_lane
        )
        head_unbiased = head_score
        head_expert = producer * v428__JIT_EXPERTS_PER_CTA + head_local
        for output_idx in tl.range(0, v428__JIT_TOPK, loop_unroll_factor=1):
            head_key = routing__ordered_score_key(head_score).to(tl.uint32)
            best_expert = v368__quadlane4_stable_best_expert(head_key, head_expert)
            source_list = best_expert // v428__JIT_EXPERTS_PER_CTA
            best_score = v368__quadlane4_broadcast_f32(head_unbiased, source_list)
            owns_best = head_expert == best_expert
            selected_score0 = tl.where(
                (output_idx < 4) & (subwarp_lane == output_idx),
                best_score,
                selected_score0,
            )
            selected_expert0 = tl.where(
                (output_idx < 4) & (subwarp_lane == output_idx),
                best_expert,
                selected_expert0,
            )
            selected_score1 = tl.where(
                (output_idx >= 4) & (output_idx < 8) & (subwarp_lane == output_idx - 4),
                best_score,
                selected_score1,
            )
            selected_expert1 = tl.where(
                (output_idx >= 4) & (output_idx < 8) & (subwarp_lane == output_idx - 4),
                best_expert,
                selected_expert1,
            )
            selected_score2 = tl.where(
                (output_idx >= 8) & (subwarp_lane == output_idx - 8),
                best_score,
                selected_score2,
            )
            selected_expert2 = tl.where(
                (output_idx >= 8) & (subwarp_lane == output_idx - 8),
                best_expert,
                selected_expert2,
            )
            has_next = output_idx + 1 < v428__JIT_TOPK
            cursor += (has_next & owns_best).to(tl.uint32)
            next_candidate = (
                candidate_scratch_ptr
                + (safe_token * v428__JIT_NUM_PRODUCERS + producer)
                * v428__JIT_COMPACT_CANDIDATE_WORDS
                + cursor
            )
            next_score, next_local = v402__load_packed_candidate_word_l1(
                next_candidate, valid_token & has_next & owns_best
            )
            next_expert = producer * v428__JIT_EXPERTS_PER_CTA + next_local
            update = has_next & owns_best
            head_score = tl.where(update, next_score, head_score)
            head_unbiased = tl.where(update, next_score, head_unbiased)
            head_expert = tl.where(update, next_expert, head_expert)
        top1 = v368__quadlane4_broadcast_f32(
            selected_score0, tl.zeros((128,), dtype=tl.uint32)
        )
        has_finite = top1 != -float("inf")
        exp0 = tl.where(has_finite, libdevice.exp(selected_score0 - top1), 0.0)
        exp1 = tl.where(has_finite, libdevice.exp(selected_score1 - top1), 0.0)
        valid_rank2 = subwarp_lane < 2
        exp2 = tl.where(
            valid_rank2 & has_finite, libdevice.exp(selected_score2 - top1), 0.0
        )
        local_sum = exp0 + exp2
        local_sum += exp1
        denominator = v368__quadlane4_sum_f32(local_sum) + 1e-20
        output_base = token * v428__JIT_TOPK
        tl.store(
            topk_indices_ptr + output_base + subwarp_lane,
            selected_expert0.to(tl.int32),
            mask=valid_token,
        )
        tl.store(
            topk_weights_ptr + output_base + subwarp_lane,
            exp0 / denominator,
            mask=valid_token,
        )
        tl.store(
            topk_indices_ptr + output_base + subwarp_lane + 4,
            selected_expert1.to(tl.int32),
            mask=valid_token,
        )
        tl.store(
            topk_weights_ptr + output_base + subwarp_lane + 4,
            exp1 / denominator,
            mask=valid_token,
        )
        tl.store(
            topk_indices_ptr + output_base + subwarp_lane + 8,
            selected_expert2.to(tl.int32),
            mask=valid_token & valid_rank2,
        )
        tl.store(
            topk_weights_ptr + output_base + subwarp_lane + 8,
            exp2 / denominator,
            mask=valid_token & valid_rank2,
        )


v431__KERNEL_MAXNREG = v428__KERNEL_MAXNREG
v431__JIT_BLOCK_TOKENS = v428__JIT_BLOCK_TOKENS
v431__JIT_ACTIVATION_MAIN_TOKENS = v428__JIT_ACTIVATION_MAIN_TOKENS
v431__JIT_ACTIVATION_RESIDUE_TOKENS = v428__JIT_ACTIVATION_RESIDUE_TOKENS
v431__JIT_EXPERTS_PER_CTA = v428__JIT_EXPERTS_PER_CTA
v431__JIT_LOGICAL_EXPERT_CTAS = v428__JIT_LOGICAL_EXPERT_CTAS
v431__JIT_BLOCK_K = v428__JIT_BLOCK_K
v431__JIT_PIPE_STAGES = v428__JIT_PIPE_STAGES
v431__JIT_TOTAL_THREADS = v428__JIT_TOTAL_THREADS
v431__JIT_MATH_MAXNREG = v428__JIT_MATH_MAXNREG
v431__JIT_STATE_WORDS_PER_TILE = v428__JIT_STATE_WORDS_PER_TILE
v431__JIT_TICKET_WORD = v428__JIT_TICKET_WORD
v431__JIT_NUM_PRODUCERS = v428__JIT_NUM_PRODUCERS


@triton.jit
def v431__math_publish_arrive_merge_static_last(
    reader,
    logits_ptr,
    candidate_scratch_ptr,
    state_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    logits_ready_barrier,
    candidates_ready_barrier,
    route_handoff_barrier,
    token_base,
    expert_base,
    token_tile,
    tokens: tl.constexpr,
    MATH_ID: tl.constexpr,
    ROLE_ID: tl.constexpr,
):
    if MATH_ID < 2:
        v390__gemm_head_tail32_wait0(
            reader, logits_ptr, token_base, expert_base, tokens, MATH_ID
        )
    else:
        v390__gemm_middle64_wait0(
            reader, logits_ptr, token_base, expert_base, tokens, MATH_ID
        )
    runtime_zero = tl.program_id(0).to(tl.int32) * 0
    runtime_role_id = runtime_zero + ROLE_ID
    tle.gpu.barrier_wait(logits_ready_barrier)
    v428__publish_local_top10_phase_nw4(
        logits_ptr,
        candidate_scratch_ptr,
        token_base,
        expert_base,
        runtime_role_id,
        tokens,
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)
    if MATH_ID == 3:
        ticket_ptr = (
            state_ptr
            + token_tile * v431__JIT_STATE_WORDS_PER_TILE
            + v431__JIT_TICKET_WORD
        )
        old_ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="gpu")
        terminal = old_ticket % v431__JIT_NUM_PRODUCERS + 1 == v431__JIT_NUM_PRODUCERS
        tl.store(tle.gpu.local_ptr(terminal_smem, (0,)), terminal.to(tl.int32))
    tle.gpu.barrier_wait(route_handoff_barrier)
    v428__terminal_merge_phase_shared_nw4(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        runtime_role_id,
        tokens,
    )


@triton.jit
def v431__fused_qwen_h4096_b160_e128_static_last_kernel(
    weight_desc,
    x128_desc,
    x32_desc,
    logits_ptr,
    candidate_scratch_ptr,
    state_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    tokens: tl.constexpr,
):
    pid = tl.program_id(0)
    logical_rank = pid % v431__JIT_LOGICAL_EXPERT_CTAS
    token_tile = pid // v431__JIT_LOGICAL_EXPERT_CTAS
    token_base = token_tile * v431__JIT_BLOCK_TOKENS
    expert_base = logical_rank * v431__JIT_EXPERTS_PER_CTA
    weight = tle.gpu.alloc(
        [v431__JIT_PIPE_STAGES, v431__JIT_EXPERTS_PER_CTA, v431__JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    x128 = tle.gpu.alloc(
        [v431__JIT_PIPE_STAGES, v431__JIT_ACTIVATION_MAIN_TOKENS, v431__JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    x32 = tle.gpu.alloc(
        [v431__JIT_PIPE_STAGES, v431__JIT_ACTIVATION_RESIDUE_TOKENS, v431__JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    pipe = tle.pipe(
        capacity=v431__JIT_PIPE_STAGES,
        scope="cta",
        name="qwen_h4096_t5240_b160_e128_static_last_v431",
        readers=("math0", "math1", "math2", "math3"),
        weight=weight,
        x128=x128,
        x32=x32,
    )
    terminal_smem = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    logits_ready_barrier = tle.gpu.alloc_barrier(arrive_count=v431__JIT_TOTAL_THREADS)
    candidates_ready_barrier = tle.gpu.alloc_barrier(
        arrive_count=v431__JIT_TOTAL_THREADS
    )
    route_handoff_barrier = tle.gpu.alloc_barrier(arrive_count=v431__JIT_TOTAL_THREADS)
    tle.gpu.warp_specialize(
        [
            (
                v402__loader_publish_merge0,
                (
                    pipe.writer(),
                    weight_desc,
                    x128_desc,
                    x32_desc,
                    logits_ptr,
                    candidate_scratch_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    terminal_smem,
                    logits_ready_barrier,
                    candidates_ready_barrier,
                    route_handoff_barrier,
                    token_base,
                    expert_base,
                    tokens,
                ),
            ),
            (
                v431__math_publish_arrive_merge_static_last,
                (
                    pipe.reader(name="math0", fields=("weight", "x128", "x32")),
                    logits_ptr,
                    candidate_scratch_ptr,
                    state_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    terminal_smem,
                    logits_ready_barrier,
                    candidates_ready_barrier,
                    route_handoff_barrier,
                    token_base,
                    expert_base,
                    token_tile,
                    tokens,
                    0,
                    1,
                ),
            ),
            (
                v431__math_publish_arrive_merge_static_last,
                (
                    pipe.reader(name="math1", fields=("weight", "x128", "x32")),
                    logits_ptr,
                    candidate_scratch_ptr,
                    state_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    terminal_smem,
                    logits_ready_barrier,
                    candidates_ready_barrier,
                    route_handoff_barrier,
                    token_base,
                    expert_base,
                    token_tile,
                    tokens,
                    1,
                    2,
                ),
            ),
            (
                v431__math_publish_arrive_merge_static_last,
                (
                    pipe.reader(name="math2", fields=("weight", "x128")),
                    logits_ptr,
                    candidate_scratch_ptr,
                    state_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    terminal_smem,
                    logits_ready_barrier,
                    candidates_ready_barrier,
                    route_handoff_barrier,
                    token_base,
                    expert_base,
                    token_tile,
                    tokens,
                    2,
                    3,
                ),
            ),
            (
                v431__math_publish_arrive_merge_static_last,
                (
                    pipe.reader(name="math3", fields=("weight", "x128")),
                    logits_ptr,
                    candidate_scratch_ptr,
                    state_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    terminal_smem,
                    logits_ready_barrier,
                    candidates_ready_barrier,
                    route_handoff_barrier,
                    token_base,
                    expert_base,
                    token_tile,
                    tokens,
                    3,
                    4,
                ),
            ),
        ],
        [4, 4, 4, 4],
        [
            v431__JIT_MATH_MAXNREG,
            v431__JIT_MATH_MAXNREG,
            v431__JIT_MATH_MAXNREG,
            v431__JIT_MATH_MAXNREG,
        ],
    )


KERNEL = v431__fused_qwen_h4096_b160_e128_static_last_kernel
MAXNREG = v431__KERNEL_MAXNREG
SOURCE_LEAF = "v431"
__all__ = ["KERNEL", "MAXNREG", "SOURCE_LEAF"]
