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
from .qwen_h4096_primitives import pair2_broadcast_f32 as v369__pair2_broadcast_f32
from .qwen_h4096_primitives import pair2_max_u32 as v388__pair2_max_u32
from .qwen_h4096_primitives import pair2_sum_f32 as v369__pair2_sum_f32
from .qwen_h4096_primitives import sort4_pack as v166__sort4_pack
from .qwen_h4096_primitives import sorted_head as v166__sorted_head
from .routing_primitives import ordered_score_key as routing__ordered_score_key
from .routing_primitives import warp_broadcast_f32 as routing__warp_broadcast_f32
from .routing_primitives import warp_max_u32 as routing__warp_max_u32

v369__HIDDEN = 4096
v369__EXPERTS = 512
v369__TOPK = 10
v369__LOCAL_TOPK = v369__TOPK
v369__BLOCK_TOKENS = 128
v369__EXPERTS_PER_CTA = 256
v369__EXPERTS_PER_MATH = 64
v369__LOGICAL_EXPERT_CTAS = 2
v369__BLOCK_K = 128
v369__NUM_K_TILES = v369__HIDDEN // v369__BLOCK_K
v369__PIPE_STAGES = 2
v369__MATH_ROLES = 4
v369__LOGICAL_ROLES = 1 + v369__MATH_ROLES
v369__THREADS_PER_ROLE = 128
v369__TOTAL_THREADS = v369__LOGICAL_ROLES * v369__THREADS_PER_ROLE
v369__MATH_MAXNREG = 112
v369__KERNEL_MAXNREG = 96
v369__STATE_WORDS_PER_TILE = 16
v369__TICKET_WORD = 4
v369__JIT_EXPERTS = tl.constexpr(v369__EXPERTS)
v369__JIT_BLOCK_TOKENS = tl.constexpr(v369__BLOCK_TOKENS)
v369__JIT_EXPERTS_PER_CTA = tl.constexpr(v369__EXPERTS_PER_CTA)
v369__JIT_EXPERTS_PER_MATH = tl.constexpr(v369__EXPERTS_PER_MATH)
v369__JIT_BLOCK_K = tl.constexpr(v369__BLOCK_K)
v369__JIT_NUM_K_TILES = tl.constexpr(v369__NUM_K_TILES)


@triton.jit
def v369__load_ticket_b128_e256(
    writer, weight_desc, x_desc, token_base, expert_base, ticket
):
    slot = writer.acquire(ticket)
    k_base = ticket * v369__JIT_BLOCK_K
    tle.gpu.copy(
        weight_desc,
        slot.weight,
        [v369__JIT_EXPERTS_PER_CTA, v369__JIT_BLOCK_K],
        [expert_base, k_base],
    )
    tle.gpu.copy(
        x_desc,
        slot.x,
        [v369__JIT_BLOCK_TOKENS, v369__JIT_BLOCK_K],
        [token_base, k_base],
    )
    writer.commit(ticket)


@triton.jit
def v369__load_b128_e256(writer, weight_desc, x_desc, token_base, expert_base):
    for ticket in tl.range(0, v369__JIT_NUM_K_TILES):
        v369__load_ticket_b128_e256(
            writer, weight_desc, x_desc, token_base, expert_base, ticket
        )


@triton.jit
def v369__issue_b128(stage, accumulator, MATH_ID: tl.constexpr):
    weight_atom = stage.slot.weight.subslice(
        MATH_ID * v369__JIT_EXPERTS_PER_MATH, v369__JIT_EXPERTS_PER_MATH, 0
    )
    accumulator = tle.gpu.wgmma(weight_atom, stage.slot.x, accumulator, trans_b=True)
    return accumulator


@triton.jit
def v369__store_atom(accumulator, logits_ptr, token_base, expert_base, tokens):
    token = token_base + tl.arange(0, v369__JIT_BLOCK_TOKENS)[None, :]
    expert = expert_base + tl.arange(0, v369__JIT_EXPERTS_PER_MATH)[:, None]
    bounded = accumulator.to(tl.bfloat16).to(tl.float32)
    tl.store(
        logits_ptr + token * v369__JIT_EXPERTS + expert, bounded, mask=token < tokens
    )


@triton.jit
def v369__gemm_b128(
    reader, logits_ptr, token_base, expert_base, tokens, MATH_ID: tl.constexpr
):
    accumulator = tl.zeros(
        (v369__JIT_EXPERTS_PER_MATH, v369__JIT_BLOCK_TOKENS), dtype=tl.float32
    )
    for ticket in tl.range(0, v369__JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        accumulator = v369__issue_b128(stage, accumulator, MATH_ID)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        reader.release(ticket)
    v369__store_atom(accumulator, logits_ptr, token_base, expert_base, tokens)


@triton.jit
def v369__loader_route0(
    writer,
    weight_desc,
    x_desc,
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
    v369__load_b128_e256(writer, weight_desc, x_desc, token_base, expert_base)
    tle.gpu.barrier_wait(logits_ready_barrier)
    tle.gpu.barrier_wait(candidates_ready_barrier)
    tle.gpu.barrier_wait(route_handoff_barrier)


@triton.jit
def v388__packed_global_rank(score, expert):
    score_rank = routing__ordered_score_key(score).to(tl.uint32) >> 16
    return score_rank << 9 | 511 - expert.to(tl.uint32)


@triton.jit
def v388__store_packed_candidate_word(ptr, score, local_expert, valid):
    address = ptr.to(tl.uint64)
    score_bits = score.to(tl.uint32, bitcast=True)
    packed = score_bits & 4294901760 | local_expert.to(tl.uint32) & 255
    _ = tl.inline_asm_elementwise(
        "\n        {\n            .reg .pred p;\n            setp.ne.u32 p, $3, 0;\n            @p st.global.u32 [$1], $2;\n            mov.u32 $0, $3;\n        }\n        ",
        "=r,l,r,r,~{memory}",
        [address, packed, valid.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


v391__EXPERTS = v369__EXPERTS
v391__TOPK = v369__TOPK
v391__LOCAL_TOPK = v369__LOCAL_TOPK
v391__BLOCK_TOKENS = v369__BLOCK_TOKENS
v391__EXPERTS_PER_CTA = v369__EXPERTS_PER_CTA
v391__EXPERTS_PER_MATH = v369__EXPERTS_PER_MATH
v391__LOGICAL_EXPERT_CTAS = v369__LOGICAL_EXPERT_CTAS
v391__BLOCK_K = v369__BLOCK_K
v391__PIPE_STAGES = v369__PIPE_STAGES
v391__TOTAL_THREADS = v369__TOTAL_THREADS
v391__KERNEL_MAXNREG = v369__KERNEL_MAXNREG
v391__MATH_MAXNREG = v369__MATH_MAXNREG
v391__STATE_WORDS_PER_TILE = v369__STATE_WORDS_PER_TILE
v391__TICKET_WORD = v369__TICKET_WORD
v391__ROUTE_ROUNDS = 8
v391__TERMINAL_WARPS = 8
v391__COMPACT_CANDIDATE_WORDS = v391__LOCAL_TOPK
v391__JIT_EXPERTS = tl.constexpr(v391__EXPERTS)
v391__JIT_LOCAL_TOPK = tl.constexpr(v391__LOCAL_TOPK)
v391__JIT_EXPERTS_PER_CTA = tl.constexpr(v391__EXPERTS_PER_CTA)
v391__JIT_EXPERTS_PER_MATH = tl.constexpr(v391__EXPERTS_PER_MATH)
v391__JIT_LOGICAL_EXPERT_CTAS = tl.constexpr(v391__LOGICAL_EXPERT_CTAS)
v391__JIT_ROUTE_ROUNDS = tl.constexpr(v391__ROUTE_ROUNDS)
v391__JIT_COMPACT_CANDIDATE_WORDS = tl.constexpr(v391__COMPACT_CANDIDATE_WORDS)


@triton.jit(noinline=True)
def v391__publish_local_top10_compact_packed(
    logits_ptr,
    candidate_scratch_ptr,
    expert_base,
    token_base,
    tokens,
    role_id: tl.constexpr,
):
    route = tl.arange(0, 128)
    routing_warp = route // 32
    lane = route % 32
    lane_u32 = lane.to(tl.uint32)
    for owner_round in tl.range(0, v391__JIT_ROUTE_ROUNDS, loop_unroll_factor=1):
        local_token = role_id * 4 + routing_warp + owner_round * 16
        token = token_base + local_token
        valid_token = token < tokens
        score_row = token * v391__JIT_EXPERTS
        local_expert_base = lane * 4
        s0, s1, s2, s3 = v370__load_score_float4(
            logits_ptr + score_row + expert_base + local_expert_base, valid_token
        )
        s4, s5, s6, s7 = v370__load_score_float4(
            logits_ptr
            + score_row
            + expert_base
            + local_expert_base
            + 2 * v391__JIT_EXPERTS_PER_MATH,
            valid_token,
        )
        s0 = s0.to(tl.bfloat16).to(tl.float32)
        s1 = s1.to(tl.bfloat16).to(tl.float32)
        s2 = s2.to(tl.bfloat16).to(tl.float32)
        s3 = s3.to(tl.bfloat16).to(tl.float32)
        s4 = s4.to(tl.bfloat16).to(tl.float32)
        s5 = s5.to(tl.bfloat16).to(tl.float32)
        s6 = s6.to(tl.bfloat16).to(tl.float32)
        s7 = s7.to(tl.bfloat16).to(tl.float32)
        s0 = tl.where(s0 == s0, s0, -float("inf"))
        s1 = tl.where(s1 == s1, s1, -float("inf"))
        s2 = tl.where(s2 == s2, s2, -float("inf"))
        s3 = tl.where(s3 == s3, s3, -float("inf"))
        s4 = tl.where(s4 == s4, s4, -float("inf"))
        s5 = tl.where(s5 == s5, s5, -float("inf"))
        s6 = tl.where(s6 == s6, s6, -float("inf"))
        s7 = tl.where(s7 == s7, s7, -float("inf"))
        s0, s1, s2, s3, permutation0 = v166__sort4_pack(s0, s1, s2, s3)
        s4, s5, s6, s7, permutation1 = v166__sort4_pack(s4, s5, s6, s7)
        r0 = routing__ordered_score_key(s0).to(tl.uint32) >> 16
        r1 = routing__ordered_score_key(s1).to(tl.uint32) >> 16
        r2 = routing__ordered_score_key(s2).to(tl.uint32) >> 16
        r3 = routing__ordered_score_key(s3).to(tl.uint32) >> 16
        r4 = routing__ordered_score_key(s4).to(tl.uint32) >> 16
        r5 = routing__ordered_score_key(s5).to(tl.uint32) >> 16
        r6 = routing__ordered_score_key(s6).to(tl.uint32) >> 16
        r7 = routing__ordered_score_key(s7).to(tl.uint32) >> 16
        cursor0 = tl.zeros((128,), dtype=tl.uint32)
        cursor1 = tl.zeros((128,), dtype=tl.uint32)
        selected_score = tl.zeros((128,), dtype=tl.float32)
        selected_local = tl.zeros((128,), dtype=tl.uint32)
        for output_idx in tl.static_range(0, v391__JIT_LOCAL_TOPK):
            head0 = v166__sorted_head(cursor0, s0, s1, s2, s3)
            head1 = v166__sorted_head(cursor1, s4, s5, s6, s7)
            rank0 = tl.where(
                cursor0 == 0,
                r0,
                tl.where(
                    cursor0 == 1,
                    r1,
                    tl.where(cursor0 == 2, r2, tl.where(cursor0 == 3, r3, 0)),
                ),
            )
            rank1 = tl.where(
                cursor1 == 0,
                r4,
                tl.where(
                    cursor1 == 1,
                    r5,
                    tl.where(cursor1 == 2, r6, tl.where(cursor1 == 3, r7, 0)),
                ),
            )
            offset0 = permutation0 >> cursor0 * 2 & 3
            offset1 = permutation1 >> cursor1 * 2 & 3
            local0 = lane_u32 * 4 + offset0
            local1 = 2 * v391__JIT_EXPERTS_PER_MATH + lane_u32 * 4 + offset1
            take_second = (rank1 > rank0) | (rank1 == rank0) & (local1 < local0)
            local_head = tl.where(take_second, head1, head0)
            local_expert = tl.where(take_second, local1, local0)
            local_rank = tl.where(take_second, rank1, rank0)
            packed_rank = local_rank << 8 | 255 - local_expert
            best_packed = routing__warp_max_u32(packed_rank)
            best_local = 255 - (best_packed & 255)
            source_lane = (best_local & 127) >> 2
            best_score = routing__warp_broadcast_f32(local_head, source_lane)
            owns_output = lane == output_idx
            selected_score = tl.where(owns_output, best_score, selected_score)
            selected_local = tl.where(owns_output, best_local, selected_local)
            owns_best = local_expert == best_local
            cursor0 += (owns_best & ~take_second).to(tl.uint32)
            cursor1 += (owns_best & take_second).to(tl.uint32)
        producer = expert_base // v391__JIT_EXPERTS_PER_CTA
        candidate = (
            candidate_scratch_ptr
            + (token * v391__JIT_LOGICAL_EXPERT_CTAS + producer)
            * v391__JIT_COMPACT_CANDIDATE_WORDS
            + lane
        )
        v388__store_packed_candidate_word(
            candidate,
            selected_score,
            selected_local,
            valid_token & (lane < v391__JIT_LOCAL_TOPK),
        )


v394__TOPK = v391__TOPK
v394__BLOCK_TOKENS = v391__BLOCK_TOKENS
v394__EXPERTS_PER_CTA = v391__EXPERTS_PER_CTA
v394__LOGICAL_EXPERT_CTAS = v391__LOGICAL_EXPERT_CTAS
v394__BLOCK_K = v391__BLOCK_K
v394__PIPE_STAGES = v391__PIPE_STAGES
v394__TOTAL_THREADS = v391__TOTAL_THREADS
v394__MATH_MAXNREG = v391__MATH_MAXNREG
v394__KERNEL_MAXNREG = v391__KERNEL_MAXNREG
v394__STATE_WORDS_PER_TILE = v391__STATE_WORDS_PER_TILE
v394__TICKET_WORD = v391__TICKET_WORD
v394__COMPACT_CANDIDATE_WORDS = v391__COMPACT_CANDIDATE_WORDS
v394__TERMINAL_WARPS = v391__TERMINAL_WARPS
v394__JIT_EXPERTS_PER_CTA = tl.constexpr(v394__EXPERTS_PER_CTA)
v394__JIT_LOGICAL_EXPERT_CTAS = tl.constexpr(v394__LOGICAL_EXPERT_CTAS)
v394__JIT_BLOCK_TOKENS = tl.constexpr(v394__BLOCK_TOKENS)
v394__JIT_BLOCK_K = tl.constexpr(v394__BLOCK_K)
v394__JIT_PIPE_STAGES = tl.constexpr(v394__PIPE_STAGES)
v394__JIT_TOTAL_THREADS = tl.constexpr(v394__TOTAL_THREADS)
v394__JIT_MATH_MAXNREG = tl.constexpr(v394__MATH_MAXNREG)
v394__JIT_STATE_WORDS_PER_TILE = tl.constexpr(v394__STATE_WORDS_PER_TILE)
v394__JIT_TICKET_WORD = tl.constexpr(v394__TICKET_WORD)
v394__JIT_TERMINAL_WARPS = tl.constexpr(v394__TERMINAL_WARPS)
v394__JIT_COMPACT_CANDIDATE_WORDS = tl.constexpr(v394__COMPACT_CANDIDATE_WORDS)
v394__JIT_TOPK = tl.constexpr(v394__TOPK)


@triton.jit
def v394__load_packed_candidate_word_l1(ptr, valid):
    """Load one compact head while retaining its 40-byte list in L1."""
    address = ptr.to(tl.uint64)
    packed = tl.inline_asm_elementwise(
        "\n        {\n            .reg .pred p;\n            mov.u32 $0, 0xff8000ff;\n            setp.ne.u32 p, $2, 0;\n            @p ld.weak.global.L1::evict_last.b32 $0, [$1];\n        }\n        ",
        "=r,l,r,~{memory}",
        [address, valid.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    score_bits = packed & 4294901760
    return (score_bits.to(tl.float32, bitcast=True), packed & 255)


@triton.jit
def v394__merge_two_local_top10_compact_l1_role(
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    token_base,
    tokens,
    role_id: tl.constexpr,
):
    if role_id * 4 < v394__JIT_TERMINAL_WARPS:
        terminal = tl.load(tle.gpu.local_ptr(terminal_smem, (0,)))
        if terminal != 0:
            route = tl.arange(0, 128)
            routing_warp = route // 32
            lane = route % 32
            token_group = lane // 2
            subwarp_lane = lane % 2
            terminal_warp = role_id * 4 + routing_warp
            local_token = terminal_warp * 16 + token_group
            token = token_base + local_token
            valid_token = token < tokens
            safe_token = tl.where(valid_token, token, token_base)
            producer = subwarp_lane.to(tl.uint32)
            candidate = (
                candidate_scratch_ptr
                + (safe_token * v394__JIT_LOGICAL_EXPERT_CTAS + producer)
                * v394__JIT_COMPACT_CANDIDATE_WORDS
            )
            head_score, head_local = v394__load_packed_candidate_word_l1(
                candidate, valid_token
            )
            head_expert = producer * v394__JIT_EXPERTS_PER_CTA + head_local
            head_rank = v388__packed_global_rank(head_score, head_expert)
            cursor = tl.zeros((128,), dtype=tl.uint32)
            selected_score0 = tl.zeros((128,), dtype=tl.float32)
            selected_score1 = tl.zeros((128,), dtype=tl.float32)
            selected_score2 = tl.zeros((128,), dtype=tl.float32)
            selected_score3 = tl.zeros((128,), dtype=tl.float32)
            selected_score4 = tl.zeros((128,), dtype=tl.float32)
            selected_expert0 = tl.zeros((128,), dtype=tl.uint32)
            selected_expert1 = tl.zeros((128,), dtype=tl.uint32)
            selected_expert2 = tl.zeros((128,), dtype=tl.uint32)
            selected_expert3 = tl.zeros((128,), dtype=tl.uint32)
            selected_expert4 = tl.zeros((128,), dtype=tl.uint32)
            top1 = tl.zeros((128,), dtype=tl.float32)
            for output_idx in tl.static_range(0, v394__JIT_TOPK):
                best_rank = v388__pair2_max_u32(head_rank)
                best_expert = 511 - (best_rank & 511)
                source_list = best_expert // v394__JIT_EXPERTS_PER_CTA
                best_score = v369__pair2_broadcast_f32(head_score, source_list)
                if output_idx == 0:
                    top1 = best_score
                output_slot = output_idx // 2
                owns_output = subwarp_lane == output_idx & 1
                if output_slot == 0:
                    selected_score0 = tl.where(owns_output, best_score, selected_score0)
                    selected_expert0 = tl.where(
                        owns_output, best_expert, selected_expert0
                    )
                elif output_slot == 1:
                    selected_score1 = tl.where(owns_output, best_score, selected_score1)
                    selected_expert1 = tl.where(
                        owns_output, best_expert, selected_expert1
                    )
                elif output_slot == 2:
                    selected_score2 = tl.where(owns_output, best_score, selected_score2)
                    selected_expert2 = tl.where(
                        owns_output, best_expert, selected_expert2
                    )
                elif output_slot == 3:
                    selected_score3 = tl.where(owns_output, best_score, selected_score3)
                    selected_expert3 = tl.where(
                        owns_output, best_expert, selected_expert3
                    )
                else:
                    selected_score4 = tl.where(owns_output, best_score, selected_score4)
                    selected_expert4 = tl.where(
                        owns_output, best_expert, selected_expert4
                    )
                owns_best = head_expert == best_expert
                if output_idx + 1 < v394__JIT_TOPK:
                    cursor += owns_best.to(tl.uint32)
                    next_candidate = (
                        candidate_scratch_ptr
                        + (safe_token * v394__JIT_LOGICAL_EXPERT_CTAS + producer)
                        * v394__JIT_COMPACT_CANDIDATE_WORDS
                        + cursor
                    )
                    next_score, next_local = v394__load_packed_candidate_word_l1(
                        next_candidate, valid_token & owns_best
                    )
                    next_expert = producer * v394__JIT_EXPERTS_PER_CTA + next_local
                    next_rank = v388__packed_global_rank(next_score, next_expert)
                    head_score = tl.where(owns_best, next_score, head_score)
                    head_expert = tl.where(owns_best, next_expert, head_expert)
                    head_rank = tl.where(owns_best, next_rank, head_rank)
            has_finite = top1 != -float("inf")
            exp0 = tl.where(has_finite, libdevice.exp(selected_score0 - top1), 0.0)
            exp1 = tl.where(has_finite, libdevice.exp(selected_score1 - top1), 0.0)
            exp2 = tl.where(has_finite, libdevice.exp(selected_score2 - top1), 0.0)
            exp3 = tl.where(has_finite, libdevice.exp(selected_score3 - top1), 0.0)
            exp4 = tl.where(has_finite, libdevice.exp(selected_score4 - top1), 0.0)
            local_sum = exp0 + exp4 + exp2
            local_sum += exp1 + exp3
            denominator = v369__pair2_sum_f32(local_sum) + 1e-20
            output_base = token * v394__JIT_TOPK + subwarp_lane
            tl.store(
                topk_indices_ptr + output_base,
                selected_expert0.to(tl.int32),
                mask=valid_token,
            )
            tl.store(
                topk_weights_ptr + output_base, exp0 / denominator, mask=valid_token
            )
            tl.store(
                topk_indices_ptr + output_base + 2,
                selected_expert1.to(tl.int32),
                mask=valid_token,
            )
            tl.store(
                topk_weights_ptr + output_base + 2, exp1 / denominator, mask=valid_token
            )
            tl.store(
                topk_indices_ptr + output_base + 4,
                selected_expert2.to(tl.int32),
                mask=valid_token,
            )
            tl.store(
                topk_weights_ptr + output_base + 4, exp2 / denominator, mask=valid_token
            )
            tl.store(
                topk_indices_ptr + output_base + 6,
                selected_expert3.to(tl.int32),
                mask=valid_token,
            )
            tl.store(
                topk_weights_ptr + output_base + 6, exp3 / denominator, mask=valid_token
            )
            tl.store(
                topk_indices_ptr + output_base + 8,
                selected_expert4.to(tl.int32),
                mask=valid_token,
            )
            tl.store(
                topk_weights_ptr + output_base + 8, exp4 / denominator, mask=valid_token
            )


@triton.jit
def v394__math_route_compact_l1(
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
    math_id: tl.constexpr,
    role_id: tl.constexpr,
):
    local_expert = expert_base + math_id * v369__JIT_EXPERTS_PER_MATH
    v369__gemm_b128(reader, logits_ptr, token_base, local_expert, tokens, math_id)
    tle.gpu.barrier_wait(logits_ready_barrier)
    v391__publish_local_top10_compact_packed(
        logits_ptr, candidate_scratch_ptr, expert_base, token_base, tokens, role_id
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)
    if math_id == 3:
        ticket_ptr = (
            state_ptr
            + token_tile * v394__JIT_STATE_WORDS_PER_TILE
            + v394__JIT_TICKET_WORD
        )
        old_ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="gpu")
        terminal = (
            old_ticket % v394__JIT_LOGICAL_EXPERT_CTAS + 1
            == v394__JIT_LOGICAL_EXPERT_CTAS
        )
        tl.store(tle.gpu.local_ptr(terminal_smem, (0,)), terminal.to(tl.int32))
    tle.gpu.barrier_wait(route_handoff_barrier)
    v394__merge_two_local_top10_compact_l1_role(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        tokens,
        role_id,
    )


@triton.jit
def v394__fused_qwen_h4096_t8214_compact_l1_kernel(
    weight_desc,
    x_desc,
    logits_ptr,
    candidate_scratch_ptr,
    state_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    tokens: tl.constexpr,
):
    pid = tl.program_id(0)
    logical_rank = pid % v394__JIT_LOGICAL_EXPERT_CTAS
    token_tile = pid // v394__JIT_LOGICAL_EXPERT_CTAS
    token_base = token_tile * v394__JIT_BLOCK_TOKENS
    expert_base = logical_rank * v394__JIT_EXPERTS_PER_CTA
    weight_smem = tle.gpu.alloc(
        [v394__JIT_PIPE_STAGES, v394__JIT_EXPERTS_PER_CTA, v394__JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    x_smem = tle.gpu.alloc(
        [v394__JIT_PIPE_STAGES, v394__JIT_BLOCK_TOKENS, v394__JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    pipe = tle.pipe(
        capacity=v394__JIT_PIPE_STAGES,
        scope="cta",
        name="qwen_h4096_t8214_compact_l1_v394",
        readers=("math0", "math1", "math2", "math3"),
        weight=weight_smem,
        x=x_smem,
    )
    terminal_smem = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    logits_ready_barrier = tle.gpu.alloc_barrier(arrive_count=v394__JIT_TOTAL_THREADS)
    candidates_ready_barrier = tle.gpu.alloc_barrier(
        arrive_count=v394__JIT_TOTAL_THREADS
    )
    route_handoff_barrier = tle.gpu.alloc_barrier(arrive_count=v394__JIT_TOTAL_THREADS)
    tle.gpu.warp_specialize(
        [
            (
                v369__loader_route0,
                (
                    pipe.writer(),
                    weight_desc,
                    x_desc,
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
                v394__math_route_compact_l1,
                (
                    pipe.reader(name="math0", fields=("weight", "x")),
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
                    0,
                ),
            ),
            (
                v394__math_route_compact_l1,
                (
                    pipe.reader(name="math1", fields=("weight", "x")),
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
                    1,
                ),
            ),
            (
                v394__math_route_compact_l1,
                (
                    pipe.reader(name="math2", fields=("weight", "x")),
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
                    2,
                ),
            ),
            (
                v394__math_route_compact_l1,
                (
                    pipe.reader(name="math3", fields=("weight", "x")),
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
                    3,
                ),
            ),
        ],
        [4, 4, 4, 4],
        [
            v394__JIT_MATH_MAXNREG,
            v394__JIT_MATH_MAXNREG,
            v394__JIT_MATH_MAXNREG,
            v394__JIT_MATH_MAXNREG,
        ],
    )


KERNEL = v394__fused_qwen_h4096_t8214_compact_l1_kernel
MAXNREG = v394__KERNEL_MAXNREG
SOURCE_LEAF = "v394"
__all__ = ["KERNEL", "MAXNREG", "SOURCE_LEAF"]
