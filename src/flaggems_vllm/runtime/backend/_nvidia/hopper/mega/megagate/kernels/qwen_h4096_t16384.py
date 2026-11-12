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
from .qwen_h4096_primitives import pair2_sum_f32 as v369__pair2_sum_f32
from .qwen_h4096_primitives import sort4_pack as v166__sort4_pack
from .qwen_h4096_primitives import sorted_head as v166__sorted_head
from .routing_primitives import ordered_score_key as routing__ordered_score_key
from .routing_primitives import warp_broadcast_f32 as routing__warp_broadcast_f32
from .routing_primitives import warp_max_u32 as routing__warp_max_u32
from .routing_primitives import warp_min_u32 as routing__warp_min_u32

v369__EXPERTS = 512
v369__TOPK = 10
v369__LOCAL_TOPK = v369__TOPK
v369__BLOCK_TOKENS = 128
v369__EXPERTS_PER_CTA = 256
v369__EXPERTS_PER_MATH = 64
v369__LOGICAL_EXPERT_CTAS = 2
v369__CANDIDATE_WORDS = 4
v369__CANDIDATE_RECORD_WORDS = v369__LOCAL_TOPK * v369__CANDIDATE_WORDS
v369__JIT_EXPERTS = tl.constexpr(v369__EXPERTS)
v369__JIT_TOPK = tl.constexpr(v369__TOPK)
v369__JIT_LOCAL_TOPK = tl.constexpr(v369__LOCAL_TOPK)
v369__JIT_BLOCK_TOKENS = tl.constexpr(v369__BLOCK_TOKENS)
v369__JIT_EXPERTS_PER_CTA = tl.constexpr(v369__EXPERTS_PER_CTA)
v369__JIT_EXPERTS_PER_MATH = tl.constexpr(v369__EXPERTS_PER_MATH)
v369__JIT_NUM_PRODUCERS = tl.constexpr(v369__LOGICAL_EXPERT_CTAS)
v369__JIT_CANDIDATE_WORDS = tl.constexpr(v369__CANDIDATE_WORDS)
v369__JIT_CANDIDATE_RECORD_WORDS = tl.constexpr(v369__CANDIDATE_RECORD_WORDS)


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
def v369__store_candidate_record_with_key(ptr, score, expert, ordered_key, valid):
    """Store score, expert and the already-computed ordered key as one v4."""
    address = ptr.to(tl.uint64)
    score_bits = score.to(tl.uint32, bitcast=True)
    _ = tl.inline_asm_elementwise(
        "\n        {\n            .reg .pred p;\n            setp.ne.u32 p, $5, 0;\n            @p st.global.v4.u32 [$1], {$2, $2, $3, $4};\n            mov.u32 $0, $5;\n        }\n        ",
        "=r,l,r,r,r,r,~{memory}",
        [
            address,
            score_bits,
            expert.to(tl.uint32),
            ordered_key.to(tl.uint32),
            valid.to(tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def v369__publish_local_top10_e256_role(
    logits_ptr,
    candidate_scratch_ptr,
    expert_base,
    token_base,
    tokens,
    ROLE_ID: tl.constexpr,
    OWNER_ROUNDS: tl.constexpr,
):
    """Publish one E256 local K10 using the four high-register math roles."""
    route = tl.arange(0, 128)
    routing_warp = route // 32
    lane = route % 32
    lane_u32 = lane.to(tl.uint32)
    for owner_round in tl.range(0, OWNER_ROUNDS, loop_unroll_factor=1):
        local_token = ROLE_ID * 4 + routing_warp + owner_round * 16
        token = token_base + local_token
        valid_token = token < tokens
        score_row = token * v369__JIT_EXPERTS
        local_expert_base = expert_base + lane * 4
        s0, s1, s2, s3 = v370__load_score_float4(
            logits_ptr + score_row + local_expert_base, valid_token
        )
        s4, s5, s6, s7 = v370__load_score_float4(
            logits_ptr + score_row + local_expert_base + 2 * v369__JIT_EXPERTS_PER_MATH,
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
        k0 = routing__ordered_score_key(s0).to(tl.uint32)
        k1 = routing__ordered_score_key(s1).to(tl.uint32)
        k2 = routing__ordered_score_key(s2).to(tl.uint32)
        k3 = routing__ordered_score_key(s3).to(tl.uint32)
        k4 = routing__ordered_score_key(s4).to(tl.uint32)
        k5 = routing__ordered_score_key(s5).to(tl.uint32)
        k6 = routing__ordered_score_key(s6).to(tl.uint32)
        k7 = routing__ordered_score_key(s7).to(tl.uint32)
        cursor0 = tl.zeros((128,), dtype=tl.uint32)
        cursor1 = tl.zeros((128,), dtype=tl.uint32)
        selected_score = tl.zeros((128,), dtype=tl.float32)
        selected_expert = tl.zeros((128,), dtype=tl.uint32)
        selected_key = tl.zeros((128,), dtype=tl.uint32)
        for output_idx in tl.static_range(0, v369__JIT_LOCAL_TOPK):
            head0 = v166__sorted_head(cursor0, s0, s1, s2, s3)
            head1 = v166__sorted_head(cursor1, s4, s5, s6, s7)
            key0 = tl.where(
                cursor0 == 0,
                k0,
                tl.where(
                    cursor0 == 1,
                    k1,
                    tl.where(
                        cursor0 == 2,
                        k2,
                        tl.where(cursor0 == 3, k3, tl.zeros((128,), dtype=tl.uint32)),
                    ),
                ),
            )
            key1 = tl.where(
                cursor1 == 0,
                k4,
                tl.where(
                    cursor1 == 1,
                    k5,
                    tl.where(
                        cursor1 == 2,
                        k6,
                        tl.where(cursor1 == 3, k7, tl.zeros((128,), dtype=tl.uint32)),
                    ),
                ),
            )
            offset0 = permutation0 >> cursor0 * 2 & 3
            offset1 = permutation1 >> cursor1 * 2 & 3
            expert0 = (expert_base + lane_u32 * 4 + offset0).to(tl.uint32)
            expert1 = (
                expert_base + 2 * v369__JIT_EXPERTS_PER_MATH + lane_u32 * 4 + offset1
            ).to(tl.uint32)
            take_second = (key1 > key0) | (key1 == key0) & (expert1 < expert0)
            local_head = tl.where(take_second, head1, head0)
            local_expert = tl.where(take_second, expert1, expert0)
            local_key = tl.where(take_second, key1, key0)
            best_key = routing__warp_max_u32(local_key)
            best_expert = routing__warp_min_u32(
                tl.where(local_key == best_key, local_expert, 4294967295).to(tl.uint32)
            )
            source_lane = (best_expert - expert_base & 127) >> 2
            best_score = routing__warp_broadcast_f32(local_head, source_lane)
            owns_output = lane == output_idx
            selected_score = tl.where(owns_output, best_score, selected_score)
            selected_expert = tl.where(owns_output, best_expert, selected_expert)
            selected_key = tl.where(owns_output, best_key, selected_key)
            owns_best = local_expert == best_expert
            cursor0 += (owns_best & ~take_second).to(tl.uint32)
            cursor1 += (owns_best & take_second).to(tl.uint32)
        producer = expert_base // v369__JIT_EXPERTS_PER_CTA
        candidate = (
            candidate_scratch_ptr
            + (token * v369__JIT_NUM_PRODUCERS + producer)
            * v369__JIT_CANDIDATE_RECORD_WORDS
            + lane * v369__JIT_CANDIDATE_WORDS
        )
        v369__store_candidate_record_with_key(
            candidate,
            selected_score,
            selected_expert,
            selected_key,
            valid_token & (lane < v369__JIT_LOCAL_TOPK),
        )


@triton.jit
def v369__pair2_stable_best_expert(score_key, expert):
    return tl.inline_asm_elementwise(
        "\n        {\n            .reg .b32 best_key, best_expert, peer_key, peer_expert;\n            .reg .pred p_key_gt, p_idx_lt, p_tie, p_take;\n            mov.u32 best_key, $1;\n            mov.u32 best_expert, $2;\n\n            shfl.sync.down.b32 peer_key, best_key, 1, 0x1e01, 0xffffffff;\n            shfl.sync.down.b32 peer_expert, best_expert, 1, 0x1e01, 0xffffffff;\n            setp.gt.u32 p_key_gt, peer_key, best_key;\n            setp.lt.u32 p_idx_lt, peer_expert, best_expert;\n            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;\n            or.pred p_take, p_key_gt, p_tie;\n            selp.u32 best_expert, peer_expert, best_expert, p_take;\n\n            shfl.sync.idx.b32 $0, best_expert, 0, 0x1e01, 0xffffffff;\n        }\n        ",
        "=r,r,r",
        [score_key.to(tl.uint32), expert.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def v369__merge_two_local_top10_width2_role(
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    token_base,
    tokens,
    ROLE_ID: tl.constexpr,
    TERMINAL_WARPS: tl.constexpr,
):
    """Merge two E256 lists with eight B128 or thirteen B208 warps."""
    if ROLE_ID * 4 < TERMINAL_WARPS:
        terminal = tl.load(tle.gpu.local_ptr(terminal_smem, (0,)))
        if terminal != 0:
            route = tl.arange(0, 128)
            routing_warp = route // 32
            lane = route % 32
            token_group = lane // 2
            subwarp_lane = lane % 2
            terminal_warp = ROLE_ID * 4 + routing_warp
            active_warp = terminal_warp < TERMINAL_WARPS
            local_token = terminal_warp * 16 + token_group
            token = token_base + local_token
            valid_token = active_warp & (token < tokens)
            safe_token = tl.where(valid_token, token, token_base)
            producer = subwarp_lane.to(tl.uint32)
            candidate = (
                candidate_scratch_ptr
                + (safe_token * v369__JIT_NUM_PRODUCERS + producer)
                * v369__JIT_CANDIDATE_RECORD_WORDS
            )
            head_score, _, head_expert, head_key = v370__load_candidate_record(
                candidate, valid_token
            )
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
            for output_idx in tl.static_range(0, v369__JIT_TOPK):
                best_expert = v369__pair2_stable_best_expert(head_key, head_expert)
                source_list = best_expert // v369__JIT_EXPERTS_PER_CTA
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
                if output_idx + 1 < v369__JIT_TOPK:
                    cursor += owns_best.to(tl.uint32)
                    next_candidate = (
                        candidate_scratch_ptr
                        + (safe_token * v369__JIT_NUM_PRODUCERS + producer)
                        * v369__JIT_CANDIDATE_RECORD_WORDS
                        + cursor * v369__JIT_CANDIDATE_WORDS
                    )
                    next_score, _, next_expert, next_key = v370__load_candidate_record(
                        next_candidate, valid_token & owns_best
                    )
                    head_score = tl.where(owns_best, next_score, head_score)
                    head_expert = tl.where(owns_best, next_expert, head_expert)
                    head_key = tl.where(owns_best, next_key, head_key)
            has_finite = top1 != -float("inf")
            exp0 = tl.where(has_finite, libdevice.exp(selected_score0 - top1), 0.0)
            exp1 = tl.where(has_finite, libdevice.exp(selected_score1 - top1), 0.0)
            exp2 = tl.where(has_finite, libdevice.exp(selected_score2 - top1), 0.0)
            exp3 = tl.where(has_finite, libdevice.exp(selected_score3 - top1), 0.0)
            exp4 = tl.where(has_finite, libdevice.exp(selected_score4 - top1), 0.0)
            local_sum = exp0 + exp4 + exp2
            local_sum += exp1 + exp3
            denominator = v369__pair2_sum_f32(local_sum) + 1e-20
            output_base = token * v369__JIT_TOPK + subwarp_lane
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
def v370__load_candidate_record(ptr, valid):
    """Load one CUDA-v020 16-byte candidate in a single instruction."""
    address = ptr.to(tl.uint64)
    ranking, unbiased, expert, ordered = tl.inline_asm_elementwise(
        "\n        {\n            .reg .pred p;\n            .reg .b64 cache_hint;\n            mov.u32 $0, 0xff800000;\n            mov.u32 $1, 0xff800000;\n            mov.u32 $2, 0xffffffff;\n            mov.u32 $3, 0;\n            mov.u64 cache_hint, 0x12f0000000000000;\n            setp.ne.u32 p, $5, 0;\n            @p ld.weak.global.L1::no_allocate.L2::cache_hint.v4.b32\n                {$0, $1, $2, $3}, [$4], cache_hint;\n        }\n        ",
        "=r,=r,=r,=r,l,r,~{memory}",
        [address, valid.to(tl.uint32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    return (
        ranking.to(tl.float32, bitcast=True),
        unbiased.to(tl.float32, bitcast=True),
        expert,
        ordered,
    )


v387__HIDDEN = 4096
v387__BLOCK_TOKENS = 256
v387__PHASE_TOKENS = 128
v387__EXPERTS_PER_CTA = 256
v387__EXPERTS_PER_MATH = 64
v387__LOGICAL_EXPERT_CTAS = 2
v387__BLOCK_K = 128
v387__NUM_K_TILES = v387__HIDDEN // v387__BLOCK_K
v387__PIPE_STAGES = 2
v387__MATH_ROLES = 4
v387__LOGICAL_ROLES = 1 + v387__MATH_ROLES
v387__THREADS_PER_ROLE = 128
v387__TOTAL_THREADS = v387__LOGICAL_ROLES * v387__THREADS_PER_ROLE
v387__MATH_MAXNREG = 112
v387__KERNEL_MAXNREG = 96
v387__STATE_WORDS_PER_TILE = 16
v387__TICKET_WORD = 4
v387__OWNER_ROUNDS = 16
v387__TERMINAL_WARPS = 16
v387__JIT_BLOCK_TOKENS = tl.constexpr(v387__BLOCK_TOKENS)
v387__JIT_PHASE_TOKENS = tl.constexpr(v387__PHASE_TOKENS)
v387__JIT_EXPERTS_PER_CTA = tl.constexpr(v387__EXPERTS_PER_CTA)
v387__JIT_EXPERTS_PER_MATH = tl.constexpr(v387__EXPERTS_PER_MATH)
v387__JIT_LOGICAL_EXPERT_CTAS = tl.constexpr(v387__LOGICAL_EXPERT_CTAS)
v387__JIT_BLOCK_K = tl.constexpr(v387__BLOCK_K)
v387__JIT_NUM_K_TILES = tl.constexpr(v387__NUM_K_TILES)
v387__JIT_PIPE_STAGES = tl.constexpr(v387__PIPE_STAGES)
v387__JIT_TOTAL_THREADS = tl.constexpr(v387__TOTAL_THREADS)
v387__JIT_MATH_MAXNREG = tl.constexpr(v387__MATH_MAXNREG)
v387__JIT_STATE_WORDS_PER_TILE = tl.constexpr(v387__STATE_WORDS_PER_TILE)
v387__JIT_TICKET_WORD = tl.constexpr(v387__TICKET_WORD)
v387__JIT_NUM_PRODUCERS = tl.constexpr(v387__LOGICAL_EXPERT_CTAS)
v387__JIT_OWNER_ROUNDS = tl.constexpr(v387__OWNER_ROUNDS)
v387__JIT_TERMINAL_WARPS = tl.constexpr(v387__TERMINAL_WARPS)


@triton.jit
def v387__load_b256_e256(writer, weight_desc, x128_desc, token_base, expert_base):
    """Transport two sequential B128 phases through one physical stage."""
    for k_tile in tl.range(0, v387__JIT_NUM_K_TILES):
        slot = writer.acquire(k_tile)
        k_base = k_tile * v387__JIT_BLOCK_K
        tle.gpu.copy(
            weight_desc,
            slot.weight,
            [v387__JIT_EXPERTS_PER_CTA, v387__JIT_BLOCK_K],
            [expert_base, k_base],
        )
        tle.gpu.copy(
            x128_desc,
            slot.x,
            [v387__JIT_PHASE_TOKENS, v387__JIT_BLOCK_K],
            [token_base, k_base],
        )
        writer.commit(k_tile)
    for k_tile in tl.range(0, v387__JIT_NUM_K_TILES):
        ticket = v387__JIT_NUM_K_TILES + k_tile
        slot = writer.acquire(ticket)
        k_base = k_tile * v387__JIT_BLOCK_K
        tle.gpu.copy(
            weight_desc,
            slot.weight,
            [v387__JIT_EXPERTS_PER_CTA, v387__JIT_BLOCK_K],
            [expert_base, k_base],
        )
        tle.gpu.copy(
            x128_desc,
            slot.x,
            [v387__JIT_PHASE_TOKENS, v387__JIT_BLOCK_K],
            [token_base + v387__JIT_PHASE_TOKENS, k_base],
        )
        writer.commit(ticket)


@triton.jit
def v387__loader_route0(
    writer,
    weight_desc,
    x128_desc,
    logits_ready_barrier,
    candidates_ready_barrier,
    route_handoff_barrier,
    token_base,
    expert_base,
):
    v387__load_b256_e256(writer, weight_desc, x128_desc, token_base, expert_base)
    tle.gpu.barrier_wait(logits_ready_barrier)
    tle.gpu.barrier_wait(candidates_ready_barrier)
    tle.gpu.barrier_wait(route_handoff_barrier)


v415__KERNEL_MAXNREG = v387__KERNEL_MAXNREG
v415__JIT_BLOCK_TOKENS = v387__JIT_BLOCK_TOKENS
v415__JIT_PHASE_TOKENS = v387__JIT_PHASE_TOKENS
v415__JIT_EXPERTS_PER_CTA = v387__JIT_EXPERTS_PER_CTA
v415__JIT_EXPERTS_PER_MATH = v387__JIT_EXPERTS_PER_MATH
v415__JIT_LOGICAL_EXPERT_CTAS = v387__JIT_LOGICAL_EXPERT_CTAS
v415__JIT_BLOCK_K = v387__JIT_BLOCK_K
v415__JIT_NUM_K_TILES = v387__JIT_NUM_K_TILES
v415__JIT_PIPE_STAGES = v387__JIT_PIPE_STAGES
v415__JIT_TOTAL_THREADS = v387__JIT_TOTAL_THREADS
v415__JIT_MATH_MAXNREG = v387__JIT_MATH_MAXNREG
v415__JIT_STATE_WORDS_PER_TILE = v387__JIT_STATE_WORDS_PER_TILE
v415__JIT_TICKET_WORD = v387__JIT_TICKET_WORD
v415__JIT_NUM_PRODUCERS = v387__JIT_NUM_PRODUCERS
v415__JIT_OWNER_ROUNDS = v387__JIT_OWNER_ROUNDS
v415__JIT_TERMINAL_WARPS = v387__JIT_TERMINAL_WARPS


@triton.jit
def v415__gemm_two_b128_phases_loop(
    reader, logits_ptr, token_base, local_expert, tokens, MATH_ID: tl.constexpr
):
    """Reuse one B128 accumulator across two non-unrolled phase iterations."""
    for phase in tl.range(0, 2, loop_unroll_factor=1):
        accumulator = tl.zeros(
            (v415__JIT_EXPERTS_PER_MATH, v415__JIT_PHASE_TOKENS), dtype=tl.float32
        )
        for k_tile in tl.range(0, v415__JIT_NUM_K_TILES):
            ticket = phase * v415__JIT_NUM_K_TILES + k_tile
            stage = reader.wait(ticket)
            accumulator = v369__issue_b128(stage, accumulator, MATH_ID)
            accumulator = tle.gpu.wgmma_wait(0, accumulator)
            reader.release(ticket)
        v369__store_atom(
            accumulator,
            logits_ptr,
            token_base + phase * v415__JIT_PHASE_TOKENS,
            local_expert,
            tokens,
        )


@triton.jit
def v415__math_route_loop_reuse(
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
    tokens,
    MATH_ID: tl.constexpr,
    ROLE_ID: tl.constexpr,
):
    local_expert = expert_base + MATH_ID * v415__JIT_EXPERTS_PER_MATH
    v415__gemm_two_b128_phases_loop(
        reader, logits_ptr, token_base, local_expert, tokens, MATH_ID
    )
    tle.gpu.barrier_wait(logits_ready_barrier)
    v369__publish_local_top10_e256_role(
        logits_ptr,
        candidate_scratch_ptr,
        expert_base,
        token_base,
        tokens,
        ROLE_ID,
        v415__JIT_OWNER_ROUNDS,
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)
    if MATH_ID == 3:
        ticket_ptr = (
            state_ptr
            + token_tile * v415__JIT_STATE_WORDS_PER_TILE
            + v415__JIT_TICKET_WORD
        )
        old_ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="gpu")
        terminal = old_ticket % v415__JIT_NUM_PRODUCERS + 1 == v415__JIT_NUM_PRODUCERS
        tl.store(tle.gpu.local_ptr(terminal_smem, (0,)), terminal.to(tl.int32))
    tle.gpu.barrier_wait(route_handoff_barrier)
    v369__merge_two_local_top10_width2_role(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        tokens,
        ROLE_ID,
        v415__JIT_TERMINAL_WARPS,
    )


@triton.jit
def v415__fused_qwen_h4096_t16384_b256_e256_loop_reuse_kernel(
    weight_desc,
    x128_desc,
    logits_ptr,
    candidate_scratch_ptr,
    state_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    tokens,
):
    pid = tl.program_id(0)
    logical_rank = pid % v415__JIT_LOGICAL_EXPERT_CTAS
    token_tile = pid // v415__JIT_LOGICAL_EXPERT_CTAS
    token_base = token_tile * v415__JIT_BLOCK_TOKENS
    expert_base = logical_rank * v415__JIT_EXPERTS_PER_CTA
    weight_smem = tle.gpu.alloc(
        [v415__JIT_PIPE_STAGES, v415__JIT_EXPERTS_PER_CTA, v415__JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    x_smem = tle.gpu.alloc(
        [v415__JIT_PIPE_STAGES, v415__JIT_PHASE_TOKENS, v415__JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    pipe = tle.pipe(
        capacity=v415__JIT_PIPE_STAGES,
        scope="cta",
        name="qwen_h4096_t16384_b256_e256_loop_reuse_v415",
        readers=("math0", "math1", "math2", "math3"),
        weight=weight_smem,
        x=x_smem,
    )
    terminal_smem = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    logits_ready_barrier = tle.gpu.alloc_barrier(arrive_count=v415__JIT_TOTAL_THREADS)
    candidates_ready_barrier = tle.gpu.alloc_barrier(
        arrive_count=v415__JIT_TOTAL_THREADS
    )
    route_handoff_barrier = tle.gpu.alloc_barrier(arrive_count=v415__JIT_TOTAL_THREADS)
    tle.gpu.warp_specialize(
        [
            (
                v387__loader_route0,
                (
                    pipe.writer(),
                    weight_desc,
                    x128_desc,
                    logits_ready_barrier,
                    candidates_ready_barrier,
                    route_handoff_barrier,
                    token_base,
                    expert_base,
                ),
            ),
            (
                v415__math_route_loop_reuse,
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
                v415__math_route_loop_reuse,
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
                v415__math_route_loop_reuse,
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
                v415__math_route_loop_reuse,
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
            v415__JIT_MATH_MAXNREG,
            v415__JIT_MATH_MAXNREG,
            v415__JIT_MATH_MAXNREG,
            v415__JIT_MATH_MAXNREG,
        ],
    )


KERNEL = v415__fused_qwen_h4096_t16384_b256_e256_loop_reuse_kernel
MAXNREG = v415__KERNEL_MAXNREG
SOURCE_LEAF = "v415"
__all__ = ["KERNEL", "MAXNREG", "SOURCE_LEAF"]
