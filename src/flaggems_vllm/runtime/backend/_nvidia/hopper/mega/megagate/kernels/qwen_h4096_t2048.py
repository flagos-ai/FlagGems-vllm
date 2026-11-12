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
from .qwen_h4096_primitives import sort4_pack as v166__sort4_pack
from .qwen_h4096_primitives import sorted_head as v166__sorted_head
from .routing_primitives import ordered_score_key as routing__ordered_score_key
from .routing_primitives import warp_broadcast_f32 as routing__warp_broadcast_f32
from .routing_primitives import warp_max_u32 as routing__warp_max_u32
from .routing_primitives import warp_min_u32 as routing__warp_min_u32


@triton.jit
def v370__halfwarp16_stable_best_expert(score_key, expert):
    """CUDA-v020 stable argmax in two independent width-16 subwarps."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b32 best_key, best_expert, peer_key, peer_expert;
            .reg .pred p_key_gt, p_idx_lt, p_tie, p_take;
            mov.u32 best_key, $1;
            mov.u32 best_expert, $2;

            shfl.sync.down.b32 peer_key, best_key, 8, 0x100f, 0xffffffff;
            shfl.sync.down.b32 peer_expert, best_expert, 8, 0x100f, 0xffffffff;
            setp.gt.u32 p_key_gt, peer_key, best_key;
            setp.lt.u32 p_idx_lt, peer_expert, best_expert;
            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;
            or.pred p_take, p_key_gt, p_tie;
            selp.u32 best_key, peer_key, best_key, p_take;
            selp.u32 best_expert, peer_expert, best_expert, p_take;

            shfl.sync.down.b32 peer_key, best_key, 4, 0x100f, 0xffffffff;
            shfl.sync.down.b32 peer_expert, best_expert, 4, 0x100f, 0xffffffff;
            setp.gt.u32 p_key_gt, peer_key, best_key;
            setp.lt.u32 p_idx_lt, peer_expert, best_expert;
            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;
            or.pred p_take, p_key_gt, p_tie;
            selp.u32 best_key, peer_key, best_key, p_take;
            selp.u32 best_expert, peer_expert, best_expert, p_take;

            shfl.sync.down.b32 peer_key, best_key, 2, 0x100f, 0xffffffff;
            shfl.sync.down.b32 peer_expert, best_expert, 2, 0x100f, 0xffffffff;
            setp.gt.u32 p_key_gt, peer_key, best_key;
            setp.lt.u32 p_idx_lt, peer_expert, best_expert;
            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;
            or.pred p_take, p_key_gt, p_tie;
            selp.u32 best_key, peer_key, best_key, p_take;
            selp.u32 best_expert, peer_expert, best_expert, p_take;

            shfl.sync.down.b32 peer_key, best_key, 1, 0x100f, 0xffffffff;
            shfl.sync.down.b32 peer_expert, best_expert, 1, 0x100f, 0xffffffff;
            setp.gt.u32 p_key_gt, peer_key, best_key;
            setp.lt.u32 p_idx_lt, peer_expert, best_expert;
            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;
            or.pred p_take, p_key_gt, p_tie;
            selp.u32 best_key, peer_key, best_key, p_take;
            selp.u32 best_expert, peer_expert, best_expert, p_take;

            shfl.sync.idx.b32 $0, best_expert, 0, 0x100f, 0xffffffff;
        }
        """,
        "=r,r,r",
        [score_key.to(tl.uint32), expert.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def v370__halfwarp16_broadcast_f32(value, source_lane):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b32 input_bits, output_bits;
            mov.b32 input_bits, $1;
            shfl.sync.idx.b32 output_bits, input_bits, $2, 0x100f, 0xffffffff;
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
def v370__halfwarp16_sum_f32(value):
    """Reduce and broadcast one sum independently in each width-16 group."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 current, peer;
            .reg .b32 current_bits, peer_bits;
            mov.f32 current, $1;

            mov.b32 current_bits, current;
            shfl.sync.down.b32 peer_bits, current_bits, 8, 0x100f, 0xffffffff;
            mov.b32 peer, peer_bits;
            add.rn.f32 current, current, peer;

            mov.b32 current_bits, current;
            shfl.sync.down.b32 peer_bits, current_bits, 4, 0x100f, 0xffffffff;
            mov.b32 peer, peer_bits;
            add.rn.f32 current, current, peer;

            mov.b32 current_bits, current;
            shfl.sync.down.b32 peer_bits, current_bits, 2, 0x100f, 0xffffffff;
            mov.b32 peer, peer_bits;
            add.rn.f32 current, current, peer;

            mov.b32 current_bits, current;
            shfl.sync.down.b32 peer_bits, current_bits, 1, 0x100f, 0xffffffff;
            mov.b32 peer, peer_bits;
            add.rn.f32 current, current, peer;

            mov.b32 current_bits, current;
            shfl.sync.idx.b32 peer_bits, current_bits, 0, 0x100f, 0xffffffff;
            mov.b32 $0, peer_bits;
        }
        """,
        "=f,f",
        [value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


v386__HIDDEN = 4096
v386__EXPERTS = 512
v386__TOPK = 10
v386__LOCAL_TOPK = v386__TOPK
v386__BLOCK_TOKENS = 64
v386__EXPERTS_PER_CTA = 128
v386__EXPERTS_PER_MATH = 64
v386__LOGICAL_EXPERT_CTAS = 4
v386__BLOCK_K = 128
v386__NUM_K_TILES = v386__HIDDEN // v386__BLOCK_K
v386__PIPE_STAGES = 4
v386__LOGICAL_ROLES = 4
v386__THREADS_PER_ROLE = 128
v386__TOTAL_THREADS = v386__LOGICAL_ROLES * v386__THREADS_PER_ROLE
v386__MATH_MAXNREG = 160
v386__ROUTE_MAXNREG = 128
v386__KERNEL_MAXNREG = 128
v386__STATE_WORDS_PER_TILE = 16
v386__TICKET_WORD = 4
v386__PUBLISH_OWNER_ROUNDS = 4
v386__MERGE_OWNER_ROUNDS = 2
v386__JIT_EXPERTS = tl.constexpr(v386__EXPERTS)
v386__JIT_BLOCK_TOKENS = tl.constexpr(v386__BLOCK_TOKENS)
v386__JIT_EXPERTS_PER_CTA = tl.constexpr(v386__EXPERTS_PER_CTA)
v386__JIT_EXPERTS_PER_MATH = tl.constexpr(v386__EXPERTS_PER_MATH)
v386__JIT_BLOCK_K = tl.constexpr(v386__BLOCK_K)
v386__JIT_NUM_K_TILES = tl.constexpr(v386__NUM_K_TILES)


@triton.jit
def v386__load(writer, weight_desc, x_desc, token_base, expert_base):
    for ticket in tl.range(0, v386__JIT_NUM_K_TILES):
        slot = writer.acquire(ticket)
        k_base = ticket * v386__JIT_BLOCK_K
        tle.gpu.copy(
            weight_desc,
            slot.weight,
            [v386__JIT_EXPERTS_PER_CTA, v386__JIT_BLOCK_K],
            [expert_base, k_base],
        )
        tle.gpu.copy(
            x_desc,
            slot.x,
            [v386__JIT_BLOCK_TOKENS, v386__JIT_BLOCK_K],
            [token_base, k_base],
        )
        writer.commit(ticket)


@triton.jit
def v386__math_wait0(
    reader,
    logits_ptr,
    token_base,
    expert_base,
    tokens: tl.constexpr,
    MATH_ID: tl.constexpr,
):
    accumulator = tl.zeros(
        (v386__JIT_EXPERTS_PER_MATH, v386__JIT_BLOCK_TOKENS), dtype=tl.float32
    )
    for ticket in tl.range(0, v386__JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        weight_half = stage.slot.weight.subslice(
            MATH_ID * v386__JIT_EXPERTS_PER_MATH, v386__JIT_EXPERTS_PER_MATH, 0
        )
        accumulator = tle.gpu.wgmma(
            weight_half, stage.slot.x, accumulator, trans_b=True
        )
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        reader.release(ticket)
    token = token_base + tl.arange(0, v386__JIT_BLOCK_TOKENS)[:, None]
    expert = (
        expert_base
        + MATH_ID * v386__JIT_EXPERTS_PER_MATH
        + tl.arange(0, v386__JIT_EXPERTS_PER_MATH)[None, :]
    )
    bounded = tl.trans(accumulator).to(tl.bfloat16).to(tl.float32)
    tl.store(
        logits_ptr + token * v386__JIT_EXPERTS + expert, bounded, mask=token < tokens
    )


@triton.jit
def v388__store_packed_candidate_word(ptr, score, local_expert, valid):
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


v400__EXPERTS = v386__EXPERTS
v400__TOPK = v386__TOPK
v400__LOCAL_TOPK = v386__LOCAL_TOPK
v400__BLOCK_TOKENS = v386__BLOCK_TOKENS
v400__EXPERTS_PER_CTA = v386__EXPERTS_PER_CTA
v400__LOGICAL_EXPERT_CTAS = v386__LOGICAL_EXPERT_CTAS
v400__BLOCK_K = v386__BLOCK_K
v400__PIPE_STAGES = v386__PIPE_STAGES
v400__TOTAL_THREADS = v386__TOTAL_THREADS
v400__MATH_MAXNREG = v386__MATH_MAXNREG
v400__ROUTE_MAXNREG = v386__ROUTE_MAXNREG
v400__KERNEL_MAXNREG = v386__KERNEL_MAXNREG
v400__STATE_WORDS_PER_TILE = v386__STATE_WORDS_PER_TILE
v400__TICKET_WORD = v386__TICKET_WORD
v400__PUBLISH_OWNER_ROUNDS = v386__PUBLISH_OWNER_ROUNDS
v400__MERGE_OWNER_ROUNDS = v386__MERGE_OWNER_ROUNDS
v400__COMPACT_CANDIDATE_WORDS = v400__LOCAL_TOPK
v400__JIT_EXPERTS = tl.constexpr(v400__EXPERTS)
v400__JIT_TOPK = tl.constexpr(v400__TOPK)
v400__JIT_LOCAL_TOPK = tl.constexpr(v400__LOCAL_TOPK)
v400__JIT_BLOCK_TOKENS = tl.constexpr(v400__BLOCK_TOKENS)
v400__JIT_EXPERTS_PER_CTA = tl.constexpr(v400__EXPERTS_PER_CTA)
v400__JIT_LOGICAL_EXPERT_CTAS = tl.constexpr(v400__LOGICAL_EXPERT_CTAS)
v400__JIT_BLOCK_K = tl.constexpr(v400__BLOCK_K)
v400__JIT_PIPE_STAGES = tl.constexpr(v400__PIPE_STAGES)
v400__JIT_TOTAL_THREADS = tl.constexpr(v400__TOTAL_THREADS)
v400__JIT_MATH_MAXNREG = tl.constexpr(v400__MATH_MAXNREG)
v400__JIT_ROUTE_MAXNREG = tl.constexpr(v400__ROUTE_MAXNREG)
v400__JIT_STATE_WORDS_PER_TILE = tl.constexpr(v400__STATE_WORDS_PER_TILE)
v400__JIT_TICKET_WORD = tl.constexpr(v400__TICKET_WORD)
v400__JIT_PUBLISH_OWNER_ROUNDS = tl.constexpr(v400__PUBLISH_OWNER_ROUNDS)
v400__JIT_MERGE_OWNER_ROUNDS = tl.constexpr(v400__MERGE_OWNER_ROUNDS)
v400__JIT_COMPACT_CANDIDATE_WORDS = tl.constexpr(v400__COMPACT_CANDIDATE_WORDS)


@triton.jit
def v400__publish_local_top10_b64_compact_role(
    logits_ptr,
    candidate_scratch_ptr,
    expert_base,
    token_base,
    tokens: tl.constexpr,
    role_id: tl.constexpr,
):
    """Run v386's E128 selector and compact only its output representation."""
    route = tl.arange(0, 128)
    routing_warp = route // 32
    lane = route % 32
    lane_u32 = lane.to(tl.uint32)
    invalid_expert = tl.full((128,), 4294967295, dtype=tl.uint32)
    for owner_round in tl.range(
        0, v400__JIT_PUBLISH_OWNER_ROUNDS, loop_unroll_factor=1
    ):
        local_token = role_id * 4 + routing_warp + owner_round * 16
        token = token_base + local_token
        valid_token = token < tokens
        safe_token = tl.where(valid_token, token, token_base)
        score_row = safe_token * v400__JIT_EXPERTS
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
        for output_idx in tl.static_range(0, v400__JIT_LOCAL_TOPK):
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
        producer = expert_base // v400__JIT_EXPERTS_PER_CTA
        candidate = (
            candidate_scratch_ptr
            + (safe_token * v400__JIT_LOGICAL_EXPERT_CTAS + producer)
            * v400__JIT_COMPACT_CANDIDATE_WORDS
            + lane
        )
        v388__store_packed_candidate_word(
            candidate,
            selected_score,
            selected_expert - expert_base,
            valid_token & (lane < v400__JIT_LOCAL_TOPK),
        )


@triton.jit
def v400__load_packed_candidate_word_l1(ptr, valid):
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
def v400__merge_four_local_top10_b64_compact_role(
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    token_base,
    tokens: tl.constexpr,
    role_id: tl.constexpr,
):
    """Run v386's stable four-list merge over contiguous packed heads."""
    terminal = tl.load(tle.gpu.local_ptr(terminal_smem, (0,)))
    if terminal != 0:
        route = tl.arange(0, 128)
        routing_warp = route // 32
        lane = route % 32
        token_half = lane // 16
        subwarp_lane = lane % 16
        producer = subwarp_lane.to(tl.uint32)
        producer_lane = subwarp_lane < v400__JIT_LOGICAL_EXPERT_CTAS
        invalid_expert = tl.full((128,), 4294967295, dtype=tl.uint32)
        for owner_round in tl.range(
            0, v400__JIT_MERGE_OWNER_ROUNDS, loop_unroll_factor=1
        ):
            local_token = role_id * 8 + routing_warp * 2 + token_half + owner_round * 32
            token = token_base + local_token
            valid_token = token < tokens
            safe_token = tl.where(valid_token, token, token_base)
            cursor = tl.zeros((128,), dtype=tl.uint32)
            selected_score = tl.zeros((128,), dtype=tl.float32)
            selected_expert = tl.zeros((128,), dtype=tl.uint32)
            candidate = (
                candidate_scratch_ptr
                + (safe_token * v400__JIT_LOGICAL_EXPERT_CTAS + producer)
                * v400__JIT_COMPACT_CANDIDATE_WORDS
            )
            head_score, head_local = v400__load_packed_candidate_word_l1(
                candidate, valid_token & producer_lane
            )
            head_expert = producer * v400__JIT_EXPERTS_PER_CTA + head_local
            head_expert = tl.where(producer_lane, head_expert, invalid_expert)
            for output_idx in tl.static_range(0, v400__JIT_TOPK):
                head_key = routing__ordered_score_key(head_score).to(tl.uint32)
                best_expert = v370__halfwarp16_stable_best_expert(
                    head_key, tl.where(producer_lane, head_expert, invalid_expert)
                )
                source_list = best_expert // v400__JIT_EXPERTS_PER_CTA
                best_score = v370__halfwarp16_broadcast_f32(head_score, source_list)
                owns_best = producer_lane & (head_expert == best_expert)
                selected_score = tl.where(
                    subwarp_lane == output_idx, best_score, selected_score
                )
                selected_expert = tl.where(
                    subwarp_lane == output_idx, best_expert, selected_expert
                )
                if output_idx + 1 < v400__JIT_TOPK:
                    cursor += owns_best.to(tl.uint32)
                    next_candidate = (
                        candidate_scratch_ptr
                        + (safe_token * v400__JIT_LOGICAL_EXPERT_CTAS + producer)
                        * v400__JIT_COMPACT_CANDIDATE_WORDS
                        + cursor
                    )
                    next_score, next_local = v400__load_packed_candidate_word_l1(
                        next_candidate, valid_token & owns_best
                    )
                    next_expert = producer * v400__JIT_EXPERTS_PER_CTA + next_local
                    head_score = tl.where(owns_best, next_score, head_score)
                    head_expert = tl.where(owns_best, next_expert, head_expert)
            output_lane = subwarp_lane < v400__JIT_TOPK
            top1 = v370__halfwarp16_broadcast_f32(
                selected_score, tl.zeros((128,), dtype=tl.uint32)
            )
            has_finite = top1 != -float("inf")
            selected_exp = tl.where(
                output_lane & has_finite, libdevice.exp(selected_score - top1), 0.0
            )
            denominator = v370__halfwarp16_sum_f32(selected_exp) + 1e-20
            output = safe_token * v400__JIT_TOPK + subwarp_lane
            store_mask = valid_token & output_lane
            tl.store(
                topk_indices_ptr + output, selected_expert.to(tl.int32), mask=store_mask
            )
            tl.store(
                topk_weights_ptr + output, selected_exp / denominator, mask=store_mask
            )


@triton.jit
def v400__loader_publish_merge0(
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
    v386__load(writer, weight_desc, x_desc, token_base, expert_base)
    tle.gpu.barrier_wait(logits_ready_barrier)
    v400__publish_local_top10_b64_compact_role(
        logits_ptr, candidate_scratch_ptr, expert_base, token_base, tokens, 0
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)
    tle.gpu.barrier_wait(route_handoff_barrier)
    v400__merge_four_local_top10_b64_compact_role(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        tokens,
        0,
    )


@triton.jit
def v400__math_publish_arrive_merge(
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
    v386__math_wait0(reader, logits_ptr, token_base, expert_base, tokens, math_id)
    tle.gpu.barrier_wait(logits_ready_barrier)
    v400__publish_local_top10_b64_compact_role(
        logits_ptr, candidate_scratch_ptr, expert_base, token_base, tokens, role_id
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)
    if math_id == 1:
        ticket_ptr = (
            state_ptr
            + token_tile * v400__JIT_STATE_WORDS_PER_TILE
            + v400__JIT_TICKET_WORD
        )
        old_ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="gpu")
        terminal = (
            old_ticket % v400__JIT_LOGICAL_EXPERT_CTAS + 1
            == v400__JIT_LOGICAL_EXPERT_CTAS
        )
        tl.store(tle.gpu.local_ptr(terminal_smem, (0,)), terminal.to(tl.int32))
    tle.gpu.barrier_wait(route_handoff_barrier)
    v400__merge_four_local_top10_b64_compact_role(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        tokens,
        role_id,
    )


@triton.jit
def v400__publish_merge3(
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
    tle.gpu.barrier_wait(logits_ready_barrier)
    v400__publish_local_top10_b64_compact_role(
        logits_ptr, candidate_scratch_ptr, expert_base, token_base, tokens, 3
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)
    tle.gpu.barrier_wait(route_handoff_barrier)
    v400__merge_four_local_top10_b64_compact_role(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        tokens,
        3,
    )


@triton.jit
def v400__fused_qwen_h4096_t2048_b64_e128_compact_packed_kernel(
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
    rank = pid % v400__JIT_LOGICAL_EXPERT_CTAS
    token_tile = pid // v400__JIT_LOGICAL_EXPERT_CTAS
    token_base = token_tile * v400__JIT_BLOCK_TOKENS
    expert_base = rank * v400__JIT_EXPERTS_PER_CTA
    weight_smem = tle.gpu.alloc(
        [v400__JIT_PIPE_STAGES, v400__JIT_EXPERTS_PER_CTA, v400__JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    x_smem = tle.gpu.alloc(
        [v400__JIT_PIPE_STAGES, v400__JIT_BLOCK_TOKENS, v400__JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    pipe = tle.pipe(
        capacity=v400__JIT_PIPE_STAGES,
        scope="cta",
        name="qwen_h4096_t2048_b64_e128_compact_packed_v400",
        readers=("math0", "math1"),
        weight=weight_smem,
        x=x_smem,
    )
    terminal_smem = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    logits_ready_barrier = tle.gpu.alloc_barrier(arrive_count=v400__JIT_TOTAL_THREADS)
    candidates_ready_barrier = tle.gpu.alloc_barrier(
        arrive_count=v400__JIT_TOTAL_THREADS
    )
    route_handoff_barrier = tle.gpu.alloc_barrier(arrive_count=v400__JIT_TOTAL_THREADS)
    tle.gpu.warp_specialize(
        [
            (
                v400__loader_publish_merge0,
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
                v400__math_publish_arrive_merge,
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
                    1,
                ),
            ),
            (
                v400__math_publish_arrive_merge,
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
                    2,
                ),
            ),
            (
                v400__publish_merge3,
                (
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
        ],
        [4, 4, 4],
        [v400__JIT_MATH_MAXNREG, v400__JIT_MATH_MAXNREG, v400__JIT_ROUTE_MAXNREG],
    )


KERNEL = v400__fused_qwen_h4096_t2048_b64_e128_compact_packed_kernel
MAXNREG = v400__KERNEL_MAXNREG
SOURCE_LEAF = "v400"
__all__ = ["KERNEL", "MAXNREG", "SOURCE_LEAF"]
