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


@triton.jit
def routing__ordered_score_key(score):
    score = tl.where(score == score, score, -float("inf"))
    bits = score.to(tl.uint32, bitcast=True)
    bits = tl.where(score == 0.0, 0, bits)
    return tl.where(bits & 2147483648 != 0, ~bits, bits ^ 2147483648)


@triton.jit
def routing__subwarp8_sum_f32(value):
    """Reduce four independent eight-lane groups in one physical warp."""
    return tl.inline_asm_elementwise(
        "\n        {\n            .reg .f32 current, peer;\n            .reg .b32 current_bits, peer_bits;\n            mov.f32 current, $1;\n\n            mov.b32 current_bits, current;\n            shfl.sync.down.b32 peer_bits, current_bits, 4, 0x1807, 0xffffffff;\n            mov.b32 peer, peer_bits;\n            add.rn.f32 current, current, peer;\n\n            mov.b32 current_bits, current;\n            shfl.sync.down.b32 peer_bits, current_bits, 2, 0x1807, 0xffffffff;\n            mov.b32 peer, peer_bits;\n            add.rn.f32 current, current, peer;\n\n            mov.b32 current_bits, current;\n            shfl.sync.down.b32 peer_bits, current_bits, 1, 0x1807, 0xffffffff;\n            mov.b32 peer, peer_bits;\n            add.rn.f32 $0, current, peer;\n        }\n        ",
        "=f,f",
        [value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def routing__warp_max_u32(value):
    return tl.inline_asm_elementwise(
        "redux.sync.max.u32 $0, $1, 0xffffffff;",
        "=r,r",
        [value.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def routing__warp_min_u32(value):
    return tl.inline_asm_elementwise(
        "redux.sync.min.u32 $0, $1, 0xffffffff;",
        "=r,r",
        [value.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def routing__warp_broadcast_f32(value, source_lane):
    return tl.inline_asm_elementwise(
        "\n        {\n            .reg .b32 input_bits, output_bits;\n            mov.b32 input_bits, $1;\n            shfl.sync.idx.b32 output_bits, input_bits, $2, 0x1f, 0xffffffff;\n            mov.b32 $0, output_bits;\n        }\n        ",
        "=f,f,r",
        [value, source_lane.to(tl.uint32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def v166__compare_swap_local(lhs_score, lhs_offset, rhs_score, rhs_offset):
    take_rhs = (rhs_score > lhs_score) | (rhs_score == lhs_score) & (
        rhs_offset < lhs_offset
    )
    return (
        tl.where(take_rhs, rhs_score, lhs_score),
        tl.where(take_rhs, rhs_offset, lhs_offset),
        tl.where(take_rhs, lhs_score, rhs_score),
        tl.where(take_rhs, lhs_offset, rhs_offset),
    )


@triton.jit
def v166__sort4_pack(s0, s1, s2, s3):
    """CUDA sort4 network with offsets retired into one packed register."""
    i0 = tl.zeros((128,), dtype=tl.uint32)
    i1 = tl.full((128,), 1, dtype=tl.uint32)
    i2 = tl.full((128,), 2, dtype=tl.uint32)
    i3 = tl.full((128,), 3, dtype=tl.uint32)
    s0, i0, s1, i1 = v166__compare_swap_local(s0, i0, s1, i1)
    s2, i2, s3, i3 = v166__compare_swap_local(s2, i2, s3, i3)
    s0, i0, s2, i2 = v166__compare_swap_local(s0, i0, s2, i2)
    s1, i1, s3, i3 = v166__compare_swap_local(s1, i1, s3, i3)
    s1, i1, s2, i2 = v166__compare_swap_local(s1, i1, s2, i2)
    permutation = i0 | i1 << 2 | i2 << 4 | i3 << 6
    return (s0, s1, s2, s3, permutation)


@triton.jit
def v166__sorted_head(cursor, s0, s1, s2, s3):
    return tl.where(
        cursor == 0,
        s0,
        tl.where(
            cursor == 1,
            s1,
            tl.where(cursor == 2, s2, tl.where(cursor == 3, s3, -float("inf"))),
        ),
    )


@triton.jit
def cluster8_common__load_ranked_wave(
    logits_ptr, score_row, lane, wave_base: tl.constexpr
):
    """Load native BF16 logits and retain v172's packed local sort4."""
    expert_base = wave_base + lane * 4
    e0 = expert_base + 0
    e1 = expert_base + 1
    e2 = expert_base + 2
    e3 = expert_base + 3
    s0 = tl.load(logits_ptr + score_row + e0).to(tl.float32)
    s1 = tl.load(logits_ptr + score_row + e1).to(tl.float32)
    s2 = tl.load(logits_ptr + score_row + e2).to(tl.float32)
    s3 = tl.load(logits_ptr + score_row + e3).to(tl.float32)
    s0 = tl.where(s0 == s0, s0, -float("inf"))
    s1 = tl.where(s1 == s1, s1, -float("inf"))
    s2 = tl.where(s2 == s2, s2, -float("inf"))
    s3 = tl.where(s3 == s3, s3, -float("inf"))
    return v166__sort4_pack(s0, s1, s2, s3)


large_common__HIDDEN = 4096
large_common__EXPERTS_PER_CTA = 64
large_common__BLOCK_K = 128
large_common__NUM_K_TILES = large_common__HIDDEN // large_common__BLOCK_K
large_common__JIT_EXPERTS_PER_CTA = tl.constexpr(large_common__EXPERTS_PER_CTA)
large_common__JIT_BLOCK_K = tl.constexpr(large_common__BLOCK_K)
large_common__JIT_NUM_K_TILES = tl.constexpr(large_common__NUM_K_TILES)


@triton.jit
def large_common__stage4_loader_role(
    writer, weight_desc, x_desc, expert_base, token_base, block_tokens: tl.constexpr
):
    """v164/v196 serial loader specialized from H5120 to H4096."""
    for ticket in tl.range(0, large_common__JIT_NUM_K_TILES):
        slot = writer.acquire(ticket)
        k_offset = ticket * large_common__JIT_BLOCK_K
        tle.gpu.copy(
            weight_desc,
            slot.weight,
            [large_common__JIT_EXPERTS_PER_CTA, large_common__JIT_BLOCK_K],
            [expert_base, k_offset],
        )
        tle.gpu.copy(
            x_desc,
            slot.x,
            [block_tokens, large_common__JIT_BLOCK_K],
            [token_base, k_offset],
        )
        writer.commit(ticket)
    tle.distributed_barrier()


v371__EXPERTS = 512
v371__TOPK = 10
v371__BLOCK_TOKENS = 128
v371__EXPERTS_PER_CTA = 64
v371__BLOCK_K = 128
v371__NUM_K_TILES = 32
v371__PIPE_STAGES = 4
v371__V014_CLUSTER_SIZE = 8
v371__V014_MATH_MAXNREG = 240
v371__V014_KERNEL_MAXNREG = v371__V014_MATH_MAXNREG
v371__V014_ROUTING_WARPS = 4
v371__V014_TOKENS_PER_OWNER_ROUND = v371__V014_CLUSTER_SIZE * v371__V014_ROUTING_WARPS
v371__V014_OWNER_ROUNDS = v371__BLOCK_TOKENS // v371__V014_TOKENS_PER_OWNER_ROUND
v371__JIT_EXPERTS = tl.constexpr(v371__EXPERTS)
v371__JIT_TOPK = tl.constexpr(v371__TOPK)
v371__JIT_BLOCK_TOKENS = tl.constexpr(v371__BLOCK_TOKENS)
v371__JIT_EXPERTS_PER_CTA = tl.constexpr(v371__EXPERTS_PER_CTA)
v371__JIT_BLOCK_K = tl.constexpr(v371__BLOCK_K)
v371__JIT_NUM_K_TILES = tl.constexpr(v371__NUM_K_TILES)
v371__JIT_PIPE_STAGES = tl.constexpr(v371__PIPE_STAGES)
v371__JIT_V014_CLUSTER_SIZE = tl.constexpr(v371__V014_CLUSTER_SIZE)
v371__JIT_V014_ROUTING_WARPS = tl.constexpr(v371__V014_ROUTING_WARPS)
v371__JIT_V014_TOKENS_PER_OWNER_ROUND = tl.constexpr(v371__V014_TOKENS_PER_OWNER_ROUND)
v371__JIT_V014_OWNER_ROUNDS = tl.constexpr(v371__V014_OWNER_ROUNDS)
v371__JIT_V014_MATH_MAXNREG = tl.constexpr(v371__V014_MATH_MAXNREG)


@triton.jit
def v371__v014_route_packed_runtime_owner_masked(
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    token_base,
    rank,
    tokens: tl.constexpr,
):
    """CUDA-v014 cluster8 packed owner with a masked residue tile."""
    route = tl.arange(0, 128)
    routing_warp = route // 32
    lane = route % 32
    lane_u32 = lane.to(tl.uint32)
    invalid_expert = tl.full((128,), 4294967295, dtype=tl.uint32)
    one = tl.full((128,), 1, dtype=tl.uint32)
    first_owned_token = token_base + rank * v371__JIT_V014_ROUTING_WARPS
    remaining_owned_tokens = tl.maximum(tokens - first_owned_token, 0)
    active_owner_rounds = tl.minimum(
        (remaining_owned_tokens + v371__JIT_V014_TOKENS_PER_OWNER_ROUND - 1)
        // v371__JIT_V014_TOKENS_PER_OWNER_ROUND,
        v371__JIT_V014_OWNER_ROUNDS,
    )
    for owner_round in tl.range(0, active_owner_rounds):
        local_token = (
            rank * v371__JIT_V014_ROUTING_WARPS
            + routing_warp
            + owner_round * v371__JIT_V014_TOKENS_PER_OWNER_ROUND
        )
        global_token = token_base + local_token
        valid_token = global_token < tokens
        safe_token = tl.where(valid_token, global_token, token_base)
        score_row = safe_token * v371__JIT_EXPERTS
        s00, s01, s02, s03, permutation0 = cluster8_common__load_ranked_wave(
            logits_ptr, score_row, lane, 0
        )
        s10, s11, s12, s13, permutation1 = cluster8_common__load_ranked_wave(
            logits_ptr, score_row, lane, 128
        )
        s20, s21, s22, s23, permutation2 = cluster8_common__load_ranked_wave(
            logits_ptr, score_row, lane, 256
        )
        s30, s31, s32, s33, permutation3 = cluster8_common__load_ranked_wave(
            logits_ptr, score_row, lane, 384
        )
        packed_cursors = tl.zeros((128,), dtype=tl.uint32)
        selected_for_lane = tl.zeros((128,), dtype=tl.int32)
        lane_expert_base = lane_u32 * 4
        for output_idx in tl.static_range(0, v371__JIT_TOPK):
            cursor0 = packed_cursors & 7
            cursor1 = packed_cursors >> 3 & 7
            cursor2 = packed_cursors >> 6 & 7
            cursor3 = packed_cursors >> 9 & 7
            head0 = v166__sorted_head(cursor0, s00, s01, s02, s03)
            head1 = v166__sorted_head(cursor1, s10, s11, s12, s13)
            head2 = v166__sorted_head(cursor2, s20, s21, s22, s23)
            head3 = v166__sorted_head(cursor3, s30, s31, s32, s33)
            offset0 = permutation0 >> cursor0 * 2 & 3
            offset1 = permutation1 >> cursor1 * 2 & 3
            offset2 = permutation2 >> cursor2 * 2 & 3
            offset3 = permutation3 >> cursor3 * 2 & 3
            expert0 = lane_expert_base + offset0
            expert1 = lane_expert_base + offset1 + 128
            expert2 = lane_expert_base + offset2 + 256
            expert3 = lane_expert_base + offset3 + 384
            take1 = head1 > head0
            local_score = tl.where(take1, head1, head0)
            local_expert = tl.where(take1, expert1, expert0)
            local_wave = tl.where(take1, 1, 0).to(tl.uint32)
            take2 = head2 > local_score
            local_score = tl.where(take2, head2, local_score)
            local_expert = tl.where(take2, expert2, local_expert)
            local_wave = tl.where(take2, 2, local_wave).to(tl.uint32)
            take3 = head3 > local_score
            local_score = tl.where(take3, head3, local_score)
            local_expert = tl.where(take3, expert3, local_expert)
            local_wave = tl.where(take3, 3, local_wave).to(tl.uint32)
            local_key = routing__ordered_score_key(local_score).to(tl.uint32)
            best_key = routing__warp_max_u32(local_key)
            best_expert = routing__warp_min_u32(
                tl.where(local_key == best_key, local_expert, invalid_expert)
            )
            owns_best = local_expert == best_expert
            packed_cursors += tl.where(
                owns_best, one << local_wave * 3, tl.zeros((128,), dtype=tl.uint32)
            )
            selected_for_lane = tl.where(
                lane == output_idx, best_expert.to(tl.int32), selected_for_lane
            )
        valid_output = lane < v371__JIT_TOPK
        selected_expert = tl.where(valid_output, selected_for_lane, 0)
        selected_logit = tl.load(
            logits_ptr + score_row + selected_expert,
            mask=valid_output,
            other=-float("inf"),
        ).to(tl.float32)
        selected_logit = tl.where(
            selected_logit == selected_logit, selected_logit, -float("inf")
        )
        top1 = routing__warp_broadcast_f32(
            selected_logit, tl.zeros((128,), dtype=tl.uint32)
        )
        has_finite_input = top1 != -float("inf")
        selected_exp = tl.where(
            valid_output & has_finite_input, libdevice.exp(selected_logit - top1), 0.0
        )
        partial = routing__subwarp8_sum_f32(selected_exp)
        denominator = routing__warp_broadcast_f32(
            partial, tl.zeros((128,), dtype=tl.uint32)
        )
        denominator += routing__warp_broadcast_f32(
            partial, tl.full((128,), 8, dtype=tl.uint32)
        )
        denominator += 1e-20
        output_offset = safe_token * v371__JIT_TOPK + lane
        store_mask = valid_token & valid_output
        tl.store(topk_indices_ptr + output_offset, selected_for_lane, mask=store_mask)
        tl.store(
            topk_weights_ptr + output_offset,
            selected_exp / denominator,
            mask=store_mask,
        )


@triton.jit
def v371__v014_math_and_route_role(
    reader,
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    token_base,
    expert_base,
    rank,
    tokens: tl.constexpr,
):
    """Retain the v014 B128/E64 producer and 32-warp cluster router."""
    accumulator = tl.zeros(
        (v371__JIT_EXPERTS_PER_CTA, v371__JIT_BLOCK_TOKENS), dtype=tl.float32
    )
    first = reader.wait(0)
    accumulator = tle.gpu.wgmma(
        first.slot.weight, first.slot.x, accumulator, trans_b=True
    )
    for ticket in tl.range(1, v371__JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        accumulator = tle.gpu.wgmma(
            stage.slot.weight, stage.slot.x, accumulator, trans_b=True
        )
        accumulator = tle.gpu.wgmma_wait(1, accumulator)
        reader.release(ticket - 1)
    accumulator = tle.gpu.wgmma_wait(0, accumulator)
    reader.release(v371__JIT_NUM_K_TILES - 1)
    token = token_base + tl.arange(0, v371__JIT_BLOCK_TOKENS)[:, None]
    expert = expert_base + tl.arange(0, v371__JIT_EXPERTS_PER_CTA)[None, :]
    bounded = tl.trans(accumulator).to(tl.bfloat16).to(tl.float32)
    tl.store(
        logits_ptr + token * v371__JIT_EXPERTS + expert, bounded, mask=token < tokens
    )
    tle.distributed_barrier()
    v371__v014_route_packed_runtime_owner_masked(
        logits_ptr, topk_indices_ptr, topk_weights_ptr, token_base, rank, tokens
    )


@triton.jit
def v371__fused_qwen_h4096_b128_e64_v014_cluster8_kernel(
    weight_desc,
    x_desc,
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    mesh: tl.constexpr,
    tokens: tl.constexpr,
):
    pid = tl.program_id(0)
    rank = tle.shard_id(mesh, "cluster_x")
    token_tile = pid // v371__JIT_V014_CLUSTER_SIZE
    token_base = token_tile * v371__JIT_BLOCK_TOKENS
    expert_base = rank * v371__JIT_EXPERTS_PER_CTA
    weight = tle.gpu.alloc(
        [v371__JIT_PIPE_STAGES, v371__JIT_EXPERTS_PER_CTA, v371__JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    x_smem = tle.gpu.alloc(
        [v371__JIT_PIPE_STAGES, v371__JIT_BLOCK_TOKENS, v371__JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    pipe = tle.pipe(
        capacity=v371__JIT_PIPE_STAGES,
        scope="cta",
        name="qwen_h4096_b128_e64_v014_cluster8_v371",
        readers=("math",),
        weight=weight,
        x=x_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                large_common__stage4_loader_role,
                (
                    pipe.writer(),
                    weight_desc,
                    x_desc,
                    expert_base,
                    token_base,
                    v371__JIT_BLOCK_TOKENS,
                ),
            ),
            (
                v371__v014_math_and_route_role,
                (
                    pipe.reader(name="math", fields=("weight", "x")),
                    logits_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    token_base,
                    expert_base,
                    rank,
                    tokens,
                ),
            ),
        ],
        [4],
        [v371__JIT_V014_MATH_MAXNREG],
    )


KERNEL = v371__fused_qwen_h4096_b128_e64_v014_cluster8_kernel
MAXNREG = v371__V014_KERNEL_MAXNREG
MESH = tle.device_mesh(tle.MeshConfig(block_cluster=[("cluster_x", 8)]))
SOURCE_LEAF = "v371"
__all__ = ["KERNEL", "MAXNREG", "MESH", "SOURCE_LEAF"]
