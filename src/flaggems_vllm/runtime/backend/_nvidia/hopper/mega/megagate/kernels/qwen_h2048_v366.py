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

"""Qwen-H2048 v014c small6 ``B128/E64/S4`` cluster-4 migration.

The five shapes through T=2048 use the prepared-generic direct leaf: one TMA
role and one WGMMA role, with the math role routing 32 rows owned by its
hardware-cluster rank after a four-producer ticket/ready hand-off.  T=3104 is
the CUDA heuristic boundary and uses a distinct compile-time pipeline leaf
with one dedicated routing role.  The two JIT definitions intentionally do
not share a runtime device branch.

Both leaves materialize FP32 accumulators, apply Qwen's BF16 producer boundary
inside the shared stable softmax/top-8/selected-renorm row helper, and launch
exactly one fused device kernel.  No padding, arbitrary-T admission, or
fallback is provided.
"""

from __future__ import annotations

from typing import Final

import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language.extra import libdevice

from . import qwen_h2048_helpers as _packed
from . import qwen_h2048_helpers as _v370_helpers
from .routing_primitives import ordered_score_key as _ordered_score_key
from .routing_primitives import subwarp8_sum_f32 as _subwarp8_sum_f32
from .routing_primitives import warp_broadcast_f32 as _warp_broadcast_f32
from .routing_primitives import warp_max_u32 as _warp_max_u32
from .routing_primitives import warp_min_u32 as _warp_min_u32

HIDDEN: Final = 2048
EXPERTS: Final = 256
TOPK: Final = 8
LOCAL_TOPK: Final = 8
BLOCK_TOKENS: Final = 128
EXPERTS_PER_CTA: Final = 64
CLUSTER_SIZE: Final = 4
BLOCK_K: Final = 128
NUM_K_TILES: Final = HIDDEN // BLOCK_K
PIPE_STAGES: Final = 4
STATE_WORDS_PER_TILE: Final = 16
TICKET_WORD: Final = 4
READY_WORD: Final = 12
NUM_PRODUCERS_PER_TILE: Final = CLUSTER_SIZE
DIRECT_TOKENS: Final = (1034, 1035, 1036, 1352, 2048)
PIPELINE_TOKENS: Final = (3104,)
SUPPORTED_TOKENS: Final = DIRECT_TOKENS + PIPELINE_TOKENS
OWNER_TOKENS_PER_RANK: Final = BLOCK_TOKENS // CLUSTER_SIZE

THREADS_PER_ROLE: Final = 128
PHYSICAL_WARPS_PER_ROLE: Final = THREADS_PER_ROLE // 32
TOKENS_PER_ROUTE_ROUND: Final = CLUSTER_SIZE * PHYSICAL_WARPS_PER_ROLE
NUM_ROUTE_ROUNDS: Final = BLOCK_TOKENS // TOKENS_PER_ROUTE_ROUND
DIRECT_ROLES: Final = 2
PIPELINE_ROLES: Final = 3
DIRECT_THREADS: Final = DIRECT_ROLES * THREADS_PER_ROLE
PIPELINE_THREADS: Final = PIPELINE_ROLES * THREADS_PER_ROLE
MATH_MAXNREG: Final = 224
ROUTE_MAXNREG: Final = 96
KERNEL_MAXNREG: Final = 128
DIRECT_REGISTER_BUDGET: Final = THREADS_PER_ROLE * (32 + MATH_MAXNREG)
PIPELINE_REGISTER_BUDGET: Final = THREADS_PER_ROLE * (64 + MATH_MAXNREG + ROUTE_MAXNREG)
PIPE_BYTES_PER_STAGE: Final = EXPERTS_PER_CTA * BLOCK_K * 2 + BLOCK_TOKENS * BLOCK_K * 2
NOMINAL_PIPE_BYTES: Final = PIPE_STAGES * PIPE_BYTES_PER_STAGE

assert SUPPORTED_TOKENS == (1034, 1035, 1036, 1352, 2048, 3104)
assert (HIDDEN, EXPERTS, TOPK, LOCAL_TOPK) == (2048, 256, 8, 8)
assert (BLOCK_TOKENS, EXPERTS_PER_CTA, CLUSTER_SIZE) == (128, 64, 4)
assert (BLOCK_K, NUM_K_TILES, PIPE_STAGES) == (128, 16, 4)
assert OWNER_TOKENS_PER_RANK == 32
assert (PHYSICAL_WARPS_PER_ROLE, TOKENS_PER_ROUTE_ROUND) == (4, 16)
assert NUM_ROUTE_ROUNDS == 8
assert (DIRECT_THREADS, PIPELINE_THREADS) == (256, 384)
assert NOMINAL_PIPE_BYTES == 196_608
assert STATE_WORDS_PER_TILE * 8 == 128

JIT_EXPERTS = tl.constexpr(EXPERTS)
JIT_TOPK = tl.constexpr(TOPK)
JIT_LOCAL_TOPK = tl.constexpr(LOCAL_TOPK)
JIT_BLOCK_TOKENS = tl.constexpr(BLOCK_TOKENS)
JIT_EXPERTS_PER_CTA = tl.constexpr(EXPERTS_PER_CTA)
JIT_CLUSTER_SIZE = tl.constexpr(CLUSTER_SIZE)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_NUM_K_TILES = tl.constexpr(NUM_K_TILES)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_STATE_WORDS_PER_TILE = tl.constexpr(STATE_WORDS_PER_TILE)
JIT_TICKET_WORD = tl.constexpr(TICKET_WORD)
JIT_READY_WORD = tl.constexpr(READY_WORD)
JIT_NUM_PRODUCERS_PER_TILE = tl.constexpr(NUM_PRODUCERS_PER_TILE)
JIT_OWNER_TOKENS_PER_RANK = tl.constexpr(OWNER_TOKENS_PER_RANK)
JIT_PHYSICAL_WARPS_PER_ROLE = tl.constexpr(PHYSICAL_WARPS_PER_ROLE)
JIT_TOKENS_PER_ROUTE_ROUND = tl.constexpr(TOKENS_PER_ROUTE_ROUND)
JIT_NUM_ROUTE_ROUNDS = tl.constexpr(NUM_ROUTE_ROUNDS)
JIT_MATH_MAXNREG = tl.constexpr(MATH_MAXNREG)
JIT_ROUTE_MAXNREG = tl.constexpr(ROUTE_MAXNREG)

CLUSTER4_MESH = tle.device_mesh(
    tle.MeshConfig(block_cluster=[("cluster_x", CLUSTER_SIZE)])
)


@triton.jit
def _loader_role(writer, weight_desc, x_desc, expert_base, token_base):
    for ticket in tl.range(0, JIT_NUM_K_TILES):
        slot = writer.acquire(ticket)
        k_base = ticket * JIT_BLOCK_K
        tle.gpu.copy(
            weight_desc,
            slot.weight,
            [JIT_EXPERTS_PER_CTA, JIT_BLOCK_K],
            [expert_base, k_base],
        )
        tle.gpu.copy(
            x_desc,
            slot.x,
            [JIT_BLOCK_TOKENS, JIT_BLOCK_K],
            [token_base, k_base],
        )
        writer.commit(ticket)


@triton.jit
def _loader_direct_role(
    writer,
    weight_desc,
    x_desc,
    expert_base,
    token_base,
):
    _loader_role(writer, weight_desc, x_desc, expert_base, token_base)
    # CUDA v014c's direct leaf has both resident roles participate in the
    # hardware cluster barrier after the final score slice is published.
    tle.distributed_barrier()


@triton.jit
def _gemm_store(reader, logits_ptr, expert_base, token_base, tokens: tl.constexpr):
    accumulator = tl.zeros((JIT_EXPERTS_PER_CTA, JIT_BLOCK_TOKENS), dtype=tl.float32)
    first = reader.wait(0)
    accumulator = tle.gpu.wgmma(
        first.slot.weight,
        first.slot.x,
        accumulator,
        trans_b=True,
    )
    for ticket in tl.range(1, JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        accumulator = tle.gpu.wgmma(
            stage.slot.weight,
            stage.slot.x,
            accumulator,
            trans_b=True,
        )
        accumulator = tle.gpu.wgmma_wait(1, accumulator)
        reader.release(ticket - 1)
    accumulator = tle.gpu.wgmma_wait(0, accumulator)
    reader.release(JIT_NUM_K_TILES - 1)

    token = token_base + tl.arange(0, JIT_BLOCK_TOKENS)[:, None]
    expert = expert_base + tl.arange(0, JIT_EXPERTS_PER_CTA)[None, :]
    tl.store(
        logits_ptr + token * JIT_EXPERTS + expert,
        tl.trans(accumulator),
        mask=token < tokens,
    )


@triton.jit
def _publish_ticket_target(
    state_ptr,
    token_tile,
):
    state_base = token_tile * JIT_STATE_WORDS_PER_TILE
    ticket_ptr = state_ptr + state_base + JIT_TICKET_WORD
    ready_ptr = state_ptr + state_base + JIT_READY_WORD
    old_ticket = tl.atomic_add(ticket_ptr, 1, sem="release", scope="gpu")
    epoch_offset = old_ticket % JIT_NUM_PRODUCERS_PER_TILE
    target = old_ticket + (JIT_NUM_PRODUCERS_PER_TILE - epoch_offset)
    published_ticket = old_ticket + 1
    if epoch_offset + 1 == JIT_NUM_PRODUCERS_PER_TILE:
        acquired = tl.atomic_add(ticket_ptr, 0, sem="acquire", scope="gpu")
        while acquired != published_ticket:
            acquired = tl.atomic_add(ticket_ptr, 0, sem="acquire", scope="gpu")
        tl.atomic_xchg(ready_ptr, acquired, sem="release", scope="gpu")
    return target


@triton.jit
def _wait_target(
    state_ptr,
    token_tile,
    target,
):
    ready_ptr = state_ptr + token_tile * JIT_STATE_WORDS_PER_TILE + JIT_READY_WORD
    observed = tl.atomic_add(ready_ptr, 0, sem="acquire", scope="gpu")
    while observed != target:
        observed = tl.atomic_add(ready_ptr, 0, sem="acquire", scope="gpu")


@triton.jit
def _sorted_key(cursor, k0, k1, k2, k3):
    return tl.where(
        cursor == 0,
        k0,
        tl.where(
            cursor == 1,
            k1,
            tl.where(
                cursor == 2,
                k2,
                tl.where(
                    cursor == 3,
                    k3,
                    tl.zeros((128,), dtype=tl.uint32),
                ),
            ),
        ),
    )


@triton.jit
def _route_rank_rows(
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    rank,
    token_base,
    tokens: tl.constexpr,
):
    route = tl.arange(0, 128)
    routing_warp = route // 32
    lane = route % 32
    lane_u32 = lane.to(tl.uint32)
    invalid_expert = tl.full((128,), 0xFFFFFFFF, dtype=tl.uint32)

    # CUDA v014c assigns one token to each physical math warp.  Four cluster
    # ranks x four warps finalize sixteen rows per round, so B128 needs eight
    # rounds.  Every lane owns two sorted quartets (eight experts total).
    for owner_round in tl.range(0, JIT_NUM_ROUTE_ROUNDS, loop_unroll_factor=1):
        local_token = (
            rank * JIT_PHYSICAL_WARPS_PER_ROLE
            + routing_warp
            + owner_round * JIT_TOKENS_PER_ROUTE_ROUND
        )
        token = token_base + local_token
        valid_token = (local_token < JIT_BLOCK_TOKENS) & (token < tokens)
        score_row = token * JIT_EXPERTS
        local_expert_base = lane * 4

        s0, s1, s2, s3 = _v370_helpers._load_score_float4(
            logits_ptr + score_row + local_expert_base,
            valid_token,
        )
        s4, s5, s6, s7 = _v370_helpers._load_score_float4(
            logits_ptr + score_row + local_expert_base + 128,
            valid_token,
        )
        s0 = tl.where(s0 == s0, s0.to(tl.bfloat16).to(tl.float32), -float("inf"))
        s1 = tl.where(s1 == s1, s1.to(tl.bfloat16).to(tl.float32), -float("inf"))
        s2 = tl.where(s2 == s2, s2.to(tl.bfloat16).to(tl.float32), -float("inf"))
        s3 = tl.where(s3 == s3, s3.to(tl.bfloat16).to(tl.float32), -float("inf"))
        s4 = tl.where(s4 == s4, s4.to(tl.bfloat16).to(tl.float32), -float("inf"))
        s5 = tl.where(s5 == s5, s5.to(tl.bfloat16).to(tl.float32), -float("inf"))
        s6 = tl.where(s6 == s6, s6.to(tl.bfloat16).to(tl.float32), -float("inf"))
        s7 = tl.where(s7 == s7, s7.to(tl.bfloat16).to(tl.float32), -float("inf"))
        s0, s1, s2, s3, permutation0 = _packed._sort4_pack(s0, s1, s2, s3)
        s4, s5, s6, s7, permutation1 = _packed._sort4_pack(s4, s5, s6, s7)
        k0 = _ordered_score_key(s0).to(tl.uint32)
        k1 = _ordered_score_key(s1).to(tl.uint32)
        k2 = _ordered_score_key(s2).to(tl.uint32)
        k3 = _ordered_score_key(s3).to(tl.uint32)
        k4 = _ordered_score_key(s4).to(tl.uint32)
        k5 = _ordered_score_key(s5).to(tl.uint32)
        k6 = _ordered_score_key(s6).to(tl.uint32)
        k7 = _ordered_score_key(s7).to(tl.uint32)

        cursor0 = tl.zeros((128,), dtype=tl.uint32)
        cursor1 = tl.zeros((128,), dtype=tl.uint32)
        selected_score = tl.zeros((128,), dtype=tl.float32)
        selected_expert = tl.zeros((128,), dtype=tl.uint32)
        for output_idx in tl.static_range(0, JIT_TOPK):
            head0 = _packed._sorted_head(cursor0, s0, s1, s2, s3)
            head1 = _packed._sorted_head(cursor1, s4, s5, s6, s7)
            key0 = _sorted_key(cursor0, k0, k1, k2, k3)
            key1 = _sorted_key(cursor1, k4, k5, k6, k7)
            offset0 = (permutation0 >> (cursor0 * 2)) & 3
            offset1 = (permutation1 >> (cursor1 * 2)) & 3
            expert0 = (lane_u32 * 4 + offset0).to(tl.uint32)
            expert1 = (128 + lane_u32 * 4 + offset1).to(tl.uint32)
            take_second = (key1 > key0) | ((key1 == key0) & (expert1 < expert0))
            local_head = tl.where(take_second, head1, head0)
            local_expert = tl.where(take_second, expert1, expert0)
            local_key = tl.where(take_second, key1, key0)
            best_key = _warp_max_u32(local_key)
            best_expert = _warp_min_u32(
                tl.where(
                    local_key == best_key,
                    local_expert,
                    invalid_expert,
                )
            )
            source_lane = (best_expert & 127) >> 2
            best_score = _warp_broadcast_f32(local_head, source_lane)
            owns_output = lane == output_idx
            selected_score = tl.where(owns_output, best_score, selected_score)
            selected_expert = tl.where(owns_output, best_expert, selected_expert)
            owns_best = local_expert == best_expert
            cursor0 += (owns_best & ~take_second).to(tl.uint32)
            cursor1 += (owns_best & take_second).to(tl.uint32)

        top1 = _warp_broadcast_f32(selected_score, tl.zeros((128,), dtype=tl.uint32))
        active_output = valid_token & (lane < JIT_TOPK)
        selected_exp = tl.where(
            active_output & (top1 != -float("inf")),
            libdevice.exp(selected_score - top1),
            0.0,
        )
        partial = _subwarp8_sum_f32(selected_exp)
        denominator = (
            _warp_broadcast_f32(partial, tl.zeros((128,), dtype=tl.uint32)) + 1.0e-20
        )
        output_offset = token * JIT_TOPK + lane
        tl.store(
            topk_indices_ptr + output_offset,
            selected_expert.to(tl.int32),
            mask=active_output,
        )
        tl.store(
            topk_weights_ptr + output_offset,
            selected_exp / denominator,
            mask=active_output,
        )


@triton.jit
def _math_direct_role(
    reader,
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    rank,
    expert_base,
    token_base,
    tokens: tl.constexpr,
):
    _gemm_store(reader, logits_ptr, expert_base, token_base, tokens)
    tle.distributed_barrier()
    _route_rank_rows(
        logits_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        rank,
        token_base,
        tokens,
    )


@triton.jit
def _math_pipeline_role(
    reader,
    logits_ptr,
    state_ptr,
    target_smem,
    publish_barrier,
    route_start_barrier,
    token_tile,
    expert_base,
    token_base,
    tokens: tl.constexpr,
):
    _gemm_store(reader, logits_ptr, expert_base, token_base, tokens)
    tle.gpu.barrier_wait(publish_barrier)
    target = _publish_ticket_target(state_ptr, token_tile)
    tl.store(tle.gpu.local_ptr(target_smem, (0,)), target)
    tle.gpu.barrier_wait(route_start_barrier)


@triton.jit
def _pipeline_route_role(
    logits_ptr,
    state_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    target_smem,
    route_start_barrier,
    token_tile,
    rank,
    token_base,
    tokens: tl.constexpr,
):
    tle.gpu.barrier_wait(route_start_barrier)
    target = tl.load(tle.gpu.local_ptr(target_smem, (0,)))
    _wait_target(state_ptr, token_tile, target)
    _route_rank_rows(
        logits_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        rank,
        token_base,
        tokens,
    )


@triton.jit
def _fused_qwen_h2048_v014c_direct_kernel(
    weight_desc,
    x_desc,
    logits_ptr,
    state_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    tokens: tl.constexpr,
    mesh: tl.constexpr,
):
    pid = tl.program_id(0)
    rank = tle.shard_id(mesh, "cluster_x")
    token_tile = pid // JIT_CLUSTER_SIZE
    expert_base = rank * JIT_EXPERTS_PER_CTA
    token_base = token_tile * JIT_BLOCK_TOKENS

    weight_smem = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_EXPERTS_PER_CTA, JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    x_smem = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_BLOCK_TOKENS, JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    pipe = tle.pipe(
        capacity=JIT_PIPE_STAGES,
        scope="cta",
        name="qwen_h2048_v014c_b128_e64_direct_v366",
        weight=weight_smem,
        x=x_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _loader_direct_role,
                (pipe.writer(), weight_desc, x_desc, expert_base, token_base),
            ),
            (
                _math_direct_role,
                (
                    pipe.reader(),
                    logits_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    rank,
                    expert_base,
                    token_base,
                    tokens,
                ),
            ),
        ],
        [4],
        [JIT_MATH_MAXNREG],
    )


@triton.jit
def _fused_qwen_h2048_v014c_pipeline_kernel(
    weight_desc,
    x_desc,
    logits_ptr,
    state_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    tokens: tl.constexpr,
    mesh: tl.constexpr,
):
    pid = tl.program_id(0)
    rank = tle.shard_id(mesh, "cluster_x")
    token_tile = pid // JIT_CLUSTER_SIZE
    expert_base = rank * JIT_EXPERTS_PER_CTA
    token_base = token_tile * JIT_BLOCK_TOKENS

    weight_smem = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_EXPERTS_PER_CTA, JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    x_smem = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_BLOCK_TOKENS, JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    pipe = tle.pipe(
        capacity=JIT_PIPE_STAGES,
        scope="cta",
        name="qwen_h2048_v014c_b128_e64_pipeline_v366",
        weight=weight_smem,
        x=x_smem,
    )
    target_smem = tle.gpu.alloc(
        [1],
        dtype=tl.int64,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    publish_barrier = tle.gpu.alloc_barrier(arrive_count=128)
    route_start_barrier = tle.gpu.alloc_barrier(arrive_count=256)
    tle.gpu.warp_specialize(
        [
            (
                _loader_role,
                (pipe.writer(), weight_desc, x_desc, expert_base, token_base),
            ),
            (
                _math_pipeline_role,
                (
                    pipe.reader(),
                    logits_ptr,
                    state_ptr,
                    target_smem,
                    publish_barrier,
                    route_start_barrier,
                    token_tile,
                    expert_base,
                    token_base,
                    tokens,
                ),
            ),
            (
                _pipeline_route_role,
                (
                    logits_ptr,
                    state_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    target_smem,
                    route_start_barrier,
                    token_tile,
                    rank,
                    token_base,
                    tokens,
                ),
            ),
        ],
        [4, 4],
        [JIT_MATH_MAXNREG, JIT_ROUTE_MAXNREG],
    )
