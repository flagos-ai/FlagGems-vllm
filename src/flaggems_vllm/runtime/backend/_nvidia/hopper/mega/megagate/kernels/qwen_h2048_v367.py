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

"""Compile-light exact T16384/B128 Qwen H2048 E256 migration.

This leaf isolates the power-of-two CUDA-v032 schedule from v363's
non-power token-fragment machinery.  It keeps one native N128 accumulator per
math warpgroup and one N128 TensorDescriptor, so the TLE compiler does not
clone the 64+8/32/16 fragment graph or its per-fragment softmax tail.

One CTA owns E256.  The resource-fixed schedule mirrors the GPU-closed v369
topology: an independent loader plus four balanced single-E64 math
warpgroups.  A single packed E256 weight field replaces four E64 pipe fields
and avoids the split-field descriptor lowering defect observed in the old
dual+single topology.  Accumulators are stored in their native
``[expert, token]`` layout, avoiding the MMA-to-blocked transpose scratch.
All five roles join one CTA-local logits barrier and then reuse their twenty
physical warps to route disjoint token rows.
This leaf retains the CUDA-v032/v369 schedule provenance.  The profile entry
and operator tests own SM90 correctness, lifecycle, resource, and performance
admission for the exact shape dispatched here.
"""

from __future__ import annotations

from typing import Final

import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language.extra import libdevice

from . import qwen_h2048_helpers as _packed
from . import qwen_h2048_helpers as _v370_helpers
from . import qwen_h2048_v363 as _v363
from .routing_primitives import ordered_score_key as _ordered_score_key
from .routing_primitives import subwarp8_sum_f32 as _subwarp8_sum_f32
from .routing_primitives import warp_broadcast_f32 as _warp_broadcast_f32
from .routing_primitives import warp_max_u32 as _warp_max_u32
from .routing_primitives import warp_min_u32 as _warp_min_u32

TOKENS: Final = 16384
HIDDEN: Final = 2048
EXPERTS: Final = 256
TOPK: Final = 8
LOCAL_TOPK: Final = 8
BLOCK_TOKENS: Final = 128
EXPERTS_PER_MATH: Final = 64
BLOCK_K: Final = 128
# Packing the four E64 weight fields into one E256 field prevents the SM90
# backend from dropping the first B descriptor initialization in later math
# roles.  Keep the same proven two-stage SPMC topology as v369; native-layout
# stores remove the old 29 KB transpose scratch from this footprint.
PIPE_STAGES: Final = 2
NUM_K_TILES: Final = HIDDEN // BLOCK_K
EXPERT_ATOMS: Final = 4
PHYSICAL_MATH_ROLES: Final = 4
LOGICAL_ROLES: Final = 1 + PHYSICAL_MATH_ROLES
TOTAL_THREADS: Final = LOGICAL_ROLES * 128
LOADER_MAXNREG: Final = 32
MATH_MAXNREG: Final = 112
KERNEL_MAXNREG: Final = 96
PHYSICAL_WARPS_PER_ROLE: Final = 4
TOKENS_PER_ROUTE_ROUND: Final = LOGICAL_ROLES * PHYSICAL_WARPS_PER_ROLE
NUM_ROUTE_ROUNDS: Final = (
    BLOCK_TOKENS + TOKENS_PER_ROUTE_ROUND - 1
) // TOKENS_PER_ROUTE_ROUND

assert (HIDDEN, EXPERTS, TOPK) == (2048, 256, 8)
assert EXPERTS == EXPERT_ATOMS * EXPERTS_PER_MATH
assert NUM_K_TILES == 16
assert TOTAL_THREADS == 640
assert TOKENS_PER_ROUTE_ROUND == 20
assert NUM_ROUTE_ROUNDS == 7
assert _v363.TOKEN_TO_BLOCK[TOKENS] == BLOCK_TOKENS

JIT_EXPERTS = tl.constexpr(EXPERTS)
JIT_TOPK = tl.constexpr(TOPK)
JIT_LOCAL_TOPK = tl.constexpr(LOCAL_TOPK)
JIT_BLOCK_TOKENS = tl.constexpr(BLOCK_TOKENS)
JIT_EXPERTS_PER_MATH = tl.constexpr(EXPERTS_PER_MATH)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_NUM_K_TILES = tl.constexpr(NUM_K_TILES)
JIT_LOGICAL_ROLES = tl.constexpr(LOGICAL_ROLES)
JIT_TOTAL_THREADS = tl.constexpr(TOTAL_THREADS)
JIT_MATH_MAXNREG = tl.constexpr(MATH_MAXNREG)
JIT_PHYSICAL_WARPS_PER_ROLE = tl.constexpr(PHYSICAL_WARPS_PER_ROLE)
JIT_TOKENS_PER_ROUTE_ROUND = tl.constexpr(TOKENS_PER_ROUTE_ROUND)
JIT_NUM_ROUTE_ROUNDS = tl.constexpr(NUM_ROUTE_ROUNDS)
JIT_TIE_MASK = tl.constexpr(0xFFFF)


SCHEDULE: Final = {
    "backend": "tle_sm90_v367_qwen_h2048_e256_t16384_b128_compile_light",
    "lineage": "v363_without_nonpower_fragment_graph",
    "source_cuda": "v032_qwen_h2048_single_producer_direct_finalize",
    "kernel_count": 1,
    "tokens": TOKENS,
    "block_tokens": BLOCK_TOKENS,
    "experts_per_cta": EXPERTS,
    "experts_per_math_warpgroup": EXPERTS_PER_MATH,
    "math_warpgroups": PHYSICAL_MATH_ROLES,
    "logical_expert_atoms": EXPERT_ATOMS,
    "threads": TOTAL_THREADS,
    "route_loop": "five_roles_x_four_warps_direct_finalize",
    "cluster_size": 1,
    "pipe_stages": PIPE_STAGES,
    "x_fragments": (BLOCK_TOKENS,),
    "score": "bf16_boundary_stable_top8_selected_logit_renorm",
    "candidate_workspace": False,
    "workspace_contract": "full_fp32_logits_no_candidate_workspace",
    "padding": False,
    "fallback": False,
    "resource_fix": (
        "packed_e256_weight_field_native_layout_store_removes_transpose_scratch"
    ),
    "lowering_workaround": (
        "avoid_split_field_first_b_descriptor_mis_lowering_without_patching_flagtree"
    ),
    "thread_fix": "independent_loader_plus_four_balanced_single_math_wg",
    "release_status": "formal_2026_11_11_no_graph_admitted",
}


@triton.jit
def _load_ticket_b128(
    writer,
    weight_desc,
    x_desc,
    token_base,
    ticket,
):
    slot = writer.acquire(ticket)
    k_base = ticket * JIT_BLOCK_K
    tle.gpu.copy(
        weight_desc,
        slot.weight,
        [JIT_EXPERTS, JIT_BLOCK_K],
        [0, k_base],
    )
    tle.gpu.copy(
        x_desc,
        slot.x,
        [JIT_BLOCK_TOKENS, JIT_BLOCK_K],
        [token_base, k_base],
    )
    writer.commit(ticket)


@triton.jit
def _load_b128(writer, weight_desc, x_desc, token_base):
    for ticket in tl.range(0, JIT_NUM_K_TILES):
        _load_ticket_b128(writer, weight_desc, x_desc, token_base, ticket)


@triton.jit
def _issue_native_b128(
    stage,
    accumulator,
    MATH_ID: tl.constexpr,
):
    weight_atom = stage.slot.weight.subslice(
        MATH_ID * JIT_EXPERTS_PER_MATH,
        JIT_EXPERTS_PER_MATH,
        0,
    )
    accumulator = tle.gpu.wgmma(weight_atom, stage.slot.x, accumulator, trans_b=True)
    return accumulator


@triton.jit
def _gemm_native_b128(
    reader,
    logits_ptr,
    token_base,
    expert_base,
    MATH_ID: tl.constexpr,
):
    accumulator = tl.zeros((JIT_EXPERTS_PER_MATH, JIT_BLOCK_TOKENS), dtype=tl.float32)
    for ticket in tl.range(0, JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        accumulator = _issue_native_b128(stage, accumulator, MATH_ID)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        reader.release(ticket)

    token = token_base + tl.arange(0, JIT_BLOCK_TOKENS)[None, :]
    expert = expert_base + tl.arange(0, JIT_EXPERTS_PER_MATH)[:, None]
    # Preserve the WGMMA accumulator's native [expert, token] layout.  A
    # transpose here forces an MMA-to-blocked conversion through shared
    # memory.  Apply the production BF16 logits boundary before publication.
    bounded = accumulator.to(tl.bfloat16).to(tl.float32)
    tl.store(
        logits_ptr + token * JIT_EXPERTS + expert,
        bounded,
    )


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
def _route_rows(
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    logits_barrier,
    token_base,
    ROLE_ID: tl.constexpr,
):
    tle.gpu.barrier_wait(logits_barrier)
    route = tl.arange(0, 128)
    routing_warp = route // 32
    lane = route % 32
    lane_u32 = lane.to(tl.uint32)
    invalid_expert = tl.full((128,), 0xFFFFFFFF, dtype=tl.uint32)

    # The loader and four math roles contribute four physical warps each.
    # Twenty rows are finalized per round and seven rounds cover B128.
    for owner_round in tl.range(0, JIT_NUM_ROUTE_ROUNDS, loop_unroll_factor=1):
        local_token = (
            ROLE_ID * JIT_PHYSICAL_WARPS_PER_ROLE
            + routing_warp
            + owner_round * JIT_TOKENS_PER_ROUTE_ROUND
        )
        token = token_base + local_token
        valid_token = local_token < JIT_BLOCK_TOKENS
        score_row = token * JIT_EXPERTS
        local_expert_base = lane * 4

        s0, s1, s2, s3 = _v370_helpers._load_score_float4(
            logits_ptr + score_row + local_expert_base,
            valid_token,
        )
        s4, s5, s6, s7 = _v370_helpers._load_score_float4(
            logits_ptr + score_row + local_expert_base + 2 * JIT_EXPERTS_PER_MATH,
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
            expert1 = (2 * JIT_EXPERTS_PER_MATH + lane_u32 * 4 + offset1).to(tl.uint32)
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
        denominator = _warp_broadcast_f32(partial, tl.zeros((128,), dtype=tl.uint32))
        denominator += 1.0e-20
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
def _loader_route0(
    writer,
    weight_desc,
    x_desc,
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    logits_barrier,
    token_base,
):
    _load_b128(writer, weight_desc, x_desc, token_base)
    _route_rows(
        logits_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        logits_barrier,
        token_base,
        0,
    )


@triton.jit
def _math_route(
    reader,
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    logits_barrier,
    token_base,
    expert_base,
    MATH_ID: tl.constexpr,
    ROLE_ID: tl.constexpr,
):
    _gemm_native_b128(reader, logits_ptr, token_base, expert_base, MATH_ID)
    _route_rows(
        logits_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        logits_barrier,
        token_base,
        ROLE_ID,
    )


@triton.jit
def _fused_t16384_b128_kernel(
    weight_desc,
    x_desc,
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
):
    token_base = tl.program_id(0) * JIT_BLOCK_TOKENS

    weight_smem = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_EXPERTS, JIT_BLOCK_K],
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
        name="v367_qwen_h2048_e256_t16384_b128",
        readers=("math0", "math1", "math2", "math3"),
        weight=weight_smem,
        x=x_smem,
    )
    logits_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_TOTAL_THREADS)

    tle.gpu.warp_specialize(
        [
            (
                _loader_route0,
                (
                    pipe.writer(),
                    weight_desc,
                    x_desc,
                    logits_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    logits_barrier,
                    token_base,
                ),
            ),
            (
                _math_route,
                (
                    pipe.reader(name="math0", fields=("weight", "x")),
                    logits_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    logits_barrier,
                    token_base,
                    0,
                    0,
                    1,
                ),
            ),
            (
                _math_route,
                (
                    pipe.reader(name="math1", fields=("weight", "x")),
                    logits_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    logits_barrier,
                    token_base,
                    JIT_EXPERTS_PER_MATH,
                    1,
                    2,
                ),
            ),
            (
                _math_route,
                (
                    pipe.reader(name="math2", fields=("weight", "x")),
                    logits_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    logits_barrier,
                    token_base,
                    2 * JIT_EXPERTS_PER_MATH,
                    2,
                    3,
                ),
            ),
            (
                _math_route,
                (
                    pipe.reader(name="math3", fields=("weight", "x")),
                    logits_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    logits_barrier,
                    token_base,
                    3 * JIT_EXPERTS_PER_MATH,
                    3,
                    4,
                ),
            ),
        ],
        [4, 4, 4, 4],
        [
            JIT_MATH_MAXNREG,
            JIT_MATH_MAXNREG,
            JIT_MATH_MAXNREG,
            JIT_MATH_MAXNREG,
        ],
    )
