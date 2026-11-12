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

"""Single-CTA E256 direct-finalize migration for Qwen H2048/K8.

This module migrates the CUDA v032/v033/v034 leaf topology used by the six
large-token Qwen H2048/E256/K8 winners.  One CTA owns the complete 256-expert
row.  The resource-fixed implementation uses an independent loader plus four
balanced E64 math roles, a packed E256 weight field, and a two-stage pipe.
All five roles route disjoint token rows with their physical warps after the
CTA-local logits barrier.  There is no cluster, global ticket, candidate
workspace, or second kernel.

``BLOCK_TOKENS`` is a launch-time ``tl.constexpr``.  The public factory admits
only the exact CUDA-provenance leaves B72/B96/B112/B128, so Triton emits a
separate device specialization for every schedule without a runtime device
branch.  GPU compile/correctness/performance admission remains a separate
gate; this source is intentionally not release-allowlisted yet.
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
EXPERTS_PER_MATH: Final = 64
BLOCK_K: Final = 128
PIPE_STAGES: Final = 2
NUM_K_TILES: Final = HIDDEN // BLOCK_K
EXPERT_ATOMS: Final = 4
PHYSICAL_MATH_ROLES: Final = 4
LOGICAL_ROLES: Final = 1 + PHYSICAL_MATH_ROLES
THREADS_PER_ROLE: Final = 128
PHYSICAL_WARPS_PER_ROLE: Final = THREADS_PER_ROLE // 32
TOKENS_PER_ROUTE_ROUND: Final = LOGICAL_ROLES * PHYSICAL_WARPS_PER_ROLE
TOTAL_THREADS: Final = LOGICAL_ROLES * THREADS_PER_ROLE
LOADER_MAXNREG: Final = 64
SINGLE_MATH_MAXNREG: Final = 112
KERNEL_MAXNREG: Final = 96
NOMINAL_REGISTER_BUDGET: Final = THREADS_PER_ROLE * (
    LOADER_MAXNREG + 4 * SINGLE_MATH_MAXNREG
)

B72_TOKENS: Final = (9309, 9330)
B96_TOKENS: Final = (11469,)
B112_TOKENS: Final = (13421, 13422)
B128_TOKENS: Final = (16384,)
SUPPORTED_TOKENS: Final = (
    *B72_TOKENS,
    *B96_TOKENS,
    *B112_TOKENS,
    *B128_TOKENS,
)
TOKEN_TO_BLOCK: Final = {
    **{tokens: 72 for tokens in B72_TOKENS},
    **{tokens: 96 for tokens in B96_TOKENS},
    **{tokens: 112 for tokens in B112_TOKENS},
    **{tokens: 128 for tokens in B128_TOKENS},
}
BLOCK_TO_FRAGMENTS: Final = {
    72: (64, 8, 8, 2),
    96: (64, 32, 8, 2),
    112: (64, 32, 16, 3),
    128: (64, 64, 8, 2),
}

assert (HIDDEN, EXPERTS, TOPK, LOCAL_TOPK) == (2048, 256, 8, 8)
assert EXPERTS == EXPERT_ATOMS * EXPERTS_PER_MATH
assert NUM_K_TILES == 16
assert TOTAL_THREADS == 640
assert (PHYSICAL_WARPS_PER_ROLE, TOKENS_PER_ROUTE_ROUND) == (4, 20)
assert NOMINAL_REGISTER_BUDGET == 65_536
assert tuple(sorted(TOKEN_TO_BLOCK)) == tuple(sorted(SUPPORTED_TOKENS))
assert set(TOKEN_TO_BLOCK.values()) == {72, 96, 112, 128}
assert {
    block: frag0 + frag1 + (frag2 if count == 3 else 0)
    for block, (frag0, frag1, frag2, count) in BLOCK_TO_FRAGMENTS.items()
} == {72: 72, 96: 96, 112: 112, 128: 128}

JIT_EXPERTS = tl.constexpr(EXPERTS)
JIT_TOPK = tl.constexpr(TOPK)
JIT_LOCAL_TOPK = tl.constexpr(LOCAL_TOPK)
JIT_EXPERTS_PER_MATH = tl.constexpr(EXPERTS_PER_MATH)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_NUM_K_TILES = tl.constexpr(NUM_K_TILES)
JIT_LOGICAL_ROLES = tl.constexpr(LOGICAL_ROLES)
JIT_TOTAL_THREADS = tl.constexpr(TOTAL_THREADS)
JIT_PHYSICAL_WARPS_PER_ROLE = tl.constexpr(PHYSICAL_WARPS_PER_ROLE)
JIT_TOKENS_PER_ROUTE_ROUND = tl.constexpr(TOKENS_PER_ROUTE_ROUND)
JIT_SINGLE_MATH_MAXNREG = tl.constexpr(SINGLE_MATH_MAXNREG)


@triton.jit
def _load_e256_fragment_ticket(
    writer,
    weight_desc,
    x0_desc,
    x1_desc,
    x2_desc,
    token_base,
    FRAG0: tl.constexpr,
    FRAG1: tl.constexpr,
    FRAG2: tl.constexpr,
    NUM_FRAGMENTS: tl.constexpr,
    ticket,
):
    """TMA-load only power-of-two token fragments.

    TensorDescriptor rejects the CUDA leaves' native B72/B96/B112 block
    shapes.  The exact compile-time decomposition is 64+8, 64+32, and
    64+32+16 respectively; B128 uses 64+64.  This preserves the CUDA token
    tile/grid while satisfying the descriptor and native WGMMA contracts.
    """

    slot = writer.acquire(ticket)
    k_base = ticket * JIT_BLOCK_K
    tle.gpu.copy(
        weight_desc,
        slot.weight,
        [JIT_EXPERTS, JIT_BLOCK_K],
        [0, k_base],
    )
    tle.gpu.copy(
        x0_desc,
        slot.x0,
        [FRAG0, JIT_BLOCK_K],
        [token_base, k_base],
    )
    tle.gpu.copy(
        x1_desc,
        slot.x1,
        [FRAG1, JIT_BLOCK_K],
        [token_base + FRAG0, k_base],
    )
    if NUM_FRAGMENTS == 3:
        tle.gpu.copy(
            x2_desc,
            slot.x2,
            [FRAG2, JIT_BLOCK_K],
            [token_base + FRAG0 + FRAG1, k_base],
        )
    else:
        # The recent FlagTree pipe verifier requires every declared field to
        # have a proven producer before commit.  NUM_FRAGMENTS==2 prunes all
        # x2 readers, so one dummy local write satisfies that contract without
        # issuing an otherwise useless third-fragment TMA transaction.
        zero = tl.zeros((1,), dtype=tl.int32)
        tl.store(
            tle.gpu.local_ptr(slot.x2, (zero, zero)),
            tl.zeros((1,), dtype=tl.bfloat16),
        )
    writer.commit(ticket)


@triton.jit
def _load_e256_fragments(
    writer,
    weight_desc,
    x0_desc,
    x1_desc,
    x2_desc,
    token_base,
    FRAG0: tl.constexpr,
    FRAG1: tl.constexpr,
    FRAG2: tl.constexpr,
    NUM_FRAGMENTS: tl.constexpr,
):
    for ticket in tl.range(0, JIT_NUM_K_TILES):
        _load_e256_fragment_ticket(
            writer,
            weight_desc,
            x0_desc,
            x1_desc,
            x2_desc,
            token_base,
            FRAG0,
            FRAG1,
            FRAG2,
            NUM_FRAGMENTS,
            ticket,
        )


@triton.jit
def _issue_fragmented_math(
    stage,
    accumulator0,
    accumulator1,
    accumulator2,
    MATH_ID: tl.constexpr,
    NUM_FRAGMENTS: tl.constexpr,
):
    weight_atom = stage.slot.weight.subslice(
        MATH_ID * JIT_EXPERTS_PER_MATH,
        JIT_EXPERTS_PER_MATH,
        0,
    )
    accumulator0 = tle.gpu.wgmma(weight_atom, stage.slot.x0, accumulator0, trans_b=True)
    accumulator1 = tle.gpu.wgmma(weight_atom, stage.slot.x1, accumulator1, trans_b=True)
    if NUM_FRAGMENTS == 3:
        accumulator2 = tle.gpu.wgmma(
            weight_atom,
            stage.slot.x2,
            accumulator2,
            trans_b=True,
        )
    return accumulator0, accumulator1, accumulator2


@triton.jit
def _gemm_e64_fragmented(
    reader,
    logits_ptr,
    token_base,
    expert_base,
    TOKENS,
    FRAG0: tl.constexpr,
    FRAG1: tl.constexpr,
    FRAG2: tl.constexpr,
    NUM_FRAGMENTS: tl.constexpr,
    MATH_ID: tl.constexpr,
):
    accumulator0 = tl.zeros((JIT_EXPERTS_PER_MATH, FRAG0), dtype=tl.float32)
    accumulator1 = tl.zeros((JIT_EXPERTS_PER_MATH, FRAG1), dtype=tl.float32)
    accumulator2 = tl.zeros((JIT_EXPERTS_PER_MATH, FRAG2), dtype=tl.float32)

    for ticket in tl.range(0, JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        accumulator0, accumulator1, accumulator2 = _issue_fragmented_math(
            stage,
            accumulator0,
            accumulator1,
            accumulator2,
            MATH_ID,
            NUM_FRAGMENTS,
        )
        accumulator0 = tle.gpu.wgmma_wait(0, accumulator0)
        accumulator1 = tle.gpu.wgmma_wait(0, accumulator1)
        if NUM_FRAGMENTS == 3:
            accumulator2 = tle.gpu.wgmma_wait(0, accumulator2)
        reader.release(ticket)

    expert = expert_base + tl.arange(0, JIT_EXPERTS_PER_MATH)[:, None]
    token0 = token_base + tl.arange(0, FRAG0)[None, :]
    tl.store(
        logits_ptr + token0 * JIT_EXPERTS + expert,
        accumulator0.to(tl.bfloat16).to(tl.float32),
        mask=token0 < TOKENS,
    )
    token1 = token_base + FRAG0 + tl.arange(0, FRAG1)[None, :]
    tl.store(
        logits_ptr + token1 * JIT_EXPERTS + expert,
        accumulator1.to(tl.bfloat16).to(tl.float32),
        mask=token1 < TOKENS,
    )
    if NUM_FRAGMENTS == 3:
        token2 = token_base + FRAG0 + FRAG1 + tl.arange(0, FRAG2)[None, :]
        tl.store(
            logits_ptr + token2 * JIT_EXPERTS + expert,
            accumulator2.to(tl.bfloat16).to(tl.float32),
            mask=token2 < TOKENS,
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
def _route_role_rows(
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    logits_barrier,
    token_base,
    TOKENS,
    BLOCK_TOKENS: tl.constexpr,
    ROLE_ID: tl.constexpr,
):
    tle.gpu.barrier_wait(logits_barrier)
    route = tl.arange(0, 128)
    routing_warp = route // 32
    lane = route % 32
    lane_u32 = lane.to(tl.uint32)
    invalid_expert = tl.full((128,), 0xFFFFFFFF, dtype=tl.uint32)
    num_route_rounds: tl.constexpr = (
        BLOCK_TOKENS + JIT_TOKENS_PER_ROUTE_ROUND - 1
    ) // JIT_TOKENS_PER_ROUTE_ROUND

    # Four roles contribute four independent physical warps each.  Sixteen
    # rows are finalized per round; each lane owns two sorted expert quartets.
    for owner_round in tl.range(0, num_route_rounds, loop_unroll_factor=1):
        local_token = (
            ROLE_ID * JIT_PHYSICAL_WARPS_PER_ROLE
            + routing_warp
            + owner_round * JIT_TOKENS_PER_ROUTE_ROUND
        )
        token = token_base + local_token
        valid_token = (local_token < BLOCK_TOKENS) & (token < TOKENS)
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
        # GEMM publication already rounded each accumulator to BF16 before
        # storing it as FP32, so repeating the BF16 round-trip here is both
        # redundant and instruction-heavy in the routing hot loop.
        s0 = tl.where(s0 == s0, s0, -float("inf"))
        s1 = tl.where(s1 == s1, s1, -float("inf"))
        s2 = tl.where(s2 == s2, s2, -float("inf"))
        s3 = tl.where(s3 == s3, s3, -float("inf"))
        s4 = tl.where(s4 == s4, s4, -float("inf"))
        s5 = tl.where(s5 == s5, s5, -float("inf"))
        s6 = tl.where(s6 == s6, s6, -float("inf"))
        s7 = tl.where(s7 == s7, s7, -float("inf"))
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
                tl.where(local_key == best_key, local_expert, invalid_expert)
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
def _fragmented_loader_and_route_role(
    writer,
    weight_desc,
    x0_desc,
    x1_desc,
    x2_desc,
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    logits_barrier,
    token_base,
    TOKENS,
    BLOCK_TOKENS: tl.constexpr,
    FRAG0: tl.constexpr,
    FRAG1: tl.constexpr,
    FRAG2: tl.constexpr,
    NUM_FRAGMENTS: tl.constexpr,
):
    _load_e256_fragments(
        writer,
        weight_desc,
        x0_desc,
        x1_desc,
        x2_desc,
        token_base,
        FRAG0,
        FRAG1,
        FRAG2,
        NUM_FRAGMENTS,
    )
    _route_role_rows(
        logits_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        logits_barrier,
        token_base,
        TOKENS,
        BLOCK_TOKENS,
        0,
    )


@triton.jit
def _fragmented_math_and_route_role(
    reader,
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    logits_barrier,
    token_base,
    expert_base,
    TOKENS,
    BLOCK_TOKENS: tl.constexpr,
    FRAG0: tl.constexpr,
    FRAG1: tl.constexpr,
    FRAG2: tl.constexpr,
    NUM_FRAGMENTS: tl.constexpr,
    MATH_ID: tl.constexpr,
    ROLE_ID: tl.constexpr,
):
    _gemm_e64_fragmented(
        reader,
        logits_ptr,
        token_base,
        expert_base,
        TOKENS,
        FRAG0,
        FRAG1,
        FRAG2,
        NUM_FRAGMENTS,
        MATH_ID,
    )
    _route_role_rows(
        logits_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        logits_barrier,
        token_base,
        TOKENS,
        BLOCK_TOKENS,
        ROLE_ID,
    )


@triton.jit
def _fused_qwen_h2048_e256_direct_finalize_kernel(
    weight_desc,
    x0_desc,
    x1_desc,
    x2_desc,
    logits_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    TOKENS,
    BLOCK_TOKENS: tl.constexpr,
    FRAG0: tl.constexpr,
    FRAG1: tl.constexpr,
    FRAG2: tl.constexpr,
    NUM_FRAGMENTS: tl.constexpr,
):
    token_tile = tl.program_id(0)
    token_base = token_tile * BLOCK_TOKENS

    weight_smem = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_EXPERTS, JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    x0_smem = tle.gpu.alloc(
        [JIT_PIPE_STAGES, FRAG0, JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    x1_smem = tle.gpu.alloc(
        [JIT_PIPE_STAGES, FRAG1, JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    x2_smem = tle.gpu.alloc(
        [JIT_PIPE_STAGES, FRAG2, JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    pipe = tle.pipe(
        capacity=JIT_PIPE_STAGES,
        scope="cta",
        name="v363_qwen_h2048_e256_direct_finalize",
        readers=("math0", "math1", "math2", "math3"),
        weight=weight_smem,
        x0=x0_smem,
        x1=x1_smem,
        x2=x2_smem,
    )
    logits_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_TOTAL_THREADS)

    tle.gpu.warp_specialize(
        [
            (
                _fragmented_loader_and_route_role,
                (
                    pipe.writer(),
                    weight_desc,
                    x0_desc,
                    x1_desc,
                    x2_desc,
                    logits_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    logits_barrier,
                    token_base,
                    TOKENS,
                    BLOCK_TOKENS,
                    FRAG0,
                    FRAG1,
                    FRAG2,
                    NUM_FRAGMENTS,
                ),
            ),
            (
                _fragmented_math_and_route_role,
                (
                    pipe.reader(
                        name="math0",
                        fields=("weight", "x0", "x1", "x2"),
                    ),
                    logits_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    logits_barrier,
                    token_base,
                    0,
                    TOKENS,
                    BLOCK_TOKENS,
                    FRAG0,
                    FRAG1,
                    FRAG2,
                    NUM_FRAGMENTS,
                    0,
                    1,
                ),
            ),
            (
                _fragmented_math_and_route_role,
                (
                    pipe.reader(name="math1", fields=("weight", "x0", "x1", "x2")),
                    logits_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    logits_barrier,
                    token_base,
                    JIT_EXPERTS_PER_MATH,
                    TOKENS,
                    BLOCK_TOKENS,
                    FRAG0,
                    FRAG1,
                    FRAG2,
                    NUM_FRAGMENTS,
                    1,
                    2,
                ),
            ),
            (
                _fragmented_math_and_route_role,
                (
                    pipe.reader(name="math2", fields=("weight", "x0", "x1", "x2")),
                    logits_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    logits_barrier,
                    token_base,
                    2 * JIT_EXPERTS_PER_MATH,
                    TOKENS,
                    BLOCK_TOKENS,
                    FRAG0,
                    FRAG1,
                    FRAG2,
                    NUM_FRAGMENTS,
                    2,
                    3,
                ),
            ),
            (
                _fragmented_math_and_route_role,
                (
                    pipe.reader(name="math3", fields=("weight", "x0", "x1", "x2")),
                    logits_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    logits_barrier,
                    token_base,
                    3 * JIT_EXPERTS_PER_MATH,
                    TOKENS,
                    BLOCK_TOKENS,
                    FRAG0,
                    FRAG1,
                    FRAG2,
                    NUM_FRAGMENTS,
                    3,
                    4,
                ),
            ),
        ],
        [4, 4, 4, 4],
        [
            JIT_SINGLE_MATH_MAXNREG,
            JIT_SINGLE_MATH_MAXNREG,
            JIT_SINGLE_MATH_MAXNREG,
            JIT_SINGLE_MATH_MAXNREG,
        ],
    )
