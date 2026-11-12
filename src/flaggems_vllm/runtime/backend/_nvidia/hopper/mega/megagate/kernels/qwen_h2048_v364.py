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

"""B80/E128 clusterless TLE port of the Qwen-H2048 CUDA-v036 leaf.

The three admitted residue shapes form exactly 56 B80 token tiles on H800.
Each tile launches two independent CTAs, one for each E128 half.  A CTA has
one TMA role, two E64 WGMMA roles, and one routing role.  Each math role emits
the donor's native m64n80k16 accumulator and retires every K tile with
wait_group<0> before releasing the corresponding pipeline stage.

After both E64 math roles publish BF16-boundary logits, all 16 physical warps
in each producer CTA compress that CTA's E128 slice into a local top-8 stored
in a dedicated candidate scratch tensor.  The FP32 producer logits remain
complete.  Only then does one lane perform the GPU-scope release arrival.  The
second of the two expert CTAs becomes the owner, and all physical warps merge
the two sorted local lists for all 80 token rows with CUDA-v030's halfwarp-16
mapping.  The single kernel therefore
produces stable softmax top-8 and selected-score renormalization without a
grid barrier or a second launch.

This leaf retains the v036 schedule provenance.  The profile entry and
operator tests own SM90 correctness, lifecycle, resource, and performance
admission for the exact shapes dispatched here.
"""

from __future__ import annotations

from typing import Final

import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language.extra import libdevice

from . import qwen_h2048_helpers as _packed
from . import qwen_h2048_helpers as _v369_helpers
from . import qwen_h2048_helpers as _v370_helpers
from .routing_primitives import ordered_score_key as _ordered_score_key
from .routing_primitives import warp_broadcast_f32 as _warp_broadcast_f32
from .routing_primitives import warp_max_u32 as _warp_max_u32

HIDDEN: Final = 2048
EXPERTS: Final = 256
TOPK: Final = 8
LOCAL_TOPK: Final = 8
BLOCK_TOKENS: Final = 80
EXPERTS_PER_CTA: Final = 128
EXPERTS_PER_MATH: Final = 64
LOGICAL_EXPERT_CTAS: Final = 2
BLOCK_K: Final = 128
NUM_K_TILES: Final = HIDDEN // BLOCK_K
PIPE_STAGES: Final = 4
LOGICAL_ROLES: Final = 4
THREADS_PER_ROLE: Final = 128
TOTAL_THREADS: Final = 512
PHYSICAL_WARPS_PER_ROLE: Final = THREADS_PER_ROLE // 32
TOKENS_PER_ROUTE_ROUND: Final = LOGICAL_ROLES * PHYSICAL_WARPS_PER_ROLE
NUM_ROUTE_ROUNDS: Final = (
    BLOCK_TOKENS + TOKENS_PER_ROUTE_ROUND - 1
) // TOKENS_PER_ROUTE_ROUND
CANDIDATE_WORDS: Final = 4
CANDIDATE_RECORD_WORDS: Final = LOCAL_TOPK * CANDIDATE_WORDS
TOKENS_PER_TERMINAL_WARP: Final = 2
TERMINAL_TOKENS_PER_ROUND: Final = (
    LOGICAL_ROLES * PHYSICAL_WARPS_PER_ROLE * TOKENS_PER_TERMINAL_WARP
)
NUM_TERMINAL_MERGE_ROUNDS: Final = (
    BLOCK_TOKENS + TERMINAL_TOKENS_PER_ROUND - 1
) // TERMINAL_TOKENS_PER_ROUND
TOKENS_PER_WIDTH2_WARP: Final = 16
TERMINAL_MERGE_WARPS: Final = (
    BLOCK_TOKENS + TOKENS_PER_WIDTH2_WARP - 1
) // TOKENS_PER_WIDTH2_WARP
LOADER_MAXNREG: Final = 64
MATH_MAXNREG: Final = 176
ROUTE_MAXNREG: Final = 96
POST_GEMM_MAXNREG: Final = 96
CUDA_TMA_MAXNREG: Final = 40
KERNEL_MAXNREG: Final = 128
REGISTER_BUDGET: Final = THREADS_PER_ROLE * (
    LOADER_MAXNREG + 2 * MATH_MAXNREG + ROUTE_MAXNREG
)
STATE_WORDS_PER_TILE: Final = 16
TICKET_WORD: Final = 4
NUM_PRODUCERS_PER_TILE: Final = LOGICAL_EXPERT_CTAS
SUPPORTED_TOKENS: Final = (4434, 4435, 4436)

PIPE_BYTES_PER_STAGE: Final = EXPERTS_PER_CTA * BLOCK_K * 2 + BLOCK_TOKENS * BLOCK_K * 2
NOMINAL_PIPE_BYTES: Final = PIPE_STAGES * PIPE_BYTES_PER_STAGE

assert (HIDDEN, EXPERTS, TOPK, LOCAL_TOPK) == (2048, 256, 8, 8)
assert BLOCK_TOKENS == 80
assert EXPERTS_PER_CTA == 2 * EXPERTS_PER_MATH
assert EXPERTS == LOGICAL_EXPERT_CTAS * EXPERTS_PER_CTA
assert NUM_K_TILES == 16
assert TOTAL_THREADS == LOGICAL_ROLES * THREADS_PER_ROLE
assert TOTAL_THREADS == 512
assert PHYSICAL_WARPS_PER_ROLE == 4
assert TOKENS_PER_ROUTE_ROUND == 16
assert NUM_ROUTE_ROUNDS == 5
assert CANDIDATE_RECORD_WORDS == 32
assert CANDIDATE_RECORD_WORDS <= EXPERTS_PER_CTA
assert TERMINAL_TOKENS_PER_ROUND == 32
assert NUM_TERMINAL_MERGE_ROUNDS == 3
assert TERMINAL_MERGE_WARPS == 5
assert REGISTER_BUDGET == 65_536
assert PIPE_BYTES_PER_STAGE == 53_248
assert NOMINAL_PIPE_BYTES == 212_992

JIT_EXPERTS = tl.constexpr(EXPERTS)
JIT_HIDDEN = tl.constexpr(HIDDEN)
JIT_TOPK = tl.constexpr(TOPK)
JIT_LOCAL_TOPK = tl.constexpr(LOCAL_TOPK)
JIT_BLOCK_TOKENS = tl.constexpr(BLOCK_TOKENS)
JIT_EXPERTS_PER_CTA = tl.constexpr(EXPERTS_PER_CTA)
JIT_EXPERTS_PER_MATH = tl.constexpr(EXPERTS_PER_MATH)
JIT_LOGICAL_EXPERT_CTAS = tl.constexpr(LOGICAL_EXPERT_CTAS)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_NUM_K_TILES = tl.constexpr(NUM_K_TILES)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_LOGICAL_ROLES = tl.constexpr(LOGICAL_ROLES)
JIT_TOTAL_THREADS = tl.constexpr(TOTAL_THREADS)
JIT_PHYSICAL_WARPS_PER_ROLE = tl.constexpr(PHYSICAL_WARPS_PER_ROLE)
JIT_TOKENS_PER_ROUTE_ROUND = tl.constexpr(TOKENS_PER_ROUTE_ROUND)
JIT_NUM_ROUTE_ROUNDS = tl.constexpr(NUM_ROUTE_ROUNDS)
JIT_CANDIDATE_WORDS = tl.constexpr(CANDIDATE_WORDS)
JIT_CANDIDATE_RECORD_WORDS = tl.constexpr(CANDIDATE_RECORD_WORDS)
JIT_NUM_TERMINAL_MERGE_ROUNDS = tl.constexpr(NUM_TERMINAL_MERGE_ROUNDS)
JIT_TERMINAL_MERGE_WARPS = tl.constexpr(TERMINAL_MERGE_WARPS)
JIT_MATH_MAXNREG = tl.constexpr(MATH_MAXNREG)
JIT_ROUTE_MAXNREG = tl.constexpr(ROUTE_MAXNREG)
JIT_STATE_WORDS_PER_TILE = tl.constexpr(STATE_WORDS_PER_TILE)
JIT_TICKET_WORD = tl.constexpr(TICKET_WORD)
JIT_NUM_PRODUCERS_PER_TILE = tl.constexpr(NUM_PRODUCERS_PER_TILE)

CUDA_V030_PATCH: Final = (
    "v030_qwen_h2048_b80_e128_clusterless/"
    "0025-sm90-qwen-h2048-b80-e128-clusterless.patch"
)
CUDA_V030_SHA256: Final = (
    "85dfc676c58f29a350f0b8a782397972" "8c4b0a8976eaa5d1f9e3f90659737f24"
)

SCHEDULE: Final = {
    "backend": "tle_sm90_qwen_h2048_b80_e128_clusterless",
    "lineage": "cuda_v036_b80_e128_clusterless_native_m64n80k16",
    "release_status": "formal_2026_11_11_no_graph_admitted",
    "kernel_count": 1,
    "hidden": HIDDEN,
    "experts": EXPERTS,
    "topk": TOPK,
    "tokens": SUPPORTED_TOKENS,
    "block_tokens": BLOCK_TOKENS,
    "experts_per_cta": EXPERTS_PER_CTA,
    "logical_expert_ctas": LOGICAL_EXPERT_CTAS,
    "cluster_size": 1,
    "block_k": BLOCK_K,
    "pipe_stages": PIPE_STAGES,
    "physical_wgmma_shape": "m64n80k16",
    "wgmma_wait_group": 0,
    "weight_tma": "single_e128_then_e64_smem_subslice",
    "tma_copies_per_k_tile": 2,
    "role_layout": (
        "loader+local_top8+merge0 / math0+local_top8+merge1 / "
        "math1+local_top8+arrival+merge2 / local_top8+merge3"
    ),
    "route_loop": ("producer_local_top8_then_terminal_two_list_halfwarp16_merge"),
    "threads": TOTAL_THREADS,
    "register_budget": REGISTER_BUDGET,
    "nominal_pipe_bytes": NOMINAL_PIPE_BYTES,
    "score": "bf16_boundary_fp32_softmax_stable_top8_renorm",
    "candidate_abi": "separate_cuda_v030_16B_candidate_x8_per_e128",
    "candidate_workspace": "[T,2,32] fp32",
    "workspace_contract": "full_fp32_logits_plus_candidate_scratch",
    "synchronization": ("u32_gpu_release_inc_two_producer_last_arriver_then_acquire"),
    "padding": False,
    "fallback": False,
    "source_cuda_patch": CUDA_V030_PATCH,
    "source_cuda_patch_sha256": CUDA_V030_SHA256,
}


@triton.jit
def _setmaxnreg_dec_40():
    """Match CUDA-v030's register-light TMA warpgroup during GEMM."""

    lane = tl.arange(0, 128)
    return tl.inline_asm_elementwise(
        """
        {
            setmaxnreg.dec.sync.aligned.u32 40;
            mov.u32 $0, $1;
        }
        """,
        "=r,r,~{memory}",
        [lane.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _setmaxnreg_dec_96():
    """Return math registers before every role enters routing."""

    lane = tl.arange(0, 128)
    return tl.inline_asm_elementwise(
        """
        {
            setmaxnreg.dec.sync.aligned.u32 96;
            mov.u32 $0, $1;
        }
        """,
        "=r,r,~{memory}",
        [lane.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _setmaxnreg_inc_96():
    """Give the retired TMA warpgroup CUDA-v030's routing budget."""

    lane = tl.arange(0, 128)
    return tl.inline_asm_elementwise(
        """
        {
            setmaxnreg.inc.sync.aligned.u32 96;
            mov.u32 $0, $1;
        }
        """,
        "=r,r,~{memory}",
        [lane.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _packed_bf16_score_expert_key(score, expert):
    """One REDUX key: larger BF16 score, then smaller u8 expert."""

    score_rank = _ordered_score_key(score).to(tl.uint32) >> 16
    inverse_expert = 0xFF - expert.to(tl.uint32)
    return (score_rank << 8) | inverse_expert


@triton.jit
def _runtime_role_id(role_id: tl.constexpr):
    """Materialize a non-foldable runtime i32 for one shared helper ABI."""

    seed = tl.program_id(0).to(tl.uint32) * 0 + role_id
    return tl.inline_asm_elementwise(
        "mov.u32 $0, $1;",
        "=r,r,~{memory}",
        [seed],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    ).to(tl.int32)


@triton.jit
def _pack_candidate_word(score, expert):
    """Pack the BF16-boundary score and E256 expert into one u32.

    ``score`` has already crossed the model's BF16 logits boundary, so its
    FP32 low 16 bits are zero.  Reusing those bits for the eight-bit expert
    index preserves the score bit-for-bit while allowing the terminal merge
    to issue one scalar global load instead of a split pair of loads from the
    legacy 16-byte CUDA candidate record.
    """

    score_bits = score.to(tl.uint32, bitcast=True)
    return (score_bits & 0xFFFF0000) | (expert.to(tl.uint32) & 0xFF)


@triton.jit
def _store_packed_candidate_word(ptr, score, expert, valid):
    """Publish one packed candidate at the legacy 16-byte record stride."""

    address = ptr.to(tl.uint64)
    packed = _pack_candidate_word(score, expert)
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
def _load_packed_candidate_word(ptr, valid):
    """Load one packed candidate with CUDA's streaming cache policy."""

    address = ptr.to(tl.uint64)
    packed = tl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            .reg .b64 cache_hint;
            mov.u32 $0, 0xff8000ff;
            mov.u64 cache_hint, 0x12f0000000000000;
            setp.ne.u32 p, $2, 0;
            @p ld.weak.global.L1::no_allocate.L2::cache_hint.b32
                $0, [$1], cache_hint;
        }
        """,
        "=r,l,r,~{memory}",
        [address, valid.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    score_bits = packed & 0xFFFF0000
    return score_bits.to(tl.float32, bitcast=True), packed & 0xFF


@triton.jit
def _loader_publish_merge0(
    writer,
    weight_desc,
    x_desc,
    logits_ptr,
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    logits_ready_barrier,
    register_handoff_barrier,
    candidates_ready_barrier,
    route_handoff_barrier,
    token_base,
    expert_base,
    tokens: tl.constexpr,
):
    _ = _setmaxnreg_dec_40()
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

    tle.gpu.barrier_wait(logits_ready_barrier)
    _ = _setmaxnreg_inc_96()
    tle.gpu.barrier_wait(register_handoff_barrier)
    _publish_local_top8_role(
        logits_ptr,
        candidate_scratch_ptr,
        expert_base,
        token_base,
        tokens,
        _runtime_role_id(0),
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)
    tle.gpu.barrier_wait(route_handoff_barrier)
    _merge_two_local_top8_halfwarp16_role(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        tokens,
        0,
    )


@triton.jit
def _gemm_native_n80(
    reader,
    logits_ptr,
    token_base,
    expert_base,
    tokens: tl.constexpr,
    math_id: tl.constexpr,
):
    first = reader.wait(0)
    weight_half = first.slot.weight.subslice(
        math_id * JIT_EXPERTS_PER_MATH,
        JIT_EXPERTS_PER_MATH,
        0,
    )
    acc = tle.gpu.wgmma(weight_half, first.slot.x, trans_b=True, out_dtype=tl.float32)
    acc = tle.gpu.wgmma_wait(0, acc)
    reader.release(0)
    for ticket in tl.range(1, JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        weight_half = stage.slot.weight.subslice(
            math_id * JIT_EXPERTS_PER_MATH,
            JIT_EXPERTS_PER_MATH,
            0,
        )
        acc = tle.gpu.wgmma(weight_half, stage.slot.x, acc, trans_b=True)
        acc = tle.gpu.wgmma_wait(0, acc)
        reader.release(ticket)

    expert = (
        expert_base
        + math_id * JIT_EXPERTS_PER_MATH
        + tl.arange(0, JIT_EXPERTS_PER_MATH)[:, None]
    )
    token_lane = tl.arange(0, 128)[None, :]
    token = token_base + token_lane
    # Preserve the native [expert, token] accumulator layout and store raw
    # FP32.  The model's BF16 logits boundary is applied once by the publisher.
    tl.store(
        logits_ptr + token * JIT_EXPERTS + expert,
        acc,
        mask=(token_lane < JIT_BLOCK_TOKENS) & (token < tokens),
    )


@triton.jit(noinline=True)
def _publish_local_top8_role(
    logits_ptr,
    candidate_scratch_ptr,
    expert_base,
    token_base,
    tokens: tl.constexpr,
    role_id,
):
    """Compress one producer's E128 slice to a sorted local top-8."""

    route = tl.arange(0, 128)
    routing_warp = route // 32
    lane = route % 32
    lane_u32 = lane.to(tl.uint32)
    for owner_round in tl.range(0, JIT_NUM_ROUTE_ROUNDS, loop_unroll_factor=1):
        local_token = (
            role_id * JIT_PHYSICAL_WARPS_PER_ROLE
            + routing_warp
            + owner_round * JIT_TOKENS_PER_ROUTE_ROUND
        )
        token = token_base + local_token
        valid_token = (local_token < JIT_BLOCK_TOKENS) & (token < tokens)
        score_row = token * JIT_EXPERTS
        local_expert_base = expert_base + lane * 4
        s0, s1, s2, s3 = _v370_helpers._load_score_float4(
            logits_ptr + score_row + local_expert_base,
            valid_token,
        )
        # Match CUDA-v030: keep the GEMM store in FP32 and apply the model's
        # BF16 logits boundary once, here, immediately before ranking.
        s0 = s0.to(tl.bfloat16).to(tl.float32)
        s1 = s1.to(tl.bfloat16).to(tl.float32)
        s2 = s2.to(tl.bfloat16).to(tl.float32)
        s3 = s3.to(tl.bfloat16).to(tl.float32)
        s0 = tl.where(s0 == s0, s0, -float("inf"))
        s1 = tl.where(s1 == s1, s1, -float("inf"))
        s2 = tl.where(s2 == s2, s2, -float("inf"))
        s3 = tl.where(s3 == s3, s3, -float("inf"))
        s0, s1, s2, s3, permutation = _packed._sort4_pack(s0, s1, s2, s3)

        cursor = tl.zeros((128,), dtype=tl.uint32)
        selected_score = tl.zeros((128,), dtype=tl.float32)
        selected_expert = tl.zeros((128,), dtype=tl.uint32)
        for output_idx in tl.static_range(0, JIT_LOCAL_TOPK):
            head = _packed._sorted_head(cursor, s0, s1, s2, s3)
            offset = (permutation >> (cursor * 2)) & 3
            local_expert = (expert_base + lane_u32 * 4 + offset).to(tl.uint32)
            packed_key = _packed_bf16_score_expert_key(head, local_expert)
            best_packed = _warp_max_u32(packed_key)
            best_expert = 0xFF - (best_packed & 0xFF)
            source_lane = (best_expert - expert_base) >> 2
            best_score = _warp_broadcast_f32(head, source_lane)
            cursor += (local_expert == best_expert).to(tl.uint32)
            selected_score = tl.where(lane == output_idx, best_score, selected_score)
            selected_expert = tl.where(lane == output_idx, best_expert, selected_expert)

        candidate = (
            candidate_scratch_ptr
            + (token * JIT_LOGICAL_EXPERT_CTAS + expert_base // JIT_EXPERTS_PER_CTA)
            * JIT_CANDIDATE_RECORD_WORDS
            + lane * JIT_CANDIDATE_WORDS
        )
        _store_packed_candidate_word(
            candidate,
            selected_score,
            selected_expert,
            valid_token & (lane < JIT_LOCAL_TOPK),
        )


@triton.jit
def _merge_two_local_top8_width2_role(
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    token_base,
    tokens: tl.constexpr,
    role_id: tl.constexpr,
):
    """Merge the two producer-local lists with 16 tokens per warp."""

    if role_id * JIT_PHYSICAL_WARPS_PER_ROLE < JIT_TERMINAL_MERGE_WARPS:
        terminal = tl.load(tle.gpu.local_ptr(terminal_smem, (0,)))
        if terminal != 0:
            route = tl.arange(0, 128)
            routing_warp = route // 32
            lane = route % 32
            token_group = lane // 2
            subwarp_lane = lane % 2
            terminal_warp = role_id * JIT_PHYSICAL_WARPS_PER_ROLE + routing_warp
            active_warp = terminal_warp < JIT_TERMINAL_MERGE_WARPS
            local_token = terminal_warp * 16 + token_group
            token = token_base + local_token
            valid_token = (
                active_warp & (local_token < JIT_BLOCK_TOKENS) & (token < tokens)
            )
            safe_token = tl.where(valid_token, token, token_base)
            producer = subwarp_lane.to(tl.uint32)
            candidate = (
                candidate_scratch_ptr
                + (safe_token * JIT_LOGICAL_EXPERT_CTAS + producer)
                * JIT_CANDIDATE_RECORD_WORDS
            )
            head_score, head_expert = _load_packed_candidate_word(
                candidate, valid_token
            )
            head_key = _ordered_score_key(head_score).to(tl.uint32)
            cursor = tl.zeros((128,), dtype=tl.uint32)
            selected_score0 = tl.zeros((128,), dtype=tl.float32)
            selected_score1 = tl.zeros((128,), dtype=tl.float32)
            selected_score2 = tl.zeros((128,), dtype=tl.float32)
            selected_score3 = tl.zeros((128,), dtype=tl.float32)
            selected_expert0 = tl.zeros((128,), dtype=tl.uint32)
            selected_expert1 = tl.zeros((128,), dtype=tl.uint32)
            selected_expert2 = tl.zeros((128,), dtype=tl.uint32)
            selected_expert3 = tl.zeros((128,), dtype=tl.uint32)
            top1 = tl.zeros((128,), dtype=tl.float32)

            for output_idx in tl.static_range(0, JIT_TOPK):
                best_expert = _v369_helpers._pair2_stable_best_expert(
                    head_key, head_expert
                )
                source_list = best_expert // JIT_EXPERTS_PER_CTA
                best_score = _v369_helpers._pair2_broadcast_f32(head_score, source_list)
                if output_idx == 0:
                    top1 = best_score
                output_slot = output_idx // 2
                owns_output = subwarp_lane == (output_idx & 1)
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
                else:
                    selected_score3 = tl.where(owns_output, best_score, selected_score3)
                    selected_expert3 = tl.where(
                        owns_output, best_expert, selected_expert3
                    )
                owns_best = head_expert == best_expert
                if output_idx + 1 < JIT_TOPK:
                    cursor += owns_best.to(tl.uint32)
                    next_candidate = (
                        candidate_scratch_ptr
                        + (safe_token * JIT_LOGICAL_EXPERT_CTAS + producer)
                        * JIT_CANDIDATE_RECORD_WORDS
                        + cursor * JIT_CANDIDATE_WORDS
                    )
                    next_score, next_expert = _load_packed_candidate_word(
                        next_candidate, valid_token & owns_best
                    )
                    next_key = _ordered_score_key(next_score).to(tl.uint32)
                    head_score = tl.where(owns_best, next_score, head_score)
                    head_expert = tl.where(owns_best, next_expert, head_expert)
                    head_key = tl.where(owns_best, next_key, head_key)

            has_finite = top1 != -float("inf")
            exp0 = tl.where(has_finite, libdevice.exp(selected_score0 - top1), 0.0)
            exp1 = tl.where(has_finite, libdevice.exp(selected_score1 - top1), 0.0)
            exp2 = tl.where(has_finite, libdevice.exp(selected_score2 - top1), 0.0)
            exp3 = tl.where(has_finite, libdevice.exp(selected_score3 - top1), 0.0)
            local_sum = (exp0 + exp2) + (exp1 + exp3)
            denominator = _v369_helpers._pair2_sum_f32(local_sum) + 1.0e-20
            output_base = token * JIT_TOPK + subwarp_lane
            tl.store(
                topk_indices_ptr + output_base,
                selected_expert0.to(tl.int32),
                mask=valid_token,
            )
            tl.store(
                topk_weights_ptr + output_base,
                exp0 / denominator,
                mask=valid_token,
            )
            tl.store(
                topk_indices_ptr + output_base + 2,
                selected_expert1.to(tl.int32),
                mask=valid_token,
            )
            tl.store(
                topk_weights_ptr + output_base + 2,
                exp1 / denominator,
                mask=valid_token,
            )
            tl.store(
                topk_indices_ptr + output_base + 4,
                selected_expert2.to(tl.int32),
                mask=valid_token,
            )
            tl.store(
                topk_weights_ptr + output_base + 4,
                exp2 / denominator,
                mask=valid_token,
            )
            tl.store(
                topk_indices_ptr + output_base + 6,
                selected_expert3.to(tl.int32),
                mask=valid_token,
            )
            tl.store(
                topk_weights_ptr + output_base + 6,
                exp3 / denominator,
                mask=valid_token,
            )


@triton.jit
def _merge_two_local_top8_halfwarp16_role(
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    token_base,
    tokens: tl.constexpr,
    role_id: tl.constexpr,
):
    """Merge two lists for two tokens per warp, matching CUDA-v030."""

    terminal = tl.load(tle.gpu.local_ptr(terminal_smem, (0,)))
    if terminal != 0:
        route = tl.arange(0, 128)
        routing_warp = route // 32
        lane = route % 32
        token_half = lane // 16
        subwarp_lane = lane % 16
        producer = subwarp_lane.to(tl.uint32)
        producer_lane = subwarp_lane < JIT_NUM_PRODUCERS_PER_TILE
        invalid_expert = tl.full((128,), 0xFFFFFFFF, dtype=tl.uint32)

        # Sixteen warps x two tokens cover 32 rows per round.  The first two
        # rounds cover 64 rows.  Only roles 0 and 1 participate in the final
        # round, covering rows 64..79 exactly; roles 2 and 3 must not execute
        # an otherwise fully masked merge for rows 80..95.
        if role_id < 2:
            terminal_rounds: tl.constexpr = 3
        else:
            terminal_rounds: tl.constexpr = 2
        for owner_round in tl.range(0, terminal_rounds, loop_unroll_factor=1):
            local_token = role_id * 8 + routing_warp * 2 + token_half + owner_round * 32
            token = token_base + local_token
            valid_token = (local_token < JIT_BLOCK_TOKENS) & (token < tokens)
            safe_token = tl.where(valid_token, token, token_base)
            cursor = tl.zeros((128,), dtype=tl.uint32)
            selected_score = tl.zeros((128,), dtype=tl.float32)
            selected_expert = tl.zeros((128,), dtype=tl.uint32)

            candidate = (
                candidate_scratch_ptr
                + (safe_token * JIT_LOGICAL_EXPERT_CTAS + producer)
                * JIT_CANDIDATE_RECORD_WORDS
            )
            head_score, head_expert = _load_packed_candidate_word(
                candidate, valid_token & producer_lane
            )
            head_expert = tl.where(producer_lane, head_expert, invalid_expert)

            for output_idx in tl.static_range(0, JIT_TOPK):
                head_key = _ordered_score_key(head_score).to(tl.uint32)
                best_expert = _v370_helpers._halfwarp16_stable_best_expert(
                    head_key,
                    tl.where(producer_lane, head_expert, invalid_expert),
                )
                source_list = best_expert // JIT_EXPERTS_PER_CTA
                best_score = _v370_helpers._halfwarp16_broadcast_f32(
                    head_score, source_list
                )
                owns_best = producer_lane & (head_expert == best_expert)
                selected_score = tl.where(
                    subwarp_lane == output_idx,
                    best_score,
                    selected_score,
                )
                selected_expert = tl.where(
                    subwarp_lane == output_idx,
                    best_expert,
                    selected_expert,
                )
                if output_idx + 1 < JIT_TOPK:
                    cursor += owns_best.to(tl.uint32)
                    next_candidate = (
                        candidate_scratch_ptr
                        + (safe_token * JIT_LOGICAL_EXPERT_CTAS + producer)
                        * JIT_CANDIDATE_RECORD_WORDS
                        + cursor * JIT_CANDIDATE_WORDS
                    )
                    next_score, next_expert = _load_packed_candidate_word(
                        next_candidate, valid_token & owns_best
                    )
                    head_score = tl.where(owns_best, next_score, head_score)
                    head_expert = tl.where(owns_best, next_expert, head_expert)

            output_lane = subwarp_lane < JIT_TOPK
            top1 = _v370_helpers._halfwarp16_broadcast_f32(
                selected_score,
                tl.zeros((128,), dtype=tl.uint32),
            )
            has_finite = top1 != -float("inf")
            selected_exp = tl.where(
                output_lane & has_finite,
                libdevice.exp(selected_score - top1),
                0.0,
            )
            denominator = _v370_helpers._halfwarp16_sum_f32(selected_exp) + 1.0e-20
            output = token * JIT_TOPK + subwarp_lane
            store_mask = valid_token & output_lane
            tl.store(
                topk_indices_ptr + output,
                selected_expert.to(tl.int32),
                mask=store_mask,
            )
            tl.store(
                topk_weights_ptr + output,
                selected_exp / denominator,
                mask=store_mask,
            )


@triton.jit
def _ticket_arrive_release_inc2(ticket_ptr, active):
    """CUDA-v030's wrapping two-producer u32 release ticket."""

    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            mov.u32 $0, 0;
            setp.ne.u32 p, $2, 0;
            @p atom.release.gpu.global.inc.u32 $0, [$1], 1;
        }
        """,
        "=r,l,r,~{memory}",
        [ticket_ptr.to(tl.uint64), active.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _terminal_acquire_fence(active):
    """Acquire producer writes only in the last-arriving CTA."""

    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            setp.ne.u32 p, $1, 0;
            @p fence.acquire.gpu;
            mov.u32 $0, $1;
        }
        """,
        "=r,r,~{memory}",
        [active.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _math_publish_arrive_merge(
    reader,
    logits_ptr,
    candidate_scratch_ptr,
    state_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    logits_ready_barrier,
    register_handoff_barrier,
    candidates_ready_barrier,
    route_handoff_barrier,
    token_base,
    expert_base,
    token_tile,
    tokens: tl.constexpr,
    math_id: tl.constexpr,
    role_id: tl.constexpr,
):
    _gemm_native_n80(
        reader,
        logits_ptr,
        token_base,
        expert_base,
        tokens,
        math_id,
    )
    tle.gpu.barrier_wait(logits_ready_barrier)
    _ = _setmaxnreg_dec_96()
    tle.gpu.barrier_wait(register_handoff_barrier)
    _publish_local_top8_role(
        logits_ptr,
        candidate_scratch_ptr,
        expert_base,
        token_base,
        tokens,
        _runtime_role_id(role_id),
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)
    if math_id == 1:
        arrival_lane = tl.arange(0, 128)
        arrival_active = arrival_lane == 0
        ticket_ptr = state_ptr + token_tile * JIT_STATE_WORDS_PER_TILE + JIT_TICKET_WORD
        old_ticket = _ticket_arrive_release_inc2(ticket_ptr, arrival_active)
        terminal = arrival_active & (old_ticket == JIT_NUM_PRODUCERS_PER_TILE - 1)
        _ = _terminal_acquire_fence(terminal)
        terminal_index = tl.zeros((128,), dtype=tl.int32)
        tl.store(
            tle.gpu.local_ptr(terminal_smem, (terminal_index,)),
            terminal.to(tl.int32),
            mask=arrival_active,
        )
    tle.gpu.barrier_wait(route_handoff_barrier)
    _merge_two_local_top8_halfwarp16_role(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        tokens,
        role_id,
    )


@triton.jit
def _publish_merge3(
    logits_ptr,
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    logits_ready_barrier,
    register_handoff_barrier,
    candidates_ready_barrier,
    route_handoff_barrier,
    token_base,
    expert_base,
    tokens: tl.constexpr,
):
    tle.gpu.barrier_wait(logits_ready_barrier)
    tle.gpu.barrier_wait(register_handoff_barrier)
    _publish_local_top8_role(
        logits_ptr,
        candidate_scratch_ptr,
        expert_base,
        token_base,
        tokens,
        _runtime_role_id(3),
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)
    tle.gpu.barrier_wait(route_handoff_barrier)
    _merge_two_local_top8_halfwarp16_role(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        tokens,
        3,
    )


@triton.jit
def _fused_qwen_h2048_b80_e128_clusterless_kernel(
    weight_desc,
    x_ptr,
    logits_ptr,
    candidate_scratch_ptr,
    state_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    tokens: tl.constexpr,
):
    pid = tl.program_id(0)
    logical_rank = pid % JIT_LOGICAL_EXPERT_CTAS
    token_tile = pid // JIT_LOGICAL_EXPERT_CTAS
    expert_base = logical_rank * JIT_EXPERTS_PER_CTA
    token_base = token_tile * JIT_BLOCK_TOKENS

    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        [tokens, JIT_HIDDEN],
        [JIT_HIDDEN, 1],
        [JIT_BLOCK_TOKENS, JIT_BLOCK_K],
    )

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
        name="qwen_h2048_b80_e128_clusterless_v364",
        readers=("math0", "math1"),
        weight=weight_smem,
        x=x_smem,
    )
    terminal_smem = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    logits_ready_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_TOTAL_THREADS)
    register_handoff_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_TOTAL_THREADS)
    candidates_ready_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_TOTAL_THREADS)
    route_handoff_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_TOTAL_THREADS)

    tle.gpu.warp_specialize(
        [
            (
                _loader_publish_merge0,
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
                    register_handoff_barrier,
                    candidates_ready_barrier,
                    route_handoff_barrier,
                    token_base,
                    expert_base,
                    tokens,
                ),
            ),
            (
                _math_publish_arrive_merge,
                (
                    pipe.reader(name="math0", fields=("weight", "x")),
                    logits_ptr,
                    candidate_scratch_ptr,
                    state_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    terminal_smem,
                    logits_ready_barrier,
                    register_handoff_barrier,
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
                _math_publish_arrive_merge,
                (
                    pipe.reader(name="math1", fields=("weight", "x")),
                    logits_ptr,
                    candidate_scratch_ptr,
                    state_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    terminal_smem,
                    logits_ready_barrier,
                    register_handoff_barrier,
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
                _publish_merge3,
                (
                    logits_ptr,
                    candidate_scratch_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    terminal_smem,
                    logits_ready_barrier,
                    register_handoff_barrier,
                    candidates_ready_barrier,
                    route_handoff_barrier,
                    token_base,
                    expert_base,
                    tokens,
                ),
            ),
        ],
        [4, 4, 4],
        [JIT_MATH_MAXNREG, JIT_MATH_MAXNREG, JIT_ROUTE_MAXNREG],
    )
