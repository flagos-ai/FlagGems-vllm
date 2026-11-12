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

"""Exact v018/v036 Qwen-H2048 E64 single-list TLE migration.

This is the clusterless ``B128/E64/S4`` leaf used by the three Qwen eager
shapes ``T={4107, 4108, 4138}``.  Four independently scheduled CTAs compute
one E64 producer slice each.  After the BF16 logits boundary, every producer
compresses its completed slice to one stable local top-8 list in a dedicated
candidate scratch tensor.  A
GPU-scope acq_rel ticket elects the fourth producer, whose three static roles
merge the four sorted lists and emit softmax-equivalent selected-score
renormalization.

The record ABI occupies sixteen FP32 scratch words per token and E64 producer:
eight logits followed by eight exactly representable expert ids.  Keeping that
record separate from the producer output preserves all 256 FP32 logits for
correctness oracles while retaining CUDA v018's one-list layout and v036's
removal of the dead second-list scalar chain.  TLE statically
partitions each 24-row routing round across three roles.  Each role
vectorizes eight rows, matching the CUDA donor's aggregate 12-warp x two-row
publication width without spelling its width-16 half-warp shuffle network.
The producer lists, last-arriver ownership, single fused launch, and model
result remain identical.

This leaf retains the v018/v036 schedule provenance.  The profile entry and
operator tests own SM90 correctness, lifecycle, resource, and performance
admission for the exact shapes dispatched here.
"""

from __future__ import annotations

from typing import Final

import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language.extra import libdevice

from .routing_primitives import ordered_score_key as _ordered_score_key

HIDDEN: Final = 2048
EXPERTS: Final = 256
TOPK: Final = 8
LOCAL_TOPK: Final = 8
BLOCK_TOKENS: Final = 128
EXPERTS_PER_CTA: Final = 64
LOGICAL_EXPERT_CTAS: Final = 4
BLOCK_K: Final = 128
NUM_K_TILES: Final = HIDDEN // BLOCK_K
PIPE_STAGES: Final = 4
LOGICAL_ROLES: Final = 3
TOKENS_PER_ROLE_BATCH: Final = 8
TOKENS_PER_ROUTE_ROUND: Final = LOGICAL_ROLES * TOKENS_PER_ROLE_BATCH
NUM_ROUTE_ROUNDS: Final = (
    BLOCK_TOKENS + TOKENS_PER_ROUTE_ROUND - 1
) // TOKENS_PER_ROUTE_ROUND
THREADS_PER_ROLE: Final = 128
TOTAL_THREADS: Final = LOGICAL_ROLES * THREADS_PER_ROLE
LOADER_MAXNREG: Final = 64
MATH_MAXNREG: Final = 224
ROUTE_MAXNREG: Final = 96
KERNEL_MAXNREG: Final = 128
REGISTER_BUDGET: Final = THREADS_PER_ROLE * (
    LOADER_MAXNREG + MATH_MAXNREG + ROUTE_MAXNREG
)
STATE_WORDS_PER_TILE: Final = 16
TICKET_WORD: Final = 4
NUM_PRODUCERS_PER_TILE: Final = LOGICAL_EXPERT_CTAS
SUPPORTED_TOKENS: Final = (4107, 4108, 4138)

# Each producer owns one independent record in candidate_scratch.
CANDIDATE_SCORE_WORD: Final = 0
CANDIDATE_EXPERT_WORD: Final = LOCAL_TOPK
CANDIDATE_RECORD_WORDS: Final = 2 * LOCAL_TOPK

PIPE_BYTES_PER_STAGE: Final = EXPERTS_PER_CTA * BLOCK_K * 2 + BLOCK_TOKENS * BLOCK_K * 2
NOMINAL_PIPE_BYTES: Final = PIPE_STAGES * PIPE_BYTES_PER_STAGE

assert (HIDDEN, EXPERTS, TOPK, LOCAL_TOPK) == (2048, 256, 8, 8)
assert (BLOCK_TOKENS, EXPERTS_PER_CTA, LOGICAL_EXPERT_CTAS) == (128, 64, 4)
assert (BLOCK_K, NUM_K_TILES, PIPE_STAGES) == (128, 16, 4)
assert TOTAL_THREADS == 384
assert (TOKENS_PER_ROLE_BATCH, TOKENS_PER_ROUTE_ROUND) == (8, 24)
assert NUM_ROUTE_ROUNDS == 6
assert REGISTER_BUDGET == KERNEL_MAXNREG * TOTAL_THREADS == 49_152
assert STATE_WORDS_PER_TILE * 8 == 128
assert CANDIDATE_RECORD_WORDS <= EXPERTS_PER_CTA
assert NOMINAL_PIPE_BYTES == 196_608

JIT_EXPERTS = tl.constexpr(EXPERTS)
JIT_TOPK = tl.constexpr(TOPK)
JIT_LOCAL_TOPK = tl.constexpr(LOCAL_TOPK)
JIT_BLOCK_TOKENS = tl.constexpr(BLOCK_TOKENS)
JIT_EXPERTS_PER_CTA = tl.constexpr(EXPERTS_PER_CTA)
JIT_LOGICAL_EXPERT_CTAS = tl.constexpr(LOGICAL_EXPERT_CTAS)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_NUM_K_TILES = tl.constexpr(NUM_K_TILES)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_LOGICAL_ROLES = tl.constexpr(LOGICAL_ROLES)
JIT_TOKENS_PER_ROLE_BATCH = tl.constexpr(TOKENS_PER_ROLE_BATCH)
JIT_TOKENS_PER_ROUTE_ROUND = tl.constexpr(TOKENS_PER_ROUTE_ROUND)
JIT_NUM_ROUTE_ROUNDS = tl.constexpr(NUM_ROUTE_ROUNDS)
JIT_TOTAL_THREADS = tl.constexpr(TOTAL_THREADS)
JIT_MATH_MAXNREG = tl.constexpr(MATH_MAXNREG)
JIT_ROUTE_MAXNREG = tl.constexpr(ROUTE_MAXNREG)
JIT_STATE_WORDS_PER_TILE = tl.constexpr(STATE_WORDS_PER_TILE)
JIT_TICKET_WORD = tl.constexpr(TICKET_WORD)
JIT_NUM_PRODUCERS_PER_TILE = tl.constexpr(NUM_PRODUCERS_PER_TILE)
JIT_CANDIDATE_SCORE_WORD = tl.constexpr(CANDIDATE_SCORE_WORD)
JIT_CANDIDATE_EXPERT_WORD = tl.constexpr(CANDIDATE_EXPERT_WORD)
JIT_CANDIDATE_RECORD_WORDS = tl.constexpr(CANDIDATE_RECORD_WORDS)
JIT_TIE_MASK = tl.constexpr(0xFFFF)

CUDA_V018_PATCH: Final = (
    "v018_qwen_halfwarp_two_token/" "0014-sm90-qwen-halfwarp-two-token.patch"
)
CUDA_V018_SHA256: Final = (
    "3849eedf22c41c9957fc62655195c33df" "1a78c7fb5f553a810ca7564286ee65b"
)
CUDA_V036_PATCH: Final = (
    "v036_qwen_h2048_restore_v018_leaf/" "0031-sm90-qwen-h2048-restore-v018-leaf.patch"
)
CUDA_V036_SHA256: Final = (
    "6136b5b25d5f14a2b03c11a7865900e3" "d75e4dd712c0a1c13576accfd9a8a1e4"
)

SCHEDULE: Final = {
    "backend": "tle_sm90_qwen_h2048_v018_v036_single_list",
    "lineage": "cuda_v018_halfwarp_two_token_plus_v036_single_list_restore",
    "release_status": "formal_2026_11_11_no_graph_admitted",
    "kernel_count": 1,
    "device_kernel_count": 1,
    "runtime_kernel_launch_count": 1,
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
    "roles": ("tma_then_route", "wgmma_then_publish_route", "route"),
    "tokens_per_role_batch": TOKENS_PER_ROLE_BATCH,
    "tokens_per_route_round": TOKENS_PER_ROUTE_ROUND,
    "route_rounds": NUM_ROUTE_ROUNDS,
    "threads": TOTAL_THREADS,
    "register_budget": REGISTER_BUDGET,
    "nominal_pipe_bytes": NOMINAL_PIPE_BYTES,
    "score": "bf16_boundary_stable_softmax_top8_selected_renorm",
    "candidate_abi": "separate_one_sorted_top8_list_per_e64_producer",
    "candidate_workspace": "[T,4,16] fp32",
    "workspace_contract": "full_fp32_logits_plus_candidate_scratch",
    "synchronization": "gpu_acq_rel_four_producer_last_arriver",
    "routing_owner": "fourth_arriving_cta_all_three_static_roles",
    "padding": False,
    "fallback": False,
    "concurrent_replay_safe": False,
    "registered": True,
    "default_dispatch": True,
    "release_allowlist": True,
    "source_cuda_patches": (CUDA_V018_PATCH, CUDA_V036_PATCH),
    "source_cuda_patch_sha256": (CUDA_V018_SHA256, CUDA_V036_SHA256),
    "topology_equivalent_except": (
        "TLE vectorizes eight rows per static role; CUDA v018 uses two "
        "width16 half-warps per physical warp"
    ),
}


@triton.jit
def _publish_local_single_list(
    token,
    token_is_valid,
    logits_ptr,
    candidate_scratch_ptr,
    expert_base,
):
    """Publish one sorted local top-8 without modifying producer logits."""

    local_expert = tl.arange(0, JIT_EXPERTS_PER_CTA)[None, :]
    global_expert = expert_base + local_expert
    safe_token = tl.where(token_is_valid, token, 0)
    logits = tl.load(logits_ptr + safe_token * JIT_EXPERTS + global_expert)
    # ReplicatedLinear exposes BF16 logits to Qwen routing.  CUDA v017 fixed
    # the local publisher by applying this boundary before local selection.
    logits = logits.to(tl.bfloat16).to(tl.float32)
    logits = tl.where(logits == logits, logits, -float("inf"))
    logits = tl.where(token_is_valid, logits, -float("inf"))

    score_key = _ordered_score_key(logits).to(tl.uint32)
    tie_key = (JIT_TIE_MASK - global_expert).to(tl.uint64)
    packed = (score_key.to(tl.uint64) << 16) | tie_key
    selected = tl.topk(packed, JIT_LOCAL_TOPK)
    selected_expert = (JIT_TIE_MASK - (selected & JIT_TIE_MASK).to(tl.int32)).to(
        tl.int32
    )
    selected_logit = tl.gather(
        logits,
        selected_expert - expert_base,
        axis=1,
    )

    item = tl.arange(0, JIT_LOCAL_TOPK)[None, :]
    producer = expert_base // JIT_EXPERTS_PER_CTA
    record = (
        candidate_scratch_ptr
        + (safe_token * JIT_LOGICAL_EXPERT_CTAS + producer) * JIT_CANDIDATE_RECORD_WORDS
    )
    tl.store(
        record + JIT_CANDIDATE_SCORE_WORD + item,
        selected_logit,
        mask=token_is_valid,
    )
    # Expert ids are <=255 and therefore exactly representable as FP32.  This
    # preserves the donor record encoding in the typed scratch pointer.
    tl.store(
        record + JIT_CANDIDATE_EXPERT_WORD + item,
        selected_expert.to(tl.float32),
        mask=token_is_valid,
    )


@triton.jit
def _publish_role_rows(
    logits_ptr,
    candidate_scratch_ptr,
    expert_base,
    token_base,
    tokens: tl.constexpr,
    role_id: tl.constexpr,
):
    row = tl.arange(0, JIT_TOKENS_PER_ROLE_BATCH)[:, None]
    for route_round in tl.range(0, JIT_NUM_ROUTE_ROUNDS):
        local_token = (
            route_round * JIT_TOKENS_PER_ROUTE_ROUND
            + role_id * JIT_TOKENS_PER_ROLE_BATCH
            + row
        )
        token = token_base + local_token
        token_is_valid = (local_token < JIT_BLOCK_TOKENS) & (token < tokens)
        _publish_local_single_list(
            token,
            token_is_valid,
            logits_ptr,
            candidate_scratch_ptr,
            expert_base,
        )


@triton.jit
def _merge_four_single_lists(
    token,
    token_is_valid,
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
):
    """Merge four producer heads for eight rows, then renormalize top-8."""

    producer = tl.arange(0, JIT_LOGICAL_EXPERT_CTAS)[None, :]
    cursor = tl.zeros(
        (JIT_TOKENS_PER_ROLE_BATCH, JIT_LOGICAL_EXPERT_CTAS),
        dtype=tl.int32,
    )
    selected_max = tl.zeros((JIT_TOKENS_PER_ROLE_BATCH,), tl.float32)
    denominator = tl.zeros((JIT_TOKENS_PER_ROLE_BATCH,), tl.float32)
    safe_token = tl.where(token_is_valid, token, 0)
    output_base = safe_token * JIT_TOPK

    for output_idx in tl.static_range(0, JIT_TOPK):
        record_base = (
            safe_token[:, None] * JIT_LOGICAL_EXPERT_CTAS + producer
        ) * JIT_CANDIDATE_RECORD_WORDS
        head_score = tl.load(
            candidate_scratch_ptr + record_base + JIT_CANDIDATE_SCORE_WORD + cursor
        )
        head_expert = tl.load(
            candidate_scratch_ptr + record_base + JIT_CANDIDATE_EXPERT_WORD + cursor
        ).to(tl.int32)
        head_key = _ordered_score_key(head_score).to(tl.uint32)
        best_key = tl.max(head_key, axis=1)
        invalid_expert = tl.full(
            (
                JIT_TOKENS_PER_ROLE_BATCH,
                JIT_LOGICAL_EXPERT_CTAS,
            ),
            0xFFFFFFFF,
            tl.uint32,
        )
        best_expert = tl.min(
            tl.where(
                head_key == best_key[:, None],
                head_expert.to(tl.uint32),
                invalid_expert,
            ),
            axis=1,
        ).to(tl.int32)
        owns_best = head_expert == best_expert[:, None]
        best_score = tl.sum(tl.where(owns_best, head_score, 0.0), axis=1)
        cursor += owns_best.to(tl.int32)

        if output_idx == 0:
            selected_max = best_score
        has_finite = selected_max != -float("inf")
        selected_exp = tl.where(
            has_finite,
            libdevice.exp(best_score - selected_max),
            0.0,
        )
        denominator += selected_exp
        tl.store(
            topk_indices_ptr + output_base + output_idx,
            best_expert,
            mask=token_is_valid,
        )
        # Temporary unnormalized selected softmax numerator.  The second
        # eight-word pass below divides in place without another kernel.
        tl.store(
            topk_weights_ptr + output_base + output_idx,
            selected_exp,
            mask=token_is_valid,
        )

    denominator += 1.0e-20
    for output_idx in tl.static_range(0, JIT_TOPK):
        value = tl.load(topk_weights_ptr + output_base + output_idx)
        tl.store(
            topk_weights_ptr + output_base + output_idx,
            value / denominator,
            mask=token_is_valid,
        )


@triton.jit
def _merge_role_rows(
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    token_base,
    tokens: tl.constexpr,
    role_id: tl.constexpr,
):
    terminal = tl.load(tle.gpu.local_ptr(terminal_smem, (0,)))
    if terminal != 0:
        row = tl.arange(0, JIT_TOKENS_PER_ROLE_BATCH)
        for route_round in tl.range(0, JIT_NUM_ROUTE_ROUNDS):
            local_token = (
                route_round * JIT_TOKENS_PER_ROUTE_ROUND
                + role_id * JIT_TOKENS_PER_ROLE_BATCH
                + row
            )
            token = token_base + local_token
            token_is_valid = (local_token < JIT_BLOCK_TOKENS) & (token < tokens)
            _merge_four_single_lists(
                token,
                token_is_valid,
                candidate_scratch_ptr,
                topk_indices_ptr,
                topk_weights_ptr,
            )


@triton.jit
def _loader_then_publish_then_merge(
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
    expert_base,
    token_base,
    tokens: tl.constexpr,
):
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
    _publish_role_rows(
        logits_ptr,
        candidate_scratch_ptr,
        expert_base,
        token_base,
        tokens,
        0,
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)
    tle.gpu.barrier_wait(route_handoff_barrier)
    _merge_role_rows(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        tokens,
        0,
    )


@triton.jit
def _math_then_publish_arrive_merge(
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
    expert_base,
    token_base,
    token_tile,
    tokens: tl.constexpr,
):
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

    tle.gpu.barrier_wait(logits_ready_barrier)
    _publish_role_rows(
        logits_ptr,
        candidate_scratch_ptr,
        expert_base,
        token_base,
        tokens,
        1,
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)

    ticket_ptr = state_ptr + token_tile * JIT_STATE_WORDS_PER_TILE + JIT_TICKET_WORD
    old_ticket = tl.atomic_add(
        ticket_ptr,
        1,
        sem="acq_rel",
        scope="gpu",
    )
    terminal = (old_ticket % JIT_NUM_PRODUCERS_PER_TILE) + 1 == (
        JIT_NUM_PRODUCERS_PER_TILE
    )
    tl.store(
        tle.gpu.local_ptr(terminal_smem, (0,)),
        terminal.to(tl.int32),
    )
    tle.gpu.barrier_wait(route_handoff_barrier)
    _merge_role_rows(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        tokens,
        1,
    )


@triton.jit
def _route_publish_then_merge(
    logits_ptr,
    candidate_scratch_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    terminal_smem,
    logits_ready_barrier,
    candidates_ready_barrier,
    route_handoff_barrier,
    expert_base,
    token_base,
    tokens: tl.constexpr,
):
    tle.gpu.barrier_wait(logits_ready_barrier)
    _publish_role_rows(
        logits_ptr,
        candidate_scratch_ptr,
        expert_base,
        token_base,
        tokens,
        2,
    )
    tle.gpu.barrier_wait(candidates_ready_barrier)
    tle.gpu.barrier_wait(route_handoff_barrier)
    _merge_role_rows(
        candidate_scratch_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        terminal_smem,
        token_base,
        tokens,
        2,
    )


@triton.jit
def _fused_qwen_h2048_v018_v036_single_list_kernel(
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
    logical_rank = pid % JIT_LOGICAL_EXPERT_CTAS
    token_tile = pid // JIT_LOGICAL_EXPERT_CTAS
    expert_base = logical_rank * JIT_EXPERTS_PER_CTA
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
        name="qwen_h2048_v018_v036_b128_e64_single_list_v365",
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
    candidates_ready_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_TOTAL_THREADS)
    route_handoff_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_TOTAL_THREADS)

    tle.gpu.warp_specialize(
        [
            (
                _loader_then_publish_then_merge,
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
                    expert_base,
                    token_base,
                    tokens,
                ),
            ),
            (
                _math_then_publish_arrive_merge,
                (
                    pipe.reader(),
                    logits_ptr,
                    candidate_scratch_ptr,
                    state_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    terminal_smem,
                    logits_ready_barrier,
                    candidates_ready_barrier,
                    route_handoff_barrier,
                    expert_base,
                    token_base,
                    token_tile,
                    tokens,
                ),
            ),
            (
                _route_publish_then_merge,
                (
                    logits_ptr,
                    candidate_scratch_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    terminal_smem,
                    logits_ready_barrier,
                    candidates_ready_barrier,
                    route_handoff_barrier,
                    expert_base,
                    token_base,
                    tokens,
                ),
            ),
        ],
        [4, 4],
        [JIT_MATH_MAXNREG, JIT_ROUTE_MAXNREG],
    )
