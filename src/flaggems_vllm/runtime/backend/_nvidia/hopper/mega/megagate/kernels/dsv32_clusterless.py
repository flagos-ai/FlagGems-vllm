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

"""E64/B128 clusterless DSV3.2 MegaGate kernel (source leaf v371)."""

from __future__ import annotations

from typing import Final

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from . import dsv32_common as _common

SOURCE_LEAF: Final = "v371"

GENERIC_TOKENS: Final = (1028, 1756, 2048, 3076, 3805)
V015_TOKENS: Final = (8276,)
SUPPORTED_TOKENS: Final = GENERIC_TOKENS + V015_TOKENS
TOKEN_TO_CUDA_LEAF: Final = {
    **{tokens: "generic_bf16_mega_gate" for tokens in GENERIC_TOKENS},
    8276: "v015_e64_clusterless",
}

HIDDEN: Final = _common.HIDDEN
EXPERTS: Final = _common.EXPERTS
TOPK: Final = _common.TOPK
ROUTED_SCALING_FACTOR: Final = _common.ROUTED_SCALING_FACTOR
BLOCK_TOKENS: Final = 128
EXPERTS_PER_CTA: Final = 64
LOGICAL_EXPERT_CTAS: Final = 4
BLOCK_K: Final = 128
NUM_K_TILES: Final = HIDDEN // BLOCK_K
PIPE_STAGES: Final = 4
THREADS_PER_ROLE: Final = 128
TOTAL_ROLES: Final = 3
TOTAL_THREADS: Final = THREADS_PER_ROLE * TOTAL_ROLES
MATH_MAXNREG: Final = 224
ROUTE_MAXNREG: Final = 96
KERNEL_MAXNREG: Final = 128
REGISTER_BUDGET: Final = THREADS_PER_ROLE * (
    KERNEL_MAXNREG + MATH_MAXNREG + ROUTE_MAXNREG
)
STATE_WORDS_PER_TILE: Final = _common.STATE_WORDS_PER_TILE
TICKET_WORD: Final = _common.TICKET_WORD

PIPE_BYTES_PER_STAGE: Final = EXPERTS_PER_CTA * BLOCK_K * 2 + BLOCK_TOKENS * BLOCK_K * 2
NOMINAL_PIPE_BYTES: Final = PIPE_STAGES * PIPE_BYTES_PER_STAGE

assert tuple(sorted(SUPPORTED_TOKENS)) == SUPPORTED_TOKENS
assert (HIDDEN, EXPERTS, TOPK) == (7168, 256, 8)
assert EXPERTS == LOGICAL_EXPERT_CTAS * EXPERTS_PER_CTA
assert (BLOCK_TOKENS, BLOCK_K, NUM_K_TILES) == (128, 128, 56)
assert (PIPE_STAGES, TOTAL_THREADS) == (4, 384)
assert NOMINAL_PIPE_BYTES == 196_608
assert REGISTER_BUDGET == 57_344

JIT_EXPERTS = tl.constexpr(EXPERTS)
JIT_BLOCK_TOKENS = tl.constexpr(BLOCK_TOKENS)
JIT_EXPERTS_PER_CTA = tl.constexpr(EXPERTS_PER_CTA)
JIT_LOGICAL_EXPERT_CTAS = tl.constexpr(LOGICAL_EXPERT_CTAS)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_NUM_K_TILES = tl.constexpr(NUM_K_TILES)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_STATE_WORDS_PER_TILE = tl.constexpr(STATE_WORDS_PER_TILE)
JIT_TICKET_WORD = tl.constexpr(TICKET_WORD)
JIT_NUM_PRODUCERS = tl.constexpr(LOGICAL_EXPERT_CTAS)
JIT_MATH_MAXNREG = tl.constexpr(MATH_MAXNREG)
JIT_ROUTE_MAXNREG = tl.constexpr(ROUTE_MAXNREG)
JIT_TOTAL_ROLES = tl.constexpr(TOTAL_ROLES)
JIT_TOTAL_THREADS = tl.constexpr(TOTAL_THREADS)


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
def _route_terminal_rows(
    logits_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    terminal_smem,
    route_handoff_barrier,
    token_base,
    TOKENS,
    ROLE_ID: tl.constexpr,
):
    """Join the publication barrier and route this role's four warp rows."""

    tle.gpu.barrier_wait(route_handoff_barrier)
    terminal = tl.load(tle.gpu.local_ptr(terminal_smem, (0,)))
    if terminal != 0:
        _common.route_grouped_sigmoid_warp_rows(
            logits_ptr,
            bias_ptr,
            topk_indices_ptr,
            topk_weights_ptr,
            routed_scaling_factor,
            token_base,
            TOKENS,
            JIT_BLOCK_TOKENS,
            ROLE_ID,
            JIT_TOTAL_ROLES,
            False,
        )


@triton.jit
def _loader_then_route_role(
    writer,
    weight_desc,
    x_desc,
    logits_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    terminal_smem,
    route_handoff_barrier,
    expert_base,
    token_base,
    TOKENS,
):
    _loader_role(writer, weight_desc, x_desc, expert_base, token_base)
    _route_terminal_rows(
        logits_ptr,
        bias_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        routed_scaling_factor,
        terminal_smem,
        route_handoff_barrier,
        token_base,
        TOKENS,
        0,
    )


@triton.jit
def _math_publish_role(
    reader,
    logits_ptr,
    state_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    terminal_smem,
    publish_barrier,
    route_handoff_barrier,
    expert_base,
    token_base,
    token_tile,
    TOKENS,
):
    accumulator = tl.zeros((JIT_EXPERTS_PER_CTA, JIT_BLOCK_TOKENS), dtype=tl.float32)
    first = reader.wait(0)
    accumulator = tle.gpu.wgmma(
        first.slot.weight, first.slot.x, accumulator, trans_b=True
    )
    for ticket in tl.range(1, JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        accumulator = tle.gpu.wgmma(
            stage.slot.weight, stage.slot.x, accumulator, trans_b=True
        )
        accumulator = tle.gpu.wgmma_wait(1, accumulator)
        reader.release(ticket - 1)
    accumulator = tle.gpu.wgmma_wait(0, accumulator)
    reader.release(JIT_NUM_K_TILES - 1)

    token = token_base + tl.arange(0, JIT_BLOCK_TOKENS)[:, None]
    expert = expert_base + tl.arange(0, JIT_EXPERTS_PER_CTA)[None, :]
    bounded = tl.trans(accumulator).to(tl.bfloat16)
    tl.store(
        logits_ptr + token * JIT_EXPERTS + expert,
        bounded,
        mask=token < TOKENS,
    )

    tle.gpu.barrier_wait(publish_barrier)
    ticket_ptr = state_ptr + token_tile * JIT_STATE_WORDS_PER_TILE + JIT_TICKET_WORD
    old_ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="gpu")
    terminal = (old_ticket % JIT_NUM_PRODUCERS) + 1 == JIT_NUM_PRODUCERS
    tl.store(tle.gpu.local_ptr(terminal_smem, (0,)), terminal.to(tl.int32))
    _route_terminal_rows(
        logits_ptr,
        bias_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        routed_scaling_factor,
        terminal_smem,
        route_handoff_barrier,
        token_base,
        TOKENS,
        1,
    )


@triton.jit
def _route_role(
    logits_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    terminal_smem,
    route_handoff_barrier,
    token_base,
    TOKENS,
):
    _route_terminal_rows(
        logits_ptr,
        bias_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        routed_scaling_factor,
        terminal_smem,
        route_handoff_barrier,
        token_base,
        TOKENS,
        2,
    )


@triton.jit
def _fused_e64_clusterless_kernel(
    weight_desc,
    x_desc,
    logits_ptr,
    state_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    TOKENS,
):
    pid = tl.program_id(0)
    logical_rank = pid % JIT_LOGICAL_EXPERT_CTAS
    token_tile = pid // JIT_LOGICAL_EXPERT_CTAS
    token_base = token_tile * JIT_BLOCK_TOKENS
    expert_base = logical_rank * JIT_EXPERTS_PER_CTA

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
        name="dsv32_h7168_e256_e64_clusterless_v371",
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
    publish_barrier = tle.gpu.alloc_barrier(arrive_count=128)
    route_handoff_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_TOTAL_THREADS)

    tle.gpu.warp_specialize(
        [
            (
                _loader_then_route_role,
                (
                    pipe.writer(),
                    weight_desc,
                    x_desc,
                    logits_ptr,
                    bias_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    routed_scaling_factor,
                    terminal_smem,
                    route_handoff_barrier,
                    expert_base,
                    token_base,
                    TOKENS,
                ),
            ),
            (
                _math_publish_role,
                (
                    pipe.reader(),
                    logits_ptr,
                    state_ptr,
                    bias_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    routed_scaling_factor,
                    terminal_smem,
                    publish_barrier,
                    route_handoff_barrier,
                    expert_base,
                    token_base,
                    token_tile,
                    TOKENS,
                ),
            ),
            (
                _route_role,
                (
                    logits_ptr,
                    bias_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    routed_scaling_factor,
                    terminal_smem,
                    route_handoff_barrier,
                    token_base,
                    TOKENS,
                ),
            ),
        ],
        [4, 4],
        [JIT_MATH_MAXNREG, JIT_ROUTE_MAXNREG],
    )


__all__ = [
    "GENERIC_TOKENS",
    "KERNEL_MAXNREG",
    "SUPPORTED_TOKENS",
    "V015_TOKENS",
    "_fused_e64_clusterless_kernel",
]
