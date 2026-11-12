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

"""E128/B128 dual-math DSV3.2 MegaGate kernel (source leaf v372)."""

from __future__ import annotations

from typing import Final

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from . import dsv32_common as _common

SOURCE_LEAF: Final = "v372"

SUPPORTED_TOKENS: Final = (12292, 12372)
LARGE_T_SCREEN_TOKENS: Final = (14305, 16384)
HIDDEN: Final = _common.HIDDEN
EXPERTS: Final = _common.EXPERTS
TOPK: Final = _common.TOPK
ROUTED_SCALING_FACTOR: Final = _common.ROUTED_SCALING_FACTOR
BLOCK_TOKENS: Final = 128
EXPERTS_PER_CTA: Final = 128
EXPERTS_PER_MATH: Final = 64
LOGICAL_EXPERT_CTAS: Final = 2
BLOCK_K: Final = 128
NUM_K_TILES: Final = HIDDEN // BLOCK_K
PIPE_STAGES: Final = 3
LOGICAL_ROLES: Final = 4
TOTAL_THREADS: Final = 512
MATH_MAXNREG: Final = 176
ROUTE_MAXNREG: Final = 96
KERNEL_MAXNREG: Final = 128
STATE_WORDS_PER_TILE: Final = _common.STATE_WORDS_PER_TILE
TICKET_WORD: Final = _common.TICKET_WORD

PIPE_BYTES_PER_STAGE: Final = EXPERTS_PER_CTA * BLOCK_K * 2 + BLOCK_TOKENS * BLOCK_K * 2
NOMINAL_PIPE_BYTES: Final = PIPE_STAGES * PIPE_BYTES_PER_STAGE
REGISTER_BUDGET: Final = 128 * (40 + 2 * MATH_MAXNREG + ROUTE_MAXNREG)

assert (HIDDEN, EXPERTS, TOPK) == (7168, 256, 8)
assert EXPERTS == LOGICAL_EXPERT_CTAS * EXPERTS_PER_CTA
assert EXPERTS_PER_CTA == 2 * EXPERTS_PER_MATH
assert (BLOCK_TOKENS, BLOCK_K, NUM_K_TILES) == (128, 128, 56)
assert (PIPE_STAGES, TOTAL_THREADS) == (3, 512)
assert NOMINAL_PIPE_BYTES == 196_608
assert REGISTER_BUDGET == 62_464

JIT_EXPERTS = tl.constexpr(EXPERTS)
JIT_BLOCK_TOKENS = tl.constexpr(BLOCK_TOKENS)
JIT_EXPERTS_PER_CTA = tl.constexpr(EXPERTS_PER_CTA)
JIT_EXPERTS_PER_MATH = tl.constexpr(EXPERTS_PER_MATH)
JIT_LOGICAL_EXPERT_CTAS = tl.constexpr(LOGICAL_EXPERT_CTAS)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_NUM_K_TILES = tl.constexpr(NUM_K_TILES)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_LOGICAL_ROLES = tl.constexpr(LOGICAL_ROLES)
JIT_TOTAL_THREADS = tl.constexpr(TOTAL_THREADS)
JIT_STATE_WORDS_PER_TILE = tl.constexpr(STATE_WORDS_PER_TILE)
JIT_TICKET_WORD = tl.constexpr(TICKET_WORD)
JIT_NUM_PRODUCERS = tl.constexpr(LOGICAL_EXPERT_CTAS)
JIT_MATH_MAXNREG = tl.constexpr(MATH_MAXNREG)
JIT_ROUTE_MAXNREG = tl.constexpr(ROUTE_MAXNREG)


@triton.jit
def _load(writer, weight_desc, x_desc, token_base, expert_base):
    for ticket in tl.range(0, JIT_NUM_K_TILES):
        slot = writer.acquire(ticket)
        k_base = ticket * JIT_BLOCK_K
        tle.gpu.copy(
            weight_desc,
            slot.weight0,
            [JIT_EXPERTS_PER_MATH, JIT_BLOCK_K],
            [expert_base, k_base],
        )
        tle.gpu.copy(
            weight_desc,
            slot.weight1,
            [JIT_EXPERTS_PER_MATH, JIT_BLOCK_K],
            [expert_base + JIT_EXPERTS_PER_MATH, k_base],
        )
        tle.gpu.copy(
            x_desc,
            slot.x,
            [JIT_BLOCK_TOKENS, JIT_BLOCK_K],
            [token_base, k_base],
        )
        writer.commit(ticket)


@triton.jit
def _math(
    reader,
    logits_ptr,
    token_base,
    expert_base,
    TOKENS,
    MATH_ID: tl.constexpr,
):
    accumulator = tl.zeros((JIT_EXPERTS_PER_MATH, JIT_BLOCK_TOKENS), dtype=tl.float32)
    first = reader.wait(0)
    if MATH_ID == 0:
        accumulator = tle.gpu.wgmma(
            first.slot.weight0, first.slot.x, accumulator, trans_b=True
        )
    else:
        accumulator = tle.gpu.wgmma(
            first.slot.weight1, first.slot.x, accumulator, trans_b=True
        )
    for ticket in tl.range(1, JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        if MATH_ID == 0:
            accumulator = tle.gpu.wgmma(
                stage.slot.weight0, stage.slot.x, accumulator, trans_b=True
            )
        else:
            accumulator = tle.gpu.wgmma(
                stage.slot.weight1, stage.slot.x, accumulator, trans_b=True
            )
        accumulator = tle.gpu.wgmma_wait(1, accumulator)
        reader.release(ticket - 1)
    accumulator = tle.gpu.wgmma_wait(0, accumulator)
    reader.release(JIT_NUM_K_TILES - 1)

    token = token_base + tl.arange(0, JIT_BLOCK_TOKENS)[:, None]
    expert = (
        expert_base
        + MATH_ID * JIT_EXPERTS_PER_MATH
        + tl.arange(0, JIT_EXPERTS_PER_MATH)[None, :]
    )
    bounded = tl.trans(accumulator).to(tl.bfloat16)
    tl.store(
        logits_ptr + token * JIT_EXPERTS + expert,
        bounded,
        mask=token < TOKENS,
    )


@triton.jit
def _route(
    logits_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    terminal_smem,
    route_barrier,
    token_base,
    TOKENS,
    ROLE_ID: tl.constexpr,
):
    tle.gpu.barrier_wait(route_barrier)
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
            JIT_LOGICAL_ROLES,
            False,
        )


@triton.jit
def _loader_route0(
    writer,
    weight_desc,
    x_desc,
    logits_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    terminal_smem,
    route_barrier,
    token_base,
    expert_base,
    TOKENS,
):
    _load(writer, weight_desc, x_desc, token_base, expert_base)
    _route(
        logits_ptr,
        bias_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        routed_scaling_factor,
        terminal_smem,
        route_barrier,
        token_base,
        TOKENS,
        0,
    )


@triton.jit
def _math_route(
    reader,
    logits_ptr,
    state_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    terminal_smem,
    math_barrier,
    route_barrier,
    token_base,
    expert_base,
    token_tile,
    TOKENS,
    MATH_ID: tl.constexpr,
    ROLE_ID: tl.constexpr,
):
    _math(reader, logits_ptr, token_base, expert_base, TOKENS, MATH_ID)
    tle.gpu.barrier_wait(math_barrier)
    if MATH_ID == 1:
        ticket_ptr = state_ptr + token_tile * JIT_STATE_WORDS_PER_TILE + JIT_TICKET_WORD
        old_ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="gpu")
        terminal = (old_ticket % JIT_NUM_PRODUCERS) + 1 == JIT_NUM_PRODUCERS
        tl.store(tle.gpu.local_ptr(terminal_smem, (0,)), terminal.to(tl.int32))
    _route(
        logits_ptr,
        bias_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        routed_scaling_factor,
        terminal_smem,
        route_barrier,
        token_base,
        TOKENS,
        ROLE_ID,
    )


@triton.jit
def _route3(
    logits_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    terminal_smem,
    route_barrier,
    token_base,
    TOKENS,
):
    _route(
        logits_ptr,
        bias_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        routed_scaling_factor,
        terminal_smem,
        route_barrier,
        token_base,
        TOKENS,
        3,
    )


@triton.jit
def _fused_e128_dualmath_kernel(
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
    rank = pid % JIT_LOGICAL_EXPERT_CTAS
    token_tile = pid // JIT_LOGICAL_EXPERT_CTAS
    token_base = token_tile * JIT_BLOCK_TOKENS
    expert_base = rank * JIT_EXPERTS_PER_CTA

    weight0 = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_EXPERTS_PER_MATH, JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    weight1 = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_EXPERTS_PER_MATH, JIT_BLOCK_K],
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
        name="dsv32_h7168_e256_e128_dualmath_v372",
        readers=("math0", "math1"),
        weight0=weight0,
        weight1=weight1,
        x=x_smem,
    )
    terminal_smem = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    math_barrier = tle.gpu.alloc_barrier(arrive_count=256)
    route_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_TOTAL_THREADS)

    tle.gpu.warp_specialize(
        [
            (
                _loader_route0,
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
                    route_barrier,
                    token_base,
                    expert_base,
                    TOKENS,
                ),
            ),
            (
                _math_route,
                (
                    pipe.reader(name="math0", fields=("weight0", "x")),
                    logits_ptr,
                    state_ptr,
                    bias_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    routed_scaling_factor,
                    terminal_smem,
                    math_barrier,
                    route_barrier,
                    token_base,
                    expert_base,
                    token_tile,
                    TOKENS,
                    0,
                    1,
                ),
            ),
            (
                _math_route,
                (
                    pipe.reader(name="math1", fields=("weight1", "x")),
                    logits_ptr,
                    state_ptr,
                    bias_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    routed_scaling_factor,
                    terminal_smem,
                    math_barrier,
                    route_barrier,
                    token_base,
                    expert_base,
                    token_tile,
                    TOKENS,
                    1,
                    2,
                ),
            ),
            (
                _route3,
                (
                    logits_ptr,
                    bias_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    routed_scaling_factor,
                    terminal_smem,
                    route_barrier,
                    token_base,
                    TOKENS,
                ),
            ),
        ],
        [4, 4, 4],
        [JIT_MATH_MAXNREG, JIT_MATH_MAXNREG, JIT_ROUTE_MAXNREG],
    )


__all__ = ["KERNEL_MAXNREG", "SUPPORTED_TOKENS", "_fused_e128_dualmath_kernel"]
