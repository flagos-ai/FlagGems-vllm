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

"""B128 clusterless last-arriver TLE kernel for DSV4 MegaGate.

Each logical token tile launches four independently schedulable CTAs, one per
E64 expert shard.  The math role stores precise activated FP32 logits and then
performs a GPU-scope acq_rel ticket increment.  Only the CTA observing the
fourth arrival hands control to its routing role, which consumes the complete
E256 row and writes stable top-6 output.  Non-terminal CTAs exit without a
grid-wide wait, so large grids may execute in multiple waves.

After their producer work completes, all three static TLE roles join the
terminal-CTA handoff and route disjoint token rows. This reproduces the CUDA
twelve-warp routing topology
without changing roles dynamically: loader routes rows for
warps 0--3, math routes rows for warps 4--7, and the dedicated route role owns
warps 8--11.  B128, E64 shards, cluster_dim=1, last-arriver ownership, stage-4
BK128 GEMM, FP32 score workspace, and exact model semantics remain unchanged.
"""

from __future__ import annotations

from typing import Final

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from . import dsv4_common as _common
from . import ordered_routing as _ordered

ADMITTED_TOKENS: Final = (
    1028,
    2048,
    3076,
    3805,
    4100,
    4835,
    7884,
    8221,
    13021,
    16384,
)

HIDDEN: Final = _common.HIDDEN
EXPERTS: Final = _common.EXPERTS
TOPK: Final = _common.TOPK
ROUTED_SCALING_FACTOR: Final = _common.ROUTED_SCALING_FACTOR
BLOCK_TOKENS: Final = _common.BLOCK_TOKENS
EXPERTS_PER_CTA: Final = _common.EXPERTS_PER_CTA
LOGICAL_EXPERT_CTAS: Final = _common.LOGICAL_EXPERT_CTAS
BLOCK_K: Final = _common.BLOCK_K
NUM_K_TILES: Final = _common.NUM_K_TILES
PIPE_STAGES: Final = _common.PIPE_STAGES

THREADS_PER_ROLE: Final = 128
MATH_MAXNREG: Final = 224
ROUTE_MAXNREG: Final = 96
TOTAL_ROLES: Final = 3
TOTAL_THREADS: Final = THREADS_PER_ROLE * TOTAL_ROLES
KERNEL_MAXNREG: Final = 139
RAW_DEFAULT_MAXNREG: Final = TOTAL_ROLES * KERNEL_MAXNREG - MATH_MAXNREG - ROUTE_MAXNREG
LOADER_MAXNREG: Final = (RAW_DEFAULT_MAXNREG // 8) * 8
REGISTER_CAP_PER_CTA: Final = THREADS_PER_ROLE * (
    LOADER_MAXNREG + MATH_MAXNREG + ROUTE_MAXNREG
)
KERNEL_REGISTER_LIMIT_PER_CTA: Final = TOTAL_THREADS * KERNEL_MAXNREG

ROUTING_WARPS_PER_ROLE: Final = 4
ROUTING_ROLES: Final = TOTAL_ROLES
TOTAL_ROUTING_WARPS: Final = ROUTING_WARPS_PER_ROLE * ROUTING_ROLES
OWNER_ROUNDS: Final = (BLOCK_TOKENS + TOTAL_ROUTING_WARPS - 1) // TOTAL_ROUTING_WARPS

STATE_WORDS_PER_TILE: Final = 16
TICKET_WORD: Final = 4
NUM_PRODUCERS_PER_TILE: Final = LOGICAL_EXPERT_CTAS

PIPE_BYTES_PER_CTA: Final = PIPE_STAGES * (
    EXPERTS_PER_CTA * BLOCK_K * 2 + BLOCK_TOKENS * BLOCK_K * 2
)
BIAS_BYTES_PER_CTA: Final = _common.BIAS_SMEM_EXTENT * 4
NOMINAL_SMEM_BYTES: Final = PIPE_BYTES_PER_CTA + BIAS_BYTES_PER_CTA

assert len(ADMITTED_TOKENS) == 10
assert tuple(sorted(ADMITTED_TOKENS)) == ADMITTED_TOKENS
assert (HIDDEN, EXPERTS, TOPK) == (4096, 256, 6)
assert (BLOCK_TOKENS, BLOCK_K, PIPE_STAGES) == (128, 128, 4)
assert (EXPERTS_PER_CTA, LOGICAL_EXPERT_CTAS) == (64, 4)
assert NUM_K_TILES == 32
assert (RAW_DEFAULT_MAXNREG, LOADER_MAXNREG) == (97, 96)
assert REGISTER_CAP_PER_CTA == 53_248
assert KERNEL_REGISTER_LIMIT_PER_CTA == 53_376
assert (TOTAL_ROUTING_WARPS, OWNER_ROUNDS) == (12, 11)
assert STATE_WORDS_PER_TILE * 8 == 128

JIT_BLOCK_TOKENS = tl.constexpr(BLOCK_TOKENS)
JIT_EXPERTS = tl.constexpr(EXPERTS)
JIT_EXPERTS_PER_CTA = tl.constexpr(EXPERTS_PER_CTA)
JIT_LOGICAL_EXPERT_CTAS = tl.constexpr(LOGICAL_EXPERT_CTAS)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_NUM_K_TILES = tl.constexpr(NUM_K_TILES)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_NUM_PRODUCERS_PER_TILE = tl.constexpr(NUM_PRODUCERS_PER_TILE)
JIT_STATE_WORDS_PER_TILE = tl.constexpr(STATE_WORDS_PER_TILE)
JIT_TICKET_WORD = tl.constexpr(TICKET_WORD)
JIT_MATH_MAXNREG = tl.constexpr(MATH_MAXNREG)
JIT_ROUTE_MAXNREG = tl.constexpr(ROUTE_MAXNREG)
JIT_BIAS_SMEM_EXTENT = tl.constexpr(_common.BIAS_SMEM_EXTENT)
JIT_TOTAL_ROUTING_WARPS = tl.constexpr(TOTAL_ROUTING_WARPS)
JIT_OWNER_ROUNDS = tl.constexpr(OWNER_ROUNDS)


@triton.jit
def _loader_role(
    writer,
    weight_desc,
    x_desc,
    bias_ptr,
    bias_smem,
    activated_score_ptr,
    terminal_smem,
    bias_init_barrier,
    route_handoff_barrier,
    topk_idx_ptr,
    topk_weight_ptr,
    expert_base,
    token_base,
    tokens,
):
    lane = tl.arange(0, 128)
    tl.store(
        tle.gpu.local_ptr(bias_smem, (lane,)),
        tl.load(bias_ptr + lane).to(tl.float32),
    )
    tle.gpu.barrier_wait(bias_init_barrier)

    for ticket in tl.range(0, JIT_NUM_K_TILES):
        slot = writer.acquire(ticket)
        k_offset = ticket * JIT_BLOCK_K
        tle.gpu.copy(
            weight_desc,
            slot.weight,
            [JIT_EXPERTS_PER_CTA, JIT_BLOCK_K],
            [expert_base, k_offset],
        )
        tle.gpu.copy(
            x_desc,
            slot.x,
            [JIT_BLOCK_TOKENS, JIT_BLOCK_K],
            [token_base, k_offset],
        )
        writer.commit(ticket)

    tle.gpu.barrier_wait(route_handoff_barrier)
    terminal = tl.load(tle.gpu.local_ptr(terminal_smem, (0,)))
    if terminal != 0:
        _common._route_two_wave_owner_tokens(
            activated_score_ptr,
            bias_smem,
            topk_idx_ptr,
            topk_weight_ptr,
            token_base,
            tokens,
            0,
            JIT_TOTAL_ROUTING_WARPS,
            JIT_OWNER_ROUNDS,
        )


@triton.jit
def _math_publish_role(
    reader,
    activated_score_ptr,
    state_ptr,
    bias_ptr,
    bias_smem,
    terminal_smem,
    bias_init_barrier,
    publish_barrier,
    route_handoff_barrier,
    topk_idx_ptr,
    topk_weight_ptr,
    expert_base,
    token_base,
    token_tile,
    tokens,
):
    lane = tl.arange(0, 128)
    tl.store(
        tle.gpu.local_ptr(bias_smem, (lane + 128,)),
        tl.load(bias_ptr + lane + 128).to(tl.float32),
    )
    tle.gpu.barrier_wait(bias_init_barrier)

    accumulator = tl.zeros((JIT_EXPERTS_PER_CTA, JIT_BLOCK_TOKENS), dtype=tl.float32)
    first = reader.wait(0)
    accumulator = tle.gpu.wgmma(
        first.slot.weight,
        first.slot.x,
        accumulator,
        trans_b=True,
    )
    for ticket in tl.range(1, JIT_NUM_K_TILES):
        current = reader.wait(ticket)
        accumulator = tle.gpu.wgmma(
            current.slot.weight,
            current.slot.x,
            accumulator,
            trans_b=True,
        )
        accumulator = tle.gpu.wgmma_wait(1, accumulator)
        reader.release(ticket - 1)
    accumulator = tle.gpu.wgmma_wait(0, accumulator)
    reader.release(JIT_NUM_K_TILES - 1)

    activated = _ordered._precise_sqrt_softplus(tl.trans(accumulator))
    token = token_base + tl.arange(0, JIT_BLOCK_TOKENS)[:, None]
    expert = expert_base + tl.arange(0, JIT_EXPERTS_PER_CTA)[None, :]
    tl.store(
        activated_score_ptr + token * JIT_EXPERTS + expert,
        activated,
        mask=token < tokens,
    )

    # All 128 math threads finish their FP32 stores before the elected scalar
    # acq_rel arrival.  The fourth arrival acquires the other three releases.
    tle.gpu.barrier_wait(publish_barrier)
    ticket_ptr = state_ptr + token_tile * JIT_STATE_WORDS_PER_TILE + JIT_TICKET_WORD
    old_ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="gpu")
    terminal = (old_ticket % JIT_NUM_PRODUCERS_PER_TILE) + 1 == (
        JIT_NUM_PRODUCERS_PER_TILE
    )
    tl.store(tle.gpu.local_ptr(terminal_smem, (0,)), terminal.to(tl.int32))
    tle.gpu.barrier_wait(route_handoff_barrier)
    if terminal != 0:
        _common._route_two_wave_owner_tokens(
            activated_score_ptr,
            bias_smem,
            topk_idx_ptr,
            topk_weight_ptr,
            token_base,
            tokens,
            4,
            JIT_TOTAL_ROUTING_WARPS,
            JIT_OWNER_ROUNDS,
        )


@triton.jit
def _route_role(
    activated_score_ptr,
    bias_smem,
    terminal_smem,
    bias_init_barrier,
    route_handoff_barrier,
    topk_idx_ptr,
    topk_weight_ptr,
    token_base,
    tokens,
):
    # No third bias wave exists for E256, but this role must join the 384-way
    # publication barrier before it may consume the complete cache.
    tle.gpu.barrier_wait(bias_init_barrier)
    tle.gpu.barrier_wait(route_handoff_barrier)
    terminal = tl.load(tle.gpu.local_ptr(terminal_smem, (0,)))
    if terminal != 0:
        _common._route_two_wave_owner_tokens(
            activated_score_ptr,
            bias_smem,
            topk_idx_ptr,
            topk_weight_ptr,
            token_base,
            tokens,
            8,
            JIT_TOTAL_ROUTING_WARPS,
            JIT_OWNER_ROUNDS,
        )


@triton.jit
def _b128_clusterless_last_arriver_kernel(
    weight_desc,
    x_desc,
    activated_score_ptr,
    state_ptr,
    bias_ptr,
    topk_idx_ptr,
    topk_weight_ptr,
    tokens,
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
    # Keep the admitted pipe identity stable: FlagTree uses outlined pipe
    # identities while allocating module-wide named barriers.
    pipe = tle.pipe(
        capacity=JIT_PIPE_STAGES,
        scope="cta",
        name="dsv4_h4096_e256_b128_clusterless_last_arriver_v363",
        weight=weight_smem,
        x=x_smem,
    )
    bias_smem = tle.gpu.alloc(
        [JIT_BIAS_SMEM_EXTENT],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    terminal_smem = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    bias_init_barrier = tle.gpu.alloc_barrier(arrive_count=384)
    publish_barrier = tle.gpu.alloc_barrier(arrive_count=128)
    route_handoff_barrier = tle.gpu.alloc_barrier(arrive_count=384)

    tle.gpu.warp_specialize(
        [
            (
                _loader_role,
                (
                    pipe.writer(),
                    weight_desc,
                    x_desc,
                    bias_ptr,
                    bias_smem,
                    activated_score_ptr,
                    terminal_smem,
                    bias_init_barrier,
                    route_handoff_barrier,
                    topk_idx_ptr,
                    topk_weight_ptr,
                    expert_base,
                    token_base,
                    tokens,
                ),
            ),
            (
                _math_publish_role,
                (
                    pipe.reader(),
                    activated_score_ptr,
                    state_ptr,
                    bias_ptr,
                    bias_smem,
                    terminal_smem,
                    bias_init_barrier,
                    publish_barrier,
                    route_handoff_barrier,
                    topk_idx_ptr,
                    topk_weight_ptr,
                    expert_base,
                    token_base,
                    token_tile,
                    tokens,
                ),
            ),
            (
                _route_role,
                (
                    activated_score_ptr,
                    bias_smem,
                    terminal_smem,
                    bias_init_barrier,
                    route_handoff_barrier,
                    topk_idx_ptr,
                    topk_weight_ptr,
                    token_base,
                    tokens,
                ),
            ),
        ],
        [4, 4],
        [JIT_MATH_MAXNREG, JIT_ROUTE_MAXNREG],
    )


__all__ = [
    "ADMITTED_TOKENS",
    "KERNEL_MAXNREG",
    "_b128_clusterless_last_arriver_kernel",
]
