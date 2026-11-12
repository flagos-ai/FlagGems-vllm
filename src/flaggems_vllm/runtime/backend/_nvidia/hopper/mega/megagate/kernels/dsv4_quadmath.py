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

"""Large-T E256 quad-math TLE kernel for DSV4 MegaGate.

One E256 TMA stage is consumed by four E64 math roles in one CTA. The four
math roles preserve the FP32 GEMM to precise sqrt-softplus boundary; after a
single CTA barrier all five roles route disjoint token rows using the same
stable bias/top-6/normalization helper as the clusterless implementation.

The kernel keeps B128/BK128, two pipe stages, wait-group zero before stage
release, the activated-score workspace ABI, and one device-kernel launch.
"""

from __future__ import annotations

from typing import Final

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from . import dsv4_common as _common
from . import ordered_routing as _ordered

SUPPORTED_TOKENS: Final = (7884, 8221, 13021, 16384)
HIDDEN: Final = _common.HIDDEN
EXPERTS: Final = _common.EXPERTS
TOPK: Final = _common.TOPK
ROUTED_SCALING_FACTOR: Final = _common.ROUTED_SCALING_FACTOR
BLOCK_TOKENS: Final = 128
EXPERTS_PER_CTA: Final = 256
EXPERTS_PER_MATH: Final = 64
LOGICAL_EXPERT_CTAS: Final = 1
BLOCK_K: Final = 128
NUM_K_TILES: Final = HIDDEN // BLOCK_K
PIPE_STAGES: Final = 2
EXPERT_ATOMS: Final = 4
PHYSICAL_MATH_ROLES: Final = 4
LOGICAL_ROLES: Final = 5
ROUTING_ROLES: Final = 5
THREADS_PER_ROLE: Final = 128
TOTAL_THREADS: Final = LOGICAL_ROLES * THREADS_PER_ROLE
LOADER_MAXNREG: Final = 64
SINGLE_MATH_MAXNREG: Final = 104
KERNEL_MAXNREG: Final = 96
TOTAL_ROUTING_WARPS: Final = ROUTING_ROLES * 4
OWNER_ROUNDS: Final = (BLOCK_TOKENS + TOTAL_ROUTING_WARPS - 1) // TOTAL_ROUTING_WARPS

PIPE_BYTES_PER_STAGE: Final = EXPERTS_PER_CTA * BLOCK_K * 2 + BLOCK_TOKENS * BLOCK_K * 2
NOMINAL_PIPE_BYTES: Final = PIPE_STAGES * PIPE_BYTES_PER_STAGE
BIAS_BYTES: Final = EXPERTS * 4
NOMINAL_SMEM_BYTES: Final = NOMINAL_PIPE_BYTES + BIAS_BYTES
REGISTER_BUDGET: Final = THREADS_PER_ROLE * (
    LOADER_MAXNREG + PHYSICAL_MATH_ROLES * SINGLE_MATH_MAXNREG
)

assert (HIDDEN, EXPERTS, TOPK) == (4096, 256, 6)
assert SUPPORTED_TOKENS == (7884, 8221, 13021, 16384)
assert EXPERTS == EXPERT_ATOMS * EXPERTS_PER_MATH
assert EXPERTS == LOGICAL_EXPERT_CTAS * EXPERTS_PER_CTA
assert (BLOCK_TOKENS, BLOCK_K, NUM_K_TILES) == (128, 128, 32)
assert (PIPE_STAGES, PHYSICAL_MATH_ROLES, TOTAL_THREADS) == (2, 4, 640)
assert PIPE_BYTES_PER_STAGE == 98_304
assert NOMINAL_PIPE_BYTES == 196_608
assert NOMINAL_SMEM_BYTES == 197_632
assert REGISTER_BUDGET == 61_440
assert (TOTAL_ROUTING_WARPS, OWNER_ROUNDS) == (20, 7)

JIT_EXPERTS = tl.constexpr(EXPERTS)
JIT_BLOCK_TOKENS = tl.constexpr(BLOCK_TOKENS)
JIT_EXPERTS_PER_MATH = tl.constexpr(EXPERTS_PER_MATH)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_NUM_K_TILES = tl.constexpr(NUM_K_TILES)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_TOTAL_THREADS = tl.constexpr(TOTAL_THREADS)
JIT_SINGLE_MATH_MAXNREG = tl.constexpr(SINGLE_MATH_MAXNREG)
JIT_TOTAL_ROUTING_WARPS = tl.constexpr(TOTAL_ROUTING_WARPS)
JIT_OWNER_ROUNDS = tl.constexpr(OWNER_ROUNDS)


@triton.jit
def _route(
    activated_score_ptr,
    bias_smem,
    topk_idx_ptr,
    topk_weight_ptr,
    token_base,
    tokens,
    ROUTING_WARP_BASE: tl.constexpr,
):
    _common._route_two_wave_owner_tokens(
        activated_score_ptr,
        bias_smem,
        topk_idx_ptr,
        topk_weight_ptr,
        token_base,
        tokens,
        ROUTING_WARP_BASE,
        JIT_TOTAL_ROUTING_WARPS,
        JIT_OWNER_ROUNDS,
    )


@triton.jit
def _load_route(
    writer,
    weight_desc,
    x_desc,
    activated_score_ptr,
    bias_ptr,
    bias_smem,
    topk_idx_ptr,
    topk_weight_ptr,
    logits_barrier,
    token_base,
    tokens,
):
    lane = tl.arange(0, 128)
    tl.store(
        tle.gpu.local_ptr(bias_smem, (lane,)),
        tl.load(bias_ptr + lane).to(tl.float32),
    )
    tl.store(
        tle.gpu.local_ptr(bias_smem, (lane + 128,)),
        tl.load(bias_ptr + lane + 128).to(tl.float32),
    )
    for ticket in tl.range(0, JIT_NUM_K_TILES):
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
    tle.gpu.barrier_wait(logits_barrier)
    _route(
        activated_score_ptr,
        bias_smem,
        topk_idx_ptr,
        topk_weight_ptr,
        token_base,
        tokens,
        0,
    )


@triton.jit
def _issue_single(stage, accumulator, EXPERT_ATOM: tl.constexpr):
    weight_atom = stage.slot.weight.subslice(
        EXPERT_ATOM * JIT_EXPERTS_PER_MATH,
        JIT_EXPERTS_PER_MATH,
        0,
    )
    return tle.gpu.wgmma(
        weight_atom,
        stage.slot.x,
        accumulator,
        trans_b=True,
    )


@triton.jit
def _store_activated_atom(
    accumulator,
    activated_score_ptr,
    token_base,
    tokens,
    EXPERT_ATOM: tl.constexpr,
):
    token = token_base + tl.arange(0, JIT_BLOCK_TOKENS)[None, :]
    expert = (
        EXPERT_ATOM * JIT_EXPERTS_PER_MATH + tl.arange(0, JIT_EXPERTS_PER_MATH)[:, None]
    )
    activated = _ordered._precise_sqrt_softplus(accumulator)
    tl.store(
        activated_score_ptr + token * JIT_EXPERTS + expert,
        activated,
        mask=token < tokens,
    )


@triton.jit
def _math_single(
    reader,
    activated_score_ptr,
    token_base,
    tokens,
    EXPERT_ATOM: tl.constexpr,
):
    accumulator = tl.zeros(
        (JIT_EXPERTS_PER_MATH, JIT_BLOCK_TOKENS),
        dtype=tl.float32,
    )
    for ticket in tl.range(0, JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        accumulator = _issue_single(stage, accumulator, EXPERT_ATOM)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        reader.release(ticket)
    _store_activated_atom(
        accumulator,
        activated_score_ptr,
        token_base,
        tokens,
        EXPERT_ATOM,
    )


@triton.jit
def _math_route(
    reader,
    activated_score_ptr,
    bias_smem,
    topk_idx_ptr,
    topk_weight_ptr,
    logits_barrier,
    token_base,
    tokens,
    EXPERT_ATOM: tl.constexpr,
    ROLE_ID: tl.constexpr,
):
    _math_single(
        reader,
        activated_score_ptr,
        token_base,
        tokens,
        EXPERT_ATOM,
    )
    tle.gpu.barrier_wait(logits_barrier)
    _route(
        activated_score_ptr,
        bias_smem,
        topk_idx_ptr,
        topk_weight_ptr,
        token_base,
        tokens,
        ROLE_ID * 4,
    )


@triton.jit
def _fused_e256_quadmath_kernel(
    weight_desc,
    x_desc,
    activated_score_ptr,
    bias_ptr,
    topk_idx_ptr,
    topk_weight_ptr,
    tokens,
):
    token_tile = tl.program_id(0)
    token_base = token_tile * JIT_BLOCK_TOKENS

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
    # Keep the admitted pipe identity stable: FlagTree uses outlined pipe
    # identities while allocating module-wide named barriers.
    pipe = tle.pipe(
        capacity=JIT_PIPE_STAGES,
        scope="cta",
        name="dsv4_h4096_e256_b128_quadmath_singlewave_v376",
        readers=("math0", "math1", "math2", "math3"),
        weight=weight_smem,
        x=x_smem,
    )
    bias_smem = tle.gpu.alloc(
        [JIT_EXPERTS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    logits_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_TOTAL_THREADS)

    tle.gpu.warp_specialize(
        [
            (
                _load_route,
                (
                    pipe.writer(),
                    weight_desc,
                    x_desc,
                    activated_score_ptr,
                    bias_ptr,
                    bias_smem,
                    topk_idx_ptr,
                    topk_weight_ptr,
                    logits_barrier,
                    token_base,
                    tokens,
                ),
            ),
            (
                _math_route,
                (
                    pipe.reader(name="math0", fields=("weight", "x")),
                    activated_score_ptr,
                    bias_smem,
                    topk_idx_ptr,
                    topk_weight_ptr,
                    logits_barrier,
                    token_base,
                    tokens,
                    0,
                    1,
                ),
            ),
            (
                _math_route,
                (
                    pipe.reader(name="math1", fields=("weight", "x")),
                    activated_score_ptr,
                    bias_smem,
                    topk_idx_ptr,
                    topk_weight_ptr,
                    logits_barrier,
                    token_base,
                    tokens,
                    1,
                    2,
                ),
            ),
            (
                _math_route,
                (
                    pipe.reader(name="math2", fields=("weight", "x")),
                    activated_score_ptr,
                    bias_smem,
                    topk_idx_ptr,
                    topk_weight_ptr,
                    logits_barrier,
                    token_base,
                    tokens,
                    2,
                    3,
                ),
            ),
            (
                _math_route,
                (
                    pipe.reader(name="math3", fields=("weight", "x")),
                    activated_score_ptr,
                    bias_smem,
                    topk_idx_ptr,
                    topk_weight_ptr,
                    logits_barrier,
                    token_base,
                    tokens,
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


__all__ = [
    "KERNEL_MAXNREG",
    "SUPPORTED_TOKENS",
    "_fused_e256_quadmath_kernel",
]
