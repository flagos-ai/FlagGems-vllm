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

"""E256/B128 quad-math DSV3.2 MegaGate kernel (source leaf v435)."""

from __future__ import annotations

from typing import Final

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from . import dsv32_common as _common

SOURCE_LEAF: Final = "v435"

SUPPORTED_TOKENS: Final = (14305, 16384)
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
TOTAL_THREADS: Final = 640
LOADER_MAXNREG: Final = 64
SINGLE_MATH_MAXNREG: Final = 104
KERNEL_MAXNREG: Final = 96

PIPE_BYTES_PER_STAGE: Final = EXPERTS_PER_CTA * BLOCK_K * 2 + BLOCK_TOKENS * BLOCK_K * 2
NOMINAL_PIPE_BYTES: Final = PIPE_STAGES * PIPE_BYTES_PER_STAGE
REGISTER_BUDGET: Final = 128 * (
    LOADER_MAXNREG + PHYSICAL_MATH_ROLES * SINGLE_MATH_MAXNREG
)

assert (HIDDEN, EXPERTS, TOPK) == (7168, 256, 8)
assert EXPERTS == LOGICAL_EXPERT_CTAS * EXPERTS_PER_CTA
assert EXPERTS_PER_CTA == EXPERT_ATOMS * EXPERTS_PER_MATH
assert (BLOCK_TOKENS, BLOCK_K, NUM_K_TILES) == (128, 128, 56)
assert (PIPE_STAGES, PHYSICAL_MATH_ROLES, TOTAL_THREADS) == (2, 4, 640)
assert NOMINAL_PIPE_BYTES == 196_608
assert REGISTER_BUDGET == 61_440

JIT_EXPERTS = tl.constexpr(EXPERTS)
JIT_BLOCK_TOKENS = tl.constexpr(BLOCK_TOKENS)
JIT_EXPERTS_PER_MATH = tl.constexpr(EXPERTS_PER_MATH)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_NUM_K_TILES = tl.constexpr(NUM_K_TILES)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_LOGICAL_ROLES = tl.constexpr(LOGICAL_ROLES)
JIT_ROUTING_ROLES = tl.constexpr(ROUTING_ROLES)
JIT_TOTAL_THREADS = tl.constexpr(TOTAL_THREADS)
JIT_SINGLE_MATH_MAXNREG = tl.constexpr(SINGLE_MATH_MAXNREG)


@triton.jit
def _load(writer, weight_desc, x_desc, token_base):
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


@triton.jit
def _issue_single(stage, accumulator, EXPERT_ATOM: tl.constexpr):
    weight_atom = stage.slot.weight.subslice(
        EXPERT_ATOM * JIT_EXPERTS_PER_MATH,
        JIT_EXPERTS_PER_MATH,
        0,
    )
    return tle.gpu.wgmma(weight_atom, stage.slot.x, accumulator, trans_b=True)


@triton.jit
def _store_atom(
    accumulator,
    logits_ptr,
    token_base,
    TOKENS,
    EXPERT_ATOM: tl.constexpr,
):
    token = token_base + tl.arange(0, JIT_BLOCK_TOKENS)[None, :]
    expert = (
        EXPERT_ATOM * JIT_EXPERTS_PER_MATH + tl.arange(0, JIT_EXPERTS_PER_MATH)[:, None]
    )
    # Preserve the WGMMA accumulator's [expert, token] layout.  Transposing
    # the value forces an MMA-to-blocked conversion through shared memory.
    bounded = accumulator.to(tl.bfloat16)
    tl.store(
        logits_ptr + token * JIT_EXPERTS + expert,
        bounded,
        mask=token < TOKENS,
    )


@triton.jit
def _math_single(
    reader,
    logits_ptr,
    token_base,
    TOKENS,
    EXPERT_ATOM: tl.constexpr,
):
    accumulator = tl.zeros((JIT_EXPERTS_PER_MATH, JIT_BLOCK_TOKENS), dtype=tl.float32)
    for ticket in tl.range(0, JIT_NUM_K_TILES):
        stage = reader.wait(ticket)
        accumulator = _issue_single(stage, accumulator, EXPERT_ATOM)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        reader.release(ticket)
    _store_atom(accumulator, logits_ptr, token_base, TOKENS, EXPERT_ATOM)


@triton.jit
def _route(
    logits_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    token_base,
    TOKENS,
    ROLE_ID: tl.constexpr,
):
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
        JIT_ROUTING_ROLES,
        True,
    )


@triton.jit
def _loader_route(
    writer,
    weight_desc,
    x_desc,
    logits_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    logits_barrier,
    token_base,
    TOKENS,
):
    _load(writer, weight_desc, x_desc, token_base)
    tle.gpu.barrier_wait(logits_barrier)
    _route(
        logits_ptr,
        bias_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        routed_scaling_factor,
        token_base,
        TOKENS,
        0,
    )


@triton.jit
def _single_math_route(
    reader,
    logits_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    logits_barrier,
    token_base,
    TOKENS,
    EXPERT_ATOM: tl.constexpr,
    ROLE_ID: tl.constexpr,
):
    _math_single(reader, logits_ptr, token_base, TOKENS, EXPERT_ATOM)
    tle.gpu.barrier_wait(logits_barrier)
    _route(
        logits_ptr,
        bias_ptr,
        topk_indices_ptr,
        topk_weights_ptr,
        routed_scaling_factor,
        token_base,
        TOKENS,
        ROLE_ID,
    )


@triton.jit
def _fused_e256_quadmath_kernel(
    weight_desc,
    x_desc,
    logits_ptr,
    bias_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    routed_scaling_factor,
    TOKENS,
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
    pipe = tle.pipe(
        capacity=JIT_PIPE_STAGES,
        scope="cta",
        name="dsv32_h7168_e256_quadmath_selected_reload_v435",
        readers=("math0", "math1", "math2", "math3"),
        weight=weight_smem,
        x=x_smem,
    )
    logits_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_TOTAL_THREADS)

    tle.gpu.warp_specialize(
        [
            (
                _loader_route,
                (
                    pipe.writer(),
                    weight_desc,
                    x_desc,
                    logits_ptr,
                    bias_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    routed_scaling_factor,
                    logits_barrier,
                    token_base,
                    TOKENS,
                ),
            ),
            (
                _single_math_route,
                (
                    pipe.reader(name="math0", fields=("weight", "x")),
                    logits_ptr,
                    bias_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    routed_scaling_factor,
                    logits_barrier,
                    token_base,
                    TOKENS,
                    0,
                    1,
                ),
            ),
            (
                _single_math_route,
                (
                    pipe.reader(name="math1", fields=("weight", "x")),
                    logits_ptr,
                    bias_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    routed_scaling_factor,
                    logits_barrier,
                    token_base,
                    TOKENS,
                    1,
                    2,
                ),
            ),
            (
                _single_math_route,
                (
                    pipe.reader(name="math2", fields=("weight", "x")),
                    logits_ptr,
                    bias_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    routed_scaling_factor,
                    logits_barrier,
                    token_base,
                    TOKENS,
                    2,
                    3,
                ),
            ),
            (
                _single_math_route,
                (
                    pipe.reader(name="math3", fields=("weight", "x")),
                    logits_ptr,
                    bias_ptr,
                    topk_indices_ptr,
                    topk_weights_ptr,
                    routed_scaling_factor,
                    logits_barrier,
                    token_base,
                    TOKENS,
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


__all__ = ["KERNEL_MAXNREG", "SUPPORTED_TOKENS", "_fused_e256_quadmath_kernel"]
