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

"""Shared machinery for the DSV4 B128 clusterless MegaGate kernels.

The score workspace is token-major FP32 ``[T, 256]`` and contains precise
sqrt-softplus values without correction bias.  Bias participates only in the
stable top-6 ranking; selected unbiased values are normalized and multiplied
by 1.5, matching the CUDA and vLLM DSV4 routing contract.
"""

from __future__ import annotations

from typing import Final

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from . import ordered_routing as _ordered
from . import routing_primitives as _redux

HIDDEN: Final = 4096
EXPERTS: Final = 256
TOPK: Final = 6
ROUTED_SCALING_FACTOR: Final = 1.5
EXPERTS_PER_CTA: Final = 64
LOGICAL_EXPERT_CTAS: Final = 4
BLOCK_TOKENS: Final = 128
BLOCK_K: Final = 128
NUM_K_TILES: Final = HIDDEN // BLOCK_K
PIPE_STAGES: Final = 4
BIAS_SMEM_EXTENT: Final = 256
ROUTING_WARPS: Final = 4

assert EXPERTS == EXPERTS_PER_CTA * LOGICAL_EXPERT_CTAS
assert NUM_K_TILES == 32

JIT_EXPERTS = tl.constexpr(EXPERTS)
JIT_TOPK = tl.constexpr(TOPK)
JIT_SCALE = tl.constexpr(ROUTED_SCALING_FACTOR)
JIT_EXPERTS_PER_CTA = tl.constexpr(EXPERTS_PER_CTA)
JIT_LOGICAL_EXPERT_CTAS = tl.constexpr(LOGICAL_EXPERT_CTAS)
JIT_BLOCK_TOKENS = tl.constexpr(BLOCK_TOKENS)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_NUM_K_TILES = tl.constexpr(NUM_K_TILES)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_BIAS_SMEM_EXTENT = tl.constexpr(BIAS_SMEM_EXTENT)
JIT_ROUTING_WARPS = tl.constexpr(ROUTING_WARPS)


def _allocator(size: int, alignment: int, stream: int) -> torch.Tensor:
    del alignment, stream
    return torch.empty(
        size,
        device=torch.device("cuda", torch.cuda.current_device()),
        dtype=torch.uint8,
    )


@triton.jit
def _route_two_wave_owner_tokens(
    activated_score_ptr,
    bias_smem,
    topk_idx_ptr,
    topk_weight_ptr,
    token_base,
    tokens,
    routing_warp_base: tl.constexpr,
    total_routing_warps: tl.constexpr,
    owner_rounds: tl.constexpr,
):
    """Route disjoint B128 rows across one or more four-warp roles."""

    route = tl.arange(0, 128)
    routing_warp = routing_warp_base + route // 32
    lane = route % 32
    invalid_expert = tl.full((128,), 0xFFFFFFFF, dtype=tl.uint32)

    for owner_round in tl.range(
        0,
        owner_rounds,
        loop_unroll_factor=1,
    ):
        local_token = routing_warp + owner_round * total_routing_warps
        global_token = token_base + local_token
        valid_token = (local_token < JIT_BLOCK_TOKENS) & (global_token < tokens)
        score_row = global_token * JIT_EXPERTS

        expert00 = lane * 4 + 0
        expert01 = lane * 4 + 1
        expert02 = lane * 4 + 2
        expert03 = lane * 4 + 3
        expert10 = expert00 + 128
        expert11 = expert01 + 128
        expert12 = expert02 + 128
        expert13 = expert03 + 128

        unbiased00 = tl.load(
            activated_score_ptr + score_row + expert00,
            mask=valid_token,
            other=0.0,
        )
        unbiased01 = tl.load(
            activated_score_ptr + score_row + expert01,
            mask=valid_token,
            other=0.0,
        )
        unbiased02 = tl.load(
            activated_score_ptr + score_row + expert02,
            mask=valid_token,
            other=0.0,
        )
        unbiased03 = tl.load(
            activated_score_ptr + score_row + expert03,
            mask=valid_token,
            other=0.0,
        )
        unbiased10 = tl.load(
            activated_score_ptr + score_row + expert10,
            mask=valid_token,
            other=0.0,
        )
        unbiased11 = tl.load(
            activated_score_ptr + score_row + expert11,
            mask=valid_token,
            other=0.0,
        )
        unbiased12 = tl.load(
            activated_score_ptr + score_row + expert12,
            mask=valid_token,
            other=0.0,
        )
        unbiased13 = tl.load(
            activated_score_ptr + score_row + expert13,
            mask=valid_token,
            other=0.0,
        )

        score00 = tl.where(
            valid_token,
            unbiased00 + tl.load(tle.gpu.local_ptr(bias_smem, (expert00,))),
            -float("inf"),
        )
        score01 = tl.where(
            valid_token,
            unbiased01 + tl.load(tle.gpu.local_ptr(bias_smem, (expert01,))),
            -float("inf"),
        )
        score02 = tl.where(
            valid_token,
            unbiased02 + tl.load(tle.gpu.local_ptr(bias_smem, (expert02,))),
            -float("inf"),
        )
        score03 = tl.where(
            valid_token,
            unbiased03 + tl.load(tle.gpu.local_ptr(bias_smem, (expert03,))),
            -float("inf"),
        )
        score10 = tl.where(
            valid_token,
            unbiased10 + tl.load(tle.gpu.local_ptr(bias_smem, (expert10,))),
            -float("inf"),
        )
        score11 = tl.where(
            valid_token,
            unbiased11 + tl.load(tle.gpu.local_ptr(bias_smem, (expert11,))),
            -float("inf"),
        )
        score12 = tl.where(
            valid_token,
            unbiased12 + tl.load(tle.gpu.local_ptr(bias_smem, (expert12,))),
            -float("inf"),
        )
        score13 = tl.where(
            valid_token,
            unbiased13 + tl.load(tle.gpu.local_ptr(bias_smem, (expert13,))),
            -float("inf"),
        )

        (
            score00,
            expert00,
            score01,
            expert01,
            score02,
            expert02,
            score03,
            expert03,
        ) = _ordered._sort4_best_first(
            score00,
            expert00,
            score01,
            expert01,
            score02,
            expert02,
            score03,
            expert03,
        )
        (
            score10,
            expert10,
            score11,
            expert11,
            score12,
            expert12,
            score13,
            expert13,
        ) = _ordered._sort4_best_first(
            score10,
            expert10,
            score11,
            expert11,
            score12,
            expert12,
            score13,
            expert13,
        )

        cursor0 = tl.zeros((128,), dtype=tl.int32)
        cursor1 = tl.zeros((128,), dtype=tl.int32)
        selected_for_lane = tl.zeros((128,), dtype=tl.int32)

        for output_idx in tl.static_range(0, JIT_TOPK):
            head_score0, head_expert0 = _ordered._head4(
                cursor0,
                score00,
                expert00,
                score01,
                expert01,
                score02,
                expert02,
                score03,
                expert03,
            )
            head_score1, head_expert1 = _ordered._head4(
                cursor1,
                score10,
                expert10,
                score11,
                expert11,
                score12,
                expert12,
                score13,
                expert13,
            )
            take_wave1 = head_score1 > head_score0
            local_score = tl.where(take_wave1, head_score1, head_score0)
            local_expert = tl.where(take_wave1, head_expert1, head_expert0)
            local_wave = tl.where(take_wave1, 1, 0)

            local_key = _ordered._ordered_score_key(local_score).to(tl.uint32)
            best_key = _redux.warp_max_u32(local_key)
            best_expert = _redux.warp_min_u32(
                tl.where(local_key == best_key, local_expert, invalid_expert)
            )
            owns_best = valid_token & (local_expert == best_expert)
            cursor0 += owns_best & (local_wave == 0)
            cursor1 += owns_best & (local_wave == 1)
            selected_for_lane = tl.where(
                lane == output_idx,
                best_expert.to(tl.int32),
                selected_for_lane,
            )

        valid_output = valid_token & (lane < JIT_TOPK)
        selected_expert = tl.where(valid_output, selected_for_lane, 0)
        selected_unbiased = tl.load(
            activated_score_ptr + score_row + selected_expert,
            mask=valid_output,
            other=0.0,
        )
        selected_unbiased = tl.where(valid_output, selected_unbiased, 0.0)
        topk_sum = _redux.subwarp8_sum_f32(selected_unbiased)
        topk_sum = (
            _redux.warp_broadcast_f32(topk_sum, tl.zeros((128,), dtype=tl.uint32))
            + 1.0e-20
        )
        output_offset = global_token * JIT_TOPK + lane
        tl.store(
            topk_idx_ptr + output_offset,
            selected_for_lane.to(tl.int64),
            mask=valid_output,
        )
        tl.store(
            topk_weight_ptr + output_offset,
            selected_unbiased * (JIT_SCALE / topk_sum),
            mask=valid_output,
        )


__all__ = [
    "BIAS_SMEM_EXTENT",
    "BLOCK_K",
    "BLOCK_TOKENS",
    "EXPERTS",
    "EXPERTS_PER_CTA",
    "HIDDEN",
    "LOGICAL_EXPERT_CTAS",
    "NUM_K_TILES",
    "PIPE_STAGES",
    "ROUTED_SCALING_FACTOR",
    "TOPK",
    "TensorDescriptor",
    "_allocator",
    "_route_two_wave_owner_tokens",
]
