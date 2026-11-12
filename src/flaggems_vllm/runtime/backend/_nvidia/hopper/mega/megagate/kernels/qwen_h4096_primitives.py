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

"""Pure inline routing helpers shared by Qwen H4096 MegaGate leaves.

These helpers intentionally exclude pipeline, barrier, shared-memory, and
warp-specialized role state so exact-shape entry kernels remain independent.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def compare_swap_local(lhs_score, lhs_offset, rhs_score, rhs_offset):
    take_rhs = (rhs_score > lhs_score) | (rhs_score == lhs_score) & (
        rhs_offset < lhs_offset
    )
    return (
        tl.where(take_rhs, rhs_score, lhs_score),
        tl.where(take_rhs, rhs_offset, lhs_offset),
        tl.where(take_rhs, lhs_score, rhs_score),
        tl.where(take_rhs, lhs_offset, rhs_offset),
    )


@triton.jit
def sort4_pack(s0, s1, s2, s3):
    """CUDA sort4 network with offsets retired into one packed register."""
    i0 = tl.zeros((128,), dtype=tl.uint32)
    i1 = tl.full((128,), 1, dtype=tl.uint32)
    i2 = tl.full((128,), 2, dtype=tl.uint32)
    i3 = tl.full((128,), 3, dtype=tl.uint32)
    s0, i0, s1, i1 = compare_swap_local(s0, i0, s1, i1)
    s2, i2, s3, i3 = compare_swap_local(s2, i2, s3, i3)
    s0, i0, s2, i2 = compare_swap_local(s0, i0, s2, i2)
    s1, i1, s3, i3 = compare_swap_local(s1, i1, s3, i3)
    s1, i1, s2, i2 = compare_swap_local(s1, i1, s2, i2)
    permutation = i0 | i1 << 2 | i2 << 4 | i3 << 6
    return (s0, s1, s2, s3, permutation)


@triton.jit
def sorted_head(cursor, s0, s1, s2, s3):
    return tl.where(
        cursor == 0,
        s0,
        tl.where(
            cursor == 1,
            s1,
            tl.where(cursor == 2, s2, tl.where(cursor == 3, s3, -float("inf"))),
        ),
    )


@triton.jit
def load_score_float4(ptr, valid):
    """CUDA-v020's aligned float4 score load with invalid-row defaults."""
    address = ptr.to(tl.uint64)
    b0, b1, b2, b3 = tl.inline_asm_elementwise(
        "\n        {\n            .reg .pred p;\n            .reg .b64 cache_hint;\n            mov.u32 $0, 0xff800000;\n            mov.u32 $1, 0xff800000;\n            mov.u32 $2, 0xff800000;\n            mov.u32 $3, 0xff800000;\n            mov.u64 cache_hint, 0x12f0000000000000;\n            setp.ne.u32 p, $5, 0;\n            @p ld.weak.global.L1::no_allocate.L2::cache_hint.v4.b32\n                {$0, $1, $2, $3}, [$4], cache_hint;\n        }\n        ",
        "=r,=r,=r,=r,l,r,~{memory}",
        [address, valid.to(tl.uint32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    return (
        b0.to(tl.float32, bitcast=True),
        b1.to(tl.float32, bitcast=True),
        b2.to(tl.float32, bitcast=True),
        b3.to(tl.float32, bitcast=True),
    )


@triton.jit
def quadlane4_stable_best_expert(score_key, expert):
    """CUDA-v025 ordered best reduction in eight independent width-4 groups."""
    return tl.inline_asm_elementwise(
        "\n        {\n            .reg .b32 best_key, best_expert, peer_key, peer_expert;\n            .reg .pred p_key_gt, p_idx_lt, p_tie, p_take;\n            mov.u32 best_key, $1;\n            mov.u32 best_expert, $2;\n\n            shfl.sync.down.b32 peer_key, best_key, 2, 0x1c03, 0xffffffff;\n            shfl.sync.down.b32 peer_expert, best_expert, 2, 0x1c03, 0xffffffff;\n            setp.gt.u32 p_key_gt, peer_key, best_key;\n            setp.lt.u32 p_idx_lt, peer_expert, best_expert;\n            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;\n            or.pred p_take, p_key_gt, p_tie;\n            selp.u32 best_key, peer_key, best_key, p_take;\n            selp.u32 best_expert, peer_expert, best_expert, p_take;\n\n            shfl.sync.down.b32 peer_key, best_key, 1, 0x1c03, 0xffffffff;\n            shfl.sync.down.b32 peer_expert, best_expert, 1, 0x1c03, 0xffffffff;\n            setp.gt.u32 p_key_gt, peer_key, best_key;\n            setp.lt.u32 p_idx_lt, peer_expert, best_expert;\n            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;\n            or.pred p_take, p_key_gt, p_tie;\n            selp.u32 best_key, peer_key, best_key, p_take;\n            selp.u32 best_expert, peer_expert, best_expert, p_take;\n\n            shfl.sync.idx.b32 $0, best_expert, 0, 0x1c03, 0xffffffff;\n        }\n        ",
        "=r,r,r",
        [score_key.to(tl.uint32), expert.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def quadlane4_broadcast_f32(value, source_lane):
    return tl.inline_asm_elementwise(
        "\n        {\n            .reg .b32 input_bits, output_bits;\n            mov.b32 input_bits, $1;\n            shfl.sync.idx.b32 output_bits, input_bits, $2, 0x1c03, 0xffffffff;\n            mov.b32 $0, output_bits;\n        }\n        ",
        "=f,f,r",
        [value, source_lane.to(tl.uint32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def quadlane4_sum_f32(value):
    """Reduce and broadcast CUDA-v025's width-4 normalization sum."""
    return tl.inline_asm_elementwise(
        "\n        {\n            .reg .f32 current, peer;\n            .reg .b32 current_bits, peer_bits;\n            mov.f32 current, $1;\n\n            mov.b32 current_bits, current;\n            shfl.sync.down.b32 peer_bits, current_bits, 2, 0x1c03, 0xffffffff;\n            mov.b32 peer, peer_bits;\n            add.rn.f32 current, current, peer;\n\n            mov.b32 current_bits, current;\n            shfl.sync.down.b32 peer_bits, current_bits, 1, 0x1c03, 0xffffffff;\n            mov.b32 peer, peer_bits;\n            add.rn.f32 current, current, peer;\n\n            mov.b32 current_bits, current;\n            shfl.sync.idx.b32 peer_bits, current_bits, 0, 0x1c03, 0xffffffff;\n            mov.b32 $0, peer_bits;\n        }\n        ",
        "=f,f",
        [value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def pair2_broadcast_f32(value, source_lane):
    return tl.inline_asm_elementwise(
        "\n        {\n            .reg .b32 input_bits, output_bits;\n            mov.b32 input_bits, $1;\n            shfl.sync.idx.b32 output_bits, input_bits, $2, 0x1e01, 0xffffffff;\n            mov.b32 $0, output_bits;\n        }\n        ",
        "=f,f,r",
        [value, source_lane.to(tl.uint32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def pair2_sum_f32(value):
    return tl.inline_asm_elementwise(
        "\n        {\n            .reg .f32 current, peer;\n            .reg .b32 current_bits, peer_bits;\n            mov.f32 current, $1;\n            mov.b32 current_bits, current;\n            shfl.sync.down.b32 peer_bits, current_bits, 1, 0x1e01, 0xffffffff;\n            mov.b32 peer, peer_bits;\n            add.rn.f32 current, current, peer;\n            mov.b32 current_bits, current;\n            shfl.sync.idx.b32 peer_bits, current_bits, 0, 0x1e01, 0xffffffff;\n            mov.b32 $0, peer_bits;\n        }\n        ",
        "=f,f",
        [value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def pair2_max_u32(value):
    return tl.inline_asm_elementwise(
        "\n        {\n            .reg .b32 best, peer;\n            mov.u32 best, $1;\n            shfl.sync.down.b32 peer, best, 1, 0x1e01, 0xffffffff;\n            max.u32 best, best, peer;\n            shfl.sync.idx.b32 $0, best, 0, 0x1e01, 0xffffffff;\n        }\n        ",
        "=r,r",
        [value.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
