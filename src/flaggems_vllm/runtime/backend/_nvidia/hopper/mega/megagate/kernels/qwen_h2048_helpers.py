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

"""Device helpers shared by the admitted Qwen-H2048 MegaGate leaves."""

import triton
import triton.language as tl


@triton.jit
def _compare_swap_local(lhs_score, lhs_offset, rhs_score, rhs_offset):
    take_rhs = (rhs_score > lhs_score) | (
        (rhs_score == lhs_score) & (rhs_offset < lhs_offset)
    )
    return (
        tl.where(take_rhs, rhs_score, lhs_score),
        tl.where(take_rhs, rhs_offset, lhs_offset),
        tl.where(take_rhs, lhs_score, rhs_score),
        tl.where(take_rhs, lhs_offset, rhs_offset),
    )


@triton.jit
def _sort4_pack(s0, s1, s2, s3):
    """CUDA sort4 network with offsets retired into one packed register."""
    i0 = tl.zeros((128,), dtype=tl.uint32)
    i1 = tl.full((128,), 1, dtype=tl.uint32)
    i2 = tl.full((128,), 2, dtype=tl.uint32)
    i3 = tl.full((128,), 3, dtype=tl.uint32)
    s0, i0, s1, i1 = _compare_swap_local(s0, i0, s1, i1)
    s2, i2, s3, i3 = _compare_swap_local(s2, i2, s3, i3)
    s0, i0, s2, i2 = _compare_swap_local(s0, i0, s2, i2)
    s1, i1, s3, i3 = _compare_swap_local(s1, i1, s3, i3)
    s1, i1, s2, i2 = _compare_swap_local(s1, i1, s2, i2)
    permutation = i0 | (i1 << 2) | (i2 << 4) | (i3 << 6)
    return s0, s1, s2, s3, permutation


@triton.jit
def _sorted_head(cursor, s0, s1, s2, s3):
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
def _pair2_stable_best_expert(score_key, expert):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b32 best_key, best_expert, peer_key, peer_expert;
            .reg .pred p_key_gt, p_idx_lt, p_tie, p_take;
            mov.u32 best_key, $1;
            mov.u32 best_expert, $2;

            shfl.sync.down.b32 peer_key, best_key, 1, 0x1e01, 0xffffffff;
            shfl.sync.down.b32 peer_expert, best_expert, 1, 0x1e01, 0xffffffff;
            setp.gt.u32 p_key_gt, peer_key, best_key;
            setp.lt.u32 p_idx_lt, peer_expert, best_expert;
            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;
            or.pred p_take, p_key_gt, p_tie;
            selp.u32 best_expert, peer_expert, best_expert, p_take;

            shfl.sync.idx.b32 $0, best_expert, 0, 0x1e01, 0xffffffff;
        }
        """,
        "=r,r,r",
        [score_key.to(tl.uint32), expert.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _pair2_broadcast_f32(value, source_lane):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b32 input_bits, output_bits;
            mov.b32 input_bits, $1;
            shfl.sync.idx.b32 output_bits, input_bits, $2, 0x1e01, 0xffffffff;
            mov.b32 $0, output_bits;
        }
        """,
        "=f,f,r",
        [value, source_lane.to(tl.uint32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _pair2_sum_f32(value):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 current, peer;
            .reg .b32 current_bits, peer_bits;
            mov.f32 current, $1;
            mov.b32 current_bits, current;
            shfl.sync.down.b32 peer_bits, current_bits, 1, 0x1e01, 0xffffffff;
            mov.b32 peer, peer_bits;
            add.rn.f32 current, current, peer;
            mov.b32 current_bits, current;
            shfl.sync.idx.b32 peer_bits, current_bits, 0, 0x1e01, 0xffffffff;
            mov.b32 $0, peer_bits;
        }
        """,
        "=f,f",
        [value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _halfwarp16_stable_best_expert(score_key, expert):
    """CUDA-v020 stable argmax in two independent width-16 subwarps."""

    return tl.inline_asm_elementwise(
        """
        {
            .reg .b32 best_key, best_expert, peer_key, peer_expert;
            .reg .pred p_key_gt, p_idx_lt, p_tie, p_take;
            mov.u32 best_key, $1;
            mov.u32 best_expert, $2;

            shfl.sync.down.b32 peer_key, best_key, 8, 0x100f, 0xffffffff;
            shfl.sync.down.b32 peer_expert, best_expert, 8, 0x100f, 0xffffffff;
            setp.gt.u32 p_key_gt, peer_key, best_key;
            setp.lt.u32 p_idx_lt, peer_expert, best_expert;
            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;
            or.pred p_take, p_key_gt, p_tie;
            selp.u32 best_key, peer_key, best_key, p_take;
            selp.u32 best_expert, peer_expert, best_expert, p_take;

            shfl.sync.down.b32 peer_key, best_key, 4, 0x100f, 0xffffffff;
            shfl.sync.down.b32 peer_expert, best_expert, 4, 0x100f, 0xffffffff;
            setp.gt.u32 p_key_gt, peer_key, best_key;
            setp.lt.u32 p_idx_lt, peer_expert, best_expert;
            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;
            or.pred p_take, p_key_gt, p_tie;
            selp.u32 best_key, peer_key, best_key, p_take;
            selp.u32 best_expert, peer_expert, best_expert, p_take;

            shfl.sync.down.b32 peer_key, best_key, 2, 0x100f, 0xffffffff;
            shfl.sync.down.b32 peer_expert, best_expert, 2, 0x100f, 0xffffffff;
            setp.gt.u32 p_key_gt, peer_key, best_key;
            setp.lt.u32 p_idx_lt, peer_expert, best_expert;
            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;
            or.pred p_take, p_key_gt, p_tie;
            selp.u32 best_key, peer_key, best_key, p_take;
            selp.u32 best_expert, peer_expert, best_expert, p_take;

            shfl.sync.down.b32 peer_key, best_key, 1, 0x100f, 0xffffffff;
            shfl.sync.down.b32 peer_expert, best_expert, 1, 0x100f, 0xffffffff;
            setp.gt.u32 p_key_gt, peer_key, best_key;
            setp.lt.u32 p_idx_lt, peer_expert, best_expert;
            setp.eq.and.u32 p_tie, peer_key, best_key, p_idx_lt;
            or.pred p_take, p_key_gt, p_tie;
            selp.u32 best_key, peer_key, best_key, p_take;
            selp.u32 best_expert, peer_expert, best_expert, p_take;

            shfl.sync.idx.b32 $0, best_expert, 0, 0x100f, 0xffffffff;
        }
        """,
        "=r,r,r",
        [score_key.to(tl.uint32), expert.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _halfwarp16_broadcast_f32(value, source_lane):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b32 input_bits, output_bits;
            mov.b32 input_bits, $1;
            shfl.sync.idx.b32 output_bits, input_bits, $2, 0x100f, 0xffffffff;
            mov.b32 $0, output_bits;
        }
        """,
        "=f,f,r",
        [value, source_lane.to(tl.uint32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _halfwarp16_sum_f32(value):
    """Reduce and broadcast one sum independently in each width-16 group."""

    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 current, peer;
            .reg .b32 current_bits, peer_bits;
            mov.f32 current, $1;

            mov.b32 current_bits, current;
            shfl.sync.down.b32 peer_bits, current_bits, 8, 0x100f, 0xffffffff;
            mov.b32 peer, peer_bits;
            add.rn.f32 current, current, peer;

            mov.b32 current_bits, current;
            shfl.sync.down.b32 peer_bits, current_bits, 4, 0x100f, 0xffffffff;
            mov.b32 peer, peer_bits;
            add.rn.f32 current, current, peer;

            mov.b32 current_bits, current;
            shfl.sync.down.b32 peer_bits, current_bits, 2, 0x100f, 0xffffffff;
            mov.b32 peer, peer_bits;
            add.rn.f32 current, current, peer;

            mov.b32 current_bits, current;
            shfl.sync.down.b32 peer_bits, current_bits, 1, 0x100f, 0xffffffff;
            mov.b32 peer, peer_bits;
            add.rn.f32 current, current, peer;

            mov.b32 current_bits, current;
            shfl.sync.idx.b32 peer_bits, current_bits, 0, 0x100f, 0xffffffff;
            mov.b32 $0, peer_bits;
        }
        """,
        "=f,f",
        [value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _load_score_float4(ptr, valid):
    """CUDA-v020's aligned float4 score load with invalid-row defaults."""

    address = ptr.to(tl.uint64)
    b0, b1, b2, b3 = tl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            .reg .b64 cache_hint;
            mov.u32 $0, 0xff800000;
            mov.u32 $1, 0xff800000;
            mov.u32 $2, 0xff800000;
            mov.u32 $3, 0xff800000;
            mov.u64 cache_hint, 0x12f0000000000000;
            setp.ne.u32 p, $5, 0;
            @p ld.weak.global.L1::no_allocate.L2::cache_hint.v4.b32
                {$0, $1, $2, $3}, [$4], cache_hint;
        }
        """,
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
