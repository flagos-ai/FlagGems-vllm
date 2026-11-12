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

"""Small SM90 routing primitives shared by TLE MegaGate variants.

The functions deliberately preserve the instruction-level behavior used by
the validated T=1 and T=512 routing selectors: precise sqrt-softplus, an
ordered unsigned score key, warp-wide integer reductions, one-lane broadcast,
and an eight-lane FP32 sum.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def ordered_score_key(score):
    score = tl.where(score == score, score, -float("inf"))
    bits = score.to(tl.uint32, bitcast=True)
    bits = tl.where(score == 0.0, 0, bits)
    return tl.where(
        (bits & 0x80000000) != 0,
        ~bits,
        bits ^ 0x80000000,
    )


@triton.jit
def precise_sqrt_softplus(raw):
    exponential = libdevice.exp(raw)
    softplus = libdevice.log1p(exponential)
    return libdevice.sqrt(tl.where(raw > 20.0, raw, softplus))


@triton.jit
def subwarp8_sum_f32(value):
    """Reduce four independent eight-lane groups in one physical warp."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 current, peer;
            .reg .b32 current_bits, peer_bits;
            mov.f32 current, $1;

            mov.b32 current_bits, current;
            shfl.sync.down.b32 peer_bits, current_bits, 4, 0x1807, 0xffffffff;
            mov.b32 peer, peer_bits;
            add.rn.f32 current, current, peer;

            mov.b32 current_bits, current;
            shfl.sync.down.b32 peer_bits, current_bits, 2, 0x1807, 0xffffffff;
            mov.b32 peer, peer_bits;
            add.rn.f32 current, current, peer;

            mov.b32 current_bits, current;
            shfl.sync.down.b32 peer_bits, current_bits, 1, 0x1807, 0xffffffff;
            mov.b32 peer, peer_bits;
            add.rn.f32 $0, current, peer;
        }
        """,
        "=f,f",
        [value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def warp_max_u32(value):
    return tl.inline_asm_elementwise(
        "redux.sync.max.u32 $0, $1, 0xffffffff;",
        "=r,r",
        [value.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def warp_min_u32(value):
    return tl.inline_asm_elementwise(
        "redux.sync.min.u32 $0, $1, 0xffffffff;",
        "=r,r",
        [value.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def warp_broadcast_f32(value, source_lane):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b32 input_bits, output_bits;
            mov.b32 input_bits, $1;
            shfl.sync.idx.b32 output_bits, input_bits, $2, 0x1f, 0xffffffff;
            mov.b32 $0, output_bits;
        }
        """,
        "=f,f,r",
        [value, source_lane.to(tl.uint32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
