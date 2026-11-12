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

"""Ordered-score helpers shared by Hopper MegaGate routing code."""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _ordered_score_key(score):
    """Map finite FP32 scores to monotonically ordered unsigned keys."""

    bits = score.to(tl.uint32, bitcast=True)
    bits = tl.where(score == 0.0, 0, bits)
    return tl.where(
        (bits & 0x80000000) != 0,
        ~bits,
        bits ^ 0x80000000,
    )


@triton.jit
def _precise_sqrt_softplus(raw):
    exponential = libdevice.exp(raw)
    softplus = libdevice.log1p(exponential)
    return libdevice.sqrt(tl.where(raw > 20.0, raw, softplus))


@triton.jit
def _compare_swap_ordered(
    lhs_score,
    lhs_expert,
    rhs_score,
    rhs_expert,
):
    take_rhs = (rhs_score > lhs_score) | (
        (rhs_score == lhs_score) & (rhs_expert < lhs_expert)
    )
    return (
        tl.where(take_rhs, rhs_score, lhs_score),
        tl.where(take_rhs, rhs_expert, lhs_expert),
        tl.where(take_rhs, lhs_score, rhs_score),
        tl.where(take_rhs, lhs_expert, rhs_expert),
    )


@triton.jit
def _sort4_best_first(s0, e0, s1, e1, s2, e2, s3, e3):
    s0, e0, s1, e1 = _compare_swap_ordered(s0, e0, s1, e1)
    s2, e2, s3, e3 = _compare_swap_ordered(s2, e2, s3, e3)
    s0, e0, s2, e2 = _compare_swap_ordered(s0, e0, s2, e2)
    s1, e1, s3, e3 = _compare_swap_ordered(s1, e1, s3, e3)
    s1, e1, s2, e2 = _compare_swap_ordered(s1, e1, s2, e2)
    return s0, e0, s1, e1, s2, e2, s3, e3


@triton.jit
def _head4(cursor, s0, e0, s1, e1, s2, e2, s3, e3):
    invalid_expert = tl.full((128,), 0xFFFFFFFF, dtype=tl.uint32)
    score = tl.where(
        cursor == 0,
        s0,
        tl.where(
            cursor == 1,
            s1,
            tl.where(cursor == 2, s2, tl.where(cursor == 3, s3, -float("inf"))),
        ),
    )
    expert = tl.where(
        cursor == 0,
        e0,
        tl.where(
            cursor == 1,
            e1,
            tl.where(cursor == 2, e2, tl.where(cursor == 3, e3, invalid_expert)),
        ),
    ).to(tl.uint32)
    return score, expert


__all__ = [
    "_head4",
    "_ordered_score_key",
    "_precise_sqrt_softplus",
    "_sort4_best_first",
]
