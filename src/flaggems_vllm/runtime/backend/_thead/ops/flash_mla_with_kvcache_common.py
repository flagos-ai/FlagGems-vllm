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

"""What the two FP8 sparse decode layouts on T-Head PPU have in common.

Everything that reads the cache -- the schedule, the tile loads, the
dequantization, the attention loop -- is layout-specific and lives in
flash_mla_with_kvcache_model1.py and flash_mla_with_kvcache_v32.py. What is
left is the same for both: the attention shape, the device constants above the
cache layout, and the kernel that merges the partitions a request was split
over.
"""
import triton
import triton.language as tl

HEADS = tl.constexpr(64)
HEAD_DIM = tl.constexpr(512)
# KV tokens per tile. A 128-token tile halves the per-tile fixed costs but only
# fits with Q in registers instead of a shared tile, and measures slower both
# ways; 32 needs twice the tiles for the same work and measures twice as slow.
BLOCK_N = tl.constexpr(64)
LN2 = tl.constexpr(0.6931471805599453)
LOG2E = 1.4426950408889634
# Partitions (one program each). On the 810E this dominates every other tuning
# knob: a sweep of 8..128 is good at 20, 40 and 80 and markedly worse at 16,
# 24, 32 and 64, with 20 the best at every batch; below 20 the time scales as
# 1/PARTS. A high-water counter added to the attention kernel for the
# measurement showed the device keeping exactly 20 of its programs resident,
# and the same 20 whatever PARTS, the warp count and the shared-memory
# footprint were. Re-measure it if a schedule, a tile size or that footprint
# changes.
PARTS = 20
# combine: heads per program and warps; one head per program parallelizes best
# when a few long requests are spread over all partitions.
COMBINE_HEADS = 1
COMBINE_WARPS = 4


@triton.jit
def sink_factor(sink, h, lse):
    """exp(lse) / (exp(lse) + exp(sink)), including sink = +-inf and lse = +inf."""
    sink_value = tl.load(sink + h)
    return tl.where(
        sink_value == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink_value - lse))
    )


@triton.jit
def combine(
    partial,
    partial_lse,
    split_prefix,
    sink,
    out,
    lse,
    HAS_SINK: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Merge a request's splits (GROUP heads per program), then apply the sink."""
    b = tl.program_id(0)
    first = tl.load(split_prefix + b)
    end = tl.load(split_prefix + b + 1)
    if end - first > 1:
        h = tl.program_id(1) * GROUP + tl.arange(0, GROUP)
        rows = h[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        maximum = tl.full((GROUP,), float("-inf"), tl.float32)
        denominator = tl.zeros((GROUP,), tl.float32)
        result = tl.zeros((GROUP, HEAD_DIM), tl.float32)
        for split in range(first, end):
            value = tl.load(partial_lse + split * HEADS + h)
            new_maximum = tl.maximum(maximum, value)
            safe = tl.where(new_maximum == float("-inf"), 0.0, new_maximum)
            alpha = tl.exp2(maximum - safe)
            weight = tl.exp2(value - safe)
            result = (
                result * alpha[:, None]
                + tl.load(partial + split * HEADS * HEAD_DIM + rows) * weight[:, None]
            )
            denominator = denominator * alpha + weight
            maximum = new_maximum
        valid = denominator > 0
        natural = tl.where(valid, (tl.log2(denominator) + maximum) * LN2, float("inf"))
        factor = tl.where(valid, 1.0 / denominator, 0.0)
        if HAS_SINK:
            factor *= sink_factor(sink, h, natural)
        tl.store(
            out + b * HEADS * HEAD_DIM + rows,
            (result * factor[:, None]).to(tl.bfloat16),
        )
        tl.store(lse + b * HEADS + h, natural)
