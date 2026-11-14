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

"""Triton implementation of iHC (independent Hyper-Connections) post operator.

GCU (enflame) adaptation of the device-agnostic port of the vLLM HY V4
implementation (vllm/models/hy_v4/nvidia/triton_ihc.py). Computes the gated
residual scatter (no cross-stream mixing, unlike mHC):

    out[n, i, :] = post[n, i] * x[n, :] + residual[n, i, :]

Adaptations forced by the GCU backend (2026-11-07 toolchain):

* Fully 1-D affine accesses. The toolchain failed to compile any 2-D tile
  load/store (`memref_ext.slice_(pad_)start` operand-segment mismatch, see
  compiler_issues/vllm_moe_sum.md; the family was declared resolved
  2026-11-10), so the [HC, BLOCK_D] residual tile and the [HC, 1] gate tile
  become a `tl.static_range` loop of 1-D rows plus scalar gate loads. A
  [HC, BLOCK_D] 2-D row-unit tile was re-tried after the P01a resolution
  (2026-11-11) and rejected: it is ~2x slower than the 1-D form at every
  BLOCK_D and silently produces wrong values at BLOCK_D=4096
  (compiler_issues/vllm_ihc.md), so the 1-D form stays.
* Capped persistent grid: the (token, hidden-block) tile space is strided by
  a fixed 48 programs (one per GCU400 SP), mirroring `_enflame/ops/moe_sum.py`.
  A program per tile (98304 at N=8192) is dominated by the ~52ns/CTA dispatch
  floor, and the original layout also put num_tokens on grid.y, whose
  hardware limit is 255.
* `(gate * x + 0.0) + residual`: the GCU backend fuses mul+add into a
  single-rounding FMA at the MLIR level (`enable_fp_fusion` is not honored),
  which breaks this op's zero-tolerance bitwise contract vs the torch
  reference. The +0.0 add breaks the fusion so the fp32 product rounds
  exactly like the reference; a -0.0 product becomes +0.0, which still
  compares equal at atol=0.
"""

import torch
import triton
import triton.language as tl

# Persistent-grid program count: one program per GCU400 SP, the measured
# sweet spot for streaming kernels on this part (see _enflame/ops/moe_sum.py).
_GRID_TARGET = 48
# Hidden-slice width: 2026-11-11 BLOCK_D x num_warps sweep -- 4096/nw=8 wins
# at large N (N=8192 1994 -> 896us, -55%; N=1024 254 -> 114us), i.e. fewer,
# wider per-program slices amortize the per-tile fixed cost (4 scalar gate
# loads + masked ops) despite the 1/3-lane tail waste at hidden 6144. A
# [HC, BLOCK_D] 2-D row-unit tile variant was tried and rejected: 2x slower
# everywhere and silently wrong at BLOCK_D=4096 (compiler_issues). 6144-wide
# single tiles do not compile (BLOCK must be a power of two). Per-CTA slice
# stays ~48KB at 4096, under the compiler limit.
_POST_BLOCK_D = 4096
_NUM_WARPS = 8


def _post_block_d(hidden_size: int) -> int:
    # 4096 for wide hidden (the sweep sweet spot), next_pow2 below that, and
    # never a pathological width-1 tile.
    return min(_POST_BLOCK_D, max(2, triton.next_power_of_2(hidden_size)))


@triton.jit
def _ihc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    output_ptr,
    num_tokens,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    NUM_HB: tl.constexpr = (HIDDEN_SIZE + BLOCK_D - 1) // BLOCK_D
    total = num_tokens * NUM_HB
    for tile in range(tl.program_id(0), total, tl.num_programs(0)):
        hb = tile % NUM_HB
        token = tile // NUM_HB
        hidden_offsets = hb * BLOCK_D + tl.arange(0, BLOCK_D)
        hidden_mask = hidden_offsets < HIDDEN_SIZE
        x = tl.load(
            x_ptr + token * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        for channel in tl.static_range(HC_MULT):
            gate = tl.load(post_ptr + token * HC_MULT + channel)
            residual = tl.load(
                residual_ptr
                + (token * HC_MULT + channel) * HIDDEN_SIZE
                + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            # +0.0 keeps the fp32 mul and add unfused: the GCU backend
            # otherwise contracts this into a single-rounding FMA and the
            # zero-tolerance test fails with 1-ULP output diffs.
            output = (gate * x + 0.0) + residual
            tl.store(
                output_ptr + (token * HC_MULT + channel) * HIDDEN_SIZE + hidden_offsets,
                output.to(output_ptr.dtype.element_ty),
                mask=hidden_mask,
            )


def ihc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """iHC post: scatter a sub-block output back over the hc residual channels.

    Args:
        x: [num_tokens, hidden_size], float16/bfloat16 -- sub-block output.
        residual: [num_tokens, hc_mult, hidden_size], same dtype as x.
        post: [num_tokens, hc_mult], float32 -- per-channel gate from ihc_pre.

    Returns:
        out: [num_tokens, hc_mult, hidden_size], x.dtype.
    """
    assert x.dim() == 2, "x must be [num_tokens, hidden_size]"
    assert x.is_cuda and residual.is_cuda and post.is_cuda
    assert post.dtype == torch.float32

    x = x.contiguous()
    residual = residual.contiguous()
    post = post.contiguous()
    num_tokens, hidden_size = x.shape
    hc_mult = post.shape[-1]
    assert residual.shape == (num_tokens, hc_mult, hidden_size)
    assert post.shape == (num_tokens, hc_mult)

    if num_tokens == 0:
        return torch.empty((0, hc_mult, hidden_size), dtype=x.dtype, device=x.device)

    output = torch.empty(
        (num_tokens, hc_mult, hidden_size), dtype=x.dtype, device=x.device
    )
    block_d = _post_block_d(hidden_size)
    total = num_tokens * triton.cdiv(hidden_size, block_d)
    _ihc_post_kernel[(min(total, _GRID_TARGET),)](
        x,
        residual,
        post,
        output,
        num_tokens,
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        BLOCK_D=block_d,
        num_warps=_NUM_WARPS,
    )
    return output


def ihc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference (test/benchmark only; never imported by production)."""
    return (post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + residual.float()).to(
        x.dtype
    )
