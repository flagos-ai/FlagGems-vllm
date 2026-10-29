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

Baseline port of the vLLM HY V4 implementation
(vllm/models/hy_v4/nvidia/triton_ihc.py), adapted to FlagGems-vllm
conventions.

Computes the gated residual scatter (no cross-stream mixing, unlike mHC):
    out[n, i, :] = post[n, i] * x[n, :] + residual[n, i, :]

where x is the sub-block output [N, d], residual is [N, hc, d] and post is the
per-channel gate [N, hc] produced by ihc_pre. Pure elementwise / streaming:
bandwidth-bound, no reduction.
"""

import torch
import triton
import triton.language as tl

# All hc channels of one BLOCK_D hidden slice are handled as a single
# [HC_POW2, BLOCK_D] tile, so the hc residual loads and output stores issue as
# one 2D access each instead of a static_range of per-channel accesses. The op
# is pure streaming (18*N*D bytes at hc=4), so at small N it is latency-bound
# and the wider access window is what hides memory latency; at large N it still
# wins slightly by sustaining a better access pattern.
#
# BLOCK_D=512 / num_warps=4 measured best at every token count on H20
# (D=6144, hc=4, bf16): 1.30x at N=1, 1.26x at N=8, 1.16x at N=64, 1.11x at
# N=256, 1.06x at N=1024, 1.05x at N=4096, 1.04x at N=8192 versus the previous
# BLOCK_D=1024 / num_warps=4 per-channel loop. A narrower tile kept winning even
# at N=8192 (98304 blocks), so no per-N config split is needed; num_warps was
# flat within +-1% across 1/2/4/8. Hence the fixed config (no autotune), as in
# ihc_pre.
_BLOCK_D = 512
_NUM_WARPS = 4


@triton.jit
def _ihc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
    EVEN_HC: tl.constexpr,
):
    # D-major grid: consecutive programs walk contiguous hidden offsets inside a
    # token rather than striding hc*D apart across tokens.
    hidden_block_idx = tl.program_id(0)
    token_idx = tl.program_id(1).to(tl.int64)

    hc_idx = tl.arange(0, HC_POW2)[:, None]
    hidden_offsets = (hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D))[None, :]
    # post[n, i] broadcast over the hidden axis: [HC_POW2, 1] * [1, BLOCK_D].
    gate_ptr = post_ptr + token_idx * HC_MULT + hc_idx
    if EVEN_HC:
        gate = tl.load(gate_ptr)
    else:
        gate = tl.load(gate_ptr, mask=hc_idx < HC_MULT, other=0.0)

    offsets = token_idx * HC_MULT * HIDDEN_SIZE + hc_idx * HIDDEN_SIZE + hidden_offsets
    if EVEN_D and EVEN_HC:
        x = tl.load(x_ptr + token_idx * HIDDEN_SIZE + hidden_offsets).to(tl.float32)
        residual = tl.load(residual_ptr + offsets).to(tl.float32)
        output = gate * x + residual
        tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty))
    else:
        hidden_mask = hidden_offsets < HIDDEN_SIZE
        mask = hidden_mask if EVEN_HC else (hc_idx < HC_MULT) & hidden_mask
        x = tl.load(
            x_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        output = gate * x + residual
        tl.store(
            output_ptr + offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=mask,
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
    hc_pow2 = triton.next_power_of_2(hc_mult)
    # enable_fp_fusion=False keeps the fp32 mul and add unfused, so the result
    # is bitwise identical to the reference (the accuracy test is zero-tolerance).
    _ihc_post_kernel[(triton.cdiv(hidden_size, _BLOCK_D), num_tokens)](
        x,
        residual,
        post,
        output,
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        BLOCK_D=_BLOCK_D,
        EVEN_D=(hidden_size % _BLOCK_D == 0),
        EVEN_HC=(hc_pow2 == hc_mult),
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,
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
