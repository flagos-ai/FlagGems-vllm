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

_BLOCK_D = 1024


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
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    post = tl.load(
        post_ptr + token_idx * HC_MULT + hc_idx,
        mask=hc_mask,
        other=0.0,
    )

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x = tl.load(
        x_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    residual_row = residual_ptr + token_idx * HC_MULT * HIDDEN_SIZE
    output_row = output_ptr + token_idx * HC_MULT * HIDDEN_SIZE
    for channel_idx in tl.static_range(HC_MULT):
        residual = tl.load(
            residual_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, post, 0.0), axis=0)
        output = channel_gate * x + residual
        tl.store(
            output_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
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
    _ihc_post_kernel[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
        x,
        residual,
        post,
        output,
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=triton.next_power_of_2(hc_mult),
        BLOCK_D=_BLOCK_D,
        num_warps=4,
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
