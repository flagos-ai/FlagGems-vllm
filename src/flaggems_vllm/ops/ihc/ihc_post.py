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

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


# Reuse x and post across all HC channels of each hidden slice. Autotuning
# compares tile sizes and program order on the current device, without Graph.


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("triton_ihc_post"),
    key=["NUM_TOKENS", "HC_MULT", "HIDDEN_SIZE", "DTYPE_TAG", "LAYOUT_TAG"],
    warmup=5,
    rep=10,
    use_cuda_graph=False,
)
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
    NUM_TOKENS: tl.constexpr,
    DTYPE_TAG: tl.constexpr,
    LAYOUT_TAG: tl.constexpr,
    TOKEN_MAJOR: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    hidden_blocks = tl.cdiv(HIDDEN_SIZE, BLOCK_D)
    if TOKEN_MAJOR:
        token_idx = pid % NUM_TOKENS
        hidden_block_idx = pid // NUM_TOKENS
    else:
        hidden_block_idx = pid % hidden_blocks
        token_idx = pid // hidden_blocks
    EVEN_D: tl.constexpr = HIDDEN_SIZE % BLOCK_D == 0
    EVEN_HC: tl.constexpr = HC_MULT == HC_POW2

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
    logger.debug("GEMS IHC_POST")
    return _ihc_post(x, residual, post)


def triton_ihc_post(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    """Return post[:, :, None] * x[:, None, :] + residual with unfused FP32 arithmetic.

    FP16/BF16 inputs may be noncontiguous or alias each other. No input is
    mutated; the result always owns fresh storage. HC and D must be positive.
    """
    logger.debug("GEMS TRITON_IHC_POST")
    return _ihc_post(x, residual, post)


def _ihc_post(x, residual, post):
    if not all(isinstance(t, torch.Tensor) for t in (x, residual, post)):
        raise TypeError("x, residual and post must be tensors")
    if x.ndim != 2 or residual.ndim != 3 or post.ndim != 2:
        raise ValueError("expected x[N,D], residual[N,HC,D], post[N,HC]")
    n, d = x.shape
    hc = post.shape[1]
    if d <= 0 or hc <= 0:
        raise ValueError("D and HC must be positive")
    if post.shape != (n, hc) or residual.shape != (n, hc, d):
        raise ValueError("incompatible x, residual and post shapes")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("x must have dtype float16 or bfloat16")
    if residual.dtype != x.dtype or post.dtype != torch.float32:
        raise TypeError("residual must match x dtype and post must be float32")
    if x.device != residual.device or x.device != post.device:
        raise ValueError("all inputs must be on the same device")
    if x.device.type != runtime.device.name or x.device.type == "cpu":
        raise ValueError("inputs must be on the active accelerator device")
    layout_tag = sum(
        (not t.is_contiguous()) << i for i, t in enumerate((x, residual, post))
    )
    x = x.contiguous()
    residual = residual.contiguous()
    post = post.contiguous()
    num_tokens, hidden_size = x.shape
    hc_mult = post.shape[-1]

    if num_tokens == 0:
        return torch.empty((0, hc_mult, hidden_size), dtype=x.dtype, device=x.device)

    output = torch.empty(
        (num_tokens, hc_mult, hidden_size), dtype=x.dtype, device=x.device
    )
    hc_pow2 = triton.next_power_of_2(hc_mult)
    kernel = _ihc_post_kernel
    if runtime.device.vendor_name == "metax":
        from flaggems_vllm.runtime.backend._metax.ops.ihc_post import (
            _ihc_post_kernel as kernel,
        )
    elif runtime.device.vendor_name == "mthreads" and x.dtype == torch.float16:
        from flaggems_vllm.runtime.backend._mthreads.ops.ihc_post import (
            _ihc_post_kernel as kernel,
        )
    # Keep the native FP32 multiply and addition separate, including rounding.
    with runtime.torch_device_fn.device(x.device):
        if runtime.device.vendor_name in (
            "nvidia",
            "thead",
        ) and triton.__version__.startswith("3.6."):
            if runtime.device.vendor_name == "nvidia":
                from flaggems_vllm.runtime.backend._nvidia.ops.ihc_post import (
                    _launch_ihc_post,
                )
            else:
                from flaggems_vllm.runtime.backend._thead.ops.ihc_post import (
                    _launch_ihc_post,
                )

            _launch_ihc_post(
                x, residual, post, output, num_tokens, hc_mult, hidden_size, layout_tag
            )
            return output
        kernel[lambda meta: (num_tokens * triton.cdiv(hidden_size, meta["BLOCK_D"]),)](
            x,
            residual,
            post,
            output,
            HIDDEN_SIZE=hidden_size,
            HC_MULT=hc_mult,
            HC_POW2=hc_pow2,
            NUM_TOKENS=num_tokens,
            DTYPE_TAG=str(x.dtype),
            LAYOUT_TAG=layout_tag,
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
