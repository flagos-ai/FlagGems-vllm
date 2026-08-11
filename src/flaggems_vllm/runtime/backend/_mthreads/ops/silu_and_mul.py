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

"""Moore Threads MUSA-optimized SiLU-and-Mul fused kernel.

Fuses the SiLU activation with element-wise multiplication into a single
kernel pass, reducing memory traffic for the gate projection in LLM MLP
layers. Optimized block sizes for the S5000 architecture.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _silu_and_mul_fwd_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)

    x_fp32 = x.to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-x_fp32))
    silu = x_fp32 * sigmoid
    result = silu * y.to(tl.float32)

    tl.store(out_ptr + offs, result.to(x.dtype), mask=mask)


@triton.jit
def _silu_and_mul_bwd_kernel(
    x_ptr,
    y_ptr,
    dout_ptr,
    dx_ptr,
    dy_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    dout = tl.load(dout_ptr + offs, mask=mask, other=0.0)

    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    dout_fp32 = dout.to(tl.float32)

    sigmoid = 1.0 / (1.0 + tl.exp(-x_fp32))
    silu = x_fp32 * sigmoid
    # d(silu)/dx = sigmoid * (1 + x * (1 - sigmoid))
    d_silu_dx = sigmoid * (1.0 + x_fp32 * (1.0 - sigmoid))

    dx = dout_fp32 * y_fp32 * d_silu_dx
    dy = dout_fp32 * silu

    tl.store(dx_ptr + offs, dx.to(x.dtype), mask=mask)
    tl.store(dy_ptr + offs, dy.to(x.dtype), mask=mask)


def _select_block_size(numel):
    """Select BLOCK size tuned for Moore Threads S5000 warp width."""
    if numel >= 131072:
        return 2048
    elif numel >= 8192:
        return 1024
    return 512


class SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        logger.debug("GEMS MTHREADS SILU AND MUL FORWARD")

        assert A.shape == B.shape, "silu_and_mul: input shapes must match"
        out = torch.empty_like(A)
        numel = A.numel()
        BLOCK = _select_block_size(numel)
        grid = (triton.cdiv(numel, BLOCK),)

        _silu_and_mul_fwd_kernel[grid](A, B, out, numel, BLOCK=BLOCK, num_warps=4)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        logger.debug("GEMS MTHREADS SILU AND MUL BACKWARD")

        dx = torch.empty_like(A)
        dy = torch.empty_like(B)
        numel = A.numel()
        BLOCK = _select_block_size(numel)
        grid = (triton.cdiv(numel, BLOCK),)

        _silu_and_mul_bwd_kernel[grid](
            A, B, grad_output, dx, dy, numel, BLOCK=BLOCK, num_warps=4
        )
        return dx, dy


def silu_and_mul(A, B):
    return SiluAndMul.apply(A, B)


def silu_and_mul_out(A, B, out):
    logger.debug("GEMS MTHREADS SILU AND MUL OUT")
    assert A.shape == B.shape, "silu_and_mul: input shapes must match"
    numel = A.numel()
    BLOCK = _select_block_size(numel)
    grid = (triton.cdiv(numel, BLOCK),)

    _silu_and_mul_fwd_kernel[grid](A, B, out, numel, BLOCK=BLOCK, num_warps=4)
    return out
