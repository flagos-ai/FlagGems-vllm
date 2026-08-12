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

"""Cambricon element-wise addition kernel.

Simple pointwise addition operator for CI testing and baseline performance
validation on the Cambricon architecture.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_fwd_kernel(a_ptr, b_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)

    result = a + b
    tl.store(out_ptr + offs, result, mask=mask)


@triton.jit
def _add_bwd_kernel(dout_ptr, da_ptr, db_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    dout = tl.load(dout_ptr + offs, mask=mask, other=0.0)
    tl.store(da_ptr + offs, dout, mask=mask)
    tl.store(db_ptr + offs, dout, mask=mask)


def _select_block_size(numel):
    """Select BLOCK size tuned for Cambricon."""
    if numel >= 131072:
        return 2048
    elif numel >= 8192:
        return 1024
    return 512


class Add(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        assert A.shape == B.shape, "add: input shapes must match"
        ctx.save_for_backward(A, B)
        out = torch.empty_like(A)
        numel = A.numel()
        BLOCK = _select_block_size(numel)
        grid = (triton.cdiv(numel, BLOCK),)
        _add_fwd_kernel[grid](A, B, out, numel, BLOCK=BLOCK, num_warps=4)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        da = torch.empty_like(A)
        db = torch.empty_like(B)
        numel = A.numel()
        BLOCK = _select_block_size(numel)
        grid = (triton.cdiv(numel, BLOCK),)
        _add_bwd_kernel[grid](grad_output, da, db, numel, BLOCK=BLOCK, num_warps=4)
        return da, db


def add(A, B):
    return Add.apply(A, B)


def add_out(A, B, out):
    assert A.shape == B.shape == out.shape, "add: shapes must match"
    numel = A.numel()
    BLOCK = _select_block_size(numel)
    grid = (triton.cdiv(numel, BLOCK),)
    _add_fwd_kernel[grid](A, B, out, numel, BLOCK=BLOCK, num_warps=4)
    return out
