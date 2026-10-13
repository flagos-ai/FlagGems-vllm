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

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel_1d(
    x_ptr,
    out_ptr,
    total_elems,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems

    # For each output element i, decompose into (row, col) with row=i//N, col=i%N
    row = offs // N
    col = offs - row * N
    two_n = 2 * N
    a_offs = row * two_n + col
    b_offs = a_offs + N

    a = tl.load(x_ptr + a_offs, mask=mask, other=0.0)
    b = tl.load(x_ptr + b_offs, mask=mask, other=0.0)

    a_fp32 = a.to(tl.float32)
    b_fp32 = b.to(tl.float32)
    sig = tl.sigmoid(a_fp32)
    out_fp32 = a_fp32 * sig * b_fp32
    out = out_fp32.to(a.dtype)

    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _swiglu_kernel_2d(
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    two_n = 2 * N
    row_base_a = pid_m * two_n
    a_offs = row_base_a + offs_n
    b_offs = row_base_a + N + offs_n
    out_offs = pid_m * N + offs_n

    a = tl.load(x_ptr + a_offs, mask=mask, other=0.0)
    b = tl.load(x_ptr + b_offs, mask=mask, other=0.0)

    a_fp32 = a.to(tl.float32)
    sig = tl.sigmoid(a_fp32)
    silu_fp32 = a_fp32 * sig
    silu_t = silu_fp32.to(a.dtype)
    out = silu_t * b

    tl.store(out_ptr + out_offs, out, mask=mask)


def swiglu(input_tensor, quantizer=None):
    del quantizer
    x = input_tensor
    if not x.is_contiguous():
        x = x.contiguous()

    last_dim = x.shape[-1]
    assert last_dim % 2 == 0, "swiglu requires last dim divisible by 2"
    N = last_dim // 2
    M = x.numel() // last_dim

    out_shape = x.shape[:-1] + (N,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    if M == 0 or N == 0:
        return out

    total = M * N
    # Choose BLOCK based on total size to maximize SM occupancy on BI-V150 (16 SMs).
    if total <= 4096:
        BLOCK = 1024
        num_warps = 4
    elif total <= 65536:
        BLOCK = 1024
        num_warps = 4
    else:
        BLOCK = 2048
        num_warps = 8

    grid = (triton.cdiv(total, BLOCK),)
    _swiglu_kernel_1d[grid](
        x,
        out,
        total,
        N,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
