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
def _swiglu_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    W,
    num_programs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    col = tl.arange(0, BLOCK_N)
    col_mask = col < N

    num_row_tiles = tl.cdiv(M, BLOCK_M)
    for t in range(pid, num_row_tiles, num_programs):
        rows = t * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        mask = row_mask[:, None] & col_mask[None, :]

        a_ptrs = x_ptr + rows[:, None] * W + col[None, :]
        b_ptrs = a_ptrs + N
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        b = tl.load(b_ptrs, mask=mask, other=0.0)

        a_f32 = a.to(tl.float32)
        silu = a_f32 * tl.sigmoid(a_f32)
        silu_cast = silu.to(a.dtype)

        out = silu_cast * b

        out_ptrs = out_ptr + rows[:, None] * N + col[None, :]
        tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask)


def swiglu(input_tensor, quantizer=None):
    del quantizer
    W = input_tensor.shape[-1]
    N = W // 2
    x2d = input_tensor.reshape(-1, W)
    M = x2d.shape[0]

    out = torch.empty(
        input_tensor.shape[:-1] + (N,),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    out2d = out.reshape(-1, N)

    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_M = 16
    num_row_tiles = triton.cdiv(M, BLOCK_M)
    num_programs = min(num_row_tiles, 16)
    grid = (num_programs,)
    _swiglu_kernel[grid](
        x2d, out2d, M, N, W, num_programs, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=2
    )
    return out
