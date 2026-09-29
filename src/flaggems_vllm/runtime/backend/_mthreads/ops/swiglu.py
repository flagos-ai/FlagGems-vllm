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

"""Triton SwiGLU kernel for MTT S5000 (backend: musa).

Elementwise operator:
    a, b = X.chunk(2, dim=-1)
    silu_a = silu(a.float()).to(X.dtype)
    out = silu_a * b

Kernel design (baseline):
- Flatten leading dims to M = numel // last_dim; halve last dim to N.
- 2D grid (M, cdiv(N, BLOCK_N)) with each program tile handling one row block.
- Compute silu in fp32 with tl.sigmoid, cast to input dtype, multiply by b in input dtype
  (matches reference cast boundary), store as input dtype.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    X_ptr,  # *T, [M, 2N]
    Y_ptr,  # *T, [M, N]
    M,
    N,
    stride_xm,  # row stride of input (elements)
    stride_ym,  # row stride of output (elements)
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col < N

    a_off = pid_m * stride_xm + col
    b_off = pid_m * stride_xm + N + col
    y_off = pid_m * stride_ym + col

    a = tl.load(X_ptr + a_off, mask=mask, other=0.0)
    b = tl.load(X_ptr + b_off, mask=mask, other=0.0)

    a_f32 = a.to(tl.float32)
    silu_f32 = a_f32 * tl.sigmoid(a_f32)
    silu = silu_f32.to(a.dtype)
    y = silu * b

    tl.store(Y_ptr + y_off, y, mask=mask)


def swiglu(input_tensor, quantizer=None):
    del quantizer
    x = input_tensor
    assert x.shape[-1] % 2 == 0, "last dim must be even"
    last = x.shape[-1]
    N = last // 2
    # Flatten leading dims into a single row axis.
    x2 = x.reshape(-1, last)
    M = x2.shape[0]

    out2 = torch.empty((M, N), dtype=x.dtype, device=x.device)

    stride_xm = x2.stride(0)
    stride_ym = out2.stride(0)

    # Shape-conditional tile selection informed by prior evidence:
    #  - small N (correctness shapes): use next_pow2(N), num_warps=4
    #  - N in [2048, 3072]: BLOCK_N=256, num_warps=4   -> 8 blocks/row
    #  - N >= 4096: BLOCK_N=1024, num_warps=8         -> ~4 blocks/row, more latency-hiding warps
    if N < 2048:
        BLOCK_N = max(32, triton.next_power_of_2(N))
        num_warps = 4
        num_stages = 1
    elif N < 4096:
        BLOCK_N = 256
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_N = 1024
        num_warps = 8
        num_stages = 2

    grid = (M, triton.cdiv(N, BLOCK_N))
    _swiglu_kernel[grid](
        x2,
        out2,
        M,
        N,
        stride_xm,
        stride_ym,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out2.reshape(*x.shape[:-1], N)
