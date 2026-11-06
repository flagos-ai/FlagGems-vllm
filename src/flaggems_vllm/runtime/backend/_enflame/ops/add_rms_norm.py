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
def _add_rms_norm_kernel(
    x1_ptr,
    x2_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    ptrs = offs_m[:, None] * N + offs_n[None, :]
    x1 = tl.load(x1_ptr + ptrs, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + ptrs, mask=mask, other=0.0)
    # Add in the input dtype to mirror the reference rounding boundary,
    # then convert to fp32.
    s = (x1 + x2).to(tl.float32)

    sumsq = tl.sum(s * s, axis=1)  # [BLOCK_M]
    ms = sumsq / N
    rms = tl.rsqrt(ms + eps)  # [BLOCK_M]

    w = tl.load(w_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)  # [BLOCK_N]
    y = s * rms[:, None] * w[None, :]

    tl.store(out_ptr + ptrs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _add_rms_norm_kernel_nomask(
    x1_ptr,
    x2_ptr,
    w_ptr,
    out_ptr,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Fast path: N == BLOCK_N and M % BLOCK_M == 0, so no masking is required.
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    ptrs = offs_m[:, None] * N + offs_n[None, :]

    x1 = tl.load(x1_ptr + ptrs, cache_modifier=".cg", eviction_policy="evict_first")
    x2 = tl.load(x2_ptr + ptrs, cache_modifier=".cg", eviction_policy="evict_first")
    s = (x1 + x2).to(tl.float32)

    sumsq = tl.sum(s * s, axis=1)
    ms = sumsq / N
    rms = tl.rsqrt(ms + eps)

    w = tl.load(w_ptr + offs_n).to(tl.float32)
    y = s * rms[:, None] * w[None, :]

    tl.store(out_ptr + ptrs, y.to(out_ptr.dtype.element_ty))


def add_rms_norm(x1, x2, normalized_shape, weight, eps=1e-05):
    N = 1
    for d in normalized_shape:
        N *= int(d)
    M = x1.numel() // N

    out = torch.empty_like(x1)

    x1c = x1.contiguous()
    x2c = x2.contiguous()
    wc = weight.contiguous()

    BLOCK_N = triton.next_power_of_2(N)
    # Rows per program: the shared per-workload optimum on this Enflame target
    # is 8 rows/program at num_warps=1 (independent of N).
    BLOCK_M = min(8, M)
    grid = (triton.cdiv(M, BLOCK_M),)
    if BLOCK_N == N and M % BLOCK_M == 0:
        _add_rms_norm_kernel_nomask[grid](
            x1c,
            x2c,
            wc,
            out,
            N,
            float(eps),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=1,
        )
    else:
        _add_rms_norm_kernel[grid](
            x1c,
            x2c,
            wc,
            out,
            M,
            N,
            float(eps),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=1,
        )
    return out
