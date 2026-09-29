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
    N,
    inv_N,
    eps,
    BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    row_off = row * N + cols

    x1_off = x1_ptr + row_off
    x2_off = x2_ptr + row_off
    out_off = out_ptr + row_off
    w_off = w_ptr + cols

    tl.max_contiguous(tl.multiple_of(x1_off, 16), BLOCK_N)
    tl.max_contiguous(tl.multiple_of(x2_off, 16), BLOCK_N)
    tl.max_contiguous(tl.multiple_of(out_off, 16), BLOCK_N)
    tl.max_contiguous(tl.multiple_of(w_off, 16), BLOCK_N)

    if HAS_MASK:
        mask = cols < N
        a1 = tl.load(x1_off, mask=mask, other=0.0).to(tl.float32)
        a2 = tl.load(x2_off, mask=mask, other=0.0).to(tl.float32)
        xf = a1 + a2

        sumsq = tl.sum(xf * xf, axis=0)
        mean = sumsq * inv_N
        rms = tl.rsqrt(mean + eps)

        w = tl.load(w_off, mask=mask, other=0.0).to(tl.float32)
        y = (xf * rms) * w

        tl.store(out_off, y, mask=mask)
    else:
        a1 = tl.load(x1_off).to(tl.float32)
        a2 = tl.load(x2_off).to(tl.float32)
        xf = a1 + a2

        sumsq = tl.sum(xf * xf, axis=0)
        mean = sumsq * inv_N
        rms = tl.rsqrt(mean + eps)

        w = tl.load(w_off).to(tl.float32)
        y = (xf * rms) * w

        tl.store(out_off, y)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def add_rms_norm(x1, x2, normalized_shape, weight, eps=1e-05):
    N = 1
    for s in normalized_shape:
        N *= int(s)
    M = x1.numel() // N

    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    x1v = x1.view(M, N)
    x2v = x2.view(M, N)
    wv = weight.view(N)

    out = torch.empty_like(x1v)

    BLOCK_N = _next_pow2(N)
    if BLOCK_N < 128:
        BLOCK_N = 128

    if BLOCK_N <= 512:
        num_warps = 4
    elif BLOCK_N <= 2048:
        num_warps = 8
    else:
        num_warps = 16

    has_mask = BLOCK_N != N

    grid = (M,)
    _add_rms_norm_kernel[grid](
        x1v,
        x2v,
        wv,
        out,
        N,
        1.0 / float(N),
        float(eps),
        BLOCK_N=BLOCK_N,
        HAS_MASK=has_mask,
        num_warps=num_warps,
    )

    return out.view(x1.shape)
