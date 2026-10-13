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


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


@triton.jit
def _add_rms_norm_kernel_single(
    X1_ptr,
    X2_ptr,
    W_ptr,
    OUT_ptr,
    M,
    N,
    eps,
    stride_row,
    BLOCK_N: tl.constexpr,
):
    """Single-pass kernel: BLOCK_N >= N; caches s = x1 + x2 in registers.
    One program per row.
    """
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x1 = tl.load(X1_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X2_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    s = x1 + x2

    ss = tl.sum(s * s, axis=0)
    mean = ss / N
    rms = tl.rsqrt(mean + eps)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = s * rms * w

    tl.store(OUT_ptr + row * stride_row + offs, y, mask=mask)


@triton.jit
def _add_rms_norm_kernel_two_pass(
    X1_ptr,
    X2_ptr,
    W_ptr,
    OUT_ptr,
    M,
    N,
    eps,
    stride_row,
    BLOCK_N: tl.constexpr,
):
    """Two-pass kernel: N > BLOCK_N. Streams tiles.
    One program per row.
    """
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return

    row_off = row * stride_row
    n_tiles = tl.cdiv(N, BLOCK_N)

    # Pass 1: accumulate sum of squares
    acc = tl.zeros((), dtype=tl.float32)
    for tile_idx in range(0, n_tiles):
        offs = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        x1 = tl.load(X1_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(X2_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        s = x1 + x2
        acc += tl.sum(s * s, axis=0)

    mean = acc / N
    rms = tl.rsqrt(mean + eps)

    # Pass 2: recompute s, scale, store
    for tile_idx in range(0, n_tiles):
        offs = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        x1 = tl.load(X1_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(X2_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s = x1 + x2
        y = s * rms * w
        tl.store(OUT_ptr + row_off + offs, y, mask=mask)


def add_rms_norm(x1, x2, normalized_shape, weight, eps=1e-05):
    ndim_norm = len(normalized_shape)

    # Flatten to [M, N]
    N = 1
    for d in normalized_shape:
        N *= int(d)
    M = 1
    for d in x1.shape[: x1.dim() - ndim_norm]:
        M *= int(d)

    x1_flat = x1.reshape(M, N)
    x2_flat = x2.reshape(M, N)
    w_flat = weight.reshape(N)

    out = torch.empty_like(x1)
    out_flat = out.reshape(M, N)

    stride_row = N  # contiguous

    # Choose kernel variant
    # For "small/medium" N, cache the row in registers (single-pass BLOCK_N>=N).
    # For very large N, do two-pass with a larger tile.
    if N <= 4096:
        BLOCK_N = _next_pow2(N)
        if BLOCK_N <= 256:
            num_warps = 4
        elif BLOCK_N <= 1024:
            num_warps = 4
        elif BLOCK_N <= 2048:
            num_warps = 8
        else:
            num_warps = 16
        grid = (M,)
        _add_rms_norm_kernel_single[grid](
            x1_flat,
            x2_flat,
            w_flat,
            out_flat,
            M,
            N,
            float(eps),
            stride_row,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
    else:
        BLOCK_N = 4096
        num_warps = 16
        grid = (M,)
        _add_rms_norm_kernel_two_pass[grid](
            x1_flat,
            x2_flat,
            w_flat,
            out_flat,
            M,
            N,
            float(eps),
            stride_row,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )

    return out
