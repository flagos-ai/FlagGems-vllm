# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Single-launch kernels for dense BF16 down projections on Hopper."""

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from flaggems_vllm.ops.qwen4.hc_down_silu import (
    _hc_down_silu_epilogue,
    _hc_down_silu_gemv,
)


@triton.jit
def _batched_dot(
    X,
    W,
    O,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    RANK: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUPS: tl.constexpr,
):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 16 + tl.arange(0, 16)
    offsets = tl.arange(0, GROUPS)[:, None] * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    # Each batch computes a disjoint part of the dot product. Its reduction
    # stays in the CTA, so there is no global workspace or second dispatch.
    acc = tl.full((GROUPS, 16, 16), 0, tl.float32)
    for block in range(tl.cdiv(K, GROUPS * BLOCK_K)):
        k = block * (GROUPS * BLOCK_K) + offsets
        x = tl.load(
            X + rows[None, :, None] * K + k[:, None, :],
            (rows[None, :, None] < M) & (k[:, None, :] < K),
            other=0,
        )
        w = tl.load(
            W + cols[None, :, None] * K + k[:, None, :],
            (cols[None, :, None] < RANK + HC) & (k[:, None, :] < K),
            other=0,
        )
        acc = tl.dot(x, tl.permute(w, (0, 2, 1)), acc)
    value = _hc_down_silu_epilogue(tl.sum(acc, 0), cols[None, :], RANK, HC)
    tl.store(
        O + rows[:, None] * N + cols[None, :],
        value,
        (rows[:, None] < M) & (cols[None, :] < RANK + HC),
    )


def make_dense_launcher(device, m, k, n, rank, hc):
    # The caller guarantees aligned BF16 pointers and dense row strides.
    # Only the device, shape and scalars specialize this kernel. The returned
    # Triton launcher resolves the current stream anew on every invocation.
    if k <= 4096 or m <= 7:
        kernel = _hc_down_silu_gemv.jit_function
        shape = (m, k, n, rank, hc, k, k, 1)
        params = (shape, 1, 1024)
        grid = (m, rank + hc, 1)
        warps, stages = 4, 1
    else:
        kernel = _batched_dot
        params = (m, k, n, rank, hc, 256, 8)
        grid = (triton.cdiv(m, 16), triton.cdiv(rank + hc, 16), 1)
        warps, stages = 8, 2
    bf16 = torch.bfloat16
    compiled = kernel.warmup(
        bf16, bf16, bf16, *params, grid=grid, num_warps=warps, num_stages=stages
    )
    run = compiled[grid]
    get_stream = driver.active.get_current_stream

    def launch(x, weight, output):
        run(x, weight, output, *params, stream=get_stream(device))

    return launch
