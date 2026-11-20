# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Grouped Gemma RMSNorm with runtime strides and epsilon on THead."""

from contextlib import nullcontext

import triton
import triton.language as tl

from flaggems_vllm import runtime


@triton.jit(do_not_specialize=["STRIDE_X", "STRIDE_W", "EPS"])
def _thead_grouped_gemma_rmsnorm_kernel(
    X,
    W,
    Y,
    STRIDE_X,
    STRIDE_W,
    EPS,
    GROUPS: tl.constexpr,
    GROUP_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CONTIGUOUS_ROWS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    group = pid % GROUPS
    y_base = pid * GROUP_DIM
    if CONTIGUOUS_ROWS:
        x_base = y_base
    else:
        row = pid // GROUPS
        x_base = row * STRIDE_X + group * GROUP_DIM
    # Both supported weight strides, 0 and GROUP_DIM, share this alignment.
    weight_alignment: tl.constexpr = min(16, GROUP_DIM & -GROUP_DIM)
    w_base = group * tl.multiple_of(STRIDE_W, weight_alignment)
    offsets = tl.arange(0, BLOCK_SIZE)

    if GROUP_DIM <= BLOCK_SIZE:
        mask = offsets < GROUP_DIM
        values = tl.load(X + x_base + offsets, mask, other=0.0).to(tl.float32)
        weight = tl.load(W + w_base + offsets, mask, other=0.0).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(values * values, axis=0) / GROUP_DIM + EPS)
        normalized = values * rrms
        tl.store(Y + y_base + offsets, tl.fma(normalized, weight, normalized), mask)
    else:
        sum_sq = tl.full((), 0.0, tl.float32)
        for start in range(0, GROUP_DIM, BLOCK_SIZE):
            columns = start + offsets
            values = tl.load(X + x_base + columns, columns < GROUP_DIM, other=0.0).to(
                tl.float32
            )
            sum_sq += tl.sum(values * values, axis=0)
        rrms = tl.rsqrt(sum_sq / GROUP_DIM + EPS)
        for start in range(0, GROUP_DIM, BLOCK_SIZE):
            columns = start + offsets
            mask = columns < GROUP_DIM
            values = tl.load(X + x_base + columns, mask, other=0.0).to(tl.float32)
            weight = tl.load(W + w_base + columns, mask, other=0.0).to(tl.float32)
            normalized = values * rrms
            tl.store(Y + y_base + columns, tl.fma(normalized, weight, normalized), mask)


def _launch_grouped_gemma_rmsnorm(x, weight, output, eps, groups):
    n, dim = x.shape
    group_dim = dim // groups
    row_stride = x.stride(0)
    # A single row has no row offset. Reuse the contiguous path even if the
    # unused row stride has padding or would require a 64-bit scalar.
    contiguous_rows = n == 1 or row_stride == dim
    weight_stride = 0 if weight.numel() == group_dim else group_dim
    block_size = min(triton.next_power_of_2(group_dim), 1024)
    device = runtime.torch_device_fn
    context = (
        nullcontext()
        if device.current_device() == x.device.index
        else device.device(x.device.index)
    )
    with context:
        _thead_grouped_gemma_rmsnorm_kernel.run(
            x,
            weight,
            output,
            0 if contiguous_rows else row_stride,
            weight_stride,
            eps,
            groups,
            group_dim,
            block_size,
            contiguous_rows,
            grid=(n * groups,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
