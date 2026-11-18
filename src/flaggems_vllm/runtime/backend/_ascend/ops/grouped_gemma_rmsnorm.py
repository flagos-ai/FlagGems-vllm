# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded RMSNorm tiles with runtime metadata to limit cold specializations."""

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry


@libentry()
@triton.jit(
    do_not_specialize=[
        "GROUPS",
        "GROUP_SHIFT",
        "STRIDE_X",
        "STRIDE_W",
        "EPS",
    ]
)
def _grouped_gemma_rmsnorm_kernel(
    X,
    W,
    Y,
    GROUPS,
    GROUP_DIM: tl.constexpr,
    GROUP_SHIFT,
    STRIDE_X,
    STRIDE_W,
    EPS,
    BLOCK_SIZE: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    CONTIGUOUS_ROWS: tl.constexpr,
    POWER_OF_TWO_GROUPS: tl.constexpr,
    WEIGHT_ALIGNMENT: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if POWER_OF_TWO_GROUPS:
        row = pid >> GROUP_SHIFT
        group = pid & (GROUPS - 1)
    else:
        row = pid // GROUPS
        group = pid - row * GROUPS
    # Use 64-bit arithmetic for large tensor storage offsets.
    if CONTIGUOUS_ROWS:
        x_base = pid.to(tl.int64) * GROUP_DIM
    else:
        x_base = row.to(tl.int64) * STRIDE_X + group.to(tl.int64) * GROUP_DIM
    # The validated contiguous weight is either shared (stride 0) or has one
    # full group per stride. Preserve that alignment without specializing it.
    STRIDE_W = tl.multiple_of(STRIDE_W, WEIGHT_ALIGNMENT)
    w_base = group.to(tl.int64) * STRIDE_W
    y_base = pid.to(tl.int64) * GROUP_DIM
    offsets = tl.arange(0, BLOCK_SIZE)
    if SINGLE_PASS:
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
    # Specialize the reduction and coarse scheduling paths, not batch size,
    # exact strides, sharing or eps. A runtime shift handles power-of-two group
    # counts without either integer division or one binary per group count.
    # A bounded 4096-element tile avoids extra passes for medium-width groups
    # while retaining two passes for wider groups.
    group_dim = x.shape[1] // groups
    block_size = min(triton.next_power_of_2(group_dim), 4096)
    power_of_two_groups = groups & (groups - 1) == 0
    args = (
        x,
        weight,
        output,
        groups,
        group_dim,
        groups.bit_length() - 1 if power_of_two_groups else 0,
        x.stride(0),
        0 if weight.numel() == group_dim else group_dim,
        eps,
    )
    options = {
        "BLOCK_SIZE": block_size,
        "SINGLE_PASS": group_dim <= block_size,
        "CONTIGUOUS_ROWS": group_dim >= 1024 and x.stride(0) == x.shape[1],
        "POWER_OF_TWO_GROUPS": power_of_two_groups,
        "WEIGHT_ALIGNMENT": min(16, group_dim & -group_dim),
        "num_warps": 4,
        "num_stages": 1,
    }
    grid = (x.shape[0] * groups,)
    if runtime.torch_device_fn.current_device() == x.device.index:
        _grouped_gemma_rmsnorm_kernel[grid](*args, **options)
    else:
        with runtime.torch_device_fn.device(x.device):
            _grouped_gemma_rmsnorm_kernel[grid](*args, **options)
