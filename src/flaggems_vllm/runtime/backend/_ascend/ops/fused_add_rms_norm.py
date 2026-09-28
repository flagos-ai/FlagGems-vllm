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

import logging
import math

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.fused_add_rms_norm import (
    fused_add_rms_norm as generic_fused_add_rms_norm,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


# =============================================================================
# Dispatch constants
# =============================================================================

_MULTIROW_MIN_M = 8192
_DIRECT_MULTIROW_MAX_M = 32768
_MULTIROW_MAX_N = 4096

_N1_BLOCK_SIZE = 1024
_WIDE_BLOCK_N = 4096

# N == 100 specialized persistent batched-row path.
# Keep this path only for small M, where benchmarking showed a large win.
_N100 = 100
_N100_SPECIAL_MAX_M = 1024
_N100_BLOCK_M_CANDIDATES = (4, 8, 16, 32)

# N == 100 medium-M masked persistent path.
# Keep the efficient 128-wide vectorized layout and tune only launch
# parallelism. BLOCK_M is fixed at 16; CORE_MULT controls the number of
# persistent programs relative to the physical Vector Core count.
_N100_MEDIUM_MIN_M = 1024
_N100_MEDIUM_MAX_M = 32768
_N100_MEDIUM_BLOCK_M = 16
_N100_MEDIUM_CORE_MULT_CANDIDATES = (2, 4, 8, 16)

# Exact benchmark target: shape [100, 256, 100] => M=25600, N=100.
# Use a dedicated physical-core batched kernel so experiments on this shape do
# not perturb other N=100 cases. All candidate BLOCK_M values divide 25600,
# which lets the kernel remove the row-tail mask completely.
_N100_TARGET_M = 25600
_N100_TARGET_BLOCK_M_CANDIDATES = (8, 16, 32, 64)

# N == 256 persistent batched-row path.
_N256 = 256
_N256_PERSISTENT_MIN_M = 8192
_N256_BLOCK_M_CANDIDATES = (4, 8, 16, 32)

# N == 1024 physical-core persistent batched-row path.
_N1024 = 1024
_N1024_PERSISTENT_MAX_M = 16384
_N1024_BLOCK_M_CANDIDATES = (1, 2, 4, 8)

# N == 4096 FP16/BF16 persistent single-row path.
# Only the benchmarked M=4096 case is intercepted to avoid changing other
# already-stable shapes while this experiment is being evaluated.
_N4096 = 4096
_N4096_HALF_M = 4096
_N4096_HALF_CORE_MULT_CANDIDATES = (2, 4, 8, 16)


# =============================================================================
# Ascend Vector Core helper
# =============================================================================

_CACHED_CORE_NUM = None


def _get_core_num():
    global _CACHED_CORE_NUM

    if _CACHED_CORE_NUM is None:
        try:
            import torch_npu  # noqa: F401

            current_device = torch.npu.current_device()
            torch.npu.set_device(current_device)
            limits = torch.npu.get_device_limit(current_device)
            _CACHED_CORE_NUM = limits["vector_core_num"]
        except (ImportError, AttributeError, KeyError, TypeError):
            _CACHED_CORE_NUM = None

    return _CACHED_CORE_NUM


# =============================================================================
# Existing multi-row grid policy
# =============================================================================

_MULTIROW_GRID_SMALL = 16384
_MULTIROW_GRID_LARGE = 32768
_MULTIROW_LARGE_GRID_THRESHOLD = 131072


# =============================================================================
# Autotune helpers
# =============================================================================


def _get_n100_autotune_configs():
    return [triton.Config({"BLOCK_M": block_m}) for block_m in _N100_BLOCK_M_CANDIDATES]


def _get_n100_medium_autotune_configs():
    return [
        triton.Config({"CORE_MULT": core_mult})
        for core_mult in _N100_MEDIUM_CORE_MULT_CANDIDATES
    ]


def _get_n100_target_autotune_configs():
    return [
        triton.Config({"BLOCK_M": block_m})
        for block_m in _N100_TARGET_BLOCK_M_CANDIDATES
    ]


def _get_n256_autotune_configs():
    return [triton.Config({"BLOCK_M": block_m}) for block_m in _N256_BLOCK_M_CANDIDATES]


_MULTIROW_BLOCK_M_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)


def _get_multirow_autotune_configs():
    return [
        triton.Config({"BLOCK_M": block_m}) for block_m in _MULTIROW_BLOCK_M_CANDIDATES
    ]


def _prune_multirow_configs(configs, named_args, **kwargs):
    block_n = kwargs.get("BLOCK_N")
    if block_n is None:
        block_n = named_args.get("BLOCK_N", None)
    if block_n is None:
        return configs

    block_n = int(block_n)
    valid_configs = [
        config for config in configs if config.kwargs["BLOCK_M"] * block_n <= 8192
    ]
    return valid_configs if valid_configs else [configs[0]]


def _get_n1024_autotune_configs():
    return [
        triton.Config({"BLOCK_M": block_m}) for block_m in _N1024_BLOCK_M_CANDIDATES
    ]


def _get_n4096_half_autotune_configs():
    return [
        triton.Config({"CORE_MULT": core_mult})
        for core_mult in _N4096_HALF_CORE_MULT_CANDIDATES
    ]


# =============================================================================
# Grid helpers
# =============================================================================


def _n100_persistent_grid(M, cores):
    def grid(meta):
        total_tiles = triton.cdiv(M, meta["BLOCK_M"])
        return (min(cores, total_tiles),)

    return grid


def _n100_medium_persistent_grid(M, cores):
    def grid(meta):
        total_tiles = triton.cdiv(M, _N100_MEDIUM_BLOCK_M)
        program_count = cores * meta["CORE_MULT"]
        return (min(total_tiles, program_count),)

    return grid


def _n100_target_physical_grid(cores):
    def grid(meta):
        total_tiles = triton.cdiv(_N100_TARGET_M, meta["BLOCK_M"])
        return (min(cores, total_tiles),)

    return grid


def _n1024_persistent_grid(M, cores):
    def grid(meta):
        total_tiles = triton.cdiv(M, meta["BLOCK_M"])
        return (min(cores, total_tiles),)

    return grid


def _n4096_half_persistent_grid(M, cores):
    def grid(meta):
        program_count = cores * meta["CORE_MULT"]
        return (min(M, program_count),)

    return grid


def _direct_multirow_grid(M):
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    return grid


def _grid_stride_multirow_grid(M):
    def grid(meta):
        total_tiles = triton.cdiv(M, meta["BLOCK_M"])
        if total_tiles <= _MULTIROW_GRID_SMALL:
            program_count = total_tiles
        elif total_tiles <= _MULTIROW_LARGE_GRID_THRESHOLD:
            program_count = _MULTIROW_GRID_SMALL
        else:
            program_count = _MULTIROW_GRID_LARGE
        return (program_count,)

    return grid


# =============================================================================
# N == 1 fast path
# =============================================================================


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_n1_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    M,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    added_native = x + residual

    tl.store(residual_ptr + offsets, added_native, mask=mask)

    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        added = added_native.to(tl.float32)
    else:
        added = added_native

    weight = tl.load(weight_ptr)
    rrms = 1.0 / tl.sqrt(added * added + eps)
    output = (added * rrms * weight).to(input_ptr.dtype.element_ty)
    tl.store(input_ptr + offsets, output, mask=mask)


# =============================================================================
# N == 100 specialized physical-core persistent + batched rows
# =============================================================================


@libentry()
@triton.autotune(
    configs=_get_n100_autotune_configs(),
    key=["M"],
    restore_value=["input_ptr", "residual_ptr"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_n100_persistent_batched_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    M,
    eps,
    BLOCK_M: tl.constexpr,
):
    """N=100 specialized persistent kernel.

    BLOCK_N is fixed at 128 because Triton reduction width is power-of-two.
    Only the 28-column tail is masked. Each physical program preloads weight
    once and processes multiple BLOCK_M x 128 tiles with a grid-stride loop.
    """

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    rows_lane = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 128)
    col_mask = cols < 100

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
    )

    total_tiles = tl.cdiv(M, BLOCK_M)

    for tile_id in tl.range(pid, total_tiles, num_programs):
        rows = tile_id * BLOCK_M + rows_lane
        row_mask = rows < M

        offsets = rows[:, None] * 100 + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(
            input_ptr + offsets,
            mask=mask,
            other=0.0,
        )
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
        )

        added_native = x + residual
        tl.store(
            residual_ptr + offsets,
            added_native,
            mask=mask,
        )

        if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
            input_ptr.dtype.element_ty == tl.bfloat16
        ):
            added = added_native.to(tl.float32)
        else:
            added = added_native

        # The 28 invalid lanes are zero because masked loads use other=0.0.
        sum_sq = tl.sum(added * added, axis=1)
        variance = sum_sq / 100
        rrms = 1.0 / tl.sqrt(variance + eps)

        output = (added * rrms[:, None] * weight[None, :]).to(
            input_ptr.dtype.element_ty
        )

        tl.store(
            input_ptr + offsets,
            output,
            mask=mask,
        )


# =============================================================================
# N == 100, M == 25600 dedicated physical-core batched persistent path
# =============================================================================


@libentry()
@triton.autotune(
    configs=_get_n100_target_autotune_configs(),
    # M is fixed for this specialized kernel. Use an explicit dtype id so
    # FP16 / FP32 / BF16 tune independently instead of sharing a winner.
    key=["DTYPE_ID"],
    restore_value=["input_ptr", "residual_ptr"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_n100_m25600_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    eps,
    DTYPE_ID,
    BLOCK_M: tl.constexpr,
):
    """Exact fast path for shape [100, 256, 100].

    M is exactly 25600 and every candidate BLOCK_M in {8, 16, 32, 64}
    divides M, so there is no row-tail mask. The N dimension stays 128-wide
    with only the 28-column tail masked, preserving efficient vectorization.

    Launch exactly up to the physical Vector Core count. BLOCK_M controls how
    many contiguous rows are packed into each runtime tile; increasing it cuts
    the number of grid-stride loop iterations without multiplying the number
    of launched programs.
    """

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    rows_lane = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 128)
    col_mask = cols < 100

    # Weight is reused by every row tile handled by this persistent program.
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0)

    # 25600 is divisible by every BLOCK_M candidate, therefore every row in
    # every generated tile is valid and row_mask can be eliminated entirely.
    total_tiles = 25600 // BLOCK_M

    for tile_id in tl.range(pid, total_tiles, num_programs):
        rows = tile_id * BLOCK_M + rows_lane
        offsets = rows[:, None] * 100 + cols[None, :]
        mask = col_mask[None, :]

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

        added_native = x + residual
        tl.store(residual_ptr + offsets, added_native, mask=mask)

        if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
            input_ptr.dtype.element_ty == tl.bfloat16
        ):
            added = added_native.to(tl.float32)
        else:
            added = added_native

        sum_sq = tl.sum(added * added, axis=1)
        rrms = 1.0 / tl.sqrt(sum_sq / 100 + eps)

        output = (added * rrms[:, None] * weight[None, :]).to(
            input_ptr.dtype.element_ty
        )
        tl.store(input_ptr + offsets, output, mask=mask)


# =============================================================================
# N == 100 medium-M 128-wide masked persistent + batched rows
# =============================================================================


@libentry()
@triton.autotune(
    configs=_get_n100_medium_autotune_configs(),
    key=["M"],
    restore_value=["input_ptr", "residual_ptr"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_n100_medium_masked_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    M,
    eps,
    CORE_MULT: tl.constexpr,
):
    """N=100 medium-M persistent kernel.

    Keep the original 128-wide vectorized layout because the exact 64+32+4
    segmentation was substantially slower on Ascend. BLOCK_M is fixed at 16
    to keep each tile coarse enough, while CORE_MULT autotunes launch
    parallelism over {2, 4, 8, 16} x physical Vector Core count.

    Each persistent program preloads weight[128] once and processes multiple
    16 x 128 tiles through a runtime grid-stride loop. Only the 28-column tail
    and final row tile are masked.
    """

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    rows_lane = tl.arange(0, 16)
    cols = tl.arange(0, 128)
    col_mask = cols < 100

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
    )

    total_tiles = tl.cdiv(M, 16)

    for tile_id in tl.range(pid, total_tiles, num_programs):
        rows = tile_id * 16 + rows_lane
        row_mask = rows < M

        offsets = rows[:, None] * 100 + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(
            input_ptr + offsets,
            mask=mask,
            other=0.0,
        )
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
        )

        added_native = x + residual
        tl.store(
            residual_ptr + offsets,
            added_native,
            mask=mask,
        )

        if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
            input_ptr.dtype.element_ty == tl.bfloat16
        ):
            added = added_native.to(tl.float32)
        else:
            added = added_native

        sum_sq = tl.sum(added * added, axis=1)
        variance = sum_sq / 100
        rrms = 1.0 / tl.sqrt(variance + eps)

        output = (added * rrms[:, None] * weight[None, :]).to(
            input_ptr.dtype.element_ty
        )

        tl.store(
            input_ptr + offsets,
            output,
            mask=mask,
        )


# =============================================================================
# N == 256 physical-core persistent + batched rows
# =============================================================================


@libentry()
@triton.autotune(
    configs=_get_n256_autotune_configs(),
    key=["M"],
    restore_value=["input_ptr", "residual_ptr"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_n256_persistent_batched_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    M,
    eps,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    rows_lane = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 256)
    weight = tl.load(weight_ptr + cols)

    total_tiles = tl.cdiv(M, BLOCK_M)

    for tile_id in tl.range(pid, total_tiles, num_programs):
        rows = tile_id * BLOCK_M + rows_lane
        row_mask = rows < M
        offsets = rows[:, None] * 256 + cols[None, :]
        mask = row_mask[:, None]

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        added_native = x + residual
        tl.store(residual_ptr + offsets, added_native, mask=mask)

        if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
            input_ptr.dtype.element_ty == tl.bfloat16
        ):
            added = added_native.to(tl.float32)
        else:
            added = added_native

        sum_sq = tl.sum(added * added, axis=1)
        variance = sum_sq / 256
        rrms = 1.0 / tl.sqrt(variance + eps)
        output = (added * rrms[:, None] * weight[None, :]).to(
            input_ptr.dtype.element_ty
        )

        tl.store(input_ptr + offsets, output, mask=mask)


# =============================================================================
# N == 1024 physical-core persistent + batched rows
# =============================================================================


@libentry()
@triton.autotune(
    configs=_get_n1024_autotune_configs(),
    key=["M"],
    restore_value=["input_ptr", "residual_ptr"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_n1024_persistent_batched_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    M,
    eps,
    BLOCK_M: tl.constexpr,
):
    """Dedicated N=1024 persistent batched-row kernel.

    Each physical Vector Core preloads weight[1024] once and processes
    multiple BLOCK_M x 1024 row tiles with a runtime grid-stride loop.
    N=1024 exactly matches the tile width, so the column dimension is
    completely mask-free; only the final row tile needs a row mask.
    """

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    rows_lane = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 1024)

    # N == 1024 exactly: preload weight once, with no column mask.
    weight = tl.load(weight_ptr + cols)

    total_tiles = tl.cdiv(M, BLOCK_M)

    for tile_id in tl.range(pid, total_tiles, num_programs):
        rows = tile_id * BLOCK_M + rows_lane
        row_mask = rows < M

        offsets = rows[:, None] * 1024 + cols[None, :]
        mask = row_mask[:, None]

        x = tl.load(
            input_ptr + offsets,
            mask=mask,
            other=0.0,
        )
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
        )

        # Keep residual add in native dtype. Promote only the fused value
        # for RMS accumulation to reduce FP32 temporary pressure.
        added_native = x + residual
        tl.store(
            residual_ptr + offsets,
            added_native,
            mask=mask,
        )

        if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
            input_ptr.dtype.element_ty == tl.bfloat16
        ):
            added = added_native.to(tl.float32)
        else:
            added = added_native

        # BLOCK_M independent 1024-element RMS reductions.
        sum_sq = tl.sum(added * added, axis=1)
        variance = sum_sq / 1024
        rrms = 1.0 / tl.sqrt(variance + eps)

        output = (added * rrms[:, None] * weight[None, :]).to(
            input_ptr.dtype.element_ty
        )

        tl.store(
            input_ptr + offsets,
            output,
            mask=mask,
        )


# =============================================================================
# N == 4096 FP16/BF16 persistent single-row path
# =============================================================================


@libentry()
@triton.autotune(
    configs=_get_n4096_half_autotune_configs(),
    key=["M"],
    restore_value=["input_ptr", "residual_ptr"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_n4096_half_persistent_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    M,
    eps,
    CORE_MULT: tl.constexpr,
):
    """Dedicated N=4096 FP16/BF16 persistent single-row kernel.

    CORE_MULT is an autotuned launch meta-parameter. The grid callback sweeps
    {2, 4, 8, 16} x physical Vector Core count. Each launched program preloads
    weight[4096] once, then processes rows with a runtime grid-stride loop.

    N=4096 is exact and power-of-two, so the column dimension is fully
    mask-free. Unlike the N=256 case, one FP16 row is already 8 KiB, making
    single-row persistent work sufficiently coarse-grained for this experiment.
    """

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    cols = tl.arange(0, 4096)

    # Reuse the same 4096-element weight vector across all rows handled by
    # this persistent program.
    weight = tl.load(weight_ptr + cols)

    for row in tl.range(pid, M, num_programs):
        offsets = row * 4096 + cols

        # N == 4096 exactly: no column-tail mask.
        x = tl.load(input_ptr + offsets)
        residual = tl.load(residual_ptr + offsets)

        # Keep the residual add in FP16/BF16 and promote only the fused value for
        # RMS accumulation.
        added_native = x + residual
        tl.store(residual_ptr + offsets, added_native)

        added = added_native.to(tl.float32)

        sum_sq = tl.sum(added * added, axis=0)
        variance = sum_sq / 4096
        rrms = 1.0 / tl.sqrt(variance + eps)

        output = (added * rrms * weight).to(input_ptr.dtype.element_ty)
        tl.store(input_ptr + offsets, output)


# =============================================================================
# Existing direct multi-row path
# =============================================================================


@libentry()
@triton.autotune(
    configs=_get_multirow_autotune_configs(),
    key=["M", "N"],
    prune_configs_by={"early_config_prune": _prune_multirow_configs},
    restore_value=["input_ptr", "residual_ptr"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_multirow_direct_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_id = tl.program_id(0)
    rows_lane = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    rows = tile_id * BLOCK_M + rows_lane
    row_mask = rows < M
    col_mask = cols < N
    offsets = rows[:, None] * N + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    added_native = x + residual
    tl.store(residual_ptr + offsets, added_native, mask=mask)

    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        added = added_native.to(tl.float32)
    else:
        added = added_native

    sum_sq = tl.sum(added * added, axis=1)
    variance = sum_sq / N
    rrms = 1.0 / tl.sqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0)

    output = (added * rrms[:, None] * weight[None, :]).to(input_ptr.dtype.element_ty)

    tl.store(input_ptr + offsets, output, mask=mask)


# =============================================================================
# Existing grid-stride multi-row path
# =============================================================================


@libentry()
@triton.autotune(
    configs=_get_multirow_autotune_configs(),
    key=["M", "N"],
    prune_configs_by={"early_config_prune": _prune_multirow_configs},
    restore_value=["input_ptr", "residual_ptr"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_multirow_grid_stride_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total_tiles = tl.cdiv(M, BLOCK_M)

    rows_lane = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0)

    for tile_id in tl.range(pid, total_tiles, num_programs):
        rows = tile_id * BLOCK_M + rows_lane
        row_mask = rows < M
        offsets = rows[:, None] * N + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        added_native = x + residual
        tl.store(residual_ptr + offsets, added_native, mask=mask)

        if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
            input_ptr.dtype.element_ty == tl.bfloat16
        ):
            added = added_native.to(tl.float32)
        else:
            added = added_native

        sum_sq = tl.sum(added * added, axis=1)
        variance = sum_sq / N
        rrms = 1.0 / tl.sqrt(variance + eps)
        output = (added * rrms[:, None] * weight[None, :]).to(
            input_ptr.dtype.element_ty
        )

        tl.store(input_ptr + offsets, output, mask=mask)


# =============================================================================
# Existing wide-N two-pass path
# =============================================================================


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_wide_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_base = pid * N
    cols = tl.arange(0, BLOCK_N)

    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        compute_dtype = tl.float32
    else:
        compute_dtype = input_ptr.dtype.element_ty

    sum_sq = tl.zeros([1], dtype=compute_dtype)

    for start in tl.range(0, N, BLOCK_N):
        offsets = start + cols
        mask = offsets < N

        x = tl.load(
            input_ptr + row_base + offsets,
            mask=mask,
            other=0.0,
        )
        residual = tl.load(
            residual_ptr + row_base + offsets,
            mask=mask,
            other=0.0,
        )
        added_native = x + residual
        tl.store(
            residual_ptr + row_base + offsets,
            added_native,
            mask=mask,
        )

        if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
            input_ptr.dtype.element_ty == tl.bfloat16
        ):
            added = added_native.to(tl.float32)
        else:
            added = added_native

        sum_sq += tl.sum(added * added, axis=0)

    variance = sum_sq / N
    rrms = 1.0 / tl.sqrt(variance + eps)

    for start in tl.range(0, N, BLOCK_N):
        offsets = start + cols
        mask = offsets < N

        added_native = tl.load(
            residual_ptr + row_base + offsets,
            mask=mask,
            other=0.0,
        )

        if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
            input_ptr.dtype.element_ty == tl.bfloat16
        ):
            added = added_native.to(tl.float32)
        else:
            added = added_native

        weight = tl.load(
            weight_ptr + offsets,
            mask=mask,
            other=0.0,
        )
        output = (added * rrms * weight).to(input_ptr.dtype.element_ty)
        tl.store(
            input_ptr + row_base + offsets,
            output,
            mask=mask,
        )


# =============================================================================
# Python entry
# =============================================================================


def fused_add_rms_norm(
    x,
    residual,
    normalized_shape,
    weight,
    eps=1e-5,
):
    logger.debug(
        "GEMS_ASCEND FUSED_ADD_RMS_NORM FORWARD, "
        "[input shape]: %s, [residual shape]: %s, [weight shape]: %s",
        x.size(),
        residual.size(),
        weight.size(),
    )

    assert x.shape == residual.shape, (
        "Input and residual shapes must match: " f"{x.shape} vs {residual.shape}"
    )

    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    if M == 0 or N == 0:
        return x, residual

    # -------------------------------------------------------------------------
    # N == 1
    # -------------------------------------------------------------------------
    if N == 1:
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()

        grid = (triton.cdiv(M, _N1_BLOCK_SIZE),)

        with torch_device_fn.device(x.device):
            fused_add_rms_norm_n1_kernel[grid](
                x,
                residual,
                weight,
                M,
                eps,
                BLOCK_SIZE=_N1_BLOCK_SIZE,
            )

        return x, residual

    # -------------------------------------------------------------------------
    # N == 100 specialized path for small M only.
    #
    # Benchmarking showed a large gain for M=100, while the same persistent
    # path regressed the M=25600 and very-large-M N=100 cases. Keep the
    # specialized kernel only where it is beneficial; larger M falls through
    # to the existing generic / multi-row paths below.
    # -------------------------------------------------------------------------
    if N == _N100 and M <= _N100_SPECIAL_MAX_M:
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()

        cores = _get_core_num()
        if cores is None:
            cores = 24

        grid = _n100_persistent_grid(M, cores)

        with torch_device_fn.device(x.device):
            fused_add_rms_norm_n100_persistent_batched_kernel[grid](
                x,
                residual,
                weight,
                M=M,
                eps=eps,
                multibuffer=True,
                limit_auto_multi_buffer_only_for_local_buffer=False,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
            )

        return x, residual

    # -------------------------------------------------------------------------
    # Exact target shape [100, 256, 100] => M=25600, N=100.
    #
    # Use physical-core persistent execution and tune only BLOCK_M. Because
    # every candidate BLOCK_M divides 25600, this fast path removes row-tail
    # masking completely while keeping the efficient 128-wide column layout.
    # -------------------------------------------------------------------------
    if N == _N100 and M == _N100_TARGET_M:
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()

        cores = _get_core_num()
        if cores is None:
            cores = 24

        if x.dtype == torch.float16:
            dtype_id = 0
        elif x.dtype == torch.float32:
            dtype_id = 1
        elif x.dtype == torch.bfloat16:
            dtype_id = 2
        else:
            dtype_id = 3

        grid = _n100_target_physical_grid(cores)

        with torch_device_fn.device(x.device):
            fused_add_rms_norm_n100_m25600_kernel[grid](
                x,
                residual,
                weight,
                eps=eps,
                DTYPE_ID=dtype_id,
                multibuffer=True,
                limit_auto_multi_buffer_only_for_local_buffer=False,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
            )

        return x, residual

    # -------------------------------------------------------------------------
    # N == 100 medium-M 128-wide masked persistent path.
    #
    # The exact 64+32+4 segmentation regressed badly, so keep the efficient
    # 128-wide vectorized layout and tune only persistent launch parallelism.
    # BLOCK_M is fixed at 16; CORE_MULT sweeps {2, 4, 8, 16}.
    # -------------------------------------------------------------------------
    if N == _N100 and M > _N100_MEDIUM_MIN_M and M <= _N100_MEDIUM_MAX_M:
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()

        cores = _get_core_num()
        if cores is None:
            cores = 24

        grid = _n100_medium_persistent_grid(M, cores)

        with torch_device_fn.device(x.device):
            fused_add_rms_norm_n100_medium_masked_kernel[grid](
                x,
                residual,
                weight,
                M=M,
                eps=eps,
                multibuffer=True,
                limit_auto_multi_buffer_only_for_local_buffer=False,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
            )

        return x, residual

    # -------------------------------------------------------------------------
    # N == 256 persistent batched-row path
    # -------------------------------------------------------------------------
    if N == _N256 and M >= _N256_PERSISTENT_MIN_M:
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()

        cores = _get_core_num()
        if cores is None:
            cores = 24

        with torch_device_fn.device(x.device):
            fused_add_rms_norm_n256_persistent_batched_kernel[(cores,)](
                x,
                residual,
                weight,
                M=M,
                eps=eps,
                multibuffer=True,
                limit_auto_multi_buffer_only_for_local_buffer=False,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
            )

        return x, residual

    # -------------------------------------------------------------------------
    # N == 1024 physical-core persistent batched-row path
    #
    # Replaces the previous ROWS_PER_PROGRAM + tl.static_range row-group
    # implementation. BLOCK_M is autotuned over {1, 2, 4, 8}; each program
    # processes multiple row tiles and reuses a single preload of weight[1024].
    # -------------------------------------------------------------------------
    if N == _N1024 and M <= _N1024_PERSISTENT_MAX_M:
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()

        cores = _get_core_num()
        if cores is None:
            cores = 24

        with torch_device_fn.device(x.device):
            grid = _n1024_persistent_grid(M, cores)
            fused_add_rms_norm_n1024_persistent_batched_kernel[grid](
                x,
                residual,
                weight,
                M=M,
                eps=eps,
                multibuffer=True,
                limit_auto_multi_buffer_only_for_local_buffer=False,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
            )

        return x, residual

    # -------------------------------------------------------------------------
    # N == 4096, M == 4096, FP16/BF16 persistent single-row path
    #
    # FP32 already matches/exceeds the native baseline, while both FP16 and
    # BF16 benefit from the specialized persistent path. Keep FP32 unchanged.
    # CORE_MULT is autotuned over {2, 4, 8, 16}.
    # -------------------------------------------------------------------------
    if (
        N == _N4096
        and M == _N4096_HALF_M
        and x.dtype in (torch.float16, torch.bfloat16)
    ):
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()

        cores = _get_core_num()
        if cores is None:
            cores = 24

        grid = _n4096_half_persistent_grid(M, cores)

        with torch_device_fn.device(x.device):
            fused_add_rms_norm_n4096_half_persistent_kernel[grid](
                x,
                residual,
                weight,
                M=M,
                eps=eps,
                multibuffer=True,
                limit_auto_multi_buffer_only_for_local_buffer=False,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
            )

        return x, residual

    # -------------------------------------------------------------------------
    # Small-M generic fallback
    # -------------------------------------------------------------------------
    if M < _MULTIROW_MIN_M:
        return generic_fused_add_rms_norm(
            x,
            residual,
            normalized_shape,
            weight,
            eps,
        )

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    with torch_device_fn.device(x.device):
        if N <= _MULTIROW_MAX_N:
            block_n = triton.next_power_of_2(N)

            if M <= _DIRECT_MULTIROW_MAX_M:
                grid = _direct_multirow_grid(M)
                fused_add_rms_norm_multirow_direct_kernel[grid](
                    x,
                    residual,
                    weight,
                    M=M,
                    N=N,
                    eps=eps,
                    BLOCK_N=block_n,
                    multibuffer=True,
                    limit_auto_multi_buffer_only_for_local_buffer=False,
                    limit_auto_multi_buffer_of_local_buffer="no-limit",
                )
            else:
                grid = _grid_stride_multirow_grid(M)
                fused_add_rms_norm_multirow_grid_stride_kernel[grid](
                    x,
                    residual,
                    weight,
                    M=M,
                    N=N,
                    eps=eps,
                    BLOCK_N=block_n,
                    multibuffer=True,
                    limit_auto_multi_buffer_only_for_local_buffer=False,
                    limit_auto_multi_buffer_of_local_buffer="no-limit",
                )
        else:
            fused_add_rms_norm_wide_kernel[(M,)](
                x,
                residual,
                weight,
                N,
                eps,
                BLOCK_N=_WIDE_BLOCK_N,
            )

    return x, residual
