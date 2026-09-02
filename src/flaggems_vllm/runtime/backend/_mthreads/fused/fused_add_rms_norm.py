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

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tuned constants
# -----------------------------------------------------------------------------
#
# Large-N whole-row/tiled thresholds are intentionally kept close to the
# original NVIDIA baseline for this first MThreads/MUSA port.
_WHOLE_ROW_SAFE_N = 16384
_LARGE_N_THRESHOLD = 32768
_SMALL_M_THRESHOLD = 8

_TILE_SIZE = 1024

# Small-N / extreme-M persistent 2D multi-row specialization.
#
# For [100, 256, 100] with normalized_shape=[100]:
#   M = 25600, N = 100 -> ordinary whole-row.
#
# For [100, 65536, 100] with normalized_shape=[100]:
#   M = 6553600, N = 100 -> persistent 2D multi-row.
#
# Each persistent program handles BLOCK_M rows in parallel in one loop
# iteration, and then advances by num_programs * BLOCK_M rows.
_SMALL_N_2D_PERSISTENT_MAX_N = 128
_2D_PERSISTENT_MIN_M = 262144
_2D_BLOCK_M = 4
_2D_PERSISTENT_NUM_PROGRAMS = 4096
_2D_PERSISTENT_NUM_WARPS = 4
_2D_PERSISTENT_NUM_STAGES = 1


# -----------------------------------------------------------------------------
# 1) Whole-row baseline
# -----------------------------------------------------------------------------
@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    in_stride_r,
    in_stride_c,
    r_stride_r,
    r_stride_c,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        compute_dtype = tl.float32
    else:
        compute_dtype = input_ptr.dtype.element_ty

    pid = ext.program_id(0)

    input_ptr += pid * in_stride_r
    residual_ptr += pid * r_stride_r

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(
        input_ptr + cols * in_stride_c,
        mask=mask,
        other=0.0,
    ).to(compute_dtype)

    residual = tl.load(
        residual_ptr + cols * r_stride_c,
        mask=mask,
        other=0.0,
    ).to(compute_dtype)

    added = x + residual

    # residual <- input + residual
    tl.store(
        residual_ptr + cols * r_stride_c,
        added,
        mask=mask,
    )

    sum_sq = tl.sum(added * added, axis=0)
    variance = sum_sq / N
    rrms = 1.0 / tl.sqrt(variance + eps)

    weight = tl.load(
        w_ptr + cols,
        mask=mask,
        other=0.0,
    )

    output = (added * rrms * weight).to(compute_dtype)

    # input <- RMSNorm(input + residual) * weight
    tl.store(
        input_ptr + cols * in_stride_c,
        output,
        mask=mask,
    )


# -----------------------------------------------------------------------------
# 2) Small-N / extreme-M persistent 2D multi-row path
# -----------------------------------------------------------------------------
@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_2d_persistent_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    in_stride_r,
    in_stride_c,
    r_stride_r,
    r_stride_c,
    M,
    N,
    num_row_groups,
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Persistent 2D multi-row kernel for very large M and short rows.

    One persistent program processes BLOCK_M independent normalized rows in
    parallel per loop iteration. For BLOCK_M=4 and BLOCK_N=128 the logical
    tile is [4, 128]. Reduction is performed along axis=1, producing one RMS
    value for each of the four rows.

    Program pid processes row groups:
      pid,
      pid + num_programs,
      pid + 2 * num_programs,
      ...

    where one row group contains BLOCK_M consecutive rows.

    Weight is intentionally loaded *inside* the persistent loop. The vector is
    therefore only live for one row-group iteration instead of remaining live
    across the entire persistent kernel, reducing long register live ranges.
    """
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        compute_dtype = tl.float32
    else:
        compute_dtype = input_ptr.dtype.element_ty

    pid = ext.program_id(0)
    group_step = tl.num_programs(0)

    row_offsets = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N

    for group_id in tl.range(
        pid,
        num_row_groups,
        group_step,
        num_stages=1,
    ):
        rows = group_id * BLOCK_M + row_offsets

        # [BLOCK_M, BLOCK_N]
        mask = (rows[:, None] < M) & col_mask[None, :]

        input_offsets = rows[:, None] * in_stride_r + cols[None, :] * in_stride_c
        residual_offsets = rows[:, None] * r_stride_r + cols[None, :] * r_stride_c

        x = tl.load(
            input_ptr + input_offsets,
            mask=mask,
            other=0.0,
        ).to(compute_dtype)

        residual = tl.load(
            residual_ptr + residual_offsets,
            mask=mask,
            other=0.0,
        ).to(compute_dtype)

        added = x + residual

        # residual <- input + residual
        tl.store(
            residual_ptr + residual_offsets,
            added,
            mask=mask,
        )

        # Reduce each row independently: [BLOCK_M, BLOCK_N] -> [BLOCK_M].
        sum_sq = tl.sum(added * added, axis=1)
        variance = sum_sq / N
        rrms = 1.0 / tl.sqrt(variance + eps)

        # Intentionally inside the persistent loop: shorter register live range.
        weight = tl.load(
            w_ptr + cols,
            mask=col_mask,
            other=0.0,
        )

        # Broadcast rrms over columns and weight over rows.
        output = (added * rrms[:, None] * weight[None, :]).to(compute_dtype)

        # input <- RMSNorm(input + residual) * weight
        tl.store(
            input_ptr + input_offsets,
            output,
            mask=mask,
        )


# -----------------------------------------------------------------------------
# 3) Single-CTA fixed-tile two-pass path
# -----------------------------------------------------------------------------
@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_tiled_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    in_stride_r,
    in_stride_c,
    r_stride_r,
    r_stride_c,
    N,
    eps,
    TILE_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        compute_dtype = tl.float32
    else:
        compute_dtype = input_ptr.dtype.element_ty

    pid = ext.program_id(0)

    input_row = input_ptr + pid * in_stride_r
    residual_row = residual_ptr + pid * r_stride_r

    offsets = tl.arange(0, TILE_SIZE)
    sum_sq = tl.zeros([1], dtype=compute_dtype)

    # Pass 1: compute the row sum of squares.
    for tile_start in tl.range(
        0,
        N,
        TILE_SIZE,
        num_stages=1,
    ):
        cols = tile_start + offsets
        mask = cols < N

        x = tl.load(
            input_row + cols * in_stride_c,
            mask=mask,
            other=0.0,
        ).to(compute_dtype)

        residual = tl.load(
            residual_row + cols * r_stride_c,
            mask=mask,
            other=0.0,
        ).to(compute_dtype)

        added = x + residual
        sum_sq += tl.sum(added * added, axis=0)

    variance = sum_sq / N
    rrms = 1.0 / tl.sqrt(variance + eps)

    # Pass 2: re-read, update residual in-place, and write normalized output.
    for tile_start in tl.range(
        0,
        N,
        TILE_SIZE,
        num_stages=1,
    ):
        cols = tile_start + offsets
        mask = cols < N

        x = tl.load(
            input_row + cols * in_stride_c,
            mask=mask,
            other=0.0,
        ).to(compute_dtype)

        residual = tl.load(
            residual_row + cols * r_stride_c,
            mask=mask,
            other=0.0,
        ).to(compute_dtype)

        weight = tl.load(
            w_ptr + cols,
            mask=mask,
            other=0.0,
        )

        added = x + residual
        output = (added * rrms * weight).to(compute_dtype)

        tl.store(
            residual_row + cols * r_stride_c,
            added,
            mask=mask,
        )

        tl.store(
            input_row + cols * in_stride_c,
            output,
            mask=mask,
        )


# -----------------------------------------------------------------------------
# Dispatch policy
# -----------------------------------------------------------------------------
def _prefer_2d_persistent(M, N):
    """Use the persistent 2D path only for short rows with extreme M."""
    return N <= _SMALL_N_2D_PERSISTENT_MAX_N and M >= _2D_PERSISTENT_MIN_M


def _prefer_whole_row(x, M, N):
    """Initial MThreads selector copied from the NVIDIA benchmark policy.

    Small-N/extreme-M is handled before this selector. For the remaining cases,
    this keeps the NVIDIA whole-row/tiled crossover policy as a migration
    baseline. NVIDIA Cluster-selected cases are represented here as
    "not whole-row" and therefore fall back to the single-CTA tiled kernel.
    """
    if N <= _WHOLE_ROW_SAFE_N:
        return True

    # NVIDIA measurements showed a next_power_of_2 cliff immediately above
    # 16384, so use the tiled path in this interval for the initial baseline.
    if N < 24576:
        return False

    # In the original NVIDIA BLOCK_SIZE=32768 region:
    #   * FP16/BF16 preferred whole-row.
    #   * FP32 with small M preferred Cluster.
    # Since this implementation intentionally has no Cluster path, those
    # small-M FP32 cases use the single-CTA tiled kernel instead.
    if N <= _LARGE_N_THRESHOLD:
        if M <= _SMALL_M_THRESHOLD and x.dtype == torch.float32:
            return False
        return True

    # Above 32768, the original NVIDIA implementation avoided whole-row because
    # next_power_of_2(N) becomes 65536. Use the tiled path here as well.
    return False


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """Fused residual addition + RMSNorm, in-place.

    MThreads/MUSA implementation with three execution paths:

      A) Small-N / extreme-M persistent 2D multi-row kernel
         - N <= 128 and M >= 262144.
         - BLOCK_M=4 rows are processed in parallel per loop iteration.
         - BLOCK_N=next_power_of_2(N), so N=100 uses BLOCK_N=128.
         - Launches at most 4096 persistent programs.
         - Program pid processes row groups pid, pid+grid, pid+2*grid, ...
         - Weight is loaded inside each persistent loop iteration.
         - Initial launch: num_warps=4, num_stages=1.

         This means:
           * [100, 256, 100], normalized_shape=[100]
               M=25600, N=100 -> ordinary whole-row.
           * [100, 65536, 100], normalized_shape=[100]
               M=6553600, N=100 -> persistent 2D multi-row.

      B) Whole-row Triton kernel
         - Used for ordinary small/medium N cases not captured by A.
         - BLOCK_SIZE = triton.next_power_of_2(N).

      C) Single-CTA fixed-tile two-pass Triton kernel
         - TILE_SIZE=1024.
         - num_warps=4, num_stages=1.

    Large-N dispatch policy is kept from the NVIDIA baseline:

      1) 16384 < N < 24576
           -> single-CTA tiled.

      2) 24576 <= N <= 32768
           -> FP16/BF16: whole-row.
           -> FP32, M <= 8: single-CTA tiled.
           -> FP32, M > 8: whole-row.

      3) N > 32768
           -> single-CTA tiled.

    Notes:
      * No CUDA capability checks.
      * No TLE device mesh / CTA Cluster code.
      * The 2D BLOCK_M, persistent grid, warp count, and threshold are first-pass
        MThreads tuning values and should be benchmarked on MTT S5000.
    """
    logger.debug(
        "GEMS FUSED_ADD_RMS_NORM FORWARD (MTHREADS 2D PERSISTENT), "
        "[input shape]: %s, [residual shape]: %s, [weight shape]: %s",
        x.size(),
        residual.size(),
        weight.size(),
    )

    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    prefer_2d_persistent = _prefer_2d_persistent(M, N)
    prefer_whole = _prefer_whole_row(x, M, N)

    with torch_device_fn.device(x.device):
        if prefer_2d_persistent:
            block_n = triton.next_power_of_2(N)
            num_row_groups = triton.cdiv(M, _2D_BLOCK_M)
            num_programs = min(num_row_groups, _2D_PERSISTENT_NUM_PROGRAMS)

            fused_add_rms_norm_2d_persistent_kernel[(num_programs,)](
                x,
                residual,
                weight,
                N,
                1,
                N,
                1,
                M,
                N,
                num_row_groups,
                eps,
                BLOCK_N=block_n,
                BLOCK_M=_2D_BLOCK_M,
                num_warps=_2D_PERSISTENT_NUM_WARPS,
                num_stages=_2D_PERSISTENT_NUM_STAGES,
            )

        elif prefer_whole:
            block_size = triton.next_power_of_2(N)

            fused_add_rms_norm_kernel[(M,)](
                x,
                residual,
                weight,
                N,
                1,
                N,
                1,
                N,
                eps,
                BLOCK_SIZE=block_size,
            )

        else:
            fused_add_rms_norm_tiled_kernel[(M,)](
                x,
                residual,
                weight,
                N,
                1,
                N,
                1,
                N,
                eps,
                TILE_SIZE=_TILE_SIZE,
                num_warps=4,
                num_stages=1,
            )

    return x, residual
