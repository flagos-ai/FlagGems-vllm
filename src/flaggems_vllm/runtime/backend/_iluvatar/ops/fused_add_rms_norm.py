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
# Generic-path constants
# -----------------------------------------------------------------------------

_WHOLE_ROW_SAFE_N = 16384
_LARGE_N_THRESHOLD = 32768


# -----------------------------------------------------------------------------
# Iluvatar small-N / large-M fast path
# -----------------------------------------------------------------------------
#
# The generic fused_add_rms_norm whole-row kernel launches one Triton program
# for every logical row.  That is efficient for moderate/large hidden sizes,
# but becomes launch/scheduling heavy when N is narrow and M is very large.
#
# BI-V150 benefits from "fattening" each program so that one program processes
# several rows at once.  The same TILE_M idea is already used by other Iluvatar
# RMSNorm kernels in this backend.
#
# Keep the initial dispatch conservative:
#   * FP16/BF16 only (the performance target used by vLLM-Iluvatar);
#   * 64 <= N <= 256;
#   * only sufficiently large M.
#
# This intentionally leaves the already-good small-M cases on the generic path.

_ILUVATAR_MULTIROW_MIN_N = 64
_ILUVATAR_MULTIROW_MAX_N = 256
_ILUVATAR_MULTIROW_M_THRESHOLD_N128 = 1024
_ILUVATAR_MULTIROW_M_THRESHOLD_N256 = 4096


def _should_use_iluvatar_multirow(x, M, N):
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False

    if N < _ILUVATAR_MULTIROW_MIN_N or N > _ILUVATAR_MULTIROW_MAX_N:
        return False

    if N <= 128:
        return M >= _ILUVATAR_MULTIROW_M_THRESHOLD_N128

    return M >= _ILUVATAR_MULTIROW_M_THRESHOLD_N256


def _select_iluvatar_multirow_config(M, N):
    """Return (TILE_M, num_warps) for the initial BI-V150 fast path.

    These are intentionally small, conservative configurations.  They are
    meant to establish the benefit of multi-row processing first; once the
    benchmark confirms the direction, the values can be autotuned or refined.
    """
    if N <= 128:
        # The important benchmark families are N=100 with M=25,600 and
        # M=6,553,600.  Eight rows/program amortizes program overhead while
        # keeping TILE_M * BLOCK_SIZE modest.
        return 8, 4

    # N in (128, 256].  Four rows/program keeps the per-program working set
    # similar to the N<=128 case.
    return 4, 4


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _fused_add_rms_norm_multirow_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    in_stride_r,
    in_stride_c,
    r_stride_r,
    r_stride_c,
    M,
    eps,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
):
    """BI-V150 fast path: one Triton program processes TILE_M rows."""

    pid_m = tl.program_id(0).to(tl.int64)

    rows = pid_m * TILE_M + tl.arange(0, TILE_M)
    cols = tl.arange(0, BLOCK_SIZE)

    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    input_offsets = rows[:, None] * in_stride_r + cols[None, :] * in_stride_c
    residual_offsets = rows[:, None] * r_stride_r + cols[None, :] * r_stride_c

    x = tl.load(
        input_ptr + input_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    residual = tl.load(
        residual_ptr + residual_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    added = x + residual

    # fused_add_rms_norm is in-place:
    # residual <- input + residual
    tl.store(
        residual_ptr + residual_offsets,
        added,
        mask=mask,
    )

    sum_sq = tl.sum(
        added * added,
        axis=1,
    )
    rrms = tl.rsqrt(sum_sq / N + eps)

    # Weight is shared by all TILE_M rows handled by this program.
    weight = tl.load(
        w_ptr + cols,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    output = added * rrms[:, None] * weight[None, :]

    # input <- RMSNorm(input + residual)
    tl.store(
        input_ptr + input_offsets,
        output,
        mask=mask,
    )


# -----------------------------------------------------------------------------
# Generic whole-row fallback
# -----------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _fused_add_rms_norm_whole_row_kernel(
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

    tl.store(
        residual_ptr + cols * r_stride_c,
        added,
        mask=mask,
    )

    variance = tl.sum(added * added / N, axis=0)
    rrms = 1.0 / tl.sqrt(variance + eps)

    weight = tl.load(
        w_ptr + cols,
        mask=mask,
        other=0.0,
    )

    output = (added * rrms * weight).to(compute_dtype)

    tl.store(
        input_ptr + cols * in_stride_c,
        output,
        mask=mask,
    )


# -----------------------------------------------------------------------------
# Generic tiled fallback for large hidden sizes
# -----------------------------------------------------------------------------


def _get_tiled_autotune_configs():
    return [
        triton.Config(
            {
                "TILE_SIZE": tile_size,
                "LOOP_NUM_STAGES": num_stages,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for tile_size in (256, 512, 1024, 2048)
        for num_warps in (2, 4, 8)
        for num_stages in (1, 2, 3)
    ]


@libentry()
@triton.autotune(
    configs=_get_tiled_autotune_configs(),
    key=["M", "N"],
    # The kernel modifies both input and residual in-place.  Every autotune
    # candidate must therefore start from the same values.
    restore_value=["input_ptr", "residual_ptr"],
)
@triton.jit(do_not_specialize=["eps"])
def _fused_add_rms_norm_tiled_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    in_stride_r,
    in_stride_c,
    r_stride_r,
    r_stride_c,
    M,
    N,
    eps,
    TILE_SIZE: tl.constexpr,
    LOOP_NUM_STAGES: tl.constexpr,
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

    # Pass 1:
    #   residual <- input + residual
    #   accumulate sum of squares.
    for tile_start in tl.range(
        0,
        N,
        TILE_SIZE,
        num_stages=LOOP_NUM_STAGES,
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

        tl.store(
            residual_row + cols * r_stride_c,
            added,
            mask=mask,
        )

    variance = sum_sq / N
    rrms = 1.0 / tl.sqrt(variance + eps)

    # Pass 2:
    #   read the updated residual and write normalized output to input.
    for tile_start in tl.range(
        0,
        N,
        TILE_SIZE,
        num_stages=LOOP_NUM_STAGES,
    ):
        cols = tile_start + offsets
        mask = cols < N

        added = tl.load(
            residual_row + cols * r_stride_c,
            mask=mask,
            other=0.0,
        ).to(compute_dtype)

        weight = tl.load(
            w_ptr + cols,
            mask=mask,
            other=0.0,
        )

        output = (added * rrms * weight).to(compute_dtype)

        tl.store(
            input_row + cols * in_stride_c,
            output,
            mask=mask,
        )


def _prefer_whole_row(N):
    if N <= _WHOLE_ROW_SAFE_N:
        return True

    if N < 24576:
        return False

    if N <= _LARGE_N_THRESHOLD:
        return True

    return False


# -----------------------------------------------------------------------------
# Public backend implementation
# -----------------------------------------------------------------------------


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """Iluvatar fused residual addition + RMSNorm, in-place.

    Dispatch:
      1. FP16/BF16 + narrow hidden size + sufficiently many rows
           -> BI-V150 multi-row fast path.

      2. Other shapes with a safe whole-row hidden size
           -> generic whole-row path.

      3. Large hidden sizes where next_power_of_2 would be expensive
           -> generic tiled two-pass path.

    The function preserves the standard fused_add_rms_norm semantics:
      residual <- x + residual
      x        <- RMSNorm(residual) * weight

    Returns:
      (x, residual)
    """
    logger.debug(
        "ILUVATAR FUSED_ADD_RMS_NORM FORWARD, "
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

    with torch_device_fn.device(x.device):
        if _should_use_iluvatar_multirow(x, M, N):
            block_size = triton.next_power_of_2(N)
            tile_m, num_warps = _select_iluvatar_multirow_config(M, N)

            grid = (triton.cdiv(M, tile_m),)

            _fused_add_rms_norm_multirow_kernel[grid](
                x,
                residual,
                weight,
                N,
                1,
                N,
                1,
                M,
                eps,
                N=N,
                BLOCK_SIZE=block_size,
                TILE_M=tile_m,
                num_warps=num_warps,
                num_stages=1,
            )

        elif _prefer_whole_row(N):
            block_size = triton.next_power_of_2(N)

            _fused_add_rms_norm_whole_row_kernel[(M,)](
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
            _fused_add_rms_norm_tiled_kernel[(M,)](
                x,
                residual,
                weight,
                N,
                1,
                N,
                1,
                M,
                N,
                eps,
            )

    return x, residual


__all__ = ["fused_add_rms_norm"]
