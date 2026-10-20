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
from flaggems_vllm.ops._memory_overlap import tensor_ranges_overlap
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tuned constants
# -----------------------------------------------------------------------------
_WHOLE_ROW_SAFE_N = 16384
_LARGE_N_THRESHOLD = 32768


# -----------------------------------------------------------------------------
# 1) Original whole-row baseline
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
    GEMMA: tl.constexpr = False,
    GEMMA_N: tl.constexpr = 0,
    GEMMA_EPS: tl.constexpr = 0.0,
):
    if GEMMA:
        N = GEMMA_N
        eps = GEMMA_EPS
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = ext.program_id(0)
    input_ptr += pid * in_stride_r
    residual_ptr += pid * r_stride_r

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(
        input_ptr + cols * in_stride_c,
        mask=mask,
        other=0.0,
    ).to(cdtype)

    r = tl.load(
        residual_ptr + cols * r_stride_c,
        mask=mask,
        other=0.0,
    ).to(cdtype)

    x += r

    tl.store(
        residual_ptr + cols * r_stride_c,
        x,
        mask=mask,
    )

    if GEMMA:
        var = tl.sum(x * x, axis=0) / N
        rrms = tl.rsqrt(var + eps)
    else:
        var = tl.sum(x * x / N, axis=0)
        rrms = 1.0 / tl.sqrt(var + eps)

    w = tl.load(
        w_ptr + cols,
        mask=mask,
        other=0.0,
    )

    if GEMMA:
        y = (x * rrms) * (1.0 + w.to(tl.float32))
    else:
        y = (x * rrms * w).to(cdtype)

    tl.store(
        input_ptr + cols * in_stride_c,
        y,
        mask=mask,
    )


# -----------------------------------------------------------------------------
# 2) Single-CTA tiled two-pass path with Triton autotune
# -----------------------------------------------------------------------------
def _get_tiled_autotune_configs():
    # Tune TILE_SIZE, num_warps and num_stages jointly.
    #
    # LOOP_NUM_STAGES mirrors Config.num_stages into tl.range(...).  This is
    # intentional: tl.range(num_stages=...) controls software pipelining for
    # loads in this non-dot reduction loop, while Config.num_stages is also
    # supplied as the normal Triton compilation option.
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
    # This kernel updates input_ptr and residual_ptr in-place.  Autotune runs
    # every candidate multiple times, so restore the original tensors before
    # benchmarking each candidate to keep all configs semantically equivalent.
    restore_value=["input_ptr", "residual_ptr"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_tiled_kernel(
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

    # Pass 1: compute sum of squares and immediately materialize
    # added = input + residual into residual.  This removes the second-pass
    # input load and the repeated addition for all supported dtypes.
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

    # Pass 2: read the already-updated residual and produce only the output.
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
    """Empirical whole-row selector without the removed Cluster paths."""
    if N <= _WHOLE_ROW_SAFE_N:
        return True

    # Just above 16384 is a severe next_power_of_2 cliff.
    if N < 24576:
        return False

    # BLOCK_SIZE=32768 remains efficient in this region.
    if N <= _LARGE_N_THRESHOLD:
        return True

    # Once next_power_of_2(N) is 65536, use the tiled path.
    return False


def _gemma_residual(x, residual, normalized_shape, weight, eps):
    if x.device.type != "cuda" or x.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise NotImplementedError("Gemma residual mode requires CUDA BF16/FP16/FP32")
    if len(normalized_shape) != 1 or x.shape[-1] != normalized_shape[0]:
        raise ValueError("Gemma residual mode normalizes the last dimension")
    n = x.shape[-1]
    if n <= 0 or triton.next_power_of_2(n) > _WHOLE_ROW_SAFE_N:
        raise NotImplementedError("Gemma residual mode requires 1 <= N <= 16384")
    if (
        residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
    ):
        raise ValueError("Residual must match input")
    if weight.shape != (n,) or weight.device != x.device or not weight.is_contiguous():
        raise ValueError("Invalid Gemma weight")
    if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise NotImplementedError("Unsupported Gemma weight dtype")
    if not math.isfinite(float(eps)) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")

    def matrix(t):
        if t.ndim == 2 and t.stride(-1) == 1:
            return t
        if t.is_contiguous():
            return t.view(-1, n)
        raise NotImplementedError(
            "Gemma residual requires viewable rows and unit column stride"
        )

    a, r = matrix(x), matrix(residual)
    if not a.numel():
        return x, residual
    if tensor_ranges_overlap(x, residual):
        raise ValueError("Input and residual must not alias")
    with torch_device_fn.device(x.device):
        fused_add_rms_norm_kernel[(a.shape[0],)](
            a,
            r,
            weight,
            a.stride(0),
            1,
            r.stride(0),
            1,
            n,
            eps,
            BLOCK_SIZE=triton.next_power_of_2(n),
            GEMMA=True,
            GEMMA_N=n,
            GEMMA_EPS=float(eps),
            num_warps=4,
        )
    return x, residual


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5, *, gemma=False):
    """Fused residual addition + RMSNorm, in-place.

    Dispatch:
      A) N <= 16384
           -> whole-row baseline.

      B) 16384 < N < 24576
           -> Single-CTA tiled.

      C) 24576 <= N <= 32768
           -> whole-row baseline.

      D) N > 32768
           -> Single-CTA tiled.
    """
    if not isinstance(gemma, bool):
        raise TypeError("gemma must be a bool")
    if gemma:
        return _gemma_residual(x, residual, normalized_shape, weight, eps)
    logger.debug(
        "GEMS FUSED_ADD_RMS_NORM FORWARD, "
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

    prefer_whole = _prefer_whole_row(N)

    with torch_device_fn.device(x.device):
        if prefer_whole:
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
                M,
                N,
                eps,
            )

    return x, residual
