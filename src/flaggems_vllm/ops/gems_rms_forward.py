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

# Forward kernels adapted from FlagGems v5.4.0 (6ed2f390).
import math
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils import triton_lang_extension as ext


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@libtuner(configs=runtime.get_tuned_config("gems_rms_forward"), key=["N"])
@triton.jit(do_not_specialize=["eps"])
def _rms_norm_kernel(
    out_ptr,  # pointer to the output
    in_ptr,  # pointer to the input
    w_ptr,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tl.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + cols * x_stride_c, mask, other=0.0).to(cdtype)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    # Cast x_normed back to input dtype before multiplying with weight
    # to align with vLLM native: x.to(weight.dtype) * weight
    x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
    y = x_normed * w
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("gems_rms_forward_loop"), key=["N"])
@triton.jit(do_not_specialize=["eps"])
def _rms_norm_loop_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = ext.program_id(0)

    # Pass 1: compute sum(x^2) in chunks
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)

    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        acc += x * x

    # last step with mask
    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    acc += x * x

    var = tl.sum(acc) / N
    rrms = 1 / tl.sqrt(var + eps)

    # Pass 2: normalize in reverse order (better L2 cache reuse)
    prev_multiple = prev_multiple_of(N, TILE_N)

    # first reverse step with mask
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0)
        # Cast x_normed back to input dtype before multiplying with weight
        x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
        y = x_normed * w
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets)
        # Cast x_normed back to input dtype before multiplying with weight
        x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
        y = x_normed * w
        tl.store(out_ptr + pid * N + n_offsets, y)


# FlagGems v5.4.0 uses different residual rounding on CUDA and Ascend.
@libentry()
@triton.jit(do_not_specialize=["eps"])
def _cuda_residual_kernel(x_ptr, residual_ptr, w_ptr, N, eps, BLOCK_SIZE: tl.constexpr):
    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty
    row_start = ext.program_id(0) * N
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offsets = row_start + cols
    x = tl.load(x_ptr + offsets, mask, other=0.0).to(cdtype)
    residual = tl.load(residual_ptr + offsets, mask, other=0.0).to(cdtype)
    summed = x + residual
    tl.store(residual_ptr + offsets, summed, mask)

    var = tl.sum(summed * summed / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)
    weight = tl.load(w_ptr + cols, mask, other=0.0)
    tl.store(x_ptr + offsets, (summed * rrms * weight).to(cdtype), mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _cuda_residual_loop_kernel(
    x_ptr, residual_ptr, w_ptr, N, eps, BLOCK_SIZE: tl.constexpr
):
    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty
    row_start = ext.program_id(0) * N
    var_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    num_steps = tl.cdiv(N, BLOCK_SIZE)
    lanes = tl.arange(0, BLOCK_SIZE)

    for step in range(num_steps):
        cols = step * BLOCK_SIZE + lanes
        mask = cols < N
        offsets = row_start + cols
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(cdtype)
        residual = tl.load(residual_ptr + offsets, mask, other=0.0).to(cdtype)
        summed = x + residual
        tl.store(residual_ptr + offsets, summed, mask)
        var_acc += summed * summed

    rrms = 1 / tl.sqrt(tl.sum(var_acc) / N + eps)

    for step in range(num_steps):
        cols = step * BLOCK_SIZE + lanes
        mask = cols < N
        offsets = row_start + cols
        # The original CUDA loop normalizes the rounded residual.
        summed = tl.load(residual_ptr + offsets, mask, other=0.0).to(cdtype)
        weight = tl.load(w_ptr + cols, mask, other=0.0)
        tl.store(x_ptr + offsets, (summed * rrms * weight).to(cdtype), mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _ascend_residual_kernel(
    x_ptr, residual_ptr, w_ptr, N, eps, row_offset, BLOCK_SIZE: tl.constexpr
):
    row_start = (tl.program_id(0) + row_offset) * N
    var_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    lanes = tl.arange(0, BLOCK_SIZE)

    for start in range(0, N, BLOCK_SIZE):
        cols = start + lanes
        mask = cols < N
        offsets = row_start + cols
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask, other=0.0).to(tl.float32)
        summed = x + residual
        var_acc += summed * summed / N

    rrms = 1 / tl.sqrt(tl.sum(var_acc) + eps)

    for start in range(0, N, BLOCK_SIZE):
        cols = start + lanes
        mask = cols < N
        offsets = row_start + cols
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask, other=0.0).to(tl.float32)
        summed = x + residual
        weight = tl.load(w_ptr + cols, mask, other=0.0)
        # The original Ascend kernel rounds the normalized value before scaling.
        out = (summed * rrms).to(x_ptr.dtype.element_ty) * weight
        tl.store(residual_ptr + offsets, summed, mask)
        tl.store(x_ptr + offsets, out, mask)


def _gems_rms_forward_residual(x, residual, weight, eps):
    if runtime.device.vendor_name not in ("nvidia", "ascend"):
        from flaggems_vllm import fused_add_rms_norm

        return fused_add_rms_norm(x, residual, list(weight.shape), weight, eps)

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    if not x.numel():
        return x, residual

    n = weight.numel()
    m = math.prod(x.shape[: x.ndim - weight.ndim])
    with torch_device_fn.device(x.device):
        if runtime.device.vendor_name == "ascend":
            block_size = min(triton.next_power_of_2(n), 8192)
            # Ascend limits the grid size. Launch each group of rows.
            max_grid = 65535
            for row_offset in range(0, m, max_grid):
                grid = (min(m - row_offset, max_grid),)
                # The 8192-element tile can exhaust unified buffer with multibuffering.
                _ascend_residual_kernel[grid](
                    x,
                    residual,
                    weight,
                    n,
                    eps,
                    row_offset,
                    BLOCK_SIZE=block_size,
                    multibuffer=False,
                )
        elif n <= 4096:
            _cuda_residual_kernel[(m,)](
                x, residual, weight, n, eps, BLOCK_SIZE=triton.next_power_of_2(n)
            )
        else:
            _cuda_residual_loop_kernel[(m,)](
                x, residual, weight, n, eps, BLOCK_SIZE=1024
            )
    return x, residual


def gems_rms_forward(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run the FlagGems RMSNorm inference interface.

    Without a residual, return a new tensor. With a residual, contiguous input
    buffers are updated in place and the pair of updated buffers is returned.
    The fused path applies its existing layout rules to noncontiguous inputs.
    """
    if residual is not None:
        return _gems_rms_forward_residual(x, residual, weight, eps)

    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(x)
    if not out.numel():
        return out
    n = weight.numel()
    m = math.prod(x.shape[: x.ndim - weight.ndim])
    with torch_device_fn.device(x.device):
        if n <= 4096:
            _rms_norm_kernel[(m,)](
                out, x, weight, n, 1, n, 1, n, eps, BLOCK_SIZE=triton.next_power_of_2(n)
            )
        else:
            _rms_norm_loop_kernel[(m,)](out, x, weight, n, eps)
    return out
