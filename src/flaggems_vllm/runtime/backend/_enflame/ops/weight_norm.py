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

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._enflame.ops.weightnorm import (
    weight_norm_interface,
    weight_norm_interface_backward,
)
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_norm_kernel"),
    key=["v_shape0", "v_shape1", "v_shape2"],
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_except_dim_kernel(
    output,
    norm,
    v,
    g,
    v_shape0,
    v_shape1,
    v_shape2,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    # GCU workaround: 2D tile load/store hits the broken memref_ext.slice_*
    # lowering (compiler_issues/vllm_moe_sum.md); use 1D affine access only.
    pid = ext.program_id(axis=0)
    row_base = pid * BLOCK_ROW_SIZE
    total = v_shape0 * v_shape2
    for ri in tl.static_range(0, BLOCK_ROW_SIZE):
        row = row_base + ri
        if row < v_shape1:
            vacc = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)
            for base in range(0, total, BLOCK_COL_SIZE):
                flat = base + tl.arange(0, BLOCK_COL_SIZE)
                flat_mask = flat < total
                m_idx = flat // v_shape2
                k_idx = flat - m_idx * v_shape2
                v_offsets = m_idx * (v_shape1 * v_shape2) + row * v_shape2 + k_idx
                v_value = tl.load(v + v_offsets, mask=flat_mask, other=0.0).to(
                    tl.float32
                )
                vacc += v_value * v_value
            normalized = tl.sqrt(tl.sum(vacc) + eps)
            tl.store(norm + row, normalized)
            g_value = tl.load(g + row).to(tl.float32)
            scale = g_value / normalized
            for base in range(0, total, BLOCK_COL_SIZE):
                flat = base + tl.arange(0, BLOCK_COL_SIZE)
                flat_mask = flat < total
                m_idx = flat // v_shape2
                k_idx = flat - m_idx * v_shape2
                v_offsets = m_idx * (v_shape1 * v_shape2) + row * v_shape2 + k_idx
                v_value = tl.load(v + v_offsets, mask=flat_mask, other=0.0).to(
                    tl.float32
                )
                tl.store(output + v_offsets, v_value * scale, mask=flat_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_norm_kernel"),
    key=["v_shape0", "v_shape1", "v_shape2"],
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_except_dim_bwd_kernel(
    v_grad,
    g_grad,
    grad,
    v,
    g,
    norm,
    v_shape0,
    v_shape1,
    v_shape2,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    # GCU workaround: 1D affine access only (see weight_norm_except_dim_kernel).
    pid = ext.program_id(axis=0)
    row_base = pid * BLOCK_ROW_SIZE
    total = v_shape0 * v_shape2
    for ri in tl.static_range(0, BLOCK_ROW_SIZE):
        row = row_base + ri
        if row < v_shape1:
            g_value = tl.load(g + row).to(tl.float32)
            norm_value = tl.load(norm + row).to(tl.float32)

            # norm_value is already sqrt(sum(v^2) + eps) from forward,
            # safe to divide directly
            norm_1 = 1 / norm_value
            norm_cu = norm_value * norm_value * norm_value

            vw = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)
            for base in range(0, total, BLOCK_COL_SIZE):
                flat = base + tl.arange(0, BLOCK_COL_SIZE)
                flat_mask = flat < total
                m_idx = flat // v_shape2
                k_idx = flat - m_idx * v_shape2
                v_offsets = m_idx * (v_shape1 * v_shape2) + row * v_shape2 + k_idx
                v_value = tl.load(v + v_offsets, mask=flat_mask, other=0.0).to(
                    tl.float32
                )
                grad_value = tl.load(grad + v_offsets, mask=flat_mask, other=0.0).to(
                    tl.float32
                )
                vw += v_value * grad_value
            vw_sum = tl.sum(vw)

            for base in range(0, total, BLOCK_COL_SIZE):
                flat = base + tl.arange(0, BLOCK_COL_SIZE)
                flat_mask = flat < total
                m_idx = flat // v_shape2
                k_idx = flat - m_idx * v_shape2
                v_offsets = m_idx * (v_shape1 * v_shape2) + row * v_shape2 + k_idx
                v_value = tl.load(v + v_offsets, mask=flat_mask, other=0.0).to(
                    tl.float32
                )
                grad_value = tl.load(grad + v_offsets, mask=flat_mask, other=0.0).to(
                    tl.float32
                )
                # v_grad = g * (grad / norm - v * <v, grad> / norm^3)
                v_grad_value = g_value * (
                    grad_value * norm_1 - v_value * vw_sum / norm_cu
                )
                tl.store(v_grad + v_offsets, v_grad_value, mask=flat_mask)

            # g_grad = <v, grad> / norm
            g_grad_value = vw_sum * norm_1
            tl.store(g_grad + row, g_grad_value)


def weight_norm_except_dim(v, g, dim):
    logger.debug("GEMS WEIGHT NORM EXCEPT DIM FORWARD")
    v = v.contiguous()
    output = torch.empty_like(v)
    norm = torch.empty_like(g, dtype=torch.float32)
    v_shape = [
        math.prod(v.shape[:dim]),
        v.shape[dim],
        math.prod(v.shape[dim + 1 :]),
    ]

    grid = lambda META: (triton.cdiv(v_shape[1], META["BLOCK_ROW_SIZE"]),)

    with torch_device_fn.device(v.device):
        weight_norm_except_dim_kernel[grid](
            output,
            norm,
            v,
            g,
            v_shape[0],
            v_shape[1],
            v_shape[2],
            eps=torch.finfo(torch.float32).tiny,
        )
    return output, norm


def weight_norm_except_dim_backward(grad, v, g, norm, dim):
    logger.debug("GEMS WEIGHT NORM EXCEPT DIM BACKWARD")
    grad = grad.contiguous()
    v_grad = torch.empty_like(v)
    g_grad = torch.empty_like(g)
    v_shape = [
        math.prod(v.shape[:dim]),
        v.shape[dim],
        math.prod(v.shape[dim + 1 :]),
    ]

    grid = lambda META: (triton.cdiv(v_shape[1], META["BLOCK_ROW_SIZE"]),)
    with torch_device_fn.device(v.device):
        weight_norm_except_dim_bwd_kernel[grid](
            v_grad,
            g_grad,
            grad,
            v,
            g,
            norm,
            *v_shape,
            eps=torch.finfo(torch.float32).tiny,
        )
    return v_grad, g_grad


class WeightNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, g, dim=0):
        logger.debug("GEMS WEIGHT NORM")
        dim = dim % v.ndim
        can_use_fused = dim == 0 or dim == v.ndim - 1
        if can_use_fused:
            output, norm = weight_norm_interface(v, g, dim)
        else:
            output, norm = weight_norm_except_dim(v, g, dim)
        ctx.save_for_backward(v, g, norm)
        ctx.dim = dim
        ctx.can_use_fused = can_use_fused
        return output

    @staticmethod
    def backward(ctx, grad):
        logger.debug("GEMS WEIGHT NORM BACKWARD")
        v, g, norm = ctx.saved_tensors
        dim = ctx.dim
        if ctx.can_use_fused:
            v_grad, g_grad = weight_norm_interface_backward(grad, v, g, norm, dim)
        else:
            v_grad, g_grad = weight_norm_except_dim_backward(grad, v, g, norm, dim)
        return v_grad, g_grad, None


def weight_norm(v, g, dim=0):
    return WeightNorm.apply(v, g, dim)
