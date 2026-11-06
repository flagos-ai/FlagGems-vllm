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

"""The canonical Qwen4 grouped Gemma RMSNorm forward interface."""

import logging
from numbers import Integral, Real

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_ACTIVE_DEVICE_TYPE = runtime.device.name
_ACTIVE_ACCELERATOR = _ACTIVE_DEVICE_TYPE not in ("cpu", "meta")
_IS_METAX = runtime.device.vendor_name == "metax"

# Bound live vectors, rather than rejecting larger groups. Other backends start
# with a smaller tile and a single conservative launch, not NVIDIA's candidates.
_MAX_BLOCK_SIZE = 16384 if runtime.device.vendor_name == "nvidia" else 1024
_METAX_FP32_PREFILL_MIN_ROWS = 128
_METAX_FP32_PREFILL_MIN_GROUP_DIM = 2048
_METAX_FP32_PREFILL_MAX_BLOCK_SIZE = 8192
_CONFIGS = (
    runtime.get_tuned_config("grouped_gemma_rmsnorm")
    if runtime.device.vendor_name == "nvidia"
    else [triton.Config({}, num_warps=4, num_stages=1)]
)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


# SPEC contains (N, DIM, NUM_GROUPS, W_SHARED, ROW_GAPPED, EPS, BLOCK_SIZE,
# stride_x). The compact signature reduces LibEntry's per-call parameter work.
# repr preserves all tuple fields in the SQL-backed tuner's scalar key column;
# LibTuner still appends each pointer dtype independently to the tuning key.
@libentry()
@libtuner(
    configs=_CONFIGS,
    key=["SPEC"],
    strategy=[repr],
    # Explicit warmup/rep uses the non-Graph timer on Triton 3.2+; older
    # versions receive use_cuda_graph=False directly.
    warmup=5,
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _grouped_gemma_rmsnorm_kernel(X, W, Y, SPEC: tl.constexpr):
    stride_x: tl.constexpr = SPEC[7]
    DIM: tl.constexpr = SPEC[1]
    NUM_GROUPS: tl.constexpr = SPEC[2]
    W_SHARED: tl.constexpr = SPEC[3]
    EPS: tl.constexpr = SPEC[5]
    BLOCK_SIZE: tl.constexpr = SPEC[6]
    GROUP_DIM: tl.constexpr = DIM // NUM_GROUPS
    pid = tl.program_id(0).to(tl.int64)
    row = pid // NUM_GROUPS
    group = pid % NUM_GROUPS
    group_offset = group * GROUP_DIM
    weight_offset = 0 if W_SHARED else group_offset
    offsets = tl.arange(0, BLOCK_SIZE)

    if GROUP_DIM <= BLOCK_SIZE:
        mask = offsets < GROUP_DIM
        values = tl.load(
            X + row * stride_x + group_offset + offsets, mask, other=0.0
        ).to(tl.float32)
        weight = tl.load(W + weight_offset + offsets, mask, other=0.0).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(values * values, axis=0) / GROUP_DIM + EPS)
        normalized = values * rrms
        # Preserve the native z + z * float(w) FMA, including w near -1.
        result = tl.fma(normalized, weight, normalized)
        tl.store(Y + row * DIM + group_offset + offsets, result, mask)
    else:
        # Two passes keep register/shared-memory use independent of GROUP_DIM.
        # Each chunk contributes an FP32 sum, with the true group-size divisor.
        sum_sq = tl.full((), 0.0, tl.float32)
        for start in range(0, GROUP_DIM, BLOCK_SIZE):
            columns = start + offsets
            values = tl.load(
                X + row * stride_x + group_offset + columns,
                columns < GROUP_DIM,
                other=0.0,
            ).to(tl.float32)
            sum_sq += tl.sum(values * values, axis=0)
        rrms = tl.rsqrt(sum_sq / GROUP_DIM + EPS)
        for start in range(0, GROUP_DIM, BLOCK_SIZE):
            columns = start + offsets
            mask = columns < GROUP_DIM
            values = tl.load(
                X + row * stride_x + group_offset + columns, mask, other=0.0
            ).to(tl.float32)
            weight = tl.load(W + weight_offset + columns, mask, other=0.0).to(
                tl.float32
            )
            normalized = values * rrms
            result = tl.fma(normalized, weight, normalized)
            tl.store(Y + row * DIM + group_offset + columns, result, mask)


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    """Normalize each group and apply Gemma's ``z + z * weight`` affine.

    ``x`` is a two-dimensional accelerator tensor with unit inner stride;
    nonnegative row strides, including row gaps, are supported. A contiguous
    ``weight`` has either ``DIM // num_groups`` elements shared across groups,
    or ``DIM`` elements. The tensors independently use FP16, BF16, or FP32.
    Accumulation and affine arithmetic use FP32, then round to ``x.dtype``.

    ``eps`` accepts real scalars, including zero, negative values, NaN, and Inf.
    These values follow the native FP32 sum/rsqrt expression without clamping;
    invalid square roots and nonfinite arithmetic can produce NaN or Inf.
    The result is fresh and contiguous. This is an inference-only forward API;
    CPU/meta tensors and autograd are not supported.
    """
    logger.debug("GEMS GROUPED_GEMMA_RMSNORM")

    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("x and weight must be tensors")
    if x.layout != torch.strided or weight.layout != torch.strided:
        raise NotImplementedError("grouped Gemma RMSNorm requires strided tensors")
    if x.ndim != 2:
        raise ValueError("grouped Gemma RMSNorm requires a two-dimensional x")
    if type(num_groups) is not int:
        if isinstance(num_groups, bool) or not isinstance(num_groups, Integral):
            raise TypeError("num_groups must be an integer")
        num_groups = int(num_groups)
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    n, dim = x.shape
    if dim <= 0 or dim % num_groups:
        raise ValueError("x.shape[1] must be positive and divisible by num_groups")
    row_stride = x.stride(0)
    if x.stride(1) != 1 or row_stride < 0:
        raise ValueError("x requires unit inner stride and nonnegative row stride")
    group_dim = dim // num_groups
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous with group_dim or DIM elements")
    weight_numel = weight.numel()
    if weight_numel not in (group_dim, dim):
        raise ValueError("weight must be contiguous with group_dim or DIM elements")
    supported_dtypes = _SUPPORTED_DTYPES
    x_dtype = x.dtype
    if x_dtype not in supported_dtypes or weight.dtype not in supported_dtypes:
        raise NotImplementedError("x and weight independently require FP16/BF16/FP32")
    x_device = x.device
    if x_device != weight.device:
        raise ValueError("x and weight must be on the same device")
    if not _ACTIVE_ACCELERATOR or x_device.type != _ACTIVE_DEVICE_TYPE:
        raise NotImplementedError(
            "grouped Gemma RMSNorm requires the active accelerator"
        )
    if type(eps) is not float:
        if isinstance(eps, bool) or not isinstance(eps, Real):
            raise TypeError("eps must be a real scalar")
        eps = float(eps)
    output = x.new_empty((n, dim))
    if n == 0:
        return output
    if runtime.device.vendor_name == "ascend":
        from flaggems_vllm.runtime.backend._ascend.ops.grouped_gemma_rmsnorm import (
            _launch_grouped_gemma_rmsnorm,
        )

        _launch_grouped_gemma_rmsnorm(x, weight, output, eps, num_groups)
        return output
    # C550 trials favor larger tiles for FP32 prefill; BF16 did not benefit.
    max_block_size = _MAX_BLOCK_SIZE
    if (
        _IS_METAX
        and x_dtype == torch.float32
        and (n >= _METAX_FP32_PREFILL_MIN_ROWS)
        and (group_dim >= _METAX_FP32_PREFILL_MIN_GROUP_DIM)
    ):
        max_block_size = _METAX_FP32_PREFILL_MAX_BLOCK_SIZE
    block_size = min(triton.next_power_of_2(group_dim), max_block_size)
    if runtime.torch_device_fn.current_device() == x_device.index:
        _grouped_gemma_rmsnorm_kernel.run(
            X=x,
            W=weight,
            Y=output,
            SPEC=(
                n,
                dim,
                num_groups,
                weight_numel == group_dim,
                row_stride != dim,
                eps,
                block_size,
                row_stride,
            ),
            grid=(n * num_groups,),
            warmup=False,
        )
    else:
        with runtime.torch_device_fn.device(x_device.index):
            _grouped_gemma_rmsnorm_kernel.run(
                X=x,
                W=weight,
                Y=output,
                SPEC=(
                    n,
                    dim,
                    num_groups,
                    weight_numel == group_dim,
                    row_stride != dim,
                    eps,
                    block_size,
                    row_stride,
                ),
                grid=(n * num_groups,),
                warmup=False,
            )
    return output
