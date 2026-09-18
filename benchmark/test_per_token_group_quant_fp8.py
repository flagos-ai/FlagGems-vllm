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

import inspect
import os
from functools import partial

import pytest
import torch
import triton
import triton.language as tl

import flaggems_vllm

from . import base

try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8 as vllm_per_token_group_quant_fp8,
    )

    HAS_VLLM_PER_TOKEN_GROUP_QUANT_FP8 = True
except ImportError:
    HAS_VLLM_PER_TOKEN_GROUP_QUANT_FP8 = False


def _supports_keyword(op, keyword):
    try:
        parameters = inspect.signature(op).parameters
    except (TypeError, ValueError):
        return False
    return keyword in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


VLLM_SUPPORTS_UE8M0 = HAS_VLLM_PER_TOKEN_GROUP_QUANT_FP8 and _supports_keyword(
    vllm_per_token_group_quant_fp8, "use_ue8m0"
)

IS_ASCEND = flaggems_vllm.vendor_name == "ascend"


# ---------------------------------------------------------------------------
# Ascend baseline
#
# Copied from vLLM (vllm/model_executor/layers/quantization/utils/fp8_utils.py,
# kernel `_per_token_group_quant_fp8` and the Triton-fallback branch of
# `per_token_group_quant_fp8`). On Ascend vLLM cannot use its CUDA/XPU kernel
# and always dispatches to this Triton path, so the copied prototype below is
# used as the performance baseline on the Ascend backend. Other backends keep
# calling the installed vLLM op directly.
#
# Note: the Ascend Triton backend cannot cast to `torch.float8_e4m3fn` inside
# a kernel ("unrecognized float type: 'f8E4M3FN'"), so the quantized values
# are stored as float32. The quantization math, including the fp8 clamp
# range, is unchanged.
# ---------------------------------------------------------------------------
@triton.jit
def ascend_per_token_group_quant_fp8_kernel(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    # Use multiply-by-reciprocal instead of division to match PyTorch's
    # tensor/scalar division precision (GPU fast-division for constexpr
    # divisors can introduce 1-ULP error that flips FP8 quantization at
    # representable-value boundaries).
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def ascend_triton_per_token_group_quant_fp8(
    x,
    group_size,
    scale_ue8m0,
    eps=1e-10,
):
    """Host-side replica of vLLM's Triton-fallback dispatch (row-major scales)."""
    # fp8_min/fp8_max mirror vLLM's get_fp8_min_max(); the output is stored as
    # float32 because the Ascend Triton backend cannot cast to fp8 in-kernel.
    dtype = torch.float32
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_min, fp8_max = finfo.min, finfo.max

    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    x_q = torch.empty(x.shape, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    ascend_per_token_group_quant_fp8_kernel[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        x.shape[1],
        x.stride(0),
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        use_ue8m0=scale_ue8m0,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s


# Safe fallback for custom shape files that do not define this benchmark.
# Keep it aligned with PerTokenGroupQuantFp8Benchmark in core_shapes.yaml.
CORE_SHAPES = [
    (7, 512, 512),
    (7, 4096, 256),
    (83, 512, 64),
    (2048, 4096, 256),
    (2048, 13824, 512),
]


class PerTokenGroupQuantFp8Benchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = CORE_SHAPES
    DEFAULT_SHAPE_DESC = "num_tokens, d, group_size"

    def set_shapes(self, shape_file_path=None):
        # Benchmark.init_default_config() supplies a relative path, which is
        # resolved from the caller's cwd rather than this package directory.
        if shape_file_path is None or shape_file_path == self.DEFAULT_SHAPE_FILES:
            shape_file_path = os.path.join(
                os.path.dirname(__file__), self.DEFAULT_SHAPE_FILES
            )
        super().set_shapes(shape_file_path)

    def set_more_shapes(self):
        return []


def _input_fn(shape, dtype, device, scale_ue8m0):
    num_tokens, d, group_size = shape
    x = torch.rand(num_tokens, d, dtype=dtype, device=device)

    yield (x, group_size, scale_ue8m0)


def _vllm_per_token_group_quant_fp8_wrapper(x, group_size, scale_ue8m0):
    if VLLM_SUPPORTS_UE8M0:
        return vllm_per_token_group_quant_fp8(x, group_size, use_ue8m0=scale_ue8m0)
    if scale_ue8m0:
        raise RuntimeError("installed vLLM does not support use_ue8m0")
    return vllm_per_token_group_quant_fp8(x, group_size)


def _gems_per_token_group_quant_fp8_wrapper(x, group_size, scale_ue8m0):
    return flaggems_vllm.per_token_group_quant_fp8(
        x, group_size, scale_ue8m0=scale_ue8m0
    )


@pytest.mark.per_token_group_quant_fp8
@pytest.mark.skipif(
    not (HAS_VLLM_PER_TOKEN_GROUP_QUANT_FP8),
    reason="requires vLLM",
)
@pytest.mark.parametrize(
    "scale_ue8m0", [False, True], ids=["standard_scale", "ue8m0_scale"]
)
def test_per_token_group_quant_fp8(scale_ue8m0):
    if scale_ue8m0 and not IS_ASCEND and not VLLM_SUPPORTS_UE8M0:
        pytest.skip("installed vLLM does not support use_ue8m0")

    # On Ascend the baseline is the copied vLLM Triton prototype above;
    # other backends keep using the installed vLLM op.
    torch_op = (
        ascend_triton_per_token_group_quant_fp8
        if IS_ASCEND
        else _vllm_per_token_group_quant_fp8_wrapper
    )

    bench = PerTokenGroupQuantFp8Benchmark(
        op_name="per_token_group_quant_fp8",
        input_fn=partial(_input_fn, scale_ue8m0=scale_ue8m0),
        torch_op=torch_op,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_per_token_group_quant_fp8_wrapper)
    bench.run()
