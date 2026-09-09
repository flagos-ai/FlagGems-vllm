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
    if scale_ue8m0 and not VLLM_SUPPORTS_UE8M0:
        pytest.skip("installed vLLM does not support use_ue8m0")

    bench = PerTokenGroupQuantFp8Benchmark(
        op_name="per_token_group_quant_fp8",
        input_fn=partial(_input_fn, scale_ue8m0=scale_ue8m0),
        torch_op=_vllm_per_token_group_quant_fp8_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_per_token_group_quant_fp8_wrapper)
    bench.run()
