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

import random

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


class PerTokenGroupQuantFp8Benchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _input_fn(shape, dtype, device):
    num_tokens, d, group_size = shape
    scale_ue8m0 = random.choice([True, False])
    x = torch.rand(num_tokens, d, dtype=dtype, device=device)

    yield (x, group_size, scale_ue8m0)


@pytest.mark.per_token_group_quant_fp8
@pytest.mark.skipif(
    not (HAS_VLLM_PER_TOKEN_GROUP_QUANT_FP8),
    reason="requires vLLM",
)
def test_per_token_group_quant_fp8():
    bench = PerTokenGroupQuantFp8Benchmark(
        op_name="per_token_group_quant_fp8",
        input_fn=_input_fn,
        torch_op=vllm_per_token_group_quant_fp8,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flaggems_vllm.per_token_group_quant_fp8)
    bench.run()
