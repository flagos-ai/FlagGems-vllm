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

import pytest
import torch
from vllm._custom_ops import scaled_int8_quant as vllm_scaled_int8_quant

import flaggems_vllm

from . import base

NUM_TOKENS = [1, 64, 512, 2048, 4096]
HIDDEN_SIZES = [512, 1024, 4096, 5120, 8192, 13824]
SHAPES = [(m, n) for m in NUM_TOKENS for n in HIDDEN_SIZES]


class ScaledInt8QuantBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES
        self.shape_desc = "M, N"

    def set_more_shapes(self):
        return []


def _input_fn_dynamic(shape, dtype, device):
    m, n = shape
    x = torch.randn(m, n, dtype=dtype, device=device) * 1000
    yield (x,)


def _input_fn_static(shape, dtype, device):
    m, n = shape
    x = torch.randn(m, n, dtype=dtype, device=device) * 1000
    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    yield (x, scale)


def _vllm_dynamic_symmetric(x):
    return vllm_scaled_int8_quant(x, symmetric=True)


def _vllm_static_symmetric(x, scale):
    return vllm_scaled_int8_quant(x, scale=scale, symmetric=True)


def _gems_dynamic_symmetric(x):
    return flaggems_vllm.scaled_int8_quant(x, symmetric=True)


def _gems_static_symmetric(x, scale):
    return flaggems_vllm.scaled_int8_quant(x, scale=scale, symmetric=True)


@pytest.mark.scaled_int8_quant
def test_dynamic_scaled_int8_quant():
    bench = ScaledInt8QuantBenchmark(
        op_name="dynamic_scaled_int8_quant",
        torch_op=_vllm_dynamic_symmetric,
        input_fn=_input_fn_dynamic,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_dynamic_symmetric)
    bench.run()


@pytest.mark.scaled_int8_quant
def test_static_scaled_int8_quant():
    bench = ScaledInt8QuantBenchmark(
        op_name="static_scaled_int8_quant",
        torch_op=_vllm_static_symmetric,
        input_fn=_input_fn_static,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_static_symmetric)
    bench.run()
