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
import triton
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8 as vllm_per_token_quant_int8,
)

import flaggems_vllm

from . import base


class PerTokenQuantInt8Benchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (1, 128),
        (1, 4096),
        (7, 513),
        (64, 1024),
        (512, 4096),
        (2048, 5120),
        (2048, 8192),
        (4096, 13824),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = "M, N"

    def get_latency(self, op, *args, **kwargs):
        if (
            base.Config.mode == base.consts.BenchMode.CUDAGRAPH
            and base.Config.warm_up > 0
        ):
            # The shared CUDA Graph runner does not consume --warmup.
            triton.testing.do_bench_cudagraph(
                lambda: op(*args, **kwargs), rep=base.Config.warm_up
            )
        return super().get_latency(op, *args, **kwargs)


def _input_fn(shape, dtype, device):
    yield (torch.randn(shape, dtype=dtype, device=device),)


@pytest.mark.per_token_quant_int8
def test_per_token_quant_int8():
    bench = PerTokenQuantInt8Benchmark(
        op_name="per_token_quant_int8",
        torch_op=vllm_per_token_quant_int8,
        input_fn=_input_fn,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    )
    bench.set_gems(flaggems_vllm.per_token_quant_int8)
    bench.run()
