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

import os

import pytest
import torch
import yaml

import flaggems_vllm

from . import base, consts


class PerTokenQuantInt8Benchmark(base.GenericBenchmark):
    # The token counts and hidden dimensions of PerTokenGroupQuantFp8Benchmark.
    DEFAULT_SHAPES = [(7, 512), (7, 4096), (83, 512), (2048, 4096), (2048, 13824)]
    DEFAULT_SHAPE_DESC = "num_tokens, d"
    DEFAULT_SHAPE_FILES = os.path.join(os.path.dirname(__file__), "core_shapes.yaml")

    def set_shapes(self, shape_file_path=None):
        shape_file_path = shape_file_path or self.DEFAULT_SHAPE_FILES
        with open(shape_file_path) as shape_file:
            config = yaml.safe_load(shape_file)
        if any(key in config for key in (self.op_name, type(self).__name__)):
            return super().set_shapes(shape_file_path)

        # Reuse FP8's input dimensions; INT8 has one scale for the whole row.
        # Avoid the shared Benchmark fallback, which includes unrelated ranks.
        shapes = config.get("PerTokenGroupQuantFp8Benchmark", {}).get(
            "shapes", self.DEFAULT_SHAPES
        )
        self.shapes = [tuple(shape[:2]) for shape in shapes]
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        if (
            base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
            and not base.Config.query
        ):
            self.shapes = list(dict.fromkeys(self.shapes + self.set_more_shapes()))

    def set_more_shapes(self):
        # FP8's complete token/hidden-dimension matrix, plus decode/prefill
        # workloads and tails that do not require FP8 group divisibility.
        return [
            (num_tokens, d)
            for num_tokens in (7, 83, 2048)
            for d in (512, 4096, 5120, 13824)
        ] + [
            (1, 128),
            (1, 2048),
            (1, 8192),
            (16, 2048),
            (32, 4096),
            (64, 8192),
            (128, 1024),
            (256, 2048),
            (512, 8192),
            (1024, 4096),
            (4096, 2048),
            (4096, 4096),
            (4096, 8192),
            (128, 4097),
            (7, 5137),
        ]


def _input_fn(shape, dtype, device):
    x = torch.rand(shape, dtype=dtype, device=device)
    yield (x,)


def _native_npu_dynamic_quant(x):
    from torch_npu import npu_dynamic_quant

    q, scale = npu_dynamic_quant(x, dst_type=torch.int8)
    return q, scale.view(x.shape[:-1] + (1,))


@pytest.mark.per_token_quant_int8
def test_per_token_quant_int8():
    dtypes = [torch.bfloat16, torch.float16]
    if flaggems_vllm.vendor_name == "ascend":
        reference = _native_npu_dynamic_quant
    else:
        from vllm.model_executor.layers.quantization.utils.int8_utils import (
            per_token_quant_int8,
        )

        reference = per_token_quant_int8
        dtypes.append(torch.float32)
    bench = PerTokenQuantInt8Benchmark(
        op_name="per_token_quant_int8",
        torch_op=reference,
        input_fn=_input_fn,
        dtypes=dtypes,
    )
    bench.set_gems(flaggems_vllm.per_token_quant_int8)
    bench.run()
