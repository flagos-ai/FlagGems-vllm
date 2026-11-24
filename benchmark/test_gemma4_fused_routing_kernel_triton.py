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

import flaggems_vllm

from . import base


class Gemma4RoutingBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_tokens, num_experts, topk"
    DEFAULT_SHAPES = [
        (1, 128, 8),
        (64, 128, 8),
        (131, 65, 16),
        (512, 64, 4),
        (8192, 128, 8),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for num_tokens, num_experts, topk in self.shapes:
            logits = torch.randn(
                (num_tokens, num_experts), device=self.device, dtype=dtype
            )
            scales = torch.rand(num_experts, device=self.device, dtype=dtype)
            yield logits, topk, scales


@pytest.mark.gemma4_fused_routing_kernel_triton
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend"
    and (flaggems_vllm.device != "cuda" or not torch.cuda.is_available()),
    reason="requires Ascend or CUDA-compatible tensors",
)
def test_gemma4_fused_routing_kernel_triton_perf():
    if flaggems_vllm.vendor_name == "ascend":
        from baseline.gemma4_fused_routing_kernel_triton import (
            gemma4_fused_routing_kernel_triton as reference,
        )
    else:
        try:
            from vllm.model_executor.models.gemma4 import (
                gemma4_fused_routing_kernel_triton as reference,
            )
        except ImportError as exc:
            pytest.skip(f"vLLM Gemma 4 routing is unavailable: {exc}")

    bench = Gemma4RoutingBenchmark(
        op_name="gemma4_fused_routing_kernel_triton",
        torch_op=reference,
        gems_op=flaggems_vllm.gemma4_fused_routing_kernel_triton,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    )
    bench.run()
