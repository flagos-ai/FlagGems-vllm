# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
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
    flaggems_vllm.device != "cuda" or not torch.cuda.is_available(),
    reason="requires CUDA-compatible tensors",
)
def test_gemma4_fused_routing_kernel_triton_perf():
    try:
        from vllm.model_executor.models.gemma4 import (
            gemma4_fused_routing_kernel_triton as vllm_reference,
        )
    except ImportError as exc:
        pytest.skip(f"vLLM Gemma 4 routing is unavailable: {exc}")

    bench = Gemma4RoutingBenchmark(
        op_name="gemma4_fused_routing_kernel_triton",
        torch_op=vllm_reference,
        gems_op=flaggems_vllm.gemma4_fused_routing_kernel_triton,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    )
    bench.run()
