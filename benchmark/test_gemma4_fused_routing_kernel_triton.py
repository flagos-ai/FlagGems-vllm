# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm
from tests._gemma4_routing_reference import (
    gemma4_fused_routing_kernel_triton as reference,
)

from . import base, consts


class Gemma4RoutingBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        tokens = [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            96,
            128,
            192,
            256,
            384,
            512,
            768,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
            12288,
            16384,
            24576,
            32768,
        ]
        self.shapes = [
            (t, e, k)
            for e, k in [(64, 4), (128, 4), (128, 8), (256, 8)]
            for t in tokens
        ]
        if base.Config.bench_level == consts.BenchLevel.CORE:
            self.shapes = [
                (1, 128, 8),
                (512, 64, 4),
                (1024, 128, 4),
                (8192, 128, 8),
                (2048, 256, 8),
            ]
        self.shape_desc = "synthetic Gemma 4 routing (T,E,K)"

    def get_input_iter(self, dtype):
        for index, (t, e, k) in enumerate(self.shapes):
            logits = torch.randn((t, e), device=self.device, dtype=dtype)
            scales = torch.rand(e, device=self.device, dtype=dtype)
            yield logits, k, scales


@pytest.mark.gemma4_fused_routing_kernel_triton
def test_gemma4_fused_routing_kernel_triton():
    bench = Gemma4RoutingBenchmark(
        op_name="gemma4_fused_routing_kernel_triton",
        torch_op=reference,
        gems_op=flaggems_vllm.gemma4_fused_routing_kernel_triton,
        dtypes=[torch.float32],
    )
    bench.run()
