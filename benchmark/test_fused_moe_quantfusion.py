# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm.ops.fused_moe import _int8_quantize_per_token_triton

from . import base, consts


def reference(x):
    scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-10).float() / 127
    return (x.float() / scale).round().clamp(-128, 127).to(torch.int8), scale


class OperatorBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 6144), (64, 6144), (4096, 6144), (5089, 768), (8192, 768)]
        self.shape_desc = "synthetic operator dimensions"

    def get_input_iter(self, dtype):
        for m, k in self.shapes:
            yield (torch.randn((m, k), device=self.device, dtype=dtype),)


@pytest.mark.fused_moe
def test_perf_fused_moe_quantfusion():
    bench = OperatorBenchmark(
        op_name="fused_moe",
        torch_op=reference,
        gems_op=_int8_quantize_per_token_triton,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.run()
