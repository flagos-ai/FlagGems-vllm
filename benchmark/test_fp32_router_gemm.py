# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm

from . import base, consts


def reference(x, weight):
    return torch.nn.functional.linear(x.float(), weight)


class OperatorBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]

    def set_shapes(self, shape_file_path=None):
        self.shapes = [1, 4, 8, 16, 32]
        self.shape_desc = "synthetic operator dimensions"

    def get_input_iter(self, dtype):
        for m in self.shapes:
            yield (
                torch.randn((m, 6144), device=self.device, dtype=dtype),
                torch.randn((128, 6144), device=self.device, dtype=torch.float32),
            )


@pytest.mark.fp32_router_gemm
def test_perf_fp32_router_gemm():
    bench = OperatorBenchmark(
        op_name="fp32_router_gemm",
        torch_op=reference,
        gems_op=flaggems_vllm.fp32_router_gemm,
        dtypes=[torch.bfloat16, torch.float32],
    )
    bench.run()
