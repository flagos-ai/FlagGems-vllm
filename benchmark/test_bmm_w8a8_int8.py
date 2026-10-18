# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm

from . import base
from .test_blas_perf_parallel import ParallelBenchmarkMixin


def vllm_channel_int8_bmm(activation, weight, activation_scale, weight_scale):
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm,
    )

    # vLLM exposes scaled MM; independent batch slices require no layout copies.
    return [
        triton_scaled_mm(
            activation[index],
            weight[index],
            activation_scale[index, :, None],
            weight_scale[index, :, None],
            torch.bfloat16,
        )
        for index in range(activation.shape[0])
    ]


class ChannelInt8BmmBenchmark(ParallelBenchmarkMixin, base.Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPE_DESC = "B, M, N, K"

    def get_input_iter(self, dtype):
        for batch, rows, columns, reduction in self.shapes:
            torch.manual_seed(0)
            activation = torch.randint(
                -128, 128, (batch, rows, reduction), device=self.device, dtype=dtype
            )
            weight = torch.randint(
                -128, 128, (batch, columns, reduction), device=self.device, dtype=dtype
            ).transpose(1, 2)
            activation_scale = torch.rand((batch, rows), device=self.device) / 127
            weight_scale = torch.rand((batch, columns), device=self.device) / 127
            yield activation, weight, activation_scale, weight_scale


@pytest.mark.bmm_w8a8_int8
@pytest.mark.skipif(
    "bmm_w8a8_int8" not in vars(flaggems_vllm),
    reason="backend does not export channel-wise INT8 BMM",
)
def test_perf_bmm_w8a8_int8():
    bench = ChannelInt8BmmBenchmark(
        op_name="bmm_w8a8_int8", torch_op=vllm_channel_int8_bmm, dtypes=[torch.int8]
    )
    bench.set_gems(flaggems_vllm.bmm_w8a8_int8)
    bench.run()
