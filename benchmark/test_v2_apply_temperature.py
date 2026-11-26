# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm

from . import base


class V2TemperatureBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 1031), (4, 151936), (32, 151939)]

    def get_input_iter(self, dtype):
        for rows, vocab_size in self.shapes:
            yield (
                torch.randn(rows, vocab_size, device=self.device, dtype=dtype),
                torch.arange(rows, device=self.device, dtype=torch.int32),
                torch.full((rows,), 1.05, device=self.device),
            )


@pytest.mark.v2_apply_temperature
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Ascend required")
def test_v2_apply_temperature():
    benchmark = V2TemperatureBenchmark(
        op_name="v2_apply_temperature", torch_op=None, dtypes=[torch.float32]
    )
    benchmark.set_gems(flaggems_vllm.v2_apply_temperature)
    benchmark.run()
