# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm

from . import base


class V2PenaltiesBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 1031), (4, 151936), (32, 151939)]

    def get_input_iter(self, dtype):
        for rows, vocab_size in self.shapes:
            index = torch.arange(rows, device=self.device, dtype=torch.int32)
            words = (vocab_size + 31) // 32
            yield (
                torch.randn(rows, vocab_size, device=self.device, dtype=dtype),
                index,
                torch.zeros_like(index),
                torch.zeros_like(index),
                torch.full((rows,), 1.001, device=self.device),
                torch.full((rows,), 0.01, device=self.device),
                torch.full((rows,), 0.01, device=self.device),
                torch.zeros(rows, words, dtype=torch.int32, device=self.device),
                torch.ones(rows, vocab_size, dtype=torch.int32, device=self.device),
            )


@pytest.mark.v2_apply_penalties
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Ascend required")
def test_v2_apply_penalties():
    benchmark = V2PenaltiesBenchmark(
        op_name="v2_apply_penalties", torch_op=None, dtypes=[torch.float32]
    )
    benchmark.set_gems(flaggems_vllm.v2_apply_penalties)
    benchmark.run()
