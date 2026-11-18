# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm
from tests.test_hy4_flash_gated_pool import make_inputs, reference

from . import base


class IndexPoolBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 4), (64, 4), (1024, 16), (8192, 4)]
        self.shape_desc = "Hy4 Flash prefill and decode"

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield make_inputs(shape)


@pytest.mark.hy4_flash_gated_pool
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="Hy4 Flash IndexPool requires NVIDIA CUDA",
)
def test_hy4_flash_gated_pool_benchmark():
    benchmark = IndexPoolBenchmark(
        op_name="hy4_flash_gated_pool", torch_op=reference, dtypes=[torch.bfloat16]
    )
    benchmark.set_gems(flaggems_vllm.hy4_flash_gated_pool)
    benchmark.run()
