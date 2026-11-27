# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm
from tests.test_hy4_flash_rotate_indexer import make_inputs, reference

from . import base


class IndexPoolBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 1, 64), (64, 4, 64), (1024, 4, 64), (4096, 1, 64)]
        self.shape_desc = "Hy4 Flash prefill and decode"

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield make_inputs(shape)


@pytest.mark.hy4_flash_rotate_indexer
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="Hy4 Flash IndexPool requires NVIDIA CUDA",
)
def test_hy4_flash_rotate_indexer_benchmark():
    benchmark = IndexPoolBenchmark(
        op_name="hy4_flash_rotate_indexer", torch_op=reference, dtypes=[torch.bfloat16]
    )
    benchmark.set_gems(flaggems_vllm.hy4_flash_rotate_indexer)
    benchmark.run()
