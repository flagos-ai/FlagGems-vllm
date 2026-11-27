# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm
from tests.test_hy4_flash_engram_lookup_norm import make_inputs, reference

from . import base


class FlashEngramBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 4, 128), (64, 4, 128), (67, 4, 96), (8192, 4, 128)]
        self.shape_desc = "Hy4 Flash Engram prefill and decode"

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield make_inputs(shape, fp8=True)


@pytest.mark.hy4_flash_engram_lookup_norm
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="Hy4 Flash Engram requires NVIDIA CUDA",
)
def test_hy4_flash_engram_lookup_norm_benchmark():
    bench = FlashEngramBenchmark(
        op_name="hy4_flash_engram_lookup_norm",
        torch_op=reference,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flaggems_vllm.hy4_flash_engram_lookup_norm)
    bench.run()
