# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Full channel affine operator, including caller-owned output writes."""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import fused_mm_input_norm

from . import base


def reference(inputs, outputs, weight, bias):
    return (inputs.float() * weight[None, :, None] + bias[None, :, None]).to(
        outputs.dtype
    )


class MMInputNormBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(rows, 3, 256) for rows in (256, 1024, 4096)]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield (
                torch.randint(0, 256, shape, dtype=torch.uint8, device=self.device),
                torch.empty(shape, dtype=dtype, device=self.device),
                torch.tensor([0.25, 0.125, 0.5], device=self.device),
                torch.tensor([-2.0, 1.0, -3.0], device=self.device),
            )


@pytest.mark.fused_mm_input_norm
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or flaggems_vllm.vendor_name != "nvidia",
    reason="NVIDIA CUDA required",
)
@torch.inference_mode()
def test_fused_mm_input_norm_perf():
    benchmark = MMInputNormBenchmark(
        op_name="fused_mm_input_norm",
        torch_op=reference,
        gems_op=fused_mm_input_norm,
        dtypes=[torch.bfloat16, torch.float16, torch.float32],
    )
    benchmark.run()
