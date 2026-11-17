# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm

from . import base


def _reference(x, weight, eps):
    x32 = x.float()
    normalized = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight


def _candidate(x, weight, eps):
    return flaggems_vllm.gems_rms_forward(x, None, weight, eps)


class GemsRmsForwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 129), (1, 4096), (128, 4096), (128, 8192)]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
            w = torch.rand((shape[-1],), dtype=dtype, device=x.device)
            yield x, w, 1e-5


@pytest.mark.gems_rms_forward
def test_gems_rms_forward():
    benchmark = GemsRmsForwardBenchmark(
        op_name="gems_rms_forward",
        torch_op=_reference,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    )
    benchmark.set_gems(_candidate)
    benchmark.run()
