# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Complete dynamic quantization with identical row-scale semantics."""

import pytest
import torch

import flaggems_vllm

from . import base


def reference(x):
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-10).float()
    scale = amax * torch.tensor(1.0 / 127.0, dtype=torch.float32, device=x.device)
    return (x.float() / scale).round().clamp(-128, 127).to(torch.int8), scale


class PerTokenQuantBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (rows, width) for rows in (4, 208, 1024) for width in (2048, 6144)
        ]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield (torch.randn(shape, dtype=dtype, device="cuda"),)


@pytest.mark.int8_quantize_per_token
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="NVIDIA CUDA required",
)
@torch.inference_mode()
def test_int8_quantize_per_token_perf():
    bench = PerTokenQuantBenchmark(
        op_name="int8_quantize_per_token", torch_op=reference, dtypes=[torch.bfloat16]
    )
    bench.set_gems(flaggems_vllm.int8_quantize_per_token)
    bench.run()
