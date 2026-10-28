# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Full INT8 projection and epilogue at qualified output-projection shapes."""
import pytest
import torch

from flaggems_vllm.ops.int8_scaled_mm_decode import int8_scaled_mm_decode

from . import base


def reference(a, b, scale_a, scale_b):
    accumulator = torch._int_mm(a, b).float()
    return (scale_a * (scale_b.t() * accumulator)).bfloat16()


class Int8DecodeBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(m, 1024, 6144) for m in (1, 2, 4, 8, 16, 32, 64)]
        self.shape_desc = "Hy4-preview output projection M,K,N"

    def get_input_iter(self, dtype):
        for m, k, n in self.shapes:
            yield (
                torch.randint(-16, 17, (m, k), dtype=dtype, device=self.device),
                torch.randint(-16, 17, (n, k), dtype=dtype, device=self.device).t(),
                torch.rand(m, 1, device=self.device) / 32,
                torch.rand(n, 1, device=self.device) / 32,
            )


@pytest.mark.int8_scaled_mm_decode
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_int8_scaled_mm_decode_perf():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("qualified SM90 kernel contract")
    bench = Int8DecodeBenchmark(
        op_name="int8_scaled_mm_decode",
        torch_op=reference,
        gems_op=int8_scaled_mm_decode,
        dtypes=[torch.int8],
    )
    bench.run()
