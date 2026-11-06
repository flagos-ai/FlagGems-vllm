# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BF16 gate product plus dynamic INT8 quantization, including row strides."""
import pytest
import torch

from flaggems_vllm.ops.mul_scaled_int8_quant import mul_scaled_int8_quant

from . import base


def reference(a, b):
    z = (a * b).bfloat16().float()
    absmax = z.abs().amax(-1, keepdim=True)
    inverse = torch.where(absmax == 0, 0.0, 127.0 / absmax)
    return (z * inverse).round().clamp(-128, 127).to(torch.int8), absmax / 127.0


class MulQuantBenchmark(base.Benchmark):
    def set_shapes(self, shape_file=None):
        self.shapes = [(m, 1024) for m in (1, 4, 8, 64, 256, 1024)]

    def get_input_iter(self, dtype):
        for m, n in self.shapes:
            a = torch.randn(m, n + 32, dtype=dtype, device="cuda")[:, :n]
            b = torch.randn(m, n, dtype=dtype, device="cuda")
            yield a, b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_mul_scaled_int8_quant_perf():
    bench = MulQuantBenchmark(
        op_name="mul_scaled_int8_quant", torch_op=reference, dtypes=[torch.bfloat16]
    )
    bench.set_gems(mul_scaled_int8_quant)
    bench.run()
