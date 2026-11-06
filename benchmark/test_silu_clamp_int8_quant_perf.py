# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Clamped SiLU and quantization, retaining the activation output in both arms."""

import pytest
import torch

import flaggems_vllm

from . import base


def reference(packed, output, limit):
    gate, up = packed.float().chunk(2, -1)
    gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    output.copy_((gate / (1 + torch.exp(-gate)) * up).to(output.dtype))
    z = output.float()
    amax = z.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = amax * torch.tensor(1 / 127, dtype=torch.float32, device=packed.device)
    return (z / scale).round().clamp(-128, 127).to(torch.int8), scale


class SiluQuantBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(rows, 2048) for rows in (1, 208, 256, 1024)]

    def get_input_iter(self, dtype):
        for rows, width in self.shapes:
            yield torch.randn(
                (rows, width * 2), dtype=dtype, device="cuda"
            ), torch.empty((rows, width), dtype=dtype, device="cuda"), 10.0


@pytest.mark.silu_clamp_int8_quant
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="CUDA required",
)
@torch.inference_mode()
def test_silu_clamp_int8_quant_perf():
    bench = SiluQuantBenchmark(
        op_name="silu_clamp_int8_quant", torch_op=reference, dtypes=[torch.bfloat16]
    )
    bench.set_gems(flaggems_vllm.silu_clamp_int8_quant)
    bench.run()
