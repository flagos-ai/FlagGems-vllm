# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone operator cost; not the archived complete-consumer benchmark."""

import pytest
import torch

import flaggems_vllm

from . import base


def reference(x, weight):
    z = x.float()
    y = (z / torch.sqrt((z * z).mean(-1, keepdim=True) + 1e-5)).bfloat16()
    y = (y * weight).bfloat16()
    absmax = y.float().abs().amax(-1, keepdim=True)
    scale = absmax / 127
    inv = torch.where(absmax == 0, 0.0, 127 / absmax)
    q = (y.float() * inv).round().clamp(-128, 127).to(torch.int8)
    return y, q, scale


class RMSQuantBenchmark(base.Benchmark):
    def set_shapes(self, shape_file=None):
        self.shapes = [(4, 2048), (208, 2048), (256, 2048)]

    def get_input_iter(self, dtype):
        for m, n in self.shapes:
            cpu = ((torch.arange(m * 2624) % 63 - 31).float() / 32).bfloat16()
            storage = cpu.reshape(m, 2624).to(device=flaggems_vllm.device)
            weight = torch.ones(n, dtype=dtype, device=flaggems_vllm.device)
            yield storage[:, :n], weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_rms_scaled_int8_quant_perf():
    bench = RMSQuantBenchmark(
        op_name="rms_scaled_int8_quant",
        torch_op=reference,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flaggems_vllm.rms_scaled_int8_quant)
    bench.run()
