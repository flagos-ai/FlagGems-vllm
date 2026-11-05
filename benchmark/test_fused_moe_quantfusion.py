# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm.ops.fused_moe import _int8_quantize_per_token_triton

from . import base, consts


def reference(x):
    scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-10).float() / 127
    return (x.float() / scale).round().clamp(-128, 127).to(torch.int8), scale


class OperatorBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 6144), (64, 6144), (4096, 6144), (5089, 768), (8192, 768)]
        self.shape_desc = "synthetic operator dimensions"

    def get_input_iter(self, dtype):
        for m, k in self.shapes:
            yield (torch.randn((m, k), device=self.device, dtype=dtype),)


@pytest.mark.fused_moe
def test_perf_fused_moe_quantfusion():
    bench = OperatorBenchmark(
        op_name="fused_moe",
        torch_op=reference,
        gems_op=_int8_quantize_per_token_triton,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.run()


def clamped_silu_quant_reference(packed, activation):
    gate, up = packed.float().chunk(2, dim=-1)
    gate, up = gate.clamp(max=10.0), up.clamp(-10.0, 10.0)
    rounded = (gate / (1.0 + torch.exp(-gate)) * up).to(packed.dtype)
    return reference(rounded)


def clamped_silu_quant_candidate(packed, activation):
    from flaggems_vllm.ops.fused_moe import _swigluoai_quantize_per_token_triton

    return _swigluoai_quantize_per_token_triton(
        packed, activation, 1.0, 0.0, 10.0, _silu=True
    )


class Hy4SiLUQuantBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(m * 8, 2048) for m in (1, 4, 8, 64, 1024)]
        self.shape_desc = "Hy4-preview quant2 rows=model tokens*8, expert width"

    def get_input_iter(self, dtype):
        for rows, width in self.shapes:
            yield (
                torch.randn(rows, width * 2, device=self.device, dtype=dtype),
                torch.empty(rows, width, device=self.device, dtype=dtype),
            )


@pytest.mark.fused_moe
@torch.inference_mode()
def test_perf_hy4_clamped_silu_quant():
    bench = Hy4SiLUQuantBenchmark(
        op_name="fused_moe",
        torch_op=clamped_silu_quant_reference,
        gems_op=clamped_silu_quant_candidate,
        dtypes=[torch.bfloat16],
    )
    bench.run()
