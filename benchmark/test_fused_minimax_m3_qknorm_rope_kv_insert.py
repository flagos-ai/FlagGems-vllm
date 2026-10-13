# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm

from . import base, consts


def reference(x, w, cs, pos):
    x = x.view(-1, 8, 128).float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + w.float())
    phase = cs[pos].float()
    c = phase[:, :32, None].transpose(1, 2)
    s = phase[:, 32:, None].transpose(1, 2)
    a, b = y[..., :32], y[..., 32:64]
    result = torch.cat((a * c - b * s, b * c + a * s, y[..., 64:]), -1).bfloat16()
    result[:, 6:] = x[:, 6:].bfloat16()
    return result.view(-1, 1024)


def candidate(x, w, cs, pos):
    x = x.clone()
    flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(
        x, w, w, cs, pos, 4, 2, 64, 1e-6
    )
    return x


class OperatorBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]

    def set_shapes(self, shape_file_path=None):
        self.shapes = [1, 4, 64, 4096, 8192]
        self.shape_desc = "synthetic operator dimensions"

    def get_input_iter(self, dtype):
        for m in self.shapes:
            x = torch.randn((m, 1024), device=self.device, dtype=dtype)
            w = torch.randn((128,), device=self.device, dtype=dtype) * 0.1
            cs = torch.randn((max(m, 1), 64), device=self.device, dtype=dtype)
            pos = torch.arange(m, device=self.device, dtype=torch.int64)
            yield (x, w, cs, pos)


@pytest.mark.fused_minimax_m3_qknorm_rope_kv_insert
def test_perf_fused_minimax_m3_qknorm_rope_kv_insert():
    bench = OperatorBenchmark(
        op_name="fused_minimax_m3_qknorm_rope_kv_insert",
        torch_op=reference,
        gems_op=candidate,
        dtypes=[torch.bfloat16],
    )
    bench.run()
