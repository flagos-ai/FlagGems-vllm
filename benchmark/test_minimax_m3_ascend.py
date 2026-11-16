# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm
from . import base, consts


def reference(x, weight, cs, positions):
    source = x.view(-1, 8, 128)
    gamma = (1 + weight.float()).to(source.dtype)
    y = (
        (
            source.float()
            * torch.rsqrt(source.float().square().mean(-1, keepdim=True) + 1e-6)
            * gamma.float()
        )
        .to(source.dtype)
        .float()
    )
    c = cs[positions, :32].float().unsqueeze(1)
    s = cs[positions, 32:].float().unsqueeze(1)
    a, b = y[..., :32], y[..., 32:64]
    result = torch.cat((a * c - b * s, b * c + a * s, y[..., 64:]), -1).to(source.dtype)
    result[:, 6:] = source[:, 6:]
    return result.flatten(1)


def candidate(x, weight, cs, positions):
    result = x.clone()
    flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert_ascend(
        result, weight, weight, cs, positions, 4, 2, 64, 1e-6
    )
    return result


class M3AscendBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]

    def set_shapes(self, shape_file_path=None):
        self.shapes = [1, 4, 7, 64, 4096]
        self.shape_desc = "tokens, packed Q4/K2/V2, D128, rotary64; BF16"

    def get_input_iter(self, dtype):
        for n in self.shapes:
            x = torch.randn((n, 1024), device=self.device, dtype=dtype)
            w = torch.randn(128, device=self.device, dtype=dtype) * 0.1
            cs = torch.randn((n, 64), device=self.device, dtype=dtype)
            pos = torch.arange(n, device=self.device, dtype=torch.int64)
            yield x, w, cs, pos


@pytest.mark.minimax_m3_ascend
@pytest.mark.skipif(
    flaggems_vllm.runtime.device.vendor_name != "ascend", reason="Ascend-only"
)
def test_perf_minimax_m3_ascend():
    M3AscendBenchmark(
        op_name="minimax_m3_ascend",
        torch_op=reference,
        gems_op=candidate,
        dtypes=[torch.bfloat16],
    ).run()
