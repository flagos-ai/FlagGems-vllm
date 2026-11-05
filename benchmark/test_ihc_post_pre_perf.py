# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone full post-to-pre operator cost; no serving-speed attribution."""
import pytest
import torch

from flaggems_vllm.ops.ihc_post_pre import ihc_post_pre

from . import base


def reference(x, residual, post, weight, scale, bias, magnitude, hc_eps, norm_eps):
    materialized = (
        post[:, :, None] * x.float()[:, None, :] + residual.float()
    ).bfloat16()
    flat = materialized.float().flatten(1)
    mixes = (flat @ weight.t()) * torch.rsqrt(
        (flat * flat).mean(1, keepdim=True) + norm_eps
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + hc_eps
    gates = magnitude * torch.sigmoid(mixes[:, 4:] * scale[1] + bias[4:]) + hc_eps
    return (
        (pre[:, :, None] * materialized.float()).sum(1).bfloat16(),
        gates,
        materialized,
    )


class IHCPostPreBenchmark(base.Benchmark):
    def set_shapes(self, shape_file=None):
        self.shapes = [(m, 6144) for m in (1, 3, 4, 8, 64, 1024)]

    def get_input_iter(self, dtype):
        for m, hidden in self.shapes:
            yield (
                torch.randn(m, hidden, dtype=dtype, device="cuda"),
                torch.randn(m, 4, hidden, dtype=dtype, device="cuda"),
                torch.rand(m, 4, device="cuda"),
                torch.randn(8, 4 * hidden, device="cuda") / 256,
                torch.tensor([0.5, 0.5], device="cuda"),
                torch.zeros(8, device="cuda"),
                2.0,
                1e-6,
                1e-5,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_ihc_post_pre_perf():
    bench = IHCPostPreBenchmark(
        op_name="ihc_post_pre", torch_op=reference, dtypes=[torch.bfloat16]
    )
    bench.set_gems(ihc_post_pre)
    bench.run()
