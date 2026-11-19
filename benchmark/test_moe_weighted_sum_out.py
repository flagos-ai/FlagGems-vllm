# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.moe_sum import moe_sum

from . import base


def weighted_moe_sum_reference(route_output, weights, out, *, column_start):
    tokens, topk = weights.shape
    width = route_output.shape[1]
    weighted = route_output.view(tokens, topk, width).float() * weights[:, :, None]
    reduced = torch.empty((tokens, width), device=out.device, dtype=torch.float32)
    moe_sum(weighted, reduced)
    out[:, column_start : column_start + width].copy_(reduced)


class WeightedSumBenchmark(base.Benchmark):
    def get_input_iter(self, dtype):
        for tokens, topk, width, start, hidden in self.shapes:
            values = torch.randn(
                (tokens * topk, width), dtype=dtype, device=self.device
            )
            weights = torch.rand(
                (tokens, topk), dtype=torch.float32, device=self.device
            )
            out = torch.empty((tokens, hidden), dtype=dtype, device=self.device)
            yield values, weights, out, {"column_start": start}


@pytest.mark.moe_weighted_sum_out
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX backend")
def test_perf_moe_weighted_sum_out():
    WeightedSumBenchmark(
        op_name="moe_weighted_sum_out",
        torch_op=weighted_moe_sum_reference,
        gems_op=flaggems_vllm.moe_weighted_sum_out,
        dtypes=[torch.bfloat16],
    ).run()
