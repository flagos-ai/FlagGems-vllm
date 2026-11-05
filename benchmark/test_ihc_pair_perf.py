# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired post-to-pre iHC compared with the existing unpaired direct API."""

import pytest
import torch

import flaggems_vllm

from . import base


class IHCPairBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(rows, 6144) for rows in (208, 256, 1024)]

    def get_input_iter(self, dtype):
        for rows, hidden in self.shapes:
            yield (
                torch.randn(rows, hidden, dtype=dtype, device="cuda"),
                torch.randn(rows, 4, hidden, dtype=dtype, device="cuda"),
                torch.rand(rows, 4, device="cuda"),
                torch.randn(8, 4 * hidden, device="cuda") / 256,
                torch.tensor([0.5, 0.5], device="cuda"),
                torch.zeros(8, device="cuda"),
                2.0,
                1e-6,
                1e-5,
            )


@pytest.mark.ihc_post_pre_pair
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="SM90 CUDA required",
)
@torch.inference_mode()
def test_ihc_pair_perf():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("selected paired schedule is scoped to SM90")
    bench = IHCPairBenchmark(
        op_name="ihc_post_pre_pair",
        torch_op=flaggems_vllm.ihc_post_pre,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flaggems_vllm.ihc_post_pre_pair)
    bench.run()
