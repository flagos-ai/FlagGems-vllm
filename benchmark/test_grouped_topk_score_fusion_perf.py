# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Score fusion compared with the complete current generic route."""

import importlib

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.grouped_topk import grouped_topk as generic_grouped_topk

from . import base


class ScoreFusionBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(rows, 384) for rows in (1, 64, 256, 1024)]

    def get_input_iter(self, dtype):
        for rows, experts in self.shapes:
            yield torch.randn(
                rows, experts, dtype=dtype, device="cuda"
            ), 1, 1, 8, True, 1.0, torch.randn(experts, device="cuda"), 1


@pytest.mark.grouped_topk
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="NVIDIA CUDA required",
)
@torch.inference_mode()
def test_grouped_topk_score_fusion_perf(monkeypatch):
    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._nvidia.ops.grouped_topk"
    )
    monkeypatch.setattr(module, "_GROUPED_TOPK_SCORE_FUSION", True)
    monkeypatch.setattr(module, "_GROUPED_TOPK_SINGLE_GROUP_FUSION", True)
    bench = ScoreFusionBenchmark(
        op_name="grouped_topk", torch_op=generic_grouped_topk, dtypes=[torch.float32]
    )
    bench.set_gems(flaggems_vllm.grouped_topk)
    bench.run()
