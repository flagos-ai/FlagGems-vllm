# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Compare public min-p entry points on identical inputs.

The standard timer measures repeated calls on already-filtered logits
after warmup. Each provider starts from the same original logits.
"""

import pytest
import torch
from vllm.v1.worker.gpu.sample.min_p import apply_min_p as vllm_reference

try:
    from flaggems_vllm import apply_min_p, runtime
except ImportError as exc:
    raise ImportError(
        "Failed to import apply_min_p and runtime from flaggems_vllm. "
        "Check that FlagGems-vllm and its dependencies are installed correctly."
    ) from exc

from benchmark import base

pytestmark = [
    pytest.mark.apply_min_p,
    pytest.mark.skipif(
        not runtime.torch_device_fn.is_available(),
        reason="requires the active accelerator",
    ),
]


class MinPBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "tokens, vocab"
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [(50, 1024), (1, 32000), (32, 128256), (128, 152065)]

    def __init__(self):
        super().__init__(
            op_name="apply_min_p",
            torch_op=vllm_reference,
            gems_op=apply_min_p,
            dtypes=[torch.float32],
            is_inplace=True,
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for rows, vocab in self.shapes:
            logits = torch.full((rows, vocab), 1e-2, dtype=dtype, device=self.device)
            logits[:, 0] = 10.0
            mapping = torch.arange(rows, dtype=torch.int32, device=self.device)
            min_p = torch.full((rows,), 0.1, dtype=torch.float32, device=self.device)
            self._original = logits.clone()
            yield logits, mapping, min_p

    def get_latency(self, op, *args, **kwargs):
        # Restore once per provider; min-p is idempotent for these inputs.
        args[0].copy_(self._original)
        return super().get_latency(op, *args, **kwargs)


def test_min_p_perf():
    MinPBenchmark().run()
