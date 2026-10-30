# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Framework benchmark using the native FLA workload definition."""

import importlib.util
from pathlib import Path

import pytest

from benchmark.base import Benchmark
from flaggems_vllm.ops.FLA import chunk_kda_fwd_intra_token_parallel

spec = importlib.util.spec_from_file_location(
    "kda_intra_workload",
    Path(__file__).resolve().parents[2]
    / "baseline/chunk_kda_fwd_intra_token_parallel.py",
)
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


class ChunkKDAIntraTokenBenchmark(Benchmark):
    DEFAULT_DTYPES = workload.DTYPES
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [
        (binding["B"], binding["H"], *binding["lengths"]) for binding in workload.grid()
    ]
    DEFAULT_SHAPE_DESC = "B, H, sequence_lengths... (packed when multiple)"

    def init_user_config(self):
        super().init_user_config()
        if any(len(shape) < 3 for shape in self.shapes):
            self.shapes = self.DEFAULT_SHAPES
            self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, dtype):
        for batch, heads, *lengths in self.shapes:
            binding = dict(
                B=batch, H=heads, HV=heads, K=128, BT=64, BC=16, lengths=lengths
            )
            args, kwargs = workload.build_inputs(binding, dtype, self.device)
            yield (*args, kwargs)


@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel():
    bench = ChunkKDAIntraTokenBenchmark(
        op_name=workload.OP_NAME,
        torch_op=workload.native(),
        is_inplace=workload.IS_INPLACE,
    )
    bench.set_gems(chunk_kda_fwd_intra_token_parallel)
    bench.run()
