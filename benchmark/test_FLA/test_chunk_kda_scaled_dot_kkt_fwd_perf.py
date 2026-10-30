# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest

import flaggems_vllm as gems
from benchmark.base import Benchmark

spec = importlib.util.spec_from_file_location(
    "kda_kkt_workload",
    Path(__file__).resolve().parents[2] / "baseline/chunk_kda_scaled_dot_kkt_fwd.py",
)
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


class ChunkKDAScaledDotKKTFwdBenchmark(Benchmark):
    DEFAULT_DTYPES = workload.DTYPES
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [
        (
            binding["B"],
            binding["T"],
            binding["H"],
            binding["K"],
            binding["BT"],
            *(binding["lengths"] or []),
        )
        for binding in workload.grid()
    ]
    DEFAULT_SHAPE_DESC = "B, T, H, K, BT, seq_lens... (omitted for fixed length)"

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        if any(len(shape) < 5 for shape in self.shapes):
            self.shapes = self.DEFAULT_SHAPES
            self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype):
        for b, t, h, k, bt, *lengths in self.shapes:
            binding = dict(B=b, T=t, H=h, K=k, BT=bt, lengths=lengths or None)
            args, kwargs = workload.build_inputs(binding, cur_dtype, gems.device)
            yield (*args, kwargs)


@pytest.mark.chunk_kda_scaled_dot_kkt_fwd
def test_chunk_kda_scaled_dot_kkt_fwd_perf():
    bench = ChunkKDAScaledDotKKTFwdBenchmark(
        op_name=workload.OP_NAME,
        torch_op=workload.native(),
    )
    bench.set_gems(gems.chunk_kda_scaled_dot_kkt_fwd)
    bench.run()
