# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Framework benchmark using the native FLA workload definition."""

import importlib.util
from pathlib import Path

import pytest

from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from benchmark.consts import lookup_base_record_ex
from flaggems_vllm.ops.FLA import chunk_kda_fwd_intra_token_parallel

spec = importlib.util.spec_from_file_location(
    "kda_intra_workload",
    Path(__file__).resolve().parents[2]
    / "baseline/chunk_kda_fwd_intra_token_parallel.py",
)
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


class ChunkKDAIntraTokenBenchmark(Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).resolve().parents[1] / "core_shapes.yaml")
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

    def set_more_shapes(self):
        return self.DEFAULT_SHAPES

    def record_shapes(self, *args, **kwargs):
        # Preserve the existing eager-mode key in the recorded baseline shape.
        return super().record_shapes(*args, **kwargs, use_graph=False)

    def run(self):
        self.offline_baseline = (
            None if Config.user_desired_metrics == ["latency"] else Config.base_data
        )
        # Scope per-row fallback to this benchmark, including failed runs.
        with pytest.MonkeyPatch.context() as config:
            config.setattr(Config, "base_data", self.offline_baseline)
            super().run()

    def get_input_iter(self, dtype):
        baseline_misses = []
        for batch, heads, *lengths in self.shapes:
            binding = dict(
                B=batch, H=heads, HV=heads, K=128, BT=64, BC=16, lengths=lengths
            )
            args, kwargs = workload.build_inputs(binding, dtype, self.device)
            if self.torch_op is None and self.offline_baseline is not None:
                shape = self.record_shapes(*args, **kwargs)
                record, reason = lookup_base_record_ex(
                    self.offline_baseline, self.op_name, dtype, shape, self.base_variant
                )
                if record is None or record.get("latency_ms") is None:
                    Config.base_data = None
                    baseline_misses.append(
                        {"shape": shape, "reason": reason or "offline latency missing"}
                    )
                else:
                    Config.base_data = self.offline_baseline
            yield (*args, kwargs)
        if self.torch_op is None:
            reason = (
                "native timing disabled by --metrics"
                if Config.user_desired_metrics == ["latency"]
                else workload.NATIVE_UNAVAILABLE_REASON
            )
            update_result(
                self.op_name,
                dict(
                    dtype=str(dtype),
                    native_baseline_skip_reason=reason,
                    offline_baseline_misses=baseline_misses,
                ),
            )


@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel():
    bench = ChunkKDAIntraTokenBenchmark(
        op_name=workload.OP_NAME,
        torch_op=(
            None if Config.user_desired_metrics == ["latency"] else workload.native()
        ),
        is_inplace=workload.IS_INPLACE,
    )
    bench.set_gems(chunk_kda_fwd_intra_token_parallel)
    bench.run()
