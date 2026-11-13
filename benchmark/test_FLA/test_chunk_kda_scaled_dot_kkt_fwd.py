# SPDX-License-Identifier: Apache-2.0
import importlib.util
import math
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result

spec = importlib.util.spec_from_file_location(
    "kda_kkt_workload",
    Path(__file__).resolve().parents[2] / "baseline/chunk_kda_scaled_dot_kkt_fwd.py",
)
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


def _native_slot(*args, **kwargs):
    raise RuntimeError("native slot must be resolved by get_latency")


class ChunkKDAScaledDotKKTFwdBenchmark(Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).resolve().parents[1] / "core_shapes.yaml")
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

    def set_more_shapes(self):
        return self.DEFAULT_SHAPES

    def run(self):
        gems_only = Config.user_desired_metrics == ["latency"]
        self.native_op = None
        self.native_reason = (
            "native timing disabled by --metrics" if gems_only else None
        )
        if not gems_only:
            try:
                self.native_op = workload.native()
                self.native_reason = workload.NATIVE_UNAVAILABLE_REASON
                if self.native_op is self.gems_op:
                    self.native_op = None
                    self.native_reason = "native resolves to the Gems candidate"
            except Exception as error:
                self.native_reason = f"native resolution: {error!r}"
        with pytest.MonkeyPatch.context() as config:
            if gems_only:
                config.setattr(Config, "base_data", None)
            super().run()

    def get_input_iter(self, dtype):
        misses = []
        for b, t, h, k, bt, *lengths in self.shapes:
            binding = dict(B=b, T=t, H=h, K=k, BT=bt, lengths=lengths or None)
            args, kwargs = workload.build_inputs(binding, dtype, self.device)
            self.row_native_reason = self.native_reason
            yield (*args, kwargs)
            if self.row_native_reason:
                misses.append(
                    dict(
                        shape=self.record_shapes(*args, **kwargs),
                        reason=self.row_native_reason,
                    )
                )
        if misses:
            update_result(
                self.op_name,
                dict(
                    dtype=str(dtype),
                    native_baseline_skip_reason="; ".join(
                        dict.fromkeys(row["reason"] for row in misses)
                    ),
                    native_baseline_misses=misses,
                ),
            )

    def get_latency(self, op, *args, **kwargs):
        if op is not self.torch_op:
            return super().get_latency(op, *args, **kwargs)
        if self.native_op is None:
            return None

        expected = self.gems_op(*args, **kwargs)
        try:
            actual = self.native_op(*args, **kwargs)
            for got, want in zip(actual, expected, strict=True):
                torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)
            latency = super().get_latency(self.native_op, *args, **kwargs)
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"invalid native latency: {latency}")
        except Exception as error:
            self.row_native_reason = f"native validation or timing: {error!r}"
            return None
        return latency


@pytest.mark.chunk_kda_scaled_dot_kkt_fwd
def test_chunk_kda_scaled_dot_kkt_fwd_perf():
    bench = ChunkKDAScaledDotKKTFwdBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_native_slot,
    )
    bench.set_gems(gems.chunk_kda_scaled_dot_kkt_fwd)
    bench.run()
