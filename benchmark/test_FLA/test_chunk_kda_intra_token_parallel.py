# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Framework benchmark using the native FLA workload definition."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from flaggems_vllm.ops.FLA import chunk_kda_fwd_intra_token_parallel
from tests.test_FLA.test_chunk_kda_intra_token_parallel import TOLERANCES

spec = importlib.util.spec_from_file_location(
    "kda_intra_workload",
    Path(__file__).resolve().parents[2]
    / "baseline/chunk_kda_fwd_intra_token_parallel.py",
)
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


def _native_slot(*args, **kwargs):
    raise RuntimeError("native slot must be resolved by get_latency")


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
        for batch, heads, *lengths in self.shapes:
            binding = dict(
                B=batch, H=heads, HV=heads, K=128, BT=64, BC=16, lengths=lengths
            )
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

        expected_args = (*args[:4], args[4].clone(), args[5].clone(), *args[6:])
        self.gems_op(*expected_args, **kwargs)
        try:
            native_args = (*args[:4], args[4].clone(), args[5].clone(), *args[6:])
            self.native_op(*native_args, **kwargs)
            for got, want in zip(native_args[4:6], expected_args[4:6]):
                rtol, atol = TOLERANCES[want.dtype]
                torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
            latency = super().get_latency(self.native_op, *native_args, **kwargs)
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"invalid native latency: {latency}")
        except Exception as error:
            self.row_native_reason = f"native validation or timing: {error!r}"
            return None
        return latency


@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel():
    bench = ChunkKDAIntraTokenBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_native_slot,
        is_inplace=workload.IS_INPLACE,
    )
    bench.set_gems(chunk_kda_fwd_intra_token_parallel)
    bench.run()
