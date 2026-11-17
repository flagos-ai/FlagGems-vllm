# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""No-Graph KDA native/Gems pairs with full state reset outside each event."""

import json
import math
import os
import statistics
from pathlib import Path

import pytest

import flaggems_vllm as gems
from baseline import fused_recurrent_kda as baseline
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from benchmark.consts import BenchMode
from flaggems_vllm.runtime import torch_device_fn
from tests.test_FLA.test_fused_recurrent_kda import assert_result, reference


def _measure(fn, data, state):
    for _ in range(5):
        data["initial_state"].copy_(state)
        fn(**data)
    torch_device_fn.synchronize()
    samples = []
    for _ in range(25):
        data["initial_state"].copy_(state)
        torch_device_fn.synchronize()
        start = torch_device_fn.Event(enable_timing=True)
        end = torch_device_fn.Event(enable_timing=True)
        start.record()
        fn(**data)
        end.record()
        torch_device_fn.synchronize()
        samples.append(start.elapsed_time(end))
    data["initial_state"].copy_(state)
    return samples


def _native_slot(**data):
    raise RuntimeError("native slot must be resolved by get_latency")


class FusedRecurrentKDABenchmark(Benchmark):
    DEFAULT_DTYPES = baseline.DTYPES
    DEFAULT_SHAPES = [(case.name,) for case in baseline.PERF_CASES]
    DEFAULT_SHAPE_DESC = "case (baseline.PERF_CASES)"

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def init_user_config(self):
        if Config.mode != BenchMode.KERNEL:
            raise ValueError(
                "Recurrent KDA uses --mode kernel with no-Graph device events"
            )
        super().init_user_config()

    def run(self):
        gems_only = Config.user_desired_metrics == ["latency"]
        self.native_op = None
        self.native_reason = None
        if gems_only:
            self.native_reason = "native timing disabled by --metrics"
        elif gems.vendor_name == "ascend":
            self.native_reason = (
                "Ascend native KDA disabled: native accuracy is unstable"
            )
        else:
            try:
                self.native_op = baseline.native()
                self.native_reason = baseline.NATIVE_UNAVAILABLE_REASON
            except Exception as error:
                self.native_reason = f"native resolution: {error!r}"
        with pytest.MonkeyPatch.context() as config:
            if gems_only:
                config.setattr(Config, "base_data", None)
            super().run()

    def get_input_iter(self, dtype):
        misses = []
        cases = {case.name: case for case in baseline.PERF_CASES}
        for (name,) in self.shapes:
            self.case = cases[name]
            self.dtype = dtype
            self.latencies = None
            self.row_native_reason = self.native_reason
            yield (baseline.build(self.case, dtype, self.device),)
            if self.row_native_reason:
                misses.append(dict(shape=name, reason=self.row_native_reason))
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

    def record_shapes(self, *args, **kwargs):
        return dict(case=self.case.name, inputs=super().record_shapes(*args, **kwargs))

    def get_latency(self, op, *args, **kwargs):
        if self.latencies is None:
            self.latencies = self._measure_pair(kwargs)
        return self.latencies["native_ms" if op is self.torch_op else "gems_ms"]

    def _measure_pair(self, data):
        state = data["initial_state"].clone()
        expected = reference(data)
        row = dict(
            case=self.case.name,
            dtype=str(self.dtype),
            backend=gems.vendor_name,
            graph=False,
            unit="ms",
            protocol="device-event-reset-outside-5warmup-25sample-median",
            config=baseline.config({"case": self.case}, self.dtype),
            candidate_accuracy="NOT_RUN",
            native_accuracy="NOT_RUN",
            native_ms=None,
            selected_source=None,
        )
        functions = {"gems": self.gems_op}
        if self.native_op is not None and "latency_base" in self.to_bench_metrics:
            if self.native_op is self.gems_op:
                self.row_native_reason = "native resolves to the Gems candidate"
            else:
                self.row_native_reason = baseline.unsupported_reason(
                    {"case": self.case}, self.dtype
                )
            if self.row_native_reason is None:
                functions["native"] = self.native_op
        if self.case.name in ("decode16", "decode256", "outofplace"):
            functions = dict(reversed(list(functions.items())))

        candidate_error = None
        for name, fn in functions.items():
            key = "candidate_accuracy" if name == "gems" else "native_accuracy"
            data["initial_state"].copy_(state)
            try:
                assert_result(fn(**data), expected)
                row[key] = "PASS"
                samples = _measure(fn, data, state)
                latency = statistics.median(samples)
                if not math.isfinite(latency) or latency <= 0:
                    raise ValueError(f"invalid {name} latency: {latency}")
                row[name + "_samples_ms"] = samples
                row[name + "_ms"] = latency
            except Exception as error:
                if row[key] != "PASS":
                    row[key] = "FAIL"
                    row[name + "_error"] = repr(error)
                else:
                    row[name + "_timing_error"] = repr(error)
                if name == "gems":
                    candidate_error = error
                else:
                    self.row_native_reason = repr(error)
            finally:
                data["initial_state"].copy_(state)

        if self.row_native_reason:
            row["native_error"] = self.row_native_reason
        if row["native_ms"] is not None and "gems_ms" in row:
            row["native_speedup"] = row["native_ms"] / row["gems_ms"]
            row["selected_source"] = "native"
        target = os.environ.get("RKDA_RESULTS")
        if target:
            with Path(target).open("a") as output:
                output.write(json.dumps(row) + "\n")
        if candidate_error is not None:
            raise candidate_error
        return row


@pytest.mark.fused_recurrent_kda
def test_fused_recurrent_kda():
    FusedRecurrentKDABenchmark(
        op_name=baseline.OP_NAME,
        torch_op=_native_slot,
        gems_op=gems.fused_recurrent_kda,
        is_inplace=baseline.IS_INPLACE,
    ).run()
