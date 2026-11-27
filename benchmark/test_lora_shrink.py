# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark the public LoRA shrink call, including its output initialization."""

import math
import statistics
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import pytest
import torch

import flaggems_vllm as gems
from baseline import lora_shrink as workload
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from benchmark.consts import BenchMode as Mode
from flaggems_vllm.runtime import torch_device_fn
from tests.test_lora_shrink import assert_output, reference_output

# Benchmark.run asks get_latency for this slot even when native is unavailable.
_NATIVE = object()
_TIMERS = {
    Mode.KERNEL: "device-events-reset-outside",
    Mode.OPERATOR: "synchronized-host-wall-reset-outside",
    Mode.WRAPPER: "host-dispatch-reset-outside",
}
_GRID = {tuple(b[d] for d in workload.DIMS): b for b in workload.grid()}
_EXTRA = {"prefill_uneven", "tail_zero_scale"}


class LoraShrinkBenchmark(Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPES = [shape for shape, b in _GRID.items() if b["case"] not in _EXTRA]

    def set_more_shapes(self):
        if Path(Config.shape_file).resolve() == Path(self.DEFAULT_SHAPE_FILES):
            return [shape for shape, b in _GRID.items() if b["case"] in _EXTRA]
        return []

    def run(self):
        if not Config.query and Config.mode not in _TIMERS:
            raise ValueError("lora_shrink requires no-Graph timing")
        if Config.user_desired_metrics == ["latency"]:
            with patch.object(Config, "base_data", None):
                return super().run()
        return super().run()

    def get_input_iter(self, dtype):
        self.native = (
            workload.native() if "latency_base" in self.to_bench_metrics else None
        )
        misses = []
        for shape in self.shapes:
            binding = _GRID[tuple(shape)]
            self.binding = binding
            args, kwargs = workload.build_inputs(binding, dtype, self.device)
            self.expected = reference_output(*args, **kwargs)
            self.candidate = self._qualify(self.gems_op, args)
            yield (*args, kwargs)
            if reason := self.detail["native_reason"]:
                misses.append({"shape": self.detail["config"]["key"], "reason": reason})
        if misses:
            update_result(
                self.op_name,
                dict(
                    dtype=str(dtype),
                    native_baseline_misses=misses,
                    native_baseline_skip_reason="; ".join(
                        dict.fromkeys(m["reason"] for m in misses)
                    ),
                ),
            )

    def record_shapes(self, *args, **kwargs):
        cfg = workload.config(self.binding, args[0].dtype)
        self.clock = (
            Mode.OPERATOR
            if self.mode == Mode.KERNEL and cfg["case"] == "no_lora"
            else self.mode
        )
        reason = workload.unsupported_reason(self.binding, args[0].dtype)
        if self.native is None:
            reason = (
                workload.NATIVE_UNAVAILABLE_REASON
                if "latency_base" in self.to_bench_metrics
                else "native timing disabled by --metrics"
            )
        elif self.native is self.gems_op:
            reason = "native resolves to the candidate"
        self.detail = dict(
            key_shape=torch.Size(cfg["key"]),
            args=super().record_shapes(*args, **kwargs),
            config=cfg,
            candidate_accuracy="PASS",
            native_accuracy="NOT_RUN",
            native_reason=reason,
            native_performance="N/A",
            timer=_TIMERS[self.clock],
            warmup_calls=self.warmup_calls,
            measured_calls=self.measured_calls,
            statistic="median" if self.mode == Mode.KERNEL else "mean",
            unit="ms",
        )
        return self.detail

    def _qualify(self, op, args):
        out = args[2].clone()
        call_args = (*args[:2], out, *args[3:])
        ptr = out.data_ptr()
        if op(*call_args) is not None:
            raise AssertionError("lora_shrink must return None")
        if out.data_ptr() != ptr or out.stride() != args[2].stride():
            raise AssertionError(
                "lora_shrink changed the supplied output storage/layout"
            )
        assert_output(out, self.expected, call_args, exact=args[-1] == 0)
        return call_args

    def _samples(self, op, args, initial):
        # Benchmark.get_latency has no per-call reset hook. Keep each copy and
        # its synchronization outside the public call's timing interval.
        if self.clock == Mode.KERNEL:
            start = torch_device_fn.Event(enable_timing=True)
            end = torch_device_fn.Event(enable_timing=True)
        samples = []
        for i in range(self.warmup_calls + self.measured_calls):
            args[2].copy_(initial)
            torch_device_fn.synchronize()
            if i < self.warmup_calls:
                op(*args)
            elif self.clock == Mode.KERNEL:
                start.record()
                op(*args)
                end.record()
                torch_device_fn.synchronize()
                samples.append(start.elapsed_time(end))
            else:
                begin = perf_counter_ns()
                op(*args)
                if self.clock == Mode.OPERATOR:
                    torch_device_fn.synchronize()
                samples.append((perf_counter_ns() - begin) / 1e6)
                if self.clock == Mode.WRAPPER:
                    torch_device_fn.synchronize()
        if not all(math.isfinite(ms) and ms > 0 for ms in samples):
            raise ValueError(f"invalid latency samples (ms): {samples}")
        return samples

    def get_latency(self, op, *args, **kwargs):
        native = op is _NATIVE
        if native and self.detail["native_reason"]:
            return None
        stage = "qualification"
        try:
            call_args = self._qualify(self.native, args) if native else self.candidate
            if native:
                self.detail["native_accuracy"] = "PASS"
            stage = "timing"
            samples = self._samples(self.native if native else op, call_args, args[2])
            stage = "post-timing qualification"
            assert_output(call_args[2], self.expected, call_args, exact=args[-1] == 0)
        except Exception as exc:
            if not native:
                raise
            if stage != "timing":
                self.detail["native_accuracy"] = "FAIL"
            self.detail["native_reason"] = (
                f"native {stage}: {type(exc).__name__}: {exc}"
            )
            return None
        provider = "native" if native else "candidate"
        self.detail[f"{provider}_samples_ms"] = samples
        if native:
            self.detail["native_performance"] = "PASS"
        return (
            statistics.median(samples)
            if self.mode == Mode.KERNEL
            else statistics.mean(samples)
        )


@pytest.mark.lora_shrink
def test_lora_shrink(request):
    bench = LoraShrinkBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_NATIVE,
        gems_op=gems.lora_shrink,
        dtypes=workload.DTYPES,
        is_inplace=workload.IS_INPLACE,
    )
    # Explicit CLI values are strings. Defaults here match the in-place collector.
    warmup, rep = (request.config.getoption(k) for k in ("warmup", "iter"))
    bench.warmup_calls = int(warmup) if isinstance(warmup, str) else 5
    bench.measured_calls = int(rep) if isinstance(rep, str) else 25
    if bench.warmup_calls < 0 or bench.measured_calls < 1:
        raise ValueError("lora_shrink requires warmup >= 0 and iter >= 1 calls")
    bench.run()
