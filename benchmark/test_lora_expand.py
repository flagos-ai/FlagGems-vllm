# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark the public LoRA-B projection with caller-owned output."""

import math
import multiprocessing
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from baseline import lora_expand as workload
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from benchmark.consts import BenchMode
from flaggems_vllm.runtime import torch_device_fn
from tests.test_lora_expand import assert_output, reference_output

BINDINGS = {tuple(workload.key_shape(b)): b for b in workload.grid()}


def _measure(op, args, kwargs, initial, host_noop, warmup_ms, repetition_ms):
    """Restore before each sample, outside the device events or CPU no-op timer."""
    if not host_noop:
        start = torch_device_fn.Event(enable_timing=True)
        end = torch_device_fn.Event(enable_timing=True)

    def sample():
        args[2].copy_(initial)
        torch_device_fn.synchronize()
        if host_noop:
            begin = time.perf_counter_ns()
            op(*args, **kwargs)
            return (time.perf_counter_ns() - begin) * 1e-6
        start.record()
        op(*args, **kwargs)
        end.record()
        torch_device_fn.synchronize()
        return start.elapsed_time(end)

    until = time.perf_counter() + warmup_ms / 1000
    while time.perf_counter() < until:
        sample()
    until = time.perf_counter() + repetition_ms / 1000
    samples = []
    while not samples or time.perf_counter() < until:
        samples.append(sample())
    ms = statistics.median(samples)
    if not math.isfinite(ms) or ms <= 0:
        raise ValueError(f"invalid latency in ms: {ms!r}")
    return ms


class _TimingError(RuntimeError):
    """The initial numerical qualification passed."""


def _checked_latency(op, binding, args, kwargs, expected, warmup_ms, repetition_ms):
    output = args[2]
    initial = output.clone()
    exact = binding["case"] in ("round_fp32_out", "mixed_low_out")
    returned = op(*args, **kwargs)
    assert returned is None, "lora_expand must return None"
    assert_output(output, expected, args, exact=exact)
    try:
        host_noop = binding["distribution"] == "disabled"
        ms = _measure(op, args, kwargs, initial, host_noop, warmup_ms, repetition_ms)
    except Exception as exc:
        raise _TimingError(f"{type(exc).__name__}: {exc}") from exc
    assert_output(output, expected, args, exact=exact)
    return ms


def _native_row(binding, dtype, device, warmup_ms, repetition_ms):
    native = workload.native()
    if native is None or native is gems.lora_expand or native is gems.ops.lora_expand:
        reason = (
            workload.NATIVE_UNAVAILABLE_REASON or "installed native resolves to Gems"
        )
        return None, "NOT_RUN", reason, False
    if device.index is not None:
        torch_device_fn.set_device(device.index)
    args, kwargs = workload.build_inputs(binding, dtype, device)
    expected = reference_output(*args, **kwargs)
    try:
        ms = _checked_latency(
            native, binding, args, kwargs, expected, warmup_ms, repetition_ms
        )
    except Exception as exc:
        stage = "PASS" if isinstance(exc, _TimingError) else "FAIL"
        phase = "timing" if stage == "PASS" else "qualification"
        reason = f"native {phase}: {type(exc).__name__}: {exc}"
        try:
            torch_device_fn.synchronize()
        except Exception:
            return None, stage, reason, True
        return None, stage, reason, False
    return ms, "PASS", None, False


class LoraExpandBenchmark(Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPES = list(BINDINGS)
    pool = None
    import_error = None

    def close_worker(self):
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None

    def get_input_iter(self, dtype):
        misses = []
        for shape in self.shapes:
            self.binding = BINDINGS[shape]
            self.dtype = dtype
            self.reason = self.import_error or workload.unsupported_reason(
                self.binding, dtype
            )
            if "latency_base" not in self.to_bench_metrics:
                self.reason = "native timing disabled by --metrics"
            self.stage = "NOT_RUN"
            args, kwargs = workload.build_inputs(self.binding, dtype, self.device)
            # Device-event records are keyed by workload. The CPU no-op uses a host clock.
            self.base_variant = (
                "host_noop"
                if self.binding["distribution"] == "disabled"
                else str(list(shape))
            )
            yield (*args, kwargs)
            if self.reason or "speedup" not in self.to_bench_metrics:
                misses.append(
                    dict(
                        case=self.binding["case"],
                        key=list(shape),
                        native_accuracy=self.stage,
                        reason=self.reason or "native speedup disabled by --metrics",
                    )
                )
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
        # The collector looks up the first torch.Size in the standard report.
        return dict(
            key=torch.Size(workload.key_shape(self.binding)),
            case=self.binding["case"],
            timer=(
                "host_noop_perf_counter_ns"
                if self.binding["distribution"] == "disabled"
                else "device_event"
            ),
            config=workload.config(self.binding, self.dtype),
            args=super().record_shapes(*args, **kwargs),
        )

    def get_latency(self, op, *args, **kwargs):
        budget = Config.warm_up, Config.repetition
        if op is not self.torch_op:
            expected = reference_output(*args, **kwargs)
            return _checked_latency(op, self.binding, args, kwargs, expected, *budget)
        if self.reason:
            return None
        # An illegal native access can invalidate the device context for later rows.
        if self.pool is None:
            self.pool = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
        try:
            ms, self.stage, self.reason, poisoned = self.pool.submit(
                _native_row, self.binding, self.dtype, args[0].device, *budget
            ).result()
        except BrokenProcessPool as exc:
            ms, self.stage, self.reason, poisoned = None, "FAIL", str(exc), True
        if poisoned:
            self.close_worker()
        if self.stage == "NOT_RUN":
            self.import_error = self.reason
        return ms


@pytest.mark.lora_expand
def test_lora_expand(monkeypatch, request):
    if Config.mode != BenchMode.KERNEL or Config.warm_up < 0 or Config.repetition <= 0:
        raise ValueError("lora_expand requires kernel mode, warmup >= 0 and iter > 0")
    if Config.user_desired_metrics == ["latency"]:
        monkeypatch.setattr(Config, "base_data", None)
    elif entry := (Config.base_data or {}).get(workload.OP_NAME):
        variants = {
            key: {"shapes": {key: rows}}
            for key, rows in entry.get("shapes", {}).items()
        }
        data = {**Config.base_data, workload.OP_NAME: {**entry, "variants": variants}}
        monkeypatch.setattr(Config, "base_data", data)

    bench = LoraExpandBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_native_row,  # Resolves and calls the installed op in the worker.
        gems_op=gems.lora_expand,
        dtypes=workload.DTYPES,
        is_inplace=workload.IS_INPLACE,
    )
    request.addfinalizer(bench.close_worker)
    bench.run()
