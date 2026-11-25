# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Grouped LoRA-B expand, including caller-owned output and the CPU no-op."""

import math
import multiprocessing
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import flaggems_vllm as gems
from baseline import lora_expand as workload
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from benchmark.consts import BenchMode
from flaggems_vllm.runtime import torch_device_fn
from tests.test_lora_expand import assert_output, reference_output

# Benchmark.run requests this slot only when latency_base is enabled.
_NATIVE_SLOT = object()
_HOST_TIMER = "host_noop_perf_counter_ns"


def _measure(op, args, kwargs, initial, host_noop, warmup_ms, repetition_ms):
    """Median public-call ms. Reset outside every warmup and sample."""
    output = args[2]
    if not host_noop:
        start = torch_device_fn.Event(enable_timing=True)
        end = torch_device_fn.Event(enable_timing=True)

    def sample():
        output.copy_(initial)
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

    # Config uses milliseconds. Bound the two sampling windows by wall time,
    # including the setup overhead, while each event pair spans just one call.
    until = time.perf_counter() + warmup_ms / 1000
    while time.perf_counter() < until:
        sample()
    until = time.perf_counter() + repetition_ms / 1000
    samples = []
    while not samples or time.perf_counter() < until:
        samples.append(sample())
    latency = statistics.median(samples)
    if not math.isfinite(latency) or latency < 0 or (latency == 0 and not host_noop):
        raise ValueError(f"invalid latency in ms: {latency!r}")
    return latency


class _TimingError(RuntimeError):
    """The callable passed its initial numerical qualification."""


def _checked_latency(op, binding, args, kwargs, expected, warmup_ms, repetition_ms):
    output = args[2]
    initial = output.clone()
    exact = binding["case"] in ("round_fp32_out", "mixed_low_out")
    output.copy_(initial)
    returned = op(*args, **kwargs)
    assert returned is None, "lora_expand must return None"
    assert_output(output, expected, args, exact=exact)
    try:
        latency = _measure(
            op,
            args,
            kwargs,
            initial,
            binding["distribution"] == "disabled",
            warmup_ms,
            repetition_ms,
        )
    except Exception as exc:
        raise _TimingError(f"{type(exc).__name__}: {exc}") from exc
    assert_output(output, expected, args, exact=exact)
    return latency


def _native_row(binding, dtype, device, warmup_ms, repetition_ms):
    # A device illegal-address error poisons the process even after Python
    # catches it. Only the native gets this independent accelerator context.
    native = workload.native()
    reason = workload.NATIVE_UNAVAILABLE_REASON
    if native is gems.lora_expand or native is gems.ops.lora_expand:
        reason = "installed native resolves to the Gems candidate"
    if reason or not callable(native):
        return None, "NOT_RUN", reason or "installed native is not callable", False
    if device.index is not None:
        torch_device_fn.set_device(device.index)
    args, kwargs = workload.build_inputs(binding, dtype, device)
    expected = reference_output(*args, **kwargs)
    try:
        ms = _checked_latency(
            native, binding, args, kwargs, expected, warmup_ms, repetition_ms
        )
    except Exception as exc:
        timing_error = isinstance(exc, _TimingError)
        phase = "timing" if timing_error else "qualification"
        reason = f"native {phase}: {type(exc).__name__}: {exc}"
        try:
            torch_device_fn.synchronize()
        except Exception:
            return None, "PASS" if timing_error else "FAIL", reason, True
        return None, "PASS" if timing_error else "FAIL", reason, False
    if ms == 0:
        return None, "PASS", "zero host-call latency; native ratio unavailable", False
    return ms, "PASS", None, False


class LoraExpandBenchmark(Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPES = [tuple(workload.key_shape(b)) for b in workload.grid()]
    DEFAULT_SHAPE_DESC = "LoRA expand workload key from baseline.lora_expand"

    def init_user_config(self):
        # Benchmark otherwise replaces the caller's shape file on some vendors.
        # These grouped workloads use the same configuration keys on every chip.
        self.mode = Config.mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        self.set_shapes(Config.shape_file)

    def set_shapes(self, shape_file):
        if Path(shape_file).resolve() == Path(self.DEFAULT_SHAPE_FILES).resolve():
            self.shapes = self.DEFAULT_SHAPES
        else:
            super().set_shapes(shape_file)

    def run(self):
        if Config.mode != BenchMode.KERNEL:
            raise ValueError("lora_expand requires --mode kernel (no Graph)")
        if Config.warm_up < 0 or Config.repetition <= 0:
            raise ValueError("--warmup must be >= 0 and --iter > 0 (milliseconds)")
        data = Config.base_data
        if Config.user_desired_metrics == ["latency"]:
            data = None
        elif data and self.op_name in data:
            # The NVIDIA collector times every row using device events. Its
            # no-op cannot be paired with a wall-clock measurement of a CPU call.
            entry = data[self.op_name]
            shapes = dict(entry.get("shapes", {}))
            for binding in workload.grid():
                if binding["distribution"] == "disabled":
                    shapes.pop(str(workload.key_shape(binding)), None)
            data = {**data, self.op_name: {**entry, "shapes": shapes}}
        self.native_pool = None
        self.import_error = None
        try:
            with patch.object(Config, "base_data", data):
                super().run()
        finally:
            self._close_native_worker()

    def _close_native_worker(self):
        if self.native_pool is not None:
            self.native_pool.shutdown()
            self.native_pool = None

    def get_input_iter(self, dtype):
        bindings = {tuple(workload.key_shape(b)): b for b in workload.grid()}
        misses = []
        for shape in self.shapes:
            self.binding = bindings[shape]
            self.dtype = dtype
            self.host_noop = self.binding["distribution"] == "disabled"
            self.reason = self.import_error or workload.unsupported_reason(
                self.binding, dtype
            )
            if "latency_base" not in self.to_bench_metrics:
                self.reason = "native timing disabled by --metrics"
            self.stage = "NOT_RUN"
            args, kwargs = workload.build_inputs(self.binding, dtype, self.device)
            self.expected = reference_output(*args, **kwargs)
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
                        dict.fromkeys(row["reason"] for row in misses)
                    ),
                ),
            )

    def record_shapes(self, *args, **kwargs):
        # lookup_base_record_ex indexes the first torch.Size. Use the exact
        # compound collector key so equal tensor shapes retain distinct variants.
        return dict(
            key=torch.Size(workload.key_shape(self.binding)),
            case=self.binding["case"],
            timer=_HOST_TIMER if self.host_noop else "device_event",
            config=workload.config(self.binding, self.dtype),
            args=super().record_shapes(*args, **kwargs),
        )

    def get_latency(self, op, *args, **kwargs):
        if op is not self.torch_op:
            try:
                return _checked_latency(
                    op,
                    self.binding,
                    args,
                    kwargs,
                    self.expected,
                    Config.warm_up,
                    Config.repetition,
                )
            except Exception as exc:
                raise RuntimeError(f"Gems {self.binding['case']}: {exc}") from exc
        if self.reason:
            return None
        print(f"native {self.binding['case']} {self.dtype}", flush=True)
        try:
            if self.native_pool is None:
                self.native_pool = ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn")
                )
            ms, self.stage, self.reason, poisoned = self.native_pool.submit(
                _native_row,
                self.binding,
                self.dtype,
                args[0].device,
                Config.warm_up,
                Config.repetition,
            ).result()
        except BrokenProcessPool as exc:
            # Includes native aborts and fatal signals for which Python cannot
            # send back an exception. A fresh worker handles the next row.
            self.stage, self.reason = "FAIL", f"native worker exited: {exc}"
            self._close_native_worker()
            return None
        if poisoned:
            self._close_native_worker()
        if self.stage == "NOT_RUN":
            self.import_error = self.reason
        return ms


@pytest.mark.lora_expand
def test_lora_expand():
    LoraExpandBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_NATIVE_SLOT,
        gems_op=gems.lora_expand,
        dtypes=workload.DTYPES,
        is_inplace=workload.IS_INPLACE,
    ).run()
