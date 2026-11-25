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
from benchmark import consts
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from benchmark.consts import BenchMode as Mode
from benchmark.gumbel_baseline import canonical_chip
from flaggems_vllm.runtime import torch_device_fn
from tests.test_lora_shrink import assert_output, reference_output

# Benchmark.run asks get_latency for this slot even when native is unavailable.
_NATIVE = object()
_TIMERS = {
    Mode.KERNEL: ("device-events-reset-outside", "device-event"),
    Mode.OPERATOR: ("synchronized-host-wall-reset-outside", "operator-wall"),
    Mode.WRAPPER: ("host-dispatch-reset-outside", "wrapper-wall"),
}
_GRID = {tuple(b[d] for d in workload.DIMS): b for b in workload.grid()}
_EXTRA = {"prefill_uneven", "tail_zero_scale"}


def _positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def _qualified_archive(data, cfg, dtype, protocol, info):
    """Keep raw cross-chip ratios out of the base class's normalized column."""
    entry = data.get(workload.OP_NAME, {})
    factors = data.get("_scaling_factors") or {}
    gpu = data.get("_environment", {}).get("hardware", {}).get("gpu", {})
    reference = canonical_chip(factors.get("reference_chip"))
    info.update(
        source=entry.get("native_api"), reference_chip=factors.get("reference_chip")
    )
    if factors.get("_error"):
        return {}, factors["_error"]
    if not reference or canonical_chip(gpu.get("name")) != reference:
        return {}, "scaling factors do not identify the measured NVIDIA reference"
    if entry.get("native_api") not in (
        "ops.lora_shrink",
        "vllm.lora.ops.triton_ops.lora_shrink_op.lora_shrink",
    ):
        return {}, "missing or different native source"
    # Signature fallback cannot distinguish a different grouping of the same tokens.
    row = entry.get("shapes", {}).get(str(cfg["key"]), {}).get(str(dtype))
    if not row:
        return {}, "no native record for the exact workload key and dtype"
    if row.get("native_accuracy") != "PASS":
        return {}, "archived native lacks independent accuracy qualification"
    recorded = dict(row.get("config") or {})
    native_bi = recorded.pop("native_fixed_bi_enabled", None)
    if recorded != {k: v for k, v in cfg.items() if k != "native_fixed_bi_enabled"}:
        return {}, "archived workload or grouped metadata differs"
    if cfg["batch_invariant_requested"] and not native_bi:
        return {}, "archived native did not use fixed BI reduction"
    if cfg["case"] == "no_lora":
        return {}, "host-only call has no device scaling factor"
    if row.get("timing_protocol") != protocol or not _positive(row.get("latency_ms")):
        return {}, "incompatible timing protocol or invalid native latency"

    try:
        target = torch_device_fn.get_device_name(torch_device_fn.current_device())
    except (AttributeError, RuntimeError, TypeError):
        target = consts._SELF_CHIP if gems.device == "cuda" else ""
    matches = [
        c
        for c in factors.get("chips", [])
        if c.get("vendor") == gems.vendor_name
        and canonical_chip(c.get("chip")) == canonical_chip(target)
    ]
    if not canonical_chip(target) or len(matches) != 1:
        return {}, "no unique scaling factor entry for the actual target chip"
    chip = matches[0]
    resource = row.get("bottle_neck_unit")
    if (
        resource == "mem"
        and f"{gems.vendor_name} {chip['chip']}" in consts._ACHIEVABLE_BW_RATIO
        and reference != "h800"
    ):
        return {}, "bandwidth override requires an H800 reference"
    # Restrict the public factor lookup to the verified chip on multi-chip vendors.
    archive = {**data, "_scaling_factors": {**factors, "chips": matches}}
    factor = consts.lookup_scaling_factor(archive, gems.vendor_name, resource, dtype)
    if not _positive(factor):
        return {}, "missing bottleneck or positive scaling factor"
    info.update(target_chip=chip["chip"], bottleneck=resource, scaling_factor=factor)
    return archive, None


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
        self.archive = Config.base_data
        # Scope per-row archive selection, including when a candidate raises.
        with patch.object(Config, "base_data", None):
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
        _, prefix = _TIMERS[self.mode]
        statistic = "median" if self.mode == Mode.KERNEL else "mean"
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
            timer=_TIMERS[self.clock][0],
            warmup_calls=self.warmup_calls,
            measured_calls=self.measured_calls,
            statistic=statistic,
            unit="ms",
        )
        if cfg["case"] == "no_lora":
            self.detail["offline_timer_required"] = _TIMERS[self.clock][0]
        if self.archive is not None and Config.user_desired_metrics != ["latency"]:
            info = dict(target_chip=None, bottleneck=None, scaling_factor=None)
            protocol = (
                f"no-graph-{prefix}-reset-outside-{self.warmup_calls}warmup-"
                f"{self.measured_calls}sample-{statistic}"
            )
            Config.base_data, info["reason"] = _qualified_archive(
                self.archive, cfg, args[0].dtype, protocol, info
            )
            self.detail["offline_baseline"] = info
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
        if not all(map(_positive, samples)):
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
