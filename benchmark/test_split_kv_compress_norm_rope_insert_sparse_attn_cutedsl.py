# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Measure both C128 stages as one public call with reset-outside timing."""

import math
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import flaggems_vllm as gems
from baseline import split_kv_compress_norm_rope_insert_sparse_attn_cutedsl as workload
from benchmark import conftest as report
from benchmark import consts
from benchmark.base import Benchmark
from flaggems_vllm.runtime import torch_device_fn
from tests.test_split_kv_compress_norm_rope_insert_sparse_attn_cutedsl import (
    assert_result,
    reference,
)

_NATIVE_SLOT = object()


def _backing(tensor):
    count = tensor.untyped_storage().nbytes() // tensor.element_size()
    return tensor.as_strided((count,), (1,), 0).view(torch.uint8)


def _measure(op, args, kwargs, buffers, mode):
    # The shared timer has no hook for an untimed restore on every iteration.
    # These events never enclose the copies and no capture/replay API is used.
    start = end = None
    if mode == consts.BenchMode.KERNEL:
        start, end = (torch_device_fn.Event(enable_timing=True) for _ in range(2))

    def sample():
        for backing, snapshot in buffers:
            backing.copy_(snapshot)
        torch_device_fn.synchronize()
        if mode == consts.BenchMode.KERNEL:
            start.record()
            op(*args, **kwargs)
            end.record()
            torch_device_fn.synchronize()
            ms = start.elapsed_time(end)
        else:
            before = time.perf_counter()
            op(*args, **kwargs)
            if mode == consts.BenchMode.OPERATOR:
                torch_device_fn.synchronize()
            ms = (time.perf_counter() - before) * 1000
            if mode == consts.BenchMode.WRAPPER:
                torch_device_fn.synchronize()
        if not math.isfinite(ms) or ms <= 0:
            raise ValueError(f"invalid {mode.value} latency in ms: {ms}")
        return ms

    warmups, elapsed = 0, 0.0
    while warmups < 5 or elapsed < report.Config.warm_up:
        elapsed += sample()
        warmups += 1
    samples = []
    elapsed = 0.0
    while len(samples) < 25 or elapsed < report.Config.repetition:
        ms = sample()
        samples.append(ms)
        elapsed += ms
    latency = (
        statistics.median(samples)
        if mode == consts.BenchMode.KERNEL
        else statistics.mean(samples)
    )
    return latency, samples, warmups


class SplitBenchmark(Benchmark):
    DEFAULT_SHAPES = workload.CORE
    DEFAULT_SHAPE_DESC = ", ".join(workload.FIELDS)
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_DTYPES = workload.DTYPES
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]

    def set_shapes(self, shape_file_path=None):
        path = Path(shape_file_path or self.DEFAULT_SHAPE_FILES)
        if path.resolve() == Path(self.DEFAULT_SHAPE_FILES).resolve():
            self.shapes = list(workload.CORE)
            if (
                not report.Config.query
                and report.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
            ):
                self.shapes.extend(workload.EXTRA)
        else:
            super().set_shapes(str(path))
        if (
            not self.shapes
            or any(len(row) != len(workload.FIELDS) for row in self.shapes)
            or len(set(self.shapes)) != len(self.shapes)
        ):
            raise ValueError(
                f"expected unique [{self.DEFAULT_SHAPE_DESC}] configurations"
            )

    def init_default_config(self):
        self.set_shapes(report.Config.shape_file)

    def init_user_config(self):
        super().init_user_config()
        if self.mode == consts.BenchMode.CUDAGRAPH:
            raise ValueError("C128 split benchmark requires a no-Graph mode")
        if report.Config.warm_up < 0 or report.Config.repetition <= 0:
            raise ValueError("expected --warmup >= 0 and --iter > 0 (milliseconds)")
        self.native_op, self.native_reason = None, "native timing disabled by --metrics"
        if "latency_base" in self.to_bench_metrics:
            self.native_op = workload.native()
            self.native_reason = workload.NATIVE_UNAVAILABLE_REASON
            if self.native_op is self.gems_op:
                self.native_op, self.native_reason = (
                    None,
                    "installed native resolves to Gems",
                )
        self.scaling_dtype = torch.float32
        self.protocol = dict(
            mode=self.mode.value,
            graph=False,
            scope="public split call, compression and store, including internal default-scale handling",
            reset="entire compressed_kv and k_cache backing buffers and synchronize before each call",
            reduction="median" if self.mode == consts.BenchMode.KERNEL else "mean",
            timer=(
                "device events"
                if self.mode == consts.BenchMode.KERNEL
                else "wall clock"
            ),
            unit="ms",
            warmup_budget_ms=report.Config.warm_up,
            sample_budget_ms=report.Config.repetition,
            minimum_warmups=5,
            minimum_samples=25,
        )

    def set_metrics(self, desired):
        if desired:
            desired = [
                m
                for m in dict.fromkeys([*self.DEFAULT_METRICS, *desired])
                if m in desired
            ]
        super().set_metrics(desired)

    def _offline(self, dtype):
        if "speedup" not in self.to_bench_metrics:
            return "offline comparison disabled by --metrics"
        cfg = self.row["config"]
        source = (
            self.offline_entry.get("shapes", {}).get(cfg["case"], {}).get(str(dtype))
        )
        if not isinstance(source, dict) or source.get("config") != cfg:
            return "no offline record for the complete configuration and weight dtype"
        if (
            source.get("native_status") != "PASS"
            or source.get("native_accuracy") != "PASS"
        ):
            return "offline native lacks successful numerical qualification"
        ms = source.get("latency_ms")
        if not isinstance(ms, (float, int)) or not math.isfinite(ms) or ms <= 0:
            return "offline native latency is missing or invalid"
        if source.get("timing_protocol") != self.protocol:
            return "offline timing mode, reset protocol, or sampling budget differs"

        data = report.Config.base_data
        gpu = data.get("_environment", {}).get("hardware", {}).get("gpu", {})
        chip = gpu.get("name") or data.get("_qualification", {}).get("chip")
        self.row.update(base_reference_chip=chip, base_latency_ms=ms)
        sf = data.get("_scaling_factors", {})
        reference_chip = sf.get("reference_chip")
        if not consts._chip_matches(reference_chip, chip):
            return "offline source chip differs from or lacks a scaling reference"
        unit = source.get("bottle_neck_unit")
        chips = [
            c
            for c in sf.get("chips", [])
            if c.get("vendor") == gems.vendor_name
            and consts._chip_matches(c.get("chip"), consts._SELF_CHIP)
        ]
        if len(chips) != 1:
            return "no unique scaling factor for the measured target chip"
        target = chips[0].get("chip")
        if (
            unit == "mem"
            and f"{gems.vendor_name} {target}" in consts._ACHIEVABLE_BW_RATIO
            and reference_chip != "H800"
        ):
            return "shared bandwidth calibration uses an H800 reference"
        factor = consts.lookup_scaling_factor(
            data, gems.vendor_name, unit, self.scaling_dtype
        )
        if (
            not isinstance(factor, (float, int))
            or not math.isfinite(factor)
            or factor <= 0
        ):
            return "missing target scaling factor for the measured bottleneck"
        self.row.update(
            base_scaling_factor=factor, base_target_chip=target, base_bottleneck=unit
        )
        # Select one semantic binding before calling the shared shape lookup.
        shape = str(cfg["inputs"]["state_cache"]["shape"])
        self.variants[self.base_variant] = {"shapes": {shape: {str(dtype): source}}}
        return None

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self.binding = dict(zip(workload.FIELDS, shape))
            cfg = workload.config(self.binding, dtype)
            self.base_variant = (cfg["case"], str(dtype))
            args, kwargs = workload.build_inputs(self.binding, dtype, self.device)
            expected = reference(*args, **kwargs)
            buffers = [
                (_backing(args[i]), _backing(args[i]).clone())
                for i in workload.MUTATED_INPUTS
            ]
            torch_device_fn.synchronize()
            before = time.perf_counter()
            result = self.gems_op(*args, **kwargs)
            torch_device_fn.synchronize()
            first_ms = (time.perf_counter() - before) * 1000
            assert_result(expected, result, *args, **kwargs)
            self.buffers, self.expected = buffers, expected
            self.row = dict(
                config=cfg,
                candidate_accuracy="PASS",
                candidate_first_call_wall_ms=first_ms,
                native_entry=f"{workload.NATIVE_MODULE}.{workload.OP_NAME}",
                native_status="NOT_RUN",
                native_accuracy="NOT_RUN",
                native_stage="resolution",
                native_reason=self.native_reason,
                timing_protocol=self.protocol,
            )
            try:
                self.row["offline_reason"] = self._offline(dtype)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                self.row["offline_reason"] = f"invalid offline metadata: {exc}"
            self.rows[self.base_variant] = self.row
            yield (*args, kwargs)

    def record_shapes(self, *args, **kwargs):
        return dict(
            case=self.base_variant[0], inputs=super().record_shapes(*args, **kwargs)
        )

    def get_latency(self, op, *args, **kwargs):
        if op is not _NATIVE_SLOT:
            ms, samples, warmups = _measure(op, args, kwargs, self.buffers, self.mode)
            self.row.update(candidate_samples_ms=samples, candidate_warmups=warmups)
            if "base_latency_ms" in self.row:
                self.row["base_raw_speedup"] = self.row["base_latency_ms"] / ms
            return ms
        if self.native_op is None:
            return None
        dtype = args[7].dtype
        # A failing native cannot corrupt candidate buffers or immutable inputs.
        native_args, native_kwargs = workload.build_inputs(
            self.binding, dtype, self.device
        )
        native_buffers = [
            (_backing(native_args[i]), snapshot)
            for i, (_, snapshot) in zip(workload.MUTATED_INPUTS, self.buffers)
        ]
        phase = "geometry"
        try:
            reason = workload.native_layout_reason(
                self.native_op, self.row["config"], dtype
            )
            if reason:
                self.row.update(native_stage=phase, native_reason=reason)
                return None
            phase = "execution"
            torch_device_fn.synchronize()
            before = time.perf_counter()
            result = self.native_op(*native_args, **native_kwargs)
            torch_device_fn.synchronize()
            self.row["native_first_call_wall_ms"] = (
                time.perf_counter() - before
            ) * 1000
            phase = "accuracy"
            assert_result(self.expected, result, *native_args, **native_kwargs)
            self.row["native_accuracy"] = "PASS"
            phase = "timing"
            ms, samples, warmups = _measure(
                self.native_op, native_args, native_kwargs, native_buffers, self.mode
            )
        except Exception as exc:
            self.row.update(
                native_stage=phase,
                native_status="FAIL",
                native_reason=f"native {phase}: {type(exc).__name__}: {exc}",
            )
            if phase == "accuracy":
                self.row["native_accuracy"] = "FAIL"
            return None
        self.row.update(
            native_stage="complete",
            native_status="PASS",
            native_reason=None,
            native_samples_ms=samples,
            native_warmups=warmups,
        )
        return ms

    def run(self):
        # Keep the standard dtype loop, metrics and JSON, and scope offline
        # comparisons to qualified rows with complete physical configuration keys.
        data = report.Config.base_data
        self.rows, self.variants = {}, {}
        self.offline_entry = (data or {}).get(self.op_name, {})
        if report.Config.user_desired_metrics == ["latency"]:
            data = None
        elif data is not None:
            data = {**data, self.op_name: {"variants": self.variants}}
        details = report.TEST_RESULTS.get(self.op_name, {}).get("details", [])
        first = len(details)
        with (
            patch.object(report.Config, "base_data", data),
            workload.single_cpu_thread(),
        ):
            super().run()
        for detail in report.TEST_RESULTS.get(self.op_name, {}).get("details", [])[
            first:
        ]:
            misses = []
            for row in detail["result"]:
                row.update(self.rows[row["shape_detail"]["case"], detail["dtype"]])
                if row["native_reason"]:
                    misses.append(
                        dict(
                            shape=row["shape_detail"],
                            stage=row["native_stage"],
                            reason=row["native_reason"],
                        )
                    )
                if row["compared_speedup"] is None and row["offline_reason"]:
                    row["base_miss_reason"] = row["offline_reason"]
            if misses:
                detail["native_baseline_misses"] = misses
                detail["native_baseline_skip_reason"] = "; ".join(
                    dict.fromkeys(m["reason"] for m in misses)
                )


@pytest.mark.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl
def test_split_kv_compress_norm_rope_insert_sparse_attn_cutedsl():
    SplitBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_NATIVE_SLOT,
        gems_op=gems.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl,
        is_inplace=True,
    ).run()
