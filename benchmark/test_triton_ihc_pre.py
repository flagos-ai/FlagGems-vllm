# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark the public iHC pre call against qualified native or H800 timings."""

import math
import os
import shlex
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from baseline import triton_ihc_pre as workload
from tests.test_triton_ihc_pre import ihc_pre_reference
from tools.collect_baseline_nvidia import _measure_latency

from .base import Benchmark
from .conftest import Config, update_result
from .consts import BenchMode, lookup_base_record_ex

_FIELDS = ("tokens", "channels", "hidden", "layout")
_H800_PROTOCOL = "triton-do-bench-no-graph-25ms-warmup-100ms-median"


def _native_slot(*args, **kwargs):
    raise RuntimeError("native must be qualified before timing")


@workload._cpu_single_thread()
def _check_output(output, args, expected):
    assert isinstance(output, tuple) and len(output) == 2
    for actual, ref, (rtol, atol) in zip(
        output, expected, ((1e-2, 2e-2), (1e-5, 2e-5))
    ):
        assert actual.shape == ref.shape and actual.dtype == ref.dtype
        assert actual.device == args[0].device
        torch.testing.assert_close(
            actual.cpu(), ref, rtol=rtol, atol=atol, equal_nan=True
        )


@workload._cpu_single_thread()
def _check_inputs(args, snapshots, *, restore=False):
    changed = []
    for index, (value, saved) in enumerate(zip(args, snapshots)):
        if not torch.equal(
            value.detach().cpu().contiguous().view(torch.uint8),
            saved.contiguous().view(torch.uint8),
        ):
            changed.append(index)
            if restore:
                value.copy_(saved)
    if changed and not restore:
        raise AssertionError(f"mutated positional inputs {changed}")


def _offline_status(data, key, dtype, detail):
    if data is None:
        return {"status": "N/A", "reason": "no offline NVIDIA baseline supplied"}
    record, _ = lookup_base_record_ex(data, workload.OP_NAME, dtype, detail, key)
    if not isinstance(record, dict):
        return {
            "status": "N/A",
            "reason": f"no offline NVIDIA record for shape key {key} and dtype {dtype}",
        }
    reference = record.get("config")
    fields = (
        "inputs",
        "outputs",
        "dims",
        "mutation",
        "input_aliasing",
        "arithmetic",
        "tolerance",
        "timing_scope",
    )
    latency = record.get("latency_ms")
    protocol = record.get("timing_protocol")
    if not isinstance(reference, dict) or any(
        reference.get(f) != detail["config"][f] for f in fields
    ):
        reason = "offline NVIDIA configuration differs from the benchmark"
    elif record.get("graph", False) or record.get("status") not in (
        None,
        "pass",
        "PASS",
        "ok",
    ):
        reason = "offline NVIDIA record is invalid or uses Graph"
    elif (
        type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0
    ):
        reason = "offline NVIDIA record lacks a positive finite latency"
    elif protocol != _H800_PROTOCOL:
        reason = f"offline NVIDIA timing protocol {protocol!r} does not match {_H800_PROTOCOL!r}"
    else:
        return {
            "status": "MATCH",
            "shape_key": key,
            "latency_ms": latency,
            "timing_protocol": protocol,
        }
    return {"status": "N/A", "reason": reason}


class TritonIhcPreBenchmark(Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPE_DESC = ", ".join(_FIELDS)
    DEFAULT_SHAPES = [tuple(row[f] for f in _FIELDS) for row in workload.grid()]

    def set_shapes(self, shape_file=None):
        if shape_file and Path(shape_file).resolve() != Path(self.DEFAULT_SHAPE_FILES):
            super().set_shapes(shape_file)
        else:
            self.shapes = list(self.DEFAULT_SHAPES)

    def run(self):
        data = Config.base_data
        entry = (data or {}).get(self.op_name) or {}
        self.offline_data = data
        # Dense and strided inputs share tensor dimensions. Give each full key its own variant.
        if "shapes" in entry and "variants" not in entry:
            variants = {
                key: {"shapes": {key: rows}} for key, rows in entry["shapes"].items()
            }
            self.offline_data = {**data, self.op_name: {"variants": variants}}
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Config, "base_data", None)
            super().run()

    def get_input_iter(self, dtype):
        native = None
        reason = "native timing disabled by --metrics"
        if "latency_base" in self.to_bench_metrics:
            try:
                native = workload.native()
                reason = workload.NATIVE_UNAVAILABLE_REASON
            except Exception as exc:
                reason = f"native resolution failed: {exc!r}"
            if native is self.gems_op:
                reason = "native resolves to the Gems candidate"
            elif not callable(native):
                reason = reason or "installed native unavailable"
        self.native = native
        misses = []
        for shape in self.shapes:
            binding = dict(zip(_FIELDS, shape))
            self.base_variant = workload.key_shape(binding)
            with workload._cpu_single_thread():
                args, kwargs = workload.build_inputs(binding, dtype, self.device)
                self.snapshots = tuple(t.detach().cpu().clone() for t in args[:4])
                self.expected = ihc_pre_reference(*self.snapshots, *args[4:])
                _check_output(self.gems_op(*args, **kwargs), args, self.expected)
                _check_inputs(args, self.snapshots)
            self.detail = dict(
                arguments=super().record_shapes(*args, **kwargs),
                shape_key=self.base_variant,
                config=workload.config(binding, dtype),
                native={"status": "N/A" if reason else "VALIDATING", "reason": reason},
                offline_nvidia=None,
                timing=dict(
                    scope="complete public callable, allocation and contiguous copies",
                    framework="Benchmark.get_latency / operator",
                    graph=False,
                    unit="ms",
                    statistic="synchronized wall-clock mean",
                    protocol=f"operator-wall-clock-mean-{Config.warm_up}ms-{Config.repetition}ms",
                    warmup_budget_ms=Config.warm_up,
                    repetition_budget_ms=Config.repetition,
                ),
                active_rows=len(self.shapes) * len(self.to_bench_dtypes),
            )
            Config.base_data = None
            yield (*args, kwargs)
            del self.snapshots, self.expected
            if self.detail["native"]["reason"]:
                misses.append(
                    dict(
                        shape_key=self.base_variant,
                        reason=self.detail["native"]["reason"],
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
        return self.detail

    def get_latency(self, op, *args, **kwargs):
        native = self.detail["native"]
        is_native = op is self.torch_op
        if is_native and native["reason"]:
            return None
        phase = "qualification"
        try:
            if is_native:
                op = self.native
                _check_output(op(*args, **kwargs), args, self.expected)
                _check_inputs(args, self.snapshots)
                phase = "timing"
            if (
                not is_native
                and native["status"] == "N/A"
                and "latency_base" in self.to_bench_metrics
            ):
                offline = _offline_status(
                    self.offline_data, self.base_variant, args[0].dtype, self.detail
                )
                self.detail["offline_nvidia"] = offline
                if offline["status"] == "MATCH":
                    Config.base_data = self.offline_data
                self.detail["timing"].update(
                    framework="triton.testing.do_bench",
                    statistic="device-event median",
                    protocol=_H800_PROTOCOL,
                    warmup_budget_ms=25,
                    repetition_budget_ms=100,
                )
                latency = _measure_latency(workload, op, args, kwargs)
            else:
                latency = super().get_latency(op, *args, **kwargs)
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"invalid latency: {latency!r}")
            _check_inputs(args, self.snapshots)
        except Exception as exc:
            if not is_native:
                raise
            _check_inputs(args, self.snapshots, restore=True)
            native.update(status="N/A", reason=f"native {phase} failed: {exc!r}")
            return None
        if is_native:
            native.update(status="PASS", reason=None)
        return latency


@pytest.mark.triton_ihc_pre
def test_triton_ihc_pre(pytestconfig, monkeypatch):
    if not Config.query and Config.mode != BenchMode.OPERATOR:
        opts = (
            *pytestconfig.getini("addopts"),
            *shlex.split(os.environ.get("PYTEST_ADDOPTS", "")),
            *pytestconfig.invocation_params.args,
        )
        if Config.mode != BenchMode.KERNEL or any(
            arg.partition("=")[0] in ("--mode", "--fg_mode") for arg in opts
        ):
            raise ValueError(
                f"iHC pre requires --mode operator (no Graph), got {Config.mode.value}"
            )
        monkeypatch.setattr(Config, "mode", BenchMode.OPERATOR)
    TritonIhcPreBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_native_slot,
        gems_op=gems.triton_ihc_pre,
        dtypes=workload.DTYPES,
    ).run()
