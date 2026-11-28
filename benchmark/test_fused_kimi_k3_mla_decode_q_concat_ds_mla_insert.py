# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""No-Graph timing of one public fp8_ds_mla call with two restored outputs."""

import math
import re
import statistics
import time
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from baseline import fused_kimi_k3_mla_decode_q_concat_ds_mla_insert as workload
from benchmark import conftest as report
from benchmark import consts
from benchmark.base import Benchmark
from flaggems_vllm.runtime import torch_device_fn
from tests.test_fused_kimi_k3_mla_decode_q_concat_ds_mla_insert import (
    assert_cache_equal,
    expected_outputs,
)


def _backing(tensor):
    size = tensor.untyped_storage().nbytes() // tensor.element_size()
    return tensor.as_strided((size,), (1,), 0)


def _readonly_inputs(args):
    seen = set()
    for index, tensor in enumerate(args):
        if isinstance(tensor, torch.Tensor) and index not in workload.MUTATED_INPUTS:
            data = _backing(tensor)
            if data.data_ptr() not in seen:
                seen.add(data.data_ptr())
                yield data


@workload.single_cpu_thread()
def _expected(args):
    q, cache = expected_outputs(args)
    original = args[5]
    storage = _backing(original).cpu().clone()
    storage.as_strided(
        original.shape, original.stride(), original.storage_offset()
    ).copy_(cache)
    readonly = [t.cpu().clone() for t in _readonly_inputs(args)]
    return q, cache, storage, readonly


@workload.single_cpu_thread()
def _check(op, args, expected):
    q, cache, storage, readonly = expected
    tensors = [t for t in args if isinstance(t, torch.Tensor)]

    def metadata():
        return [
            (t.data_ptr(), t.storage_offset(), t.shape, t.stride(), t.dtype)
            for t in tensors
        ]

    before = metadata()
    result = op(*args)
    torch_device_fn.synchronize()
    assert result is None and metadata() == before
    torch.testing.assert_close(
        args[4].cpu().view(torch.int16), q.view(torch.int16), rtol=0, atol=0
    )
    assert_cache_equal(args[5], cache)
    torch.testing.assert_close(_backing(args[5]).cpu(), storage, rtol=0, atol=0)
    observed = list(_readonly_inputs(args))
    assert len(observed) == len(readonly)
    for tensor, ref in zip(observed, readonly):
        torch.testing.assert_close(
            tensor.cpu().view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
        )


def _measure(op, args, saved):
    # The base kernel timer has no untimed pre-call reset hook. Direct device
    # events also keep this path free of backend-specific graph capture.
    outputs = [_backing(args[i]) for i in workload.MUTATED_INPUTS]

    def restore():
        for tensor, original in zip(outputs, saved):
            tensor.copy_(original)
        torch_device_fn.synchronize()

    start = torch_device_fn.Event(enable_timing=True)
    end = torch_device_fn.Event(enable_timing=True)
    deadline = time.perf_counter() + report.Config.warm_up / 1000
    warmups = 0
    while warmups < 5 or time.perf_counter() < deadline:
        restore()
        op(*args)
        torch_device_fn.synchronize()
        warmups += 1
    deadline = time.perf_counter() + report.Config.repetition / 1000
    samples = []
    while len(samples) < 25 or time.perf_counter() < deadline:
        restore()
        start.record()
        op(*args)
        end.record()
        torch_device_fn.synchronize()
        samples.append(float(start.elapsed_time(end)))
    if not all(_positive(sample) for sample in samples):
        raise ValueError(f"invalid device event samples: {samples}")
    return statistics.median(samples), samples, warmups


def _native_slot(*args):
    raise RuntimeError("native dispatch must be resolved before timing")


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _matches_chip(model, observed):
    if not isinstance(model, str) or not isinstance(observed, str):
        return False
    model = model.lower().removeprefix("nvidia ").strip()
    return (
        bool(model)
        and re.search(rf"(?<![a-z0-9]){re.escape(model)}(?![a-z0-9])", observed.lower())
        is not None
    )


class DecodeBenchmark(Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_DTYPES = workload.DTYPES
    DEFAULT_SHAPES = workload.CASES
    DEFAULT_SHAPE_DESC = ", ".join(workload.FIELDS)
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def set_shapes(self, shape_file_path=None):
        if (
            shape_file_path
            and Path(shape_file_path).resolve()
            != Path(self.DEFAULT_SHAPE_FILES).resolve()
        ):
            super().set_shapes(shape_file_path)
        else:
            self.shapes = workload.CASES
        if not self.shapes or any(
            len(row) != len(workload.FIELDS) for row in self.shapes
        ):
            raise ValueError(f"expected shape rows [{self.DEFAULT_SHAPE_DESC}]")
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("duplicate decode benchmark configurations")

    def init_user_config(self):
        super().init_user_config()
        if self.mode != consts.BenchMode.KERNEL:
            raise ValueError(
                "fp8_ds_mla decode benchmark requires --mode kernel (no Graph)"
            )
        self.scaling_dtype = torch.float32  # RoPE and scale arithmetic
        self.native_op = None
        self.native_reason = "native timing disabled by --metrics"
        if "latency_base" in self.to_bench_metrics:
            self.native_op = workload.native()
            self.native_reason = workload.NATIVE_UNAVAILABLE_REASON
        self.protocol = dict(
            graph=False,
            timer="current-device events enclosing one public call",
            reset="full mqa_q and cache allocations, synchronized before start",
            unit="ms",
            reduction="median",
            warmup_budget_ms=report.Config.warm_up,
            sample_budget_ms=report.Config.repetition,
            budget_clock="wall time including untimed restores",
            minimum_warmups=5,
            minimum_samples=25,
        )

    def run(self):
        self.records = {}
        self.offline = report.Config.base_data
        first = len(report.TEST_RESULTS.get(self.op_name, {}).get("details", []))
        # Only qualified offline records enter Benchmark.run(). Restore the
        # original CLI configuration even when a candidate assertion fails.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(report.Config, "base_data", None)
            super().run()
        for detail in report.TEST_RESULTS.get(self.op_name, {}).get("details", [])[
            first:
        ]:
            misses = []
            for row in detail["result"]:
                record = self.records[row["shape_detail"]["case"]]
                row.update(record)
                if row.get("base_latency_ms") and row["latency"]:
                    row["base_raw_speedup"] = row["base_latency_ms"] / row["latency"]
                if row["native_reason"]:
                    misses.append(
                        dict(shape=row["shape_detail"], reason=row["native_reason"])
                    )
                    if row["compared_speedup"] is None:
                        row["base_miss_reason"] = "; ".join(
                            str(r)
                            for r in (row["native_reason"], row["offline_reason"])
                            if r
                        )
            if misses:
                detail["native_baseline_misses"] = misses
                detail["native_baseline_skip_reason"] = "; ".join(
                    dict.fromkeys(m["reason"] for m in misses)
                )

    def record_shapes(self, *args, **kwargs):
        return {
            "case": workload.key_shape(self.binding),
            "inputs": super().record_shapes(*args, **kwargs),
        }

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self.binding = dict(zip(workload.FIELDS, shape))
            self.dtype = dtype
            args, kwargs = workload.build_inputs(self.binding, dtype, self.device)
            self.expected = _expected(args)
            self.saved = [_backing(args[i]).clone() for i in workload.MUTATED_INPUTS]
            _check(self.gems_op, args, self.expected)
            config = workload.config(self.binding, dtype)
            self.record = {
                "config": config,
                "candidate_accuracy": "PASS",
                "native_status": "NOT_RUN" if self.native_reason else "PENDING",
                "native_reason": self.native_reason,
                "native_provenance": (
                    workload.NATIVE_PROVENANCE
                    if "latency_base" in self.to_bench_metrics
                    else None
                ),
                "offline_reason": None,
                "timing_protocol": self.protocol,
            }
            try:
                offline, reason = self._qualified_offline(args)
            except (AttributeError, TypeError, ValueError) as exc:
                offline, reason = None, f"invalid offline metadata: {exc}"
            report.Config.base_data = offline
            self.record["offline_reason"] = reason
            self.records[config["case"]] = self.record
            yield (*args, kwargs)

    def get_latency(self, op, *args, **kwargs):
        if op is self.torch_op:
            if self.native_op is None:
                return None
            # A broken native must not overwrite the candidate's inputs via a
            # read-only alias. Rebuild the same seeded workload independently.
            native_args, _ = workload.build_inputs(
                self.binding, self.dtype, self.device
            )
            phase = "device dispatch"
            try:
                key = torch._C._dispatch_key_for_device(native_args[0].device.type)
                if not torch._C._dispatch_has_kernel_for_dispatch_key(
                    f"_C::{self.op_name}", key
                ):
                    self.record.update(
                        native_status="NOT_RUN",
                        native_reason=f"native has no {key} device kernel",
                    )
                    return None
                phase = "qualification"
                _check(self.native_op, native_args, self.expected)
                phase = "timing"
                latency, samples, warmups = _measure(
                    self.native_op, native_args, self.saved
                )
            except (Exception, pytest.fail.Exception) as exc:
                self.record.update(
                    native_status="FAIL", native_reason=f"native {phase}: {exc!r}"
                )
                return None
            self.record.update(
                native_status="PASS", native_samples_ms=samples, native_warmups=warmups
            )
            return latency
        latency, samples, warmups = _measure(op, args, self.saved)
        self.record.update(candidate_samples_ms=samples, candidate_warmups=warmups)
        return latency

    def _qualified_offline(self, args):
        if "speedup" not in self.to_bench_metrics:
            return None, "comparison disabled by --metrics"
        if self.offline is None:
            return None, "no offline baseline supplied"
        config = self.record["config"]
        row = (
            self.offline.get(self.op_name, {})
            .get("shapes", {})
            .get(config["case"], {})
            .get(str(self.dtype), {})
        )
        if row.get("config") != config or row.get("native_status") != "PASS":
            return (
                None,
                "no numerically qualified offline record for this exact configuration",
            )
        protocol = row.get("timing_protocol")
        if isinstance(protocol, dict):
            valid_timer = all(
                protocol.get(k) == self.protocol[k]
                for k in ("graph", "timer", "reset", "unit", "reduction")
            )
        else:
            valid_timer = (
                protocol
                == "no-graph-device-event-reset-outside-5warmup-25sample-median"
            )
        if not valid_timer or not _positive(row.get("latency_ms")):
            return None, "invalid offline latency or timing protocol"
        qualified = self.offline.get("_qualification", {}).get("chip")
        source = (
            self.offline.get("_environment", {})
            .get("hardware", {})
            .get("gpu", {})
            .get("name")
            or qualified
        )
        if not source:
            return None, "offline source chip is unidentified"
        self.record.update(
            base_latency_ms=row["latency_ms"], base_reference_chip=source
        )
        scaling = self.offline.get("_scaling_factors", {})
        reference = scaling.get("reference_chip")
        if not _matches_chip(reference, source) or (
            qualified and not _matches_chip(reference, qualified)
        ):
            return None, "scaling reference does not match the measured source chip"
        try:
            target = torch_device_fn.get_device_name(self.device)
        except (AttributeError, TypeError, RuntimeError):
            target = consts._SELF_CHIP
        chips = [
            c
            for c in scaling.get("chips", [])
            if c.get("vendor") == gems.vendor_name
            and _matches_chip(c.get("chip"), target)
        ]
        if len(chips) != 1:
            return None, "no unique scaling entry for the measured target chip"
        unit = row.get("bottle_neck_unit")
        if unit not in ("cuda", "mem"):
            return None, "offline row lacks a verified vector or memory bottleneck"
        if (
            unit == "mem"
            and gems.vendor_name == "metax"
            and chips[0]["chip"] == "C550"
            and not _matches_chip("H800", source)
        ):
            # The common lookup calibrates C550 bandwidth against H800.
            return None, "C550 bandwidth calibration requires an H800 source"
        # Map only this verified row to the base class's direct tensor-shape
        # key. Its ordinary lookup and scaling code now handles both ratios.
        data = {
            "_scaling_factors": {**scaling, "chips": chips},
            self.op_name: {
                "shapes": {str(list(args[0].shape)): {str(self.dtype): row}}
            },
        }
        factor = consts.lookup_scaling_factor(
            data, gems.vendor_name, unit, self.scaling_dtype
        )
        if not _positive(factor):
            return None, "no positive scaling factor for this chip and resource"
        self.record.update(
            base_target_chip=chips[0]["chip"],
            base_bottleneck=unit,
            base_scaling_factor=factor,
        )
        return data, None


@pytest.mark.fused_kimi_k3_mla_decode_q_concat_ds_mla_insert
def test_fused_kimi_k3_mla_decode_q_concat_ds_mla_insert():
    DecodeBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_native_slot,
        gems_op=gems.fused_kimi_k3_mla_decode_q_concat_ds_mla_insert,
        is_inplace=True,
    ).run()
