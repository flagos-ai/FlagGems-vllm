# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""No-Graph, reset-outside-event recurrent GDN benchmark with independent oracle."""

import importlib
import importlib.util
import inspect
import json
import os
import statistics
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from baseline import fused_recurrent_gated_delta_rule as workload
from benchmark import base, consts
from benchmark.conftest import Config, update_result
from flaggems_vllm.runtime import torch_device_fn
from tests.test_FLA.test_fused_recurrent_gated_delta_rule import (
    assert_result,
    reference,
)


def _measure(fn, data, state0):
    for _ in range(5):
        data["initial_state"].copy_(state0)
        fn(**data)
    torch_device_fn.synchronize()
    samples = []
    for _ in range(25):
        data["initial_state"].copy_(state0)
        torch_device_fn.synchronize()
        start = torch_device_fn.Event(enable_timing=True)
        end = torch_device_fn.Event(enable_timing=True)
        start.record()
        fn(**data)
        end.record()
        torch_device_fn.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


class RecurrentGDNBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).resolve().parents[1] / "core_shapes.yaml")
    DEFAULT_DTYPES = workload.DTYPES
    DEFAULT_SHAPES = [(case[0],) for case in workload.PERF_CASES]
    DEFAULT_SHAPE_DESC = "case (baseline.PERF_CASES)"

    def __init__(self):
        self._native_fn = None
        super().__init__(
            op_name=workload.OP_NAME, torch_op=self._native_op, is_inplace=True
        )

    def init_user_config(self):
        if Config.mode != consts.BenchMode.KERNEL:
            raise ValueError(
                "Recurrent GDN uses --mode kernel with no-Graph device events."
            )
        super().init_user_config()

    def _native_op(self, **data):
        # get_latency resolves and validates the actual native per input. Missing
        # native remains N/A without making the optional --base-data match fatal.
        if self._native_fn is None:
            raise RuntimeError(self._native_result["reason"])
        return self._native_fn(**data)

    def get_input_iter(self, dtype):
        reasons = []
        cases = {case[0]: case for case in workload.PERF_CASES}
        for entry in ("generic", "top"):
            for (name,) in self.shapes:
                case = cases[name]
                self._case, self._dtype, self._entry = case, dtype, entry
                self._latencies = None
                self._native_fn = None
                self._native_result = dict(
                    accuracy="NOT_RUN", performance="N/A", reason=None
                )
                self.set_gems(
                    getattr(
                        gems.ops if entry == "generic" else gems,
                        "fused_recurrent_gated_delta_rule_fwd",
                    )
                )
                yield (workload.build(case, dtype, self.device),)
                reason = self._native_result["reason"]
                if reason:
                    reasons.append(f"{entry}/{case[0]}: {reason}")
        if reasons:
            # Benchmark.run owns all metrics and dtype aggregation. This extra
            # metadata lets the weekly parser retain null native speedups.
            update_result(
                self.op_name,
                dict(dtype=str(dtype), native_baseline_skip_reason="; ".join(reasons)),
            )

    def record_shapes(self, *args, **kwargs):
        return dict(
            case=self._case[0],
            entry=self._entry,
            native=self._native_result,
            inputs=super().record_shapes(*args, **kwargs),
        )

    def get_latency(self, op, *args, **kwargs):
        # Keep the established 5 warmups / 25 samples and alternating case order.
        # Base.run requests native and Gems separately; measure the pair once so
        # state restores remain outside each event, unlike the generic timer.
        if self._latencies is None:
            self._latencies = self._measure_pair(kwargs)
        return self._latencies.get("native" if op == self.torch_op else "gems")

    def _measure_pair(self, data):
        case, dtype, entry = self._case, self._dtype, self._entry
        state0 = data["initial_state"].clone()
        expected = reference(data)
        candidate = self.gems_op
        row = dict(
            case=case[0],
            dtype=str(dtype),
            entry=entry,
            backend=gems.vendor_name,
            graph=False,
            unit="ms",
            protocol="device-event-reset-outside-5warmup-25sample-median",
            config={
                k: (
                    dict(
                        shape=list(v.shape), stride=list(v.stride()), dtype=str(v.dtype)
                    )
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in data.items()
            },
            candidate_accuracy="NOT_RUN",
            native_accuracy="NOT_RUN",
            native_performance="N/A",
            native_ms=None,
            native_speedup=None,
            before_accuracy="NOT_RUN",
        )
        functions = {"gems": candidate}
        try:
            module = importlib.import_module(
                "vllm.model_executor.layers.fla.ops.fused_recurrent"
            )
            native = module.fused_recurrent_gated_delta_rule
            assert native is not candidate
            self._native_fn = native
            row["native_file"] = inspect.getfile(native)
            row["native_module"] = native.__module__
            indices = data["ssm_state_indices"]
            if indices.ndim == 2 and indices.stride(1) != 1:
                # vLLM 0.24 reads table + row*stride(0) + token, ignoring
                # stride(1). This caused device illegal accesses in the native
                # baseline; retain the case, but never launch that unsafe path.
                row["native_accuracy"] = "NOT_RUN"
                row["native_error"] = (
                    "Unsupported token stride: native ignores stride_indices_tok; observed device illegal access"
                )
            else:
                functions["native"] = native
        except Exception as exc:
            row["native_error"] = repr(exc)
        before = os.environ.get("GDN_BEFORE")
        if (
            before
            and entry == "generic"
            and case[0] in ("decode1", "decode16", "decode64", "decode256")
        ):
            spec = importlib.util.spec_from_file_location("gdn_before", before)
            old = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(old)
            functions["before"] = old.fused_recurrent_gated_delta_rule_fwd
        # Alternating per-case order avoids always warming one implementation first.
        if case[0] in ("decode16", "decode256", "mtp", "tail"):
            functions = dict(reversed(list(functions.items())))
        candidate_error = None
        for name, fn in functions.items():
            data["initial_state"].copy_(state0)
            key = "candidate_accuracy" if name == "gems" else name + "_accuracy"
            try:
                result = fn(**data)
                assert_result(result, expected, dtype)
                row[key] = "PASS"
            except Exception as exc:
                row[key] = "FAIL"
                row[name + "_error"] = repr(exc)
                if name == "gems":
                    candidate_error = exc
                # Retain the failed accuracy result, but never time an invalid baseline.
                continue
            try:
                # Only implementations passing the oracle are eligible for timing.
                values = _measure(fn, data, state0)
                data["initial_state"].copy_(state0)
                assert_result(fn(**data), expected, dtype)
                row[name + "_samples_ms"] = values
                row[name + "_ms"] = statistics.median(values)
                if name == "native":
                    row["native_performance"] = "PASS"
            except Exception as exc:
                row[name + "_timing_error"] = repr(exc)
                if isinstance(exc, AssertionError):
                    row[key] = "FAIL"
                    row[name + "_error"] = repr(exc)
                if name == "gems":
                    candidate_error = exc
        if (
            row["candidate_accuracy"] == "PASS"
            and row["native_accuracy"] == "PASS"
            and (row.get("native_ms") or 0) > 0
            and (row.get("gems_ms") or 0) > 0
        ):
            row["native_speedup"] = row["native_ms"] / row["gems_ms"]
            row["selected_source"] = "native"
        else:
            row["selected_source"] = None
        self._native_result.update(
            accuracy=row["native_accuracy"],
            performance=row["native_performance"],
            reason=row.get("native_error") or row.get("native_timing_error"),
        )
        print(json.dumps(row))
        target = os.environ.get("GDN_RESULTS")
        if target:
            with Path(target).open("a") as f:
                f.write(json.dumps(row) + "\n")
        if candidate_error is not None:
            raise candidate_error
        return {"native": row["native_ms"], "gems": row.get("gems_ms")}


@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.fused_recurrent_gated_delta_rule
def test_perf_fused_recurrent_gated_delta_rule():
    RecurrentGDNBenchmark().run()
