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
from baseline.fused_recurrent_gated_delta_rule import OP_NAME, PERF_CASES, build
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


@pytest.fixture(scope="module")
def _performance_records():
    return {}


def _record_performance(row, dtype, entry, records):
    """Expose the measured case to the standard weekly-test report parser."""
    native_valid = (
        row["native_accuracy"] == "PASS" and row["native_performance"] == "PASS"
    )
    dtype = str(dtype)
    if dtype not in records:
        # The weekly parser expects one record containing all cases per dtype.
        records[dtype] = {
            "op_name": OP_NAME,
            "dtype": dtype,
            "mode": "kernel",
            "level": Config.bench_level.value,
            "result": [],
        }
        update_result(OP_NAME, records[dtype])
    record = records[dtype]
    result = {
        "shape_detail": {
            "case": row["case"],
            "entry": entry,
            "inputs": row["config"],
        },
        "latency": row.get("gems_ms"),
        "latency_base": row.get("native_ms") if native_valid else None,
        "speedup": row.get("native_speedup") if native_valid else None,
        "native_accuracy": row["native_accuracy"],
        "native_performance": row["native_performance"],
        "error_msg": row.get("gems_error") or row.get("gems_timing_error"),
    }
    record["result"].append(result)
    if not native_valid:
        reason = row.get("native_error") or row.get("native_timing_error")
        result["native_baseline_skip_reason"] = (
            f"Native accuracy {row['native_accuracy']}: "
            f"{reason or 'no validated native timing'}"
        )
        record["native_baseline_skip_reason"] = "; ".join(
            dict.fromkeys(
                r["native_baseline_skip_reason"]
                for r in record["result"]
                if r.get("native_baseline_skip_reason")
            )
        )


@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.fused_recurrent_gated_delta_rule
@pytest.mark.parametrize("case", PERF_CASES, ids=[c[0] for c in PERF_CASES])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("entry", ["generic", "top"])
def test_perf_fused_recurrent_gated_delta_rule(
    case, dtype, entry, _performance_records
):
    data = build(case, dtype, gems.device)
    state0 = data["initial_state"].clone()
    expected = reference(data)
    candidate = getattr(
        gems.ops if entry == "generic" else gems, "fused_recurrent_gated_delta_rule_fwd"
    )
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
                dict(shape=list(v.shape), stride=list(v.stride()), dtype=str(v.dtype))
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
    _record_performance(row, dtype, entry, _performance_records)
    print(json.dumps(row))
    target = os.environ.get("GDN_RESULTS")
    if target:
        with Path(target).open("a") as f:
            f.write(json.dumps(row) + "\n")
    if candidate_error is not None:
        raise candidate_error
