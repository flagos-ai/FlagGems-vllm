# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""No-Graph KDA native/Gems pairs with full state reset outside each event."""

import inspect
import json
import os
import statistics
from pathlib import Path

import pytest

import flaggems_vllm as gems
from baseline import fused_recurrent_kda as baseline
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


@pytest.mark.fused_recurrent_kda
@pytest.mark.parametrize(
    "case", baseline.PERF_CASES, ids=[case.name for case in baseline.PERF_CASES]
)
@pytest.mark.parametrize("dtype", baseline.DTYPES)
def test_fused_recurrent_kda(case, dtype):
    data = baseline.build(case, dtype, gems.device)
    state = data["initial_state"].clone()
    expected = reference(data)
    row = dict(
        case=case.name,
        dtype=str(dtype),
        backend=gems.vendor_name,
        graph=False,
        unit="ms",
        protocol="device-event-reset-outside-5warmup-25sample-median",
        config=baseline.config({"case": case}, dtype),
        candidate_accuracy="NOT_RUN",
        native_accuracy="NOT_RUN",
        selected_source=None,
    )
    functions = {"gems": gems.fused_recurrent_kda}
    original = None if gems.vendor_name == "ascend" else baseline.native()
    if gems.vendor_name == "ascend":
        row["native_accuracy"] = "SKIP"
        row["native_error"] = "Ascend native KDA disabled: native accuracy is unstable"
    elif original is None:
        row["native_error"] = baseline.NATIVE_UNAVAILABLE_REASON
    else:
        assert original is not gems.fused_recurrent_kda
        row["native_module"], row["native_file"] = original.__module__, inspect.getfile(
            original
        )
        reason = baseline.unsupported_reason({"case": case}, dtype)
        if reason:
            row["native_error"] = reason
        else:
            functions["native"] = original
    if case.name in ("decode16", "decode256", "outofplace"):
        functions = dict(reversed(list(functions.items())))
    candidate_error = None
    for name, fn in functions.items():
        key = "candidate_accuracy" if name == "gems" else "native_accuracy"
        data["initial_state"].copy_(state)
        try:
            assert_result(fn(**data), expected)
            row[key] = "PASS"
        except Exception as exc:
            row[key], row[name + "_error"] = "FAIL", repr(exc)
            if name == "gems":
                candidate_error = exc
        try:
            samples = _measure(fn, data, state)
            row[name + "_samples_ms"] = samples
            row[name + "_ms"] = statistics.median(samples)
        except Exception as exc:
            row[name + "_timing_error"] = repr(exc)
            if name == "gems":
                candidate_error = candidate_error or exc
    if (
        row["candidate_accuracy"] == row["native_accuracy"] == "PASS"
        and row.get("gems_ms", 0) > 0
        and row.get("native_ms", 0) > 0
    ):
        row["native_speedup"] = row["native_ms"] / row["gems_ms"]
        row["selected_source"] = "native"
    target = os.environ.get("RKDA_RESULTS")
    if target:
        with Path(target).open("a") as f:
            f.write(json.dumps(row) + "\n")
    if candidate_error is not None:
        raise candidate_error
