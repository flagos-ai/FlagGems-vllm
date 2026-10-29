# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from benchmark.conftest import update_result
from flaggems_vllm.runtime import torch_device_fn

spec = importlib.util.spec_from_file_location(
    "kda_kkt_workload",
    Path(__file__).resolve().parents[2] / "baseline/chunk_kda_scaled_dot_kkt_fwd.py",
)
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


def event_samples(fn):
    for _ in range(5):
        fn()
    torch_device_fn.synchronize()
    samples = []
    for _ in range(15):
        start = torch_device_fn.Event(enable_timing=True)
        end = torch_device_fn.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


@pytest.mark.chunk_kda_scaled_dot_kkt_fwd
@pytest.mark.parametrize("dtype", workload.DTYPES)
@pytest.mark.parametrize("binding", workload.grid())
def test_chunk_kda_scaled_dot_kkt_fwd_perf(binding, dtype):
    args, kw = workload.build_inputs(binding, dtype, gems.device)
    candidate = gems.chunk_kda_scaled_dot_kkt_fwd
    started = time.monotonic()
    actual = candidate(*args, **kw)
    torch_device_fn.synchronize()
    cold = time.monotonic() - started
    row = dict(
        config=workload.config(binding, dtype),
        backend=gems.vendor_name,
        graph=False,
        protocol="device-event/5warmup/15samples/median-ms",
        cold_seconds=cold,
        gems_samples_ms=event_samples(lambda: candidate(*args, **kw)),
    )
    row["gems_ms"] = statistics.median(row["gems_samples_ms"])
    update_result(workload.OP_NAME, row)
    # Native is measured only after independent numerical checks by the task harness.
    row["native_status"] = "NOT_RUN"
    output = os.environ.get("KDA_KKT_RESULTS")
    if output:
        with open(output, "a") as f:
            f.write(json.dumps(row) + "\n")
    assert all(torch.isfinite(x).all().item() for x in actual)
    assert row["gems_ms"] > 0
