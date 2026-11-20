# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Use --mode cudagraph for ACL Graph or --mode operator for synchronized eager."""

from dataclasses import asdict
from functools import partial
from statistics import mean

import pytest
import torch

import flaggems_vllm
from tests.test_mega_mhc import (
    _to_npu,
    load_mhc_cases,
    mhc_eps,
    mhc_inputs,
    mhc_operations,
)

from . import conftest
from .base import Benchmark
from .consts import BenchMode

pytestmark = [
    pytest.mark.mega_mhc,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "ascend", reason="requires Ascend ACL Graph"
    ),
]


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = load_mhc_cases(
            quick=metafunc.config.getoption("--level") == "core",
            shape_file=metafunc.config.getoption("--shape_file"),
        )
        metafunc.parametrize("case", cases, ids=[case.id for case in cases])


@torch.inference_mode()
def test_mega_mhc_latency(case):
    config = conftest.Config
    mode = config.mode
    inputs = _to_npu(mhc_inputs(case, 2026))
    calls = {name: partial(op, *inputs) for name, op, _ in mhc_operations(case)}
    baseline = next(iter(calls))
    if mode == BenchMode.OPERATOR:
        eps = mhc_eps(case)
        context = flaggems_vllm.create_mega_mhc_context(*inputs, rms_eps=eps)
        calls["prepared"] = partial(
            flaggems_vllm.mega_mhc, *inputs, rms_eps=eps, context=context
        )
        for call in calls.values():
            call()  # Exclude initialization and JIT from eager calibration.
        torch.npu.synchronize()

    benchmark = Benchmark("mega_mhc", calls[baseline])
    samples = {name: [] for name in calls}
    names = list(calls)
    for index in range(3):
        first = index % len(names)
        for name in names[first:] + names[:first]:
            samples[name].append(1000 * benchmark.get_latency(calls[name]))

    us = {name: mean(values) for name, values in samples.items()}
    mode_name, metric = {
        BenchMode.CUDAGRAPH: ("acl_graph", "event_us"),
        BenchMode.OPERATOR: ("eager", "host_us"),
        BenchMode.KERNEL: ("kernel", "kernel_us"),
        BenchMode.WRAPPER: ("wrapper", "host_us"),
    }[mode]
    result = {
        "shape": asdict(case),
        "case_id": case.id,
        "mode": mode_name,
        "latencies": {name: {metric: value} for name, value in us.items()},
        "samples": {
            name: [{metric: v} for v in values] for name, values in samples.items()
        },
    }
    if mode == BenchMode.CUDAGRAPH:
        result.update(
            warmup=config.warm_up,
            iterations=config.repetition,
            speedup_event=us[baseline] / us["candidate"],
        )
    elif mode == BenchMode.OPERATOR:
        result.update(
            warmup_ms=config.warm_up,
            repetition_ms=config.repetition,
            speedup_host={
                name: us[baseline] / us[name] for name in ("candidate", "prepared")
            },
        )
    else:
        result["speedup"] = us[baseline] / us["candidate"]
    conftest.update_result("mega_mhc", result)
