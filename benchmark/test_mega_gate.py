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

import json
from dataclasses import asdict
from statistics import mean

import pytest

import flaggems_vllm
from tests.test_mega_gate import (
    gate_inputs,
    gate_operations,
    load_gate_cases,
    time_graphs,
)

from . import conftest

pytestmark = [
    pytest.mark.mega_gate,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "ascend", reason="requires Ascend ACL Graph"
    ),
]


def run_case(op, case, operations, make_inputs):
    config = conftest.Config
    samples = time_graphs(
        operations,
        make_inputs(case, 2026),
        config.warm_up,
        config.repetition,
    )
    latencies = {
        name: {
            key: mean(sample[key] for sample in rounds)
            for key in ("event_us", "host_us")
        }
        for name, rounds in samples.items()
    }
    result = {
        "shape": asdict(case),
        "case_id": case.id,
        "mode": "acl_graph",
        "warmup": config.warm_up,
        "iterations": config.repetition,
        "latencies": latencies,
        "samples": samples,
    }
    if "candidate" in latencies:
        baseline = next(name for name in latencies if name != "candidate")
        result["speedup_event"] = (
            latencies[baseline]["event_us"] / latencies["candidate"]["event_us"]
        )
    print(json.dumps(result))
    conftest.update_result(op, result)


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = load_gate_cases(
            metafunc.config.getoption("--shape_file"),
            quick=metafunc.config.getoption("--level") == "core",
        )
        metafunc.parametrize("case", cases, ids=[case.id for case in cases])


def test_mega_gate_graph_latency(case):
    run_case("mega_gate", case, gate_operations(case), gate_inputs)
