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

"""Benchmark prepared MegaGate. Use --mode cudagraph for ACL Graph, or --mode operator for eager."""

from functools import partial
from statistics import mean

import pytest
import torch

import flaggems_vllm
from tests.test_mega_gate import (
    gate_baseline,
    gate_inputs,
    gate_kwargs,
    load_gate_cases,
    v4_route_inputs,
)

from . import conftest
from .base import Benchmark

pytestmark = [
    pytest.mark.mega_gate,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="requires Ascend"),
]


def pytest_generate_tests(metafunc):
    cases = load_gate_cases(
        metafunc.config.getoption("--shape_file"),
        quick=metafunc.config.getoption("--level") == "core",
    )
    matrix = [(case, "score") for case in cases]
    for case in cases:
        if case.profile == "v4-flash":
            matrix.extend((case, mode) for mode in ("hash", "vision", "vision-hash"))
    metafunc.parametrize(
        ("case", "route"), matrix, ids=[f"{case.id}-{mode}" for case, mode in matrix]
    )


@torch.inference_mode()
def test_mega_gate_latency(case, route):
    from vllm_ascend.utils import enable_custom_op

    from flaggems_vllm.runtime.backend._ascend.ops.mega_gate import prepare_mega_gate

    assert enable_custom_op(), "vLLM-Ascend custom operators could not be loaded"
    x, weight, bias = (
        t.npu() if t is not None else None for t in gate_inputs(case, 2026)
    )
    routing = {}
    if route != "score":
        routing = v4_route_inputs(case, route, x.device)
        if "hash" in route:
            bias = None
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        kwargs = dict(**gate_kwargs(case), **routing)
        handle = prepare_mega_gate(x, weight, bias, **kwargs)
        candidate = partial(
            flaggems_vllm.mega_gate, x, weight, bias, prepared=handle, **kwargs
        )
        native = partial(gate_baseline, case, x, weight, bias, **routing)

        # Match the model-facing output contract; charge any conversions to MegaGate.
        def candidate_for_model():
            ids, weights = candidate()
            return ids.int(), weights.to(case.logits_dtype)

        measured_candidate = (
            candidate if weight.dtype == torch.float32 else candidate_for_model
        )
        benchmark = Benchmark("mega_gate", native)
        samples = {"native": [], "prepared": []}
        for index in range(3):
            pairs = (("native", native), ("prepared", measured_candidate))
            for name, operation in pairs if index % 2 == 0 else reversed(pairs):
                samples[name].append(1000 * benchmark.get_latency(operation))
    native_us, prepared_us = mean(samples["native"]), mean(samples["prepared"])
    conftest.update_result(
        "mega_gate",
        {
            "case_id": case.id,
            "route": route,
            "mode": (
                "aclgraph"
                if conftest.Config.mode.value == "cudagraph"
                else conftest.Config.mode.value
            ),
            "input_dtype": str(x.dtype),
            "weight_dtype": str(weight.dtype),
            "logits_dtype": str(case.logits_dtype),
            "tokens": case.tokens,
            "native_us": native_us,
            "prepared_us": prepared_us,
            "speedup": native_us / prepared_us,
            "samples_us": samples,
        },
    )
