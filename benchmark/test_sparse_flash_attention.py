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

"""Matched-input shared-KV device timing; includes gather/attention/combine where used.

Baseline = installed vLLM-Ascend _C_ascend SFA with value=key (MLA absorb).
SpeedUp = baseline / candidate. Two alternating-order rounds, median; cold L2.
Host allocations and Python dispatch are not included in this device metric.
"""

import math
import statistics
from dataclasses import asdict

import pytest
import torch

import flaggems_vllm
from tests.sparse_flash_attention_utils import (
    CASES,
    MODES,
    load_vllm_sfa,
    make_inputs,
    vllm_inputs,
)

from .conftest import TEST_RESULTS, Config, emit_record_logger, update_result
from .consts import BenchmarkMetrics, BenchmarkResult

pytestmark = [
    pytest.mark.sparse_flash_attention,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "ascend", reason="requires an Ascend NPU"
    ),
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", MODES)
@torch.inference_mode()
def test_sparse_flash_attention_perf(case, mode):
    from triton.backends.ascend.testing import do_bench_npu

    inputs = make_inputs(case, mode, flaggems_vllm.device)
    reference_inputs = vllm_inputs(inputs)
    reference_op = load_vllm_sfa()

    def candidate():
        return flaggems_vllm.sparse_flash_attention(**reference_inputs)[0]

    def reference():
        return reference_op(**reference_inputs)[0]

    torch.testing.assert_close(
        candidate(), reference(), atol=0.01, rtol=0.01, equal_nan=False
    )
    # Initialize both runtime paths before the profiler counts kernel rows.
    for _ in range(5):
        candidate()
        reference()
    torch.npu.synchronize()
    samples = []
    for rep in range(2):
        funcs = (candidate, reference) if rep == 0 else (reference, candidate)
        pair = [
            float(do_bench_npu(fn, warmup=5, active=30, clear_l2_cache=True)) * 1000
            for fn in funcs
        ]
        sample = pair if rep == 0 else pair[::-1]
        samples.append(sample)
        print(
            f"SFA_SAMPLE batch={case[0]} capacity={case[1]} valid={case[2]} "
            f"mode={mode} round={rep + 1} "
            f"candidate_us={sample[0]:.4f} baseline_us={sample[1]:.4f}"
        )
    ours = statistics.median(x[0] for x in samples)
    ref = statistics.median(x[1] for x in samples)
    assert math.isfinite(ours) and ours > 0, "invalid candidate timing"
    assert math.isfinite(ref) and ref > 0, "invalid reference timing"
    print(
        f"SFA_RESULT batch={case[0]} capacity={case[1]} valid={case[2]} "
        f"mode={mode} baseline=vllm_ascend._C_ascend.npu_sparse_flash_attention "
        f"kv=shared candidate_us={ours:.4f} baseline_us={ref:.4f} "
        f"speedup={ref / ours:.4f}"
    )
    # The repository report uses milliseconds; samples above are microseconds.
    metrics = BenchmarkMetrics(
        shape_detail={
            "query": tuple(inputs["query"].shape),
            "key": tuple(inputs["key"].shape),
            "valid_kv_length": case[2],
            "index_distribution": mode,
            "baseline": "vllm_ascend._C_ascend.npu_sparse_flash_attention",
        },
        latency_base=ref / 1000,
        latency=ours / 1000,
        speedup=ref / ours,
    )
    result = BenchmarkResult(
        op_name="sparse_flash_attention",
        dtype=str(inputs["query"].dtype),
        mode=Config.mode.value,
        level=Config.bench_level.value,
        result=[metrics],
    )
    # parse_perf_data consumes one record per dtype. Aggregate parametrized
    # cases so later cases do not replace earlier shapes in the CI summary.
    records = TEST_RESULTS.get(result.op_name, {}).get("details", [])
    existing = next(
        (
            record
            for record in records
            if (record["dtype"], record["mode"], record["level"])
            == (result.dtype, result.mode, result.level)
        ),
        None,
    )
    if existing is None:
        update_result(result.op_name, asdict(result))
    else:
        existing["result"].append(asdict(metrics))
    emit_record_logger(result.to_json())
