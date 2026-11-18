# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark: fused (single-kernel) chunk_gated_delta_rule_fwd_intra_npu.

    pytest -v -s benchmark/chunk_gated_delta_rule_fwd_intra_npu_fused.py \
        -m chunk_gated_delta_rule_fwd_intra_npu_fused

Same cases and inputs as the unfused benchmark, so the two SpeedUp columns
are directly comparable.
"""

import pytest

import flaggems_vllm as gems
from baseline import chunk_gated_delta_rule_fwd_intra_npu as workload
from benchmark.base import Benchmark

# Collected on every backend's runner, including NVIDIA. The 950-only guard
# has to sit at module scope so the skip happens before any class body runs.
pytestmark = pytest.mark.skipif(
    not workload.IS_SUPPORTED_SOC,
    reason=workload.unsupported_soc_reason() or "",
)


class ChunkGatedDeltaRuleFwdIntraNpuFusedBenchmark(Benchmark):
    # One sweep entry only: each case pins its own dtype (``fp16`` is a
    # separate case), so iterating DTYPES would re-run every case on
    # identical inputs.
    DEFAULT_DTYPES = [workload.DTYPES[0]]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [binding["case"] for binding in workload.grid()]
    DEFAULT_SHAPE_DESC = "case name (shape is pinned per case, see baseline module)"

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype):
        del cur_dtype
        for case in self.shapes:
            args, kwargs = workload.build_inputs({"case": case}, None, gems.device)
            yield (*args, kwargs)


@pytest.mark.chunk_gated_delta_rule_fwd_intra_npu_fused
def test_perf_chunk_gated_delta_rule_fwd_intra_npu_fused():
    native = workload.native()
    if native is None:
        pytest.skip(f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}")
    from flaggems_vllm.runtime.backend._ascend.a5.FLA import (
        chunk_gated_delta_rule_fwd_intra_npu_fused,
    )

    bench = ChunkGatedDeltaRuleFwdIntraNpuFusedBenchmark(
        op_name=workload.OP_NAME + "_fused",
        torch_op=native,
    )
    bench.set_gems(chunk_gated_delta_rule_fwd_intra_npu_fused)
    bench.run()
