# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark: unfused (three-kernel) chunk_gated_delta_rule_fwd_intra_npu.

    pytest -v -s benchmark/chunk_gated_delta_rule_fwd_intra_npu.py \
        -m chunk_gated_delta_rule_fwd_intra_npu

Built on the repository's ``Benchmark`` base class; cases and inputs come
from ``baseline/chunk_gated_delta_rule_fwd_intra_npu.py``, the same module
the accuracy tests use.

SpeedUp follows the repository convention ``latency_base / latency`` with
the FLA reference as ``latency_base``.
"""

import pytest

import flaggems_vllm as gems
from baseline import chunk_gated_delta_rule_fwd_intra_npu as workload
from benchmark.base import Benchmark


class ChunkGatedDeltaRuleFwdIntraNpuBenchmark(Benchmark):
    # One sweep entry only: each case pins its own dtype (``fp16`` is a
    # separate case), so iterating DTYPES would re-run every case on
    # identical inputs.
    DEFAULT_DTYPES = [workload.DTYPES[0]]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [binding["case"] for binding in workload.grid()]
    DEFAULT_SHAPE_DESC = "case name (shape is pinned per case, see baseline module)"

    def set_shapes(self, shape_file_path=None):
        # Cases are named, not numeric tuples, so the generic shape-file
        # parsing does not apply; always use the workload's own list.
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype):
        # Each case pins its own dtype (fp16 is a separate case), so the
        # sweep dtype is ignored and every case is yielded once.
        del cur_dtype
        for case in self.shapes:
            args, kwargs = workload.build_inputs({"case": case}, None, gems.device)
            yield (*args, kwargs)


@pytest.mark.chunk_gated_delta_rule_fwd_intra_npu
def test_perf_chunk_gated_delta_rule_fwd_intra_npu():
    native = workload.native()
    if native is None:
        pytest.skip(f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}")
    from flaggems_vllm.runtime.backend._ascend.FLA import (
        chunk_gated_delta_rule_fwd_intra_npu_unfused,
    )

    bench = ChunkGatedDeltaRuleFwdIntraNpuBenchmark(
        op_name=workload.OP_NAME,
        torch_op=native,
    )
    bench.set_gems(chunk_gated_delta_rule_fwd_intra_npu_unfused)
    bench.run()
