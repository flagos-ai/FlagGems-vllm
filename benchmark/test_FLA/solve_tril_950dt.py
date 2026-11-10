# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark: solve_tril_950dt vs the FLA operator it replaces.

    pytest -v -s benchmark/test_FLA/solve_tril_950dt.py \
        -m solve_tril_950dt --mode operator --level core

Stage 2 of ``chunk_gated_delta_rule_fwd_intra_npu``: inverts I + M per chunk tile. Cases come
from ``baseline/chunk_gated_delta_rule_fwd_intra_npu.py``, the same module
the accuracy test uses, and the inputs are produced by the *reference*
upstream stages so both sides consume identical data.

``SpeedUp = latency_base / latency`` per the repository convention.

Read with the launch floor in mind: an empty kernel costs ~41 us on this
device while these kernels do ~5 us of work, so a single-operator SpeedUp
here is dominated by per-call overhead that both sides pay. For the kernel's
own cost see ``gdn_profile/unfused_kernel/end2end/benchmark`` (per-operator
device time over thousands of calls) or
``gdn_profile/scripts/bench_stages.py``.
"""

import pytest

import flaggems_vllm as gems
from baseline import chunk_gated_delta_rule_fwd_intra_npu as workload
from benchmark.base import Benchmark


class SolveTril950dtBenchmark(Benchmark):
    # One sweep entry: each case pins its own dtype, so iterating DTYPES
    # would re-run every case on identical inputs.
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
            args, kwargs = workload.build_inputs_solve_tril({"case": case}, None, gems.device)
            if kwargs is None:
                pytest.skip(
                    f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}"
                )
            yield (*args, kwargs)


@pytest.mark.solve_tril_950dt
def test_perf_solve_tril_950dt():
    native = workload.native_solve_tril()
    if native is None:
        pytest.skip(f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}")
    from flaggems_vllm.runtime.backend._ascend.FLA import solve_tril_950dt

    bench = SolveTril950dtBenchmark(
        op_name="solve_tril_950dt",
        torch_op=native,
    )
    bench.set_gems(solve_tril_950dt)
    bench.run()
