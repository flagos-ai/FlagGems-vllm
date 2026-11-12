# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark: chunk_scaled_dot_kkt_fwd_950dt vs the FLA operator it replaces.

    pytest -v -s benchmark/test_FLA/chunk_scaled_dot_kkt_fwd_950dt.py \
        -m chunk_scaled_dot_kkt_fwd_950dt --mode operator --level core

Stage 1 of ``chunk_gated_delta_rule_fwd_intra_npu``. Cases come from
``baseline/chunk_gated_delta_rule_fwd_intra_npu.py``, the same module the
accuracy test uses.

``SpeedUp = latency_base / latency`` per the repository convention.

Read with the launch floor in mind: an empty kernel costs ~41 us on this
device while this one does ~5 us of work, so a single-operator SpeedUp here
is dominated by per-call overhead that both sides pay. For the kernel's own
cost see ``gdn_profile/unfused_kernel/end2end/benchmark`` (per-operator
device time over thousands of calls) or ``gdn_profile/scripts/bench_stages.py``.
"""

import pytest

import flaggems_vllm as gems
from baseline import chunk_gated_delta_rule_fwd_intra_npu as workload
from benchmark.base import Benchmark


class ChunkScaledDotKktFwd950dtBenchmark(Benchmark):
    # One sweep entry: each case pins its own dtype, so iterating DTYPES
    # would re-run every case on identical inputs.
    DEFAULT_DTYPES = [workload.DTYPES[0]]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [binding["case"] for binding in workload.grid()]
    DEFAULT_SHAPE_DESC = "case name (shape is pinned per case, see baseline module)"

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_latency(self, op, *args, **kwargs):
        """Device kernel time via the toolchain's do_bench_npu.

        The base class's OPERATOR/WRAPPER modes time a host loop, which on
        this device charges ~45 us of per-call launch work to every sample
        against 5-20 us of kernel work. See the comment above
        ``kernel_latency_ms`` in the baseline module.
        """
        side = "mine" if op is self.gems_op else "ref"
        return workload.kernel_latency_ms(
            lambda: op(*args, **kwargs),
            target_kernel_name=workload.KERNEL_NAMES["kkt"][side][0],
        )

    def get_input_iter(self, cur_dtype):
        del cur_dtype
        for case in self.shapes:
            args, kwargs = workload.build_inputs_kkt({"case": case}, None, gems.device)
            yield (*args, kwargs)


@pytest.mark.chunk_scaled_dot_kkt_fwd_950dt
def test_perf_chunk_scaled_dot_kkt_fwd_950dt():
    native = workload.native_kkt()
    if native is None:
        pytest.skip(f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}")
    from flaggems_vllm.runtime.backend._ascend.FLA import chunk_scaled_dot_kkt_fwd_950dt

    bench = ChunkScaledDotKktFwd950dtBenchmark(
        op_name="chunk_scaled_dot_kkt_fwd_950dt",
        torch_op=native,
    )
    bench.set_gems(chunk_scaled_dot_kkt_fwd_950dt)
    bench.run()
