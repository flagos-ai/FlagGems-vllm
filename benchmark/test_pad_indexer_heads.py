# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Standard reports with byte-qualified compiled native and no-Graph timing."""

from pathlib import Path

import pytest
import yaml

import flaggems_vllm as gems
from baseline import pad_indexer_heads as workload
from tests.test_pad_indexer_heads import assert_contract

from . import base, consts


class PadIndexerHeadsBenchmark(base.Benchmark):
    DEFAULT_SHAPES = workload.grid()
    DEFAULT_SHAPE_DESC = "tokens, heads, tail, added_heads, layout"
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def set_shapes(self, shape_file_path):
        # The base class otherwise selects the generic `Benchmark` YAML entry,
        # whose tensor shapes cannot describe this operator's scalar/layout ABI.
        with open(shape_file_path) as stream:
            shapes = yaml.safe_load(stream)
        if self.op_name in shapes:
            super().set_shapes(shape_file_path)
        else:
            self.shapes = workload.grid()
            self.shape_desc = self.DEFAULT_SHAPE_DESC

    def init_user_config(self):
        super().init_user_config()
        # OPERATOR uses synchronized host timing. The validated backend KERNEL
        # timers use device events/NPU profiling without Graph capture; retain
        # that default for the repository's CI entry point.
        if self.mode not in (consts.BenchMode.OPERATOR, consts.BenchMode.KERNEL):
            raise ValueError(
                "pad_indexer_heads supports operator/kernel timing without Graph"
            )

    def get_input_iter(self, dtype):
        for binding in self.shapes:
            self.binding = binding
            args, kwargs = workload.build_inputs(binding, dtype, gems.device)
            assert_contract(self.gems_op, *args, **kwargs)
            self.native_status = {
                "status": "N/A",
                "reason": workload.NATIVE_UNAVAILABLE_REASON,
                "graph": False,
            }
            self.native_valid = False
            if self.torch_op is not None:
                try:
                    # Compile and qualify outside the timed region. Failures
                    # remain attached to this configuration, not to the candidate.
                    assert_contract(self.torch_op, *args, **kwargs)
                except Exception as exc:
                    self.native_status.update(status="FAIL", reason=repr(exc))
                else:
                    self.native_valid = True
                    self.native_status.update(status="PASS", reason=None)
            yield (*args, kwargs)

    def get_latency(self, op, *args, **kwargs):
        if op is self.torch_op and not self.native_valid:
            return None
        return super().get_latency(op, *args, **kwargs)

    def record_shapes(self, *args, **kwargs):
        # Per-configuration native validity cannot be represented by tensor shapes.
        return [
            *super().record_shapes(*args, **kwargs),
            {
                "native": self.native_status,
                "config": workload.config(self.binding, args[0].dtype),
            },
        ]


@pytest.mark.pad_indexer_heads
def test_pad_indexer_heads():
    bench = PadIndexerHeadsBenchmark(
        op_name=workload.OP_NAME,
        torch_op=workload.native(),
        gems_op=gems.pad_indexer_heads,
        dtypes=workload.DTYPES,
        is_inplace=workload.IS_INPLACE,
    )
    bench.run()
