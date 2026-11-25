# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Standard reports with byte-qualified compiled native and no-Graph timing."""

import math
from pathlib import Path

import pytest
import yaml

import flaggems_vllm as gems
from baseline import pad_indexer_heads as workload
from tests.test_pad_indexer_heads import assert_contract

from . import base
from . import conftest as report
from . import consts


def _native_slot(*args, **kwargs):
    raise RuntimeError("native slot must be handled by get_latency")


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
        self.native_op = None
        self.native_reason = None
        if "latency_base" not in self.to_bench_metrics:
            self.native_reason = "native timing disabled by --metrics"
            return
        try:
            native = workload.native()
        except Exception as exc:
            self.native_reason = f"native resolution: {exc!r}"
            return
        if native is self.gems_op:
            self.native_reason = "native resolves to the Gems candidate"
        elif callable(native):
            self.native_op = native
        else:
            self.native_reason = workload.NATIVE_UNAVAILABLE_REASON or (
                "installed native callable unavailable"
            )

    def run(self):
        with pytest.MonkeyPatch.context() as patch:
            if report.Config.user_desired_metrics == ["latency"]:
                patch.setattr(report.Config, "base_data", None)
            super().run()

    def get_input_iter(self, dtype):
        misses = []
        for binding in self.shapes:
            self.binding = binding
            args, kwargs = workload.build_inputs(binding, dtype, gems.device)
            assert_contract(self.gems_op, *args, **kwargs)
            self.native_status = {
                "status": "N/A",
                "reason": self.native_reason,
                "graph": False,
            }
            self.native_valid = False
            self.native_inputs = None
            if self.native_op is not None:
                # Keep a broken native from mutating the candidate's inputs.
                self.native_inputs = workload.build_inputs(binding, dtype, gems.device)
                try:
                    # Compile and qualify outside the timed region. Failures
                    # remain attached to this configuration, not to the candidate.
                    native_args, native_kwargs = self.native_inputs
                    assert_contract(self.native_op, *native_args, **native_kwargs)
                except Exception as exc:
                    self.native_status.update(
                        status="FAIL", reason=f"native qualification: {exc!r}"
                    )
                else:
                    self.native_valid = True
                    self.native_status.update(status="PASS", reason=None)
            yield (*args, kwargs)
            if self.native_status["reason"]:
                misses.append(
                    {
                        "shape": self.record_shapes(*args, **kwargs),
                        "reason": self.native_status["reason"],
                    }
                )
        if misses:
            report.update_result(
                self.op_name,
                {
                    "dtype": str(dtype),
                    "native_baseline_skip_reason": "; ".join(
                        dict.fromkeys(row["reason"] for row in misses)
                    ),
                    "native_baseline_misses": misses,
                },
            )

    def get_latency(self, op, *args, **kwargs):
        if op is not self.torch_op:
            latency = super().get_latency(op, *args, **kwargs)
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"invalid Gems latency: {latency!r}")
            return latency
        if not self.native_valid:
            return None
        try:
            native_args, native_kwargs = self.native_inputs
            latency = super().get_latency(self.native_op, *native_args, **native_kwargs)
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"invalid native latency: {latency!r}")
        except Exception as exc:
            self.native_status.update(status="FAIL", reason=f"native timing: {exc!r}")
            self.native_valid = False
            return None
        return latency

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
        torch_op=_native_slot,
        gems_op=gems.pad_indexer_heads,
        dtypes=workload.DTYPES,
        is_inplace=workload.IS_INPLACE,
    )
    bench.run()
