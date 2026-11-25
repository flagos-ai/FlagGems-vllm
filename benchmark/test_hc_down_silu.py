# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark the public hc_down_silu call with a qualified local native."""

import math
from pathlib import Path

import pytest

import flaggems_vllm as gems
from baseline import hc_down_silu as workload
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from tests.test_hc_down_silu import assert_result, reference


def _native_slot(*args, **kwargs):
    raise RuntimeError("native must be qualified before timing")


class HCDownSiluBenchmark(Benchmark):
    def set_shapes(self, shape_file):
        if shape_file and Path(shape_file).resolve() not in (
            Path(self.DEFAULT_SHAPE_FILES).resolve(),
            Path(__file__).with_name("core_shapes.yaml").resolve(),
        ):
            super().set_shapes(shape_file)
        else:
            self.shapes = [tuple(workload.key_shape(b)) for b in workload.grid()]
        self.shape_desc = "M, K, N, rank, HC, layout"

    def run(self):
        data = Config.base_data
        entry = (data or {}).get(self.op_name, {})
        # The first tensor shape is identical for dense and row-padded inputs.
        # Give the base class one variant per full collector key.
        with pytest.MonkeyPatch.context() as patch:
            if Config.user_desired_metrics == ["latency"]:
                patch.setattr(Config, "base_data", None)
            elif "shapes" in entry and "variants" not in entry:
                variants = {
                    key: {"shapes": {key: rows}}
                    for key, rows in entry["shapes"].items()
                }
                patch.setattr(
                    Config, "base_data", {**data, self.op_name: {"variants": variants}}
                )
            super().run()

    def get_input_iter(self, dtype):
        native = None
        reason = "native timing disabled by --metrics"
        if "latency_base" in self.to_bench_metrics:
            try:
                native = workload.native()
                reason = workload.NATIVE_UNAVAILABLE_REASON
            except Exception as exc:
                reason = f"native resolution: {exc!r}"
            if native is self.gems_op:
                reason = "installed native resolves to Gems"
            elif not callable(native):
                reason = reason or "installed native unavailable"
        self.native = native
        misses = []
        for shape in self.shapes:
            binding = dict(zip(workload.FIELDS, shape))
            self.base_variant = str(workload.key_shape(binding))
            args, kwargs = workload.build_inputs(binding, dtype, gems.device)
            self.expected = reference(*args)
            self.detail = dict(
                args=super().record_shapes(*args, **kwargs),
                config=workload.config(binding, dtype),
                native_reason=reason or workload.unsupported_reason(binding, dtype),
                native_status="N/A",
            )
            yield (*args, kwargs)
            if self.detail["native_reason"]:
                misses.append(
                    dict(shape=list(shape), reason=self.detail["native_reason"])
                )
        if misses:
            update_result(
                self.op_name,
                dict(
                    dtype=str(dtype),
                    native_baseline_skip_reason="; ".join(
                        dict.fromkeys(m["reason"] for m in misses)
                    ),
                    native_baseline_misses=misses,
                ),
            )

    def record_shapes(self, *args, **kwargs):
        return self.detail

    def get_latency(self, op, *args, **kwargs):
        is_native = op is self.torch_op
        if is_native:
            if self.detail["native_reason"]:
                return None
            op = self.native
        try:
            assert_result(op(*args, **kwargs), self.expected, *args)
            latency = super().get_latency(op, *args, **kwargs)
            # The NPU profiler reports zero device time when M=0 launches no kernel.
            invalid_zero = latency == 0 and (is_native or args[0].shape[0] != 0)
            if not math.isfinite(latency) or latency < 0 or invalid_zero:
                raise ValueError(f"invalid latency: {latency!r}")
        except Exception as exc:
            if not is_native:
                raise RuntimeError(f"Gems {self.base_variant}: {exc!r}") from exc
            self.detail["native_reason"] = repr(exc)
            return None
        if is_native:
            self.detail["native_status"] = "PASS"
        return latency


@pytest.mark.hc_down_silu
def test_hc_down_silu():
    HCDownSiluBenchmark(
        op_name="hc_down_silu",
        torch_op=_native_slot,
        dtypes=workload.DTYPES,
        is_inplace=workload.IS_INPLACE,
        gems_op=gems.hc_down_silu,
    ).run()
