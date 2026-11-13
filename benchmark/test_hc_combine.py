# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Public canonical entry; installed vLLM; standard reports with per-case N/A.

The local timer supplies non-Graph device events on every vendor, rather than
assuming that vendor do_bench implementations use the same timing mechanism.
For this op --warmup/--iter are iteration counts in kernel mode.
"""

import ast
import inspect
import math
import statistics
from pathlib import Path

import pytest

import flaggems_vllm as gems
from baseline import hc_combine as workload
from benchmark import consts
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from flaggems_vllm.ops.qwen4.hc_combine import _hc_combine_kernel
from flaggems_vllm.runtime import torch_device_fn
from tests.test_hc_combine import assert_result, reference


def _native_slot(*args, **kwargs):
    raise RuntimeError("hc_combine native slot must be handled by get_latency")


class HCCombineBenchmark(Benchmark):
    def set_shapes(self, shape_file):
        # The workload module is the sole default matrix. Explicit shape files
        # retain the base loader and must contain [N, HC, D, layout, injection].
        if shape_file and Path(shape_file).resolve() not in (
            Path(self.DEFAULT_SHAPE_FILES).resolve(),
            Path(__file__).with_name("core_shapes.yaml").resolve(),
        ):
            super().set_shapes(shape_file)
        else:
            self.shapes = [tuple(workload.key_shape(b)) for b in workload.grid()]
        self.shape_desc = "N, HC, D, layout, injection (none/same/fp32)"

    def init_user_config(self):
        super().init_user_config()
        if self.mode not in (consts.BenchMode.KERNEL, consts.BenchMode.OPERATOR):
            raise ValueError("hc_combine supports non-Graph kernel/operator modes")

        self.native_op = None
        self.native_reason = "native timing disabled by --metrics"
        if "latency_base" in self.to_bench_metrics:
            try:
                native = workload.native()
            except Exception as exc:
                self.native_reason = f"native resolution: {exc!r}"
            else:
                if native is self.gems_op:
                    self.native_reason = "installed native resolves to Gems"
                elif not callable(native):
                    self.native_reason = (
                        workload.NATIVE_UNAVAILABLE_REASON
                        or "installed native unavailable"
                    )
                else:
                    self.native_op = native
                    self.native_reason = None

    def run(self):
        offline = (Config.base_data or {}).get(self.op_name)
        with pytest.MonkeyPatch.context() as patch:
            if Config.user_desired_metrics == ["latency"]:
                patch.setattr(Config, "base_data", None)
            elif isinstance(offline, dict) and isinstance(offline.get("shapes"), dict):
                # The shared shape lookup cannot distinguish packed/same/fp32
                # rows with equal tensor dimensions. Select the exact collector
                # key first, then expose that record to the framework lookup.
                variants = {}
                for key, rows in offline["shapes"].items():
                    if not isinstance(rows, dict):
                        continue
                    try:
                        n, hc, d, _, _ = ast.literal_eval(key)
                    except (SyntaxError, TypeError, ValueError):
                        continue
                    if not all(type(value) is int for value in (n, hc, d)):
                        continue
                    exact_rows = {
                        dtype: row
                        for dtype, row in rows.items()
                        if isinstance(row, dict)
                        and type(row.get("latency_ms")) in (int, float)
                        and math.isfinite(row["latency_ms"])
                        and row["latency_ms"] > 0
                    }
                    variants[key] = {"shapes": {str([n, hc * d]): exact_rows}}
                patch.setattr(
                    Config,
                    "base_data",
                    {**Config.base_data, self.op_name: {"variants": variants}},
                )
            super().run()

    def get_input_iter(self, dtype):
        misses = []
        for shape in self.shapes:
            self.binding = dict(zip(("n", "hc", "d", "layout", "injection"), shape))
            self.base_variant = str(workload.key_shape(self.binding))
            args, kwargs = workload.build_inputs(self.binding, dtype, gems.device)
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
                        dict.fromkeys(row["reason"] for row in misses)
                    ),
                    native_baseline_misses=misses,
                ),
            )

    def record_shapes(self, *args, **kwargs):
        # Base shape records omit strides, mixed dtypes, and optional variants.
        self.detail = dict(
            args=super().record_shapes(*args, **kwargs),
            config=workload.config(self.binding, args[0].dtype),
            candidate_module=gems.hc_combine.__module__,
            candidate_file=inspect.getfile(gems.hc_combine),
            native_status="N/A" if self.native_op is None else "NOT_RUN",
            native_reason=self.native_reason,
            graph=False,
            protocol=(
                "device-events-median"
                if self.mode == consts.BenchMode.KERNEL
                else "operator-wallclock"
            ),
            warmup=Config.warm_up,
            iterations=Config.repetition,
        )
        return self.detail

    def get_latency(self, op, *args, **kwargs):
        is_native = op is self.torch_op
        if is_native and self.native_op is None:
            return None

        expected = reference(*args)
        if is_native:
            op = self.native_op
            phase = "native qualification"
            try:
                actual = op(*args, **kwargs)
                assert_result(actual, expected)
                phase = "native timing"
                latency = self._measure(op, True, *args, **kwargs)
            except Exception as exc:
                self.detail.update(
                    native_status="N/A", native_reason=f"{phase}: {exc!r}"
                )
                return None
            self.detail.update(
                native_status="PASS",
                native_reason=None,
                native_file=getattr(inspect.getmodule(op), "__file__", None),
            )
            return latency

        actual = op(*args, **kwargs)
        assert_result(actual, expected)
        self.detail["candidate_status"] = "PASS"
        self._record_tuned_config(args, actual)
        return self._measure(op, False, *args, **kwargs)

    def _record_tuned_config(self, args, actual):
        r, b, i, hc = args
        shape = (
            r.shape[0],
            b.shape[1],
            hc,
            r.stride(0),
            b.stride(0),
            0 if i is None else i.stride(0),
            i is not None,
        )
        entry_key = _hc_combine_kernel.key([r, b, i, actual], [], [shape])
        entry = _hc_combine_kernel.kernel_cache[torch_device_fn.current_device()].get(
            entry_key
        )
        if entry is not None:
            self.detail["tuned_config"] = {
                name: entry[1][name]
                for name in ("BLOCK_D", "LOOP", "num_warps", "num_stages")
            }
            self.detail["tuning_cache_table"] = _hc_combine_kernel.fn.cache.table_name

    def _measure(self, op, is_native, *args, **kwargs):
        if self.mode == consts.BenchMode.OPERATOR:
            latency = super().get_latency(op, *args, **kwargs)
        else:
            latency = self._event_latency(op, is_native, *args, **kwargs)
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"invalid hc_combine latency: {latency!r}")
        return latency

    def _event_latency(self, op, is_native, *args, **kwargs):
        for _ in range(max(1, Config.warm_up)):
            op(*args, **kwargs)
        torch_device_fn.synchronize()
        samples = []
        for _ in range(max(1, Config.repetition)):
            start = torch_device_fn.Event(enable_timing=True)
            end = torch_device_fn.Event(enable_timing=True)
            start.record()
            op(*args, **kwargs)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        if is_native:
            self.detail["native_samples_ms"] = samples
        else:
            self.detail["candidate_samples_ms"] = samples
        return statistics.median(samples)


@pytest.mark.hc_combine
def test_hc_combine():
    bench = HCCombineBenchmark(
        op_name="hc_combine",
        torch_op=_native_slot,
        dtypes=workload.DTYPES,
        is_inplace=workload.IS_INPLACE,
        gems_op=gems.hc_combine,
    )
    bench.run()
