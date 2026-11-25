# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Public canonical entry; installed vLLM; standard reports with per-case N/A.

For matching H800 rows on the four collector backends, use the same non-Graph
25 ms / 100 ms median as the reference. The standalone event timer takes
--warmup/--iter as counts.
"""

import ast
import importlib
import importlib.metadata
import importlib.util
import inspect
import math
import statistics
import sys
from pathlib import Path

import pytest
from triton.testing import do_bench

import flaggems_vllm as gems
from baseline import hc_combine as workload
from benchmark import consts
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from flaggems_vllm.ops.qwen4.hc_combine import _hc_combine_kernel
from flaggems_vllm.runtime import torch_device_fn
from tests.test_hc_combine import assert_result, reference

_COLLECTOR_BACKENDS = {"ascend", "hygon", "metax", "mthreads"}
_COLLECTOR_PROTOCOL = "triton-do-bench-no-graph-25ms-warmup-100ms-median"


def _matches_collector(row, binding, dtype):
    if not row or row.get("timing_protocol") != _COLLECTOR_PROTOCOL:
        return False
    if row.get("status") not in (None, "ok", "pass", "PASS"):
        return False
    stored = row.get("config")
    expected = workload.config(binding, dtype)
    fields = ("inputs", "outputs", "dims", "native_api", "mutation", "alias", "graph")
    return isinstance(stored, dict) and all(
        stored.get(k) == expected[k] for k in fields
    )


def _ascend_native():
    module = "vllm_ascend.ops.triton.qwen4_exp.hc"
    try:
        return importlib.import_module(module).hc_combine
    except ModuleNotFoundError as exc:
        # Some vendor wheels import engine warmup from ops/__init__.py even
        # though their standalone HC module has no engine dependency.
        if exc.name != "vllm.model_executor.warmup.jit_warmup":
            raise
    dist = importlib.metadata.distribution("vllm-ascend")
    path = dist.locate_file(module.replace(".", "/") + ".py")
    spec = importlib.util.spec_from_file_location(module, path)
    native = importlib.util.module_from_spec(spec)
    # Triton resolves source through the defining module during compilation.
    sys.modules[module] = native
    try:
        spec.loader.exec_module(native)
        return native.hc_combine
    except Exception:
        del sys.modules[module]
        raise


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
                if gems.vendor_name == "ascend":
                    try:
                        native = _ascend_native()
                    except (ImportError, AttributeError) as exc:
                        native = workload.native()
                        if native is None:
                            raise RuntimeError(
                                f"vllm_ascend: {exc!r}, vllm: {workload.NATIVE_UNAVAILABLE_REASON}"
                            ) from exc
                else:
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
        base = Config.base_data or {}
        offline = base.get(self.op_name)
        self.collector_rows = set()
        h800 = base.get("_environment", {}).get("reference_chip") == "H800"
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
                        dimensions = ast.literal_eval(key)
                        n, hc, d, _, _ = dimensions
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
                    if gems.vendor_name in _COLLECTOR_BACKENDS and h800:
                        # The collector's semantic key includes layout and injection.
                        # Check tensor contracts too before comparing its device time.
                        binding = dict(
                            zip(("n", "hc", "d", "layout", "injection"), dimensions)
                        )
                        exact_rows = {
                            str(dtype): exact_rows[str(dtype)]
                            for dtype in workload.DTYPES
                            if Config.mode == consts.BenchMode.KERNEL
                            and _matches_collector(
                                exact_rows.get(str(dtype)), binding, dtype
                            )
                        }
                        self.collector_rows.update((key, dtype) for dtype in exact_rows)
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
        self.collector_timer = (
            self.base_variant,
            str(args[0].dtype),
        ) in self.collector_rows
        if self.collector_timer:
            self.detail.update(
                protocol=_COLLECTOR_PROTOCOL,
                warmup=None,
                iterations=None,
                warmup_budget_ms=25,
                repetition_budget_ms=100,
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
                if gems.vendor_name in _COLLECTOR_BACKENDS and (
                    actual.device != args[0].device or not actual.is_contiguous()
                ):
                    raise ValueError("native output device or layout differs")
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
            if gems.vendor_name == "ascend":
                self.detail["native_api"] = f"{op.__module__}.{op.__qualname__}"
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
        elif self.collector_timer:
            latency = do_bench(
                lambda: op(*args, **kwargs), warmup=25, rep=100, return_mode="median"
            )
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
