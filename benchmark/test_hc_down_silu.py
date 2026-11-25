# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Compare each public call to qualified local native or a matched H800 record."""

import inspect
import math
import re
import statistics
from pathlib import Path

import pytest
import triton

import flaggems_vllm as gems
from baseline import hc_down_silu as workload
from benchmark import consts
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from flaggems_vllm.runtime import torch_device_fn
from tests.test_hc_down_silu import assert_result, reference

EVENT_PROTOCOL = "device-events-median-counts-v1"
H800_PROTOCOL = "triton-do-bench-no-graph-25ms-warmup-100ms-median"


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _target_chip():
    try:
        name = torch_device_fn.get_device_name()
    except (AttributeError, RuntimeError):
        return ""
    # DTK can return just the family ("BW"). optest resolves the board model
    # from hy-smi, and lists multiple models separately on heterogeneous hosts.
    if gems.vendor_name == "hygon" and name == "BW":
        try:
            from flagos_optest.gpu_check import probe_accelerator_metadata
        except ImportError:
            return name
        board = probe_accelerator_metadata("hygon").get("device")
        if board and "," not in board:
            return board
    return name


def _native_slot(*args, **kwargs):
    raise RuntimeError("hc_down_silu native slot must be handled by get_latency")


class HCDownSiluBenchmark(Benchmark):
    def set_shapes(self, shape_file):
        # The workload module is the sole default matrix. Explicit shape files
        # retain the base loader and must contain [M, K, N, rank, HC, layout].
        if shape_file and Path(shape_file).resolve() not in (
            Path(self.DEFAULT_SHAPE_FILES).resolve(),
            Path(__file__).with_name("core_shapes.yaml").resolve(),
        ):
            super().set_shapes(shape_file)
        else:
            self.shapes = [tuple(workload.key_shape(b)) for b in workload.grid()]
        self.shape_desc = "M, K, N, rank, HC, layout"

    def init_user_config(self):
        super().init_user_config()
        if self.mode not in (consts.BenchMode.KERNEL, consts.BenchMode.OPERATOR):
            raise ValueError("hc_down_silu supports non-Graph kernel/operator modes")

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
        self.reference_data = _mapping(Config.base_data)
        if Config.user_desired_metrics == ["latency"]:
            self.reference_data = {}
        self.target_chip = (
            _target_chip() if self.reference_data and not Config.query else ""
        )
        # Benchmark.run computes compared_speedup for every visible record,
        # even without a scaling factor. Expose only the selected fallback.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Config, "base_data", None)
            super().run()

    def _select_reference(self, dtype):
        data = self.reference_data
        entry = _mapping(data.get(self.op_name))
        key = self.base_variant
        shape = str(self.detail["config"]["inputs"]["x"]["shape"])
        rows = _mapping(entry.get("shapes")).get(key)
        if "variants" in entry:
            variant = _mapping(_mapping(entry["variants"]).get(key))
            shapes = _mapping(variant.get("shapes"))
            rows = shapes.get(shape, shapes.get(key, {}))
        row = _mapping(_mapping(rows).get(str(dtype)))
        latency = row.get("latency_ms")
        self.detail["offline_latency_ms"] = latency
        if self.detail["native_status"] == "PASS":
            return "native"

        scaling = _mapping(data.get("_scaling_factors"))
        config = self.detail["config"]
        reason = None
        if not row:
            reason = "no H800 record for this configuration and dtype"
        elif not _positive(latency):
            reason = row.get("reason") or "offline latency is not positive and finite"
        elif row.get("status") not in (None, "ok") or config["scope"] != "native-model":
            reason = row.get("reason") or "unsupported H800 native configuration"
        elif (
            self.mode != consts.BenchMode.KERNEL
            or row.get("timing_protocol") != H800_PROTOCOL
        ):
            reason = "offline timing protocol does not match"
        elif (
            _mapping(data.get("_environment")).get("reference_chip") != "H800"
            or scaling.get("reference_chip") != "H800"
        ):
            reason = "reference and scaling factors must both identify H800"
        elif any(
            _mapping(row.get("config")).get(k) != v
            for k, v in config.items()
            if k != "native_version"
        ):
            reason = "offline configuration or public call scope does not match"
        elif row.get("bottle_neck_unit") not in ("mem", "tensor", "cuda"):
            reason = "offline bottleneck is missing or unknown"
        else:
            # The common lookup accepts a vendor's sole chip even on a different
            # SKU. Filter by actual model first, including MTT S5000 / MTT-S5000.
            name = self.target_chip.lower().replace("-", " ")
            chips = [
                chip
                for chip in scaling.get("chips") or []
                if isinstance(chip, dict)
                and chip.get("vendor") == gems.vendor_name
                and isinstance(chip.get("chip"), str)
                and chip["chip"]
                and re.search(
                    r"\b" + re.escape(chip["chip"].lower().replace("-", " ")) + r"\b",
                    name,
                )
            ]
            scoped = {
                "_scaling_factors": {**scaling, "chips": chips},
                self.op_name: {"shapes": {shape: {str(dtype): row}}},
            }
            unit = row["bottle_neck_unit"]
            factor = consts.lookup_scaling_factor(scoped, gems.vendor_name, unit, dtype)
            if len(chips) == 1 and _positive(factor):
                Config.base_data = scoped
                self.detail.update(
                    offline_scaling_factor=factor,
                    offline_bottleneck=unit,
                    offline_target_chip=chips[0]["chip"],
                )
                return "H800"
            reason = f"no matching {unit} scaling factor for {self.target_chip!r}"

        self.detail["offline_rejection"] = dict(
            reason=reason, original_latency_ms=latency
        )
        return "gems_only"

    def get_input_iter(self, dtype):
        misses = []
        for shape in self.shapes:
            Config.base_data = None
            self.binding = dict(zip(workload.FIELDS, shape))
            self.base_variant = str(workload.key_shape(self.binding))
            args, kwargs = workload.build_inputs(self.binding, dtype, gems.device)
            self.expected = reference(*args)
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
            candidate_module=gems.hc_down_silu.__module__,
            candidate_file=inspect.getfile(gems.hc_down_silu),
            native_status="N/A" if self.native_op is None else "NOT_RUN",
            native_reason=self.native_reason,
            graph=False,
            protocol=(
                EVENT_PROTOCOL
                if self.mode == consts.BenchMode.KERNEL
                else "operator-wallclock"
            ),
            warmup=max(1, Config.warm_up),
            iterations=max(1, Config.repetition),
            budget_unit="counts" if self.mode == consts.BenchMode.KERNEL else "ms",
            offline_rejection=None,
        )
        return self.detail

    def get_latency(self, op, *args, **kwargs):
        is_native = op is self.torch_op
        if is_native and self.native_op is None:
            return None

        expected = self.expected
        unsupported = workload.unsupported_reason(self.binding, args[0].dtype)
        if is_native and unsupported:
            self.detail.update(native_status="NOT_RUN", native_reason=unsupported)
            return None
        if is_native:
            op = self.native_op
            phase = "native qualification"
            try:
                actual = op(*args, **kwargs)
                assert_result(actual, expected, *args)
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

        self.detail["selected_source"] = self._select_reference(args[0].dtype)
        try:
            actual = op(*args, **kwargs)
            assert_result(actual, expected, *args)
            latency = self._measure(op, False, *args, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Gems {self.base_variant}: {exc!r}") from exc
        self.detail["candidate_status"] = "PASS"
        return latency

    def _measure(self, op, is_native, *args, **kwargs):
        if self.mode == consts.BenchMode.OPERATOR:
            latency = super().get_latency(op, *args, **kwargs)
        elif not is_native and self.detail["selected_source"] == "H800":
            self.detail.update(
                protocol=H800_PROTOCOL, warmup=25, iterations=100, budget_unit="ms"
            )
            latency = triton.testing.do_bench(
                lambda: op(*args, **kwargs), warmup=25, rep=100, return_mode="median"
            )
        else:
            latency = self._event_latency(op, is_native, *args, **kwargs)
        if not _positive(latency):
            raise ValueError(f"invalid hc_down_silu latency: {latency!r}")
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


@pytest.mark.hc_down_silu
def test_hc_down_silu():
    bench = HCDownSiluBenchmark(
        op_name="hc_down_silu",
        torch_op=_native_slot,
        dtypes=workload.DTYPES,
        is_inplace=workload.IS_INPLACE,
        gems_op=gems.hc_down_silu,
    )
    bench.run()
