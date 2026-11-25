# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark the public hc_combine call with the repository timer."""

import importlib
import importlib.metadata
import importlib.util
import math
import sys
from pathlib import Path

import pytest

import flaggems_vllm as gems
from baseline import hc_combine as workload
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from tests.test_hc_combine import assert_result, reference


def _local_native(device):
    if device != "npu":
        return workload.native()
    module = "vllm_ascend.ops.triton.qwen4_exp.hc"
    try:
        return importlib.import_module(module).hc_combine
    except (ImportError, AttributeError) as exc:
        # Some vendor wheels import engine warmup from ops/__init__.py even
        # though their standalone HC module has no engine dependency.
        if getattr(exc, "name", None) != "vllm.model_executor.warmup.jit_warmup":
            return workload.native()
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
    raise RuntimeError("native must be qualified before timing")


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

    def run(self):
        data = Config.base_data
        entry = (data or {}).get(self.op_name, {})
        # Dense/packed and same/fp32 can share every tensor shape.
        # Present each full collector key as a separate base-class variant.
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

    def init_user_config(self):
        super().init_user_config()
        self.native = None
        self.native_reason = "native timing disabled by --metrics"
        if "latency_base" not in self.to_bench_metrics:
            return
        try:
            self.native = _local_native(self.device)
            self.native_reason = None
            if self.native is self.gems_op:
                self.native_reason = "installed native resolves to Gems"
            elif not callable(self.native):
                self.native_reason = (
                    workload.NATIVE_UNAVAILABLE_REASON or "native unavailable"
                )
        except Exception as exc:
            self.native_reason = f"native resolution: {exc!r}"

    def get_input_iter(self, dtype):
        misses = []
        for shape in self.shapes:
            binding = dict(zip(("n", "hc", "d", "layout", "injection"), shape))
            self.base_variant = str(workload.key_shape(binding))
            args, kwargs = workload.build_inputs(binding, dtype, self.device)
            self.expected = reference(*args)
            config = workload.config(binding, dtype)
            config["graph"] = self.mode.value == "cudagraph"
            self.detail = dict(
                args=super().record_shapes(*args, **kwargs),
                config=config,
                protocol=f"Benchmark.get_latency/{self.mode.value}",
                native_reason=self.native_reason,
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
            if self.native_reason:
                return None
            op = self.native
        try:
            actual = op(*args, **kwargs)
            assert_result(actual, self.expected)
            if actual.device != args[0].device or not actual.is_contiguous():
                raise ValueError("output device or layout differs")
            latency = super().get_latency(op, *args, **kwargs)
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"invalid latency: {latency!r}")
        except Exception as exc:
            if not is_native:
                raise
            self.detail["native_reason"] = repr(exc)
            return None
        if is_native:
            self.detail.update(
                native_status="PASS", native_api=f"{op.__module__}.{op.__qualname__}"
            )
        else:
            self.detail["candidate_status"] = "PASS"
        return latency


@pytest.mark.hc_combine
def test_hc_combine():
    HCCombineBenchmark(
        op_name="hc_combine",
        torch_op=_native_slot,
        dtypes=workload.DTYPES,
        is_inplace=workload.IS_INPLACE,
        gems_op=gems.hc_combine,
    ).run()
