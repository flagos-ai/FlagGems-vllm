# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import math
import statistics
import time
from pathlib import Path

import pytest

import flaggems_vllm
from baseline import triton_reshape_and_cache_flash_per_token_head_quant as workload
from flaggems_vllm.runtime import torch_device_fn

from . import base, conftest, consts


class PerTokenHeadQuantBenchmark(base.Benchmark):
    DEFAULT_SHAPES = workload.CORE_SHAPES
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPE_DESC = ", ".join(workload.SHAPE_FIELDS)

    def set_shapes(self, shape_file_path):
        if Path(shape_file_path).resolve() != Path(self.DEFAULT_SHAPE_FILES).resolve():
            return super().set_shapes(shape_file_path)
        self.shapes = list(workload.CORE_SHAPES)
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        if (
            conftest.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
            and not conftest.Config.query
        ):
            self.shapes.extend(workload.EXTRA_SHAPES)

    def init_user_config(self):
        super().init_user_config()
        if self.mode != consts.BenchMode.KERNEL:
            raise ValueError("per-token-head cache benchmark uses --mode kernel")
        if "latency_base" in self.to_bench_metrics:
            self.torch_op = workload.native()

    def run(self):
        # The shared offline lookup keys on the first tensor shape. This op
        # needs matching K/V widths, four layouts, cache format and ABI.
        config = conftest.Config
        details = conftest.TEST_RESULTS.get(self.op_name, {}).get("details", [])
        start = len(details)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(config, "base_data", None)
            super().run()
        for detail in conftest.TEST_RESULTS.get(self.op_name, {}).get("details", [])[
            start:
        ]:
            misses = []
            for row in detail["result"]:
                shape = row["shape_detail"]
                row.update(shape["native"])
                if row["native_reason"] is not None:
                    misses.append({"shape": shape, "reason": row["native_reason"]})
            if misses:
                detail["native_baseline_misses"] = misses
                detail["native_baseline_skip_reason"] = "; ".join(
                    dict.fromkeys(miss["reason"] for miss in misses)
                )

    def record_shapes(self, *args, **kwargs):
        return {
            "case": workload.key_shape(self._binding),
            "inputs": super().record_shapes(*args, **kwargs),
            "config": workload.config(self._binding, self._dtype),
            "native": self._row,
        }

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            if len(shape) != len(workload.SHAPE_FIELDS):
                raise ValueError(
                    f"expected benchmark shape fields {self.DEFAULT_SHAPE_DESC}"
                )
            self._binding = dict(zip(workload.SHAPE_FIELDS, shape))
            self._dtype = dtype
            args, _ = workload.build_inputs(self._binding, dtype, self.device)
            self._initial = [out.clone() for out in workload.snapshot_outputs(args)]
            self._expected = workload.reference(args)
            self._row = {
                "native_status": "NOT_RUN",
                "native_reason": (
                    workload.NATIVE_UNAVAILABLE_REASON
                    if "latency_base" in self.to_bench_metrics
                    else "native disabled by --metrics"
                ),
                "native_provenance": (
                    workload.NATIVE_PROVENANCE if self.torch_op else None
                ),
            }
            # The shared runner measures native first. Qualify the candidate
            # here before giving native an opportunity to run on this row.
            result = self.gems_op(*args)
            assert result is None
            workload.assert_outputs(args, self._expected, scale_rtol=2e-7)
            self._candidate = [
                out.cpu().clone() for out in workload.snapshot_outputs(args)
            ]
            self._reset(args)
            yield args

    def _reset(self, args):
        for out, initial in zip(workload.snapshot_outputs(args), self._initial):
            out.copy_(initial)
        torch_device_fn.synchronize()

    def _measure(self, op, args, kwargs):
        warmup_end = time.perf_counter() + conftest.Config.warm_up / 1000
        while time.perf_counter() < warmup_end:
            self._reset(args)
            op(*args, **kwargs)
            torch_device_fn.synchronize()
        samples = []
        sample_end = time.perf_counter() + conftest.Config.repetition / 1000
        while not samples or time.perf_counter() < sample_end:
            self._reset(args)
            start = torch_device_fn.Event(enable_timing=True)
            end = torch_device_fn.Event(enable_timing=True)
            start.record()
            op(*args, **kwargs)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        latency = statistics.median(samples)
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"invalid median latency {latency}")
        return latency

    def get_latency(self, op, *args, **kwargs):
        if op is self.gems_op:
            return self._measure(op, args, kwargs)

        native_args, _ = workload.build_inputs(self._binding, self._dtype, self.device)
        phase = "binding"
        try:
            op, native_kwargs = op.bind(native_args)
            phase = "qualification"
            assert op(*native_args, **native_kwargs) is None
            workload.assert_outputs(native_args, self._expected, scale_rtol=2e-7)
            for index, actual, expected in zip(
                workload.MUTATED_INPUTS,
                workload.snapshot_outputs(native_args),
                self._candidate,
            ):
                # Compare scale bytes and untouched gaps as well as quantized codes.
                if not actual.cpu().equal(expected):
                    raise AssertionError(f"{workload.ARGUMENTS[index]} backing differs")
            phase = "timing"
            latency = self._measure(op, native_args, native_kwargs)
        except Exception as exc:
            self._row.update(
                native_status="FAIL",
                native_reason=f"native {phase}: {type(exc).__name__}: {exc}",
            )
            return None
        self._row.update(native_status="PASS", native_reason=None)
        return latency


@pytest.mark.triton_reshape_and_cache_flash_per_token_head_quant
def test_triton_reshape_and_cache_flash_per_token_head_quant_perf():
    PerTokenHeadQuantBenchmark(
        op_name=workload.OP_NAME,
        torch_op=None,
        gems_op=flaggems_vllm.triton_reshape_and_cache_flash_per_token_head_quant,
        dtypes=workload.DTYPES,
        is_inplace=workload.IS_INPLACE,
    ).run()
