# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Public canonical entry; installed vLLM; standard reports with per-case N/A.

The local timer supplies non-Graph device events on every vendor, rather than
assuming that vendor do_bench implementations use the same timing mechanism.
For this op --warmup/--iter are iteration counts in kernel mode.
"""

import inspect
import statistics
from pathlib import Path

import pytest

import flaggems_vllm as gems
from baseline import hc_combine as workload
from benchmark import consts
from benchmark.base import Benchmark
from benchmark.conftest import Config
from flaggems_vllm.ops.qwen4.hc_combine import _hc_combine_kernel
from flaggems_vllm.runtime import torch_device_fn
from tests.test_hc_combine import assert_result, reference


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

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self.binding = dict(zip(("n", "hc", "d", "layout", "injection"), shape))
            args, kwargs = workload.build_inputs(self.binding, dtype, gems.device)
            yield (*args, kwargs)

    def record_shapes(self, *args, **kwargs):
        # Base shape records omit strides, mixed dtypes, and optional variants.
        self.detail = dict(
            args=super().record_shapes(*args, **kwargs),
            config=workload.config(self.binding, args[0].dtype),
            candidate_module=gems.hc_combine.__module__,
            candidate_file=inspect.getfile(gems.hc_combine),
            native_status="N/A" if self.torch_op is None else "NOT_RUN",
            native_reason=workload.NATIVE_UNAVAILABLE_REASON,
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
        # The base runner treats all exceptions as candidate failures. Native
        # runtime/timer failures instead retain this row as N/A and continue.
        try:
            return self._validated_latency(op, *args, **kwargs)
        except Exception as exc:
            if op is not self.torch_op:
                raise
            self.detail.update(native_status="N/A", native_reason=repr(exc))
            return None

    def _validated_latency(self, op, *args, **kwargs):
        is_native = op is self.torch_op
        expected = reference(*args)
        try:
            actual = op(*args, **kwargs)
            assert_result(actual, expected)
        except Exception as exc:
            if not is_native:
                raise
            self.detail.update(native_status="N/A", native_reason=repr(exc))
            return None
        if not is_native:
            self.detail["candidate_status"] = "PASS"
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
            entry = _hc_combine_kernel.kernel_cache[
                torch_device_fn.current_device()
            ].get(entry_key)
            if entry is not None:
                self.detail["tuned_config"] = {
                    name: entry[1][name]
                    for name in ("BLOCK_D", "LOOP", "num_warps", "num_stages")
                }
                self.detail["tuning_cache_table"] = (
                    _hc_combine_kernel.fn.cache.table_name
                )
        if is_native:
            self.detail.update(
                native_status="PASS",
                native_reason=None,
                native_file=inspect.getfile(op),
            )
        if self.mode == consts.BenchMode.OPERATOR:
            return super().get_latency(op, *args, **kwargs)
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
        torch_op=workload.native(),
        dtypes=workload.DTYPES,
        is_inplace=workload.IS_INPLACE,
        gems_op=gems.hc_combine,
    )
    bench.run()
