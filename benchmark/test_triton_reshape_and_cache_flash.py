# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Standard report path with per-call cache reset and no Graph timing."""

import statistics
import time
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from baseline import triton_reshape_and_cache_flash as workloads
from flaggems_vllm.runtime import torch_device_fn

from . import base, conftest, consts


class CacheFlashBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPE_DESC = "T,H,D,BS,layout,padding,distribution,cache"

    def set_shapes(self, shape_file_path=None):
        # The baseline owns the matrix. Preserve explicit external shape files.
        super().set_shapes(shape_file_path)
        if shape_file_path == self.DEFAULT_SHAPE_FILES:
            names = ("T", "H", "D", "BS", "layout", "padding", "distribution", "cache")
            self.shapes = [
                tuple(binding[name] for name in names) for binding in workloads.grid()
            ]

    def run(self):
        # Base metrics have no native-qualification/raw-sample fields. Preserve
        # the inherited execution and enrich its dictionaries before the pytest
        # JSON hook writes them; runtime facts must not change shape identities.
        self._records = {}
        super().run()
        for detail in conftest.TEST_RESULTS.get(self.op_name, {}).get("details", []):
            for row in detail.get("result", []):
                key = (detail["dtype"], row["shape_detail"]["case"])
                if key in self._records:
                    row.update(self._records[key])
                    if (
                        row["native_status"] != "PASS"
                        and row.get("base_miss_reason") is None
                        and row.get("compared_speedup") is None
                    ):
                        row["base_miss_reason"] = "native N/A: " + str(
                            row["native_reason"]
                        )

    def record_shapes(self, *args, **kwargs):
        # Stable input metadata only: no timing samples, provenance paths or
        # native availability, which differ between runs and devices.
        return {
            "case": workloads.key_shape(self._binding),
            "config": workloads.config(self._binding, self._dtype),
            "tensors": [
                {
                    "shape": list(t.shape),
                    "stride": list(t.stride()),
                    "dtype": str(t.dtype),
                    "storage_offset": t.storage_offset(),
                }
                for t in (*args, *kwargs.values())
                if isinstance(t, torch.Tensor)
            ],
        }

    def _reset(self, args):
        for cache, saved in zip(args[2:4], self._poison):
            workloads.backing(cache).view(torch.uint8).copy_(saved)
        torch_device_fn.synchronize()

    def _check(self, op, args, kwargs):
        self._reset(args)
        result = op(*args, **kwargs)
        torch_device_fn.synchronize()
        assert result is None
        for cache, expected, identity in zip(
            args[2:4], self._expected, self._identities
        ):
            assert (
                cache.data_ptr(),
                cache.untyped_storage().data_ptr(),
                cache.storage_offset(),
                cache.stride(),
            ) == identity
            torch.testing.assert_close(
                workloads.backing(cache).view(torch.uint8).cpu(),
                expected,
                rtol=0,
                atol=0,
            )
        for tensor, snapshot in self._readonly:
            torch.testing.assert_close(
                workloads.backing(tensor).view(torch.uint8).cpu(),
                snapshot,
                rtol=0,
                atol=0,
            )

    def get_input_iter(self, dtype):
        native = workloads.native()
        for shape in self.shapes:
            names = ("T", "H", "D", "BS", "layout", "padding", "distribution", "cache")
            if len(shape) != len(names):
                raise ValueError(
                    "cache flash shapes must have T,H,D,BS,layout,padding,distribution,cache"
                )
            self._binding, self._dtype = dict(zip(names, shape)), dtype
            args, kwargs = workloads.build_inputs(self._binding, dtype, self.device)
            self._poison = [
                workloads.backing(t).view(torch.uint8).clone() for t in args[2:4]
            ]
            self._expected = workloads.reference(args)
            self._identities = [
                (
                    t.data_ptr(),
                    t.untyped_storage().data_ptr(),
                    t.storage_offset(),
                    t.stride(),
                )
                for t in args[2:4]
            ]
            self._readonly = [
                (t, workloads.backing(t).view(torch.uint8).cpu().clone())
                for index, t in enumerate(args)
                if isinstance(t, torch.Tensor) and index not in workloads.MUTATED_INPUTS
            ]
            # Candidate errors are hard failures and never classified as native N/A.
            self._check(self.gems_op, args, kwargs)
            self._native_status, self._native_reason = (
                "UNAVAILABLE",
                workloads.NATIVE_UNAVAILABLE_REASON,
            )
            self.torch_op = native
            if native is not None:
                try:
                    self._check(native, args, kwargs)
                    self._native_status, self._native_reason = "PASS", None
                except Exception as exc:
                    self._native_status, self._native_reason = (
                        "FAIL",
                        f"native qualification failed: {type(exc).__name__}: {exc}",
                    )
                    self.torch_op = None
                    # Restore readonly inputs too before validating candidate again.
                    for tensor, saved in self._readonly:
                        workloads.backing(tensor).view(torch.uint8).copy_(
                            saved.to(tensor.device)
                        )
                    self._check(self.gems_op, args, kwargs)
            self._row_metadata = {
                "candidate_accuracy": "PASS",
                "native_status": self._native_status,
                "native_reason": self._native_reason,
                "native_provenance": workloads.NATIVE_PROVENANCE,
                "timing_protocol": {
                    "graph": False,
                    "reset": "full allocations before every invocation, outside event interval",
                    "warmup_budget_ms": conftest.Config.warm_up,
                    "sample_budget_ms": conftest.Config.repetition,
                    "minimum_samples": 25,
                    "budget_clock": "wall clock including untimed reset/synchronize",
                    "reduction": "median",
                    "unit": "ms",
                },
            }
            self._records[(str(dtype), workloads.key_shape(self._binding))] = (
                self._row_metadata
            )
            self._reset(args)
            # Benchmark.unpack_to_args_kwargs ignores bare strings. Keep the
            # selector and following scale arguments in named kwargs.
            yield (
                *args[:5],
                {
                    "kv_cache_dtype": args[5],
                    "k_scale": args[6],
                    "v_scale": args[7],
                    **kwargs,
                },
            )

    def get_latency(self, op, *args, **kwargs):
        # Base timers repeatedly call a closure without an outside-the-event reset
        # hook. This local adapter is necessary for two mutated cache allocations.
        candidate = op is self.gems_op
        try:
            self._check(op, args, kwargs)
            start_budget = time.perf_counter()
            while time.perf_counter() - start_budget < conftest.Config.warm_up / 1000:
                self._reset(args)
                op(*args, **kwargs)
                torch_device_fn.synchronize()
            samples = []
            start_budget = time.perf_counter()
            while (
                len(samples) < 25
                or time.perf_counter() - start_budget
                < conftest.Config.repetition / 1000
            ):
                self._reset(args)
                begin = torch_device_fn.Event(enable_timing=True)
                end = torch_device_fn.Event(enable_timing=True)
                begin.record()
                op(*args, **kwargs)
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end))
            self._check(op, args, kwargs)
            self._row_metadata.setdefault("samples_ms", {})[
                "candidate" if candidate else "native"
            ] = samples
            return statistics.median(samples)
        except Exception as exc:
            if candidate:
                raise
            # Preserve native timing failures as N/A in the same standard record.
            self._native_status, self._native_reason = (
                "FAIL",
                f"native timing failed: {type(exc).__name__}: {exc}",
            )
            self.torch_op = None
            self._row_metadata.update(
                native_status=self._native_status, native_reason=self._native_reason
            )
            return None


@pytest.mark.triton_reshape_and_cache_flash
def test_triton_reshape_and_cache_flash():
    if conftest.Config.mode != consts.BenchMode.KERNEL:
        raise ValueError("cache benchmark requires --mode kernel with no Graph")
    bench = CacheFlashBenchmark(
        op_name=workloads.OP_NAME,
        torch_op=workloads.native(),
        gems_op=flaggems_vllm.triton_reshape_and_cache_flash,
        dtypes=workloads.DTYPES,
        is_inplace=True,
    )
    bench.run()
