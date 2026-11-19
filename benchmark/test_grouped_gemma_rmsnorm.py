# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard kernel benchmark of the complete public grouped Gemma callable."""

import gc
import hashlib
import json
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from baseline import grouped_gemma_rmsnorm as baseline

from . import base, conftest, consts

LOGGER = logging.getLogger(__name__)
TOLERANCES = {
    torch.float16: (0.002, 0.002),
    torch.bfloat16: (0.02, 0.02),
    torch.float32: (2e-6, 2e-5),
}


def _check_output(output, args, snapshots):
    x, weight, eps, groups = args
    assert isinstance(output, torch.Tensor)
    assert output.shape == x.shape and output.dtype == x.dtype
    assert output.device == x.device and output.is_contiguous()
    assert output.untyped_storage().data_ptr() not in (
        x.untyped_storage().data_ptr(),
        weight.untyped_storage().data_ptr(),
    ), "output must own storage independent of both inputs"
    n, dim = x.shape
    group_dim = dim // groups
    weight64 = (
        snapshots[1]
        .double()
        .reshape(1, 1 if weight.numel() == group_dim else groups, group_dim)
    )
    actual = output.detach().cpu()
    atol, rtol = TOLERANCES[x.dtype]
    for start in range(0, n, 32):
        values = (
            snapshots[0][start : start + 32].double().reshape(-1, groups, group_dim)
        )
        z = values * torch.rsqrt((values * values).mean(-1, keepdim=True) + eps)
        expected = (z + z * weight64).reshape(-1, dim).to(x.dtype)
        torch.testing.assert_close(
            actual[start : start + 32], expected, atol=atol, rtol=rtol, equal_nan=True
        )


def _changed_inputs(args, snapshots, restore=False):
    changed = []
    for index, (value, saved) in enumerate(zip(args[:2], snapshots)):
        if not torch.equal(value.detach().cpu(), saved):
            changed.append(index)
            if restore:
                value.copy_(saved)
    return changed


class GroupedGemmaRmsnormBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, DIM, num_groups"

    def set_more_shapes(self):
        return baseline.SHAPES

    def record_shapes(self, *args, **kwargs):
        # Base records dtype separately; retain every other binding in this key.
        return baseline.key_shape(self._binding)

    def _compare_offline(self, metric, dtype, data):
        # A tensor-shape match alone can conflate weight dtypes and bindings.
        entry = data.get(self.op_name)
        if not entry:
            metric.base_miss_reason = f"operator '{self.op_name}' not found in baseline"
            return
        records = entry.get("shapes", {}).get(metric.shape_detail)
        record = records.get(str(dtype)) if records else None
        if record is None:
            metric.base_miss_reason = (
                f"no exact binding and dtype {dtype} in NVIDIA baseline"
            )
            return
        config = record.get("config")
        if config is not None and config != baseline.config(self._binding, dtype):
            metric.base_miss_reason = "NVIDIA baseline configuration mismatch"
            return
        latency = record.get("latency_ms")
        if (
            type(latency) not in (int, float)
            or not math.isfinite(latency)
            or latency <= 0
        ):
            metric.base_miss_reason = (
                record.get("reason") or "invalid NVIDIA baseline latency"
            )
            return
        factor = consts.lookup_scaling_factor(
            data, flaggems_vllm.vendor_name, record.get("bottle_neck_unit"), dtype
        )
        if type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0:
            metric.base_miss_reason = "no valid NVIDIA baseline scaling factor"
            return
        metric.compared_speedup = latency / metric.latency / factor

    def run(self):
        cfg = conftest.Config
        if cfg.query:
            return super().run()

        # Base.run aborts on a double baseline miss before publishing the row.
        self.init_user_config()
        for dtype in self.to_bench_dtypes:
            metrics = []
            native_skip_reason = None
            for inputs in self.get_input_iter(dtype):
                args, kwargs = self.unpack_to_args_kwargs(inputs)
                metric = consts.BenchmarkMetrics(
                    shape_detail=self.record_shapes(*args, **kwargs)
                )
                if (
                    "latency_base" in self.to_bench_metrics
                    and self.torch_op is not None
                ):
                    metric.latency_base = self.get_latency(
                        self.torch_op, *args, **kwargs
                    )
                if "latency" in self.to_bench_metrics:
                    metric.latency = self.get_latency(self.gems_op, *args, **kwargs)
                    if cfg.base_data is not None:
                        self._compare_offline(metric, dtype, cfg.base_data)
                if (
                    "speedup" in self.to_bench_metrics
                    and metric.latency_base is not None
                    and metric.latency is not None
                ):
                    metric.speedup = metric.latency_base / metric.latency
                else:
                    native_skip_reason = (
                        native_skip_reason
                        or self._native_error
                        or "native speedup not requested"
                    )
                metrics.append(metric)
                gc.collect()

            result = consts.BenchmarkResult(
                level=cfg.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=cfg.mode.value,
                result=metrics,
            )
            print(result)
            data = asdict(result)
            # The weekly optest parser uses this field to exclude null ratios.
            if native_skip_reason:
                data["native_baseline_skip_reason"] = native_skip_reason
            conftest.update_result(self.op_name, data)
            conftest.emit_record_logger(json.dumps(data))

    def _record(self, role, latency_ms, status, reason=None):
        config = baseline.config(self._binding, self._dtype)
        identity = json.dumps(config, sort_keys=True, separators=(",", ":"))
        cfg = conftest.Config
        ascend = flaggems_vllm.vendor_name == "ascend"
        row = {
            "op": baseline.OP_NAME,
            "variant": "canonical",
            "role": role,
            "status": status,
            "measurement_executed": status == "PASS",
            "reason": reason,
            "case_id": hashlib.sha256(identity.encode()).hexdigest(),
            "shape_key": baseline.key_shape(self._binding),
            "dtype": str(self._dtype),
            "config": config,
            "vendor": flaggems_vllm.vendor_name,
            "device": str(self.device),
            "unit": "ms",
            "latency_ms": latency_ms,
            "native_ms": self._native_ms,
            "native_speedup": (
                self._native_ms / latency_ms
                if role == "candidate" and self._native_ms is not None
                else None
            ),
            "native_provenance": self._native_provenance,
            "native_unavailable_reason": self._native_error,
            "accuracy": "PASS active-grid CPU FP64 check" if status == "PASS" else None,
            "mode": cfg.mode.value,
            "level": cfg.bench_level.value,
            "graph": False,
            "scope": "full public callable including allocation and dispatch",
            "timing_protocol": {
                "framework": "benchmark.base.Benchmark.get_latency",
                "timer": (
                    "triton.backends.ascend.testing.do_bench_npu"
                    if ascend
                    else "triton.testing.do_bench"
                ),
                "requested_warmup": cfg.warm_up,
                "requested_repetition": cfg.repetition,
                "effective_warmup_ms": (
                    cfg.warm_up if not ascend and status == "PASS" else None
                ),
                "effective_repetition_ms": (
                    cfg.repetition if not ascend and status == "PASS" else None
                ),
                "reduction": "backend default" if ascend else "median",
                "note": (
                    "Base Ascend path uses do_bench_npu defaults and does not forward Config durations."
                    if ascend
                    else "Base forwards Config durations; backend timer chooses effective iterations."
                ),
            },
            "active_counts": self._active_counts,
        }
        payload = json.dumps(row, sort_keys=True, allow_nan=False)
        LOGGER.info("%s", payload)
        destination = os.environ.get("FGV_GEMMA_STANDARD_JSONL")
        if destination:
            with Path(destination).open("a", encoding="utf-8") as stream:
                stream.write(payload + "\n")

    def get_input_iter(self, dtype):
        unknown = set(self.shapes) - set(baseline.SHAPES)
        if unknown:
            raise ValueError(f"shapes have no canonical baseline binding: {unknown}")
        bindings = [
            binding
            for binding in baseline.grid()
            if (binding["n"], binding["dim"], binding["num_groups"]) in self.shapes
        ]
        self._active_counts = {
            "bindings": len(bindings),
            "dtypes": len(self.to_bench_dtypes),
            "candidate_rows": len(bindings) * len(self.to_bench_dtypes),
        }
        native = baseline.native(backend=flaggems_vllm.vendor_name)
        unavailable = baseline.NATIVE_UNAVAILABLE_REASON
        self._native_provenance = dict(baseline.NATIVE_PROVENANCE)
        if native is self.gems_op:
            native, unavailable = None, "native aliases the candidate callable"
        for binding in bindings:
            self._binding, self._dtype = binding, dtype
            self._native_ms = None
            self._native_error = unavailable if native is None else None
            self.torch_op = native
            args, kwargs = baseline.build_inputs(binding, dtype, self.device)
            self._snapshots = tuple(value.detach().cpu().clone() for value in args[:2])
            if native is not None:
                try:
                    _check_output(native(*args, **kwargs), args, self._snapshots)
                except Exception as exc:
                    self._native_error = f"{type(exc).__name__}: {exc}"
                changed = _changed_inputs(args, self._snapshots, restore=True)
                if changed:
                    reason = f"native mutated inputs {changed}"
                    self._native_error = (
                        f"{self._native_error}; {reason}"
                        if self._native_error
                        else reason
                    )
            if native is None or self._native_error is not None:
                self.torch_op = None
                self._record("native", None, "N/A", self._native_error)
            # Candidate failures stay outside the native exception scope.
            _check_output(self.gems_op(*args, **kwargs), args, self._snapshots)
            assert not _changed_inputs(
                args, self._snapshots
            ), "candidate mutated inputs"
            yield (*args, kwargs)

    def get_latency(self, op, *args, **kwargs):
        candidate = op is self.gems_op
        try:
            latency = super().get_latency(op, *args, **kwargs)
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"invalid latency: {latency}")
            changed = _changed_inputs(args, self._snapshots, restore=not candidate)
            if changed:
                raise AssertionError(f"timed callable mutated inputs {changed}")
        except Exception as exc:
            if candidate:
                raise
            # Base.run aborts on exceptions; retain invalid native as a row N/A.
            _changed_inputs(args, self._snapshots, restore=True)
            self.torch_op = None
            self._native_error = f"native timing invalid: {type(exc).__name__}: {exc}"
            self._record("native", None, "N/A", self._native_error)
            return None
        if not candidate:
            self._native_ms = latency
        self._record("candidate" if candidate else "native", latency, "PASS")
        return latency


@pytest.mark.grouped_gemma_rmsnorm
def test_grouped_gemma_rmsnorm():
    if conftest.Config.mode != consts.BenchMode.KERNEL:
        raise ValueError("grouped Gemma benchmark requires --mode kernel (no Graph)")
    bench = GroupedGemmaRmsnormBenchmark(
        op_name=baseline.OP_NAME,
        torch_op=None,
        gems_op=flaggems_vllm.grouped_gemma_rmsnorm,
        dtypes=baseline.DTYPES,
    )
    bench.run()
