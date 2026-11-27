# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""In-place BF16 Q and cache benchmark with untimed per-call state restoration."""

import math
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import flaggems_vllm
from baseline import fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert as work
from benchmark import conftest as report
from benchmark.base import Benchmark
from benchmark.consts import BenchLevel, BenchMode, lookup_scaling_factor
from benchmark.gumbel_baseline import canonical_chip
from flaggems_vllm.runtime import torch_device_fn
from tests.test_fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert import (
    assert_cache,
    reference,
)

# Benchmark.run requests this slot for each row, including rows with no native.
_NATIVE_SLOT = object()
_NATIVE_API = f"torch.ops._C.{work.OP_NAME}"


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


class FullCacheBF16Benchmark(Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPES = work.CORE
    DEFAULT_SHAPE_DESC = ", ".join(work.FIELDS)

    def set_shapes(self, shape_file_path=None):
        if Path(shape_file_path).resolve() != Path(self.DEFAULT_SHAPE_FILES).resolve():
            super().set_shapes(shape_file_path)
        else:
            self.shapes = list(work.CORE)
            if (
                report.Config.bench_level == BenchLevel.COMPREHENSIVE
                and not report.Config.query
            ):
                self.shapes.extend(work.EXTRA)
        if any(len(shape) != len(work.FIELDS) for shape in self.shapes):
            raise ValueError(f"expected workload fields {self.DEFAULT_SHAPE_DESC}")

    def init_user_config(self):
        super().init_user_config()
        if self.mode not in (BenchMode.OPERATOR, BenchMode.KERNEL):
            raise ValueError(
                "full BF16 cache insert requires eager operator or kernel mode"
            )
        clock = (
            "device-event-median"
            if self.mode == BenchMode.KERNEL
            else "wall-clock-mean"
        )
        self.protocol = (
            f"no-graph-{clock}-reset-outside-min25-"
            f"{report.Config.warm_up}ms-warmup-{report.Config.repetition}ms-sampling"
        )
        # Norm and rotary arithmetic use FP32 CUDA cores; BF16 is the I/O type.
        self.scaling_dtype = torch.float32
        self.native_op = None
        self.resolution_reason = "native disabled by --metrics"
        if "latency_base" in self.to_bench_metrics:
            self.native_op = work.native()
            self.resolution_reason = work.NATIVE_UNAVAILABLE_REASON
            if self.native_op is self.gems_op:
                self.native_op = None
                self.resolution_reason = (
                    "installed vLLM reference resolves to the candidate"
                )

    def run(self):
        cfg = report.Config
        # Select one exact configuration before the shared shape lookup. Its
        # fallback otherwise conflates layouts or slot distributions.
        self.archive = (
            None if cfg.user_desired_metrics == ["latency"] else cfg.base_data
        )
        self.offline = {}
        first = len(report.TEST_RESULTS.get(self.op_name, {}).get("details", []))
        with patch.object(cfg, "base_data", None):
            super().run()
        # The summary consumer reads base_miss_reason from the standard row.
        for detail in report.TEST_RESULTS.get(self.op_name, {}).get("details", [])[
            first:
        ]:
            for row in detail.get("result", []):
                row.update(self.offline[row["shape_detail"]["case"]])

    def _offline_row(self, dtype, config):
        record = self.archive[self.op_name]["shapes"][self.case_key][str(dtype)]
        if (
            record.get("config") != config
            or record.get("timing_protocol") != self.protocol
        ):
            raise ValueError(
                "offline workload configuration or timing protocol differs"
            )
        entry = record.get("native_api", self.archive[self.op_name].get("native_api"))
        if entry != _NATIVE_API:
            raise ValueError(
                "offline source is not the registered vLLM native operator"
            )
        status = record.get("native_status")
        if (
            str(record.get("status", "PASS")).lower() not in ("ok", "pass", "passed")
            or status not in (None, "PASS")
            or record.get("native_accuracy", status) != "PASS"
            or record.get("native_timing", status) != "PASS"
            or not _positive(record.get("latency_ms"))
        ):
            raise ValueError("offline native accuracy or latency is unqualified")
        bottleneck = record.get("bottle_neck_unit")
        if bottleneck not in ("mem", "cuda"):
            raise ValueError(
                "missing or incompatible native memory/FP32-vector bottleneck"
            )

        env = self.archive["_environment"]
        gpu = env["hardware"]["gpu"]
        reference = self.archive["_scaling_factors"]["reference_chip"]
        source = gpu["name"]
        if not (source.lower().startswith("nvidia ") or gpu.get("vendor") == "nvidia"):
            raise ValueError("offline latency lacks a measured NVIDIA source device")
        if not canonical_chip(reference) or any(
            canonical_chip(name) != canonical_chip(reference)
            for name in (source, env.get("reference_chip", reference))
        ):
            raise ValueError(
                "scaling factor reference differs from the measured NVIDIA SKU"
            )

        vendor = flaggems_vllm.vendor_name
        target = canonical_chip(torch_device_fn.get_device_name(0))
        matches = [
            t
            for t in self.archive["_scaling_factors"]["chips"]
            if t.get("vendor") == vendor
            and target
            and canonical_chip(t.get("chip")) == target
        ]
        if len(matches) != 1:
            raise ValueError("missing or ambiguous exact target vendor/SKU")
        chip = matches[0]
        measured = chip["measured"]
        factor = (
            measured.get("bandwidth_gbps")
            if bottleneck == "mem"
            else measured.get("vector", {}).get("fp32")
        )
        if not _positive(factor):
            raise ValueError(
                "missing positive measured factor for the native bottleneck"
            )
        if canonical_chip(reference) == target and vendor == "nvidia" and factor != 1.0:
            raise ValueError("measured same-chip factor must be one")
        # Use only the exact measured entry. Reject a built-in achievable factor
        # if its value differs from the factor for this archive's denominator.
        data = {
            self.op_name: {
                "shapes": {str(config["inputs"]["q"]["shape"]): {str(dtype): record}}
            },
            "_scaling_factors": {
                "reference_chip": reference,
                "chips": [
                    {"vendor": vendor, "chip": chip["chip"], "measured": measured}
                ],
            },
        }
        if (
            lookup_scaling_factor(data, vendor, bottleneck, self.scaling_dtype)
            != factor
        ):
            raise ValueError(
                "framework factor differs from the measured archive factor"
            )
        return data, {
            "base_data_source": _NATIVE_API,
            "base_source_chip": source,
            "base_target_chip": chip["chip"],
            "base_bottleneck": bottleneck,
            "base_scaling_factor": factor,
            "base_latency_ms": record["latency_ms"],
        }

    def _select_offline(self, dtype, config):
        report.Config.base_data = None
        if self.archive is None:
            return {}
        try:
            report.Config.base_data, info = self._offline_row(dtype, config)
        except (KeyError, TypeError, AttributeError, ValueError, RuntimeError) as exc:
            return {"base_miss_reason": f"offline NVIDIA baseline: {exc}"}
        return info

    def record_shapes(self, *args, **kwargs):
        return {
            "case": self.case_key,
            "arguments": super().record_shapes(*args, **kwargs),
        }

    def get_input_iter(self, dtype):
        misses, details = [], []
        for shape in self.shapes:
            self.binding = dict(zip(work.FIELDS, shape))
            self.case_key = work.key_shape(self.binding)
            self.row_reason = self.resolution_reason or work.unsupported_reason(
                self.binding, dtype
            )
            self.native_accuracy = "NOT_RUN"
            self.native_status = "NOT_RUN"
            config = work.config(self.binding, dtype)
            self.offline[self.case_key] = self._select_offline(dtype, config)
            args, kwargs = work.build_inputs(self.binding, dtype, self.device)
            self.expected = reference(*args, **kwargs)
            self.samples = {}
            yield (*args, kwargs)
            shape_detail = self.record_shapes(*args, **kwargs)
            details.append(
                dict(
                    shape=shape_detail,
                    config=config,
                    timing_protocol=self.protocol,
                    sample_count=self.samples,
                    native_accuracy=self.native_accuracy,
                    native_timing=self.native_status,
                    native_reason=self.row_reason,
                    native_api=_NATIVE_API if self.native_op is not None else None,
                )
            )
            if self.row_reason:
                misses.append({"shape": shape_detail, "reason": self.row_reason})
        metadata = dict(dtype=str(dtype), workload_details=details)
        if misses:
            metadata.update(
                native_baseline_skip_reason="; ".join(
                    dict.fromkeys(m["reason"] for m in misses)
                ),
                native_baseline_misses=misses,
            )
        report.update_result(self.op_name, metadata)

    def _qualify(self, op, args, kwargs, backing, initial_backing):
        q, kv, cache, slots, positions, cis = args[:6]
        readonly = (kv, slots, positions, cis)
        saved = [tensor.detach().cpu().clone() for tensor in readonly]
        result = op(*args, **kwargs)
        torch_device_fn.synchronize()
        assert result is None
        expected_q, expected_cache = self.expected
        torch.testing.assert_close(q.cpu(), expected_q, rtol=0.01, atol=0.01)
        if not self.binding["apply_q_norm"]:
            torch.testing.assert_close(
                q.cpu()[..., :448].view(torch.int16),
                expected_q[..., :448].view(torch.int16),
                rtol=0,
                atol=0,
            )
        expected_backing = initial_backing.detach().cpu().clone()
        expected_backing.as_strided(cache.shape, cache.stride()).copy_(expected_cache)
        assert_cache(backing, expected_backing, cache, slots, self.binding["S"])
        for actual, before in zip(readonly, saved):
            torch.testing.assert_close(
                actual.cpu().view(torch.uint8), before.view(torch.uint8), rtol=0, atol=0
            )

    def _measure(self, op, args, kwargs, restore):
        # The common timer has no hook outside every sample for in-place reset.
        # --warmup and --iter remain millisecond budgets, as in cache flash.
        until = time.perf_counter() + report.Config.warm_up / 1000
        while time.perf_counter() < until:
            restore()
            op(*args, **kwargs)
            torch_device_fn.synchronize()
        kernel = self.mode == BenchMode.KERNEL
        if kernel:
            begin = torch_device_fn.Event(enable_timing=True)
            end = torch_device_fn.Event(enable_timing=True)
        samples = []
        until = time.perf_counter() + report.Config.repetition / 1000
        while len(samples) < 25 or time.perf_counter() < until:
            restore()
            if kernel:
                begin.record()
                op(*args, **kwargs)
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end))
            else:
                start = time.perf_counter()
                op(*args, **kwargs)
                torch_device_fn.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
        ms = statistics.median(samples) if kernel else statistics.mean(samples)
        if not math.isfinite(ms) or ms <= 0:
            raise ValueError(f"invalid latency {ms!r}")
        return ms, len(samples)

    def get_latency(self, op, *args, **kwargs):
        native = op is _NATIVE_SLOT
        if native:
            if self.row_reason:
                return None
            op = self.native_op
            # Isolate even read-only arguments from a defective native kernel.
            args, kwargs = work.build_inputs(self.binding, args[0].dtype, self.device)
        q = args[0]
        backing = work.cache_backing(args[2])
        q0, backing0 = q.clone(), backing.clone()

        def restore():
            q.copy_(q0)
            backing.copy_(backing0)
            torch_device_fn.synchronize()

        phase = "qualification"
        try:
            self._qualify(op, args, kwargs, backing, backing0)
            if native:
                self.native_accuracy = "PASS"
            phase = "timing"
            ms, count = self._measure(op, args, kwargs, restore)
        except Exception as exc:
            if not native:
                raise
            self.row_reason = f"native {phase}: {type(exc).__name__}: {exc}"
            if phase == "qualification":
                self.native_accuracy = "FAIL"
            else:
                self.native_status = "FAIL"
            return None
        self.samples["native" if native else "candidate"] = count
        if native:
            self.native_status = "PASS"
        return ms


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert
def test_fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert():
    FullCacheBF16Benchmark(
        op_name=work.OP_NAME,
        torch_op=_NATIVE_SLOT,
        gems_op=flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert,
        dtypes=work.DTYPES,
        is_inplace=True,
    ).run()
