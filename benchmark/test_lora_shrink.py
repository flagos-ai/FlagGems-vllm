# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""No-Graph timing of the public LoRA shrink call, with output reset per sample."""

import math
import statistics
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import pytest

import flaggems_vllm as gems
from baseline import lora_shrink as workload
from benchmark import consts
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from benchmark.gumbel_baseline import canonical_chip
from flaggems_vllm.runtime import torch_device_fn
from tests.test_lora_shrink import assert_output, reference_output

_NATIVE_SLOT = object()
_CORE_OMIT = {"tail_zero_scale", "prefill_uneven"}
_HOST_TIMER = "synchronized-host-wall-reset-outside"
_WRAPPER_TIMER = "host-dispatch-reset-outside"
_EVENT_TIMER = "device-events-reset-outside"
# Timer name in the result, collector protocol prefix, statistic.
_MODES = {
    consts.BenchMode.KERNEL: (
        _EVENT_TIMER,
        "no-graph-device-event-reset-outside",
        "median",
    ),
    consts.BenchMode.OPERATOR: (
        _HOST_TIMER,
        "no-graph-operator-wall-reset-outside",
        "mean",
    ),
    consts.BenchMode.WRAPPER: (
        _WRAPPER_TIMER,
        "no-graph-wrapper-wall-reset-outside",
        "mean",
    ),
}


def _positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def _archive_scaling(data):
    scaling = data.get("_scaling_factors") or {}
    if scaling.get("_error"):
        raise ValueError(scaling["_error"])
    ref = canonical_chip(scaling.get("reference_chip"))
    env = data.get("_environment", {})
    measured = env.get("hardware", {}).get("gpu", {}).get("name")
    if not ref or ref != canonical_chip(measured):
        raise ValueError("factor reference chip differs from measured NVIDIA GPU")
    try:
        chip = torch_device_fn.get_device_name(torch_device_fn.current_device())
    except (AttributeError, RuntimeError, TypeError):
        chip = consts._SELF_CHIP if gems.device == "cuda" else ""
    if not canonical_chip(chip):
        raise ValueError("cannot identify the measured target chip")
    targets = [
        item
        for item in scaling.get("chips", [])
        if item.get("vendor") == gems.vendor_name
        and canonical_chip(item.get("chip")) == canonical_chip(chip)
    ]
    if len(targets) != 1:
        raise ValueError(f"no unique scaling entry for {gems.vendor_name} {chip}")
    # The shared lookup accepts a sole same-vendor entry when no name matches.
    # Only give it the entry verified against the actual target device.
    return {**scaling, "chips": targets}


class LoraShrinkBenchmark(Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPE_DESC = "tokens, hidden, rank, slices, adapter slots"

    def set_shapes(self, shape_file):
        grid = workload.grid()
        if Path(shape_file).resolve() != Path(self.DEFAULT_SHAPE_FILES).resolve():
            super().set_shapes(shape_file)
            return
        if Config.query or Config.bench_level == consts.BenchLevel.CORE:
            grid = [b for b in grid if b["case"] not in _CORE_OMIT]
        self.shapes = [tuple(b[d] for d in workload.DIMS) for b in grid]

    def _archive_row(self, rec, cfg, source):
        if not isinstance(rec, dict):
            raise ValueError("no record for the exact workload key and dtype")
        if source not in (
            "ops.lora_shrink",
            "vllm.lora.ops.triton_ops.lora_shrink_op.lora_shrink",
        ):
            raise ValueError("missing or different vLLM native source")
        # The collector records timing only. Qualifying its exported rows uses
        # native_accuracy from this benchmark's matching independent CPU check.
        if rec.get("native_accuracy") != "PASS":
            raise ValueError(
                "missing or failed independent native accuracy qualification"
            )
        observed = rec.get("config") or {}
        if any(
            field not in observed or observed[field] != value
            for field, value in cfg.items()
            if field != "native_fixed_bi_enabled"
        ):
            raise ValueError(
                "NVIDIA workload config, layout or grouped metadata differs"
            )
        if cfg["batch_invariant_requested"] and not observed.get(
            "native_fixed_bi_enabled"
        ):
            raise ValueError("archived native did not use the requested BI reduction")
        if cfg["case"] == "no_lora":
            raise ValueError(
                "host-only no-LoRA call has no GPU resource scaling factor"
            )
        protocol = (
            f"{self.protocol_prefix}-{self.warmup_calls}warmup-"
            f"{self.measured_calls}sample-{self.statistic}"
        )
        if rec.get("timing_protocol") != protocol:
            raise ValueError("incompatible no-Graph sampling protocol")
        if not _positive(rec.get("latency_ms")):
            raise ValueError("missing or invalid NVIDIA latency_ms")
        if rec.get("bottle_neck_unit") not in ("mem", "tensor", "cuda"):
            raise ValueError("missing NVIDIA bottleneck classification")

    def _project_archive(self, data):
        entry = data.get(self.op_name) or {}
        try:
            scaling = _archive_scaling(data)
            scaling_error = None
        except (ValueError, AttributeError, TypeError) as exc:
            scaling = {}
            scaling_error = str(exc)
        variants = {}
        for binding in workload.grid():
            key = str(workload.key_shape(binding))
            rows = (entry.get("shapes") or {}).get(key) or {}
            matched = {}
            for dtype in self.dtypes:
                dt = str(dtype)
                rec = rows.get(dt)
                factor = None
                reasons = [scaling_error] if scaling_error else []
                try:
                    self._archive_row(
                        rec,
                        workload.config(binding, dtype),
                        entry.get("native_api"),
                    )
                    if not scaling_error:
                        target = scaling["chips"][0]
                        resource = rec["bottle_neck_unit"]
                        # The public C550 bandwidth override was measured against
                        # the H800-class reference, not an arbitrary NVIDIA GPU.
                        target_key = f"{target['vendor']} {target['chip']}"
                        if (
                            resource == "mem"
                            and target_key in consts._ACHIEVABLE_BW_RATIO
                            and canonical_chip(scaling["reference_chip"]) != "h800"
                        ):
                            raise ValueError(
                                "achievable bandwidth override has a different reference GPU"
                            )
                        factor = consts.lookup_scaling_factor(
                            {"_scaling_factors": scaling},
                            gems.vendor_name,
                            resource,
                            dtype,
                        )
                        if not _positive(factor):
                            raise ValueError(
                                "missing or invalid factor for target chip, bottleneck and dtype"
                            )
                except (ValueError, AttributeError, TypeError) as exc:
                    reasons.append(str(exc))
                self.archive_info[key, dt] = {
                    "reason": "; ".join(reasons) or None,
                    "source": entry.get("native_api"),
                    "reference_chip": (data.get("_scaling_factors") or {}).get(
                        "reference_chip"
                    ),
                    "target_chip": scaling["chips"][0]["chip"] if scaling else None,
                    "bottleneck": (
                        rec.get("bottle_neck_unit") if isinstance(rec, dict) else None
                    ),
                    "scaling_factor": factor if not reasons else None,
                }
                if not reasons:
                    matched[dt] = rec
            if matched:
                variants[key] = {"shapes": {key: matched}}
        return {
            **data,
            "_scaling_factors": scaling,
            self.op_name: {"variants": variants},
        }

    def run(self):
        if Config.query:
            return super().run()
        if Config.mode not in _MODES:
            raise ValueError(
                "lora_shrink requires no-Graph timing; cudagraph is unsupported"
            )
        self.timer, self.protocol_prefix, self.statistic = _MODES[Config.mode]
        self.archive_info = {}
        data = Config.base_data
        if Config.user_desired_metrics == ["latency"]:
            data = None
        elif data is not None:
            # Expose only exact, qualified rows to Benchmark.run(). The base
            # otherwise writes an unnormalized ratio when a factor is missing.
            data = self._project_archive(data)
        with patch.object(Config, "base_data", data):
            super().run()

    def get_input_iter(self, dtype):
        grid = {tuple(b[d] for d in workload.DIMS): b for b in workload.grid()}
        native = None
        reason = "native timing disabled by --metrics"
        if "latency_base" in self.to_bench_metrics:
            native = workload.native()
            reason = workload.NATIVE_UNAVAILABLE_REASON
            if native is self.gems_op or native is getattr(gems.ops, self.op_name):
                reason = "native resolves to the Gems candidate"
        misses = []
        for shape in self.shapes:
            if tuple(shape) not in grid:
                raise ValueError(f"unknown lora_shrink workload {shape}: use --query")
            binding = grid[tuple(shape)]
            cfg = workload.config(binding, dtype)
            key = cfg["key"]
            self.base_variant = str(key)
            args, kwargs = workload.build_inputs(binding, dtype, self.device)
            self.expected = reference_output(*args, **kwargs)
            self.native = native
            self.host_only = binding["case"] == "no_lora"
            self.row_timer = (
                _HOST_TIMER
                if self.host_only and self.mode == consts.BenchMode.KERNEL
                else self.timer
            )
            self.detail = {
                "args": super().record_shapes(*args, **kwargs),
                "config": cfg,
                "native_accuracy": "NOT_RUN",
                "native_reason": reason or workload.unsupported_reason(binding, dtype),
                "native_performance": "N/A",
                "timer": self.row_timer,
                "warmup_calls": self.warmup_calls,
                "measured_calls": self.measured_calls,
                "statistic": self.statistic,
                "unit": "ms",
            }
            archive = self.archive_info.get((self.base_variant, str(dtype)))
            if archive is not None:
                self.detail["offline_baseline"] = archive
            # The collector's no-op event pair is incompatible with a host-call timer.
            if self.host_only:
                self.detail["offline_timer_required"] = self.row_timer
            # Candidate failures must propagate even when only native latency is
            # requested. Each implementation gets a fresh writable output buffer.
            self.candidate_args = self._qualify(self.gems_op, args)
            self.detail["candidate_accuracy"] = "PASS"
            yield (*args, kwargs)
            if self.detail["native_reason"]:
                misses.append({"shape": key, "reason": self.detail["native_reason"]})
        if misses:
            update_result(
                self.op_name,
                {
                    "dtype": str(dtype),
                    "native_baseline_misses": misses,
                    "native_baseline_skip_reason": "; ".join(
                        dict.fromkeys(m["reason"] for m in misses)
                    ),
                },
            )

    def record_shapes(self, *args, **kwargs):
        return self.detail

    def _qualify(self, op, args):
        out = args[2].clone()
        call_args = (*args[:2], out, *args[3:])
        ptr = out.data_ptr()
        if op(*call_args) is not None:
            raise AssertionError("lora_shrink must return None")
        if out.data_ptr() != ptr or out.stride() != args[2].stride():
            raise AssertionError(
                "lora_shrink changed the supplied output storage/layout"
            )
        assert_output(out, self.expected, call_args, exact=args[-1] == 0)
        return call_args

    def _samples(self, op, call_args, initial):
        # Base.get_latency() repeats fn() inside one timed loop with no reset
        # hook. Restore the entire output outside every warmup and event pair.
        out = call_args[2]
        for _ in range(self.warmup_calls):
            out.copy_(initial)
            torch_device_fn.synchronize()
            op(*call_args)
        samples = []
        if self.row_timer == _EVENT_TIMER:
            start = torch_device_fn.Event(enable_timing=True)
            end = torch_device_fn.Event(enable_timing=True)
        for _ in range(self.measured_calls):
            out.copy_(initial)
            torch_device_fn.synchronize()
            if self.row_timer == _EVENT_TIMER:
                start.record()
                op(*call_args)
                end.record()
                torch_device_fn.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                begin = perf_counter_ns()
                op(*call_args)
                if self.row_timer == _HOST_TIMER:
                    torch_device_fn.synchronize()
                elapsed = (perf_counter_ns() - begin) / 1e6
                if self.row_timer == _WRAPPER_TIMER:
                    torch_device_fn.synchronize()
            samples.append(elapsed)
        if not all(math.isfinite(ms) and ms > 0 for ms in samples):
            raise ValueError(f"invalid latency samples (ms): {samples}")
        return samples

    def get_latency(self, op, *args, **kwargs):
        if op is _NATIVE_SLOT:
            if self.detail["native_reason"]:
                return None
            stage = "qualification"
            try:
                call_args = self._qualify(self.native, args)
                self.detail["native_accuracy"] = "PASS"
                stage = "timing"
                samples = self._samples(self.native, call_args, args[2])
                stage = "post-timing qualification"
                assert_output(
                    call_args[2], self.expected, call_args, exact=args[-1] == 0
                )
            except Exception as exc:
                if stage != "timing":
                    self.detail["native_accuracy"] = "FAIL"
                self.detail["native_reason"] = (
                    f"native {stage}: {type(exc).__name__}: {exc}"
                )
                return None
            self.detail["native_performance"] = "PASS"
            self.detail["native_samples_ms"] = samples
        else:
            samples = self._samples(op, self.candidate_args, args[2])
            assert_output(
                self.candidate_args[2],
                self.expected,
                self.candidate_args,
                exact=args[-1] == 0,
            )
            self.detail["candidate_samples_ms"] = samples
        return (
            statistics.median(samples)
            if self.statistic == "median"
            else statistics.mean(samples)
        )


@pytest.mark.lora_shrink
def test_lora_shrink(request):
    bench = LoraShrinkBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_NATIVE_SLOT,
        gems_op=gems.lora_shrink,
        is_inplace=workload.IS_INPLACE,
        dtypes=workload.DTYPES,
    )
    # pytest leaves the framework defaults as ints and supplies explicit CLI
    # values as strings. Match the collector's 5/25-call protocol by default.
    warmup = request.config.getoption("warmup")
    repetition = request.config.getoption("iter")
    bench.warmup_calls = int(warmup) if isinstance(warmup, str) else 5
    bench.measured_calls = int(repetition) if isinstance(repetition, str) else 25
    if bench.warmup_calls < 0 or bench.measured_calls < 1:
        raise ValueError("lora_shrink requires warmup >= 0 and iter >= 1 calls")
    bench.run()
