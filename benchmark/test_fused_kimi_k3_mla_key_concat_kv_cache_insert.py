# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""State-restored, no-Graph timing of the complete Kimi-K3 prefill epilogue."""

import math
import statistics
import time
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from baseline import fused_kimi_k3_mla_key_concat_kv_cache_insert as workload
from benchmark import conftest as report
from benchmark import consts
from benchmark.base import Benchmark
from benchmark.consts import BenchMode
from flaggems_vllm.runtime import torch_device_fn
from tests.test_fused_kimi_k3_mla_key_concat_kv_cache_insert import reference_inplace


def _storage(tensor):
    n = tensor.untyped_storage().nbytes() // tensor.element_size()
    return tensor.as_strided((n,), (1,), 0)


@workload.single_cpu_thread()
def _expected(binding, dtype):
    args, kwargs = workload.build_inputs(binding, dtype, "cpu")
    reference_inplace(*args, **kwargs)
    return args


@workload.single_cpu_thread()
def _qualify(op, args, expected):
    pairs = [(x, ref) for x, ref in zip(args, expected) if isinstance(x, torch.Tensor)]
    pointers = [x.data_ptr() for x, _ in pairs]
    result = op(*args)
    torch_device_fn.synchronize()
    assert result is None, "prefill epilogue must return None"
    for (x, ref), original_pointer in zip(pairs, pointers):
        assert x.data_ptr() == original_pointer
        assert (x.shape, x.stride(), x.storage_offset(), x.dtype) == (
            ref.shape,
            ref.stride(),
            ref.storage_offset(),
            ref.dtype,
        )
        # Check the whole backing buffer, including read-only projection slices,
        # page/row padding, unused slots and the unrotated portion of Q.
        actual, wanted = _storage(x).cpu(), _storage(ref)
        if not torch.equal(actual, wanted):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def _measure(op, args, snapshots):
    # Benchmark.get_latency has no untimed reset hook and may use a
    # graph-based backend timer. Use identical device events for both providers.
    buffers = [_storage(args[i]) for i in workload.MUTATED_INPUTS]
    start = torch_device_fn.Event(enable_timing=True)
    end = torch_device_fn.Event(enable_timing=True)

    def reset():
        for tensor, initial in zip(buffers, snapshots):
            tensor.copy_(initial)
        torch_device_fn.synchronize()

    deadline = time.perf_counter() + report.Config.warm_up / 1000
    warmups = 0
    while warmups < 5 or time.perf_counter() < deadline:
        reset()
        op(*args)
        torch_device_fn.synchronize()
        warmups += 1

    samples = []
    deadline = time.perf_counter() + report.Config.repetition / 1000
    while len(samples) < 25 or time.perf_counter() < deadline:
        reset()
        start.record()
        op(*args)
        end.record()
        torch_device_fn.synchronize()
        samples.append(float(start.elapsed_time(end)))
    if any(not math.isfinite(t) or t <= 0 for t in samples):
        raise ValueError(f"invalid device-event samples in ms: {samples}")
    return statistics.median(samples), samples, warmups


def _native_slot(*args):
    raise RuntimeError("the benchmark must resolve native before dispatch")


def _positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


class PrefillBenchmark(Benchmark):
    DEFAULT_SHAPES = workload.CASES
    DEFAULT_SHAPE_DESC = ", ".join(workload.FIELDS)
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_DTYPES = workload.DTYPES
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]

    def init_default_config(self):
        self.set_shapes(report.Config.shape_file or self.DEFAULT_SHAPE_FILES)

    def set_metrics(self, user_desired_metrics):
        # The common dependency validator expects prerequisites before speedup.
        if user_desired_metrics:
            ordered = dict.fromkeys([*self.DEFAULT_METRICS, *user_desired_metrics])
            user_desired_metrics = [m for m in ordered if m in user_desired_metrics]
        super().set_metrics(user_desired_metrics)

    def set_shapes(self, shape_file_path=None):
        shape_file_path = shape_file_path or self.DEFAULT_SHAPE_FILES
        if Path(shape_file_path).resolve() == Path(self.DEFAULT_SHAPE_FILES).resolve():
            self.shapes = workload.CASES
        else:
            super().set_shapes(shape_file_path)
        if not self.shapes or any(
            len(row) != len(workload.FIELDS) for row in self.shapes
        ):
            raise ValueError(
                f"expected [{self.DEFAULT_SHAPE_DESC}] for each configuration"
            )
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("duplicate prefill benchmark configurations")

    def init_user_config(self):
        super().init_user_config()
        if (
            self.mode != BenchMode.KERNEL
            or report.Config.warm_up < 0
            or report.Config.repetition <= 0
        ):
            raise ValueError(
                "prefill requires --mode kernel, --warmup >= 0, and --iter > 0"
            )
        self.native_op = None
        self.native_reason = "native timing disabled by --metrics"
        if "latency_base" in self.to_bench_metrics:
            self.native_op = workload.native()
            self.native_reason = workload.NATIVE_UNAVAILABLE_REASON
        self.protocol = dict(
            graph=False,
            timer="device events enclosing one preallocated public call",
            reset="complete q, k_out, k_cache backing buffers; synchronize before start event",
            unit="ms",
            reduction="median",
            warmup_budget_ms=report.Config.warm_up,
            sample_budget_ms=report.Config.repetition,
            budget_clock="wall time including untimed restores",
            minimum_warmups=5,
            minimum_samples=25,
        )

    def _select_offline(self, config, dtype):
        if "speedup" not in self.to_bench_metrics:
            return "offline comparison disabled by --metrics"
        if not self.offline_entry:
            return "no offline baseline for this operator"
        case = config["case"]
        row = self.offline_entry.get("shapes", {}).get(case, {}).get(str(dtype))
        if not isinstance(row, dict) or row.get("config") != config:
            return "no offline record for this exact configuration and dtype"
        if (
            row.get("native_status") != "PASS"
            or row.get("native_accuracy", "PASS") != "PASS"
            or row.get("status") in ("unsupported", "FAIL")
        ):
            return (
                row.get("reason") or "offline row lacks successful native qualification"
            )
        protocol = row.get("timing_protocol")
        if isinstance(protocol, dict):
            # The command-line budgets may differ across runs. The timer,
            # sample floors, reduction and reset must describe the same work.
            compatible = all(
                protocol.get(k) == v
                for k, v in self.protocol.items()
                if k not in ("warmup_budget_ms", "sample_budget_ms")
            )
        else:
            compatible = (
                protocol
                == "no-graph-device-event-reset-outside-5warmup-25sample-median"
            )
        if not compatible:
            return "offline timing protocol is missing or incompatible"
        latency = row.get("latency_ms")
        if not _positive(latency):
            return "offline latency missing or invalid"

        data = self.offline
        gpu = data.get("_environment", {}).get("hardware", {}).get("gpu", {})
        qualified = data.get("_qualification", {}).get("chip")
        source = gpu.get("name") or qualified
        if not source or (gpu.get("vendor") and "nvidia" not in gpu["vendor"].lower()):
            return "offline NVIDIA source chip is unidentified"
        sf = data.get("_scaling_factors") or {}
        reference = sf.get("reference_chip")
        unit = row.get("bottle_neck_unit")
        if qualified and not consts._chip_matches(qualified + " ", source + " "):
            return "offline device and qualification identify different chips"
        self.row.update(
            base_latency_ms=latency, base_reference_chip=source, base_bottleneck=unit
        )
        # The shared lookup may fall back to a raw ratio or to a vendor's sole
        # chip. Only expose scored rows with a compatible source and resource.
        if not reference or not consts._chip_matches(reference + " ", source + " "):
            return "scaling reference does not match the measured source chip"
        if unit != "mem" and not (unit == "cuda" and self.binding["rope"]):
            return "no compatible memory or FP32 vector bottleneck"
        try:
            device = torch_device_fn.get_device_name()
        except (AttributeError, TypeError, RuntimeError):
            device = consts._SELF_CHIP
        targets = [
            c
            for c in sf.get("chips", [])
            if c.get("vendor") == gems.vendor_name
            and c.get("chip")
            and consts._chip_matches(c["chip"] + " ", device + " ")
        ]
        if len(targets) != 1:
            return "no unique scaling entry for measured target chip"
        target = targets[0]["chip"]
        if (
            unit == "mem"
            and f"{gems.vendor_name} {target}" in consts._ACHIEVABLE_BW_RATIO
            and reference != "H800"
        ):
            return "shared bandwidth calibration references H800"
        resource = (gems.vendor_name, unit, self.scaling_dtype)
        factor = consts.lookup_scaling_factor(data, *resource)
        selected = {"_scaling_factors": {"chips": targets}}
        if not _positive(factor) or factor != consts.lookup_scaling_factor(
            selected, *resource
        ):
            return "shared lookup lacks a positive factor for the measured target"
        self.row.update(base_scaling_factor=factor, base_target_chip=target)
        q_shape = str(config["inputs"]["q"]["shape"])
        self.variants[self.base_variant] = {"shapes": {q_shape: {str(dtype): row}}}
        return None

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self.binding = dict(zip(workload.FIELDS, shape))
            config = workload.config(self.binding, dtype)
            self.base_variant = (config["case"], str(dtype))
            self.scaling_dtype = torch.float32 if self.binding["rope"] else dtype
            args, kwargs = workload.build_inputs(self.binding, dtype, self.device)
            self.expected = _expected(self.binding, dtype)
            self.snapshots = [
                _storage(args[i]).clone() for i in workload.MUTATED_INPUTS
            ]
            _qualify(self.gems_op, args, self.expected)
            self.row = dict(
                config=config,
                candidate_accuracy="PASS",
                native_status="NOT_RUN",
                native_reason=self.native_reason,
                native_provenance=(
                    workload.NATIVE_PROVENANCE
                    if "latency_base" in self.to_bench_metrics
                    else None
                ),
                timing_protocol=self.protocol,
            )
            try:
                self.row["offline_reason"] = self._select_offline(config, dtype)
            except (AttributeError, TypeError, ValueError) as exc:
                self.row["offline_reason"] = f"invalid offline metadata: {exc}"
            self.rows[self.base_variant] = self.row
            yield (*args, kwargs)

    def record_shapes(self, *args, **kwargs):
        return dict(
            case=self.row["config"]["case"],
            inputs=super().record_shapes(*args, **kwargs),
        )

    def get_latency(self, op, *args, **kwargs):
        if op is not self.torch_op:
            latency, samples, warmups = _measure(op, args, self.snapshots)
            self.row.update(candidate_samples_ms=samples, candidate_warmups=warmups)
            if "base_latency_ms" in self.row:
                self.row["base_raw_speedup"] = self.row["base_latency_ms"] / latency
            return latency
        if self.native_op is None:
            return None
        # A failing native may write through a read-only argument. Its entire
        # input set is independent of the candidate and the reference.
        native_args, _ = workload.build_inputs(self.binding, args[0].dtype, self.device)
        phase = "qualification"
        try:
            _qualify(self.native_op, native_args, self.expected)
            self.row["native_accuracy"] = "PASS"
            phase = "timing"
            latency, samples, warmups = _measure(
                self.native_op, native_args, self.snapshots
            )
        except Exception as exc:
            unavailable = phase == "qualification" and isinstance(
                exc, NotImplementedError
            )
            self.row.update(
                native_status="NOT_RUN" if unavailable else "FAIL",
                native_reason=f"native {phase}: {exc!r}",
            )
            return None
        self.row.update(
            native_status="PASS", native_samples_ms=samples, native_warmups=warmups
        )
        return latency

    def run(self):
        with pytest.MonkeyPatch.context() as patch:
            self.rows, self.variants = {}, {}
            self.offline = report.Config.base_data
            self.offline_entry = (self.offline or {}).get(self.op_name)
            if self.offline_entry is not None:
                # Other operators and their scaling data stay visible. This
                # operator's scoped variants include only qualified comparisons.
                patch.setitem(self.offline, self.op_name, {"variants": self.variants})
            first = len(report.TEST_RESULTS.get(self.op_name, {}).get("details", []))
            super().run()

        # Preserve the standard per-dtype result and decorate its measured rows.
        for detail in report.TEST_RESULTS.get(self.op_name, {}).get("details", [])[
            first:
        ]:
            misses = []
            for row in detail["result"]:
                row.update(self.rows[row["shape_detail"]["case"], detail["dtype"]])
                if row["compared_speedup"] is None and row["offline_reason"]:
                    row["base_miss_reason"] = row["offline_reason"]
                if row["native_reason"]:
                    misses.append(
                        dict(shape=row["shape_detail"], reason=row["native_reason"])
                    )
                    if row["compared_speedup"] is None:
                        row["base_miss_reason"] = "; ".join(
                            filter(None, (row["native_reason"], row["offline_reason"]))
                        )
            if misses:
                detail.update(
                    native_baseline_misses=misses,
                    native_baseline_skip_reason="; ".join(
                        dict.fromkeys(m["reason"] for m in misses)
                    ),
                )


@pytest.mark.fused_kimi_k3_mla_key_concat_kv_cache_insert
def test_fused_kimi_k3_mla_key_concat_kv_cache_insert():
    PrefillBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_native_slot,
        gems_op=gems.fused_kimi_k3_mla_key_concat_kv_cache_insert,
        is_inplace=workload.IS_INPLACE,
    ).run()
