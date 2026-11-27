# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""No-Graph decode timing with a full output/cache reset before each call."""

import math
import re
import statistics
import time
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from baseline import fused_kimi_k3_mla_decode_q_concat_kv_cache_insert as workload
from benchmark import conftest as report
from benchmark import consts
from benchmark.base import Benchmark
from benchmark.consts import BenchMode
from flaggems_vllm.runtime import torch_device_fn


def _storage(tensor):
    elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    return tensor.as_strided((elements,), (1,), 0)


def _unique_readonly(args):
    seen = set()
    for i, tensor in enumerate(args):
        if isinstance(tensor, torch.Tensor) and i not in workload.MUTATED_INPUTS:
            data = _storage(tensor)
            if data.data_ptr() not in seen:
                yield data
                seen.add(data.data_ptr())


@workload.single_cpu_thread()
def _reference(args):
    ql, qp, kv, kp, out, cache, slots, s, positions, table = args
    # Transfer logical read-only views, and the full cache allocation including
    # page/row padding. as_strided restores the offset inside its CPU backing.
    qp, kp = qp.cpu(), kp.cpu()
    if positions is not None:
        cos, sin = table.cpu()[positions.cpu()].chunk(2, dim=-1)

        def rotate(x, c, s):
            even, odd = x[..., 0::2].float(), x[..., 1::2].float()
            return torch.stack((even * c - odd * s, odd * c + even * s), -1).flatten(-2)

        qp = rotate(qp, cos[:, None], sin[:, None]).to(ql.dtype)
        kp = rotate(kp, cos, sin).to(kv.dtype)
    q = torch.cat((ql.cpu(), qp), dim=-1)
    cache_data = _storage(cache).cpu().clone()
    cache_view = cache_data.as_strided(
        cache.shape, cache.stride(), cache.storage_offset()
    )
    slots = slots.cpu()
    valid = slots >= 0
    cache_view[slots[valid] // s, slots[valid] % s] = torch.cat((kv.cpu(), kp), -1)[
        valid
    ]
    readonly = [data.cpu().clone() for data in _unique_readonly(args)]
    return q.reshape(-1), cache_data, readonly


@workload.single_cpu_thread()
def _check(op, args, expected):
    tensors = [t for t in args if isinstance(t, torch.Tensor)]
    metadata = [
        (t.data_ptr(), t.storage_offset(), t.shape, t.stride()) for t in tensors
    ]
    result = op(*args)
    torch_device_fn.synchronize()
    assert result is None, "decode must return None"
    assert metadata == [
        (t.data_ptr(), t.storage_offset(), t.shape, t.stride()) for t in tensors
    ]
    q, cache, readonly = expected
    actual = [
        *(_storage(args[i]) for i in workload.MUTATED_INPUTS),
        *_unique_readonly(args),
    ]
    assert len(actual) == 2 + len(readonly)
    for data, ref in zip(actual, [q, cache, *readonly]):
        # Inputs are finite dyadic values; even rotated lanes are exact here.
        observed = data.cpu().view(torch.uint8)
        wanted = ref.view(torch.uint8)
        if not torch.equal(observed, wanted):
            torch.testing.assert_close(observed, wanted, rtol=0, atol=0)


def _measure(op, args, initial):
    # Benchmark.get_latency provides neither an untimed reset hook nor a
    # universal no-Graph kernel timer (Ascend selects do_bench_npu).
    outputs = [_storage(args[i]) for i in workload.MUTATED_INPUTS]

    def reset():
        for output, saved in zip(outputs, initial):
            output.copy_(saved)
        torch_device_fn.synchronize()

    start = torch_device_fn.Event(enable_timing=True)
    end = torch_device_fn.Event(enable_timing=True)
    deadline = time.perf_counter() + report.Config.warm_up / 1000
    warmups = 0
    while warmups < 1 or time.perf_counter() < deadline:
        reset()
        op(*args)
        torch_device_fn.synchronize()
        warmups += 1
    deadline = time.perf_counter() + report.Config.repetition / 1000
    samples = []
    while len(samples) < 25 or time.perf_counter() < deadline:
        reset()
        start.record()
        op(*args)
        end.record()
        torch_device_fn.synchronize()
        samples.append(float(start.elapsed_time(end)))
    if any(not math.isfinite(t) or t <= 0 for t in samples):
        raise ValueError(f"invalid device event samples: {samples}")
    return statistics.median(samples), samples, warmups


def _native_slot(*args):
    raise RuntimeError("native availability must be resolved before timing")


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _matches_chip(model, observed):
    if not isinstance(model, str) or not isinstance(observed, str):
        return False
    model = model.strip().lower().removeprefix("nvidia ")
    observed = observed.strip().lower().removeprefix("nvidia ")
    # The common match accepts vendor prefixes and capacity suffixes. Require
    # model boundaries as well, e.g. H20 must not match H200.
    return (
        consts._chip_matches(model, observed)
        and re.search(rf"(?<![a-z0-9]){re.escape(model)}(?![a-z0-9])", observed)
        is not None
    )


class DecodeBenchmark(Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_DTYPES = workload.DTYPES
    DEFAULT_SHAPES = workload.CASES
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPE_DESC = ", ".join(workload.FIELDS)

    def set_shapes(self, shape_file_path=None):
        if (
            shape_file_path
            and Path(shape_file_path).resolve()
            != Path(self.DEFAULT_SHAPE_FILES).resolve()
        ):
            super().set_shapes(shape_file_path)
        else:
            self.shapes = workload.CASES
        if any(len(row) != len(workload.FIELDS) for row in self.shapes):
            raise ValueError(f"expected shape rows [{self.DEFAULT_SHAPE_DESC}]")
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("duplicate decode benchmark configurations")

    def init_user_config(self):
        super().init_user_config()
        if self.mode != BenchMode.KERNEL:
            raise ValueError("decode benchmark requires --mode kernel (no Graph)")
        self.native_op = None
        self.native_reason = "native timing disabled by --metrics"
        if "latency_base" in self.to_bench_metrics:
            self.native_op = workload.native()
            self.native_reason = workload.NATIVE_UNAVAILABLE_REASON
        self.protocol = dict(
            graph=False,
            timer="current-device events enclosing one public call",
            reset="full mqa_q and cache allocations, synchronized before start",
            unit="ms",
            reduction="median",
            warmup_budget_ms=report.Config.warm_up,
            sample_budget_ms=report.Config.repetition,
            budget_clock="wall time including untimed restores",
            minimum_warmups=1,
            minimum_samples=25,
        )

    def run(self):
        self.records = {}
        self.offline = report.Config.base_data
        details = report.TEST_RESULTS.get(self.op_name, {}).get("details", [])
        first = len(details)
        # The common shape lookup cannot distinguish slot/stride patterns and
        # emits raw ratios when scaling is unavailable. Qualify the exact row here.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(report.Config, "base_data", None)
            super().run()
        for detail in report.TEST_RESULTS.get(self.op_name, {}).get("details", [])[
            first:
        ]:
            misses = []
            for row in detail["result"]:
                record = self.records[detail["dtype"], row["shape_detail"]["case"]]
                row.update(record)
                if row["native_reason"]:
                    misses.append(
                        dict(shape=row["shape_detail"], reason=row["native_reason"])
                    )
                    if row["compared_speedup"] is None:
                        row["base_miss_reason"] = "; ".join(
                            r
                            for r in (row["native_reason"], record["offline_reason"])
                            if r
                        )
            if misses:
                detail["native_baseline_misses"] = misses
                detail["native_baseline_skip_reason"] = "; ".join(
                    dict.fromkeys(m["reason"] for m in misses)
                )

    def _compare_offline(self, candidate_ms):
        if "speedup" not in self.to_bench_metrics:
            return "comparison disabled by --metrics"
        if self.offline is None:
            return "no offline baseline supplied"
        config, dtype = self.record["config"], self.dtype
        entry = self.offline.get(self.op_name, {})
        record = entry.get("shapes", {}).get(config["case"], {}).get(str(dtype), {})
        if record.get("config") != config:
            return "no offline record for this exact configuration"
        if record.get("status") == "unsupported":
            return f"offline native: {record.get('reason', 'unsupported')}"
        protocol = record.get("timing_protocol")
        if record.get("native_status") != "PASS":
            return "offline native lacks per-row numerical qualification"
        if isinstance(protocol, dict):
            comparable = all(
                protocol.get(k) == self.protocol[k]
                for k in ("graph", "timer", "reset", "unit", "reduction")
            )
        else:
            # An ordinary collector's 5/25 measurements can be paired with
            # a qualification from the corresponding native build and row.
            comparable = (
                protocol
                == "no-graph-device-event-reset-outside-5warmup-25sample-median"
            )
        if not comparable:
            return "offline timing protocol mismatch"
        latency = record.get("latency_ms")
        if not _positive(latency):
            return "invalid offline latency"
        env = self.offline.get("_environment", {})
        gpu = env.get("hardware", {}).get("gpu", {}).get("name")
        qualified = self.offline.get("_qualification", {}).get("chip")
        source = gpu or qualified
        if not isinstance(source, str) or not source.strip():
            return "offline native source chip is unidentified"
        raw = latency / candidate_ms
        if not _positive(raw):
            return "invalid raw offline ratio"
        self.record.update(
            base_latency_ms=latency, base_reference_chip=source, base_raw_speedup=raw
        )

        scaling = self.offline.get("_scaling_factors") or {}
        reference = scaling.get("reference_chip")
        if not _matches_chip(reference, source) or (
            qualified and not _matches_chip(reference, qualified)
        ):
            return "scaling reference does not match the measured source chip"
        unit = record.get("bottle_neck_unit")
        if unit not in consts._BOTTLENECK_TO_SF:
            return "offline row lacks a qualified bottleneck resource"
        try:
            target_name = torch_device_fn.get_device_name(self.device)
        except (AttributeError, TypeError, RuntimeError):
            target_name = consts._SELF_CHIP
        targets = [
            chip
            for chip in scaling.get("chips", [])
            if isinstance(chip, dict)
            and chip.get("vendor") == gems.vendor_name
            and _matches_chip(chip.get("chip"), target_name)
        ]
        if len(targets) != 1:
            return "no unique scaling entry for the measured target chip"
        target = targets[0]
        if unit != "mem":
            # The only floating-point arithmetic in this operator is FP32 RoPE.
            if unit != "cuda" or "scalar" in config["inputs"]["position_ids"]:
                return "unqualified compute resource or precision for this workload"
            dtype = torch.float32
        if (
            unit == "mem"
            and gems.vendor_name == "metax"
            and target["chip"] == "C550"
            and not _matches_chip("H800", source)
        ):
            # The shared lookup overrides C550 bandwidth with a calibration
            # against H800-class H100. Use only the supplied same-reference data.
            factor = next(
                (
                    target[k]["bandwidth_gbps"]
                    for k in ("achievable", "measured", "nominal")
                    if isinstance(target.get(k), dict)
                    and target[k].get("bandwidth_gbps") is not None
                ),
                None,
            )
        else:
            factor = consts.lookup_scaling_factor(
                {"_scaling_factors": {"reference_chip": reference, "chips": targets}},
                gems.vendor_name,
                unit,
                dtype,
            )
        if not _positive(factor) or not _positive(raw / factor):
            return "no positive finite scaling factor for this resource and chip"
        self.record.update(
            base_scaling_factor=factor,
            base_target_chip=target["chip"],
            base_bottleneck=unit,
            compared_speedup=raw / factor,
        )
        return None

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self.binding = dict(zip(workload.FIELDS, shape))
            self.dtype = dtype
            args, kwargs = workload.build_inputs(self.binding, dtype, self.device)
            config = workload.config(self.binding, dtype)
            self.expected = _reference(args)
            self.initial = [_storage(args[i]).clone() for i in workload.MUTATED_INPUTS]
            # Candidate failures must propagate even if the local native fails.
            _check(self.gems_op, args, self.expected)
            self.record = {
                "config": config,
                "candidate_accuracy": "PASS",
                "timing_protocol": self.protocol,
                "native_status": "NOT_RUN" if self.native_reason else "PENDING",
                "native_reason": self.native_reason,
                "offline_reason": None,
                "native_provenance": (
                    workload.NATIVE_PROVENANCE
                    if self.native_op or "latency_base" in self.to_bench_metrics
                    else None
                ),
            }
            self.records[str(dtype), config["case"]] = self.record
            yield (*args, kwargs)

    def record_shapes(self, *args, **kwargs):
        return {
            "case": workload.key_shape(self.binding),
            "inputs": super().record_shapes(*args, **kwargs),
        }

    def get_latency(self, op, *args, **kwargs):
        if op is self.torch_op:
            if self.native_op is None:
                return None
            # Separate, deterministically regenerated allocations prevent a
            # faulty native writing through a read-only alias into candidate data.
            native_args, _ = workload.build_inputs(
                self.binding, self.dtype, self.device
            )
            phase = "device dispatch"
            try:
                key = torch._C._dispatch_key_for_device(native_args[0].device.type)
                if not torch._C._dispatch_has_kernel_for_dispatch_key(
                    f"_C::{self.op_name}", key
                ):
                    self.record.update(
                        native_status="NOT_RUN",
                        native_reason=f"native has no {key} device kernel",
                    )
                    return None
                phase = "qualification"
                _check(self.native_op, native_args, self.expected)
                phase = "timing"
                latency, samples, warmups = _measure(
                    self.native_op, native_args, self.initial
                )
            except Exception as exc:
                self.record.update(
                    native_status="FAIL", native_reason=f"native {phase}: {exc!r}"
                )
                return None
            self.record.update(
                native_status="PASS", native_samples_ms=samples, native_warmups=warmups
            )
            return latency
        latency, samples, warmups = _measure(op, args, self.initial)
        try:
            self.record["offline_reason"] = self._compare_offline(latency)
        except (AttributeError, TypeError, ValueError) as exc:
            # Malformed optional JSON metadata is a per-row offline miss.
            self.record["offline_reason"] = f"invalid offline metadata: {exc}"

        self.record.update(candidate_samples_ms=samples, candidate_warmups=warmups)
        return latency


@pytest.mark.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert
def test_fused_kimi_k3_mla_decode_q_concat_kv_cache_insert():
    DecodeBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_native_slot,
        gems_op=gems.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert,
        is_inplace=True,
    ).run()
