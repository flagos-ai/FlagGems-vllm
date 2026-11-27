# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""No-Graph timing of the complete public call, including its TP collective."""

import math
import os
import shlex
import statistics
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import flaggems_vllm as gems
from baseline import fused_moe_lora as workload
from benchmark import conftest as report
from benchmark.base import Benchmark
from benchmark.consts import BenchLevel, BenchMode, lookup_scaling_factor
from flaggems_vllm.runtime import torch_device_fn
from tests import test_fused_moe_lora as accuracy

_NATIVE_SLOT = object()
_COLLECTOR_PROTOCOL = "no-graph-device-event-reset-outside-5warmup-25sample-median"


def _rank_error(error, world):
    if world == 1:
        return error
    return "; ".join(accuracy.tp_errors(error)) or None


def _rank_max(samples, device, world):
    if world == 1:
        return samples
    # Reduce corresponding samples, before taking any statistic. No measurement
    # includes this reporting collective or the barrier preceding each sample.
    times = torch.tensor(samples, dtype=torch.float32, device=device)
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    return times.tolist()


def _measure(op, args, kwargs, budget, world):
    config = report.Config
    out = args[0]
    initial = out.clone()
    start = torch_device_fn.Event(enable_timing=True)
    end = torch_device_fn.Event(enable_timing=True)

    def event_sample():
        out.copy_(initial)
        torch_device_fn.synchronize()
        if world > 1:
            dist.barrier()
            torch_device_fn.synchronize()
        start.record()
        op(*args, **kwargs)
        end.record()
        torch_device_fn.synchronize()
        return start.elapsed_time(end)

    def sample():
        if world == 1:
            return event_sample()
        error = None
        try:
            ms = event_sample()
        except Exception as exc:
            error = repr(exc)
        # Make a rank-local timing failure visible before a peer starts its
        # next collective. Exchange diagnostics on the TP CPU group.
        error = _rank_error(error, world)
        if error:
            raise RuntimeError(error)
        return ms

    try:
        warmups, repetitions = 5, 25
        if budget:
            pilot = _rank_max([sample() for _ in range(3)], out.device, world)
            typical_ms = statistics.median(pilot)
            if not math.isfinite(typical_ms) or typical_ms <= 0:
                raise ValueError(f"invalid event calibration {pilot}")
            if "--warmup" in budget:
                warmups = max(0, int(config.warm_up / typical_ms))
            if "--iter" in budget:
                repetitions = max(1, int(config.repetition / typical_ms))
        for _ in range(warmups):
            sample()
        samples = _rank_max([sample() for _ in range(repetitions)], out.device, world)
        latency = statistics.median(samples)
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"invalid event median {latency} ms")
        return latency, warmups, repetitions, samples
    finally:
        out.copy_(initial)
        torch_device_fn.synchronize()


class FusedMoELoRABenchmark(Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPE_DESC = "full workload key from baseline.fused_moe_lora.key_shape"

    def __init__(self, pytestconfig, world=1, rank=0, device=None):
        super().__init__(
            op_name=workload.OP_NAME,
            torch_op=_NATIVE_SLOT,
            gems_op=gems.fused_moe_lora,
            is_inplace=True,
            dtypes=workload.DTYPES,
        )
        self.rank, self.world = rank, world
        self.device = device or gems.device
        self.rows = workload.tp_grid(world) if world > 1 else workload.grid()
        self.by_key = {tuple(workload.key_shape(b)): b for b in self.rows}
        if len(self.by_key) != len(self.rows):
            raise ValueError("duplicate fused_moe_lora workload key")
        options = [
            *pytestconfig.getini("addopts"),
            *shlex.split(os.environ.get("PYTEST_ADDOPTS", "")),
            *pytestconfig.invocation_params.args,
        ]
        self.budget = {arg.partition("=")[0] for arg in options} & {
            "--warmup",
            "--iter",
        }
        self.protocol = _COLLECTOR_PROTOCOL
        if self.budget:
            warm = f"{report.Config.warm_up}ms" if "--warmup" in self.budget else "5"
            rep = f"{report.Config.repetition}ms" if "--iter" in self.budget else "25"
            self.protocol = (
                f"no-graph-device-event-reset-outside-{warm}warmup-{rep}sample-median"
            )
        if self.world > 1:
            self.protocol += f"-max-across-{self.world}-ranks-per-sample"

    def set_shapes(self, shape_file_path):
        if Path(shape_file_path).resolve() == Path(self.DEFAULT_SHAPE_FILES):
            self.shapes = list(self.by_key)
            if self.world == 1 and (
                report.Config.query or report.Config.bench_level == BenchLevel.CORE
            ):
                core = {"decode_single", "sorted_prefill", "rank129"}
                self.shapes = [k for k in self.shapes if self.by_key[k]["case"] in core]
        else:
            super().set_shapes(shape_file_path)
            unknown = [key for key in self.shapes if key not in self.by_key]
            if unknown:
                raise ValueError(f"unknown fused_moe_lora workload keys: {unknown}")
        if not self.shapes:
            raise ValueError("fused_moe_lora workload selection is empty")

    def run(self):
        cfg = report.Config
        if not cfg.query and cfg.mode != BenchMode.KERNEL:
            raise ValueError(
                "fused_moe_lora requires --mode kernel for no-Graph events"
            )
        self.offline = (cfg.base_data or {}).get(self.op_name, {}).get("shapes", {})
        self.lookup_entry = {"shapes": {}}
        data = cfg.base_data
        if cfg.user_desired_metrics == ["latency"]:
            data = None
        elif data is not None:
            # Give the shared lookup only an exact, protocol-compatible row.
            data = {**data, self.op_name: self.lookup_entry}
        with patch.object(cfg, "base_data", data):
            super().run()

    def _inputs(self, binding, dtype):
        if self.world == 1:
            return workload.build_inputs(binding, dtype, self.device)
        return workload.build_tp_inputs(binding, dtype, self.device, self.rank)

    def get_input_iter(self, dtype):
        misses = []
        rows = []
        for key in self.shapes:
            binding = self.by_key[key]
            self.binding = binding
            args, kwargs = self._inputs(binding, dtype)
            if self.world == 1:
                self.expected = accuracy.reference(binding, args)
            else:
                shards = [
                    workload.build_tp_inputs(binding, dtype, "cpu", rank)[0]
                    for rank in range(self.world)
                ]
                self.expected = accuracy.reference_tp(binding, shards, self.rank)
                del shards
            self.detail = {
                "shape_key": list(key),
                "config": workload.config(binding, dtype, rank=self.rank),
                "candidate": {"accuracy": "NOT_RUN", "timing": "NOT_RUN"},
                "native": {"accuracy": "NOT_RUN", "timing": "NOT_RUN"},
                "timing_protocol": self.protocol,
                "graph": False,
                "unit": "ms",
            }
            self._select_offline(args[0])
            if "latency_base" not in self.to_bench_metrics:
                self.detail["native"]["reason"] = "native timing disabled by --metrics"
            self.candidate_output = None
            self._qualify(self.gems_op, args, kwargs, native=False)
            yield (*args, kwargs)
            rows.append(self.detail)
            reason = self.detail["native"].get("reason")
            if reason:
                misses.append({"shape": list(key), "reason": reason})
        note = {"dtype": str(dtype), "workload_rows": rows}
        if misses:
            note.update(
                native_baseline_misses=misses,
                native_baseline_skip_reason="; ".join(
                    dict.fromkeys(m["reason"] for m in misses)
                ),
            )
        report.update_result(self.op_name, note)

    def _select_offline(self, output):
        self.lookup_entry["shapes"] = {}
        if report.Config.base_data is None:
            self.detail["offline_reason"] = "no offline NVIDIA comparison requested"
            return
        key = str(self.detail["shape_key"])
        dtype = str(output.dtype)
        row = self.offline.get(key, {}).get(dtype)
        if not row:
            self.detail["offline_reason"] = "no exact workload key and dtype"
            return
        ms = row.get("latency_ms")
        if (
            row.get("config") != self.detail["config"]
            or row.get("timing_protocol") != self.protocol
            or row.get("status") not in (None, "PASS", "pass", "ok")
            or not isinstance(ms, (float, int))
            or not math.isfinite(ms)
            or ms <= 0
        ):
            self.detail["offline_reason"] = "invalid offline timing or configuration"
            return
        # The shared collector can time a native call without qualifying its
        # output. A timed offset/high-rank row alone cannot establish parity.
        if row.get("accuracy", row.get("status")) not in ("PASS", "pass"):
            self.detail["offline_reason"] = "offline native accuracy unverified"
            return
        data = report.Config.base_data
        factors = data.get("_scaling_factors", {})
        source = data.get("_environment", {}).get("hardware", {}).get("gpu", {})
        reference = factors.get("reference_chip")
        actual = torch_device_fn.get_device_name(output.device)
        if not reference or reference.lower() not in (source.get("name") or "").lower():
            self.detail["offline_reason"] = (
                "reference chip identity is missing or mismatched"
            )
            return
        self.scaling_dtype = None
        if reference.lower() not in actual.lower():
            if not any(
                chip.get("vendor") == gems.vendor_name
                and chip.get("chip")
                and chip["chip"].lower() in actual.lower()
                for chip in factors.get("chips", ())
            ):
                self.detail["offline_reason"] = (
                    "no scaling record for the measured target chip"
                )
                return
            resource = row.get("bottle_neck_unit")
            compute = row.get("compute_dtype") if resource != "mem" else None
            if resource != "mem" and compute not in (
                "torch.float16",
                "torch.bfloat16",
                "torch.float32",
            ):
                self.detail["offline_reason"] = "offline compute precision unverified"
                return
            self.scaling_dtype = (
                getattr(torch, compute.split(".")[-1]) if compute else None
            )
            factor = lookup_scaling_factor(
                data, gems.vendor_name, resource, self.scaling_dtype or output.dtype
            )
            if (
                not factor
                or not math.isfinite(factor)
                or factor <= 0
                or factor
                != lookup_scaling_factor(
                    data,
                    gems.vendor_name,
                    resource,
                    self.scaling_dtype or output.dtype,
                    actual,
                )
            ):
                self.detail["offline_reason"] = (
                    "no positive scaling factor for the measured resource"
                )
                return
        self.lookup_entry["shapes"] = {str(list(output.shape)): {dtype: row}}

    def record_shapes(self, *args, **kwargs):
        # The optest consumer stringifies shape_detail as the stable row ID.
        # Keep samples, status and machine-specific reasons in workload_rows.
        return {
            "shape_key": self.detail["shape_key"],
            "arguments": super().record_shapes(*args, **kwargs),
        }

    def get_latency(self, op, *args, **kwargs):
        native = op is _NATIVE_SLOT
        status = self.detail["native" if native else "candidate"]
        if native:
            reason = None
            try:
                op = workload.native()
                if not callable(op):
                    reason = workload.NATIVE_UNAVAILABLE_REASON or "native unavailable"
                elif op is self.gems_op:
                    reason = "installed native resolves to the Gems candidate"
                else:
                    reason = workload.unsupported_reason(self.binding, args[0].dtype)
            except Exception as exc:
                reason = f"native resolution: {exc!r}"
            reason = _rank_error(reason, self.world)
            if reason:
                status["reason"] = reason
                return None
            # A defective native can modify inputs. Keep candidate storage pristine.
            args, kwargs = self._inputs(self.binding, args[0].dtype)

            if not self._qualify(op, args, kwargs, native=True):
                return None

        reason = None
        try:
            ms, warmups, repetitions, samples = _measure(
                op, args, kwargs, self.budget, self.world
            )
        except Exception as exc:
            reason = f"timing: {exc!r}"
        reason = _rank_error(reason, self.world)
        status["timing"] = "FAIL" if reason else "PASS"
        if reason:
            status["reason"] = reason
            if not native:
                self._fail_candidate(args, reason)
            return None
        status.update(warmups=warmups, repetitions=repetitions, samples_ms=samples)
        return ms

    def _qualify(self, op, args, kwargs, *, native):
        status = self.detail["native" if native else "candidate"]
        initial = args[0].clone()
        snapshots = accuracy.snapshot_inputs(args)
        reason = None
        stage = "oracle qualification"
        try:
            result = op(*args, **kwargs)
            torch_device_fn.synchronize()
            accuracy.assert_result(self.binding, args, result, self.expected)
            accuracy.assert_inputs_unchanged(args, snapshots)
            if native:
                stage = "native vs candidate"
                accuracy.assert_result(
                    self.binding, args, result, self.candidate_output
                )
            elif "latency_base" in self.to_bench_metrics:
                self.candidate_output = args[0].detach().to("cpu", copy=True)
        except Exception as exc:
            reason = f"{stage}: {exc!r}"
        finally:
            args[0].copy_(initial)
        reason = _rank_error(reason, self.world)
        status["accuracy"] = "FAIL" if reason else "PASS"
        if reason:
            status["reason"] = reason
            if not native:
                self._fail_candidate(args, reason)
        return reason is None

    def _fail_candidate(self, args, reason):
        report.update_result(
            self.op_name,
            {"dtype": str(args[0].dtype), "workload_rows": [self.detail]},
        )
        raise RuntimeError(f"Gems {self.binding['case']}: {reason}")


@pytest.mark.fused_moe_lora
def test_fused_moe_lora(pytestconfig):
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        pytest.skip("single-device rows require an ordinary pytest launch")
    FusedMoELoRABenchmark(pytestconfig).run()


@pytest.mark.fused_moe_lora
def test_fused_moe_lora_tp(pytestconfig):
    world = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    if rank and report.Config.record_json:
        # pytest's session-end hook on each torchrun worker writes REPORT_FILE.
        path = Path(report.REPORT_FILE)
        report.REPORT_FILE = str(path.with_name(f"{path.stem}.rank{rank}{path.suffix}"))
    if report.Config.query:
        FusedMoELoRABenchmark(pytestconfig, world=max(2, world)).run()
    else:
        with accuracy.tp_context() as (rank, world, device):
            FusedMoELoRABenchmark(pytestconfig, world, rank, device).run()
