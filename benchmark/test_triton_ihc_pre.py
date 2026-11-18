# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Standard no-Graph benchmark of the complete iHC pre public callable."""

import math
import os
import shlex
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from baseline import triton_ihc_pre as workloads

from . import base, conftest, consts


def _native_slot(*args, **kwargs):
    raise RuntimeError("native slot must be resolved by get_latency")


@workloads._cpu_single_thread()
def _check_output(output, args, expected):
    assert isinstance(output, tuple) and len(output) == 2
    for index, (actual, ref) in enumerate(zip(output, expected)):
        assert actual.shape == ref.shape and actual.dtype == ref.dtype
        assert actual.device == args[0].device
        torch.testing.assert_close(
            actual.cpu(),
            ref.cpu(),
            rtol=1e-2 if index == 0 else 1e-5,
            atol=2e-2 if index == 0 else 2e-5,
            equal_nan=True,
        )


@workloads._cpu_single_thread()
def _changed_inputs(args, snapshots, restore=False):
    changed = []
    for index, (value, saved) in enumerate(zip(args[:4], snapshots)):
        actual = value.detach().cpu().contiguous().view(torch.uint8)
        if not torch.equal(actual, saved.contiguous().view(torch.uint8)):
            changed.append(index)
            if restore:
                value.copy_(saved)
    return changed


class TritonIhcPreBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPE_DESC = "tokens, channels, hidden, layout"
    DEFAULT_SHAPES = [
        (row["tokens"], row["channels"], row["hidden"], row["layout"])
        for row in workloads.grid()
    ]

    def set_shapes(self, shape_file_path=None):
        # The repository's generic Benchmark YAML must not replace this grid.
        # Explicit external shape files retain the normal base-class behavior.
        default = Path(__file__).with_name("core_shapes.yaml")
        if (
            shape_file_path is not None
            and Path(shape_file_path).resolve() == default.resolve()
        ):
            self.shapes = list(self.DEFAULT_SHAPES)
            self.shape_desc = self.DEFAULT_SHAPE_DESC
            return
        super().set_shapes(shape_file_path)

    def run(self, *, mode_is_explicit=False):
        config = conftest.Config
        # optest leaves --mode at its kernel default. Use the existing eager
        # public-call timer for that entry point as well as --mode operator.
        with pytest.MonkeyPatch.context() as scoped:
            if not mode_is_explicit and config.mode == consts.BenchMode.KERNEL:
                print("iHC pre uses --mode operator (synchronized mean, no Graph)")
                scoped.setattr(config, "mode", consts.BenchMode.OPERATOR)
            self._offline_shapes = None
            self._offline_entry = {"shapes": {}}
            if config.user_desired_metrics == ["latency"]:
                scoped.setattr(config, "base_data", None)
            elif config.base_data is not None:
                entry = config.base_data.get(self.op_name, {})
                self._offline_shapes = (entry or {}).get("shapes") or {}
                scoped.setattr(
                    config,
                    "base_data",
                    {**config.base_data, self.op_name: self._offline_entry},
                )
            super().run()

    def init_user_config(self):
        super().init_user_config()
        if self.mode != consts.BenchMode.OPERATOR:
            raise ValueError(
                f"iHC pre requires --mode operator (no Graph), got {self.mode.value}"
            )
        config = conftest.Config
        self._timing_protocol = (
            f"operator-wall-clock-mean-{config.warm_up}ms-{config.repetition}ms"
        )

    def _select_offline_row(self, shape_key, config):
        if self._offline_shapes is None:
            return None

        # The shared matcher can fall back to tensor sizes alone. Expose only
        # this row's exact key/config, then let it do the normal ratio calculation.
        self._offline_entry["shapes"] = {}
        dtype = str(self._dtype)
        record = (self._offline_shapes.get(shape_key) or {}).get(dtype)
        reason = "no offline NVIDIA record for this shape key and dtype"
        if isinstance(record, dict):
            reference = record.get("config") or {}
            fields = (
                "inputs",
                "outputs",
                "dims",
                "mutation",
                "input_aliasing",
                "arithmetic",
                "tolerance",
                "timing_scope",
            )
            latency = record.get("latency_ms")
            protocol = record.get("timing_protocol")
            if not isinstance(reference, dict) or not all(
                reference.get(field) == config[field] for field in fields
            ):
                reason = "offline NVIDIA configuration differs from the benchmark"
            elif (
                record.get("status") not in (None, "pass", "PASS", "ok")
                or record.get("graph", False)
                or not isinstance(latency, (int, float))
                or isinstance(latency, bool)
                or not math.isfinite(latency)
                or latency <= 0
            ):
                reason = "offline NVIDIA record is invalid or lacks a positive latency"
            elif protocol != self._timing_protocol:
                reason = (
                    f"offline NVIDIA timing protocol {protocol!r} does not match "
                    f"{self._timing_protocol!r}"
                )
            else:
                key = str(config["inputs"]["x"]["shape"])
                self._offline_entry["shapes"] = {key: {dtype: record}}
                return {
                    "status": "MATCH",
                    "shape_key": shape_key,
                    "latency_ms": latency,
                    "timing_protocol": protocol,
                }
        return {"status": "N/A", "reason": reason}

    def record_shapes(self, *args, **kwargs):
        # Preserve normal tensor shapes while retaining layout and per-row N/A.
        return {
            "arguments": super().record_shapes(*args, **kwargs),
            "shape_key": workloads.key_shape(self._binding),
            "config": self._config,
            "offline_nvidia": self._offline_status,
            "native": self._native_status,
            "timing": {
                "scope": "complete public callable, allocation and contiguous copies",
                "framework": "Benchmark.get_latency / operator",
                "graph": False,
                "unit": "ms",
                "statistic": "synchronized wall-clock mean",
                "protocol": self._timing_protocol,
                "warmup_budget_ms": conftest.Config.warm_up,
                "repetition_budget_ms": conftest.Config.repetition,
            },
            "active_rows": len(self.shapes) * len(self.to_bench_dtypes),
        }

    def get_input_iter(self, dtype):
        native = None
        if "latency_base" not in self.to_bench_metrics:
            native_reason = "native timing disabled by --metrics"
        else:
            native = workloads.native()
            native_reason = workloads.NATIVE_UNAVAILABLE_REASON
            if native is self.gems_op:
                native = None
                native_reason = "native resolves to the Gems candidate"
            if native is None and not native_reason:
                native_reason = "installed native callable unavailable"
        self._native = native
        misses = []
        for n, hc, d, layout in self.shapes:
            binding = {"tokens": n, "channels": hc, "hidden": d, "layout": layout}
            with workloads._cpu_single_thread():
                args, kwargs = workloads.build_inputs(binding, dtype, self.device)
                snapshots = tuple(value.detach().cpu().clone() for value in args[:4])
                self._binding, self._dtype = binding, dtype
                self._snapshots = snapshots
                from tests.test_triton_ihc_pre import ihc_pre_reference

                expected = ihc_pre_reference(*snapshots, *args[4:])
                self._native_status = {
                    "status": "N/A" if native is None else "VALIDATING",
                    "reason": native_reason,
                }
                if native is not None:
                    try:
                        output = native(*args, **kwargs)
                        _check_output(output, args, expected)
                        del output
                        changed = _changed_inputs(args, snapshots, restore=True)
                        if changed:
                            raise AssertionError(
                                f"native mutated positional inputs {changed}"
                            )
                    except Exception as exc:
                        _changed_inputs(args, snapshots, restore=True)
                        self._native_status.update(
                            status="N/A",
                            reason=f"native qualification failed: {type(exc).__name__}: {exc}",
                        )
                    else:
                        self._native_status.update(status="ACCURACY_PASS", reason=None)
                # Candidate failures remain outside the native exception scope.
                output = self.gems_op(*args, **kwargs)
                _check_output(output, args, expected)
                assert not _changed_inputs(args, snapshots), "candidate mutated inputs"
                del output
                self._config = workloads.config(binding, dtype)
                shape_key = workloads.key_shape(binding)
                self._offline_status = self._select_offline_row(shape_key, self._config)
            # Restore CPU threads before the framework enters device timing.
            yield (*args, kwargs)
            if self._native_status["reason"]:
                misses.append(
                    {"shape_key": shape_key, "reason": self._native_status["reason"]}
                )
        if misses:
            # optest selects its nullable-ratio parser from this metadata.
            conftest.update_result(
                self.op_name,
                {
                    "dtype": str(dtype),
                    "native_baseline_skip_reason": "; ".join(
                        dict.fromkeys(row["reason"] for row in misses)
                    ),
                    "native_baseline_misses": misses,
                },
            )

    def get_latency(self, op, *args, **kwargs):
        # Base.run aborts the iterator for any timing failure. Limit recovery to
        # native failures so candidate rows and standard JSON remain available.
        candidate = op is self.gems_op
        if not candidate and self._native_status["status"] != "ACCURACY_PASS":
            return None
        try:
            latency = super().get_latency(
                op if candidate else self._native, *args, **kwargs
            )
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"invalid latency: {latency}")
            changed = _changed_inputs(args, self._snapshots, restore=not candidate)
            if changed:
                raise AssertionError(f"timed callable mutated inputs {changed}")
        except Exception as exc:
            if candidate:
                raise
            _changed_inputs(args, self._snapshots, restore=True)
            self._native_status.update(
                status="N/A",
                reason=f"native timing failed: {type(exc).__name__}: {exc}",
            )
            return None
        if not candidate:
            self._native_status.update(status="PASS", reason=None)
        return latency


@pytest.mark.triton_ihc_pre
def test_triton_ihc_pre(pytestconfig):
    bench = TritonIhcPreBenchmark(
        op_name=workloads.OP_NAME,
        torch_op=_native_slot,
        gems_op=flaggems_vllm.triton_ihc_pre,
        dtypes=workloads.DTYPES,
    )
    pytest_args = (
        *pytestconfig.getini("addopts"),
        *shlex.split(os.environ.get("PYTEST_ADDOPTS", "")),
        *pytestconfig.invocation_params.args,
    )
    bench.run(
        mode_is_explicit=any(
            arg.partition("=")[0] in ("--mode", "--fg_mode") for arg in pytest_args
        )
    )
