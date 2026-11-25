# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Standard no-Graph benchmark of the complete iHC post public callable."""

import math
import os
import shlex
from pathlib import Path

import pytest
import torch
import triton

import flaggems_vllm
from baseline import triton_ihc_post as workloads

from . import base, conftest, consts

_H800_PROTOCOL = "triton-do-bench-no-graph-25ms-warmup-100ms-median"


def _positive_finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _native_slot(*args, **kwargs):
    raise RuntimeError("native slot must be resolved by get_latency")


def _check_output(output, args, snapshots):
    x, residual, _ = args
    assert isinstance(output, torch.Tensor)
    assert output.shape == residual.shape
    assert output.dtype == x.dtype and output.device == x.device
    assert output.is_contiguous()
    if output.numel():
        assert all(
            output.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
            for value in args
        ), "output aliases input storage"
    actual = output.detach().cpu()
    for start in range(0, x.shape[0], 32):
        stop = start + 32
        x32, residual32, post32 = (value[start:stop].float() for value in snapshots)
        product = torch.mul(post32.unsqueeze(-1), x32.unsqueeze(1))
        expected = torch.add(product, residual32).to(x.dtype)
        part = actual[start:stop]
        torch.testing.assert_close(part, expected, rtol=0, atol=0, equal_nan=True)
        zeros = expected == 0
        assert torch.equal(torch.signbit(part[zeros]), torch.signbit(expected[zeros]))


def _changed_inputs(args, snapshots, restore=False):
    changed = []
    for index, (value, saved) in enumerate(zip(args, snapshots)):
        actual = value.detach().cpu().contiguous().view(torch.uint8)
        if not torch.equal(actual, saved.view(torch.uint8)):
            changed.append(index)
            if restore:
                value.copy_(saved)
    return changed


def _check_native_arithmetic(native, dtype, channels, device):
    """Qualify separate FP32 mul/add before accepting any native timing row."""
    value = 1025 if dtype == torch.float16 else 129
    snapshots = (
        torch.full((1, 1), value, dtype=dtype),
        torch.full((1, channels, 1), -value, dtype=dtype),
        torch.full((1, channels), 1 + 2**-23, dtype=torch.float32),
    )
    args = tuple(value.to(device) for value in snapshots)
    try:
        output = native(*args)
        _check_output(output, args, snapshots)
        changed = _changed_inputs(args, snapshots)
        if changed:
            raise AssertionError(f"native mutated arithmetic fixture inputs {changed}")
    except Exception as exc:
        raise AssertionError(
            "native arithmetic protocol requires FP32 multiply then FP32 add "
            f"without FMA: {type(exc).__name__}: {exc}"
        ) from exc


class TritonIhcPostBenchmark(base.Benchmark):
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
        # Keep the synchronized public-call comparison for qualified native rows.
        # Expose an offline record to Benchmark.run only after candidate timing.
        with pytest.MonkeyPatch.context() as scoped:
            if not mode_is_explicit and config.mode == consts.BenchMode.KERNEL:
                scoped.setattr(config, "mode", consts.BenchMode.OPERATOR)
            data = config.base_data
            self._offline_data = None
            if data and config.user_desired_metrics != ["latency"]:
                entry = data.get(self.op_name, {})
                if "shapes" in entry and "variants" not in entry:
                    # Two layouts can have the same tensor shapes. Select the
                    # collector's full key through Benchmark.base_variant.
                    variants = {
                        key: {"shapes": {key: rows}}
                        for key, rows in entry["shapes"].items()
                    }
                    data = {**data, self.op_name: {"variants": variants}}
                self._offline_data = data
            scoped.setattr(config, "base_data", None)
            super().run()

    def init_user_config(self):
        super().init_user_config()
        if self.mode != consts.BenchMode.OPERATOR:
            raise ValueError(
                f"iHC post requires --mode operator (no Graph), got {self.mode.value}"
            )
        config = conftest.Config
        self._timing_protocol = (
            f"operator-wall-clock-mean-{config.warm_up}ms-{config.repetition}ms"
        )

    def _select_offline_row(self, args):
        data = self._offline_data
        if data is None:
            return None
        shape_key = self.base_variant
        dtype = str(self._dtype)
        try:
            variant = data[self.op_name]["variants"][shape_key]
            record = variant["shapes"][shape_key][dtype]
        except (KeyError, TypeError):
            return {"status": "N/A", "reason": "no exact H800 shape key and dtype"}

        reference = record.get("config") if isinstance(record, dict) else None
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
        source = reference.get("source") if isinstance(reference, dict) else None
        if (
            not isinstance(source, dict)
            or source.get("native_api") != self._config["source"]["native_api"]
            or any(reference.get(field) != self._config[field] for field in fields)
        ):
            return {
                "status": "N/A",
                "reason": "offline public call or input contract differs",
            }
        if (
            record.get("timing_protocol") != _H800_PROTOCOL
            or record.get("graph", False)
            or record.get("status") not in (None, "pass", "PASS", "ok")
            or not _positive_finite(record.get("latency_ms"))
        ):
            return {
                "status": "N/A",
                "reason": "invalid offline latency or timing protocol",
            }

        environment = data.get("_environment")
        scaling = data.get("_scaling_factors")
        if not all(
            isinstance(meta, dict) and meta.get("reference_chip") == "H800"
            for meta in (environment, scaling)
        ):
            return {
                "status": "N/A",
                "reason": "latency and scaling must both reference H800",
            }
        vendor = flaggems_vllm.vendor_name
        chip = base.torch_device_fn.get_device_name(args[0].device)
        result = {
            "status": "N/A",
            "shape_key": shape_key,
            "reference_chip": "H800",
            "target_chip": chip,
            "latency_ms": record["latency_ms"],
            "timing_protocol": _H800_PROTOCOL,
        }
        models = [
            entry["chip"]
            for entry in scaling.get("chips", [])
            if isinstance(entry, dict)
            and entry.get("vendor") == vendor
            and entry.get("chip")
        ]
        # A concrete SKU must agree with the table (e.g. 910B1 vs 910B4).
        # Leave a family-only device name to the framework's single-vendor rule.
        device_name = " " + chip.lower().replace("-", " ") + " "
        if any(char.isdigit() for char in chip) and not any(
            " " + model.lower().replace("-", " ") + " " in device_name
            for model in models
        ):
            return {**result, "reason": f"no H800 scaling entry for {chip}"}
        factor = consts.lookup_scaling_factor(
            data, vendor, record.get("bottle_neck_unit"), self._dtype
        )
        if not _positive_finite(factor):
            return {**result, "reason": "no valid scaling factor for this bottleneck"}
        conftest.Config.base_data = data
        return {
            **result,
            "status": "MATCH",
            "scaling_factor": factor,
            "scaling_chip": models[0] if len(models) == 1 else chip,
        }

    def record_shapes(self, *args, **kwargs):
        # Preserve normal tensor shapes while retaining layout and per-row N/A.
        self._row = {
            "arguments": super().record_shapes(*args, **kwargs),
            "shape_key": self.base_variant,
            "config": self._config,
            "offline_nvidia": None,
            "selected_source": None,
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
        return self._row

    def get_input_iter(self, dtype):
        native = None
        if "latency_base" not in self.to_bench_metrics:
            native_reason = "native timing disabled by --metrics"
        else:
            try:
                native = workloads.native()
                native_reason = workloads.NATIVE_UNAVAILABLE_REASON
            except Exception as exc:
                native_reason = f"native resolution failed: {type(exc).__name__}: {exc}"
            if native is self.gems_op:
                native = None
                native_reason = "native resolves to the Gems candidate"
            if native is None and not native_reason:
                native_reason = "installed native callable unavailable"
        self._native = native
        misses = []
        for n, hc, d, layout in self.shapes:
            conftest.Config.base_data = None
            binding = {"tokens": n, "channels": hc, "hidden": d, "layout": layout}
            args, kwargs = workloads.build_inputs(binding, dtype, self.device)
            snapshots = tuple(value.detach().cpu().clone() for value in args)
            self.base_variant = workloads.key_shape(binding)
            self._dtype = dtype
            self._snapshots = snapshots
            self._native_status = {
                "status": "N/A" if native is None else "VALIDATING",
                "reason": native_reason,
            }
            if native is not None:
                try:
                    _check_native_arithmetic(native, dtype, hc, self.device)
                    output = native(*args, **kwargs)
                    _check_output(output, args, snapshots)
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
            _check_output(output, args, snapshots)
            assert not _changed_inputs(args, snapshots), "candidate mutated inputs"
            del output
            self._config = workloads.config(binding, dtype)
            for value, name in zip(args, ("x", "residual", "post")):
                spec = self._config["inputs"][name]
                assert (
                    list(value.shape) == spec["shape"]
                    and str(value.dtype) == spec["dtype"]
                )
                assert (
                    list(value.stride()) == spec["stride"]
                    and value.storage_offset() == spec["storage_offset"]
                ), name
            yield (*args, kwargs)
            if self._native_status["reason"]:
                misses.append(
                    {
                        "shape_key": self.base_variant,
                        "reason": self._native_status["reason"],
                    }
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
        if op is self.gems_op:
            if self._native_status["status"] == "PASS":
                latency = super().get_latency(op, *args, **kwargs)
                self._row["selected_source"] = "native"
            else:
                latency = triton.testing.do_bench(
                    lambda: op(*args, **kwargs),
                    warmup=25,
                    rep=100,
                    return_mode="median",
                )
                self._row["timing"].update(
                    framework="triton.testing.do_bench",
                    statistic="median",
                    protocol=_H800_PROTOCOL,
                    warmup_budget_ms=25,
                    repetition_budget_ms=100,
                )
                offline = self._select_offline_row(args)
                self._row["offline_nvidia"] = offline
                if offline and offline["status"] == "MATCH":
                    self._row["selected_source"] = "nv_baseline"
            if not _positive_finite(latency):
                raise ValueError(f"invalid Gems latency: {latency}")
            assert not _changed_inputs(
                args, self._snapshots
            ), "timed Gems callable mutated inputs"
            return latency

        if self._native_status["status"] != "ACCURACY_PASS":
            return None
        try:
            latency = super().get_latency(self._native, *args, **kwargs)
            if not _positive_finite(latency):
                raise ValueError(f"invalid native latency: {latency}")
            changed = _changed_inputs(args, self._snapshots, restore=True)
            if changed:
                raise AssertionError(f"timed native callable mutated inputs {changed}")
        except Exception as exc:
            _changed_inputs(args, self._snapshots, restore=True)
            self._native_status.update(
                status="N/A",
                reason=f"native timing failed: {type(exc).__name__}: {exc}",
            )
            return None
        self._native_status.update(status="PASS", reason=None)
        return latency


@pytest.mark.triton_ihc_post
def test_triton_ihc_post(pytestconfig):
    bench = TritonIhcPostBenchmark(
        op_name=workloads.OP_NAME,
        torch_op=_native_slot,
        gems_op=flaggems_vllm.triton_ihc_post,
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
