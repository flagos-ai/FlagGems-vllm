# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Standard report path with per-call cache reset and no Graph timing."""

import math
import statistics
import time
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from baseline import triton_reshape_and_cache_flash as workloads
from flaggems_vllm.runtime import torch_device_fn

from . import base, conftest, consts

_SHAPE_FIELDS = ("T", "H", "D", "BS", "layout", "padding", "distribution", "cache")
_TIMING_PROTOCOL = {
    "graph": False,
    "reset": "full allocations before every invocation, outside event interval",
    "minimum_samples": 25,
    "budget_clock": "wall clock including untimed reset/synchronize",
    "reduction": "median",
    "unit": "ms",
}


def _positive_finite(value):
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _chip_key(name):
    name = str(name or "").strip().lower()
    return name[7:] if name.startswith("nvidia ") else name


class CacheFlashBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPE_DESC = ",".join(_SHAPE_FIELDS)

    def set_shapes(self, shape_file_path=None):
        # The baseline owns the matrix. Preserve explicit external shape files.
        super().set_shapes(shape_file_path)
        if Path(shape_file_path).resolve() == Path(self.DEFAULT_SHAPE_FILES).resolve():
            self.shapes = [
                tuple(binding[name] for name in _SHAPE_FIELDS)
                for binding in workloads.grid()
            ]

    def init_user_config(self):
        super().init_user_config()
        if self.mode != consts.BenchMode.KERNEL:
            raise ValueError("cache benchmark requires --mode kernel with no Graph")
        self._records = {}
        self.torch_op = None
        self._native_reason = "native timing disabled by --metrics"
        if "latency_base" in self.to_bench_metrics:
            try:
                native = workloads.native()
                if native is self.gems_op:
                    raise RuntimeError("native resolves to the Gems candidate")
                self.torch_op = native
                self._native_reason = workloads.NATIVE_UNAVAILABLE_REASON
                if native is None and self._native_reason is None:
                    self._native_reason = "installed vLLM callable unavailable"
            except Exception as exc:
                self._native_reason = (
                    f"native resolution failed: {type(exc).__name__}: {exc}"
                )

    def run(self):
        # Only expose a validated per-case projection to the shared lookup.
        # Its normal miss path raises when there is also no installed native.
        config = conftest.Config
        self._offline_data = config.base_data
        details = conftest.TEST_RESULTS.get(self.op_name, {}).get("details", [])
        first_new_detail = len(details)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(config, "base_data", None)
            super().run()
        for detail in conftest.TEST_RESULTS.get(self.op_name, {}).get("details", [])[
            first_new_detail:
        ]:
            misses = []
            for row in detail["result"]:
                shape = row["shape_detail"]
                row.update(self._records[detail["dtype"], shape["case"]])
                if row["speedup"] is not None:
                    row["selected_source"] = "native"
                elif row["compared_speedup"] is not None:
                    row["selected_source"] = "nv_baseline"
                else:
                    row["selected_source"] = None
                if row["native_reason"]:
                    misses.append({"shape": shape, "reason": row["native_reason"]})
            # Optest uses this field for latency-only rows and mixed sources.
            if misses:
                detail["native_baseline_skip_reason"] = "; ".join(
                    dict.fromkeys(miss["reason"] for miss in misses)
                )
                detail["native_baseline_misses"] = misses

    def _enable_offline(self, args):
        shape = self._shape_detail
        case = shape["case"]
        dtype = str(self._dtype)
        message = f"[native N/A] {case} {dtype}: {self._row_metadata['native_reason']}"
        try:
            record, scaling, factor = self._offline_record(case, shape["config"], dtype)
        except ValueError as exc:
            reason = f"{self._row_metadata['native_reason']}; offline NV: {exc}"
            self._row_metadata["base_miss_reason"] = reason
            message += f"; offline NV: {exc}"
        else:
            # The shared lookup indexes simple records by their first tensor
            # shape. Expose only the record already validated by exact case.
            conftest.Config.base_data = {
                self.op_name: {"shapes": {str(list(args[0].shape)): {dtype: record}}},
                "_scaling_factors": scaling,
            }
            self._row_metadata.update(
                base_latency_ms=record["latency_ms"],
                base_scaling_factor=factor,
                base_shape_key=case,
                base_data_source=f"offline NVIDIA ({scaling['reference_chip']})",
            )
        print(message)
        conftest.emit_record_logger(message)

    def _offline_record(self, case, config, dtype):
        data = self._offline_data
        if data is None:
            raise ValueError("no baseline data")
        try:
            record = data[self.op_name]["shapes"][case][dtype]
        except (KeyError, TypeError) as exc:
            raise ValueError("no record for exact case and dtype") from exc
        if not isinstance(record, dict) or record.get("config") != config:
            raise ValueError("configuration mismatch")
        if record.get("native_status") != "PASS" or not _positive_finite(
            record.get("latency_ms")
        ):
            raise ValueError("no qualified positive native latency")
        protocol = record.get("timing_protocol")
        expected_protocol = self._row_metadata["timing_protocol"]
        if not isinstance(protocol, dict) or any(
            protocol.get(key) != value for key, value in expected_protocol.items()
        ):
            raise ValueError("incompatible timing protocol")

        scaling = data.get("_scaling_factors")
        if not isinstance(scaling, dict) or not scaling.get("reference_chip"):
            raise ValueError("missing reference chip metadata")
        reference = scaling["reference_chip"]
        try:
            chip = _chip_key(torch_device_fn.get_device_name(0))
        except (AttributeError, RuntimeError, TypeError):
            chip = _chip_key(consts._SELF_CHIP)
        if not chip:
            raise ValueError("missing target chip")
        if flaggems_vllm.vendor_name == "nvidia" and _chip_key(reference) == chip:
            # The common unscaled ratio is valid for the reference chip itself.
            return record, {"reference_chip": reference}, 1.0
        # The shared factor helper may fall back to another SKU from the same
        # vendor. Only expose the exact target SKU to that helper.
        chips = scaling.get("chips")
        targets = [
            target
            for target in (chips if isinstance(chips, list) else [])
            if isinstance(target, dict)
            and target.get("vendor") == flaggems_vllm.vendor_name
            and _chip_key(target.get("chip")) == chip
        ]
        if len(targets) != 1:
            raise ValueError("no unique scaling entry for the target chip")
        scaling = {"reference_chip": reference, "chips": targets}
        factor = consts.lookup_scaling_factor(
            {"_scaling_factors": scaling},
            flaggems_vllm.vendor_name,
            record.get("bottle_neck_unit"),
            dtype,
        )
        if not _positive_finite(factor):
            raise ValueError("no valid scaling factor for the target chip and resource")
        return record, scaling, factor

    def record_shapes(self, *args, **kwargs):
        # Stable, actual tensor shapes with variant metadata for optest keys.
        self._shape_detail = {
            "case": workloads.key_shape(self._binding),
            "inputs": super().record_shapes(*args, **kwargs),
            "config": workloads.config(self._binding, self._dtype),
        }
        return self._shape_detail

    @staticmethod
    def _bytes(tensor):
        return workloads.backing(tensor).view(torch.uint8)

    def _reset(self, args):
        for cache, saved in zip(args[2:4], self._poison):
            self._bytes(cache).copy_(saved)
        torch_device_fn.synchronize()

    @staticmethod
    def _identity(tensor):
        return (
            tensor.data_ptr(),
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tensor.shape,
            tensor.stride(),
            tensor.dtype,
        )

    def _check_result(self, result, tensors, identities):
        assert result is None
        for tensor, identity, expected in zip(tensors, identities, self._expected):
            assert self._identity(tensor) == identity
            torch.testing.assert_close(
                self._bytes(tensor).cpu(), expected, rtol=0, atol=0
            )

    def _build_inputs(self, dtype):
        args, kwargs = workloads.build_inputs(self._binding, dtype, self.device)
        # Benchmark.unpack_to_args_kwargs ignores bare strings. Use the same
        # kwargs convention for both implementations.
        return args[:5], dict(
            kv_cache_dtype=args[5], k_scale=args[6], v_scale=args[7], **kwargs
        )

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            conftest.Config.base_data = None
            if len(shape) != len(_SHAPE_FIELDS):
                raise ValueError(
                    f"cache flash shapes must have {self.DEFAULT_SHAPE_DESC}"
                )
            self._binding, self._dtype = dict(zip(_SHAPE_FIELDS, shape)), dtype
            args, kwargs = self._build_inputs(dtype)
            self._poison = [self._bytes(t).clone() for t in args[2:4]]
            self._expected = [
                *(self._bytes(t).cpu().clone() for t in args[:2]),
                *workloads.reference((*args, *kwargs.values())),
                *(
                    self._bytes(t).cpu().clone()
                    for t in (args[4], kwargs["k_scale"], kwargs["v_scale"])
                ),
            ]
            self._row_metadata = {
                "native_status": (
                    "NOT_RUN"
                    if "latency_base" not in self.to_bench_metrics
                    else "UNAVAILABLE"
                ),
                "native_reason": self._native_reason,
                "native_provenance": (
                    workloads.NATIVE_PROVENANCE if self.torch_op else None
                ),
                "timing_protocol": {
                    **_TIMING_PROTOCOL,
                    "warmup_budget_ms": conftest.Config.warm_up,
                    "sample_budget_ms": conftest.Config.repetition,
                },
            }
            self._records[str(dtype), workloads.key_shape(self._binding)] = (
                self._row_metadata
            )
            yield (*args, kwargs)

    def get_latency(self, op, *args, **kwargs):
        # The common timer has no reset hook outside the measured interval.
        candidate = op is self.gems_op
        phase = "qualification"
        try:
            if not candidate:
                # A failed native call must not contaminate the candidate's
                # input allocations or tensor metadata.
                args, kwargs = self._build_inputs(self._dtype)
            tensors = (*args, kwargs["k_scale"], kwargs["v_scale"])
            identities = [self._identity(t) for t in tensors]
            self._reset(args)
            result = op(*args, **kwargs)
            torch_device_fn.synchronize()
            self._check_result(result, tensors, identities)
            phase = "timing"
            start_budget = time.perf_counter()
            while time.perf_counter() - start_budget < conftest.Config.warm_up / 1000:
                self._reset(args)
                op(*args, **kwargs)
                torch_device_fn.synchronize()
            samples = []
            start_budget = time.perf_counter()
            while (
                len(samples) < _TIMING_PROTOCOL["minimum_samples"]
                or time.perf_counter() - start_budget
                < conftest.Config.repetition / 1000
            ):
                self._reset(args)
                begin = torch_device_fn.Event(enable_timing=True)
                end = torch_device_fn.Event(enable_timing=True)
                begin.record()
                result = op(*args, **kwargs)
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end))
            self._check_result(result, tensors, identities)
            latency = statistics.median(samples)
            if not _positive_finite(latency):
                raise ValueError(f"invalid median latency: {latency}")
        except Exception as exc:
            if candidate:
                raise
            self._row_metadata.update(
                native_status="FAIL",
                native_reason=f"native {phase} failed: {type(exc).__name__}: {exc}",
            )
            return None

        self._row_metadata.setdefault("samples_ms", {})[
            "candidate" if candidate else "native"
        ] = samples
        if candidate:
            self._row_metadata["candidate_accuracy"] = "PASS"
            if "speedup" not in self.to_bench_metrics:
                self._row_metadata["base_miss_reason"] = "speedup disabled by --metrics"
            elif self._row_metadata["native_status"] != "PASS":
                self._enable_offline(args)
        else:
            self._row_metadata.update(native_status="PASS", native_reason=None)
        return latency


@pytest.mark.triton_reshape_and_cache_flash
def test_triton_reshape_and_cache_flash():
    bench = CacheFlashBenchmark(
        op_name=workloads.OP_NAME,
        torch_op=None,
        gems_op=flaggems_vllm.triton_reshape_and_cache_flash,
        dtypes=workloads.DTYPES,
        is_inplace=workloads.IS_INPLACE,
    )
    bench.run()
