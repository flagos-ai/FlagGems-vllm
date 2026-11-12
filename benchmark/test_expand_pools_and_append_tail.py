# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Standard Benchmark adapter for the frozen pool-expansion matrix.

EXPAND_BENCH_ENTRY selects candidate (default), installed, pinned, or before.
Independent provider processes use the same Benchmark.run/get_latency path.
The historical explicit-event diagnostic remains in its separate file.
"""

import hashlib
import importlib.util
import inspect
import json
import logging
import math
import os
import sys
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from baseline import expand_pools_and_append_tail as baseline
from benchmark import conftest as benchmark_config
from benchmark.base import Benchmark
from flaggems_vllm.runtime import torch_device_fn

pytestmark = pytest.mark.expand_pools_and_append_tail
_LOG = logging.getLogger(__name__)
_PROTOCOL = "benchmark-base-kernel-v1"
_BEFORE_SHA256 = "77cb5928f7570a3943668ab1adf79e3a624ff32e8290be3d93d57a99297140b0"
_ACCURACY = (
    "CPU int64 oracle; two negative sentinels; fresh output; inputs/backing unchanged"
)


def _native_slot(*args, **kwargs):
    raise RuntimeError("native slot sentinel must be intercepted before timing")


def _unavailable_provider(*args, **kwargs):
    raise RuntimeError("unavailable provider sentinel must never be called")


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _describe(op):
    path = inspect.getsourcefile(op)
    return {
        "module": getattr(op, "__module__", None),
        "qualname": getattr(op, "__qualname__", None),
        "file": path,
        "file_sha256": _sha256(Path(path).read_bytes()) if path else None,
        "source_sha256": _sha256(inspect.getsource(op).encode()),
    }


def _independent_native(op):
    assert op is not flaggems_vllm.expand_pools_and_append_tail
    descriptor = _describe(op)
    assert "flaggems_vllm" not in (descriptor["file"] or "")
    assert "flaggems_vllm" not in (descriptor["module"] or "")
    assert "flaggems_vllm" not in inspect.getsource(op)
    return descriptor


def _load_before():
    path = Path(os.environ["EXPAND_BEFORE_FILE"]).expanduser().resolve(strict=True)
    if _sha256(path.read_bytes()) != _BEFORE_SHA256:
        raise ValueError(
            "EXPAND_BEFORE_FILE does not match the original before snapshot"
        )
    name = "_expand_original_before_" + _BEFORE_SHA256[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    op = getattr(module, baseline.OP_NAME)
    if not callable(op) or op is flaggems_vllm.expand_pools_and_append_tail:
        raise RuntimeError("before must resolve to the original independent callable")
    return op, {"kind": "before", "file": str(path), "file_sha256": _BEFORE_SHA256}


def _resolve_native(entry):
    pinned = os.environ.get("EXPAND_PINNED_NATIVE_FILE")
    if entry == "installed" and pinned:
        raise ValueError("installed provider requires EXPAND_PINNED_NATIVE_FILE unset")
    if entry == "pinned":
        if not pinned:
            raise ValueError("pinned provider requires EXPAND_PINNED_NATIVE_FILE")
        if flaggems_vllm.vendor_name != "nvidia":
            raise ValueError("pinned native provider is an explicit NVIDIA-only opt-in")
    if pinned and flaggems_vllm.vendor_name != "nvidia":
        raise ValueError("pinned native source extraction is NVIDIA-only")
    op = baseline.native()
    if op is None:
        raise RuntimeError(baseline.NATIVE_UNAVAILABLE_REASON)
    _independent_native(op)
    return op, baseline.NATIVE_PROVENANCE


def _oracle(pool_ids, seq_lens, size):
    pools = pool_ids.to(device="cpu", dtype=torch.int64).tolist()
    lengths = seq_lens.to(device="cpu", dtype=torch.int64).tolist()
    rows, groups = pool_ids.shape
    result = []
    for row in range(rows):
        values = []
        for group in range(groups):
            pid = pools[row][group]
            values.extend(
                pid * size + offset if pid >= 0 else -1 for offset in range(size)
            )
        start = lengths[row] // size * size
        count = lengths[row] % size
        values.extend(start + j if j < count else -1 for j in range(size - 1))
        assert all(-1 <= value <= 2**31 - 1 for value in values)
        result.append(values)
    return torch.tensor(result, dtype=torch.int32).reshape(
        rows, groups * size + size - 1
    )


def _snapshot(args):
    pools, lengths, _ = args
    backing = pools._base if pools._base is not None else pools
    return [(value, value.cpu().clone()) for value in (backing, lengths)]


def _assert_unchanged(saved):
    for value, original in saved:
        assert torch.equal(value.cpu(), original), "input or row padding was modified"


def _restore(saved):
    for value, original in saved:
        value.copy_(original)
    torch_device_fn.synchronize()


def _assert_output(output, expected, args):
    assert output.shape == expected.shape
    assert output.dtype == torch.int32
    assert output.device == args[0].device
    assert output.is_contiguous()
    assert torch.equal(output.cpu(), expected), "output differs from CPU int64 oracle"
    for value in args[:2]:
        if value.numel():
            assert (
                output.untyped_storage().data_ptr()
                != value.untyped_storage().data_ptr()
            )


def _validate(op, binding, dtype, args, kwargs):
    for sentinel in (-1, -17):
        check_args, check_kwargs = (
            (args, kwargs)
            if sentinel == -1
            else baseline.build_inputs(
                {**binding, "sentinel": sentinel}, dtype, flaggems_vllm.device
            )
        )
        expected = _oracle(*check_args)
        saved = _snapshot(check_args)
        try:
            output = op(*check_args, **check_kwargs)
            second = op(*check_args, **check_kwargs)
            torch_device_fn.synchronize()
            _assert_output(output, expected, check_args)
            _assert_output(second, expected, check_args)
            assert (
                output.untyped_storage().data_ptr()
                != second.untyped_storage().data_ptr()
            )
            _assert_unchanged(saved)
        finally:
            _restore(saved)


class ExpandPoolsBenchmark(Benchmark):
    DEFAULT_DTYPES = baseline.DTYPES
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = baseline.grid()
    DEFAULT_SHAPE_DESC = "R/G/P, pool/seq dtype, row stride, tail distribution"

    def __init__(self, record_property):
        self.entry = os.environ.get("EXPAND_BENCH_ENTRY", "candidate")
        if self.entry not in {"candidate", "installed", "pinned", "before"}:
            raise ValueError(
                "EXPAND_BENCH_ENTRY must be candidate/installed/pinned/before"
            )
        self.record_property = record_property
        self.selected = None
        self.selected_reason = None
        self.provenance = None
        self.native_op = None
        self.native_reason = None
        self.native_provenance = None
        self.gems_only = (
            self.entry == "candidate"
            and benchmark_config.Config.user_desired_metrics == ["latency"]
        )
        self.external_path = os.environ.get("EXPAND_STANDARD_NATIVE_RESULTS")
        self.external_rows = {}
        self.external_reason = None
        try:
            if self.entry == "candidate":
                self.selected = flaggems_vllm.expand_pools_and_append_tail
                if self.gems_only:
                    self.native_reason = "native timing disabled by --metrics"
                elif self.external_path:
                    self._load_external()
                else:
                    try:
                        self.native_op, self.native_provenance = _resolve_native("auto")
                    except Exception as exc:
                        self.native_reason = repr(exc)
            elif self.entry == "before":
                self.selected, self.provenance = _load_before()
            else:
                self.selected, self.provenance = _resolve_native(self.entry)
            self.selected_description = (
                _describe(self.selected) if self.selected else None
            )
            if self.entry == "candidate":
                self.provenance = {
                    "kind": "candidate",
                    "callable": self.selected_description,
                }
        except Exception as exc:
            if self.entry == "candidate":
                raise
            self.selected_reason = repr(exc)
            self.selected_description = None
        super().__init__(
            op_name=baseline.OP_NAME,
            torch_op=_native_slot,
            dtypes=baseline.DTYPES,
            gems_op=self.selected or _unavailable_provider,
        )

    def _load_external(self):
        try:
            with Path(self.external_path).expanduser().open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (row["case"], row["dtype"])
                    if key in self.external_rows:
                        raise ValueError(
                            f"ambiguous duplicate independent native row: {key}"
                        )
                    self.external_rows[key] = row
        except Exception as exc:
            self.external_rows = {}
            self.external_reason = repr(exc)

    def set_shapes(self, shape_file_path=None):
        # Preserve the frozen matrix for both --level values and shape files.
        self.shapes = baseline.grid()

    def init_user_config(self):
        super().init_user_config()
        if benchmark_config.Config.mode.value != "kernel":
            raise ValueError("standard expansion benchmark requires --mode kernel")
        if set(self.to_bench_dtypes) != set(baseline.DTYPES):
            raise ValueError(
                "the frozen expansion matrix requires both int32 and int64"
            )
        if not self.gems_only and set(self.to_bench_metrics) != set(
            self.DEFAULT_METRICS
        ):
            raise ValueError(
                "use latency_base/latency/speedup, or --metrics latency for the candidate"
            )

    def run(self):
        with pytest.MonkeyPatch.context() as config:
            if self.gems_only:
                config.setattr(benchmark_config.Config, "base_data", None)
            super().run()

    def get_input_iter(self, dtype):
        misses = []
        for binding in self.shapes:
            self.binding = binding
            self.current_dtype = dtype
            self.row_native_ms = None
            self.row_native_reason = self.native_reason
            self.row_native_accuracy = "unavailable"
            self.row_native_provenance = self.native_provenance
            self.row_native_callable = None
            args, kwargs = baseline.build_inputs(binding, dtype, flaggems_vllm.device)
            yield (*args, kwargs)
            if self.row_native_reason:
                misses.append(
                    dict(
                        shape=self.record_shapes(*args, **kwargs),
                        reason=self.row_native_reason,
                    )
                )
        if misses:
            benchmark_config.update_result(
                self.op_name,
                dict(
                    dtype=str(dtype),
                    native_baseline_skip_reason="; ".join(
                        dict.fromkeys(row["reason"] for row in misses)
                    ),
                    native_baseline_misses=misses,
                ),
            )

    def record_shapes(self, *args, **kwargs):
        config = baseline.config(self.binding, self.current_dtype)
        config["inputs"]["pool_ids"]["stride"] = list(args[0].stride())
        config["inputs"]["pool_ids"]["dtype"] = str(args[0].dtype)
        config["inputs"]["seq_lens"]["stride"] = list(args[1].stride())
        config["inputs"]["seq_lens"]["dtype"] = str(args[1].dtype)
        config["inputs"]["pool_size"] = {"scalar": args[2]}
        self.case_config = config
        return dict(
            case=baseline.key_shape(self.binding),
            config=config,
            inputs=super().record_shapes(*args, **kwargs),
        )

    @staticmethod
    def _settings():
        cfg = benchmark_config.Config
        return {
            "protocol": _PROTOCOL,
            "mode": cfg.mode.value,
            "warmup_ms": cfg.warm_up,
            "rep_ms": cfg.repetition,
            "level": cfg.bench_level.value,
        }

    @staticmethod
    def _source_identity():
        return {
            "scope_id": os.environ.get("EXPAND_STANDARD_SCOPE", "").strip() or None,
            "benchmark_file_sha256": _sha256(Path(__file__).read_bytes()),
            "baseline_file_sha256": _sha256(Path(baseline.__file__).read_bytes()),
        }

    @staticmethod
    def _effective_timer():
        if flaggems_vllm.vendor_name == "ascend":
            return {
                "callable": "triton.backends.ascend.testing.do_bench_npu",
                "arguments": "backend defaults; base does not forward warmup/iter",
            }
        return {
            "callable": "triton.testing.do_bench",
            "warmup": benchmark_config.Config.warm_up,
            "rep": benchmark_config.Config.repetition,
            "return_mode": "median",
            "grad_to_none": None,
        }

    def _external_native(self):
        if self.external_reason:
            raise ValueError(self.external_reason)
        row = self.external_rows.get(
            (baseline.key_shape(self.binding), str(self.current_dtype))
        )
        if row is None:
            raise ValueError("independent native table has no matching case/dtype")
        for key, expected in self._settings().items():
            if row.get(key) != expected:
                raise ValueError(
                    f"independent native {key} does not match: {row.get(key)!r}"
                )
        identity = self._source_identity()
        if not identity["scope_id"]:
            raise ValueError(
                "independent native pairing requires nonempty EXPAND_STANDARD_SCOPE"
            )
        for key, expected in identity.items():
            if row.get(key) != expected:
                raise ValueError(f"independent native {key} does not match")
        if row.get("vendor") != flaggems_vllm.vendor_name:
            raise ValueError(
                "independent native vendor does not match the current backend"
            )
        if row.get("device") != str(flaggems_vllm.device):
            raise ValueError(
                "independent native device does not match the current device"
            )
        if row.get("kind") not in {"installed", "pinned"}:
            raise ValueError("independent reference must be installed/pinned native")
        if row.get("op") != baseline.OP_NAME or row.get("unit") != "ms":
            raise ValueError("independent native op/unit mismatch")
        if row.get("config") != self.case_config:
            raise ValueError("independent native config mismatch")
        self.row_native_provenance = {
            "kind": "independent_standard_results",
            "results_file": str(Path(self.external_path).expanduser().resolve()),
            "provider": row["kind"],
            "provenance": row["provenance"],
            "effective_timer": row.get("effective_timer"),
            "vendor": row.get("vendor"),
            "device": row.get("device"),
            "scope_id": row.get("scope_id"),
            "benchmark_file_sha256": row.get("benchmark_file_sha256"),
            "baseline_file_sha256": row.get("baseline_file_sha256"),
        }
        self.row_native_callable = row.get("selected_callable")
        if row.get("accuracy") != "pass" or row.get("status") != "measured":
            raise ValueError(
                "independent native unavailable: "
                f"{row.get('unavailable_reason')!r}; "
                f"status={row.get('status')!r}, accuracy={row.get('accuracy')!r}"
            )
        if not row.get("selected_callable") or not row.get("provenance"):
            raise ValueError("independent native lacks callable/provenance")
        timer = row.get("effective_timer") or {}
        if timer.get("callable") not in {
            "triton.testing.do_bench",
            "triton.backends.ascend.testing.do_bench_npu",
        }:
            raise ValueError(
                "independent native does not identify a standard kernel timer"
            )
        latency = row.get("latency_ms")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError("independent native latency must be finite and positive")
        return float(latency)

    def _checked_latency(self, op, args, kwargs):
        saved = _snapshot(args)
        self.last_accuracy = "not_run"
        self.last_phase = "validation"
        try:
            _validate(op, self.binding, self.current_dtype, args, kwargs)
            self.last_accuracy = "pass"
            self.last_phase = "timing"
            # Validation, copies, and recovery are outside the standard timer.
            latency = float(super().get_latency(op, *args, **kwargs))
            torch_device_fn.synchronize()
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(
                    "standard timer returned nonpositive/nonfinite latency"
                )
            self.last_phase = "input_immutability"
            _assert_unchanged(saved)
            return latency
        except Exception:
            if self.last_phase in {"validation", "input_immutability"}:
                self.last_accuracy = "failed"
            raise
        finally:
            _restore(saved)

    def _emit(self, latency, accuracy, reason=None):
        row = {
            **self._settings(),
            **self._source_identity(),
            "op": baseline.OP_NAME,
            "case": baseline.key_shape(self.binding),
            "dtype": str(self.current_dtype),
            "kind": self.entry,
            "source": self.entry,
            "selected_callable": self.selected_description,
            "provenance": self.provenance,
            "device": str(flaggems_vllm.device),
            "vendor": flaggems_vllm.vendor_name,
            "config": self.case_config,
            "effective_timer": self._effective_timer(),
            "unit": "ms",
            "timing_scope": "complete allocating public callable via Benchmark.get_latency",
            "status": "measured" if latency is not None else "N/A",
            "accuracy": accuracy,
            "validation": _ACCURACY,
            "latency_ms": latency,
            "unavailable_reason": reason,
            "native_ms": self.row_native_ms,
            "native_status": "measured" if self.row_native_ms is not None else "N/A",
            "native_accuracy": self.row_native_accuracy,
            "native_unavailable_reason": self.row_native_reason,
            "native_provenance": self.row_native_provenance,
            "native_callable": self.row_native_callable,
            "native_speedup": (
                self.row_native_ms / latency
                if self.row_native_ms is not None and latency
                else None
            ),
        }
        encoded = json.dumps(row, sort_keys=True)
        _LOG.info("%s", encoded)
        self.record_property("expand_performance", encoded)
        destination = os.environ.get("EXPAND_PERFORMANCE_RESULTS")
        if destination:
            path = Path(destination).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")

    def get_latency(self, op, *args, **kwargs):
        if op is self.torch_op:
            if self.entry != "candidate":
                return None
            try:
                if self.external_path:
                    latency = self._external_native()
                elif self.native_op is not None:
                    self.row_native_callable = _independent_native(self.native_op)
                    latency = self._checked_latency(self.native_op, args, kwargs)
                else:
                    return None
                self.row_native_ms = latency
                self.row_native_accuracy = "pass"
                self.row_native_reason = None
                return latency
            except Exception as exc:
                self.row_native_reason = (
                    f"{getattr(self, 'last_phase', 'native_resolution')}: {exc!r}"
                )
                self.row_native_accuracy = (
                    "unavailable"
                    if self.external_path
                    else getattr(self, "last_accuracy", "not_run")
                )
                return None
        assert op is self.gems_op, "unexpected callable in standard benchmark slot"
        if self.selected is None:
            self._emit(None, "unavailable", self.selected_reason)
            return None
        try:
            latency = self._checked_latency(op, args, kwargs)
        except Exception as exc:
            self._emit(
                None,
                getattr(self, "last_accuracy", "not_run"),
                f"{getattr(self, 'last_phase', 'unknown')}: {exc!r}",
            )
            if self.entry == "candidate":
                raise
            return None
        self._emit(latency, "pass")
        return latency


def test_expand_pools_and_append_tail(record_property):
    bench = ExpandPoolsBenchmark(record_property)
    bench.run()
