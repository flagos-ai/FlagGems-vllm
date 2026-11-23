# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Standard JSON and private JSONL timing for the immutable Gumbel workload.

CUDA-compatible backends use the collector's no-Graph do_bench protocol.
All backends use runtime device events through their Triton implementation.
Native device failures terminate the process even when the runner omits -x.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import triton

import flaggems_vllm
from baseline import gumbel_sample as workload
from flaggems_vllm.runtime import torch_device_fn

from . import base, conftest
from .gumbel_baseline import normalize_case

pytestmark = pytest.mark.gumbel_sample
_DETAIL = None
_NATIVE_SKIP_REASONS = []


def _chip_name():
    try:
        return torch_device_fn.get_device_name()
    except Exception:
        return None


def _record(row, metric, native_reason):
    """One dtype group and unique case keys preserve all 16 measurements."""
    global _DETAIL
    if _DETAIL is None:
        _DETAIL = {
            "op_name": "gumbel_sample",
            "dtype": "torch.float32",
            "mode": "kernel",
            "level": conftest.Config.bench_level.value,
            "result": [],
        }
        conftest.update_result("gumbel_sample", _DETAIL)
    if native_reason and native_reason not in _NATIVE_SKIP_REASONS:
        _NATIVE_SKIP_REASONS.append(native_reason)
    if _NATIVE_SKIP_REASONS:
        _DETAIL["native_baseline_skip_reason"] = "; ".join(_NATIVE_SKIP_REASONS)
    _DETAIL["result"].append(metric)
    print("GUMBEL_BENCH " + json.dumps(row, sort_keys=True))
    output = os.environ.get("GUMBEL_BENCH_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


class GumbelSampleBenchmark(base.Benchmark):
    """Per-case adapter for the frozen sampler measurement protocol.

    Keep the dedicated run loop: the generic loop uses a different Ascend timer,
    normalization fallback and native-failure policy. Pytest owns the 16-case grid.
    """

    def __init__(self, binding, request):
        super().__init__(
            op_name=workload.OP_NAME,
            torch_op=None,
            gems_op=flaggems_vllm.gumbel_sample,
            dtypes=workload.DTYPES,
            is_inplace=workload.IS_INPLACE,
        )
        self.binding = binding
        self.request = request

    def get_latency(self, op, *args, **kwargs):
        return triton.testing.do_bench(
            lambda: op(*args, **kwargs),
            warmup=25,
            rep=100,
            return_mode="median",
        )

    def _measure(self, op, args, kwargs):
        latency_ms = self.get_latency(op, *args, **kwargs)
        return {
            "median_us": latency_ms * 1000,
            "latency_ms": latency_ms,
            "timing_protocol": workload.TIMING_PROTOCOL,
        }

    def run(self):
        binding, request = self.binding, self.request
        if conftest.Config.mode.value != "kernel":
            raise ValueError("gumbel supports only the frozen no-Graph kernel mode")
        requested = conftest.Config.user_desired_dtypes
        if requested and requested != [torch.float32]:
            raise ValueError("gumbel performance coverage is FP32 only")
        self.set_dtypes(requested)
        args, kwargs = workload.build_inputs(
            binding, torch.float32, flaggems_vllm.device, qualify=False
        )
        workload.assert_output(binding, self.gems_op(*args, **kwargs))
        workload.assert_output(binding, self.gems_op(*args, **kwargs))
        candidate = self._measure(self.gems_op, args, kwargs)
        workload.assert_output(binding, self.gems_op(*args, **kwargs))
        cfg = workload.config(binding, torch.float32)
        base_path = request.config.getoption("--base-data")
        source = None
        if base_path:
            path = Path(base_path)
            source = {"path": str(path.resolve())}
            if path.is_file():
                source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        compared = normalize_case(
            conftest.Config.base_data,
            workload.key_shape(binding),
            "torch.float32",
            cfg,
            candidate["latency_ms"],
            flaggems_vllm.vendor_name,
            _chip_name(),
            base_data_source=source,
            timing_protocol=candidate["timing_protocol"],
        )
        native_reason = None
        native = None
        identity = {"status": "unavailable"}
        if flaggems_vllm.vendor_name == "nvidia":
            native = self.torch_op = workload.native()
            identity = workload.native_identity()
            native_reason = workload.NATIVE_UNAVAILABLE_REASON
        else:
            native_reason = (
                "pinned provider device qualification covers NVIDIA only; "
                "no qualified on-device native for this backend"
            )
            identity["reason"] = native_reason
        metric = {
            "shape_detail": workload.key_shape(binding),
            "latency": candidate["latency_ms"],
            "latency_base": None,
            "speedup": None,
            "timing_protocol": candidate["timing_protocol"],
            "workload_config": cfg,
            "native_identity": identity,
            **compared,
        }
        row = {
            "vendor": flaggems_vllm.vendor_name,
            "device": str(flaggems_vllm.device),
            "chip": _chip_name(),
            "shape": binding["shape"],
            "case_key": workload.key_shape(binding),
            "dtype": "float32",
            "stride": args[0].stride(),
            "temperature": binding["temperature"],
            "graph": False,
            "scope": cfg["timing"]["scope"],
            "rng": cfg["rng"],
            "timing_protocol": candidate["timing_protocol"],
            "candidate": candidate,
            "native_identity": identity,
            "baseline_kind": "unavailable",
            "native_ratio": None,
            "normalized_ratio": compared["compared_speedup"],
            "normalization": compared,
        }
        if native is not None:
            try:
                qualification = workload.qualify_native(binding, args, kwargs, native)
                native_time = self._measure(native, args, kwargs)
                workload.qualify_native(binding, args, kwargs, native)
                metric["native_qualification"] = qualification
                metric["latency_base"] = native_time["latency_ms"]
                metric["speedup"] = native_time["latency_ms"] / candidate["latency_ms"]
                row.update(
                    native=native_time,
                    native_ratio=metric["speedup"],
                    baseline_kind="native_vllm",
                    native_qualification=qualification,
                )
            except Exception as exc:
                native_reason = f"native qualification/execution failed: {type(exc).__name__}: {exc}"
                metric["error_msg"] = native_reason
                row["native_error"] = native_reason
                _record(row, metric, native_reason)
                # Central runners do not all pass -x. A failed native call may
                # poison the device context, so stop before another case can run.
                # pytest.exit need not emit a failed call report: persist it first.
                conftest.TEST_RESULTS.setdefault("gumbel_sample", {}).update(
                    result="failed", reason=native_reason, test_case=request.node.nodeid
                )
                pytest.exit(native_reason, returncode=1)
        _record(row, metric, native_reason)
        if metric["speedup"] is not None:
            assert metric["speedup"] >= 0.9, row


@pytest.mark.parametrize("binding", workload.grid(), ids=workload.key_shape)
def test_gumbel_sample_benchmark(binding, request):
    GumbelSampleBenchmark(binding, request).run()
