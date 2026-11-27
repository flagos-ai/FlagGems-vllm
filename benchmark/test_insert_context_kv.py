# SPDX-License-Identifier: Apache-2.0
"""Measure KV-only insertion with a private, restored cache for each replay."""

import hashlib
import math
import statistics
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from tests.test_insert_context_kv import cpu_insert_context_kv, make_case

from . import base, consts
from .conftest import Config, emit_record_logger, update_result

OP_NAME = "insert_context_kv"
FORMATS = ("bf16", "fp8", "v4", "v41")
TOKENS = (1, 17, 255, 1024)
BLOCK_SIZES = (64, 256)
WARMUP = 10
SAMPLES = 25
torch_device_fn = flaggems_vllm.runtime.torch_device_fn


_NATIVE_SYMBOL = "_C::fused_deepseek_v4_kv_rope_insert"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_registration(dispatch):
    key = flaggems_vllm.runtime.device.dispatch_key
    return next(
        (line for line in dispatch.splitlines() if line.startswith(key + ":")), ""
    )


def _native():
    identity = {"kind": "installed vllm"}
    try:
        identity["backend"] = flaggems_vllm.runtime.device.vendor_name
        import vllm._C as extension
        import vllm._custom_ops as custom_ops

        binary = Path(extension.__file__).resolve()
        identity.update(
            version=version("vllm"),
            module=custom_ops.__file__,
            binary=str(binary),
            binary_sha256=_sha256(binary),
        )
        op = torch.ops._C.fused_deepseek_v4_kv_rope_insert.default
        dispatch = torch._C._dispatch_dump_table(_NATIVE_SYMBOL)
        identity.update(schema=str(op._schema), dispatch=dispatch)
        registration = _device_registration(dispatch)
        if "[kernel]" not in registration or "fallback" in registration.lower():
            raise ValueError(
                "target operator has no native kernel for the active device"
            )
        if any(
            marker in registration.lower()
            for marker in ("flaggems", "flag_gems", "vllm-plugin-fl")
        ):
            raise ValueError(
                "native registration points to a candidate/plugin implementation"
            )
        if (
            "registered at /dev/null" in registration
            or "registered at [python]" in registration.lower()
        ):
            raise ValueError(
                "Python operator registration is not an audited binary reference"
            )
    except Exception as exc:
        # Missing native support does not prevent candidate measurements.
        return None, identity, f"{type(exc).__name__}: {exc}"
    return op, identity, None


class InsertContextKVBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shape_desc = "format,tokens,block_size"
        self.shapes = [
            (fmt, tokens, block)
            for fmt in FORMATS
            for tokens in TOKENS
            for block in BLOCK_SIZES
        ]

    def get_input_iter(self, dtype):
        for fmt, tokens, block in self.shapes:
            # The native implementation uses vector stores. The 16-byte pad
            # preserves alignment while making unintended writes observable.
            yield make_case(fmt, tokens=tokens, block_size=block)

    def get_latency(self, op, case, expected):
        actual = case.to(self.device)
        args = actual.arguments()
        initial = actual.backing.clone()

        def reset():
            actual.backing.copy_(initial)
            torch_device_fn.synchronize()

        def check(result):
            torch_device_fn.synchronize()
            assert result is None
            torch.testing.assert_close(actual.backing.cpu(), expected, atol=0, rtol=0)
            for name in (
                "kv",
                "slot_mapping",
                "position_ids",
                "cos_sin_cache",
                "fp8_scale",
            ):
                observed = getattr(actual, name)
                original = getattr(case, name)
                if observed is not None:
                    torch.testing.assert_close(
                        observed.cpu().reshape(-1).view(torch.uint8),
                        original.reshape(-1).view(torch.uint8),
                        atol=0,
                        rtol=0,
                        msg=f"read-only input changed: {name}",
                    )

        reset()
        check(op(*args))
        for _ in range(WARMUP):
            reset()
            op(*args)
        begin = torch_device_fn.Event(enable_timing=True)
        end = torch_device_fn.Event(enable_timing=True)
        samples = []
        for _ in range(SAMPLES):
            reset()
            begin.record()
            result = op(*args)
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end))
        # Check the final timed invocation itself, without another write that
        # could hide a bad final cache state.
        check(result)
        latency = statistics.median(samples)
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"invalid device-event latency {latency}")
        self._last_timing = {
            "samples_ms": samples,
            "median_ms": latency,
            "min_ms": min(samples),
            "max_ms": max(samples),
        }
        return latency

    def run(self):
        if Config.query:
            return super().run()
        self.init_user_config()
        if self.mode != consts.BenchMode.KERNEL:
            raise ValueError("insert_context_kv requires --mode kernel")

        missing_reason = None
        dtype = torch.bfloat16
        if dtype not in self.to_bench_dtypes:
            raise ValueError("insert_context_kv requires bfloat16 input")
        metrics = []
        pending_native = []
        candidate_errors = []
        candidate_runtime_failure = None
        for case in self.get_input_iter(dtype):
            metric = consts.BenchmarkMetrics()
            metric.shape_detail = {
                "format": case.cache_format,
                "tokens": case.kv.shape[0],
                "block_size": case.cache_block_size,
                "cache_stride": list(case.k_cache.stride()),
                "candidate_status": "NOT_RUN",
                "native_status": "NOT_RUN",
                "timing": {
                    "timer": "device_event",
                    "graph": False,
                    "warmup_calls": WARMUP,
                    "samples": SAMPLES,
                    "aggregate": "median",
                    "unit": "ms",
                    "reset": "restore full backing outside event interval",
                },
            }
            metric.base_miss_reason = missing_reason
            metrics.append(metric)
            if candidate_runtime_failure:
                metric.error_msg = candidate_runtime_failure
                metric.base_miss_reason = missing_reason or "candidate not executed"
                continue
            expected = cpu_insert_context_kv(case)
            try:
                metric.latency = self.get_latency(self.gems_op, case, expected)
            except (AssertionError, ValueError, RuntimeError) as exc:
                key = (case.cache_format, case.kv.shape[0], case.cache_block_size)
                metric.error_msg = str(exc).strip() or type(exc).__name__
                metric.base_miss_reason = missing_reason or "candidate did not qualify"
                candidate_errors.append(f"{key}: {metric.error_msg}")
                if isinstance(exc, RuntimeError):
                    candidate_runtime_failure = (
                        f"candidate runtime failed on {key}: {metric.error_msg}"
                    )
                    metric.shape_detail["candidate_status"] = "FAILED_EXECUTION"
                else:
                    metric.shape_detail["candidate_status"] = (
                        "FAILED_QUALIFICATION"
                        if isinstance(exc, AssertionError)
                        else "FAILED_TIMING"
                    )
            else:
                metric.shape_detail["candidate_status"] = "PASS"
                metric.shape_detail["candidate_timing"] = self._last_timing
                pending_native.append((case, expected, metric))

        result = consts.BenchmarkResult(
            level=Config.bench_level.value,
            op_name=self.op_name,
            dtype=str(dtype),
            mode=Config.mode.value,
            result=metrics,
        )
        # Resolve/import references only after every candidate row: a vendor
        # extension must not prevent independent candidate measurements.
        if candidate_runtime_failure:
            native, native_identity, missing_reason = (
                None,
                None,
                candidate_runtime_failure,
            )
        else:
            native, native_identity, missing_reason = _native()
        for metric in metrics:
            metric.shape_detail["native_identity"] = native_identity
            if not native:
                metric.shape_detail["native_status"] = "UNAVAILABLE"
                metric.base_miss_reason = missing_reason
            elif metric.shape_detail["candidate_status"] != "PASS":
                metric.shape_detail["native_status"] = "NOT_RUN_CANDIDATE_FAILED"
                metric.base_miss_reason = "candidate did not qualify"
        native_launch_failure = None
        for case, expected, metric in pending_native if native else ():
            if native_launch_failure:
                metric.base_miss_reason = native_launch_failure
                metric.shape_detail["native_status"] = "NOT_RUN_AFTER_NATIVE_FAILURE"
                continue
            try:
                metric.latency_base = self.get_latency(native, case, expected)
            except (AssertionError, ValueError) as exc:
                metric.base_miss_reason = str(exc).strip() or type(exc).__name__
                metric.shape_detail["native_status"] = (
                    "FAILED_QUALIFICATION"
                    if isinstance(exc, AssertionError)
                    else "FAILED_TIMING"
                )
            except Exception as exc:
                # Unexpected compiler/runtime failures may invalidate context;
                # stop native calls while retaining all candidate results.
                native_launch_failure = (
                    f"native launch/sync failed: {type(exc).__name__}: {exc}"
                )
                metric.base_miss_reason = native_launch_failure
                metric.shape_detail["native_status"] = "FAILED_EXECUTION"
            else:
                metric.speedup = metric.latency_base / metric.latency
                metric.shape_detail["native_status"] = "PASS"
                metric.shape_detail["native_timing"] = self._last_timing

        print(result)
        update_result(self.op_name, asdict(result))
        emit_record_logger(result.to_json())
        if candidate_errors:
            pytest.fail("\n".join(candidate_errors))


@pytest.mark.insert_context_kv
def test_insert_context_kv():
    InsertContextKVBenchmark(
        op_name=OP_NAME,
        torch_op=None,
        gems_op=flaggems_vllm.insert_context_kv,
        dtypes=[torch.bfloat16],
        is_inplace=True,
    ).run()
