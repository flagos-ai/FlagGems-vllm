# SPDX-License-Identifier: Apache-2.0
"""No-Graph public-host latency; reset the aliased value outside every event."""

import json
import math
import os
import statistics
from pathlib import Path

import pytest
import torch
import triton.testing

import flaggems_vllm
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from benchmark.consts import BenchMode
from flaggems_vllm.runtime import torch_device_fn
from tests.test_FLA.test_chunk_kda_with_fused_gate import build, check, reference


@pytest.fixture
def no_graph(monkeypatch):
    monkeypatch.setenv("FLA_USE_CUDA_GRAPH", "0")
    monkeypatch.setenv("FLA_USE_FAST_OPS", "0")
    monkeypatch.setenv("FG_KDA_DEVICE", flaggems_vllm.device)

    def forbidden(*args, **kwargs):
        raise RuntimeError("Graph is forbidden for this benchmark")

    monkeypatch.setattr(triton.testing, "do_bench_cudagraph", forbidden)
    for api in {torch.cuda, torch_device_fn}:
        if hasattr(api, "graph"):
            monkeypatch.setattr(api, "graph", forbidden)
        for name in ("CUDAGraph", "MUSAGraph", "NPUGraph", "PPUGraph"):
            cls = getattr(api, name, None)
            if cls is not None and hasattr(cls, "capture_begin"):
                monkeypatch.setattr(cls, "capture_begin", forbidden)

    def stage_bench(fn, quantiles=None, **kwargs):
        fn()
        torch_device_fn.synchronize()
        values = []
        for _ in range(5):
            begin = torch_device_fn.Event(enable_timing=True)
            end = torch_device_fn.Event(enable_timing=True)
            begin.record()
            fn()
            end.record()
            end.synchronize()
            values.append(begin.elapsed_time(end))
        values.sort()
        if quantiles:
            return [values[round(q * (len(values) - 1))] for q in quantiles]
        return statistics.median(values)

    monkeypatch.setattr(triton.testing, "do_bench", stage_bench)


def _latency(fn, args, kwargs, original):
    for _ in range(5):
        args[2].copy_(original)
        fn(*args, **kwargs)
    torch_device_fn.synchronize()
    samples = []
    for _ in range(25):
        args[2].copy_(original)
        torch_device_fn.synchronize()
        start = torch_device_fn.Event(enable_timing=True)
        end = torch_device_fn.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _resolve_native():
    from vllm.model_executor.layers.fla.ops.kda import (
        chunk_kda_with_fused_gate as native,
    )

    if not callable(native):
        raise AttributeError("installed vLLM export is not callable")
    return native


def _native(*args, **kwargs):
    return _resolve_native()(*args, **kwargs)


def _measure(shape, dtype, gems, args, kwargs, measure_native):
    saved = args[2].clone()
    expected = reference(args, kwargs)
    result = gems(*args, **kwargs)
    check(result[0], expected[0], dtype)
    check(result[1], expected[1], dtype)
    gems_expected = (
        tuple(value.detach().cpu().clone() for value in result)
        if measure_native
        else None
    )
    samples = _latency(gems, args, kwargs, saved)
    gems_ms = statistics.median(samples)
    if not math.isfinite(gems_ms) or gems_ms <= 0:
        raise ValueError(f"invalid Gems latency: {gems_ms}")
    row = dict(
        shape=shape,
        dtype=str(dtype),
        graph=False,
        gems_samples_ms=samples,
        gems_ms=gems_ms,
        native_ms=None,
        selected_source=None,
    )
    if measure_native:
        try:
            native = _resolve_native()
            if native is gems:
                raise ValueError("native resolves to the Gems candidate")
            args[2].copy_(saved)
            native_result = native(*args, **kwargs)
            check(native_result[0], expected[0], dtype)
            check(native_result[1], expected[1], dtype)
            for actual, candidate in zip(native_result, gems_expected, strict=True):
                check(actual, candidate, dtype)
            native_samples = _latency(native, args, kwargs, saved)
            native_ms = statistics.median(native_samples)
            if not math.isfinite(native_ms) or native_ms <= 0:
                raise ValueError(f"invalid native latency: {native_ms}")
        except Exception as error:
            row["native_unavailable"] = (
                f"native resolution, validation or timing: {error!r}"
            )
        else:
            row.update(
                native_samples_ms=native_samples,
                native_ms=native_ms,
                selected_source="native",
                native_speedup=native_ms / row["gems_ms"],
            )
        finally:
            args[2].copy_(saved)
    else:
        row["native_unavailable"] = "native timing disabled by --metrics"
    path = os.environ.get("FG_KDA_PERF_RAW")
    if path:
        with open(path, "a") as out:
            out.write(json.dumps(row) + "\n")
    return row


class ChunkKDAFusedGateBenchmark(Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).resolve().parents[1] / "core_shapes.yaml")
    DEFAULT_DTYPES = [torch.float16, torch.bfloat16]

    def set_shapes(self, shape_file_path=None):
        path = Path(shape_file_path or self.DEFAULT_SHAPE_FILES)
        if path.parent.name in {"_kunlunxin", "_enflame"}:
            path = Path(self.DEFAULT_SHAPE_FILES)
        super().set_shapes(path)

    def set_more_shapes(self):
        return [
            (1, 512, 8, 128, 128),
            (1, 2048, 16, 128, 128),
            (1, 8192, 96, 128, 128),
            (1, 8192, 96, 128, 128, 1300, 547, 2048, 963, 271, 3063),
            (1, 8192, 96, 128, 128) + (1024,) * 8,
        ]

    def init_user_config(self):
        if Config.mode != BenchMode.KERNEL:
            raise ValueError("Fused-gate KDA uses --mode kernel with no-Graph events")
        super().init_user_config()

    def run(self):
        with pytest.MonkeyPatch.context() as config:
            if Config.user_desired_metrics == ["latency"]:
                config.setattr(Config, "base_data", None)
            super().run()

    def get_input_iter(self, dtype):
        misses = []
        self.dtype = dtype
        for b, t, h, dk, dv, *lengths in self.shapes:
            self.case = (b, t, h, dk, dv, tuple(lengths) or None)
            self.row = None
            args, kwargs = build(self.case, dtype)
            yield (*args, kwargs)
            if self.row and self.row.get("native_unavailable"):
                misses.append(
                    dict(shape=self.case, reason=self.row["native_unavailable"])
                )
        if misses:
            # The weekly parser uses this metadata to accept missing speedups.
            update_result(
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
        return dict(case=self.case, inputs=super().record_shapes(*args, **kwargs))

    def get_latency(self, op, *args, **kwargs):
        # Use one oracle and snapshot for both providers and keep the aliased
        # v reset outside each event. Benchmark.run still owns the metrics.
        if self.row is None:
            self.row = _measure(
                self.case,
                self.dtype,
                self.gems_op,
                args,
                kwargs,
                measure_native="latency_base" in self.to_bench_metrics,
            )
        return self.row["native_ms" if op is self.torch_op else "gems_ms"]


@pytest.mark.chunk_kda_with_fused_gate
def test_chunk_kda_with_fused_gate_perf(no_graph):
    ChunkKDAFusedGateBenchmark(
        op_name="chunk_kda_with_fused_gate",
        # Resolve lazily in get_latency so Gems-only does not import vLLM.
        torch_op=_native,
        gems_op=flaggems_vllm.chunk_kda_with_fused_gate,
        is_inplace=True,
    ).run()
