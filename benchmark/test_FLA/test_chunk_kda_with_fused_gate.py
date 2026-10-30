# SPDX-License-Identifier: Apache-2.0
"""No-Graph public-host latency; reset the aliased value outside every event."""

import json
import os
import statistics

import pytest
import torch
import triton.testing

import flaggems_vllm
from benchmark.conftest import emit_record_logger, update_result
from flaggems_vllm.runtime import torch_device_fn
from tests.test_FLA.test_chunk_kda_with_fused_gate import build, check, reference

SHAPES = [
    (1, 64, 8, 64, 64, None),
    (1, 512, 8, 128, 128, None),
    (1, 2048, 16, 128, 128, None),
    (1, 8192, 96, 128, 128, None),
    (1, 8192, 96, 128, 128, (1300, 547, 2048, 963, 271, 3063)),
    (1, 8192, 96, 128, 128, (1024,) * 8),
]


@pytest.fixture
def no_graph(monkeypatch):
    monkeypatch.setenv("FLA_USE_CUDA_GRAPH", "0")
    monkeypatch.setenv("FLA_USE_FAST_OPS", "0")

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


@pytest.mark.chunk_kda_with_fused_gate
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_chunk_kda_with_fused_gate_perf(shape, dtype, no_graph):
    os.environ["FG_KDA_DEVICE"] = flaggems_vllm.device
    args, kwargs = build(shape, dtype)
    saved = args[2].clone()
    gems = flaggems_vllm.chunk_kda_with_fused_gate
    expected = reference(args, kwargs)
    result = gems(*args, **kwargs)
    check(result[0], expected[0], dtype)
    check(result[1], expected[1], dtype)
    samples = _latency(gems, args, kwargs, saved)
    row = dict(
        shape=shape,
        dtype=str(dtype),
        graph=False,
        gems_samples_ms=samples,
        gems_ms=statistics.median(samples),
        native_ms=None,
        selected_source=None,
    )
    try:
        from vllm.model_executor.layers.fla.ops.kda import (
            chunk_kda_with_fused_gate as native,
        )
    except (ImportError, AttributeError) as error:
        row["native_unavailable"] = repr(error)
    else:
        args[2].copy_(saved)
        native_result = native(*args, **kwargs)
        check(native_result[0], expected[0], dtype)
        check(native_result[1], expected[1], dtype)
        # The native call also aliases v; materialize before the next call.
        native_expected = native_result[0].clone(), native_result[1].clone()
        args[2].copy_(saved)
        result = gems(*args, **kwargs)
        check(result[0], native_expected[0].cpu(), dtype)
        check(result[1], native_expected[1].cpu(), dtype)
        native_samples = _latency(native, args, kwargs, saved)
        row.update(
            native_samples_ms=native_samples,
            native_ms=statistics.median(native_samples),
            selected_source="native",
        )
        row["native_speedup"] = row["native_ms"] / row["gems_ms"]
    path = os.environ.get("FG_KDA_PERF_RAW")
    if path:
        with open(path, "a") as out:
            out.write(json.dumps(row) + "\n")

    update_result("chunk_kda_with_fused_gate", row)
    emit_record_logger(json.dumps(row))
