# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Latency benchmark for the MiniMax M3 sparse attention shim.

The baseline is vLLM's own MiniMax M3 operator, per this repository's
convention (``SpeedUp = latency_vllm_baseline / latency_flaggems``). It is
imported lazily because it only exists in vLLM builds that ship the packed
4D KV layout ``[num_blocks, num_kv_heads, 128, 2 * head_dim]``; older builds
use a 5D layout and are skipped rather than silently compared.

Each record carries the selected-index span statistics. Sparse-attention
latency depends heavily on how clustered the top-k ids are, so a speedup is
only interpretable next to the distribution that produced it.
"""

import json
import statistics

import pytest
import torch

import flaggems_vllm
from tests.test_minimax_m3_sparse_attn import (
    _native_takes_packed_pages,
    make_decode_inputs,
    make_inputs,
    topk_span_stats,
)

from . import conftest as bench_cfg

native = pytest.importorskip(
    "vllm.models.minimax_m3.common.ops.sparse_attn",
    reason="requires the vLLM MiniMax M3 packed-KV operator",
)

pytestmark = [
    pytest.mark.minimax_m3_sparse_attn,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available(),
        reason="requires an NVIDIA GPU",
    ),
    # The module imports on older vLLM too, so importorskip above is not
    # enough: those builds only fail once called, on kv_cache.stride(4).
    pytest.mark.skipif(
        not _native_takes_packed_pages(),
        reason="installed vLLM operator predates the packed 4D KV layout",
    ),
]

WARMUP = 20
SAMPLES = 50


def _median_us(fn):
    """Interleaved CUDA-event timing; returns the median microseconds."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(SAMPLES)
    ]
    for start, end in pairs:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(end) * 1000 for start, end in pairs)
    return statistics.median(values), values[SAMPLES // 10], values[-SAMPLES // 10]


def _emit(op_name, record, native_us, gems_us):
    """SpeedUp = latency_vllm_baseline / latency_flaggems (repo convention)."""
    record["flaggems_us"] = round(gems_us, 2)
    record["vllm_us"] = round(native_us, 2)
    record["speedup"] = round(native_us / gems_us, 3)
    print(json.dumps(record), flush=True)
    if bench_cfg.Config.record_json:
        bench_cfg.update_result(op_name, record)


def _kv_dtype():
    return (
        torch.float8_e4m3fn
        if torch.cuda.get_device_capability() >= (8, 9)
        else torch.bfloat16
    )


@pytest.mark.parametrize("distribution", ["scattered", "compact"])
@pytest.mark.parametrize("qlen", [512, 2048, 4096, 8192])
def test_perf_minimax_m3_sparse_attn(qlen, distribution):
    """Prefill latency across query lengths and index distributions."""
    if bench_cfg.Config.bench_level.value == "core" and (
        qlen not in (512, 2048) or distribution != "scattered"
    ):
        pytest.skip("non-core shape")
    dtype = _kv_dtype()
    args = make_inputs(
        qlens=(qlen,), d=128, kv_heads=1, dtype=dtype, mode=distribution, context=32768
    )
    out = torch.empty_like(args[0])
    out_native = torch.empty_like(args[0])

    def candidate():
        flaggems_vllm.minimax_m3_sparse_attn(*args, output=out)

    def baseline():
        native.minimax_m3_sparse_attn(*args, output=out_native)

    # Timing a wrong kernel is meaningless; agree with the baseline first.
    baseline()
    candidate()
    torch.cuda.synchronize()
    finite = torch.isfinite(out) & torch.isfinite(out_native)
    if bool(finite.any()):
        torch.testing.assert_close(
            out.float()[finite], out_native.float()[finite], atol=0.02, rtol=0
        )

    native_us = _median_us(baseline)[0]
    gems_us, p10, p90 = _median_us(candidate)
    record = {
        "stage": "prefill",
        "q": qlen,
        "heads": 8,
        "D": 128,
        "kv_heads": 1,
        "context": 32768,
        "TOPK": 16,
        "page": 128,
        "distribution": distribution,
        "kv_dtype": str(dtype),
        "warmup": WARMUP,
        "samples": SAMPLES,
        "p10_us": round(p10, 2),
        "p90_us": round(p90, 2),
        "index_span": topk_span_stats(args[2]),
        "device": torch.cuda.get_device_name(),
    }
    _emit("minimax_m3_sparse_attn", record, native_us, gems_us)


@pytest.mark.parametrize("num_reqs", [1, 16, 64, 128])
def test_perf_minimax_m3_sparse_attn_decode(num_reqs):
    """Decode latency across the batch sizes that move the split-K count."""
    if bench_cfg.Config.bench_level.value == "core" and num_reqs not in (16, 64):
        pytest.skip("non-core shape")
    dtype = _kv_dtype()
    args = make_decode_inputs(
        num_reqs=num_reqs,
        d=128,
        kv_heads=1,
        dtype=dtype,
        mode="scattered",
        context=32768,
    )
    out = torch.empty_like(args[0])

    out_native = torch.empty_like(args[0])

    def candidate():
        flaggems_vllm.minimax_m3_sparse_attn_decode(
            *args[:7], output=out, decode_query_len=args[7]
        )

    def baseline():
        native.minimax_m3_sparse_attn_decode(
            *args[:7], output=out_native, decode_query_len=args[7]
        )

    baseline()
    candidate()
    torch.cuda.synchronize()
    finite = torch.isfinite(out) & torch.isfinite(out_native)
    if bool(finite.any()):
        torch.testing.assert_close(
            out.float()[finite], out_native.float()[finite], atol=0.02, rtol=0
        )

    native_us = _median_us(baseline)[0]
    gems_us, p10, p90 = _median_us(candidate)
    record = {
        "stage": "decode",
        "num_reqs": num_reqs,
        "decode_query_len": 1,
        "heads": 8,
        "D": 128,
        "kv_heads": 1,
        "context": 32768,
        "TOPK": 16,
        "page": 128,
        "distribution": "scattered",
        "kv_dtype": str(dtype),
        "warmup": WARMUP,
        "samples": SAMPLES,
        "p10_us": round(p10, 2),
        "p90_us": round(p90, 2),
        "index_span": topk_span_stats(args[2]),
        "device": torch.cuda.get_device_name(),
    }
    _emit("minimax_m3_sparse_attn_decode", record, native_us, gems_us)
