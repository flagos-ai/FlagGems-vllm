# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import time
from statistics import median

import pytest
import torch

import flaggems_vllm
from benchmark.conftest import update_result
from tests.lightning_attention_utils import (
    MINIMAX_PREFIX_LENGTHS,
    make_initial,
    make_qkv,
    make_slope,
)

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend", reason="requires Ascend"
)


def run_pair(request, op, shape, providers, reset, calls, rounds):
    mode = request.config.getoption("--mode")
    if mode not in {"operator", "kernel"}:
        pytest.skip(
            "Lightning Attention measures synchronized operator or device-event latency"
        )
    with torch.inference_mode():
        results = {name: [] for name in providers}
        for fn in providers.values():
            reset()
            for _ in range(3):
                fn()
            torch.npu.synchronize()
        for i in range(rounds):
            order = providers.items() if i % 2 == 0 else reversed(providers.items())
            for name, fn in order:
                reset()
                torch.npu.synchronize()
                if mode == "operator":
                    start = time.perf_counter_ns()
                    for _ in range(calls):
                        fn()
                    torch.npu.synchronize()
                    latency = (time.perf_counter_ns() - start) / 1e6
                else:
                    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                        enable_timing=True
                    )
                    start.record()
                    for _ in range(calls):
                        fn()
                    end.record()
                    end.synchronize()
                    latency = start.elapsed_time(end)
                results[name].append(latency / calls)
    baseline, optimized = (median(results[key]) for key in ("vllm", "gems"))
    row = {
        "shape": shape,
        "dtype": "float32",
        "state_dtype": "bfloat16",
        "timing": "operator" if mode == "operator" else "device_event",
        "latency_base": baseline,
        "latency": optimized,
        "speedup": baseline / optimized,
    }
    print(json.dumps(row))
    update_result(op, row)


@pytest.mark.lightning_attention
@pytest.mark.parametrize("tokens", MINIMAX_PREFIX_LENGTHS)
def test_lightning_attention_benchmark(tokens, request):
    native = pytest.importorskip("vllm_ascend.ops.triton.mamba.lightning_attn")
    torch.manual_seed(932)
    qkv, slope, initial = make_qkv(1, tokens), make_slope(), make_initial(1)
    run_pair(
        request,
        "lightning_attention",
        [1, 8, tokens, 128],
        {
            "vllm": lambda: native.lightning_attention_npu(*qkv, slope, 256, initial),
            "gems": lambda: flaggems_vllm.lightning_attention(
                *qkv, slope, 256, initial
            ),
        },
        lambda: None,
        8 if tokens <= 1024 else 3,
        9,
    )
