# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import json
import statistics

import pytest
import torch

import flaggems_vllm
from benchmark.conftest import update_result
from tests.lightning_attention_reference import prefix_reference

pytestmark = pytest.mark.skipif(
    not hasattr(flaggems_vllm, "lightning_attention_decode"),
    reason="requires Hopper with FlagTree TLE",
)

# Shape order is B,H,T,D. Sequence and decode retain their native return contracts.
_SEQUENCE = [
    (1, 8, 8, 64),
    (1, 8, 64, 64),
    (1, 8, 257, 64),
    (2, 8, 320, 64),
    (1, 16, 512, 128),
    (1, 16, 1024, 128),
    (1, 16, 2048, 128),
    (1, 8, 4096, 128),
    (1, 8, 8192, 128),
]
_DECODE = [
    (1, 8, 1, 64),
    (8, 8, 1, 64),
    (32, 16, 1, 64),
    (1, 8, 1, 128),
    (8, 8, 1, 128),
    (32, 16, 1, 128),
    (64, 16, 1, 128),
]


def _measure(fns, resets, repeats):
    graphs = {}
    for name, fn in fns.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats[name]):
                fn()
        graphs[name] = graph
    for graph in graphs.values():
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in fns}
    for round_idx in range(11):
        order = list(fns) if round_idx % 2 == 0 else list(reversed(fns))
        for name in order:
            if resets:
                resets[name]()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / repeats[name])
    return {name: statistics.median(times) for name, times in samples.items()}


@pytest.mark.parametrize(
    "phase,shape",
    [("prefix", shape) for shape in _SEQUENCE]
    + [("decode", shape) for shape in _DECODE],
)
def test_lightning_attention_cudagraph(phase, shape, request):
    if request.config.getoption("--level") == "core" and shape not in (
        _SEQUENCE[2],
        _SEQUENCE[-2],
        _DECODE[0],
        _DECODE[-2],
    ):
        pytest.skip("comprehensive shape")
    native = pytest.importorskip("vllm.model_executor.layers.lightning_attn")
    torch.manual_seed(932)
    b, h, t, d = shape
    device = flaggems_vllm.device
    q, k, v = [torch.randn(b, h, t, d, device=device) * 0.125 for _ in range(3)]
    slopes = torch.linspace(0.002, 0.12, h, device=device)
    initial = torch.randn(b, h, d, d, device=device) * 0.125
    slots = torch.randperm(b, device=device)
    resets = {}
    if phase == "prefix":
        fns = {
            "torch": lambda: prefix_reference(q, k, v, slopes, initial, torch.float32),
            "vllm": lambda: native.lightning_attention(
                q, k, v, slopes, kv_history=initial
            ),
            "tle": lambda: flaggems_vllm.lightning_attention(
                q, k, v, slopes, kv_history=initial
            ),
        }
    else:
        pools = {name: initial.clone() for name in ("torch", "vllm", "tle")}
        resets = {
            name: (lambda pool=pool: pool.copy_(initial))
            for name, pool in pools.items()
        }

        def torch_decode():
            state = pools["torch"].index_select(0, slots)
            state = (
                torch.exp(-slopes[None, :, None, None]) * state
                + k[:, :, 0, :, None] * v[:, :, 0, None, :]
            )
            pools["torch"].index_copy_(0, slots, state)
            return (q @ state).reshape(b, h * d)

        fns = {
            "torch": torch_decode,
            "vllm": lambda: native.linear_decode_forward_triton(
                q, k, v, pools["vllm"], slopes, slots
            ),
            "tle": lambda: flaggems_vllm.lightning_attention_decode(
                q, k, v, pools["tle"], slopes, slots
            ),
        }
    calls = 32 if phase == "decode" else 12
    repeats = {"vllm": calls, "tle": calls, "torch": 1 if t >= 512 else calls}
    median = _measure(fns, resets, repeats)
    row = dict(
        phase=phase,
        shape=list(shape),
        dtype="float32",
        median_us=median,
        vllm_over_tle=median["vllm"] / median["tle"],
        torch_over_tle=median["torch"] / median["tle"],
    )
    print("LIGHTNING_BENCH " + json.dumps(row), flush=True)
    update_result("lightning_attention_" + phase, row)
