# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import json
import statistics

import pytest
import torch

import flaggems_vllm
from benchmark.conftest import update_result
from tests.test_linear_attention import (
    DECODE_CASES,
    PREFIX_CASES,
    PrefixCase,
    case_id,
    make_decode_inputs,
    make_initial,
    make_qkv,
    make_slope,
    make_state,
    native_decode_reason,
    native_prefix_reason,
    prefix_reference,
)

pytestmark = pytest.mark.skipif(
    not hasattr(flaggems_vllm, "lightning_attention_decode"),
    reason="requires Hopper with FlagTree TLE",
)


def _measure(fns, reset, repeats):
    graphs = {}
    for name, fn in fns.items():
        if reset:
            reset()
        for _ in range(5):
            fn()
        if reset:
            reset()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats[name]):
                fn()
        graphs[name] = graph
    for graph in graphs.values():
        if reset:
            reset()
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in fns}
    for round_idx in range(11):
        order = list(fns)
        shift = round_idx % len(order)
        order = order[shift:] + order[:shift]
        if round_idx % 2:
            order.reverse()
        for name in order:
            if reset:
                reset()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / repeats[name])
    return {name: statistics.median(times) for name, times in samples.items()}


def benchmark_case(case, native, candidates, include_torch=True):
    """Measure the same tensors, layouts and state-update contract for every provider."""
    torch.manual_seed(932)
    device = flaggems_vllm.device
    q, k, v = make_qkv(case, device)
    fns = {}
    if isinstance(case, PrefixCase):
        reset = None
        phase = "prefix"
        initial = make_initial(case, device)
        slope = make_slope(case, device)
        native_reason = native_prefix_reason(case)
        if include_torch:

            def torch_prefix():
                y, states = prefix_reference(q, k, v, slope, initial, torch.float32)
                return y.to(q.dtype), states

            fns["torch"] = torch_prefix
        if native_reason is None:
            fns["vllm"] = lambda: native.lightning_attention(
                q, k, v, slope, kv_history=initial
            )
        for name, (prefix, _) in candidates.items():
            fns[name] = lambda prefix=prefix: prefix(q, k, v, slope, kv_history=initial)
        calls = 12 if case.t < 4096 else 6
    else:
        phase = "decode"
        slope, slots, initial = make_decode_inputs(case, device)
        native_reason = native_decode_reason((q, k, v), slope, slots)

        # All graphs replay serially on the same physical cache address. A separate
        # allocation per provider can change bandwidth on sparse slot patterns.
        cache = make_state(case, initial.size(0), device)

        def reset():
            cache.copy_(initial)

        if include_torch:
            # Valid rows are fixed for benchmarking. Prepare indices outside capture.
            rows = (slots >= 0).nonzero().flatten()
            ids = slots.index_select(0, rows).long()

            def torch_decode():
                state = cache.index_select(0, ids)
                ks = k.index_select(0, rows)[:, :, 0, :, None].float()
                vs = v.index_select(0, rows)[:, :, 0, None, :].float()
                state = torch.exp(-slope.reshape(1, -1, 1, 1)) * state + ks * vs
                cache.index_copy_(0, ids, state.to(cache.dtype))
                product = q.index_select(0, rows).float() @ state
                y = q.new_zeros((case.b, case.h * case.d))
                y.index_copy_(
                    0, rows, product.reshape(rows.numel(), case.h * case.d).to(q.dtype)
                )
                return y

            fns["torch"] = torch_decode
        if native_reason is None:
            fns["vllm"] = lambda: native.linear_decode_forward_triton(
                q, k, v, cache, slope, slots, case.block
            )
        for name, (_, decode) in candidates.items():
            fns[name] = lambda decode=decode: decode(
                q, k, v, cache, slope, slots, case.block
            )
        calls = 32
    repeats = {
        key: (1 if key == "torch" and getattr(case, "t", 1) >= 512 else calls)
        for key in fns
    }
    times = _measure(fns, reset, repeats)
    return dict(
        id=case_id(case),
        phase=phase,
        group=case.group,
        median_us=times,
        native_unavailable=native_reason,
    )


_CORE = {
    "p-minimax-tp8-b1h8t128d128e128",
    "p-minimax-tp8-b1h8t1024d128e128",
    "d-minimax-tp8-b1h8t1d128e128",
    "d-minimax-tp8-b8h8t1d128e128",
}


@pytest.mark.parametrize("case", PREFIX_CASES + DECODE_CASES, ids=case_id)
def test_lightning_attention_cudagraph(case, request):
    if request.config.getoption("--level") == "core" and case_id(case) not in _CORE:
        pytest.skip("comprehensive shape")
    native = pytest.importorskip("vllm.model_executor.layers.lightning_attn")
    row = benchmark_case(
        case,
        native,
        {
            "tle": (
                flaggems_vllm.lightning_attention,
                flaggems_vllm.lightning_attention_decode,
            )
        },
    )
    print("LIGHTNING_BENCH " + json.dumps(row), flush=True)
    update_result("lightning_attention_" + row["phase"], row)
