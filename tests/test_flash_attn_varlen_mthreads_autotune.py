# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Check every compiled configuration, then the automatic cache/Graph route."""

import json
import os
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime import get_tuned_config
from tests.test_flash_attn_varlen_mthreads import make_case, reference
from tests.test_flash_attn_varlen_mthreads_full import compare, compile_resources

pytestmark = [
    pytest.mark.flash_attn_varlen_func,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "mthreads", reason="MThreads only"),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "kind", ["direct", "split", "decode", "single_kv", "mixed", "prefill"]
)
def test_candidate_space(kind, dtype, monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import autotune
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import (
        capture_call,
        use_config,
    )
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.policy import (
        candidates,
    )

    qlens, klens, heads, dim = {
        "direct": ([17, 3], [63, 0], (8, 2), 64),
        "split": ([1, 1], [8193, 17], (8, 1), 256),
        "decode": ([1, 1], [17, 0], (4, 1), 256),
        "single_kv": ([1, 0, 1], [1, 1, 0], (8, 1), 256),
        "mixed": ([257, 0, 256], [1023, 0, 17], (8, 1), 256),
        "prefill": ([513], [1023], (16, 2), 256),
    }[kind]
    args = make_case(dtype, True, qlens, klens, heads, dim, page=32)
    args.update(causal=True, softmax_scale=dim**-0.5, out=torch.empty_like(args["q"]))
    monkeypatch.setenv("MTHREADS_FA_AUTOTUNE", "0")
    with capture_call() as info:
        compile_resources([], args, monkeypatch)
    pool = candidates(
        info["problem"], info["default"], get_tuned_config("flash_attn_varlen_mthreads")
    )
    records, valid = [], []
    # Resource compilation completes before running or checking candidate outputs.
    for candidate in pool:
        row = dict(config=candidate.values())
        try:
            row["resources"] = autotune.compile_candidate(
                lambda: flaggems_vllm.flash_attn_varlen_func(**args), candidate
            )
        except autotune.RESOURCE_ERRORS as error:
            row.update(status="resource_rejected", error=str(error))
        else:
            valid.append((candidate, row))
        records.append(row)
    assert valid
    expected = reference(args, qlens, klens)
    for candidate, row in valid:
        with use_config(candidate):
            got = flaggems_vllm.flash_attn_varlen_func(**args)
        torch.musa.synchronize()
        compare(got[0], expected[0], 0.02, 0.02)
        compare(got[1], expected[1], 0.002, 0.002)
        row["status"] = "correct"
    # Each configuration must also obey current metadata inside Graph replay.
    for candidate, row in valid:
        stream = torch.musa.Stream()
        stream.wait_stream(torch.musa.current_stream())
        with use_config(candidate), torch.musa.stream(stream):
            flaggems_vllm.flash_attn_varlen_func(**args)
        torch.musa.synchronize()
        graph = torch.musa.MUSAGraph()
        with use_config(candidate), torch.musa.graph(graph, stream=stream):
            got = flaggems_vllm.flash_attn_varlen_func(**args)
        args["q"].add_(0.0625)
        graph.replay()
        torch.musa.synchronize()
        expected = reference(args, qlens, klens)
        compare(got[0], expected[0], 0.02, 0.02)
        compare(got[1], expected[1], 0.002, 0.002)
        row["status"] = "graph_passed"
    monkeypatch.setenv("MTHREADS_FA_AUTOTUNE", "1")
    got = flaggems_vllm.flash_attn_varlen_func(**args)
    compare(got[0], expected[0], 0.02, 0.02)
    compare(got[1], expected[1], 0.002, 0.002)
    key = info["key"]
    selected = autotune._MEMORY[key]
    monkeypatch.setattr(
        autotune,
        "graph_bench",
        lambda *a, **kw: pytest.fail("Retuned an exact cache hit"),
    )
    flaggems_vllm.flash_attn_varlen_func(**args)
    autotune._MEMORY.pop(key)
    flaggems_vllm.flash_attn_varlen_func(**args)
    assert autotune._MEMORY[key] == selected
    with monkeypatch.context() as patch:
        patch.setattr(torch.musa, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="Warm up this exact attention shape"):
            flaggems_vllm.flash_attn_varlen_func(
                **dict(args, max_seqlen_q=args["max_seqlen_q"] + 1)
            )
    graph = torch.musa.MUSAGraph()
    torch.musa.synchronize()
    with torch.musa.graph(graph):
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    args["q"].add_(0.03125)
    graph.replay()
    torch.musa.synchronize()
    expected = reference(args, qlens, klens)
    compare(got[0], expected[0], 0.02, 0.02)
    compare(got[1], expected[1], 0.002, 0.002)
    record = dict(
        kind=kind, dtype=str(dtype), candidates=records, selected=selected.values()
    )
    if os.environ.get("MTHREADS_FA_VALIDATION_OUT"):
        folder = Path(os.environ["MTHREADS_FA_VALIDATION_OUT"])
        with (folder / "candidate-space.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")
    print("AUTOTUNE_SPACE_RESULT", json.dumps(record), flush=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_no_tle_writes_all_outputs(dtype, monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import cached

    monkeypatch.setattr(cached, "launch_prefill", None)
    monkeypatch.setattr(cached, "create_2d_tma_descriptor", None)
    monkeypatch.setenv("MTHREADS_FA_AUTOTUNE", "0")
    qlens, klens = [257, 256], [17, 0]
    args = make_case(dtype, True, qlens, klens, (8, 1), 256, page=32)
    args.update(causal=True, out=torch.full_like(args["q"], float("nan")))
    expected = reference(args, qlens, klens)
    got = flaggems_vllm.flash_attn_varlen_func(**args)
    compare(got[0], expected[0], 0.02, 0.02)
    compare(got[1], expected[1], 0.002, 0.002)


def test_short_decode_without_tle(monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import launcher
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import (
        capture_call,
    )

    monkeypatch.setattr(launcher, "supports_tle", lambda: False)
    monkeypatch.setenv("MTHREADS_FA_AUTOTUNE", "0")
    qlens, klens = [1, 0, 1], [17, 0, 31]
    args = make_case(torch.bfloat16, True, qlens, klens, (8, 1), 256, page=32)
    args.update(causal=True, out=torch.full_like(args["q"], float("nan")))
    with capture_call() as info:
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    assert "tle_decode" not in info["problem"].routes
    expected = reference(args, qlens, klens)
    compare(got[0], expected[0], 0.02, 0.02)
    compare(got[1], expected[1], 0.002, 0.002)


@pytest.mark.parametrize(
    "observations,expected",
    [
        ({"seed": [1.0] * 5, "vector": [0.6, 0.61, 3.0, 4.0, 5.0]}, "vector"),
        ({"seed": [1.0] * 5, "outlier": [0.1, 2.0, 2.0, 2.0, 2.0]}, "seed"),
        ({"seed": [1.0] * 5, "near_tie": [0.995] * 5}, "seed"),
        ({"seed": [1.0] * 5, "first": [0.9] * 5, "third": [0.75] * 5}, "third"),
    ],
)
def test_finalist_timing_interference(observations, expected, monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import autotune

    pending = {name: iter(values) for name, values in observations.items()}
    calls = []

    def measure(fn, candidate, rep):
        calls.append((candidate, rep))
        return next(pending[candidate])

    monkeypatch.setattr(autotune, "measure_candidate", measure)
    winner, samples = autotune.confirm_selection(None, "seed", tuple(observations))
    assert winner == expected
    assert samples == observations
    assert len(calls) == len(observations) * 5
    assert all(rep == 20 for _, rep in calls)
