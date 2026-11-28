# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Former M192 workloads use the standard compiler for eager and Graph calls."""

import pytest
import torch

import flaggems_vllm
from tests.test_flash_attn_varlen_mthreads import make_case
from tests.test_flash_attn_varlen_mthreads_bounded import check
from tests.test_flash_attn_varlen_mthreads_full import compile_resources

pytest_plugins = ("tests.mthreads_attention_test_utils",)

pytestmark = [
    pytest.mark.flash_attn_varlen_func,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "mthreads", reason="MThreads only"),
]


@pytest.mark.parametrize("qlen", [577, 769, 960, 1035, 1345, 2047])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mode", ["bounded", "online", "mixed"])
def test_standard_prefill_numerics(qlen, causal, mode, monkeypatch):
    import triton.backends.mthreads.compiler as compiler
    import triton.compiler.compiler as compile_module

    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import (
        capture_call,
    )

    original_backend, original_compile = (
        compiler.MUSABackend.make_llir,
        compile_module.compile,
    )
    args = make_case(torch.bfloat16, True, [qlen], [1088], (16, 2), 256)
    args.update(causal=causal, max_seqlen_q=4096, out=torch.empty_like(args["q"]))
    args["seqused_k"].fill_(1025)
    if mode == "online":
        args["q"].mul_(8)
    elif mode == "mixed":
        args["q"][:192].mul_(8)
    monkeypatch.setenv("MTHREADS_FA_AUTOTUNE", "0")
    with capture_call() as info:
        resources = compile_resources([], args, monkeypatch)
    assert all(r["spills"] < 40 for r in resources), resources
    assert set(info["problem"].routes) <= {"generic", "bounded", "tle", "descriptor"}
    check(args, [qlen], [1025])
    assert compiler.MUSABackend.make_llir is original_backend
    assert compile_module.compile is original_compile


@pytest.mark.parametrize("qlen,heads", [(577, (4, 1)), (1035, (16, 2))])
def test_standard_prefill_mutable_graph(qlen, heads, monkeypatch):
    monkeypatch.setenv("MTHREADS_FA_AUTOTUNE", "0")
    args = make_case(torch.bfloat16, True, [qlen], [1088], heads, 256)
    args.update(causal=True, max_seqlen_q=4096, out=torch.empty_like(args["q"]))
    args["seqused_k"].fill_(1025)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    for mode, length in [
        ("online", 1025),
        ("mixed", 1025),
        ("pages", 1023),
        ("short", 17),
        ("empty", 0),
        ("zero", 1025),
    ]:
        if mode == "online":
            args["q"].mul_(8)
        elif mode == "mixed":
            args["q"].div_(8)
            args["q"][:192].mul_(8)
        elif mode == "pages":
            args["k"].neg_()
            args["v"].mul_(-0.75)
            args["block_table"].copy_(args["block_table"].flip(1))
        elif mode == "zero":
            args["q"].zero_()
        args["seqused_k"].fill_(length)
        got[0].fill_(float("nan"))
        got[1].fill_(float("nan"))
        graph.replay()
        torch.musa.synchronize()
        check(args, [qlen], [length], got)
