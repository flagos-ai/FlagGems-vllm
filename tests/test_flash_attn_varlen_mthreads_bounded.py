# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent references and dynamic Graph checks for norm-bounded paths."""

import pytest
import torch

import flaggems_vllm
from tests.test_flash_attn_varlen_mthreads import make_case, reference

pytest_plugins = ("tests.mthreads_attention_test_utils",)

pytestmark = [
    pytest.mark.flash_attn_varlen_func,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "mthreads", reason="MThreads only"),
]


def check(args, qlens, klens, got=None):
    expected = reference(args, qlens, klens)
    if got is None:
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    assert got[0] is args["out"]
    torch.testing.assert_close(got[0].cpu(), expected[0], atol=0.02, rtol=0.02)
    torch.testing.assert_close(got[1].cpu(), expected[1], atol=0.002, rtol=0.002)


@pytest.mark.parametrize("page", [16, 32])
@pytest.mark.parametrize("scale", [0.0625, 0.125])
@pytest.mark.parametrize("causal", [False, True])
def test_bounded_prefill_mixed_ranges(page, scale, causal):
    qlens, klens = [0, 1, 127, 257, 512], [0, 0, 64, 509, 1023]
    args = make_case(
        torch.bfloat16, True, qlens, klens, heads=(4, 1), dim=256, page=page
    )
    args.update(softmax_scale=scale, causal=causal, out=torch.empty_like(args["q"]))
    check(args, qlens, klens)
    # Adjacent tiles use both fixed-reference and online softmax.
    args["q"][129:257].mul_(4)
    check(args, qlens, klens)


@pytest.mark.parametrize("causal", [False, True])
def test_bounded_prefill_graph_updates(causal):
    qlens, klens = [129, 385], [257, 1023]
    args = make_case(
        torch.bfloat16, True, qlens, klens, heads=(16, 2), dim=256, page=32
    )
    args.update(softmax_scale=0.0625, causal=causal, out=torch.empty_like(args["q"]))
    check(args, qlens, klens)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    for step in range(4):
        args["q"].normal_()
        args["k"].normal_()
        args["v"].normal_()
        if step % 2:
            args["q"][:128].mul_(8)
            args["q"][256:].mul_(3)
        qnew = [257, 257] if step % 2 else qlens
        knew = [0, 255] if step == 2 else klens
        args["cu_seqlens_q"].copy_(
            torch.tensor([0, qnew[0], sum(qnew)], device="musa", dtype=torch.int32)
        )
        args["seqused_k"].copy_(torch.tensor(knew, device="musa", dtype=torch.int32))
        args["block_table"].copy_(torch.flip(args["block_table"], [1]))
        for _ in range(3):
            graph.replay()
            check(args, qnew, knew, got)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("large_q", [False, True])
def test_bounded_splitkv_ranges(causal, large_q):
    qlens, klens = [1, 3, 17], [8193, 0, 9000]
    args = make_case(
        torch.bfloat16, True, qlens, klens, heads=(16, 2), dim=256, page=32
    )
    args.update(softmax_scale=0.0625, causal=causal, out=torch.empty_like(args["q"]))
    if large_q:
        args["q"][4:].mul_(4)
    check(args, qlens, klens)


def test_bounded_splitkv_graph_updates():
    qlens, klens = [1, 3, 17], [8193, 0, 9000]
    args = make_case(
        torch.bfloat16, True, qlens, klens, heads=(16, 2), dim=256, page=32
    )
    args.update(softmax_scale=0.0625, causal=True, out=torch.empty_like(args["q"]))
    check(args, qlens, klens)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    for large_q, knew in [(True, [0, 8193, 1]), (False, klens)]:
        args["q"].normal_()
        args["k"].normal_()
        args["v"].normal_()
        if large_q:
            args["q"][4:].mul_(4)
        args["seqused_k"].copy_(torch.tensor(knew, device="musa", dtype=torch.int32))
        args["block_table"].copy_(torch.flip(args["block_table"], [1]))
        for _ in range(3):
            graph.replay()
            check(args, qlens, knew, got)
