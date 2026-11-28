# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""GQA4 compact prefill: mutable ragged rows, pages, bounds and masked rows."""
import itertools

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


@pytest.mark.parametrize("causal", [False, True])
def test_gqa4_compact_prefill_graph(causal, monkeypatch):
    qlens, klens = [257, 1, 0, 769], [1025, 0, 32, 1023]
    args = make_case(torch.bfloat16, True, qlens, klens, heads=(4, 1), dim=256, page=16)
    args.update(causal=causal, out=torch.empty_like(args["q"]))
    resources = compile_resources([], args, monkeypatch)
    assert all(r["spills"] < 40 for r in resources)
    check(args, qlens, klens)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    for qnew, knew, large in [
        (qlens, klens, False),
        ([1, 0, 513, 513], [0, 1025, 17, 1023], True),
        (qlens, [0, 0, 0, 0], False),
    ]:
        args["q"].normal_()
        args["k"].normal_()
        args["v"].normal_()
        if large:
            args["q"][1:65].mul_(8)
        args["cu_seqlens_q"].copy_(
            torch.tensor(
                [0, *itertools.accumulate(qnew)], device="musa", dtype=torch.int32
            )
        )
        args["seqused_k"].copy_(torch.tensor(knew, device="musa", dtype=torch.int32))
        args["block_table"].copy_(args["block_table"].flip(1))
        for _ in range(2):
            got[0].fill_(float("nan"))
            got[1].fill_(float("nan"))
            graph.replay()
            check(args, qnew, knew, got)
