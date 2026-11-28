# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Medium-cache prefill: conservative Q bounds and mutable online fallback."""
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


@pytest.mark.parametrize("heads", [(4, 1), (16, 2)])
@pytest.mark.parametrize("causal", [False, True])
def test_medium_cache_prefill_graph(heads, causal, monkeypatch):
    args = make_case(torch.bfloat16, True, [577], [4097], heads=heads, dim=256, page=32)
    args.update(causal=causal, max_seqlen_q=4108, out=torch.empty_like(args["q"]))
    args["seqused_k"].fill_(1025)
    resources = compile_resources([], args, monkeypatch)
    assert all(r["spills"] < 40 for r in resources)
    check(args, [577], [1025])
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    for length, large in [(4097, False), (1025, True), (17, False), (0, False)]:
        args["q"].normal_()
        args["k"].normal_()
        args["v"].normal_()
        if large:
            args["q"][:192].mul_(8)
        args["seqused_k"].fill_(length)
        args["block_table"].copy_(args["block_table"].flip(1))
        got[0].fill_(float("nan"))
        got[1].fill_(float("nan"))
        graph.replay()
        check(args, [577], [length], got)
