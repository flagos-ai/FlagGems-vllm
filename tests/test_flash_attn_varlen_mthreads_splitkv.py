# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent references for compact Split-KV and dynamic grouped bounds."""

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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("groups", [4, 8])
@pytest.mark.parametrize("causal", [False, True])
def test_compact_splitkv_graph_metadata_and_ranges(dtype, groups, causal, monkeypatch):
    qlens, klens = [132, 1, 0, 1], [8193, 0, 53, 17]
    heads, page = ((4, 1), 16) if groups == 4 else ((16, 2), 32)
    args = make_case(dtype, True, qlens, klens, heads=heads, dim=256, page=page)
    args.update(causal=causal, softmax_scale=0.0625, out=torch.empty_like(args["q"]))
    resources = compile_resources([], args, monkeypatch)
    assert all(row["spills"] < 40 for row in resources), resources
    check(args, qlens, klens)

    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        got = flaggems_vllm.flash_attn_varlen_func(**args)

    # Move the long query between requests while keeping the host bounds valid.
    # The causal second replay includes both zero-KV and fully masked query rows.
    for qnew, knew, large in [
        (qlens, klens, False),
        ([1, 0, 1, 132], [0, 8193, 53, 17], True),
    ]:
        args["q"].normal_()
        args["k"].normal_()
        args["v"].normal_()
        if large:
            args["q"][2:66].mul_(4)
        args["cu_seqlens_q"].copy_(
            torch.tensor(
                [0, *itertools.accumulate(qnew)], device="musa", dtype=torch.int32
            )
        )
        args["seqused_k"].copy_(torch.tensor(knew, device="musa", dtype=torch.int32))
        args["block_table"].copy_(torch.flip(args["block_table"], [1]))
        got[0].fill_(float("nan"))
        got[1].fill_(float("nan"))
        graph.replay()
        check(args, qnew, knew, got)
