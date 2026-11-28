# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent CPU references and dynamic Graph checks for packed attention paths."""
import pytest
import torch

import flaggems_vllm
from tests.test_flash_attn_varlen_mthreads import make_case
from tests.test_flash_attn_varlen_mthreads_bounded import check

pytest_plugins = ("tests.mthreads_attention_test_utils",)

pytestmark = [
    pytest.mark.flash_attn_varlen_func,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "mthreads", reason="MThreads only"),
]


@pytest.mark.parametrize("page", [16, 32])
@pytest.mark.parametrize("heads", [(4, 1), (16, 2)])
@pytest.mark.parametrize("scale", [-0.125, 0.0, 0.0625])
@pytest.mark.parametrize("causal", [False, True])
def test_decode_single_tile_edges(page, heads, scale, causal, monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import launcher

    lengths = [0, 1, 15, 16, 17, 31, 32, 0]
    queries = [1, 1, 1, 1, 1, 1, 1, 0]
    args = make_case(torch.bfloat16, True, queries, lengths, heads, 256, page=page)
    args.update(softmax_scale=scale, causal=causal, out=torch.empty_like(args["q"]))
    original = launcher.launch_decode_tile
    called = []

    def capture(*args, **kwargs):
        kernel = original(*args, **kwargs)
        kernel._init_handles()
        assert kernel.n_spills < 40
        called.append(True)
        return kernel

    monkeypatch.setattr(launcher, "launch_decode_tile", capture)
    check(args, queries, lengths)
    assert called
    # KV page/head strides are accepted independently of the packed Q/O layout.
    args["k"] = torch.stack([args["k"], args["k"]], 1)[:, 0]
    args["v"] = torch.stack([args["v"], args["v"], args["v"]], 1)[:, 1]
    check(args, queries, lengths)


@pytest.mark.parametrize("heads", [(4, 1), (16, 2)])
def test_decode_single_tile_graph_updates(heads):
    queries, lengths = [1, 0, 1, 1], [32, 1, 0, 17]
    args = make_case(torch.bfloat16, True, queries, lengths, heads, 256, page=16)
    args.update(softmax_scale=0.0625, causal=True, out=torch.empty_like(args["q"]))
    check(args, queries, lengths)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    for qnew, knew in [([0, 1, 1, 1], [0, 32, 1, 16]), (queries, lengths)]:
        for name in ("q", "k", "v"):
            args[name].normal_()
        args["cu_seqlens_q"].copy_(
            torch.tensor(
                [0, qnew[0], sum(qnew[:2]), sum(qnew[:3]), sum(qnew)],
                device="musa",
                dtype=torch.int32,
            )
        )
        args["seqused_k"].copy_(torch.tensor(knew, device="musa", dtype=torch.int32))
        args["block_table"].copy_(torch.flip(args["block_table"], [1]))
        for _ in range(2):
            graph.replay()
            check(args, qnew, knew, got)


@pytest.mark.parametrize("page", [16, 32])
@pytest.mark.parametrize("causal", [False, True])
def test_small_prefill_gather_graph_updates(page, causal):
    queries, lengths = [513], [1023]
    args = make_case(torch.bfloat16, True, queries, lengths, (4, 1), 256, page=page)
    args.update(softmax_scale=0.0625, causal=causal, out=torch.empty_like(args["q"]))
    check(args, queries, lengths)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    for amplitude, knew in [(5.0, [517]), (1.0, [0]), (1.0, lengths)]:
        for name in ("q", "k", "v"):
            args[name].normal_()
        args["q"].mul_(amplitude)
        args["seqused_k"].copy_(torch.tensor(knew, device="musa", dtype=torch.int32))
        args["block_table"].copy_(torch.flip(args["block_table"], [1]))
        for _ in range(2):
            graph.replay()
            check(args, queries, knew, got)


@pytest.mark.parametrize("page", [16, 32])
@pytest.mark.parametrize("heads", [(8, 1), (16, 2)])
@pytest.mark.parametrize("causal", [False, True])
def test_mixed_short_kv_packed_graph_updates(page, heads, causal, monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import (
        bounded_prefill,
    )

    original = bounded_prefill.attention
    packs = []

    class RecordPacked:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                packs.append(kwargs["PACK"])
                kernel = original[grid](*args, **kwargs)
                assert kernel.n_spills < 40
                return kernel

            return run

    monkeypatch.setattr(bounded_prefill, "attention", RecordPacked())
    queries, lengths = [0, 1, 127, 385], [0, 0, 97, 1023]
    args = make_case(torch.bfloat16, True, queries, lengths, heads, 256, page=page)
    args.update(softmax_scale=0.0625, causal=causal, out=torch.empty_like(args["q"]))
    check(args, queries, lengths)
    assert packs and set(packs) == {8}
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    for qnew, knew, large in [
        ([17, 0, 129, 367], [31, 0, 0, 513], True),
        (queries, lengths, False),
    ]:
        for name in ("q", "k", "v"):
            args[name].normal_()
        if large:
            args["q"][129:].mul_(4)
        args["cu_seqlens_q"].copy_(
            torch.tensor(
                [0, qnew[0], sum(qnew[:2]), sum(qnew[:3]), sum(qnew)],
                device="musa",
                dtype=torch.int32,
            )
        )
        args["seqused_k"].copy_(torch.tensor(knew, device="musa", dtype=torch.int32))
        args["block_table"].copy_(torch.flip(args["block_table"], [1]))
        for _ in range(2):
            graph.replay()
            check(args, qnew, knew, got)


@pytest.mark.parametrize("page", [16, 32])
@pytest.mark.parametrize("heads", [(4, 1), (16, 2)])
@pytest.mark.parametrize("causal", [False, True])
def test_small_prefill_norm_tail_and_head_graph(page, heads, causal, monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import (
        bounded_prefill,
    )

    original = bounded_prefill.attention
    gather_tiles = []

    class RecordGatherTile:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                gather_tiles.append(kwargs["GN"])
                kernel = original[grid](*args, **kwargs)
                kernel._init_handles()
                assert kernel.n_spills < 40
                return kernel

            return run

    monkeypatch.setattr(bounded_prefill, "attention", RecordGatherTile())
    queries, lengths = [513], [1023]
    args = make_case(torch.bfloat16, True, queries, lengths, heads, 256, page=page)
    args.update(softmax_scale=0.0625, causal=causal, out=torch.empty_like(args["q"]))
    args["q"].fill_(0.5)

    def set_key_peak(head):
        args["k"].fill_(0.125)
        if head is not None:
            # This logical page is beyond the first half of the norm records.
            # A wrong GN=32 mask misses it when the actual gather uses GN=16.
            column = args["block_table"].shape[1] * 3 // 4
            physical = int(args["block_table"][0, column].item())
            args["k"][physical, :, head, :].fill_(16.0)

    set_key_peak(heads[1] - 1)
    check(args, queries, lengths)
    assert gather_tiles and set(gather_tiles) == {16}
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    for peak_head in [None, 0, heads[1] - 1]:
        args["block_table"].copy_(torch.flip(args["block_table"], [1]))
        set_key_peak(peak_head)
        args["v"].normal_()
        for _ in range(2):
            graph.replay()
            check(args, queries, lengths, got)
