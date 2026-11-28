# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Cached prefill must gather current strided KV on every Graph replay."""

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
@pytest.mark.parametrize("layout", ["interleaved", "independent_strides"])
@pytest.mark.parametrize("causal", [False, True])
def test_cached_strided_kv_graph(dtype, layout, causal, monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import launcher

    queries, lengths = [257, 0, 256], [1023, 0, 17]
    args = make_case(dtype, True, queries, lengths, (8, 1), 256, page=32)
    key, value = args["k"], args["v"]
    pages, page, heads, dim = key.shape
    if layout == "interleaved":
        storage = torch.full(
            (pages, 2, page, heads, dim), float("nan"), device="musa", dtype=dtype
        )
        args["k"], args["v"] = storage[:, 0], storage[:, 1]
    else:
        key_storage = torch.full(
            (pages * 2, page * 2, heads * 2, dim),
            float("nan"),
            device="musa",
            dtype=dtype,
        )
        value_storage = torch.full(
            (pages * 3, page * 3, heads * 3, dim),
            float("nan"),
            device="musa",
            dtype=dtype,
        )
        args["k"] = key_storage[::2, ::2, ::2]
        args["v"] = value_storage[1::3, 1::3, 1::3]
    args["k"].copy_(key)
    args["v"].copy_(value)
    assert not args["k"].is_contiguous() and not args["v"].is_contiguous()
    output = torch.full((sum(queries) + 1, 8, 256), 13, device="musa", dtype=dtype)
    args.update(causal=causal, softmax_scale=0.0625, max_seqlen_q=513, out=output[:-1])
    calls = []
    original = launcher.launch_cached

    def record_cached(*values, **options):
        calls.append(True)
        return original(*values, **options)

    monkeypatch.setattr(launcher, "launch_cached", record_cached)
    resources = compile_resources([], args, monkeypatch)
    assert all(row["spills"] < 40 for row in resources), resources
    check(args, queries, lengths)
    assert calls, "The intended cached route was not exercised"
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    for qnew, knew, amplitude in [
        ([1, 0, 512], [0, 0, 1023], 5.0),
        (queries, lengths, 1.0),
    ]:
        for name in ("q", "k", "v"):
            args[name].normal_()
        args["q"].mul_(amplitude)
        args["cu_seqlens_q"].copy_(
            torch.tensor(
                [0, qnew[0], qnew[0], sum(qnew)], device="musa", dtype=torch.int32
            )
        )
        args["seqused_k"].copy_(torch.tensor(knew, device="musa", dtype=torch.int32))
        args["block_table"].copy_(args["block_table"].flip(1))
        for _ in range(2):
            got[0].fill_(float("nan"))
            got[1].fill_(float("nan"))
            graph.replay()
            check(args, qnew, knew, got)
            assert (output[-1] == 13).all().item()
