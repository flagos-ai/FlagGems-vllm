# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Device lengths may cross the short-KV boundary during graph replay."""
import math

import pytest
import torch

import flaggems_vllm as gems
from tests.accuracy_utils import gems_assert_close

pytestmark = [
    pytest.mark.flash_attn_varlen_func_w8a8,
    pytest.mark.skipif(gems.vendor_name != "metax", reason="MetaX decode scheduling"),
]


@pytest.mark.parametrize("batch", [255, 256])
@pytest.mark.parametrize("group", [4, 8])
@pytest.mark.parametrize("graph_replay", [False, True])
def test_decode_mutable_lengths(group, graph_replay, batch):
    kvheads, dim = 2, 256
    heads = kvheads * group
    device = gems.device
    q = torch.zeros((batch, heads, dim), dtype=torch.int8, device=device)
    k = torch.zeros((4, 16, kvheads, dim), dtype=torch.int8, device=device)
    values = torch.arange(-32, 32, dtype=torch.int8, device=device)
    v = values[:, None, None].expand(64, kvheads, dim).reshape_as(k).contiguous()
    used = torch.full((batch,), 32, dtype=torch.int32, device=device)
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device)
    table = torch.arange(4, dtype=torch.int32, device=device)[None, :].expand(batch, 4)
    qs = torch.ones((batch, heads, 1), device=device)
    ks = torch.ones((batch, kvheads, 1), device=device).expand(batch, kvheads, 576)
    vs = torch.full((batch, kvheads, 1), 1.5, device=device).expand(batch, kvheads, 576)
    out = torch.empty(q.shape, device=device, dtype=torch.bfloat16)

    def run():
        return gems.flash_attn_varlen_func(
            q,
            k,
            v,
            1,
            cu,
            73728,
            seqused_k=used,
            block_table=table,
            q_descale=qs,
            k_descale=ks,
            v_descale=vs,
            causal=True,
            num_splits=32,
            return_softmax_lse=True,
            out=out,
        )

    result = run()
    if graph_replay:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run()
    for length in [0, 1, 2, 31, 32, 33, 64, 32]:
        used.fill_(length)
        if graph_replay:
            graph.replay()
        else:
            result = run()
        expected = torch.full_like(out, (-32 + (length - 1) / 2) * 1.5 if length else 0)
        gems_assert_close(
            out.float(), expected.float(), dtype=torch.float32, atol=0.025
        )
        expected_lse = torch.full_like(
            result[1], math.log(length) if length else float("inf")
        )
        gems_assert_close(result[1], expected_lse, dtype=torch.float32, atol=0.0015)
