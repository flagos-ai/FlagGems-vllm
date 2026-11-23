# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Residual dispatch must follow changing scales, including graph replay."""
import math

import pytest
import torch

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close

pytestmark = [
    pytest.mark.flash_attn_varlen_func_w8a8,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "metax", reason="MetaX residual dispatch"
    ),
]


@pytest.mark.parametrize("batch", [2, 17])
@pytest.mark.parametrize("graph_replay", [False, True])
def test_residual_flags_follow_mutable_scales(batch, graph_replay):
    device = flaggems_vllm.device
    length, keys, heads, dim = 128, 2048, 4, 256
    gap = 0.000244
    q = torch.full((2 * length, heads, dim), 127, dtype=torch.int8, device=device)
    q[:, :, 0] = 1
    k = torch.full((256, 16, 1, dim), -127, dtype=torch.int8, device=device)
    v = torch.zeros_like(k)
    for page in (0, 128):
        k[page, :2] = 0
        k[page, 1, :, 0] = 1
        v[page, 0] = 127
        v[page, 1] = -127
    cuq = torch.tensor(
        [0, length] + [2 * length] * (batch - 1), dtype=torch.int32, device=device
    )
    used = torch.full((batch,), keys, dtype=torch.int32, device=device)
    table = torch.zeros((batch, 128), dtype=torch.int32, device=device)
    table[:2] = torch.arange(256, dtype=torch.int32, device=device).reshape(2, 128)
    qs = torch.ones((batch, heads, 1), device=device)
    ks = torch.full((batch, 1, 1), gap * math.sqrt(dim), device=device).expand(
        batch, 1, keys // 128
    )
    values = torch.full((batch, 1, 1), 0.01, device=device)
    vs = values.expand(batch, 1, keys // 128)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=device)

    def run():
        return flaggems_vllm.flash_attn_varlen_func(
            q,
            k,
            v,
            length,
            cuq,
            keys,
            seqused_k=used,
            block_table=table,
            q_descale=qs,
            k_descale=ks,
            v_descale=vs,
            causal=True,
            out=out,
        )

    run()
    if graph_replay:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()

    for scales in [(0.01, 0.01), (1.7, 0.01), (0.01, -1.7), (16.0, -16.0)]:
        # Empty requests carry large scales too. They must not affect output.
        values.fill_(16.0)
        values[:2, 0, 0].copy_(torch.tensor(scales, device=device))
        if graph_replay:
            graph.replay()
        else:
            run()
        expected = torch.tensor(scales, device=device) * (-127 * math.tanh(gap / 2))
        expected = expected[:, None, None, None].expand(2, length, heads, dim)
        gems_assert_close(
            out.float(), expected.reshape_as(out), dtype=torch.float32, atol=0.025
        )
