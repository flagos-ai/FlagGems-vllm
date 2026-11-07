# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

"""Numerical and changing-input Graph coverage for the C550 TP2 profile."""

import importlib
import inspect
import math

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm

pytestmark = [
    pytest.mark.fused_experts_impl,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "metax", reason="MetaX TP2 configuration coverage"
    ),
]

E, H, I, TOPK = 256, 2048, 256, 8


@pytest.fixture(scope="module")
def expert_weights():
    cache = {}

    def get(dtype):
        if dtype not in cache:
            torch.manual_seed(1067)
            w1 = torch.randn(E, 2 * I, H, device=flaggems_vllm.device, dtype=dtype)
            w2 = torch.randn(E, H, I, device=flaggems_vllm.device, dtype=dtype)
            cache[dtype] = w1 / math.sqrt(H), w2 / math.sqrt(I)
        return cache[dtype]

    return get


def reference(x, w1, w2, weights, ids):
    """Independent FP32 expert projections, activation and weighted reduction."""
    output = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    for expert in range(E):
        positions = (ids == expert).nonzero(as_tuple=False)
        if positions.numel() == 0:
            continue
        token, slot = positions[:, 0], positions[:, 1]
        projected = x[token].float() @ w1[expert].float().T
        activated = F.silu(projected[:, :I]) * projected[:, I:]
        contribution = activated @ w2[expert].float().T
        output.index_add_(0, token, contribution * weights[token, slot, None])
    return output.to(x.dtype)


@pytest.mark.parametrize(
    "m,routing,dtype",
    [(m, "uniform", torch.bfloat16) for m in (1, 63, 64, 65, 8191, 8192, 8193)]
    + [(m, "concentrated", torch.bfloat16) for m in (64, 8192)]
    + [(m, "uniform", torch.float16) for m in (64, 8192)],
)
def test_tp2_public_api_changing_graph(
    expert_weights, m, routing, dtype, record_property
):
    w1, w2 = expert_weights(dtype)
    torch.manual_seed(1067 + m)
    torch.backends.cuda.matmul.allow_tf32 = False
    public = flaggems_vllm.fused_experts_impl
    module = importlib.import_module(public.__module__)
    assert "._metax.ops.fused_moe" in module.__name__
    assert public is module.fused_experts_impl
    record_property("backend_source", inspect.getsourcefile(public))
    record_property("dtype", str(w1.dtype))
    x = torch.randn(m, H, device=w1.device, dtype=w1.dtype)
    original = x.clone()
    if routing == "uniform":
        ids = torch.topk(torch.rand(m, E, device=x.device), TOPK, dim=-1).indices.int()
    else:
        ids = torch.arange(TOPK, device=x.device).expand(m, TOPK).contiguous().int()
    weights = torch.rand(m, TOPK, device=x.device)
    weights /= weights.sum(-1, keepdim=True)

    def call():
        return public(x, w1, w2, weights, ids, inplace=False, activation="silu")

    expected = reference(x, w1, w2, weights, ids)
    for _ in range(5):
        actual = call()
    torch.cuda.synchronize()
    # Same tolerance as the independently validated local TP2 configuration sweep.
    torch.testing.assert_close(actual, expected, rtol=0.1, atol=0.01)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = call()
    assert actual.data_ptr() != x.data_ptr()
    for scale in (0.5, -0.75):
        x.copy_(original * scale)
        expected = reference(x, w1, w2, weights, ids)
        actual.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0.1, atol=0.01)
        record_property(
            f"max_abs_scale_{scale}", (actual - expected).abs().max().item()
        )
    record_property("changed_input_graph_replays", 2)
