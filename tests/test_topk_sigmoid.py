# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.topk_sigmoid,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def reference(logits, bias, k, renormalize):
    p = torch.nan_to_num(torch.sigmoid(logits.float()), nan=0.0)
    choice = p if bias is None else p + bias
    ids = torch.argsort(choice, dim=-1, descending=True, stable=True)[:, :k]
    weights = p.gather(1, ids)
    if renormalize:
        denom = weights.sum(-1, keepdim=True)
        weights = weights / torch.where(denom > 0, denom, torch.ones_like(denom))
    m = logits.shape[0]
    source = (
        torch.arange(k, device=logits.device)[None, :] * m
        + torch.arange(m, device=logits.device)[:, None]
    )
    return weights, ids, source.int()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(0, 32, 4), (1, 64, 8), (7, 128, 8), (64, 256, 4)])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("bias_on", [False, True])
def test_topk_sigmoid(dtype, shape, renormalize, bias_on):
    m, e, k = shape
    torch.manual_seed(31)
    logits = torch.randn((m, e), device="cuda", dtype=dtype)
    bias = torch.randn((e,), device="cuda") * 0.1 if bias_on else None
    w = torch.empty((m, k), device="cuda")
    ids = torch.empty((m, k), device="cuda", dtype=torch.int64)
    source = torch.empty((m, k), device="cuda", dtype=torch.int32)
    flaggems_vllm.topk_sigmoid(w, ids, source, logits, renormalize, bias)
    expected = reference(logits, bias, k, renormalize)
    torch.testing.assert_close(w, expected[0], atol=2e-7, rtol=2e-6)
    assert torch.equal(ids, expected[1]) and torch.equal(source, expected[2])


def test_topk_sigmoid_ties_nan_graph():
    logits = torch.zeros((3, 128), device="cuda")
    logits[1] = float("nan")
    logits[2] = -float("inf")
    w = torch.empty((3, 8), device="cuda")
    ids = torch.empty((3, 8), device="cuda", dtype=torch.int32)
    src = torch.empty_like(ids)
    flaggems_vllm.topk_sigmoid(w, ids, src, logits, True)
    expected = reference(logits, None, 8, True)
    assert torch.equal(ids, expected[1].int())
    torch.testing.assert_close(w, expected[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flaggems_vllm.topk_sigmoid(w, ids, src, logits, True)
    logits.copy_(torch.randn_like(logits))
    graph.replay()
    expected = reference(logits, None, 8, True)
    assert torch.equal(ids, expected[1].int())
    torch.testing.assert_close(w, expected[0], atol=2e-7, rtol=2e-6)
