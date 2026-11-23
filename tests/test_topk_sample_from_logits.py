# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import flaggems_vllm

pytestmark = pytest.mark.skipif(
    flaggems_vllm.device != "musa", reason="MUSA-specific fused sampler"
)


def reference(logits, k, noise):
    ordered = logits.sort(dim=-1).values
    cutoff = ordered.gather(1, (logits.shape[1] - k.long())[:, None])
    masked = logits.masked_fill(logits < cutoff, -float("inf"))
    return (masked.softmax(dim=-1) / noise).argmax(dim=-1)


@pytest.mark.topk_sample_from_logits
@pytest.mark.parametrize("batch,vocab", [(1, 1), (4, 17), (16, 6761), (4, 8192)])
@pytest.mark.parametrize("k_dtype", [torch.int32, torch.int64])
def test_topk_sample_from_logits(batch, vocab, k_dtype):
    rng = torch.Generator(device="musa").manual_seed(19)
    logits = torch.randn((batch, vocab), device="musa", generator=rng)
    noise = torch.empty_like(logits).exponential_(generator=rng)
    cases = [min(value, vocab) for value in (1, 25, 32, 33, 64, 65, vocab)]
    k = torch.tensor(
        [cases[row % len(cases)] for row in range(batch)], dtype=k_dtype, device="musa"
    )
    actual = flaggems_vllm.topk_sample_from_logits(logits, k, noise)
    torch.testing.assert_close(actual, reference(logits, k, noise), rtol=0, atol=0)


@pytest.mark.topk_sample_from_logits
def test_ties_and_nonfinite_logits():
    logits = torch.full((4, 17), 0.5, device="musa")
    logits[1].fill_(-float("inf"))
    logits[2, -1] = float("nan")
    logits[3, -1] = float("inf")
    k = torch.full((4,), 2, dtype=torch.int32, device="musa")
    noise = torch.ones_like(logits)
    noise[0, -1] = 0.1
    before = logits.clone()
    actual = flaggems_vllm.topk_sample_from_logits(logits, k, noise)
    torch.testing.assert_close(actual, reference(logits, k, noise), rtol=0, atol=0)
    torch.testing.assert_close(logits, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.topk_sample_from_logits
def test_empty_batch_and_bad_dtype():
    logits = torch.empty((0, 17), device="musa")
    k = torch.empty(0, dtype=torch.int32, device="musa")
    actual = flaggems_vllm.topk_sample_from_logits(logits, k, logits)
    assert actual.shape == (0,)
    assert actual.dtype == torch.int64
    with pytest.raises(ValueError, match="FP32"):
        flaggems_vllm.topk_sample_from_logits(logits.half(), k, logits)


@pytest.mark.topk_sample_from_logits
def test_musa_graph_replay(monkeypatch):
    monkeypatch.setenv("TRITON_MUSA_ENABLE_LLC_OPT", "1")
    op = flaggems_vllm.topk_sample_from_logits
    logits = torch.randn((16, 6761), device="musa")
    k = torch.full((16,), 25, dtype=torch.int32, device="musa")
    noise = torch.empty_like(logits).exponential_()
    op(logits, k, noise)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        actual = op(logits, k, noise)
    graph.replay()
    torch.testing.assert_close(actual, reference(logits, k, noise), rtol=0, atol=0)
    k.fill_(33)
    logits.normal_()
    graph.replay()
    torch.testing.assert_close(actual, reference(logits, k, noise), rtol=0, atol=0)
    assert os.environ["TRITON_MUSA_ENABLE_LLC_OPT"] == "1"
