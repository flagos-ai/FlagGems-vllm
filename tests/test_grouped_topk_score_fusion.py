# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-group score fusion with independent unbiased-weight selection."""

import importlib

import pytest
import torch

import flaggems_vllm


def reference(scores, bias, count, scale, groups, keep_groups):
    processed = (0.5 * torch.tanh(0.5 * scores.float()) + 0.5).float()
    corrected = processed + bias.float()
    grouped = corrected.reshape(scores.shape[0], groups, -1)
    group_scores = torch.topk(grouped, 2, dim=-1).values.sum(-1)
    selected_groups = torch.argsort(group_scores, dim=-1, descending=True, stable=True)[
        :, :keep_groups
    ]
    mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(
        1, selected_groups, True
    )
    eligible = corrected.masked_fill(
        ~mask[:, :, None].expand_as(grouped).reshape_as(corrected), -torch.inf
    )
    selected = torch.argsort(eligible, dim=-1, descending=True, stable=True)[:, :count]
    weights = processed.gather(1, selected)
    weights = weights * (scale / (weights.sum(1, keepdim=True) + 1e-20))
    return weights, selected.to(torch.int32)


@pytest.mark.grouped_topk
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("rows", [1, 64, 256])
@pytest.mark.parametrize("groups", [1, 4])
@torch.inference_mode()
def test_grouped_score_single_group_reference_and_graph(rows, groups, monkeypatch):
    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._nvidia.ops.grouped_topk"
    )
    monkeypatch.setattr(module, "_GROUPED_TOPK_SCORE_FUSION", True)
    monkeypatch.setattr(module, "_GROUPED_TOPK_SINGLE_GROUP_FUSION", True)
    assert flaggems_vllm.grouped_topk is module.grouped_topk
    generator = torch.Generator().manual_seed(2026 + rows)
    cpu = torch.randn(rows, 384, generator=generator)
    # Distinct, well-separated bias ranks avoid converting an unrelated tanh
    # approximation difference into a selection-tie test.
    cpu_bias = torch.arange(384).float() * 2
    scores, bias = cpu.cuda(), cpu_bias.cuda()

    def run():
        return flaggems_vllm.grouped_topk(
            scores, groups, min(groups, 2), 8, True, 1.0, bias, 1
        )

    def check(actual, source):
        expected = reference(source, cpu_bias, 8, 1.0, groups, min(groups, 2))
        assert torch.equal(actual[1].cpu(), expected[1])
        torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=2e-5, atol=2e-6)

    check(run(), cpu)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        run()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
    torch.cuda.current_stream().wait_stream(stream)
    scores.copy_(-cpu)
    captured[0].fill_(17)
    captured[1].fill_(-17)
    graph.replay()
    torch.cuda.synchronize()
    check(captured, -cpu)
    scores.zero_()
    bias.zero_()
    captured[0].fill_(17)
    captured[1].fill_(-17)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(
        captured[1], torch.arange(8, device="cuda", dtype=torch.int32).expand(rows, 8)
    )
    assert torch.equal(captured[0], torch.full_like(captured[0], 0.125))
