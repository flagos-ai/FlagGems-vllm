# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent reference and changed-input graph tests for MetaX."""
import pytest
import torch

import flaggems_vllm as ops

pytestmark = pytest.mark.skipif(
    ops.vendor_name != "metax", reason="MetaX specialization"
)


def check_replays(call, update, check, outputs=()):
    # No compilation, allocation growth or tuning is permitted during capture.
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.5, -0.75):
        update(value)
        for output in outputs:
            output.fill_(float("nan") if output.is_floating_point() else -999)
        graph.replay()
        torch.cuda.synchronize()
        check()


@pytest.mark.topk_softplus_sqrt
@pytest.mark.parametrize("hashed", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_public_sqrtsoftplus_routing_reference_and_replay(hashed, renormalize, dtype):
    rows, experts, k = 3, 256, 6
    logits = torch.randn(rows, experts, device="cuda", dtype=dtype)
    bias = torch.randn(experts, device="cuda") * 0.1
    tokens = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int64)
    table = torch.arange(18, device="cuda", dtype=torch.int32).view(3, 6)
    result = []

    def call():
        weights = torch.empty((rows, k), device=logits.device, dtype=torch.float32)
        indices = torch.empty((rows, k), device=logits.device, dtype=torch.int32)
        slots = torch.empty((rows, k), device=logits.device, dtype=torch.int32)
        ops.topk_softplus_sqrt(
            weights,
            indices,
            slots,
            logits,
            renormalize,
            2.5,
            bias,
            tokens if hashed else None,
            table if hashed else None,
        )
        result[:] = (weights, indices)

    def check():
        score = torch.nn.functional.softplus(logits.cpu().float()).sqrt()
        if hashed:
            selected = table.cpu()[tokens.cpu()].long()
        else:
            selected = torch.argsort(score + bias.cpu(), descending=True, stable=True)[
                :, :k
            ]
        weights = score.gather(1, selected)
        if renormalize:
            weights /= weights.sum(1, keepdim=True)
        weights *= 2.5
        assert torch.equal(result[1].cpu().long(), selected)
        torch.testing.assert_close(result[0].cpu(), weights, rtol=1e-4, atol=1e-6)

    call()
    check()
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.125, -0.25):
        logits.add_(value)
        result[0].fill_(float("nan"))
        result[1].fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        check()
