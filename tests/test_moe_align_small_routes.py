# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Full-buffer correctness for the MetaX small-route non-TLE specialization."""
import pytest
import torch

from flaggems_vllm import runtime
from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_no_tle

pytestmark = pytest.mark.skipif(
    runtime.device.vendor_name != "metax", reason="MetaX fast path"
)


def _expected(ids, e=1024, block=16):
    ids = ids.cpu().flatten()
    n = ids.numel()
    capacity = n + e * (block - 1)
    if n < e:
        capacity = min(n * block, capacity)
    sorted_ids = torch.full((capacity,), n, dtype=torch.int32)
    expert_ids = torch.full(((capacity + block - 1) // block,), -1, dtype=torch.int32)
    position = 0
    for expert in range(e):
        routes = torch.nonzero(ids == expert).flatten().to(torch.int32)
        count = routes.numel()
        if count:
            padded = (count + block - 1) // block * block
            sorted_ids[position : position + count] = routes
            expert_ids[position // block : (position + padded) // block] = expert
            position += padded
    return sorted_ids, expert_ids, torch.tensor([position], dtype=torch.int32)


@pytest.mark.parametrize("m", [2, 3, 12, 24, 64])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("pattern", ["spread", "random", "last_experts"])
def test_small_route_full_buffers(m, dtype, pattern):
    torch.manual_seed(611 + m)
    if pattern == "spread":
        ids = torch.arange(m * 16).reshape(m, 16) % 1024
    elif pattern == "last_experts":
        ids = (torch.arange(16) + 1008).expand(m, -1).clone()
    else:
        ids = torch.randint(0, 1024, (m, 16))
    ids = ids.to(dtype).cuda()
    expected = _expected(ids)
    for pad_sorted_ids in (False, True):
        actual = moe_align_block_size_no_tle(
            ids, 16, 1024, pad_sorted_ids=pad_sorted_ids
        )
        assert all(torch.equal(a.cpu(), b) for a, b in zip(actual, expected))


def test_small_route_changing_graph_last_bin():
    ids = torch.randint(0, 1024, (12, 16), device="cuda", dtype=torch.int32)
    moe_align_block_size_no_tle(ids, 16, 1024)
    moe_align_block_size_no_tle(ids, 16, 1024)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = moe_align_block_size_no_tle(ids, 16, 1024)
    for mode in range(3):
        if mode == 0:
            ids.fill_(1023)
        elif mode == 1:
            ids.random_(0, 1024)
        else:
            ids.copy_(torch.arange(192, device="cuda").reshape(12, 16))
        graph.replay()
        expected = _expected(ids)
        assert all(torch.equal(a.cpu(), b) for a, b in zip(output, expected))


def test_caller_workspace_retains_native_fallback():
    torch.manual_seed(710)
    ids = torch.randint(0, 1024, (12, 16), dtype=torch.int32, device="cuda")
    workspace = (
        torch.zeros(1025, dtype=torch.int32, device="cuda"),
        torch.zeros(1025, 1024, dtype=torch.int32, device="cuda"),
    )
    expected = _expected(ids)
    actual = moe_align_block_size_no_tle(ids, 16, 1024, workspace=workspace)
    assert all(torch.equal(a.cpu(), b) for a, b in zip(actual, expected))
