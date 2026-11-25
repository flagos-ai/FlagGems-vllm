# SPDX-License-Identifier: Apache-2.0
"""Exact pool-index expansion and refreshed CUDA replay coverage."""

import pytest
import torch

from flaggems_vllm import expand_pools_and_append_tail

from .glm5_reference import expand_pools_and_append_tail as reference

pytestmark = [pytest.mark.gpu, pytest.mark.expand_pools_and_append_tail]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("pool_size", [1, 4, 16])
@pytest.mark.parametrize(
    "rows,groups,padded",
    [(1, 1, False), (7, 3, True), (24, 128, False), (257, 3, True)],
)
def test_matches_reference(dtype, pool_size, rows, groups, padded):
    storage = torch.arange(
        rows * (groups + 2 * padded), device="cuda", dtype=dtype
    ).reshape(rows, -1)
    pools = storage[:, :groups]
    pools[::2, -1] = -1
    lengths = torch.arange(rows, device="cuda", dtype=dtype) + 2 * pool_size
    result = expand_pools_and_append_tail(pools, lengths, pool_size)
    assert result.dtype == torch.int32
    assert result.shape == (rows, groups * pool_size + pool_size - 1)
    assert result.device == pools.device
    torch.testing.assert_close(
        result, reference(pools, lengths, pool_size), rtol=0, atol=0
    )


@pytest.mark.parametrize("rows,groups,pool_size", [(0, 3, 4), (7, 0, 1)])
def test_empty_output(rows, groups, pool_size):
    pools = torch.empty((rows, groups), device="cuda", dtype=torch.int32)
    lengths = torch.empty(rows, device="cuda", dtype=torch.int32)
    result = expand_pools_and_append_tail(pools, lengths, pool_size)
    assert result.shape == (rows, groups * pool_size + pool_size - 1)
    assert result.dtype == torch.int32 and result.numel() == 0


@pytest.mark.parametrize("pool_size", [4, 16])
def test_graph_replay_refreshes_history_and_tail(pool_size):
    pools = torch.tensor(
        [[0, 1, -1], [3, -1, 2], [5, 0, 1]], device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor(
        [pool_size, pool_size + 1, 2 * pool_size - 1], device="cuda", dtype=torch.int32
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expand_pools_and_append_tail(pools, lengths, pool_size)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = expand_pools_and_append_tail(pools, lengths, pool_size)
    for revision in range(3):
        pools.copy_(
            torch.tensor(
                [[revision + 1, -1, 2], [-1, revision + 3, 0], [7, 2, revision]],
                device="cuda",
                dtype=torch.int32,
            )
        )
        lengths.copy_(
            torch.tensor(
                [2 * pool_size + revision, 3 * pool_size - revision - 1, pool_size + 2],
                device="cuda",
                dtype=torch.int32,
            )
        )
        expected = reference(pools, lengths, pool_size)
        result.fill_(-777)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "invalid", ["pool_size", "rank", "rows", "dtype", "columns", "seq_stride"]
)
def test_rejects_invalid_contract(invalid):
    pools = torch.zeros((2, 3), device="cuda", dtype=torch.int32)
    lengths = torch.ones(2, device="cuda", dtype=torch.int32)
    pool_size = 4
    if invalid == "pool_size":
        pool_size = 0
    elif invalid == "rank":
        pools = pools[0]
    elif invalid == "rows":
        lengths = lengths[:1]
    elif invalid == "dtype":
        pools = pools.float()
    elif invalid == "columns":
        pools = torch.zeros((2, 6), device="cuda", dtype=torch.int32)[:, ::2]
    else:
        lengths = torch.ones(4, device="cuda", dtype=torch.int32)[::2]
    with pytest.raises((TypeError, ValueError)):
        expand_pools_and_append_tail(pools, lengths, pool_size)


def test_rejects_cpu():
    with pytest.raises(NotImplementedError, match="CUDA"):
        expand_pools_and_append_tail(
            torch.zeros((2, 3), dtype=torch.int32), torch.ones(2, dtype=torch.int32), 4
        )
