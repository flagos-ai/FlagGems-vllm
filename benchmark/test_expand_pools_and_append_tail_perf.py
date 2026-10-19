# SPDX-License-Identifier: Apache-2.0
"""Frozen small/hot/odd/large pool-expansion workload against Torch composition."""

import json

import pytest
import torch
import triton

from flaggems_vllm import expand_pools_and_append_tail
from tests.glm5_reference import expand_pools_and_append_tail as reference

pytestmark = pytest.mark.expand_pools_and_append_tail


@pytest.mark.parametrize(
    "rows,groups,pool_size", [(1, 1, 4), (7, 3, 16), (24, 128, 16), (257, 128, 16)]
)
def test_expand_pools_and_append_tail_perf(rows, groups, pool_size):
    pools = torch.arange(rows * groups, device="cuda", dtype=torch.int32).reshape(
        rows, groups
    )
    pools[:, -1] = -1
    lengths = torch.arange(rows, device="cuda", dtype=torch.int32) + 2 * pool_size
    torch.testing.assert_close(
        expand_pools_and_append_tail(pools, lengths, pool_size),
        reference(pools, lengths, pool_size),
        rtol=0,
        atol=0,
    )
    baseline_us = 1000 * triton.testing.do_bench(
        lambda: reference(pools, lengths, pool_size),
        warmup=100,
        rep=500,
        return_mode="median",
    )
    candidate_us = 1000 * triton.testing.do_bench(
        lambda: expand_pools_and_append_tail(pools, lengths, pool_size),
        warmup=100,
        rep=500,
        return_mode="median",
    )
    print(
        json.dumps(
            {
                "op": "expand_pools_and_append_tail",
                "rows": rows,
                "groups": groups,
                "pool_size": pool_size,
                "baseline_us": baseline_us,
                "candidate_us": candidate_us,
                "speedup": baseline_us / candidate_us,
            }
        )
    )
