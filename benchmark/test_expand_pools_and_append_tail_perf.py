# SPDX-License-Identifier: Apache-2.0
"""Frozen small/hot/odd/large pool-expansion workload against Torch composition."""

import json
from functools import partial

import pytest
import torch
import triton

from benchmark.conftest import Config
from flaggems_vllm import device, expand_pools_and_append_tail, vendor_name
from tests.glm5_reference import expand_pools_and_append_tail as reference

pytestmark = pytest.mark.expand_pools_and_append_tail


@pytest.mark.parametrize(
    "rows,groups,pool_size", [(1, 1, 4), (7, 3, 16), (24, 128, 16), (257, 128, 16)]
)
def test_expand_pools_and_append_tail_perf(rows, groups, pool_size):
    pools = torch.arange(rows * groups, device=device, dtype=torch.int32).reshape(
        rows, groups
    )
    pools[:, -1] = -1
    lengths = torch.arange(rows, device=device, dtype=torch.int32) + 2 * pool_size
    torch.testing.assert_close(
        expand_pools_and_append_tail(pools, lengths, pool_size),
        reference(pools, lengths, pool_size),
        rtol=0,
        atol=0,
    )
    if vendor_name == "ascend":
        from triton.backends.ascend.testing import do_bench_npu as timer
    else:
        timer = partial(
            triton.testing.do_bench, warmup=100, rep=500, return_mode="median"
        )
    baseline_us = None
    if Config.user_desired_metrics != ["latency"]:
        baseline_us = 1000 * timer(lambda: reference(pools, lengths, pool_size))
    candidate_us = 1000 * timer(
        lambda: expand_pools_and_append_tail(pools, lengths, pool_size)
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
                "speedup": (
                    baseline_us / candidate_us if baseline_us is not None else None
                ),
            }
        )
    )
