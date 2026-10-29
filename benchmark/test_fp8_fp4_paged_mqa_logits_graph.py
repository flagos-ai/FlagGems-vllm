# SPDX-License-Identifier: Apache-2.0
"""Compare the unchanged public API and static-grid graph API on HY4 shapes."""

import json

import pytest
import triton

from flaggems_vllm import fp8_fp4_paged_mqa_logits_graph
from flaggems_vllm.ops import fp8_fp4_paged_mqa_logits
from tests.test_fp8_fp4_paged_mqa_logits_graph import make_inputs


@pytest.mark.fp8_fp4_paged_mqa_logits_graph
@pytest.mark.parametrize("batch,length", [(1, 64), (1, 2048), (4, 2048), (4, 4096)])
def test_graph_api_latency(batch, length):
    args = make_inputs(batch, 1, 64, False)
    args[3].fill_(length)
    old = lambda: fp8_fp4_paged_mqa_logits(*args, None, 4096)
    new = lambda: fp8_fp4_paged_mqa_logits_graph(*args, None, 4096)
    old_ms = triton.testing.do_bench(old, warmup=100, rep=300)
    new_ms = triton.testing.do_bench(new, warmup=100, rep=300)
    graph_ms = triton.testing.do_bench_cudagraph(new, rep=300)
    print(
        json.dumps(
            {
                "batch": batch,
                "length": length,
                "legacy_us": old_ms * 1000,
                "graph_api_eager_us": new_ms * 1000,
                "graph_replay_us": graph_ms * 1000,
                "eager_speedup": old_ms / new_ms,
                "graph_speedup": old_ms / graph_ms,
            }
        )
    )
    assert old_ms / new_ms >= 0.9
