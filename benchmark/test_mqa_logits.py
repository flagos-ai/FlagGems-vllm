# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Public API benchmarks; component timings are not end-to-end model gains."""
import pytest
import torch

import flaggems_vllm

from . import base


def torch_mqa_into(q, kv, weights, ks, ke, out, row_shard=False):
    scores = torch.einsum("mhd,nd->mhn", q.float(), kv.float()).relu()
    result = (scores * weights[:, :, None]).sum(1)
    columns = torch.arange(kv.shape[0], device=q.device)
    result.masked_fill_(
        (columns[None, :] < ks[:, None]) | (columns[None, :] >= ke[:, None]), -torch.inf
    )
    out.copy_(result)
    return out


class MQAIndexBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "query_rows, key_rows, heads, head_dim"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(16, 1024, 64, 128), (64, 4096, 64, 128), (64, 16384, 64, 128)]

    def get_input_iter(self, dtype):
        for m, n, h, d in self.shapes:
            q = torch.randn(m, h, d, dtype=dtype, device=self.device)
            kv = torch.randn(n, d, dtype=dtype, device=self.device)
            weights = torch.randn(m, h, device=self.device)
            ks = torch.zeros(m, dtype=torch.int32, device=self.device)
            ke = torch.full((m,), n, dtype=torch.int32, device=self.device)
            out = torch.empty(m, n, device=self.device)
            yield q, kv, weights, ks, ke, {"out": out}


@pytest.mark.mqa_logits
def test_mqa_public_api():
    benchmark = MQAIndexBenchmark(
        "mqa_logits",
        torch_mqa_into,
        [torch.bfloat16],
        gems_op=flaggems_vllm.mqa_logits,
    )
    benchmark.run()
