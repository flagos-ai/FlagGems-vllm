# SPDX-License-Identifier: Apache-2.0
from functools import partial

import pytest
import torch

from flaggems_vllm import qsa_sparse_paged_attention

from . import base
from .qwen4_references import sparse_reference


class SparseBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(r, k) for r in (1, 8, 64) for k in (33, 513, 2051)]
        self.shape_desc = "rows,topk"


def inputs(shape, dtype, device, *, splits):
    rows, topk = shape
    pages = (topk + 15) // 16
    q = torch.randn(rows, 3, 256, device=device, dtype=dtype)
    k = torch.randn(rows * pages, 16, 1, 256, device=device, dtype=dtype)
    table = torch.arange(rows * pages, device=device, dtype=torch.int32).reshape(
        rows, pages
    )
    indices = torch.arange(topk, device=device, dtype=torch.int32).expand(rows, -1)
    requests = torch.arange(rows, device=device, dtype=torch.int32)
    gate = torch.randn_like(q)
    workspace = (
        torch.empty(rows, 3, 8, 256, device=device, dtype=torch.float32),
        torch.empty(rows, 3, 8, device=device, dtype=torch.float32),
        torch.empty(rows, 3, 8, device=device, dtype=torch.float32),
    )
    yield q, k, torch.randn_like(k), indices, table, requests, {
        "gate": gate,
        "split_workspace": workspace if splits > 1 else None,
    }


@pytest.mark.qsa_attention
@pytest.mark.parametrize("splits", [1, 8])
def test_sparse_attention_perf(splits):
    SparseBenchmark(
        input_fn=partial(inputs, splits=splits),
        op_name="qsa_sparse_paged_attention",
        torch_op=sparse_reference,
        gems_op=qsa_sparse_paged_attention,
        dtypes=[torch.bfloat16, torch.float16],
    ).run()
