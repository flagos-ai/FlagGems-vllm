# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_vllm as gems

from . import base

pytestmark = pytest.mark.skipif(
    gems.vendor_name != "nvidia", reason="NVIDIA CUDA required"
)


def torch_logits(q, k, scales, weights, starts, ends):
    m, h, d = q.shape
    dots = torch._int_mm(q.view(m * h, d), k.T.contiguous()).float().view(m, h, -1)
    return ((dots * scales[None, None, :]).relu() * weights[:, :, None]).sum(1)


def gems_logits(q, k, scales, weights, starts, ends):
    return gems.int8_mqa_logits((q, None), (k, scales), weights, starts, ends, False)


class IndexerBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [(1, 64, 32), (7, 4096, 32), (128, 4096, 32)]
    DEFAULT_SHAPE_DESC = "query rows, key rows, heads; head_dim=128"

    def set_more_shapes(self):
        return []


@pytest.mark.int8_mqa_logits
def test_int8_prefill():
    def inputs(shape, dtype, device):
        m, n, h = shape
        yield (
            torch.randint(-127, 128, (m, h, 128), dtype=torch.int8, device=device),
            torch.randint(-127, 128, (n, 128), dtype=torch.int8, device=device),
            torch.rand(n, device=device) * 0.01,
            torch.randn(m, h, device=device),
            torch.zeros(m, dtype=torch.int32, device=device),
            torch.full((m,), n, dtype=torch.int32, device=device),
        )

    IndexerBenchmark(
        op_name="int8_mqa_logits",
        input_fn=inputs,
        torch_op=torch_logits,
        gems_op=gems_logits,
        dtypes=[torch.int8],
    ).run()
