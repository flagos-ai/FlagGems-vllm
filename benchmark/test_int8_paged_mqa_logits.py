# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_vllm as gems

from . import base

pytestmark = pytest.mark.skipif(
    gems.vendor_name != "nvidia", reason="NVIDIA CUDA required"
)


def torch_paged(q, cache, weights, lengths, tables, schedule, capacity, clean):
    rows, heads, d = q[0].shape
    page = cache.shape[1]
    flat = cache.view(cache.shape[0], -1)
    keys = flat[:, : page * d].view(torch.int8).reshape(-1, d)
    scales = flat[:, page * d :].view(torch.float32).reshape(-1)
    dots = torch._int_mm(q[0].view(rows * heads, d), keys.T.contiguous()).float()
    return (
        (dots.view(rows, heads, capacity) * scales[None, None, :]).relu()
        * weights[:, :, None]
    ).sum(1)


class PagedBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [(1, 64, 32), (7, 4096, 32), (32, 4096, 32)]
    DEFAULT_SHAPE_DESC = "query rows, context capacity, heads; page=64, D=128"

    def set_more_shapes(self):
        return []


@pytest.mark.int8_paged_mqa_logits
def test_int8_decode():
    def inputs(shape, dtype, device):
        rows, capacity, heads = shape
        k = torch.randn(capacity, 128, dtype=torch.bfloat16, device=device)
        cache = torch.empty(capacity // 64, 64, 132, dtype=torch.uint8, device=device)
        gems.indexer_k_quant_and_cache_int8(
            k, cache, torch.arange(capacity, device=device), 128, "ue8m0"
        )
        yield (
            (
                torch.randint(
                    -127, 128, (rows, heads, 128), dtype=torch.int8, device=device
                ),
                None,
            ),
            cache.unsqueeze(2),
            torch.randn(rows, heads, device=device),
            torch.full((rows,), capacity, dtype=torch.int32, device=device),
            torch.arange(capacity // 64, dtype=torch.int32, device=device).repeat(
                rows, 1
            ),
            None,
            capacity,
            False,
        )

    PagedBenchmark(
        op_name="int8_paged_mqa_logits",
        input_fn=inputs,
        torch_op=torch_paged,
        gems_op=gems.int8_paged_mqa_logits,
        dtypes=[torch.int8],
    ).run()
