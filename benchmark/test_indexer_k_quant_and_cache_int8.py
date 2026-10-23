# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_vllm as gems

from . import base

pytestmark = pytest.mark.skipif(
    gems.vendor_name != "nvidia", reason="NVIDIA CUDA required"
)


def torch_cache(k, cache, slots, group, scale_fmt="ue8m0"):
    # Benchmark workload uses consecutive slots covering full pages.
    values = k.float()
    scales = torch.exp2(
        torch.ceil(torch.log2(values.abs().amax(-1).clamp_min(1e-4) / 127))
    )
    q = (values / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
    page = cache.shape[1]
    flat = cache.view(cache.shape[0], -1)
    flat[:, : page * 128].copy_(q.view(cache.shape[0], -1).view(torch.uint8))
    flat[:, page * 128 :].view(torch.float32).copy_(scales.view(-1, page))


class CacheBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [(64, 128), (256, 128), (4096, 128)]
    DEFAULT_SHAPE_DESC = "key rows, head_dim; page=64"

    def set_more_shapes(self):
        return []


@pytest.mark.indexer_k_quant_and_cache_int8
def test_int8_cache():
    def inputs(shape, dtype, device):
        rows, d = shape
        yield (
            torch.randn(rows, d, dtype=dtype, device=device),
            torch.empty(rows // 64, 64, 132, dtype=torch.uint8, device=device),
            torch.arange(rows, device=device),
            128,
            {"scale_fmt": "ue8m0"},
        )

    CacheBenchmark(
        op_name="indexer_k_quant_and_cache_int8",
        input_fn=inputs,
        torch_op=torch_cache,
        gems_op=gems.indexer_k_quant_and_cache_int8,
        dtypes=[torch.bfloat16],
    ).run()
