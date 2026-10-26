# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_vllm as gems

from . import base

pytestmark = pytest.mark.skipif(
    gems.vendor_name != "nvidia", reason="NVIDIA CUDA required"
)


def torch_quant(x, group_size, *, use_ue8m0=True):
    shaped = x.float().view(-1, group_size)
    scales = shaped.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
    scales = torch.exp2(torch.ceil(torch.log2(scales)))
    return (shaped / scales).round().clamp(-127, 127).to(torch.int8).view_as(x), scales


class QuantBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [(1, 128), (32, 128), (4096, 128)]
    DEFAULT_SHAPE_DESC = "rows, head_dim"

    def set_more_shapes(self):
        return []


@pytest.mark.per_token_group_quant_int8
def test_int8_query_quantization():
    def inputs(shape, dtype, device):
        yield torch.randn(shape, dtype=dtype, device=device), 128

    QuantBenchmark(
        op_name="per_token_group_quant_int8",
        input_fn=inputs,
        torch_op=torch_quant,
        gems_op=gems.per_token_group_quant_int8,
        dtypes=[torch.bfloat16],
    ).run()
