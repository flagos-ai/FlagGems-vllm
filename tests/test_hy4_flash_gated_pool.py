# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
        reason="Hy4 Flash IndexPool requires NVIDIA CUDA",
    ),
]

op = flaggems_vllm.hy4_flash_gated_pool


def make_inputs(shape, dtype=torch.bfloat16):
    rows, ratio = shape
    values = torch.randn(rows, ratio, 128, dtype=dtype, device="cuda")
    gates = torch.randn_like(values)
    ape = torch.randn(ratio, 128, device="cuda")
    norm_weight = torch.randn(128, device="cuda")
    return values, gates, ape, norm_weight, 1e-6


def reference(values, gates, ape, norm_weight, eps):
    weights = (gates.float() + ape.float()).softmax(dim=1)
    pooled = (values.float() * weights).sum(dim=1)
    return (
        pooled
        * (pooled.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
        * norm_weight
    )


@pytest.mark.parametrize("shape", [(0, 4), (1, 2), (17, 4), (5, 8), (5, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.hy4_flash_gated_pool
def test_gated_pool(shape, dtype):
    args = make_inputs(shape, dtype)
    torch.testing.assert_close(op(*args), reference(*args), atol=2e-5, rtol=2e-5)


@pytest.mark.hy4_flash_gated_pool
def test_gated_pool_rejects_unsupported_ratio():
    with pytest.raises(ValueError, match="shape"):
        op(*make_inputs((1, 3)))
