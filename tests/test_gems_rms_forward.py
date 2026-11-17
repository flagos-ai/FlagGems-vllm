# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm


@pytest.mark.gems_rms_forward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 129), (2, 4097), (2, 3, 17)])
def test_gems_rms_forward(dtype, shape):
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    normalized_shape = shape[1:]
    weight = torch.rand(normalized_shape, dtype=dtype, device=x.device)
    eps = 1e-5
    original = x.clone()
    dims = tuple(range(1, x.ndim))
    x32 = x.float()
    expected = (x32 * torch.rsqrt(x32.square().mean(dims, keepdim=True) + eps)).to(
        dtype
    )
    expected = expected * weight

    result = flaggems_vllm.gems_rms_forward(x, None, weight, eps)

    assert flaggems_vllm.gems_rms_forward is flaggems_vllm.ops.gems_rms_forward
    assert result.data_ptr() != x.data_ptr()
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.gems_rms_forward
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gems_rms_forward_rounds_before_weight(dtype):
    x = torch.ones((2, 17), dtype=dtype, device=flaggems_vllm.device)
    weight = torch.full((17,), 7, dtype=dtype, device=x.device)
    expected = torch.full_like(x, 1 / 3) * weight
    actual = flaggems_vllm.gems_rms_forward(x, None, weight, 8.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gems_rms_forward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gems_rms_forward_with_residual(dtype):
    x = torch.randn((2, 129), dtype=dtype, device=flaggems_vllm.device)
    residual = torch.randn_like(x)
    weight = torch.rand((129,), dtype=dtype, device=x.device)
    eps = 1e-5
    summed = x.float() + residual.float()
    expected = (
        summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + eps) * weight
    ).to(dtype)

    output, residual_out = flaggems_vllm.gems_rms_forward(x, residual, weight, eps)

    assert output.data_ptr() == x.data_ptr()
    assert residual_out.data_ptr() == residual.data_ptr()
    torch.testing.assert_close(x, expected)
    torch.testing.assert_close(residual, summed.to(dtype), rtol=0, atol=0)
