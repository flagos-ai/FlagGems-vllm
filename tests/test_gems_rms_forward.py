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
    normalized = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + eps)
    if flaggems_vllm.vendor_name == "ascend":
        normalized = normalized.to(dtype)
    expected = (normalized * weight).to(dtype)

    output, residual_out = flaggems_vllm.gems_rms_forward(x, residual, weight, eps)

    assert output.data_ptr() == x.data_ptr()
    assert residual_out.data_ptr() == residual.data_ptr()
    torch.testing.assert_close(x, expected)
    torch.testing.assert_close(residual, summed.to(dtype), rtol=0, atol=0)


@pytest.mark.gems_rms_forward
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia", reason="Checks NVIDIA residual reload"
)
@pytest.mark.parametrize(
    "dtype,residual_value", [(torch.float16, 2**-11), (torch.bfloat16, 2**-8)]
)
def test_gems_rms_forward_nvidia_reloads_rounded_residual(dtype, residual_value):
    x = torch.ones((2, 8192), dtype=dtype, device=flaggems_vllm.device)
    residual = torch.full_like(x, residual_value)
    weight = torch.ones_like(x[0])
    eps = 1e-6

    # Use values whose sum is rounded to 1 when written back to residual.
    raw_sum = 1 + residual_value
    expected = torch.full_like(x, 1 / (raw_sum**2 + eps) ** 0.5)
    actual, residual_out = flaggems_vllm.gems_rms_forward(x, residual, weight, eps)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(residual_out, torch.ones_like(residual), rtol=0, atol=0)


@pytest.mark.gems_rms_forward
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend", reason="Checks Ascend residual rounding"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 8193])
def test_gems_rms_forward_ascend_rounds_residual_output_before_weight(dtype, width):
    x = torch.ones((2, width), dtype=dtype, device=flaggems_vllm.device)
    residual = torch.zeros_like(x)
    weight = torch.full_like(x[0], 7)
    # 1 / sqrt(1 + 8) rounds differently before and after multiplying by 7.
    expected = torch.full_like(x, 1 / 3) * weight
    actual, residual_out = flaggems_vllm.gems_rms_forward(x, residual, weight, 8.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(residual_out, torch.ones_like(residual), rtol=0, atol=0)


@pytest.mark.gems_rms_forward
def test_gems_rms_forward_rejects_mismatched_residual_shape():
    # Shape validation must run before device checks or kernel dispatch.
    x = torch.empty((2, 128))
    residual = torch.empty((1, 128))
    weight = torch.empty((128,))
    with pytest.raises(ValueError, match="x and residual must have the same shape"):
        flaggems_vllm.gems_rms_forward(x, residual, weight, 1e-5)


@pytest.mark.gems_rms_forward
@pytest.mark.parametrize("with_residual", [False, True], ids=["standalone", "residual"])
@pytest.mark.parametrize(
    "x_shape,weight_shape",
    [((2, 128), (129,)), ((2, 3, 17), (51,)), ((2, 128), (1, 2, 128))],
    ids=["wrong-width", "same-size-wrong-shape", "extra-axis"],
)
def test_gems_rms_forward_rejects_mismatched_weight_shape(
    with_residual, x_shape, weight_shape
):
    # Invalid shapes are rejected before any accelerator dispatch.
    x = torch.empty(x_shape)
    residual = torch.empty_like(x) if with_residual else None
    weight = torch.empty(weight_shape)

    with pytest.raises(
        ValueError, match=r"weight\.shape .* must match the trailing dimensions"
    ) as exc:
        flaggems_vllm.gems_rms_forward(x, residual, weight, 1e-5)

    assert str(x_shape) in str(exc.value)
    assert str(weight_shape) in str(exc.value)


@pytest.mark.gems_rms_forward
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend", reason="Checks Ascend reduction order"
)
@pytest.mark.parametrize("width", [128, 4097, 8193])
def test_gems_rms_forward_ascend_scales_before_reduction(width):
    x = torch.full(
        (2, width), float(2**62), device=flaggems_vllm.device, dtype=torch.float32
    )
    weight = torch.ones_like(x[0])
    # Squaring each value is finite, but the unscaled sum of squares overflows.
    actual = flaggems_vllm.gems_rms_forward(x, None, weight, 1e-5)
    torch.testing.assert_close(actual, torch.ones_like(x))
