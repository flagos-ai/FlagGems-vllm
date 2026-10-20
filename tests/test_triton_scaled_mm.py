import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.triton_scaled_mm import triton_scaled_mm

from . import accuracy_utils as utils
from .accuracy_utils import gems_assert_close

device = flaggems_vllm.device

# Check if vLLM is available and has triton_scaled_mm
try:
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm as vllm_triton_scaled_mm,
    )

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    vllm_triton_scaled_mm = None


def torch_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a * out
    out = scale_b.T * out
    out = out.to(out_dtype)
    if bias is not None:
        out = out + bias

    return out


def get_8bit_types():
    types = [torch.int8]
    # # Check if fp8 is supported by trying to create a tensor
    # try:
    #     _ = torch.empty(1, dtype=torch.float8_e4m3fn, device=device)
    #     types.append(torch.float8_e4m3fn)
    # except (RuntimeError, AttributeError):
    #     pass
    return types


MNK_FACTORS = [
    (1, 256, 128),
    (33, 256, 496),
    (64, 971, 1024),
    (64, 20486, 128),
    (512, 256, 496),
    (512, 20486, 1024),
]


@pytest.mark.triton_scaled_mm
@pytest.mark.parametrize("M,N,K", MNK_FACTORS)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("in_dtype", get_8bit_types())
@pytest.mark.parametrize("use_scalar_scale_a", [True, False])
@pytest.mark.parametrize("use_scalar_scale_b", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_scaled_mm(
    M, N, K, in_dtype, out_dtype, use_scalar_scale_a, use_scalar_scale_b, use_bias
):
    is_floating_point_type = lambda t: torch.tensor([1, 1], dtype=t).is_floating_point()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # NOTE: There are cases, where if the matrix is large enough, an output
    # like 65504.4 can be produced, and can easily turn into inf when
    # multiplied when using float16/bfloat16.  This means one function, e.g.,
    # testing function, and another function, e.g. golden function, can
    # produce a non-inf value while the other produces an inf value, and
    # will cause assert_close/allclose to fail, even though if overflow
    # wouldn't have occurred, the values would have been "close."
    #
    # So, the values here are kept small enough to avoid this situation.
    if is_floating_point_type(in_dtype):
        a = (0.25 * torch.rand((M, K), dtype=torch.float32, device=device)).to(in_dtype)
        b = (0.25 * torch.rand((K, N), dtype=torch.float32, device=device)).to(in_dtype)
    else:
        a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device=device)
        b = torch.randint(-32, 32, (K, N), dtype=in_dtype, device=device)

    if use_scalar_scale_a:
        scale_a = torch.rand((1, 1), device=device)
    else:
        scale_a = 0.25 * torch.rand((M, 1), device=device)

    if use_scalar_scale_b:
        scale_b = torch.rand((1, 1), device=device)
    else:
        scale_b = 0.25 * torch.rand((N, 1), device=device)

    bias = None
    if use_bias:
        bias = torch.rand((N,), device=device, dtype=out_dtype)

    c_check = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    c_actual = torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    gems_assert_close(c_check, c_actual, out_dtype, reduce_dim=1, atol=1e-1)


# ============================================================================
# Tests comparing against vLLM triton_scaled_mm implementation
# ============================================================================


@pytest.mark.triton_scaled_mm
@pytest.mark.parametrize("M,N,K", MNK_FACTORS)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("in_dtype", get_8bit_types())
@pytest.mark.parametrize("use_scalar_scale_a", [True, False])
@pytest.mark.parametrize("use_scalar_scale_b", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
def test_triton_scaled_mm_vs_vllm(
    M, N, K, in_dtype, out_dtype, use_scalar_scale_a, use_scalar_scale_b, use_bias
):
    """Test triton_scaled_mm accuracy against vLLM implementation."""
    is_floating_point_type = lambda t: torch.tensor([1, 1], dtype=t).is_floating_point()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # NOTE: There are cases, where if the matrix is large enough, an output
    # like 65504.4 can be produced, and can easily turn into inf when
    # multiplied when using float16/bfloat16.  This means one function, e.g.,
    # testing function, and another function, e.g. golden function, can
    # produce a non-inf value while the other produces an inf value, and
    # will cause assert_close/allclose to fail, even though if overflow
    # wouldn't have occurred, the values would have been "close."
    #
    # So, the values here are kept small enough to avoid this situation.
    if is_floating_point_type(in_dtype):
        a = (0.25 * torch.rand((M, K), dtype=torch.float32, device=device)).to(in_dtype)
        b = (0.25 * torch.rand((K, N), dtype=torch.float32, device=device)).to(in_dtype)
    else:
        a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device=device)
        b = torch.randint(-32, 32, (K, N), dtype=in_dtype, device=device)

    if use_scalar_scale_a:
        scale_a = torch.rand((1, 1), device=device)
    else:
        scale_a = 0.25 * torch.rand((M, 1), device=device)

    if use_scalar_scale_b:
        scale_b = torch.rand((1, 1), device=device)
    else:
        scale_b = 0.25 * torch.rand((N, 1), device=device)

    bias = None
    if use_bias:
        bias = torch.rand((N,), device=device, dtype=out_dtype)

    # Run vLLM reference
    ref_output = vllm_triton_scaled_mm(
        a.clone(), b.clone(), scale_a.clone(), scale_b.clone(), out_dtype, bias
    )
    ref_output = utils.to_reference(ref_output)

    # Run FlagGems implementation
    res_output = triton_scaled_mm(
        a.clone(), b.clone(), scale_a.clone(), scale_b.clone(), out_dtype, bias
    )

    gems_assert_close(res_output, ref_output, out_dtype, reduce_dim=1, atol=1e-1)


requires_channel_int8_bmm = pytest.mark.skipif(
    "bmm_w8a8_int8" not in vars(flaggems_vllm),
    reason="backend does not export channel-wise INT8 BMM",
)


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1, 1), (2, 33, 65, 129), (4, 64, 128, 256), (2, 129, 17, 257)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("scale_rank", [2, 3])
@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
def test_channel_int8_bmm(shape, dtype, scale_rank, use_out):
    torch.manual_seed(0)
    batch, rows, columns, reduction = shape
    activation = torch.randint(
        -128,
        128,
        (batch, rows, reduction),
        device=flaggems_vllm.device,
        dtype=torch.int8,
    )
    weight = torch.randint(
        -128,
        128,
        (batch, columns, reduction),
        device=activation.device,
        dtype=torch.int8,
    ).transpose(1, 2)
    activation_scale = torch.rand((batch, rows), device=activation.device) / 127
    weight_scale = torch.rand((batch, columns), device=activation.device) / 127
    reference = torch.stack(
        [
            torch_scaled_mm(
                activation[index],
                weight[index],
                activation_scale[index, :, None],
                weight_scale[index, :, None],
                torch.float32,
            )
            for index in range(batch)
        ]
    )
    activation_scale = (
        activation_scale[:, :, None] if scale_rank == 3 else activation_scale
    )
    weight_scale = weight_scale[:, None, :] if scale_rank == 3 else weight_scale
    output = (
        torch.empty((batch, rows, columns), device=activation.device, dtype=dtype)
        if use_out
        else None
    )
    actual = flaggems_vllm.bmm_w8a8_int8(
        activation, weight, activation_scale, weight_scale, out_dtype=dtype, out=output
    )
    gems_assert_close(actual, reference.to(dtype), dtype, reduce_dim=reduction)


@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize(
    "shape", [(0, 5, 7, 32), (2, 0, 7, 32), (2, 5, 0, 32), (2, 5, 7, 0)]
)
@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
def test_channel_int8_bmm_empty(shape, use_out):
    batch, rows, columns, reduction = shape
    activation = torch.empty(
        shape[:2] + (reduction,), device=flaggems_vllm.device, dtype=torch.int8
    )
    weight = torch.empty(
        (batch, columns, reduction), device=activation.device, dtype=torch.int8
    ).transpose(1, 2)
    activation_scale = torch.ones((batch, rows), device=activation.device)
    weight_scale = torch.ones((batch, columns), device=activation.device)
    output = (
        torch.empty(
            (batch, rows, columns), device=activation.device, dtype=torch.bfloat16
        )
        if use_out
        else None
    )
    actual = flaggems_vllm.bmm_w8a8_int8(
        activation, weight, activation_scale, weight_scale, out=output
    )
    reference = torch.zeros(
        (batch, rows, columns), device=activation.device, dtype=torch.bfloat16
    )
    gems_assert_close(actual, reference, torch.bfloat16)


@pytest.mark.parametrize(
    "invalid_input", ["batch", "weight_layout", "scale_shape", "scale_dtype"]
)
@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
def test_channel_int8_bmm_input_contract(invalid_input):
    activation = torch.empty((2, 8, 16), device=flaggems_vllm.device, dtype=torch.int8)
    weight = torch.empty(
        (2, 32, 16), device=activation.device, dtype=torch.int8
    ).transpose(1, 2)
    activation_scale = torch.ones((2, 8), device=activation.device)
    weight_scale = torch.ones((2, 32), device=activation.device)
    error_type = ValueError
    if invalid_input == "batch":
        weight = weight[:1]
    elif invalid_input == "weight_layout":
        weight = weight.contiguous()
    elif invalid_input == "scale_shape":
        activation_scale = activation_scale[:, :7]
    else:
        weight_scale = weight_scale.to(torch.int8)
        error_type = TypeError
    with pytest.raises(error_type):
        flaggems_vllm.bmm_w8a8_int8(activation, weight, activation_scale, weight_scale)
