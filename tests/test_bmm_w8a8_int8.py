# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm

from .accuracy_utils import gems_assert_close

pytestmark = [
    pytest.mark.bmm_w8a8_int8,
    pytest.mark.skipif(
        "bmm_w8a8_int8" not in vars(flaggems_vllm),
        reason="backend does not export channel-wise INT8 BMM",
    ),
]


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1, 1), (2, 33, 65, 129), (4, 64, 128, 256), (2, 129, 17, 257)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("scale_rank", [2, 3])
@pytest.mark.parametrize("use_out", [False, True])
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
    reference = torch.bmm(activation.double(), weight.double())
    reference *= activation_scale.double()[:, :, None]
    reference *= weight_scale.double()[:, None, :]
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


@pytest.mark.parametrize(
    "shape", [(0, 5, 7, 32), (2, 0, 7, 32), (2, 5, 0, 32), (2, 5, 7, 0)]
)
def test_channel_int8_bmm_empty(shape):
    batch, rows, columns, reduction = shape
    activation = torch.empty(
        shape[:2] + (reduction,), device=flaggems_vllm.device, dtype=torch.int8
    )
    weight = torch.empty(
        (batch, columns, reduction), device=activation.device, dtype=torch.int8
    ).transpose(1, 2)
    activation_scale = torch.ones((batch, rows), device=activation.device)
    weight_scale = torch.ones((batch, columns), device=activation.device)
    actual = flaggems_vllm.bmm_w8a8_int8(
        activation, weight, activation_scale, weight_scale
    )
    reference = torch.zeros(
        (batch, rows, columns), device=activation.device, dtype=torch.bfloat16
    )
    gems_assert_close(actual, reference, torch.bfloat16)


@pytest.mark.parametrize(
    "invalid_input", ["batch", "weight_layout", "scale_shape", "scale_dtype"]
)
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
