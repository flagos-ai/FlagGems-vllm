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

op = flaggems_vllm.hy4_flash_rotate_indexer


def make_inputs(shape, dtype=torch.bfloat16, axes=False, interleaved=False):
    rows, heads, rope_dim = shape
    x = torch.randn(rows, heads, 128, device="cuda", dtype=dtype)
    freq = 1.0 / (
        10000.0 ** (torch.arange(0, rope_dim, 2, device="cuda").float() / rope_dim)
    )
    parameters = {}
    if axes:
        axis_shape = (3, rows)
        parameters["mrope_interleaved"] = interleaved
        parameters["mrope_section"] = (rope_dim // 2 - 20, 10, 10)
    else:
        axis_shape = (rows,)
    positions = torch.randint(0, 204800, axis_shape, device="cuda")
    return x, positions, rope_dim, parameters, freq


def reference(x, positions, rope_dim, parameters, inverse_freq):
    half = rope_dim // 2
    if positions.ndim == 1:
        angles = positions.float()[:, None] * inverse_freq
    else:
        a, b, c = parameters["mrope_section"]
        lanes = torch.arange(half, device=x.device)
        if parameters["mrope_interleaved"]:
            axis = torch.where((lanes % 3 == 1) & (lanes < b * 3), 1, 0)
            axis = torch.where((lanes % 3 == 2) & (lanes < c * 3), 2, axis)
        else:
            axis = torch.where(lanes >= a + b, 2, (lanes >= a).int())
        angles = positions[axis, :].transpose(0, 1).float() * inverse_freq
    c = angles.cos().unsqueeze(1)
    s = angles.sin().unsqueeze(1)
    left = x[..., -rope_dim::2].float()
    right = x[..., -rope_dim + 1 :: 2].float()
    return torch.cat(
        (
            x[..., :-rope_dim],
            (left * c - right * s).to(x.dtype),
            (right * c + left * s).to(x.dtype),
        ),
        dim=-1,
    )


@pytest.mark.parametrize("rows", [0, 1, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "axes,interleaved", [(False, False), (True, False), (True, True)]
)
@pytest.mark.hy4_flash_rotate_indexer
def test_indexer_rope(rows, dtype, axes, interleaved):
    args = make_inputs((rows, 2, 64), dtype, axes, interleaved)
    torch.testing.assert_close(op(*args), reference(*args))


@pytest.mark.hy4_flash_rotate_indexer
def test_indexer_rope_uses_noop_for_zero_dim():
    x, positions, _, parameters, inverse_freq = make_inputs((1, 1, 64))
    assert op(x, positions, 0, parameters, inverse_freq) is x
