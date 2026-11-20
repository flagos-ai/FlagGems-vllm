# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.hy4_flash_engram_residual,
    pytest.mark.skipif(
        not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
        reason="Hy4 Flash Engram requires NVIDIA CUDA",
    ),
]
op = flaggems_vllm.hy4_flash_engram_residual


def make_inputs(shape, dtype=torch.bfloat16):
    m, h, n = shape
    hidden = torch.randn(m, h, 2 * n, dtype=dtype, device="cuda")[..., ::2]
    value = torch.randn(m, n, dtype=dtype, device="cuda")
    gate = torch.rand(m, h, dtype=dtype, device="cuda")
    gamma = torch.randn(n, dtype=dtype, device="cuda")
    scales = torch.randn(h, n, dtype=dtype, device="cuda")
    valid = torch.arange(m, device="cuda") % 2 == 1
    return hidden, value, gate, gamma, scales, valid


def reference(hidden, value, gate, gamma, scales, valid):
    x = value.float()
    rms = (x.square().mean(-1, keepdim=True) + torch.finfo(torch.float32).eps).rsqrt()
    signal = gamma.float() * x * rms
    delta = ((gate.float().unsqueeze(-1) + scales.float()) * signal.unsqueeze(1)).to(
        hidden.dtype
    )
    return hidden + delta.masked_fill(~valid[:, None, None], 0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(0, 1, 1), (7, 2, 129), (9, 4, 4096)])
@pytest.mark.hy4_flash_engram_residual
def test_engram_residual(shape, dtype):
    hidden, value, gate, gamma, scales, valid = make_inputs(shape, dtype)
    # Stand-in for the BF16 value projection after the framework's TP reduction.
    value = value + (value * 0.25).to(dtype)
    value[::2] = float("nan")
    args = hidden, value, gate, gamma, scales, valid
    got = op(*args)
    tol = 2e-6 if dtype == torch.float32 else 0.008
    torch.testing.assert_close(got, reference(*args), atol=tol, rtol=tol)
    torch.testing.assert_close(got[::2], hidden[::2], atol=0, rtol=0)
