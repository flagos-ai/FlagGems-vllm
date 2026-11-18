# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.hy4_flash_engram_gate,
    pytest.mark.skipif(
        not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
        reason="Hy4 Flash Engram requires NVIDIA CUDA",
    ),
]
op = flaggems_vllm.hy4_flash_engram_gate


def make_inputs(shape, dtype=torch.bfloat16):
    m, n = shape
    a = torch.randn((m, n), device="cuda", dtype=dtype)
    b = torch.randn((m, 2 * n), device="cuda", dtype=dtype)[:, ::2]
    return a, b


def reference(key, query):
    gate = (key * query).sum(-1) * (key.shape[-1] ** -0.5)
    return (gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(0, 1), (7, 131), (17, 4096)])
@pytest.mark.hy4_flash_engram_gate
def test_engram_gate(shape, dtype):
    a, b = make_inputs(shape, dtype)
    if a.numel():
        a[0].zero_()
        a[1].fill_(-2)
        b[1].fill_(1)
    tol = 3e-6 if dtype == torch.float32 else 0.004
    torch.testing.assert_close(op(a, b), reference(a, b), atol=tol, rtol=tol)
    if a.numel():
        assert op(a, b)[0].item() == 0.5


@pytest.mark.hy4_flash_engram_gate
def test_engram_gate_rejects_incompatible_shape():
    a, b = make_inputs((4, 9))
    with pytest.raises(ValueError, match="matching"):
        op(a, b[:, :-1])
