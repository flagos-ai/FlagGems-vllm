# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.fp32_router_gemm,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("m", [0, 1, 3, 4, 8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_fp32_router_gemm(m, dtype, strided):
    torch.manual_seed(29)
    x = torch.randn((m, 12288 if strided else 6144), device="cuda", dtype=dtype)
    w = (
        torch.randn(
            (128, 12288 if strided else 6144), device="cuda", dtype=torch.float32
        )
        * 0.01
    )
    if strided:
        x, w = x[:, ::2], w[:, ::2]
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected = torch.nn.functional.linear(x.float(), w)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    actual = flaggems_vllm.fp32_router_gemm(x, w)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    if m:
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))
        assert torch.equal(actual, flaggems_vllm.fp32_router_gemm(x, w))


def test_fp32_router_graph():
    x = torch.randn((4, 6144), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((128, 6144), device="cuda", dtype=torch.float32) * 0.01
    flaggems_vllm.fp32_router_gemm(x, w)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = flaggems_vllm.fp32_router_gemm(x, w)
    x.mul_(-1)
    g.replay()
    torch.testing.assert_close(
        out, flaggems_vllm.fp32_router_gemm(x, w), rtol=0, atol=0
    )
    with pytest.raises(NotImplementedError):
        flaggems_vllm.fp32_router_gemm(x.repeat(9, 1), w)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.fp32_router_gemm(x, w.bfloat16())
