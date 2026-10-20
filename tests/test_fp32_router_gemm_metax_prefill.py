# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import flaggems_vllm as fgv
import pytest
import torch
from flaggems_vllm.runtime import device

pytestmark = pytest.mark.skipif(
    device.vendor_name != "metax", reason="MetaX prefill path"
)


def check(actual, expected):
    a = actual.detach().cpu().float()
    r = expected.detach().cpu().float()
    d = a - r
    assert torch.isfinite(a).all()
    assert (d.norm() / r.norm().clamp_min(1e-20)).item() <= 2e-6
    assert (d.abs().max() / r.abs().max().clamp_min(1e-20)).item() <= 5e-6


@pytest.mark.parametrize("scale", [0.001, 0.2, 1.0])
def test_prefill_fp32_reference(scale):
    torch.manual_seed(560)
    x = (torch.randn(8192, 6144, device="cuda") * scale).bfloat16()
    w = (torch.randn(128, 6144, device="cuda") * scale).float()
    result = fgv.fp32_router_gemm(x, w)
    assert result.dtype == torch.float32 and result.shape == (8192, 128)
    check(result, torch.nn.functional.linear(x.cpu().float(), w.cpu()))


@pytest.mark.parametrize("m", [1, 4, 16, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_existing_small_path(m, dtype):
    torch.manual_seed(560)
    x = torch.randn(m, 6144, device="cuda").to(dtype)
    w = torch.randn(128, 6144, device="cuda")
    result = fgv.fp32_router_gemm(x, w)
    check(result, torch.nn.functional.linear(x.cpu().float(), w.cpu()))


@pytest.mark.parametrize(
    "m,dtype",
    [
        (33, torch.bfloat16),
        (8191, torch.bfloat16),
        (8192, torch.float32),
        (8193, torch.bfloat16),
    ],
)
def test_unsupported_scope_stays_explicit(m, dtype):
    x = torch.empty(m, 6144, device="cuda", dtype=dtype)
    w = torch.empty(128, 6144, device="cuda", dtype=torch.float32)
    with pytest.raises(NotImplementedError):
        fgv.fp32_router_gemm(x, w)


def test_prefill_strides_are_rejected():
    x = torch.empty(8192, 12288, device="cuda", dtype=torch.bfloat16)[:, ::2]
    w = torch.empty(128, 6144, device="cuda", dtype=torch.float32)
    with pytest.raises(NotImplementedError):
        fgv.fp32_router_gemm(x, w)


def test_prefill_graph_changed_inputs():
    torch.manual_seed(560)
    x = torch.randn(8192, 6144, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 6144, device="cuda")
    for _ in range(3):
        fgv.fp32_router_gemm(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fgv.fp32_router_gemm(x, w)
    x.mul_(0.25)
    graph.replay()
    torch.cuda.synchronize()
    check(result, torch.nn.functional.linear(x.cpu().float(), w.cpu()))


def test_prefill_nonfinite_and_input_preservation():
    x = torch.zeros(8192, 6144, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(128, 6144, device="cuda", dtype=torch.float32)
    x[0, 0] = float("nan")
    x[1, 0] = float("inf")
    result = fgv.fp32_router_gemm(x, w)
    assert torch.isnan(result[0]).all()
    assert torch.isposinf(result[1]).all()
    assert torch.equal(result[2:], torch.zeros_like(result[2:]))
    assert torch.isnan(x[0, 0]) and torch.isposinf(x[1, 0])


def test_empty():
    x = torch.empty(0, 6144, device="cuda", dtype=torch.bfloat16)
    w = torch.empty(128, 6144, device="cuda", dtype=torch.float32)
    assert fgv.fp32_router_gemm(x, w).shape == (0, 128)
