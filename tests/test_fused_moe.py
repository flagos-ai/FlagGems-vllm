# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest
import torch

moe = importlib.import_module("flaggems_vllm.ops.fused_moe")
pytestmark = [
    pytest.mark.fused_moe,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def literal_quant(x):
    scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-10).float() / 127
    return (x.float() / scale).round().clamp(-128, 127).to(torch.int8), scale


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(0, 384), (1, 127), (7, 384), (32, 768), (64, 6144)])
@pytest.mark.fused_moe
def test_quant(dtype, shape):
    torch.manual_seed(43)
    x = torch.randn(shape, device="cuda", dtype=dtype) * 8
    q, s = moe._int8_quantize_per_token_triton(x)
    refq, refs = literal_quant(x)
    assert torch.equal(q, refq)
    torch.testing.assert_close(s, refs, rtol=0, atol=0)


def test_quant_nan_halfway_graph():
    x = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [127.0, 0.5, 1.5, 2.5], [float("nan"), 1.0, -2.0, 3.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, s = moe._int8_quantize_per_token_triton(x)
    rq, rs = literal_quant(x)
    assert torch.equal(q, rq)
    torch.testing.assert_close(s, rs, rtol=0, atol=0, equal_nan=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        q, s = moe._int8_quantize_per_token_triton(x)
    x.copy_(torch.randn_like(x))
    g.replay()
    rq, rs = literal_quant(x)
    assert torch.equal(q, rq)
    torch.testing.assert_close(s, rs, rtol=0, atol=0)


def test_quant_is_independent_of_flaggems_registration():
    """The kernel owns the arithmetic, so operator overrides cannot change it.

    The previous eager chain dispatched through abs/amax/clamp/div/round, which
    meant a full, partial, or absent FlagGems registration each produced
    different results.  Nothing here may depend on that registration.
    """
    import flag_gems

    # 52 and the half-way values below are the cases that used to diverge
    # between the FlagGems and plain-Torch division paths.
    x = torch.tensor(
        [
            [52.0, 26.0, 13.0, 0.0],
            [127.0, 0.5, 1.5, 2.5],
            [float("nan"), 1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    baseline = moe._int8_quantize_per_token_triton(x)
    registrations = (
        ["abs", "amax", "clamp", "true_divide", "round"],  # full chain
        ["abs"],  # partial: used to be rejected outright
        [],  # empty include registers every operator
    )
    for include in registrations:
        with flag_gems.use_gems(include=include):
            q, s = moe._int8_quantize_per_token_triton(x)
        assert torch.equal(q, baseline[0]), include
        torch.testing.assert_close(s, baseline[1], rtol=0, atol=0, equal_nan=True)

    # The row with a NaN is what used to swing: FlagGems' amax drops NaN and
    # produced a finite scale, while PyTorch's propagates it.  The kernel
    # reduces NaN itself, so the poison stays visible in the scale.
    assert torch.isnan(baseline[1][2]).item()
    with flag_gems.use_gems(include=["amax"]):
        assert not torch.isnan(x[2].abs().amax(-1)).item()


@pytest.mark.parametrize("k", [8192, 8193, 16384])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_quant_wide_rows(k, dtype):
    """Rows wider than one block take the two-pass kernel."""
    torch.manual_seed(13)
    x = torch.randn((3, k), device="cuda", dtype=dtype) * 8
    q, s = moe._int8_quantize_per_token_triton(x)
    rq, rs = literal_quant(x)
    assert torch.equal(q, rq)
    torch.testing.assert_close(s, rs, rtol=0, atol=0)


def test_quant_single_and_two_pass_kernels_agree(monkeypatch):
    torch.manual_seed(17)
    x = torch.randn((4, 4096), device="cuda", dtype=torch.bfloat16) * 8
    single = moe._int8_quantize_per_token_triton(x)
    # Force the same data through the wide kernel.
    monkeypatch.setattr(moe, "_INT8_QUANT_MAX_SINGLE_PASS_K", 1024)
    wide = moe._int8_quantize_per_token_triton(x)
    assert torch.equal(single[0], wide[0])
    torch.testing.assert_close(single[1], wide[1], rtol=0, atol=0)


def test_quant_wide_row_keeps_nan_scale():
    x = torch.randn((2, 16384), device="cuda", dtype=torch.bfloat16)
    x[0, 9999] = float("nan")
    _, s = moe._int8_quantize_per_token_triton(x)
    assert torch.isnan(s[0]).item()
    assert not torch.isnan(s[1]).item()


def test_quant_offsets_promote_past_int32():
    """A large base tensor makes a small view's row offset exceed int32."""
    rows, k = 3, 384
    stride = 2**30  # row offsets reach 2 * 2**30 > 2**31 - 1
    # Reaching the int32 boundary at 2 bytes per element needs ~4 GiB; there
    # is no smaller buffer that can produce the overflow.
    needed = (stride * (rows - 1) + k) * 2
    free, _ = torch.cuda.mem_get_info()
    if free < needed + (1 << 30):
        pytest.skip(f"needs ~{needed >> 30} GiB free, has {free >> 30} GiB")
    base = torch.zeros(stride * (rows - 1) + k, device="cuda", dtype=torch.bfloat16)
    view = base.as_strided((rows, k), (stride, 1))
    view.copy_(torch.randn((rows, k), device="cuda", dtype=torch.bfloat16) * 8)
    q, s = moe._int8_quantize_per_token_triton(view)
    rq, rs = literal_quant(view)
    assert torch.equal(q, rq)
    torch.testing.assert_close(s, rs, rtol=0, atol=0)
