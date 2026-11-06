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


def literal_oai(x):
    gate, up = x.float().chunk(2, dim=-1)
    gate = gate.clamp(max=7.0)
    up = up.clamp(-7.0, 7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(x.dtype)


def literal_quant(x):
    scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-10).float() / 127
    return (x.float() / scale).round().clamp(-128, 127).to(torch.int8), scale


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(0, 384), (1, 127), (7, 384), (32, 768), (64, 6144)])
@pytest.mark.fused_moe
def test_quant_and_oai_quant(dtype, shape):
    torch.manual_seed(43)
    x = torch.randn(shape, device="cuda", dtype=dtype) * 8
    q, s = moe._int8_quantize_per_token_triton(x)
    refq, refs = literal_quant(x)
    assert torch.equal(q, refq)
    torch.testing.assert_close(s, refs, rtol=0, atol=0)
    packed = torch.randn((shape[0], 2 * shape[1]), device="cuda", dtype=dtype) * 8
    activated = torch.empty(shape, device="cuda", dtype=dtype)
    q, s = moe._swigluoai_quantize_per_token_triton(packed, activated, 1.702, 1.0, 7.0)
    expected = torch.empty_like(activated)
    if shape[0]:
        moe.apply_moe_activation(
            moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            expected,
            packed,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
        )
    torch.testing.assert_close(activated, expected, rtol=0, atol=0)
    refq, refs = literal_quant(expected)
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


def test_oai_parameter_validation():
    x = torch.randn((2, 768), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((2, 384), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        moe.apply_moe_activation(moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE, out, x)
    with pytest.raises(ValueError):
        moe.apply_moe_activation(
            moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            out,
            x,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=-1.0,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,i", [(1, 384), (7, 768), (32, 1536)])
@pytest.mark.parametrize("mode", ["off", "quant_oai"])
def test_w8a8_oai_experts(dtype, m, i, mode, monkeypatch):
    monkeypatch.setattr(moe, "_W8A8_QUANT_FUSION_MODE", mode)
    torch.manual_seed(47)
    e, h, k = 4, 128, 2
    x = torch.randn((m, h), device="cuda", dtype=dtype)
    w1 = torch.randint(-64, 64, (e, 2 * i, h), device="cuda", dtype=torch.int8)
    w2 = torch.randint(-64, 64, (e, h, i), device="cuda", dtype=torch.int8)
    s1 = torch.rand((e, 2 * i), device="cuda") * 0.001 + 0.001
    s2 = torch.rand((e, h), device="cuda") * 0.001 + 0.001
    ids = (torch.arange(m * k, device="cuda").view(m, k) % e).int()
    weights = torch.rand((m, k), device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    actual = moe.fused_experts_impl(
        x,
        w1,
        w2,
        weights,
        ids,
        activation="swigluoai_uninterleave",
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=s1,
        w2_scale=s2,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    if m == 1 and mode == "off":
        kwargs = dict(
            activation="swigluoai_uninterleave",
            use_int8_w8a8=True,
            per_channel_quant=True,
            w1_scale=s1,
            w2_scale=s2,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
        )
        out = moe.outplace_fused_experts(x, w1, w2, weights, ids, **kwargs)
        torch.testing.assert_close(out, actual, rtol=0, atol=0)
        in_place = x.clone()
        assert (
            moe.inplace_fused_experts(in_place, w1, w2, weights, ids, **kwargs) is None
        )
        torch.testing.assert_close(in_place, actual, rtol=0, atol=0)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        qx, sx = literal_quant(x)
        contributions = []
        for t in range(m):
            row = []
            for rank in range(k):
                expert = int(ids[t, rank])
                up = (
                    (qx[t : t + 1].float() @ w1[expert].float().t())
                    * sx[t]
                    * s1[expert]
                )
                activated = literal_oai(up.to(dtype))
                qa, sa = literal_quant(activated)
                down = (qa.float() @ w2[expert].float().t()) * sa * s2[expert]
                row.append((down * weights[t, rank]).to(dtype))
            contributions.append(torch.stack(row).float().sum(0).to(dtype))
        expected = torch.cat(contributions)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    torch.testing.assert_close(
        actual,
        expected,
        atol=0.0005 if dtype == torch.float16 else 0.004,
        rtol=0.003 if dtype == torch.float16 else 0.016,
    )


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


@pytest.mark.parametrize("column", [0, 384])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_oai_nan_survives_clamp(column, dtype):
    """A NaN gate or up value must not be replaced by the clamp bound."""
    packed = torch.ones((2, 768), device="cuda", dtype=dtype)
    packed[0, column] = float("nan")
    activated = torch.empty((2, 384), device="cuda", dtype=dtype)
    moe._swigluoai_quantize_per_token_triton(packed, activated, 1.702, 1.0, 7.0)
    torch.testing.assert_close(
        activated, literal_oai(packed), rtol=0, atol=0, equal_nan=True
    )
    assert torch.isnan(activated[0, 0])
    assert not torch.isnan(activated[1]).any()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_uninterleave_nan_survives_clamp(dtype):
    packed = torch.ones((2, 768), device="cuda", dtype=dtype)
    packed[0, 0] = float("nan")
    packed[1, 384] = float("nan")
    out = torch.empty((2, 384), device="cuda", dtype=dtype)
    moe.apply_moe_activation(
        moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        out,
        packed,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    torch.testing.assert_close(out, literal_oai(packed), rtol=0, atol=0, equal_nan=True)
    assert torch.isnan(out[0, 0]) and torch.isnan(out[1, 0])


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


def test_oai_scalars_do_not_specialize_the_kernel():
    """Scalars are runtime args, so distinct values must not recompile."""
    kernel = moe._swigluoai_quantize_per_token_kernel
    packed = torch.randn((4, 768), device="cuda", dtype=torch.bfloat16)
    activated = torch.empty((4, 384), device="cuda", dtype=torch.bfloat16)
    moe._swigluoai_quantize_per_token_triton(packed, activated, 1.702, 1.0, 7.0)
    device = torch.cuda.current_device()
    compiled = len(kernel.device_caches[device][0])
    for alpha in (1.5, 1.6, 1.8, 1.9):
        moe._swigluoai_quantize_per_token_triton(packed, activated, alpha, 1.0, 7.0)
    assert len(kernel.device_caches[device][0]) == compiled


def test_oai_quant_rejects_mismatched_device_metadata():
    packed = torch.randn((2, 768), device="cuda", dtype=torch.bfloat16)
    activated = torch.empty((2, 384), device="cpu", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        moe._swigluoai_quantize_per_token_triton(packed, activated, 1.702, 1.0, 7.0)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("value", [-1e39, 1e39])
def test_oai_fp32_overflow_rejected_before_tensor_access(index, value):
    params = [1.702, 1.0, 7.0]
    params[index] = value
    # Scalars are checked before any tensor metadata is read, so passing None
    # tensors still produces the scalar error rather than an AttributeError.
    with pytest.raises(ValueError, match="finite FP32"):
        moe._swigluoai_quantize_per_token_triton(None, None, *params)
