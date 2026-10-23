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
def test_quant_and_oai_quant(dtype, shape, monkeypatch):
    monkeypatch.setattr(moe, "_int8_quant_dispatch_policy", lambda: False)
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


def test_quant_nan_halfway_graph(monkeypatch):
    monkeypatch.setattr(moe, "_int8_quant_dispatch_policy", lambda: False)
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
    monkeypatch.setattr(moe, "_int8_quant_dispatch_policy", lambda: False)
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


def test_quant_full_and_mixed_dispatch():
    import flag_gems

    x = torch.tensor(
        [[52.0, 26.0, 13.0, 0.0], [float("nan"), 1.0, 2.0, 3.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    with flag_gems.use_gems(include=["abs", "amax", "clamp", "true_divide", "round"]):
        assert moe._int8_quant_dispatch_policy() is True
        q, s = moe._int8_quantize_per_token_triton(x)
        rq, rs = literal_quant(x)
    assert torch.equal(q, rq)
    torch.testing.assert_close(s, rs, rtol=0, atol=0, equal_nan=True)
    with flag_gems.use_gems(include=["abs"]):
        assert moe._int8_quant_dispatch_policy() is None
    assert moe._int8_quant_dispatch_policy() is False


@pytest.fixture
def thead_moe_original_layout(monkeypatch):
    from flaggems_vllm.runtime import device

    if device.vendor_name != "thead":
        pytest.skip("T-Head weight-layout regression")
    thead_moe = importlib.import_module(
        "flaggems_vllm.runtime.backend._thead.ops.fused_moe"
    )
    permute = importlib.import_module("flaggems_vllm.ops.permute_copy")

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("MoE must not copy weights or query free memory")

    monkeypatch.setattr(permute, "permute_copy", unexpected_conversion)
    monkeypatch.setattr(torch.cuda, "mem_get_info", unexpected_conversion)
    return thead_moe


def _thead_moe_inputs(m, dtype, padded=False, hidden=128):
    # E/topk keeps M<=12 on the naive path and M>=13 on the aligned path.
    # I=80 exercises partial K/N tiles. hidden=1024 also covers GEMV at M=8.
    e, intermediate, topk = 512, 80, 10
    x = torch.randn((m, hidden), device="cuda", dtype=dtype)

    def weight(n, k):
        shape = (e, n + 1, k + 16) if padded else (e, n, k)
        storage = torch.randn(shape, device="cuda", dtype=dtype) * (k**-0.5)
        return storage[:, :n, :k]

    w1 = weight(2 * intermediate, hidden)
    w2 = weight(hidden, intermediate)
    ids = (torch.arange(m * topk, device="cuda").view(m, topk) % e).int()
    weights = torch.full((m, topk), 1.0 / topk, device="cuda", dtype=torch.float32)
    return x, w1, w2, weights, ids


def _thead_moe_reference(x, w1, w2, weights, ids):
    # Independent CPU reference; production code never calls this helper.
    x, w1, w2, weights, ids = [t.cpu() for t in (x, w1, w2, weights, ids)]
    result = torch.zeros(x.shape, dtype=torch.float32)
    for rank in range(ids.shape[1]):
        up = torch.einsum("mh,mnh->mn", x.float(), w1[ids[:, rank]].float())
        gate, value = up.chunk(2, dim=-1)
        activated = (torch.nn.functional.silu(gate) * value).to(x.dtype)
        down = torch.einsum("mi,mhi->mh", activated.float(), w2[ids[:, rank]].float())
        result += (down * weights[:, rank, None]).to(x.dtype).float()
    return result.to(x.dtype)


@pytest.mark.fused_moe
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [0, 1, 8, 12, 13, 64, 128, 256])
@pytest.mark.parametrize("padded", [False, True])
def test_thead_fused_moe_original_weight_layout(
    thead_moe_original_layout, dtype, m, padded, monkeypatch
):
    torch.manual_seed(67)
    inputs = _thead_moe_inputs(m, dtype, padded)
    x, w1, w2, weights, ids = inputs
    original_strides = (w1.stride(), w2.stride())
    seen = []
    dispatch = thead_moe_original_layout.dispatch_fused_moe_kernel

    def check_weight(a, b, *args, **kwargs):
        expected = (w1, w2)[len(seen) % 2]
        assert b is expected
        seen.append(b.stride())
        return dispatch(a, b, *args, **kwargs)

    monkeypatch.setattr(
        thead_moe_original_layout, "dispatch_fused_moe_kernel", check_weight
    )
    expected = _thead_moe_reference(*inputs)
    actual = thead_moe_original_layout.fused_experts_impl(*inputs)
    assert actual.shape == x.shape and actual.dtype == dtype
    assert (w1.stride(), w2.stride()) == original_strides
    assert seen == list(original_strides) * bool(m)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.01)


@pytest.mark.fused_moe
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_thead_fused_moe_weight_reuse_and_update(thead_moe_original_layout, dtype):
    torch.manual_seed(68)
    # Three live pairs exceed the former four-tensor module cache capacity.
    layers = [_thead_moe_inputs(1, dtype) for _ in range(3)]
    for _ in range(2):
        for inputs in layers:
            actual = thead_moe_original_layout.fused_experts_impl(*inputs)
            torch.testing.assert_close(
                actual.cpu(), _thead_moe_reference(*inputs), rtol=0.05, atol=0.01
            )
    x, w1, w2, weights, ids = layers[0]
    w2.neg_()
    expected = _thead_moe_reference(x, w1, w2, weights, ids)
    actual = thead_moe_original_layout.fused_experts_impl(
        x, w1, w2, weights, ids, inplace=True
    )
    assert actual is x
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.01)


@pytest.mark.fused_moe
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,hidden", [(1, 128), (8, 1024), (13, 128), (64, 128)])
@pytest.mark.parametrize("padded", [False, True])
def test_thead_fused_moe_graph_reads_current_weights(
    thead_moe_original_layout, dtype, m, hidden, padded
):
    torch.manual_seed(69)
    inputs = _thead_moe_inputs(m, dtype, padded, hidden)
    x, w1, w2, weights, ids = inputs
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            thead_moe_original_layout.fused_experts_impl(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = thead_moe_original_layout.fused_experts_impl(*inputs)

    # Update inputs and both original weights at their captured addresses.
    # Graphs must observe each change, not a stale cached transpose.
    original_w1, original_w2 = w1.clone(), w2.clone()
    previous = _thead_moe_reference(*inputs)
    for change in ("input", "weights", "restore", "input"):
        if change == "input":
            x.copy_(torch.randn_like(x))
        elif change == "weights":
            w1.copy_(original_w1 * 0.5)
            w2.copy_(-original_w2)
        else:
            w1.copy_(original_w1)
            w2.copy_(original_w2)
        expected = _thead_moe_reference(*inputs)
        assert not torch.allclose(previous, expected, rtol=0.05, atol=0.01)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.01)
        previous = expected
