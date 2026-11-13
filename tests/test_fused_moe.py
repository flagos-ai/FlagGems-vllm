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
@pytest.mark.parametrize(
    "m,hidden", [(1, 128), (1, 1024), (8, 1024), (13, 128), (64, 128)]
)
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


def _thead_dense_moe_inputs(m, dtype):
    x, w1, w2, _, ids = _thead_moe_inputs(m, dtype, padded=True)
    # Keep padded strides and a nonzero expert storage offset on both weights.
    w1, w2 = w1[1:9], w2[1:9]
    ids = ids[:, :2].remainder(8)
    weights = torch.full((m, 2), 0.5, device="cuda", dtype=torch.float32)
    return x, w1, w2, weights, ids


@pytest.mark.fused_moe
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [64, 128, 512])
def test_thead_fused_moe_dense_routing(thead_moe_original_layout, dtype, m):
    import flaggems_vllm

    torch.manual_seed(71)
    inputs = _thead_dense_moe_inputs(m, dtype)
    assert inputs[1].storage_offset() and inputs[2].storage_offset()
    actual = flaggems_vllm.ops_experts_impl(*inputs)
    torch.testing.assert_close(
        actual.cpu(), _thead_moe_reference(*inputs), rtol=0.05, atol=0.01
    )


@pytest.mark.fused_moe
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [128, 512])
def test_thead_fused_moe_dense_graph_reads_current_weights(
    thead_moe_original_layout, dtype, m
):
    import flaggems_vllm

    torch.manual_seed(72)
    inputs = _thead_dense_moe_inputs(m, dtype)
    x, w1, w2, _, _ = inputs
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flaggems_vllm.ops_experts_impl(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = flaggems_vllm.ops_experts_impl(*inputs)
    previous = _thead_moe_reference(*inputs)
    for _ in range(2):
        x.copy_(torch.randn_like(x))
        w1.mul_(0.75)
        w2.neg_()
        actual.fill_(float("nan"))
        expected = _thead_moe_reference(*inputs)
        assert not torch.allclose(previous, expected, rtol=0.05, atol=0.01)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.01)
        previous = expected


@pytest.mark.fused_moe
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "m,e,h", [(1, 8, 4096), (4, 32, 1024), (16, 8, 2048), (4, 8, 2048)]
)
def test_thead_fused_moe_long_k_reference(thead_moe_original_layout, dtype, m, e, h):
    import flaggems_vllm

    torch.manual_seed(73)
    intermediate, topk = 512, 2
    x = torch.randn((m, h), device="cuda", dtype=dtype)
    w1 = torch.randn((e, 2 * intermediate, h), device="cuda", dtype=dtype)
    w1 *= h**-0.5
    w2 = torch.randn((e, h, intermediate), device="cuda", dtype=dtype)
    w2 *= intermediate**-0.5
    weights = torch.full((m, topk), 0.5, device="cuda", dtype=torch.float32)
    ids = (torch.arange(m * topk, device="cuda").view(m, topk) % e).int()
    inputs = x, w1, w2, weights, ids
    actual = flaggems_vllm.ops_experts_impl(*inputs)
    torch.testing.assert_close(
        actual.cpu(), _thead_moe_reference(*inputs), rtol=0.05, atol=0.01
    )


@pytest.mark.fused_moe
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("quantization", ["use_int8_w8a16", "use_int4_w4a16"])
def test_thead_fused_moe_dequantized_gemv(
    thead_moe_original_layout, dtype, quantization
):
    import flaggems_vllm

    torch.manual_seed(74)
    e, hidden, intermediate, topk = 8, 1024, 512, 2
    limit = 8 if quantization == "use_int4_w4a16" else 32
    x = torch.randn((1, hidden), device="cuda", dtype=dtype)
    # T-Head stores W4 unpacked in int8 containers, with the full K dimension.
    w1 = torch.randint(
        -limit, limit, (e, 2 * intermediate, hidden), device="cuda", dtype=torch.int8
    )
    w2 = torch.randint(
        -limit, limit, (e, hidden, intermediate), device="cuda", dtype=torch.int8
    )
    scale1 = torch.full(
        (e, 2 * intermediate),
        (3 / hidden) ** 0.5 / limit,
        device="cuda",
        dtype=torch.float32,
    )
    scale2 = torch.full(
        (e, hidden),
        (3 / intermediate) ** 0.5 / limit,
        device="cuda",
        dtype=torch.float32,
    )
    weights = torch.full((1, topk), 0.5, device="cuda", dtype=torch.float32)
    ids = torch.arange(topk, device="cuda", dtype=torch.int32).view(1, topk)
    kwargs = {quantization: True, "w1_scale": scale1, "w2_scale": scale2}

    def reference():
        # Match the public contract's dequantization rounding, then use the
        # independent CPU MoE formula. No production helper is the oracle.
        q1, q2 = w1.cpu().to(dtype), w2.cpu().to(dtype)
        s1, s2 = scale1.cpu().to(dtype), scale2.cpu().to(dtype)
        return _thead_moe_reference(
            x.cpu(),
            q1 * s1.unsqueeze(-1),
            q2 * s2.unsqueeze(-1),
            weights.cpu(),
            ids.cpu(),
        )

    expected = reference()
    actual = flaggems_vllm.ops_experts_impl(x, w1, w2, weights, ids, **kwargs)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.01)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flaggems_vllm.ops_experts_impl(x, w1, w2, weights, ids, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = flaggems_vllm.ops_experts_impl(x, w1, w2, weights, ids, **kwargs)
    for change in ("input", "weights"):
        previous = expected
        if change == "input":
            x.copy_(torch.randn_like(x))
        else:
            w2.zero_()
        expected = reference()
        assert not torch.allclose(previous, expected, rtol=0.05, atol=0.01)
        actual.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.01)


@pytest.mark.fused_moe
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "m,hidden,intermediate,topk",
    [
        (9, 3072, 1024, 4),
        (16, 3072, 1024, 3),
        (16, 2048, 1024, 4),
        (16, 3072, 1024, 4),
        (16, 3073, 1040, 4),
    ],
)
@pytest.mark.parametrize("padded", [False, True])
def test_thead_fused_moe_long_k_swap_graph(
    thead_moe_original_layout, monkeypatch, dtype, m, hidden, intermediate, topk, padded
):
    import flaggems_vllm

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(105)
        experts = 128
        x = torch.randn((m, hidden), device="cuda", dtype=dtype)

        def weight(n, k):
            shape = (experts, n + 1, k + 16) if padded else (experts, n, k)
            storage = torch.randn(shape, device="cuda", dtype=dtype) * (k**-0.5)
            return storage[:, :n, 8 : 8 + k] if padded else storage

        w1 = weight(2 * intermediate, hidden)
        w2 = weight(hidden, intermediate)
        weights = torch.full((m, topk), 1 / topk, device="cuda")
        ids = (torch.arange(m * topk, device="cuda").view(m, topk) % 8).int()
        values = x, w1, w2, weights, ids
        generic = importlib.import_module("flaggems_vllm.ops.fused_moe")
        invoke = generic.invoke_fused_moe_triton_kernel
        seen = []

        def observe(a, b, *args, **kwargs):
            seen.append(b)
            return invoke(a, b, *args, **kwargs)

        expected = _thead_moe_reference(*values)
        with monkeypatch.context() as context:
            context.setattr(generic, "invoke_fused_moe_triton_kernel", observe)
            actual = flaggems_vllm.ops_experts_impl(*values)
        if m == 16 and hidden == 3072 and topk == 4:
            assert len(seen) == 2 and seen[0] is w1 and seen[1] is w2
        else:
            # Small workloads and partial tiles retain the vendor schedule.
            assert not seen
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.01)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                flaggems_vllm.ops_experts_impl(*values)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = flaggems_vllm.ops_experts_impl(*values)
        # Different expert histograms exercise new padded ranges in one graph.
        for active_experts in (experts, 5, 2):
            ids.copy_(
                (
                    torch.arange(m * topk, device="cuda").view(m, topk) % active_experts
                ).int()
            )
            x.copy_(torch.randn_like(x))
            w1.mul_(0.75)
            w2.neg_()
            expected = _thead_moe_reference(*values)
            actual.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.01)

        for value in (float("nan"), float("inf"), -float("inf"), 0.0):
            x[0, 0] = value
            expected = _thead_moe_reference(*values)
            actual = flaggems_vllm.ops_experts_impl(*values)
            torch.testing.assert_close(
                actual.cpu(), expected, rtol=0.05, atol=0.01, equal_nan=True
            )
        expected = _thead_moe_reference(*values)
        actual = flaggems_vllm.ops_experts_impl(*values, inplace=True)
        assert actual is x
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.01)


@pytest.mark.fused_moe
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden,intermediate", [(1024, 512), (3072, 1024)])
def test_thead_fused_moe_long_k_swap_upper_batch(
    thead_moe_original_layout, dtype, hidden, intermediate
):
    import flaggems_vllm

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(106)
        m, experts, topk = 256, 128, 4
        # Repeat diverse rows to bound CPU-reference memory at the batch limit.
        x = torch.randn((16, hidden), device="cuda", dtype=dtype).repeat(16, 1)
        w1 = torch.randn(
            (experts, 2 * intermediate, hidden), device="cuda", dtype=dtype
        ) * (hidden**-0.5)
        w2 = torch.randn(
            (experts, hidden, intermediate), device="cuda", dtype=dtype
        ) * (intermediate**-0.5)
        ids = (
            (torch.arange(16 * topk, device="cuda").view(16, topk) % experts)
            .int()
            .repeat(16, 1)
        )
        weights = torch.full((m, topk), 1 / topk, device="cuda")
        values = x, w1, w2, weights, ids
        expected = _thead_moe_reference(x[:16], w1, w2, weights[:16], ids[:16]).repeat(
            16, 1
        )
        actual = flaggems_vllm.ops_experts_impl(*values)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.01)


@pytest.mark.fused_moe
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_thead_fused_moe_long_k_keeps_naive_gemv(
    thead_moe_original_layout, monkeypatch, dtype
):
    import flaggems_vllm

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(107)
        m, experts, hidden, intermediate, topk = 16, 128, 3072, 1024, 4
        x = torch.randn((m, hidden), device="cuda", dtype=dtype)
        w1 = torch.randn(
            (experts, 2 * intermediate, hidden), device="cuda", dtype=dtype
        ) * (hidden**-0.5)
        w2 = torch.randn(
            (experts, hidden, intermediate), device="cuda", dtype=dtype
        ) * (intermediate**-0.5)
        ids = (torch.arange(m * topk, device="cuda").view(m, topk) % experts).int()
        weights = torch.full((m, topk), 1 / topk, device="cuda")
        values = x, w1, w2, weights, ids
        expected = _thead_moe_reference(*values)
        generic = importlib.import_module("flaggems_vllm.ops.fused_moe")

        def unexpected_swap(*args, **kwargs):
            pytest.fail("Naive routing must retain the vendor GEMV path")

        monkeypatch.setattr(generic, "invoke_fused_moe_triton_kernel", unexpected_swap)
        actual = flaggems_vllm.ops_experts_impl(*values, global_num_experts=1024)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.01)


@pytest.mark.fused_moe
def test_thead_fused_moe_keeps_quantized_swap_rejection(thead_moe_original_layout):
    import triton.language as tl

    # Invalid quantized layout flags must keep their pre-existing error contract.
    # CPU buffers ensure this rejection happens before any device launch.
    a = torch.empty((16, 16), dtype=torch.int8)
    b = torch.empty((1, 16, 16), dtype=torch.int8)
    c = torch.empty((16, 1, 16), dtype=torch.bfloat16)
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_K": 16,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
        "INT8_ACCUMULATOR": True,
        "SWAP_AB": True,
    }
    with pytest.raises(ValueError, match="INT8_ACCUMULATOR requires"):
        thead_moe_original_layout.invoke_fused_moe_triton_kernel(
            a,
            b,
            c,
            None,
            None,
            None,
            torch.arange(16, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([16], dtype=torch.int32),
            False,
            1,
            config,
            tl.bfloat16,
            False,
            True,
            False,
            False,
            True,
            None,
        )


@pytest.mark.fused_moe
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("topk", [8, 9])
def test_thead_fused_moe_sparse_gm_policy_boundary(
    thead_moe_original_layout, monkeypatch, dtype, topk
):
    """Public R=E / R>E policy boundary for the sparse G1 group order."""
    import flaggems_vllm

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(111)
        m, e, h, i = 16, 128, 3072, 1024
        x = torch.randn((m, h), device="cuda", dtype=dtype)
        storage1 = torch.randn((e, 2 * i + 1, h + 16), device="cuda", dtype=dtype)
        storage2 = torch.randn((e, h + 1, i + 16), device="cuda", dtype=dtype)
        w1 = storage1[:, : 2 * i, 8 : 8 + h]
        w2 = storage2[:, :h, 8 : 8 + i]
        w1.mul_(h**-0.5)
        w2.mul_(i**-0.5)
        weights = torch.full((m, topk), 1 / topk, device="cuda")
        ids = (torch.arange(m * topk, device="cuda").view(m, topk) % 8).int()
        values = x, w1, w2, weights, ids
        expected = _thead_moe_reference(*values)

        mod = thead_moe_original_layout
        dispatch = mod.dispatch_fused_moe_kernel
        seen = []

        def observe(*args, **kwargs):
            seen.append(args)
            return dispatch(*args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(mod, "dispatch_fused_moe_kernel", observe)
            actual = flaggems_vllm.ops_experts_impl(*values)
        assert len(seen) == 2
        g1, g2 = seen[0][12], seen[1][12]
        assert g1["GROUP_SIZE_M"] == (1 if m * topk <= e else 8)
        assert g2["GROUP_SIZE_M"] == 8
        assert seen[0][1] is w1 and seen[1][1] is w2
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.01)


@pytest.mark.fused_moe
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_thead_fused_moe_long_k_three_live_banks(
    thead_moe_original_layout, monkeypatch, dtype
):
    """Three live weight banks, changed state, and public graph replay."""
    import flaggems_vllm

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(112)
        m, e, h, i, k = 16, 128, 3072, 1024, 4
        banks = []
        for _ in range(3):
            x = torch.randn((m, h), device="cuda", dtype=dtype)
            w1 = torch.randn((e, 2 * i, h), device="cuda", dtype=dtype) * (h**-0.5)
            w2 = torch.randn((e, h, i), device="cuda", dtype=dtype) * (i**-0.5)
            weights = torch.full((m, k), 1 / k, device="cuda")
            ids = (torch.arange(m * k, device="cuda").view(m, k) % 256).int()
            banks.append([x, w1, w2, weights, ids])
        originals = [[t.clone() for t in bank] for bank in banks]

        mod = thead_moe_original_layout
        dispatch = mod.dispatch_fused_moe_kernel
        seen = []

        def observe(*args, **kwargs):
            if args[1].size(1) == 2 * i:
                seen.append(args[1])
            return dispatch(*args, **kwargs)

        def run():
            return [flaggems_vllm.ops_experts_impl(*bank) for bank in banks]

        with monkeypatch.context() as context:
            context.setattr(mod, "dispatch_fused_moe_kernel", observe)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = run()

        # Warmups and graph capture may all pass through the observer; every
        # GEMM1 weight must reference one of the three live caller banks.
        assert len(seen) >= 9
        bank_w1 = [bank[1] for bank in banks]
        used = set()
        for b in seen:
            assert any(b is w for w in bank_w1)
            used.add(next(i for i, w in enumerate(bank_w1) if b is w))
        assert used == {0, 1, 2}

        for step, active in enumerate((256, 2, 8)):
            expected = []
            for bank in banks:
                bank[0].copy_(torch.randn_like(bank[0]))
                bank[1].mul_(0.75)
                bank[2].neg_()
                bank[4].copy_(
                    (torch.arange(m * k, device="cuda").view(m, k) % active).int()
                )
                expected.append(_thead_moe_reference(*bank))
            for out in outputs:
                out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for out, ref in zip(outputs, expected):
                torch.testing.assert_close(out.cpu(), ref, rtol=0.05, atol=0.01)

        for bank, original in zip(banks, originals):
            for value, saved in zip(bank, original):
                value.copy_(saved)
