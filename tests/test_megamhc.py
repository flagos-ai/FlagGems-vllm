# SPDX-License-Identifier: Apache-2.0
"""Test MegaMHC accuracy and CUDA Graph replay."""

import pytest
import torch

import flaggems_vllm

HIDDEN = 4096
STREAMS = 4
PROJECTIONS = 24
EPS = 1e-6
POST_SCALE = 2.0
SINKHORN_ROUNDS = 20
MAX_DIFF = 5e-5
OUTPUTS = ("new_residual", "new_post_mix", "new_comb_res_mix", "y_bf16")


def make_inputs(tokens):
    generator = torch.Generator(device="cuda").manual_seed(20261111 + tokens)

    def randn(shape, dtype=torch.float32):
        return torch.randn(shape, dtype=dtype, device="cuda", generator=generator)

    comb = torch.rand((tokens, STREAMS, STREAMS), device="cuda", generator=generator)
    for _ in range(3):
        comb /= comb.sum(-1, keepdim=True)
        comb /= comb.sum(-2, keepdim=True)
    return (
        randn((tokens, HIDDEN), torch.bfloat16),
        randn((tokens, STREAMS, HIDDEN), torch.bfloat16),
        randn((tokens, STREAMS, 1)).sigmoid_(),
        comb,
        randn((PROJECTIONS, STREAMS * HIDDEN)).mul_(1e-2),
        randn((3,)).mul_(0.1),
        randn((PROJECTIONS,)).mul_(0.1),
        randn((HIDDEN,)).mul_(0.1).add_(1).bfloat16(),
    )


def reference(inputs):
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch

    x, residual, post, comb, fn, scale, base, norm = inputs
    updated = mhc_post_torch(x, residual, post, comb)
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        next_post, next_comb, layer_input = mhc_pre_torch(
            updated, fn, scale, base, EPS, EPS, EPS, POST_SCALE, SINKHORN_ROUNDS
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
    value = layer_input.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS)
    return updated, next_post, next_comb, (normalized * norm.float()).bfloat16()


def difference(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"Got {actual.shape} {actual.dtype}, expected {expected.shape} {expected.dtype}"
        )
    if not torch.isfinite(actual).all().item():
        return float("inf")
    actual, expected = actual.double(), expected.double()
    denominator = (actual.square() + expected.square()).sum().item()
    if denominator == 0:
        return 0.0
    return (actual - expected).square().sum().item() / denominator


GRAPH_TOKENS = (1, 2, 4, 8, 16, *range(24, 257, 8), *range(272, 513, 16))
EAGER_TOKENS = (1028, 2048, 3076, 3805, 4100, 12991, 13005, 13021, 16384)


def invoke(op, inputs, **kwargs):
    *values, norm = inputs
    return op(
        *values,
        EPS,
        EPS,
        EPS,
        POST_SCALE,
        SINKHORN_ROUNDS,
        1,
        1,
        norm,
        EPS,
        **kwargs,
    )


@pytest.fixture(scope="module")
def megamhc_op():
    if flaggems_vllm.vendor_name == "hygon" and torch.cuda.is_available():
        return flaggems_vllm.mega_mhc
    pytest.skip("MegaMHC currently requires a Hygon DCU with FlagTree")


def verify(results, expected):
    for actual, target in zip(results, expected, strict=True):
        assert difference(actual, target) < MAX_DIFF


@pytest.mark.parametrize("tokens", GRAPH_TOKENS)
@torch.inference_mode()
@pytest.mark.megamhc
def test_megamhc_graph(megamhc_op, tokens):
    inputs = make_inputs(tokens)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        invoke(megamhc_op, inputs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            results = invoke(megamhc_op, inputs)
        graph.replay()
        verify(results, reference(inputs))

        if tokens in (1, 64, 512):
            inputs[0].mul_(0.75)
            graph.replay()
            verify(results, reference(inputs))
        stream.synchronize()


@pytest.mark.parametrize("tokens", EAGER_TOKENS)
@torch.inference_mode()
@pytest.mark.megamhc
def test_megamhc_eager(megamhc_op, tokens):
    inputs = make_inputs(tokens)
    verify(invoke(megamhc_op, inputs), reference(inputs))


@pytest.mark.megamhc
@pytest.mark.skipif(flaggems_vllm.vendor_name != "hygon", reason="Hygon export")
def test_megamhc_hygon_entry(megamhc_op):
    from flaggems_vllm.runtime.backend._hygon.ops.megamhc import mega_mhc

    assert megamhc_op is mega_mhc


@pytest.mark.megamhc
def test_megamhc_empty(megamhc_op):
    outputs = invoke(megamhc_op, make_inputs(0))
    assert [tuple(out.shape) for out in outputs] == [
        (0, 4, 4096),
        (0, 4, 1),
        (0, 4, 4),
        (0, 4096),
    ]
    assert [out.dtype for out in outputs] == [
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.bfloat16,
    ]


@pytest.mark.parametrize("invalid", ("dtype", "layout", "gradient", "parameters"))
@pytest.mark.megamhc
def test_megamhc_unsupported(megamhc_op, invalid):
    x, *rest = make_inputs(2)
    *values, norm = rest
    rounds = 20
    if invalid == "dtype":
        x = x.float()
    elif invalid == "layout":
        x = x.T.contiguous().T
    elif invalid == "gradient":
        x.requires_grad_()
    else:
        rounds = 19
    with pytest.raises(NotImplementedError):
        megamhc_op(x, *values, 1e-6, 1e-6, 1e-6, 2.0, rounds, 1, 1, norm, 1e-6)


@pytest.mark.parametrize(
    "tokens", (1, 24, 32, 33, 64, 127, 128, 129, 512, 1028, 2048, 3076, 16384)
)
@torch.inference_mode()
@pytest.mark.megamhc
def test_megamhc_prepared_graph(megamhc_op, tokens):
    inputs = make_inputs(tokens)
    prepare = flaggems_vllm.mega_mhc_prepare
    fn = inputs[4]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared = prepare(fn, max_decode_tokens=128 if tokens <= 129 else None)
        expected = reference(inputs)
        verify(invoke(megamhc_op, inputs, prepared=prepared), expected)
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            first = invoke(megamhc_op, inputs, prepared=prepared)
            second = invoke(megamhc_op, inputs, prepared=prepared)

        for _ in range(3):
            inputs[0].mul_(0.75)
            graph.replay()
            expected = reference(inputs)
            verify(first, expected)
            verify(second, expected)

        fn.mul_(-1.5)
        assert prepare(fn, out=prepared) is prepared
        graph.replay()
        expected = reference(inputs)
        verify(first, expected)
        verify(second, expected)
        stream.synchronize()
