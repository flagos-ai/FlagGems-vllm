# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems

pytestmark = [
    pytest.mark.skipif(
        gems.vendor_name != "thead" or not torch.cuda.is_available(),
        reason="requires T-Head PPU",
    ),
]

EPS = (1e-6, 1e-6, 1e-6, 2.0, 20)


@pytest.fixture(scope="module")
def vllm_reference():
    # Load the installed pure-torch reference without initializing vLLM's
    # serving runtime or requiring its CUDA/TileLang extensions on PPU.
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("vLLM is not installed")
    path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "model_executor/kernels/mhc/torch.py"
    )
    if not path.is_file():
        pytest.skip("installed vLLM has no mHC torch reference")
    spec = importlib.util.spec_from_file_location("_vllm_mhc_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(tokens, hidden, streams, strided=False):
    gen = torch.Generator().manual_seed(20261117 + tokens)

    def randn(shape, dtype=torch.float32):
        return torch.randn(shape, generator=gen, dtype=dtype).cuda()

    mixes = streams * (streams + 2)
    x = randn((tokens, hidden), torch.bfloat16)
    residual = randn((tokens, streams, hidden), torch.bfloat16)
    post = randn((tokens, streams, 1)).sigmoid()
    comb = randn((tokens, streams, streams)).softmax(-1)
    if strided:
        comb = comb.mT
    fn = randn((mixes, streams * hidden)) * 0.01
    scale = randn((3,)) * 0.1
    base = randn((mixes,)) * 0.1
    norm = (randn((hidden,)) * 0.1 + 1).bfloat16()
    return x, residual, post, comb, fn, scale, base, norm


def reference(args, use_norm=True, *, eps=None, norm_eps=1e-6):
    x, residual, post, comb, fn, scale, base, norm = (a.cpu() for a in args)
    streams = residual.shape[1]
    new_residual = (
        x.float().unsqueeze(1) * post
        + torch.einsum("tij,tih->tjh", comb, residual.float())
    ).bfloat16()
    flat = new_residual.flatten(1).float()
    rms_eps, pre_eps, sink_eps, multiplier, repeats = EPS if eps is None else eps
    projection = (flat.double() @ fn.double().T).float() * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + rms_eps
    )
    p, q, c = projection.split((streams, streams, streams * streams), dim=-1)
    b_pre, b_post, b_comb = base.split((streams, streams, streams * streams))
    pre = (torch.sigmoid(p * scale[0] + b_pre) + pre_eps).unsqueeze(-1)
    new_post = (torch.sigmoid(q * scale[1] + b_post) * multiplier).unsqueeze(-1)
    new_comb = (c * scale[2] + b_comb).reshape(-1, streams, streams).softmax(
        -1
    ) + sink_eps
    new_comb /= new_comb.sum(-2, keepdim=True) + sink_eps
    for _ in range(1, repeats):
        new_comb /= new_comb.sum(-1, keepdim=True) + sink_eps
        new_comb /= new_comb.sum(-2, keepdim=True) + sink_eps
    y = (new_residual.float() * pre).sum(1).bfloat16()
    if use_norm:
        y = y.float()
        y = (
            y
            * torch.rsqrt(y.square().mean(-1, keepdim=True) + norm_eps)
            * norm.bfloat16().float()
        ).bfloat16()
    return new_residual, new_post, new_comb, y


def candidate(args, use_norm=True, *, packed_fn=None, implementation="auto"):
    return gems.mega_mhc(
        *args[:-1],
        *EPS,
        norm_weight=args[-1] if use_norm else None,
        packed_fn=packed_fn,
        implementation=implementation,
    )


def verify(result, expected):
    for actual, target in zip(result, expected, strict=True):
        if target.dtype is torch.float32:
            torch.testing.assert_close(
                actual.cpu(), target, rtol=3e-4, atol=2e-5, equal_nan=True
            )
        else:
            torch.testing.assert_close(
                actual.cpu(), target, rtol=0.016, atol=0.008, equal_nan=True
            )


@pytest.mark.parametrize(
    "tokens,hidden,streams",
    [
        (1, 4096, 4),
        (17, 259, 4),
        (3, 513, 2),
        (5, 127, 1),
        (3, 129, 8),
        (3, 16384, 8),
        (3, 1, 1),
        (3, 1, 8),
        (1029, 259, 1),
        (4097, 259, 4),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.megamhc
def test_mega_mhc(tokens, hidden, streams, strided):
    args = make_inputs(tokens, hidden, streams, strided=strided)
    verify(candidate(args), reference(args))


@pytest.mark.megamhc
def test_mega_mhc_graph_replay():
    args = make_inputs(13, 5120, 4)
    candidate(args, use_norm=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = candidate(args, use_norm=False)
    for _ in range(2):
        args[0].mul_(-0.75)
        graph.replay()
        verify(out, reference(args, use_norm=False))


@pytest.mark.parametrize("implementation", ["auto", "triton"])
@pytest.mark.parametrize(
    "tokens,hidden",
    [(128, 4096), (256, 4096), (384, 4096), (512, 4096), (4097, 521), (8193, 521)],
)
@pytest.mark.megamhc
def test_mega_mhc_projection_cancellation(tokens, hidden, implementation):
    streams = 4
    args = make_inputs(tokens, hidden, streams)
    x, residual, post, comb, fn, scale, base, norm = args
    x.zero_()
    residual.fill_(1)
    post.zero_()
    comb.copy_(torch.eye(streams, device=comb.device))
    fn.zero_()
    scale.fill_(1)
    base.zero_()
    norm.fill_(1)

    # The first post coefficient depends on a small FP32 remainder in each pair.
    # Converting the weights to two BF16 terms can erase the remainder.
    weight = 1.0 + 2**-9
    fn[streams, 0::2] = weight + 2**-19
    fn[streams, 1::2] = -weight

    rows = torch.tensor([0, tokens // 2, tokens - 1], device=x.device)

    def check(outputs):
        sampled = tuple(a[rows] if i < 4 else a for i, a in enumerate(args))
        verify(tuple(t[rows] for t in outputs), reference(sampled))

    check(candidate(args, implementation=implementation))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = candidate(args, implementation=implementation)
    for _ in range(2):
        fn.neg_()
        graph.replay()
        check(outputs)


@pytest.mark.parametrize("implementation", ["auto", "triton"])
@pytest.mark.parametrize(
    "hidden,streams,unaligned",
    [
        (256, 2, False),
        (256, 4, False),
        (512, 2, False),
        (512, 4, False),
        (512, 4, True),
        (768, 2, False),
        (768, 4, False),
        (4096, 4, False),
        (16384, 2, False),
        (521, 4, False),
        (259, 2, False),
    ],
)
@pytest.mark.megamhc
def test_mega_mhc_prepacked_graph_replay(
    hidden, streams, unaligned, implementation, monkeypatch
):
    args = make_inputs(4097, hidden, streams, strided=True)
    if unaligned:
        x = args[0]
        offset_x = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)[1:]
        offset_x = offset_x.view_as(x).copy_(x)
        args = (offset_x, *args[1:])
    if implementation == "triton":
        mod = importlib.import_module(
            "flaggems_vllm.runtime.backend._thead.ops.megamhc"
        )

        def unexpected_bitcode():
            pytest.fail("Triton implementation requested device bitcode")

        monkeypatch.setattr(mod, "device_bitcode", unexpected_bitcode)
    packed = gems.pack_mega_mhc_weights(args[4])
    # Keep the original weights for the reference and change the input
    # weights after capture to verify the graph reads the packed snapshot.
    frozen_args = (*args[:4], args[4].clone(), *args[5:])
    candidate(args, packed_fn=packed, implementation=implementation)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = candidate(args, packed_fn=packed, implementation=implementation)

    args[4].neg_()
    sample_rows = [0, 1, 2, 3, 6, 7, 8, 63, 64, 2048, 4095, 4096]
    for _ in range(2):
        args[0].mul_(-0.75)
        graph.replay()
        sampled = tuple(
            value[sample_rows] if i < 4 else value
            for i, value in enumerate(frozen_args)
        )
        verify(tuple(value[sample_rows] for value in outputs), reference(sampled))


@pytest.mark.megamhc
def test_mega_mhc_empty():
    args = make_inputs(0, 127, 4)
    assert [t.shape for t in candidate(args)] == [
        (0, 4, 127),
        (0, 4, 1),
        (0, 4, 4),
        (0, 127),
    ]


@pytest.mark.parametrize("implementation", ["auto", "triton"])
@pytest.mark.parametrize(
    "tokens,hidden,streams,use_norm",
    [
        (127, 3072, 4, True),
        (129, 4104, 4, False),
        (257, 4096, 4, True),
        (385, 4096, 4, True),
        (385, 4088, 4, True),
        (385, 4104, 4, True),
        (385, 4096, 4, False),
        (385, 2048, 4, False),
        (385, 8192, 2, True),
        (385, 16384, 2, True),
        (385, 16384, 4, True),
        (129, 16384, 2, True),
    ],
)
@pytest.mark.megamhc
def test_mega_mhc_short_graph_replay(tokens, hidden, streams, use_norm, implementation):
    args = make_inputs(tokens, hidden, streams, strided=True)
    graph = torch.cuda.CUDAGraph()
    # Warm up the device helper and Triton compilation before capture.
    candidate(args, use_norm=use_norm, implementation=implementation)
    with torch.cuda.graph(graph):
        outputs = candidate(args, use_norm=use_norm, implementation=implementation)
    for _ in range(3):
        args[0].mul_(-0.75)
        args[4].mul_(-0.5)
        graph.replay()
        verify(outputs, reference(args, use_norm=use_norm))


@pytest.mark.parametrize("repeats", [1, 31])
@pytest.mark.parametrize("streams,hidden", [(4, 4096), (2, 16384)])
@pytest.mark.parametrize("use_norm", [False, True])
@pytest.mark.megamhc
def test_mega_mhc_sinkhorn_graph_replay(
    repeats, streams, hidden, use_norm, monkeypatch
):
    monkeypatch.setitem(globals(), "EPS", (*EPS[:-1], repeats))
    args = make_inputs(385, hidden, streams, strided=True)
    candidate(args, use_norm=use_norm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = candidate(args, use_norm=use_norm)
    for _ in range(3):
        args[0].mul_(-0.75)
        args[4].mul_(-0.5)
        graph.replay()
        verify(outputs, reference(args, use_norm=use_norm))


@pytest.mark.parametrize("implementation", ["auto", "triton"])
@pytest.mark.parametrize("tokens", [385, 4097])
@pytest.mark.megamhc
def test_mega_mhc_broadcast_comb_graph_replay(tokens, implementation):
    args = make_inputs(tokens, 4096, 4)
    comb = args[3][:, :1, :1].expand(tokens, 4, 4)
    args = (*args[:3], comb, *args[4:])
    packed = gems.pack_mega_mhc_weights(args[4])
    candidate(args, packed_fn=packed, implementation=implementation)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = candidate(args, packed_fn=packed, implementation=implementation)
    rows = [0, tokens // 2, tokens - 1]
    for _ in range(3):
        args[0].mul_(-0.75)
        args[5].mul_(-0.5)
        graph.replay()
        sampled = tuple(value[rows] if i < 4 else value for i, value in enumerate(args))
        verify(tuple(value[rows] for value in outputs), reference(sampled))


@pytest.mark.parametrize("implementation", ["auto", "triton"])
@pytest.mark.parametrize(
    "outer,hidden,streams,norm_dtype",
    [
        ((), 127, 1, None),
        ((1,), 129, 2, torch.float64),
        ((0, 2), 129, 8, torch.float32),
        ((2, 3), 259, 8, torch.bfloat16),
        ((3, 43), 4096, 4, torch.bfloat16),
        ((3, 43), 4104, 4, torch.float32),
        ((3, 171), 768, 2, torch.float16),
        ((1, 1029), 259, 1, torch.float32),
        ((2, 1024), 4096, 4, torch.float32),
        ((4097,), 521, 4, torch.bfloat16),
    ],
)
@pytest.mark.megamhc
def test_mega_mhc_vllm_reference(
    outer, hidden, streams, norm_dtype, implementation, vllm_reference
):
    """Compare with vLLM's own post/pre, independently of our FP64 oracle."""
    tokens = 1
    for dim in outer:
        tokens *= dim
    args = list(make_inputs(tokens, hidden, streams, strided=True))
    args[0] = args[0].view(*outer, hidden)
    args[1] = args[1].view(*outer, streams, hidden)
    # vLLM accepts post coefficients with or without a final singleton.
    args[2] = args[2].view(*outer, streams)
    args[3] = args[3].view(*outer, streams, streams)
    norm = None
    if norm_dtype is not None:
        # Strided FP32/FP16 weights exercise vLLM's BF16 conversion as well.
        norm = torch.empty(2 * hidden, dtype=norm_dtype, device=args[0].device)[::2]
        norm.copy_(args[-1].float() + 0.0021)
    eps = (2e-4, 3e-3, 4e-3, 1.25, 3)
    norm_eps = 0.03

    def call():
        return gems.mega_mhc(
            *args[:-1],
            *eps,
            n_splits=3,
            tile_n=7,
            norm_weight=norm,
            norm_eps=norm_eps,
            implementation=implementation,
        )

    frozen = [t.cpu().clone() for t in args]
    outputs = call()
    for value, original in zip(args, frozen, strict=True):
        torch.testing.assert_close(value.cpu(), original, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for replay in (False, True):
        if replay:
            args[0].mul_(-0.75)
            args[4].neg_()
            args[2].mul_(-0.5)
            args[3].mul_(-0.75)
            args[5].neg_()
            args[6].add_(0.01)
            if norm is not None:
                norm.mul_(-0.8)
            graph.replay()
            outputs = captured
        cpu = [a.cpu() for a in args[:-1]]
        # The official torch pre uses view(-1, ...), which is ambiguous at zero.
        if tokens:
            updated = vllm_reference.mhc_post_torch(
                cpu[0], cpu[1], cpu[2].unsqueeze(-1), cpu[3]
            )
            post, comb, y = vllm_reference.mhc_pre_torch(updated, *cpu[4:], *eps)
            if norm is not None:
                value = y.float()
                y = (
                    value
                    * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm_eps)
                    * norm.cpu().bfloat16().float()
                ).bfloat16()
            expected = (updated, post, comb, y)
            verify(outputs, expected)
        expected_shapes = (
            (*outer, streams, hidden),
            (*outer, streams, 1),
            (*outer, streams, streams),
            (*outer, hidden),
        )
        assert tuple(t.shape for t in outputs) == expected_shapes


@pytest.mark.parametrize("tokens,hidden", [(129, 4096), (513, 521)])
@pytest.mark.parametrize("disable_tle", [False, True])
@pytest.mark.parametrize("implementation", ["auto", "triton"])
@pytest.mark.megamhc
def test_mega_mhc_without_native_helpers(
    tokens, hidden, disable_tle, implementation, monkeypatch
):
    mod = importlib.import_module("flaggems_vllm.runtime.backend._thead.ops.megamhc")

    def unexpected_bitcode():
        pytest.fail("Triton implementation requested device bitcode")

    monkeypatch.setattr(
        mod,
        "device_bitcode",
        unexpected_bitcode if implementation == "triton" else lambda: None,
    )
    if disable_tle:
        monkeypatch.setattr(mod, "has_triton_tle_attrs", lambda attrs: False)
        original_import = builtins.__import__

        def no_tle_module(name, *args, **kwargs):
            if name.endswith("megamhc_tle"):
                pytest.fail("basic fusion or weight packing imported the TLE module")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_tle_module)
    args = make_inputs(tokens, hidden, 4)
    packed = gems.pack_mega_mhc_weights(args[4])
    verify(
        candidate(args, packed_fn=packed, implementation=implementation),
        reference(args),
    )


@pytest.mark.parametrize("implementation", ["auto", "triton"])
@pytest.mark.parametrize("tokens,hidden", [(129, 4104), (513, 521)])
@pytest.mark.parametrize("special", ["zero", "nan", "inf"])
@pytest.mark.megamhc
def test_mega_mhc_special_values(implementation, tokens, hidden, special):
    args = make_inputs(tokens, hidden, 4)
    if special == "zero":
        args[0].zero_()
        args[1].zero_()
    else:
        args[0][tokens // 2, 0] = float(special)
    verify(candidate(args, implementation=implementation), reference(args))


@pytest.mark.parametrize("tokens", [0, 1])
@pytest.mark.parametrize("invalid", ["packed", "post", "norm", "fn", "repeats"])
@pytest.mark.megamhc
def test_mega_mhc_invalid_inputs(tokens, invalid):
    args = list(make_inputs(tokens, 127, 4))
    kwargs = {}
    eps = EPS
    if invalid == "packed":
        kwargs["packed_fn"] = torch.empty(1, device=args[0].device)
    elif invalid == "post":
        args[2] = args[2].to(torch.bfloat16)
    elif invalid == "norm":
        args[-1] = args[-1][:-1]
    elif invalid == "fn":
        args[4] = args[4].t()
    else:
        eps = (*EPS[:-1], 0)
    with pytest.raises(ValueError):
        gems.mega_mhc(*args[:-1], *eps, norm_weight=args[-1], **kwargs)


@pytest.mark.parametrize("implementation", ["auto", "triton"])
@pytest.mark.parametrize("prepacked", [False, True])
@pytest.mark.megamhc
def test_mega_mhc_concurrent_graphs(implementation, prepacked):
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    inputs = [make_inputs(1029, 768, 4), make_inputs(2049, 521, 2)]
    packed = [gems.pack_mega_mhc_weights(a[4]) if prepacked else None for a in inputs]
    runs = []
    for stream, args, weights in zip(streams, inputs, packed, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            candidate(args, packed_fn=weights, implementation=implementation)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = candidate(
                    args, packed_fn=weights, implementation=implementation
                )
            runs.append((graph, outputs))
    torch.cuda.synchronize()
    for _ in range(8):
        for stream, args, (graph, _) in zip(streams, inputs, runs, strict=True):
            with torch.cuda.stream(stream):
                args[0].mul_(-0.75)
                if not prepacked:
                    args[4].neg_()
                graph.replay()
        torch.cuda.synchronize()
    for args, (_, outputs) in zip(inputs, runs, strict=True):
        rows = [0, args[0].shape[0] // 2, args[0].shape[0] - 1]
        sampled = tuple(value[rows] if i < 4 else value for i, value in enumerate(args))
        verify(tuple(value[rows] for value in outputs), reference(sampled))
