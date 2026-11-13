# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Exercise the public package entry, never an experimental candidate loader."""

import importlib
import inspect
import os
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from baseline import matmul_persistent as workload

impl = importlib.import_module("flaggems_vllm.ops.matmul_persistent")
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


@pytest.fixture(scope="module")
def gpu():
    available = torch.cuda.is_available() and torch.version.hip is None
    if not available:
        if os.environ.get("MATMUL_PERSISTENT_REQUIRE_CUDA") == "1":
            pytest.fail("NVIDIA CUDA is required for acceptance")
        pytest.skip("NVIDIA CUDA unavailable")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("SM80 or newer required")
    return torch.device("cuda", torch.cuda.current_device())


def tensors(a_shape, b_shape, dtype, device, *, signed=True):
    gen = torch.Generator(device=device).manual_seed(42)
    factory = torch.randn if signed else torch.rand
    return (
        factory(tuple(a_shape), dtype=dtype, device=device, generator=gen),
        factory(tuple(b_shape), dtype=dtype, device=device, generator=gen),
    )


def assert_case(a, b, out=None):
    before = (a.clone(), b.clone())
    expected = workload.native()(a, b)
    actual = gems.matmul_persistent(a, b, out=out)
    assert actual.shape == expected.shape and actual.dtype == a.dtype
    assert actual.device == a.device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if out is not None:
        assert actual is out
    for x, saved in zip((a, b), before):
        torch.testing.assert_close(x, saved, rtol=0, atol=0, equal_nan=True)
    return actual


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("binding", workload.grid())
@pytest.mark.parametrize("dtype", workload.DTYPES)
def test_matmul_persistent(binding, dtype, gpu):
    a, b = workload.build_inputs(binding, dtype, gpu)[0]
    actual = assert_case(a, b)
    tol = 0.1 if dtype == torch.bfloat16 else 0.01
    torch.testing.assert_close(actual, torch.matmul(a, b), rtol=tol, atol=tol)


@pytest.mark.matmul_persistent
def test_public_contract(gpu):
    expected = (
        Path(__file__).resolve().parents[1]
        / "src/flaggems_vllm/ops/matmul_persistent.py"
    )
    assert Path(inspect.getfile(gems.matmul_persistent)).resolve() == expected
    assert gems.matmul_persistent is impl.matmul_persistent
    assert str(inspect.signature(gems.matmul_persistent)) == "(a, b, *, out=None)"
    assert "matmul_persistent" in gems._LIBRARY_ONLY_OPS
    assert "matmul_persistent" not in dict(gems._FULL_CONFIG)


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("m", [1, 31, 32, 33, 64, 128, 129])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batched", [False, True])
def test_boundaries_and_signed_values(m, dtype, batched, gpu):
    shapes = ((2, m, 64), (2, 64, 32)) if batched else ((m, 64), (64, 32))
    assert_case(*tensors(*shapes, dtype, gpu))


@pytest.mark.matmul_persistent
@pytest.mark.parametrize(
    "k,n", [(16, 16), (32, 32), (64, 16), (65, 32), (128, 33), (256, 128)]
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_reduction_and_general_fallback(k, n, dtype, gpu, monkeypatch):
    if k not in (16, 32, 64) or n not in (16, 32):

        def forbidden(*args, **kwargs):
            pytest.fail("New batch preparation escaped its M/N/K domain")

        monkeypatch.setattr(impl, "_optimized_bmm", forbidden)
    assert_case(*tensors((1, 2, 33, k), (4, 2, k, n), dtype, gpu))


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("dtype", workload.DTYPES)
@pytest.mark.parametrize(
    "layout", ["transpose_b", "stride_a", "offset", "strided_out", "batch_permute"]
)
def test_layouts_offsets_and_out(dtype, layout, gpu):
    a, b = tensors((1, 2, 32, 64), (4, 2, 64, 16), dtype, gpu)
    out = None
    if layout == "transpose_b":
        b = b.transpose(-1, -2).contiguous().transpose(-1, -2)
    elif layout == "stride_a":
        backing = torch.empty((1, 2, 32, 128), dtype=dtype, device=gpu)
        backing[..., ::2].copy_(a)
        a = backing[..., ::2]
    elif layout == "offset":
        backing = torch.empty(a.numel() + 7, dtype=dtype, device=gpu)
        backing[7:].copy_(a.reshape(-1))
        a = backing[7:].view(a.shape)
    elif layout == "strided_out":
        out = torch.empty((4, 2, 32, 32), dtype=dtype, device=gpu)[..., ::2]
    elif layout == "batch_permute":
        a, b = tensors((2, 4, 32, 64), (4, 2, 64, 16), dtype, gpu)
        a = a.transpose(0, 1)
    assert_case(a, b, out=out)


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("binding", workload.grid())
@pytest.mark.parametrize("dtype", workload.DTYPES)
def test_repeated_batch_invariance(binding, dtype, gpu):
    a, b = tensors(binding["a_shape"], binding["b_shape"], dtype, gpu)
    single = gems.matmul_persistent(a, b)
    shape = tuple(torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
    # Expand both operands to the same batch rank BEFORE adding the repeat axis.
    aa = (
        a.expand(shape + tuple(a.shape[-2:]))
        .unsqueeze(0)
        .expand((2,) + shape + tuple(a.shape[-2:]))
    )
    bb = (
        b.expand(shape + tuple(b.shape[-2:]))
        .unsqueeze(0)
        .expand((2,) + shape + tuple(b.shape[-2:]))
    )
    repeated = gems.matmul_persistent(aa, bb)
    assert torch.equal(single, repeated[0]) and torch.equal(single, repeated[1])


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("dtype", workload.DTYPES)
@pytest.mark.parametrize("m", [1, 31, 32, 33, 128, 129])
def test_row_invariance(dtype, m, gpu):
    a, b = tensors((m, 64), (64, 16), dtype, gpu)
    single = gems.matmul_persistent(a[:1], b)
    full = gems.matmul_persistent(a, b)
    assert torch.equal(single[0], full[0])


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((0, 64), (64, 16)),
        ((2, 32, 64), (2, 64, 0)),
        ((0, 32, 64), (1, 64, 16)),
        ((2, 32, 0), (2, 0, 16)),
    ],
)
def test_empty_and_zero_reduction(a_shape, b_shape, dtype, gpu):
    a, b = tensors(a_shape, b_shape, dtype, gpu)
    actual = gems.matmul_persistent(a, b)
    torch.testing.assert_close(actual, torch.matmul(a, b), rtol=0, atol=0)


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("kind", ["shape", "dtype", "overlap", "alias"])
def test_invalid_out(kind, gpu):
    a, b = tensors((32, 64), (64, 16), torch.float16, gpu)
    out = torch.empty((32, 16), dtype=a.dtype, device=gpu)
    if kind == "shape":
        out = out[:1]
    elif kind == "dtype":
        out = out.float()
    elif kind == "overlap":
        out = out[:1].expand(32, 16)
    else:
        out = a[:, :16]
    with pytest.raises(ValueError):
        gems.matmul_persistent(a, b, out=out)


@pytest.mark.matmul_persistent
@pytest.mark.parametrize(
    "kind", ["rank", "inner", "broadcast", "dtype", "integer", "grad"]
)
def test_unsupported_inputs(kind, gpu):
    a, b = tensors((2, 32, 64), (2, 64, 16), torch.float16, gpu)
    if kind == "rank":
        a = a.reshape(-1)
    elif kind == "inner":
        b = b[:, :32]
    elif kind == "broadcast":
        b = b[:1].expand(3, 64, 16)
    elif kind == "dtype":
        b = b.float()
    elif kind == "integer":
        a, b = a.int(), b.int()
    elif kind == "grad":
        a.requires_grad_(True)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        gems.matmul_persistent(a, b)


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("dtype", workload.DTYPES)
def test_special_values(dtype, gpu):
    a, b = tensors((32, 64), (64, 16), dtype, gpu)
    a[0, 0], a[1, 0], a[2, 0], a[3, 0] = float("nan"), float("inf"), -float("inf"), -0.0
    assert_case(a, b)


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("dtype", workload.DTYPES)
def test_no_copy_launch_storage(dtype, gpu, monkeypatch):
    if not impl._device_policy(gpu.index)["h20"]:
        pytest.skip("H20-specific path")
    original = impl._broadcast_bmm
    calls = []
    a, b = tensors((1, 2, 32, 64), (4, 2, 64, 16), dtype, gpu)

    def observe(aa, bb, shape, policy):
        assert aa.untyped_storage().data_ptr() == a.untyped_storage().data_ptr()
        assert bb.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        calls.append(tuple(shape))
        return original(aa, bb, shape, policy)

    monkeypatch.setattr(impl, "_broadcast_bmm", observe)
    assert_case(a, b)
    assert calls == [(4, 2)]


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("dtype", workload.DTYPES)
@pytest.mark.parametrize("batched", [False, True])
def test_cuda_graph(dtype, batched, gpu):
    shapes = ((1, 2, 32, 64), (4, 2, 64, 16)) if batched else ((32, 64), (64, 16))
    a, b = tensors(*shapes, dtype, gpu)
    reference = assert_case(a, b)
    side = torch.cuda.Stream(device=gpu)
    side.wait_stream(torch.cuda.current_stream(gpu))
    with torch.cuda.stream(side):
        for _ in range(3):
            gems.matmul_persistent(a, b)
    torch.cuda.current_stream(gpu).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = gems.matmul_persistent(a, b)
    graph.replay()
    torch.cuda.synchronize(gpu)
    assert torch.equal(result, reference)


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("m", [8, 32, 256, 2048])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_vllm_tuned_bucket_invariance(m, transpose_b, gpu):
    a, b = tensors((m, 2048), (2048, 2048), torch.bfloat16, gpu)
    if transpose_b:
        b = b.t()
    actual = assert_case(a, b)
    assert torch.equal(actual[0], gems.matmul_persistent(a[:1], b)[0])
