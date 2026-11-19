# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Exact iHC post checks against an independent, two-step CPU FP32 oracle."""

import itertools

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.ihc import ihc_post
from flaggems_vllm.runtime import torch_device_fn

DTYPES = (torch.float16, torch.bfloat16)
SHAPES = (
    (0, 3, 513),
    (1, 1, 1),
    (1, 3, 513),
    (7, 1, 1),
    (7, 3, 1),
    (7, 3, 513),
    (1, 4, 513),
    (1, 4, 6144),
    (7, 4, 6144),
    (64, 4, 6144),
    (256, 4, 6144),
    (1024, 4, 6144),
)
LAYOUTS = tuple(itertools.product((False, True), repeat=3))
ENTRY_NAMES = ("canonical", "legacy")
pytestmark = pytest.mark.triton_ihc_post


@pytest.fixture
def device():
    selected = torch.device(flaggems_vllm.device)
    assert selected.type not in ("cpu", "meta"), "accelerator runtime required"
    return selected


@pytest.fixture(params=ENTRY_NAMES)
def entry(request):
    return flaggems_vllm.triton_ihc_post if request.param == "canonical" else ihc_post


def _oracle(x, residual, post):
    # Separate eager CPU operations force the required FP32 rounding boundary.
    x32 = x.detach().cpu().float()
    residual32 = residual.detach().cpu().float()
    post32 = post.detach().cpu().float()
    product = torch.mul(post32.unsqueeze(-1), x32.unsqueeze(1))
    summed = torch.add(product, residual32)
    return summed.to(x.dtype)


def _bytes(value):
    return value.detach().cpu().contiguous().view(torch.uint8).clone()


def _view(shape, dtype, device, seed, strided=False):
    parent_shape = (*shape[:-1], 2 * shape[-1] + 2) if strided else shape
    owner = torch.randn(parent_shape, generator=torch.Generator().manual_seed(seed))
    owner = owner.to(dtype=dtype, device=device)
    return (owner[..., 1:-1:2] if strided else owner), owner


def _inputs(shape, dtype, device, layout=(False, False, False)):
    n, hc, d = shape
    values = [
        _view(s, dt, device, seed, step)
        for s, dt, seed, step in zip(
            ((n, d), (n, hc, d), (n, hc)),
            (dtype, dtype, torch.float32),
            (13, 29, 43),
            layout,
        )
    ]
    return tuple(v[0] for v in values), tuple(v[1] for v in values)


def _assert_exact(actual, expected):
    actual = actual.detach().cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


def _check(entry, args, owners=None):
    owners = args if owners is None else owners
    before = tuple(_bytes(owner) for owner in owners)
    expected = _oracle(*args)
    out = entry(*args)
    assert isinstance(out, torch.Tensor)
    assert out.shape == args[1].shape
    assert out.dtype == args[0].dtype
    assert out.device == args[0].device
    assert out.is_contiguous()
    assert all(out is not arg for arg in args)
    if out.numel():
        assert all(
            out.untyped_storage().data_ptr() != arg.untyped_storage().data_ptr()
            for arg in args
        ), "output aliases input storage"
    _assert_exact(out, expected)
    again = entry(*args)
    assert again is not out
    if out.numel():
        assert out.untyped_storage().data_ptr() != again.untyped_storage().data_ptr()
    _assert_exact(again, expected)
    for owner, saved in zip(owners, before):
        assert torch.equal(_bytes(owner), saved), "input storage or canary was modified"
    return out.detach().cpu()


@pytest.mark.triton_ihc_post
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_triton_ihc_post_shapes(shape, dtype, device, entry):
    args, owners = _inputs(shape, dtype, device)
    _check(entry, args, owners)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_triton_ihc_post_layouts(layout, dtype, device, entry):
    args, owners = _inputs((7, 3, 513), dtype, device, layout)
    _check(entry, args, owners)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gate", [0.0, -0.0, -3.0, 2.0, 1e20, -1e20])
def test_triton_ihc_post_gates(gate, dtype, device, entry):
    args, owners = _inputs((7, 4, 513), dtype, device)
    args[2].fill_(gate)
    _check(entry, args, owners)


@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_ihc_post_special_values(dtype, device, entry):
    x = torch.tensor(
        [[float("nan"), float("inf"), -float("inf"), 0, -0.0, 1, -1, 2]],
        dtype=dtype,
        device=device,
    )
    residual = torch.tensor(
        [[[1, -float("inf"), float("inf"), -0.0, 0, float("nan"), 1, -1]] * 4],
        dtype=dtype,
        device=device,
    )
    post = torch.tensor([[0, -2, float("inf"), float("nan")]], device=device)
    _check(entry, (x, residual, post))


@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_ihc_post_signed_zero(dtype, device, entry):
    x = torch.tensor([[0.0, -0.0, 0.0, -0.0]], dtype=dtype, device=device)
    residual = torch.tensor([[[-0.0, -0.0, 0.0, 0.0]] * 3], dtype=dtype, device=device)
    # Nonzero gates isolate the FP32 multiplication/addition signed-zero rules.
    post = torch.tensor([[1.0, -1.0, 2.0]], dtype=torch.float32, device=device)
    _check(entry, (x, residual, post))
    # A negative-zero gate and negative-zero residual must keep a negative zero.
    _check(
        entry,
        (
            torch.ones((1, 513), dtype=dtype, device=device),
            torch.full((1, 3, 513), -0.0, dtype=dtype, device=device),
            torch.tensor([[-0.0, 0.0, -0.0]], dtype=torch.float32, device=device),
        ),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_ihc_post_dtype_boundaries(dtype, device, entry):
    if dtype == torch.float16:
        gates = [
            1 + 2**-11,
            1 + 3 * 2**-11,
            0.723388671875,
            2**-25,
            3 * 2**-25,
            2**-14,
            65504.0,
            65520.0,
        ]
    else:
        gates = [
            1 + 2**-8,
            1 + 3 * 2**-8,
            2**-134,
            3 * 2**-134,
            2**-126,
            (2 - 2**-7) * 2**127,
            torch.finfo(torch.float32).max,
        ]
    gates = gates + [-value for value in gates]
    channels = len(gates)
    x = torch.ones((1, 513), dtype=dtype, device=device)
    residual = torch.zeros((1, channels, 513), dtype=dtype, device=device)
    post = torch.tensor([gates], dtype=torch.float32, device=device)
    _check(entry, (x, residual, post))


def _fma_fixture(dtype):
    # Exact FP64 arithmetic emulates fused FP32 here (the product has <=35 bits).
    value = 1025 if dtype == torch.float16 else 129
    x = torch.tensor([[value]], dtype=dtype)
    residual = -x.reshape(1, 1, 1)
    post = torch.tensor([[1 + 2**-23]], dtype=torch.float32)
    expected = _oracle(x, residual, post)
    fused = post.double().unsqueeze(-1) * x.double().unsqueeze(1) + residual.double()
    fused = fused.float().to(dtype)
    assert not torch.equal(expected, fused), "fixture must distinguish FMA after cast"
    return x, residual, post


@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_ihc_post_fma_rounding(dtype, device, entry):
    _check(entry, tuple(value.to(device) for value in _fma_fixture(dtype)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("all_shared", [False, True])
def test_triton_ihc_post_input_aliases(dtype, all_shared, device, entry):
    owner = torch.randn((7, 3, 514), dtype=dtype, device=device)
    residual = owner[..., :513]
    x = residual[:, 0, :]
    if all_shared:
        post = owner.reshape(-1).view(torch.float32)[:21].reshape(7, 3)
        owners = (owner,)
    else:
        post = torch.randn((7, 3), device=device)
        owners = (owner, post)
    _check(entry, (x, residual, post), owners)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "all_strided"])
def test_triton_ihc_post_model_layouts(dtype, layout, device):
    args, owners = _inputs((7, 4, 6144), dtype, device, (layout == "all_strided",) * 3)
    _check(flaggems_vllm.triton_ihc_post, args, owners)


@pytest.mark.parametrize("argument", range(3))
def test_triton_ihc_post_non_tensor(argument, device):
    args, _ = _inputs((1, 3, 5), torch.float16, device)
    args = list(args)
    args[argument] = 1.0
    with pytest.raises(TypeError):
        flaggems_vllm.triton_ihc_post(*args)


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 3, 5), (1, 3, 5), (1, 3)),
        ((1, 5), (1, 15), (1, 3)),
        ((1, 5), (1, 3, 5), (3,)),
        ((1, 5), (2, 3, 5), (1, 3)),
        ((1, 5), (1, 4, 5), (1, 3)),
        ((1, 5), (1, 3, 7), (1, 3)),
        ((1, 5), (1, 3, 5), (2, 3)),
        ((1, 0), (1, 3, 0), (1, 3)),
        ((1, 5), (1, 0, 5), (1, 0)),
        ((0, 0), (0, 3, 0), (0, 3)),
        ((0, 5), (0, 0, 5), (0, 0)),
    ],
)
def test_triton_ihc_post_invalid_shapes(shapes, device):
    args = [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in zip(shapes, (torch.float16, torch.float16, torch.float32))
    ]
    with pytest.raises(ValueError):
        flaggems_vllm.triton_ihc_post(*args)


@pytest.mark.parametrize("argument", range(3))
@pytest.mark.parametrize("dtype", [torch.int32, torch.float64, torch.float32])
def test_triton_ihc_post_invalid_dtype(argument, dtype, device):
    if argument == 2 and dtype == torch.float32:
        dtype = torch.float16
    args, _ = _inputs((1, 3, 5), torch.float16, device)
    args = list(args)
    # Some adapters silently lower a float64 conversion to float32. Allocate
    # metadata directly so this case actually presents the requested dtype.
    args[argument] = torch.empty(args[argument].shape, dtype=dtype, device=device)
    assert args[argument].dtype == dtype
    with pytest.raises(TypeError):
        flaggems_vllm.triton_ihc_post(*args)


def test_triton_ihc_post_mismatched_dtype(device):
    args, _ = _inputs((1, 3, 5), torch.float16, device)
    with pytest.raises(TypeError):
        flaggems_vllm.triton_ihc_post(args[0], args[1].bfloat16(), args[2])


@pytest.mark.parametrize("host", ["cpu", "meta"])
def test_triton_ihc_post_unsupported_device(host):
    args, _ = _inputs((1, 3, 5), torch.float16, host)
    with pytest.raises(ValueError):
        flaggems_vllm.triton_ihc_post(*args)


@pytest.mark.parametrize("argument", range(3))
def test_triton_ihc_post_device_mismatch(argument, device):
    args, _ = _inputs((1, 3, 5), torch.float16, device)
    args = list(args)
    args[argument] = args[argument].cpu()
    with pytest.raises(ValueError):
        flaggems_vllm.triton_ihc_post(*args)


def test_triton_ihc_post_cross_accelerator(device):
    if torch_device_fn.device_count() < 2:
        pytest.skip("cross-device rejection requires two visible accelerator devices")
    current = device.index
    if current is None:
        current = torch_device_fn.current_device()
    other = (current + 1) % torch_device_fn.device_count()
    args, _ = _inputs((1, 3, 5), torch.float16, torch.device(device.type, current))
    with pytest.raises(ValueError):
        flaggems_vllm.triton_ihc_post(
            args[0], args[1].to(torch.device(device.type, other)), args[2]
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset", [0, 1])
def test_triton_ihc_post_fresh_buffers_and_streams(dtype, offset, device, entry):
    # Keep prior allocations alive while exercising an already compiled shape.
    # Offset-one inputs are contiguous but have a different pointer alignment.
    live = []
    streams = [torch_device_fn.current_stream(), torch_device_fn.Stream()]
    for iteration in range(4):
        with torch_device_fn.stream(streams[iteration % 2]):
            args, owners = [], []
            for shape, value_dtype, fill in (
                ((7, 513), dtype, iteration + 1),
                ((7, 3, 513), dtype, -iteration),
                ((7, 3), torch.float32, iteration + 0.25),
            ):
                count = 1
                for dimension in shape:
                    count *= dimension
                owner = torch.full(
                    (count + offset,), fill, dtype=value_dtype, device=device
                )
                owners.append(owner)
                args.append(owner[offset:].view(shape))
            live.append(owners)
            _check(entry, tuple(args), tuple(owners))
