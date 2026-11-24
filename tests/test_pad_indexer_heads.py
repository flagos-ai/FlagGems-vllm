# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent byte oracle, layout/alias contracts, and compiled-native checks."""

import math

import pytest
import torch

import flaggems_vllm as gems
from baseline import pad_indexer_heads as workload

COPY_DTYPES = workload.DTYPES
EXTRA_DTYPES = [
    torch.float8_e5m2,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float64,
    torch.complex64,
    torch.complex128,
]


def storage_input(shape, dtype, layout="contiguous"):
    """Transfer arbitrary bits, including every FP8 encoding, without FP casts."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    source_shape = list(shape)
    if layout == "row_gap":
        source_shape[0] *= 2
    elif layout == "tail_gap":
        source_shape[-1] *= 2
    elif layout == "transpose":
        source_shape[0], source_shape[1] = source_shape[1], source_shape[0]
    elif layout == "broadcast":
        source_shape[0] = 1
    nbytes = math.prod(source_shape) * itemsize
    generator = torch.Generator().manual_seed(20261106)
    raw = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=generator)
    # Exhaustive prefix includes positive/negative zero and FP8 NaN/Inf codes.
    raw[: min(nbytes, 256)] = torch.arange(min(nbytes, 256), dtype=torch.uint8)
    if dtype == torch.bool:
        raw &= 1
    x = raw.to(gems.device).view(dtype).reshape(source_shape)
    if layout == "row_gap":
        x = x[::2]
    elif layout == "tail_gap":
        x = x[..., ::2]
    elif layout == "transpose":
        x = x.transpose(0, 1)
    elif layout == "broadcast":
        x = x.expand(shape)
    return x


def byte_view(x):
    # Copy physical uint8 storage, then reconstruct logical strides on CPU.
    # Device-side FP8 contiguous/slice can dispatch to unsupported AiCPU ops.
    itemsize = x.element_size()
    storage = x.untyped_storage()
    raw = (
        torch.empty(0, dtype=torch.uint8, device=x.device)
        .set_(storage, 0, (storage.nbytes(),), (1,))
        .cpu()
    )
    return raw.as_strided(
        (*x.shape, itemsize),
        (*[stride * itemsize for stride in x.stride()], 1),
        x.storage_offset() * itemsize,
    )


def assert_contract(op, x, pad):
    before = byte_view(x).clone()
    expected = torch.cat(
        [
            before,
            torch.zeros(
                (x.shape[0], pad, *x.shape[2:], x.element_size()), dtype=torch.uint8
            ),
        ],
        dim=1,
    )
    out = op(x, pad)
    assert out.shape == (x.shape[0], x.shape[1] + pad, *x.shape[2:])
    assert out.dtype == x.dtype and out.device == x.device
    assert out.is_contiguous()
    assert out is not x
    assert out.untyped_storage()._cdata != x.untyped_storage()._cdata
    if out.numel() and x.numel():
        assert out.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    actual = byte_view(out)
    if not torch.equal(actual, expected):
        mismatch = (actual.reshape(-1) != expected.reshape(-1)).nonzero().flatten()
        raise AssertionError(
            f"storage mismatch: {mismatch.numel()} bytes; "
            f"first offsets {mismatch[:8].tolist()}"
        )
    assert torch.equal(byte_view(x), before)
    return out


CASES = [
    ((t, h, *tail), (0, 1, max(0, 32 - h), 64 - h, 3)[(ti + hi + di) % 5])
    for ti, t in enumerate((0, 1, 7, 128))
    for hi, h in enumerate((1, 16, 31, 32, 63))
    for di, tail in enumerate(((), (1,), (7,), (128,), (3, 5)))
]
CASES += [((7, 0), 3), ((7, 0, 7), 3), ((7, 16, 0), 16), ((0, 0, 0), 0)]


@pytest.mark.parametrize("dtype", COPY_DTYPES)
@pytest.mark.parametrize("shape,pad", CASES)
@pytest.mark.pad_indexer_heads
def test_pad_indexer_heads(shape, pad, dtype):
    assert_contract(gems.pad_indexer_heads, storage_input(shape, dtype), pad)


@pytest.mark.parametrize("dtype", COPY_DTYPES)
@pytest.mark.parametrize("shape", [(7, 16), (7, 16, 7), (7, 16, 3, 5)])
@pytest.mark.parametrize("layout", ["row_gap", "tail_gap", "transpose", "broadcast"])
@pytest.mark.parametrize("pad", [0, 3])
@pytest.mark.pad_indexer_heads
def test_pad_indexer_heads_strides(shape, layout, pad, dtype):
    assert_contract(gems.pad_indexer_heads, storage_input(shape, dtype, layout), pad)


@pytest.mark.parametrize("dtype", COPY_DTYPES)
@pytest.mark.parametrize("tail", [(), (7,), (3, 5)])
@pytest.mark.pad_indexer_heads
def test_pad_indexer_heads_storage_offset(tail, dtype):
    x = storage_input((9, 16, *tail), dtype)[1:8]
    assert x.storage_offset() > 0
    assert_contract(gems.pad_indexer_heads, x, 3)


@pytest.mark.parametrize("dtype", COPY_DTYPES)
@pytest.mark.parametrize("tail", [(), (7,), (3, 5)])
@pytest.mark.pad_indexer_heads
def test_pad_indexer_heads_inner_offset(tail, dtype):
    shape = (7, 16, *tail)
    source_shape = (*shape[:-1], shape[-1] + 1)
    x = storage_input(source_shape, dtype)[..., 1:]
    assert x.storage_offset() == 1
    assert_contract(gems.pad_indexer_heads, x, 3)


@pytest.mark.parametrize("dtype", EXTRA_DTYPES)
@pytest.mark.parametrize("shape", [(7, 31), (7, 31, 9), (1, 31, 3, 5)])
@pytest.mark.parametrize("pad", [0, 2])
@pytest.mark.pad_indexer_heads
def test_pad_indexer_heads_copy_dtypes(shape, pad, dtype):
    assert_contract(
        gems.pad_indexer_heads, storage_input(shape, dtype, "tail_gap"), pad
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.pad_indexer_heads
def test_pad_indexer_heads_special_bits(dtype):
    x = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), -float("inf")], dtype=dtype
    )
    x = x.reshape(1, 1, 5).expand(7, 16, 5).contiguous().to(gems.device)
    assert_contract(gems.pad_indexer_heads, x, 1)


@pytest.mark.parametrize("shape,error", [((), IndexError), ((7,), RuntimeError)])
@pytest.mark.pad_indexer_heads
def test_pad_indexer_heads_wrong_rank(shape, error):
    with pytest.raises(error):
        gems.pad_indexer_heads(torch.empty(shape, device=gems.device), 1)


@pytest.mark.pad_indexer_heads
def test_pad_indexer_heads_invalid_pad():
    x = torch.empty((7, 16), device=gems.device)
    with pytest.raises(RuntimeError, match="negative"):
        gems.pad_indexer_heads(x, -1)
    with pytest.raises(TypeError):
        gems.pad_indexer_heads(x, 1.5)


@pytest.mark.pad_indexer_heads
def test_pad_indexer_heads_lazy_conjugate():
    x = torch.empty((7, 16), dtype=torch.complex64, device=gems.device).conj()
    with pytest.raises(NotImplementedError, match="lazy"):
        gems.pad_indexer_heads(x, 1)


@pytest.mark.parametrize("dtype", COPY_DTYPES)
@pytest.mark.parametrize("binding", workload.grid())
@pytest.mark.pad_indexer_heads
def test_pad_indexer_heads_native(binding, dtype):
    op = workload.native()
    if op is None:
        pytest.skip(f"native N/A: {workload.NATIVE_UNAVAILABLE_REASON}")
    args, kwargs = workload.build_inputs(binding, dtype, gems.device)
    assert_contract(op, *args, **kwargs)


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.pad_indexer_heads
def test_pad_indexer_heads_legacy_api(heads):
    # Retain the target-total-heads contract and old rank3 callable.
    x = torch.randn((7, 16, 18), device=gems.device)[..., ::2]
    actual = gems.pad_attention_heads(x, heads)
    expected = torch.cat((x.cpu(), torch.zeros(7, heads - 16, 9)), dim=1)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
