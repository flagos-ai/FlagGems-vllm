# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm
from baseline import triton_reshape_and_cache_flash_per_token_head_quant as cases
from baseline.triton_reshape_and_cache_flash import backing
from flaggems_vllm import (
    triton_reshape_and_cache_flash_per_token_head_quant as cache_kv,
)


def _inputs(shape, dtype, cache, **binding):
    binding.update(zip(cases.SHAPE_FIELDS[:5], shape))
    binding.setdefault("layout", "strided")
    binding.setdefault("slots", "random")
    return cases.build_inputs(dict(binding, cache=cache), dtype, flaggems_vllm.device)[
        0
    ]


def _check(args):
    expected = cases.reference(args)
    strides = [t.stride() for t in args]
    offsets = [t.storage_offset() for t in args]
    addresses = [t.data_ptr() for t in args]
    source_snapshots = {i: backing(args[i]).cpu().clone() for i in (0, 1, 6)}
    assert cache_kv(*args) is None
    cases.assert_outputs(args, expected, scale_rtol=2e-7)
    for i, source in source_snapshots.items():
        torch.testing.assert_close(backing(args[i]).cpu(), source, rtol=0, atol=0)
    assert [t.stride() for t in args] == strides
    assert [t.storage_offset() for t in args] == offsets
    assert [t.data_ptr() for t in args] == addresses


@pytest.mark.parametrize("dtype", cases.DTYPES)
@pytest.mark.parametrize("cache", ["int8", "fp8"])
@pytest.mark.parametrize(
    "shape,slot_dtype,slot_stride",
    [
        ((0, 1, 17, 33, 7), torch.int64, 1),
        ((1, 1, 1, 1, 7), torch.int32, 2),
        ((17, 3, 80, 128, 7), torch.int64, 1),
        ((21, 5, 160, 96, 8), torch.int32, 2),
        ((21, 5, 96, 96, 8), torch.int64, 1),
        ((8, 3, 32, 512, 7), torch.int64, 1),
        ((8, 3, 512, 32, 7), torch.int64, 1),
    ],
)
@pytest.mark.triton_reshape_and_cache_flash_per_token_head_quant
def test_strided_cache_writes(shape, slot_dtype, slot_stride, dtype, cache):
    _check(_inputs(shape, dtype, cache, slot_dtype=slot_dtype, slot_stride=slot_stride))


@pytest.mark.parametrize("cache", ["int8", "fp8"])
@pytest.mark.triton_reshape_and_cache_flash_per_token_head_quant
def test_all_negative_slots(cache):
    args = _inputs((8, 3, 80, 128, 7), torch.float16, cache, slots="negative")
    args[6][0] = -(2**32)
    _check(args)


@pytest.mark.parametrize("cache", ["int8", "fp8"])
@pytest.mark.triton_reshape_and_cache_flash_per_token_head_quant
def test_reused_layout_writes_to_new_buffers(cache):
    shape = (2, 3, 80, 128, 7)
    first = _inputs(shape, torch.float16, cache)
    second = _inputs(shape, torch.float16, cache, slots="negative")
    second[0].fill_(1.0)
    second[1].fill_(-4.0)
    second[6][1] = 7
    _check(first)
    saved = [out.cpu().clone() for out in cases.snapshot_outputs(first)]
    _check(second)
    for out, expected in zip(cases.snapshot_outputs(first), saved):
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", cases.DTYPES)
@pytest.mark.parametrize("cache", ["int8", "fp8"])
@pytest.mark.triton_reshape_and_cache_flash_per_token_head_quant
def test_scale_floor_and_rounding(dtype, cache):
    args = _inputs((6, 3, 80, 128, 3), dtype, cache)
    maximum = {
        torch.int8: 127.0,
        torch.float8_e4m3fn: 448.0,
        torch.float8_e4m3fnuz: 224.0,
    }[args[2].dtype]
    subnormal_midpoint = 2 ** (-11 if maximum == 224.0 else -10)
    edges = torch.tensor(
        [
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            maximum,
            -maximum,
            1.0625,
            -1.0625,
            1.1875,
            -1.1875,
            subnormal_midpoint,
            -subnormal_midpoint,
            1.5 * subnormal_midpoint,
            -1.5 * subnormal_midpoint,
        ]
    )
    for i, source in enumerate(args[:2]):
        rows = torch.zeros(source.shape)
        rows[1, i, : len(edges)] = edges
        rows[2, 2 - i, : len(edges)] = edges * 4
        rows[3, i, : len(edges)] = edges * 2**-24
        rows[4, 0, : len(edges)] = edges.flip(0)
        rows[4, 1, : len(edges)] = edges * (i + 1)
        rows[4, 2, : len(edges)] = edges * 0.25
        source.copy_(rows.to(source))
    _check(args)

    # For the scale=1 row, pin the signed rounding and format-specific zero.
    page, offset = divmod(int(args[6][1].item()), 3)
    raw = args[2].view(torch.uint8)[page, offset, 0].cpu()
    assert float(args[4][page, offset, 0].item()) == 1.0
    if cache == "int8":
        assert raw[2:8].view(torch.int8).tolist() == [1, -1, 2, -2, 127, -127]
    else:
        assert raw[1].item() == (128 if maximum == 448.0 else 0)


@pytest.mark.parametrize(
    "fp8_dtype,qmax,format_max",
    [(torch.float8_e4m3fn, 448, 448), (torch.float8_e4m3fnuz, 224, 240)],
)
@pytest.mark.triton_reshape_and_cache_flash_per_token_head_quant
def test_fp8_range_matches_cache_encoding(fp8_dtype, qmax, format_max):
    assert torch.finfo(fp8_dtype).max == format_max
    args = _inputs((1, 1, 5, 7, 4), torch.float16, "fp8", fp8_dtype=fp8_dtype)
    for source in args[:2]:
        row = torch.zeros(source.shape)
        row[..., :4] = torch.tensor([qmax, -qmax, 0.0, -0.0])
        source.copy_(row.to(source))
    _check(args)
    assert args[4][0, 0, 0].item() == 1.0
    assert args[5][0, 0, 0].item() == 1.0
    high = 0x7E
    zero = 0x80 if qmax == 448 else 0
    for cache in args[2:4]:
        assert cache.view(torch.uint8)[0, 0, 0, :4].cpu().tolist() == [
            high,
            high | 0x80,
            0,
            zero,
        ]


@pytest.mark.parametrize("cache", ["int8", "fp8"])
@pytest.mark.triton_reshape_and_cache_flash_per_token_head_quant
def test_invalid_metadata(cache):
    args = _inputs((3, 3, 80, 128, 7), torch.float16, cache)
    with pytest.raises(ValueError, match="slot_mapping"):
        cache_kv(*args[:6], args[6][:-1])
    with pytest.raises(ValueError, match="scale caches"):
        cache_kv(*args[:4], args[4][:, :, :1], *args[5:])
    with pytest.raises(ValueError, match="data caches"):
        cache_kv(*args[:3], args[3].view(torch.uint8), *args[4:])
    with pytest.raises(ValueError, match="unit stride"):
        cache_kv(args[0].transpose(1, 2).contiguous().transpose(1, 2), *args[1:])
