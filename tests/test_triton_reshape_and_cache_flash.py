# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Exact full-allocation mutation tests against independent CPU indexing."""

import importlib

import pytest
import torch

import flaggems_vllm
from baseline import triton_reshape_and_cache_flash as cases

pytestmark = pytest.mark.triton_reshape_and_cache_flash
_ORDINARY_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def check_case(binding, dtype, op=None):
    args, kwargs = cases.build_inputs(binding, dtype, flaggems_vllm.device)
    check_arguments(args, kwargs, op=op)


def check_arguments(args, kwargs, op=None, expected=None):
    if expected is None:
        expected = cases.reference(args)
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    snapshots = [cases.backing(a).view(torch.uint8).cpu().clone() for a in tensors]
    identities = [
        (a.data_ptr(), a.untyped_storage().data_ptr(), a.storage_offset(), a.stride())
        for a in tensors
    ]
    result = (op or flaggems_vllm.triton_reshape_and_cache_flash)(*args, **kwargs)
    flaggems_vllm.runtime.torch_device_fn.synchronize()
    assert result is None
    for index, tensor in enumerate(tensors):
        assert (
            tensor.data_ptr(),
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tensor.stride(),
        ) == identities[index]
        actual = cases.backing(tensor).view(torch.uint8).cpu()
        wanted = expected[index - 2] if index in (2, 3) else snapshots[index]
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", _ORDINARY_DTYPES)
@pytest.mark.parametrize("layout", ["flash", "head_major"])
@pytest.mark.parametrize("padding", [0, 13])
@pytest.mark.parametrize(
    "shape",
    [(0, 1, 1, 1), (1, 1, 63, 7), (7, 3, 80, 7), (17, 8, 128, 16), (128, 16, 128, 32)],
)
@pytest.mark.triton_reshape_and_cache_flash
def test_ordinary(dtype, layout, padding, shape):
    t, h, d, bs = shape
    check_case(
        dict(T=t, H=h, D=d, BS=bs, layout=layout, padding=padding, cache="auto"), dtype
    )


@pytest.mark.parametrize("dtype", _ORDINARY_DTYPES)
@pytest.mark.parametrize("layout", ["flash", "head_major"])
@pytest.mark.parametrize(
    "selector", ["auto", "float16", "bfloat16", "float32", "half", "float"]
)
def test_ordinary_selector(dtype, layout, selector):
    check_case(
        dict(T=7, H=3, D=80, BS=7, layout=layout, padding=11, selector=selector), dtype
    )


@pytest.mark.parametrize("layout", ["flash", "head_major"])
@pytest.mark.parametrize("slots,all_negative", [(0, False), (3, False), (7, True)])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_slot_prefix_and_negative(layout, slots, all_negative, slot_dtype):
    check_case(
        dict(
            T=7,
            H=3,
            D=80,
            BS=7,
            layout=layout,
            padding=13,
            slots=slots,
            all_negative=all_negative,
            slot_dtype=slot_dtype,
        ),
        torch.float16,
    )


@pytest.mark.parametrize("layout", ["flash", "head_major"])
@pytest.mark.parametrize("cache", ["fp8", "uint8"])
@pytest.mark.parametrize("dtype", _ORDINARY_DTYPES)
@pytest.mark.parametrize("scales", [(0.5, 2.0), (0.0, 2**-100), (1.0, 1.0), (3.0, 7.0)])
def test_fp8_global(layout, cache, dtype, scales):
    check_case(
        dict(
            T=17,
            H=3,
            D=80,
            BS=7,
            layout=layout,
            padding=13,
            cache=cache,
            edges=True,
            k_scale=scales[0],
            v_scale=scales[1],
        ),
        dtype,
    )


@pytest.mark.parametrize("layout", ["flash", "head_major"])
@pytest.mark.parametrize("cache", ["fp8", "uint8"])
@pytest.mark.parametrize("scale", [0.0, 0.5, 7.0])
def test_fp8_prequantized(layout, cache, scale):
    check_case(
        dict(
            T=17,
            H=3,
            D=80,
            BS=7,
            layout=layout,
            padding=13,
            cache=cache,
            prequantized=True,
            edges=True,
            k_scale=scale,
            v_scale=scale,
        ),
        torch.float16,
    )


@pytest.mark.parametrize("format", [torch.float8_e4m3fn, torch.float8_e4m3fnuz])
def test_all_fp8_codes_and_midpoints(format):
    # Auto can store either explicit software format regardless of device cast support.
    codes = torch.arange(256, dtype=torch.uint8)
    decoded = codes.view(format).float()
    finite = decoded[torch.isfinite(decoded)].sort().values.unique()
    mid = ((finite[:-1].double() + finite[1:].double()) / 2).float()
    values = torch.cat(
        (
            decoded,
            mid,
            torch.nextafter(mid, torch.full_like(mid, float("inf"))),
            torch.nextafter(mid, torch.full_like(mid, -float("inf"))),
        )
    )
    binding = dict(
        T=1, H=1, D=values.numel(), BS=1, layout="flash", cache="auto", padding=5
    )
    args, _ = cases.build_inputs(binding, format, flaggems_vllm.device)
    args = list(args)
    args[0] = values.to(flaggems_vllm.device).reshape(1, 1, -1)
    args[1] = args[0].clone()
    expected = cases.reference(args)
    assert flaggems_vllm.triton_reshape_and_cache_flash(*args) is None
    for cache, wanted in zip(args[2:4], expected):
        torch.testing.assert_close(
            cases.backing(cache).view(torch.uint8).cpu(), wanted, rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "problem",
    [
        "slot_length",
        "slot_dtype",
        "slot_stride",
        "source_stride",
        "cache_layout",
        "cache_stride",
        "scale_shape",
        "scale_dtype",
        "selector",
        "device",
        "cache_dtype",
    ],
)
def test_invalid_metadata(problem):
    args, _ = cases.build_inputs(
        dict(T=7, H=3, D=80, BS=7, layout="flash", padding=13),
        torch.float16,
        flaggems_vllm.device,
    )
    args = list(args)
    if problem == "slot_length":
        args[4] = torch.zeros(8, dtype=torch.int64, device=args[0].device)
    elif problem == "slot_dtype":
        args[4] = args[4].float()
    elif problem == "slot_stride":
        args[4] = torch.zeros(14, dtype=torch.int64, device=args[0].device)[::2]
    elif problem == "source_stride":
        args[0] = torch.empty((7, 3, 160), device=args[0].device, dtype=args[0].dtype)[
            ..., ::2
        ]
    elif problem == "cache_layout":
        args[2] = args[2].transpose(1, 2)
    elif problem == "cache_stride":
        args[3] = args[3].contiguous()
    elif problem == "scale_shape":
        args[6] = torch.ones(2, device=args[0].device)
    elif problem == "scale_dtype":
        args[6] = args[6].half()
    elif problem == "selector":
        args[5] = "int8"
    elif problem == "device":
        args[6] = args[6].cpu()
    else:
        args[2] = args[2].to(torch.int32)
        args[3] = args[3].to(torch.int32)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        flaggems_vllm.triton_reshape_and_cache_flash(*args)


def test_legacy_c_extension_switch(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.reshape_and_cache_flash")
    recorded = []
    monkeypatch.setattr(module, "use_c_extension", True)
    monkeypatch.setattr(
        torch.ops.flag_gems,
        "reshape_and_cache_flash",
        lambda *args: recorded.append(args),
        raising=False,
    )
    arguments = tuple(object() for _ in range(8))
    assert module.reshape_and_cache_flash(*arguments) is None
    assert recorded == [arguments]
    # Canonical must use its own hand-written Triton route, regardless of switch.
    check_case(dict(T=7, H=3, D=80, BS=7, layout="flash"), torch.float16)
    assert recorded == [arguments]


def test_legacy_triton_regression(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.reshape_and_cache_flash")
    monkeypatch.setattr(module, "use_c_extension", False)
    check_case(
        dict(T=7, H=3, D=80, BS=7, layout="flash"),
        torch.float16,
        module.reshape_and_cache_flash,
    )


@pytest.mark.parametrize("layout", ["flash", "head_major"])
@pytest.mark.parametrize("dtype", _ORDINARY_DTYPES)
def test_exact_page_boundaries(layout, dtype):
    args, _ = cases.build_inputs(
        dict(T=7, H=3, D=80, BS=7, layout=layout, padding=13, edges=True),
        dtype,
        flaggems_vllm.device,
    )
    args = list(args)
    bs = 7
    blocks = args[2].shape[0]
    args[4] = torch.tensor(
        [0, bs - 1, bs, 2 * bs - 1, (blocks - 1) * bs, blocks * bs - 1, -1],
        dtype=torch.int64,
        device=args[0].device,
    )
    expected = cases.reference(args)
    assert flaggems_vllm.triton_reshape_and_cache_flash(*args) is None
    for cache, wanted in zip(args[2:4], expected):
        torch.testing.assert_close(
            cases.backing(cache).view(torch.uint8).cpu(), wanted, rtol=0, atol=0
        )


@pytest.mark.parametrize("layout", ["flash", "head_major"])
@pytest.mark.parametrize("x", [1, 8, 16])
def test_shared_padded_inner_cache_strides(layout, x):
    args, _ = cases.build_inputs(
        dict(T=7, H=3, D=80, BS=7, layout=layout, padding=13, x=x),
        torch.float16,
        flaggems_vllm.device,
    )
    args = list(args)
    b, h, d, bs = args[2].shape[0], 3, 80, 7
    if layout == "flash":
        ch, cp = d + 3, h * (d + 3) + 5
        cb = bs * cp + 11
        shapes = [(b, bs, h, d)] * 2
        strides = [(cb, cp, ch, 1)] * 2
    else:
        kd, vd = bs * x + 8, bs + 1
        ch = max((d // x) * kd, d * vd) + 9
        cb = h * ch + 13
        shapes = [(b, h, d // x, bs, x), (b, h, d, bs)]
        strides = [(cb, ch, kd, x, 1), (cb, ch, vd, 1)]
    for index, (shape, stride) in enumerate(zip(shapes, strides)):
        raw = torch.full(
            ((b + 2) * cb,), -113.25, dtype=torch.float16, device=args[0].device
        )
        args[index + 2] = raw.as_strided(shape, stride, cb)
    expected = cases.reference(args)
    assert flaggems_vllm.triton_reshape_and_cache_flash(*args) is None
    for cache, wanted in zip(args[2:4], expected):
        torch.testing.assert_close(
            cases.backing(cache).view(torch.uint8).cpu(), wanted, rtol=0, atol=0
        )


@pytest.mark.parametrize("layout", ["flash", "head_major"])
@pytest.mark.parametrize("cache_dtype", _ORDINARY_DTYPES)
def test_output_dtype_and_independent_source_dtype(layout, cache_dtype):
    args, _ = cases.build_inputs(
        dict(T=7, H=3, D=80, BS=7, layout=layout), cache_dtype, flaggems_vllm.device
    )
    args = list(args)
    args[0] = args[0].to(torch.float16)
    args[1] = args[1].to(torch.bfloat16)
    expected = cases.reference(args)
    assert flaggems_vllm.triton_reshape_and_cache_flash(*args) is None
    for cache, wanted in zip(args[2:4], expected):
        torch.testing.assert_close(
            cases.backing(cache).view(torch.uint8).cpu(), wanted, rtol=0, atol=0
        )


@pytest.mark.parametrize("selector", ["fp8", "fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("t", [0, 1, 7])
def test_fp8_selector_and_empty(selector, t):
    # Native selectors all use platform.fp8_dtype(), not a suffix-based format.
    check_case(
        dict(T=t, H=3, D=80, BS=7, layout="flash", cache="uint8", selector=selector),
        torch.float32,
    )


@pytest.mark.parametrize("layout", ["flash", "head_major"])
@pytest.mark.parametrize("cache", ["fp8", "uint8"])
@pytest.mark.parametrize("ratio", [1.0, -2.0])
@pytest.mark.parametrize(
    "dtype,exponent",
    [
        (torch.float32, -126),
        (torch.float32, -140),
        (torch.float32, -149),
        (torch.bfloat16, -126),
        (torch.bfloat16, -130),
        (torch.bfloat16, -133),
    ],
)
def test_fp8_subnormal_scale(layout, cache, ratio, dtype, exponent):
    args, kwargs = cases.build_inputs(
        dict(T=17, H=3, D=80, BS=7, layout=layout, padding=13, cache=cache),
        dtype,
        flaggems_vllm.device,
    )
    args = list(args)
    # Build the powers of two as raw IEEE words, avoiding host/device FTZ in
    # the test's input construction. All ratios are independently known.
    width, mantissa = (32, 23) if dtype == torch.float32 else (16, 7)
    source_exponent = exponent + int(abs(ratio) == 2)
    bits = (
        (source_exponent + 127) << mantissa
        if source_exponent >= -126
        else 1 << (source_exponent + 126 + mantissa)
    )
    if ratio < 0:
        bits -= 1 << (width - 1)
    word_dtype = torch.int32 if width == 32 else torch.int16
    for source in args[:2]:
        raw = torch.full((cases.backing(source).numel(),), bits, dtype=word_dtype)
        cases.backing(source).view(torch.uint8).copy_(
            raw.view(torch.uint8).to(source.device)
        )
    scale_bits = (exponent + 127) << 23 if exponent >= -126 else 1 << (exponent + 149)
    for index in (6, 7):
        raw = torch.tensor([scale_bits], dtype=torch.int32).view(torch.uint8)
        args[index] = raw.to(args[0].device).view(torch.float32).reshape(())
    # The exact quotient is ratio; encode/index normal CPU values instead of
    # relying on the host processor's handling of subnormal arithmetic.
    oracle = list(args)
    oracle[0] = torch.full(args[0].shape, ratio, dtype=torch.float32)
    oracle[1] = torch.full(args[1].shape, ratio, dtype=torch.float32)
    oracle[6] = oracle[7] = torch.ones((), dtype=torch.float32)
    check_arguments(args, kwargs, expected=cases.reference(oracle))
