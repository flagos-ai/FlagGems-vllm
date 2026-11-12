# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent CPU oracle, public dispatch, boundaries, and legacy regression."""

import builtins
import itertools
from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm as gems
from baseline import hc_combine as baseline

DTYPES = baseline.DTYPES
SHAPES = [
    (0, 4, 17),
    (1, 1, 1),
    (1, 3, 7),
    (7, 4, 127),
    (17, 3, 513),
    (1, 4, 4096),
    (1024, 4, 1024),
    (4096, 4, 127),
    (3, 8, 33),
    (1, 4, 0),
]


def reference(residual, block, injection, hc):
    # Explicit HC broadcast; no flattened repeat/indexing shared with the kernel.
    n, width = residual.shape
    d = width // hc
    res = residual.detach().cpu().float().reshape(n, hc, d)
    blk = block.detach().cpu().float().reshape(n, 1, d)
    if injection is not None:
        logits = injection.detach().cpu().float().reshape(n, hc, 1)
        gate = 2.0 * torch.sigmoid(logits / hc)
        # Native's FP32 multiply-add may fuse. Widen only the multiply-add,
        # after FP32 gate evaluation, to model one final FP32 rounding.
        result = (res.double() + blk.double() * gate.double()).float()
    else:
        result = res + blk
    return result.reshape(n, width).to(residual.dtype)


def assert_result(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    actual = actual.detach().cpu()
    torch.testing.assert_close(torch.isnan(actual), torch.isnan(expected))
    torch.testing.assert_close(torch.isposinf(actual), torch.isposinf(expected))
    torch.testing.assert_close(torch.isneginf(actual), torch.isneginf(expected))
    tol = (
        2e-6 if expected.dtype == torch.float32 else 2 * torch.finfo(expected.dtype).eps
    )
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol, equal_nan=True)


def check(data, fn=gems.hc_combine):
    tensors = [x for x in data if isinstance(x, torch.Tensor)]
    copies = [x.detach().cpu().clone() for x in tensors]
    expected = reference(*data)
    out = fn(*data)
    assert_result(out, expected)
    assert out.device == data[0].device and out.is_contiguous()
    if out.numel():
        assert all(out.data_ptr() != x.data_ptr() for x in tensors)
    for x, old in zip(tensors, copies):
        torch.testing.assert_close(
            x.detach().cpu(), old, rtol=0, atol=0, equal_nan=True
        )
    return out


@pytest.mark.hc_combine
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("injection", ["none", "same", "fp32"])
@pytest.mark.parametrize("layout", ["dense", "packed"])
def test_hc_combine(shape, dtype, injection, layout):
    n, hc, d = shape
    data, _ = baseline.build_inputs(
        dict(n=n, hc=hc, d=d, layout=layout, injection=injection), dtype, gems.device
    )
    check(data)
    if injection != "none":
        r, b, i, h = data
        check((r, b, i, h), lambda r, b, i, h: gems.qwen4_hc_inject_combine(i, b, r, h))


@pytest.mark.hc_combine
@pytest.mark.parametrize(
    "rd,bd,idt", list(itertools.product(DTYPES, DTYPES, [None] + DTYPES))
)
def test_hc_combine_dtype_matrix(rd, bd, idt):
    r = torch.randn((7, 51), device=gems.device, dtype=rd)
    b = torch.randn((7, 17), device=gems.device, dtype=bd)
    i = None if idt is None else torch.randn((7, 3), device=gems.device, dtype=idt)
    check((r, b, i, 3))
    check((r, b, i, 3), lambda r, b, i, h: gems.qwen4_hc_inject_combine(i, b, r, h))


@pytest.mark.hc_combine
@pytest.mark.parametrize(
    "value", [None, 0.0, 1000.0, -1000.0, float("nan"), float("inf"), -float("inf")]
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_hc_combine_extremes(value, dtype):
    values = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            torch.finfo(dtype).max,
            -torch.finfo(dtype).max,
            float("inf"),
            -float("inf"),
            float("nan"),
            torch.finfo(dtype).tiny,
        ],
        dtype=dtype,
    )
    r = values.repeat(3, 3).to(gems.device)
    b = values.repeat(3, 1).to(gems.device)
    i = None if value is None else torch.full((3, 3), value, device=gems.device)
    check((r, b, i, 3))


@pytest.mark.hc_combine
@pytest.mark.parametrize("injection", [None, 0.5])
@pytest.mark.parametrize("stride", [0, 1, 8])
def test_hc_combine_overlapping_rows(injection, stride):
    def make(width):
        storage = torch.randn((3 * stride + width,), device=gems.device)
        return storage.as_strided((4, width), (stride, 1))

    data = (make(21), make(7), None if injection is None else make(3), 3)
    check(data)
    check(data, lambda r, b, i, h: gems.qwen4_hc_inject_combine(i, b, r, h))


@pytest.mark.hc_combine
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("injection_dtype", DTYPES)
def test_hc_combine_projection_slice(dtype, injection_dtype):
    # Injection is the tail of a packed [lowrank | HC] projection.
    packed = torch.randn((7, 259), dtype=injection_dtype, device=gems.device)
    injection = packed[:, 256:]
    residual = torch.randn((9, 72), dtype=dtype, device=gems.device)[1:8, 3:72]
    block = torch.randn((9, 31), dtype=dtype, device=gems.device)[1:8, 5:28]
    check((residual, block, injection, 3))
    check(
        (residual, block, injection, 3),
        lambda r, b, i, h: gems.qwen4_hc_inject_combine(i, b, r, h),
    )


@pytest.mark.hc_combine
@pytest.mark.parametrize(
    "kind",
    [
        "hc0",
        "hcnegative",
        "hcfloat",
        "hcbool",
        "dim",
        "block",
        "injection",
        "rank",
        "res_stride",
        "block_stride",
        "inj_stride",
        "dtype",
        "device",
    ],
)
def test_hc_combine_errors(kind):
    r = torch.empty((2, 12), device=gems.device)
    b = torch.empty((2, 4), device=gems.device)
    i = torch.empty((2, 3), device=gems.device)
    h = 3
    if kind == "hc0":
        h = 0
    elif kind == "hcnegative":
        h = -3
    elif kind == "hcfloat":
        h = 3.0
    elif kind == "hcbool":
        h = True
    elif kind == "dim":
        h = 5
    elif kind == "block":
        b = b[:, :3]
    elif kind == "injection":
        i = i[:, :2]
    elif kind == "rank":
        r = r.unsqueeze(0)
    elif kind == "res_stride":
        r = torch.empty((2, 24), device=gems.device)[:, ::2]
    elif kind == "block_stride":
        b = torch.empty((2, 8), device=gems.device)[:, ::2]
    elif kind == "inj_stride":
        i = torch.empty((2, 6), device=gems.device)[:, ::2]
    elif kind == "dtype":
        b = b.to(torch.int32)
    elif kind == "device":
        i = i.cpu()
    with pytest.raises((ValueError, TypeError)):
        gems.hc_combine(r, b, i, h)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        gems.qwen4_hc_inject_combine(i, b, r, h)
    if kind == "hcbool":
        # Reject bool even when it describes an otherwise valid HC=1 workload.
        with pytest.raises((ValueError, TypeError)):
            gems.qwen4_hc_inject_combine(i[:, :1], b, b, True)


@pytest.mark.hc_combine
def test_hc_combine_legacy_leading_dims():
    r = torch.randn((2, 3, 21), device=gems.device, dtype=torch.float16)
    b = torch.randn((2, 3, 7), device=gems.device, dtype=torch.float16)
    i = torch.randn((2, 3, 3), device=gems.device)
    actual = gems.qwen4_hc_inject_combine(i, b, r, 3)
    assert_result(
        actual.reshape(6, 21),
        reference(r.reshape(6, 21), b.reshape(6, 7), i.reshape(6, 3), 3),
    )
    for index in range(3):
        data = [i, b, r]
        # Swapping equal leading dimensions preserves shape but prevents flattening.
        data = [tensor[:2, :2].contiguous() for tensor in data]
        data[index] = data[index].transpose(0, 1)
        with pytest.raises(ValueError, match="flattenable"):
            gems.qwen4_hc_inject_combine(*data, 3)


@pytest.mark.hc_combine
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("injection", ["none", "same", "fp32"])
def test_hc_combine_cpu_reference_tail(dtype, injection):
    # Keep all nine packed projection-tail configurations against the CPU oracle.
    # Native qualification belongs to the benchmark, not accuracy tests.
    data, _ = baseline.build_inputs(
        dict(n=7, hc=3, d=513, layout="packed", injection=injection), dtype, gems.device
    )
    check(data)
    if injection != "none":
        r, b, i, h = data
        check((r, b, i, h), lambda r, b, i, h: gems.qwen4_hc_inject_combine(i, b, r, h))


@pytest.mark.hc_combine
@pytest.mark.parametrize(
    "mode",
    ["ready", "versionless", "missing_module", "missing_attribute", "not_callable"],
)
def test_hc_combine_native_resolution(monkeypatch, mode):
    # Resolver fault injection is separate from candidate numerical accuracy.
    # Successful resolution must return and call the original imported function.
    calls = []
    output = object()

    def imported_native(*args):
        calls.append(args)
        return output

    original_import = builtins.__import__

    def import_vllm(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm":
            return (
                SimpleNamespace()
                if mode == "versionless"
                else SimpleNamespace(__version__="test-version")
            )
        if name == "vllm.models.qwen4_exp.nvidia.ops.hc":
            if mode == "missing_module":
                raise ImportError("vLLM hc module unavailable")
            if mode == "missing_attribute":
                raise AttributeError("vLLM hc_combine unavailable")
            return SimpleNamespace(
                hc_combine=None if mode == "not_callable" else imported_native
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_vllm)
    monkeypatch.setattr(baseline, "NATIVE_UNAVAILABLE_REASON", "previous failure")
    monkeypatch.setattr(baseline, "NATIVE_VERSION", None)
    monkeypatch.setenv("HC_COMBINE_NATIVE_SOURCE", "/unused/pinned/source.py")
    native = baseline.native()
    if mode in ("ready", "versionless"):
        assert native is imported_native
        args = (object(), object(), None, 3)
        assert native(*args) is output and calls == [args]
        assert baseline.NATIVE_UNAVAILABLE_REASON is None
        assert baseline.NATIVE_VERSION == (
            None if mode == "versionless" else "test-version"
        )
    else:
        assert native is None and not calls
        assert baseline.NATIVE_UNAVAILABLE_REASON
