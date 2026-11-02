# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Offline host contracts: real module, fake tensor metadata, no GPU calls."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src/flaggems_vllm/ops/minimax_m3_sparse_attn.py"
spec = importlib.util.spec_from_file_location("draft_attention_cpu", SOURCE)
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)


class Meta:
    def __init__(
        self,
        shape,
        dtype=torch.bfloat16,
        device="cuda:0",
        contiguous=True,
        strides=None,
        requires_grad=False,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.ndim = len(shape)
        self.requires_grad = requires_grad
        self._contiguous = contiguous
        self._stride = []
        current = 1
        for n in reversed(shape):
            self._stride.append(current)
            current *= max(1, n)
        self._stride = (
            tuple(reversed(self._stride)) if strides is None else tuple(strides)
        )

    def stride(self, dim=None):
        return self._stride if dim is None else self._stride[dim]

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return self._contiguous

    def view(self, *shape):
        return Meta(shape, self.dtype, strides=(self.shape[-1], 1))


def inputs(n=17, d=96, kv_heads=1, dtype=torch.bfloat16, topk=16):
    return [
        Meta((n, 8 * kv_heads, d)),
        Meta((32, kv_heads, 128, 2 * d), dtype),
        Meta((kv_heads, n, topk), torch.int32),
        Meta((1, 32), torch.int32),
        Meta((2,), torch.int32),
        Meta((1,), torch.int32),
        Meta((1,), torch.int32),
        n,
        kv_heads,
        d**-0.5,
        Meta((n, 8 * kv_heads, d)),
    ]


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("CUDA API called by offline gate")

    for name in (
        "get_device_capability",
        "get_device_properties",
        "current_stream",
        "synchronize",
        "init",
    ):
        monkeypatch.setattr(torch.cuda, name, fail)
    initialized = torch.cuda.is_initialized()
    yield
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("cap", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0)])
def test_bf16_preserved(cap):
    assert op._resolve_qk_dtype("bf16", torch.bfloat16, cap) == "bf16"


@pytest.mark.parametrize("cap", [(8, 0), (8, 6)])
def test_only_explicit_fp8_can_fallback(cap):
    assert op._resolve_qk_dtype("fp8", torch.bfloat16, cap) == "int8"
    with pytest.raises(ValueError, match="precision planner"):
        op._resolve_qk_dtype("fp8", torch.float8_e4m3fn, cap)


@pytest.mark.parametrize(
    "requested,dtype,cap,error",
    [
        ("fp8", torch.bfloat16, (9, 0), ValueError),
        ("int8", torch.float8_e4m3fn, (9, 0), ValueError),
        ("invalid", torch.bfloat16, (9, 0), ValueError),
        ("bf16", torch.bfloat16, (7, 5), NotImplementedError),
    ],
)
def test_reject_precision(requested, dtype, cap, error):
    with pytest.raises(error):
        op._resolve_qk_dtype(requested, dtype, cap)


@pytest.mark.parametrize(
    "n,d,heads,topk",
    [
        (1, 96, 1, 1),
        (17, 96, 2, 17),
        (384, 128, 2, 16),
        (8192, 128, 1, 32),
        (0, 96, 1, 16),
    ],
)
def test_valid_metadata(n, d, heads, topk):
    op._validate_inputs(*inputs(n, d, heads, topk=topk), None, None)


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda a: setattr(a[0], "dtype", torch.float16), NotImplementedError),
        (lambda a: setattr(a[10], "_contiguous", False), NotImplementedError),
        (lambda a: setattr(a[0], "requires_grad", True), NotImplementedError),
        (lambda a: setattr(a[0], "device", torch.device("cpu")), NotImplementedError),
        (lambda a: setattr(a[2], "dtype", torch.int64), ValueError),
        (
            lambda a: a.__setitem__(2, Meta((1, 17, 0), torch.int32)),
            NotImplementedError,
        ),
        (lambda a: a.__setitem__(6, Meta((2,), torch.int32)), ValueError),
        (
            lambda a: a.__setitem__(1, Meta((32, 1, 128, 192), torch.uint8)),
            NotImplementedError,
        ),
        (lambda a: a.__setitem__(1, Meta((32, 1, 64, 192))), ValueError),
        (lambda a: a.__setitem__(3, Meta((1, 0), torch.int32)), ValueError),
        (
            lambda a: a.__setitem__(4, Meta((2,), torch.int32, strides=(2,))),
            NotImplementedError,
        ),
        (
            lambda a: a.__setitem__(3, Meta((1, 32), torch.int32, device="cuda:1")),
            ValueError,
        ),
        (lambda a: a.__setitem__(7, 0), ValueError),
        (lambda a: a.__setitem__(7, -1), ValueError),
        (lambda a: a.__setitem__(7, True), ValueError),
        (lambda a: a.__setitem__(9, float("nan")), ValueError),
        (lambda a: a.__setitem__(9, 1e39), ValueError),
        (
            lambda a: a.__setitem__(
                2, Meta((1, 17, 16), torch.int32, strides=(100, 2**30, 1))
            ),
            NotImplementedError,
        ),
    ],
)
def test_reject_metadata(change, error):
    a = inputs()
    change(a)
    with pytest.raises(error):
        op._validate_inputs(*a, None, None)


@pytest.mark.parametrize(
    "scales,error",
    [
        ((Meta((), torch.float32), None), ValueError),
        ((Meta((2,), torch.float32), Meta((), torch.float32)), ValueError),
        ((Meta((), torch.bfloat16), Meta((), torch.float32)), ValueError),
        (
            (Meta((), torch.float32, device="cuda:1"), Meta((), torch.float32)),
            ValueError,
        ),
    ],
)
def test_reject_scales(scales, error):
    with pytest.raises(error):
        op._validate_inputs(*inputs(), *scales)


def test_valid_scalar_scales():
    op._validate_inputs(*inputs(), Meta((), torch.float32), Meta((), torch.float32))


@pytest.mark.parametrize(
    "cap,cache,requested,expected",
    [
        ((8, 0), torch.bfloat16, "bf16", "bf16"),
        ((8, 6), torch.bfloat16, "fp8", "int8"),
        ((9, 0), torch.bfloat16, "bf16", "bf16"),
        ((9, 0), torch.float8_e4m3fn, "fp8", "fp8"),
        ((9, 0), torch.bfloat16, "int8", "int8"),
    ],
)
def test_selected_dispatch(monkeypatch, cap, cache, requested, expected):
    calls = []
    monkeypatch.setattr(op, "_device_capability", lambda _: cap)
    monkeypatch.setattr(op, "_launch_selected", lambda *args: calls.append(args))
    a = inputs(dtype=cache)
    assert (
        op.minimax_m3_sparse_attn(*a, qk_dtype=requested, schedule="selected") is a[-1]
    )
    assert calls[0][14] == expected
    assert calls[0][15] is None


def test_union_publishes_shared_flags_and_keeps_bf16(monkeypatch):
    launches = []
    selected = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, args, kwargs))

    monkeypatch.setattr(op, "_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(op, "_is_h100", lambda _: True)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=33)
    )
    monkeypatch.setattr(torch, "empty", lambda shape, **kw: Meta(shape, kw["dtype"]))
    monkeypatch.setattr(op, "_union_span_kernel", Kernel())
    monkeypatch.setattr(op, "_launch_selected", lambda *args: selected.append(args))
    a = inputs(n=512)
    op.minimax_m3_sparse_attn(*a)
    assert launches[0][0] == (64, 1, 1)
    assert launches[0][2]["AUTO_SPAN"] == 32
    assert selected[0][14] == "bf16"
    assert selected[0][15] is launches[0][2]["auto_flags_ptr"]
    assert selected[0][16] == 8


def test_empty_returns_without_probe_or_launch():
    a = inputs(n=0)
    assert op.minimax_m3_sparse_attn(*a) is a[-1]


def test_library_export_is_excluded_from_aten_registration():
    p = REPO / "src/flaggems_vllm/ops/__init__.py"
    t = ast.parse(p.read_text())
    exports = next(
        n.value
        for n in t.body
        if isinstance(n, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id == "__all__" for x in n.targets)
    )
    assert "minimax_m3_sparse_attn" in ast.literal_eval(exports)
    assert any(
        isinstance(n, ast.ImportFrom)
        and n.module == "flaggems_vllm.ops.minimax_m3_sparse_attn"
        for n in t.body
    )
    s = (REPO / "src/flaggems_vllm/__init__.py").read_text()
    assert 'getattr(_ops_module, "__all__", [])' in s
    assert "op_name not in _LIBRARY_ONLY_OPS" in s
    assert '_LIBRARY_ONLY_OPS = frozenset({"minimax_m3_sparse_attn"})' in s
