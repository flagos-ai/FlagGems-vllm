# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent CPU references; native absence never hides candidate correctness."""

from itertools import product

import pytest
import torch

import flaggems_vllm
from baseline import grouped_gemma_rmsnorm as baseline
from flaggems_vllm.runtime import torch_device_fn

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
DTYPE_PAIRS = list(product(DTYPES, repeat=2))
# Fixed before candidate measurements; compare after the final output cast.
TOLERANCES = {
    torch.float32: (2e-6, 2e-5),
    torch.float16: (2e-3, 2e-3),
    torch.bfloat16: (2e-2, 2e-2),
}
SHAPES = [(0, 12, 4), (17, 4, 4)] + baseline.SHAPES
ERRORS = (TypeError, ValueError, RuntimeError, NotImplementedError)


def _reference(x, weight, eps, groups, compute_dtype=torch.float64):
    n, dim = x.shape
    group_dim = dim // groups
    values = x.detach().cpu().to(compute_dtype).reshape(n, groups, group_dim)
    affine = weight.detach().cpu().to(compute_dtype).reshape(-1)
    affine = affine.reshape(1, 1 if affine.numel() == group_dim else groups, group_dim)
    normalized = values * torch.rsqrt(
        (values * values).sum(-1, keepdim=True) / group_dim + eps
    )
    return (normalized + normalized * affine).reshape(n, dim).to(x.dtype)


def _assert_close(actual, expected):
    atol, rtol = TOLERANCES[actual.dtype]
    torch.testing.assert_close(
        actual.detach().cpu(), expected, atol=atol, rtol=rtol, equal_nan=True
    )


def _check_case(x, weight, eps, groups, compute_dtype=torch.float64):
    # Snapshot the entire storage, including padding, to detect unintended writes.
    x_storage = torch.as_strided(
        x, (x.untyped_storage().nbytes() // x.element_size(),), (1,), 0
    )
    saved_x = x_storage.clone()
    saved_weight = weight.clone()
    expected = _reference(x, weight, eps, groups, compute_dtype)
    actual = flaggems_vllm.grouped_gemma_rmsnorm(x, weight, eps, groups)
    assert actual is not x and actual is not weight
    assert (
        actual.shape == x.shape
        and actual.dtype == x.dtype
        and actual.device == x.device
    )
    assert actual.is_contiguous()
    if actual.numel():
        assert actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        assert (
            actual.untyped_storage().data_ptr() != weight.untyped_storage().data_ptr()
        )
    _assert_close(actual, expected)
    torch.testing.assert_close(x_storage, saved_x, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(weight, saved_weight, atol=0, rtol=0, equal_nan=True)
    return actual


def _inputs(shape, x_dtype, w_dtype, shared, row_gap, weight_rank=1):
    n, dim, groups = shape
    binding = {
        "n": n,
        "dim": dim,
        "num_groups": groups,
        "weight_dtype": str(w_dtype).split(".")[-1],
        "shared": shared,
        "row_gap": row_gap,
        "weight_rank": weight_rank,
        "eps": 1e-5,
        "distribution": "normal",
        "seed": 20261102,
    }
    args, _ = baseline.build_inputs(binding, x_dtype, flaggems_vllm.device)
    return args


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("x_dtype,w_dtype", DTYPE_PAIRS)
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("row_gap", [0, 7])
def test_grouped_gemma_rmsnorm(shape, x_dtype, w_dtype, shared, row_gap):
    args = _inputs(shape, x_dtype, w_dtype, shared, row_gap, weight_rank=2)
    _check_case(*args)


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize("binding", baseline.grid(), ids=baseline.key_shape)
@pytest.mark.parametrize("dtype", DTYPES)
def test_grouped_gemma_rmsnorm_performance_workload(binding, dtype):
    args, _ = baseline.build_inputs(binding, dtype, flaggems_vllm.device)
    _check_case(*args)


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize("x_dtype,w_dtype", DTYPE_PAIRS)
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("row_stride", [0, 5])
def test_grouped_gemma_rmsnorm_overlapping_rows(x_dtype, w_dtype, shared, row_stride):
    x, weight, eps, groups = _inputs((3, 12, 4), x_dtype, w_dtype, shared, 7)
    x = torch.as_strided(x, (3, 12), (row_stride, 1))
    _check_case(x, weight, eps, groups)


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize("x_dtype,w_dtype", DTYPE_PAIRS)
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_grouped_gemma_rmsnorm_constant_weight(x_dtype, w_dtype, shared, value):
    x, weight, eps, groups = _inputs((3, 508, 4), x_dtype, w_dtype, shared, 7)
    weight.fill_(value)
    _check_case(x, weight, eps, groups)


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize("x_dtype,w_dtype", DTYPE_PAIRS)
@pytest.mark.parametrize("direction", [float("inf"), -float("inf")])
def test_grouped_gemma_rmsnorm_near_minus_one(x_dtype, w_dtype, direction):
    x, weight, eps, groups = _inputs((3, 508, 4), x_dtype, w_dtype, True, 7)
    near = torch.nextafter(
        torch.tensor(-1.0, dtype=w_dtype), torch.tensor(direction, dtype=w_dtype)
    ).item()
    weight.fill_(near)
    _check_case(x, weight, eps, groups)


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_grouped_gemma_rmsnorm_weight_rank(rank):
    x, weight, eps, groups = _inputs((3, 4, 4), torch.float32, torch.float16, True, 7)
    weight = weight.reshape([1] * rank)
    _check_case(x, weight, eps, groups)


EDGE_CASES = [
    ("zero", 1e-5),
    ("zero", 0.0),
    ("random", 0.0),
    ("random", 1e-30),
    ("random", -1e-6),
    ("random", -10.0),
    ("random", float("inf")),
    ("random", -float("inf")),
    ("random", float("nan")),
    ("nan", 1e-5),
    ("posinf", 1e-5),
    ("neginf", 1e-5),
    ("large", 1e-5),
]


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize("x_dtype,w_dtype", DTYPE_PAIRS)
@pytest.mark.parametrize("kind,eps", EDGE_CASES)
def test_grouped_gemma_rmsnorm_edges(x_dtype, w_dtype, kind, eps):
    x, weight, _, groups = _inputs((3, 508, 4), x_dtype, w_dtype, True, 7)
    if kind == "zero":
        x.zero_()
    elif kind in ("nan", "posinf", "neginf"):
        x[0, 0] = {
            "nan": float("nan"),
            "posinf": float("inf"),
            "neginf": -float("inf"),
        }[kind]
    elif kind == "large":
        x.fill_(100.0 if x_dtype == torch.float16 else 1e10)
    _check_case(x, weight, eps, groups)


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_grouped_gemma_rmsnorm_fp32_overflow(dtype):
    x, weight, eps, groups = _inputs((3, 508, 4), dtype, torch.float32, False, 0)
    x.fill_(1e20)
    # An FP64 mathematical reference remains finite here. Native intentionally
    # squares/reduces in FP32, overflows to infinity, and normalizes to zero.
    _check_case(x, weight, eps, groups, compute_dtype=torch.float32)


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize("groups", [0, -1, 5, 20, 1.5, True, "4", None])
def test_grouped_gemma_rmsnorm_invalid_groups(groups):
    x, weight, eps, _ = _inputs((3, 12, 4), torch.float32, torch.float32, False, 0)
    with pytest.raises(ERRORS):
        flaggems_vllm.grouped_gemma_rmsnorm(x, weight, eps, groups)


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize(
    "kind",
    [
        "rank0",
        "rank1",
        "rank3",
        "dim0",
        "inner_stride",
        "weight_size",
        "weight_stride",
        "x_integer",
        "x_bool",
        "weight_integer",
        "weight_bool",
        "eps_string",
        "eps_complex",
    ],
)
def test_grouped_gemma_rmsnorm_invalid_inputs(kind):
    x, weight, eps, groups = _inputs((3, 12, 4), torch.float32, torch.float32, False, 0)
    if kind == "rank0":
        x = x[0, 0]
    elif kind == "rank1":
        x = x[0]
    elif kind == "rank3":
        x = x.unsqueeze(0)
    elif kind == "dim0":
        x = x[:, :0]
    elif kind == "inner_stride":
        x = x[:, ::2]
    elif kind == "weight_size":
        weight = weight[:2]
    elif kind == "weight_stride":
        weight = torch.empty(24, device=weight.device)[::2]
    elif kind in ("x_integer", "x_bool"):
        x = x.to(torch.int32 if kind == "x_integer" else torch.bool)
    elif kind in ("weight_integer", "weight_bool"):
        weight = weight.to(torch.int32 if kind == "weight_integer" else torch.bool)
    elif kind == "eps_string":
        eps = "1e-5"
    elif kind == "eps_complex":
        eps = 1e-5 + 1j
    with pytest.raises(ERRORS):
        flaggems_vllm.grouped_gemma_rmsnorm(x, weight, eps, groups)


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize("cpu_input", ["x", "weight", "both"])
def test_grouped_gemma_rmsnorm_invalid_device(cpu_input):
    x, weight, eps, groups = _inputs((3, 12, 4), torch.float32, torch.float32, False, 0)
    if cpu_input in ("x", "both"):
        x = x.cpu()
    if cpu_input in ("weight", "both"):
        weight = weight.cpu()
    with pytest.raises(ERRORS):
        flaggems_vllm.grouped_gemma_rmsnorm(x, weight, eps, groups)


@pytest.mark.grouped_gemma_rmsnorm
def test_grouped_gemma_rmsnorm_cross_accelerator_device():
    if torch_device_fn.device_count() < 2:
        pytest.skip("cross-accelerator rejection requires two visible devices")
    x, weight, eps, groups = _inputs((3, 12, 4), torch.float32, torch.float32, False, 0)
    weight = weight.to(f"{x.device.type}:{1 if x.device.index == 0 else 0}")
    with pytest.raises(ERRORS):
        flaggems_vllm.grouped_gemma_rmsnorm(x, weight, eps, groups)


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "x_dtype,w_dtype", list(product((torch.float16, torch.bfloat16), repeat=2))
)
def test_grouped_gemma_rmsnorm_legacy_consistency(shape, x_dtype, w_dtype):
    args = _inputs(shape, x_dtype, w_dtype, False, 0)
    x, weight, eps, groups = args
    actual = _check_case(*args)
    legacy = flaggems_vllm.qwen4_grouped_gemma_rmsnorm(x, weight, groups, eps)
    _assert_close(actual, legacy.detach().cpu())


@pytest.mark.grouped_gemma_rmsnorm
@pytest.mark.parametrize("binding", baseline.grid(), ids=baseline.key_shape)
@pytest.mark.parametrize("dtype", DTYPES)
def test_grouped_gemma_rmsnorm_native(binding, dtype):
    native = baseline.native(backend=flaggems_vllm.vendor_name)
    if native is None:
        pytest.skip(baseline.NATIVE_UNAVAILABLE_REASON)
    args, kwargs = baseline.build_inputs(binding, dtype, flaggems_vllm.device)
    expected = native(*args, **kwargs)
    actual = _check_case(*args)
    _assert_close(actual, expected.detach().cpu())


@pytest.mark.grouped_gemma_rmsnorm
def test_grouped_gemma_rmsnorm_public_dispatch():
    assert (
        flaggems_vllm.grouped_gemma_rmsnorm is flaggems_vllm.ops.grouped_gemma_rmsnorm
    )
    assert all(
        item[0] != "grouped_gemma_rmsnorm" for item in flaggems_vllm._FULL_CONFIG
    )
