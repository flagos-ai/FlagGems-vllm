# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent forward tests for the GLM-5 indexer K LayerNorm leaf."""

import importlib
import inspect
import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm
import flaggems_vllm.ops as ops

from . import conftest as cfg

OP_NAME = "fused_indexer_k_norm"
DEVICE = flaggems_vllm.device
DEVICE_AVAILABLE = flaggems_vllm.runtime.torch_device_fn.is_available()
DEVICE_REQUIRED = pytest.mark.skipif(
    not DEVICE_AVAILABLE, reason="accelerator required for fused_indexer_k_norm"
)

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
TOLERANCES = {
    torch.float16: (2e-3, 2e-3),
    torch.bfloat16: (1.6e-2, 1.6e-2),
    torch.float32: (2e-5, 2e-5),
}
EPS_VALUES = (1e-6, 1e-5, 1e-3)
FP32_EDGE_EPS_VALUES = (2**-149, 2**-126)
NVIDIA_RAW_CONFIGS = ((1, 4), (4, 4), (8, 4), (1, 8))
NVIDIA_WIDE_RAW_CONFIGS = ((1, 4), (1, 8))
NVIDIA_REQUIRED = pytest.mark.skipif(
    flaggems_vllm.runtime.device.vendor_name != "nvidia",
    reason="NVIDIA raw-candidate coverage requires the NVIDIA backend",
)

QUICK_CASES = ((0, 128), (3, 129), (17, 128))
FULL_CASES = QUICK_CASES + (
    (1, 1),
    (2, 3),
    (3, 127),
    (128, 128),
    (1024, 128),
    (4096, 128),
    (128, 256),
    (17, 257),
    (64, 1024),
    (1, 4095),
    (3, 4096),
)
FINITE_CASES = QUICK_CASES if cfg.QUICK_MODE else FULL_CASES


def _eps32(eps):
    """Match the frozen FP32-scalar contract without using a candidate helper."""
    return torch.tensor(eps, dtype=torch.float32).item()


def _h20z_d128_fast_path_runtime_available(module):
    return (
        module._is_h20z()
        and module.triton.__version__ == module._H20Z_D128_TRITON_VERSION
    )


def cpu_fp64_oracle(x, weight, bias, eps):
    """Formula oracle; it deliberately does not call layer_norm or candidate code."""
    eps32 = _eps32(eps)
    u = x.detach().to(device="cpu", dtype=torch.float32).to(torch.float64)
    w = weight.detach().to(device="cpu", dtype=torch.float32).to(torch.float64)
    b = bias.detach().to(device="cpu", dtype=torch.float32).to(torch.float64)
    variance = (u - u.mean(dim=-1, keepdim=True)).square().mean(dim=-1, keepdim=True)
    normalized = (u - u.mean(dim=-1, keepdim=True)) * torch.rsqrt(
        variance + float(eps32)
    )
    return (normalized * w + b).to(dtype=x.dtype)


def _round_fp32(value):
    return torch.tensor(value, dtype=torch.float32).item()


def cpu_fp32_semantics_oracle(x, weight, bias, eps):
    """C1 oracle with a stable mean and explicit FP32 operation rounding."""
    eps32 = _eps32(eps)
    u = x.detach().to(device="cpu", dtype=torch.float32)
    w = weight.detach().to(device="cpu", dtype=torch.float32)
    b = bias.detach().to(device="cpu", dtype=torch.float32)
    if not u.shape[0]:
        return torch.empty_like(u, dtype=x.dtype)

    rows = []
    for row in u.tolist():
        mean32 = _round_fp32(math.fsum(row) / len(row))
        centered = [_round_fp32(value - mean32) for value in row]
        squared = [_round_fp32(value * value) for value in centered]
        sum_sq = _round_fp32(0.0)
        for value in squared:
            sum_sq = _round_fp32(sum_sq + value)
        variance32 = _round_fp32(sum_sq / len(row))
        denominator32 = _round_fp32(variance32 + eps32)
        rstd32 = _round_fp32(1.0 / math.sqrt(denominator32))

        output = []
        for centered_value, weight_value, bias_value in zip(
            centered, w.tolist(), b.tolist()
        ):
            normalized32 = _round_fp32(centered_value * rstd32)
            scaled32 = _round_fp32(normalized32 * weight_value)
            output.append(_round_fp32(scaled32 + bias_value))
        rows.append(output)

    return torch.tensor(rows, dtype=torch.float32).to(dtype=x.dtype)


def source_shadow_reference(x, weight, bias, eps):
    """The upstream expression, retained only for special-value parity tests."""
    return F.layer_norm(x.float(), (x.shape[-1],), weight, bias, eps).to(x.dtype)


def _assert_close(actual, expected, dtype):
    rtol, atol = TOLERANCES[dtype]
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    try:
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise AssertionError(_error_statistics(actual_cpu, expected_cpu)) from exc


def _error_statistics(actual, expected):
    if actual.numel() == 0:
        return "assert_close failed for an empty tensor"

    actual64 = actual.to(torch.float64)
    expected64 = expected.to(torch.float64)
    diff = (actual64 - expected64).abs()
    denominator = expected64.abs()
    relative = torch.where(
        denominator == 0,
        torch.where(
            diff == 0, torch.zeros_like(diff), torch.full_like(diff, float("inf"))
        ),
        diff / denominator,
    )
    max_index = tuple(
        int(index) for index in torch.unravel_index(diff.argmax(), diff.shape)
    )
    rms = torch.sqrt(diff.square().mean()).item()
    return (
        "assert_close failed: "
        f"shape={tuple(actual.shape)}, dtype={actual.dtype}, strides={actual.stride()}, "
        f"max_abs={diff.max().item():.8g}, max_rel={relative.max().item():.8g}, "
        f"rms={rms:.8g}, max_abs_index={max_index}, "
        f"actual={actual[max_index].item():.8g}, expected={expected[max_index].item():.8g}"
    )


def _assert_source_parity(actual, expected, dtype):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if finite.any():
        _assert_close(actual[finite], expected[finite], dtype)


def _source_compatibility_diagnostic(actual, expected):
    actual_cpu = actual.detach().cpu().to(torch.float64)
    expected_cpu = expected.detach().cpu().to(torch.float64)
    finite = torch.isfinite(actual_cpu) & torch.isfinite(expected_cpu)
    diagnostic = {
        "kind": "eager_source_shadow_compatibility_only",
        "nan_mismatch": int(
            (torch.isnan(actual_cpu) != torch.isnan(expected_cpu)).sum().item()
        ),
        "inf_mismatch": int(
            (torch.isinf(actual_cpu) != torch.isinf(expected_cpu)).sum().item()
        ),
        "finite_overlap": int(finite.sum().item()),
    }
    if finite.any():
        diff = (actual_cpu[finite] - expected_cpu[finite]).abs()
        diagnostic["finite_max_abs"] = diff.max().item()
        diagnostic["finite_rms"] = torch.sqrt(diff.square().mean()).item()
    return json.dumps(diagnostic, sort_keys=True)


def _make_inputs(shape, dtype, seed=20261106):
    torch.manual_seed(seed)
    m, d = shape
    x = torch.randn((m, d), dtype=dtype, device=DEVICE)
    weight = torch.randn((d,), dtype=torch.float32, device=DEVICE)
    bias = torch.randn((d,), dtype=torch.float32, device=DEVICE)
    return x, weight, bias


def _assert_fresh_contiguous(out, x, weight, bias):
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert out.is_contiguous()
    assert out.stride() == (x.shape[1], 1)
    assert out is not x
    if out.numel():
        out_storage = out.untyped_storage().data_ptr()
        assert out_storage != x.untyped_storage().data_ptr()
        assert out_storage != weight.untyped_storage().data_ptr()
        assert out_storage != bias.untyped_storage().data_ptr()


def _constant_row(dtype, d=128, m=3):
    x = torch.full((m, d), 2.0, dtype=dtype, device=DEVICE)
    weight = torch.linspace(-1.0, 1.0, d, device=DEVICE, dtype=torch.float32)
    bias = torch.linspace(-0.75, 0.75, d, device=DEVICE, dtype=torch.float32)
    return x, weight, bias


def _c1_affine(dim):
    weight = torch.linspace(0.5, 1.5, dim, device=DEVICE, dtype=torch.float32)
    bias = torch.linspace(-0.7, 0.9, dim, device=DEVICE, dtype=torch.float32)
    return weight, bias


def _c1_original_extreme_rows():
    dim = 128
    signs = torch.where(
        torch.arange(dim, device=DEVICE) % 2 == 0,
        torch.ones(dim, device=DEVICE),
        -torch.ones(dim, device=DEVICE),
    )
    return torch.stack(
        (
            signs * 1e20,
            torch.full_like(signs, 1e20),
            torch.full_like(signs, float(2**125)),
        )
    )


def _c1_non_power_of_two_rows(dim):
    constant = torch.full((dim,), float(2**125), device=DEVICE, dtype=torch.float32)
    mixed = torch.full((dim,), float(2**100), device=DEVICE, dtype=torch.float32)
    mixed[0] = float(2**125)
    return torch.stack((constant, mixed))


def _run_raw_nvidia_candidate(x, weight, bias, eps, block_m, num_warps):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    rows, dim = x.shape
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    with flaggems_vllm.runtime.torch_device_fn.device(x.device):
        module._fused_indexer_k_norm_kernel[((rows + block_m - 1) // block_m,)](
            x,
            weight,
            bias,
            out,
            rows,
            dim,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            bias.stride(0),
            _eps32(eps),
            BLOCK_M=block_m,
            num_warps=num_warps,
            num_stages=1,
        )
    return out


def test_cpu_fp64_oracle_closed_form_and_properties():
    x = torch.tensor([[1.0, 3.0], [2.0, 2.0]], dtype=torch.float32)
    weight = torch.tensor([2.0, -1.0], dtype=torch.float32)
    bias = torch.tensor([0.5, 1.25], dtype=torch.float32)
    eps = 1e-4

    actual = cpu_fp64_oracle(x, weight, bias, eps)
    expected = torch.tensor([[-1.4999, 0.2501], [0.5, 1.25]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    singleton = cpu_fp64_oracle(x[:, :1], weight[:1], bias[:1], eps)
    torch.testing.assert_close(singleton, bias[:1].expand_as(singleton), rtol=0, atol=0)

    perm = torch.tensor([1, 0])
    torch.testing.assert_close(
        cpu_fp64_oracle(x[perm], weight, bias, eps), actual[perm], rtol=0, atol=0
    )

    scaled = cpu_fp64_oracle(x * 2.0 + 8.0, weight, bias, eps * 4.0)
    torch.testing.assert_close(scaled, actual, rtol=2e-5, atol=2e-5)


def test_cpu_fp64_oracle_agrees_with_source_for_bounded_values():
    x = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0], [3.0, 1.0, -1.0, -3.0]], dtype=torch.float32
    )
    weight = torch.tensor([0.5, -1.0, 1.5, 2.0], dtype=torch.float32)
    bias = torch.tensor([0.25, 0.5, -0.75, 1.0], dtype=torch.float32)
    expected = source_shadow_reference(x, weight, bias, 1e-6)
    _assert_close(cpu_fp64_oracle(x, weight, bias, 1e-6), expected, torch.float32)


def test_cpu_fp32_semantics_oracle_c1_extreme_rows_reduce_to_bias():
    dim = 128
    signs = torch.where(torch.arange(dim) % 2 == 0, torch.ones(dim), -torch.ones(dim))
    x = torch.stack(
        (
            signs * 1e20,
            torch.full_like(signs, 1e20),
            torch.full_like(signs, float(2**125)),
        )
    )
    weight = torch.linspace(0.5, 1.5, dim, dtype=torch.float32)
    bias = torch.linspace(-0.7, 0.9, dim, dtype=torch.float32)

    _assert_close(
        cpu_fp32_semantics_oracle(x, weight, bias, 1e-6),
        bias.expand_as(x),
        x.dtype,
    )


def test_export_and_library_only_contract():
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    public_op = flaggems_vllm.fused_indexer_k_norm

    assert public_op is ops.fused_indexer_k_norm
    assert public_op is module.fused_indexer_k_norm
    assert OP_NAME in ops.__all__
    assert OP_NAME not in {name for name, _ in flaggems_vllm._FULL_CONFIG}
    assert tuple(inspect.signature(public_op).parameters) == (
        "x",
        "weight",
        "bias",
        "dim",
        "eps",
        "out",
    )
    assert inspect.signature(public_op).parameters["out"].default is None


def test_nvidia_prune_norm_configs():
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    configs = tuple(
        SimpleNamespace(kwargs={"BLOCK_M": block_m}) for block_m in (1, 4, 8, 1)
    )

    assert module._prune_norm_configs(configs, {"D": 256}) == configs
    wide = module._prune_norm_configs(configs, {"D": 257})
    assert tuple(config.kwargs["BLOCK_M"] for config in wide) == (1, 1)


@pytest.mark.parametrize(
    "rows,expected",
    (
        (1, (1, 4)),
        (2, (4, 4)),
        (3, (4, 4)),
        (4, (4, 4)),
        (7, (4, 4)),
        (8, (4, 4)),
        (9, (4, 4)),
        (17, (4, 4)),
        (4095, (4, 4)),
        (4096, (8, 4)),
        (4097, (8, 4)),
        (8192, (8, 4)),
    ),
)
def test_h20z_d128_config_thresholds(rows, expected):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    assert module._h20z_d128_config(rows) == expected


@pytest.mark.parametrize(
    "metadata,expected",
    (
        ((1, 128, 128, 1, 1, 1), True),
        ((17, 128, 144, 1, 1, 1), True),
        ((8192, 128, 192, 1, 1, 1), True),
        ((0, 128, 192, 1, 1, 1), False),
        ((8193, 128, 192, 1, 1, 1), False),
        ((17, 129, 129, 1, 1, 1), False),
        ((17, 128, 160, 1, 1, 1), False),
        ((17, 128, 256, 2, 1, 1), False),
        ((17, 128, 128, 1, 2, 1), False),
        ((17, 128, 128, 1, 1, 2), False),
    ),
)
def test_h20z_d128_fast_path_layout_gate(monkeypatch, metadata, expected):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    monkeypatch.setattr(module, "_is_h20z", lambda device_index=None: True)
    monkeypatch.setattr(module.triton, "__version__", "3.6.0")
    assert module._use_h20z_d128_fast_path(*metadata) is expected


def test_h20z_d128_fast_path_device_gate(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    monkeypatch.setattr(module, "_is_h20z", lambda device_index=None: False)
    assert not module._use_h20z_d128_fast_path(17, 128, 192, 1, 1, 1)


def test_h20z_d128_fast_path_compiler_gate(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    monkeypatch.setattr(module, "_is_h20z", lambda device_index=None: True)
    monkeypatch.setattr(module.triton, "__version__", "3.5.0")
    assert not module._use_h20z_d128_fast_path(17, 128, 192, 1, 1, 1)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@NVIDIA_REQUIRED
@torch.inference_mode()
def test_h20z_d128_launcher_dispatches_fast_path_and_fallback(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    calls = []

    class Recorder:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((self.name, grid, args, kwargs))

            return launch

    monkeypatch.setattr(module, "_is_h20z", lambda device_index=None: True)
    monkeypatch.setattr(module.triton, "__version__", "3.6.0")
    monkeypatch.setattr(module, "_fused_indexer_k_norm_kernel", Recorder("fast"))
    monkeypatch.setattr(module, "_norm_kernel", Recorder("fallback"))

    parent = torch.empty((2, 192), dtype=torch.float32, device=DEVICE)
    x = parent[:, :128]
    weight = torch.empty((128,), dtype=torch.float32, device=DEVICE)
    bias = torch.empty_like(weight)
    out = torch.empty_like(x)
    module._launch_fused_indexer_k_norm(x, weight, bias, out, _eps32(1e-6))
    assert [call[0] for call in calls] == ["fast"]
    assert calls[0][3]["BLOCK_M"] == 4

    calls.clear()
    weight_parent = torch.empty((256,), dtype=torch.float32, device=DEVICE)
    strided_weight = weight_parent[::2]
    module._launch_fused_indexer_k_norm(x, strided_weight, bias, out, _eps32(1e-6))
    assert [call[0] for call in calls] == ["fallback"]
    assert "BLOCK_M" not in calls[0][3]
    grid = calls[0][1]
    assert callable(grid)
    for block_m in (1, 4, 8):
        assert grid({"BLOCK_M": block_m}) == ((x.shape[0] + block_m - 1) // block_m,)


@pytest.mark.parametrize("rows", (1, 8))
@pytest.mark.parametrize("dim", (127, 128, 129))
@pytest.mark.parametrize("gap", (0, 16))
def test_benchmark_records_row_gap(rows, dim, gap):
    from benchmark.test_fused_indexer_k_norm import FusedIndexerKNormBenchmark

    parent = torch.empty((rows, dim + gap), dtype=torch.float32)
    x = parent[:, :dim]
    weight = torch.empty((dim,), dtype=torch.float32)
    bias = torch.empty_like(weight)
    metadata = FusedIndexerKNormBenchmark().record_shapes(x, weight, bias, dim, 1e-6)

    assert metadata["M"] == rows
    assert metadata["D"] == dim
    assert metadata["row_gap"] == gap
    assert metadata["x_stride"] == [dim + gap, 1]


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("shape", FINITE_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_indexer_k_norm_matches_cpu_fp64_oracle(shape, dtype):
    x, weight, bias = _make_inputs(shape, dtype, seed=shape[0] * 8191 + shape[1])
    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, shape[1], 1e-6)

    _assert_close(out, cpu_fp64_oracle(x, weight, bias, 1e-6), dtype)
    _assert_fresh_contiguous(out, x, weight, bias)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", EPS_VALUES)
def test_fused_indexer_k_norm_eps_values(dtype, eps):
    x, weight, bias = _make_inputs((17, 128), dtype, seed=77)
    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, eps)
    _assert_close(out, cpu_fp64_oracle(x, weight, bias, eps), dtype)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", FP32_EDGE_EPS_VALUES)
def test_fp32_edge_eps_constant_and_d1_source_parity(dtype, eps):
    for d in (1, 128):
        x, weight, bias = _constant_row(dtype, d=d)
        out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, d, eps)
        shadow = source_shadow_reference(x, weight, bias, eps)
        _assert_source_parity(out, shadow, dtype)
        _assert_close(out, bias.to(dtype).expand_as(out), dtype)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("dtype", DTYPES)
def test_constant_rows_and_d1_reduce_to_bias(dtype):
    for d in (1, 128):
        x, weight, bias = _constant_row(dtype, d=d)
        out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, d, 1e-6)
        expected = bias.to(dtype).expand_as(out)
        _assert_close(out, expected, dtype)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@NVIDIA_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("block_m,num_warps", NVIDIA_RAW_CONFIGS)
def test_nvidia_raw_candidates_d128(block_m, num_warps):
    x, weight, bias = _make_inputs((17, 128), torch.float32, seed=119)
    out = _run_raw_nvidia_candidate(x, weight, bias, 1e-6, block_m, num_warps)
    _assert_close(out, cpu_fp64_oracle(x, weight, bias, 1e-6), torch.float32)
    _assert_fresh_contiguous(out, x, weight, bias)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@NVIDIA_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("block_m,num_warps", NVIDIA_WIDE_RAW_CONFIGS)
def test_nvidia_raw_wide_candidates_pruned_to_bm1(block_m, num_warps):
    x, weight, bias = _make_inputs((3, 4095), torch.float32, seed=120)
    out = _run_raw_nvidia_candidate(x, weight, bias, 1e-6, block_m, num_warps)
    _assert_close(out, cpu_fp64_oracle(x, weight, bias, 1e-6), torch.float32)
    _assert_fresh_contiguous(out, x, weight, bias)


def _layout_case(kind, dtype):
    m, d = 17, 128
    torch.manual_seed(9182)
    weight_parent = torch.randn((2 * d + 3,), dtype=torch.float32, device=DEVICE)
    bias_parent = torch.randn((2 * d + 5,), dtype=torch.float32, device=DEVICE)
    weight = weight_parent[1 : 1 + 2 * d : 2]
    bias = bias_parent[3 : 3 + 2 * d : 2]

    if kind == "rowgap144":
        x_parent = torch.randn((m + 2, d + 16), dtype=dtype, device=DEVICE)
        x = x_parent[1 : 1 + m, 3 : 3 + d]
        parents = (x_parent, weight_parent, bias_parent)
    elif kind == "rowgap192":
        x_parent = torch.randn((m + 2, d + 64), dtype=dtype, device=DEVICE)
        x = x_parent[1 : 1 + m, 3 : 3 + d]
        parents = (x_parent, weight_parent, bias_parent)
    elif kind == "inner_stride2":
        x_parent = torch.randn((m + 1, d * 2 + 2), dtype=dtype, device=DEVICE)
        x = x_parent[1:, 1 : 1 + 2 * d : 2]
        parents = (x_parent, weight_parent, bias_parent)
    elif kind == "transpose":
        x_parent = torch.randn((d, m + 1), dtype=dtype, device=DEVICE)
        x = x_parent[:, 1:].t()
        parents = (x_parent, weight_parent, bias_parent)
    elif kind == "row_broadcast":
        x_parent = torch.randn((1, d), dtype=dtype, device=DEVICE)
        x = x_parent.expand(m, d)
        parents = (x_parent, weight_parent, bias_parent)
    elif kind == "feature_broadcast":
        x_parent = torch.randn((m, 1), dtype=dtype, device=DEVICE)
        x = x_parent.expand(m, d)
        parents = (x_parent, weight_parent, bias_parent)
    elif kind == "affine_broadcast":
        x_parent = torch.randn((m, d), dtype=dtype, device=DEVICE)
        weight_parent = torch.randn((1,), dtype=torch.float32, device=DEVICE)
        bias_parent = torch.randn((1,), dtype=torch.float32, device=DEVICE)
        x = x_parent
        weight = weight_parent.expand(d)
        bias = bias_parent.expand(d)
        parents = (x_parent, weight_parent, bias_parent)
    else:
        raise AssertionError(f"unknown layout case: {kind}")

    return x, weight, bias, parents


def _h20z_fast_path_offset_case(rows, dtype):
    torch.manual_seed(rows + 2718)
    x_parent = torch.randn((rows + 1, 192), dtype=dtype, device=DEVICE)
    weight_parent = torch.randn((130,), dtype=torch.float32, device=DEVICE)
    bias_parent = torch.randn((132,), dtype=torch.float32, device=DEVICE)
    x = x_parent[1 : 1 + rows, 3:131]
    weight = weight_parent[1:129]
    bias = bias_parent[3:131]
    return x, weight, bias, (x_parent, weight_parent, bias_parent)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize(
    "kind",
    (
        "rowgap144",
        "rowgap192",
        "inner_stride2",
        "transpose",
        "row_broadcast",
        "feature_broadcast",
        "affine_broadcast",
    ),
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_strides_offsets_and_no_alias(kind, dtype):
    x, weight, bias, parents = _layout_case(kind, dtype)
    before_inputs = tuple(t.clone() for t in (x, weight, bias))
    before_parents = tuple(t.clone() for t in parents)

    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, x.shape[1], 1e-6)

    _assert_close(out, cpu_fp64_oracle(x, weight, bias, 1e-6), dtype)
    _assert_fresh_contiguous(out, x, weight, bias)
    for actual, before in zip((x, weight, bias), before_inputs):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
    for actual, before in zip(parents, before_parents):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)

    out.fill_(0)
    for actual, before in zip((x, weight, bias), before_inputs):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@NVIDIA_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("rows", (1, 17, 4096))
@pytest.mark.parametrize("dtype", DTYPES)
def test_h20z_d128_real_layout_configs(rows, dtype):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    if not _h20z_d128_fast_path_runtime_available(module):
        pytest.skip("H20Z/Triton-specific production fast path")
    torch.manual_seed(rows + 864)
    parent = torch.randn((rows, 192), dtype=dtype, device=DEVICE)
    x = parent[:, :128]
    weight = torch.randn((128,), dtype=torch.float32, device=DEVICE)
    bias = torch.randn((128,), dtype=torch.float32, device=DEVICE)

    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)

    assert x.stride() == (192, 1)
    assert module._use_h20z_d128_fast_path(rows, 128, 192, 1, 1, 1, x.device.index)
    _assert_close(out, cpu_fp64_oracle(x, weight, bias, 1e-6), dtype)
    _assert_fresh_contiguous(out, x, weight, bias)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@NVIDIA_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("dtype", DTYPES)
def test_h20z_d128_non_aligned_offsets_use_fast_path(dtype):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    if not _h20z_d128_fast_path_runtime_available(module):
        pytest.skip("H20Z/Triton-specific production fast path")
    x, weight, bias, parents = _h20z_fast_path_offset_case(17, dtype)
    before = tuple(t.clone() for t in (x, weight, bias, *parents))

    assert x.stride() == (192, 1)
    assert weight.stride() == (1,)
    assert bias.stride() == (1,)
    assert x.data_ptr() % 16 != 0
    assert weight.data_ptr() % 16 != 0
    assert bias.data_ptr() % 16 != 0
    assert module._use_h20z_d128_fast_path(17, 128, 192, 1, 1, 1, x.device.index)

    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)

    _assert_close(out, cpu_fp64_oracle(x, weight, bias, 1e-6), dtype)
    _assert_fresh_contiguous(out, x, weight, bias)
    for actual, expected in zip((x, weight, bias, *parents), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@NVIDIA_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("rows", (1, 2, 3, 4, 7, 8, 9, 17, 4095, 4096, 4097, 8192))
def test_h20z_d128_tiny_eps_thresholds(rows):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    if not _h20z_d128_fast_path_runtime_available(module):
        pytest.skip("H20Z/Triton-specific production fast path")
    parent = torch.full((rows, 192), 2.0, dtype=torch.float32, device=DEVICE)
    x = parent[:, :128]
    weight, bias = _c1_affine(128)
    eps = 2.0**-149

    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, eps)

    _assert_close(out, bias.expand_as(out), x.dtype)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@NVIDIA_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("rows", (1, 17, 4096))
def test_h20z_d128_extreme_and_tiny_variance_configs(rows):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    if not _h20z_d128_fast_path_runtime_available(module):
        pytest.skip("H20Z/Triton-specific production fast path")
    weight, bias = _c1_affine(128)
    parent = torch.empty((rows, 192), dtype=torch.float32, device=DEVICE)

    for c1_row in _c1_original_extreme_rows():
        parent[:, :128] = c1_row
        extreme = parent[:, :128]
        extreme_out = flaggems_vllm.fused_indexer_k_norm(
            extreme, weight, bias, 128, 1e-6
        )
        _assert_close(extreme_out, bias.expand_as(extreme_out), extreme.dtype)

    signs = torch.where(
        torch.arange(128, device=DEVICE) % 2 == 0,
        torch.ones(128, device=DEVICE),
        -torch.ones(128, device=DEVICE),
    )
    parent[:, :128] = signs * 2.0**-70
    tiny = parent[:, :128]
    for eps in FP32_EDGE_EPS_VALUES + (2.0**-125,):
        tiny_out = flaggems_vllm.fused_indexer_k_norm(tiny, weight, bias, 128, eps)
        _assert_close(tiny_out, cpu_fp64_oracle(tiny, weight, bias, eps), tiny.dtype)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@NVIDIA_REQUIRED
@torch.inference_mode()
def test_h20z_d128_non_default_stream(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    if not _h20z_d128_fast_path_runtime_available(module):
        pytest.skip("H20Z/Triton-specific production fast path")
    x, weight, bias, _ = _h20z_fast_path_offset_case(17, torch.float32)
    gate_calls = []
    original_gate = module._use_h20z_d128_fast_path

    def record_gate(*args, **kwargs):
        result = original_gate(*args, **kwargs)
        gate_calls.append((args, kwargs, result))
        return result

    assert original_gate(17, 128, 192, 1, 1, 1, x.device.index)
    monkeypatch.setattr(module, "_use_h20z_d128_fast_path", record_gate)
    stream = flaggems_vllm.runtime.torch_device_fn.Stream()
    stream.wait_stream(flaggems_vllm.runtime.torch_device_fn.current_stream())
    with flaggems_vllm.runtime.torch_device_fn.stream(stream):
        out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)
    stream.synchronize()

    assert len(gate_calls) == 1
    assert gate_calls[0][2] is True
    _assert_close(out, cpu_fp64_oracle(x, weight, bias, 1e-6), x.dtype)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@NVIDIA_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("rows", (1, 4096))
def test_h20z_d128_nan_inf_source_parity_for_bm1_and_bm8(rows):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    if not _h20z_d128_fast_path_runtime_available(module):
        pytest.skip("H20Z/Triton-specific production fast path")
    x, weight, bias, _ = _h20z_fast_path_offset_case(rows, torch.float32)
    assert module._use_h20z_d128_fast_path(rows, 128, 192, 1, 1, 1, x.device.index)

    x[0, 3] = float("nan")
    x[min(rows - 1, 1), 7] = float("inf")
    x[min(rows - 1, 2), 9] = float("-inf")
    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)
    _assert_source_parity(out, source_shadow_reference(x, weight, bias, 1e-6), x.dtype)

    finite_x, weight, bias, _ = _h20z_fast_path_offset_case(rows, torch.float32)
    weight[11] = float("nan")
    bias[29] = float("inf")
    out = flaggems_vllm.fused_indexer_k_norm(finite_x, weight, bias, 128, 1e-6)
    _assert_source_parity(
        out, source_shadow_reference(finite_x, weight, bias, 1e-6), finite_x.dtype
    )


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("dtype", DTYPES)
def test_row_permutation_and_large_offset(dtype):
    x, weight, bias = _make_inputs((17, 128), dtype, seed=442)
    perm = torch.tensor([16, 0, 8, 2, 5, 1, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15])
    base = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)
    permuted = flaggems_vllm.fused_indexer_k_norm(x[perm], weight, bias, 128, 1e-6)
    _assert_close(permuted, base[perm], dtype)

    centered = torch.arange(128, device=DEVICE, dtype=torch.float32) - 63.5
    if dtype == torch.float32:
        values = 2**20 + 0.25 * centered
    elif dtype == torch.float16:
        values = 2048.0 + 2.0 * centered
    else:
        values = 512.0 + 4.0 * centered
    large_offset = values.to(dtype).expand(3, -1)
    out = flaggems_vllm.fused_indexer_k_norm(large_offset, weight, bias, 128, 1e-6)
    _assert_close(out, cpu_fp64_oracle(large_offset, weight, bias, 1e-6), dtype)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("dtype", DTYPES)
def test_near_zero_cpu_fp64_oracle(dtype):
    values = torch.linspace(-1e-4, 1e-4, 128, device=DEVICE, dtype=torch.float32)
    x = torch.stack((values, -values, values.roll(17))).to(dtype)
    weight = torch.linspace(-1.0, 1.0, 128, device=DEVICE, dtype=torch.float32)
    bias = torch.linspace(-0.5, 0.5, 128, device=DEVICE, dtype=torch.float32)

    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)
    _assert_close(out, cpu_fp64_oracle(x, weight, bias, 1e-6), dtype)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("dtype", DTYPES)
def test_nan_inf_source_parity(dtype):
    x, weight, bias = _make_inputs((4, 128), dtype, seed=998)
    x[1, 3] = float("nan")
    x[2, 7] = float("inf")
    x[3, 9] = float("-inf")

    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)
    shadow = source_shadow_reference(x, weight, bias, 1e-6)
    _assert_source_parity(out, shadow, dtype)
    _assert_close(out[0], cpu_fp64_oracle(x[0:1], weight, bias, 1e-6)[0], dtype)

    finite_x, weight, bias = _make_inputs((3, 128), dtype, seed=999)
    weight[11] = float("nan")
    bias[29] = float("inf")
    out = flaggems_vllm.fused_indexer_k_norm(finite_x, weight, bias, 128, 1e-6)
    shadow = source_shadow_reference(finite_x, weight, bias, 1e-6)
    _assert_source_parity(out, shadow, dtype)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
def test_extreme_finite_c1_oracle_and_source_compatibility_diagnostic(record_property):
    x = _c1_original_extreme_rows()
    weight, bias = _c1_affine(x.shape[1])

    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, x.shape[1], 1e-6)
    expected = cpu_fp32_semantics_oracle(x, weight, bias, 1e-6)
    _assert_close(out, expected, x.dtype)
    _assert_close(expected, bias.expand_as(expected), x.dtype)
    record_property(
        "compatibility_eager_shadow",
        _source_compatibility_diagnostic(
            out, source_shadow_reference(x, weight, bias, 1e-6)
        ),
    )


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("dim", (127, 129))
def test_c1_non_power_of_two_high_amplitude_rows(dim, record_property):
    x = _c1_non_power_of_two_rows(dim)
    weight, bias = _c1_affine(dim)

    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, dim, 1e-6)
    expected = cpu_fp32_semantics_oracle(x, weight, bias, 1e-6)
    _assert_close(out, expected, x.dtype)
    _assert_close(expected, bias.expand_as(expected), x.dtype)
    record_property(
        "compatibility_eager_shadow",
        _source_compatibility_diagnostic(
            out, source_shadow_reference(x, weight, bias, 1e-6)
        ),
    )


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize(
    "invalid",
    (
        "rank",
        "d0",
        "d4097",
        "dim_bool",
        "dim_mismatch",
        "x_dtype",
        "weight_dtype",
        "bias_shape",
        "weight_shape",
        "weight_none",
        "eps_zero",
        "eps_negative",
        "eps_nan",
        "eps_inf",
        "eps_underflow",
        "eps_overflow",
        "eps_bool",
    ),
)
def test_invalid_metadata_rejected_before_launch(monkeypatch, invalid):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    x, weight, bias = _make_inputs((2, 128), torch.float32, seed=19)
    dim, eps = 128, 1e-6

    if invalid == "rank":
        x = x[0]
    elif invalid == "d0":
        x = torch.empty((2, 0), dtype=torch.float32, device=DEVICE)
        weight = torch.empty((0,), dtype=torch.float32, device=DEVICE)
        bias = torch.empty((0,), dtype=torch.float32, device=DEVICE)
        dim = 0
    elif invalid == "d4097":
        x, weight, bias = _make_inputs((1, 4097), torch.float32, seed=20)
        dim = 4097
    elif invalid == "dim_bool":
        dim = True
    elif invalid == "dim_mismatch":
        dim = 127
    elif invalid == "x_dtype":
        # Ascend silently converts a requested device FP64 tensor back to FP32.
        # Use a genuinely unsupported device dtype; FP64 metadata is checked
        # separately below without relying on device FP64 allocation support.
        x = x.to(torch.int32)
        assert x.dtype == torch.int32
    elif invalid == "weight_dtype":
        weight = weight.to(torch.float16)
    elif invalid == "bias_shape":
        bias = bias.unsqueeze(0)
    elif invalid == "weight_shape":
        weight = weight[:-1]
    elif invalid == "weight_none":
        weight = None
    elif invalid == "eps_zero":
        eps = 0.0
    elif invalid == "eps_negative":
        eps = -1e-6
    elif invalid == "eps_nan":
        eps = float("nan")
    elif invalid == "eps_inf":
        eps = float("inf")
    elif invalid == "eps_underflow":
        eps = 1e-50
    elif invalid == "eps_overflow":
        eps = 1e40
    elif invalid == "eps_bool":
        eps = True
    else:
        raise AssertionError(f"unknown invalid case: {invalid}")

    calls = []

    def reject_launch(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid metadata reached the kernel launcher")

    monkeypatch.setattr(module, "_launch_fused_indexer_k_norm", reject_launch)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        flaggems_vllm.fused_indexer_k_norm(x, weight, bias, dim, eps)
    assert calls == []


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
def test_device_mismatch_rejected_before_launch(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    x, _, bias = _make_inputs((2, 128), torch.float32, seed=51)
    weight = torch.randn(128, dtype=torch.float32, device="cpu")
    calls = []
    monkeypatch.setattr(
        module,
        "_launch_fused_indexer_k_norm",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises((ValueError, NotImplementedError)):
        flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)
    assert calls == []


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
def test_grad_contract():
    x, weight, bias = _make_inputs((3, 128), torch.float32, seed=711)
    weight.requires_grad_()
    bias.requires_grad_()

    with torch.no_grad():
        out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)
    _assert_close(out, cpu_fp64_oracle(x, weight, bias, 1e-6), torch.float32)

    x.requires_grad_()
    with torch.enable_grad(), pytest.raises(NotImplementedError, match="inference"):
        flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)


def test_eps_contract_is_finite_after_fp32_conversion():
    assert _eps32(1e-6) > 0.0
    assert math.isfinite(_eps32(1e-6))
    assert _eps32(1e-50) == 0.0
    assert math.isinf(_eps32(1e40))


def test_fp64_metadata_rejected_before_device_check(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    x = torch.empty((2, 128), dtype=torch.float64, device="cpu")
    weight = torch.empty((128,), dtype=torch.float32, device="cpu")
    bias = torch.empty_like(weight)

    def reject_launch(*args, **kwargs):
        raise AssertionError("unsupported FP64 reached the launcher")

    monkeypatch.setattr(module, "_launch_fused_indexer_k_norm", reject_launch)
    with pytest.raises(NotImplementedError, match="FP16, BF16, or FP32"):
        flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("eps", (2.0**-149, 2.0**-126, 2.0**-125))
def test_subnormal_and_normal_variance_nonconstant(dtype, eps):
    # Exact powers of two keep the CPU FP64 oracle independent of FTZ behavior.
    # These rows exercise a subnormal variance and the normal boundary, rather
    # than allowing a constant-row shortcut to hide a broken rsqrt path.
    signs = torch.tensor([-1.0, 1.0] * 64, dtype=torch.float32, device="cpu")
    x = torch.stack((signs * 2.0**-70, signs * 2.0**-63)).to(device=DEVICE, dtype=dtype)
    weight = torch.linspace(0.5, 1.5, 128, device=DEVICE, dtype=torch.float32)
    bias = torch.linspace(-0.25, 0.25, 128, device=DEVICE, dtype=torch.float32)
    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, eps)
    _assert_close(out, cpu_fp64_oracle(x, weight, bias, eps), dtype)


ROW_TAIL_D128_CASES = (
    (7, 0),
    (9, 16),
    (63, 0),
    (65, 16),
    (4095, 0),
    (4097, 16),
)


def _d128_offset_inputs(rows, row_gap, dtype, seed):
    torch.manual_seed(seed)
    parent = torch.randn((rows + 1, 128 + row_gap), dtype=dtype, device=DEVICE)
    # Both layouts have a nonzero storage offset.  Their row strides are 128
    # and 144 respectively, so this checks gap=0 and gap=16 without a view
    # that changes the logical D=128 input.
    x = parent[1:, :128]
    weight_parent = torch.randn((130,), dtype=torch.float32, device=DEVICE)
    bias_parent = torch.randn((131,), dtype=torch.float32, device=DEVICE)
    weight = weight_parent[1:129]
    bias = bias_parent[2:130]
    assert x.shape == (rows, 128)
    assert x.stride() == (128 + row_gap, 1)
    assert x.storage_offset() > 0
    assert weight.storage_offset() > 0
    assert bias.storage_offset() > 0
    return x, weight, bias, (parent, weight_parent, bias_parent)


def _call_d128_with_optional_out(x, weight, bias, provided):
    before = tuple(t.clone() for t in (x, weight, bias))
    supplied = torch.full_like(x, float("nan")) if provided else None
    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6, out=supplied)
    if supplied is None:
        _assert_fresh_contiguous(out, x, weight, bias)
    else:
        assert out is supplied
        assert out.data_ptr() == supplied.data_ptr()
        assert out.is_contiguous()
    for actual, expected in zip((x, weight, bias), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return out


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("rows,row_gap", ROW_TAIL_D128_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("provided", (False, True))
def test_d128_row_tail_offsets_all_dtypes_and_out_modes(rows, row_gap, dtype, provided):
    x, weight, bias, parents = _d128_offset_inputs(
        rows, row_gap, dtype, seed=6100 + rows + row_gap
    )
    parents_before = tuple(parent.clone() for parent in parents)

    out = _call_d128_with_optional_out(x, weight, bias, provided)

    _assert_close(out, cpu_fp64_oracle(x, weight, bias, 1e-6), dtype)
    for actual, expected in zip(parents, parents_before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("provided", (False, True))
@pytest.mark.parametrize("rows", (9, 129, 1025, 4097))
def test_d128_fp32_nonconstant_tiny_eps_row_tail(provided, rows):
    x, weight, bias, _ = _d128_offset_inputs(rows, 16, torch.float32, seed=6201)
    signs = torch.where(
        torch.arange(128, device=DEVICE) % 2 == 0,
        torch.ones(128, device=DEVICE),
        -torch.ones(128, device=DEVICE),
    )
    patterns = torch.stack((signs * 2.0**-70, signs * 2.0**-63))
    x.copy_(patterns.repeat((rows + 1) // 2, 1)[:rows])
    supplied = torch.empty_like(x) if provided else None

    out = flaggems_vllm.fused_indexer_k_norm(
        x, weight, bias, 128, 2.0**-149, out=supplied
    )

    if supplied is not None:
        assert out is supplied
    _assert_close(out, cpu_fp64_oracle(x, weight, bias, 2.0**-149), torch.float32)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("provided", (False, True))
@pytest.mark.parametrize("rows", (65, 129, 1025, 4097))
def test_d128_fp32_high_finite_c1_row_tail(provided, rows):
    x, weight, bias, _ = _d128_offset_inputs(rows, 16, torch.float32, seed=6301)
    signs = torch.where(
        torch.arange(128, device=DEVICE) % 2 == 0,
        torch.ones(128, device=DEVICE),
        -torch.ones(128, device=DEVICE),
    )
    alternating = signs * float(2**125)
    mixed = torch.full((128,), float(2**100), dtype=torch.float32, device=DEVICE)
    mixed[0] = float(2**125)
    x.copy_(torch.stack((alternating, mixed)).repeat((rows + 1) // 2, 1)[:rows])
    supplied = torch.empty_like(x) if provided else None

    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6, out=supplied)

    if supplied is not None:
        assert out is supplied
    # Each row repeats one of two known input patterns. Compute the independent
    # scalar FP32 oracle once per pattern, then repeat its expected rows.
    expected = cpu_fp32_semantics_oracle(x[:2], weight, bias, 1e-6)
    expected = expected.repeat((rows + 1) // 2, 1)[:rows]
    _assert_close(out, expected, torch.float32)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("provided", (False, True))
@pytest.mark.parametrize("rows", (129, 4097))
def test_d128_nonfinite_rows_across_tiles(dtype, provided, rows):
    x, weight, bias, _ = _d128_offset_inputs(rows, 16, dtype, seed=6401)
    x[64, 17] = float("inf")
    x[-1, 31] = float("nan")
    supplied = torch.empty_like(x) if provided else None

    out = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6, out=supplied)

    if supplied is not None:
        assert out is supplied
    _assert_source_parity(out, source_shadow_reference(x, weight, bias, 1e-6), dtype)
    _assert_close(out[:1], cpu_fp64_oracle(x[:1], weight, bias, 1e-6), dtype)


@pytest.mark.fused_indexer_k_norm
@DEVICE_REQUIRED
@pytest.mark.skipif(
    flaggems_vllm.runtime.device.vendor_name != "ascend",
    reason="Ascend shared-tile dispatch coverage",
)
@torch.inference_mode()
def test_ascend_shared_tile_dispatch_and_compatibility(monkeypatch):
    shared = importlib.import_module("flaggems_vllm.ops.fused_indexer_k_norm")
    ascend = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.ops.fused_indexer_k_norm"
    )
    x, weight, bias, _ = _d128_offset_inputs(129, 16, torch.float32, seed=6501)
    if not ascend._shared_tile_validated_device(x.device.index):
        pytest.skip("shared tile is not enabled for this device/compiler")
    expected = cpu_fp64_oracle(x, weight, bias, 1e-6)
    calls = []

    class ForwardingKernel:
        def __init__(self, name, kernel):
            self.name = name
            self.kernel = kernel

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(self.name)
                return self.kernel[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(
        shared,
        "_fused_indexer_k_norm_kernel",
        ForwardingKernel("shared", shared._fused_indexer_k_norm_kernel),
    )
    monkeypatch.setattr(
        ascend,
        "_ascend_fused_indexer_k_norm_kernel",
        ForwardingKernel("compatibility", ascend._ascend_fused_indexer_k_norm_kernel),
    )
    actual = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)
    assert calls == ["shared"]
    _assert_close(actual, expected, torch.float32)

    calls.clear()
    monkeypatch.setattr(ascend, "_shared_tile_validated_device", lambda _index: False)
    actual = flaggems_vllm.fused_indexer_k_norm(x, weight, bias, 128, 1e-6)
    assert calls == ["compatibility"]
    _assert_close(actual, expected, torch.float32)
