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

"""Numerical contract and strict installed-vLLM kernel comparisons."""

import ast
import importlib
import importlib.util
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

import flaggems_vllm

from . import conftest as cfg

OP = flaggems_vllm.silu_mul_per_token_group_quant_fp8_colmajor
_CORE = importlib.import_module(
    "flaggems_vllm.ops.silu_mul_per_token_group_quant_fp8_colmajor"
)
pytestmark = [
    pytest.mark.silu_mul_per_token_group_quant_fp8_colmajor,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name
        not in ("nvidia", "metax", "thead", "mthreads", "hygon"),
        reason="this backend has no validated fused FP8 implementation",
    ),
]


def test_backend_routing():
    prefix = "flaggems_vllm.ops"
    if flaggems_vllm.vendor_name != "nvidia":
        prefix = f"flaggems_vllm.runtime.backend._{flaggems_vllm.vendor_name}.ops"
    expected_module = f"{prefix}.silu_mul_per_token_group_quant_fp8_colmajor"
    assert OP.__module__ == expected_module
    assert OP is getattr(
        importlib.import_module(expected_module),
        "silu_mul_per_token_group_quant_fp8_colmajor",
    )


@lru_cache(maxsize=1)
def _installed_vllm_reference():
    """Load unchanged installed function ASTs, isolating vLLM plugin setup.

    Use the same platform FP8 dtype as the operator, with scale mode passed
    explicitly. This checks the kernel, not end-to-end vLLM platform support.
    """
    spec = importlib.util.find_spec("vllm")
    if spec is None:
        pytest.skip("installed vLLM reference is unavailable")
    path = Path(next(iter(spec.submodule_search_locations))) / (
        "model_executor/layers/quantization/utils/fp8_utils.py"
    )
    source = path.read_text()
    names = {
        "_silu_mul_per_token_group_quant_fp8_colmajor",
        "silu_mul_per_token_group_quant_fp8_colmajor",
    }
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(functions) != 2:
        pytest.skip("installed vLLM has no compatible Triton reference")
    host = next(node for node in functions if not node.name.startswith("_"))
    if "clamp_limit" not in {arg.arg for arg in host.args.args}:
        pytest.skip("installed vLLM predates the v0.23 clamp/rounding contract")
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    namespace = {
        "torch": torch,
        "triton": triton,
        "tl": tl,
        "current_platform": current_platform,
        "is_deep_gemm_e8m0_used": is_deep_gemm_e8m0_used,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["silu_mul_per_token_group_quant_fp8_colmajor"]


def _cpu_reference(x, use_ue8m0=False, eps=1e-10, clamp_limit=None):
    gate, up = x.cpu().chunk(2, dim=-1)
    if clamp_limit is not None:
        gate = gate.float().clamp(max=clamp_limit).to(x.dtype)
        up = up.float().clamp(-clamp_limit, clamp_limit).to(x.dtype)
    gate_f32 = gate.float()
    activated = (gate_f32 / (1 + torch.exp(-gate_f32))).to(x.dtype)
    values = (activated * up).to(x.dtype).float()
    groups = values.reshape(x.shape[0], x.shape[1] // 256, 128)
    fp8_dtype = _CORE._platform_fp8_dtype()
    fp8_max = 224.0 if fp8_dtype == torch.float8_e4m3fnuz else 448.0
    scale = groups.abs().amax(-1).clamp(min=eps) * (1.0 / fp8_max)
    if use_ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (groups / scale[..., None]).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return q.reshape(x.shape[0], x.shape[1] // 2), scale


def _assert_reference_equal(actual, expected):
    q, s = actual
    ref_q, ref_s = expected
    assert q.dtype == _CORE._platform_fp8_dtype()
    assert q.element_size() == 1
    assert s.dtype == torch.float32
    assert s.stride() == (1, q.shape[0])
    # NaN payload/sign need not be stable. Every non-NaN FP8 byte, including
    # signed zeros, must match the installed kernel exactly.
    actual_bytes = q.view(torch.uint8).cpu()
    expected_bytes = ref_q.view(torch.uint8).cpu()
    expected_float = ref_q.cpu().float()
    actual_float = q.cpu().float()
    assert torch.equal(torch.isnan(actual_float), torch.isnan(expected_float))
    finite_mask = ~torch.isnan(expected_float)
    assert torch.equal(actual_bytes[finite_mask], expected_bytes[finite_mask])
    torch.testing.assert_close(s.cpu(), ref_s.cpu(), rtol=1e-6, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [(128, 256), (128, 4096), (512, 4096), (1024, 1792), (8192, 4096)],
)
@pytest.mark.parametrize("use_ue8m0", [False, True])
@pytest.mark.parametrize("clamp_limit", [None, 0.0, 2.0])
def test_matches_installed_vllm(dtype, shape, use_ue8m0, clamp_limit):
    if cfg.QUICK_MODE and shape[0] > 128:
        pytest.skip("large shape omitted in quick mode")
    torch.manual_seed(1337)
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    before = x.clone()
    kwargs = dict(use_ue8m0=use_ue8m0, clamp_limit=clamp_limit)
    ref = _installed_vllm_reference()(x, **kwargs)
    actual = OP(x, **kwargs)
    _assert_reference_equal(actual, ref)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    # Separate independent mathematical check; CPU/GPU exp may straddle an
    # FP8 rounding boundary. This is not the strict GPU equality criterion.
    cpu_q, cpu_s = _cpu_reference(x, **kwargs)
    torch.testing.assert_close(actual[1].cpu(), cpu_s, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(
        actual[0].cpu().float(), cpu_q.float(), rtol=0.125, atol=2**-9
    )


@pytest.mark.parametrize("special", ["zero", "nan", "inf", "negative_inf"])
@pytest.mark.parametrize("use_ue8m0", [False, True])
def test_special_values_match_vllm(special, use_ue8m0):
    x = torch.zeros((128, 256), dtype=torch.float32, device=flaggems_vllm.device)
    if special == "zero":
        x[:, 1] = -0.0
        x[:, 129] = 1.0
    else:
        x[:, 0] = {"nan": torch.nan, "inf": torch.inf, "negative_inf": -torch.inf}[
            special
        ]
        x[:, 128:] = 1.0
    ref = _installed_vllm_reference()(x, use_ue8m0=use_ue8m0)
    _assert_reference_equal(OP(x, use_ue8m0=use_ue8m0), ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_signed_zero_clamp_cache(dtype):
    """Scalar signed zero must remain correct across repeated compiled calls."""
    torch.manual_seed(2027)
    x = torch.randn((1024, 1280), dtype=dtype, device=flaggems_vllm.device)
    reference = _installed_vllm_reference()
    for clamp_limit in (0.0, -0.0, 0.0):
        kwargs = dict(use_ue8m0=False, clamp_limit=clamp_limit)
        _assert_reference_equal(OP(x, **kwargs), reference(x, **kwargs))


def test_output_and_default_scale_mode():
    x = torch.randn((1024, 1280), dtype=torch.bfloat16, device=flaggems_vllm.device)
    output = torch.empty(
        (1024, 640), dtype=_CORE._platform_fp8_dtype(), device=x.device
    )
    actual = OP(x, output=output)
    assert actual[0] is output
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    _assert_reference_equal(actual, _installed_vllm_reference()(x))
    _assert_reference_equal(actual, OP(x, use_ue8m0=is_deep_gemm_e8m0_used()))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_ue8m0", [False, True])
@pytest.mark.parametrize("clamp_limit", [None, 0.0, 2.0])
def test_independent_cpu_reference(dtype, use_ue8m0, clamp_limit):
    """Check numerical semantics against an independent CPU calculation."""
    torch.manual_seed(2026)
    x = torch.randn((1024, 1280), dtype=dtype, device=flaggems_vllm.device)
    kwargs = dict(use_ue8m0=use_ue8m0, clamp_limit=clamp_limit)
    before = x.cpu()
    ref_q, ref_s = _cpu_reference(before, **kwargs)
    q, s = OP(x, **kwargs)
    torch.testing.assert_close(x.cpu(), before, rtol=0, atol=0)
    assert q.dtype == _CORE._platform_fp8_dtype()
    assert q.element_size() == 1
    assert q.device == x.device and s.device == x.device
    assert q.is_contiguous()
    assert s.stride() == (1, x.shape[0])
    torch.testing.assert_close(s.cpu(), ref_s, rtol=1e-5, atol=1e-8)
    # CPU and accelerator exp may straddle an FP8 rounding boundary. Strict
    # byte comparisons against the native reference remain separate above.
    torch.testing.assert_close(q.cpu().float(), ref_q.float(), rtol=0.125, atol=2**-9)


@pytest.mark.parametrize("shape", [(0, 256), (128, 0)])
def test_empty(shape):
    x = torch.empty(shape, dtype=torch.float16, device=flaggems_vllm.device)
    q, s = OP(x)
    assert q.shape == (shape[0], shape[1] // 2)
    assert q.dtype == _CORE._platform_fp8_dtype()
    assert s.shape == (shape[0], shape[1] // 256)
    assert s.dtype == torch.float32


@pytest.mark.parametrize(
    "kind",
    [
        "rank",
        "rows",
        "columns",
        "dtype",
        "layout",
        "out_shape",
        "out_layout",
        "alias",
        "device",
    ],
)
def test_invalid_contract(kind):
    x = torch.empty((128, 256), dtype=torch.float16, device=flaggems_vllm.device)
    kwargs = {}
    if kind == "rank":
        x = x.view(128, 2, 128)
    elif kind == "rows":
        x = x[:127]
    elif kind == "columns":
        x = torch.empty((128, 258), dtype=x.dtype, device=x.device)
    elif kind == "dtype":
        x = torch.empty(x.shape, dtype=torch.int32, device=x.device)
    elif kind == "layout":
        x = x.T
    elif kind.startswith("out_"):
        shape = (128, 127) if kind == "out_shape" else (128, 128)
        dtype = torch.float16 if kind == "out_dtype" else _CORE._platform_fp8_dtype()
        output = torch.empty(shape, dtype=dtype, device=x.device)
        kwargs["output"] = output.T if kind == "out_layout" else output
    elif kind == "alias":
        kwargs["output"] = (
            x.view(_CORE._platform_fp8_dtype()).view(-1)[: 128 * 128].view(128, 128)
        )
    elif kind == "device":
        kwargs["output"] = torch.empty((128, 128), dtype=_CORE._platform_fp8_dtype())
    with pytest.raises((ValueError, TypeError, NotImplementedError, AssertionError)):
        OP(x, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eps": 0.0},
        {"eps": -1.0},
        {"eps": float("nan")},
        {"eps": float("inf")},
        {"eps": -float("inf")},
        {"clamp_limit": -1.0},
        {"clamp_limit": float("nan")},
        {"clamp_limit": float("inf")},
        {"clamp_limit": -float("inf")},
        {"use_ue8m0": 0},
        {"use_ue8m0": 1},
    ],
)
def test_scalar_arguments_match_vllm(kwargs):
    torch.manual_seed(16)
    x = torch.randn((128, 256), dtype=torch.bfloat16, device=flaggems_vllm.device)
    _assert_reference_equal(OP(x, **kwargs), _installed_vllm_reference()(x, **kwargs))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_preallocated_float_output_matches_vllm(dtype):
    x = torch.randn((128, 256), dtype=torch.bfloat16, device=flaggems_vllm.device)
    output = torch.empty((128, 128), dtype=dtype, device=x.device)
    ref_output = torch.empty_like(output)
    q, s = OP(x, output=output, use_ue8m0=False)
    rq, rs = _installed_vllm_reference()(x, output=ref_output, use_ue8m0=False)
    assert q is output
    torch.testing.assert_close(q, rq, rtol=1e-6, atol=0)
    torch.testing.assert_close(s, rs, rtol=1e-6, atol=0)


def test_disjoint_shared_storage():
    # Both tensors share storage, but their contiguous byte ranges do not overlap.
    storage = torch.empty(128 * 256 + 128 * 128, device=flaggems_vllm.device)
    x = storage[: 128 * 256].view(128, 256)
    x.normal_()
    output = storage[128 * 256 :].view(128, 128)
    actual = OP(x, output=output, use_ue8m0=False)
    reference = _installed_vllm_reference()(
        x, output=torch.empty_like(output), use_ue8m0=False
    )
    assert actual[0] is output
    torch.testing.assert_close(actual[0], reference[0], rtol=1e-6, atol=0)
    torch.testing.assert_close(actual[1], reference[1], rtol=1e-6, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1024, 1280), (1024, 4096)])
@pytest.mark.parametrize(
    "eps",
    [
        2.0**-110,
        2.0**-101,
        2.0**-100,
        2.0**-99,
        1e-10,
        1e20,
        torch.finfo(torch.float32).max,
    ],
)
def test_ue8m0_eps_fast_path_boundary(dtype, shape, eps):
    torch.manual_seed(20261109)
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    kwargs = dict(use_ue8m0=True, eps=eps)
    _assert_reference_equal(OP(x, **kwargs), _installed_vllm_reference()(x, **kwargs))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1024, 1280), (1024, 4096), (8192, 4096)])
def test_ue8m0_infinite_scale_preserves_negative_zero(dtype, shape):
    # Infinite scales must preserve the sign of finite negative activations.
    x = torch.empty(shape, dtype=dtype, device=flaggems_vllm.device)
    x[:, : shape[1] // 2] = -1.0
    x[:, shape[1] // 2 :] = 1.0
    x[0, 0] = torch.inf
    kwargs = dict(use_ue8m0=True)
    actual = OP(x, **kwargs)
    _assert_reference_equal(actual, _installed_vllm_reference()(x, **kwargs))
    assert torch.all(actual[0].view(torch.uint8).cpu()[0, 1:128] == 0x80)


def test_platform_fp8_dtype_selection(monkeypatch):
    """Follow vLLM's platform dtype rather than assuming every device uses FN."""
    import importlib
    import sys
    from types import ModuleType

    core = importlib.import_module(
        "flaggems_vllm.ops.silu_mul_per_token_group_quant_fp8_colmajor"
    )
    platforms = ModuleType("vllm.platforms")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    for dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        platforms.current_platform = SimpleNamespace(fp8_dtype=lambda: dtype)
        assert core._platform_fp8_dtype() == dtype
    platforms.current_platform = SimpleNamespace(fp8_dtype=lambda: torch.float32)
    assert core._platform_fp8_dtype() == torch.float32
