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

"""Accuracy tests against vLLM and an independent CPU reference."""

import ast
import importlib
import importlib.util
from functools import lru_cache
from pathlib import Path

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
    kwargs = dict(use_ue8m0=use_ue8m0, clamp_limit=clamp_limit)
    ref = _installed_vllm_reference()(x, **kwargs)
    actual = OP(x, **kwargs)
    _assert_reference_equal(actual, ref)


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
def test_signed_zero_clamp_matches_vllm(dtype):
    """Scalar signed zero must remain correct across repeated compiled calls."""
    torch.manual_seed(2027)
    x = torch.randn((128, 4096), dtype=dtype, device=flaggems_vllm.device)
    reference = _installed_vllm_reference()
    for clamp_limit in (0.0, -0.0, 0.0):
        kwargs = dict(use_ue8m0=False, clamp_limit=clamp_limit)
        _assert_reference_equal(OP(x, **kwargs), reference(x, **kwargs))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_ue8m0", [False, True])
@pytest.mark.parametrize("clamp_limit", [None, 0.0, 2.0])
def test_independent_cpu_reference(dtype, use_ue8m0, clamp_limit):
    """Check numerical semantics against an independent CPU calculation."""
    torch.manual_seed(2026)
    x = torch.randn((128, 4096), dtype=dtype, device=flaggems_vllm.device)
    kwargs = dict(use_ue8m0=use_ue8m0, clamp_limit=clamp_limit)
    ref_q, ref_s = _cpu_reference(x, **kwargs)
    q, s = OP(x, **kwargs)
    torch.testing.assert_close(s.cpu(), ref_s, rtol=1e-5, atol=1e-8)
    # CPU and accelerator exp may straddle an FP8 rounding boundary. Strict
    # byte comparisons against the native reference remain separate above.
    torch.testing.assert_close(q.cpu().float(), ref_q.float(), rtol=0.125, atol=2**-9)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(128, 4096), (1024, 1792)])
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
@pytest.mark.parametrize("shape", [(128, 256), (1024, 1792), (8192, 4096)])
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
