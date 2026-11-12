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

import importlib

import pytest
import torch
import torch.nn.functional as F
import triton.language.math as tl_math

import flaggems_vllm

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


@pytest.mark.geglu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is required")
def test_geglu(shape, dtype):
    if len(shape) == 0:
        # GEGLU does not support 0-dim scalar tensors.
        return

    if shape[-1] % 2 != 0:
        shape = list(shape)
        shape[-1] += 1
        shape = tuple(shape)

    input_tensor = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)

    ref_out = tex.geglu(input_tensor, None)
    ref_out = utils.to_reference(ref_out)

    with flaggems_vllm.use_gems():
        res_out = flaggems_vllm.geglu(input_tensor)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.geglu
def test_geglu_import_with_triton_math(monkeypatch):
    # Use real core-math functions and real Triton fallbacks, never fake exports.
    import flaggems_vllm.utils as gems_utils
    from flaggems_vllm.utils.triton_lang_helper import _complete_math_shim

    modules = [
        importlib.import_module(f"flaggems_vllm.ops.{name}")
        for name in ("geglu", "gelu_and_mul", "swiglu", "tanh")
    ]
    shim = _complete_math_shim(tl_math)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(gems_utils, "tl_extra_shim", shim)
            for module in modules:
                importlib.reload(module)
            x = torch.linspace(-6, 6, 130, device=flaggems_vllm.device).view(1, 130)
            a, b = x.cpu().double().chunk(2, dim=-1)
            result = modules[0].geglu(x)
            reference = F.gelu(a, approximate="tanh") * b
            utils.gems_assert_close(result.cpu(), reference, torch.float32)
            grad = torch.ones_like(result)
            ref_x = x.cpu().double().requires_grad_(True)
            a, b = ref_x.chunk(2, dim=-1)
            (reference_grad,) = torch.autograd.grad(
                F.gelu(a, approximate="tanh") * b, ref_x, grad.cpu().double()
            )
            utils.gems_assert_close(
                modules[0].dgeglu(grad, x).cpu(), reference_grad, torch.float32
            )
    finally:
        for module in modules:
            importlib.reload(module)


@pytest.mark.geglu
@pytest.mark.parametrize("shape", [(1, 2), (7, 130), (64, 512), (1024, 4096)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_geglu_torch_reference(shape, dtype):
    torch.manual_seed(0)
    input_tensor = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    original = input_tensor.cpu().clone()
    a, b = original.double().chunk(2, dim=-1)
    reference = F.gelu(a, approximate="tanh") * b

    result = flaggems_vllm.geglu(input_tensor)

    assert result.shape == reference.shape
    assert result.device == input_tensor.device
    utils.gems_assert_close(result.cpu(), reference, dtype)
    torch.testing.assert_close(input_tensor.cpu(), original, rtol=0, atol=0)
