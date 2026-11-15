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
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils

try:
    importlib.import_module("vllm._custom_ops")
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
except ImportError:
    SiluAndMulWithClamp = None
    set_current_vllm_config = None


HAS_VLLM_BASELINE = SiluAndMulWithClamp is not None and hasattr(
    getattr(torch.ops, "_C", None), "silu_and_mul_with_clamp"
)

SILU_AND_MUL_WITH_CLAMP_LIMITS = [3.0, 7.0]
# Valid packed inputs: cover tiny/linear, decode/prefill, irregular and ND cases.
SILU_AND_MUL_WITH_CLAMP_SHAPES = (
    [(2,), (16, 1024), (2, 3, 100)]
    if utils.QUICK_MODE
    else [
        (2,),
        (1024,),
        (1, 1024),
        (16, 1024),
        (496, 1024),
        (513, 514),
        (4096, 2048),
        (2, 3, 100),
        (2, 16, 1024),
        (2, 3, 4, 128),
        (2, 3, 4, 5, 64),
    ]
)


@contextmanager
def _vllm_custom_op_config():
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            custom_ops=["all"],
            enabled_custom_ops=set(),
            disabled_custom_ops=set(),
        )
    )
    with set_current_vllm_config(config):
        yield


@pytest.fixture(scope="module")
def vllm_silu_and_mul_with_clamp_cls():
    if not HAS_VLLM_BASELINE:
        pytest.skip("vLLM SiluAndMulWithClamp baseline is unavailable")
    return SiluAndMulWithClamp


@pytest.mark.silu_and_mul_with_clamp
@pytest.mark.parametrize("shape", SILU_AND_MUL_WITH_CLAMP_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("limit", SILU_AND_MUL_WITH_CLAMP_LIMITS)
def test_silu_and_mul_with_clamp(shape, dtype, limit, vllm_silu_and_mul_with_clamp_cls):
    packed = torch.randn(
        shape, dtype=dtype, device=flaggems_vllm.device, requires_grad=True
    )
    d = packed.shape[-1] // 2
    inp1 = packed[..., :d]
    inp2 = packed[..., d:]
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    with _vllm_custom_op_config():
        vllm_op = vllm_silu_and_mul_with_clamp_cls(swiglu_limit=limit).to(
            flaggems_vllm.device
        )
        # OOT platforms may dispatch __call__ to forward_native. Explicitly
        # select the fused baseline, including platforms using PrivateUse1.
        vllm_op.op = torch.ops._C.silu_and_mul_with_clamp
        vllm_out = utils.to_reference(vllm_op.forward_cuda(packed.detach()))
    # Use vLLM's differentiable reference for backward, not its fused kernel.
    ref_out = vllm_op.forward_native(torch.cat((ref_inp1, ref_inp2), dim=-1))
    res_out = flaggems_vllm.silu_and_mul_with_clamp(inp1, inp2, limit)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)

    ref_inp1_grad, ref_inp2_grad = torch.autograd.grad(
        ref_out, (ref_inp1, ref_inp2), ref_grad
    )

    res_inp1_grad, res_inp2_grad = torch.autograd.grad(res_out, (inp1, inp2), out_grad)

    utils.gems_assert_close(res_out, vllm_out, dtype)
    utils.gems_assert_close(res_inp1_grad, ref_inp1_grad, dtype)
    utils.gems_assert_close(res_inp2_grad, ref_inp2_grad, dtype)


@pytest.mark.silu_and_mul_with_clamp_out
@pytest.mark.parametrize("shape", SILU_AND_MUL_WITH_CLAMP_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("limit", SILU_AND_MUL_WITH_CLAMP_LIMITS)
def test_silu_and_mul_with_clamp_out(
    shape, dtype, limit, vllm_silu_and_mul_with_clamp_cls
):
    packed = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    d = packed.shape[-1] // 2
    inp1 = packed[..., :d]
    inp2 = packed[..., d:]

    with _vllm_custom_op_config():
        vllm_op = vllm_silu_and_mul_with_clamp_cls(swiglu_limit=limit).to(
            flaggems_vllm.device
        )
        vllm_op.op = torch.ops._C.silu_and_mul_with_clamp
        vllm_out = utils.to_reference(vllm_op.forward_cuda(packed))

    out = torch.empty_like(inp1)
    ret = flaggems_vllm.silu_and_mul_with_clamp_out(inp1, inp2, out, limit)

    assert ret is out
    utils.gems_assert_close(out, vllm_out, dtype)
