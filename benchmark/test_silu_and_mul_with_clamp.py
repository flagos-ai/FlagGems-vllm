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

try:
    importlib.import_module("vllm._custom_ops")
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
except ImportError:
    SiluAndMulWithClamp = None
    set_current_vllm_config = None

from . import base, consts, utils

HAS_VLLM_BASELINE = SiluAndMulWithClamp is not None and hasattr(
    torch.ops._C, "silu_and_mul_with_clamp"
)
SWIGLU_LIMIT = 7.0


def packed_input_fn(shape, dtype, device):
    """Generate packed input with twice the configured gate/up width."""
    if not shape:
        shape = (1,)
    packed_shape = (*shape[:-1], shape[-1] * 2)
    yield utils.generate_tensor_input(packed_shape, dtype, device),


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
def vllm_silu_and_mul_with_clamp():
    if not HAS_VLLM_BASELINE:
        pytest.skip("vLLM SiluAndMulWithClamp baseline is unavailable")
    with _vllm_custom_op_config():
        vllm_op = SiluAndMulWithClamp(swiglu_limit=SWIGLU_LIMIT).to(
            flaggems_vllm.device
        )
        # Bypass OOT dispatch to forward_native so timings measure the fused op.
        vllm_op.op = torch.ops._C.silu_and_mul_with_clamp
        return vllm_op.forward_cuda


@pytest.mark.silu_and_mul_with_clamp
def test_silu_and_mul_with_clamp(vllm_silu_and_mul_with_clamp):

    def gems_op(packed):
        d = packed.shape[-1] // 2
        return flaggems_vllm.silu_and_mul_with_clamp(
            packed[..., :d], packed[..., d:], SWIGLU_LIMIT
        )

    def vllm_op(packed):
        return vllm_silu_and_mul_with_clamp(packed)

    bench = base.GenericBenchmark(
        input_fn=packed_input_fn,
        op_name="silu_and_mul_with_clamp",
        gems_op=gems_op,
        torch_op=vllm_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.silu_and_mul_with_clamp_out
def test_silu_and_mul_with_clamp_out(vllm_silu_and_mul_with_clamp):

    def gems_op(packed):
        d = packed.shape[-1] // 2
        x = packed[..., :d]
        y = packed[..., d:]
        out = torch.empty_like(x)
        return flaggems_vllm.silu_and_mul_with_clamp_out(x, y, out, SWIGLU_LIMIT)

    def vllm_op(packed):
        return vllm_silu_and_mul_with_clamp(packed)

    bench = base.GenericBenchmark(
        input_fn=packed_input_fn,
        op_name="silu_and_mul_with_clamp_out",
        gems_op=gems_op,
        torch_op=vllm_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
