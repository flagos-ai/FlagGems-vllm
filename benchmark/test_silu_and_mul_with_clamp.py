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
from pathlib import Path
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

# Each shape is the full packed input (..., 2D), not the gate/up shape.
SILU_AND_MUL_WITH_CLAMP_CORE_SHAPES = [
    # Representative model shapes: decode, prefill and the most frequent large inputs.
    (1, 512),
    (1, 1024),
    (16, 512),
    (16, 1024),
    (496, 512),
    (512, 1024),
    (16384, 512),
    (16384, 1024),
    # Existing general coverage.
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]


class SiluAndMulWithClampBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = SILU_AND_MUL_WITH_CLAMP_CORE_SHAPES
    DEFAULT_SHAPE_DESC = "packed (..., 2D)"
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def set_shapes(self, shape_file_path=None):
        if (
            shape_file_path
            and Path(shape_file_path).resolve()
            != Path(self.DEFAULT_SHAPE_FILES).resolve()
        ):
            return super().set_shapes(shape_file_path)
        self.shapes = list(SILU_AND_MUL_WITH_CLAMP_CORE_SHAPES)
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_more_shapes(self):
        # Custom shape files should not inherit unrelated generic shapes.
        return []


def packed_input_fn(shape, dtype, device):
    """Treat the configured shape as the full packed input shape (..., 2 * d)."""
    if not shape or shape[-1] <= 0 or shape[-1] % 2:
        return
    yield utils.generate_tensor_input(shape, dtype, device),


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

    bench = SiluAndMulWithClampBenchmark(
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

    bench = SiluAndMulWithClampBenchmark(
        input_fn=packed_input_fn,
        op_name="silu_and_mul_with_clamp_out",
        gems_op=gems_op,
        torch_op=vllm_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
