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

import pytest
import torch

import flaggems_vllm

from . import base, consts, utils

try:
    import torch_npu

    NPU_SWIGLU = getattr(torch_npu, "npu_swiglu", None)
except ImportError:
    torch_npu = None
    NPU_SWIGLU = None

# Note: Importing transformer_engine (especially in some versions like py 3.10) may automatically
# configure the Root Logger (adding handlers). This may cause subsequent `logging.basicConfig`
# calls (used by FlagGems benchmark) to be ignored/no-op, leading to missing result log files.
# See: https://github.com/NVIDIA/TransformerEngine/issues/1065
try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "swiglu")
    TE_AVAILABLE = True
    GEMS_OP = getattr(flaggems_vllm, "swiglu")
except ImportError:
    TE_AVAILABLE = False
    TE_OP = None
    GEMS_OP = None


@pytest.mark.swiglu
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine not installed")
@pytest.mark.skipif(TE_OP is None, reason="'swiglu' not found in TransformerEngine")
@pytest.mark.skipif(GEMS_OP is None, reason="'swiglu' not found in FlagGems")
def test_swiglu():
    bench = base.TexGluForwardBenchmark(
        op_name="swiglu",
        torch_op=TE_OP,
        gems_op=GEMS_OP,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# swiglu splits the last dim in half, so it must be even; shapes come from
# core_shapes.yaml like other ops.
def npu_swiglu_golden(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch_npu.npu_swiglu(input_tensor, dim=-1)


class SwigluBenchmark(base.GenericBenchmark):
    """Filters out shapes with an odd last dim, which swiglu cannot accept."""

    def set_more_shapes(self):
        return [s for s in super().set_more_shapes() if s[-1] % 2 == 0]


@pytest.mark.swiglu
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="swiglu is only specialized on the ascend backend",
)
@pytest.mark.skipif(
    NPU_SWIGLU is None, reason="golden torch_npu.npu_swiglu is unavailable"
)
def test_swiglu_ascend():
    bench = SwigluBenchmark(
        op_name="swiglu",
        input_fn=utils.unary_input_fn,
        torch_op=npu_swiglu_golden,
        gems_op=flaggems_vllm.swiglu,
        dtypes=[torch.float16],
    )
    bench.run()
