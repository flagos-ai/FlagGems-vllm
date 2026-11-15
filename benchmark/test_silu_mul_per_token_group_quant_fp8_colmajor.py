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

"""Fused FP8 latency (framework milliseconds); speedup = vLLM / Gems.

The baseline is the unchanged installed-vLLM AST with only its platform dtype
selected by vLLM. No FP32-output substitute or alternate quantizer is timed.
Use ``--mode cudagraph`` to measure captured device work through graph replay.
"""

import ast
import importlib.util
from functools import partial
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
import yaml

import flaggems_vllm

from . import base

if flaggems_vllm.vendor_name == "mthreads":
    import torchada  # noqa: F401


def _installed_vllm_reference():
    """Load unchanged installed function ASTs, isolating vLLM plugin setup.

    Use the vLLM platform FP8 dtype, with scale mode passed explicitly.
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

    namespace = {
        "torch": torch,
        "triton": triton,
        "tl": tl,
        "current_platform": current_platform,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["silu_mul_per_token_group_quant_fp8_colmajor"]


# Small, common, two narrow-group regressions, and bandwidth-heavy workloads.
ACTIVE_SHAPES = [
    (128, 256),
    (128, 4096),
    (512, 4096),
    (1024, 1792),
    (8192, 4096),
]


class SiluMulQuantFp8Benchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = ACTIVE_SHAPES
    DEFAULT_SHAPE_DESC = "M, packed_gate_up_N"

    def set_shapes(self, shape_file_path=None):
        if shape_file_path is None or shape_file_path == self.DEFAULT_SHAPE_FILES:
            shape_file_path = str(Path(__file__).with_name(self.DEFAULT_SHAPE_FILES))
        with open(shape_file_path) as stream:
            config = yaml.safe_load(stream) or {}
        # Generic class entries are not valid workloads for this fused op.
        # Custom files without an operator-specific entry retain the frozen set.
        if self.op_name not in config:
            self.shapes = list(ACTIVE_SHAPES)
            self.shape_desc = self.DEFAULT_SHAPE_DESC
            return
        super().set_shapes(shape_file_path)

    def set_more_shapes(self):
        return []


def _inputs(shape, dtype, device, use_ue8m0=False):
    x = torch.randn(shape, dtype=dtype, device=device)
    before = x.clone()
    native = _installed_vllm_reference()
    expected = native(x, use_ue8m0=use_ue8m0, clamp_limit=None)
    actual = flaggems_vllm.silu_mul_per_token_group_quant_fp8_colmajor(
        x, use_ue8m0=use_ue8m0, clamp_limit=None
    )
    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert actual[0].data_ptr() != x.data_ptr()
    del before, actual, expected
    yield (x, {"use_ue8m0": use_ue8m0, "clamp_limit": None})


@pytest.mark.silu_mul_per_token_group_quant_fp8_colmajor(
    recommended_shapes=ACTIVE_SHAPES,
    shape_desc="M, packed_gate_up_N",
)
@pytest.mark.skipif(
    flaggems_vllm.vendor_name not in ("nvidia", "metax", "thead", "mthreads", "hygon"),
    reason="this backend has no validated fused FP8 implementation",
)
@pytest.mark.parametrize("use_ue8m0", [False, True])
def test_silu_mul_per_token_group_quant_fp8_colmajor(use_ue8m0):
    bench = SiluMulQuantFp8Benchmark(
        op_name="silu_mul_per_token_group_quant_fp8_colmajor",
        input_fn=partial(_inputs, use_ue8m0=use_ue8m0),
        torch_op=_installed_vllm_reference(),
        gems_op=flaggems_vllm.silu_mul_per_token_group_quant_fp8_colmajor,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
