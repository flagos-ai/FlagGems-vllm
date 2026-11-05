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

from contextlib import nullcontext

import pytest
import torch

import flaggems_vllm

from .accuracy_utils import gems_assert_close, gems_assert_equal
from .conftest import QUICK_MODE

API_NAME = "fused_indexer_weight_scale"
pytestmark = pytest.mark.skipif(
    API_NAME not in vars(flaggems_vllm), reason="backend has no indexer weight scaling"
)
SHAPES = (
    [(0, 32), (1, 32), (7, 13), (1036, 32)]
    if QUICK_MODE
    else [
        (0, 32),
        (7, 0),
        (1, 1),
        (1, 32),
        (7, 13),
        (33, 64),
        (1003, 16),
        (64, 129),
        (16384, 32),
        (4096, 64),
        (1036, 32),
    ]
)


@pytest.fixture(scope="module")
def native_weight_scale():
    # Match each backend's native compile policy outside timing.
    native_context = (
        torch.compiler.set_stance("force_eager")
        if flaggems_vllm.vendor_name == "ascend"
        else nullcontext()
    )
    with native_context:
        try:
            from vllm.models.glm5next.common.attention import (
                _fused_indexer_weight_scale,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The native indexer baseline requires vLLM 0.33.0+empty; "
                "vLLM 0.24+empty has no such entry. Install it in the vendor "
                "test environment following the PR's validation instructions."
            ) from exc

        yield _fused_indexer_weight_scale


def make_inputs(tokens, heads, device, is_strided=False, dtype=torch.float32):
    width = heads * 2 if is_strided else heads
    weights = torch.randn((tokens, width), dtype=dtype, device=device)
    q_scale = torch.exp2(
        torch.randint(-8, 1, (tokens, width, 1), device=device).float()
    )
    if is_strided:
        weights = weights[:, ::2]
        q_scale = q_scale[:, ::2]
    return weights, q_scale


@pytest.mark.fused_indexer_weight_scale
@pytest.mark.parametrize("tokens,heads", SHAPES)
@pytest.mark.parametrize("is_strided", [False, True])
@pytest.mark.parametrize("scale", [0.0, 128**-0.5 * 32**-0.5, -0.125])
def test_fused_indexer_weight_scale(
    tokens, heads, is_strided, scale, native_weight_scale
):
    torch.manual_seed(0)
    weights, q_scale = make_inputs(tokens, heads, flaggems_vllm.device, is_strided)
    expected = native_weight_scale(weights, q_scale, scale)
    output = flaggems_vllm.fused_indexer_weight_scale(weights, q_scale, scale)
    gems_assert_close(output, expected, torch.float32, atol=1e-6)


@pytest.mark.fused_indexer_weight_scale
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,heads", [(7, 13), (1036, 32)])
@pytest.mark.parametrize("is_strided", [False, True])
def test_fused_indexer_weight_scale_promotion(
    dtype, tokens, heads, is_strided, native_weight_scale
):
    torch.manual_seed(0)
    weights, q_scale = make_inputs(
        tokens, heads, flaggems_vllm.device, is_strided, dtype
    )
    expected = native_weight_scale(weights, q_scale, 0.015625)
    output = flaggems_vllm.fused_indexer_weight_scale(weights, q_scale, 0.015625)
    gems_assert_close(output, expected, torch.float32, atol=1e-6)


@pytest.mark.fused_indexer_weight_scale
@pytest.mark.parametrize("contiguous_input", ["weights", "q_scale"])
def test_fused_indexer_weight_scale_independent_strides(
    contiguous_input, native_weight_scale
):
    torch.manual_seed(0)
    weights, q_scale = make_inputs(7, 13, flaggems_vllm.device, True)
    if contiguous_input == "weights":
        weights = weights.contiguous()
    else:
        q_scale = q_scale.contiguous()
    expected = native_weight_scale(weights, q_scale, 0.037)
    output = flaggems_vllm.fused_indexer_weight_scale(weights, q_scale, 0.037)
    gems_assert_close(output, expected, torch.float32, atol=1e-6)


@pytest.mark.fused_indexer_weight_scale
@pytest.mark.parametrize("scale", [0.0, 0.125, float("inf"), float("nan")])
def test_fused_indexer_weight_scale_nonfinite(native_weight_scale, scale):
    torch.manual_seed(0)
    weights, q_scale = make_inputs(2, 32, flaggems_vllm.device)
    weights[0, :3] = torch.tensor([1.0, 0.0, -1.0], device=weights.device)
    q_scale[0, :3, 0] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], device=weights.device
    )
    expected = native_weight_scale(weights, q_scale, scale)
    output = flaggems_vllm.fused_indexer_weight_scale(weights, q_scale, scale)
    gems_assert_close(output, expected, torch.float32, atol=1e-6, equal_nan=True)


@pytest.mark.fused_indexer_weight_scale
@pytest.mark.parametrize("offset", [1, 8])
def test_fused_indexer_weight_scale_storage_offset(native_weight_scale, offset):
    torch.manual_seed(0)
    total = 35 * 32
    weights_base = torch.randn(total + offset, device=flaggems_vllm.device)
    query_base = torch.randn(total + offset, device=flaggems_vllm.device)
    weights_saved = weights_base.clone()
    query_saved = query_base.clone()
    weights = weights_base[offset:].view(35, 32)
    query = query_base[offset:].view(35, 32, 1)
    expected = native_weight_scale(weights, query, 0.015625)
    output = flaggems_vllm.fused_indexer_weight_scale(weights, query, 0.015625)
    gems_assert_close(output, expected, torch.float32, atol=1e-6)
    gems_assert_equal(weights_base, weights_saved)
    gems_assert_equal(query_base, query_saved)
