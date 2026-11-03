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

from .conftest import QUICK_MODE

pytestmark = [
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "metax", reason="MetaX backend implementation"
    ),
]


def native_weight_scale(weights, q_scale, scale):
    # vLLM v0.33.0, commit 4ba787ec9959e566d3b587a796dff07673324da5,
    # vllm/models/glm5next/common/attention.py; undecorated reference.
    return (weights.unsqueeze(-1) * q_scale * scale).squeeze(-1)


def make_inputs(tokens, heads, dtype, layout):
    width = heads * 2 + 1
    weights = torch.randn(tokens + 1, width, dtype=dtype, device="cuda")
    query = torch.randn(tokens + 1, width, 1, dtype=torch.float32, device="cuda")
    if layout == "dense":
        return (
            weights[:tokens, :heads].contiguous(),
            query[:tokens, :heads].contiguous(),
        )
    if layout == "offset":
        weights = torch.randn(tokens * heads + 1, dtype=dtype, device="cuda")
        query = torch.randn(tokens * heads + 1, device="cuda")
        return weights[1:].view(tokens, heads), query[1:].view(tokens, heads, 1)
    if layout == "broadcast":
        return weights[:1, :heads].expand(tokens, heads), query[:tokens, :1].expand(
            tokens, heads, 1
        )
    return weights[:tokens, : heads * 2 : 2], query[:tokens, : heads * 2 : 2]


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
        (64, 129),
        (1036, 32),
        (16384, 32),
    ]
)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["dense", "strided", "offset", "broadcast"])
@pytest.mark.parametrize("scale", [0.0, 128**-0.5 * 32**-0.5, -0.037])
@pytest.mark.fused_indexer_weight_scale
def test_fused_indexer_weight_scale(shape, dtype, layout, scale):
    torch.manual_seed(0)
    weights, query = make_inputs(*shape, dtype, layout)
    saved_weights, saved_query = weights.clone(), query.clone()
    output = flaggems_vllm.fused_indexer_weight_scale(weights, query, scale)
    expected = native_weight_scale(weights, query, scale)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-7)
    assert output.dtype == torch.float32
    assert output.is_contiguous()
    torch.testing.assert_close(weights, saved_weights, rtol=0, atol=0)
    torch.testing.assert_close(query, saved_query, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0.0, -0.125, float("inf"), float("nan")])
@pytest.mark.fused_indexer_weight_scale
def test_fused_indexer_weight_scale_nonfinite(scale):
    weights = torch.tensor(
        [[0.0, -0.0, 1.0, -1.0, float("inf"), float("nan")]], device="cuda"
    )
    query = torch.tensor(
        [[[1.0], [1.0], [float("inf")], [float("nan")], [0.0], [1.0]]], device="cuda"
    )
    output = flaggems_vllm.fused_indexer_weight_scale(weights, query, scale)
    expected = native_weight_scale(weights, query, scale)
    torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        torch.signbit(output[finite]), torch.signbit(expected[finite])
    )


@pytest.mark.fused_indexer_weight_scale
def test_fused_indexer_weight_scale_independent_strides():
    weights, query = make_inputs(35, 32, torch.float32, "strided")
    for dense_weights in (False, True):
        w, q = (
            (weights.contiguous(), query)
            if dense_weights
            else (weights, query.contiguous())
        )
        output = flaggems_vllm.fused_indexer_weight_scale(w, q, 0.037)
        torch.testing.assert_close(
            output, native_weight_scale(w, q, 0.037), rtol=1e-6, atol=1e-7
        )


@pytest.mark.fused_indexer_weight_scale
def test_fused_indexer_weight_scale_invalid_inputs():
    weights = torch.empty(4, 32, device="cuda")
    query = torch.empty(4, 32, 1, device="cuda")
    with pytest.raises(ValueError, match="weights must have shape"):
        flaggems_vllm.fused_indexer_weight_scale(weights[0], query, 1.0)
    with pytest.raises(ValueError, match="q_scale must have shape"):
        flaggems_vllm.fused_indexer_weight_scale(weights, query.squeeze(-1), 1.0)
    with pytest.raises(TypeError, match="weights must be"):
        flaggems_vllm.fused_indexer_weight_scale(weights.int(), query, 1.0)
    with pytest.raises(TypeError, match="q_scale must be FP32"):
        flaggems_vllm.fused_indexer_weight_scale(weights, query.half(), 1.0)
    with pytest.raises(NotImplementedError, match="requires a GPU"):
        flaggems_vllm.fused_indexer_weight_scale(weights.cpu(), query.cpu(), 1.0)


@pytest.mark.fused_indexer_weight_scale
def test_fused_indexer_weight_scale_library_export():
    assert callable(flaggems_vllm.fused_indexer_weight_scale)
    assert "fused_indexer_weight_scale" not in flaggems_vllm.FULL_CONFIG_BY_FUNC


@pytest.mark.parametrize("offset", [1, 4])
@pytest.mark.fused_indexer_weight_scale
def test_fused_indexer_weight_scale_vector_offset(offset):
    torch.manual_seed(0)
    total = 1036 * 32
    weights = torch.randn(total + offset, device="cuda")[offset:].view(1036, 32)
    query = torch.randn(total + offset, device="cuda")[offset:].view(1036, 32, 1)
    output = flaggems_vllm.fused_indexer_weight_scale(weights, query, 0.037)
    torch.testing.assert_close(
        output, native_weight_scale(weights, query, 0.037), rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("scale", [0.0, 0.125, -0.125, float("inf"), float("nan")])
@pytest.mark.fused_indexer_weight_scale
def test_fused_indexer_weight_scale_vector_nonfinite(scale):
    weights = torch.tensor(
        [[0.0, -0.0, 1.0, -1.0, float("inf"), float("nan")]], device="cuda"
    ).repeat(2048, 1)
    query = torch.tensor(
        [[[1.0], [1.0], [float("inf")], [float("nan")], [0.0], [1.0]]], device="cuda"
    ).repeat(2048, 1, 1)
    output = flaggems_vllm.fused_indexer_weight_scale(weights, query, scale)
    # MACA eager loses zero signs for vectorized muls; use an independent CPU oracle.
    expected = native_weight_scale(weights.cpu(), query.cpu(), scale).to(weights.device)
    torch.testing.assert_close(
        output,
        native_weight_scale(weights, query, scale),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        torch.signbit(output[finite]), torch.signbit(expected[finite])
    )


@pytest.mark.fused_indexer_weight_scale
def test_fused_indexer_weight_scale_without_gluon(monkeypatch):
    import importlib

    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.fused_indexer_weight_scale"
    )
    monkeypatch.setattr(module, "HAS_GLUON", False)
    weights, query = make_inputs(1036, 32, torch.float32, "dense")
    output = flaggems_vllm.fused_indexer_weight_scale(weights, query, 0.037)
    torch.testing.assert_close(
        output, native_weight_scale(weights, query, 0.037), rtol=1e-6, atol=1e-7
    )
