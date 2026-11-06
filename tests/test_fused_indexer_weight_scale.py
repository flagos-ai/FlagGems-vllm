# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
import torch

import flaggems_vllm

from .accuracy_utils import gems_assert_close, gems_assert_equal
from .conftest import QUICK_MODE

API_NAME = "fused_indexer_weight_scale"
pytestmark = pytest.mark.skipif(
    API_NAME not in vars(flaggems_vllm),
    reason="backend has no indexer weight scaling",
)


@pytest.fixture(scope="module")
def native_weight_scale():
    from vllm.models.glm5next.common.attention import _fused_indexer_weight_scale

    # Benchmarks use the decorated function with its native compile policy.
    return _fused_indexer_weight_scale


def make_inputs(tokens, heads, device, dtype=torch.float32, layout="contiguous"):
    weight_width = heads * 2 if layout in ("weights", "both") else heads
    query_width = heads * 2 if layout in ("query", "both") else heads
    weights = torch.randn((tokens, weight_width), dtype=dtype, device=device)
    q_scale = torch.exp2(
        torch.randint(-8, 1, (tokens, query_width, 1), device=device).float()
    )
    if layout in ("weights", "both"):
        weights = weights[:, ::2]
    if layout in ("query", "both"):
        q_scale = q_scale[:, ::2]
    return weights, q_scale


@pytest.mark.fused_indexer_weight_scale
@pytest.mark.parametrize(
    "tokens,heads",
    (
        [(1, 32), (7, 13)]
        if QUICK_MODE
        else [(0, 32), (7, 0), (1, 1), (7, 13), (65, 32), (16384, 32)]
    ),
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "layout",
    (
        ["contiguous", "both"]
        if QUICK_MODE
        else ["contiguous", "weights", "query", "both"]
    ),
)
def test_fused_indexer_weight_scale(tokens, heads, dtype, layout, native_weight_scale):
    torch.manual_seed(0)
    weights, query = make_inputs(tokens, heads, flaggems_vllm.device, dtype, layout)
    scale = 128**-0.5 * 32**-0.5
    # Correctness covers eager promotion/rounding; performance retains compile.
    expected = inspect.unwrap(native_weight_scale)(weights, query, scale)
    output = flaggems_vllm.fused_indexer_weight_scale(weights, query, scale)
    assert output.shape == weights.shape
    assert output.dtype == torch.float32
    assert output.is_contiguous()
    gems_assert_close(output, expected, torch.float32, atol=1e-6)


@pytest.mark.fused_indexer_weight_scale
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.037, float("inf"), float("nan")])
def test_fused_indexer_weight_scale_special_values(scale, native_weight_scale):
    torch.manual_seed(0)
    weights, query = make_inputs(3, 5, flaggems_vllm.device)
    weights[0, :4] = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf")], device=weights.device
    )
    query[1, :3, 0] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], device=weights.device
    )
    expected = inspect.unwrap(native_weight_scale)(weights, query, scale)
    output = flaggems_vllm.fused_indexer_weight_scale(weights, query, scale)
    gems_assert_close(output, expected, torch.float32, atol=1e-6, equal_nan=True)


@pytest.mark.fused_indexer_weight_scale
@pytest.mark.parametrize("offset", [1, 8])
def test_fused_indexer_weight_scale_storage_offset(offset, native_weight_scale):
    torch.manual_seed(0)
    total = 35 * 32
    weights_base = torch.randn(total + offset, device=flaggems_vllm.device)
    query_base = torch.randn(total + offset, device=flaggems_vllm.device)
    weights_saved, query_saved = weights_base.clone(), query_base.clone()
    weights = weights_base[offset:].view(35, 32)
    query = query_base[offset:].view(35, 32, 1)
    expected = inspect.unwrap(native_weight_scale)(weights, query, 0.015625)
    output = flaggems_vllm.fused_indexer_weight_scale(weights, query, 0.015625)
    gems_assert_close(output, expected, torch.float32, atol=1e-6)
    gems_assert_equal(weights_base, weights_saved)
    gems_assert_equal(query_base, query_saved)
