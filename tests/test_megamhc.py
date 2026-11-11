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

"""Tests for the public Hopper MegaMHC entry points."""

from __future__ import annotations

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

pytestmark = pytest.mark.megamhc


def _is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.megamhc
@pytest.mark.skipif(
    not (_is_hopper() and has_triton_tle(3, 6, 0)),
    reason="requires Hopper with FlagTree TLE",
)
def test_megamhc_is_exported_on_hopper():
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops import (
        megamhc as hopper_megamhc,
    )

    assert flaggems_vllm.create_mega_mhc_context is (
        hopper_megamhc.create_mega_mhc_context
    )
    assert flaggems_vllm.mega_mhc is hopper_megamhc.mega_mhc


def _make_inputs(tokens: int):
    generator = torch.Generator(device="cuda").manual_seed(20261111 + tokens)

    def randn(shape, dtype):
        return torch.randn(shape, dtype=dtype, device="cuda", generator=generator)

    comb = torch.rand((tokens, 4, 4), device="cuda", generator=generator)
    for _ in range(3):
        comb /= comb.sum(-1, keepdim=True)
        comb /= comb.sum(-2, keepdim=True)
    return {
        "x": randn((tokens, 4096), torch.bfloat16),
        "residual": randn((tokens, 4, 4096), torch.bfloat16),
        "post": randn((tokens, 4, 1), torch.float32).sigmoid(),
        "comb": comb,
        "fn": randn((24, 16384), torch.float32).mul_(1e-2),
        "scale": randn((3,), torch.float32).mul_(0.1),
        "base": randn((24,), torch.float32).mul_(0.1),
        "norm": randn((4096,), torch.float32).mul_(0.1).add_(1).bfloat16(),
    }


def _reference(inputs):
    x = inputs["x"]
    residual = inputs["residual"]
    post = inputs["post"]
    new_residual = x.float().unsqueeze(1) * post
    new_residual += torch.einsum("tij,tih->tjh", inputs["comb"], residual.float())
    new_residual = new_residual.bfloat16()
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        mixes = new_residual.flatten(1).float() @ inputs["fn"].mT
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
    mixes *= torch.rsqrt(
        new_residual.float().square().sum((1, 2)) / (4 * 4096) + 1e-6
    ).unsqueeze(1)
    scales = torch.cat(
        (
            inputs["scale"][0].expand(4),
            inputs["scale"][1].expand(4),
            inputs["scale"][2].expand(16),
        )
    )
    mixes = mixes * scales + inputs["base"]
    pre = mixes[:, :4].sigmoid().unsqueeze(2) + 1e-6
    new_post = (mixes[:, 4:8].sigmoid() * 2.0).unsqueeze(2)
    new_comb = mixes[:, 8:].view(-1, 4, 4).softmax(dim=-1) + 1e-6
    new_comb /= new_comb.sum(dim=-2, keepdim=True) + 1e-6
    for _ in range(19):
        new_comb /= new_comb.sum(dim=-1, keepdim=True) + 1e-6
        new_comb /= new_comb.sum(dim=-2, keepdim=True) + 1e-6
    norm_input = (new_residual.float() * pre).sum(dim=1).bfloat16().float()
    y = (
        norm_input
        * torch.rsqrt(norm_input.square().mean(1) + 1e-6).unsqueeze(1)
        * inputs["norm"].float()
    ).bfloat16()
    return new_residual, new_post, new_comb, y


def _relative_diff(actual, expected):
    actual = actual.double()
    expected = expected.double()
    denominator = (actual.square() + expected.square()).sum()
    if float(denominator) == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (actual * expected).sum() / denominator)


@pytest.mark.megamhc
@pytest.mark.skipif(
    not (_is_hopper() and has_triton_tle(3, 6, 0)),
    reason="requires Hopper with FlagTree TLE",
)
def test_megamhc_mixed_token_contexts_are_accurate():
    cases = []
    for tokens in (1, 64, 512):
        inputs = _make_inputs(tokens)
        context = flaggems_vllm.create_mega_mhc_context(
            inputs["x"],
            inputs["residual"],
            inputs["post"],
            inputs["comb"],
            inputs["fn"],
            inputs["scale"],
            inputs["base"],
            inputs["norm"],
        )
        assert context.supported_tokens == frozenset({tokens})
        cases.append((inputs, context, _reference(inputs)))

    for inputs, context, expected in (*reversed(cases), *cases):
        actual = flaggems_vllm.mega_mhc(
            inputs["x"],
            inputs["residual"],
            inputs["post"],
            inputs["comb"],
            inputs["fn"],
            inputs["scale"],
            inputs["base"],
            1e-6,
            1e-6,
            1e-6,
            2.0,
            20,
            1,
            1,
            inputs["norm"],
            1e-6,
            context=context,
        )
        torch.cuda.synchronize()
        for result, reference in zip(actual, expected, strict=True):
            assert result.shape == reference.shape
            assert result.dtype == reference.dtype
            assert torch.isfinite(result).all()
            assert _relative_diff(result, reference) < 5e-5
