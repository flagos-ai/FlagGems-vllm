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

"""BF16 public-dispatch regressions for head counts, tails and special values."""

import importlib

import pytest
import torch
import triton

import flaggems_vllm
from tests import test_fused_inv_rope_fp8_quant as native

mod = importlib.import_module("flaggems_vllm.ops.fused_inv_rope_fp8_quant")


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(
    not native.HAS_FP8_QUANT or not mod._HAS_LOAD_REORDER,
    reason="requires FP8 and compiler load reorder support",
)
@pytest.mark.skipif(
    flaggems_vllm.fused_inv_rope_fp8_quant is not mod.fused_inv_rope_fp8_quant,
    reason="backend uses its own fused inverse RoPE implementation",
)
@pytest.mark.parametrize(
    "tokens,heads,groups,expect_window",
    [(256, 32, 4, True), (256, 64, 8, True), (260, 128, 8, True), (257, 128, 8, False)],
)
@pytest.mark.parametrize("packed", [False, True])
def test_public_shapes(monkeypatch, tokens, heads, groups, expect_window, packed):
    observed = []
    original = triton.runtime.jit.JITFunction.run

    def capture(self, *args, **kwargs):
        if self is mod._fused_inv_rope_fp8_quant_window:
            observed.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(triton.runtime.jit.JITFunction, "run", capture)
    monkeypatch.setattr(mod, "_HAS_LOAD_REORDER", False)
    before = native._run_case(tokens, heads, groups, packed, seed=42)
    assert not observed
    monkeypatch.setattr(mod, "_HAS_LOAD_REORDER", True)
    after = native._run_case(tokens, heads, groups, packed, seed=42)
    assert bool(observed) == expect_window
    assert (
        after["out"].shape == before["out"].shape
        and after["out"].stride() == before["out"].stride()
    )
    assert (
        after["scale"].shape == before["scale"].shape
        and after["scale"].stride() == before["scale"].stride()
    )
    native._assert_dequant_close(
        after["out"], after["scale"], before["out"], before["scale"], heads // groups
    )
    native._assert_dequant_close(
        after["out"],
        after["scale"],
        after["ref_out"],
        after["ref_scale"],
        heads // groups,
    )


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(
    not native.HAS_FP8_QUANT or not mod._HAS_LOAD_REORDER,
    reason="requires FP8 and compiler load reorder support",
)
@pytest.mark.skipif(
    flaggems_vllm.fused_inv_rope_fp8_quant is not mod.fused_inv_rope_fp8_quant,
    reason="backend uses its own fused inverse RoPE implementation",
)
@pytest.mark.parametrize("packed", [False, True])
def test_special_values_against_upstream(monkeypatch, packed):
    torch.manual_seed(123)
    o = torch.randn((256, 128, 512), dtype=torch.bfloat16, device=flaggems_vllm.device)
    o[0, 0, 0] = float("nan")
    o[1, 0, 0] = float("inf")
    o[2, 0, 0] = float("-inf")
    pos = torch.arange(256, dtype=torch.int64, device=flaggems_vllm.device)
    cache = native._make_cos_sin_cache(4096, 64, torch.device(flaggems_vllm.device))
    monkeypatch.setattr(mod, "_HAS_LOAD_REORDER", False)
    baseline = flaggems_vllm.fused_inv_rope_fp8_quant(
        o, pos, cache, 8, 16, tma_aligned_scales=packed
    )
    monkeypatch.setattr(mod, "_HAS_LOAD_REORDER", True)
    actual = flaggems_vllm.fused_inv_rope_fp8_quant(
        o, pos, cache, 8, 16, tma_aligned_scales=packed
    )
    flaggems_vllm.runtime.torch_device_fn.synchronize()
    assert torch.equal(actual[0].view(torch.uint8), baseline[0].view(torch.uint8))
    torch.testing.assert_close(actual[1], baseline[1], rtol=0, atol=0, equal_nan=True)
