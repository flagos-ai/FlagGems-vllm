# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import os

import pytest
import torch

import flaggems_vllm as gems
from tests.gdn_reference import CASES, assert_result, build, reference


@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("entry", ["generic", "top"])
def test_recurrent_contract(case, dtype, normalize, entry):
    data = build(case, dtype, gems.device, normalize)
    expected = reference(data)
    original = (
        data["initial_state"].clone() if data["initial_state"] is not None else None
    )
    fn = getattr(
        gems.ops if entry == "generic" else gems, "fused_recurrent_gated_delta_rule_fwd"
    )
    result = fn(**data)
    assert_result(result, expected, dtype)
    if data["inplace_final_state"]:
        assert result[1] is data["initial_state"]
        # Unwritten slots are part of the alias contract, not a tolerance check.
        original_cpu = original.cpu()
        untouched = (expected[1] == original_cpu).flatten(1).all(1)
        torch.testing.assert_close(
            result[1].cpu()[untouched], original_cpu[untouched], rtol=0, atol=0
        )
    elif original is not None:
        torch.testing.assert_close(data["initial_state"], original, rtol=0, atol=0)


@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tiny", [False, True])
def test_recurrent_optional_gate_and_beta(dtype, tiny):
    data = build(CASES[1], dtype, gems.device)
    data.update(g=None, beta=None)
    if tiny:
        data["q"].mul_(1e-5)
        data["k"].mul_(1e-5)
    expected = reference(data)
    assert_result(gems.fused_recurrent_gated_delta_rule_fwd(**data), expected, dtype)


@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.parametrize(
    "kind", ["layout", "scale", "indices", "accepted", "state", "dtype"]
)
def test_recurrent_invalid_metadata(kind):
    data = build(CASES[1], torch.float32, gems.device)
    if kind == "layout":
        data["initial_state"] = data["initial_state"].transpose(-1, -2)
    if kind == "scale":
        data["scale"] = 0
    if kind == "indices":
        data["ssm_state_indices"] = data["ssm_state_indices"][:1]
    if kind == "accepted":
        data["num_accepted_tokens"] = torch.ones(
            3, dtype=torch.int32, device=gems.device
        )
    if kind == "state":
        data["initial_state"] = None
    if kind == "dtype":
        data["q"] = data["q"].to(torch.int32)
    with pytest.raises((ValueError, NotImplementedError)):
        gems.fused_recurrent_gated_delta_rule_fwd(**data)


@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.parametrize("entry", ["generic", "top"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(9, 4, 12, 64, 32), (1, 16, 64, 128, 128)])
def test_recurrent_state_updated_once(entry, dtype, shape):
    # More logical tiles than AIV cores; task zero must be valid to expose replay.
    n, h, hv, k, v = shape
    case = ("multistep", [1] * n, h, hv, k, v, "index", True, True)
    data = build(case, dtype, gems.device)
    ref_data = {
        key: value.cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }
    parallel = os.environ.get("TRITON_ALL_BLOCKS_PARALLEL")
    fn = getattr(
        gems.ops if entry == "generic" else gems, "fused_recurrent_gated_delta_rule_fwd"
    )
    for _ in range(4):
        expected = reference(ref_data)
        ref_data["initial_state"] = expected[1]
        result = fn(**data)
        assert result[1] is data["initial_state"]
        assert_result(result, expected, dtype)
        assert os.environ.get("TRITON_ALL_BLOCKS_PARALLEL") == parallel
