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

"""Shared, device-independent contract tests for Hygon MegaMoE.

Multi-rank correctness belongs to ``benchmark/test_megamoe.py`` because the
caller, rather than the operator, owns the DUSHMEM process lifecycle and
symmetric workspace. These tests intentionally remain collectable and useful
on hosts without a Hygon device.
"""

import inspect
from dataclasses import FrozenInstanceError, fields

import pytest

from benchmark.megamoe_hygon_vllm_baseline import (
    COMMUNICATION_BACKEND,
    VLLM_REQUIRED_VERSION,
    HcuNativeEpInt8Baseline,
    NativeEpContract,
    PrequantizedHcuTritonExperts,
)
from flaggems_vllm.runtime import device
from flaggems_vllm.runtime.backend._hygon.mega.megamoe.api import (
    MegaMoEContext,
    MegaMoETileConfig,
    MegaMoEWeights,
    default_tile_config,
    megamoe_int8,
)


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
def test_megamoe_int8_python_contract():
    assert list(inspect.signature(megamoe_int8).parameters) == [
        "input_int8",
        "input_scales",
        "topk_ids",
        "topk_weights",
        "weights",
        "context",
        "output",
    ]
    assert [field.name for field in fields(MegaMoEWeights)] == [
        "w1_gate_t",
        "w1_up_t",
        "w2_t",
        "w1_gate_scale",
        "w1_up_scale",
        "w2_scale",
    ]
    context_fields = {field.name for field in fields(MegaMoEContext)}
    assert {
        "rank",
        "world_size",
        "tokens_per_rank",
        "tile",
        "q_local",
        "q_table",
        "route_queue",
        "sync_status",
        "output",
        "epoch",
        "lock",
        "_poisoned",
    } <= context_fields
    assert callable(MegaMoEContext.raise_if_sync_failed)


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
@pytest.mark.parametrize(
    "tokens,expected",
    [
        (1, MegaMoETileConfig(32, 64, 128, 64, 32, 4)),
        (4, MegaMoETileConfig(32, 64, 128, 64, 32, 4)),
        (5, MegaMoETileConfig(32, 128, 128, 64, 32, 4)),
        (64, MegaMoETileConfig(32, 128, 128, 64, 32, 4)),
        (65, MegaMoETileConfig(32, 128, 128, 64, 32, 8)),
        (512, MegaMoETileConfig(32, 128, 128, 64, 32, 8)),
        (513, MegaMoETileConfig(128, 128, 128, 64, 64, 8)),
        (16384, MegaMoETileConfig(128, 128, 128, 64, 64, 8)),
    ],
)
def test_default_tile_config_boundaries(tokens, expected):
    assert default_tile_config(tokens) == expected


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
@pytest.mark.parametrize(
    "tokens,error",
    [
        (False, TypeError),
        (1.0, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (16385, ValueError),
    ],
)
def test_default_tile_config_rejects_unsupported_values(tokens, error):
    with pytest.raises(error):
        default_tile_config(tokens)


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
def test_prepacked_weight_descriptor_is_frozen():
    marker = object()
    weights = MegaMoEWeights(marker, marker, marker, marker, marker, marker)
    with pytest.raises(FrozenInstanceError):
        weights.w2_t = object()


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
def test_native_ep_baseline_contract_is_device_independent():
    """The paired benchmark contract must remain inspectable without HCU."""

    contract = NativeEpContract(
        world_size=4,
        rank=2,
        tokens_per_rank=64,
        hidden_size=2048,
        intermediate_size=512,
        num_experts=256,
        top_k=8,
    )
    assert contract.experts_per_rank == 64
    assert contract.local_expert_begin == 128
    assert contract.local_expert_end == 192
    assert VLLM_REQUIRED_VERSION == "0.18.1"
    assert COMMUNICATION_BACKEND == "allgather_reducescatter"
    assert HcuNativeEpInt8Baseline.first_input_quantization_in_timed_region is False
    assert HcuNativeEpInt8Baseline.intermediate_quantization_in_timed_region is True

    source = inspect.getsource(PrequantizedHcuTritonExperts.apply)
    assert "ignore_invalid_experts=True" in source


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
def test_native_ep_baseline_contract_rejects_unsupported_world_size():
    with pytest.raises(ValueError, match="world_size"):
        NativeEpContract(
            world_size=3,
            rank=0,
            tokens_per_rank=8,
            hidden_size=2048,
            intermediate_size=512,
            num_experts=256,
            top_k=8,
        )


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
def test_hygon_public_export_smoke():
    """Hygon dispatch must expose exactly the backend implementation."""

    if device.vendor_name != "hygon":
        pytest.skip("public MegaMoE dispatch is installed only on Hygon")

    import flaggems_vllm
    from flaggems_vllm.runtime.backend._hygon.ops import megamoe as backend

    assert callable(backend.megamoe_int8)
    assert flaggems_vllm.megamoe_int8 is backend.megamoe_int8
