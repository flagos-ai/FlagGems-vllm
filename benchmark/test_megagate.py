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

"""Eager-only multi-profile MegaGate benchmark against native vLLM chains.

``MegaGateContext.prepare`` runs while constructing each benchmark input and
is deliberately excluded from timing.  The measured candidate is the public
``flaggems_vllm.megagate`` call, so Python dispatch, output allocation, and its
single fused launch remain inside the measured operation.  CUDA Graph capture
or replay is not a supported mode for this benchmark.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate.registry import (
    PROFILES as _REGISTERED_PROFILES,
)
from flaggems_vllm.utils.tle_capabilities import (
    supports_megagate_tle,
    supports_module_wide_named_barriers,
)

from . import base, consts
from .conftest import Config


@dataclass(frozen=True)
class _Profile:
    name: str
    hidden: int
    experts: int
    topk: int
    routed_scaling_factor: float
    scoring_func: str
    correction_bias: bool
    linear_kind: str
    num_expert_group: int | None = None
    topk_group: int | None = None


_PROFILES = (
    _Profile(
        "dsv4_flash",
        4096,
        256,
        6,
        1.5,
        "sqrtsoftplus",
        True,
        "GateLinear",
    ),
    _Profile(
        "dsv32",
        7168,
        256,
        8,
        2.5,
        "sigmoid",
        True,
        "GateLinear",
        num_expert_group=8,
        topk_group=4,
    ),
    _Profile(
        "qwen_h2048_e256_k8",
        2048,
        256,
        8,
        1.0,
        "softmax",
        False,
        "ReplicatedLinear",
    ),
    _Profile(
        "qwen_h4096_e512_k10",
        4096,
        512,
        10,
        1.0,
        "softmax",
        False,
        "ReplicatedLinear",
    ),
)
_PROFILE_BY_SIGNATURE = {
    (profile.hidden, profile.experts, profile.topk): profile for profile in _PROFILES
}

# ``MegaGateProfile.routing`` is the production semantic contract.  Keep the
# benchmark-only vLLM constructor arguments in one explicit translation table
# so a registry routing change cannot silently continue timing a different
# baseline chain.
_ROUTING_BASELINE_CONTRACTS = {
    "sqrt_softplus_bias": {
        "scoring_func": "sqrtsoftplus",
        "correction_bias": True,
        "linear_kind": "GateLinear",
        "num_expert_group": None,
        "topk_group": None,
    },
    "grouped_sigmoid_bias": {
        "scoring_func": "sigmoid",
        "correction_bias": True,
        "linear_kind": "GateLinear",
        "num_expert_group": 8,
        "topk_group": 4,
    },
    "softmax": {
        "scoring_func": "softmax",
        "correction_bias": False,
        "linear_kind": "ReplicatedLinear",
        "num_expert_group": None,
        "topk_group": None,
    },
}
for _profile in _PROFILES:
    _registered = _REGISTERED_PROFILES[_profile.name]
    assert (_profile.hidden, _profile.experts, _profile.topk) == (
        _registered.hidden,
        _registered.experts,
        _registered.topk,
    )
    assert _profile.routed_scaling_factor == _registered.routed_scaling_factor
    _contract = _ROUTING_BASELINE_CONTRACTS[_registered.routing]
    assert {
        "scoring_func": _profile.scoring_func,
        "correction_bias": _profile.correction_bias,
        "linear_kind": _profile.linear_kind,
        "num_expert_group": _profile.num_expert_group,
        "topk_group": _profile.topk_group,
    } == _contract

# Core keeps one representative per model profile. Comprehensive adds one
# shape per admitted schedule family; adjacent T values sharing a schedule are
# deliberately omitted from the shared benchmark.
CORE_SHAPES = (
    (4096, 256, 6, 1028),
    (7168, 256, 8, 12292),
    (2048, 256, 8, 4434),
    (4096, 512, 10, 8214),
)
COMPREHENSIVE_SHAPES = (
    (4096, 256, 6, 1028),
    (4096, 256, 6, 16384),
    (7168, 256, 8, 1028),
    (7168, 256, 8, 12292),
    (7168, 256, 8, 16384),
    (2048, 256, 8, 1034),
    (2048, 256, 8, 4107),
    (2048, 256, 8, 4434),
    (2048, 256, 8, 9309),
    (2048, 256, 8, 16384),
    (4096, 512, 10, 1035),
    (4096, 512, 10, 2048),
    (4096, 512, 10, 4107),
    (4096, 512, 10, 4434),
    (4096, 512, 10, 5240),
    (4096, 512, 10, 8214),
    (4096, 512, 10, 13421),
    (4096, 512, 10, 16384),
)

_VLLM_CONFIG_CONTEXT: Any | None = None
_VLLM_STORE_PATH: str | None = None
_VLLM_INITIALIZED_HERE = False


def _has_hopper_megagate() -> bool:
    if flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available():
        return False
    try:
        if tuple(torch.cuda.get_device_capability()) != (9, 0):
            return False
        import triton.experimental.tle.language as tle
    except (ImportError, RuntimeError):
        return False
    return supports_megagate_tle(tle) and supports_module_wide_named_barriers()


def _load_vllm_gate() -> tuple[dict[str, Any] | None, str | None]:
    try:
        gate_module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.router.gate_linear"
        )
        linear_module = importlib.import_module("vllm.model_executor.layers.linear")
        bias_module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
        )
        flat_module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.router.fused_topk_router"
        )
        grouped_module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.router.grouped_topk_router"
        )
        return (
            {
                "GateLinear": gate_module.GateLinear,
                "ReplicatedLinear": linear_module.ReplicatedLinear,
                "fused_topk_bias": bias_module.fused_topk_bias,
                "fused_topk": flat_module.fused_topk,
                "grouped_topk": grouped_module.grouped_topk,
            },
            None,
        )
    except Exception as error:  # pragma: no cover - depends on benchmark image
        return None, repr(error)


VLLM_GATE_API, VLLM_GATE_ERROR = _load_vllm_gate()


def _ensure_vllm_single_rank() -> None:
    """Initialize the TP=PP=1 state required by vLLM ``GateLinear``."""

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    global _VLLM_CONFIG_CONTEXT
    global _VLLM_INITIALIZED_HERE
    global _VLLM_STORE_PATH
    if model_parallel_is_initialized():
        return

    _VLLM_CONFIG_CONTEXT = set_current_vllm_config(VllmConfig())
    _VLLM_CONFIG_CONTEXT.__enter__()
    descriptor, store_path = tempfile.mkstemp(prefix="vllm-megagate-benchmark-")
    os.close(descriptor)
    _VLLM_STORE_PATH = store_path
    _VLLM_INITIALIZED_HERE = True
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{store_path}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
    except Exception:
        _cleanup_vllm_single_rank()
        raise


def _cleanup_vllm_single_rank() -> None:
    global _VLLM_CONFIG_CONTEXT
    global _VLLM_INITIALIZED_HERE
    global _VLLM_STORE_PATH
    try:
        if _VLLM_INITIALIZED_HERE:
            from vllm.distributed import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            try:
                destroy_model_parallel()
            finally:
                destroy_distributed_environment()
    finally:
        _VLLM_INITIALIZED_HERE = False
        if _VLLM_CONFIG_CONTEXT is not None:
            _VLLM_CONFIG_CONTEXT.__exit__(None, None, None)
            _VLLM_CONFIG_CONTEXT = None
        if _VLLM_STORE_PATH is not None:
            try:
                os.unlink(_VLLM_STORE_PATH)
            except FileNotFoundError:
                pass
            _VLLM_STORE_PATH = None


def _construct_vllm_gate(
    vllm_api: dict[str, Any], weight: torch.Tensor, profile: _Profile
) -> Any:
    linear_type = vllm_api[profile.linear_kind]
    kwargs: dict[str, Any] = {
        "input_size": profile.hidden,
        "output_size": profile.experts,
        "bias": False,
        "params_dtype": torch.bfloat16,
        "prefix": f"benchmark.megagate.{profile.name}",
    }
    if profile.linear_kind == "GateLinear":
        # DSV4 fixes FP32 router logits. DSV3.2 leaves out_dtype unset and
        # follows GateLinear's model-faithful producer precision policy.
        kwargs["out_dtype"] = torch.float32 if profile.name == "dsv4_flash" else None
    layer = linear_type(
        **kwargs,
    ).cuda()
    layer.eval()
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer


def _unwrap_linear_output(value: Any) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def _candidate(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    *,
    context: Any,
    baseline: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    del baseline
    return flaggems_vllm.megagate(
        x,
        gate_weight,
        correction_bias,
        context=context,
        routed_scaling_factor=context.profile_spec.routed_scaling_factor,
        out=None,
    )


def _vllm_baseline(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    *,
    context: Any,
    baseline: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    del x, gate_weight, correction_bias, context
    return baseline()


class MegaGateBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = list(CORE_SHAPES)
    DEFAULT_SHAPE_DESC = "H, E, K, T"

    def set_shapes(self, shape_file_path=None) -> None:
        # MegaGate only admits the exact model-derived token counts below. Do
        # not inherit the generic ``Benchmark`` YAML shapes through the MRO.
        del shape_file_path
        shapes = (
            COMPREHENSIVE_SHAPES
            if Config.bench_level is consts.BenchLevel.COMPREHENSIVE
            else CORE_SHAPES
        )
        self.shapes = list(shapes)
        self.shape_desc = self.DEFAULT_SHAPE_DESC


def _megagate_inputs(
    shape: tuple[int, int, int, int], dtype: torch.dtype, device: str
) -> Generator[tuple[Any, ...], None, None]:
    if dtype != torch.bfloat16:
        raise ValueError("MegaGate only supports BF16 x and gate_weight")
    if VLLM_GATE_API is None:
        pytest.skip(f"native vLLM MegaGate baseline is unavailable: {VLLM_GATE_ERROR}")
    if not hasattr(flaggems_vllm, "create_megagate_context") or not hasattr(
        flaggems_vllm, "megagate"
    ):
        raise RuntimeError("Hopper MegaGate public exports are unavailable")

    hidden, experts, topk, tokens = (int(value) for value in shape)
    try:
        profile = _PROFILE_BY_SIGNATURE[(hidden, experts, topk)]
    except KeyError as error:
        raise ValueError(f"unsupported MegaGate benchmark shape {shape}") from error
    generator = torch.Generator(device=device).manual_seed(20261112 + tokens)
    x = torch.randn((tokens, hidden), dtype=dtype, device=device, generator=generator)
    gate_weight = torch.randn(
        (experts, hidden), dtype=dtype, device=device, generator=generator
    ).mul_(hidden**-0.5)
    correction_bias = (
        torch.randn(
            (experts,), dtype=torch.float32, device=device, generator=generator
        ).mul_(0.01)
        if profile.correction_bias
        else None
    )

    # Prepare/compile belongs to model initialization and is intentionally
    # outside both timed callables.
    context = flaggems_vllm.create_megagate_context(profile=profile.name)
    context.prepare(
        x,
        gate_weight,
        correction_bias,
        routed_scaling_factor=profile.routed_scaling_factor,
    )

    try:
        layer = _construct_vllm_gate(VLLM_GATE_API, gate_weight, profile)

        def run_vllm() -> tuple[torch.Tensor, torch.Tensor]:
            logits = _unwrap_linear_output(layer(x))
            if profile.num_expert_group is not None:
                return VLLM_GATE_API["grouped_topk"](
                    hidden_states=x,
                    gating_output=logits,
                    topk=profile.topk,
                    renormalize=True,
                    num_expert_group=profile.num_expert_group,
                    topk_group=profile.topk_group,
                    scoring_func=profile.scoring_func,
                    routed_scaling_factor=profile.routed_scaling_factor,
                    e_score_correction_bias=correction_bias,
                )
            if profile.correction_bias:
                return VLLM_GATE_API["fused_topk_bias"](
                    hidden_states=x,
                    gating_output=logits,
                    scoring_func=profile.scoring_func,
                    e_score_correction_bias=correction_bias,
                    topk=profile.topk,
                    renormalize=True,
                    routed_scaling_factor=profile.routed_scaling_factor,
                    indices_type=(
                        torch.int64 if profile.name == "dsv4_flash" else None
                    ),
                )
            weights, indices, _ = VLLM_GATE_API["fused_topk"](
                hidden_states=x,
                gating_output=logits,
                topk=profile.topk,
                renormalize=True,
                indices_type=None,
                scoring_func=profile.scoring_func,
            )
            return weights, indices

        yield x, gate_weight, correction_bias, {
            "context": context,
            "baseline": run_vllm,
        }
    finally:
        context.close()


@pytest.mark.megagate
@pytest.mark.skipif(
    not _has_hopper_megagate(),
    reason="MegaGate benchmark requires NVIDIA SM90 and its TLE APIs",
)
def test_megagate() -> None:
    if Config.mode != consts.BenchMode.OPERATOR:
        pytest.skip(
            "MegaGate acceptance uses eager operator timing so public dispatch "
            "is included; CUDA Graph and GPU-only timing are excluded"
        )
    if VLLM_GATE_API is None:
        pytest.skip(f"native vLLM MegaGate baseline is unavailable: {VLLM_GATE_ERROR}")

    _ensure_vllm_single_rank()
    try:
        benchmark = MegaGateBenchmark(
            op_name="megagate",
            input_fn=_megagate_inputs,
            torch_op=_vllm_baseline,
            gems_op=_candidate,
            dtypes=[torch.bfloat16],
        )
        benchmark.run()
    finally:
        _cleanup_vllm_single_rank()
