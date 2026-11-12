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

"""Eager-only DSV4 MegaGate benchmark against the native vLLM chain.

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
from typing import Any

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate.api import (
    EXPERTS,
    HIDDEN,
    PROFILE,
    ROUTED_SCALING_FACTOR,
    TOPK,
)
from flaggems_vllm.utils.tle_capabilities import (
    supports_megagate_tle,
    supports_module_wide_named_barriers,
)

from . import base, consts
from .conftest import Config

CORE_TOKENS = (1028, 4100, 16384)
COMPREHENSIVE_TOKENS = (
    1028,
    2048,
    3076,
    3805,
    4100,
    4835,
    7884,
    8221,
    13021,
    16384,
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


def _load_vllm_gate() -> tuple[tuple[type, Callable[..., Any]] | None, str | None]:
    try:
        gate_module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.router.gate_linear"
        )
        router_module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
        )
        return (gate_module.GateLinear, router_module.fused_topk_bias), None
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


def _construct_vllm_gate(gate_linear: type, weight: torch.Tensor) -> Any:
    layer = gate_linear(
        input_size=HIDDEN,
        output_size=EXPERTS,
        bias=False,
        out_dtype=torch.float32,
        params_dtype=torch.bfloat16,
        prefix="benchmark.megagate.dsv4",
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
    correction_bias: torch.Tensor,
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
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        out=None,
    )


def _vllm_baseline(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    context: Any,
    baseline: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    del x, gate_weight, correction_bias, context
    return baseline()


class MegaGateBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [(tokens,) for tokens in CORE_TOKENS]
    DEFAULT_SHAPE_DESC = "T"

    def set_shapes(self, shape_file_path=None) -> None:
        # MegaGate only admits the exact model-derived token counts below. Do
        # not inherit the generic ``Benchmark`` YAML shapes through the MRO.
        del shape_file_path
        tokens = (
            COMPREHENSIVE_TOKENS
            if Config.bench_level is consts.BenchLevel.COMPREHENSIVE
            else CORE_TOKENS
        )
        self.shapes = [(tokens_value,) for tokens_value in tokens]
        self.shape_desc = self.DEFAULT_SHAPE_DESC


def _megagate_inputs(
    shape: tuple[int], dtype: torch.dtype, device: str
) -> Generator[tuple[Any, ...], None, None]:
    if dtype != torch.bfloat16:
        raise ValueError("MegaGate only supports BF16 x and gate_weight")
    if VLLM_GATE_API is None:
        pytest.skip(f"native vLLM MegaGate baseline is unavailable: {VLLM_GATE_ERROR}")
    if not hasattr(flaggems_vllm, "create_megagate_context") or not hasattr(
        flaggems_vllm, "megagate"
    ):
        raise RuntimeError("Hopper MegaGate public exports are unavailable")

    tokens = int(shape[0])
    generator = torch.Generator(device=device).manual_seed(20261112 + tokens)
    x = torch.randn((tokens, HIDDEN), dtype=dtype, device=device, generator=generator)
    gate_weight = torch.randn(
        (EXPERTS, HIDDEN), dtype=dtype, device=device, generator=generator
    ).mul_(HIDDEN**-0.5)
    correction_bias = torch.randn(
        (EXPERTS,), dtype=torch.float32, device=device, generator=generator
    ).mul_(0.01)

    # Prepare/compile belongs to model initialization and is intentionally
    # outside both timed callables.
    context = flaggems_vllm.create_megagate_context(profile=PROFILE)
    context.prepare(
        x,
        gate_weight,
        correction_bias,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )

    gate_linear, fused_topk_bias = VLLM_GATE_API
    layer = _construct_vllm_gate(gate_linear, gate_weight)

    def run_vllm() -> tuple[torch.Tensor, torch.Tensor]:
        logits = _unwrap_linear_output(layer(x))
        return fused_topk_bias(
            hidden_states=x,
            gating_output=logits,
            scoring_func="sqrtsoftplus",
            e_score_correction_bias=correction_bias,
            topk=TOPK,
            renormalize=True,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            indices_type=torch.int64,
        )

    try:
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
