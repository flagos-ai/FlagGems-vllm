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

"""Functional eager MegaGate entry for DSV4 on Hopper.

This module exposes the admitted DSV4 kernel selection through a functional
operator boundary: every invocation binds the current input, weight and bias,
allocates fresh semantic outputs, and submits exactly one fused kernel without
CUDA Graph capture or replay.

Compilation and the SM90 architecture probe belong to :meth:`prepare`.
Steady-state calls perform an O(1) exact-T lookup and reuse only internal,
per-device/per-stream workspaces.  Tensor descriptors are rebound when their
backing storage changes.  The raw compiled runner receives the caller's
current stream explicitly; the old ``Plan.launch`` path is never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import torch

from .capabilities import require_megagate_tle
from .context import (
    HostContext,
    InputContract,
    validate_prepare_inputs,
    validate_runtime_inputs,
)
from .errors import MegaGateContextError, UnpreparedShapeError
from .kernels.runtime import ExactGridUnboundCompiledRunner
from .registry import get_module_dispatch
from .types import KernelPlan

PROFILE: Final = "dsv4_flash"
TOKEN_TO_LEAF: Final = get_module_dispatch(__name__)
SUPPORTED_TOKENS: Final = tuple(TOKEN_TO_LEAF)
CLUSTERLESS_TOKENS: Final = tuple(
    tokens for tokens, leaf in TOKEN_TO_LEAF.items() if leaf == "v363"
)
QUADMATH_TOKENS: Final = tuple(
    tokens for tokens, leaf in TOKEN_TO_LEAF.items() if leaf == "v376"
)
HIDDEN: Final = 4096
EXPERTS: Final = 256
TOPK: Final = 6
ROUTED_SCALING_FACTOR: Final = 1.5
BLOCK_TOKENS: Final = 128
BLOCK_K: Final = 128
_INPUT_CONTRACT: Final = InputContract(
    name="DSV4 MegaGate",
    supported_tokens=SUPPORTED_TOKENS,
    hidden=HIDDEN,
    experts=EXPERTS,
    routed_scaling_factor=ROUTED_SCALING_FACTOR,
    requires_correction_bias=True,
)


@dataclass(frozen=True)
class _LeafSpec:
    implementation: str
    experts_per_cta: int
    logical_expert_ctas: int
    needs_state: bool
    expected_argument_count: int


_CLUSTERLESS_SPEC: Final = _LeafSpec(
    implementation="clusterless",
    experts_per_cta=64,
    logical_expert_ctas=4,
    needs_state=True,
    expected_argument_count=8,
)
_QUADMATH_SPEC: Final = _LeafSpec(
    implementation="quadmath",
    experts_per_cta=256,
    logical_expert_ctas=1,
    needs_state=False,
    expected_argument_count=7,
)
_LEAF_SPECS: Final = {"v363": _CLUSTERLESS_SPEC, "v376": _QUADMATH_SPEC}
LEAF_IMPLEMENTATION_BY_TOKENS: Final = {
    tokens: _LEAF_SPECS[leaf].implementation for tokens, leaf in TOKEN_TO_LEAF.items()
}


@dataclass(frozen=True)
class _CompiledLeaf:
    tokens: int
    spec: _LeafSpec
    device_index: int
    grid: tuple[int, ...]
    runner: ExactGridUnboundCompiledRunner


@dataclass
class _Workspace:
    logits: torch.Tensor
    state: torch.Tensor | None
    x_descriptor_entry: (
        tuple[tuple[int, tuple[int, ...], tuple[int, ...]], Any] | None
    ) = None


def _load_leaves() -> dict[str, Any]:
    # Keep the public module importable in CPU-only source/AST tests.
    from .kernels import (
        sm90_sqrt_softplus_h4096_e256_k6_clusterless_b128_e64 as clusterless,
    )
    from .kernels import sm90_sqrt_softplus_h4096_e256_k6_quadmath_b128_e256 as quadmath

    return {"clusterless": clusterless, "quadmath": quadmath}


class MegaGateContext(HostContext[_CompiledLeaf, _Workspace]):
    """Compiled exact-T leaves and per-device/per-stream internal storage."""

    api_kind = "functional_unbound_eager_dsv4"
    kernel_count_per_call = 1

    def __init__(self, *, profile: str = PROFILE) -> None:
        if profile != PROFILE:
            raise ValueError(
                f"unsupported MegaGate profile {profile!r}; expected {PROFILE!r}"
            )
        self.profile = profile
        super().__init__()
        # Retain one atomically published entry per leaf descriptor layout.
        # This keeps both descriptor layouts hot without retaining every prior
        # weight storage address for the lifetime of the context.
        self._weight_descriptor_entries: dict[
            int, tuple[tuple[int, int, int, int], Any] | None
        ] = {64: None, 256: None}
        self._modules: dict[str, Any] | None = None

    @staticmethod
    def _allocate_workspace(
        tokens: int, device: torch.device, spec: _LeafSpec
    ) -> _Workspace:
        token_tiles = (tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        state = (
            torch.zeros((token_tiles, 16), device=device, dtype=torch.int64)
            if spec.needs_state
            else None
        )
        return _Workspace(
            logits=torch.empty((tokens, EXPERTS), device=device, dtype=torch.float32),
            state=state,
        )

    def _weight_desc(
        self,
        common: Any,
        gate_weight: torch.Tensor,
        spec: _LeafSpec,
        device_index: int,
    ) -> Any:
        # The two implementations use different descriptor E blocks. Include that ABI
        # in the key; address-only caching would incorrectly reuse E64 for E256.
        key = (
            int(gate_weight.data_ptr()),
            device_index,
            spec.experts_per_cta,
            BLOCK_K,
        )
        entry = self._weight_descriptor_entries[spec.experts_per_cta]
        if entry is not None and entry[0] == key:
            return entry[1]
        descriptor = common.TensorDescriptor(
            gate_weight,
            shape=[EXPERTS, HIDDEN],
            strides=[HIDDEN, 1],
            block_shape=[spec.experts_per_cta, BLOCK_K],
        )
        with self._lock:
            # Publish key and descriptor with one Python reference assignment.
            self._weight_descriptor_entries[spec.experts_per_cta] = (
                key,
                descriptor,
            )
        return descriptor

    def _x_desc(
        self,
        common: Any,
        workspace: _Workspace,
        x: torch.Tensor,
        tokens: int,
    ) -> Any:
        key = (
            int(x.data_ptr()),
            tuple(int(value) for value in x.shape),
            tuple(int(value) for value in x.stride()),
        )
        entry = workspace.x_descriptor_entry
        if entry is not None and entry[0] == key:
            return entry[1]
        descriptor = common.TensorDescriptor(
            x,
            shape=[tokens, HIDDEN],
            strides=[HIDDEN, 1],
            block_shape=[BLOCK_TOKENS, BLOCK_K],
        )
        with self._lock:
            workspace.x_descriptor_entry = (key, descriptor)
        return descriptor

    def clear_cached_descriptors(self) -> None:
        """Release strong references retained by current-address descriptors."""

        with self._lock:
            for workspace in self._workspaces.values():
                workspace.x_descriptor_entry = None
            self._weight_descriptor_entries = {64: None, 256: None}

    def _clear_schedule_state(self) -> None:
        self._modules = None

    def prepare(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor,
        *,
        plan: KernelPlan,
        routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    ) -> None:
        """Prepare one exact DSV4 leaf and the current stream without launching."""

        require_megagate_tle()
        tokens, device = validate_prepare_inputs(
            _INPUT_CONTRACT,
            x,
            gate_weight,
            correction_bias,
            routed_scaling_factor,
        )
        device_index = self._resolve_device_index(device)
        if int(torch.cuda.current_device()) != device_index:
            raise ValueError(
                "current CUDA device differs from the prepared tensor device"
            )
        modules = _load_leaves()
        spec = _LEAF_SPECS[plan.source_leaf]
        module = modules[spec.implementation]
        common = modules["clusterless"]._common
        with self._lock, torch.cuda.device(device):
            self._require_prepare_device(device_index)
            stream = torch.cuda.current_stream(device)
            workspace_key = self._workspace_key(device, tokens, stream=stream)
            previous_device_index = self._device_index
            previous_modules = self._modules
            previous_last_workspace_entry = self._last_workspace_entry
            previous_weight_entries = self._weight_descriptor_entries.copy()
            previous_workspace = self._workspaces.get(workspace_key)
            previous_x_descriptor_entry = (
                previous_workspace.x_descriptor_entry
                if previous_workspace is not None
                else None
            )

            try:
                self._device_index = device_index
                self._modules = modules

                # The old Plan constructor installed this allocator before compile.
                # Keep that prepare-time prerequisite without constructing a Plan.
                module.triton.set_allocator(common._allocator)
                workspace = self._get_or_create_workspace(
                    workspace_key,
                    lambda: self._allocate_workspace(tokens, device, spec),
                )
                weight_desc = self._weight_desc(common, gate_weight, spec, device_index)
                x_desc = self._x_desc(common, workspace, x, tokens)
                existing = self._compiled.get(tokens)
                if existing is not None:
                    return

                topk_weights = torch.empty(
                    (tokens, TOPK), device=device, dtype=torch.float32
                )
                topk_indices = torch.empty(
                    (tokens, TOPK), device=device, dtype=torch.int64
                )
                token_tiles = (tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
                grid = (token_tiles * spec.logical_expert_ctas,)
                if spec.implementation == "clusterless":
                    if workspace.state is None:
                        raise AssertionError(
                            "clusterless implementation requires monotonic ticket state"
                        )
                    compiled = module._b128_clusterless_last_arriver_kernel.warmup(
                        weight_desc,
                        x_desc,
                        workspace.logits,
                        workspace.state,
                        correction_bias,
                        topk_indices,
                        topk_weights,
                        tokens,
                        grid=grid,
                        num_ctas=1,
                        num_warps=4,
                        maxnreg=module.KERNEL_MAXNREG,
                    )
                else:
                    compiled = module._fused_e256_quadmath_kernel.warmup(
                        weight_desc,
                        x_desc,
                        workspace.logits,
                        correction_bias,
                        topk_indices,
                        topk_weights,
                        tokens,
                        grid=grid,
                        num_ctas=1,
                        num_warps=4,
                        maxnreg=module.KERNEL_MAXNREG,
                    )
                runner = ExactGridUnboundCompiledRunner(
                    compiled=compiled,
                    grid=grid,
                    expected_argument_count=spec.expected_argument_count,
                )
                self._compiled[tokens] = _CompiledLeaf(
                    tokens=tokens,
                    spec=spec,
                    device_index=device_index,
                    grid=grid,
                    runner=runner,
                )
            except Exception:
                if previous_workspace is None:
                    self._workspaces.pop(workspace_key, None)
                else:
                    previous_workspace.x_descriptor_entry = previous_x_descriptor_entry
                self._last_workspace_entry = previous_last_workspace_entry
                self._weight_descriptor_entries = previous_weight_entries
                self._modules = previous_modules
                self._device_index = previous_device_index
                raise

    def run(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor,
        *,
        routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, device = validate_runtime_inputs(
            _INPUT_CONTRACT,
            x,
            gate_weight,
            correction_bias,
            routed_scaling_factor,
        )
        leaf, workspace, device_index, stream_handle = self._prepared_launch(
            tokens, device
        )
        modules = self._modules
        if modules is None:
            raise UnpreparedShapeError("context module state was not prepared")
        common = modules["clusterless"]._common
        spec = leaf.spec
        weight_desc = self._weight_desc(common, gate_weight, spec, device_index)
        x_desc = self._x_desc(common, workspace, x, tokens)
        topk_weights = torch.empty((tokens, TOPK), device=device, dtype=torch.float32)
        topk_indices = torch.empty((tokens, TOPK), device=device, dtype=torch.int64)

        if spec.implementation == "clusterless":
            state = workspace.state
            if state is None:
                raise AssertionError(
                    "clusterless implementation requires monotonic ticket state"
                )
            launch_args = (
                weight_desc,
                x_desc,
                workspace.logits,
                state,
                correction_bias,
                topk_indices,
                topk_weights,
                tokens,
            )
        else:
            launch_args = (
                weight_desc,
                x_desc,
                workspace.logits,
                correction_bias,
                topk_indices,
                topk_weights,
                tokens,
            )
        leaf.runner[leaf.grid](*launch_args, stream=stream_handle)
        return topk_weights, topk_indices


def megagate(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    context: MegaGateContext,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch one prepared DSV4 leaf as one eager, functional fused kernel."""

    if not isinstance(context, MegaGateContext):
        raise TypeError("context must be MegaGateContext")
    return context.run(
        x,
        gate_weight,
        correction_bias,
        routed_scaling_factor=routed_scaling_factor,
    )


__all__ = [
    "CLUSTERLESS_TOKENS",
    "EXPERTS",
    "HIDDEN",
    "LEAF_IMPLEMENTATION_BY_TOKENS",
    "MegaGateContext",
    "MegaGateContextError",
    "PROFILE",
    "QUADMATH_TOKENS",
    "ROUTED_SCALING_FACTOR",
    "SUPPORTED_TOKENS",
    "TOKEN_TO_LEAF",
    "TOPK",
    "UnpreparedShapeError",
    "megagate",
]
