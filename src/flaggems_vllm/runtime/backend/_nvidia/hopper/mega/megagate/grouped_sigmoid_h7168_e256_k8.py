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

"""Functional no-Graph MegaGate entry for DSV3.2 on Hopper.

This module exposes the ten exact H7168/E256/K8 eager shapes admitted by the
final dispatcher while fixing the lifecycle mismatch of the historical
``Plan`` interface. Compilation selects the v371, v372, or v435 source leaf
for the exact token count, but a steady-state call binds the *current* X,
weight, correction-bias, output, stream, and per-stream workspace objects.

``MegaGateContext.prepare`` is setup work: it validates SM90, verifies the
required TLE capabilities, constructs one exact specialization, and loads its
raw compiled runner. ``megagate`` is the measured eager operation: exact-T dict
lookup, current-address descriptor lookup/construction, two fresh semantic
outputs (unless caller-owned outputs are explicit), and one fused kernel on
the current stream.  There is no CUDA Graph, helper kernel, padding, fallback,
or address-bound plan launch in this path.
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

PROFILE: Final = "dsv32"
TOKEN_TO_LEAF: Final = get_module_dispatch(__name__)
SUPPORTED_TOKENS: Final = tuple(TOKEN_TO_LEAF)
V371_TOKENS: Final = tuple(
    tokens for tokens, leaf in TOKEN_TO_LEAF.items() if leaf == "v371"
)
V372_TOKENS: Final = tuple(
    tokens for tokens, leaf in TOKEN_TO_LEAF.items() if leaf == "v372"
)
V435_TOKENS: Final = tuple(
    tokens for tokens, leaf in TOKEN_TO_LEAF.items() if leaf == "v435"
)

HIDDEN: Final = 7168
EXPERTS: Final = 256
TOPK: Final = 8
ROUTED_SCALING_FACTOR: Final = 2.5
BLOCK_TOKENS: Final = 128
BLOCK_K: Final = 128
_INPUT_CONTRACT: Final = InputContract(
    name="functional DSV32 eager10",
    supported_tokens=SUPPORTED_TOKENS,
    hidden=HIDDEN,
    experts=EXPERTS,
    routed_scaling_factor=ROUTED_SCALING_FACTOR,
    requires_correction_bias=True,
)
STATE_WORDS_PER_TILE: Final = 16

assert len(SUPPORTED_TOKENS) == 10
assert tuple(sorted(SUPPORTED_TOKENS)) == SUPPORTED_TOKENS
assert set(TOKEN_TO_LEAF) == set(SUPPORTED_TOKENS)


@dataclass(frozen=True)
class _CompiledLeaf:
    tokens: int
    device_index: int
    family: str
    grid: tuple[int, ...]
    descriptor_experts: int
    requires_state: bool
    runner: ExactGridUnboundCompiledRunner


@dataclass
class _Workspace:
    logits: torch.Tensor
    state: torch.Tensor | None
    x_descriptor_entry: (
        tuple[tuple[int, tuple[int, ...], tuple[int, ...]], Any] | None
    ) = None


def _allocator(size: int, alignment: int, stream: int) -> torch.Tensor:
    del alignment, stream
    return torch.empty(
        size,
        device=torch.device("cuda", torch.cuda.current_device()),
        dtype=torch.uint8,
    )


def _load_dispatch_modules() -> dict[str, Any]:
    """Import TLE modules only when a CUDA preparation is requested."""

    from .kernels import (
        sm90_grouped_sigmoid_h7168_e256_k8_clusterless_b128_e64 as clusterless,
    )
    from .kernels import (
        sm90_grouped_sigmoid_h7168_e256_k8_dualmath_b128_e128 as dualmath,
    )
    from .kernels import (
        sm90_grouped_sigmoid_h7168_e256_k8_quadmath_b128_e256 as quadmath,
    )

    return {
        "v371": clusterless,
        "v372": dualmath,
        "v435": quadmath,
    }


class MegaGateContext(HostContext[_CompiledLeaf, _Workspace]):
    """Compiled exact leaves plus per-device/per-stream internal workspaces."""

    api_kind = "functional_unbound_eager_dsv32"
    kernel_count_per_call = 1
    final_scope = "dsv32_h7168_e256_k8_eager10"

    def __init__(self, *, profile: str = PROFILE) -> None:
        if profile != PROFILE:
            raise ValueError(
                f"unsupported MegaGate profile {profile!r}; expected {PROFILE!r}"
            )
        super().__init__()
        self._weight_descriptor_entries: dict[
            int, tuple[tuple[int, int, int, int], Any]
        ] = {}
        self._descriptor_type: Any | None = None

    @staticmethod
    def _allocate_workspace(
        tokens: int, device: torch.device, *, requires_state: bool
    ) -> _Workspace:
        token_tiles = (tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        return _Workspace(
            logits=torch.empty((tokens, EXPERTS), device=device, dtype=torch.bfloat16),
            state=(
                torch.zeros(
                    (token_tiles, STATE_WORDS_PER_TILE),
                    device=device,
                    dtype=torch.int64,
                )
                if requires_state
                else None
            ),
        )

    def _weight_desc(
        self,
        gate_weight: torch.Tensor,
        *,
        descriptor_experts: int,
        block_k: int,
        device_index: int,
    ) -> Any:
        descriptor_type = self._descriptor_type
        if descriptor_type is None:
            raise UnpreparedShapeError("context descriptor state was not prepared")
        key = (
            int(gate_weight.data_ptr()),
            device_index,
            descriptor_experts,
            block_k,
        )
        entry = self._weight_descriptor_entries.get(descriptor_experts)
        if entry is not None and entry[0] == key:
            return entry[1]
        descriptor = descriptor_type(
            gate_weight,
            shape=[EXPERTS, HIDDEN],
            strides=[HIDDEN, 1],
            block_shape=[descriptor_experts, block_k],
        )
        with self._lock:
            # One dict value contains both key and immutable descriptor.
            self._weight_descriptor_entries[descriptor_experts] = (key, descriptor)
        return descriptor

    def _x_desc(
        self,
        workspace: _Workspace,
        x: torch.Tensor,
        tokens: int,
    ) -> Any:
        descriptor_type = self._descriptor_type
        if descriptor_type is None:
            raise UnpreparedShapeError("context descriptor state was not prepared")
        key = (
            int(x.data_ptr()),
            tuple(int(value) for value in x.shape),
            tuple(int(value) for value in x.stride()),
        )
        entry = workspace.x_descriptor_entry
        if entry is not None and entry[0] == key:
            return entry[1]
        descriptor = descriptor_type(
            x,
            shape=[tokens, HIDDEN],
            strides=[HIDDEN, 1],
            block_shape=[BLOCK_TOKENS, BLOCK_K],
        )
        with self._lock:
            workspace.x_descriptor_entry = (key, descriptor)
        return descriptor

    def clear_cached_descriptors(self) -> None:
        """Release strong references held by last-address descriptors."""

        with self._lock:
            for workspace in self._workspaces.values():
                workspace.x_descriptor_entry = None
            self._weight_descriptor_entries = {}

    def _clear_schedule_state(self) -> None:
        self._descriptor_type = None

    def prepare(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor | None,
        *,
        plan: KernelPlan,
        routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    ) -> Any:
        """Compile one exact final-dispatch leaf without launching it."""

        tokens, device = validate_prepare_inputs(
            _INPUT_CONTRACT,
            x,
            gate_weight,
            correction_bias,
            routed_scaling_factor,
        )
        require_megagate_tle()
        modules = _load_dispatch_modules()
        import triton
        from triton.tools.tensor_descriptor import TensorDescriptor

        with self._lock, torch.cuda.device(device):
            device_index = self._resolve_device_index(device)
            self._require_prepare_device(device_index)
            self._device_index = device_index
            self._descriptor_type = TensorDescriptor
            triton.set_allocator(_allocator)

            family = plan.source_leaf
            module = modules[family]
            if family == "v371":
                kernel = module._fused_e64_clusterless_kernel
                descriptor_experts = module.EXPERTS_PER_CTA
                logical_expert_ctas = module.LOGICAL_EXPERT_CTAS
                requires_state = True
                expected_argument_count = 9
            elif family == "v372":
                kernel = module._fused_e128_dualmath_kernel
                descriptor_experts = module.EXPERTS_PER_MATH
                logical_expert_ctas = module.LOGICAL_EXPERT_CTAS
                requires_state = True
                expected_argument_count = 9
            else:
                kernel = module._fused_e256_quadmath_kernel
                descriptor_experts = module.EXPERTS
                logical_expert_ctas = module.LOGICAL_EXPERT_CTAS
                requires_state = False
                expected_argument_count = 8

            stream = torch.cuda.current_stream(device)
            workspace_key = self._workspace_key(device, tokens, stream=stream)
            workspace = self._get_or_create_workspace(
                workspace_key,
                lambda: self._allocate_workspace(
                    tokens, device, requires_state=requires_state
                ),
            )
            self._x_desc(workspace, x, tokens)
            existing = self._compiled.get(tokens)
            if existing is not None:
                return existing.runner.compiled

            weight_desc = self._weight_desc(
                gate_weight,
                descriptor_experts=descriptor_experts,
                block_k=BLOCK_K,
                device_index=device_index,
            )
            x_desc = self._x_desc(workspace, x, tokens)
            topk_weights = torch.empty(
                (tokens, TOPK), device=device, dtype=torch.float32
            )
            topk_indices = torch.empty((tokens, TOPK), device=device, dtype=torch.int32)
            token_tiles = (tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
            grid = (token_tiles * logical_expert_ctas,)
            if requires_state:
                state = workspace.state
                if state is None:
                    raise RuntimeError("stateful leaf has no per-stream ticket state")
                compiled = kernel.warmup(
                    weight_desc,
                    x_desc,
                    workspace.logits,
                    state,
                    correction_bias,
                    topk_indices,
                    topk_weights,
                    routed_scaling_factor,
                    tokens,
                    grid=grid,
                    num_ctas=1,
                    num_warps=4,
                    maxnreg=module.KERNEL_MAXNREG,
                )
            else:
                compiled = kernel.warmup(
                    weight_desc,
                    x_desc,
                    workspace.logits,
                    correction_bias,
                    topk_indices,
                    topk_weights,
                    routed_scaling_factor,
                    tokens,
                    grid=grid,
                    num_ctas=1,
                    num_warps=4,
                    maxnreg=module.KERNEL_MAXNREG,
                )
            runner = ExactGridUnboundCompiledRunner(
                compiled=compiled,
                grid=grid,
                expected_argument_count=expected_argument_count,
            )
            self._compiled[tokens] = _CompiledLeaf(
                tokens=tokens,
                device_index=device_index,
                family=family,
                grid=grid,
                descriptor_experts=descriptor_experts,
                requires_state=requires_state,
                runner=runner,
            )
            return compiled

    def run(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor | None,
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
        weight_desc = self._weight_desc(
            gate_weight,
            descriptor_experts=leaf.descriptor_experts,
            block_k=BLOCK_K,
            device_index=device_index,
        )
        x_desc = self._x_desc(workspace, x, tokens)
        topk_weights = torch.empty((tokens, TOPK), device=device, dtype=torch.float32)
        topk_indices = torch.empty((tokens, TOPK), device=device, dtype=torch.int32)

        # correction_bias is deliberately the tensor from this invocation;
        # unlike the old Plan it is never retained as a bound launch pointer.
        if leaf.requires_state:
            state = workspace.state
            if state is None:
                raise RuntimeError("stateful leaf has no per-stream ticket state")
            leaf.runner[leaf.grid](
                weight_desc,
                x_desc,
                workspace.logits,
                state,
                correction_bias,
                topk_indices,
                topk_weights,
                routed_scaling_factor,
                tokens,
                stream=stream_handle,
            )
        else:
            leaf.runner[leaf.grid](
                weight_desc,
                x_desc,
                workspace.logits,
                correction_bias,
                topk_indices,
                topk_weights,
                routed_scaling_factor,
                tokens,
                stream=stream_handle,
            )
        return topk_weights, topk_indices


def megagate(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    *,
    context: MegaGateContext,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch one exact-T functional DSV3.2 fused kernel without a Graph."""

    if not isinstance(context, MegaGateContext):
        raise TypeError("context must be MegaGateContext")
    return context.run(
        x,
        gate_weight,
        correction_bias,
        routed_scaling_factor=routed_scaling_factor,
    )


__all__ = [
    "EXPERTS",
    "HIDDEN",
    "MegaGateContext",
    "MegaGateContextError",
    "PROFILE",
    "ROUTED_SCALING_FACTOR",
    "SUPPORTED_TOKENS",
    "TOKEN_TO_LEAF",
    "TOPK",
    "UnpreparedShapeError",
    "V371_TOKENS",
    "V372_TOKENS",
    "V435_TOKENS",
    "megagate",
]
