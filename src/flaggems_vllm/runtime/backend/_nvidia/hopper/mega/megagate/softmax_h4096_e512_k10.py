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

"""Functional eager MegaGate entry for the Qwen-H4096 eager12 map.

Eight admitted TLE leaves are flattened into production modules.  This entry
selects them by exact token count and provides a public functional lifetime:

* ``prepare`` performs the exact leaf warmup and loads one raw compiled runner;
* every call dispatches on the current T and binds the current x/weight;
* internal logits, candidate scratch and state are owned per CUDA stream/T;
* semantic outputs are fresh by default (or may be supplied explicitly); and
* one raw runner invocation launches exactly one fused kernel on the explicitly
  selected current stream.

No CUDA Graph path or runtime research-tree dependency is introduced here.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import MappingProxyType
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

PROFILE: Final = "qwen_h4096_e512_k10"
TOKEN_TO_LEAF: Final = get_module_dispatch(__name__)
SUPPORTED_TOKENS: Final = tuple(TOKEN_TO_LEAF)
HIDDEN: Final = 4096
EXPERTS: Final = 512
TOPK: Final = 10
ROUTED_SCALING_FACTOR: Final = 1.0
_INPUT_CONTRACT: Final = InputContract(
    name="functional Qwen-H4096",
    supported_tokens=SUPPORTED_TOKENS,
    hidden=HIDDEN,
    experts=EXPERTS,
    routed_scaling_factor=ROUTED_SCALING_FACTOR,
    requires_correction_bias=False,
)


@dataclass(frozen=True)
class _LeafDefinition:
    name: str
    module_name: str
    weight_block: int
    x_blocks: tuple[tuple[int, int | None], ...]
    candidate_producers: int
    candidate_words: int
    candidate_dtype: torch.dtype
    state_tiles: int
    grid: int
    expected_abi_names: tuple[str, ...]
    uses_mesh: bool = False

    @property
    def expected_argument_count(self) -> int:
        return len(self.expected_abi_names)


_LEAF_DEFINITIONS: Final = (
    _LeafDefinition(
        name="v371",
        module_name="sm90_softmax_h4096_e512_k10_small_cluster8",
        weight_block=64,
        x_blocks=((128, None),),
        candidate_producers=0,
        candidate_words=0,
        candidate_dtype=torch.float32,
        state_tiles=0,
        grid=9,
        expected_abi_names=(
            "weight_desc",
            "x_desc",
            "logits_ptr",
            "topk_indices_ptr",
            "topk_weights_ptr",
            "mesh",
            "tokens",
        ),
        uses_mesh=True,
    ),
    _LeafDefinition(
        name="v400",
        module_name="sm90_softmax_h4096_e512_k10_t2048",
        weight_block=128,
        x_blocks=((64, None),),
        candidate_producers=4,
        candidate_words=10,
        candidate_dtype=torch.int32,
        state_tiles=32,
        grid=128,
        expected_abi_names=(
            "weight_desc",
            "x_desc",
            "logits_ptr",
            "candidate_scratch_ptr",
            "state_ptr",
            "topk_indices_ptr",
            "topk_weights_ptr",
            "tokens",
        ),
    ),
    _LeafDefinition(
        name="v412",
        module_name="sm90_softmax_h4096_e512_k10_t4107_t4108",
        weight_block=128,
        x_blocks=((128, None),),
        candidate_producers=4,
        candidate_words=10,
        candidate_dtype=torch.int32,
        state_tiles=33,
        grid=132,
        expected_abi_names=(
            "weight_desc",
            "x_desc",
            "logits_ptr",
            "candidate_scratch_ptr",
            "state_ptr",
            "topk_indices_ptr",
            "topk_weights_ptr",
            "tokens",
        ),
    ),
    _LeafDefinition(
        name="v414",
        module_name="sm90_softmax_h4096_e512_k10_t4434_t4435",
        weight_block=64,
        x_blocks=((64, None), (8, None)),
        candidate_producers=4,
        candidate_words=10,
        candidate_dtype=torch.int32,
        state_tiles=33,
        grid=132,
        expected_abi_names=(
            "weight_desc",
            "x64_desc",
            "x8_desc",
            "logits_ptr",
            "candidate_scratch_ptr",
            "state_ptr",
            "topk_indices_ptr",
            "topk_weights_ptr",
            "tokens",
        ),
    ),
    _LeafDefinition(
        name="v431",
        module_name="sm90_softmax_h4096_e512_k10_t5240",
        weight_block=128,
        x_blocks=((128, None), (32, None)),
        candidate_producers=4,
        candidate_words=10,
        candidate_dtype=torch.int32,
        state_tiles=33,
        grid=132,
        expected_abi_names=(
            "weight_desc",
            "x128_desc",
            "x32_desc",
            "logits_ptr",
            "candidate_scratch_ptr",
            "state_ptr",
            "topk_indices_ptr",
            "topk_weights_ptr",
            "tokens",
        ),
    ),
    _LeafDefinition(
        name="v394",
        module_name="sm90_softmax_h4096_e512_k10_t8214",
        weight_block=256,
        x_blocks=((128, None),),
        candidate_producers=2,
        candidate_words=10,
        candidate_dtype=torch.int32,
        state_tiles=65,
        grid=130,
        expected_abi_names=(
            "weight_desc",
            "x_desc",
            "logits_ptr",
            "candidate_scratch_ptr",
            "state_ptr",
            "topk_indices_ptr",
            "topk_weights_ptr",
            "tokens",
        ),
    ),
    _LeafDefinition(
        name="v430",
        module_name="sm90_softmax_h4096_e512_k10_t13421_t13422",
        weight_block=256,
        x_blocks=((128, None), (128, 80)),
        candidate_producers=2,
        candidate_words=10,
        candidate_dtype=torch.int32,
        state_tiles=65,
        grid=130,
        expected_abi_names=(
            "weight_desc",
            "x128_desc",
            "x80_desc",
            "logits_ptr",
            "candidate_scratch_ptr",
            "state_ptr",
            "topk_indices_ptr",
            "topk_weights_ptr",
            "tokens",
        ),
    ),
    _LeafDefinition(
        name="v415",
        module_name="sm90_softmax_h4096_e512_k10_t16384",
        weight_block=256,
        x_blocks=((128, None),),
        candidate_producers=2,
        candidate_words=40,
        candidate_dtype=torch.float32,
        state_tiles=64,
        grid=128,
        expected_abi_names=(
            "weight_desc",
            "x128_desc",
            "logits_ptr",
            "candidate_scratch_ptr",
            "state_ptr",
            "topk_indices_ptr",
            "topk_weights_ptr",
            "tokens",
        ),
    ),
)

_DEFINITION_BY_LEAF: Final = {
    definition.name: definition for definition in _LEAF_DEFINITIONS
}

# Public, immutable dispatch metadata.  Runtime module objects and callables are
# intentionally absent so importing this API remains CPU/static-test friendly.
DISPATCH: Final = MappingProxyType(
    {
        token: MappingProxyType(
            {
                "tle_leaf": leaf,
                "module": _DEFINITION_BY_LEAF[leaf].module_name,
                "kernel_count": 1,
            }
        )
        for token, leaf in TOKEN_TO_LEAF.items()
    }
)


@dataclass(frozen=True)
class _DescriptorTemplate:
    descriptor_type: type
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    block_shape: tuple[int, ...]
    padding: str
    stride_alignment: int | None
    logical_block_shape: tuple[int, ...] | None

    @classmethod
    def from_descriptor(cls, descriptor: Any) -> "_DescriptorTemplate":
        return cls(
            descriptor_type=type(descriptor),
            shape=tuple(int(value) for value in descriptor.shape),
            strides=tuple(int(value) for value in descriptor.strides),
            block_shape=tuple(int(value) for value in descriptor.block_shape),
            padding=str(getattr(descriptor, "padding", "zero")),
            stride_alignment=(
                int(descriptor.stride_alignment)
                if hasattr(descriptor, "stride_alignment")
                else None
            ),
            logical_block_shape=(
                tuple(int(value) for value in descriptor.logical_block_shape)
                if hasattr(descriptor, "logical_block_shape")
                else None
            ),
        )

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        return (
            self.descriptor_type,
            self.shape,
            self.strides,
            self.block_shape,
            self.padding,
            self.stride_alignment,
            self.logical_block_shape,
        )

    def bind(self, tensor: torch.Tensor) -> Any:
        kwargs = {
            "shape": list(self.shape),
            "strides": list(self.strides),
            "block_shape": list(self.block_shape),
            "padding": self.padding,
        }
        if self.stride_alignment is not None:
            kwargs["stride_alignment"] = self.stride_alignment
        descriptor = self.descriptor_type(tensor, **kwargs)
        if self.logical_block_shape is not None:
            descriptor.logical_block_shape = list(self.logical_block_shape)
        return descriptor


@dataclass(frozen=True)
class _TensorTemplate:
    shape: tuple[int, ...]
    dtype: torch.dtype

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "_TensorTemplate":
        return cls(tuple(int(value) for value in tensor.shape), tensor.dtype)

    def empty(self, device: torch.device) -> torch.Tensor:
        return torch.empty(self.shape, device=device, dtype=self.dtype)

    def zeros(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.shape, device=device, dtype=self.dtype)


@dataclass(frozen=True)
class _WorkspaceBlueprint:
    logits: _TensorTemplate
    candidate_scratch: _TensorTemplate
    state: _TensorTemplate


@dataclass(frozen=True)
class _CompiledLeaf:
    definition: _LeafDefinition
    tokens: int
    device_index: int
    module: Any
    grid: tuple[int, ...]
    runner: ExactGridUnboundCompiledRunner
    weight_descriptor_template: _DescriptorTemplate
    x_descriptor_templates: tuple[_DescriptorTemplate, ...]
    workspace_blueprint: _WorkspaceBlueprint
    mesh: Any | None
    abi_arg_names: tuple[str, ...]


@dataclass
class _Workspace:
    logits: torch.Tensor
    candidate_scratch: torch.Tensor
    state: torch.Tensor
    x_descriptor_entry: (
        tuple[
            tuple[int, tuple[int, ...], tuple[int, ...]],
            tuple[Any, ...],
        ]
        | None
    ) = None


def _load_leaf_module(definition: _LeafDefinition) -> Any:
    # Lazy import keeps CPU-only API/source tests independent of the TLE wheel.
    return importlib.import_module(f".kernels.{definition.module_name}", __package__)


def _allocator(size: int, alignment: int, stream: int) -> torch.Tensor:
    del alignment, stream
    return torch.empty(size, device="cuda", dtype=torch.uint8)


def _compiled_abi_arg_names(compiled: Any) -> tuple[str, ...]:
    try:
        names = tuple(str(name) for name in compiled.src.fn.arg_names)
    except AttributeError as error:
        raise MegaGateContextError(
            "compiled leaf does not expose the raw high-level ABI"
        ) from error
    if not names:
        raise MegaGateContextError("compiled leaf exposes an empty raw ABI")
    return names


class MegaGateContext(HostContext[_CompiledLeaf, _Workspace]):
    """Compiled exact leaves plus per-stream internal workspaces."""

    api_kind = "functional_unbound_eager_qwen_h4096_final50"
    kernel_count_per_call = 1
    final_scope = "qwen_h4096_e512_k10_eager12"

    def __init__(self) -> None:
        super().__init__()
        # One current weight storage may need E64/E128/E256 descriptors.  A
        # copy-on-write descriptor map retains all required variants without
        # retaining an unbounded history of old model weights.
        self._weight_descriptor_cache_entry: (
            tuple[tuple[int, int], dict[tuple[Any, ...], Any]] | None
        ) = None

    @property
    def prepared_dispatch(self) -> dict[int, str]:
        with self._lock:
            return {
                token: leaf.definition.name
                for token, leaf in sorted(self._compiled.items())
            }

    @property
    def prepared_abi(self) -> dict[int, tuple[str, ...]]:
        with self._lock:
            return {
                token: leaf.abi_arg_names
                for token, leaf in sorted(self._compiled.items())
            }

    @staticmethod
    def _workspace_from_blueprint(
        blueprint: _WorkspaceBlueprint,
        device: torch.device,
    ) -> _Workspace:
        return _Workspace(
            logits=blueprint.logits.empty(device),
            candidate_scratch=blueprint.candidate_scratch.empty(device),
            state=blueprint.state.zeros(device),
        )

    def _workspace(self, device: torch.device, tokens: int) -> _Workspace:
        leaf = self._compiled.get(tokens)
        if leaf is None:
            raise UnpreparedShapeError(
                f"T={tokens} was not prepared; call context.prepare first"
            )
        stream = torch.cuda.current_stream(device)
        key = self._workspace_key(device, tokens, stream=stream)
        return self._get_or_create_workspace(
            key,
            lambda: self._workspace_from_blueprint(leaf.workspace_blueprint, device),
        )

    @staticmethod
    def _weight_storage_key(gate_weight: torch.Tensor) -> tuple[int, int]:
        return (
            int(gate_weight.data_ptr()),
            (
                int(gate_weight.device.index)
                if gate_weight.device.index is not None
                else int(torch.cuda.current_device())
            ),
        )

    def _install_weight_descriptor(
        self,
        template: _DescriptorTemplate,
        gate_weight: torch.Tensor,
        descriptor: Any,
    ) -> None:
        storage_key = self._weight_storage_key(gate_weight)
        entry = self._weight_descriptor_cache_entry
        descriptor_map = (
            dict(entry[1]) if entry is not None and entry[0] == storage_key else {}
        )
        descriptor_map[template.fingerprint] = descriptor
        self._weight_descriptor_cache_entry = (storage_key, descriptor_map)

    def _weight_desc(
        self,
        leaf: _CompiledLeaf,
        gate_weight: torch.Tensor,
    ) -> Any:
        storage_key = self._weight_storage_key(gate_weight)
        fingerprint = leaf.weight_descriptor_template.fingerprint
        entry = self._weight_descriptor_cache_entry
        if entry is not None and entry[0] == storage_key:
            descriptor = entry[1].get(fingerprint)
            if descriptor is not None:
                return descriptor
        descriptor = leaf.weight_descriptor_template.bind(gate_weight)
        with self._lock:
            self._install_weight_descriptor(
                leaf.weight_descriptor_template, gate_weight, descriptor
            )
        return descriptor

    def _x_descs(
        self,
        leaf: _CompiledLeaf,
        workspace: _Workspace,
        x: torch.Tensor,
    ) -> tuple[Any, ...]:
        key = (
            int(x.data_ptr()),
            tuple(int(value) for value in x.shape),
            tuple(int(value) for value in x.stride()),
        )
        entry = workspace.x_descriptor_entry
        if entry is not None and entry[0] == key:
            return entry[1]
        descriptors = tuple(
            template.bind(x) for template in leaf.x_descriptor_templates
        )
        with self._lock:
            workspace.x_descriptor_entry = (key, descriptors)
        return descriptors

    def clear_cached_descriptors(self) -> None:
        """Release strong references retained by last-address descriptors."""

        with self._lock:
            for workspace in self._workspaces.values():
                workspace.x_descriptor_entry = None
            self._weight_descriptor_cache_entry = None

    def prepare(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        *,
        plan: KernelPlan,
        routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    ) -> Any:
        """Warm up and load one exact leaf without launching it."""

        require_megagate_tle()
        tokens, device = validate_prepare_inputs(
            _INPUT_CONTRACT,
            x,
            gate_weight,
            correction_bias,
            routed_scaling_factor,
        )
        definition = _DEFINITION_BY_LEAF[plan.source_leaf]
        module = _load_leaf_module(definition)
        with self._lock, torch.cuda.device(device):
            device_index = self._resolve_device_index(device)
            self._require_prepare_device(device_index)
            self._device_index = device_index
            existing = self._compiled.get(tokens)
            if existing is not None:
                stream = torch.cuda.current_stream(device)
                key = self._workspace_key(device, tokens, stream=stream)
                workspace = self._get_or_create_workspace(
                    key,
                    lambda: self._workspace_from_blueprint(
                        existing.workspace_blueprint, device
                    ),
                )
                self._weight_desc(existing, gate_weight)
                self._x_descs(existing, workspace, x)
                return existing.runner.compiled

            import triton
            from triton.tools.tensor_descriptor import TensorDescriptor

            triton.set_allocator(_allocator)
            grid = (definition.grid,)
            weight_descriptor = TensorDescriptor(
                gate_weight,
                shape=[EXPERTS, HIDDEN],
                strides=[HIDDEN, 1],
                block_shape=[definition.weight_block, 128],
            )
            x_descriptors = []
            for physical_tokens, logical_tokens in definition.x_blocks:
                descriptor = TensorDescriptor(
                    x,
                    shape=[tokens, HIDDEN],
                    strides=[HIDDEN, 1],
                    block_shape=[physical_tokens, 128],
                )
                if logical_tokens is not None:
                    descriptor.logical_block_shape = [logical_tokens, 128]
                x_descriptors.append(descriptor)

            logits = torch.empty((tokens, EXPERTS), device=device, dtype=torch.float32)
            if definition.candidate_producers:
                candidate_scratch = torch.empty(
                    (
                        tokens,
                        definition.candidate_producers,
                        definition.candidate_words,
                    ),
                    device=device,
                    dtype=definition.candidate_dtype,
                )
            else:
                candidate_scratch = torch.empty(
                    (0,), device=device, dtype=definition.candidate_dtype
                )
            state_tiles = definition.state_tiles
            state = torch.zeros(
                (state_tiles, 16) if state_tiles else (0,),
                device=device,
                dtype=torch.int64,
            )
            topk_weights = torch.empty(
                (tokens, TOPK), device=device, dtype=torch.float32
            )
            topk_indices = torch.empty((tokens, TOPK), device=device, dtype=torch.int32)
            mesh = module.MESH if definition.uses_mesh else None
            if mesh is not None:
                warmup_args = (
                    weight_descriptor,
                    x_descriptors[0],
                    logits,
                    topk_indices,
                    topk_weights,
                    mesh,
                    tokens,
                )
            else:
                warmup_args = (
                    weight_descriptor,
                    *x_descriptors,
                    logits,
                    candidate_scratch,
                    state,
                    topk_indices,
                    topk_weights,
                    tokens,
                )
            compiled = module.KERNEL.warmup(
                *warmup_args,
                grid=grid,
                num_ctas=1,
                num_warps=4,
                maxnreg=module.MAXNREG,
            )
            abi_arg_names = _compiled_abi_arg_names(compiled)
            if abi_arg_names != definition.expected_abi_names:
                raise MegaGateContextError(
                    f"{definition.name} raw ABI mismatch: expected "
                    f"{definition.expected_abi_names}, got {abi_arg_names}"
                )
            runner = ExactGridUnboundCompiledRunner(
                compiled=compiled,
                grid=grid,
                expected_argument_count=definition.expected_argument_count,
            )
            weight_template = _DescriptorTemplate.from_descriptor(weight_descriptor)
            x_templates = tuple(
                _DescriptorTemplate.from_descriptor(descriptor)
                for descriptor in x_descriptors
            )
            blueprint = _WorkspaceBlueprint(
                logits=_TensorTemplate.from_tensor(logits),
                candidate_scratch=_TensorTemplate.from_tensor(candidate_scratch),
                state=_TensorTemplate.from_tensor(state),
            )
            leaf = _CompiledLeaf(
                definition=definition,
                tokens=tokens,
                device_index=device_index,
                module=module,
                grid=grid,
                runner=runner,
                weight_descriptor_template=weight_template,
                x_descriptor_templates=x_templates,
                workspace_blueprint=blueprint,
                mesh=mesh,
                abi_arg_names=abi_arg_names,
            )

            stream = torch.cuda.current_stream(device)
            workspace_key = self._workspace_key(device, tokens, stream=stream)
            seed_workspace = _Workspace(
                logits=logits,
                candidate_scratch=candidate_scratch,
                state=state,
                x_descriptor_entry=(
                    (
                        int(x.data_ptr()),
                        tuple(int(value) for value in x.shape),
                        tuple(int(value) for value in x.stride()),
                    ),
                    tuple(x_descriptors),
                ),
            )
            self._workspaces[workspace_key] = seed_workspace
            self._last_workspace_entry = (workspace_key, seed_workspace)
            self._install_weight_descriptor(
                weight_template, gate_weight, weight_descriptor
            )
            # Publish last: a lock-free run can never see a leaf without its
            # initial per-stream storage and descriptor state.
            self._compiled[tokens] = leaf
            return compiled

    @staticmethod
    def _launch_args(
        leaf: _CompiledLeaf,
        workspace: _Workspace,
        weight_descriptor: Any,
        x_descriptors: tuple[Any, ...],
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> tuple[Any, ...]:
        if leaf.mesh is not None:
            if len(x_descriptors) != 1:
                raise MegaGateContextError("mesh leaf requires one x descriptor")
            return (
                weight_descriptor,
                x_descriptors[0],
                workspace.logits,
                topk_indices,
                topk_weights,
                leaf.mesh,
                leaf.tokens,
            )
        return (
            weight_descriptor,
            *x_descriptors,
            workspace.logits,
            workspace.candidate_scratch,
            workspace.state,
            topk_indices,
            topk_weights,
            leaf.tokens,
        )

    def run(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
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
        leaf, workspace, _, stream_handle = self._prepared_launch(tokens, device)
        weight_descriptor = self._weight_desc(leaf, gate_weight)
        x_descriptors = self._x_descs(leaf, workspace, x)
        topk_weights = torch.empty((tokens, TOPK), device=device, dtype=torch.float32)
        topk_indices = torch.empty((tokens, TOPK), device=device, dtype=torch.int32)
        launch_args = self._launch_args(
            leaf,
            workspace,
            weight_descriptor,
            x_descriptors,
            topk_weights,
            topk_indices,
        )
        leaf.runner[leaf.grid](*launch_args, stream=stream_handle)
        return topk_weights, topk_indices


def megagate(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None = None,
    *,
    context: MegaGateContext,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch and launch one exact final50 functional MegaGate leaf."""

    if not isinstance(context, MegaGateContext):
        raise TypeError("context must be MegaGateContext")
    return context.run(
        x,
        gate_weight,
        correction_bias,
        routed_scaling_factor=routed_scaling_factor,
    )


__all__ = [
    "DISPATCH",
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
    "megagate",
]
