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
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import torch

from .errors import MegaGateContextError, UnpreparedShapeError
from .kernels.runtime import ExactGridUnboundCompiledRunner

PROFILE: Final = "qwen_h4096_e512_k10"
SUPPORTED_TOKENS: Final = (
    1035,
    1036,
    2048,
    4107,
    4108,
    4434,
    4435,
    5240,
    8214,
    13421,
    13422,
    16384,
)
HIDDEN: Final = 4096
EXPERTS: Final = 512
TOPK: Final = 10
ROUTED_SCALING_FACTOR: Final = 1.0


@dataclass(frozen=True)
class _LeafDefinition:
    name: str
    module_name: str
    tokens: tuple[int, ...]
    weight_block: int
    x_blocks: tuple[tuple[int, int | None], ...]
    candidate_producers: int
    candidate_words: int
    candidate_dtype: torch.dtype
    state_tiles: Mapping[int, int]
    grid: Mapping[int, int]
    expected_abi_names: tuple[str, ...]
    uses_mesh: bool = False

    @property
    def expected_argument_count(self) -> int:
        return len(self.expected_abi_names)


_LEAF_DEFINITIONS: Final = (
    _LeafDefinition(
        name="v371",
        module_name="qwen_h4096_e64_cluster8",
        tokens=(1035, 1036),
        weight_block=64,
        x_blocks=((128, None),),
        candidate_producers=0,
        candidate_words=0,
        candidate_dtype=torch.float32,
        state_tiles=MappingProxyType({1035: 0, 1036: 0}),
        grid=MappingProxyType({1035: 9, 1036: 9}),
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
        module_name="qwen_h4096_t2048",
        tokens=(2048,),
        weight_block=128,
        x_blocks=((64, None),),
        candidate_producers=4,
        candidate_words=10,
        candidate_dtype=torch.int32,
        state_tiles=MappingProxyType({2048: 32}),
        grid=MappingProxyType({2048: 128}),
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
        module_name="qwen_h4096_t410x",
        tokens=(4107, 4108),
        weight_block=128,
        x_blocks=((128, None),),
        candidate_producers=4,
        candidate_words=10,
        candidate_dtype=torch.int32,
        state_tiles=MappingProxyType({4107: 33, 4108: 33}),
        grid=MappingProxyType({4107: 132, 4108: 132}),
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
        module_name="qwen_h4096_t443x",
        tokens=(4434, 4435),
        weight_block=64,
        x_blocks=((64, None), (8, None)),
        candidate_producers=4,
        candidate_words=10,
        candidate_dtype=torch.int32,
        state_tiles=MappingProxyType({4434: 33, 4435: 33}),
        grid=MappingProxyType({4434: 132, 4435: 132}),
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
        module_name="qwen_h4096_t5240",
        tokens=(5240,),
        weight_block=128,
        x_blocks=((128, None), (32, None)),
        candidate_producers=4,
        candidate_words=10,
        candidate_dtype=torch.int32,
        state_tiles=MappingProxyType({5240: 33}),
        grid=MappingProxyType({5240: 132}),
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
        module_name="qwen_h4096_t8214",
        tokens=(8214,),
        weight_block=256,
        x_blocks=((128, None),),
        candidate_producers=2,
        candidate_words=10,
        candidate_dtype=torch.int32,
        state_tiles=MappingProxyType({8214: 65}),
        grid=MappingProxyType({8214: 130}),
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
        module_name="qwen_h4096_t1342x",
        tokens=(13421, 13422),
        weight_block=256,
        x_blocks=((128, None), (128, 80)),
        candidate_producers=2,
        candidate_words=10,
        candidate_dtype=torch.int32,
        state_tiles=MappingProxyType({13421: 65, 13422: 65}),
        grid=MappingProxyType({13421: 130, 13422: 130}),
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
        module_name="qwen_h4096_t16384",
        tokens=(16384,),
        weight_block=256,
        x_blocks=((128, None),),
        candidate_producers=2,
        candidate_words=40,
        candidate_dtype=torch.float32,
        state_tiles=MappingProxyType({16384: 64}),
        grid=MappingProxyType({16384: 128}),
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

_TOKEN_TO_DEFINITION: Final = {
    token: definition for definition in _LEAF_DEFINITIONS for token in definition.tokens
}
assert tuple(_TOKEN_TO_DEFINITION) == SUPPORTED_TOKENS

TOKEN_TO_LEAF: Final = MappingProxyType(
    {token: definition.name for token, definition in _TOKEN_TO_DEFINITION.items()}
)

# Public, immutable dispatch metadata.  Runtime module objects and callables are
# intentionally absent so importing this API remains CPU/static-test friendly.
DISPATCH: Final = MappingProxyType(
    {
        token: MappingProxyType(
            {
                "tle_leaf": definition.name,
                "module": definition.module_name,
                "kernel_count": 1,
            }
        )
        for token, definition in _TOKEN_TO_DEFINITION.items()
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


@dataclass(frozen=True)
class _WorkspaceKey:
    device_index: int
    stream_handle: int
    tokens: int


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


def _require_tle_capabilities() -> None:
    """Fail clearly when the installed TLE runtime lacks required SM90 APIs."""

    try:
        import triton.experimental.tle.language as tle
    except ImportError as error:
        raise RuntimeError("MegaGate requires Triton TLE support") from error

    from flaggems_vllm.utils.tle_capabilities import (
        supports_megagate_tle,
        supports_module_wide_named_barriers,
    )

    if not supports_megagate_tle(tle):
        raise RuntimeError(
            "MegaGate requires TLE pipe, warp-specialization, barrier, "
            "TMA, and WGMMA APIs"
        )
    if not supports_module_wide_named_barriers():
        raise RuntimeError(
            "MegaGate requires module-wide named-barrier allocation support"
        )


def _require_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_inputs(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
    *,
    prepare: bool,
) -> tuple[int, torch.device]:
    if correction_bias is not None:
        raise ValueError("Qwen softmax routing does not accept correction_bias")
    if not isinstance(x, torch.Tensor) or not isinstance(gate_weight, torch.Tensor):
        raise TypeError("x and gate_weight must be torch.Tensor objects")
    if x.ndim != 2:
        raise ValueError(f"x must be rank two, got rank {x.ndim}")
    tokens = int(x.shape[0])
    if tokens not in _TOKEN_TO_DEFINITION:
        raise ValueError(
            f"functional Qwen-H4096 rejects T={tokens}; "
            f"supported={SUPPORTED_TOKENS}"
        )
    device = x.device
    if device.type != "cuda":
        raise ValueError(f"x must be CUDA, got {device}")
    _require_tensor(
        x,
        name="x",
        shape=(tokens, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
    )
    _require_tensor(
        gate_weight,
        name="gate_weight",
        shape=(EXPERTS, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
    )
    scale = float(routed_scaling_factor)
    if not math.isfinite(scale) or scale != ROUTED_SCALING_FACTOR:
        raise ValueError("routed_scaling_factor must be exactly 1.0")
    if prepare and tuple(torch.cuda.get_device_capability(device)) != (9, 0):
        raise RuntimeError("functional Qwen-H4096 MegaGate requires SM90")
    return tokens, device


def _validate_runtime_inputs(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
) -> tuple[int, torch.device]:
    """Straight-line hot ABI checks without prepare-time architecture I/O."""

    if correction_bias is not None:
        raise ValueError("Qwen softmax routing does not accept correction_bias")
    if not isinstance(x, torch.Tensor) or not isinstance(gate_weight, torch.Tensor):
        raise TypeError("x and gate_weight must be torch.Tensor objects")
    if x.ndim != 2:
        raise ValueError(f"x must be rank two, got rank {x.ndim}")
    tokens = int(x.shape[0])
    if tokens not in _TOKEN_TO_DEFINITION:
        raise ValueError(
            f"functional Qwen-H4096 rejects T={tokens}; "
            f"supported={SUPPORTED_TOKENS}"
        )
    device = x.device
    if (
        tuple(x.shape) != (tokens, HIDDEN)
        or tuple(gate_weight.shape) != (EXPERTS, HIDDEN)
        or x.dtype != torch.bfloat16
        or gate_weight.dtype != torch.bfloat16
        or device.type != "cuda"
        or gate_weight.device != device
        or not x.is_contiguous()
        or not gate_weight.is_contiguous()
    ):
        raise ValueError("x or gate_weight violates the Qwen-H4096 ABI")
    scale = float(routed_scaling_factor)
    if not math.isfinite(scale) or scale != ROUTED_SCALING_FACTOR:
        raise ValueError("routed_scaling_factor must be exactly 1.0")
    return tokens, device


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


class MegaGateContext:
    """Compiled exact leaves plus per-stream internal workspaces."""

    api_kind = "functional_unbound_eager_qwen_h4096_final50"
    kernel_count_per_call = 1
    final_scope = "qwen_h4096_e512_k10_eager12"

    def __init__(self) -> None:
        self._compiled: dict[int, _CompiledLeaf] = {}
        self._workspaces: dict[_WorkspaceKey, _Workspace] = {}
        self._last_workspace_entry: tuple[_WorkspaceKey, _Workspace] | None = None
        # One current weight storage may need E64/E128/E256 descriptors.  A
        # copy-on-write descriptor map retains all required variants without
        # retaining an unbounded history of old model weights.
        self._weight_descriptor_cache_entry: (
            tuple[tuple[int, int], dict[tuple[Any, ...], Any]] | None
        ) = None
        self._device_index: int | None = None
        self._lock = threading.RLock()

    @property
    def prepared_tokens(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._compiled))

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
    def _resolve_device_index(device: torch.device) -> int:
        if device.index is not None:
            return int(device.index)
        return int(torch.cuda.current_device())

    def _workspace_key(
        self,
        device: torch.device,
        tokens: int,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> _WorkspaceKey:
        device_index = (
            torch.cuda.current_device() if device.index is None else int(device.index)
        )
        current_stream = stream or torch.cuda.current_stream(device)
        return _WorkspaceKey(device_index, int(current_stream.cuda_stream), tokens)

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

    def _workspace_for_key(
        self,
        key: _WorkspaceKey,
        leaf: _CompiledLeaf,
        *,
        device: torch.device,
    ) -> _Workspace:
        last_entry = self._last_workspace_entry
        if last_entry is not None and last_entry[0] == key:
            return last_entry[1]
        workspace = self._workspaces.get(key)
        if workspace is None:
            with self._lock:
                workspace = self._workspaces.get(key)
                if workspace is None:
                    workspace = self._workspace_from_blueprint(
                        leaf.workspace_blueprint, device
                    )
                    self._workspaces[key] = workspace
        self._last_workspace_entry = (key, workspace)
        return workspace

    def _workspace(self, device: torch.device, tokens: int) -> _Workspace:
        leaf = self._compiled.get(tokens)
        if leaf is None:
            raise UnpreparedShapeError(
                f"T={tokens} was not prepared; call context.prepare first"
            )
        stream = torch.cuda.current_stream(device)
        key = self._workspace_key(device, tokens, stream=stream)
        return self._workspace_for_key(key, leaf, device=device)

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

    def clear_cached_input_descriptors(self) -> None:
        """Release strong references retained by last-address descriptors."""

        with self._lock:
            for workspace in self._workspaces.values():
                workspace.x_descriptor_entry = None
            self._weight_descriptor_cache_entry = None

    def close(self) -> None:
        """Release all compiled leaves, workspaces, and tensor descriptors."""

        with self._lock:
            for workspace in self._workspaces.values():
                workspace.x_descriptor_entry = None
            self._compiled.clear()
            self._workspaces.clear()
            self._last_workspace_entry = None
            self._weight_descriptor_cache_entry = None
            self._device_index = None

    def prepare(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        *,
        routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    ) -> Any:
        """Warm up and load one exact leaf without launching it."""

        _require_tle_capabilities()
        tokens, device = _validate_inputs(
            x,
            gate_weight,
            correction_bias,
            routed_scaling_factor,
            prepare=True,
        )
        definition = _TOKEN_TO_DEFINITION[tokens]
        module = _load_leaf_module(definition)
        with self._lock, torch.cuda.device(device):
            device_index = (
                torch.cuda.current_device()
                if device.index is None
                else int(device.index)
            )
            if self._device_index is not None and self._device_index != device_index:
                raise ValueError(
                    "MegaGateContext is single-device; construct one context "
                    "per CUDA device"
                )
            self._device_index = device_index
            existing = self._compiled.get(tokens)
            if existing is not None:
                stream = torch.cuda.current_stream(device)
                key = self._workspace_key(device, tokens, stream=stream)
                workspace = self._workspace_for_key(key, existing, device=device)
                self._weight_desc(existing, gate_weight)
                self._x_descs(existing, workspace, x)
                return existing.runner.compiled

            import triton
            from triton.tools.tensor_descriptor import TensorDescriptor

            triton.set_allocator(_allocator)
            grid = (int(definition.grid[tokens]),)
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
            state_tiles = int(definition.state_tiles[tokens])
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
        tokens, device = _validate_runtime_inputs(
            x,
            gate_weight,
            correction_bias,
            routed_scaling_factor,
        )
        leaf = self._compiled.get(tokens)
        if leaf is None:
            raise UnpreparedShapeError(
                f"T={tokens} was not prepared; call context.prepare first"
            )
        device_index = self._resolve_device_index(device)
        if leaf.device_index != device_index or self._device_index != device_index:
            raise ValueError("runtime tensor device differs from prepared context")
        if int(torch.cuda.current_device()) != device_index:
            raise ValueError(
                "current CUDA device differs from the prepared tensor device"
            )

        stream = torch.cuda.current_stream(device)
        stream_handle = int(stream.cuda_stream)
        workspace_key = _WorkspaceKey(device_index, stream_handle, tokens)
        last_entry = self._last_workspace_entry
        workspace = (
            last_entry[1]
            if last_entry is not None and last_entry[0] == workspace_key
            else self._workspaces.get(workspace_key)
        )
        if workspace is None:
            raise UnpreparedShapeError(
                f"T={tokens} was not prepared on the current CUDA stream"
            )
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
