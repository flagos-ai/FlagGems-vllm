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

"""Host-only input and workspace contracts shared by MegaGate profiles."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable, Generic, Protocol, TypeVar

import torch

from .errors import UnpreparedShapeError


@dataclass(frozen=True)
class InputContract:
    """Static tensor ABI shared by prepare and eager runtime validation."""

    name: str
    supported_tokens: tuple[int, ...]
    hidden: int
    experts: int
    routed_scaling_factor: float
    requires_correction_bias: bool
    supported_token_set: frozenset[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "supported_token_set", frozenset(self.supported_tokens)
        )


@dataclass(frozen=True)
class WorkspaceKey:
    """Per-device, per-stream, exact-token workspace identity."""

    device_index: int
    stream_handle: int
    tokens: int


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


def validate_prepare_inputs(
    contract: InputContract,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
) -> tuple[int, torch.device]:
    """Validate the full tensor ABI after the public API selects a plan."""

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"x must be rank two, got rank {x.ndim}")
    tokens = int(x.shape[0])
    if tokens not in contract.supported_token_set:
        raise ValueError(
            f"{contract.name} rejects T={tokens}; "
            f"supported={contract.supported_tokens}"
        )
    device = x.device
    if device.type != "cuda":
        raise ValueError(f"x must be CUDA, got {device}")
    _require_tensor(
        x,
        name="x",
        shape=(tokens, contract.hidden),
        dtype=torch.bfloat16,
        device=device,
    )
    _require_tensor(
        gate_weight,
        name="gate_weight",
        shape=(contract.experts, contract.hidden),
        dtype=torch.bfloat16,
        device=device,
    )
    if contract.requires_correction_bias:
        if correction_bias is None:
            raise ValueError(f"{contract.name} requires correction_bias")
        _require_tensor(
            correction_bias,
            name="correction_bias",
            shape=(contract.experts,),
            dtype=torch.float32,
            device=device,
        )
    elif correction_bias is not None:
        raise ValueError(f"{contract.name} does not accept correction_bias")
    scale = float(routed_scaling_factor)
    if not math.isfinite(scale) or scale != contract.routed_scaling_factor:
        raise ValueError(
            "routed_scaling_factor must be exactly " f"{contract.routed_scaling_factor}"
        )
    return tokens, device


def validate_runtime_inputs(
    contract: InputContract,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
) -> tuple[int, torch.device]:
    """Validate the hot ABI without repeating prepare-only architecture I/O."""

    if not isinstance(x, torch.Tensor) or not isinstance(gate_weight, torch.Tensor):
        raise TypeError("x and gate_weight must be torch.Tensor objects")
    if contract.requires_correction_bias:
        if not isinstance(correction_bias, torch.Tensor):
            raise TypeError("correction_bias must be a torch.Tensor")
    elif correction_bias is not None:
        raise ValueError(f"{contract.name} does not accept correction_bias")
    if x.ndim != 2:
        raise ValueError(f"x must be rank two, got rank {x.ndim}")
    tokens = int(x.shape[0])
    if tokens not in contract.supported_token_set:
        raise ValueError(
            f"{contract.name} rejects T={tokens}; "
            f"supported={contract.supported_tokens}"
        )
    device = x.device
    invalid = (
        tuple(x.shape) != (tokens, contract.hidden)
        or tuple(gate_weight.shape) != (contract.experts, contract.hidden)
        or x.dtype != torch.bfloat16
        or gate_weight.dtype != torch.bfloat16
        or device.type != "cuda"
        or gate_weight.device != device
        or not x.is_contiguous()
        or not gate_weight.is_contiguous()
    )
    if contract.requires_correction_bias:
        assert correction_bias is not None
        invalid = invalid or (
            tuple(correction_bias.shape) != (contract.experts,)
            or correction_bias.dtype != torch.float32
            or correction_bias.device != device
            or not correction_bias.is_contiguous()
        )
    if invalid:
        raise ValueError(f"inputs violate the {contract.name} ABI")
    scale = float(routed_scaling_factor)
    if not math.isfinite(scale) or scale != contract.routed_scaling_factor:
        raise ValueError(
            "routed_scaling_factor must be exactly " f"{contract.routed_scaling_factor}"
        )
    return tokens, device


class _DeviceBoundLeaf(Protocol):
    @property
    def device_index(self) -> int: ...


_Leaf = TypeVar("_Leaf", bound=_DeviceBoundLeaf)
_Workspace = TypeVar("_Workspace")


class HostContext(Generic[_Leaf, _Workspace]):
    """Own the compiled and per-stream state shared by the Hopper schedules."""

    def __init__(self) -> None:
        self._compiled: dict[int, _Leaf] = {}
        self._workspaces: dict[WorkspaceKey, _Workspace] = {}
        self._last_workspace_entry: tuple[WorkspaceKey, _Workspace] | None = None
        self._device_index: int | None = None
        self._lock = threading.RLock()

    @property
    def prepared_tokens(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._compiled))

    @staticmethod
    def _resolve_device_index(device: torch.device) -> int:
        if device.index is None:
            return int(torch.cuda.current_device())
        return int(device.index)

    def _require_prepare_device(self, device_index: int) -> None:
        if self._device_index is not None and self._device_index != device_index:
            raise ValueError(
                "MegaGateContext is single-device; construct one context per CUDA device"
            )

    def _workspace_key(
        self,
        device: torch.device,
        tokens: int,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> WorkspaceKey:
        current_stream = stream or torch.cuda.current_stream(device)
        return WorkspaceKey(
            self._resolve_device_index(device), int(current_stream.cuda_stream), tokens
        )

    def _get_or_create_workspace(
        self, key: WorkspaceKey, allocate: Callable[[], _Workspace]
    ) -> _Workspace:
        """Provision per-stream storage while preparing the schedule."""

        last_entry = self._last_workspace_entry
        if last_entry is not None and last_entry[0] == key:
            return last_entry[1]
        workspace = self._workspaces.get(key)
        if workspace is None:
            with self._lock:
                workspace = self._workspaces.get(key)
                if workspace is None:
                    workspace = allocate()
                    self._workspaces[key] = workspace
        self._last_workspace_entry = (key, workspace)
        return workspace

    def _prepared_launch(
        self, tokens: int, device: torch.device
    ) -> tuple[_Leaf, _Workspace, int, int]:
        """Get the prepared leaf and workspace without allocating or compiling."""

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
        stream_handle = int(torch.cuda.current_stream(device).cuda_stream)
        key = WorkspaceKey(device_index, stream_handle, tokens)
        last_entry = self._last_workspace_entry
        workspace = (
            last_entry[1]
            if last_entry is not None and last_entry[0] == key
            else self._workspaces.get(key)
        )
        if workspace is None:
            raise UnpreparedShapeError(
                f"T={tokens} was not prepared on the current CUDA stream; "
                "call context.prepare on every stream used for execution"
            )
        return leaf, workspace, device_index, stream_handle

    def clear_cached_descriptors(self) -> None:
        raise NotImplementedError

    def _clear_schedule_state(self) -> None:
        """Release any schedule-specific compile state."""

    def close(self) -> None:
        """Release compiled runners, descriptors, and per-stream workspaces."""

        with self._lock:
            self.clear_cached_descriptors()
            self._compiled.clear()
            self._workspaces.clear()
            self._last_workspace_entry = None
            self._device_index = None
            self._clear_schedule_state()


__all__ = [
    "HostContext",
    "InputContract",
    "WorkspaceKey",
    "validate_prepare_inputs",
    "validate_runtime_inputs",
]
