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
allocates fresh semantic outputs when ``out`` is omitted, and submits exactly
one fused kernel without CUDA Graph capture or replay.

Compilation and the SM90 architecture probe belong to :meth:`prepare`.
Steady-state calls perform an O(1) exact-T lookup and reuse only internal,
per-device/per-stream workspaces.  Tensor descriptors are rebound when their
backing storage changes.  The raw compiled runner receives the caller's
current stream explicitly; the old ``Plan.launch`` path is never used.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Final

import torch

from .kernels.runtime import ExactGridUnboundCompiledRunner

PROFILE: Final = "dsv4_flash"
SUPPORTED_TOKENS: Final = (
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
CLUSTERLESS_TOKENS: Final = (1028, 2048, 3076, 3805, 4100, 4835, 7884, 8221)
QUADMATH_TOKENS: Final = (13021, 16384)
HIDDEN: Final = 4096
EXPERTS: Final = 256
TOPK: Final = 6
ROUTED_SCALING_FACTOR: Final = 1.5
BLOCK_TOKENS: Final = 128
BLOCK_K: Final = 128


class MegaGateContextError(RuntimeError):
    """Base error for the functional DSV4 entry."""


class UnpreparedShapeError(MegaGateContextError):
    """Raised when an exact T or stream was not prepared before a call."""


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
_LEAF_SPEC_BY_TOKENS: Final = {
    **{tokens: _CLUSTERLESS_SPEC for tokens in CLUSTERLESS_TOKENS},
    **{tokens: _QUADMATH_SPEC for tokens in QUADMATH_TOKENS},
}
LEAF_IMPLEMENTATION_BY_TOKENS: Final = {
    tokens: spec.implementation for tokens, spec in _LEAF_SPEC_BY_TOKENS.items()
}


@dataclass(frozen=True)
class _CompiledLeaf:
    tokens: int
    implementation: str
    device_index: int
    grid: tuple[int, ...]
    runner: ExactGridUnboundCompiledRunner


@dataclass(frozen=True)
class _WorkspaceKey:
    device_index: int
    stream_handle: int
    tokens: int


@dataclass
class _Workspace:
    logits: torch.Tensor
    state: torch.Tensor | None
    x_descriptor_entry: (
        tuple[tuple[int, tuple[int, ...], tuple[int, ...]], Any] | None
    ) = None


def _load_leaves() -> dict[str, Any]:
    # Keep the public module importable in CPU-only source/AST tests.
    from .kernels import dsv4_clusterless as clusterless
    from .kernels import dsv4_quadmath as quadmath

    return {"clusterless": clusterless, "quadmath": quadmath}


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
            "MegaGate requires FlagTree frontend and compiler support for "
            "module-wide named-barrier allocation (upstream fix #1278)"
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
    correction_bias: torch.Tensor,
    routed_scaling_factor: float,
) -> tuple[int, torch.device]:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"x must be rank two, got rank {x.ndim}")
    tokens = int(x.shape[0])
    if tokens not in _LEAF_SPEC_BY_TOKENS:
        raise ValueError(
            f"DSV4 MegaGate rejects T={tokens}; supported={SUPPORTED_TOKENS}"
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
    if correction_bias is None:
        raise ValueError("DSV4 sqrt-softplus routing requires correction_bias")
    _require_tensor(
        correction_bias,
        name="correction_bias",
        shape=(EXPERTS,),
        dtype=torch.float32,
        device=device,
    )
    scale = float(routed_scaling_factor)
    if not math.isfinite(scale) or scale != ROUTED_SCALING_FACTOR:
        raise ValueError("routed_scaling_factor must be exactly 1.5")
    if tuple(torch.cuda.get_device_capability(device)) != (9, 0):
        raise RuntimeError("DSV4 MegaGate requires SM90")
    return tokens, device


def _validate_runtime_inputs(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    routed_scaling_factor: float,
) -> tuple[int, torch.device]:
    """Validate the hot ABI without repeating prepare-only architecture I/O."""

    if not isinstance(x, torch.Tensor) or not isinstance(gate_weight, torch.Tensor):
        raise TypeError("x and gate_weight must be torch.Tensor objects")
    if not isinstance(correction_bias, torch.Tensor):
        raise TypeError("correction_bias must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"x must be rank two, got rank {x.ndim}")
    tokens = int(x.shape[0])
    if tokens not in _LEAF_SPEC_BY_TOKENS:
        raise ValueError(
            f"DSV4 MegaGate rejects T={tokens}; supported={SUPPORTED_TOKENS}"
        )
    device = x.device
    if (
        tuple(x.shape) != (tokens, HIDDEN)
        or tuple(gate_weight.shape) != (EXPERTS, HIDDEN)
        or tuple(correction_bias.shape) != (EXPERTS,)
        or x.dtype != torch.bfloat16
        or gate_weight.dtype != torch.bfloat16
        or correction_bias.dtype != torch.float32
        or device.type != "cuda"
        or gate_weight.device != device
        or correction_bias.device != device
        or not x.is_contiguous()
        or not gate_weight.is_contiguous()
        or not correction_bias.is_contiguous()
    ):
        raise ValueError("x, gate_weight or correction_bias violates the DSV4 ABI")
    scale = float(routed_scaling_factor)
    if not math.isfinite(scale) or scale != ROUTED_SCALING_FACTOR:
        raise ValueError("routed_scaling_factor must be exactly 1.5")
    return tokens, device


def _validate_outputs(
    outputs: tuple[torch.Tensor, torch.Tensor],
    *,
    tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise TypeError("out must be a (topk_weights, topk_indices) tuple")
    topk_weights, topk_indices = outputs
    _require_tensor(
        topk_weights,
        name="topk_weights",
        shape=(tokens, TOPK),
        dtype=torch.float32,
        device=device,
    )
    _require_tensor(
        topk_indices,
        name="topk_indices",
        shape=(tokens, TOPK),
        dtype=torch.int64,
        device=device,
    )
    return topk_weights, topk_indices


class MegaGateContext:
    """Compiled exact-T leaves and per-device/per-stream internal storage."""

    api_kind = "functional_unbound_eager_dsv4"
    kernel_count_per_call = 1

    def __init__(self, *, profile: str = PROFILE) -> None:
        if profile != PROFILE:
            raise ValueError(
                f"unsupported MegaGate profile {profile!r}; expected {PROFILE!r}"
            )
        self.profile = profile
        self._compiled: dict[int, _CompiledLeaf] = {}
        self._workspaces: dict[_WorkspaceKey, _Workspace] = {}
        self._last_workspace_entry: tuple[_WorkspaceKey, _Workspace] | None = None
        # Retain one atomically published entry per leaf descriptor layout.
        # This keeps both descriptor layouts hot without retaining every prior
        # weight storage address for the lifetime of the context.
        self._weight_descriptor_entries: dict[
            int, tuple[tuple[int, int, int, int], Any] | None
        ] = {64: None, 256: None}
        self._modules: dict[str, Any] | None = None
        self._device_index: int | None = None
        self._lock = threading.RLock()

    @property
    def prepared_tokens(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._compiled))

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
        current_stream = stream or torch.cuda.current_stream(device)
        return _WorkspaceKey(
            self._resolve_device_index(device),
            int(current_stream.cuda_stream),
            tokens,
        )

    def _workspace_for_key(
        self,
        key: _WorkspaceKey,
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
                    spec = _LEAF_SPEC_BY_TOKENS[key.tokens]
                    token_tiles = (key.tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
                    state = (
                        torch.zeros(
                            (token_tiles, 16),
                            device=device,
                            dtype=torch.int64,
                        )
                        if spec.needs_state
                        else None
                    )
                    workspace = _Workspace(
                        logits=torch.empty(
                            (key.tokens, EXPERTS),
                            device=device,
                            dtype=torch.float32,
                        ),
                        state=state,
                    )
                    self._workspaces[key] = workspace
        self._last_workspace_entry = (key, workspace)
        return workspace

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

    def close(self) -> None:
        """Release compiled runners, descriptors, and per-stream workspaces."""

        with self._lock:
            self._compiled.clear()
            self._workspaces.clear()
            self._last_workspace_entry = None
            self._weight_descriptor_entries = {64: None, 256: None}
            self._modules = None
            self._device_index = None

    def prepare(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor,
        *,
        routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    ) -> None:
        """Prepare one exact DSV4 leaf and the current stream without launching."""

        _require_tle_capabilities()
        tokens, device = _validate_inputs(
            x, gate_weight, correction_bias, routed_scaling_factor
        )
        device_index = self._resolve_device_index(device)
        if int(torch.cuda.current_device()) != device_index:
            raise ValueError(
                "current CUDA device differs from the prepared tensor device"
            )
        modules = _load_leaves()
        spec = _LEAF_SPEC_BY_TOKENS[tokens]
        module = modules[spec.implementation]
        common = modules["clusterless"]._common
        with self._lock, torch.cuda.device(device):
            if self._device_index is not None and self._device_index != device_index:
                raise ValueError(
                    "MegaGateContext is single-device; construct one context "
                    "per CUDA device"
                )
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
                workspace = self._workspace_for_key(workspace_key, device=device)
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
                    implementation=spec.implementation,
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
        out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, device = _validate_runtime_inputs(
            x, gate_weight, correction_bias, routed_scaling_factor
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
        modules = self._modules
        if modules is None:
            raise UnpreparedShapeError("context module state was not prepared")
        spec = _LEAF_SPEC_BY_TOKENS[tokens]
        common = modules["clusterless"]._common

        stream = torch.cuda.current_stream(device)
        stream_handle = int(stream.cuda_stream)
        workspace_key = _WorkspaceKey(device_index, stream_handle, tokens)
        workspace = self._workspaces.get(workspace_key)
        if workspace is None:
            raise UnpreparedShapeError(
                f"T={tokens} was not prepared on the current CUDA stream; "
                "call context.prepare on every stream used for execution"
            )
        self._last_workspace_entry = (workspace_key, workspace)
        weight_desc = self._weight_desc(common, gate_weight, spec, device_index)
        x_desc = self._x_desc(common, workspace, x, tokens)
        if out is None:
            topk_weights = torch.empty(
                (tokens, TOPK), device=device, dtype=torch.float32
            )
            topk_indices = torch.empty((tokens, TOPK), device=device, dtype=torch.int64)
        else:
            topk_weights, topk_indices = _validate_outputs(
                out, tokens=tokens, device=device
            )

        if leaf.implementation == "clusterless":
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
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch one prepared DSV4 leaf as one eager, functional fused kernel."""

    if not isinstance(context, MegaGateContext):
        raise TypeError("context must be MegaGateContext")
    return context.run(
        x,
        gate_weight,
        correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        out=out,
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
    "TOPK",
    "UnpreparedShapeError",
    "megagate",
]
