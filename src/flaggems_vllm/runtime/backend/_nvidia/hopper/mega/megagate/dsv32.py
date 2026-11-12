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

import math
import threading
from dataclasses import dataclass
from typing import Any, Final

import torch

from .errors import MegaGateContextError, UnpreparedShapeError
from .kernels.runtime import ExactGridUnboundCompiledRunner

PROFILE: Final = "dsv32"
SUPPORTED_TOKENS: Final = (
    1028,
    1756,
    2048,
    3076,
    3805,
    8276,
    12292,
    12372,
    14305,
    16384,
)
V371_TOKENS: Final = SUPPORTED_TOKENS[:6]
V372_TOKENS: Final = SUPPORTED_TOKENS[6:8]
V435_TOKENS: Final = SUPPORTED_TOKENS[8:]
TOKEN_TO_LEAF: Final = {
    **{tokens: "v371" for tokens in V371_TOKENS},
    **{tokens: "v372" for tokens in V372_TOKENS},
    **{tokens: "v435" for tokens in V435_TOKENS},
}

HIDDEN: Final = 7168
EXPERTS: Final = 256
TOPK: Final = 8
ROUTED_SCALING_FACTOR: Final = 2.5
BLOCK_TOKENS: Final = 128
BLOCK_K: Final = 128
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


def _allocator(size: int, alignment: int, stream: int) -> torch.Tensor:
    del alignment, stream
    return torch.empty(
        size,
        device=torch.device("cuda", torch.cuda.current_device()),
        dtype=torch.uint8,
    )


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


def _load_dispatch_modules() -> dict[str, Any]:
    """Import TLE modules only when a CUDA preparation is requested."""

    from .kernels import dsv32_clusterless as clusterless
    from .kernels import dsv32_dualmath as dualmath
    from .kernels import dsv32_quadmath as quadmath

    return {
        "v371": clusterless,
        "v372": dualmath,
        "v435": quadmath,
    }


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
) -> tuple[int, torch.device]:
    """Full setup-time validation, including the architecture probe."""

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"x must be rank two, got rank {x.ndim}")
    tokens = int(x.shape[0])
    if tokens not in SUPPORTED_TOKENS:
        raise ValueError(
            f"functional DSV32 eager10 rejects T={tokens}; "
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
    if correction_bias is None:
        raise ValueError("DSV3.2 grouped-sigmoid routing requires correction_bias")
    _require_tensor(
        correction_bias,
        name="correction_bias",
        shape=(EXPERTS,),
        dtype=torch.float32,
        device=device,
    )
    scale = float(routed_scaling_factor)
    if not math.isfinite(scale) or scale != ROUTED_SCALING_FACTOR:
        raise ValueError("routed_scaling_factor must be exactly 2.5")
    if tuple(torch.cuda.get_device_capability(device)) != (9, 0):
        raise RuntimeError("functional DSV32 eager10 requires SM90")
    return tokens, device


def _validate_runtime_inputs(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
) -> tuple[int, torch.device]:
    """Straight-line eager ABI checks without repeating setup-time probes."""

    if (
        not isinstance(x, torch.Tensor)
        or not isinstance(gate_weight, torch.Tensor)
        or not isinstance(correction_bias, torch.Tensor)
    ):
        raise TypeError("x, gate_weight, and correction_bias must be tensors")
    if x.ndim != 2:
        raise ValueError(f"x must be rank two, got rank {x.ndim}")
    tokens = int(x.shape[0])
    if tokens not in SUPPORTED_TOKENS:
        raise ValueError(
            f"functional DSV32 eager10 rejects T={tokens}; "
            f"supported={SUPPORTED_TOKENS}"
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
        raise ValueError("inputs violate the functional DSV32 eager10 ABI")
    scale = float(routed_scaling_factor)
    if not math.isfinite(scale) or scale != ROUTED_SCALING_FACTOR:
        raise ValueError("routed_scaling_factor must be exactly 2.5")
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
        dtype=torch.int32,
        device=device,
    )
    return topk_weights, topk_indices


class MegaGateContext:
    """Compiled exact leaves plus per-device/per-stream internal workspaces."""

    api_kind = "functional_unbound_eager_dsv32"
    kernel_count_per_call = 1
    final_scope = "dsv32_h7168_e256_k8_eager10"

    def __init__(self, *, profile: str = PROFILE) -> None:
        if profile != PROFILE:
            raise ValueError(
                f"unsupported MegaGate profile {profile!r}; expected {PROFILE!r}"
            )
        self._compiled: dict[int, _CompiledLeaf] = {}
        self._workspaces: dict[_WorkspaceKey, _Workspace] = {}
        self._last_workspace_entry: tuple[_WorkspaceKey, _Workspace] | None = None
        self._weight_descriptor_entries: dict[
            int, tuple[tuple[int, int, int, int], Any]
        ] = {}
        self._descriptor_type: Any | None = None
        self._device_index: int | None = None
        self._lock = threading.RLock()

    @property
    def prepared_tokens(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._compiled))

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

    def _workspace_for_key(
        self,
        key: _WorkspaceKey,
        *,
        device: torch.device,
    ) -> _Workspace:
        # Cache hits perform only one immutable tuple read.  Miss publication
        # is serialized, and the key/value pair is replaced atomically as one
        # Python object so a lock-free reader cannot observe a torn pair.
        last_entry = self._last_workspace_entry
        if last_entry is not None and last_entry[0] == key:
            return last_entry[1]
        workspace = self._workspaces.get(key)
        if workspace is None:
            with self._lock:
                workspace = self._workspaces.get(key)
                if workspace is None:
                    token_tiles = (key.tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
                    workspace = _Workspace(
                        logits=torch.empty(
                            (key.tokens, EXPERTS),
                            device=device,
                            dtype=torch.bfloat16,
                        ),
                        state=(
                            None
                            if key.tokens in V435_TOKENS
                            else torch.zeros(
                                (token_tiles, STATE_WORDS_PER_TILE),
                                device=device,
                                dtype=torch.int64,
                            )
                        ),
                    )
                    self._workspaces[key] = workspace
        self._last_workspace_entry = (key, workspace)
        return workspace

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

    def clear_cached_input_descriptors(self) -> None:
        """Release strong references held by last-address descriptors."""

        with self._lock:
            for workspace in self._workspaces.values():
                workspace.x_descriptor_entry = None
            self._weight_descriptor_entries = {}

    def close(self) -> None:
        """Release all host and device objects owned by this context."""

        with self._lock:
            for workspace in self._workspaces.values():
                workspace.x_descriptor_entry = None
            self._compiled.clear()
            self._workspaces.clear()
            self._last_workspace_entry = None
            self._weight_descriptor_entries.clear()
            self._descriptor_type = None
            self._device_index = None

    def prepare(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor | None,
        *,
        routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    ) -> Any:
        """Compile one exact final-dispatch leaf without launching it."""

        tokens, device = _validate_inputs(
            x, gate_weight, correction_bias, routed_scaling_factor
        )
        _require_tle_capabilities()
        modules = _load_dispatch_modules()
        import triton
        from triton.tools.tensor_descriptor import TensorDescriptor

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
            self._descriptor_type = TensorDescriptor
            triton.set_allocator(_allocator)

            family = TOKEN_TO_LEAF[tokens]
            if family == "v371":
                module = modules[family]
                kernel = module._fused_e64_clusterless_kernel
                descriptor_experts = module.EXPERTS_PER_CTA
                logical_expert_ctas = module.LOGICAL_EXPERT_CTAS
                requires_state = True
                expected_argument_count = 9
            elif family == "v372":
                module = modules[family]
                kernel = module._fused_e128_dualmath_kernel
                descriptor_experts = module.EXPERTS_PER_MATH
                logical_expert_ctas = module.LOGICAL_EXPERT_CTAS
                requires_state = True
                expected_argument_count = 9
            else:
                module = modules[family]
                kernel = module._fused_e256_quadmath_kernel
                descriptor_experts = module.EXPERTS
                logical_expert_ctas = module.LOGICAL_EXPERT_CTAS
                requires_state = False
                expected_argument_count = 8

            stream = torch.cuda.current_stream(device)
            workspace_key = self._workspace_key(device, tokens, stream=stream)
            workspace = self._workspace_for_key(workspace_key, device=device)
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
        device_index = leaf.device_index if device.index is None else int(device.index)
        if leaf.device_index != device_index or self._device_index != device_index:
            raise ValueError("runtime tensor device differs from prepared context")
        if int(torch.cuda.current_device()) != device_index:
            raise ValueError(
                "current CUDA device differs from the prepared tensor device"
            )

        # Current storage addresses are rebound on every public call, but all
        # per-stream storage must already have been provisioned by prepare().
        # The measured path never allocates a workspace for a new stream.
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
        weight_desc = self._weight_desc(
            gate_weight,
            descriptor_experts=leaf.descriptor_experts,
            block_k=BLOCK_K,
            device_index=device_index,
        )
        x_desc = self._x_desc(workspace, x, tokens)
        if out is None:
            topk_weights = torch.empty(
                (tokens, TOPK), device=device, dtype=torch.float32
            )
            topk_indices = torch.empty((tokens, TOPK), device=device, dtype=torch.int32)
        else:
            topk_weights, topk_indices = _validate_outputs(
                out, tokens=tokens, device=device
            )

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
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch one exact-T functional DSV3.2 fused kernel without a Graph."""

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
