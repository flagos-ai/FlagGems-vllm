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

"""Functional no-Graph public MegaGate entry for Qwen-H2048 eager18.

The historical v375 dispatcher returns address-bound plans.  This module
keeps its exact 18-shape leaf mapping while exposing the operator lifecycle
required by FlagGems operator-mode benchmarking: JIT compilation is done by
``prepare``; every measured call performs exact-T host dispatch, binds the
current X and weight storage, allocates fresh semantic outputs when ``out``
is omitted, and submits exactly one fused kernel on the current CUDA stream.

Internal logits, tickets, candidate lists, and compiler global scratch are
owned per device, stream, and exact T.  The v364 lowering's global scratch is
preallocated per stream and passed to an unbound low-level runner, avoiding
both fixed user tensor addresses and a hot-path allocator round trip.  There
is no CUDA Graph, padding, fallback, helper kernel, or ``Plan.launch`` path.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Final

import torch

from .errors import MegaGateContextError, UnpreparedShapeError
from .kernels.qwen_h2048_runtime import (
    ExactGridUnboundPreallocatedScratchCompiledRunner,
)
from .kernels.runtime import ExactGridUnboundCompiledRunner

PROFILE: Final = "qwen_h2048_e256_k8"
SUPPORTED_TOKENS: Final = (
    1034,
    1035,
    1036,
    1352,
    2048,
    3104,
    4107,
    4108,
    4138,
    4434,
    4435,
    4436,
    9309,
    9330,
    11469,
    13421,
    13422,
    16384,
)
V366_TOKENS: Final = SUPPORTED_TOKENS[:6]
V365_TOKENS: Final = SUPPORTED_TOKENS[6:9]
V364_TOKENS: Final = SUPPORTED_TOKENS[9:12]
V363_TOKENS: Final = SUPPORTED_TOKENS[12:17]
V367_TOKENS: Final = SUPPORTED_TOKENS[17:]
TOKEN_TO_LEAF: Final = {
    **{tokens: "v366" for tokens in V366_TOKENS},
    **{tokens: "v365" for tokens in V365_TOKENS},
    **{tokens: "v364" for tokens in V364_TOKENS},
    **{tokens: "v363" for tokens in V363_TOKENS},
    **{tokens: "v367" for tokens in V367_TOKENS},
}

HIDDEN: Final = 2048
EXPERTS: Final = 256
TOPK: Final = 8
ROUTED_SCALING_FACTOR: Final = 1.0
BLOCK_K: Final = 128

assert len(SUPPORTED_TOKENS) == 18
assert tuple(sorted(SUPPORTED_TOKENS)) == SUPPORTED_TOKENS
assert set(TOKEN_TO_LEAF) == set(SUPPORTED_TOKENS)


@dataclass(frozen=True)
class _CompiledLeaf:
    tokens: int
    device_index: int
    family: str
    grid: tuple[int, ...]
    weight_block: int
    x_blocks: tuple[int, ...]
    constexpr_args: tuple[Any, ...]
    abi_arg_names: tuple[str, ...]
    runner: Any


@dataclass(frozen=True)
class _WorkspaceKey:
    device_index: int
    stream_handle: int
    tokens: int


@dataclass
class _Workspace:
    logits: torch.Tensor
    state: torch.Tensor | None
    candidate_scratch: torch.Tensor | None
    global_scratch: torch.Tensor | None = None
    x_descriptor_entries: dict[
        int, tuple[tuple[int, tuple[int, ...], tuple[int, ...]], Any]
    ] = field(default_factory=dict)


def _load_dispatch_modules() -> dict[str, Any]:
    # Lazy imports keep CPU-only source and AST tests independent of TLE.
    import triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    from .kernels import qwen_h2048_v363 as v363
    from .kernels import qwen_h2048_v364 as v364
    from .kernels import qwen_h2048_v365 as v365
    from .kernels import qwen_h2048_v366 as v366
    from .kernels import qwen_h2048_v367 as v367

    return {
        "descriptor_type": TensorDescriptor,
        "triton": triton,
        "v363": v363,
        "v364": v364,
        "v365": v365,
        "v366": v366,
        "v367": v367,
    }


def _allocator(size: int, alignment: int, stream: int) -> torch.Tensor:
    del alignment, stream
    return torch.empty(size, device="cuda", dtype=torch.uint8)


def _require_tle_capabilities() -> None:
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
) -> tuple[int, torch.device]:
    if correction_bias is not None:
        raise ValueError("Qwen softmax routing does not accept correction_bias")
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"x must be rank two, got rank {x.ndim}")
    tokens = int(x.shape[0])
    if tokens not in TOKEN_TO_LEAF:
        raise ValueError(
            f"functional Qwen-H2048 eager18 rejects T={tokens}; "
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
    if tuple(torch.cuda.get_device_capability(device)) != (9, 0):
        raise RuntimeError("functional Qwen-H2048 eager18 requires SM90")
    return tokens, device


def _validate_runtime_inputs(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
) -> tuple[int, torch.device]:
    """Straight-line hot ABI validation without prepare-only probes."""

    if correction_bias is not None:
        raise ValueError("Qwen softmax routing does not accept correction_bias")
    if not isinstance(x, torch.Tensor) or not isinstance(gate_weight, torch.Tensor):
        raise TypeError("x and gate_weight must be torch.Tensor objects")
    if x.ndim != 2:
        raise ValueError(f"x must be rank two, got rank {x.ndim}")
    tokens = int(x.shape[0])
    if tokens not in TOKEN_TO_LEAF:
        raise ValueError(
            f"functional Qwen-H2048 eager18 rejects T={tokens}; "
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
        raise ValueError("inputs violate the functional Qwen-H2048 eager18 ABI")
    scale = float(routed_scaling_factor)
    if not math.isfinite(scale) or scale != ROUTED_SCALING_FACTOR:
        raise ValueError("routed_scaling_factor must be exactly 1.0")
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
    """Exact compiled leaves plus per-device/per-stream internal storage."""

    api_kind = "functional_unbound_eager_qwen_h2048_dispatch"
    kernel_count_per_call = 1
    final_scope = "qwen_h2048_e256_k8_eager18"

    def __init__(self) -> None:
        self._compiled: dict[int, _CompiledLeaf] = {}
        self._workspaces: dict[_WorkspaceKey, _Workspace] = {}
        self._last_workspace_entry: tuple[_WorkspaceKey, _Workspace] | None = None
        self._weight_descriptor_entries: dict[
            int, tuple[tuple[int, int, int, int], Any]
        ] = {}
        self._descriptor_type: Any | None = None
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
                    family = TOKEN_TO_LEAF[key.tokens]
                    if family == "v364":
                        block_tokens = 80
                        state_dtype = torch.int32
                        candidate_shape = (key.tokens, 2, 32)
                    elif family == "v365":
                        block_tokens = 128
                        state_dtype = torch.int64
                        candidate_shape = (key.tokens, 4, 16)
                    elif family == "v366":
                        block_tokens = 128
                        state_dtype = torch.int64
                        candidate_shape = None
                    else:
                        block_tokens = 128
                        state_dtype = None
                        candidate_shape = None
                    token_tiles = (key.tokens + block_tokens - 1) // block_tokens
                    workspace = _Workspace(
                        logits=torch.empty(
                            (key.tokens, EXPERTS),
                            device=device,
                            dtype=torch.float32,
                        ),
                        state=(
                            torch.zeros(
                                (token_tiles, 16),
                                device=device,
                                dtype=state_dtype,
                            )
                            if state_dtype is not None
                            else None
                        ),
                        candidate_scratch=(
                            torch.empty(
                                candidate_shape,
                                device=device,
                                dtype=torch.float32,
                            )
                            if candidate_shape is not None
                            else None
                        ),
                    )
                    self._workspaces[key] = workspace
        self._last_workspace_entry = (key, workspace)
        return workspace

    def _weight_desc(
        self,
        gate_weight: torch.Tensor,
        *,
        block_experts: int,
        device_index: int,
    ) -> Any:
        descriptor_type = self._descriptor_type
        if descriptor_type is None:
            raise UnpreparedShapeError("context descriptor state was not prepared")
        key = (
            int(gate_weight.data_ptr()),
            device_index,
            block_experts,
            BLOCK_K,
        )
        entry = self._weight_descriptor_entries.get(block_experts)
        if entry is not None and entry[0] == key:
            return entry[1]
        descriptor = descriptor_type(
            gate_weight,
            shape=[EXPERTS, HIDDEN],
            strides=[HIDDEN, 1],
            block_shape=[block_experts, BLOCK_K],
        )
        with self._lock:
            self._weight_descriptor_entries[block_experts] = (key, descriptor)
        return descriptor

    def _x_desc(
        self,
        workspace: _Workspace,
        x: torch.Tensor,
        *,
        tokens: int,
        block_tokens: int,
    ) -> Any:
        descriptor_type = self._descriptor_type
        if descriptor_type is None:
            raise UnpreparedShapeError("context descriptor state was not prepared")
        key = (
            int(x.data_ptr()),
            tuple(int(value) for value in x.shape),
            tuple(int(value) for value in x.stride()),
        )
        entry = workspace.x_descriptor_entries.get(block_tokens)
        if entry is not None and entry[0] == key:
            return entry[1]
        descriptor = descriptor_type(
            x,
            shape=[tokens, HIDDEN],
            strides=[HIDDEN, 1],
            block_shape=[block_tokens, BLOCK_K],
        )
        with self._lock:
            workspace.x_descriptor_entries[block_tokens] = (key, descriptor)
        return descriptor

    def _ensure_global_scratch(
        self,
        workspace: _Workspace,
        leaf: _CompiledLeaf,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        scratch = workspace.global_scratch
        if scratch is not None:
            return scratch
        required = int(leaf.runner.required_global_scratch_bytes)
        with self._lock:
            scratch = workspace.global_scratch
            if scratch is None:
                scratch = torch.empty((required,), device=device, dtype=torch.uint8)
                workspace.global_scratch = scratch
        return scratch

    def clear_cached_input_descriptors(self) -> None:
        """Release strong references retained by last-address descriptors."""

        with self._lock:
            for workspace in self._workspaces.values():
                workspace.x_descriptor_entries.clear()
            self._weight_descriptor_entries.clear()

    def close(self) -> None:
        """Release all compiled leaves, workspaces, and tensor descriptors."""

        with self._lock:
            for workspace in self._workspaces.values():
                workspace.x_descriptor_entries.clear()
                workspace.global_scratch = None
            self._compiled.clear()
            self._workspaces.clear()
            self._last_workspace_entry = None
            self._weight_descriptor_entries.clear()
            self._descriptor_type = None
            self._modules = None
            self._device_index = None

    def prepare(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        *,
        routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    ) -> Any:
        """Compile one exact final-dispatch leaf without launching it."""

        tokens, device = _validate_inputs(
            x, gate_weight, correction_bias, routed_scaling_factor
        )
        _require_tle_capabilities()
        modules = _load_dispatch_modules()
        family = TOKEN_TO_LEAF[tokens]
        module = modules[family]
        with self._lock, torch.cuda.device(device):
            device_index = self._resolve_device_index(device)
            if self._device_index is not None and self._device_index != device_index:
                raise ValueError(
                    "MegaGateContext is single-device; construct one context "
                    "per CUDA device"
                )
            self._device_index = device_index
            self._modules = modules
            self._descriptor_type = modules["descriptor_type"]
            modules["triton"].set_allocator(_allocator)

            stream = torch.cuda.current_stream(device)
            workspace_key = self._workspace_key(device, tokens, stream=stream)
            workspace = self._workspace_for_key(workspace_key, device=device)
            existing = self._compiled.get(tokens)
            if existing is not None:
                if isinstance(
                    existing.runner,
                    ExactGridUnboundPreallocatedScratchCompiledRunner,
                ):
                    # Compilation is shared across streams, but global scratch
                    # is deliberately stream-local.  A prepare call on a new
                    # stream must provision it before the first timed run.
                    self._ensure_global_scratch(
                        workspace,
                        existing,
                        device=device,
                    )
                return existing.runner.compiled

            topk_weights = torch.empty(
                (tokens, TOPK), device=device, dtype=torch.float32
            )
            topk_indices = torch.empty((tokens, TOPK), device=device, dtype=torch.int32)
            constexpr_args: tuple[Any, ...] = ()

            if family == "v366":
                weight_block = module.EXPERTS_PER_CTA
                x_blocks = (module.BLOCK_TOKENS,)
                weight_desc = self._weight_desc(
                    gate_weight,
                    block_experts=weight_block,
                    device_index=device_index,
                )
                x_desc = self._x_desc(
                    workspace,
                    x,
                    tokens=tokens,
                    block_tokens=x_blocks[0],
                )
                state = workspace.state
                if state is None:
                    raise RuntimeError("v366 workspace has no ticket state")
                token_tiles = (tokens + module.BLOCK_TOKENS - 1) // module.BLOCK_TOKENS
                grid = (token_tiles,)
                kernel = (
                    module._fused_qwen_h2048_v014c_direct_kernel
                    if tokens in module.DIRECT_TOKENS
                    else module._fused_qwen_h2048_v014c_pipeline_kernel
                )
                compiled = kernel.warmup(
                    weight_desc,
                    x_desc,
                    workspace.logits,
                    state,
                    topk_indices,
                    topk_weights,
                    tokens,
                    module.CLUSTER4_MESH,
                    grid=grid,
                    num_ctas=1,
                    num_warps=4,
                    maxnreg=module.KERNEL_MAXNREG,
                )
                expected_argument_count = 8
            elif family == "v365":
                weight_block = module.EXPERTS_PER_CTA
                x_blocks = (module.BLOCK_TOKENS,)
                weight_desc = self._weight_desc(
                    gate_weight,
                    block_experts=weight_block,
                    device_index=device_index,
                )
                x_desc = self._x_desc(
                    workspace,
                    x,
                    tokens=tokens,
                    block_tokens=x_blocks[0],
                )
                if workspace.state is None or workspace.candidate_scratch is None:
                    raise RuntimeError("v365 workspace is incomplete")
                token_tiles = (tokens + module.BLOCK_TOKENS - 1) // module.BLOCK_TOKENS
                grid = (token_tiles * module.LOGICAL_EXPERT_CTAS,)
                compiled = module._fused_qwen_h2048_v018_v036_single_list_kernel.warmup(
                    weight_desc,
                    x_desc,
                    workspace.logits,
                    workspace.candidate_scratch,
                    workspace.state,
                    topk_indices,
                    topk_weights,
                    tokens,
                    grid=grid,
                    num_ctas=1,
                    num_warps=4,
                    maxnreg=module.KERNEL_MAXNREG,
                )
                expected_argument_count = 8
            elif family == "v364":
                weight_block = module.EXPERTS_PER_CTA
                x_blocks = ()
                weight_desc = self._weight_desc(
                    gate_weight,
                    block_experts=weight_block,
                    device_index=device_index,
                )
                if workspace.state is None or workspace.candidate_scratch is None:
                    raise RuntimeError("v364 workspace is incomplete")
                token_tiles = (tokens + module.BLOCK_TOKENS - 1) // module.BLOCK_TOKENS
                grid = (token_tiles * module.LOGICAL_EXPERT_CTAS,)
                compiled = module._fused_qwen_h2048_b80_e128_clusterless_kernel.warmup(
                    weight_desc,
                    x,
                    workspace.logits,
                    workspace.candidate_scratch,
                    workspace.state,
                    topk_indices,
                    topk_weights,
                    tokens,
                    grid=grid,
                    num_ctas=1,
                    num_warps=4,
                    maxnreg=module.KERNEL_MAXNREG,
                )
                expected_argument_count = 8
            elif family == "v363":
                weight_block = module.EXPERTS
                block_tokens = module.TOKEN_TO_BLOCK[tokens]
                frag0, frag1, frag2, num_fragments = module.BLOCK_TO_FRAGMENTS[
                    block_tokens
                ]
                x_blocks = (frag0, frag1, frag2)
                weight_desc = self._weight_desc(
                    gate_weight,
                    block_experts=weight_block,
                    device_index=device_index,
                )
                x_descs = tuple(
                    self._x_desc(
                        workspace,
                        x,
                        tokens=tokens,
                        block_tokens=fragment,
                    )
                    for fragment in x_blocks
                )
                grid = ((tokens + block_tokens - 1) // block_tokens,)
                compiled = module._fused_qwen_h2048_e256_direct_finalize_kernel.warmup(
                    weight_desc,
                    *x_descs,
                    workspace.logits,
                    topk_indices,
                    topk_weights,
                    tokens,
                    BLOCK_TOKENS=block_tokens,
                    FRAG0=frag0,
                    FRAG1=frag1,
                    FRAG2=frag2,
                    NUM_FRAGMENTS=num_fragments,
                    grid=grid,
                    num_ctas=1,
                    num_warps=4,
                    maxnreg=module.KERNEL_MAXNREG,
                )
                # CompiledKernel's raw callable retains tl.constexpr entries
                # in its high-level positional ABI.  Keep the exact values
                # selected during warmup and pass them on every unbound call.
                constexpr_args = (
                    block_tokens,
                    frag0,
                    frag1,
                    frag2,
                    num_fragments,
                )
                expected_argument_count = 13
            else:
                weight_block = module.EXPERTS
                x_blocks = (module.BLOCK_TOKENS,)
                weight_desc = self._weight_desc(
                    gate_weight,
                    block_experts=weight_block,
                    device_index=device_index,
                )
                x_desc = self._x_desc(
                    workspace,
                    x,
                    tokens=tokens,
                    block_tokens=x_blocks[0],
                )
                grid = (tokens // module.BLOCK_TOKENS,)
                compiled = module._fused_t16384_b128_kernel.warmup(
                    weight_desc,
                    x_desc,
                    workspace.logits,
                    topk_indices,
                    topk_weights,
                    grid=grid,
                    num_ctas=1,
                    num_warps=4,
                    maxnreg=module.KERNEL_MAXNREG,
                )
                expected_argument_count = 5

            abi_arg_names = _compiled_abi_arg_names(compiled)
            if len(abi_arg_names) != expected_argument_count:
                raise MegaGateContextError(
                    f"{family} raw ABI length mismatch: expected "
                    f"{expected_argument_count}, got {abi_arg_names}"
                )
            if family == "v363":
                expected_v363_names = (
                    "weight_desc",
                    "x0_desc",
                    "x1_desc",
                    "x2_desc",
                    "logits_ptr",
                    "topk_indices_ptr",
                    "topk_weights_ptr",
                    "TOKENS",
                    "BLOCK_TOKENS",
                    "FRAG0",
                    "FRAG1",
                    "FRAG2",
                    "NUM_FRAGMENTS",
                )
                if abi_arg_names != expected_v363_names:
                    raise MegaGateContextError(
                        "v363 raw ABI mismatch: expected "
                        f"{expected_v363_names}, got {abi_arg_names}"
                    )

            if family == "v364" and int(compiled.metadata.global_scratch_size):
                runner: Any = ExactGridUnboundPreallocatedScratchCompiledRunner(
                    compiled=compiled,
                    grid=grid,
                    expected_argument_count=expected_argument_count,
                )
            else:
                runner = ExactGridUnboundCompiledRunner(
                    compiled=compiled,
                    grid=grid,
                    expected_argument_count=expected_argument_count,
                )
            leaf = _CompiledLeaf(
                tokens=tokens,
                device_index=device_index,
                family=family,
                grid=grid,
                weight_block=weight_block,
                x_blocks=x_blocks,
                constexpr_args=constexpr_args,
                abi_arg_names=abi_arg_names,
                runner=runner,
            )
            self._compiled[tokens] = leaf
            if isinstance(runner, ExactGridUnboundPreallocatedScratchCompiledRunner):
                self._ensure_global_scratch(workspace, leaf, device=device)
            return compiled

    def run(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
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
            block_experts=leaf.weight_block,
            device_index=device_index,
        )
        if out is None:
            topk_weights = torch.empty(
                (tokens, TOPK), device=device, dtype=torch.float32
            )
            topk_indices = torch.empty((tokens, TOPK), device=device, dtype=torch.int32)
        else:
            topk_weights, topk_indices = _validate_outputs(
                out, tokens=tokens, device=device
            )

        if leaf.family == "v366":
            state = workspace.state
            if state is None:
                raise RuntimeError("v366 workspace has no ticket state")
            x_desc = self._x_desc(
                workspace,
                x,
                tokens=tokens,
                block_tokens=leaf.x_blocks[0],
            )
            launch_args = (
                weight_desc,
                x_desc,
                workspace.logits,
                state,
                topk_indices,
                topk_weights,
                tokens,
                modules["v366"].CLUSTER4_MESH,
            )
        elif leaf.family == "v365":
            if workspace.state is None or workspace.candidate_scratch is None:
                raise RuntimeError("v365 workspace is incomplete")
            x_desc = self._x_desc(
                workspace,
                x,
                tokens=tokens,
                block_tokens=leaf.x_blocks[0],
            )
            launch_args = (
                weight_desc,
                x_desc,
                workspace.logits,
                workspace.candidate_scratch,
                workspace.state,
                topk_indices,
                topk_weights,
                tokens,
            )
        elif leaf.family == "v364":
            if workspace.state is None or workspace.candidate_scratch is None:
                raise RuntimeError("v364 workspace is incomplete")
            launch_args = (
                weight_desc,
                x,
                workspace.logits,
                workspace.candidate_scratch,
                workspace.state,
                topk_indices,
                topk_weights,
                tokens,
            )
        elif leaf.family == "v363":
            x_descs = tuple(
                self._x_desc(
                    workspace,
                    x,
                    tokens=tokens,
                    block_tokens=fragment,
                )
                for fragment in leaf.x_blocks
            )
            launch_args = (
                weight_desc,
                *x_descs,
                workspace.logits,
                topk_indices,
                topk_weights,
                tokens,
            )
        else:
            x_desc = self._x_desc(
                workspace,
                x,
                tokens=tokens,
                block_tokens=leaf.x_blocks[0],
            )
            launch_args = (
                weight_desc,
                x_desc,
                workspace.logits,
                topk_indices,
                topk_weights,
            )

        if leaf.constexpr_args:
            launch_args += leaf.constexpr_args

        if isinstance(leaf.runner, ExactGridUnboundPreallocatedScratchCompiledRunner):
            global_scratch = workspace.global_scratch
            if global_scratch is None:
                raise UnpreparedShapeError(
                    f"T={tokens} global scratch was not prepared on the current stream"
                )
            leaf.runner[leaf.grid](
                *launch_args,
                global_scratch=global_scratch,
                stream=stream_handle,
            )
        else:
            leaf.runner[leaf.grid](*launch_args, stream=stream_handle)
        return topk_weights, topk_indices


def megagate(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None = None,
    *,
    context: MegaGateContext,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch one exact-T functional Qwen-H2048 fused kernel without Graph."""

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
    "V363_TOKENS",
    "V364_TOKENS",
    "V365_TOKENS",
    "V366_TOKENS",
    "V367_TOKENS",
    "megagate",
]
