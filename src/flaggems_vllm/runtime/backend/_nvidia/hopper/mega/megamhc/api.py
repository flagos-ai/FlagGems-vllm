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

"""Host contract for the exact-profile Hopper MegaMHC implementation.

This module intentionally contains no kernel implementation. It freezes the
public allocation, validation, static-binding, and stream-local workspace
contract used by the prepared TLE launcher factory.
"""

from __future__ import annotations

import operator
import threading
from dataclasses import dataclass
from typing import FrozenSet, NoReturn, Optional, Protocol, Tuple

import torch


@dataclass(frozen=True)
class MegaMHCProfile:
    """A fully specified model and numerical profile admitted by one kernel set."""

    model_name: str
    model_revision: str
    mode: str
    hidden_size: int
    residual_streams: int
    projection_outputs: int
    rms_eps: float
    hc_pre_eps: float
    hc_sinkhorn_eps: float
    hc_post_mult_value: float
    sinkhorn_repeat: int
    norm_eps: float
    norm_scale: float


DEEPSEEK_V4_FLASH_PROFILE = MegaMHCProfile(
    model_name="deepseek-ai/DeepSeek-V4-Flash",
    model_revision="60d8d70770c6776ff598c94bb586a859a38244f1",
    mode="normal",
    hidden_size=4096,
    residual_streams=4,
    projection_outputs=24,
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
    norm_eps=1e-6,
    norm_scale=1.0,
)


@dataclass(frozen=True)
class MegaMHCTensorBinding:
    """Storage and metadata contract for one layer-static tensor.

    Tensor contents may change in place. Object, storage, pointer and layout
    changes require rebuilding the compiled context because some launchers
    retain TensorMap descriptors whose base address is fixed at preparation.
    """

    object_id: int
    data_ptr: int
    storage_base: int
    storage_nbytes: int
    storage_offset: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "MegaMHCTensorBinding":
        storage = tensor.untyped_storage()
        return cls(
            object_id=id(tensor),
            data_ptr=int(tensor.data_ptr()),
            storage_base=int(storage.data_ptr()),
            storage_nbytes=int(storage.nbytes()),
            storage_offset=int(tensor.storage_offset()),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=bool(tensor.requires_grad),
        )

    def matches(self, tensor: torch.Tensor) -> bool:
        """Compare the live tensor without allocating another binding."""
        storage = tensor.untyped_storage()
        return bool(
            self.object_id == id(tensor)
            and self.data_ptr == int(tensor.data_ptr())
            and self.storage_base == int(storage.data_ptr())
            and self.storage_nbytes == int(storage.nbytes())
            and self.storage_offset == int(tensor.storage_offset())
            and self.shape == tensor.shape
            and self.stride == tensor.stride()
            and self.dtype == tensor.dtype
            and self.device == tensor.device
            and self.requires_grad == bool(tensor.requires_grad)
        )


@dataclass(frozen=True)
class MegaMHCStaticBinding:
    """Identity of layer-static tensors captured by the compiled launcher."""

    fn: MegaMHCTensorBinding
    hc_scale: MegaMHCTensorBinding
    hc_base: MegaMHCTensorBinding
    norm_weight: MegaMHCTensorBinding

    @classmethod
    def from_tensors(
        cls,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
    ) -> "MegaMHCStaticBinding":
        return cls(
            fn=MegaMHCTensorBinding.from_tensor(fn),
            hc_scale=MegaMHCTensorBinding.from_tensor(hc_scale),
            hc_base=MegaMHCTensorBinding.from_tensor(hc_base),
            norm_weight=MegaMHCTensorBinding.from_tensor(norm_weight),
        )

    def matches_tensors(
        self,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
    ) -> bool:
        return bool(
            self.fn.matches(fn)
            and self.hc_scale.matches(hc_scale)
            and self.hc_base.matches(hc_base)
            and self.norm_weight.matches(norm_weight)
        )


@dataclass
class MegaMHCWorkspace:
    """One reusable workspace lease owned by one ordered CUDA stream.

    ``state`` is kernel-family specific. It may hold partial buffers, monotonic
    counters, TensorMap storage, or other launcher-owned objects. A failed or
    interrupted cooperative launch must mark the lease ``poisoned``; poisoned
    state is never reset in the hot path and must be rebuilt by the caller.
    """

    device: torch.device
    stream_handle: Optional[int]
    state: object
    poisoned: bool = False


class MegaMHCLauncher(Protocol):
    """Address-rebindable one-kernel launcher bound to static model tensors."""

    @property
    def supported_tokens(self) -> FrozenSet[int]: ...

    @property
    def static_alignments(self) -> Tuple[int, int, int, int]: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def stream_handle(self) -> Optional[int]: ...

    @property
    def owner_thread_id(self) -> int: ...

    def dynamic_alignments(self, tokens: int) -> Tuple[int, ...]: ...

    def launch(
        self,
        workspace: MegaMHCWorkspace,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        new_residual: torch.Tensor,
        new_post_mix: torch.Tensor,
        new_comb_res_mix: torch.Tensor,
        y_bf16: torch.Tensor,
    ) -> None: ...


@dataclass(frozen=True)
class MegaMHCContext:
    """Layer-owned launch context created outside the timed operator path.

    A production factory must create one context per model layer and per
    ordered CUDA stream on an SM90 device. It compiles every admitted token
    specialization, binds the four layer-static tensors, and gives this
    context exclusive ownership of one ``MegaMHCWorkspace``. CPU devices are
    permitted only so the pure host contract can be tested without a GPU.
    ``trusted_fast_path`` is reserved for the production factory after every
    dynamic tensor ABI has been validated. It keeps only cheap per-call
    semantic and ownership guards.
    """

    profile: MegaMHCProfile
    supported_tokens: FrozenSet[int]
    static_binding: MegaMHCStaticBinding
    static_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    workspace: MegaMHCWorkspace
    launcher: MegaMHCLauncher
    owner_thread_id: int
    trusted_fast_path: bool = False

    @classmethod
    def bind(
        cls,
        *,
        profile: MegaMHCProfile,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        workspace: MegaMHCWorkspace,
        launcher: MegaMHCLauncher,
        trusted_fast_path: bool = False,
    ) -> "MegaMHCContext":
        supported_tokens = launcher.supported_tokens
        if not supported_tokens:
            raise ValueError("supported_tokens must not be empty")
        normalized_values = []
        for value in supported_tokens:
            if isinstance(value, bool):
                raise TypeError("supported_tokens must contain exact integers")
            try:
                normalized_values.append(int(operator.index(value)))
            except TypeError as exc:
                raise TypeError("supported_tokens must contain exact integers") from exc
        normalized_tokens = frozenset(normalized_values)
        if any(value <= 0 for value in normalized_tokens):
            raise ValueError("supported_tokens must contain only positive values")
        if normalized_tokens != supported_tokens:
            raise TypeError("launcher.supported_tokens must be exact integers")
        if launcher.device != workspace.device:
            raise ValueError("launcher and workspace devices differ")
        if launcher.stream_handle != workspace.stream_handle:
            raise ValueError("launcher and workspace streams differ")
        if launcher.owner_thread_id != threading.get_ident():
            raise RuntimeError("launcher belongs to a different host thread")
        if trusted_fast_path:
            _validate_trusted_context_binding(
                profile,
                fn,
                hc_scale,
                hc_base,
                norm_weight,
                workspace,
                launcher.static_alignments,
            )
        return cls(
            profile=profile,
            supported_tokens=normalized_tokens,
            static_binding=MegaMHCStaticBinding.from_tensors(
                fn, hc_scale, hc_base, norm_weight
            ),
            static_tensors=(fn, hc_scale, hc_base, norm_weight),
            workspace=workspace,
            launcher=launcher,
            owner_thread_id=threading.get_ident(),
            trusted_fast_path=trusted_fast_path,
        )


MegaMHCOutputs = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _unsupported(message: str) -> NoReturn:
    raise NotImplementedError(f"unsupported Hopper MegaMHC profile: {message}")


def _normalize_alignments(values: object, *, count: int, name: str) -> Tuple[int, ...]:
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of alignments") from exc
    if len(raw_values) != count:
        raise ValueError(f"{name} must contain exactly {count} values")
    normalized = []
    for value in raw_values:
        if isinstance(value, bool):
            raise TypeError(f"{name} must contain exact integers")
        try:
            alignment = int(operator.index(value))
        except TypeError as exc:
            raise TypeError(f"{name} must contain exact integers") from exc
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"{name} values must be positive powers of two")
        normalized.append(alignment)
    return tuple(normalized)


def _require_contiguous(
    name: str, tensor: torch.Tensor, pointer_alignment: int
) -> None:
    if not tensor.is_contiguous():
        _unsupported(f"{name} must be contiguous")
    if tensor.storage_offset() != 0:
        _unsupported(f"{name}.storage_offset must be 0")
    if tensor.data_ptr() % pointer_alignment != 0:
        _unsupported(f"{name} pointer must be {pointer_alignment}-byte aligned")


def _require_tensor(name: str, tensor: object) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _exact_integer(name: str, value: object) -> int:
    if isinstance(value, bool):
        _unsupported(f"{name} must be an exact integer, got {value!r}")
    try:
        return int(operator.index(value))
    except TypeError:
        _unsupported(f"{name} must be an exact integer, got {value!r}")


def _validate_trusted_context_binding(
    profile: MegaMHCProfile,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    workspace: MegaMHCWorkspace,
    static_alignments: Tuple[int, ...],
) -> None:
    """Validate layer-static state once before enabling the trusted hot path."""
    if profile != DEEPSEEK_V4_FLASH_PROFILE:
        _unsupported("only DeepSeek-V4-Flash normal mode is admitted")
    expected = (
        (
            "fn",
            fn,
            (
                profile.projection_outputs,
                profile.residual_streams * profile.hidden_size,
            ),
            torch.float32,
        ),
        ("hc_scale", hc_scale, (3,), torch.float32),
        ("hc_base", hc_base, (profile.projection_outputs,), torch.float32),
        ("norm_weight", norm_weight, (profile.hidden_size,), torch.bfloat16),
    )
    alignments = _normalize_alignments(
        static_alignments, count=4, name="launcher.static_alignments"
    )
    for (name, tensor, shape, dtype), alignment in zip(expected, alignments):
        _require_tensor(name, tensor)
        if tuple(tensor.shape) != shape:
            _unsupported(f"{name}.shape={tuple(tensor.shape)}, expected {shape}")
        if tensor.dtype != dtype:
            _unsupported(f"{name}.dtype={tensor.dtype}, expected {dtype}")
        if tensor.device != workspace.device:
            raise ValueError(f"{name} is not on context device")
        if tensor.requires_grad:
            _unsupported(f"{name} must not require gradients")
        _require_contiguous(name, tensor, alignment)
    if workspace.device.type == "cuda":
        if workspace.device.index is None:
            raise ValueError("CUDA MegaMHC workspace requires an explicit device index")
        if workspace.stream_handle is None:
            raise ValueError("CUDA MegaMHC workspace requires a stream handle")
    elif workspace.stream_handle is not None:
        raise ValueError("non-CUDA test workspace must not carry a CUDA stream handle")


def _validate_scalar_contract(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    tile_n: int,
    norm_eps: float,
    profile: MegaMHCProfile,
) -> None:
    scalar_contract = (
        ("rms_eps", float(rms_eps), profile.rms_eps),
        ("hc_pre_eps", float(hc_pre_eps), profile.hc_pre_eps),
        ("hc_sinkhorn_eps", float(hc_sinkhorn_eps), profile.hc_sinkhorn_eps),
        (
            "hc_post_mult_value",
            float(hc_post_mult_value),
            profile.hc_post_mult_value,
        ),
        ("norm_eps", float(norm_eps), profile.norm_eps),
    )
    for name, value, required in scalar_contract:
        if value != required:
            _unsupported(f"{name}={value}, expected {required}")
    if _exact_integer("sinkhorn_repeat", sinkhorn_repeat) != profile.sinkhorn_repeat:
        _unsupported(
            f"sinkhorn_repeat={sinkhorn_repeat}, expected {profile.sinkhorn_repeat}"
        )
    if _exact_integer("n_splits", n_splits) != 1:
        _unsupported(f"n_splits={n_splits}, expected 1")
    if _exact_integer("tile_n", tile_n) != 1:
        _unsupported(f"tile_n={tile_n}, expected 1")


def _validate_trusted_scalar_contract(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    tile_n: int,
    norm_eps: float,
    profile: MegaMHCProfile,
) -> None:
    """Validate canonical Python scalar arguments without normalization work."""
    if type(rms_eps) is not float or rms_eps != profile.rms_eps:
        _unsupported(f"rms_eps={rms_eps}, expected {profile.rms_eps}")
    if type(hc_pre_eps) is not float or hc_pre_eps != profile.hc_pre_eps:
        _unsupported(f"hc_pre_eps={hc_pre_eps}, expected {profile.hc_pre_eps}")
    if type(hc_sinkhorn_eps) is not float or hc_sinkhorn_eps != profile.hc_sinkhorn_eps:
        _unsupported(
            f"hc_sinkhorn_eps={hc_sinkhorn_eps}, expected {profile.hc_sinkhorn_eps}"
        )
    if (
        type(hc_post_mult_value) is not float
        or hc_post_mult_value != profile.hc_post_mult_value
    ):
        _unsupported(
            "hc_post_mult_value="
            f"{hc_post_mult_value}, expected {profile.hc_post_mult_value}"
        )
    if type(norm_eps) is not float or norm_eps != profile.norm_eps:
        _unsupported(f"norm_eps={norm_eps}, expected {profile.norm_eps}")
    if type(sinkhorn_repeat) is not int or sinkhorn_repeat != profile.sinkhorn_repeat:
        _unsupported(
            f"sinkhorn_repeat={sinkhorn_repeat}, expected {profile.sinkhorn_repeat}"
        )
    if type(n_splits) is not int or n_splits != 1:
        _unsupported(f"n_splits={n_splits}, expected 1")
    if type(tile_n) is not int or tile_n != 1:
        _unsupported(f"tile_n={tile_n}, expected 1")


def _validate_trusted_call(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    tile_n: int,
    norm_eps: float,
    context: MegaMHCContext,
) -> Tuple[int, Tuple[int, ...]]:
    """Run only guards that cannot be delegated to a compiled launcher ABI."""
    dynamic_tensors = (x, residual, post_layer_mix, comb_res_mix)
    for name, tensor in zip(
        ("x", "residual", "post_layer_mix", "comb_res_mix"),
        dynamic_tensors,
    ):
        _require_tensor(name, tensor)
    if x.ndim != 2 or residual.ndim != 3:
        raise ValueError("x/residual ranks must be 2/3")
    tokens = int(x.shape[0])
    if tokens not in context.supported_tokens:
        _unsupported(f"T={tokens} has no compiled specialization")
    dynamic_alignments = _normalize_alignments(
        context.launcher.dynamic_alignments(tokens),
        count=8,
        name=f"dynamic alignments for T={tokens}",
    )
    profile = context.profile
    expected = (
        ("x", x, (tokens, profile.hidden_size), torch.bfloat16),
        (
            "residual",
            residual,
            (tokens, profile.residual_streams, profile.hidden_size),
            torch.bfloat16,
        ),
        (
            "post_layer_mix",
            post_layer_mix,
            (tokens, profile.residual_streams, 1),
            torch.float32,
        ),
        (
            "comb_res_mix",
            comb_res_mix,
            (tokens, profile.residual_streams, profile.residual_streams),
            torch.float32,
        ),
    )
    for (name, tensor, shape, dtype), alignment in zip(
        expected, dynamic_alignments[:4]
    ):
        if tensor.shape != shape:
            _unsupported(f"{name}.shape={tuple(tensor.shape)}, expected {shape}")
        if tensor.dtype != dtype:
            _unsupported(f"{name}.dtype={tensor.dtype}, expected {dtype}")
        if tensor.device != context.workspace.device:
            raise ValueError(f"{name} is not on context device")
        if tensor.requires_grad:
            _unsupported(f"{name} must not require gradients")
        _require_contiguous(name, tensor, alignment)
    if norm_weight is None:
        _unsupported("norm_weight is required")
    if (
        fn is not context.static_tensors[0]
        or hc_scale is not context.static_tensors[1]
        or hc_base is not context.static_tensors[2]
        or norm_weight is not context.static_tensors[3]
    ):
        raise RuntimeError("layer-static tensor object changed; rebuild context")
    if not context.static_binding.matches_tensors(fn, hc_scale, hc_base, norm_weight):
        raise RuntimeError(
            "layer-static tensor storage/pointer contract changed; rebuild context"
        )
    _validate_trusted_scalar_contract(
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        tile_n,
        norm_eps,
        context.profile,
    )
    return tokens, dynamic_alignments


def _validate_profile(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    tile_n: int,
    norm_eps: float,
    context: MegaMHCContext,
) -> Tuple[int, Tuple[int, ...]]:
    profile = context.profile
    if profile != DEEPSEEK_V4_FLASH_PROFILE:
        _unsupported("only DeepSeek-V4-Flash normal mode is admitted")
    if norm_weight is None:
        _unsupported("norm_weight is required")
    named_tensors = (
        ("x", x),
        ("residual", residual),
        ("post_layer_mix", post_layer_mix),
        ("comb_res_mix", comb_res_mix),
        ("fn", fn),
        ("hc_scale", hc_scale),
        ("hc_base", hc_base),
        ("norm_weight", norm_weight),
    )
    for name, tensor in named_tensors:
        _require_tensor(name, tensor)
    if x.ndim != 2 or residual.ndim != 3:
        raise ValueError("x/residual ranks must be 2/3")
    tokens = int(x.shape[0])
    if tokens not in context.supported_tokens:
        _unsupported(f"T={tokens} has no compiled specialization")
    dynamic_alignments = _normalize_alignments(
        context.launcher.dynamic_alignments(tokens),
        count=8,
        name=f"dynamic alignments for T={tokens}",
    )
    static_alignments = _normalize_alignments(
        context.launcher.static_alignments,
        count=4,
        name="launcher.static_alignments",
    )
    pointer_alignments = dict(
        zip(
            (
                "x",
                "residual",
                "post_layer_mix",
                "comb_res_mix",
                "fn",
                "hc_scale",
                "hc_base",
                "norm_weight",
            ),
            (*dynamic_alignments[:4], *static_alignments),
        )
    )
    expected = {
        "x": (x, (tokens, profile.hidden_size), torch.bfloat16),
        "residual": (
            residual,
            (tokens, profile.residual_streams, profile.hidden_size),
            torch.bfloat16,
        ),
        "post_layer_mix": (
            post_layer_mix,
            (tokens, profile.residual_streams, 1),
            torch.float32,
        ),
        "comb_res_mix": (
            comb_res_mix,
            (tokens, profile.residual_streams, profile.residual_streams),
            torch.float32,
        ),
        "fn": (
            fn,
            (
                profile.projection_outputs,
                profile.residual_streams * profile.hidden_size,
            ),
            torch.float32,
        ),
        "hc_scale": (hc_scale, (3,), torch.float32),
        "hc_base": (
            hc_base,
            (profile.projection_outputs,),
            torch.float32,
        ),
        "norm_weight": (
            norm_weight,
            (profile.hidden_size,),
            torch.bfloat16,
        ),
    }
    for name, (tensor, shape, dtype) in expected.items():
        if tuple(tensor.shape) != shape:
            _unsupported(f"{name}.shape={tuple(tensor.shape)}, expected {shape}")
        if tensor.dtype != dtype:
            _unsupported(f"{name}.dtype={tensor.dtype}, expected {dtype}")
        if tensor.device != context.workspace.device:
            raise ValueError(f"{name} is not on context device")
        if tensor.requires_grad:
            _unsupported(f"{name} must not require gradients")
        _require_contiguous(name, tensor, pointer_alignments[name])
    _validate_scalar_contract(
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        tile_n,
        norm_eps,
        profile,
    )
    current_binding = MegaMHCStaticBinding.from_tensors(
        fn, hc_scale, hc_base, norm_weight
    )
    if current_binding != context.static_binding:
        raise RuntimeError("layer-static tensor identity changed; rebuild context")
    return tokens, dynamic_alignments


def _validate_workspace(context: MegaMHCContext) -> None:
    workspace = context.workspace
    if threading.get_ident() != context.owner_thread_id:
        raise RuntimeError("MegaMHC context belongs to a different host thread")
    if workspace.poisoned:
        raise RuntimeError("MegaMHC workspace is poisoned; rebuild it")
    if workspace.device.type == "cuda":
        if workspace.device.index is None:
            raise ValueError("CUDA MegaMHC workspace requires an explicit device index")
        if workspace.stream_handle is None:
            raise ValueError("CUDA MegaMHC workspace requires a stream handle")
        if torch.cuda.current_device() != workspace.device.index:
            raise RuntimeError("MegaMHC workspace belongs to a different CUDA device")
        current_stream = torch.cuda.current_stream(workspace.device)
        if int(current_stream.cuda_stream) != int(workspace.stream_handle):
            raise RuntimeError("MegaMHC workspace belongs to a different CUDA stream")
    elif workspace.stream_handle is not None:
        raise ValueError("non-CUDA test workspace must not carry a CUDA stream handle")


def _allocate_outputs(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_streams: int,
) -> MegaMHCOutputs:
    tokens, hidden = x.shape
    return (
        torch.empty_like(residual),
        torch.empty(
            (tokens, residual_streams, 1),
            dtype=torch.float32,
            device=x.device,
        ),
        torch.empty(
            (tokens, residual_streams, residual_streams),
            dtype=torch.float32,
            device=x.device,
        ),
        torch.empty((tokens, hidden), dtype=torch.bfloat16, device=x.device),
    )


def _storage_address(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def _validate_outputs(
    outputs: MegaMHCOutputs,
    x: torch.Tensor,
    residual: torch.Tensor,
    profile: MegaMHCProfile,
    protected_inputs: Tuple[torch.Tensor, ...],
    output_alignments: Tuple[int, ...],
) -> None:
    if not isinstance(outputs, tuple):
        raise TypeError("out must be a tuple of four tensors")
    if len(outputs) != 4:
        raise ValueError("out must contain exactly four tensors")
    if any(not isinstance(tensor, torch.Tensor) for tensor in outputs):
        raise TypeError("every out element must be a torch.Tensor")
    tokens = int(x.shape[0])
    expected = (
        ("new_residual", outputs[0], tuple(residual.shape), torch.bfloat16),
        (
            "new_post_mix",
            outputs[1],
            (tokens, profile.residual_streams, 1),
            torch.float32,
        ),
        (
            "new_comb_res_mix",
            outputs[2],
            (tokens, profile.residual_streams, profile.residual_streams),
            torch.float32,
        ),
        ("y_bf16", outputs[3], tuple(x.shape), torch.bfloat16),
    )
    alignments = _normalize_alignments(
        output_alignments, count=4, name="output alignments"
    )
    for (name, tensor, shape, dtype), alignment in zip(expected, alignments):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name}.shape must be {shape}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name}.dtype must be {dtype}")
        if tensor.device != x.device:
            raise ValueError(f"{name} is not on the input device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.storage_offset() != 0:
            raise ValueError(f"{name}.storage_offset must be 0")
        if tensor.data_ptr() % alignment != 0:
            raise ValueError(f"{name} pointer must be {alignment}-byte aligned")
    output_storages = [_storage_address(tensor) for tensor in outputs]
    if len(set(output_storages)) != len(output_storages):
        raise ValueError("out tensors must not share storage")
    protected_storages = {_storage_address(tensor) for tensor in protected_inputs}
    if any(address in protected_storages for address in output_storages):
        raise ValueError("out tensors must not alias any input or static tensor")


def _validate_allocated_output_alignments(
    outputs: MegaMHCOutputs, output_alignments: Tuple[int, ...]
) -> None:
    alignments = _normalize_alignments(
        output_alignments, count=4, name="output alignments"
    )
    for name, tensor, alignment in zip(
        ("new_residual", "new_post_mix", "new_comb_res_mix", "y_bf16"),
        outputs,
        alignments,
    ):
        if tensor.data_ptr() % alignment != 0:
            raise RuntimeError(
                f"allocator returned {name} below the required "
                f"{alignment}-byte alignment"
            )


def mega_mhc(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
    *,
    context: MegaMHCContext,
    out: Optional[MegaMHCOutputs] = None,
) -> MegaMHCOutputs:
    """Allocate/validate outputs and enqueue exactly one bound kernel.

    Only the exact DeepSeek-V4-Flash Normal BF16 profile is admitted. Every
    other model, dtype, layout, scalar contract, or token count raises
    ``NotImplementedError``; there is no torch compute fallback.

    ``comb_res_mix`` uses vLLM's contiguous stored-transposed convention:
    ``comb_res_mix[t, source, destination]`` weights residual ``source`` when
    producing residual ``destination``. ``new_comb_res_mix`` has the same
    convention. The return order is ``(new_residual, new_post_mix,
    new_comb_res_mix, y_bf16)``; ``y_bf16`` is the final RMS-normalized BF16
    output, not the pre-normalization weighted residual sum.
    """
    if context.trusted_fast_path:
        tokens, dynamic_alignments = _validate_trusted_call(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            norm_weight,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_eps,
            context,
        )
    else:
        tokens, dynamic_alignments = _validate_profile(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            norm_weight,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_eps,
            context,
        )
    _validate_workspace(context)
    outputs = (
        _allocate_outputs(x, residual, context.profile.residual_streams)
        if out is None
        else out
    )
    if out is None and context.trusted_fast_path:
        _validate_allocated_output_alignments(outputs, dynamic_alignments[4:])
    if out is not None or not context.trusted_fast_path:
        protected_inputs = (
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            norm_weight,
        )
        _validate_outputs(
            outputs,
            x,
            residual,
            context.profile,
            protected_inputs,
            dynamic_alignments[4:],
        )
    try:
        context.launcher.launch(
            context.workspace,
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
        )
    except BaseException:
        # A failed or interrupted cooperative launch leaves cross-CTA state
        # indeterminate. Reusing the lease could deadlock a later invocation.
        context.workspace.poisoned = True
        raise
    return outputs


__all__ = [
    "DEEPSEEK_V4_FLASH_PROFILE",
    "MegaMHCContext",
    "MegaMHCLauncher",
    "MegaMHCProfile",
    "MegaMHCStaticBinding",
    "MegaMHCTensorBinding",
    "MegaMHCWorkspace",
    "mega_mhc",
]
