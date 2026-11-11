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

"""Production context factory for prepared Hopper MegaMHC launchers."""

from __future__ import annotations

import operator
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional, Protocol, Tuple

import torch

from .api import (
    DEEPSEEK_V4_FLASH_PROFILE,
    MegaMHCContext,
    MegaMHCProfile,
    MegaMHCWorkspace,
)


class PreparedMegaMHCLauncher(Protocol):
    """Exact ABI exposed by a launch-free prepared TLE specialization."""

    token_count: int
    dynamic_alignments: Tuple[int, ...]
    static_bound_values: Tuple[object, ...]
    static_alignments: Tuple[int, ...]
    device: object
    stream_handle: Optional[int]
    owner_thread_id: int

    def validate_environment(self) -> None: ...

    def validate_static_bindings(self, values: Sequence[object]) -> None: ...

    def run_trusted(self, dynamic_values: Sequence[object]) -> None: ...


def _exact_token(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an exact integer")
    try:
        token = int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an exact integer") from exc
    if token <= 0:
        raise ValueError(f"{name} must be positive")
    return token


def _alignments(values: object, count: int, name: str) -> Tuple[int, ...]:
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable") from exc
    if len(raw_values) != count:
        raise ValueError(f"{name} must contain exactly {count} values")
    result = []
    for value in raw_values:
        if isinstance(value, bool):
            raise TypeError(f"{name} must contain exact integers")
        try:
            alignment = int(operator.index(value))
        except TypeError as exc:
            raise TypeError(f"{name} must contain exact integers") from exc
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"{name} values must be positive powers of two")
        result.append(alignment)
    return tuple(result)


def _device(value: object) -> torch.device:
    if isinstance(value, bool):
        raise TypeError("prepared launcher device must not be bool")
    if isinstance(value, int):
        return torch.device("cuda", value)
    try:
        return torch.device(value)
    except (RuntimeError, TypeError) as exc:
        raise TypeError("prepared launcher exposes an invalid device") from exc


def _stream_handle(value: object, device: torch.device) -> Optional[int]:
    if value is None:
        if device.type == "cuda":
            raise ValueError("CUDA prepared launcher requires a stream handle")
        return None
    if isinstance(value, bool):
        raise TypeError("stream_handle must be an exact integer")
    try:
        handle = int(operator.index(value))
    except TypeError as exc:
        raise TypeError("stream_handle must be an exact integer") from exc
    if device.type != "cuda":
        raise ValueError("non-CUDA prepared launcher must not carry a stream handle")
    return handle


@dataclass(frozen=True)
class _TokenBinding:
    prepared: PreparedMegaMHCLauncher
    dynamic_alignments: Tuple[int, ...]


class MegaMHCPreparedLauncherRegistry:
    """Immutable O(1) exact-token dispatch over prepared launchers."""

    def __init__(
        self,
        prepared_launchers: Mapping[int, PreparedMegaMHCLauncher],
        static_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        if not isinstance(prepared_launchers, Mapping):
            raise TypeError("prepared_launchers must be a mapping")
        if not prepared_launchers:
            raise ValueError("prepared_launchers must not be empty")

        owner_thread_id = threading.get_ident()
        bindings = {}
        registry_device = None
        registry_stream = None
        static_alignment_rows = []
        for raw_token, prepared in prepared_launchers.items():
            token = _exact_token(raw_token, "prepared launcher registry key")
            prepared_token = _exact_token(
                prepared.token_count, "prepared launcher token_count"
            )
            if prepared_token != token:
                raise ValueError(
                    f"registry key T={token} != launcher token_count={prepared_token}"
                )
            if token in bindings:
                raise ValueError(f"duplicate prepared launcher for T={token}")
            prepared.validate_environment()
            launcher_owner = _exact_token(
                prepared.owner_thread_id, "prepared launcher owner_thread_id"
            )
            if launcher_owner != owner_thread_id:
                raise RuntimeError(
                    "prepared launcher belongs to a different host thread"
                )
            launcher_device = _device(prepared.device)
            launcher_stream = _stream_handle(prepared.stream_handle, launcher_device)
            if registry_device is None:
                registry_device = launcher_device
                registry_stream = launcher_stream
            elif (
                launcher_device != registry_device or launcher_stream != registry_stream
            ):
                raise ValueError(
                    "all prepared launchers must share one device and CUDA stream"
                )

            static_values = tuple(prepared.static_bound_values)
            if len(static_values) != 4:
                raise ValueError(
                    "prepared launcher static_bound_values must contain four tensors"
                )
            if any(
                actual is not expected
                for actual, expected in zip(static_values, static_tensors, strict=True)
            ):
                raise RuntimeError(
                    f"prepared launcher T={token} is bound to different static tensors"
                )
            static_validator = getattr(prepared, "validate_static_bindings", None)
            if not callable(static_validator):
                raise TypeError(
                    "prepared launcher must expose validate_static_bindings()"
                )
            static_validator(static_tensors)
            dynamic_alignments = _alignments(
                prepared.dynamic_alignments,
                8,
                f"prepared launcher T={token} dynamic_alignments",
            )
            static_alignment_rows.append(
                _alignments(
                    prepared.static_alignments,
                    4,
                    f"prepared launcher T={token} static_alignments",
                )
            )
            bindings[token] = _TokenBinding(prepared, dynamic_alignments)

        self._bindings = MappingProxyType(bindings)
        self._supported_tokens = frozenset(bindings)
        self._device = registry_device
        self._stream_handle = registry_stream
        self._owner_thread_id = owner_thread_id
        self._static_alignments = tuple(
            max(row[index] for row in static_alignment_rows) for index in range(4)
        )

    @property
    def supported_tokens(self) -> frozenset[int]:
        return self._supported_tokens

    @property
    def static_alignments(self) -> Tuple[int, int, int, int]:
        return self._static_alignments  # type: ignore[return-value]

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def stream_handle(self) -> Optional[int]:
        return self._stream_handle

    @property
    def owner_thread_id(self) -> int:
        return self._owner_thread_id

    def dynamic_alignments(self, tokens: int) -> Tuple[int, ...]:
        try:
            return self._bindings[tokens].dynamic_alignments
        except KeyError as exc:
            raise NotImplementedError(
                f"unsupported Hopper MegaMHC profile: T={tokens} has no "
                "compiled specialization"
            ) from exc

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
    ) -> None:
        del workspace
        tokens = int(x.shape[0])
        try:
            binding = self._bindings[tokens]
        except KeyError as exc:
            raise NotImplementedError(
                f"unsupported Hopper MegaMHC profile: T={tokens} has no "
                "compiled specialization"
            ) from exc
        binding.prepared.run_trusted(
            (
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                new_residual,
                new_post_mix,
                new_comb_res_mix,
                y_bf16,
            )
        )


def create_mega_mhc_context(
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    prepared_launchers: Mapping[int, PreparedMegaMHCLauncher],
    *,
    profile: MegaMHCProfile = DEEPSEEK_V4_FLASH_PROFILE,
) -> MegaMHCContext:
    """Create one layer-, stream-, device- and host-thread-owned context.

    Compilation and launcher preparation must finish before this function.
    The registry keys are the sole source of admitted token specializations;
    callers cannot provide a divergent ``supported_tokens`` set.
    """
    static_tensors = (fn, hc_scale, hc_base, norm_weight)
    registry = MegaMHCPreparedLauncherRegistry(prepared_launchers, static_tensors)
    if registry.device.type == "cuda":
        capability = torch.cuda.get_device_capability(registry.device)
        if capability[0] != 9:
            raise NotImplementedError(
                "unsupported Hopper MegaMHC profile: prepared launchers "
                f"require SM90, got sm_{capability[0]}{capability[1]}"
            )
        current_stream = torch.cuda.current_stream(registry.device)
        if int(current_stream.cuda_stream) != registry.stream_handle:
            raise RuntimeError(
                "prepared launcher registry belongs to a different CUDA stream"
            )
    workspace = MegaMHCWorkspace(
        device=registry.device,
        stream_handle=registry.stream_handle,
        state=registry,
    )
    return MegaMHCContext.bind(
        profile=profile,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        norm_weight=norm_weight,
        workspace=workspace,
        launcher=registry,
        trusted_fast_path=True,
    )


__all__ = [
    "MegaMHCPreparedLauncherRegistry",
    "PreparedMegaMHCLauncher",
    "create_mega_mhc_context",
]
