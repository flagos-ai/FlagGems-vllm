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

"""Production context factory for the single-kernel Hopper MegaMHC path."""

from __future__ import annotations

import operator
import threading
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from .api import (
    DEEPSEEK_V4_FLASH_PROFILE,
    MegaMHCContext,
    MegaMHCProfile,
    MegaMHCWorkspace,
    _normalize_alignments,
)

if TYPE_CHECKING:
    from .tle.prepared_launcher import PreparedCompiledLauncher


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


class _ExactPreparedLauncher:
    """Adapt one prepared token specialization to the public launcher ABI."""

    def __init__(
        self,
        prepared: PreparedCompiledLauncher,
        static_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        token = _exact_token(prepared.token_count, "prepared launcher token_count")
        prepared.validate_environment()
        owner_thread_id = threading.get_ident()
        launcher_owner = _exact_token(
            prepared.owner_thread_id, "prepared launcher owner_thread_id"
        )
        if launcher_owner != owner_thread_id:
            raise RuntimeError("prepared launcher belongs to a different host thread")

        device = _device(prepared.device)
        stream_handle = _stream_handle(prepared.stream_handle, device)
        static_values = tuple(prepared.static_bound_values)
        if len(static_values) != 4:
            raise ValueError(
                "prepared launcher static_bound_values must contain four tensors"
            )
        if any(
            actual is not expected
            for actual, expected in zip(static_values, static_tensors)
        ):
            raise RuntimeError(
                f"prepared launcher T={token} is bound to different static tensors"
            )
        static_validator = getattr(prepared, "validate_static_bindings", None)
        if not callable(static_validator):
            raise TypeError("prepared launcher must expose validate_static_bindings()")
        static_validator(static_tensors)

        self._prepared = prepared
        self._token = token
        self.supported_tokens = frozenset((token,))
        self._dynamic_alignments = _normalize_alignments(
            prepared.dynamic_alignments,
            count=8,
            name=f"prepared launcher T={token} dynamic_alignments",
        )
        self.static_alignments = _normalize_alignments(
            prepared.static_alignments,
            count=4,
            name=f"prepared launcher T={token} static_alignments",
        )
        self.device = device
        self.stream_handle = stream_handle
        self.owner_thread_id = owner_thread_id

    def dynamic_alignments(self, tokens: int) -> Tuple[int, ...]:
        if tokens != self._token:
            raise NotImplementedError(
                f"unsupported Hopper MegaMHC profile: T={tokens} has no "
                "compiled specialization"
            )
        return self._dynamic_alignments

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
        if tokens != self._token:
            raise NotImplementedError(
                f"unsupported Hopper MegaMHC profile: T={tokens} has no "
                "compiled specialization"
            )
        self._prepared.run_trusted(
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


def _create_context_from_prepared(
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    prepared: PreparedCompiledLauncher,
    *,
    profile: MegaMHCProfile = DEEPSEEK_V4_FLASH_PROFILE,
) -> MegaMHCContext:
    """Bind an internally prepared launcher to one layer and ordered stream.

    Compilation and launcher preparation must finish before this function.
    The prepared launcher's token count is the sole admitted specialization.
    """
    static_tensors = (fn, hc_scale, hc_base, norm_weight)
    launcher = _ExactPreparedLauncher(prepared, static_tensors)
    if launcher.device.type == "cuda":
        capability = torch.cuda.get_device_capability(launcher.device)
        if capability[0] != 9:
            raise NotImplementedError(
                "unsupported Hopper MegaMHC profile: prepared launchers "
                f"require SM90, got sm_{capability[0]}{capability[1]}"
            )
        current_stream = torch.cuda.current_stream(launcher.device)
        if int(current_stream.cuda_stream) != launcher.stream_handle:
            raise RuntimeError("prepared launcher belongs to a different CUDA stream")
    workspace = MegaMHCWorkspace(
        device=launcher.device,
        stream_handle=launcher.stream_handle,
        state=launcher,
    )
    return MegaMHCContext.bind(
        profile=profile,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        norm_weight=norm_weight,
        workspace=workspace,
        launcher=launcher,
        trusted_fast_path=True,
    )


def create_mega_mhc_context(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    profile: MegaMHCProfile = DEEPSEEK_V4_FLASH_PROFILE,
) -> MegaMHCContext:
    """Compile and bind one exact-token Hopper MegaMHC specialization.

    This setup API runs outside the operator hot path and outside CUDA Graph
    capture.  The first four tensors describe the dynamic ABI for one physical
    token bucket; the four model tensors are retained as layer-static
    bindings.  Kernel selection, LaunchSpec construction, and FlagTree JIT
    preparation are entirely internal to FlagGems-vLLM.
    """
    if profile != DEEPSEEK_V4_FLASH_PROFILE:
        raise NotImplementedError(
            "only the DeepSeek-V4-Flash normal MegaMHC profile is supported"
        )
    dynamic_tensors = (x, residual, post_layer_mix, comb_res_mix)
    static_tensors = (fn, hc_scale, hc_base, norm_weight)
    if any(
        not isinstance(tensor, torch.Tensor)
        for tensor in (*dynamic_tensors, *static_tensors)
    ):
        raise TypeError("MegaMHC context inputs must all be torch.Tensor objects")
    device = x.device
    if device.type != "cuda":
        raise ValueError("MegaMHC production context requires CUDA tensors")
    if any(tensor.device != device for tensor in (*dynamic_tensors, *static_tensors)):
        raise ValueError("all MegaMHC context tensors must share one CUDA device")
    # FlagTree resolves both the compilation device and the ordered launch
    # stream from the current CUDA context.  Prepare under the tensors' device
    # instead of whichever device the caller happened to leave current.
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMHC context must be prepared before CUDA Graph capture"
            )
        capability = torch.cuda.get_device_capability(device)
        if capability[0] != 9:
            raise NotImplementedError(
                f"Hopper MegaMHC requires SM90, got sm_{capability[0]}{capability[1]}"
            )

        outputs = {
            "new_residual": torch.empty_like(residual),
            "new_prev_mix": None,
            "new_post_mix": torch.empty_like(post_layer_mix),
            "new_comb_res_mix": torch.empty_like(comb_res_mix),
            "y_bf16": torch.empty_like(x),
        }
        inputs = {
            "x": x,
            "residual": residual,
            "post_mix": post_layer_mix,
            "comb_res_mix": comb_res_mix,
            "shifted_prev_mix": None,
            "fn": fn,
            "mix_scales": hc_scale,
            "mix_bases": hc_base,
            "rmsnorm_weight": norm_weight,
            "hc_mult": profile.residual_streams,
            "hc_norm_eps": profile.rms_eps,
            "hc_pre_eps": profile.hc_pre_eps,
            "hc_post_scale": profile.hc_post_mult_value,
            "sinkhorn_eps": profile.hc_sinkhorn_eps,
            "num_sinkhorn_iters": profile.sinkhorn_repeat,
            "rmsnorm_eps": profile.norm_eps,
            "rmsnorm_scale": profile.norm_scale,
        }
        from .tle.dispatch import MissingTLECaseError
        from .tle.isolated_dispatch import prepare_isolated_safe_launcher

        try:
            prepared = prepare_isolated_safe_launcher(inputs, outputs)
        except MissingTLECaseError as exc:
            raise NotImplementedError(str(exc)) from exc
        return _create_context_from_prepared(
            fn,
            hc_scale,
            hc_base,
            norm_weight,
            prepared,
            profile=profile,
        )


__all__ = [
    "create_mega_mhc_context",
]
