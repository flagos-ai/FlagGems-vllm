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
"""Platform-isolated resolution of the vendor's native operator.

The host APIs are not global: on Ascend they live in ``fla_npu`` (the AscendC
kernels vllm-ascend runs in production) and ``vllm_ascend``, while every other
backend uses vLLM's own implementation.  Resolving them here keeps that split
in one place, so a test or benchmark never probes the wrong vendor's package.

Every resolver returns ``(callable, source)``, or ``(None, None)`` when the
platform has no native entry.  ``source`` tells the caller which tensor
layouts to pass:

``"fla_npu.ascendc"``
    The AscendC kernels take *kernel* layouts, which differ from this repo's
    operator contracts (see each resolver's note).

``"vllm"`` / ``"vllm_ascend"``
    These take the operator's own layouts, so a caller can pass the same
    tensors it would give the FlagGems op.
"""

from __future__ import annotations

import importlib


def vendor():
    """The backend this process is running on (``flaggems_vllm.vendor_name``)."""
    import flaggems_vllm

    return flaggems_vllm.vendor_name


def _attr(module_name, attr):
    try:
        module = importlib.import_module(module_name)
    except Exception:  # ImportError, or a module that raises while importing
        return None
    fn = getattr(module, attr, None)
    return fn if callable(fn) else None


# vLLM moved the KDA/conv modules between releases; probe the known locations.
_VLLM_FLA_KDA_MODULES = (
    "vllm.model_executor.layers.fla.ops.kda",
    "vllm.third_party.flash_linear_attention.ops.kda",
)
_VLLM_CONV1D_MODULE = "vllm.model_executor.layers.mamba.ops.causal_conv1d"


def _resolve(ascend_specs, vllm_attr, vllm_modules):
    """Resolve the native entry for the current platform.

    ``ascend_specs`` is an ordered tuple of ``(module, attr)`` pairs tried only
    on Ascend -- the AscendC kernels and the vllm-ascend entries expose
    different names, so each pair carries its own.  ``vllm_modules`` are tried
    only on the other backends.
    """
    vendor_name = vendor()
    if vendor_name == "ascend":
        for module_name, attr in ascend_specs:
            fn = _attr(module_name, attr)
            if fn is not None:
                return fn, module_name
        # Never fall through to vLLM's CUDA/Triton kernels on Ascend: they
        # import cleanly but cannot run on the NPU.
        return None, None
    for module_name in vllm_modules:
        fn = _attr(module_name, vllm_attr)
        if fn is not None:
            return fn, module_name
    return None, None


def causal_conv1d_fn():
    """Native ``causal_conv1d_fn`` for this platform."""
    return _resolve(
        ascend_specs=(
            ("fla_npu.ops.ascendc", "causal_conv1d_fn"),
            ("vllm_ascend.ops.causal_conv1d", "causal_conv1d_fn"),
        ),
        vllm_attr="causal_conv1d_fn",
        vllm_modules=(_VLLM_CONV1D_MODULE,),
    )


def causal_conv1d_update():
    """Native ``causal_conv1d_update`` for this platform."""
    return _resolve(
        ascend_specs=(
            ("fla_npu.ops.ascendc", "causal_conv1d_update"),
            ("vllm_ascend.ops.causal_conv1d", "causal_conv1d_update"),
        ),
        vllm_attr="causal_conv1d_update",
        vllm_modules=(_VLLM_CONV1D_MODULE,),
    )


def chunk_kda():
    """Native KDA prefill for this platform.

    The AscendC entry (``fla_npu.ops.ascendc.chunk_kda_fwd``) applies the
    bounded gate from the raw gate plus ``A_log``/``dt_bias`` but wants
    ``dt_bias`` flattened to ``[H*K]`` fp32, pre-normalised q/k and a
    pre-sigmoided beta.  The other entries expose vLLM's FLA signature.
    """
    return _resolve(
        ascend_specs=(
            ("fla_npu.ops.ascendc", "chunk_kda_fwd"),
            ("vllm_ascend.ops.triton.kda.kda", "chunk_kda_with_fused_gate"),
        ),
        vllm_attr="chunk_kda_with_fused_gate",
        vllm_modules=_VLLM_FLA_KDA_MODULES,
    )


def fused_recurrent_kda():
    """Native recurrent KDA for this platform.

    ``fla_npu``'s AscendC entry is compiled for bfloat16 only on Ascend 910B
    (see the shipped ``kernel/ascend910b/recurrent_kda`` variants), so an
    fp16/fp32 launch fails at tiling time with ``161002``.
    """
    return _resolve(
        ascend_specs=(
            ("fla_npu.ops.ascendc", "recurrent_kda"),
            ("vllm_ascend.ops.triton.kda.kda", "fused_recurrent_kda"),
        ),
        vllm_attr="fused_recurrent_kda",
        vllm_modules=_VLLM_FLA_KDA_MODULES,
    )
