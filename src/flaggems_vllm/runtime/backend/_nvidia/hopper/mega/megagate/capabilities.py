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

"""Compiler capability checks owned by the Hopper MegaGate operator."""

from functools import lru_cache
from inspect import Parameter, signature


def supports_megagate_tle(language) -> bool:
    """Whether ``language`` exposes every TLE primitive used by MegaGate."""

    gpu = getattr(language, "gpu", None)
    return (
        all(
            callable(getattr(language, name, None))
            for name in (
                "pipe",
                "device_mesh",
                "MeshConfig",
                "distributed_barrier",
                "shard_id",
            )
        )
        and gpu is not None
        and hasattr(gpu, "smem")
        and all(
            callable(getattr(gpu, name, None))
            for name in (
                "alloc",
                "local_ptr",
                "copy",
                "warp_specialize",
                "alloc_barrier",
                "barrier_wait",
                "wgmma",
                "wgmma_wait",
            )
        )
    )


def _has_module_wide_named_barrier_support(core, libtriton) -> bool:
    """Recognize the frontend/pass contract added by FlagTree fix #1278."""

    passes = getattr(getattr(libtriton, "tle", None), "passes", None)
    reserve = getattr(core, "_reserve_named_barrier_ids", None)
    try:
        generator = signature(reserve).parameters.get("_generator")
    except (TypeError, ValueError):
        generator = None
    return (
        getattr(core, "_FIRST_VIRTUAL_NAMED_BARRIER_ID", None) == 16
        and callable(reserve)
        and generator is not None
        and generator.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        and generator.default is None
        and callable(getattr(passes, "add_allocate_named_barriers", None))
    )


@lru_cache(maxsize=None)
def supports_module_wide_named_barriers() -> bool:
    """Whether the installed FlagTree carries MegaGate's barrier fix."""

    try:
        from triton._C import libtriton
        from triton.experimental.tle.language.gpu import core
    except (ImportError, RuntimeError):
        return False
    return _has_module_wide_named_barrier_support(core, libtriton)


def is_megagate_available() -> bool:
    """Whether this process can import and compile Hopper MegaGate kernels."""

    try:
        import triton.experimental.tle.language as language
    except (ImportError, RuntimeError):
        return False
    return supports_megagate_tle(language) and supports_module_wide_named_barriers()


def require_megagate_tle() -> None:
    """Raise a stable setup error when MegaGate's TLE contract is unavailable."""

    try:
        import triton.experimental.tle.language as language
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("MegaGate requires Triton TLE support") from error
    if not supports_megagate_tle(language):
        raise RuntimeError(
            "MegaGate requires TLE pipe, warp-specialization, barrier, "
            "TMA, WGMMA, and cluster APIs"
        )
    if not supports_module_wide_named_barriers():
        raise RuntimeError(
            "MegaGate requires module-wide named-barrier allocation support"
        )


__all__ = [
    "is_megagate_available",
    "require_megagate_tle",
    "supports_megagate_tle",
    "supports_module_wide_named_barriers",
]
