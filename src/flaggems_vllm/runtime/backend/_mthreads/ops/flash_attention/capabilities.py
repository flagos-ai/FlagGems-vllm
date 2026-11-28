# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Require registered TLE primitives and their public Python interfaces."""

from functools import lru_cache

_TLE = (
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.alloc_barriers",
    "gpu.barrier_wait",
    "gpu.wgmma",
    "gpu.wgmma_wait",
)


@lru_cache(None)
def supports_tle():
    try:
        import triton.experimental.tle as extension
        import triton.experimental.tle.language as language
        from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401

        supported = extension.get_supported_primitives("mthreads")
        for name in _TLE:
            # Unknown primitive names can be accepted by is_primitive_supported.
            # Require explicit registration as well as the Python attribute.
            if name not in supported or not extension.is_primitive_supported(
                "mthreads", name
            ):
                return False
            value = language
            for component in name.split("."):
                value = getattr(value, component)
        return True
    except (ImportError, AttributeError, OSError, TypeError):
        return False
