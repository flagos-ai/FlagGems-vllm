# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Optional API contracts used by TLE kernels, without modifying the runtime."""

from functools import cache


def _has_gpu_apis(language, names):
    gpu = getattr(language, "gpu", None)
    return (
        gpu is not None
        and hasattr(gpu, "smem")
        and all(callable(getattr(gpu, name, None)) for name in names)
    )


def supports_topk_tle(language):
    return callable(getattr(language, "cumsum", None)) and _has_gpu_apis(
        language, ("alloc", "local_ptr")
    )


def supports_sparse_mla_tle(language):
    return callable(getattr(language, "pipe", None)) and _has_gpu_apis(
        language, ("alloc", "local_ptr", "copy", "warp_specialize")
    )


def supports_megagate_tle(language):
    return callable(getattr(language, "pipe", None)) and _has_gpu_apis(
        language,
        (
            "alloc",
            "local_ptr",
            "copy",
            "warp_specialize",
            "alloc_barrier",
            "barrier_wait",
            "wgmma",
            "wgmma_wait",
        ),
    )


def _has_module_wide_named_barrier_support(core, libtriton):
    """Check both halves of FlagTree's module-wide barrier allocation fix."""

    passes = getattr(getattr(libtriton, "tle", None), "passes", None)
    return (
        getattr(core, "_FIRST_VIRTUAL_NAMED_BARRIER_ID", None) == 16
        and callable(getattr(core, "_reserve_named_barrier_ids", None))
        and callable(getattr(passes, "add_allocate_named_barriers", None))
    )


@cache
def supports_module_wide_named_barriers():
    """Whether FlagTree has frontend and backend support for fix #1278.

    A package-version comparison is insufficient because local/vendor wheels
    may share the same version while carrying different compiler revisions.
    Probe the Python-side virtual-ID allocator and the compiled allocation pass
    together, fail closed, and cache the result outside operator dispatch.
    """

    try:
        from triton._C import libtriton
        from triton.experimental.tle.language.gpu import core
    except (ImportError, RuntimeError):
        return False
    return _has_module_wide_named_barrier_support(core, libtriton)
