# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Optional API contracts used by TLE kernels, without modifying the runtime."""

from functools import lru_cache
from inspect import Parameter, signature


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
    """Recognize the FlagTree frontend contract added by fix #1278.

    Older FlagTree releases already expose virtual barrier IDs and the named
    barrier allocation pass, but reserve IDs per outlined function.  The fix
    adds the ``_generator`` argument so the frontend can reserve IDs on the
    shared module owner.  Checking that signature avoids enabling MegaGate on
    the currently pinned 0.6/0.7 compiler builds, where helper-local barriers
    can collide.

    FlagTree ships the Python frontend and compiled pass as one wheel; the
    pass-presence check below additionally rejects incomplete installations.
    """

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
def supports_module_wide_named_barriers():
    """Whether the installed FlagTree wheel carries fix #1278's contract.

    A package-version comparison is insufficient because local/vendor wheels
    may share the same version while carrying different compiler revisions.
    Probe the distinguishing Python-side module-owner API plus the compiled
    allocation pass, fail closed, and cache the result outside dispatch.
    """

    try:
        from triton._C import libtriton
        from triton.experimental.tle.language.gpu import core
    except (ImportError, RuntimeError):
        return False
    return _has_module_wide_named_barrier_support(core, libtriton)
